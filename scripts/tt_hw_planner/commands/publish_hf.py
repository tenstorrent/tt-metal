# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``publish-hf`` — publish a tool-optimized model to Hugging Face.

Assembles a model repo from an optimize run: the optimized model implementation (the demo dir, which
holds the committed wins after ``commit-wins``) plus an auto-generated model card whose perf table is
drawn straight from the run's own metrics (throughput per-user, batch, TTFT/TPOT/E2E, PCC gate,
tt-metal commit). Weights are REFERENCED (``base_model`` pointer), never re-uploaded.

Pushes with ``huggingface_hub`` (token from ``--token`` / ``HF_TOKEN`` / the CLI login file). Use
``--dry-run`` to stage + preview the card without pushing. A ``tt-model.yaml`` manifest is written so
the repo can later be turned into a servable ``tt-model`` bundle.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from ..optimize_dashboard import (
    collect_state,
    find_run_dir,
    repo_root_for_run,
    run_slug,
    state_dir_candidates,
)
from .optimize import _repo_root, _resolve_target


# box (planner) -> (arch, hardware, mesh_device) for the tt-model.yaml serve block. hardware and
# mesh_device must be values the vLLM plugin's closed table accepts; override with --hardware/--mesh.
_BOX_TARGET = {
    "QB2": ("blackhole", "p300x2", "P300x2"),
    "GalaxyBH": ("blackhole", "p150x4", "P150x4"),
    "P150": ("blackhole", "p150", "P150"),
    "P300": ("blackhole", "p300", "P300"),
    "T3K": ("wormhole_b0", "n300x4", "N300x4"),
    "GalaxyWH": ("wormhole_b0", "galaxy", "TG"),
    "N150": ("wormhole_b0", "n150", "N150"),
    "N300": ("wormhole_b0", "n300", "N300"),
}


def _yaml_quote(s: str) -> str:
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _write_tt_model_yaml(
    path: Path,
    *,
    state: dict,
    slug: str,
    repo_id: str,
    checkout: str,
    weights: str | None,
    box: str | None,
    arch: str | None,
    hardware: str | None,
    mesh_device: str | None,
    kind: str,
    plugin_ref: str,
    vllm_version: str,
    extra_models_dir: str,
    commit: str | None,
) -> None:
    """Emit a schema-5.1 tt-model.yaml describing how to build+serve this optimized model as a
    v5.1 container package. Fields the run can't provide (the vLLM adapter dir, plugin) are stated
    as sane defaults/overrides; `tt-model package --container` resolves and validates the rest."""
    a, hw, mesh = _BOX_TARGET.get(box or "", ("blackhole", "p300x2", "P300x2"))
    arch = arch or a
    hardware = hardware or hw
    mesh_device = mesh_device or mesh
    thr = state.get("throughput") or {}
    sv = state.get("serving") or {}
    pt = sv.get("per_token") or {}
    batch = state.get("batch")
    perf_bits = []
    if thr.get("current") is not None:
        perf_bits.append(f"{thr['current']:.1f} tok/s/user decode")
        if thr.get("baseline"):
            perf_bits.append(f"(+{(thr['current']-thr['baseline'])/thr['baseline']*100:.0f}% vs baseline)")
    if pt.get("ms") is not None:
        perf_bits.append(f"{pt['ms']:.1f} ms/token")
    if batch:
        perf_bits.append(f"at batch {batch}")
    if hardware:
        perf_bits.append(f"on {hardware}")
    perf = " ".join(perf_bits) or "measured with tt_hw_planner; see the dashboard."
    lines = [
        'schema: "5.1"',
        f"repo: {repo_id}",
        f"name: {slug}",
        f"weights: {weights or 'REPLACE/with-base-weights-repo'}",
        f"kind: {kind}",
        f"arch: {arch}",
        "",
        "source:",
        f"  tt_metal: {checkout}",
        "  code:",
        "    - models/common",
        f"    - models/demos/{slug}",
        '  ubuntu: "22.04"',
        '  python: "3.12"',
        "",
        "runtime:",
        f'  vllm: {{version: "{vllm_version}"}}',
        f"  plugin: {{repo: https://github.com/tenstorrent/vllm-tt-plugin, ref: {plugin_ref}}}",
        f"  extra_models_dir: {extra_models_dir}",
        "  lock: requirements.lock",
        "",
        "serve:",
        "  port: 8000",
        "  block_size: 64",
        f"  max_num_seqs: {batch or 32}",
        f"  hardware: {hardware}",
        f"  mesh_device: {mesh_device}",
        "  env:",
        f"    ARCH_NAME: {arch}",
        "  args: [--trust-remote-code]",
        "",
        "verify:",
        f'  - "import models.demos.{slug} as m; assert m"',
        "",
        "card:",
        f"  description: >",
        f"    {slug}, brought up and optimized on Tenstorrent {hardware} with tt_hw_planner",
        f"    (kernel/lever autotuning against a PCC-gated end-to-end pipeline).",
        f"  performance: >",
        f"    {perf}.",
        f"  limitations: >",
        f"    Community bring-up via tt_hw_planner; only {hardware} was validated.",
        f"  architecture: autoport ({slug})",
        f"  status: Experimental community bring-up",
        "  license:",
        "    id: apache-2.0",
        "  pipeline_tag: text-generation",
    ]
    if weights:
        lines.append(f"  base_model: [{weights}]")
    if commit:
        lines.append(f"  related: tt-metal commit {commit}")
    path.write_text("\n".join(lines) + "\n")


def _fetch_dashboard_state(url: str) -> dict:
    """Read a live/served dashboard's /api/state — the exact metrics shown on the dashboard (throughput
    per-user, batch, serving, etc.). More reliable than out-of-process state-dir discovery for a run
    that is still in flight (its ledger lives in the running process's PERF_MCP_STATE_DIR)."""
    import urllib.request

    u = url.rstrip("/") + "/api/state"
    with urllib.request.urlopen(u, timeout=20) as r:
        return json.loads(r.read().decode())


def _git_commit(repo_root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except Exception:
        pass
    return None


def _fmt(v, nd=2) -> str:
    return "—" if v is None else f"{float(v):.{nd}f}"


def _build_card(state: dict, slug: str, base_weights: str | None, commit: str | None) -> str:
    cfg = state.get("config") or {}
    thr = state.get("throughput") or {}
    sv = state.get("serving") or {}
    m = state.get("metric") or {}
    batch = state.get("batch")
    pt = sv.get("per_token") or {}
    ft = sv.get("first_token") or {}
    e2 = sv.get("e2e_latency") or {}

    L: list[str] = ["---"]
    if base_weights:
        L.append(f"base_model: {base_weights}")
        L.append("base_model_relation: finetune")
    L += [
        "library_name: tt-metal",
        "pipeline_tag: text-generation",
        "tags:",
        "  - tenstorrent",
        "  - tt-metal",
        "  - ttnn",
        "  - blackhole",
        "---",
        "",
        f"# {slug} — Tenstorrent-optimized",
        "",
        "Optimized for Tenstorrent hardware with `tt_hw_planner` (kernel/lever autotuning "
        "against a PCC-gated end-to-end pipeline).",
        "",
    ]
    if base_weights:
        L.append(f"- **Base weights:** [{base_weights}](https://huggingface.co/{base_weights})")
    if commit:
        L.append(f"- **tt-metal commit:** `{commit}`")
    if cfg.get("pcc_test"):
        L.append(f"- **Accuracy gate:** `{str(cfg['pcc_test']).split('::')[-1]}` (PCC)")
    if cfg.get("devices"):
        L.append(f"- **Devices:** {cfg['devices']}")
    L += ["", "## Performance", "", "| Metric | Value |", "| --- | --- |"]
    if batch is not None:
        L.append(f"| Batch (concurrent users) | {batch} |")
    if thr.get("current") is not None:
        L.append(f"| Throughput | {_fmt(thr['current'])} tok/s/user |")
        if thr.get("baseline"):
            gain = (thr["current"] - thr["baseline"]) / thr["baseline"] * 100.0
            L.append(
                f"| Throughput vs. baseline | {_fmt(thr['baseline'])} → {_fmt(thr['current'])} tok/s/user (+{gain:.0f}%) |"
            )
    if ft.get("ms") is not None:
        L.append(f"| TTFT | {_fmt(ft['ms'], 1)} ms |")
    if pt.get("ms") is not None:
        L.append(f"| TPOT / ITL | {_fmt(pt['ms'], 1)} ms |")
    if e2.get("ms") is not None:
        L.append(f"| E2E latency | {_fmt(e2['ms'], 1)} ms |")
    if m.get("name") and m.get("current") is not None:
        L.append(f"| {m['name']} | {_fmt(m.get('current'), 1)} {m.get('unit', '')} |")
    L += [
        "",
        "## Run on Tenstorrent",
        "",
        "```bash",
        "# vLLM, OpenAI-compatible API via tt-inference-server, from an optimized tt-metal checkout:",
        f"python3 run.py --model {slug} --tt-device <device> --workflow server \\",
        "  --local-server --tt-metal-home <your tt-metal checkout>",
        "```",
        "",
        "_Published with `tt_hw_planner publish-hf`. The optimized model implementation lives under "
        "`models/`; weights are referenced from the base repo above (not stored here)._",
        "",
    ]
    return "\n".join(L)


# Architectures the tt-metal vLLM plugin already serves via a stock generator in
# models/tt_transformers/tt/generator_vllm.py — any of these needs only a thin bundle pointing at
# the stock class (no new code). Anything else gets a scaffolded adapter stub to complete.
_STOCK_GENERATORS = {
    "LlamaForCausalLM": "LlamaForCausalLM",
    "MistralForCausalLM": "MistralForCausalLM",
    "Qwen2ForCausalLM": "QwenForCausalLM",
    "Qwen3ForCausalLM": "QwenForCausalLM",
    "Gemma3ForConditionalGeneration": "Gemma3ForConditionalGeneration",
    "Gemma3ForCausalLM": "Gemma3ForConditionalGeneration",
    "Exaone4ForCausalLM": "Exaone4_5_ForConditionalGeneration",
}
_GEN_MOD = "models.tt_transformers.tt.generator_vllm"


def _detect_arch_and_type(demo_dir: Path) -> tuple[str | None, str | None]:
    """The HF architecture name + model_type from any config.json under the demo (works for every
    model, not just one): arch is what the vLLM plugin registers as TT<arch>."""
    import glob as _g

    for cfg in _g.glob(str(Path(demo_dir) / "**" / "config.json"), recursive=True):
        try:
            d = json.loads(Path(cfg).read_text())
        except Exception:
            continue
        arch = (d.get("architectures") or [None])[0]
        if arch:
            return arch, d.get("model_type")
    return None, None


def _pick_base_generator(arch: str, model_type: str | None) -> tuple[str, bool]:
    """(stock_generator_class, is_stub) for an arch. Stock arches map to a supported generator
    (is_stub=False, trivially servable). Novel arches pick the closest base (hybrid vs plain) and
    are a stub (is_stub=True) whose body the author completes. Generic across all models."""
    if arch in _STOCK_GENERATORS:
        return _STOCK_GENERATORS[arch], False
    hay = f"{arch} {model_type or ''}".lower()
    hybrid = any(t in hay for t in ("_h", "hybrid", "mamba", "jamba", "nemotronh", "recurrent"))
    return ("HybridAttentionForCausalLM" if hybrid else "LlamaForCausalLM"), True


def _scaffold_vllm_bundle(
    checkout: Path, extra_models_dir: str, arch: str, model_type: str | None, weights: str | None, slug: str
) -> tuple[bool, bool, str]:
    """Create the vLLM adapter bundle (vllm_metadata.json + adapter.py) under the checkout if absent,
    so the container manifest is complete for ANY model. Returns (created, is_stub, path).

    - Stock arch (Llama/Qwen/Mistral/Gemma/...) -> adapter is a trivial subclass of the stock
      generator: servable as-is.
    - Novel arch -> adapter subclasses the closest base with a clearly-marked TODO body."""
    bundle = Path(checkout) / extra_models_dir
    meta = bundle / "vllm_metadata.json"
    base_cls, is_stub = _pick_base_generator(arch, model_type)
    if meta.is_file():
        return False, is_stub, str(bundle)
    bundle.mkdir(parents=True, exist_ok=True)
    tt_arch = arch if arch.startswith("TT") else "TT" + arch
    meta.write_text(
        json.dumps(
            {
                "arch": arch,
                "main_class": "adapter:" + tt_arch,
                "weights": weights or "REPLACE/with-base-weights-repo",
                "note": "Generated by tt_hw_planner publish-hf.",
            },
            indent=2,
        )
        + "\n"
    )
    todo = (
        "    # TODO(author): this arch is not a plugin built-in. Wire the base generator to this\n"
        "    #   demo's optimized model builder / weights loader (see the demo's tt/pipeline.py) and\n"
        "    #   set any arch-specific config the base expects.\n    pass\n"
        if is_stub
        else "    # Stock arch: the base generator already serves this architecture. Nothing to override.\n"
        "    pass\n"
    )
    (bundle / "adapter.py").write_text(
        "# SPDX-License-Identifier: Apache-2.0\n"
        f'"""vLLM generator adapter for {slug} ({arch}) on Tenstorrent.\n\n'
        f"The tt-metal vLLM plugin registers this class as {tt_arch} via vllm_metadata.json.\n"
        f'"""\n\n'
        "try:\n"
        f"    from {_GEN_MOD} import {base_cls} as _Base\n"
        "except Exception:  # available only inside the serving image\n"
        "    _Base = object\n\n\n"
        f"class {tt_arch}(_Base):\n"
        f'    """TT generator for {arch} (base: {base_cls})."""\n\n' + todo
    )
    return True, is_stub, str(bundle)


def _checkout_of(demo_dir: Path) -> Path:
    """The tt-metal checkout root a demo dir belongs to (the path before '/models/')."""
    parts = Path(demo_dir).resolve().parts
    if "models" in parts:
        return Path(*parts[: parts.index("models")])
    return Path(demo_dir).resolve().parents[2]


def _run_container(args, state: dict, slug: str, demo_dir, commit: str | None) -> int:
    """Build + push a real v5.1 container bundle via tt-model (exactly like the published TT repos):
    generate tt-model.yaml, then `tt-model package --container` (2.5-4h OCI build) and `tt-model push`."""
    import subprocess
    import tempfile

    checkout = _checkout_of(Path(demo_dir))
    ttm = getattr(args, "tt_model_bin", None) or "tt-model"
    out = getattr(args, "out", None) or str(Path.home() / "tt-model-builds")
    extra = getattr(args, "extra_models_dir", None) or f"models/demos/{slug}/vllm_bundle"

    # Make every model publishable this way: ensure the vLLM adapter bundle exists (scaffold it from
    # the model's own HF arch when missing). Stock arches are servable as-is; novel arches get a stub.
    arch_det, mtype = _detect_arch_and_type(Path(demo_dir))
    if arch_det and not getattr(args, "no_scaffold", False):
        created, is_stub, bpath = _scaffold_vllm_bundle(
            checkout, extra, arch_det, mtype, getattr(args, "weights", None), slug
        )
        state_note = "STUB — complete adapter.py before serving" if is_stub else "stock generator — servable as-is"
        print(f"  [publish-hf] vLLM bundle {'created' if created else 'exists'}: {bpath}  ({state_note})")

    yaml_path = Path(tempfile.mkdtemp(prefix="tt_ttmodel_")) / "tt-model.yaml"
    _write_tt_model_yaml(
        yaml_path,
        state=state,
        slug=slug,
        repo_id=args.repo,
        checkout=str(checkout),
        weights=getattr(args, "weights", None),
        box=getattr(args, "box", None),
        arch=getattr(args, "arch", None),
        hardware=getattr(args, "hardware", None),
        mesh_device=getattr(args, "mesh", None),
        kind=getattr(args, "kind", None) or "vllm-plugin",
        plugin_ref=getattr(args, "plugin_ref", None) or "main",
        vllm_version=getattr(args, "vllm_version", None) or "0.24.0",
        extra_models_dir=extra,
        commit=commit,
    )
    print(f"  [publish-hf] tt-model.yaml -> {yaml_path}")
    print("  " + "-" * 60)
    for ln in yaml_path.read_text().splitlines():
        print("  | " + ln)
    print("  " + "-" * 60)

    # Load-time validation (no hardware/build) — surfaces missing source.code / adapter dirs early.
    # tt_kernel lives in tt-model's own venv, so validate with the python next to the tt-model binary.
    ttm_p = Path(ttm)
    val_py = str(ttm_p.parent / "python") if (ttm_p.parent / "python").exists() else "python3"
    val = subprocess.run(
        [
            val_py,
            "-c",
            "import sys;from tt_kernel.container_manifest import load_container_manifest as L;"
            "m=L(sys.argv[1], check_sources=True);p=m.resolve_profile();"
            "print('VALID:', m.name, m.kind, p.hardware, p.mesh_device)",
            str(yaml_path),
        ],
        capture_output=True,
        text=True,
    )
    if val.stdout.strip():
        print("  [publish-hf] " + val.stdout.strip())
    if val.returncode != 0:
        print("  [publish-hf] manifest validation failed:\n" + (val.stderr.strip()[-800:] or "(no detail)"))
        print(
            "  [publish-hf] Most commonly: the vLLM adapter dir (runtime.extra_models_dir) with a "
            "vllm_metadata.json must exist and be under source.code. Author it, then re-run."
        )
        return 3

    if getattr(args, "dry_run", False):
        print(
            "  [publish-hf] --dry-run: manifest generated + validated; NOT building the image "
            "(tt-model package --container is a 2.5-4h OCI build)."
        )
        return 0

    pkg = [ttm, "package", "--container", str(yaml_path), "--out", out]
    print(f"  [publish-hf] building container (2.5-4h): {' '.join(pkg)}")
    rc = subprocess.run(pkg).returncode
    if rc != 0:
        print(f"  [publish-hf] tt-model package failed (rc={rc}).")
        return 4
    staged = str(Path(out) / slug)
    push = [ttm, "push", staged]
    if getattr(args, "public", False):
        push.append("--public")
    if getattr(args, "publish", False):
        push.append("--publish")
    print(f"  [publish-hf] pushing: {' '.join(push)}")
    rc = subprocess.run(push).returncode
    if rc != 0:
        print(f"  [publish-hf] tt-model push failed (rc={rc}).")
        return 4
    print(f"  [publish-hf] published container bundle: https://huggingface.co/{args.repo}")
    return 0


def cmd_publish_hf(args) -> int:
    repo_root = _repo_root()
    slug = None
    demo_dir = None
    target = getattr(args, "target", None)
    if target:
        demo_dir = _resolve_target(target, repo_root)
        if demo_dir is not None:
            slug = demo_dir.name

    from_dash = getattr(args, "from_dashboard", None)
    if from_dash:
        # Pull the exact metrics the dashboard shows (best for an in-flight run).
        try:
            state = _fetch_dashboard_state(from_dash)
        except Exception as e:
            print(f"  [publish-hf] could not read dashboard state from {from_dash}: {e}")
            return 2
        slug = slug or (state.get("model") or {}).get("slug")
        run_dir = None
        state_root = repo_root
    else:
        run_dir = find_run_dir(repo_root, slug=slug, run_ref=getattr(args, "run", None))
        if run_dir is None:
            what = f"for '{slug}' " if slug else ""
            print(
                f"  [publish-hf] no optimize run found {what}under {repo_root}. "
                f"Pass a target, --run, or --from-dashboard <url>."
            )
            return 2
        slug = slug or run_slug(run_dir)
        state_root = repo_root_for_run(run_dir, repo_root)
        state = collect_state(run_dir, state_dir_candidates(state_root, slug), slug)
    if not slug:
        print("  [publish-hf] could not determine the model slug. Pass a target.")
        return 2

    # The publishable model source is the demo dir in the MAIN checkout — after `commit-wins` it holds
    # the optimized code. Fall back to the run's own model_root only if the checkout lacks it.
    if demo_dir is None:
        try:
            from ..bringup_loop import find_demo_dir

            d = find_demo_dir(slug, repo_root)
            demo_dir = d.resolve() if d else None
        except Exception:
            demo_dir = None
    if demo_dir is None or not Path(demo_dir).is_dir():
        mr = (state.get("model") or {}).get("root")
        if mr and Path(mr).is_dir():
            demo_dir = Path(mr)
    if demo_dir is None or not Path(demo_dir).is_dir():
        print(f"  [publish-hf] could not locate the model demo dir for '{slug}'. Run `commit-wins` first.")
        return 2

    commit = _git_commit(state_root)
    base_weights = getattr(args, "weights", None)

    if getattr(args, "container", False):
        return _run_container(args, state, slug, demo_dir, commit)

    card = _build_card(state, slug, base_weights, commit)

    stage = Path(tempfile.mkdtemp(prefix="tt_publish_"))
    (stage / "README.md").write_text(card)
    manifest = {
        "repo": args.repo,
        "name": slug,
        "source": {"tt_metal": str(state_root), "code": [f"models/demos/{slug}", "models/common"]},
        "weights": base_weights,
        "serve": {"block_size": 64, "max_num_seqs": state.get("batch") or 32},
        "provenance": {
            "tt_metal_commit": commit,
            "run": run_dir.name if run_dir else (state.get("run") or {}).get("id"),
            "throughput_tok_s_user": (state.get("throughput") or {}).get("current"),
            "batch": state.get("batch"),
        },
    }
    (stage / "tt-model.yaml").write_text(json.dumps(manifest, indent=2))

    if not getattr(args, "card_only", False):
        dst = stage / "models" / "demos" / slug
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            demo_dir,
            dst,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git", "*.pt", "*.bin", "*.safetensors"),
        )

    print(f"  [publish-hf] repo:   {args.repo}")
    print(f"  [publish-hf] model:  {slug}   (source: {demo_dir})")
    print(
        f"  [publish-hf] commit: {commit}   batch: {state.get('batch')}   "
        f"throughput: {_fmt((state.get('throughput') or {}).get('current'))} tok/s/user"
    )
    print(f"  [publish-hf] staged: {stage}")

    if getattr(args, "dry_run", False):
        print("  [publish-hf] --dry-run: NOT pushing. Model card preview:")
        print("  " + "-" * 60)
        for ln in card.splitlines():
            print("  | " + ln)
        print("  " + "-" * 60)
        return 0

    try:
        from huggingface_hub import create_repo, upload_folder
    except Exception:
        print(
            "  [publish-hf] huggingface_hub is not installed. `pip install huggingface_hub`, "
            "or re-run with --dry-run to preview."
        )
        return 3

    token = getattr(args, "token", None) or os.environ.get("HF_TOKEN")
    try:
        create_repo(
            args.repo, repo_type="model", private=bool(getattr(args, "private", False)), exist_ok=True, token=token
        )
        upload_folder(
            repo_id=args.repo,
            folder_path=str(stage),
            repo_type="model",
            token=token,
            commit_message=f"Publish {slug} (tt_hw_planner; tt-metal {commit})",
        )
    except Exception as e:
        print(f"  [publish-hf] push failed: {e}")
        return 4

    url = f"https://huggingface.co/{args.repo}"
    print(f"  [publish-hf] published: {url}")
    return 0
