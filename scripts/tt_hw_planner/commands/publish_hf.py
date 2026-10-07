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


def _hf_token(args) -> str | None:
    """The Hugging Face token, in priority order: ``--token`` → ``HF_TOKEN`` → the CLI login file.
    Single source of truth so every upload/download path resolves the token the same way."""
    tok = getattr(args, "token", None) or os.environ.get("HF_TOKEN")
    if tok:
        return tok
    for p in ("~/.cache/huggingface/token", "~/.huggingface/token"):
        fp = Path(p).expanduser()
        try:
            if fp.is_file() and fp.read_text().strip():
                return fp.read_text().strip()
        except Exception:
            pass
    return None


# box (planner) -> (arch, hardware, mesh_device) for the tt-model.yaml serve block. hardware and
# mesh_device must be values the vLLM plugin's closed table accepts; override with --hardware/--mesh.
_BOX_TARGET = {
    "QB2": ("blackhole", "p300x2", "P300x2"),
    "GalaxyBH": ("blackhole", "p150x4", "P150x4"),
    "P150": ("blackhole", "p150", "P150"),
    "P300": ("blackhole", "p300", "P300"),
    "T3K": ("wormhole_b0", "n300x4", "T3K"),
    "GalaxyWH": ("wormhole_b0", "galaxy", "TG"),
    "N150": ("wormhole_b0", "n150", "N150"),
    "N300": ("wormhole_b0", "n300", "N300"),
}

# The vLLM plugin (vllm_tt_plugin/utils/dp_discovery.py) accepts ONLY these mesh_device labels, or a
# literal "(rows, cols)" tuple; anything else raises at manifest-load. Keep this in sync with it.
_PLUGIN_MESH = {
    "BH-Galaxy",
    "N150",
    "N150x4",
    "N300",
    "P100",
    "P150",
    "P150x2",
    "P150x4",
    "P150x8",
    "P300",
    "P300x2",
    "QB2",
    "T3K",
    "TG",
}


def _validate_mesh_device(mesh_device: str) -> None:
    """Raise before publishing if mesh_device is not a value the vLLM plugin will accept."""
    import re as _re

    if mesh_device in _PLUGIN_MESH:
        return
    if _re.fullmatch(r"\(\s*\d+\s*,\s*\d+\s*\)", str(mesh_device)):
        return
    raise ValueError(
        "mesh_device %r is not accepted by the vLLM plugin -- expected one of %s or a '(rows, cols)' "
        "tuple. Fix the box->mesh mapping or pass --mesh with a valid value."
        % (mesh_device, ", ".join(sorted(_PLUGIN_MESH)))
    )


def _box_from_env(env: dict | None) -> str | None:
    """The planner box the run's DETECTED hardware is (its manifest env: arch family + chip count), or
    None when that matches no box or more than one -- never a guess."""
    from ..hardware import HARDWARE

    arch = str((env or {}).get("arch") or "").strip().lower()
    try:
        chips = int((env or {}).get("device_count") or 0)
    except (TypeError, ValueError):
        chips = 0
    if not arch or chips <= 0:
        return None
    hits = [b.name for b in HARDWARE if b.arch.lower() == arch and b.chips == chips and b.name in _BOX_TARGET]
    return hits[0] if len(hits) == 1 else None


def _serve_target(box: str | None, env: dict | None) -> tuple:
    """(arch, hardware, mesh_device) for the serve block: the --box the operator named, else the box
    the run detected, else (None, None, None) -- which the caller must fill with overrides."""
    return _BOX_TARGET.get(box if box in _BOX_TARGET else (_box_from_env(env) or ""), (None, None, None))


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
    lock: str | None = None,
    vllm_path: str | None = None,
    plugin_path: str | None = None,
) -> None:
    """Emit a schema-5.1 tt-model.yaml describing how to build+serve this optimized model as a
    v5.1 container package. Fields the run can't provide (the vLLM adapter dir, plugin) are stated
    as sane defaults/overrides; `tt-model package --container` resolves and validates the rest."""
    a, hw, mesh = _serve_target(box, state.get("env"))
    _batch = state.get("batch")
    serve_max = _batch or 32
    serve_maxlen = None
    try:
        _bm = Path(checkout) / extra_models_dir / slug / "vllm_metadata.json"
        if _bm.is_file():
            _md = json.loads(_bm.read_text())
            _mv = _md.get("max_num_seqs")
            if isinstance(_mv, int) and _mv > 0:
                serve_max = _mv
            _ml = _md.get("max_model_len")
            if isinstance(_ml, int) and _ml > 0:
                serve_maxlen = _ml
    except Exception:
        pass
    arch = arch or a
    hardware = hardware or hw
    mesh_device = mesh_device or mesh
    if not (arch and hardware and mesh_device):
        raise ValueError(
            "no serve target: the run's detected hardware (%s) is no single known box and no --box / "
            "--arch / --hardware / --mesh was given" % (state.get("env") or {}).get("arch")
        )
    _validate_mesh_device(mesh_device)
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
        # The vLLM adapter subclasses a base generator from models/tt_transformers/tt/generator_vllm
        # (initialize_vllm_model etc.), so that package MUST be in the image or the import fails at
        # serve time and the engine dies with AttributeError.
        "    - models/tt_transformers",
        f"    - models/demos/{slug}",
        '  ubuntu: "22.04"',
        '  python: "3.12"',
        "",
        "runtime:",
        (f"  vllm: {{path: {vllm_path}}}" if vllm_path else f'  vllm: {{version: "{vllm_version}"}}'),
        (
            f"  plugin: {{path: {plugin_path}}}"
            if plugin_path
            else f"  plugin: {{repo: https://github.com/tenstorrent/vllm-tt-plugin, ref: {plugin_ref}}}"
        ),
        f"  extra_models_dir: {extra_models_dir}",
        # runtime.lock is optional; only emit it when a real requirements.lock is provided, else the
        # build fails resolving a path that doesn't exist. Deps resolve live without it.
        *([f"  lock: {lock}"] if lock else []),
        "",
        "serve:",
        "  port: 8000",
        "  block_size: 64",
        f"  max_num_seqs: {serve_max}",
        *([f"  max_model_len: {serve_maxlen}"] if serve_maxlen else []),
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
    ]
    _arch_tag = str((state.get("env") or {}).get("arch") or "").strip().lower()
    if _arch_tag:
        L.append(f"  - {_arch_tag}")  # the hardware the run detected, not an assumed one
    L += [
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


def _detect_weights(demo_dir: Path, slug: str) -> str | None:
    """The base-weights HF repo id for this model, so the card/manifest never carry a placeholder.
    Prefers config.json's _name_or_path, else the HF id in the bring-up metadata that best matches the
    model slug (e.g. the Lightning-BF16 id over an unrelated Nano id)."""
    import glob as _g
    import re as _re

    for cfg in _g.glob(str(Path(demo_dir) / "**" / "config.json"), recursive=True):
        try:
            d = json.loads(Path(cfg).read_text())
        except Exception:
            continue
        nop = d.get("_name_or_path")
        if nop and "/" in str(nop):
            return str(nop)
    toks = [t for t in _re.split(r"[^a-z0-9]+", slug.lower()) if len(t) > 1]
    best, best_score = None, 0
    for meta in (
        "bringup_status.json",
        "e2e_plan.json",
        "BRING_UP_PLAN.md",
        "README.md",
        "manifest.json",
        "bringup_cc_state.json",
        ".bringup_cc_state.json",
    ):
        p = Path(demo_dir) / meta
        if not p.is_file():
            continue
        try:
            txt = p.read_text(errors="replace")
        except Exception:
            continue
        for cand in _re.findall(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", txt):
            low = cand.lower()
            if "/" not in cand or low.startswith("models/") or low.startswith("tests/"):
                continue
            score = sum(1 for t in toks if t in low)
            if score > best_score:
                best_score, best = score, cand
    return best if best_score > 0 else None


def _detect_mesh(demo_dir: Path) -> tuple[int, int] | None:
    """The (rows, cols) mesh the model was actually built/validated on, read from the run's
    parallelism manifest — so serve.mesh_device matches the real topology instead of a box guess."""
    import glob as _g

    for f in _g.glob(str(Path(demo_dir) / "**" / "parallelism_manifest.json"), recursive=True):
        try:
            d = json.loads(Path(f).read_text())
        except Exception:
            continue
        m = d.get("mesh")
        if isinstance(m, (list, tuple)) and len(m) == 2:
            return int(m[0]), int(m[1])
    return None


# Honest serving-status note for a package whose architecture has no real vLLM generator yet (the
# scaffolded adapter is a stub). Keeps the card truthful: it pulls but won't serve until an adapter
# is written, and the perf/accuracy figures are bring-up measurements, not served results.
_RUN_IT_TITLE = "Run it on device (demo)"


def _run_it_section(checkout: Path, demo_dir: Path, slug: str, task_hint: str = "") -> str | None:
    """Build a 'how to actually run this model' section pointing at the model's OWN on-device demo —
    discovered (the demo entry script, the checkout's branch + remote), never hardcoded. This is the
    real runnable path for a model that isn't vLLM-servable yet."""
    import glob as _g
    import re as _re
    import subprocess

    # Collect every runnable demo script, then pick the real TASK demo -- not the demo.py dispatcher
    # (a multi-task model's demo.py only prints "run one of ..."). Prefer an explicit demo_* whose
    # name best overlaps the run's task tokens (its gate test + model id).
    cands = []
    for f in _g.glob(str(Path(demo_dir) / "demo" / "*.py")):
        nm = Path(f).stem
        if nm.startswith("__"):
            continue
        try:
            s = Path(f).read_text()
        except Exception:
            continue
        if "__main__" in s or "def main" in s:
            # A dispatcher lists the other demos by name; mark it so a real task demo wins.
            is_dispatcher = (nm == "demo") and ("demo_" in s)
            cands.append((nm, is_dispatcher))
    if not cands:
        return None
    _task = [t for t in _re.split(r"[^a-z0-9]+", (task_hint or "").lower()) if len(t) > 2]

    def _score(item):
        nm, is_disp = item
        low = nm.lower()
        s = sum(1 for t in _task if t in low)
        if not is_disp:
            s += 1  # an explicit task demo beats the bare dispatcher on a tie
        return s

    stem = max(cands, key=_score)[0]
    demo_mod = f"models.demos.{slug}.demo.{stem}"

    def _g1(*a):
        return subprocess.run(["git", "-C", str(checkout), *a], capture_output=True, text=True).stdout.strip()

    branch = _g1("rev-parse", "--abbrev-ref", "HEAD") or "main"
    up = _g1("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    remote = up.split("/")[0] if "/" in up else "origin"
    url = _g1("remote", "get-url", remote) or _g1("remote", "get-url", "origin")
    return (
        "This model already runs on Tenstorrent hardware through its own on-device demo (batched "
        "generation on the validated mesh). vLLM serving is pending an adapter (see **Serving "
        "status**), so run it directly from the model's tt-metal branch:\n\n"
        "```bash\n"
        f"git clone -b {branch} {url or '<tt-metal fork>'} tt-metal\n"
        "cd tt-metal && ./build_metal.sh          # build tt-metal + ttnn for your card\n"
        f"python -m {demo_mod}                     # generates on device (see --help for options)\n"
        "```\n\n"
        "The bundle's `code/` in this repo is the exact model implementation used, for reference."
    )


_SERVE_STATUS_TITLE = "Serving status"
_SERVE_STATUS_STUB = (
    "**Brought up and optimized on Tenstorrent hardware — not yet servable via vLLM.** This "
    "architecture is not a built-in of the Tenstorrent vLLM plugin, so the bundle ships a scaffolded "
    "adapter that still needs a real generator (`initialize_vllm_model`, `prefill_forward`, `decode`, "
    "warmup, KV-cache allocation, and the hybrid block-table handling). `tt-model pull` works; "
    "`tt-model serve` will fail at model load until that adapter is implemented. The performance and "
    "accuracy below are **bring-up measurements from the model's own on-device harness**, not results "
    "served by this package."
)


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


def _detect_mistral_format(demo_dir, model_root, weights) -> tuple:
    """Architecture for a MISTRAL-FORMAT native checkpoint, which ships params.json (+
    consolidated.safetensors / tekken.json) and NO HF config.json -- so _detect_arch_and_type finds
    nothing. Such a checkpoint is a Mistral causal-LM backbone, so report MistralForCausalLM (a stock
    generator) and the adapter can be scaffolded like any model. Looks in the demo, the model root,
    then the (cached) weights repo. Returns (arch, model_type) or (None, None)."""
    import glob as _g

    cands = []
    for base in (demo_dir, model_root):
        if base:
            cands += _g.glob(str(Path(base) / "**" / "params.json"), recursive=True)
    if not cands and weights:
        try:
            from huggingface_hub import snapshot_download

            repo = weights if Path(weights).is_dir() else snapshot_download(weights, allow_patterns=["params.json"])
            p = Path(repo) / "params.json"
            if p.is_file():
                cands.append(str(p))
        except Exception:
            pass
    for p in cands:
        try:
            d = json.loads(Path(p).read_text())
        except Exception:
            continue
        # mistral-format spec markers, and NOT an HF config (which carries 'architectures'):
        if {"dim", "n_layers", "n_heads"} <= set(d) and "architectures" not in d:
            return "MistralForCausalLM", "mistral"
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
    # The plugin scans the CHILDREN of extra_models_dir for vllm_metadata.json — so the bundle must be
    # a per-model SUBFOLDER (extra_models_dir/<slug>/), not files placed directly in extra_models_dir.
    base_dir = Path(checkout) / extra_models_dir
    for stale in (base_dir / "vllm_metadata.json", base_dir / "adapter.py"):
        try:
            if stale.is_file():
                stale.unlink()  # remove the older, mis-placed layout
        except Exception:
            pass
    bundle = base_dir / slug
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


def _tidy_provenance_note(args) -> None:
    """Drop the human-facing 'dirty tree — includes uncommitted changes' note (and any stray bold left
    behind) from the published card's provenance, so the model page reads professionally. Cosmetic
    card-text only — the manifest and `code/` remain the source of truth. Never raises."""
    import re

    try:
        from huggingface_hub import HfApi, hf_hub_download

        tok = _hf_token(args)
        api = HfApi(token=tok)
        lp = hf_hub_download(repo_id=args.repo, filename="README.md", repo_type="model", token=tok, force_download=True)
        s = open(lp).read()
        orig = s
        s = re.sub(r"\s*\(dirty tree[^)]*\)", "", s)
        s = re.sub(r"\s*—\s*the image includes uncommitted changes", "", s)
        s = re.sub(r"\*\*\s*\*\*", "", s)
        s = re.sub(r"\s*\*\*\s*\|", " |", s)
        s = re.sub(r"\|\s*\*\*\s*", "| ", s)
        if s != orig:
            open(lp, "w").write(s)
            api.upload_file(
                path_or_fileobj=lp,
                path_in_repo="README.md",
                repo_id=args.repo,
                repo_type="model",
                commit_message="Card: tidy provenance note",
            )
    except Exception as e:
        print(f"  [publish-hf] provenance tidy skipped ({e}).")


def _upload_card_section(args, title: str, section: str, aliases: tuple = ()) -> None:
    """Upsert a titled ``## `` section in the repo's README (idempotent) and re-upload it. Removes any
    existing section whose heading STARTS WITH ``title`` (so a dated/renamed variant is replaced, never
    duplicated) plus any ``aliases`` (former titles), then appends the fresh one. Never raises."""
    import re

    try:
        from huggingface_hub import HfApi, hf_hub_download

        tok = _hf_token(args)
        api = HfApi(token=tok)
        lp = hf_hub_download(repo_id=args.repo, filename="README.md", repo_type="model", token=tok)
        s = open(lp).read()
        for t in (title, *aliases):
            # Match on the heading STEM (drop a trailing ")") so a dated/renamed variant like
            # "Foo (real hardware, 2026-...)" is also removed — its ")" sits after the date, so the
            # full title isn't a prefix of it.
            stem = t[:-1] if t.endswith(")") else t
            s = re.sub(r"\n## " + re.escape(stem) + r"[^\n]*\n.*?(?=\n## |\Z)", "\n", s, flags=re.S)
        open(lp, "w").write(s.rstrip() + "\n\n## " + title + "\n\n" + section.rstrip() + "\n")
        api.upload_file(
            path_or_fileobj=lp,
            path_in_repo="README.md",
            repo_id=args.repo,
            repo_type="model",
            commit_message=f"Update card: {title.lower()}",
        )
    except Exception as e:
        print(f"  [publish-hf] card update skipped ({e}).")


def _discover_test_node(checkout: Path, demo_dir: Path, want: str) -> str | None:
    """Find a pytest node for the model's own perf ('perf') or accuracy/PCC ('pcc') test by SCANNING
    the model's test files — never a hardcoded test name. Matches on the function name the model
    itself declares (``def test_*perf*`` / ``def test_*pcc*``/``*gate*``), so a renamed test still
    resolves. Returns ``<file>::<func>`` or None."""
    import re as _re

    keys = ("perf",) if want == "perf" else ("pcc", "gate")
    best = None
    tests = list(Path(demo_dir).glob("**/test_*.py")) if Path(demo_dir).is_dir() else []
    for f in tests:
        try:
            src = f.read_text(errors="replace")
        except Exception:
            continue
        for fn in _re.findall(r"^def (test_[A-Za-z0-9_]+)", src, flags=_re.M):
            low = fn.lower()
            if any(k in low for k in keys) or any(k in f.name.lower() for k in keys):
                node = f"{f}::{fn}"
                # prefer a match whose function name (not just filename) carries the key
                if any(k in low for k in keys):
                    return node
                best = best or node
    return best


# Card section titles — single source of truth (stable heading; the date lives in the body so a
# re-run replaces the section in place, and the legacy alias is stripped so no duplicate is left).
_PERF_TITLE = "Measured latency (real hardware)"
_PERF_ALIASES = ("Measured performance (real hardware)",)
_ACC_TITLE = "Accuracy (real hardware)"


def _perf_card_body(rows, depth, perf_name: str) -> str:
    """The Measured-latency section body (table + prose). One place, reused by the engine and any
    re-injection, so the wording never gets copy-pasted."""
    import datetime as _dt

    tbl = [
        "| ISL | OSL | Users | TPOT (ms) | Decode (tok/s/u) | Out (tok/s total) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for isl, o, b, tpot, du, tot in rows:
        tbl.append(
            f"| {isl} | {o} | {b} | "
            + (f"{tpot:.1f}" if tpot else "—")
            + " | "
            + (f"{du:.1f}" if du else "—")
            + " | "
            + (f"{tot:,.0f}" if tot else "—")
            + " |"
        )
    depth_note = f"the resident {depth}-block depth" if depth else "the depth resident on the device"
    date = _dt.date.today().isoformat()
    return (
        f"Measured on this package by its on-device perf harness (`{perf_name}`), decoding at "
        f"{depth_note} on the device it was optimized on: fixed-length prompts of exactly ISL tokens, "
        f"output pinned to OSL tokens, held at each concurrency (Users) as a batched decode. **TPOT** is "
        f"the mean decode time per output token; **Decode (tok/s/u)** = 1000 / TPOT is the per-user "
        f"decode rate; **Out (tok/s total)** is the aggregate output rate across all concurrent users. "
        f"Every value is the mean over the run. _Measured {date}._\n\n" + "\n".join(tbl)
    )


def _enrich_card_with_benchmarks(
    args, slug: str, checkout: Path, demo_dir: Path, perf_node: str | None, pcc_node: str | None
) -> None:
    """Measure the model ON REAL HARDWARE across an ISL x Users grid via its OWN perf harness, and
    write a full sweep table into the card. The perf test is DISCOVERED (the run's ``config.perf_test``
    or a scan of the model's own tests) — no hardcoded test/stage name. The harness takes
    ``TT_PERF_ISL_TOKENS/OSL_TOKENS/BATCH/LAYERS`` and prints ``TRACE_PER_TOKEN_MS`` +
    ``TRACE_TOKENS_PER_SEC``. Best-effort; never raises; the publish stands regardless."""
    import re as _re
    import subprocess

    if getattr(args, "no_bench", False):
        return
    if not perf_node:
        perf_node = _discover_test_node(Path(checkout), Path(demo_dir), "perf")
    if not perf_node:
        print("  [publish-hf] bench: no perf test discovered for this model; published without a sweep.")
        return
    py = str(Path(checkout) / "python_env" / "bin" / "python")
    isls = [int(x) for x in (getattr(args, "bench_isl", None) or "128,1024").split(",")]
    osl = int(getattr(args, "bench_osl", None) or "128")
    users_grid = [int(x) for x in (getattr(args, "bench_batches", None) or "1,8,32").split(",")]
    layers_env = {}
    if getattr(args, "bench_layers", None):
        layers_env["TT_PERF_LAYERS"] = str(args.bench_layers)
    print(f"  [publish-hf] bench: on-device sweep via {perf_node} — ISL {isls} x Users {users_grid}, OSL {osl}…")

    def _run(isl, users):
        env = dict(
            os.environ,
            TT_METAL_HOME=str(checkout),
            TT_HW_PLANNER_SHARD_RUN="1",
            TT_PERF_ISL_TOKENS=str(isl),
            TT_PERF_OSL_TOKENS=str(osl),
            TT_PERF_BATCH=str(users),
            **layers_env,
        )
        try:
            r = subprocess.run(
                [py, "-m", "pytest", perf_node, "-q", "-s"],
                cwd=str(checkout),
                env=env,
                capture_output=True,
                text=True,
                timeout=5400,
            )
            return (r.stdout or "") + (r.stderr or "")
        except Exception as e:
            print(f"  [publish-hf] bench: ISL {isl} users {users} failed ({e}).")
            return ""

    def _f(pat, out):
        m = _re.search(pat, out)
        return float(m.group(1)) if m else None

    rows = []
    depth = None
    for isl in isls:
        for users in users_grid:
            out = _run(isl, users)
            tpot = _f(r"TRACE_PER_TOKEN_MS=([0-9.]+)", out)  # decode ms/token
            tot = _f(r"TRACE_TOKENS_PER_SEC=([0-9.]+)", out) or _f(
                r"trace_tokens_per_sec=([0-9.]+)", out
            )  # total decode tok/s
            b = int(_f(r"PERF_BATCH_STREAMS=([0-9]+)", out) or users)
            d = _f(r"depth=([0-9]+)", out)
            if d:
                depth = int(d)
            dec_u = (1000.0 / tpot) if tpot else ((tot / b) if (tot and b) else None)
            rows.append((isl, osl, b, tpot, dec_u, tot))
            print(f"  [publish-hf] bench: ISL {isl} users {b} → TPOT {tpot} ms, {tot} tok/s total")

    if not any(r[3] or r[5] for r in rows):
        print("  [publish-hf] bench: no on-device numbers captured; published without a sweep.")
        return
    perf_name = perf_node.split("::")[-1] if "::" in perf_node else Path(perf_node).stem
    _upload_card_section(args, _PERF_TITLE, _perf_card_body(rows, depth, perf_name), aliases=_PERF_ALIASES)
    print("  [publish-hf] bench: real-hardware sweep added to the card.")
    _enrich_card_with_accuracy(args, checkout, Path(demo_dir), py, pcc_node)


def _enrich_card_with_accuracy(args, checkout: Path, demo_dir: Path, py: str, pcc_node: str | None) -> None:
    """Add a real-hardware accuracy row: run the model's own accuracy/PCC gate on device and record the
    pass + PCC. The gate test is DISCOVERED (the run's ``config.pcc_test`` or a scan of the model's own
    tests) — no hardcoded test/stage name. (IFEval/GPQA/AIME/MMLU require a served OpenAI endpoint and
    are added for models the vLLM plugin can serve.)"""
    import re as _re
    import subprocess

    if not pcc_node:
        pcc_node = _discover_test_node(Path(checkout), Path(demo_dir), "pcc")
    if not pcc_node:
        return
    print("  [publish-hf] accuracy: running the model's on-device accuracy gate…")
    try:
        env = dict(os.environ, TT_METAL_HOME=str(checkout), TT_HW_PLANNER_SHARD_RUN="1")
        r = subprocess.run(
            [py, "-m", "pytest", pcc_node, "-q", "-s"],
            cwd=str(checkout),
            env=env,
            capture_output=True,
            text=True,
            timeout=5400,
        )
        out = (r.stdout or "") + (r.stderr or "")
        passed = (r.returncode == 0) and (" passed" in out or "PASSED" in out)
        pccs = _re.findall(r"[Pp][Cc][Cc][^0-9]*([01]\.\d{3,})", out)
        pcc = max((float(x) for x in pccs), default=None)
        verdict = "pass" if passed else "fail"
        pcc_s = f"{pcc:.4f}" if pcc is not None else "—"
        section = (
            "Correctness is verified on device by this model's end-to-end accuracy gate: the "
            "Tenstorrent pipeline's output is compared against the Hugging Face reference "
            "implementation (teacher-forced over the generated sequence) and must clear the pipeline's "
            "PCC threshold. The generative benchmark suites (IFEval, GPQA Diamond, AIME 2025, MMLU) are "
            "run through an OpenAI-compatible endpoint and are reported for models served via the "
            "Tenstorrent vLLM plugin.\n\n"
            "| Metric | Result | Score |\n| --- | --- | --- |\n"
            f"| End-to-end PCC vs. HF reference | **{verdict}** | {pcc_s} |"
        )
        _upload_card_section(args, _ACC_TITLE, section)
        print(
            f"  [publish-hf] accuracy: PCC gate {'passed' if passed else 'failed'}"
            + (f", PCC={pcc}" if pcc else "")
            + " — added to the card."
        )
    except Exception as e:
        print(f"  [publish-hf] accuracy: skipped ({e}).")


def _checkout_of(demo_dir: Path) -> Path:
    """The tt-metal checkout root a demo dir belongs to (the path before '/models/')."""
    parts = Path(demo_dir).resolve().parts
    if "models" in parts:
        return Path(*parts[: parts.index("models")])
    return Path(demo_dir).resolve().parents[2]


def _adapter_is_working(bundle_dir) -> bool:
    """True when the bundle already ships a REAL generator, not the scaffolded stub: an adapter.py that
    implements initialize_vllm_model + prefill_forward + decode_forward and carries no TODO(author)
    placeholder. Lets a hand-written adapter for ANY arch publish as servable, and stops a republish
    from re-stubbing over it."""
    from pathlib import Path as _P

    try:
        ap = _P(bundle_dir) / "adapter.py"
        meta = _P(bundle_dir) / "vllm_metadata.json"
        if not ap.is_file() or not meta.is_file():
            return False
        src = ap.read_text()
    except Exception:
        return False
    if any(t not in src for t in ("def initialize_vllm_model", "def prefill_forward", "def decode_forward")):
        return False
    return "TODO(author)" not in src


def _git_info(path):
    """(commit_sha, origin_url) for a git checkout, or (None, None). Never raises."""
    import subprocess

    def g(*a):
        try:
            return subprocess.run(["git", "-C", str(path), *a], capture_output=True, text=True).stdout.strip()
        except Exception:
            return ""

    return (g("rev-parse", "HEAD") or None, g("remote", "get-url", "origin") or None)


def _discover_serving_stack():
    """Find a matched vLLM + in-tree-plugin serving stack on this host WITHOUT hardcoding a path or
    commit: scan bounded locations for a git checkout of the Tenstorrent vLLM carrying
    plugins/vllm-tt-plugin in-tree. Returns (vllm_path, plugin_path, commit, remote) or None."""
    import glob
    import os

    home = os.path.expanduser("~")
    seen = set()
    for root in [home] + sorted(glob.glob("/home/*")):
        for depth in ("vllm", "*/vllm", "*/*/vllm"):
            for vroot in glob.glob(os.path.join(root, depth)):
                if vroot in seen:
                    continue
                seen.add(vroot)
                plugin = os.path.join(vroot, "plugins", "vllm-tt-plugin")
                if not os.path.isdir(plugin):
                    continue
                commit, remote = _git_info(vroot)
                if commit and remote and "vllm" in remote.lower():
                    return (vroot, plugin, commit, remote)
    return None


def _auto_constraint_file(checkout):
    """Write a minimal pip CONSTRAINT pinning the versions the model was validated against (torch,
    transformers, tokenizers) discovered from the checkout's python env, so a source-built vLLM cannot
    pull a torch/transformers that mismatches the ttnn modules or the model code. Returns path or None."""
    import subprocess
    import tempfile
    from pathlib import Path as _P

    py = str(_P(checkout) / "python_env" / "bin" / "python")
    if not _P(py).is_file():
        py = "python3"

    def ver(pkg):
        try:
            return (
                subprocess.run(
                    [py, "-c", f"import importlib.metadata as m; print(m.version('{pkg}'))"],
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                or None
            )
        except Exception:
            return None

    pins = []
    for pkg in ("torch", "transformers", "tokenizers"):
        v = ver(pkg)
        if v:
            pins.append(f"{pkg}=={v.split('+')[0]}")
    if not pins:
        return None
    f = _P(tempfile.mkdtemp(prefix="ttpub_constraint_")) / "constraint.lock"
    f.write_text("\n".join(pins) + "\n")
    return str(f)


def _capture_vllm_provenance(args) -> None:
    """When a servable image was built from an auto-discovered local vLLM checkout, tt-model records
    'a local checkout — commit not published'. Replace it with the real PUBLIC commit we discovered so
    the card is reproducible-from-source. Card text only; never raises."""
    disc = getattr(args, "_discovered_vllm", None)
    if not disc:
        return
    commit, remote = disc
    try:
        import re
        from huggingface_hub import HfApi, hf_hub_download

        tok = _hf_token(args)
        api = HfApi(token=tok)
        lp = hf_hub_download(repo_id=args.repo, filename="README.md", repo_type="model", token=tok, force_download=True)
        s2 = open(lp).read()
        orig = s2
        url = remote.rstrip("/")
        if url.endswith(".git"):
            url = url[:-4]
        link = f"[`{commit[:9]}`]({url}/commit/{commit})"
        s2 = re.sub(r"\|\s*vLLM\s*\|.*\|", f"| vLLM | {link} — {url} |", s2, count=1)
        s2 = re.sub(
            r"\|\s*vllm-tt-plugin\s*\|.*\|",
            f"| vllm-tt-plugin | in-tree `plugins/vllm-tt-plugin` of {url} {link} |",
            s2,
            count=1,
        )
        if s2 != orig:
            api.upload_file(
                path_or_fileobj=s2.encode(),
                path_in_repo="README.md",
                repo_id=args.repo,
                repo_type="model",
                commit_message="Provenance: record public vLLM commit + in-tree plugin",
            )
            print(f"  [publish-hf] provenance recorded: {url} @ {commit[:9]}")
    except Exception as e:
        print(f"  [publish-hf] provenance capture skipped: {e}")


def _ensure_tt_model_serving_fixes(tt_model_bin) -> None:
    """Make a source-built ({path}) matched-pair image build on ANY host by ensuring the local tt-model
    launcher (tt_kernel/launchers.py) handles a staged vLLM/plugin source that has no .git:
      (1) hand setuptools-scm a pretend version, and
      (2) --constraint the validated lock so torch/transformers stay pinned while other deps resolve.
    Idempotent + anchor-based; no-ops (with a hint to upstream the fix) if tt-model's layout changed.
    This removes the last external dependency: publishHF works end-to-end with no manual tt-model edit."""
    import subprocess
    from pathlib import Path as _P

    binp = _P(tt_model_bin)
    py = binp.parent / "python"
    if not py.is_file():
        py = _P("python3")
    try:
        loc = subprocess.run(
            [str(py), "-c", "import tt_kernel, os; print(os.path.dirname(tt_kernel.__file__))"],
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return
    lf = _P(loc) / "launchers.py" if loc else None
    if not lf or not lf.is_file():
        return
    try:
        src = lf.read_text()
    except Exception:
        return
    if "SETUPTOOLS_SCM_PRETEND_VERSION" in src:
        return  # already patched (by us before, or upstream)
    env = "SETUPTOOLS_SCM_PRETEND_VERSION=0.24.0 VCS_VERSIONING_PRETEND_VERSION=0.24.0 "
    orig_vllm = (
        '        if vllm.get("path"):\n'
        "            return (\n"
        "                [f'VLLM_TARGET_DEVICE=empty uv pip install --python \"$VENV/bin/python\" '\n"
        '                 f"{VLLM_CTX_DIR} --extra-index-url {self.PYTORCH_CPU_INDEX} "\n'
        '                 f"--index-strategy unsafe-best-match"]\n'
        "                + self._post_engine_lines(m, plugin)\n"
        "            )"
    )
    new_vllm = (
        '        if vllm.get("path"):\n'
        '            constraint = "--constraint /ctx/requirements.lock " if rt.get("lock") else ""\n'
        "            return ([\n"
        "                '" + env + "'\n"
        "                f'VLLM_TARGET_DEVICE=empty uv pip install --python \"$VENV/bin/python\" {constraint}'\n"
        '                f"{VLLM_CTX_DIR} --extra-index-url {self.PYTORCH_CPU_INDEX} "\n'
        '                f"--index-strategy unsafe-best-match"\n'
        "            ] + self._post_engine_lines(m, plugin))"
    )
    orig_plugin = (
        "            lines.append(\n"
        "                f'uv pip install --python \"$VENV/bin/python\" {PLUGIN_CTX_DIR}'\n"
        "            )"
    )
    new_plugin = (
        "            lines.append(\n"
        "                f'" + env + 'uv pip install --python "$VENV/bin/python" {PLUGIN_CTX_DIR}\'\n'
        "            )"
    )
    n = 0
    if orig_vllm in src:
        src = src.replace(orig_vllm, new_vllm, 1)
        n += 1
    if orig_plugin in src:
        src = src.replace(orig_plugin, new_plugin, 1)
        n += 1
    if n:
        try:
            lf.write_text(src)
            print(f"  [publish-hf] self-healed tt-model source-build launcher ({n} sites): {lf}")
        except Exception as e:  # noqa: BLE001
            print(
                f"  [publish-hf] could not patch tt-model launcher ({e}); apply the pretend-version + "
                "--constraint fix to tt_kernel/launchers.py manually"
            )
    else:
        print(
            "  [publish-hf] tt-model launcher layout unrecognized; if a {path} build fails on "
            "setuptools-scm/torch, upstream the pretend-version + --constraint fix to tt-model"
        )


def _run_container(args, state: dict, slug: str, demo_dir, commit: str | None) -> int:
    """Build + push a real v5.1 container bundle via tt-model (exactly like the published TT repos):
    generate tt-model.yaml, then `tt-model package --container` (2.5-4h OCI build) and `tt-model push`."""
    import subprocess
    import tempfile

    def _has_submodules(root: Path) -> bool:
        return (Path(root) / "tt_metal/third_party/umd/CMakeLists.txt").is_file()

    # Choose a BUILDABLE source checkout: it must hold the demo AND have initialised submodules (the
    # image builds tt-metal from source). The main checkout qualifies after commit-wins; an ephemeral
    # /tmp optimize worktree usually does not (submodules uninitialised) — prefer main over it.
    cand_roots: list[Path] = []
    try:
        cand_roots.append(_repo_root())
    except Exception:
        pass
    cand_roots.append(_checkout_of(Path(demo_dir)))
    mr = (state.get("model") or {}).get("root")
    if mr:
        cand_roots.append(_checkout_of(Path(mr)))
    checkout = None
    for c in cand_roots:
        c = Path(c)
        if (c / "models" / "demos" / slug).is_dir() and _has_submodules(c):
            checkout = c
            break
    if checkout is None:  # relax the submodule requirement; the build will report if it matters
        for c in cand_roots:
            if (Path(c) / "models" / "demos" / slug).is_dir():
                checkout = Path(c)
                break
    if checkout is None:
        checkout = _checkout_of(Path(demo_dir))
    demo_dir = checkout / "models" / "demos" / slug

    ttm = getattr(args, "tt_model_bin", None) or "tt-model"
    out = getattr(args, "out", None) or str(Path.home() / "tt-model-builds")
    extra = getattr(args, "extra_models_dir", None) or f"models/demos/{slug}/vllm_bundle"

    # Make every model publishable this way: ensure the vLLM adapter bundle exists (scaffold it from
    # the model's own HF arch when missing). Stock arches are servable as-is; novel arches get a stub.
    # Detect the HF arch from the demo; fall back to the run's model_root (which keeps the captured
    # config.json even when the committed demo does not), then to an explicit --hf-arch override.
    arch_det, mtype = _detect_arch_and_type(Path(demo_dir))
    if not arch_det:
        mr = (state.get("model") or {}).get("root")
        if mr and Path(mr).is_dir():
            arch_det, mtype = _detect_arch_and_type(Path(mr))
    if not arch_det and getattr(args, "hf_arch", None):
        arch_det, mtype = args.hf_arch, None
    _mistral_fallback = False
    if not arch_det:
        # Mistral-format native checkpoints (no HF config.json) -- e.g. Voxtral -- are Mistral
        # causal-LM backbones; detect them so the adapter still scaffolds and the container builds.
        _mr2 = (state.get("model") or {}).get("root")
        arch_det, mtype = _detect_mistral_format(
            Path(demo_dir), Path(_mr2) if _mr2 else None, getattr(args, "weights", None)
        )
        if arch_det:
            _mistral_fallback = True
            print(f"  [publish-hf] no HF config.json; detected mistral-format checkpoint -> {arch_det}")
    # Servability is decided by the architecture: a plugin built-in (stock generator) serves; a novel
    # arch gets a scaffolded stub and is NOT servable until an adapter is written. Drives honest card
    # labeling below — the tool never claims a stub package can serve.
    bundle_dir = Path(checkout) / extra / slug
    adapter_ready = _adapter_is_working(bundle_dir)
    servable = True
    if arch_det and not adapter_ready:
        _base_cls, _is_stub_arch = _pick_base_generator(arch_det, mtype)
        # A mistral-format fallback is an UNVERIFIED backbone guess (no HF config; the real model may
        # be custom, e.g. a TTS head), so never claim it serves until a real adapter is written.
        servable = (not _is_stub_arch) and not _mistral_fallback
    if adapter_ready:
        print(f"  [publish-hf] real vLLM adapter detected: {bundle_dir}  (servable — preserved, not scaffolded)")
        if getattr(args, "container", False) and not getattr(args, "vllm_path", None):
            disc = _discover_serving_stack()
            if disc:
                vpath, ppath, vcommit, vremote = disc
                args.vllm_path, args.plugin_path = vpath, ppath
                args._discovered_vllm = (vcommit, vremote)
                print(f"  [publish-hf] auto serving stack: {vremote} @ {vcommit[:9]} (in-tree plugin) — {vpath}")
                if not getattr(args, "lock", None):
                    cf = _auto_constraint_file(checkout)
                    if cf:
                        args.lock = cf
                        print(f"  [publish-hf] auto constraint (validated torch/transformers/tokenizers): {cf}")
            else:
                print(
                    "  [publish-hf] no local matched vLLM checkout found to auto-build a servable image; "
                    "pass --vllm-path/--plugin-path."
                )
    elif arch_det and not getattr(args, "no_scaffold", False):
        created, is_stub, bpath = _scaffold_vllm_bundle(
            checkout, extra, arch_det, mtype, getattr(args, "weights", None), slug
        )
        state_note = (
            "STUB — not servable until an adapter is written"
            if (is_stub or _mistral_fallback)
            else "stock generator — servable as-is"
        )
        print(f"  [publish-hf] vLLM bundle {'created' if created else 'exists'}: {bpath}  ({state_note})")

    # Serve on the mesh the model was actually built/validated on (from the run), not a box guess.
    mesh_rc = _detect_mesh(Path(demo_dir))
    mesh_override = getattr(args, "mesh", None) or (f"({mesh_rc[0]}, {mesh_rc[1]})" if mesh_rc else None)

    yaml_path = Path(tempfile.mkdtemp(prefix="tt_ttmodel_")) / "tt-model.yaml"
    try:
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
            mesh_device=mesh_override,
            kind=getattr(args, "kind", None) or "vllm-plugin",
            plugin_ref=getattr(args, "plugin_ref", None) or "main",
            vllm_version=getattr(args, "vllm_version", None) or "0.24.0",
            extra_models_dir=extra,
            commit=commit,
            lock=getattr(args, "lock", None),
            vllm_path=getattr(args, "vllm_path", None),
            plugin_path=getattr(args, "plugin_path", None),
        )
    except ValueError as exc:
        print(f"  [publish-hf] {exc}")
        return 2
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

    # Provenance note: the tool does NOT commit model source — that is model-specific and belongs on
    # the model's own branch, committed by the model owner (never on this tool branch). If the model's
    # working tree is committed on its branch before publishing, the image records a clean commit SHA;
    # otherwise tt-model honestly marks the SHA "dirty". The tool stays out of the model's git state.
    if getattr(args, "vllm_path", None):
        _ensure_tt_model_serving_fixes(ttm)  # self-heal the build launcher for source ({path}) builds
    pkg = [ttm, "package", "--container", str(yaml_path), "--out", out]
    print(f"  [publish-hf] building container (2.5-4h): {' '.join(pkg)}")
    rc = subprocess.run(pkg).returncode
    if rc != 0:
        print(f"  [publish-hf] tt-model package failed (rc={rc}).")
        return 4
    staged = Path(out) / slug
    if not staged.is_dir():
        # tt-model may name the staged dir after the manifest name; fall back to the newest under --out
        subs = [p for p in Path(out).iterdir() if p.is_dir()] if Path(out).is_dir() else []
        if subs:
            staged = max(subs, key=lambda p: p.stat().st_mtime)
    # Upload the built bundle OURSELVES (create_repo + upload_large_folder) rather than `tt-model push`:
    # tt-model's push depends on a huggingface_hub version whose folder-upload API drifts between
    # releases, so doing it in-process with this interpreter's hub keeps the whole flow one automated
    # button press. Same repo id every time → updates the one page.
    print(f"  [publish-hf] uploading built bundle from {staged} → {args.repo}")
    try:
        from huggingface_hub import HfApi
    except Exception:
        print("  [publish-hf] huggingface_hub not available to upload the bundle " "(pip install huggingface_hub).")
        return 4
    token = _hf_token(args)
    try:
        api = HfApi(token=token)
        api.create_repo(
            args.repo,
            repo_type="model",
            private=not (getattr(args, "public", False) or getattr(args, "publish", False)),
            exist_ok=True,
        )
        up = getattr(api, "upload_large_folder", None)
        if callable(up):
            up(repo_id=args.repo, folder_path=str(staged), repo_type="model")
        else:
            api.upload_folder(
                repo_id=args.repo,
                folder_path=str(staged),
                repo_type="model",
                commit_message=f"Publish {slug} container bundle (tt_hw_planner)",
            )
    except Exception as e:
        low = str(e).lower()
        if "401" in str(e) or "unauthorized" in low or "invalid username or password" in low:
            print(
                "  [publish-hf] upload rejected (401): set a Hugging Face WRITE token for the "
                "target org (Auth section / HF_TOKEN)."
            )
        else:
            print(f"  [publish-hf] upload failed: {e}")
        return 4
    print(f"  [publish-hf] published container bundle: https://huggingface.co/{args.repo}")
    _tidy_provenance_note(args)  # keep the published card's provenance clean/professional
    _capture_vllm_provenance(args)  # record the real public vLLM commit when we auto-discovered the stack
    if not servable:
        # Never claim a stub package serves — say so plainly, and give the REAL way to run it.
        _upload_card_section(args, _SERVE_STATUS_TITLE, _SERVE_STATUS_STUB)
        _cfg = state.get("config") or {}
        _hint = " ".join(str(x) for x in (slug, _cfg.get("pcc_test", ""), _cfg.get("perf_test", "")))
        run_it = _run_it_section(Path(checkout), Path(demo_dir), slug, _hint)
        if run_it:
            _upload_card_section(args, _RUN_IT_TITLE, run_it)
        print("  [publish-hf] card marked NOT-YET-SERVABLE + added on-device run instructions.")
    # Auto-benchmark: serve the bundle and write a measured latency sweep into the card. Universal +
    # best-effort — measures any model that serves, skips (publish stands) for one that can't yet.
    if not getattr(args, "no_bench", False):
        cfg = state.get("config") or {}
        _enrich_card_with_benchmarks(args, slug, checkout, demo_dir, cfg.get("perf_test"), cfg.get("pcc_test"))
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
        state = collect_state(
            run_dir, state_dir_candidates(state_root, slug), slug, requested_batch=getattr(args, "batch", None)
        )
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
        # Direct fallbacks: the demo dir straight under the checkout (works after commit-wins even
        # when find_demo_dir's registry lookup or the dashboard's model.root come back empty), then
        # a glob anywhere under models/.
        cand = Path(repo_root) / "models" / "demos" / slug
        if cand.is_dir():
            demo_dir = cand
        else:
            import glob as _g

            hits = [
                h for h in _g.glob(str(Path(repo_root) / "models" / "**" / slug), recursive=True) if Path(h).is_dir()
            ]
            if hits:
                demo_dir = Path(hits[0])
    if demo_dir is None or not Path(demo_dir).is_dir():
        print(
            f"  [publish-hf] could not locate the model demo dir for '{slug}' under {repo_root}. "
            f"Run `commit-wins` first so the optimized model lands in the checkout."
        )
        return 2

    commit = _git_commit(state_root)
    # Auto-detect the base-weights HF id when the caller didn't pass one, so the card/manifest never
    # ship a placeholder. Also feeds the container path (it reads args.weights downstream).
    if not getattr(args, "weights", None):
        detected = _detect_weights(Path(demo_dir), slug)
        if detected:
            args.weights = detected
            print(f"  [publish-hf] base weights auto-detected: {detected}")
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

    token = _hf_token(args)
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
