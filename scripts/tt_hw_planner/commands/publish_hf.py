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


# Chips per board, keyed by the base of a serve.hardware label ("p300x2" = two p300 boards = 4 chips).
# Mirrors tt-model's _BOARD_CHIPS: its manifest check refuses a mesh_device that opens a different
# number of chips than the hardware label implies.
_BOARD_CHIPS = {"p100": 1, "p150": 1, "n150": 1, "p300": 2, "n300": 2}
_SINGLE_CHIP_BOARD = {"blackhole": "p150", "wormhole_b0": "n150"}


def _hardware_for_chips(hardware: str, arch: str | None, chips: int) -> str:
    """The serve.hardware label for a model that opens `chips` chips of the box `hardware` names: the
    box's own board when the chips fill whole boards (2 chips of a p300x2 -> p300), else the arch's
    single-chip board (1 chip of a p300x2 -> p150). Unchanged when the label already matches or is not
    a <board>[xN] label."""
    import re as _re

    m = _re.fullmatch(r"([a-z]+\d+)[a-z]?(?:x(\d+))?", str(hardware).lower())
    if not m or m.group(1) not in _BOARD_CHIPS or chips < 1:
        return hardware
    base, per = m.group(1), _BOARD_CHIPS[m.group(1)]
    if per * int(m.group(2) or 1) == chips:
        return hardware
    if chips % per:
        single = _SINGLE_CHIP_BOARD.get(str(arch or "").lower())
        if not single:
            return hardware
        base, per = single, 1
    n = chips // per
    return base if n == 1 else f"{base}x{n}"


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
    serve_hf_override = None
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
            _ovr = _md.get("serve_hf_override_arch")
            if isinstance(_ovr, str) and _ovr:
                serve_hf_override = _ovr
    except Exception:
        pass
    arch = arch or a
    # A novel-arch adapter is not a vLLM/Transformers built-in, so the engine must be told which
    # registered TT class to use (there is no built-in to ride); stock arches omit this.
    _serve_args = ["--trust-remote-code"]
    if serve_hf_override:
        _serve_args += ["--hf-overrides", json.dumps({"architectures": [serve_hf_override]})]
    hardware_given = hardware
    hardware = hardware or hw
    mesh_device = mesh_device or mesh
    if not (arch and hardware and mesh_device):
        raise ValueError(
            "no serve target: the run's detected hardware (%s) is no single known box and no --box / "
            "--arch / --hardware / --mesh was given" % (state.get("env") or {}).get("arch")
        )
    _validate_mesh_device(mesh_device)
    # A run that opened fewer chips than its box (1 chip of a QB2) serves on that smaller mesh, and
    # tt-model refuses a hardware label naming more chips than the mesh opens. So the label names the
    # chips opened, and the card says which box they are on.
    import re as _re

    hw_phrase, perf_hw = _hardware_phrase(state, hardware), hardware
    _mt = _re.fullmatch(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", str(mesh_device))
    if _mt and not hardware_given:
        opened = int(_mt.group(1)) * int(_mt.group(2))
        fitted = _hardware_for_chips(hardware, arch, opened)
        if fitted != hardware:
            unit = "chip" if opened == 1 else "chips"
            hw_phrase = f"{opened} Tenstorrent {_arch_name({'env': {'arch': arch}})} {unit} of a {hardware}"
            perf_hw = f"{opened} {unit} of a {hardware}"
            hardware = fitted
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
        perf_bits.append(f"on {perf_hw}")
    perf = " ".join(perf_bits) or "measured on Tenstorrent; see the dashboard."
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
        f"  args: {json.dumps(_serve_args)}",
        "",
        "verify:",
        f'  - "import models.demos.{slug} as m; assert m"',
        "",
        "card:",
        f"  description: >",
        f"    {slug}, brought up and optimized on {hw_phrase}",
        f"    (kernel/lever autotuning against a PCC-gated end-to-end pipeline).",
        f"  performance: >",
        f"    {perf}.",
        f"  limitations: >",
        f"    Validated on {hw_phrase}.",
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


def _device_count(state: dict) -> int:
    try:
        return int((state.get("env") or {}).get("device_count") or 0)
    except (TypeError, ValueError):
        return 0


def _arch_name(state: dict) -> str:
    a = str((state.get("env") or {}).get("arch") or "").strip().lower()
    if "blackhole" in a:
        return "Blackhole"
    if "wormhole" in a:
        return "Wormhole"
    return a or "Tenstorrent"


def _ensure_hw_env(state: dict, demo_dir=None, box: str | None = None) -> dict:
    """Fill state.env with the real hardware when the run env didn't record it: device_count from the
    run's parallelism manifest (chips, else mesh product), arch from the operator's --box. Best-effort
    and idempotent, so every card the publish builds can name the chips automatically."""
    env = dict(state.get("env") or {})
    if not env.get("device_count") and demo_dir:
        try:
            import glob as _g

            for f in _g.glob(str(Path(demo_dir) / "**" / "parallelism_manifest.json"), recursive=True):
                d = json.loads(Path(f).read_text())
                chips = d.get("chips")
                if not chips and isinstance(d.get("mesh"), (list, tuple)) and len(d["mesh"]) == 2:
                    chips = int(d["mesh"][0]) * int(d["mesh"][1])
                if chips:
                    env["device_count"] = int(chips)
                    break
        except Exception:  # noqa: BLE001
            pass
    if not env.get("arch") and box:
        tgt = _BOX_TARGET.get(box)
        if tgt:
            env["arch"] = tgt[0]
    state = dict(state)
    state["env"] = env
    return state


def _hardware_phrase(state: dict, hardware: str | None) -> str:
    """The hardware the run ACTUALLY used, in plain words. One chip reads as one chip — never the
    whole box — because the packaging ``hardware`` tag (e.g. ``p300x2``) names the box, not the number
    of chips the model opened. ``device_count`` comes from the run's own manifest env."""
    arch = _arch_name(state)
    dc = _device_count(state)
    # Prefer the packaging label (e.g. "p300x2") when known -- it names the exact Tenstorrent box and
    # is how the other model packages describe their hardware. Fall back to a chip count only when no
    # package tag is available.
    if hardware:
        return f"Tenstorrent {hardware}"
    if dc == 1:
        return "a single Tenstorrent chip" if arch == "Tenstorrent" else f"a single Tenstorrent {arch} chip"
    if dc > 1:
        return f"{dc} Tenstorrent chips" if arch == "Tenstorrent" else f"{dc} Tenstorrent {arch} chips"
    return "Tenstorrent" if arch == "Tenstorrent" else f"Tenstorrent {arch}"


def _fetch_upstream_card(base_weights: str, token: str | None) -> str | None:
    """The upstream model's README body (front-matter + leading H1 stripped), or None if unavailable.
    Lets a published card MIRROR the upstream HF model card the port is based on."""
    try:
        from huggingface_hub import hf_hub_download

        p = hf_hub_download(repo_id=base_weights, filename="README.md", repo_type="model", token=token)
        txt = open(p, encoding="utf-8", errors="replace").read()
    except Exception:
        return None
    if txt.startswith("---"):
        parts = txt.split("\n---\n", 1)
        if len(parts) == 2:
            txt = parts[1]
    import re as _re

    txt = _re.sub(r"\A\s*#\s[^\n]*\n", "", txt, count=1)  # drop leading H1; we add a TT title
    return txt.strip() or None


def _tt_validated_block(state: dict, hw_phrase: str, commit: str | None) -> str:
    """The ONE section whose every number is measured on Tenstorrent in this run."""
    cfg = state.get("config") or {}
    thr = state.get("throughput") or {}
    sv = state.get("serving") or {}
    m = state.get("metric") or {}
    pt = sv.get("per_token") or {}
    ft = sv.get("first_token") or {}
    batch = state.get("batch")
    rows = [f"| Hardware | {hw_phrase} |"]
    if batch is not None:
        rows.append(f"| Concurrency (batch) | {batch} |")
    if thr.get("current") is not None:
        rows.append(f"| Decode throughput | {_fmt(thr['current'])} tok/s/user |")
    if ft.get("ms") is not None:
        rows.append(f"| TTFT | {_fmt(ft['ms'], 1)} ms |")
    if pt.get("ms") is not None:
        rows.append(f"| TPOT / ITL | {_fmt(pt['ms'], 1)} ms |")
    if m.get("name") and m.get("current") is not None:
        rows.append(f"| {m['name']} | {_fmt(m.get('current'), 1)} {m.get('unit', '')} |")
    if cfg.get("pcc_test"):
        rows.append(f"| Accuracy gate | `{str(cfg['pcc_test']).split('::')[-1]}` (PCC) |")
    if commit:
        rows.append(f"| tt-metal commit | `{commit}` |")
    table = "\n".join(["| Measured on Tenstorrent | Value |", "| --- | --- |", *rows])
    return (
        "## Validated on Tenstorrent\n\n"
        f"Brought up and measured on {hw_phrase} through a PCC-gated end-to-end pipeline. "
        "Every figure in this section is measured on Tenstorrent hardware; the sections below are the "
        "upstream base-model card (measured by the base-model authors on GPU), reproduced for reference.\n\n"
        + table
        + "\n"
    )


_GPU_TITLE_KEYS = (
    "vllm",
    "sglang",
    "tensorrt",
    "trt-llm",
    "gpu",
    "cuda",
    "nvidia",
    "deploy",
    "deployment",
    "quick start",
    "quickstart",
    "serving",
    "serve",
    "run with",
    "run on",
    "installation",
    "docker",
)


def _strip_gpu_sections(body: str) -> str:
    """Drop GPU-specific deployment/usage/hardware sections from the upstream card body (keeps model
    description, architecture, training, datasets, benchmarks). Conservative + best-effort: a section
    is dropped only when its ## title is clearly a GPU/deployment one, or its body is dominated by GPU
    serve commands (vllm serve / sglang / cuda / nvidia docker)."""
    import re

    parts = re.split(r"(?m)^(?=##\s)", body)
    kept = []
    for p in parts:
        m = re.match(r"##\s+(.*)", p)
        title = m.group(1).strip().lower() if m else ""
        low = p.lower()
        gpu_title = bool(title) and any(k in title for k in _GPU_TITLE_KEYS)
        gpu_body = (
            "vllm serve" in low
            or "sglang" in low
            or ("docker run" in low and "nvidia" in low)
            or ("--tensor-parallel" in low)
        )
        if gpu_title or gpu_body:
            continue
        kept.append(p)
    out = "".join(kept).strip()
    return out or body  # never return empty (fall back to the whole body)


def _load_tt_eval(demo_dir) -> dict:
    """On-device accuracy results written by a bounded TT eval, if any:
    {"<task>": {"tt": <score>, "n": <sampled>}}. Empty when none have been run yet."""
    from pathlib import Path

    for name in ("tt_eval.json", "tt_accuracy.json"):
        p = Path(demo_dir) / name
        if p.is_file():
            try:
                d = json.load(open(p))
                return d if isinstance(d, dict) else {}
            except Exception:
                return {}
    return {}


def _augment_benchmark_tables(body: str, base_weights: str, tt_results: dict) -> str:
    """Add a Tenstorrent column to the upstream BENCHMARK tables — the ones whose header names the
    model (base/other models compared). Each row's TT cell is the measured on-device score for that
    task (from tt_results) or 'pending (TT)'. Dataset / hardware tables (header has no model name) are
    left untouched. Best-effort: on any parse trouble, the table is returned unchanged."""
    import re

    model_tokens = [t for t in re.split(r"[^A-Za-z0-9]+", (base_weights or "").split("/")[-1].lower()) if len(t) > 2]

    def _is_benchmark_header(header: str) -> bool:
        h = header.lower()
        if any(t in h for t in model_tokens):
            return True
        return any(k in h for k in ("benchmark", "task", "metric", "eval", "accuracy", "score"))

    score_re = re.compile(r"^\**\s*~?\s*\d{1,3}(?:\.\d+)?\s*%?\s*\**$")

    def _rows_mostly_numeric(rows):
        data = [r for r in rows if "|" in r]
        if not data:
            return False
        hits = 0
        for r in data:
            cells = [c.strip() for c in r.strip().strip("|").split("|")]
            if any(score_re.match(c) for c in cells[1:]):
                hits += 1
        return hits >= 2 and hits >= 0.5 * len(data)

    def repl(m):
        tbl = m.group(0)
        lines = [ln for ln in tbl.split("\n") if ln.strip()]
        if len(lines) < 2 or not re.match(r"^\s*\|?[\s:|-]+\|?\s*$", lines[1]):
            return tbl
        if not (_is_benchmark_header(lines[0]) and _rows_mostly_numeric(lines[2:])):
            return tbl
        if "tenstorrent" in lines[0].lower():
            return tbl
        new = [lines[0].rstrip().rstrip("|") + " | Tenstorrent |", lines[1].rstrip().rstrip("|") + " | --- |"]
        for ln in lines[2:]:
            if "|" not in ln:
                new.append(ln)
                continue
            task = ln.strip().strip("|").split("|")[0].strip().strip("*").strip()
            res = (tt_results or {}).get(task)
            if isinstance(res, dict) and res.get("tt") is not None:
                cell = f"{res['tt']}" + (f" (n={res['n']})" if res.get("n") else "")
            else:
                cell = "pending (TT)"
            new.append(ln.rstrip().rstrip("|") + f" | {cell} |")
        return "\n".join(new) + "\n"

    return re.sub(r"(?m)^(\|.*\|[ \t]*\n\|[\s:|-]+\|[ \t]*\n(?:\|.*\|[ \t]*\n?)+)", repl, body)


_GPU_RUN_ROW_RE = __import__("re").compile(
    r"\b(A100|H100|H200|B100|B200|B300|GB200|HGX|SXM|NVIDIA|CUDA|cuda|ROCm|vllm serve|sglang|TensorRT)\b"
)


def _scrub_gpu_run_rows(body: str) -> str:
    """Drop lines that describe how to RUN on GPU — GPU hardware-requirement / compute / serve-command
    rows (A100/H100/B200/NVIDIA/CUDA/vllm serve/sglang/...). Benchmark SCORE rows (no GPU-hardware
    token) are kept, so the GPU numbers remain as the comparison column next to Tenstorrent. Prose
    lines mentioning a GPU serve command are dropped too; plain prose is left alone."""
    out = []
    for ln in body.splitlines():
        st = ln.strip()
        is_table_row = st.startswith("|")
        if _GPU_RUN_ROW_RE.search(ln) and (is_table_row or "vllm serve" in ln.lower() or "sglang" in ln.lower()):
            continue
        out.append(ln)
    return "\n".join(out)


def _mirror_body_tt(body: str, base_weights: str, demo_dir, state: dict) -> str:
    """Transform the upstream body for a TT card: strip GPU-specific sections and turn benchmark
    tables into TT-vs-GPU comparisons. Fully best-effort — any error returns the body unchanged so
    publishing never breaks."""
    try:
        tt_results = _load_tt_eval(demo_dir) if demo_dir else {}
        b = _strip_gpu_sections(body)
        b = _scrub_gpu_run_rows(b)
        b = _augment_benchmark_tables(b, base_weights, tt_results)
        return b
    except Exception:  # noqa: BLE001
        return body


def _tt_run_section(repo_id: str | None, base_weights: str | None = None) -> str:
    """How to actually serve this model on Tenstorrent: the `tt` CLI (Tenstorrent's packaged serving
    flow) pulls the published package and serves it as an OpenAI-compatible API, then any OpenAI
    client can query it. This is the same path the other Tenstorrent model packages use."""
    rid = repo_id or "<your-published-repo>"
    model = base_weights or rid
    return (
        "## Run on Tenstorrent\n\n"
        "```bash\n"
        "uv tool install tenstorrent          # once \u2014 the Tenstorrent CLI, `tt`\n"
        f"tt model pull {rid}\n"
        f"tt serve {rid}                       # OpenAI-compatible API on http://localhost:20000/v1\n"
        "```\n\n"
        "Then query it with any OpenAI client:\n\n"
        "```python\n"
        "from openai import OpenAI\n"
        'client = OpenAI(base_url="http://localhost:20000/v1", api_key="EMPTY")\n'
        "resp = client.chat.completions.create(\n"
        f'    model="{model}",\n'
        '    messages=[{"role": "user", "content": "What is the capital of Japan?"}],\n'
        "    max_tokens=256,\n"
        ")\n"
        "print(resp.choices[0].message.content)\n"
        "```\n"
    )


def _tt_accuracy_section(tt_results: dict) -> str:
    """Render the on-TT inference accuracy table from tt_eval.json. Shows a GPU reference column when
    the eval recorded one (key 'gpu'), else the TT-measured numbers only. Empty when no eval ran."""
    if not tt_results:
        return ""
    has_gpu = any(isinstance(v, dict) and v.get("gpu") is not None for v in tt_results.values())
    out = [
        "## Inference accuracy -- measured on Tenstorrent",
        "",
        "Bounded inference benchmarks run through the live `tt serve` endpoint on Tenstorrent (greedy "
        "decoding; zero-shot / direct answer, step-by-step reasoning for GSM8K). A confirmation sample "
        "over small N -- not the base-model authors' full eval-framework (few-shot / chain-of-thought) "
        "-- so treat these as a sanity check that the model's inference quality holds on device, not a "
        "leaderboard reproduction.",
        "",
    ]
    if has_gpu:
        out += ["| Benchmark | GPU (reference) | Tenstorrent (measured) | Sample |", "| --- | --- | --- | --- |"]
        for k, v in tt_results.items():
            if isinstance(v, dict):
                out.append(f"| {k} | {v.get('gpu', '-')} | {v.get('tt', '-')} | n={v.get('n', '?')} |")
    else:
        out += ["| Benchmark | Tenstorrent (measured) | Sample |", "| --- | --- | --- |"]
        for k, v in tt_results.items():
            if isinstance(v, dict):
                out.append(f"| {k} | {v.get('tt', '-')} | n={v.get('n', '?')} |")
    out.append("")
    return "\n".join(out) + "\n"


def _maybe_run_container_eval(args, slug, demo_dir) -> None:
    """After a container publish, serve the model and run a bounded on-TT inference eval, writing
    tt_eval.json into the demo dir so the mirrored card shows the GPU-vs-TT comparison automatically.
    Default on; --no-eval to skip. Fully best-effort: any serving/eval failure is caught so it can
    never break publishing."""
    import subprocess as _sp
    import time as _t

    if getattr(args, "dry_run", False) or not demo_dir:
        return
    try:
        from .tt_infer_eval import run_inference_eval
    except Exception:  # noqa: BLE001
        return
    repo = args.repo
    ttm = getattr(args, "tt_model_bin", None) or "tt-model"
    cname = "tt-model-%s-default" % slug
    model_id = getattr(args, "weights", None) or slug
    try:
        print("  [publish-hf] on-TT eval: serving %s to measure inference accuracy (bounded)..." % repo)
        _sp.run([ttm, "serve", repo, "--refresh", "--detach"], check=False, timeout=2400)
        ready = False
        for _ in range(70):
            lg = _sp.run(["docker", "logs", cname], capture_output=True, text=True)
            if "Application startup complete" in (lg.stdout + lg.stderr):
                ready = True
                break
            st = _sp.run(
                ["docker", "ps", "-a", "--filter", "name=" + cname, "--format", "{{.Status}}"],
                capture_output=True,
                text=True,
            ).stdout
            if "Exited" in st:
                break
            _t.sleep(30)
        if not ready:
            print("  [publish-hf] on-TT eval: server did not become ready; skipping")
            return
        out_path = Path(demo_dir) / "tt_eval.json"
        res = run_inference_eval("http://localhost:20000/v1/chat/completions", model_id, out_path)
        if res:
            print("  [publish-hf] on-TT eval done: " + ", ".join(f"{k}={v['tt']}(n={v['n']})" for k, v in res.items()))
    except Exception as exc:  # noqa: BLE001
        print("  [publish-hf] on-TT eval skipped: %s" % exc)
    finally:
        try:
            _sp.run(["docker", "rm", "-f", cname], capture_output=True)
        except Exception:  # noqa: BLE001
            pass


def _compose_mirrored_card(
    state: dict,
    slug: str,
    base_weights: str | None,
    commit: str | None,
    token: str | None,
    demo_dir=None,
    repo_id: str | None = None,
) -> str | None:
    """A card that MIRRORS the upstream HF model card, with a TT-validated header whose numbers all
    come from this run and hardware named as the chips actually used. None -> caller falls back."""
    if not base_weights:
        return None
    body = _fetch_upstream_card(base_weights, token)
    if not body:
        return None
    body = _mirror_body_tt(body, base_weights, demo_dir, state)  # strip GPU, TT-vs-GPU benchmarks
    state = _ensure_hw_env(state, demo_dir)
    _, hw_tag, _ = _serve_target(None, state.get("env"))
    hw = _hardware_phrase(state, hw_tag)
    arch_tag = str((state.get("env") or {}).get("arch") or "").strip().lower()
    fm = [
        "---",
        f"base_model: {base_weights}",
        "base_model_relation: finetune",
        "library_name: tt-metal",
        "pipeline_tag: text-generation",
        "tags:",
        "  - tenstorrent",
        "  - tt-metal",
        "  - ttnn",
    ]
    if arch_tag:
        fm.append(f"  - {arch_tag}")
    fm.append("---")
    intro = (
        f"A Tenstorrent deployment of [`{base_weights}`](https://huggingface.co/{base_weights}) — the same "
        f"weights, brought up on {hw}. This card mirrors the upstream model card: the "
        "**Validated on Tenstorrent** section holds figures measured on Tenstorrent hardware. GPU-specific "
        "deployment/usage has been removed; the base-model benchmark tables are shown with a Tenstorrent "
        "column (the on-device score when a bounded TT eval has been run, else 'pending (TT)')."
    )
    return (
        "\n".join(fm)
        + f"\n\n# {base_weights} — on Tenstorrent\n\n{intro}\n\n"
        + _tt_validated_block(state, hw, commit)
        + "\n"
        + _tt_run_section(repo_id, base_weights)
        + _tt_accuracy_section(_load_tt_eval(demo_dir) if demo_dir else {})
        + "\n## Base model (upstream) — reference\n\n"
        + body
        + "\n"
    )


def _build_card(
    state: dict,
    slug: str,
    base_weights: str | None,
    commit: str | None,
    token: str | None = None,
    repo_id: str | None = None,
) -> str:
    mirrored = _compose_mirrored_card(state, slug, base_weights, commit, token, repo_id=repo_id)
    if mirrored is not None:
        return mirrored
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
        "Optimized for Tenstorrent hardware with kernel/lever autotuning " "against a PCC-gated end-to-end pipeline.",
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
        *_tt_run_section(repo_id, base_weights).rstrip("\n").split("\n"),
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


def _arch_from_weights(repo: str, token: str | None) -> tuple:
    """HF architecture + model_type read from the BASE-WEIGHTS repo's config.json (hub/cache).

    Demos ported from HF often ship no config.json of their own (it lives in the weights repo), so
    _detect_arch_and_type returns (None, None) and the vLLM bundle never scaffolds -- then the
    container's EXTRA_MODELS_DIR verify fails with FileNotFoundError. Reading the architecture from
    the auto-detected base weights fixes that for ANY arch (stock or novel, e.g. kolibri1)."""
    if not repo:
        return (None, None)
    try:
        from huggingface_hub import hf_hub_download

        p = hf_hub_download(repo_id=repo, filename="config.json", repo_type="model", token=token)
        d = json.load(open(p))
        arch = (d.get("architectures") or [None])[0]
        return (arch, d.get("model_type")) if arch else (None, None)
    except Exception:  # noqa: BLE001
        return (None, None)


def _fallback_arch(slug: str) -> str:
    """A generic architecture name when none can be detected anywhere (no demo/weights config.json,
    not mistral-format). Lets the container ALWAYS get a scaffolded bundle so it builds; always a
    stub (never servable) since the real architecture is unknown."""
    import re

    camel = "".join(p.capitalize() for p in re.split(r"[^A-Za-z0-9]+", slug) if p) or "Model"
    return f"{camel}ForCausalLM"


def _pick_base_generator(arch: str, model_type: str | None) -> tuple[str, bool]:
    """(stock_generator_class, is_stub) for an arch. Stock arches map to a supported generator
    (is_stub=False, trivially servable). Novel arches pick the closest base (hybrid vs plain) and
    are a stub (is_stub=True) whose body the author completes. Generic across all models."""
    if arch in _STOCK_GENERATORS:
        return _STOCK_GENERATORS[arch], False
    hay = f"{arch} {model_type or ''}".lower()
    hybrid = any(t in hay for t in ("_h", "hybrid", "mamba", "jamba", "nemotronh", "recurrent"))
    return ("HybridAttentionForCausalLM" if hybrid else "LlamaForCausalLM"), True


_PIPELINE_ADAPTER_TMPL = '''# SPDX-License-Identifier: Apache-2.0
"""vLLM generator for @SLUG@ (@ARCH@) on Tenstorrent.

The tt-metal vLLM plugin registers this class as @TTARCH@ via vllm_metadata.json and drives it through
the generator contract (initialize_vllm_model -> allocate_kv_cache_per_layer -> warmups ->
prefill_forward / decode_forward). @ARCH@ is not a plugin built-in, so this wraps the demo's own
resident pipeline (@PMOD@) rather than a stock generator.

Starting serve profile (matches the validated demo): ONE sequence at a time (--max-num-seqs 1), prompt
and generated tokens bounded by the pipeline KV capacity (--max-model-len @KV@), host-side sampling
(logits are returned to vLLM). Concurrent users / long context / on-device sampling are model work; see
the TODOs below.

Decode is traced whenever vLLM asks for it (enable_trace): one step of the pipeline's decode_trace_step is
captured and replayed for every token, with vLLM's sampled token written into the resident tok_u32 buffer
before each replay -- the speed the demo was optimized for. Prefill stays eager (its length follows the
prompt). A prefill at a length not seen before compiles new programs whose buffers may overlap the
captured trace, so the trace is released then and captured again at the next decode.
"""
from __future__ import annotations

import logging

import torch

try:
    import ttnn  # noqa: F401
    from @PMOD@ import build_pipeline as _build_pipeline, read_host as _read_host
    try:
        from @PMOD@ import KV_CAPACITY as _KV_CAPACITY
    except Exception:  # noqa: BLE001
        _KV_CAPACITY = @KV@
except Exception:  # available only inside the serving image
    _build_pipeline = None
    _read_host = None
    _KV_CAPACITY = @KV@

log = logging.getLogger(__name__)


class @TTARCH@:
    """Generator contract over the demo resident pipeline (@PMOD@)."""

    # Read by the plugin off the CLASS before the model exists.
    model_capabilities = {"supports_sample_on_device": False}

    def __init__(self, pipe, max_batch_size, max_seq_len, mesh_device=None):
        self.pipe = pipe
        self.mesh_device = mesh_device
        self.max_batch_size = int(max_batch_size)
        self.max_seq_len = int(max_seq_len)
        self.vocab_size = int(getattr(pipe.config, "vocab_size", 0)) or None
        self._state = None
        self._active = False
        self.already_warmed_up_prefill = False
        self._traceable = mesh_device is not None and callable(getattr(pipe, "decode_trace_step", None))
        self._trace_id = None
        self._trace_logits = None
        self._decode_compiled = False
        self._prefill_lens = set()

    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size, max_seq_len=None, n_layers=None, **_ignored):
        if int(max_batch_size) != 1:
            raise ValueError(
                "%s serves one sequence at a time; launch vLLM with --max-num-seqs 1 (got %s)."
                % (cls.__name__, max_batch_size)
            )
        msl = int(max_seq_len or _KV_CAPACITY)
        if msl > _KV_CAPACITY:
            raise ValueError(
                "%s: resident KV capacity is %d tokens; use --max-model-len %d or less (got %d)."
                % (cls.__name__, _KV_CAPACITY, _KV_CAPACITY, msl)
            )
        if _build_pipeline is None:
            raise RuntimeError("pipeline module @PMOD@ not importable; run inside the serving image")
        pipe = _build_pipeline(mesh_device, layers=n_layers)
        log.info("built @SLUG@ pipeline (device rows=%s, KV=%s)", getattr(pipe, "batch", "?"), _KV_CAPACITY)
        return cls(pipe, max_batch_size, msl, mesh_device)

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        return [None for _ in per_layer_specs]  # the pipeline owns its cache

    def allocate_kv_cache(self, kv_cache_shape=None, dtype=None, num_layers=None):
        return [None for _ in range(int(num_layers or 0))]

    def _host_logits_row0(self, logits_dev):
        host = _read_host(logits_dev).reshape(-1)
        v = self.vocab_size or host.numel()
        return host[:v].float().reshape(1, 1, v)

    def _capture_decode_trace(self):
        """Record one decode step (recording runs nothing on the device). False when it cannot be traced."""
        tid = None
        try:
            tid = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
            out = self.pipe.decode_trace_step()
            ttnn.end_trace_capture(self.mesh_device, tid, cq_id=0)
        except Exception as e:  # noqa: BLE001
            if tid is not None:
                try:
                    ttnn.end_trace_capture(self.mesh_device, tid, cq_id=0)
                except Exception:  # noqa: BLE001
                    pass
                try:
                    ttnn.release_trace(self.mesh_device, tid)
                except Exception:  # noqa: BLE001
                    pass
            self._traceable = False
            log.warning("@SLUG@: decode trace capture failed (%s); decoding eagerly from here, which is much slower", e)
            return False
        if isinstance(out, dict):
            out = out["logits"]
        elif isinstance(out, (tuple, list)):
            out = out[0]
        self._trace_id, self._trace_logits = tid, out
        log.info("@SLUG@: decode trace captured; every decode token replays it")
        return True

    def _release_trace(self):
        if self._trace_id is not None:
            try:
                ttnn.release_trace(self.mesh_device, self._trace_id)
            except Exception:  # noqa: BLE001
                pass
            self._trace_id = self._trace_logits = None

    def prefill_forward(self, tokens, page_table=None, *, prompt_lens=None, start_pos=None, **_compat):
        if tokens.shape[0] != 1:
            raise ValueError("one sequence at a time: got a prefill batch of %d" % tokens.shape[0])
        if start_pos is not None and int(torch.as_tensor(start_pos).max()) != 0:
            raise ValueError("chunked / prefix-cached prefill is not supported by this generator")
        T = int(prompt_lens[0]) if prompt_lens is not None else int(tokens.shape[1])
        if T > self.max_seq_len:
            raise ValueError("prompt of %d tokens exceeds max_model_len=%d" % (T, self.max_seq_len))
        ids = tokens[0, :T].to(torch.int64).tolist()
        self._state = self.pipe.decode_prefill([ids])  # seeds the resident caches for every device row
        self._active = True
        plen = int(getattr(self.pipe, "prefill_len", T) or T)
        if plen not in self._prefill_lens:
            self._prefill_lens.add(plen)
            self._release_trace()
        return self._host_logits_row0(self._state["logits"])

    def decode_forward(self, tokens, start_pos=None, page_table=None, *, enable_trace=False, **_compat):
        if not self._active:
            raise RuntimeError("decode_forward before prefill_forward")
        # Honor vLLM's host-sampled token: the demo pipeline commits its OWN device-sampled token, so
        # overwrite its resident input-token buffer when it exposes one (the generated-pipeline
        # convention: a uint32 ``tok_u32`` [1, batch] written in place via ``_write``). Greedy decode
        # matches even without this; the writeback gives full host-sampler fidelity.
        try:
            _tok = int(torch.as_tensor(tokens).reshape(-1)[0])
            if hasattr(self.pipe, "tok_u32") and hasattr(self.pipe, "_write"):
                self.pipe._write(
                    "tok_u32",
                    torch.full((1, int(self.pipe.batch)), _tok, dtype=torch.int32),
                    ttnn.uint32,
                    ttnn.ROW_MAJOR_LAYOUT,
                )
        except Exception:  # noqa: BLE001
            pass
        # Capturing before the decode programs have run once would compile them mid-capture.
        if (
            enable_trace
            and self._traceable
            and self._decode_compiled
            and (self._trace_id is not None or self._capture_decode_trace())
        ):
            ttnn.execute_trace(self.mesh_device, self._trace_id, cq_id=0, blocking=False)
            return self._host_logits_row0(self._trace_logits)
        self._state = self.pipe.decode_step(self._state)
        self._decode_compiled = True
        return self._host_logits_row0(self._state["logits"])

    def warmup_model_prefill(self, *, kv_cache=None, can_sample_on_device=False, enable_trace=False):
        if enable_trace:
            return  # prefill stays eager: its length follows the prompt
        bos = int(getattr(self.pipe.config, "bos_token_id", 1) or 1)
        self.prefill_forward(torch.full((1, 8), bos, dtype=torch.int64), None, prompt_lens=[8])
        self.already_warmed_up_prefill = True

    def warmup_model_decode(self, *, kv_cache=None, max_batch_size=1, num_blocks=0, can_sample_on_device=False, enable_trace=False):
        if enable_trace:
            if not self._traceable:
                log.warning(
                    "@SLUG@: vLLM asked for traced decode but @PMOD@ has no decode_trace_step; "
                    "decoding eagerly, which is much slower"
                )
            elif self._trace_id is None and self._decode_compiled:
                self._capture_decode_trace()
            return
        if not self._active:
            return
        self.decode_forward(torch.zeros((1, 1), dtype=torch.int64), None, None)
        self._active = False  # warmup sequence done; next prefill reseeds

    def cleanup(self):
        self._release_trace()
'''


def _demo_pipeline_module(checkout, slug):
    """Dotted import path to the demo resident pipeline if it exposes build_pipeline, else None."""
    pf = Path(checkout) / "models" / "demos" / slug / "tt" / "pipeline.py"
    try:
        if pf.is_file() and "def build_pipeline" in pf.read_text(errors="replace"):
            return "models.demos.%s.tt.pipeline" % slug
    except Exception:  # noqa: BLE001
        pass
    return None


def _pipeline_serve_profile(checkout, slug):
    """(kv_capacity, mesh_str) for the serve note, from the demo pipeline + parallelism manifest."""
    import re as _re

    kv = 2048
    try:
        pf = Path(checkout) / "models" / "demos" / slug / "tt" / "pipeline.py"
        m = _re.search(r"KV_CAPACITY\\s*=\\s*(\\d+)", pf.read_text(errors="replace"))
        if m:
            kv = int(m.group(1))
    except Exception:  # noqa: BLE001
        pass
    mesh = None
    try:
        d = json.loads((Path(checkout) / "models" / "demos" / slug / "parallelism_manifest.json").read_text())
        if isinstance(d.get("mesh"), (list, tuple)) and len(d["mesh"]) == 2:
            mesh = "(%d, %d)" % (int(d["mesh"][0]), int(d["mesh"][1]))
    except Exception:  # noqa: BLE001
        pass
    return kv, mesh


def _pipeline_adapter_src(slug, arch, tt_arch, pipeline_mod, kv):
    """Render the pipeline-wrapping adapter for a novel arch that has its own demo pipeline."""
    return (
        _PIPELINE_ADAPTER_TMPL.replace("@SLUG@", slug)
        .replace("@TTARCH@", tt_arch)
        .replace("@ARCH@", arch)
        .replace("@PMOD@", pipeline_mod)
        .replace("@KV@", str(kv))
    )


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
    created = not meta.is_file()
    bundle.mkdir(parents=True, exist_ok=True)
    tt_arch = arch if arch.startswith("TT") else "TT" + arch

    # Novel arch WITH its own resident demo pipeline -> ship a real pipeline-wrapping generator
    # skeleton (the Nemotron-shaped serve contract) instead of a no-op stub, so publish-hf always
    # produces a servable-shaped adapter and never leaves "no adapter" for the next run.
    pipeline_mod = _demo_pipeline_module(checkout, slug) if is_stub else None
    if pipeline_mod:
        kv, mesh = _pipeline_serve_profile(checkout, slug)
        note = (
            "Single-sequence generator over the demo tt/pipeline.py; serve with "
            "--max-num-seqs 1 --max-model-len %d" % kv
        )
        if mesh:
            note += ' and MESH_DEVICE="%s"' % mesh
        note += "."
        meta.write_text(
            json.dumps(
                {
                    "arch": arch,
                    "main_class": "adapter:" + tt_arch,
                    "weights": weights or "REPLACE/with-base-weights-repo",
                    "note": note,
                    "max_num_seqs": 1,
                    "max_model_len": kv,
                    "serve_hf_override_arch": tt_arch,
                },
                indent=2,
            )
            + "\n"
        )
        (bundle / "adapter.py").write_text(_pipeline_adapter_src(slug, arch, tt_arch, pipeline_mod, kv))
        return created, is_stub, str(bundle)

    meta.write_text(
        json.dumps(
            {
                "arch": arch,
                "main_class": "adapter:" + tt_arch,
                "weights": weights or "REPLACE/with-base-weights-repo",
                "note": "Auto-generated vLLM bundle for Tenstorrent.",
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
    return created, is_stub, str(bundle)


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


_EAGER_PIPELINE_ADAPTER_MARK = "return  # eager generator; trace warmup is a no-op"


def _refresh_eager_pipeline_adapter(checkout, slug, bundle_dir) -> bool:
    """Re-render an adapter.py written by an earlier _PIPELINE_ADAPTER_TMPL, which ignored vLLM's
    enable_trace and decoded every token eagerly: a package published with it serves several times
    slower than the traced speed its card reports (kolibri_1: ~8 tok/s served vs 39.5 traced). Only the
    tool's own rendering is replaced, never a hand-written adapter. True when the file was rewritten."""
    ap = Path(bundle_dir) / "adapter.py"
    try:
        src = ap.read_text()
        meta = json.loads((Path(bundle_dir) / "vllm_metadata.json").read_text())
    except Exception:  # noqa: BLE001
        return False
    if _EAGER_PIPELINE_ADAPTER_MARK not in src:
        return False
    pipeline_mod = _demo_pipeline_module(checkout, slug)
    arch = meta.get("arch")
    tt_arch = str(meta.get("main_class") or "").partition(":")[2]
    if not (pipeline_mod and arch and tt_arch):
        return False
    kv, _mesh = _pipeline_serve_profile(checkout, slug)
    ap.write_text(_pipeline_adapter_src(slug, arch, tt_arch, pipeline_mod, kv))
    return True


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


_TT_PLUGIN_NOVEL_ARCH_SNIPPET = '''

# ---- Tenstorrent novel-architecture serving support (added by tt_hw_planner publish-hf) -----------
def _tt_register_novel_configs() -> None:
    """Register a permissive Transformers config for any EXTRA_MODELS_DIR bundle whose base weights
    declare a model_type this Transformers build does not know (brand-new archs ship no remote code),
    so vLLM can build its ModelConfig. The TT generator, not Transformers, builds the real model."""
    import os, json, glob
    try:
        from transformers import AutoConfig, PretrainedConfig
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    except Exception:
        return
    emd = os.getenv("EXTRA_MODELS_DIR")
    hf_home = os.getenv("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    if not emd:
        return
    for meta in glob.glob(os.path.join(emd, "*", "vllm_metadata.json")):
        try:
            w = (json.load(open(meta)) or {}).get("weights")
            if not w or "/" not in w:
                continue
            folder = "models--" + w.replace("/", "--")
            cfgs = glob.glob(os.path.join(hf_home, "hub", folder, "snapshots", "*", "config.json"))
            if not cfgs:
                continue
            mt = (json.load(open(cfgs[0])) or {}).get("model_type")
            if not mt or mt in CONFIG_MAPPING:
                continue
            cls = type("TTAutoConfig_" + mt, (PretrainedConfig,), {"model_type": mt})
            AutoConfig.register(mt, cls, exist_ok=True)
        except Exception:
            continue


def _tt_force_generate_capability() -> None:
    """A TT-prefixed arch is a custom adapter driven through the TT generator contract, not vLLM's
    native model __init__, so it never satisfies vLLM's structural is_vllm_model check (stock models
    ride a built-in arch; a novel arch has none). Declare TT archs generative so ModelConfig accepts
    them; the TT runner drives prefill/decode."""
    try:
        from vllm.model_executor.models.registry import ModelRegistry
        RegClass = type(ModelRegistry)
    except Exception:
        return
    if getattr(RegClass, "_tt_gen_patched", False):
        return
    _orig = RegClass.is_text_generation_model

    def _wrap(self, architectures, *a, **k):
        try:
            archs = architectures if isinstance(architectures, (list, tuple)) else [architectures]
            if any(isinstance(x, str) and x.startswith("TT") for x in archs):
                return True
        except Exception:
            pass
        return _orig(self, architectures, *a, **k)

    RegClass.is_text_generation_model = _wrap
    RegClass._tt_gen_patched = True
# ---- end Tenstorrent novel-architecture serving support -------------------------------------------
'''


def _ensure_plugin_novel_arch_support(plugin_path) -> bool:
    """Make the in-tree vLLM TT plugin serve a brand-new architecture out of the box, by adding two
    small hooks to its platform.py and calling them during EXTRA_MODELS_DIR registration. Idempotent,
    additive, and a no-op for stock archs (the hooks only fire on a novel EXTRA_MODELS_DIR bundle), so
    the image built from this plugin needs no hand-fixing. Returns True when support is present."""
    from pathlib import Path as _P

    pf = _P(plugin_path) / "src" / "vllm_tt_plugin" / "platform.py"
    try:
        src = pf.read_text()
    except Exception:
        return False
    if "_tt_register_novel_configs" in src:
        return True
    anchor = "def _register_models_from_extra_dir(ModelRegistry) -> int:\n"
    if anchor not in src:
        return False
    calls = (
        anchor
        + "    try:\n        _tt_register_novel_configs()\n    except Exception:\n        pass\n"
        + "    try:\n        _tt_force_generate_capability()\n    except Exception:\n        pass\n"
    )
    src = src.replace(anchor, _TT_PLUGIN_NOVEL_ARCH_SNIPPET + "\n\n" + calls, 1)
    try:
        pf.write_text(src)
        return True
    except Exception:
        return False


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
    if not arch_det and getattr(args, "weights", None):
        # The demo ships no HF config.json; read the arch from the auto-detected base weights so
        # novel archs (e.g. kolibri1) still scaffold a bundle and the container verify passes.
        arch_det, mtype = _arch_from_weights(args.weights, _hf_token(args))
        if arch_det:
            print(f"  [publish-hf] arch from base weights {args.weights}: {arch_det}")
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
    if adapter_ready and _refresh_eager_pipeline_adapter(checkout, slug, bundle_dir):
        print(
            f"  [publish-hf] re-rendered {bundle_dir / 'adapter.py'}: it was an earlier copy of the tool's "
            "pipeline adapter, which decoded every token eagerly; decode is now traced when vLLM asks"
        )
    servable = True
    if not adapter_ready:
        if arch_det:
            _base_cls, _is_stub_arch = _pick_base_generator(arch_det, mtype)
            # A mistral-format fallback is an UNVERIFIED backbone guess (no HF config; the real model
            # may be custom, e.g. a TTS head), so never claim it serves until a real adapter is written.
            servable = (not _is_stub_arch) and not _mistral_fallback
        else:
            servable = False  # no arch detected anywhere -> slug-derived stub, never servable
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
    elif not getattr(args, "no_scaffold", False):
        # ALWAYS scaffold a bundle for a container build, even when no arch could be detected
        # anywhere (no demo/weights config.json, not mistral-format): a slug-derived stub makes the
        # image's EXTRA_MODELS_DIR verify pass so the container builds. A stub is clearly not-servable.
        _arch_for_bundle = arch_det or _fallback_arch(slug)
        created, is_stub, bpath = _scaffold_vllm_bundle(
            checkout, extra, _arch_for_bundle, mtype, getattr(args, "weights", None), slug
        )
        state_note = (
            "STUB — not servable until an adapter is written"
            if (is_stub or _mistral_fallback or not arch_det)
            else "stock generator — servable as-is"
        )
        print(f"  [publish-hf] vLLM bundle {'created' if created else 'exists'}: {bpath}  ({state_note})")

    # A novel-arch pipeline adapter must serve out of the box: build the image from the MATCHED
    # local vLLM+plugin (no version skew) and give that plugin novel-arch support. Stock arches are
    # unaffected (they ride a vLLM built-in; the hooks only fire on a novel EXTRA_MODELS_DIR bundle).
    _is_pipeline_adapter = (not adapter_ready) and bool(_demo_pipeline_module(checkout, slug))
    if getattr(args, "container", False) and _is_pipeline_adapter:
        if not getattr(args, "vllm_path", None):
            disc = _discover_serving_stack()
            if disc:
                vpath, ppath, vcommit, vremote = disc
                args.vllm_path, args.plugin_path = vpath, ppath
                args._discovered_vllm = (vcommit, vremote)
                print(f"  [publish-hf] auto serving stack (novel arch): {vremote} @ {vcommit[:9]} -- {vpath}")
                if not getattr(args, "lock", None):
                    cf = _auto_constraint_file(checkout)
                    if cf:
                        args.lock = cf
        if getattr(args, "plugin_path", None):
            _ok = _ensure_plugin_novel_arch_support(args.plugin_path)
            print(f"  [publish-hf] plugin novel-arch serving support: {'present' if _ok else 'could not inject'}")

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
        if any(k in val.stderr for k in ("extra_models_dir", "vllm_metadata", "source.code")):
            print(
                "  [publish-hf] The vLLM adapter dir (runtime.extra_models_dir) with a "
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
                commit_message=f"Publish {slug} container bundle",
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
    _verify_published_package(args, slug, servable)
    return 0


_VERIFY_TITLE = "Published-package verification"


def _verify_published_package(args, slug: str, servable: bool) -> None:
    """Pull THE JUST-PUBLISHED package back and run its own quickstart, so the page is proven, not
    asserted. `tt model pull` must succeed; `tt serve` must then reach either `Application startup
    complete` (a servable package) or the documented scaffold stop (`initialize_vllm_model` missing)
    for a not-yet-servable one. The VERIFIED outcome is written into the card and printed; a result
    that CONTRADICTS the card's serve claim is flagged loudly. Best-effort: skips (the publish
    stands) when the `tt` CLI or docker is absent, and never raises."""
    import datetime as _dt
    import shutil
    import subprocess
    import time

    if getattr(args, "no_verify_publish", False):
        return
    tt = shutil.which("tt") or shutil.which("tt-model")
    docker = shutil.which("docker")
    if not tt or not docker:
        print("  [publish-hf] verify: tt CLI / docker not on this host; skipped (publish stands).")
        return
    repo = args.repo
    sub_model = "model" if os.path.basename(tt) == "tt" else None  # `tt model pull` vs `tt-model pull`

    def _tt(*a, timeout=3600):
        cmd = [tt] + ([sub_model] if sub_model else []) + list(a)
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    print(f"  [publish-hf] verify: pulling the PUBLISHED package back ({repo}) …")
    try:
        pull = _tt("pull", repo, timeout=5400)
    except Exception as e:  # noqa: BLE001
        print(f"  [publish-hf] verify: pull errored ({e}); skipped (publish stands).")
        return
    pull_ok = pull.returncode == 0
    if not pull_ok:
        print("  [publish-hf] verify: tt model pull FAILED for the just-published package.")

    outcome, detail, cname = "unknown", "", None
    if pull_ok:
        print(f"  [publish-hf] verify: starting `tt serve {repo}` (first start compiles kernels) …")
        try:
            # serve is a TOP-LEVEL subcommand for BOTH CLIs (tt serve / tt-model serve),
            # not `tt model serve` -- only pull lives under `tt model`.
            subprocess.run([tt, "serve", repo], capture_output=True, text=True, timeout=300)
        except Exception:  # noqa: BLE001
            pass
        key = slug.split("/")[-1].replace("-", "_")[:12]
        deadline = time.time() + 1200
        beat = 0
        while time.time() < deadline:
            ps = subprocess.run([docker, "ps", "-a", "--format", "{{.Names}}"], capture_output=True, text=True)
            cands = [n for n in ps.stdout.split() if key in n.replace("-", "_")]
            cname = cands[0] if cands else cname
            if cname:
                r = subprocess.run([docker, "logs", cname], capture_output=True, text=True)
                low = ((r.stdout or "") + (r.stderr or "")).lower()
                if "application startup complete" in low:
                    outcome, detail = "ready", "Application startup complete"
                    break
                if (
                    "initialize_vllm_model" in low
                    or "engine core initialization failed" in low
                    or "has no attribute" in low
                ):
                    outcome, detail = "scaffold_stop", "vLLM adapter not implemented (initialize_vllm_model)"
                    break
                st = subprocess.run(
                    [docker, "inspect", "-f", "{{.State.Status}} {{.State.ExitCode}}", cname],
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                if st.startswith("exited") and not st.endswith(" 0"):
                    outcome, detail = "error", f"container exited: {st}"
                    break
            if beat % 6 == 0:
                print(
                    f"  [publish-hf] verify: … waiting for the served container ({int(deadline - time.time())}s left)"
                )
            beat += 1
            time.sleep(10)
        if cname:
            subprocess.run([docker, "rm", "-f", cname], capture_output=True, text=True)

    if not pull_ok:
        verdict = "FAIL (the published package did not pull)"
    elif servable and outcome == "ready":
        verdict = "PASS (pulled + served: Application startup complete)"
    elif (not servable) and outcome == "scaffold_stop":
        verdict = "PASS as documented (pulled; serve stops at the pending vLLM adapter)"
    elif servable and outcome != "ready":
        verdict = f"MISMATCH (card claims servable, but serve did not start: {outcome} {detail})"
    elif (not servable) and outcome == "ready":
        verdict = "NOTE (card says not-yet-servable, but the package actually served)"
    else:
        verdict = f"INCONCLUSIVE (pull ok; serve outcome {outcome} {detail})"
    loud = verdict.startswith(("FAIL", "MISMATCH"))
    print(f"  [publish-hf] verify: {'FAIL -> ' if loud else 'OK -> '}{verdict}")

    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    section = (
        "Automated post-publish check of THIS uploaded package on real hardware "
        "(`tt model pull` + `tt serve`), so the page is proven rather than asserted. "
        f"_Verified {ts}._\n\n"
        "| step | result |\n| --- | --- |\n"
        f"| `tt model pull` | {'pass' if pull_ok else 'FAIL'} |\n"
        f"| `tt serve` | {verdict} |\n"
    )
    try:
        _upload_card_section(args, _VERIFY_TITLE, section)
    except Exception as e:  # noqa: BLE001
        print(f"  [publish-hf] verify: card section not written ({e}).")


def _maybe_autocommit(args, demo_dir) -> None:
    """When publish-hf finishes cleanly, commit the model's demo dir and push it (default on; disable
    with --no-commit-push). Reuses the stage auto-commit: scoped to the model, hook-free, refuses the
    tt_hw_planner tool branch, best-effort."""
    if getattr(args, "dry_run", False):
        return
    try:
        from .autocommit import commit_and_push_stage

        commit_and_push_stage(args, str(demo_dir), "publish-hf")
    except Exception as exc:  # noqa: BLE001
        print(f"  [publish-hf] auto-commit skipped: {exc}")


def _maybe_mirror_container_card(args, slug, state, base_weights, commit, demo_dir=None) -> None:
    """After a container publish, re-publish the README MIRRORING the upstream HF model card the port
    is based on — a TT-validated header (numbers measured in this run) + correct hardware + the upstream
    body for reference. Default on; --no-mirror-upstream to keep tt-model's card. Best-effort."""
    if getattr(args, "dry_run", False) or not getattr(args, "mirror_upstream", True) or not base_weights:
        return
    try:
        tok = _hf_token(args)
        card = _compose_mirrored_card(state, slug, base_weights, commit, tok, demo_dir, args.repo)
        if not card:
            print("  [publish-hf] upstream mirror: upstream card unavailable; kept tt-model card")
            return
        from huggingface_hub import HfApi

        HfApi(token=tok).upload_file(
            path_or_fileobj=card.encode(),
            path_in_repo="README.md",
            repo_id=args.repo,
            repo_type="model",
            commit_message="Card: mirror upstream model card + TT-validated results",
        )
        print("  [publish-hf] card re-published mirroring the upstream model card (TT-validated).")
    except Exception as exc:  # noqa: BLE001
        print(f"  [publish-hf] upstream mirror skipped: {exc}")


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

    state = _ensure_hw_env(state, demo_dir, getattr(args, "box", None))
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
        rc = _run_container(args, state, slug, demo_dir, commit)
        if rc == 0:
            _maybe_run_container_eval(args, slug, demo_dir)
            _maybe_mirror_container_card(args, slug, state, base_weights, commit, demo_dir)
            _maybe_autocommit(args, demo_dir)
        return rc

    card = _build_card(state, slug, base_weights, commit, _hf_token(args), args.repo)

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
            commit_message=f"Publish {slug} (tt-metal {commit})",
        )
    except Exception as e:
        print(f"  [publish-hf] push failed: {e}")
        return 4

    url = f"https://huggingface.co/{args.repo}"
    print(f"  [publish-hf] published: {url}")
    _maybe_autocommit(args, demo_dir)
    return 0
