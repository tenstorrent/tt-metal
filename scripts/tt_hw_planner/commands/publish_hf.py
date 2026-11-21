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
import sys
import tempfile
from pathlib import Path

from ..discovery import TTT_DIRS, ttt_dir
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


def _current_repo_id(repo_id: str, token: str | None) -> str:
    """The id HF serves ``repo_id`` under now. A renamed account or repo still answers to its old id, so a card that
    names the old one still resolves, but under a name that is no longer the owner's. ``repo_id`` itself when the repo
    does not exist yet or HF cannot be reached."""
    try:
        from huggingface_hub import HfApi

        return HfApi(token=token).model_info(repo_id).id or repo_id
    except Exception:  # noqa: BLE001
        return repo_id


def _upstream_card_terms(weights: str | None, token: str | None) -> dict:
    """The base model's licence and pipeline tag as its HF card states them. The package inherits both, so they are
    read from upstream, not assumed. Empty when there is no base model or HF cannot be reached."""
    if not weights:
        return {}
    try:
        from huggingface_hub import HfApi

        info = HfApi(token=token).model_info(weights)
    except Exception:  # noqa: BLE001
        return {}
    card = info.card_data or {}
    terms = {k: card.get(k) for k in ("license", "license_name", "license_link")}
    terms["pipeline_tag"] = info.pipeline_tag or card.get("pipeline_tag")
    return {k: v for k, v in terms.items() if isinstance(v, str) and v}


def _card_terms_lines(terms: dict, default_tag: str | None) -> list:
    """tt-model.yaml's card licence and pipeline-tag lines: upstream's terms, apache-2.0 when upstream states no
    licence, and ``default_tag`` when it names no pipeline."""
    lines = ["  license:", f"    id: {terms.get('license') or 'apache-2.0'}"]
    if terms.get("license") == "other":
        lines += [
            f"    {k}: {json.dumps(terms[f'license_{k}'])}" for k in ("name", "link") if terms.get(f"license_{k}")
        ]
    tag = terms.get("pipeline_tag") or default_tag
    return lines + ([f"  pipeline_tag: {tag}"] if tag else [])


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


# The vLLM TT plugin's own names for on-chip sampling: the capability a generator class declares, and
# the launch setting that turns it on. The plugin's vocabulary, not any model's.
_CAPS_ATTR = "model_capabilities"
_CAP_SAMPLE_ON_DEVICE = "supports_sample_on_device"
_PLUGIN_CONFIG_KEY = "tt"
_SAMPLE_MODE_KEY = "sample_on_device_mode"
# Decode tokens sampled on the chip; a request's single prefill sample stays on the host. Unset, the
# plugin samples every decode step on the host: a stock-generator MiniCPM package served 32 users at
# 54 tok/s/user that way, ~89 with decode sampled on the chip.
_SAMPLE_MODE_DEFAULT = "decode_only"
# How far to follow a class's bases looking for its capabilities. Generators sit one or two levels
# above the adapter class, so this only bounds a cycle.
_CAPS_MAX_DEPTH = 6
# Where the plugin is published, the ref a package follows unless told otherwise, and the script in it that
# installs the vLLM release it is built against. tt-model rebuilds that script's environment from
# runtime.vllm.version, so the version has to be the one the script at the pinned commit installs.
_PLUGIN_REPO = "https://github.com/tenstorrent/vllm-tt-plugin"
_PLUGIN_DEFAULT_REF = "main"
_PLUGIN_INSTALL_RECIPE = "docs/install-vllm-tt.sh"


# Finds a module's file the way the import system would, importing only its parent packages.
_FIND_MODULE_FILE = (
    "import importlib, importlib.machinery, sys\n"
    "m = sys.argv[1]\n"
    "parent = importlib.import_module(m.rpartition('.')[0])\n"
    "s = importlib.machinery.PathFinder.find_spec(m, list(parent.__path__))\n"
    "print(s.origin if s and s.origin else '')\n"
)


def _module_file(checkout: Path, module: str) -> Path | None:
    """The source file of dotted `module` inside `checkout`, or None. A dotted path can be an import-only
    alias with no file of its own (a package moved, its old name kept by a meta-path finder), so when
    nothing sits at the plain path the parent packages are imported in a child interpreter rooted at
    the checkout -- nothing of the checkout is loaded into this process -- and the file they resolve
    to is read instead. The child skips site-packages (-S) so an installed package of the same name
    cannot stand in for the checkout's."""
    base = Path(checkout) / Path(*module.split("."))
    for cand in (base.with_suffix(".py"), base / "__init__.py"):
        if cand.is_file():
            return cand
    if "." not in module:
        return None
    try:
        r = subprocess.run(
            [sys.executable, "-S", "-W", "ignore", "-c", _FIND_MODULE_FILE, module],
            cwd=str(checkout),
            env={**os.environ, "PYTHONPATH": str(checkout)},
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception:  # noqa: BLE001
        return None
    origin = (r.stdout or "").strip().splitlines()[-1:] if r.returncode == 0 else []
    return Path(origin[0]) if origin and Path(origin[0]).is_file() else None


def _class_capabilities(checkout: Path, module_file: Path, cls: str, _depth: int = 0) -> dict | None:
    """The literal `model_capabilities` dict class `cls` in `module_file` declares or inherits, following
    its bases through that file's `from X import Y [as Z]` imports into `checkout`. Read from source, so
    nothing is imported. None when no class in the chain states it as a literal: unknown, not False."""
    import ast

    if module_file is None or _depth > _CAPS_MAX_DEPTH:
        return None
    try:
        tree = ast.parse(Path(module_file).read_text())
    except Exception:  # noqa: BLE001
        return None
    imported = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                imported[alias.asname or alias.name] = (node.module, alias.name)
    for node in tree.body:
        if not (isinstance(node, ast.ClassDef) and node.name == cls):
            continue
        for st in node.body:
            if isinstance(st, ast.Assign):
                targets, value = st.targets, st.value
            elif isinstance(st, ast.AnnAssign):
                targets, value = [st.target], st.value
            else:
                continue
            if value is None or not any(isinstance(t, ast.Name) and t.id == _CAPS_ATTR for t in targets):
                continue
            try:
                caps = ast.literal_eval(value)
            except Exception:  # noqa: BLE001
                return None
            return dict(caps) if isinstance(caps, dict) else None
        for base in node.bases:
            if not isinstance(base, ast.Name):
                continue
            if base.id in imported:
                mod, name = imported[base.id]
                caps = _class_capabilities(checkout, _module_file(checkout, mod), name, _depth + 1)
            else:
                caps = _class_capabilities(checkout, module_file, base.id, _depth + 1)
            if caps is not None:
                return caps
        return None
    return None


def _with_declared_device_sampling(additional: dict, checkout: Path, bundle_dir: Path, main_class) -> dict:
    """`additional` (the bundle's additional_config) with on-chip decode sampling turned on when the
    class the package serves declares it supports it and the author stated no sampling mode. The
    author's settings are never changed, including an explicit mode of None."""
    tt = (additional or {}).get(_PLUGIN_CONFIG_KEY)
    if (tt is not None and not isinstance(tt, dict)) or (isinstance(tt, dict) and _SAMPLE_MODE_KEY in tt):
        return additional
    mod, _, cls = str(main_class or "").partition(":")
    if not (mod and cls):
        return additional
    caps = _class_capabilities(checkout, _module_file(bundle_dir, mod), cls)
    if not (caps or {}).get(_CAP_SAMPLE_ON_DEVICE):
        return additional
    print(
        f"  [publish-hf] serve: {cls} declares {_CAP_SAMPLE_ON_DEVICE} and the bundle sets no "
        f"{_SAMPLE_MODE_KEY}; serving with {_SAMPLE_MODE_KEY}={_SAMPLE_MODE_DEFAULT} (decode sampled on the chip)"
    )
    return {**(additional or {}), _PLUGIN_CONFIG_KEY: {**(tt or {}), _SAMPLE_MODE_KEY: _SAMPLE_MODE_DEFAULT}}


def _serve_hardware(state: dict, box, arch, hardware, mesh_device) -> dict:
    """The serve target and how the card names it: {arch, hardware, mesh_device, hw_phrase, perf_hw}, each given
    value kept, the rest derived from the run's box. Raises ValueError when there is no target."""
    import re as _re

    a, hw, mesh = _serve_target(box, state.get("env"))
    arch = arch or a
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
    return {"arch": arch, "hardware": hardware, "mesh_device": mesh_device, "hw_phrase": hw_phrase, "perf_hw": perf_hw}


def _adapter_demo_modules(checkout, adapter_file, demo_pkg) -> list:
    """Modules of the demo package `demo_pkg` that the bundle adapter imports, anywhere in it (an adapter imports its
    pipeline inside try/except, so a missing dependency only shows when a request is served), sorted. `from M import
    name` counts M.name when that is a module of the checkout, else M."""
    import ast

    try:
        tree = ast.parse(Path(adapter_file).read_text(errors="replace"))
    except (OSError, SyntaxError, ValueError):
        return []
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            base = Path(checkout, *node.module.split("."))
            for a in node.names:
                sub = base / a.name
                is_module = sub.with_suffix(".py").is_file() or (sub / "__init__.py").is_file()
                mods.add(f"{node.module}.{a.name}" if is_module else node.module)
    return sorted(m for m in mods if m.startswith(demo_pkg + "."))


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
    overrides: list[str] | None = None,
    card_terms: dict | None = None,
) -> None:
    """Emit a schema-5.1 tt-model.yaml describing how to build+serve this optimized model as a
    v5.1 container package. Fields the run can't provide (the vLLM adapter dir, plugin) are stated
    as sane defaults/overrides; `tt-model package --container` resolves and validates the rest."""
    _batch = state.get("batch")
    serve_max = _batch or 32
    serve_maxlen = None
    serve_hf_override = None
    # The adapter's own launch settings (on-chip sampling, trace memory, batched-token budget):
    # the speed it was measured at depends on them, so a republish must carry them, not drop them.
    serve_additional = {}
    serve_extra_args = []
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
            _ac = _md.get("additional_config")
            if isinstance(_ac, dict):
                serve_additional = _ac
            _sa = _md.get("serve_args")
            if isinstance(_sa, list):
                serve_extra_args = [str(x) for x in _sa]
            serve_additional = _with_declared_device_sampling(
                serve_additional, Path(checkout), _bm.parent, _md.get("main_class")
            )
    except Exception:
        pass
    # A novel-arch adapter is not a vLLM/Transformers built-in, so the engine must be told which
    # registered TT class to use (there is no built-in to ride); stock arches omit this.
    _serve_args = ["--trust-remote-code"]
    if serve_hf_override:
        _serve_args += ["--hf-overrides", json.dumps({"architectures": [serve_hf_override]})]
    _serve_args += serve_extra_args
    target = _serve_hardware(state, box, arch, hardware, mesh_device)
    arch, hardware, mesh_device = target["arch"], target["hardware"], target["mesh_device"]
    hw_phrase, perf_hw = target["hw_phrase"], target["perf_hw"]
    demo_pkg = f"models.demos.{slug}"
    verify = [f"import {demo_pkg} as m; assert m"] + [
        f"import {m}"
        for m in _adapter_demo_modules(checkout, Path(checkout) / extra_models_dir / slug / "adapter.py", demo_pkg)
    ]
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
        # serve time and the engine dies with AttributeError. After the layout move the old path is
        # only an import alias for the new one, so every layout the checkout has ships.
        *[f"    - {d}" for d in ([d for d in TTT_DIRS if (Path(checkout) / d).is_dir()] or [ttt_dir(checkout)])],
        f"    - models/demos/{slug}",
        '  ubuntu: "22.04"',
        '  python: "3.12"',
        "",
        "runtime:",
        (f"  vllm: {{path: {vllm_path}}}" if vllm_path else f'  vllm: {{version: "{vllm_version}"}}'),
        (
            f"  plugin: {{path: {plugin_path}}}"
            if plugin_path
            else f"  plugin: {{repo: {_PLUGIN_REPO}, ref: {plugin_ref}}}"
        ),
        f"  extra_models_dir: {extra_models_dir}",
        # runtime.lock is optional; only emit it when a real requirements.lock is provided, else the
        # build fails resolving a path that doesn't exist. Deps resolve live without it.
        *([f"  lock: {lock}"] if lock else []),
        *([f"  overrides: {json.dumps(list(overrides))}"] if overrides else []),
        "",
        "serve:",
        "  port: 8000",
        "  block_size: 64",
        f"  max_num_seqs: {serve_max}",
        *([f"  max_model_len: {serve_maxlen}"] if serve_maxlen else []),
        *([f"  additional_config: {json.dumps(serve_additional)}"] if serve_additional else []),
        f"  hardware: {hardware}",
        f"  mesh_device: {mesh_device}",
        "  env:",
        f"    ARCH_NAME: {arch}",
        f"  args: {json.dumps(_serve_args)}",
        "",
        "verify:",
        *(f"  - {json.dumps(v)}" for v in verify),
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
        *_card_terms_lines(card_terms or {}, "text-generation"),
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
    served = state.get("served") or {}
    measured_on = served.get("measured_on") or hw_phrase
    rows = [f"| Hardware | {hw_phrase} |"]
    if measured_on != hw_phrase:
        rows.append(f"| Measured on | {measured_on} |")
    if served.get("rows"):
        # What a customer gets from `tt serve`, measured on this package before it was uploaded. The
        # perf test's figures are bring-up measurements (fixed batch, no server) and go below it.
        rows.append(f"| Concurrent users (`tt serve`) | up to {served.get('max_num_seqs')} |")
        if served.get("max_model_len"):
            rows.append(
                f"| Context per request (`tt serve`) | {int(served['max_model_len']):,} tokens, prompt + answer |"
            )
        for users, isl, osl, ttft, tpot_ms, per_user, total in served["rows"]:
            who = "1 user" if users == 1 else f"{users} users at once"
            rows.append(
                f"| Served, {who} (ISL {isl} / OSL {osl}) | {_fmt(per_user, 1)} tok/s/user, "
                f"{_fmt(total, 0)} tok/s total, TTFT {_fmt(ttft, 2)} s |"
            )
        if thr.get("current") is not None:
            rows.append(
                f"| Optimized decode (on-device perf test, batch {batch}) | {_fmt(thr['current'])} tok/s/user |"
            )
    else:
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
        f"Brought up and measured on {measured_on} through a PCC-gated end-to-end pipeline. "
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


# tenstorrent 1.2.1 accepts any typer>=0.27.2, and typer 0.29.0 dropped the click module its CLI
# imports, so a plain `uv tool install tenstorrent` gives a `tt` that fails on its first command.
_TT_CLI_INSTALL = "uv tool install tenstorrent --with 'typer<0.29'"


def _tt_run_section(repo_id: str | None, base_weights: str | None = None) -> str:
    """How to actually serve this model on Tenstorrent: the `tt` CLI (Tenstorrent's packaged serving
    flow) pulls the published package and serves it as an OpenAI-compatible API, then any OpenAI
    client can query it. This is the same path the other Tenstorrent model packages use."""
    rid = repo_id or "<your-published-repo>"
    model = base_weights or rid
    return (
        "## Run on Tenstorrent\n\n"
        "```bash\n"
        f"{_TT_CLI_INSTALL}   # once \u2014 the Tenstorrent CLI, `tt`\n"
        f"tt model pull {rid}\n"
        f"tt serve {rid}                       # OpenAI-compatible API on http://localhost:20000/v1\n"
        "```\n\n"
        "Then query it with any OpenAI client (`pip install openai`):\n\n"
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
    measured_on = (state.get("served") or {}).get("measured_on") or hw
    tables = (
        "the base-model benchmark tables are shown with a Tenstorrent column (the on-device score when a "
        "bounded TT eval has been run, else 'pending (TT)')."
        if "| Tenstorrent |" in body
        else "the base-model benchmark results are the base-model authors' own, reproduced unchanged."
    )
    intro = (
        f"A Tenstorrent deployment of [`{base_weights}`](https://huggingface.co/{base_weights}) — the same "
        f"weights, brought up on {measured_on}. This card mirrors the upstream model card: the "
        "**Validated on Tenstorrent** section holds figures measured on Tenstorrent hardware. GPU-specific "
        f"deployment/usage has been removed; {tables}"
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


def _demo_tt_dir(checkout, slug) -> Path:
    return Path(checkout) / "models" / "demos" / slug / "tt"


def _demo_tt_module(slug, stem) -> str:
    return "models.demos.%s.tt.%s" % (slug, stem)


def _demo_pipeline_module(checkout, slug):
    """Dotted import path to the demo resident pipeline if it exposes build_pipeline, else None."""
    pf = _demo_tt_dir(checkout, slug) / "pipeline.py"
    try:
        if pf.is_file() and "def build_pipeline" in pf.read_text(errors="replace"):
            return _demo_tt_module(slug, pf.stem)
    except Exception:  # noqa: BLE001
        pass
    return None


# The row-per-request serving hooks a demo module declares to be served by _ROWS_ADAPTER_TMPL.
_SERVING_BUILDER = "build_serving_pipeline"
_SERVING_ROWS = "SERVING_ROWS"
_SERVING_KV = "SERVING_KV_CAPACITY"
_SERVING_DEVICE_PARAMS = "SERVING_DEVICE_PARAMS"
# Extra `vllm serve` flags the pipeline's measured speed depends on (e.g. a prefill token budget that fits
# SERVING_ROWS prompts in one step), rendered into the bundle's serve_args.
_SERVING_SERVE_ARGS = "SERVING_SERVE_ARGS"
# The rendered adapter's first docstring line: a bundle adapter carrying it is the tool's rendering, re-rendered on
# every publish so it follows the template and the demo's current hooks.
_ROWS_ADAPTER_MARK = "Rendered by tt_hw_planner publish-hf from the demo's serving hooks"


_NOT_LITERAL = type("_NotLiteral", (), {"__repr__": lambda self: "an expression, not a literal"})()


def _literal_assignments(tree) -> dict:
    """Module-level NAME = <value> assignments of a parsed module: the literal value, or _NOT_LITERAL."""
    import ast

    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target, value = node.target, node.value
        else:
            continue
        if isinstance(target, ast.Name):
            try:
                out[target.id] = ast.literal_eval(value)
            except (ValueError, TypeError, SyntaxError):
                out[target.id] = _NOT_LITERAL
    return out


def _demo_serving_hooks(checkout, slug):
    """The demo's row-per-request serving hooks, read from source (never imported), or None when no module of its
    tt/ package defines a top-level build_serving_pipeline.

    Returns {"module", "rows", "kv", "device_params", "serve_args"}, or {"module", "problem"} when the hooks are
    incomplete: the module must also assign SERVING_ROWS and SERVING_KV_CAPACITY positive int literals;
    SERVING_DEVICE_PARAMS, when present, is a literal dict of the device-open settings the pipeline needs (the plugin's
    tt config keys); SERVING_SERVE_ARGS, when present, a literal list of extra `vllm serve` arguments."""
    import ast

    tt_dir = _demo_tt_dir(checkout, slug)
    hits = []
    for pf in sorted(tt_dir.glob("*.py")) if tt_dir.is_dir() else []:
        try:
            tree = ast.parse(pf.read_text(errors="replace"))
        except (SyntaxError, ValueError):
            continue
        if any(isinstance(n, ast.FunctionDef) and n.name == _SERVING_BUILDER for n in tree.body):
            hits.append((_demo_tt_module(slug, pf.stem), _literal_assignments(tree)))
    if not hits:
        return None
    module, consts = hits[0]
    if len(hits) > 1:
        return {"module": module, "problem": "%s is defined in %s" % (_SERVING_BUILDER, ", ".join(m for m, _ in hits))}
    rows, kv, params = consts.get(_SERVING_ROWS), consts.get(_SERVING_KV), consts.get(_SERVING_DEVICE_PARAMS, {})
    for name, value in ((_SERVING_ROWS, rows), (_SERVING_KV, kv)):
        if not (isinstance(value, int) and not isinstance(value, bool) and value > 0):
            return {"module": module, "problem": "%s is not a positive int literal (got %r)" % (name, value)}
    if not (isinstance(params, dict) and all(isinstance(k, str) for k in params)):
        return {"module": module, "problem": "%s is not a literal dict (got %r)" % (_SERVING_DEVICE_PARAMS, params)}
    serve_args = consts.get(_SERVING_SERVE_ARGS, [])
    if not (isinstance(serve_args, (list, tuple)) and all(isinstance(a, str) for a in serve_args)):
        return {
            "module": module,
            "problem": "%s is not a literal list of strings (got %r)" % (_SERVING_SERVE_ARGS, serve_args),
        }
    return {"module": module, "rows": rows, "kv": kv, "device_params": params, "serve_args": list(serve_args)}


# A model whose output is not tokens (speech) or whose input the vLLM plugin cannot carry (audio) serves through its
# OWN HTTP server, packaged as tt-model's http-server kind. One module of the demo package declares, as literals, the
# command `tt serve` runs ({host}/{port} placeholders), the log line it prints once it serves, the device mesh it
# opens and, optionally, the pip packages it needs beyond tt-metal's own install. The server must answer GET
# /v1/models: that is how the serve gate knows it came up.
_SERVING_HTTP_COMMAND = "SERVING_HTTP_COMMAND"
_SERVING_HTTP_READY = "SERVING_HTTP_READY_LINE"
_SERVING_HTTP_MESH = "SERVING_HTTP_MESH"
_SERVING_HTTP_PACKAGES = "SERVING_HTTP_PACKAGES"
_HTTP_SERVER_KIND = "http-server"


def _demo_http_server_hooks(demo_dir):
    """The demo's own-HTTP-server hooks, read from source (never imported), or None when no module of the demo
    package (its top level or tt/) assigns SERVING_HTTP_COMMAND.

    Returns {"module", "command", "ready_line", "mesh", "packages"}, or {"module", "problem"} when they are
    incomplete: the command must be a literal list of strings naming {port}, the ready line a non-empty string, the
    mesh a "(rows, cols)" string and the packages, when present, a literal list of strings."""
    import ast
    import re as _re

    demo_dir = Path(demo_dir).resolve()
    checkout = _checkout_of(demo_dir)
    hits = []
    for pf in sorted(list(demo_dir.glob("*.py")) + list((demo_dir / "tt").glob("*.py"))):
        try:
            tree = ast.parse(pf.read_text(errors="replace"))
        except (SyntaxError, ValueError):
            continue
        consts = _literal_assignments(tree)
        if _SERVING_HTTP_COMMAND in consts:
            hits.append((".".join(pf.relative_to(checkout).with_suffix("").parts), consts))
    if not hits:
        return None
    module, consts = hits[0]
    if len(hits) > 1:
        return {
            "module": module,
            "problem": "%s is assigned in %s" % (_SERVING_HTTP_COMMAND, ", ".join(m for m, _ in hits)),
        }
    command, ready, mesh = (consts.get(k) for k in (_SERVING_HTTP_COMMAND, _SERVING_HTTP_READY, _SERVING_HTTP_MESH))
    packages = consts.get(_SERVING_HTTP_PACKAGES, [])
    if not (
        isinstance(command, (list, tuple))
        and all(isinstance(a, str) and a for a in command)
        and any("{port}" in a for a in command)
    ):
        problem = "%s is not a literal list of strings naming {port} (got %r)" % (_SERVING_HTTP_COMMAND, command)
    elif not (isinstance(ready, str) and ready.strip()):
        problem = "%s is not a non-empty string (got %r)" % (_SERVING_HTTP_READY, ready)
    elif not (isinstance(mesh, str) and _re.fullmatch(r"\(\s*\d+\s*,\s*\d+\s*\)", mesh)):
        problem = '%s is not a "(rows, cols)" string (got %r)' % (_SERVING_HTTP_MESH, mesh)
    elif not (isinstance(packages, (list, tuple)) and all(isinstance(p, str) and p for p in packages)):
        problem = "%s is not a literal list of strings (got %r)" % (_SERVING_HTTP_PACKAGES, packages)
    else:
        return {
            "module": module,
            "command": list(command),
            "ready_line": ready,
            "mesh": mesh,
            "packages": list(packages),
        }
    return {"module": module, "problem": problem}


def _rendered_rows_adapter(bundle_dir) -> bool:
    try:
        return _ROWS_ADAPTER_MARK in (Path(bundle_dir) / "adapter.py").read_text(errors="replace")
    except OSError:
        return False


def _pipeline_serve_profile(checkout, slug):
    """(kv_capacity, mesh_str) for the serve note, from the demo pipeline + parallelism manifest."""
    import re as _re

    kv = 2048
    try:
        pf = _demo_tt_dir(checkout, slug) / "pipeline.py"
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


_ROWS_ADAPTER_TMPL = '''# SPDX-License-Identifier: Apache-2.0
"""@MARK@ (@SMOD@); edit those, not this file.

vLLM generator for @SLUG@ (@ARCH@) on Tenstorrent: the demo's own optimized pipeline, one device row per request.
The tt-metal vLLM plugin registers this class as @TTARCH@ via vllm_metadata.json (a bundle registers ahead of the
plugin's built-in generators) and drives it through the generator contract (initialize_vllm_model ->
allocate_kv_cache_per_layer -> warmups -> prefill_forward / decode_forward).

Serve profile: up to @ROWS@ concurrent sequences (--max-num-seqs @ROWS@, the pipeline's device rows), prompt and
generated tokens bounded by the resident KV capacity (--max-model-len @KV@).

Sampling: with --additional-config '{"tt": {"sample_on_device_mode": "decode_only"}}' every decode token is sampled
on device by the pipeline's own sampler, each row by its request's temperature / top_k / top_p (greedy is
temperature 0 or top_k 1), among the row's top pipe.top_k candidates, from the request's seed (a random one when
unset); only the token ids are read back. vLLM samples on the host whenever a request needs what the device does not
do (logprobs, min_p, logit bias, ...): that step returns the logits. A request with a repetition / presence /
frequency penalty is sampled on the host here, from the logits and its token history. Prefill always returns the
logits (vLLM samples the first token).

Each request owns one device row for its whole life: the plugin hands prefill the row (`empty_slots`), the prompt's
K, V go into that row only, and the row is remembered by the request's first KV block id (unique per live request;
block 0 is vLLM's null block, so padding rows never match). Decode writes every row's input token before the step,
and the positions only when they change (a request joined or left). Rows with no request repeat a served row's token
and position; whatever they write into their own cache rows is rewritten by that row's next prefill -- a preempted
request is re-prefilled on resume.

Decode is traced whenever vLLM asks for it (enable_trace): one step of each kind (sampled ids, logits) is captured
and replayed for every token, the per-row inputs written in place before each replay. Under vLLM's async scheduling
a sampled step's ids are read back without waiting and the next step is queued before they arrive (its input ids are
the ones the last step left on device), so the chip runs while vLLM handles the last step. Prefill stays eager (its
length follows the prompt). A prefill that compiled programs the device had not run before may have placed their
buffers over a captured trace's, so the traces are released then and captured again at the next decode.
"""
from __future__ import annotations

import logging
import secrets
from types import SimpleNamespace

import torch

try:
    import @SMOD@ as _serving_module
    import ttnn  # noqa: F401

    _build_serving_pipeline = _serving_module.build_serving_pipeline
    _import_error = None
except Exception as e:  # noqa: BLE001  (available only inside the serving image)
    _build_serving_pipeline = None
    _import_error = e

log = logging.getLogger(__name__)

_SLUG = "@SLUG@"
_SERVING_MODULE = "@SMOD@"
_KV_CAPACITY = @KV@
# Concurrent prompts of this many tokens are compiled at startup, in every power-of-two count up to --max-num-seqs,
# rather than under a client's first requests.
_WARM_PREFILL_LEN = 128
_GREEDY = (1.0, 1, 1.0, None)  # a row's device rule: (1 / temperature, top_k, top_p, seed)
# Decode step kinds in recording order. Every recording records all of them in this order: one kind recorded
# beside the other's held trace lays L1 out differently, and its circular buffers can then clash with that
# trace's L1 outputs.
_KINDS = ("tokens", "logits")


def _request_seed(seed):
    return int(seed) if seed is not None and int(seed) >= 0 else secrets.randbits(31)


class _DeviceIds:
    """A sampled decode step's ids on device (`dev`), each request's device row (`rows`, vLLM's batch order) and,
    once a read is queued, its host destination."""

    __slots__ = ("dev", "rows", "host")

    def __init__(self, dev, rows):
        self.dev, self.rows, self.host = dev, rows, None


class @TTARCH@:
    """Generator contract over the demo's serving pipeline (@SMOD@)."""

    # Read by the plugin off the CLASS before the model exists.
    model_capabilities = {"supports_sample_on_device": True, "supports_async_decode": True}

    def __init__(self, pipe, max_batch_size, max_seq_len, mesh_device=None):
        self.pipe = pipe
        self.mesh_device = mesh_device
        self.max_batch_size = int(max_batch_size)
        self.max_seq_len = int(max_seq_len)
        self.vocab_size = int(getattr(pipe.config, "vocab_size", 0)) or None
        self.rows = int(pipe.batch)
        self.pad_id = int(pipe.pad_id)
        self._row_of_block = {}  # a live request's first KV block id -> its device row
        self._dev_pos = None  # every row's position on device now (None: unknown, write them all)
        self._dev_rule = None  # every row's rule on device now (None: never written)
        self._seeds = [None] * self.rows  # each row's request seed
        self._fresh = set()  # rows prefilled since their request's seed was taken
        self._host_gens = {}  # row -> the host sampler's generator for its request
        self.already_warmed_up_prefill = False
        self._traceable = mesh_device is not None
        self._traces = {}  # "tokens" / "logits" -> (trace id, its output tensor)
        self._untraceable = set()
        self._compiled = set()
        self._last_kind = None  # the last decode step's kind; "tokens": every row's next input is on device

    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size, max_seq_len=None, n_layers=None, **_ignored):
        msl = int(max_seq_len or _KV_CAPACITY)
        if msl > _KV_CAPACITY:
            raise ValueError(
                "%s: resident KV capacity is %d tokens; use --max-model-len %d or less (got %d)."
                % (cls.__name__, _KV_CAPACITY, _KV_CAPACITY, msl)
            )
        if _build_serving_pipeline is None:
            raise RuntimeError("serving module %s not importable: %s" % (_SERVING_MODULE, _import_error))
        pipe = _build_serving_pipeline(mesh_device, capacity=msl, layers=n_layers)
        if not 1 <= int(max_batch_size) <= int(pipe.batch):
            raise ValueError(
                "%s serves up to %d concurrent sequences (the pipeline's device rows); launch vLLM with "
                "--max-num-seqs %d or less (got %s)." % (cls.__name__, pipe.batch, pipe.batch, max_batch_size)
            )
        log.info(
            "built %s serving pipeline (device rows=%s, KV=%s, max seqs=%s)", _SLUG, pipe.batch, msl, max_batch_size
        )
        return cls(pipe, max_batch_size, msl, mesh_device)

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        return [None for _ in per_layer_specs]  # the pipeline owns its cache

    def allocate_kv_cache(self, kv_cache_shape=None, dtype=None, num_layers=None):
        return [None for _ in range(int(num_layers or 0))]

    def _vocab(self, logits):
        return logits[..., : self.vocab_size or logits.shape[-1]].float()

    def _programs(self):
        """Programs the device has compiled (None when the device does not say)."""
        count = getattr(self.mesh_device, "num_program_cache_entries", None)
        try:
            return int(count()) if callable(count) else None
        except Exception:  # noqa: BLE001
            return None

    def _run(self, kind):
        """One decode step on device: "tokens" -> the sampled ids, "logits" -> the logits."""
        return self.pipe.decode_tokens() if kind == "tokens" else self.pipe.decode_logits()

    def _capture(self, kind):
        """Record one decode step of this kind (recording runs nothing on the device); None when it cannot be."""
        tid = None
        try:
            tid = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
            out = self._run(kind)
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
            self._untraceable.add(kind)
            log.warning(
                "%s: %s decode trace capture failed (%s); those steps decode eagerly, much slower", _SLUG, kind, e
            )
            return None
        self._traces[kind] = (tid, out)
        log.info("%s: %s decode trace captured; every such decode token replays it", _SLUG, kind)
        return self._traces[kind]

    def _release_traces(self):
        for tid, _ in self._traces.values():
            try:
                ttnn.release_trace(self.mesh_device, tid)
            except Exception:  # noqa: BLE001
                pass
        self._traces = {}

    def _capture_all(self):
        """Record every compiled step kind afresh, in _KINDS order; a kind that cannot be recorded decodes eagerly."""
        self._release_traces()
        for kind in _KINDS:
            if kind in self._compiled and kind not in self._untraceable:
                self._capture(kind)

    def _step(self, kind, enable_trace):
        # Capturing before the step's programs have run once would compile them mid-capture.
        if enable_trace and self._traceable and kind in self._compiled and kind not in self._untraceable:
            if kind not in self._traces:
                self._capture_all()
            trace = self._traces.get(kind)
            if trace is not None:
                ttnn.execute_trace(self.mesh_device, trace[0], cq_id=0, blocking=False)
                return trace[1]
        # An eager step beside a held trace can clash with its L1 outputs, and a first run compiles programs whose
        # buffers may overlap it.
        self._release_traces()
        out = self._run(kind)
        self._compiled.add(kind)
        return out

    @staticmethod
    def _generator(seed):
        g = torch.Generator()
        g.manual_seed(_request_seed(seed))
        return g

    @staticmethod
    def _penalised(sp, n):
        return any(
            float(sp.presence_penalty[i]) != 0.0
            or float(sp.frequency_penalty[i]) != 0.0
            or float(sp.repetition_penalty[i]) != 1.0
            for i in range(n)
        )

    def _load_rule(self, sp, served):
        """Each served row's request rule -> the device sampler (written only on change); a row new to its request
        takes the request's seed (a random one when unset)."""
        K = int(self.pipe.top_k)
        seeds = getattr(sp, "seed", None)
        rule = list(self._dev_rule or [_GREEDY] * self.rows)
        for i, row in enumerate(served):
            if row in self._fresh:
                self._seeds[row] = _request_seed(seeds[i] if seeds is not None else None)
                self._fresh.discard(row)
            temp, top_k, top_p = float(sp.temperature[i]), int(sp.top_k[i]), float(sp.top_p[i])
            if temp <= 0.0 or top_k == 1:
                rule[row] = _GREEDY
            else:
                rule[row] = (1.0 / temp, min(top_k, K) if top_k > 0 else K, min(top_p, 1.0), self._seeds[row])
        if rule != self._dev_rule:
            inv_t, k, p, row_seeds = (list(c) for c in zip(*rule))
            self.pipe.set_sampling(inv_t, k, p, row_seeds)
            self._dev_rule = rule

    def _host_sample(self, logits, sp, served, prompt_tokens, output_tokens):
        """vLLM's rule on the host for a step whose requests carry penalties: logits [n, V] -> ids [n] int32."""
        ids = torch.empty((logits.shape[0],), dtype=torch.int32)
        seeds = getattr(sp, "seed", None)
        for i, row in enumerate(served):
            x = logits[i].clone()
            out = output_tokens[i][output_tokens[i] >= 0].long() if output_tokens is not None else x.new_empty(0).long()
            rep, pres, freq = (
                float(sp.repetition_penalty[i]),
                float(sp.presence_penalty[i]),
                float(sp.frequency_penalty[i]),
            )
            if rep != 1.0:
                seen = (
                    out if prompt_tokens is None else torch.cat([out, prompt_tokens[i][prompt_tokens[i] >= 0].long()])
                )
                seen = seen[seen < x.shape[0]].unique()
                x[seen] = torch.where(x[seen] > 0, x[seen] / rep, x[seen] * rep)
            if pres != 0.0 or freq != 0.0:
                count = torch.bincount(out[out < x.shape[0]], minlength=x.shape[0]).float()
                x -= freq * count + pres * (count > 0).float()
            temp, top_k, top_p = float(sp.temperature[i]), int(sp.top_k[i]), float(sp.top_p[i])
            if temp <= 0.0 or top_k == 1:
                ids[i] = int(torch.argmax(x))
                continue
            x = x / temp
            if 0 < top_k < x.shape[0]:
                x[x < torch.topk(x, top_k).values[-1]] = float("-inf")
            if top_p < 1.0:
                order = torch.argsort(x, descending=True)
                probs = torch.softmax(x[order], dim=-1)
                drop = torch.cumsum(probs, dim=-1) - probs >= top_p  # exclusive cumulative probability
                x[order[drop]] = float("-inf")
            if row not in self._host_gens:
                self._host_gens[row] = self._generator(seeds[i] if seeds is not None else None)
            ids[i] = int(torch.multinomial(torch.softmax(x, dim=-1), 1, generator=self._host_gens[row]))
        return ids

    def prefill_forward(
        self, tokens, page_table=None, *, prompt_lens=None, start_pos=None, empty_slots=None, **_compat
    ):
        n = int(tokens.shape[0])
        if start_pos is not None and int(torch.as_tensor(start_pos).max()) != 0:
            raise ValueError("chunked / prefix-cached prefill is not supported by this generator")
        rows = [int(s) for s in empty_slots] if empty_slots is not None else list(range(n))
        if len(rows) != n:
            raise ValueError("prefill of %d sequences got %d slots" % (n, len(rows)))
        lens = [int(prompt_lens[i]) if prompt_lens is not None else int(tokens.shape[1]) for i in range(n)]
        for T in lens:
            if T > self.max_seq_len:
                raise ValueError("prompt of %d tokens exceeds max_model_len=%d" % (T, self.max_seq_len))
        before = self._programs()
        logits = self.pipe.prefill_rows([tokens[i, : lens[i]].to(torch.int64).tolist() for i in range(n)], rows)
        if before is None or self._programs() != before:
            self._release_traces()
        for i, row in enumerate(rows):
            if page_table is not None:
                self._row_of_block[int(page_table[i, 0])] = row
            self._fresh.add(row)
            self._host_gens.pop(row, None)
        self._last_kind = None
        return self._vocab(torch.as_tensor(logits)).reshape(n, 1, -1)

    def decode_forward(
        self,
        tokens,
        start_pos=None,
        page_table=None,
        *,
        enable_trace=False,
        sampling_params=None,
        prompt_tokens=None,
        output_tokens=None,
        reset_batch=None,
        read_from_device=True,
        **_compat,
    ):
        tokens = torch.as_tensor(tokens).reshape(-1)
        n = int(tokens.shape[0])
        positions = (
            torch.as_tensor(start_pos).reshape(-1) if start_pos is not None else torch.zeros(n, dtype=torch.int64)
        )
        dev = self._dev_pos
        tok = torch.full((self.rows,), self.pad_id, dtype=torch.int32)
        pos = dev.clone() if dev is not None else torch.zeros((self.rows,), dtype=torch.int32)
        idle = torch.ones((self.rows,), dtype=torch.bool)
        served = []  # device row of each request, in vLLM's batch order (padding rows trail)
        for i in range(n):
            p = int(positions[i])
            if p < 0:
                break  # vLLM's padding rows
            row = self._row_of_block.get(int(page_table[i, 0])) if page_table is not None else i
            if row is None:
                raise RuntimeError("decode row %d (first KV block %d) was never prefilled" % (i, int(page_table[i, 0])))
            tok[row], pos[row], idle[row] = int(tokens[i]), p, False
            served.append(row)
        if served:
            tok[idle], pos[idle] = tok[served[0]], pos[served[0]]
        # vLLM's async decode submits a step before the last one's ids are back and applied, so its tokens are
        # placeholders and its positions one step behind: after a sampled step over the same rows (reset_batch
        # False) each row's next input id and position are the ones that step left on device.
        if reset_batch is False and self._last_kind == "tokens" and dev is not None:
            tok, pos = None, dev
        self.pipe.set_decode_rows(tok, None if dev is not None and torch.equal(pos, dev) else pos)
        self._dev_pos = None
        sp = sampling_params
        on_device = sp is not None and not self._penalised(sp, len(served))
        if on_device:
            self._load_rule(sp, served)
        kind = "tokens" if on_device else "logits"
        out = self._step(kind, enable_trace)
        self._last_kind = kind
        self._dev_pos = pos + 1
        rows = torch.tensor(served, dtype=torch.long)
        if on_device:
            ids = _DeviceIds(out, rows)
            return ids if not read_from_device else self.process_decode_output_host(ids)
        logits = self._vocab(self.pipe.logits_to_torch(out)).index_select(0, rows)
        if sp is not None:  # vLLM wants ids, but a penalty needs the token history the device does not keep
            return self._host_sample(logits, sp, served, prompt_tokens, output_tokens)
        return logits.unsqueeze(1)

    def read_decode_output(self, tt_out, async_read=False, **_compat):
        """vLLM's async decode: start reading a sampled step's ids back without waiting (the copy is queued
        behind the step); returns (the pending output, [its event]). Host outputs pass through."""
        if not isinstance(tt_out, _DeviceIds):
            return (tt_out, []) if async_read else tt_out
        if not async_read:
            return self.process_decode_output_host(tt_out)
        tt_out.host = tt_out.dev.cpu(blocking=False)
        return tt_out, [ttnn.record_event(self.mesh_device, 0)]

    def process_decode_output_host(self, tt_out, is_tokens=True, **_compat):
        """A sampled step's ids (read, or still on device) -> int32 [n] in vLLM's batch order."""
        if not isinstance(tt_out, _DeviceIds):
            return tt_out
        host = tt_out.host if tt_out.host is not None else tt_out.dev.cpu(blocking=True)
        ids = ttnn.to_torch(ttnn.get_device_tensors(host)[0]).reshape(-1)[: self.rows].to(torch.int64)  # may be uint32
        return ids.index_select(0, tt_out.rows).to(torch.int32)

    def warmup_model_prefill(self, *, kv_cache=None, can_sample_on_device=False, enable_trace=False, **_compat):
        if enable_trace:
            return  # prefill stays eager: its length follows the prompt
        bos = getattr(self.pipe.config, "bos_token_id", None)
        bos = int(bos) if bos is not None else self.pad_id
        T = min(_WARM_PREFILL_LEN, self.max_seq_len - 2)
        n = 1
        while n <= self.max_batch_size:
            self.prefill_forward(
                torch.full((n, T), bos, dtype=torch.int64), None, prompt_lens=[T] * n, empty_slots=list(range(n))
            )
            n *= 2
        self.already_warmed_up_prefill = True

    def warmup_model_decode(
        self,
        *,
        kv_cache=None,
        max_batch_size=1,
        num_blocks=0,
        can_sample_on_device=False,
        enable_trace=False,
        **_compat,
    ):
        if enable_trace:
            if not self._traceable:
                log.warning(
                    "%s: vLLM asked for traced decode without a mesh device; decoding eagerly, which is much slower",
                    _SLUG,
                )
                return
            self._capture_all()
            return
        if not self.already_warmed_up_prefill:
            return
        T = min(_WARM_PREFILL_LEN, self.max_seq_len - 2)
        self.decode_forward(torch.zeros((1, 1), dtype=torch.int64), torch.tensor([T]), None)
        greedy = SimpleNamespace(
            temperature=[0.0],
            top_k=[1],
            top_p=[1.0],
            presence_penalty=[0.0],
            frequency_penalty=[0.0],
            repetition_penalty=[1.0],
            seed=[None],
        )
        self.decode_forward(torch.zeros((1, 1), dtype=torch.int64), torch.tensor([T + 1]), None, sampling_params=greedy)

    def cleanup(self):
        self._release_traces()
'''


def _rows_adapter_src(slug, arch, tt_arch, hooks) -> str:
    """Render the row-per-request generator over the demo's serving hooks (see _demo_serving_hooks)."""
    return (
        _ROWS_ADAPTER_TMPL.replace("@MARK@", _ROWS_ADAPTER_MARK)
        .replace("@SLUG@", slug)
        .replace("@TTARCH@", tt_arch)
        .replace("@ARCH@", arch)
        .replace("@SMOD@", hooks["module"])
        .replace("@ROWS@", str(hooks["rows"]))
        .replace("@KV@", str(hooks["kv"]))
    )


def _scaffold_vllm_bundle(
    checkout: Path, extra_models_dir: str, arch: str, model_type: str | None, weights: str | None, slug: str
) -> tuple[bool, bool, str]:
    """Create the vLLM adapter bundle (vllm_metadata.json + adapter.py) under the checkout if absent,
    so the container manifest is complete for ANY model. Returns (created, is_stub, path).

    - A demo declaring the row-per-request serving hooks (_demo_serving_hooks), any arch -> adapter is
      _ROWS_ADAPTER_TMPL over them: the demo's own pipeline serves, servable (is_stub False).
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

    hooks = _demo_serving_hooks(checkout, slug)
    if hooks and hooks.get("problem"):
        print(f"  [publish-hf] serving hooks in {hooks['module']} not used: {hooks['problem']}")
    elif hooks:
        serve_args = hooks.get("serve_args") or []
        rows_meta = {
            "arch": arch,
            "main_class": "adapter:" + tt_arch,
            "weights": weights or "REPLACE/with-base-weights-repo",
            "note": "Generator over the demo's serving pipeline (%s), one device row per request; serve with "
            "--max-num-seqs %d --max-model-len %d%s."
            % (hooks["module"], hooks["rows"], hooks["kv"], "".join(" " + a for a in serve_args)),
            "max_num_seqs": hooks["rows"],
            "max_model_len": hooks["kv"],
        }
        if hooks["device_params"]:
            rows_meta["additional_config"] = {_PLUGIN_CONFIG_KEY: dict(hooks["device_params"])}
        if serve_args:
            rows_meta["serve_args"] = list(serve_args)
        if is_stub:  # not a plugin built-in: the engine must be told which registered class serves it
            rows_meta["serve_hf_override_arch"] = tt_arch
        meta.write_text(json.dumps(rows_meta, indent=2) + "\n")
        (bundle / "adapter.py").write_text(_rows_adapter_src(slug, arch, tt_arch, hooks))
        return created, False, str(bundle)

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


def _full_depth(demo_dir, weights: str | None, token: str | None) -> int | None:
    """The model's full layer count, from a config.json in the demo or else the base-weights repo."""
    import glob as _g

    cfgs = []
    for cfg in _g.glob(str(Path(demo_dir) / "**" / "config.json"), recursive=True):
        try:
            cfgs.append(json.loads(Path(cfg).read_text()))
        except Exception:  # noqa: BLE001
            continue
    if not cfgs and weights:
        try:
            from huggingface_hub import hf_hub_download

            cfgs.append(json.load(open(hf_hub_download(repo_id=weights, filename="config.json", token=token))))
        except Exception:  # noqa: BLE001
            pass
    for d in cfgs:
        for c in (d, d.get("text_config") or {}):
            for k in ("num_hidden_layers", "n_layers", "num_layers"):
                if isinstance(c.get(k), int) and c[k] > 0:
                    return c[k]
    return None


def _bench_layers_env(args, demo_dir) -> dict:
    """TT_PERF_LAYERS for the card's sweep: --bench-layers, else the model's full depth. Never the
    perf test's own default or a cap inherited from the shell: nemotron's test defaults to 7 of its 52
    layers, and a card filled from that showed a 7-layer ~80 tok/s/user as the model's speed."""
    if getattr(args, "bench_layers", None):
        return {"TT_PERF_LAYERS": str(args.bench_layers)}
    full = _full_depth(demo_dir, getattr(args, "weights", None), _hf_token(args))
    return {"TT_PERF_LAYERS": str(full)} if full else {}


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
    layers_env = _bench_layers_env(args, demo_dir)
    base_env = {k: v for k, v in os.environ.items() if k != "TT_PERF_LAYERS"}
    print(
        f"  [publish-hf] bench: on-device sweep via {perf_node} — ISL {isls} x Users {users_grid}, OSL {osl}, "
        f"layers {layers_env.get('TT_PERF_LAYERS', 'the test default (no depth found in config.json)')}…"
    )

    def _run(isl, users):
        env = dict(
            base_env,
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


def _adapter_never_traces(bundle_dir) -> bool:
    """True when the bundle's adapter runs decode itself (defines decode_forward) but never replays a
    trace, so every served token runs eagerly. An adapter riding a stock generator inherits that
    generator's traced decode and is not flagged."""
    try:
        src = (Path(bundle_dir) / "adapter.py").read_text()
    except Exception:  # noqa: BLE001
        return False
    return "def decode_forward" in src and "execute_trace" not in src


_SERVED_TITLE = "Served speed (tt serve)"
# What a served adapter logs when it decodes without its trace. The server still starts and answers,
# at a fraction of the optimized speed: kolibri_1's package served ~9.5 tok/s against a traced 39.47.
_EAGER_SERVE_MARKS = ("decoding eagerly", "decode eagerly", "decode trace capture failed")
_SERVE_GATE_MIN = 0.7


def _eager_serve_lines(log: str) -> list:
    return [ln.strip() for ln in (log or "").splitlines() if any(m in ln for m in _EAGER_SERVE_MARKS)]


def _time_served_decode(base_url: str, model: str, users: int, osl: int = 128, timeout: int = 1800) -> dict:
    """``users`` concurrent streamed greedy completions of exactly ``osl`` tokens, each with its own
    ~128-token prompt. Mean TTFT (s) and TPOT (ms), per-user and total decode tok/s. ``ignore_eos`` alone
    fixes the length: ``min_tokens`` would add a logits processor, which moves TT sampling off the device
    and so times a slower path than a plain client request takes."""
    import threading
    import time
    import urllib.request

    res = [None] * users

    def one(u):
        body = {
            "model": model,
            "prompt": f"Reader {u}: " + " ".join(["The quick brown fox jumps over the lazy dog."] * 12),
            "max_tokens": osl,
            "ignore_eos": True,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        req = urllib.request.Request(
            base_url.rstrip("/") + "/v1/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        t0, first, last, chunks, n = time.time(), None, None, 0, 0
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                for raw in r:
                    ln = raw.decode(errors="replace").strip()
                    if not ln.startswith("data:") or ln[5:].strip() == "[DONE]":
                        continue
                    d = json.loads(ln[5:])
                    n = (d.get("usage") or {}).get("completion_tokens") or n
                    if any(c.get("text") for c in d.get("choices") or []):
                        last = time.time()
                        first = first or last
                        chunks += 1
        except Exception as e:  # noqa: BLE001
            res[u] = {"error": str(e)}
            return
        n = n or chunks
        res[u] = {
            "ttft": (first - t0) if first else None,
            "tpot": (last - first) / (n - 1) if first and n > 1 else None,
        }

    threads = [threading.Thread(target=one, args=(u,)) for u in range(users)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    ok = [r for r in res if r and r.get("tpot")]
    errors = [r["error"] for r in res if r and r.get("error")]
    if not ok:
        return {"users": users, "completed": 0, "error": errors[0] if errors else "no tokens came back"}
    tpot = sum(r["tpot"] for r in ok) / len(ok)
    return {
        "users": users,
        "completed": len(ok),
        "errors": errors,
        "ttft": sum(r["ttft"] for r in ok) / len(ok),
        "tpot_ms": 1000.0 * tpot,
        "per_user": 1.0 / tpot,
        "total": len(ok) / tpot,
    }


def _serve_gate_problems(
    optimized,
    one_user: dict,
    many: dict | None,
    eager_lines: list,
    min_ratio: float,
    optimized_batch: int | None = None,
) -> list:
    """Why the built package must not be uploaded; empty when it serves at its optimized speed.

    `optimized_batch` is the batch the optimized tok/s/user was measured at. When the concurrent
    timing ran that many users, their per-user speed is held to the same bar as the 1-user one: a
    package sampling every decode step on the host served 1 user near its optimized speed and 32 at
    about half of it, and the 1-user check alone passed it."""
    problems = []
    if eager_lines:
        problems.append("the server decodes without its trace (%s)" % eager_lines[0][:240])
    if not one_user or not one_user.get("per_user"):
        problems.append("a 1-user request did not complete (%s)" % ((one_user or {}).get("error") or "no reply"))
    elif optimized and one_user["per_user"] < min_ratio * optimized:
        problems.append(
            "1 user decodes at %.1f tok/s, under %d%% of the %.1f tok/s/user the model was optimized to"
            % (one_user["per_user"], round(100 * min_ratio), optimized)
        )
    if many and many.get("completed", 0) < many["users"]:
        problems.append(
            "only %d of %d concurrent requests completed (%s)"
            % (many.get("completed", 0), many["users"], (many.get("errors") or [many.get("error")])[0])
        )
    elif (
        many
        and many.get("per_user")
        and optimized
        and optimized_batch
        and int(optimized_batch) == int(many["users"])
        and many["per_user"] < min_ratio * optimized
    ):
        problems.append(
            "%d users at once decode at %.1f tok/s/user, under %d%% of the %.1f tok/s/user the model was "
            "optimized to at batch %d"
            % (many["users"], many["per_user"], round(100 * min_ratio), optimized, optimized_batch)
        )
    return problems


def _served_speed_gate(args, state: dict, staged) -> tuple:
    """Serve the BUILT package the way a customer will (``tt-model serve`` on its manifest) BEFORE it
    is uploaded, and time real requests: 1 user, then max_num_seqs users at once. (ok, served): not ok
    when the server cannot serve, decodes without its trace, drops concurrent requests, or decodes 1
    user -- or, when the model was optimized at max_num_seqs users, that many users -- under
    --serve-gate-min (default 70%) of the optimized tok/s/user. A clean startup proves
    nothing about speed: kolibri_1's package started cleanly and served a quarter of its traced speed."""
    import datetime as _dt
    import socket
    import time
    import urllib.request

    if getattr(args, "no_serve_gate", False):
        print("  [publish-hf] serve gate: skipped (--no-serve-gate); uploading without timing the served package.")
        return True, None
    manifest = Path(staged) / "tt_kernel_manifest.json"
    docker = shutil.which("docker")
    if not docker or not manifest.is_file():
        print(
            f"  [publish-hf] serve gate: cannot serve the build here ({'no docker' if not docker else 'no ' + str(manifest)})."
        )
        return False, None
    try:
        mani = json.loads(manifest.read_text())
    except Exception:  # noqa: BLE001
        mani = {}
    seqs = int(((mani.get("container") or {}).get("serve") or {}).get("max_num_seqs") or 1)
    port = 20500
    while port < 20600:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                break
        port += 1
    base = f"http://127.0.0.1:{port}"
    ttm = getattr(args, "tt_model_bin", None) or "tt-model"
    cname_key = "tt-model-%s-" % (mani.get("name") or Path(staged).name)

    def _containers():
        ps = subprocess.run(
            [docker, "ps", "-a", "--filter", "name=" + cname_key, "--format", "{{.Names}} {{.Status}}"],
            capture_output=True,
            text=True,
        ).stdout
        return [ln.split(" ", 1) for ln in ps.splitlines() if ln.strip()]

    def _logs():
        return "".join(
            (r.stdout or "") + (r.stderr or "")
            for r in (subprocess.run([docker, "logs", c[0]], capture_output=True, text=True) for c in _containers())
        )

    print(
        f"  [publish-hf] serve gate: serving the built package on port {port} before upload (first start compiles kernels) …"
    )
    timeout = int(getattr(args, "serve_gate_timeout", None) or 5400)
    model, one, many, log = None, None, None, ""
    try:
        st = subprocess.run(
            [ttm, "serve", str(manifest), "--detach", "--port", str(port)],
            capture_output=True,
            text=True,
            timeout=4 * 3600,
        )
        if st.returncode != 0:
            print(
                "  [publish-hf] serve gate: tt-model serve failed:\n" + ((st.stdout or "") + (st.stderr or ""))[-1500:]
            )
            return False, None
        t0 = time.time()
        while time.time() < t0 + timeout:
            try:
                with urllib.request.urlopen(base + "/v1/models", timeout=10) as r:
                    model = json.loads(r.read())["data"][0]["id"]
                    break
            except Exception:  # noqa: BLE001
                pass
            cs = _containers()
            if (cs and all(c[1].startswith("Exited") for c in cs)) or (not cs and time.time() > t0 + 300):
                break
            time.sleep(15)
        if model is None:
            print("  [publish-hf] serve gate: the built package did not start serving:\n" + _logs()[-1500:])
            return False, None
        if (mani.get("container") or {}).get("kind") == _HTTP_SERVER_KIND:
            print(
                f"  [publish-hf] serve gate: PASS -- the package's own HTTP server serves {model}; it streams no "
                "tokens, so there is no decode speed to time."
            )
            return True, None
        print(f"  [publish-hf] serve gate: serving {model}; timing 1 user, then {seqs} at once …")
        _time_served_decode(base, model, 1, osl=16)  # first request: one-time compiles and trace capture
        one = _time_served_decode(base, model, 1)
        if seqs > 1:
            _time_served_decode(base, model, seqs, osl=16)
            many = _time_served_decode(base, model, seqs)
        log = _logs()
    except Exception as e:  # noqa: BLE001
        print(f"  [publish-hf] serve gate: could not serve and time the build ({e}).")
        return False, None
    finally:
        for c in _containers():
            subprocess.run([docker, "rm", "-f", c[0]], capture_output=True, text=True)
    optimized = (state.get("throughput") or {}).get("current")
    min_ratio = float(getattr(args, "serve_gate_min", None) or _SERVE_GATE_MIN)
    for r in (one, many):
        if r and r.get("per_user"):
            print(
                f"  [publish-hf] serve gate: {r['users']} user(s): {r['per_user']:.1f} tok/s/user, "
                f"{r['total']:.0f} tok/s total, TTFT {r['ttft']:.2f} s"
            )
    if optimized:
        print(f"  [publish-hf] serve gate: optimized on-device decode: {optimized:.1f} tok/s/user")
    problems = _serve_gate_problems(
        optimized, one, many, _eager_serve_lines(log), min_ratio, optimized_batch=state.get("batch")
    )
    if problems:
        for p in problems:
            print("  [publish-hf] serve gate: FAIL -> " + p)
        return False, None
    rows = [
        (r["users"], 128, 128, r["ttft"], r["tpot_ms"], r["per_user"], r["total"])
        for r in (one, many)
        if r and r.get("per_user")
    ]
    print("  [publish-hf] serve gate: PASS")
    return True, {
        "rows": rows,
        "max_num_seqs": seqs,
        "max_model_len": ((mani.get("container") or {}).get("serve") or {}).get("max_model_len"),
        "optimized": optimized,
        "date": _dt.date.today().isoformat(),
    }


def _served_headline(served: dict, perf_hw: str) -> str:
    """The card's one-line speed from the serve gate's rows: what `tt serve` measured on this package."""
    rows = sorted(served["rows"], key=lambda r: r[0], reverse=True)
    parts = [
        (
            f"{per_user:.1f} tok/s/user with 1 user"
            if users == 1
            else f"{per_user:.1f} tok/s/user with {users} users at once ({total:,.0f} tok/s total)"
        )
        for users, _isl, _osl, _ttft, _tpot, per_user, total in rows
    ]
    _u, isl, osl, *_ = rows[0]
    return (
        f"{' and '.join(parts)}, served by `tt serve` on {perf_hw} "
        f"(~{isl}-token prompts, {osl} output tokens each, greedy)."
    )


def _restate_card_speed(staged, headline: str) -> bool:
    """Replace the built package's card speed (the bring-up number tt-model wrote into README.md and the manifest's
    container.card.performance) with `headline`. True when the manifest held one to replace."""
    mpath, rpath = Path(staged) / "tt_kernel_manifest.json", Path(staged) / "README.md"
    try:
        mani = json.loads(mpath.read_text())
    except (OSError, ValueError):
        return False
    card = (mani.get("container") or {}).get("card")
    old = str((card or {}).get("performance") or "").strip()
    if not old:
        return False
    card["performance"] = headline + "\n"
    mpath.write_text(json.dumps(mani, indent=2))
    try:
        readme = rpath.read_text()
    except OSError:
        return True
    if old in readme:
        rpath.write_text(readme.replace(old, headline, 1))
    else:
        print(f"  [publish-hf] card speed: {rpath} does not carry the manifest's line; only the manifest was restated.")
    return True


def _served_card_body(served: dict) -> str:
    tbl = [
        "| Users (at once) | ISL | OSL | TTFT (s) | TPOT (ms) | tok/s/user | tok/s total |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for users, isl, osl, ttft, tpot_ms, per_user, total in served["rows"]:
        tbl.append(f"| {users} | ~{isl} | {osl} | {ttft:.2f} | {tpot_ms:.1f} | {per_user:.1f} | {total:,.0f} |")
    opt = served.get("optimized")
    tail = (
        f"\n\nThe model's optimized on-device decode (its perf test, no server) is {opt:.1f} tok/s/user." if opt else ""
    )
    return (
        f"Measured on this package with `tt serve` before it was uploaded: greedy decoding, each user "
        f"streaming its own prompt, up to {served['max_num_seqs']} users at once. TPOT is the time per "
        f"output token after the first; tok/s/user = 1000 / TPOT. _Measured {served['date']}._\n\n"
        + "\n".join(tbl)
        + tail
    )


def _git_info(path):
    """(commit_sha, origin_url) for a git checkout, or (None, None). Never raises."""
    import subprocess

    def g(*a):
        try:
            return subprocess.run(["git", "-C", str(path), *a], capture_output=True, text=True).stdout.strip()
        except Exception:
            return ""

    return (g("rev-parse", "HEAD") or None, g("remote", "get-url", "origin") or None)


def _recipe_vllm_version(recipe: str):
    """The one vLLM release a plugin install recipe installs, or None when it names none or several."""
    import re

    pins = set(re.findall(r"\bvllm==([0-9][0-9A-Za-z.+!-]*)", recipe or ""))
    return pins.pop() if len(pins) == 1 else None


def _plugin_serving_pair(repo: str, ref: str):
    """(commit, vLLM release) for the plugin at ``ref``, or None when either cannot be read: the commit ``ref``
    names now, and the release that commit's install recipe installs. A plugin imports vLLM modules of the release
    it is built against, so both come from one commit; pinning the commit keeps a later move of ``ref`` from
    pairing the image's plugin with another vLLM."""
    with tempfile.TemporaryDirectory(prefix="ttpub_plugin_") as d:

        def git(*a):
            return subprocess.run(["git", "-C", d, *a], capture_output=True, text=True, timeout=300)

        try:
            if git("init", "-q").returncode or git("fetch", "-q", "--depth", "1", repo, ref).returncode:
                return None
            commit = git("rev-parse", "FETCH_HEAD").stdout.strip()
            version = _recipe_vllm_version(git("show", f"FETCH_HEAD:{_PLUGIN_INSTALL_RECIPE}").stdout)
        except (OSError, subprocess.SubprocessError):
            return None
    return (commit, version) if commit and version else None


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


# What a vLLM install must not move: torch with the torchvision built for it (ttnn's extension modules are
# built against that torch) and the transformers stack the model code was validated with.
_VALIDATED_PACKAGES = ("torch", "torchvision", "transformers", "tokenizers")


def _auto_constraint_file(checkout):
    """Write a minimal pip CONSTRAINT of _validated_pins, so a source-built vLLM cannot pull a torch/transformers
    that mismatches the ttnn modules or the model code. Returns path or None."""
    import tempfile
    from pathlib import Path as _P

    pins = _validated_pins(checkout)
    if not pins:
        return None
    f = _P(tempfile.mkdtemp(prefix="ttpub_constraint_")) / "constraint.lock"
    f.write_text("\n".join(pins) + "\n")
    return str(f)


def _validated_pins(checkout):
    """``pkg==version`` for each of _VALIDATED_PACKAGES installed in the checkout's python env, the env the model
    was validated in."""
    import subprocess
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
    for pkg in _VALIDATED_PACKAGES:
        v = ver(pkg)
        if v:
            pins.append(f"{pkg}=={v.split('+')[0]}")
    return pins


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


def _pinned_packages(checkout, packages) -> list:
    """Each bare package name pinned to the version installed in the checkout's python env (the env the model was
    validated in); a requirement that already names a version, or a package that env lacks, is kept as given."""
    import re as _re

    py = Path(checkout) / "python_env" / "bin" / "python"
    bare = [p for p in packages if _re.fullmatch(r"[A-Za-z0-9_.-]+", p)]
    found = {}
    if bare and py.is_file():
        code = (
            "import importlib.metadata as m, json, sys\n"
            "out = {}\n"
            "for p in sys.argv[1:]:\n"
            "    try:\n"
            "        out[p] = m.version(p)\n"
            "    except m.PackageNotFoundError:\n"
            "        pass\n"
            "print(json.dumps(out))\n"
        )
        try:
            found = json.loads(subprocess.run([str(py), "-c", code, *bare], capture_output=True, text=True).stdout)
        except (OSError, ValueError):
            found = {}
    return [f"{p}=={found[p].split('+')[0]}" if p in found else p for p in packages]


def _write_http_server_yaml(
    path: Path,
    *,
    state: dict,
    slug: str,
    repo_id: str,
    checkout: str,
    demo_rel: str,
    weights: str | None,
    box: str | None,
    arch: str | None,
    hardware: str | None,
    http: dict,
    packages: list,
    commit: str | None,
    card_terms: dict | None = None,
) -> None:
    """A schema-5.1 tt-model.yaml for a demo served by its own HTTP server (tt-model's http-server kind): the
    hooks' command and ready line, the packages it needs and the mesh it opens; no vLLM engine, adapter or KV
    settings. Raises ValueError when there is no serve target."""
    target = _serve_hardware(state, box, arch, hardware, http["mesh"])
    arch, hardware, mesh_device = target["arch"], target["hardware"], target["mesh_device"]
    hw_phrase = target["hw_phrase"]
    lines = [
        'schema: "5.1"',
        f"repo: {repo_id}",
        f"name: {slug}",
        f"weights: {weights or 'REPLACE/with-base-weights-repo'}",
        f"kind: {_HTTP_SERVER_KIND}",
        f"arch: {arch}",
        "",
        "source:",
        f"  tt_metal: {checkout}",
        "  code:",
        "    - models/common",
        *[f"    - {d}" for d in ([d for d in TTT_DIRS if (Path(checkout) / d).is_dir()] or [ttt_dir(checkout)])],
        f"    - {demo_rel}",
        '  ubuntu: "22.04"',
        '  python: "3.12"',
        "",
        "runtime:",
        f"  command: {json.dumps(http['command'])}",
        f"  ready_line: {json.dumps(http['ready_line'])}",
        *([f"  packages: {json.dumps(packages)}"] if packages else []),
        "",
        "serve:",
        "  port: 8000",
        f"  hardware: {hardware}",
        f"  mesh_device: {mesh_device}",
        "  env:",
        f"    ARCH_NAME: {arch}",
        "",
        "verify:",
        f"  - {json.dumps('import ' + http['module'])}",
        "",
        "card:",
        "  description: >",
        f"    {slug}, brought up and optimized on {hw_phrase}, served by its own HTTP server.",
        "  performance: >",
        "    measured on Tenstorrent; see the card.",
        "  limitations: >",
        f"    Validated on {hw_phrase}.",
        f"  architecture: autoport ({slug})",
        "  status: Experimental community bring-up",
        *_card_terms_lines(card_terms or {}, None),
    ]
    if weights:
        lines.append(f"  base_model: [{weights}]")
    if commit:
        lines.append(f"  related: tt-metal commit {commit}")
    path.write_text("\n".join(lines) + "\n")


def _run_http_server_container(args, state: dict, slug: str, demo_dir, commit: str | None, http: dict) -> int:
    """Package a demo that declares its own HTTP server (the SERVING_HTTP_* hooks) as tt-model's http-server kind:
    no vLLM adapter is scaffolded and no engine is installed; `tt serve` runs the demo's server."""
    import tempfile

    if http.get("problem"):
        print(f"  [publish-hf] NOT publishing: {http['module']} declares its own HTTP server, but {http['problem']}.")
        return 2
    checkout = _checkout_of(demo_dir)
    demo_rel = Path(demo_dir).resolve().relative_to(checkout).as_posix()
    packages = _pinned_packages(checkout, http["packages"])
    print(
        f"  [publish-hf] {http['module']} serves the model over its own HTTP server: packaging it as tt-model's "
        f"{_HTTP_SERVER_KIND} kind, `tt serve` runs `{' '.join(http['command'])}` (no vLLM adapter)"
    )
    yaml_path = Path(tempfile.mkdtemp(prefix="tt_ttmodel_")) / "tt-model.yaml"
    try:
        _write_http_server_yaml(
            yaml_path,
            state=state,
            slug=slug,
            repo_id=args.repo,
            checkout=str(checkout),
            demo_rel=demo_rel,
            weights=getattr(args, "weights", None),
            box=getattr(args, "box", None),
            arch=getattr(args, "arch", None),
            hardware=getattr(args, "hardware", None),
            http=http,
            packages=packages,
            commit=commit,
            card_terms=_upstream_card_terms(getattr(args, "weights", None), _hf_token(args)),
        )
    except ValueError as exc:
        print(f"  [publish-hf] {exc}")
        return 2
    ttm = getattr(args, "tt_model_bin", None) or "tt-model"
    out = getattr(args, "out", None) or str(Path.home() / "tt-model-builds")
    return _package_and_publish(
        args, state, slug, checkout, demo_dir, yaml_path, ttm, out, True, http["mesh"], http["ready_line"]
    )


def _run_container(args, state: dict, slug: str, demo_dir, commit: str | None) -> int:
    """Build + push a real v5.1 container bundle via tt-model (exactly like the published TT repos):
    generate tt-model.yaml, then `tt-model package --container` (2.5-4h OCI build) and `tt-model push`."""
    import tempfile

    http = _demo_http_server_hooks(demo_dir)
    if http is not None:
        return _run_http_server_container(args, state, slug, demo_dir, commit, http)

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
    adapter_ready = _adapter_is_working(bundle_dir) and not _rendered_rows_adapter(bundle_dir)
    if adapter_ready and _refresh_eager_pipeline_adapter(checkout, slug, bundle_dir):
        print(
            f"  [publish-hf] re-rendered {bundle_dir / 'adapter.py'}: it was an earlier copy of the tool's "
            "pipeline adapter, which decoded every token eagerly; decode is now traced when vLLM asks"
        )
    if adapter_ready and _adapter_never_traces(bundle_dir) and not getattr(args, "no_serve_gate", False):
        print(
            f"  [publish-hf] NOT publishing: {bundle_dir / 'adapter.py'} runs decode itself but never calls "
            "ttnn.execute_trace, so every served token would run eagerly, several times slower than the speed "
            "the model was optimized to. Make decode_forward replay a captured trace when vLLM passes "
            "enable_trace (the tool's pipeline adapter template does), or pass --no-serve-gate to publish it anyway."
        )
        return 5
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
        rows_adapter = _rendered_rows_adapter(bpath)
        if rows_adapter and arch_det and not _mistral_fallback:
            servable = True
        if is_stub or _mistral_fallback or not arch_det:
            state_note = "STUB — not servable until an adapter is written"
        elif rows_adapter:
            state_note = "the demo's own serving pipeline, one device row per request — servable"
        else:
            state_note = "stock generator — servable as-is"
        print(f"  [publish-hf] vLLM bundle {'created' if created else 'exists'}: {bpath}  ({state_note})")
        _own_pipeline = _demo_pipeline_module(checkout, slug) if not (is_stub or rows_adapter) else None
        if _own_pipeline:
            print(
                f"  [publish-hf] NOTE: this package serves the stock "
                f"{_pick_base_generator(_arch_for_bundle, mtype)[0]} from {_GEN_MOD}, not this demo's own "
                f"optimized pipeline ({_own_pipeline}); what customers get is the served speed, not the "
                f"optimized run's number. To serve the pipeline itself, give one module of the demo's tt/ package "
                f"the serving hooks ({_SERVING_BUILDER}, {_SERVING_ROWS}, {_SERVING_KV})."
            )

    # A novel-arch pipeline adapter must serve out of the box: build the image from the MATCHED
    # local vLLM+plugin (no version skew) and give that plugin novel-arch support. Stock arches are
    # unaffected (they ride a vLLM built-in; the hooks only fire on a novel EXTRA_MODELS_DIR bundle).
    _is_pipeline_adapter = (not adapter_ready) and (
        bool(_demo_pipeline_module(checkout, slug)) or _rendered_rows_adapter(bundle_dir)
    )
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

    plugin_ref = getattr(args, "plugin_ref", None) or _PLUGIN_DEFAULT_REF
    vllm_version = getattr(args, "vllm_version", None)
    if not vllm_version and not getattr(args, "vllm_path", None):
        if getattr(args, "plugin_path", None):
            recipe = Path(args.plugin_path) / _PLUGIN_INSTALL_RECIPE
            vllm_version = _recipe_vllm_version(recipe.read_text() if recipe.is_file() else "")
            source = str(recipe)
        else:
            plugin_ref, vllm_version = _plugin_serving_pair(_PLUGIN_REPO, plugin_ref) or (plugin_ref, None)
            source = f"{_PLUGIN_REPO} @ {plugin_ref}: {_PLUGIN_INSTALL_RECIPE}"
        if not vllm_version:
            print(
                f"  [publish-hf] cannot read the vLLM release the plugin installs ({source}); "
                "pass --vllm-version with the release that plugin needs."
            )
            return 2
        print(f"  [publish-hf] serving stack: plugin {source} installs vLLM {vllm_version}")
    # vLLM installed by version resolves the CUDA build's PyPI requirements, which name their own torch and
    # transformers; with no lock to fix the set, hold those at the versions the model was validated with.
    overrides = None
    if not getattr(args, "vllm_path", None) and not getattr(args, "lock", None):
        overrides = _validated_pins(checkout)
        print(f"  [publish-hf] held at the validated env's versions: {', '.join(overrides) or 'none found'}")

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
            plugin_ref=plugin_ref,
            vllm_version=vllm_version,
            extra_models_dir=extra,
            commit=commit,
            lock=getattr(args, "lock", None),
            vllm_path=getattr(args, "vllm_path", None),
            plugin_path=getattr(args, "plugin_path", None),
            overrides=overrides,
            card_terms=_upstream_card_terms(getattr(args, "weights", None), _hf_token(args)),
        )
    except ValueError as exc:
        print(f"  [publish-hf] {exc}")
        return 2
    return _package_and_publish(args, state, slug, checkout, demo_dir, yaml_path, ttm, out, servable, mesh_override)


def _package_and_publish(
    args, state, slug, checkout, demo_dir, yaml_path, ttm, out, servable, mesh_override, ready_line=None
) -> int:
    """Validate the written tt-model.yaml, build the container, serve it through the gate, then upload it and
    finish its card. `ready_line` is the log line an own-HTTP-server package prints once it serves (None for vLLM)."""
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
    gate_ok, served = _served_speed_gate(args, state, staged) if servable else (True, None)
    if not gate_ok:
        print(
            f"  [publish-hf] NOT uploading {staged}: the package does not serve at the speed it was optimized to "
            "(see the serve gate lines above). Fix the adapter or its serve settings in vllm_metadata.json "
            "(e.g. additional_config.tt.trace_region_size when the trace capture failed) and re-run, or pass "
            "--no-serve-gate to upload it anyway."
        )
        return 5
    if served:
        target = _serve_hardware(
            state,
            getattr(args, "box", None),
            getattr(args, "arch", None),
            getattr(args, "hardware", None),
            mesh_override,
        )
        served = {**served, "measured_on": target["hw_phrase"]}
        state["served"] = served
        if _restate_card_speed(staged, _served_headline(served, target["perf_hw"])):
            print("  [publish-hf] card speed: the served numbers above (was the bring-up measurement).")
    if getattr(args, "no_upload", False):
        print(f"  [publish-hf] --no-upload: built and gated, NOT uploading. Package: {staged}")
        try:
            perf = json.loads((Path(staged) / "tt_kernel_manifest.json").read_text())["container"]["card"][
                "performance"
            ]
            print(f"  [publish-hf] card speed: {perf.strip()}")
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return 0
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
    if served:
        _upload_card_section(args, _SERVED_TITLE, _served_card_body(served))
    # Card speed: a package that serves shows what `tt serve` measured above; only one that cannot serve
    # falls back to the perf test's sweep (a bring-up measurement, run at the model's full depth). An own-HTTP-server
    # package gets neither: it serves no tokens, so a tokens-per-second sweep does not describe what customers get.
    if not getattr(args, "no_bench", False) and ready_line is None:
        cfg = state.get("config") or {}
        if served:
            py = str(Path(checkout) / "python_env" / "bin" / "python")
            _enrich_card_with_accuracy(args, checkout, Path(demo_dir), py, cfg.get("pcc_test"))
        else:
            _enrich_card_with_benchmarks(args, slug, checkout, demo_dir, cfg.get("perf_test"), cfg.get("pcc_test"))
    _verify_published_package(args, slug, servable, ready_line)
    return 0


_VERIFY_TITLE = "Published-package verification"


def _verify_published_package(args, slug: str, servable: bool, ready_line: str | None = None) -> None:
    """Pull THE JUST-PUBLISHED package back and run its own quickstart, so the page is proven, not
    asserted. `tt model pull` must succeed; `tt serve` must then reach either `Application startup
    complete` (a servable package; `ready_line` for one that serves through its own HTTP server) or the
    documented scaffold stop (`initialize_vllm_model` missing)
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
                if ready_line and ready_line.lower() in low:
                    outcome, detail = "ready", ready_line
                    break
                if "application startup complete" in low:
                    eager = _eager_serve_lines((r.stdout or "") + (r.stderr or ""))
                    if eager:
                        outcome, detail = "eager", eager[0][:240]
                    else:
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
        verdict = f"PASS (pulled + served: {detail})"
    elif outcome == "eager":
        verdict = f"MISMATCH (served, but decodes without its trace -- several times slower: {detail})"
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
    current = _current_repo_id(args.repo, _hf_token(args))
    if current != args.repo:
        print(f"  [publish-hf] {args.repo} is now {current} on HF; publishing under {current}.")
        args.repo = current

    if getattr(args, "container", False):
        rc = _run_container(args, state, slug, demo_dir, commit)
        if rc == 0 and not getattr(args, "no_upload", False):
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
