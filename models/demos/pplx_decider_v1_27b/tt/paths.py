# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Every filesystem location this model directory reads or writes, in one place.

Nothing here is machine specific. Locations come from, in order of precedence:

1. a specific ``PPLX_DECIDER_*`` variable (for example ``PPLX_DECIDER_DECISION_GOLDEN``);
2. ``$PPLX_DECIDER_ARTIFACTS``, the root of all generated files;
3. the default root ``<tt-metal>/generated/pplx_decider`` (tt-metal's ``generated/`` is gitignored).

The model path and the device weight cache follow ``models/demos/gemma4`` instead:

- ``resolve_model_path(model_path)``: the argument, else ``$HF_MODEL``, else ``$PPLX_DECIDER_SNAPSHOT``
  (legacy), else the HF cache at the pinned ``REVISION``. A local directory is used as is; an HF repo id
  is looked up in the HF cache (``$HF_HOME``, ``local_files_only``).
- ``resolve_model_cache_path(model_path)``: the root is ``$TT_CACHE_PATH``, else
  ``$PPLX_DECIDER_WEIGHT_CACHE`` (legacy), else the local model directory itself (the cache sits next to
  the weights), else ``$HF_HOME/tt_cache/<repo id with / -> -->``. The cache is
  ``<root>/tt_weight_cache/<key>``; ``<key>`` is the snapshot revision for an HF-cache snapshot directory
  (checked after resolving symlinks), else ``local-<sha256[:12]>`` of the checkpoint's ``config.json`` and
  ``model.safetensors.index.json`` bytes and the sizes of ``readout.safetensors`` and the index's shards,
  so two checkpoints never share a cache and a copied checkpoint keeps its key.

Layout below ``ARTIFACTS``::

    goldens/                  HF layer inputs (S8192/, S16384/)        reference/hf_reference.py
    goldens/decisions/        25-row text decision golden              reference/hf_decision_golden.py
    goldens/vision/           vision tower golden (inputs/, tower/, images/)  reference/hf_vision_reference.py
    goldens/vision/e2e/       8-row image decision golden              reference/hf_image_decision_golden.py
    logs/                     pcc_results.jsonl, context_probe.jsonl
    perf/                     prefill_perf.jsonl
    e2e/                      text e2e results, full-model perf, precision-config check
    image_e2e/                image e2e results and image perf
    vision/                   vision tower test results (pcc_vision.jsonl, perf JSON)
    datatype_sweep/           per-candidate results read by doc/datatype_sweep/sweep_report.py

Print the resolved locations (CPU only, needs no device and no snapshot)::

    python -m models.demos.pplx_decider_v1_27b.tt.paths
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path

MODEL_ID = "perplexity-ai/pplx-decider-v1-27b"
REVISION = "b01a5cbaca5391f73bd55103d4f27e8982cd5e60"

MODEL_ROOT = Path(__file__).resolve().parents[1]  # models/demos/pplx_decider_v1_27b
TT_METAL_ROOT = MODEL_ROOT.parents[2]


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else Path(default)


ARTIFACTS = _env_path("PPLX_DECIDER_ARTIFACTS", TT_METAL_ROOT / "generated" / "pplx_decider")

# HF reference goldens (CPU generated, read by the tests).
GOLDENS_ROOT = ARTIFACTS / "goldens"
LAYER_GOLDEN_DIR = _env_path("PPLX_DECIDER_GOLDEN_DIR", GOLDENS_ROOT)
DECISION_GOLDEN_DIR = _env_path("PPLX_DECIDER_DECISION_GOLDEN", GOLDENS_ROOT / "decisions")
VISION_GOLDEN_DIR = _env_path("PPLX_DECIDER_VISION_GOLDEN", GOLDENS_ROOT / "vision")
IMAGE_GOLDEN_DIR = _env_path("PPLX_DECIDER_IMAGE_GOLDEN", VISION_GOLDEN_DIR / "e2e")
GOLDEN_IMAGE_DIR = VISION_GOLDEN_DIR / "images"  # the 8 synthetic PNGs of the image prompt set

# Logs. (The device weight cache is located from the model path: resolve_model_cache_path below.)
PCC_LOG = _env_path("PPLX_DECIDER_PCC_LOG", ARTIFACTS / "logs" / "pcc_results.jsonl")
PROBE_LOG = _env_path("PPLX_DECIDER_PROBE_LOG", ARTIFACTS / "logs" / "context_probe.jsonl")
PERF_LOG = _env_path("PPLX_DECIDER_PERF_LOG", ARTIFACTS / "perf" / "prefill_perf.jsonl")

# Result directories of the device tests. The variable names keep the bring-up stage that added them.
E2E_DIR = _env_path("PPLX_DECIDER_STAGE6_DIR", ARTIFACTS / "e2e")
IMAGE_PERF_DIR = _env_path("PPLX_DECIDER_STAGE12B_DIR", ARTIFACTS / "image_e2e")
IMAGE_E2E_DIR = _env_path("PPLX_DECIDER_IMAGE_E2E_DIR", IMAGE_PERF_DIR)
VISION_DIR = _env_path("PPLX_DECIDER_STAGE12A_DIR", ARTIFACTS / "vision")
VISION_PCC_LOG = _env_path("PPLX_DECIDER_VISION_PCC_LOG", VISION_DIR / "pcc_vision.jsonl")
DATATYPE_SWEEP_DIR = ARTIFACTS / "datatype_sweep"

# Commands that generate each golden (CPU only, run from the tt-metal root).
_REF = "python -m models.demos.pplx_decider_v1_27b.reference"
GENERATE = {
    "layer": f"{_REF}.hf_reference --seq-len 8192 --layers 0 3 61 63 --through-final",
    "decision": f"{_REF}.hf_decision_golden",
    "vision": f"{_REF}.hf_vision_reference",
    "image": f"{_REF}.hf_image_decision_golden",
}


def require(path: Path | str, golden: str) -> Path:
    """Return ``path``; if it is missing, stop with the command that generates it.

    ``golden`` is a key of ``GENERATE``. Inside a running test this is ``pytest.fail``; elsewhere
    (demo, benchmark) a ``FileNotFoundError`` with the same message.
    """
    path = Path(path)
    if path.exists():
        return path
    message = (
        f"Missing {path}.\nGenerate it on CPU from the tt-metal root with:\n    {GENERATE[golden]}\n"
        "(or point $PPLX_DECIDER_ARTIFACTS / the specific PPLX_DECIDER_* variable at existing goldens; "
        "see models/demos/pplx_decider_v1_27b/README.md)"
    )
    if os.environ.get("PYTEST_CURRENT_TEST"):
        import pytest

        pytest.fail(message, pytrace=False)
    raise FileNotFoundError(message)


# ----------------------------------------------------------------------------------------------
# Model path and device weight cache (models/demos/gemma4 convention)
# ----------------------------------------------------------------------------------------------

CACHE_SUBDIR = "tt_weight_cache"
_RO_ERRNOS = (errno.EROFS, errno.EACCES, errno.EPERM)


def _hf_snapshot(repo_id: str, revision: str | None) -> Path:
    """Snapshot directory of ``repo_id`` in the HF cache (``local_files_only``), or a FileNotFoundError
    that names the ``hf download`` command."""
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(repo_id, revision=revision, local_files_only=True))
    except Exception as e:  # LocalEntryNotFoundError and friends
        at = f"@{revision}" if revision else ""
        flag = f" --revision {revision}" if revision else ""
        raise FileNotFoundError(
            f"{repo_id}{at} is not in the HF cache (HF_HOME={os.environ.get('HF_HOME', '<default>')}). "
            f"Download it with: hf download {repo_id}{flag}  (or pass a local checkpoint directory)"
        ) from e


def _model_dir(value: Path | str) -> Path:
    """A local checkpoint directory as is; an HF repo id through the HF cache (``MODEL_ID`` at ``REVISION``)."""
    path = Path(value).expanduser()
    if path.is_dir():
        return path
    repo_id = str(value)
    if path.is_absolute() or repo_id.startswith(".") or repo_id.count("/") != 1:
        raise FileNotFoundError(f"model path {value!r} is neither a local directory nor an HF repo id")
    return _hf_snapshot(repo_id, REVISION if repo_id == MODEL_ID else None)


def resolve_model_path(model_path: Path | str | None = None) -> tuple[Path, str]:
    """``(checkpoint directory, source)``: the argument, else ``$HF_MODEL``, else ``$PPLX_DECIDER_SNAPSHOT``
    (legacy), else the HF cache at the pinned ``REVISION``. Each value is a local directory or an HF repo id."""
    candidates = (
        (model_path, "argument"),
        (os.environ.get("HF_MODEL"), "$HF_MODEL"),
        (os.environ.get("PPLX_DECIDER_SNAPSHOT"), "$PPLX_DECIDER_SNAPSHOT (legacy)"),
    )
    for value, source in candidates:
        if value:
            return _model_dir(value), source
    return _hf_snapshot(MODEL_ID, REVISION), f"HF cache {MODEL_ID}@{REVISION[:12]}"


def snapshot_path() -> Path:
    """The checkpoint directory of ``resolve_model_path()`` (no argument)."""
    return resolve_model_path()[0]


def cache_key(model_dir: Path | str) -> str:
    """Weight cache key of a checkpoint directory.

    An HF-cache snapshot (``.../models--<org>--<name>/snapshots/<revision>``) is keyed by its revision.
    The layout is checked on the symlink-resolved path, so a symlink to a snapshot (or ``HF_MODEL=<symlink>``)
    keeps the revision key. Any other directory gets ``local-`` + sha256[:12] of its ``config.json`` and
    ``model.safetensors.index.json`` bytes and the byte sizes of ``readout.safetensors`` and of every shard
    the index names. No mtime: a plain ``cp`` of a checkpoint keeps its key.
    """
    model_dir = Path(model_dir).expanduser().resolve()
    if model_dir.parent.name == "snapshots" and model_dir.parent.parent.name.startswith("models--"):
        return model_dir.name
    digest = hashlib.sha256()
    for name in ("config.json", "model.safetensors.index.json"):
        digest.update(name.encode() + b"\0" + (model_dir / name).read_bytes() + b"\0")
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    for name in ["readout.safetensors", *sorted(set(index.get("weight_map", {}).values()))]:
        size = (model_dir / name).stat().st_size if (model_dir / name).exists() else -1
        digest.update(f"{name} {size}\0".encode())
    return "local-" + digest.hexdigest()[:12]


def _writable_cache_mirror(preferred: Path) -> Path:
    """Writable mirror of a cache dir on a read-only mount (gemma4 ``_writable_cache_mirror``)."""
    root = Path(os.environ.get("TT_METAL_HOME") or os.environ.get("HOME") or "/tmp")
    suffix = Path(*preferred.parts[-2:]) if len(preferred.parts) >= 2 else Path(preferred.name)
    return root / "generated" / "pplx_decider_tt_cache" / suffix


def _ensure_cache_dir(path: Path) -> Path:
    """Return ``path``, creating it when possible; on a read-only mount use a writable mirror (gemma4)."""
    if path.is_dir():
        return path
    try:
        path.mkdir(parents=True, exist_ok=True)
        return path
    except OSError as e:
        if e.errno not in _RO_ERRNOS:
            raise
        if path.is_dir():
            return path
        alt = _writable_cache_mirror(path)
        from loguru import logger

        logger.warning(f"pplx-decider weight cache: {path} is not writable ({e}); using {alt}")
        alt.mkdir(parents=True, exist_ok=True)
        return alt


def resolve_model_cache_path(model_path: Path | str, *, create: bool = True) -> tuple[Path, str]:
    """``(weight cache directory, source of its root)`` for a checkpoint (gemma4 ``resolve_model_cache_path``).

    Root: ``$TT_CACHE_PATH``, else ``$PPLX_DECIDER_WEIGHT_CACHE`` (legacy), else the local ``model_path``
    directory (the cache sits next to the weights), else ``$HF_HOME/tt_cache/<repo id, / -> -->``. The cache
    is ``<root>/tt_weight_cache/<cache_key>``. Every policy shares it: a cache file name carries the
    weight's dtype, shape and layout. ``create=False`` only computes the path.
    """
    for env in ("TT_CACHE_PATH", "PPLX_DECIDER_WEIGHT_CACHE"):
        if os.environ.get(env):
            root, source = Path(os.environ[env]), f"${env}"
            break
    else:
        if Path(model_path).is_dir():
            root, source = Path(model_path), "model directory"
        else:
            # An HF repo id: caching under Path(model_path) would create it as a relative dir in cwd.
            hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
            root, source = Path(hf_home) / "tt_cache" / str(model_path).replace("/", "--"), "$HF_HOME/tt_cache"
    path = root / CACHE_SUBDIR / cache_key(_model_dir(model_path))
    return (_ensure_cache_dir(path) if create else path), source


def resolved() -> dict[str, Path]:
    """Every default location, by name."""
    names = (
        "MODEL_ROOT TT_METAL_ROOT ARTIFACTS GOLDENS_ROOT LAYER_GOLDEN_DIR DECISION_GOLDEN_DIR VISION_GOLDEN_DIR "
        "IMAGE_GOLDEN_DIR GOLDEN_IMAGE_DIR PCC_LOG PROBE_LOG PERF_LOG E2E_DIR IMAGE_PERF_DIR "
        "IMAGE_E2E_DIR VISION_DIR VISION_PCC_LOG DATATYPE_SWEEP_DIR"
    ).split()
    return {n: globals()[n] for n in names}


if __name__ == "__main__":
    for name, path in resolved().items():
        print(f"{name:<20} {path}")
    try:
        model_dir, model_source = resolve_model_path()
        print(f"{'MODEL_PATH':<20} {model_dir}  (from {model_source})")
        cache_dir, cache_source = resolve_model_cache_path(model_dir, create=False)
        print(f"{'WEIGHT_CACHE':<20} {cache_dir}  (root from {cache_source})")
    except FileNotFoundError as e:
        print(f"{'MODEL_PATH':<20} not found: {e}")
