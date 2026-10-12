# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Every filesystem location this model directory reads or writes, in one place.

Nothing here is machine specific. Locations come from, in order of precedence:

1. a specific ``PPLX_DECIDER_*`` variable (for example ``PPLX_DECIDER_WEIGHT_CACHE``);
2. ``$PPLX_DECIDER_ARTIFACTS``, the root of all generated files;
3. the default root ``<tt-metal>/generated/pplx_decider`` (tt-metal's ``generated/`` is gitignored).

The HF snapshot is resolved through the HF cache (``$HF_HOME``) at the pinned revision, or taken from
``$PPLX_DECIDER_SNAPSHOT``.

Layout below ``ARTIFACTS``::

    goldens/                  HF layer inputs (S8192/, S16384/)        reference/hf_reference.py
    goldens/decisions/        25-row text decision golden              reference/hf_decision_golden.py
    goldens/vision/           vision tower golden (inputs/, tower/, images/)  reference/hf_vision_reference.py
    goldens/vision/e2e/       8-row image decision golden              reference/hf_image_decision_golden.py
    weight_cache/             ttnn disk weight cache (~27 GB)
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

# Device weight cache and logs.
WEIGHT_CACHE = _env_path("PPLX_DECIDER_WEIGHT_CACHE", ARTIFACTS / "weight_cache")
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


def snapshot_path() -> Path:
    """The immutable snapshot directory: ``$PPLX_DECIDER_SNAPSHOT``, else the HF cache at ``REVISION``."""
    override = os.environ.get("PPLX_DECIDER_SNAPSHOT")
    if override:
        return Path(override)
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True))
    except Exception as e:  # LocalEntryNotFoundError and friends
        raise FileNotFoundError(
            f"{MODEL_ID}@{REVISION} is not in the HF cache (HF_HOME={os.environ.get('HF_HOME', '<default>')}). "
            f"Download it with: hf download {MODEL_ID} --revision {REVISION}  (or set PPLX_DECIDER_SNAPSHOT)"
        ) from e


def resolved() -> dict[str, Path]:
    """Every default location, by name."""
    names = (
        "MODEL_ROOT TT_METAL_ROOT ARTIFACTS GOLDENS_ROOT LAYER_GOLDEN_DIR DECISION_GOLDEN_DIR VISION_GOLDEN_DIR "
        "IMAGE_GOLDEN_DIR GOLDEN_IMAGE_DIR WEIGHT_CACHE PCC_LOG PROBE_LOG PERF_LOG E2E_DIR IMAGE_PERF_DIR "
        "IMAGE_E2E_DIR VISION_DIR VISION_PCC_LOG DATATYPE_SWEEP_DIR"
    ).split()
    return {n: globals()[n] for n in names}


if __name__ == "__main__":
    for name, path in resolved().items():
        print(f"{name:<20} {path}")
    try:
        print(f"{'SNAPSHOT':<20} {snapshot_path()}")
    except FileNotFoundError as e:
        print(f"{'SNAPSHOT':<20} not found: {e}")
