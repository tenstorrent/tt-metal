# Domain check of an rm_issue_cost variant: the whole nhwc_rm_transpose acceptance suite
# (tests/ttnn/unit_tests/operations/g6_permute/test_g6_permute_nhwc_rm.py) re-run with the variant's kernels
# and knobs installed by an autouse fixture. RIC_VARIANT selects the variant (default rw2_unit_trid1).
import importlib
import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[7]
EXP = ROOT / "ttnn/ttnn/operations/g6_permute/perf_experiments/rm_issue_cost"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V = _load("ric_variants", EXP / "variants.py")
SUITE = _load("ric_nhwc_suite", ROOT / "tests/ttnn/unit_tests/operations/g6_permute/test_g6_permute_nhwc_rm.py")
globals().update({k: v for k, v in vars(SUITE).items() if not k.startswith("_") and k not in ("V",)})

VARIANT = os.environ.get("RIC_VARIANT", "rw2_unit_trid1")
op_mod = importlib.import_module("ttnn.operations.g6_permute.g6_permute")
nhwc = importlib.import_module("ttnn.operations.g6_permute.g6_permute_nhwc_rm_transpose")


@pytest.fixture(autouse=True)
def _ric_variant(monkeypatch):
    op_mod._PROGRAM_CACHE.clear()
    V.install(monkeypatch, nhwc, VARIANT, "full")
    print(f"\nRIC_DOMAIN variant={VARIANT}")
    yield
    op_mod._PROGRAM_CACHE.clear()
