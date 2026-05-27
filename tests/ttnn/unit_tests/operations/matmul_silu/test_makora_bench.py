"""Pytest shim for verify_makora.py — gets the device-reset-on-exit for free.

run_safe_pytest.sh resets the device after every run, which verify_makora.py
does not. Wrapping the benchmark in pytest lets us benefit from that without
duplicating the reset machinery.

Usage:
  scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/matmul_silu/test_makora_bench.py -s
  scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/matmul_silu/test_makora_bench.py::test_matmul_silu_shape -s
  scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/matmul_silu/test_makora_bench.py -k '32x1024x1024' -s

`-s` disables pytest stdout capture so the benchmark table prints live.
"""

import os
import sys
from pathlib import Path

# Env vars verify_makora requires for the profiler. Set BEFORE importing the
# script, so its top-level `import ttnn` sees them.
os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
# Drop the per-program profiler histogram flood; user can re-enable via env.
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "Warning")

# verify_makora.py lives at the repo root; ensure it's importable.
REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest  # noqa: E402

import verify_makora  # noqa: E402


def _run(args):
    """Drive verify_makora.main() with the given argv list."""
    orig_argv = sys.argv
    try:
        sys.argv = ["verify_makora.py"] + args
        verify_makora.main()
    finally:
        sys.argv = orig_argv


@pytest.mark.parametrize(
    "shape",
    [
        # (*batch, M, K, N). Narrow M, wide K=N — TTNN's tested regime for
        # matmul(activation="silu"). Square M=K=N hits an untested path that
        # produces wrong outputs in the agent-eval branch's stripped matmul;
        # on main the fused-activation header is present so behavior may differ.
        (32, 1024, 1024),
        (64, 1024, 1024),
        (32, 2048, 2048),
        (64, 2048, 2048),
    ],
    ids=lambda s: "x".join(str(d) for d in s),
)
def test_matmul_silu_shape(shape):
    """One shape per test — easier to triage than the bundled README run."""
    args = ["--shape"] + [str(d) for d in shape] + ["--iters", "5"]
    _run(args)


def test_matmul_silu_readme_shapes():
    """All 4 README shapes in one run; prints a GMEAN row at the end."""
    _run(["--readme-shapes", "--iters", "5"])
