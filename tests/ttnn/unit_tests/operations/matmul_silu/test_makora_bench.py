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
        # The agent-eval reference shapes from
        # /localdev/dnijemcevic/kernels/Tenstorrent/references/matmul_fused_activation.py
        # Reference uses literal input shapes (a.shape == b.shape, square M=K=N);
        # harness encodes as (*batch_dims, M, K, N):
        #   (32, 32)             literal -> (32, 32, 32)             — 1 output tile, 1 core
        #   (128, 128)           literal -> (128, 128, 128)          — 16 output tiles, ≤16 cores
        #   (4, 1024, 1024)      literal -> (4, 1024, 1024, 1024)    — batch=4, full 8×8 grid
        #   (2, 1, 2048, 2048)   literal -> (2, 1, 2048, 2048, 2048) — batch=(2,1), full 8×8 grid
        (32, 32, 32),
        (128, 128, 128),
        (4, 1024, 1024, 1024),
        (2, 1, 2048, 2048, 2048),
        # Batch-fused-into-M equivalents of shapes 3 & 4 — same FMAs, but B is
        # 2D so TTNN's in-kernel fused activation path accepts them.
        (4096, 1024, 1024),
        (4096, 2048, 2048),
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
