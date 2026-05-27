"""Reference shapes 3 and 4 via TTNN's kwarg path (2 programs) since the
fused in-kernel path refuses batched-B.
"""

import os
import statistics
import sys
from pathlib import Path

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "Warning")

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest  # noqa: E402
import torch  # noqa: E402

import ttnn  # noqa: E402
import verify_makora  # noqa: E402


def _ttnn_kwarg(a, b):
    """Same compute config as verify_makora's matmul_silu, but NO core_grid → 2-program kwarg path."""
    cfg = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True)
    return ttnn.matmul(a, b, memory_config=ttnn.DRAM_MEMORY_CONFIG, activation="silu", compute_kernel_config=cfg)


@pytest.mark.parametrize(
    "shape", [(4, 1024, 1024, 1024), (2, 1, 2048, 2048, 2048)], ids=lambda s: "x".join(str(d) for d in s)
)
def test_kwarg_path(device, shape):
    torch.manual_seed(0)
    makora = verify_makora._load_makora_module("new", "matmul_silu")
    (a, b), _ = verify_makora._make_inputs(shape, device)

    # Warmup + drain.
    for _ in range(2):
        ttnn.deallocate(_ttnn_kwarg(a, b))
        ttnn.deallocate(makora.host(a, b))
        ttnn.synchronize_device(device)
    verify_makora._measure_one_kernel_duration_ns(device)

    # Correctness check (one off).
    out_t = _ttnn_kwarg(a, b)
    ttnn.synchronize_device(device)
    out_m = makora.host(a, b)
    ttnn.synchronize_device(device)
    pcc, mad = verify_makora._check_numerics(out_m, out_t)
    ttnn.deallocate(out_t)
    ttnn.deallocate(out_m)

    # Measured.
    ttnn_d = verify_makora._run_and_measure(lambda: _ttnn_kwarg(a, b), device, iters=5, warmup=0)
    makora_d = verify_makora._run_and_measure(lambda: makora.host(a, b), device, iters=5, warmup=0)
    t_med = statistics.median(ttnn_d)
    m_med = statistics.median(makora_d)
    print(
        f"\n  matmul_silu   shape={str(shape):<28} "
        f"ttnn={int(t_med):>8d} ns  makora={int(m_med):>8d} ns  "
        f"speedup={t_med/m_med:>5.2f}x  pcc={pcc:.4f}  max_abs_diff={mad:.2e}"
    )
