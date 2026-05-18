"""Wall-clock perf for the fused attention_block kernel (#10 Commit 2).

Warmup + measure pattern (same as test_perf_siglip_only.py). Reports min,
median, p95 of N iterations of `SigLIPAttentionBlockFused.op(...)` calls
between two `ttnn.synchronize_device(device)` barriers. Tensors are built
once outside the loop, so the only thing being timed is program dispatch +
device execution.

Note: this does NOT include readback (no ttnn.to_torch in the timed
section). Add readback measurements separately if you care about the
end-to-end pipeline cost.
"""
import sys
import time
import statistics
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "attention_block"))
from op import (  # noqa: E402
    SigLIPAttentionBlockFused,
    build_tensors_for_fused_attention_block,
)

M, D = 256, 1152
EPS = 1e-6
WARMUP_ITERS = 5
MEASURE_ITERS = 50


def make_inputs(seed: int = 42):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(M, D, generator=g, dtype=torch.bfloat16) * 0.5
    gamma = torch.ones(D, dtype=torch.bfloat16) + torch.randn(D, generator=g, dtype=torch.bfloat16) * 0.1
    beta = torch.randn(D, generator=g, dtype=torch.bfloat16) * 0.05
    return x, gamma, beta


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_perf_attention_block_fused(device):
    import ttnn

    x, gamma, beta = make_inputs(seed=42)
    tensors = build_tensors_for_fused_attention_block(device, x, gamma, beta)

    # Warmup — first call JIT-compiles, subsequent calls hit cached binaries.
    for _ in range(WARMUP_ITERS):
        SigLIPAttentionBlockFused.op(*tensors, eps=EPS)
    ttnn.synchronize_device(device)

    timings_ms = []
    for _ in range(MEASURE_ITERS):
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        SigLIPAttentionBlockFused.op(*tensors, eps=EPS)
        ttnn.synchronize_device(device)
        timings_ms.append((time.perf_counter() - t0) * 1000.0)

    timings_ms.sort()
    n = len(timings_ms)
    p50 = timings_ms[n // 2]
    p95 = timings_ms[int(n * 0.95)]
    mn = timings_ms[0]
    mx = timings_ms[-1]
    mean = statistics.fmean(timings_ms)
    print(
        f"\nattention_block fused (LN1 + 8→36 mcast + residual), {MEASURE_ITERS} iters, ms:\n"
        f"  min={mn:.3f}  median={p50:.3f}  p95={p95:.3f}  max={mx:.3f}  mean={mean:.3f}"
    )
