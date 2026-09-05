# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DRAM bandwidth probes on [7168, 2048] DRAM-interleaved tile tensors in bfp4 / bfp8 / bf16
(576 / 1088 / 2048 B pages): clone (read+write), and a 32-row matmul against it (read-dominated)."""
import statistics
import pytest
import torch
import ttnn
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program_merged


def _med_ns(device, fn, kernel_substr=None):
    fn()
    ttnn.synchronize_device(device)
    _, per = profile_realtime_program_merged(device, lambda: [fn() for _ in range(3)])
    ents = [e for e in per.values() if kernel_substr is None or any(kernel_substr in s for s in e["kernel_sources"])]
    return statistics.median([e["duration_ns"] for e in ents])


@pytest.mark.parametrize("dtype,tile_bytes", [(ttnn.bfloat4_b, 576), (ttnn.bfloat8_b, 1088), (ttnn.bfloat16, 2048)])
def test_clone(device, dtype, tile_bytes):
    shape = (7168, 2048)
    t = ttnn.from_torch(torch.randn(*shape), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    n_bytes = shape[0] * shape[1] // 1024 * tile_bytes
    ns = _med_ns(device, lambda: ttnn.clone(t))
    print(
        f"DRAMBW clone tile={tile_bytes}B bytes(r+w)={2*n_bytes/1e6:.1f}MB ns={ns:.0f} -> {2*n_bytes/ns:.0f} GB/s",
        flush=True,
    )


@pytest.mark.parametrize("dtype,tile_bytes", [(ttnn.bfloat4_b, 576), (ttnn.bfloat8_b, 1088), (ttnn.bfloat16, 2048)])
def test_matmul_read(device, dtype, tile_bytes):
    K, N = 7168, 2048
    w = ttnn.from_torch(torch.randn(K, N), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    a = ttnn.from_torch(torch.randn(32, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    n_bytes = K * N // 1024 * tile_bytes
    ck = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    ns = _med_ns(device, lambda: ttnn.matmul(a, w, compute_kernel_config=ck, dtype=ttnn.bfloat16), "matmul")
    print(
        f"DRAMBW matmul32 tile={tile_bytes}B Wbytes={n_bytes/1e6:.1f}MB ns={ns:.0f} -> {n_bytes/ns:.0f} GB/s (weights only)",
        flush=True,
    )
