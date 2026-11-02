# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Rounding of the ttnn bf16 multiply / add / matmul-by-pair-swap vs torch round-to-nearest-even, one chip (which stage of rope_pe deviates from RNE)."""
import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import pf_tune


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [pytest.param({"l1_small_size": 16384}, id="p")], indirect=True)
@torch.no_grad()
def test_round(mesh_device):
    md = mesh_device
    rep = ttnn.ReplicateTensorToMesh(md)
    up = lambda t: ttnn.from_torch(
        t.to(torch.bfloat16).contiguous(),
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rep,
    )
    down = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()
    bf = lambda t: t.to(torch.bfloat16).float()
    torch.manual_seed(7)
    a = bf(torch.randn(2, 8, 512, 64) * 3)
    b = bf(torch.randn(2, 1, 512, 64))
    c = bf(torch.randn(2, 8, 512, 64) * 3)
    m = down(ttnn.multiply(up(a), up(b)))
    print(f"ROUND multiply (bcast) vs RNE: mismatches {int((m != bf(a * b)).sum())} of {m.numel()}", flush=True)
    m2 = down(ttnn.multiply(up(a), up(c)))
    print(f"ROUND multiply vs RNE: mismatches {int((m2 != bf(a * c)).sum())}", flush=True)
    s = down(ttnn.add(up(a), up(c)))
    print(f"ROUND add vs RNE: mismatches {int((s != bf(a + c)).sum())}", flush=True)
    tr = lambda t: ((t.float().contiguous().view(torch.int32) & ~0xFFFF).view(torch.float32))
    print(f"ROUND add vs truncation: mismatches {int((s != tr(a + c)).sum())}", flush=True)
    ref_trunc = ((a * b).view(torch.int32) & ~0xFFFF).view(torch.float32)
    print(f"ROUND multiply vs truncation: mismatches {int((m != ref_trunc).sum())}", flush=True)
    ckc = ttnn.init_device_compute_kernel_config(
        md.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    P = pf_tune.p64(md)
    p = down(ttnn.matmul(up(a), P, compute_kernel_config=ckc))
    sw = torch.zeros_like(a)
    sw[..., 0::2] = -a[..., 1::2]
    sw[..., 1::2] = a[..., 0::2]
    print(f"ROUND matmul by pair swap vs exact: mismatches {int((p != sw).sum())}", flush=True)
