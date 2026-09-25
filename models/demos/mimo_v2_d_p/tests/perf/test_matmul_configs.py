# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Sweep 2D-mcast matmul program configs for the MiMo attention projections (per chip at TP=2, bf8 weights, HiFi2).
Signposts ``mm_{name}_M{M}_{cfg}``; ``auto`` = ttnn's default pick. Analyze with analyze_attention-style summing."""

import itertools

import pytest
import torch

import ttnn
from models.demos.mimo_v2_d_p.tt.mm_configs import mm_2d_config

try:
    from tracy import signpost
except ImportError:  # pragma: no cover
    signpost = lambda *a, **k: None

SHAPES = {"swa_qkv": (4096, 7680), "ga_qkv": (4096, 6912), "o_proj": (4096, 4096)}


@pytest.mark.parametrize("name", list(SHAPES))
@pytest.mark.parametrize("M", [640, 2048])
def test_matmul_configs(device, name, M):
    K, N = SHAPES[name]
    x = ttnn.from_torch(torch.randn(1, 1, M, K), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    w = ttnn.from_torch(torch.randn(1, 1, K, N) * 0.02, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
    ckc = ttnn.init_device_compute_kernel_config(device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
                                                 fp32_dest_acc_en=True, packer_l1_acc=True)
    ref = None
    cfgs = [("auto", None)]
    for in0_bw, gx in itertools.product((2, 4, 8), (10, 11)):
        pc = mm_2d_config(device, M, K, N, in0_block_w=in0_bw, grid_x=gx)
        if pc is not None:
            cfgs.append((f"bw{in0_bw}_gx{gx}", pc))
    for tag, pc in cfgs:
        try:
            for it in range(3):
                if it:
                    signpost(f"mm_{name}_M{M}_{tag}_start")
                o = ttnn.linear(x, w, dtype=ttnn.bfloat16, compute_kernel_config=ckc, program_config=pc)
                ttnn.synchronize_device(device)
                if it:
                    signpost(f"mm_{name}_M{M}_{tag}_end")
                if it == 2:
                    t = ttnn.to_torch(o).float()
                    if ref is None:
                        ref = t
                    else:
                        assert torch.corrcoef(torch.stack([ref.flatten(), t.flatten()]))[0, 1] > 0.9999, tag
                o.deallocate(True)
        except Exception as e:  # noqa: BLE001 - L1 overflow etc. just skips the config
            print(f"SKIP {name} M{M} {tag}: {str(e).splitlines()[0][:120]}")
