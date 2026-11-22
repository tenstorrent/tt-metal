# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Skewed prefill rel bias (tt/fused/pf_rel.py): bias values bit-equal to ttnn.matmul(states, proj), and the SDPA on
the skewed layout (rel_skewed=True) bit-equal to the SDPA on the plain [.., W] bias, for local (window) and global
layers with history keys; plus the times of both paths at T = 8192."""

import time

import pytest
import torch

import ttnn
from models.demos.inkling2_mini.tt.fused.pf_rel import PfRel, skew_proj

W, NH = 1024, 24


def timed(dev, fn, reps=3):
    fn()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / reps * 1e3


@pytest.mark.parametrize("t,hist,window", [(256, 1024, 1024), (2048, 0, 1024), (1024, 2048, None), (8192, 1024, 1024)])
def test_pf_rel(device, t, hist, window):
    dev = device
    g = torch.Generator().manual_seed(t + hist)
    ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                 math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)  # fmt: skip
    st = torch.zeros(1, NH, t, 32)
    for h in range(NH):  # 16-wide heads, head h in its half of the tile
        st[0, h, :, 16 * (h % 2) : 16 * (h % 2) + 16] = torch.randn(t, 16, generator=g)
    p16 = torch.randn(16, W, generator=g) * 0.3
    ph = torch.cat([p16, p16])
    tile, rm = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT
    put = lambda a, lay=tile: ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=lay, device=dev)  # noqa: E731
    st_t = put(st)
    proj = put(ph.view(1, 1, 32, W))
    bsk = put(skew_proj(ph))
    mod = PfRel(dev)

    def plain():
        return ttnn.to_layout(ttnn.matmul(st_t, proj, dtype=ttnn.bfloat16, compute_kernel_config=ckc), rm)

    def skewed():
        return mod(ttnn.to_layout(st_t, rm), bsk)

    ref, sk = plain(), skewed()
    a, b = ttnn.to_torch(ref).float(), ttnn.to_torch(sk).float()
    r = torch.arange(t) % 32
    cols = W - torch.arange(W).view(1, W) + r.view(t, 1)  # column of d in each row
    b_d = torch.gather(b[0], 2, cols.expand(NH, t, W))
    bias_equal = torch.equal(a[0], b_d)

    q = put(torch.randn(1, NH, t, 128, generator=g) * 0.3)
    k = put(torch.randn(1, 8, hist + t, 128, generator=g) * 0.3)
    v = put(torch.randn(1, 8, hist + t, 128, generator=g))
    sink = put(torch.randn(1, NH, t, 1, generator=g))
    pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=dev.compute_with_storage_grid_size(),
                                q_chunk_size=128, k_chunk_size=128, exp_approx_mode=False)  # fmt: skip

    def sdpa(rel, skew):
        return ttnn.transformer.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=1.0, sliding_window_size=window, rel_logits=rel, rel_skewed=skew,
            attention_sink=sink, program_config=pc, compute_kernel_config=ckc)  # fmt: skip

    o1, o2 = ttnn.to_torch(sdpa(ref, False)), ttnn.to_torch(sdpa(sk, True))
    out_equal = torch.equal(o1, o2)
    print(f"PF_REL t={t} hist={hist} window={window} bias_equal={bias_equal} sdpa_equal={out_equal} "
          f"maxdiff={float((o1.float() - o2.float()).abs().max()):.3g}")  # fmt: skip
    if t == 8192:
        print(
            f"PF_REL_MS t={t} bias: matmul+untilize {timed(dev, plain):.2f} pf_rel {timed(dev, skewed):.2f}; "
            f"sdpa plain {timed(dev, lambda: sdpa(ref, False)):.2f} skewed {timed(dev, lambda: sdpa(sk, True)):.2f}"
        )
    assert bias_equal and out_equal
