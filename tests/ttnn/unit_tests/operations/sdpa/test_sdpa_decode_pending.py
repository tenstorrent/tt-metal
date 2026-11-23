# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""sdpa_decode with pending rows (speculative verify): one call for R tokens, batch b = t * Bkv + g, the ring
unchanged, the pass's K / V rows in pending_kv, token 0's position in pending_pos. Each row must be bit-equal to a
one-row call on the ring plain decode would have at that token (its own and earlier pending rows in their slots)."""

import pytest
import torch

import ttnn


@pytest.mark.parametrize("p0", [1000, 1021])
@pytest.mark.parametrize("R", [2, 3])
def test_sdpa_decode_pending_rows(device, p0, R):
    nkv, g, d, w = 8, 3, 128, 1024
    torch.manual_seed(p0 + R)
    f8 = lambda t: ttnn.to_torch(ttnn.from_torch(t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)).float()  # noqa
    put = lambda t, dt=ttnn.bfloat16: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)  # noqa
    grid = device.compute_with_storage_grid_size()
    cfg = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        q_chunk_size=0,
        k_chunk_size=256,
        exp_approx_mode=False,
        max_cores_per_head_batch=4,
        exact_exp_half_tile=True,
    )
    ck = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
    )
    ring_k, ring_v = f8(torch.randn(nkv, w, d) * 0.3), f8(torch.randn(nkv, w, d))
    new_k, new_v = f8(torch.randn(nkv, 32, d) * 0.3), f8(torch.randn(nkv, 32, d))
    q = torch.randn(R, nkv, g, d) * 0.3
    mask = torch.randn(R, nkv, g, w)
    sink = torch.randn(R, nkv, g, 32)

    def sink_t(s):
        out = torch.zeros(s.shape[0], 32, 32)
        out[:, :g] = s
        return put(out.view(-1, 32))

    def call(qt, k, v, m, s, **kw):
        o = ttnn.transformer.scaled_dot_product_attention_decode(
            put(qt),
            k,
            v,
            is_causal=False,
            attn_mask=put(m),
            attention_sink=sink_t(s),
            scale=1.0,
            program_config=cfg,
            compute_kernel_config=ck,
            **kw,
        )
        return ttnn.to_torch(o).float()[0, :, :g]

    ring = lambda t: put(t.view(nkv, 1, w, d), ttnn.bfloat8_b)  # noqa: E731
    pending = put(torch.stack([new_k, new_v]), ttnn.bfloat8_b)
    pos = ttnn.from_torch(torch.tensor([p0], dtype=torch.int32), dtype=ttnn.int32, device=device)
    out = call(
        q.reshape(1, R * nkv, g, d),
        ring(ring_k),
        ring(ring_v),
        mask.reshape(R * nkv, 1, g, w),
        sink.reshape(R * nkv, g, 32),
        pending_kv=pending,
        pending_pos=pos,
    ).reshape(R, nkv, g, d)
    for t in range(R):
        rk, rv = ring_k.clone(), ring_v.clone()
        for tp in range(t + 1):
            rk[:, (p0 + tp) % w], rv[:, (p0 + tp) % w] = new_k[:, tp], new_v[:, tp]
        ref = call(q[t].unsqueeze(0), ring(rk), ring(rv), mask[t].unsqueeze(1), sink[t])
        assert torch.equal(out[t], ref), t
