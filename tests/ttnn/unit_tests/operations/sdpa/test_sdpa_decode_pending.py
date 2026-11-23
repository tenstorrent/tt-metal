# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""sdpa_decode with pending rows (speculative verify): one call for R tokens, batch b = t * Bkv + g, the ring
unchanged, the pass's K / V rows in pending_kv, token 0's position in pending_pos. Each row must be bit-equal to a
one-row call on the ring plain decode would have at that token (its own and earlier pending rows in their slots).
pending_tok0: a later call of a split pass (its batches are tokens tok0 ..)."""

import pytest
import torch

import ttnn


@pytest.mark.parametrize("p0", [1000, 1021])
@pytest.mark.parametrize("R", [2, 3])
@pytest.mark.parametrize("tok0", [0, 2])
def test_sdpa_decode_pending_rows(device, p0, R, tok0):
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
    cfg_pass = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        q_chunk_size=0,
        k_chunk_size=256,
        exp_approx_mode=False,
        max_cores_per_head_batch=4,
        exact_exp_half_tile=True,
        pending_tok0=tok0,
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

    def call(qt, k, v, m, s, c=cfg, **kw):
        o = ttnn.transformer.scaled_dot_product_attention_decode(
            put(qt),
            k,
            v,
            is_causal=False,
            attn_mask=put(m),
            attention_sink=sink_t(s),
            scale=1.0,
            program_config=c,
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
        c=cfg_pass,
        pending_kv=pending,
        pending_pos=pos,
    ).reshape(R, nkv, g, d)
    for t in range(R):
        rk, rv = ring_k.clone(), ring_v.clone()
        for tp in range(tok0 + t + 1):
            rk[:, (p0 + tp) % w], rv[:, (p0 + tp) % w] = new_k[:, tp], new_v[:, tp]
        ref = call(q[t].unsqueeze(0), ring(rk), ring(rv), mask[t].unsqueeze(1), sink[t])
        assert torch.equal(out[t], ref), t


@pytest.mark.parametrize("R, b0", [(3, 3000 + 40), (2, 2047 - 5)])
def test_flash_mla_decode_rows_per_token(device, R, b0):
    """Causal flash MLA decode with rows_per_token = 3: R tokens' q rows in one call (cur_pos = the last token's
    bound) give each token's rows bit-equal to its own call, when all bounds are in one K chunk."""
    assert b0 // 128 == (b0 + R - 1) // 128
    torch.manual_seed(b0)
    put = lambda t, dt=ttnn.bfloat16: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)  # noqa
    kv = put(torch.randn(8, 1, 4096, 288) * 0.3, ttnn.bfloat8_b)
    grid = device.compute_with_storage_grid_size()
    ck = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
    )

    def call(q, s, pos, rpt):
        cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y),
            q_chunk_size=0,
            k_chunk_size=128,
            exp_approx_mode=False,
            exact_exp_half_tile=True,
            rows_per_token=rpt,
        )
        sk = torch.zeros(8, 32, 32)
        sk[:, : s.shape[1]] = s
        cp = ttnn.from_torch(torch.tensor([pos] * 8, dtype=torch.int32), dtype=ttnn.int32, device=device)
        o = ttnn.transformer.flash_multi_latent_attention_decode(
            put(q.unsqueeze(0)),
            kv,
            head_dim_v=160,
            is_causal=True,
            cur_pos_tensor=cp,
            attention_sink=put(sk.view(256, 32)),
            scale=1.0,
            program_config=cfg,
            compute_kernel_config=ck,
        )
        return ttnn.to_torch(o).float()[0]

    q = torch.randn(8, 3 * R, 288) * 0.3
    s = torch.randn(8, 3 * R, 32)
    out = call(q, s, b0 + R - 1, 3)
    for t in range(R):
        ref = call(q[:, 3 * t : 3 * t + 3], s[:, 3 * t : 3 * t + 3], b0 + t, 0)[:, :3]
        assert torch.equal(out[:, 3 * t : 3 * t + 3], ref), t
