# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

FILLS = {"zero": 0.0, "inf": float("inf"), "neg_inf": float("-inf"), "nan": float("nan")}


def reference(q, k, v, cur_pos, scale):
    b_count, num_q_heads = q.shape[1], q.shape[2]
    repeat = num_q_heads // k.shape[1]
    out = torch.zeros(1, b_count, num_q_heads, v.shape[3])
    for b in range(b_count):
        pos = int(cur_pos[b])
        k_b = k[b, :, : pos + 1].float().repeat_interleave(repeat, dim=0)
        v_b = v[b, :, : pos + 1].float().repeat_interleave(repeat, dim=0)
        scores = torch.einsum("hd,hsd->hs", q[0, b].float(), k_b) * scale
        out[0, b] = torch.einsum("hs,hsd->hd", torch.softmax(scores, dim=-1), v_b)
    return out


def stale_cache(clean, cur_pos, fill):
    stale = clean.clone()
    for b in range(clean.shape[0]):
        stale[b, :, int(cur_pos[b]) + 1 :] = fill
    return stale


def paged(per_user, page_table, block_size):
    blocks = per_user.shape[2] // block_size
    out = torch.zeros(int(page_table.max()) + 1, per_user.shape[1], block_size, per_user.shape[3])
    for b in range(per_user.shape[0]):
        for vb in range(blocks):
            out[int(page_table[b, vb])] = per_user[b, :, vb * block_size : (vb + 1) * block_size]
    return out


def program_config(device):
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=32,
        k_chunk_size=128,
        exp_approx_mode=False,
    )


def to_device(t, device, dtype=ttnn.bfloat16):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


PCC = {ttnn.bfloat16: 0.99, ttnn.bfloat8_b: 0.99, ttnn.bfloat4_b: 0.95}


def run_decode(device, q, k, v, page_table, cur_pos, block_size, kv_dtype, scale):
    out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        to_device(torch.nn.functional.pad(q, (0, 0, 0, 32 - q.shape[2])), device),
        to_device(paged(k, page_table, block_size), device, kv_dtype),
        to_device(paged(v, page_table, block_size), device, kv_dtype),
        page_table_tensor=ttnn.from_torch(page_table, dtype=ttnn.int32, device=device),
        cur_pos_tensor=ttnn.from_torch(cur_pos, dtype=ttnn.int32, device=device),
        scale=scale,
        program_config=program_config(device),
    )
    return ttnn.to_torch(out)[:, :, : q.shape[2], :].float()


@pytest.mark.parametrize("fill", list(FILLS), ids=list(FILLS))
@pytest.mark.parametrize("kv_dtype", list(PCC), ids=["bf16", "bfp8", "bfp4"])
@pytest.mark.parametrize("block_size", [32, 64])
@pytest.mark.parametrize(
    "cur_pos_list", [[0, 37, 100, 255], [31, 64, 129, 300], [20, 50, 147, 400]], ids=["pos_a", "pos_b", "pos_c"]
)
def test_paged_sdpa_decode_ignores_kv_past_cur_pos(device, fill, kv_dtype, block_size, cur_pos_list):
    torch.manual_seed(0)
    b_count, num_kv_heads, num_q_heads, head_dim, max_seq_len = 4, 1, 8, 64, 512
    blocks_per_user = max_seq_len // block_size
    cur_pos = torch.tensor(cur_pos_list, dtype=torch.int32)
    k = torch.randn(b_count, num_kv_heads, max_seq_len, head_dim).bfloat16().float()
    v = torch.randn(b_count, num_kv_heads, max_seq_len, head_dim).bfloat16().float()
    q = torch.randn(1, b_count, num_q_heads, head_dim).bfloat16().float()
    page_table = torch.randperm(b_count * blocks_per_user, dtype=torch.int32).reshape(b_count, blocks_per_user)
    scale = head_dim**-0.5

    clean = run_decode(device, q, k, v, page_table, cur_pos, block_size, kv_dtype, scale)
    out = run_decode(
        device,
        q,
        stale_cache(k, cur_pos, FILLS[fill]),
        stale_cache(v, cur_pos, FILLS[fill]),
        page_table,
        cur_pos,
        block_size,
        kv_dtype,
        scale,
    )

    assert torch.isfinite(out).all(), f"{int((~torch.isfinite(out)).sum())} non-finite outputs"
    assert torch.equal(out, clean), f"max diff vs clean cache {float((out - clean).abs().max())}"
    passing, msg = comp_pcc(reference(q, k, v, cur_pos, scale), out, pcc=PCC[kv_dtype])
    assert passing, msg
