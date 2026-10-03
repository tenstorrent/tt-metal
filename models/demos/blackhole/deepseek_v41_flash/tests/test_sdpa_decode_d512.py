# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""De-risk: paged SDPA decode at DeepSeek-V4.1-Flash attention shapes on Blackhole.

MQA: 64 query heads, 1 KV head (K and V are the same vector), head_dim 512, per-head attention sink
and a 128-token sliding window, paged KV cache. Nothing in-tree covers this combination
(test_sdpa_decode.py uses d=128, test_mla_decode.py has no sink or window).

The sink needs no extra scaling in the reference model (exp(sink - max) against already-scaled
logits), but the TT kernel multiplies sinks by `scale`, so sinks are passed pre-divided, exactly as
the GPT-OSS integration does (gpt_oss/tt/attention/weights.py).
"""

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference.ref_layer import pcc

NUM_HEADS = 64
HEAD_DIM = 512
WINDOW = 128
BLOCK = 64


def _ref_decode(q, k, sink, pos, scale, window):
    """q [B, H, D], k [B, S, D] (K == V), sink [H], pos [B] -> [B, H, D] (fp32 math)."""
    out = torch.zeros_like(q, dtype=torch.float32)
    for b in range(q.size(0)):
        lo = max(0, pos[b] + 1 - window) if window else 0
        kk = k[b, lo : pos[b] + 1].float()
        s = (q[b].float() @ kk.T) * scale  # [H, n]
        m = s.amax(-1, keepdim=True)
        p = torch.exp(s - m)
        denom = p.sum(-1, keepdim=True) + torch.exp(sink.float().view(-1, 1) - m)
        out[b] = (p @ kk) / denom
    return out


@pytest.mark.parametrize("batch", [4, 8])
@pytest.mark.parametrize("cur_pos", [200, 700])
@pytest.mark.parametrize("window", [WINDOW, 0], ids=["window128", "full"])
def test_sdpa_decode_mqa_d512_sink(device, batch, cur_pos, window):
    torch.manual_seed(0)
    scale = HEAD_DIM**-0.5
    max_seq = 1024
    blocks_per_user = max_seq // BLOCK
    num_blocks = batch * blocks_per_user

    q = torch.randn(batch, NUM_HEADS, HEAD_DIM) * 0.5
    k = torch.randn(batch, max_seq, HEAD_DIM) * 0.5
    sink = torch.randn(NUM_HEADS) * 0.5
    pos = [cur_pos - 3 * b for b in range(batch)]

    # paged cache [num_blocks, 1, BLOCK, D] with a shuffled page table so addressing is actually tested
    perm = torch.randperm(num_blocks)
    page_table = perm.view(batch, blocks_per_user).to(torch.int32)
    cache = torch.zeros(num_blocks, 1, BLOCK, HEAD_DIM)
    for b in range(batch):
        for j in range(blocks_per_user):
            cache[perm[b * blocks_per_user + j], 0] = k[b, j * BLOCK : (j + 1) * BLOCK]

    tt_k = ttnn.from_torch(cache, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_v = ttnn.from_torch(cache, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_pt = ttnn.from_torch(page_table, device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_pos = ttnn.from_torch(torch.tensor(pos, dtype=torch.int32), device=device, dtype=ttnn.int32)

    # [1, B, H, D]: one query token per user, all 64 heads of the single KV head together
    tt_q = ttnn.from_torch(q.unsqueeze(0), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    # decode sinks: [H, TILE] with the value in column 0, pre-divided by scale
    sinks = torch.nn.functional.pad((sink / scale).view(-1, 1), (0, ttnn.TILE_SIZE - 1), value=0.0)
    tt_sink = ttnn.from_torch(sinks, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    prog = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(batch, 1),
        q_chunk_size=0,
        k_chunk_size=128,
        exp_approx_mode=False,
    )
    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    tt_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        tt_q,
        tt_k,
        tt_v,
        cur_pos_tensor=tt_pos,
        page_table_tensor=tt_pt,
        sliding_window_size=window or None,
        attention_sink=tt_sink,
        scale=scale,
        program_config=prog,
        compute_kernel_config=ckc,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out = ttnn.to_torch(tt_out).reshape(batch, -1, HEAD_DIM)[:, :NUM_HEADS].float()

    ref = _ref_decode(q, k, sink, pos, scale, window)
    got = pcc(out, ref)
    print(f"batch={batch} pos={cur_pos} window={window}: PCC {got:.5f}")
    assert got > 0.99
