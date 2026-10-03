# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Which SDPA is more accurate on REAL data? Layer 0 (window-only) at S=24 on the real attention inputs of a prefill dump: the device q (RoPE'd) and the device
cache rows are read back and the attention is recomputed in fp32 on the CPU (golden). ``scaled_dot_product_attention_decode`` (the existing path) and
``sparse_sdpa`` (the paged path) are both compared with that golden on IDENTICAL inputs. Env DSV41_REAL_DIR (dump dir), DSV41_PROBE_LAYER (default 0), DSV41_PROBE_S.
"""

import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference import ref_kernels
from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.tt.attention import HEAD_DIM, NH, WINDOW, DSV41Attention
from models.demos.blackhole.deepseek_v41_flash.tt.paged_attention import DSV41PagedAttention, DSV41PagedStepState
from models.demos.blackhole.deepseek_v41_flash.tt.paged_ops import PagedKVPool
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager

DP = {"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 100_000_000}


def pcc(a, b):
    return R.pcc(a.reshape(-1).float(), b.reshape(-1).float())


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [pytest.param(DP, id="ring")], indirect=True)
@torch.no_grad()
def test_sdpa_numerics(mesh_device):
    ref_kernels.FAKE_QUANT = True
    md = mesh_device
    rows, cols = tuple(md.shape)
    per_row, B = 4, rows * 4
    layer_id = int(os.environ.get("DSV41_PROBE_LAYER", "0"))
    S = int(os.environ.get("DSV41_PROBE_S", "24"))
    real = os.environ["DSV41_REAL_DIR"]
    assert layer_id == 0, "window-only probe"
    blk = R.build_layer(layer_id, max_batch_size=B, max_seq_len=1024)
    a_in = torch.load(f"{real}/layer_{layer_id}.pt")["prefill"]["attn_in"][:1].to(torch.bfloat16)
    x_pre, x_dec = a_in[:, :S].expand(B, -1, -1).contiguous(), a_in[:, S : S + 1].expand(B, -1, -1).contiguous()
    blk.attn(x_pre, 0)
    window = blk.attn.window_kv_cache.clone().float()
    ref = blk.attn(x_dec, S).float().reshape(B, 5120)
    mesh_config = mesh_4x8()
    ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
    w = R.dequantized_attention_weights(blk)
    shard = ttnn.ShardTensor2dMesh(md, dims=(2, None), mesh_shape=(rows, cols))
    tt_x = ttnn.from_torch(
        x_dec.reshape(1, 1, B, 5120).to(torch.bfloat16),
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=shard,
    )
    to_host = lambda out: torch.cat(
        [ttnn.to_torch(ttnn.get_device_tensors(out)[r * cols]).reshape(-1, 5120) for r in range(rows)]
    ).float()[:B]

    cur = DSV41Attention(md, mesh_config, ccl, w, blk.attn.freqs_cis, users_per_row=per_row, max_seq=256)
    cur.load_window(window[:, :S])
    st = cur.step_inputs(torch.full((B,), S))
    got_cur = to_host(cur.forward(tt_x, st))  # full existing forward (also writes the new row into cur.cache)
    kvp = PagedKVPool(md, per_row, num_pages=per_row * 2, n_ring_layers=1, max_ctx=256)
    for b in range(B):
        kvp.admit(b, S + 1)
    kvp.sync_page_table()
    kvp.stage_begin()
    pa = DSV41PagedAttention(md, mesh_config, ccl, w, blk.attn.freqs_cis, kvp, 0, users_per_row=per_row)
    pa.load_ring(window)
    kvp.stage_commit()
    pos = ttnn.from_torch(
        torch.full((B,), S, dtype=torch.int32),
        device=md,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(rows, cols)),
    )
    got_pg = to_host(pa.forward(tt_x, DSV41PagedStepState(pa, max_pos=1024).build(pos)))
    print(
        f"PROBE full layer-{layer_id} output vs reference: existing {pcc(got_cur, ref):.5f}  paged {pcc(got_pg, ref):.5f}",
        flush=True,
    )

    # ---- the SDPA calls on identical inputs
    ss = DSV41PagedStepState(pa, max_pos=1024).build(pos)
    st2 = cur.step_inputs(torch.full((B,), S))
    q, kv, k = cur._qkv(tt_x, st2)  # q [1,T,32,512] RoPE'd (row 0 = kv vector, rows 1..8 = heads of this column)
    cur._write_cache(cur.cache, kv, st2["pos"])
    o_old = ttnn.transformer.scaled_dot_product_attention_decode(
        q,
        cur.cache,
        cur.cache,
        cur_pos_tensor=st2["pos"],
        sliding_window_size=WINDOW,
        attention_sink=cur.sinks,
        scale=cur.scale,
        program_config=cur._sdpa_cfg(128),
        compute_kernel_config=cur.ckc_sdpa,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    o_new = pa._paged_attend(q, None, ss, 0, 0, WINDOW)
    dev0 = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()
    qh, ko, oo, on = dev0(q), dev0(cur.cache), dev0(o_old), dev0(o_new)  # mesh row 0, column 0 (heads 0..7); users 0..3
    T = per_row
    sink = (w["attn_sink"].float()[:8]).reshape(8)  # model sink of column 0's heads
    worst = {"old": [], "new": []}
    for u in range(T):
        qu = qh.reshape(T, -1, HEAD_DIM)[u][1:NH]  # [8,512] heads
        K = ko.reshape(T, -1, HEAD_DIM)[u][max(0, S + 1 - WINDOW) : S + 1]  # the S+1 valid rows
        s = (qu @ K.T) * (HEAD_DIM**-0.5)
        s = torch.cat([s, sink.reshape(8, 1)], dim=-1)
        p = torch.softmax(s, dim=-1)[:, :-1]
        gold = p @ K
        worst["old"].append(pcc(oo.reshape(T, -1, HEAD_DIM)[u][1:NH], gold))
        worst["new"].append(pcc(on.reshape(T, -1, HEAD_DIM)[u][1:NH], gold))
    print(
        f"PROBE SDPA output vs fp32 golden on identical q/kv (mesh row 0, col 0, per user): "
        f"scaled_dot_product_attention_decode {[round(v, 5) for v in worst['old']]}  sparse_sdpa {[round(v, 5) for v in worst['new']]}",
        flush=True,
    )
    # q magnitude / score range, to see how peaked the real attention is
    print(
        f"PROBE q rms {qh.reshape(T, -1, HEAD_DIM)[0][1:NH].pow(2).mean().sqrt():.3f}  score absmax {float((qh.reshape(T, -1, HEAD_DIM)[0][1:NH] @ ko.reshape(T, -1, HEAD_DIM)[0][: S + 1].T).abs().max() * HEAD_DIM**-0.5):.2f}",
        flush=True,
    )
