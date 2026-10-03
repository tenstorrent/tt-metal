# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Milestone 1: paged decode attention (sparse_sdpa + page-table pool, tt/paged_attention.py) vs the CURRENT contiguous-cache attention
(tt/attention.py) and vs the checkpoint's reference attention, one decode step, real layer weights, short context.

Window-only layer 0 and the kv-source layers 2 (ratio 2) and 20 (ratio 1). The page pool is allocated in a shuffled page order so the page-table
translation is exercised. Prints ``PAGED_PCC`` lines: paged vs current, paged vs reference, current vs reference."""

import os
import random

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference import ref_kernels
from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.tt.attention import DSV41Attention, DSV41CompressedAttention
from models.demos.blackhole.deepseek_v41_flash.tt.paged_attention import (
    DSV41PagedAttention,
    DSV41PagedCompressedAttention,
    DSV41PagedStepState,
)
from models.demos.blackhole.deepseek_v41_flash.tt.paged_ops import PagedKVPool
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager

DP = {"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 100_000_000}


def to_host(out, rows, cols, B):
    devs = ttnn.get_device_tensors(out)
    return torch.cat([ttnn.to_torch(devs[r * cols]).reshape(-1, 5120) for r in range(rows)]).float()[:B]


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [pytest.param(DP, id="ring")], indirect=True)
@pytest.mark.parametrize("layer_id,S", [(0, 24), (0, 200), (2, 23), (2, 24), (2, 200), (20, 24), (20, 300)])
@torch.no_grad()
def test_paged_vs_current(mesh_device, layer_id, S):
    torch.manual_seed(0)
    ref_kernels.FAKE_QUANT = True
    md = mesh_device
    rows, cols = tuple(md.shape)
    per_row, B = 4, rows * 4
    blk = R.build_layer(layer_id, max_batch_size=B, max_seq_len=1024)
    ratio = blk.attn.compress_ratio
    comp = blk.attn.compressor

    def block_input(tok):
        h, pm = R.embed_tokens(tok)
        return blk.attn_norm(blk.hc_pre(h, pm))

    x_pre = block_input(torch.randint(1000, 100000, (B, S)))
    x_dec = block_input(torch.randint(1000, 100000, (B, 1)))
    real = os.environ.get("DSV41_REAL_DIR")  # REAL attention inputs of a prefill dump (all users identical)
    if real:
        a_in = torch.load(f"{real}/layer_{layer_id}.pt")["prefill"]["attn_in"][:1].to(torch.bfloat16)
        x_pre, x_dec = a_in[:, :S].expand(B, -1, -1).contiguous(), a_in[:, S : S + 1].expand(B, -1, -1).contiguous()
    blk.attn(x_pre, 0)
    snap = dict(window=blk.attn.window_kv_cache.clone().float())
    if ratio:
        snap.update(
            comp=blk.attn.compress_kv_cache[:, : S // ratio].clone().float(),
            kv_state=comp.kv_state.clone() if ratio > 1 else None,
            score_state=comp.score_state.clone() if ratio > 1 else None,
        )
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
    comp_w = None
    if ratio:
        comp_w = {"wkv": comp.wkv.weight.data.float(), "norm": comp.norm.weight.data.float()}
        if ratio > 1:
            comp_w["wgate"] = comp.wgate.weight.data.float()

    # ---- current (contiguous cache) attention (only where it is valid: window cache 256 slots, all compressed entries selected)
    got_cur = None
    if ratio == 0:
        if S < 128:
            cur = DSV41Attention(md, mesh_config, ccl, w, blk.attn.freqs_cis, users_per_row=per_row, max_seq=256)
            cur.load_window(snap["window"][:, :S])
    elif S // ratio <= 512:
        mc = max(128, -(-(S // ratio + 1) // 32) * 32)
        cur = DSV41CompressedAttention(
            md, mesh_config, ccl, w, blk.attn.freqs_cis, ratio, comp_w, users_per_row=per_row, max_comp=mc
        )
        cur.load_state(snap["window"], snap["comp"], snap["kv_state"], snap["score_state"])
    if "cur" in locals():
        got_cur = to_host(cur.forward(tt_x, cur.step_inputs(torch.full((B,), S))), rows, cols, B)
    # ---- paged attention
    pages = -(-(S + 8) // 128)
    kvp = PagedKVPool(md, per_row, num_pages=per_row * (pages + 2), n_ring_layers=1, max_ctx=1024)
    for a in kvp.allocs:
        random.Random(1).shuffle(a._free)
    for b in range(B):
        kvp.admit(b, S + 1)
    kvp.sync_page_table()
    kvp.stage_begin()
    if ratio == 0:
        pa = DSV41PagedAttention(md, mesh_config, ccl, w, blk.attn.freqs_cis, kvp, 0, users_per_row=per_row)
        pa.load_ring(snap["window"])
    else:
        pa = DSV41PagedCompressedAttention(
            md, mesh_config, ccl, w, blk.attn.freqs_cis, ratio, comp_w, kvp, 0, layer_id, users_per_row=per_row
        )
        pa.load_state(snap["window"], snap["comp"], snap["kv_state"], snap["score_state"], start_pos=S)
    kvp.stage_commit()
    ss = DSV41PagedStepState(pa, max_pos=1024)
    pos = ttnn.from_torch(
        torch.full((B,), S, dtype=torch.int32),
        device=md,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(rows, cols)),
    )
    out = pa.forward(tt_x, ss.build(pos))
    got = to_host(out, rows, cols, B)
    p_ref = R.pcc(got, ref)
    msg = f"PAGED_PCC layer {layer_id} S={S} ratio={ratio}: paged vs reference {p_ref:.5f}"
    if got_cur is not None:
        msg += f" | paged vs current {R.pcc(got, got_cur):.5f} | current vs reference {R.pcc(got_cur, ref):.5f}"
    print(msg, flush=True)
    assert p_ref > 0.98
    if got_cur is not None:
        assert R.pcc(got, got_cur) > 0.999


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [pytest.param(DP, id="ring")], indirect=True)
@pytest.mark.parametrize("layer_id,S", [(0, 127), (2, 127), (20, 127)])
@torch.no_grad()
def test_paged_two_steps_page_boundary(mesh_device, layer_id, S):
    """Two consecutive decode steps with the position advanced ON THE DEVICE (as the decoder's device loop does) across a page boundary
    (position 127 -> 128: ring wrap, new 128-token page allocated by the host between the two replays, page table re-uploaded in place).
    """
    torch.manual_seed(0)
    ref_kernels.FAKE_QUANT = True
    md = mesh_device
    rows, cols = tuple(md.shape)
    per_row, B = 4, rows * 4
    blk = R.build_layer(layer_id, max_batch_size=B, max_seq_len=1024)
    ratio = blk.attn.compress_ratio
    comp = blk.attn.compressor

    def block_input(tok):
        h, pm = R.embed_tokens(tok)
        return blk.attn_norm(blk.hc_pre(h, pm))

    x_pre = block_input(torch.randint(1000, 100000, (B, S)))
    x1, x2 = block_input(torch.randint(1000, 100000, (B, 1))), block_input(torch.randint(1000, 100000, (B, 1)))
    blk.attn(x_pre, 0)
    snap = dict(window=blk.attn.window_kv_cache.clone().float())
    if ratio:
        snap.update(
            comp=blk.attn.compress_kv_cache[:, : S // ratio].clone().float(),
            kv_state=comp.kv_state.clone() if ratio > 1 else None,
            score_state=comp.score_state.clone() if ratio > 1 else None,
        )
    ref1 = blk.attn(x1, S).float().reshape(B, 5120)
    ref2 = blk.attn(x2, S + 1).float().reshape(B, 5120)
    mesh_config = mesh_4x8()
    ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
    w = R.dequantized_attention_weights(blk)
    shard = ttnn.ShardTensor2dMesh(md, dims=(2, None), mesh_shape=(rows, cols))
    tx = lambda x: ttnn.from_torch(
        x.reshape(1, 1, B, 5120).to(torch.bfloat16),
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=shard,
    )
    kvp = PagedKVPool(md, per_row, num_pages=per_row * 4, n_ring_layers=1, max_ctx=512)
    for a in kvp.allocs:
        random.Random(5).shuffle(a._free)
    for b in range(B):
        kvp.admit(b, S + 1)  # exactly one page: position 128 needs a new one
    kvp.sync_page_table()
    kvp.stage_begin()
    if ratio == 0:
        pa = DSV41PagedAttention(md, mesh_config, ccl, w, blk.attn.freqs_cis, kvp, 0, users_per_row=per_row)
        pa.load_ring(snap["window"])
    else:
        comp_w = {"wkv": comp.wkv.weight.data.float(), "norm": comp.norm.weight.data.float()}
        if ratio > 1:
            comp_w["wgate"] = comp.wgate.weight.data.float()
        pa = DSV41PagedCompressedAttention(
            md, mesh_config, ccl, w, blk.attn.freqs_cis, ratio, comp_w, kvp, 0, layer_id, users_per_row=per_row
        )
        pa.load_state(snap["window"], snap["comp"], snap["kv_state"], snap["score_state"], start_pos=S)
    kvp.stage_commit()
    ss = DSV41PagedStepState(pa, max_pos=1024)
    pos = ttnn.from_torch(
        torch.full((B,), S, dtype=torch.int32),
        device=md,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(rows, cols)),
    )
    g1 = to_host(pa.forward(tx(x1), ss.build(pos)), rows, cols, B)
    ttnn.copy(ttnn.add(pos, 1), pos)  # the decoder's device loop advances the position like this
    pages_before = kvp.table_host()[0].clone()
    assert kvp.ensure(torch.full((B,), S + 1), lookahead=0), "the host must have allocated the second page"
    assert (kvp.table_host()[0] != pages_before).sum() == 1
    g2 = to_host(pa.forward(tx(x2), ss.build(pos)), rows, cols, B)
    p1, p2 = R.pcc(g1, ref1), R.pcc(g2, ref2)
    print(
        f"PAGED_PCC two steps layer {layer_id} pos {S}->{S + 1} (page boundary, device-advanced position): step 1 {p1:.5f}, step 2 {p2:.5f}",
        flush=True,
    )
    assert p1 > 0.98 and p2 > 0.98
