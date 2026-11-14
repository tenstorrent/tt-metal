# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Unit test for ``ttnn.experimental.deepseek.flash_mqa_decode`` against a torch reference of the
V4.1 decode attention: 16 local heads (TP4), one shared K == V head of 512, per-head sinks, over the
128-row sliding-window ring plus up to ``index_topk`` (512) CSA-selected rows.

    pytest models/experimental/deepseek_v41_flash/tests/decode/test_flash_mqa_decode_op.py
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc

HEADS = 16
HEAD_DIM = 512
WINDOW = 128
INDEX_TOPK = 512
SCALE = HEAD_DIM**-0.5


def _reference(q, ring, selected, ring_len, selected_len, sinks):
    """fp32 softmax attention of ``q`` ``[H, D]`` over the valid ring and selected rows (K == V)."""
    kv = torch.cat([ring[:ring_len], selected[:selected_len]], dim=0)
    scores = (q @ kv.T) * SCALE
    if sinks is not None:
        scores = torch.cat([scores, sinks[:, None]], dim=-1)
    probs = torch.softmax(scores, dim=-1)[:, : kv.shape[0]]
    return probs @ kv


def _apply_rope_torch(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_dim: int) -> torch.Tensor:
    """Interleaved RoPE on the trailing ``rope_dim`` channels of ``x`` ``[H, D]``."""
    nope, rope = x[:, :-rope_dim], x[:, -rope_dim:]
    pairs = rope.reshape(rope.shape[0], -1, 2)
    rotated = torch.stack([-pairs[..., 1], pairs[..., 0]], dim=-1).reshape_as(rope)
    return torch.cat([nope, rope * cos + rotated * sin], dim=-1)


def _rope_tables(rope_dim: int, device, rows: int = 1):
    angles = torch.rand(rows, rope_dim // 2) * 6.28
    cos_full = angles.cos().repeat_interleave(2, dim=-1).reshape(1, 1, rows, rope_dim)
    sin_full = angles.sin().repeat_interleave(2, dim=-1).reshape(1, 1, rows, rope_dim)
    tt_cos = ttnn.from_torch(cos_full, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_neg_sin = ttnn.from_torch(-sin_full, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    cos_ref = cos_full[0, 0, 0].bfloat16().float()
    neg_sin_ref = (-sin_full[0, 0, 0]).bfloat16().float()
    return tt_cos, tt_neg_sin, cos_ref, neg_sin_ref


def _q_memory_config(device):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [HEADS, HEAD_DIM], ttnn.ShardOrientation.ROW_MAJOR),
    )


def _lengths_tensor(ring_len, selected_len):
    return ttnn.from_torch(torch.tensor([[ring_len, selected_len]], dtype=torch.int32), dtype=ttnn.int32)


@pytest.mark.parametrize(
    "ring_len, selected_len",
    (
        (1, 0),
        (37, 0),
        (64, 0),
        (128, 0),
        (128, 1),
        (128, 63),
        (128, 64),
        (128, 300),
        (128, INDEX_TOPK),
        (77, 3),
    ),
)
@pytest.mark.parametrize("k_chunk_size, num_s_blocks", ((64, 10), (64, 8), (128, 5), (64, 1), (64, 3)))
@pytest.mark.parametrize("use_sink", (True, False))
def test_flash_mqa_decode(device, reset_seeds, ring_len, selected_len, k_chunk_size, num_s_blocks, use_sink):
    q = torch.randn(HEADS, HEAD_DIM).bfloat16().float()
    ring = torch.randn(WINDOW, HEAD_DIM).bfloat16().float()
    selected = torch.randn(INDEX_TOPK, HEAD_DIM).bfloat16().float()
    sinks = torch.randn(HEADS) if use_sink else None
    expected = _reference(q, ring, selected, ring_len, selected_len, sinks)

    # The model's sel_tile: the ring copy in rows [0, WINDOW), the selected rows after it.
    sel_tile = torch.cat([torch.zeros(WINDOW, HEAD_DIM), selected], dim=0)
    q_mem = _q_memory_config(device)
    tt_q = ttnn.from_torch(
        q.reshape(1, 1, HEADS, HEAD_DIM),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=q_mem,
    )
    tt_ring = ttnn.from_torch(
        ring.reshape(1, 1, WINDOW, HEAD_DIM), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    tt_sel = ttnn.from_torch(
        sel_tile.reshape(1, 1, WINDOW + INDEX_TOPK, HEAD_DIM),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    tt_sink = None
    if use_sink:
        sink_tile = torch.zeros(HEADS, ttnn.TILE_SIZE)
        sink_tile[:, 0] = sinks / SCALE
        tt_sink = ttnn.from_torch(sink_tile, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_lengths = ttnn.from_torch(
        torch.tensor([[ring_len, selected_len]], dtype=torch.int32), dtype=ttnn.int32, device=device
    )

    # Twice: the second call runs from the program cache.
    for _ in range(2):
        tt_out = ttnn.experimental.deepseek.flash_mqa_decode(
            tt_q,
            tt_ring,
            tt_lengths,
            selected_kv=tt_sel,
            selected_row_offset=WINDOW,
            attention_sink=tt_sink,
            scale=SCALE,
            k_chunk_size=k_chunk_size,
            num_s_blocks=num_s_blocks,
        )
    assert tt_out.memory_config() == q_mem
    actual = ttnn.to_torch(tt_out).float().reshape(HEADS, HEAD_DIM)
    passing, pcc = comp_pcc(expected, actual, 0.999)
    logger.info(f"ring {ring_len} selected {selected_len}: PCC {pcc}")
    assert passing, f"PCC {pcc}"


@pytest.mark.parametrize("kv_dtype", (ttnn.bfloat16, ttnn.bfloat8_b))
def test_flash_mqa_decode_lengths_update(device, reset_seeds, kv_dtype):
    """Lengths are read on device: rewriting them in place changes the attended rows without a new program."""
    q = torch.randn(HEADS, HEAD_DIM).bfloat16().float()
    ring = torch.randn(WINDOW, HEAD_DIM).bfloat16().float()
    selected = torch.randn(INDEX_TOPK, HEAD_DIM).bfloat16().float()
    sinks = torch.randn(HEADS)
    tt_q = ttnn.from_torch(
        q.reshape(1, 1, HEADS, HEAD_DIM),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=_q_memory_config(device),
    )
    tt_ring = ttnn.from_torch(
        ring.reshape(1, 1, WINDOW, HEAD_DIM), dtype=kv_dtype, layout=ttnn.TILE_LAYOUT, device=device
    )
    tt_sel = ttnn.from_torch(
        selected.reshape(1, 1, INDEX_TOPK, HEAD_DIM), dtype=kv_dtype, layout=ttnn.TILE_LAYOUT, device=device
    )
    ring_q = ttnn.to_torch(tt_ring).float().reshape(WINDOW, HEAD_DIM)
    sel_q = ttnn.to_torch(tt_sel).float().reshape(INDEX_TOPK, HEAD_DIM)
    sink_tile = torch.zeros(HEADS, ttnn.TILE_SIZE)
    sink_tile[:, 0] = sinks / SCALE
    tt_sink = ttnn.from_torch(sink_tile, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_lengths = ttnn.from_torch(torch.tensor([[1, 0]], dtype=torch.int32), dtype=ttnn.int32, device=device)

    cache_entries = None
    for ring_len, selected_len in ((1, 0), (50, 0), (128, 2), (128, 200), (128, INDEX_TOPK), (90, 0)):
        ttnn.copy_host_to_device_tensor(_lengths_tensor(ring_len, selected_len), tt_lengths)
        tt_out = ttnn.experimental.deepseek.flash_mqa_decode(
            tt_q, tt_ring, tt_lengths, selected_kv=tt_sel, attention_sink=tt_sink, scale=SCALE
        )
        actual = ttnn.to_torch(tt_out).float().reshape(HEADS, HEAD_DIM)
        expected = _reference(q, ring_q, sel_q, ring_len, selected_len, sinks)
        passing, pcc = comp_pcc(expected, actual, 0.999)
        logger.info(f"ring {ring_len} selected {selected_len}: PCC {pcc}")
        assert passing, f"ring {ring_len} selected {selected_len}: PCC {pcc}"
        if cache_entries is None:
            cache_entries = device.num_program_cache_entries()
    assert device.num_program_cache_entries() == cache_entries


def test_flash_mqa_decode_session_slots(device, reset_seeds):
    """Several sessions' rings in one ``kv`` (the server's slots): ``ring_rows`` caps the ring at one window and
    the lengths' third entry picks the slot, rewritten on device between calls."""
    slots = 4
    q = torch.randn(HEADS, HEAD_DIM).bfloat16().float()
    rings = torch.randn(slots * WINDOW, HEAD_DIM).bfloat16().float()
    selected = torch.randn(INDEX_TOPK, HEAD_DIM).bfloat16().float()
    sinks = torch.randn(HEADS)
    tt_q = ttnn.from_torch(
        q.reshape(1, 1, HEADS, HEAD_DIM),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=_q_memory_config(device),
    )
    tt_rings = ttnn.from_torch(
        rings.reshape(1, 1, slots * WINDOW, HEAD_DIM), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    tt_sel = ttnn.from_torch(
        selected.reshape(1, 1, INDEX_TOPK, HEAD_DIM), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    sink_tile = torch.zeros(HEADS, ttnn.TILE_SIZE)
    sink_tile[:, 0] = sinks / SCALE
    tt_sink = ttnn.from_torch(sink_tile, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_lengths = ttnn.from_torch(torch.tensor([[1, 0, 0]], dtype=torch.int32), dtype=ttnn.int32, device=device)

    cache_entries = None
    for slot, ring_len, selected_len in ((0, 128, 512), (3, 128, 300), (1, 40, 0), (2, 128, 7), (3, 1, 0)):
        host = torch.tensor([[ring_len, selected_len, slot * WINDOW]], dtype=torch.int32)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(host, dtype=ttnn.int32), tt_lengths)
        tt_out = ttnn.experimental.deepseek.flash_mqa_decode(
            tt_q, tt_rings, tt_lengths, selected_kv=tt_sel, ring_rows=WINDOW, attention_sink=tt_sink, scale=SCALE
        )
        actual = ttnn.to_torch(tt_out).float().reshape(HEADS, HEAD_DIM)
        ring = rings[slot * WINDOW : (slot + 1) * WINDOW]
        expected = _reference(q, ring, selected, ring_len, selected_len, sinks)
        passing, pcc = comp_pcc(expected, actual, 0.999)
        logger.info(f"slot {slot} ring {ring_len} selected {selected_len}: PCC {pcc}")
        assert passing, f"slot {slot} ring {ring_len} selected {selected_len}: PCC {pcc}"
        if cache_entries is None:
            cache_entries = device.num_program_cache_entries()
    assert device.num_program_cache_entries() == cache_entries


def test_flash_mqa_decode_selected_slots(device, reset_seeds):
    """Several sessions' rings and selected rows (the server's ``dense_kv`` slots): ``selected_rows`` caps the
    selected segment at ``index_topk`` and the lengths' fourth entry picks its slot, rewritten on device."""
    slots = 3
    q = torch.randn(HEADS, HEAD_DIM).bfloat16().float()
    rings = torch.randn(slots * WINDOW, HEAD_DIM).bfloat16().float()
    dense = torch.randn(slots * INDEX_TOPK, HEAD_DIM).bfloat16().float()
    sinks = torch.randn(HEADS)
    tt_q = ttnn.from_torch(
        q.reshape(1, 1, HEADS, HEAD_DIM),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=_q_memory_config(device),
    )
    tt_rings = ttnn.from_torch(
        rings.reshape(1, 1, slots * WINDOW, HEAD_DIM), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    tt_dense = ttnn.from_torch(
        dense.reshape(1, 1, slots * INDEX_TOPK, HEAD_DIM), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    sink_tile = torch.zeros(HEADS, ttnn.TILE_SIZE)
    sink_tile[:, 0] = sinks / SCALE
    tt_sink = ttnn.from_torch(sink_tile, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_lengths = ttnn.from_torch(torch.tensor([[1, 0, 0, 0]], dtype=torch.int32), dtype=ttnn.int32, device=device)

    cache_entries = None
    for slot, ring_len, selected_len in ((0, 128, 512), (2, 128, 512), (1, 40, 3), (2, 128, 300), (1, 1, 0)):
        host = torch.tensor([[ring_len, selected_len, slot * WINDOW, slot * INDEX_TOPK]], dtype=torch.int32)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(host, dtype=ttnn.int32), tt_lengths)
        tt_out = ttnn.experimental.deepseek.flash_mqa_decode(
            tt_q,
            tt_rings,
            tt_lengths,
            selected_kv=tt_dense,
            ring_rows=WINDOW,
            selected_rows=INDEX_TOPK,
            attention_sink=tt_sink,
            scale=SCALE,
        )
        actual = ttnn.to_torch(tt_out).float().reshape(HEADS, HEAD_DIM)
        ring = rings[slot * WINDOW : (slot + 1) * WINDOW]
        selected = dense[slot * INDEX_TOPK : (slot + 1) * INDEX_TOPK]
        expected = _reference(q, ring, selected, ring_len, selected_len, sinks)
        passing, pcc = comp_pcc(expected, actual, 0.999)
        logger.info(f"slot {slot} ring {ring_len} selected {selected_len}: PCC {pcc}")
        assert passing, f"slot {slot} ring {ring_len} selected {selected_len}: PCC {pcc}"
        if cache_entries is None:
            cache_entries = device.num_program_cache_entries()
    assert device.num_program_cache_entries() == cache_entries


@pytest.mark.parametrize(
    "ring_len, selected_len",
    (
        (1, 0),
        (37, 0),
        (128, INDEX_TOPK),
        (77, 3),
    ),
)
@pytest.mark.parametrize("num_s_blocks", (1, 8))
@pytest.mark.parametrize("use_sink", (True, False))
@pytest.mark.parametrize("kv_dtype", (ttnn.bfloat16, ttnn.bfloat8_b))
def test_flash_mqa_decode_rope(device, reset_seeds, ring_len, selected_len, num_s_blocks, use_sink, kv_dtype):
    """Fused inverse RoPE on the trailing 64 columns across single/multi-chunk, S-blocks, sinks, and KV dtypes."""
    rope_dim = 64
    q = torch.randn(HEADS, HEAD_DIM).bfloat16().float()
    ring = torch.randn(WINDOW, HEAD_DIM).bfloat16().float()
    selected = torch.randn(INDEX_TOPK, HEAD_DIM).bfloat16().float()
    sinks = torch.randn(HEADS) if use_sink else None

    q_mem = _q_memory_config(device)
    tt_q = ttnn.from_torch(
        q.reshape(1, 1, HEADS, HEAD_DIM),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=q_mem,
    )
    tt_ring = ttnn.from_torch(
        ring.reshape(1, 1, WINDOW, HEAD_DIM), dtype=kv_dtype, layout=ttnn.TILE_LAYOUT, device=device
    )
    sel_tile = torch.cat([torch.zeros(WINDOW, HEAD_DIM), selected], dim=0)
    tt_sel = ttnn.from_torch(
        sel_tile.reshape(1, 1, WINDOW + INDEX_TOPK, HEAD_DIM),
        dtype=kv_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    ring_q = ttnn.to_torch(tt_ring).float().reshape(WINDOW, HEAD_DIM)
    sel_q = ttnn.to_torch(tt_sel).float().reshape(WINDOW + INDEX_TOPK, HEAD_DIM)[WINDOW:]

    tt_sink = None
    if use_sink:
        sink_tile = torch.zeros(HEADS, ttnn.TILE_SIZE)
        sink_tile[:, 0] = sinks / SCALE
        tt_sink = ttnn.from_torch(sink_tile, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_lengths = ttnn.from_torch(
        torch.tensor([[ring_len, selected_len]], dtype=torch.int32), dtype=ttnn.int32, device=device
    )
    tt_cos, tt_neg_sin, cos_ref, neg_sin_ref = _rope_tables(rope_dim, device)
    attn_ref = _reference(q, ring_q, sel_q, ring_len, selected_len, sinks)
    expected = _apply_rope_torch(attn_ref, cos_ref, neg_sin_ref, rope_dim)

    for _ in range(2):
        tt_out = ttnn.experimental.deepseek.flash_mqa_decode(
            tt_q,
            tt_ring,
            tt_lengths,
            selected_kv=tt_sel,
            selected_row_offset=WINDOW,
            attention_sink=tt_sink,
            scale=SCALE,
            k_chunk_size=64,
            num_s_blocks=num_s_blocks,
            cos=tt_cos,
            sin=tt_neg_sin,
        )
    assert tt_out.memory_config() == q_mem
    actual = ttnn.to_torch(tt_out).float().reshape(HEADS, HEAD_DIM)
    passing, pcc = comp_pcc(expected, actual, 0.999)
    rope_passing, rope_pcc = comp_pcc(expected[:, -rope_dim:], actual[:, -rope_dim:], 0.999)
    logger.info(f"ring {ring_len} selected {selected_len} s={num_s_blocks}: PCC {pcc}, rope PCC {rope_pcc}")
    assert passing, f"PCC {pcc}"
    assert rope_passing, f"rope PCC {rope_pcc}"


@pytest.mark.parametrize("rope_dim", (32, 64))
def test_flash_mqa_decode_rope_runtime_args_and_dims(device, reset_seeds, rope_dim):
    """Back-to-back calls with distinct cos/sin buffers reuse the cached program and update runtime args."""
    ring_len, selected_len = 128, 256
    q = torch.randn(HEADS, HEAD_DIM).bfloat16().float()
    ring = torch.randn(WINDOW, HEAD_DIM).bfloat16().float()
    selected = torch.randn(INDEX_TOPK, HEAD_DIM).bfloat16().float()
    sinks = torch.randn(HEADS)

    tt_q = ttnn.from_torch(
        q.reshape(1, 1, HEADS, HEAD_DIM),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=_q_memory_config(device),
    )
    tt_ring = ttnn.from_torch(
        ring.reshape(1, 1, WINDOW, HEAD_DIM), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    tt_sel = ttnn.from_torch(
        selected.reshape(1, 1, INDEX_TOPK, HEAD_DIM), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    sink_tile = torch.zeros(HEADS, ttnn.TILE_SIZE)
    sink_tile[:, 0] = sinks / SCALE
    tt_sink = ttnn.from_torch(sink_tile, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_lengths = ttnn.from_torch(
        torch.tensor([[ring_len, selected_len]], dtype=torch.int32), dtype=ttnn.int32, device=device
    )

    attn_ref = _reference(q, ring, selected, ring_len, selected_len, sinks)
    tables = [_rope_tables(rope_dim, device, rows=64) for _ in range(3)]
    assert len({t[0].buffer_address() for t in tables}) == 3

    cache_entries = None
    for idx, (tt_cos, tt_neg_sin, cos_ref, neg_sin_ref) in enumerate(tables):
        tt_out = ttnn.experimental.deepseek.flash_mqa_decode(
            tt_q,
            tt_ring,
            tt_lengths,
            selected_kv=tt_sel,
            attention_sink=tt_sink,
            scale=SCALE,
            cos=tt_cos,
            sin=tt_neg_sin,
        )
        actual = ttnn.to_torch(tt_out).float().reshape(HEADS, HEAD_DIM)
        expected = _apply_rope_torch(attn_ref, cos_ref, neg_sin_ref, rope_dim)
        passing, pcc = comp_pcc(expected, actual, 0.999)
        rope_passing, rope_pcc = comp_pcc(expected[:, -rope_dim:], actual[:, -rope_dim:], 0.999)
        logger.info(f"rope_dim={rope_dim} step {idx}: PCC {pcc}, rope PCC {rope_pcc}")
        assert passing and rope_passing, f"step {idx}: PCC {pcc}, rope PCC {rope_pcc}"
        if cache_entries is None:
            cache_entries = device.num_program_cache_entries()
        else:
            assert device.num_program_cache_entries() == cache_entries


@pytest.mark.parametrize("with_rope", (False, True), ids=("no_rope", "fused_rope"))
def test_flash_mqa_decode_rope_perf(device, reset_seeds, with_rope):
    """Warm dispatches of flash_mqa_decode with vs without fused inverse RoPE for Tracy timing:

    python -m tracy -r -p -v -m pytest models/experimental/deepseek_v41_flash/tests/decode/test_flash_mqa_decode_op.py -k rope_perf
    """
    rope_dim = 64
    ring_len, selected_len = WINDOW, INDEX_TOPK
    q = torch.randn(HEADS, HEAD_DIM).bfloat16().float()
    ring = torch.randn(WINDOW, HEAD_DIM).bfloat16().float()
    selected = torch.randn(INDEX_TOPK, HEAD_DIM).bfloat16().float()
    sinks = torch.randn(HEADS)

    tt_q = ttnn.from_torch(
        q.reshape(1, 1, HEADS, HEAD_DIM),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=_q_memory_config(device),
    )
    tt_ring = ttnn.from_torch(
        ring.reshape(1, 1, WINDOW, HEAD_DIM), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    tt_sel = ttnn.from_torch(
        selected.reshape(1, 1, INDEX_TOPK, HEAD_DIM), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    sink_tile = torch.zeros(HEADS, ttnn.TILE_SIZE)
    sink_tile[:, 0] = sinks / SCALE
    tt_sink = ttnn.from_torch(sink_tile, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_lengths = ttnn.from_torch(
        torch.tensor([[ring_len, selected_len]], dtype=torch.int32), dtype=ttnn.int32, device=device
    )
    tt_cos, tt_neg_sin, cos_ref, neg_sin_ref = _rope_tables(rope_dim, device)

    for _ in range(10):
        tt_out = ttnn.experimental.deepseek.flash_mqa_decode(
            tt_q,
            tt_ring,
            tt_lengths,
            selected_kv=tt_sel,
            attention_sink=tt_sink,
            scale=SCALE,
            k_chunk_size=64,
            num_s_blocks=8,
            cos=tt_cos if with_rope else None,
            sin=tt_neg_sin if with_rope else None,
        )
        ttnn.deallocate(tt_out)
    ttnn.synchronize_device(device)

    tt_out = ttnn.experimental.deepseek.flash_mqa_decode(
        tt_q,
        tt_ring,
        tt_lengths,
        selected_kv=tt_sel,
        attention_sink=tt_sink,
        scale=SCALE,
        k_chunk_size=64,
        num_s_blocks=8,
        cos=tt_cos if with_rope else None,
        sin=tt_neg_sin if with_rope else None,
    )
    expected = _reference(q, ring, selected, ring_len, selected_len, sinks)
    if with_rope:
        expected = _apply_rope_torch(expected, cos_ref, neg_sin_ref, rope_dim)
    actual = ttnn.to_torch(tt_out).float().reshape(HEADS, HEAD_DIM)
    passing, pcc = comp_pcc(expected, actual, 0.999)
    assert passing, f"with_rope={with_rope}: PCC {pcc}"
