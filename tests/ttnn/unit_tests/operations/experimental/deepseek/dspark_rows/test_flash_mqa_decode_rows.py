# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""``ttnn.experimental.deepseek.flash_mqa_decode`` with a block of ``num_rows`` query rows (DSpark verify).

Row ``j`` of a block at positions ``p .. p + R - 1`` attends:
  * ring slots of the old positions ``max(0, p + j - (W - 1)) .. p - 1`` (slot = position % W; the ring holds
    positions <= p - 1 only);
  * the block's new kv rows ``0 .. j`` (rows of the staging tensor ``new_kv``);
  * its selected rows: ``selected_kv[j * topk : j * topk + n_j]`` (per-row selections), or the first ``n_j`` rows
    of a shared ``dense_kv`` (``dense_selected=True``);
  * the per-head attention sink.
The output is the RoPE-undone block, TILE ``(R, 32)`` ``[1, 1, R, H * Dh]``.

Correctness::

    pytest tests/ttnn/unit_tests/operations/experimental/deepseek/dspark_rows/test_flash_mqa_decode_rows.py -k "not perf"

Perf (s = 1 vs s = 4 at production shapes; read with dv_prof.py <csv> FlashMqaDecode)::

    python -m tracy -r -p -v --no-web-server -t 8123 -o <out> -m pytest <this file> -k perf -s
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
ROPE_DIM = 64
SCALE = HEAD_DIM**-0.5
K_CHUNK = 64
S_BLOCKS = 8
COMPUTE = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
)


def _core00():
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})


def _sharded(rows, width):
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(_core00(), [rows, width], ttnn.ShardOrientation.ROW_MAJOR),
    )


def _attend(q, kv, sinks):
    """fp32 softmax attention of q [H, D] over kv [N, D] (K == V) with an optional per-head sink."""
    scores = (q @ kv.T) * SCALE
    if sinks is not None:
        scores = torch.cat([scores, sinks[:, None]], dim=-1)
    probs = torch.softmax(scores, dim=-1)[:, : kv.shape[0]]
    return probs @ kv


def _rope(x, cos, sin):
    """Interleaved RoPE on the trailing ROPE_DIM channels of x [H, D]; cos / sin [ROPE_DIM]."""
    nope, rope = x[:, :-ROPE_DIM], x[:, -ROPE_DIM:]
    pairs = rope.reshape(rope.shape[0], -1, 2)
    rotated = torch.stack([-pairs[..., 1], pairs[..., 0]], dim=-1).reshape_as(rope)
    return torch.cat([nope, rope * cos + rotated * sin], dim=-1)


def _row_keys(ring, new, p, j):
    """Ring rows then new rows that block row j attends (the softmax is order-free)."""
    lo = max(0, p + j - (WINDOW - 1))
    slots = [pos % WINDOW for pos in range(lo, p)]
    return torch.cat([ring[slots], new[: j + 1]], dim=0)


def _rope_tables(rows):
    angles = torch.rand(rows, ROPE_DIM // 2) * 6.28
    cos = angles.cos().repeat_interleave(2, dim=-1).bfloat16().float()
    neg_sin = (-angles.sin().repeat_interleave(2, dim=-1)).bfloat16().float()
    return cos, neg_sin


def _tile_rows(rows):
    return ttnn.Tile((rows, 32))


def _make_inputs(device, rows, p, n, mode, use_sink, kv_dtype=ttnn.bfloat16, q_memory_config=None):
    """Host data and device tensors for one block. mode: None (sliding layer), "per_row" or "dense"."""
    q = torch.randn(HEADS, rows, HEAD_DIM).bfloat16().float()  # [h, j, D]: device row h * rows + j
    ring = torch.randn(WINDOW, HEAD_DIM).bfloat16().float()
    new = torch.randn(32, HEAD_DIM).bfloat16().float()
    sinks = torch.randn(HEADS) if use_sink else None
    cos, neg_sin = _rope_tables(rows)

    tt = {}
    tt["q"] = ttnn.from_torch(
        q.reshape(1, 1, HEADS * rows, HEAD_DIM),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=q_memory_config or _sharded(HEADS * rows, HEAD_DIM),
    )
    tt["kv"] = ttnn.from_torch(
        ring.reshape(1, 1, WINDOW, HEAD_DIM), dtype=kv_dtype, layout=ttnn.TILE_LAYOUT, device=device
    )
    tt["new_kv"] = ttnn.from_torch(
        new.reshape(1, 1, 32, HEAD_DIM), dtype=kv_dtype, layout=ttnn.TILE_LAYOUT, device=device
    )
    ring = ttnn.to_torch(tt["kv"]).float().reshape(WINDOW, HEAD_DIM)
    new = ttnn.to_torch(tt["new_kv"]).float().reshape(32, HEAD_DIM)
    selected = None
    if mode is not None:
        sel_rows = rows * INDEX_TOPK if mode == "per_row" else INDEX_TOPK
        selected = torch.randn(sel_rows, HEAD_DIM).bfloat16().float()
        tt["selected_kv"] = ttnn.from_torch(
            selected.reshape(1, 1, sel_rows, HEAD_DIM), dtype=kv_dtype, layout=ttnn.TILE_LAYOUT, device=device
        )
        selected = ttnn.to_torch(tt["selected_kv"]).float().reshape(sel_rows, HEAD_DIM)
    if use_sink:
        sink_tile = torch.zeros(HEADS, 32)
        sink_tile[:, 0] = sinks / SCALE
        tt["attention_sink"] = ttnn.from_torch(sink_tile, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt["cos"] = ttnn.from_torch(
        cos.reshape(1, 1, rows, ROPE_DIM),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        tile=_tile_rows(rows),
        device=device,
    )
    tt["sin"] = ttnn.from_torch(
        neg_sin.reshape(1, 1, rows, ROPE_DIM),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        tile=_tile_rows(rows),
        device=device,
    )
    lengths = [p, 0, 0, 0] + list(n)
    tt["lengths"] = ttnn.from_torch(torch.tensor([lengths], dtype=torch.int32), dtype=ttnn.int32, device=device)

    expected = []
    for j in range(rows):
        keys = _row_keys(ring, new, p, j)
        if mode == "per_row":
            keys = torch.cat([keys, selected[j * INDEX_TOPK : j * INDEX_TOPK + n[j]]], dim=0)
        elif mode == "dense":
            keys = torch.cat([keys, selected[: n[j]]], dim=0)
        expected.append(_rope(_attend(q[:, j], keys, sinks), cos[j], neg_sin[j]))
    return tt, torch.stack(expected, dim=0)  # [j, h, D]


def _run_rows(tt, rows, mode, num_s_blocks=S_BLOCKS, memory_config=None, row_groups=0, k_chunk_size=K_CHUNK):
    return ttnn.experimental.deepseek.flash_mqa_decode(
        tt["q"],
        tt["kv"],
        tt["lengths"],
        selected_kv=tt.get("selected_kv"),
        ring_rows=WINDOW,
        selected_rows=INDEX_TOPK if mode is not None else 0,
        attention_sink=tt.get("attention_sink"),
        scale=SCALE,
        k_chunk_size=k_chunk_size,
        num_s_blocks=num_s_blocks,
        memory_config=memory_config,
        compute_kernel_config=COMPUTE,
        cos=tt["cos"],
        sin=tt["sin"],
        new_kv=tt["new_kv"],
        num_rows=rows,
        dense_selected=mode == "dense",
        row_groups=row_groups,
    )


def _out_memory_config(rows):
    return _sharded(rows, HEADS * HEAD_DIM)


def _check(tt_out, expected, rows, label):
    assert tuple(tt_out.shape) == (1, 1, rows, HEADS * HEAD_DIM), tt_out.shape
    assert tt_out.tile.tile_shape == [rows, 32] or tuple(tt_out.tile.tile_shape) == (rows, 32), tt_out.tile
    actual = ttnn.to_torch(tt_out).float().reshape(rows, HEADS, HEAD_DIM)
    pccs = [comp_pcc(expected[j], actual[j], 0.999)[1] for j in range(rows)]
    logger.info(f"{label}: per-row PCC {[f'{x:.5f}' for x in pccs]}")
    for j in range(rows):
        assert pccs[j] >= 0.999, f"{label}: row {j} PCC {pccs[j]}"


# p: 0 (empty ring), window filling (1, 5, 63, 64, 65, 100, 124..127), full (128, 129, 200, 1000, 1023)
POSITIONS = (0, 1, 5, 63, 64, 65, 100, 124, 125, 126, 127, 128, 129, 200, 1000, 1023)


@pytest.mark.parametrize("p", POSITIONS)
@pytest.mark.parametrize("use_sink", (True, False))
def test_rows_sliding(device, reset_seeds, p, use_sink):
    rows = 4
    tt, expected = _make_inputs(device, rows, p, [0] * rows, None, use_sink)
    for _ in range(2):  # second call from the program cache
        out = _run_rows(tt, rows, None, memory_config=_out_memory_config(rows))
    _check(out, expected, rows, f"sliding p={p} sink={use_sink}")


@pytest.mark.parametrize(
    "p, n",
    (
        (5, (0, 1, 2, 3)),
        (127, (1, 64, 65, 300)),
        (128, (512, 512, 512, 512)),
        (1000, (512, 511, 63, 0)),
        (300, (37, 128, 200, 450)),
    ),
)
@pytest.mark.parametrize("mode", ("per_row", "dense"))
@pytest.mark.parametrize("num_s_blocks", (8, 3, 1))
@pytest.mark.parametrize("row_groups", (0, 1, 2))
def test_rows_selected(device, reset_seeds, p, n, mode, num_s_blocks, row_groups):
    rows = 4
    tt, expected = _make_inputs(device, rows, p, n, mode, True)
    for _ in range(2):
        out = _run_rows(
            tt, rows, mode, num_s_blocks=num_s_blocks, memory_config=_out_memory_config(rows), row_groups=row_groups
        )
    _check(out, expected, rows, f"{mode} p={p} n={n} sb={num_s_blocks} rg={row_groups}")


@pytest.mark.parametrize("rows", (2, 8))
def test_rows_other_counts(device, reset_seeds, rows):
    n = tuple((37 * (j + 1)) % 513 for j in range(rows))
    tt, expected = _make_inputs(device, rows, 300, n, "per_row", True)
    out = _run_rows(tt, rows, "per_row", memory_config=_out_memory_config(rows))
    _check(out, expected, rows, f"rows={rows}")


def test_rows_lengths_update(device, reset_seeds):
    """Lengths are read on device: rewriting them changes the attended rows without a new program."""
    rows = 4
    cache_entries = None
    for p, n in ((128, (512, 512, 512, 512)), (3, (0, 0, 1, 1)), (700, (100, 200, 300, 400))):
        tt, expected = _make_inputs(device, rows, p, n, "per_row", True)
        out = _run_rows(tt, rows, "per_row", memory_config=_out_memory_config(rows))
        _check(out, expected, rows, f"update p={p}")
        if cache_entries is None:
            cache_entries = device.num_program_cache_entries()
    assert device.num_program_cache_entries() == cache_entries


# ---------------------------------------------------------------------------------------------- perf


def _perf_s1(device, mode):
    """Today's one-row call at production shapes: q / out [16, 512] on core (0, 0), fused RoPE, sink."""
    from tracy import signpost

    q = torch.randn(HEADS, HEAD_DIM)
    tt_q = ttnn.from_torch(
        q.reshape(1, 1, HEADS, HEAD_DIM),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=_sharded(HEADS, HEAD_DIM),
    )
    tt_ring = ttnn.from_torch(
        torch.randn(1, 1, WINDOW, HEAD_DIM), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    # The model's sel_tile: the window rows first, the selected rows after them.
    tt_sel = ttnn.from_torch(
        torch.randn(1, 1, WINDOW + INDEX_TOPK, HEAD_DIM), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    sink_tile = torch.zeros(HEADS, 32)
    sink_tile[:, 0] = torch.randn(HEADS)
    tt_sink = ttnn.from_torch(sink_tile, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    cos, neg_sin = _rope_tables(1)
    tt_cos = ttnn.from_torch(
        cos.reshape(1, 1, 1, ROPE_DIM), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    tt_sin = ttnn.from_torch(
        neg_sin.reshape(1, 1, 1, ROPE_DIM), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    sel = INDEX_TOPK if mode is not None else 0
    tt_len = ttnn.from_torch(torch.tensor([[WINDOW, sel]], dtype=torch.int32), dtype=ttnn.int32, device=device)

    def call():
        return ttnn.experimental.deepseek.flash_mqa_decode(
            tt_q,
            tt_ring,
            tt_len,
            selected_kv=tt_sel if mode is not None else None,
            selected_row_offset=WINDOW if mode is not None else 0,
            ring_rows=WINDOW,
            selected_rows=INDEX_TOPK if mode is not None else 0,
            attention_sink=tt_sink,
            scale=SCALE,
            k_chunk_size=K_CHUNK,
            num_s_blocks=S_BLOCKS,
            compute_kernel_config=COMPUTE,
            cos=tt_cos,
            sin=tt_sin,
        )

    ttnn.deallocate(call())
    ttnn.synchronize_device(device)
    signpost(f"s1-{mode or 'sliding'}")
    for _ in range(12):
        ttnn.deallocate(call())
    ttnn.synchronize_device(device)


def _perf_rows(device, rows, mode, num_s_blocks, row_groups, q_dram=False, k_chunk_size=K_CHUNK):
    from tracy import signpost

    n = [INDEX_TOPK] * rows if mode is not None else [0] * rows
    q_mem = ttnn.DRAM_MEMORY_CONFIG if q_dram else None
    tt, _ = _make_inputs(device, rows, 1000, n, mode, True, q_memory_config=q_mem)
    mem = _out_memory_config(rows)
    run = lambda: _run_rows(
        tt, rows, mode, num_s_blocks=num_s_blocks, memory_config=mem, row_groups=row_groups, k_chunk_size=k_chunk_size
    )
    ttnn.deallocate(run())
    ttnn.synchronize_device(device)
    kc = f"-kc{k_chunk_size}" if k_chunk_size != K_CHUNK else ""
    signpost(f"s{rows}-{mode or 'sliding'}-sb{num_s_blocks}-rg{row_groups or 'auto'}{'-qdram' if q_dram else ''}{kc}")
    for _ in range(12):
        ttnn.deallocate(run())
    ttnn.synchronize_device(device)


@pytest.mark.parametrize("mode", (None, "per_row", "dense"), ids=("sliding", "per_row", "dense"))
def test_perf_s1(device, mode):
    if mode == "dense":
        pytest.skip("s = 1 has one selected mode")
    _perf_s1(device, mode)


@pytest.mark.parametrize("num_s_blocks, row_groups", ((8, 0), (12, 0), (8, 2), (8, 1)))
@pytest.mark.parametrize("mode", (None, "per_row", "dense"), ids=("sliding", "per_row", "dense"))
def test_perf_s4(device, mode, num_s_blocks, row_groups):
    _perf_rows(device, 4, mode, num_s_blocks, row_groups)


@pytest.mark.parametrize("mode", (None, "per_row"), ids=("sliding", "per_row"))
def test_perf_s4_q_dram(device, mode):
    _perf_rows(device, 4, mode, S_BLOCKS, 0, q_dram=True)


@pytest.mark.parametrize("k_chunk_size", (64, 128))
@pytest.mark.parametrize("mode", (None, "per_row"), ids=("sliding", "per_row"))
def test_perf_zones(device, mode, k_chunk_size):
    """Short run for device-zone breakdowns (production shapes, auto row groups)."""
    _perf_rows(device, 4, mode, S_BLOCKS, 0, k_chunk_size=k_chunk_size)
