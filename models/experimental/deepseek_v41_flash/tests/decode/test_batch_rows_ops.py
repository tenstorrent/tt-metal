# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""The B=4 batch step's independent-rows modes of ``flash_mqa_decode`` and ``compress_pool_decode``: row ``j`` is one
token of its own session (slot ``s_j``, position ``p_j``) in caches that hold several sessions' slots.

* ``flash_mqa_decode(..., independent_rows=True)`` against a torch reference: row ``j`` attends its own ring's
  positions ``max(0, p_j - 127) .. p_j - 1``, its own new kv row and its own selected (or dense) rows, then the row's
  inverse RoPE.
* ``compress_pool_decode(..., independent_rows=True)`` against ``R`` one-row calls on copies of the same caches: the
  windows, ``idx_key_cache``, ``comp_kv`` and ``dense_kv`` must end up as the one-row steps leave them, and a row that
  closes no group (entry -1) must write nothing but its window rows.

    pytest models/experimental/deepseek_v41_flash/tests/decode/test_batch_rows_ops.py
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc

R = 4
HEADS = 16
HEAD_DIM = 512
WINDOW = 128
INDEX_TOPK = 512
ROPE_DIM = 64
SLOTS = 5
SCALE = HEAD_DIM**-0.5


def _rope_torch(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_dim: int = ROPE_DIM) -> torch.Tensor:
    """Interleaved RoPE on the trailing ``rope_dim`` channels of ``x`` ``[..., D]``."""
    nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
    pairs = rope.reshape(*rope.shape[:-1], -1, 2)
    rotated = torch.stack([-pairs[..., 1], pairs[..., 0]], dim=-1).reshape_as(rope)
    return torch.cat([nope, rope * cos + rotated * sin], dim=-1)


def _int32(values, device) -> ttnn.Tensor:
    return ttnn.from_torch(
        torch.tensor(values, dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _tile(x: torch.Tensor, device, tile=None, layout=ttnn.TILE_LAYOUT) -> ttnn.Tensor:
    kw = {} if tile is None else {"tile": tile}
    return ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, **kw
    )


def _row_angles(rows: int, rope_dim: int = ROPE_DIM, seed: int = 0):
    """``(cos, sin)`` ``[rows, rope_dim]``, interleaved pairs, bf16-rounded."""
    gen = torch.Generator().manual_seed(seed)
    angles = torch.rand(rows, rope_dim // 2, generator=gen) * 6.28
    cos = angles.cos().repeat_interleave(2, dim=-1).bfloat16().float()
    sin = angles.sin().repeat_interleave(2, dim=-1).bfloat16().float()
    return cos, sin


# (positions, slots, selected counts): p = 0 (only its new row), a ring not yet full, a full ring on both sides of a
# wrap, and selected counts from 0 to index_topk that do and do not fill a 64-row chunk.
_FLASH_CASES = {
    "short": ([0, 5, 63, 64], [3, 0, 4, 1], [0, 1, 63, 64]),
    "mixed": ([127, 128, 300, 1000], [1, 2, 0, 3], [300, 512, 65, 7]),
    "wrap": ([1000, 129, 0, 4095], [4, 3, 2, 0], [512, 1, 0, 511]),
}


@pytest.mark.parametrize("dense", (False, True), ids=("selected_rows", "dense_kv"))
@pytest.mark.parametrize("case", list(_FLASH_CASES))
@pytest.mark.parametrize("use_sink", (True, False))
def test_flash_mqa_decode_independent_rows(device, reset_seeds, dense, case, use_sink):
    positions, slots, counts = _FLASH_CASES[case]
    torch.manual_seed(11)
    q = torch.randn(R, HEADS, HEAD_DIM).bfloat16().float()
    ring = torch.randn(SLOTS * WINDOW, HEAD_DIM).bfloat16().float()
    staging = torch.randn(32, HEAD_DIM).bfloat16().float()
    # Selected rows: row j's own index_topk rows of sel_tile_rows, or its session's rows of dense_kv.
    selected = torch.randn((SLOTS if dense else R) * INDEX_TOPK, HEAD_DIM).bfloat16().float()
    sinks = torch.randn(HEADS) if use_sink else None
    cos, sin = _row_angles(R, seed=3)

    expected = []
    for j, (p, s, n) in enumerate(zip(positions, slots, counts)):
        ring_pos = range(max(0, p - (WINDOW - 1)), p)
        keys = [ring[s * WINDOW + pos % WINDOW] for pos in ring_pos] + [staging[j]]
        first = (s if dense else j) * INDEX_TOPK
        kv = torch.cat([torch.stack(keys), selected[first : first + n]], dim=0)
        scores = (q[j] @ kv.T) * SCALE
        if sinks is not None:
            scores = torch.cat([scores, sinks[:, None]], dim=-1)
        probs = torch.softmax(scores, dim=-1)[:, : kv.shape[0]]
        expected.append(_rope_torch(probs @ kv, cos[j], sin[j]))

    # q head-major ([1, 1, H * R, Dh], row h * R + j), 32 rows a core over row 0, as attention_qkv_decode leaves it.
    q_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(HEADS * R // 32 - 1, 0))})
    q_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(q_grid, [32, HEAD_DIM], ttnn.ShardOrientation.ROW_MAJOR),
    )
    tt_q = ttnn.from_torch(
        q.permute(1, 0, 2).reshape(1, 1, HEADS * R, HEAD_DIM),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=q_mem,
    )
    tt_ring = _tile(ring.reshape(1, 1, -1, HEAD_DIM), device)
    tt_sel = _tile(selected.reshape(1, 1, -1, HEAD_DIM), device)
    tt_staging = _tile(staging.reshape(1, 1, 32, HEAD_DIM), device)
    block_tile = ttnn.Tile((R, ttnn.TILE_SIZE))
    tt_cos = _tile(cos.reshape(1, 1, R, ROPE_DIM), device, tile=block_tile)
    tt_sin = _tile(sin.reshape(1, 1, R, ROPE_DIM), device, tile=block_tile)
    tt_sink = None
    if use_sink:
        sink_tile = torch.zeros(HEADS, ttnn.TILE_SIZE)
        sink_tile[:, 0] = sinks / SCALE
        tt_sink = _tile(sink_tile, device)
    sel_first = [s * INDEX_TOPK if dense else 0 for s in slots]
    lengths = _int32([[*counts, *positions, *[s * WINDOW for s in slots], *sel_first]], device)

    cache_entries = None
    for _ in range(2):  # the second call runs from the program cache
        out = ttnn.experimental.deepseek.flash_mqa_decode(
            tt_q,
            tt_ring,
            lengths,
            selected_kv=tt_sel,
            selected_row_offset=0,
            ring_rows=WINDOW,
            selected_rows=INDEX_TOPK,
            attention_sink=tt_sink,
            scale=SCALE,
            k_chunk_size=64,
            num_s_blocks=8,
            cos=tt_cos,
            sin=tt_sin,
            new_kv=tt_staging,
            num_rows=R,
            dense_selected=dense,
            independent_rows=True,
        )
        if cache_entries is None:
            cache_entries = device.num_program_cache_entries()
    assert device.num_program_cache_entries() == cache_entries
    actual = ttnn.to_torch(out).float().reshape(R, HEADS, HEAD_DIM)
    for j in range(R):
        passing, pcc = comp_pcc(expected[j], actual[j], 0.999)
        logger.info(f"{case} dense={dense} row {j} (p={positions[j]}, slot {slots[j]}, n={counts[j]}): PCC {pcc}")
        assert passing, f"row {j}: PCC {pcc}"


INDEX_HEAD_DIM = 128
BLOCK_ROWS = 64
LOGICAL_BLOCKS = 4  # a session's page-table row
N_BLOCKS = SLOTS * LOGICAL_BLOCKS
DENSE_TOPK = 64  # each slot's dense_kv rows in this test
EPS = 1e-6
WIN_SLOT_ROWS = 32


def _cpd_caches(device, ratio: int, seed: int = 5):
    """Host + device caches for ``SLOTS`` sessions: slotted windows (32 rows a slot), the paged ``idx_key_cache`` /
    ``comp_kv`` (each slot's logical blocks a permutation of physical ones) and a slotted ``dense_kv``."""
    gen = torch.Generator().manual_seed(seed)
    host = {
        "idx_key_cache": torch.randn(N_BLOCKS, 1, BLOCK_ROWS, INDEX_HEAD_DIM, generator=gen).bfloat16().float(),
        "comp_kv": torch.randn(N_BLOCKS, 1, BLOCK_ROWS, HEAD_DIM, generator=gen).bfloat16().float(),
        "dense_kv": torch.randn(1, 1, SLOTS * DENSE_TOPK, HEAD_DIM, generator=gen).bfloat16().float(),
    }
    if ratio == 2:
        host["win_kv"] = torch.randn(1, 1, SLOTS * WIN_SLOT_ROWS, HEAD_DIM, generator=gen).bfloat16().float()
        host["win_gate"] = torch.randn(1, 1, SLOTS * WIN_SLOT_ROWS, HEAD_DIM, generator=gen).bfloat16().float()
    layouts = {"comp_kv": ttnn.ROW_MAJOR_LAYOUT}
    dev = {name: _tile(x, device, layout=layouts.get(name, ttnn.TILE_LAYOUT)) for name, x in host.items()}
    host["page_table"] = torch.randperm(N_BLOCKS, generator=gen).to(torch.int32).reshape(SLOTS, LOGICAL_BLOCKS)
    dev["page_table"] = _int32(host["page_table"].tolist(), device)
    return host, dev


def _cpd_weights(device, seed: int = 9):
    gen = torch.Generator().manual_seed(seed)
    host = {
        "norm_weight": (1.0 + 0.1 * torch.randn(1, 1, 1, HEAD_DIM, generator=gen)).bfloat16().float(),
        "index_wk": (torch.randn(HEAD_DIM, INDEX_HEAD_DIM, generator=gen) / HEAD_DIM**0.5).bfloat16().float(),
        "index_norm_weight": (1.0 + 0.1 * torch.randn(1, 1, 1, INDEX_HEAD_DIM, generator=gen)).bfloat16().float(),
    }
    return {name: _tile(x, device) for name, x in host.items()}


# (positions, slots) of the four rows: at ratio 2 the odd positions close a group (entries 0, 1, 37 and 100), the even
# ones only write their window row; entries past DENSE_TOPK skip dense_kv.
_CPD_CASES = {
    "mixed": ([1, 2, 75, 201], [3, 0, 4, 1]),
    "all_close": ([1, 3, 75, 201], [2, 4, 0, 3]),
    "none_close": ([0, 2, 74, 200], [1, 2, 3, 4]),
}


@pytest.mark.parametrize("ratio", (2, 1))
@pytest.mark.parametrize("case", list(_CPD_CASES))
@pytest.mark.parametrize("interleave", (0, 128), ids=("separate", "kv_gate_interleave"))
def test_compress_pool_decode_independent_rows(device, reset_seeds, ratio, case, interleave):
    positions, slots = _CPD_CASES[case]
    if ratio == 1 and case == "none_close":
        pytest.skip("every row closes a group at ratio 1")
    if ratio == 1 and interleave:
        pytest.skip("a ratio-1 compressor has no gate to interleave")
    torch.manual_seed(21)
    weights = _cpd_weights(device)
    kv = torch.randn(R, HEAD_DIM).bfloat16().float()
    gate = torch.randn(R, HEAD_DIM).bfloat16().float()
    cos, sin = _row_angles(R, seed=7)
    closes = [(p + 1) % ratio == 0 for p in positions]
    entries = [p // ratio if c else -1 for p, c in zip(positions, closes)]
    dense_entries = [s * DENSE_TOPK + e if e >= 0 and e < DENSE_TOPK else -1 for s, e in zip(slots, entries)]
    win_slots = [s * WIN_SLOT_ROWS + p % 2 for p, s in zip(positions, slots)]
    prev_slots = [s * WIN_SLOT_ROWS + (p - 1) % 2 for p, s in zip(positions, slots)]

    def block_input(x: torch.Tensor) -> ttnn.Tensor:
        return _tile(x.reshape(1, 1, R, -1), device, tile=ttnn.Tile((R, ttnn.TILE_SIZE)))

    # The batch call.
    host, dev = _cpd_caches(device, ratio)
    if interleave:
        # [rank r: r's interleave kv columns | its gate columns], as the TP4 all-gather leaves the compressor rows.
        parts = []
        for r in range(HEAD_DIM // interleave):
            parts += [kv[:, r * interleave : (r + 1) * interleave], gate[:, r * interleave : (r + 1) * interleave]]
        kv_in = gate_in = block_input(torch.cat(parts, dim=-1))
    else:
        kv_in, gate_in = block_input(kv), block_input(gate)
    windows = {}
    if ratio == 2:
        windows = dict(
            gate=gate_in,
            win_kv=dev["win_kv"],
            win_gate=dev["win_gate"],
            win_slot=_int32(win_slots, device),
            prev_slot=_int32(prev_slots, device),
        )
    block_tile = ttnn.Tile((R, ttnn.TILE_SIZE))
    latent_tt, roped_tt = ttnn.experimental.deepseek.compress_pool_decode(
        kv_in,
        weights["norm_weight"],
        epsilon=EPS,
        do_pool=True,
        cos=_tile(cos.reshape(1, 1, R, ROPE_DIM), device, tile=block_tile),
        sin=_tile(sin.reshape(1, 1, R, ROPE_DIM), device, tile=block_tile),
        index_wk=weights["index_wk"],
        index_norm_weight=weights["index_norm_weight"],
        idx_key_cache=dev["idx_key_cache"],
        comp_kv=dev["comp_kv"],
        page_table=dev["page_table"],
        entry=_int32(entries, device),
        page_row=_int32(slots, device),
        dense_kv=dev["dense_kv"],
        dense_entry=_int32(dense_entries, device),
        kv_gate_interleave=interleave,
        independent_rows=True,
        **windows,
    )
    batch = {name: ttnn.to_torch(t).float() for name, t in dev.items() if name != "page_table"}
    roped = ttnn.to_torch(roped_tt).float().reshape(R, HEAD_DIM)

    # The reference: one one-row call per row on a second copy of the caches.
    ref_host, ref_dev = _cpd_caches(device, ratio)
    ref_roped = {}
    for j in range(R):
        row_windows = {}
        if ratio == 2:
            row_windows = dict(
                gate=_tile(gate[j].reshape(1, 1, 1, HEAD_DIM), device, layout=ttnn.ROW_MAJOR_LAYOUT),
                win_kv=ref_dev["win_kv"],
                win_gate=ref_dev["win_gate"],
                win_slot=_int32([win_slots[j]], device),
            )
        kv_row = _tile(kv[j].reshape(1, 1, 1, HEAD_DIM), device, layout=ttnn.ROW_MAJOR_LAYOUT)
        if not closes[j]:
            ttnn.experimental.deepseek.compress_pool_decode(
                kv_row, weights["norm_weight"], epsilon=EPS, do_pool=False, **row_windows
            )
            continue
        if ratio == 2:
            row_windows["prev_slot"] = _int32([prev_slots[j]], device)
        cos_j, sin_j = (_tile(x[j].reshape(1, 1, 1, ROPE_DIM), device) for x in (cos, sin))
        _, roped_j = ttnn.experimental.deepseek.compress_pool_decode(
            kv_row,
            weights["norm_weight"],
            epsilon=EPS,
            do_pool=True,
            cos=cos_j,
            sin=sin_j,
            index_wk=weights["index_wk"],
            index_norm_weight=weights["index_norm_weight"],
            idx_key_cache=ref_dev["idx_key_cache"],
            comp_kv=ref_dev["comp_kv"],
            page_table=ref_dev["page_table"],
            entry=_int32([entries[j]], device),
            page_row=_int32([slots[j]], device),
            dense_kv=ref_dev["dense_kv"],
            dense_entry=_int32([dense_entries[j]], device),
            **row_windows,
        )
        ref_roped[j] = ttnn.to_torch(roped_j).float().reshape(HEAD_DIM)
    ref = {name: ttnn.to_torch(t).float() for name, t in ref_dev.items() if name != "page_table"}

    for j, row in ref_roped.items():
        passing, pcc = comp_pcc(row, roped[j], 0.9999)
        logger.info(f"ratio {ratio} {case} row {j}: roped PCC {pcc}")
        assert passing, f"row {j}: roped row PCC {pcc}"
    for name in ref:
        # Rows nobody writes stay bit-exact; the written ones match the one-row steps'.
        untouched = ref_host[name] == ref[name]
        assert torch.equal(batch[name][untouched], ref[name][untouched]), f"{name}: a row the step does not own changed"
        passing, pcc = comp_pcc(ref[name], batch[name], 0.9999)
        diff = (ref[name] != batch[name]).sum().item()
        logger.info(f"ratio {ratio} {case} {name}: PCC {pcc}, {diff} values differ from the one-row steps")
        assert passing, f"{name}: PCC {pcc}"
