# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""MTP lanes: the lane forms on the torch-backed ttnn of steps 4 and 5 (no device).

The central invariant: after a lane pass, a per-lane commit and the next pass, every active lane's state equals its
own sequential 1-row stream and the B=1 verify path bitwise, and an inactive lane's region is untouched.
"""

from __future__ import annotations

import ast
import dataclasses
import itertools
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.tests.helpers.mtp_verify import accept_select
from models.demos.blackhole.qwen38_flash_next.tests.helpers.torch_ttnn import (
    BF16,
    DRAFT_WIDTH,
    FP32,
    HOST_TENSOR_CALLS,
    I32,
    LARGE_IDS,
    ROW_MAJOR,
    SMALL_IDS,
    TILE,
    TP,
    U32,
    VERIFY_BLOCKS,
    VERIFY_CONTEXT,
    DraftWorld,
    FakeChunk,
    FakeContract,
    FakeTensor,
    KVStream,
    RowsMixer,
    accept_pattern_rows,
    bf16_randn,
    cat_shards,
    draft_body_model,
    draft_history,
    eager_draft_rows,
    fp32_lanes,
    gdn_rows_selectors,
    hidden_sharded,
    make_gdn_module,
    make_mtp_lanes_ttnn,
    make_ple_module,
    make_qsa_module,
    ordered_calls,
    partial_rope_reference,
    patch_ttnn,
    qsa_verify_constants,
    replicated,
    residual_rows,
    run_qsa_verify_pass,
    same_zero,
    seed_gdn_state,
    seed_qsa_caches,
    sharded,
    source_functions,
    source_methods,
    source_segment,
    use_aliasing_view,
)
from models.demos.blackhole.qwen38_flash_next.ttnn import embedding as embedding_module
from models.demos.blackhole.qwen38_flash_next.ttnn import final_mixer as final_mixer_module
from models.demos.blackhole.qwen38_flash_next.ttnn import gdn as gdn_module
from models.demos.blackhole.qwen38_flash_next.ttnn import lanes as lanes_module
from models.demos.blackhole.qwen38_flash_next.ttnn import layer as layer_module
from models.demos.blackhole.qwen38_flash_next.ttnn import mtp as mtp_module
from models.demos.blackhole.qwen38_flash_next.ttnn import mtp_lanes, mtp_v2
from models.demos.blackhole.qwen38_flash_next.ttnn import ple as ple_module
from models.demos.blackhole.qwen38_flash_next.ttnn import qsa as qsa_module
from models.demos.blackhole.qwen38_flash_next.ttnn.layer import Qwen38TTNNLayerNamespace

ROOT = Path(__file__).resolve().parents[1]
GDN_SOURCE = ROOT / "ttnn" / "gdn.py"
PLE_SOURCE = ROOT / "ttnn" / "ple.py"
QSA_SOURCE = ROOT / "ttnn" / "qsa.py"
MTP_LANES_SOURCE = ROOT / "ttnn" / "mtp_lanes.py"
LANE_FORMS = ((4, 4), (4, 3), (2, 4), (1, 4), (6, 4), (8, 3))  # (lanes, drafts); 8 x 4 fills the tile
LANE_POSITIONS = ((5, 31, 32, 100), (29, 30, 63, 60), (0, 3, 61, 127), (96, 33, 5, 29))  # every P % 4, both block cases
KV_SCRATCH_ROWS = mtp_lanes.kv_scratch_rows(VERIFY_CONTEXT)


# --------------------------------------------------------------------------- the lane fake


@pytest.fixture
def fake(monkeypatch):
    chunk = FakeChunk()
    fake_ttnn = make_mtp_lanes_ttnn(chunk)
    patch_ttnn(
        monkeypatch,
        fake_ttnn,
        qsa_module,
        gdn_module,
        ple_module,
        layer_module,
        embedding_module,
        final_mixer_module,
        mtp_module,
        mtp_v2,
        mtp_lanes,
        lanes_module,  # the owned slice the import's pack slices go through
    )
    monkeypatch.setattr(qsa_module, "apply_partial_rope_prefill", partial_rope_reference)
    return SimpleNamespace(ttnn=fake_ttnn, chunk=chunk)


def _u32_row(values) -> FakeTensor:
    return replicated(torch.tensor([int(v) for v in values], dtype=torch.int64).reshape(1, 1, 1, 32), U32, ROW_MAJOR)


def _lane_column(values, dtype=FP32) -> FakeTensor:
    return replicated(torch.tensor([float(v) for v in values]).reshape(1, len(values), 1, 1), dtype)


def _pad32(values, fill) -> list:
    return list(values) + [fill] * (32 - len(values))


def _lane_selectors(accepted, active, constants):
    return gdn_module.build_rows_selectors_lanes(_lane_column(accepted), _lane_column(active), constants)


# --------------------------------------------------------------------------- constants (host images)


@pytest.mark.parametrize("lanes,drafts", LANE_FORMS)
def test_lane_accept_constants_are_block_diagonal_and_zero_on_the_pad_columns(
    expect_error, lanes: int, drafts: int
) -> None:
    rows = drafts + 1
    real = lanes * rows
    host = mtp_lanes.mtp_lane_constant_rows(lanes, drafts)
    lane_of, draft_of = mtp_lanes.lane_of_rows(lanes, rows)
    assert lane_of[:real] == tuple(r // rows for r in range(real)) and set(lane_of[real:]) <= {0}
    assert draft_of[:real] == tuple(r % rows for r in range(real)) and set(draft_of[real:]) <= {rows - 1}
    prefix = host["prefix_upper_block"]
    for i in range(32):
        for c in range(32):
            assert prefix[i, c] == float(i < real and c < real and lane_of[i] == lane_of[c] and i <= c), (i, c)
    block_sum = host["block_sum"]
    assert block_sum.shape == (32, lanes) and torch.count_nonzero(block_sum[real:]) == 0
    for c in range(real):
        assert block_sum[c].tolist() == [float(u == lane_of[c]) for u in range(lanes)], c
    assert torch.equal(host["block_sum_t"], block_sum.t())
    plus_one = host["draft_plus_one_row"].reshape(-1).tolist()
    assert plus_one[:real] == [float(draft_of[c] + 1) for c in range(real)] and set(plus_one[real:]) <= {64.0}
    match = host["draft_match_row"].reshape(-1).tolist()
    assert match[:real] == [float(draft_of[c]) for c in range(real)] and set(match[real:]) <= {-1.0}
    assert host["lane_base_row"].reshape(-1).tolist() == [u * rows for u in range(lanes)]
    # The accept algebra in torch on every per-lane pattern, including pad flags of 1 (sentinel == sentinel).
    for patterns in itertools.islice(itertools.product(itertools.product((0, 1), repeat=drafts), repeat=lanes), 40):
        flags = torch.zeros(1, 32)
        flags[0, real:] = 1.0
        for u, pattern in enumerate(patterns):
            flags[0, u * rows : u * rows + drafts] = torch.tensor(pattern, dtype=torch.float32)
        running = flags @ prefix
        counts = (running == host["draft_plus_one_row"]).float() @ block_sum
        expected = [float(next((j for j, m in enumerate(pattern) if not m), drafts)) for pattern in patterns]
        assert counts.reshape(-1).tolist() == expected, patterns
    for bad in ((lanes, 6), (lanes, 2)):
        with expect_error(ValueError):
            mtp_lanes.mtp_lane_constant_rows(*bad)
    with expect_error(ValueError):
        mtp_lanes.mtp_lane_constant_rows(7, 4)  # 35 rows


@pytest.mark.parametrize("lanes,rows", ((4, 5), (2, 4), (6, 5), (8, 4)))
def test_gdn_lane_selects_expand_fold_and_pick_the_history_rows(expect_error, lanes: int, rows: int) -> None:
    selects = gdn_module.lane_rows_select_tiles(lanes, rows)
    expand, fold, stack = selects["expand_select"], selects["fold_select"], selects["history_select_stack"]
    torch.manual_seed(3)
    tile = torch.randn(32, 2560)
    tile[lanes * rows :] = 0.0
    expanded = expand @ tile
    for u in range(lanes):
        assert torch.equal(expanded[u * 32 : u * 32 + rows], tile[u * rows : u * rows + rows]), u
        assert torch.count_nonzero(expanded[u * 32 + rows : (u + 1) * 32]) == 0, u
    assert torch.equal(fold @ expanded, tile) and torch.equal(fold, expand.t())
    for committed in range(rows + 1):
        select = stack[committed].reshape(32, 64)
        assert torch.count_nonzero(select[3:]) == 0
        for index in range(3):
            logical = committed + index
            assert select[index].nonzero().reshape(-1).tolist() == [logical if logical < 3 else 32 + logical - 3]
    assert torch.count_nonzero(stack[rows + 1 :]) == 0
    with expect_error(ValueError):
        gdn_module.lane_rows_select_tiles(lanes, 33 // lanes + 1)


@pytest.mark.parametrize("lanes,rows", ((4, 5), (2, 4), (6, 5), (1, 5)))
def test_qsa_lane_verify_constants_address_each_lanes_region_and_never_a_pad_row(
    expect_error, lanes: int, rows: int
) -> None:
    host = qsa_module.qsa_lane_verify_constant_rows(lanes, rows, VERIFY_CONTEXT)
    real = lanes * rows
    lane_of = host["lane_of_row"].reshape(-1).tolist()
    draft_of = host["draft_of_row"].reshape(-1).tolist()
    assert lane_of[:real] == [r // rows for r in range(real)] and set(lane_of[real:]) <= {0}
    assert draft_of[:real] == [r % rows for r in range(real)] and set(draft_of[real:]) <= {rows - 1}
    block_lane = host["block_lane_of_row"].reshape(-1).tolist()
    block_offset = host["block_offset_row"].reshape(-1).tolist()
    assert block_lane[: 2 * lanes] == [r // 2 for r in range(2 * lanes)] and set(block_lane[2 * lanes :]) <= {0}
    assert block_offset[: 2 * lanes] == [4 * (r % 2) for r in range(2 * lanes)] and set(block_offset[2 * lanes :]) <= {
        0
    }
    assert set(host["scratch_row"].reshape(-1).tolist()) == {lanes * VERIFY_CONTEXT}
    assert host["arange_per_lane"].reshape(-1).tolist() == list(range(32)) * lanes
    indicator = host["lane_indicator_rows"]
    for u in range(lanes):
        assert indicator[u, 0].tolist() == [float(c < real and lane_of[c] == u) for c in range(32)], u
        assert torch.equal(indicator[u], indicator[u, :1].expand(32, -1))
    expand = host["expand_select"]
    for u in range(lanes):
        for j in range(32):
            expected = torch.zeros(32)
            if j < rows:
                expected[u * rows + j] = 1.0
            assert torch.equal(expand[u * 32 + j], expected), (u, j)
    pick_pooled = host["pick_pooled"]
    for r in range(32):
        nonzero = pick_pooled[r].nonzero().reshape(-1).tolist()
        assert nonzero == ([(r // 2) * 32 + r % 2] if r < 2 * lanes else []), r
    for block in range(2):
        for u in range(lanes):
            assert host["pick_users"][block][u].nonzero().tolist() == [[0, 2 * u + block]], (block, u)
    offsets = qsa_module.qsa_row_constants()["block_offsets"].reshape(-1)
    for r in range(32):
        assert torch.equal(
            host["block_offsets_lanes"][0, 0, r] - offsets, torch.full_like(offsets, VERIFY_CONTEXT * lane_of[r])
        )
        assert torch.equal(
            host["arange_slots_rows"][0, 0, r],
            torch.arange(qsa_module.SPARSE_INDEX_CAPACITY) + VERIFY_CONTEXT * lane_of[r],
        )
    with expect_error(ValueError):
        qsa_module.qsa_lane_verify_constant_rows(lanes, 33 // lanes + 1, VERIFY_CONTEXT)


# --------------------------------------------------------------------------- QSA lane verify inputs


def _lane_qsa_constants(rows: int, lanes: int):
    contract = FakeContract()
    lane_constants = qsa_module.Qwen38TTNNQSALaneConstants.build(
        "mesh", contract, lanes=lanes, allocated_context=VERIFY_CONTEXT
    )
    lane_verify = qsa_module.Qwen38TTNNQSALaneVerifyConstants.build(
        "mesh", contract, lanes=lanes, rows=rows, allocated_context=VERIFY_CONTEXT
    )
    return qsa_verify_constants(rows), lane_constants, lane_verify


def _derive_lane_inputs(fake, positions, active, lane_qsa, *, single_row=False):
    (position_constants, chunk_constants, verify_constants), lane_constants, lane_verify = lane_qsa
    position_row = _u32_row(_pad32(positions, positions[0]))
    active_row = _u32_row(_pad32(active, 1))
    inactive_row = _u32_row(_pad32([1 - flag for flag in active], 0))
    lane_rows = fake.ttnn.gather(position_row, 3, lane_verify.lane_of_row)
    row_positions = fake.ttnn.add(lane_rows, lane_verify.draft_of_row)
    inputs = qsa_module.derive_qsa_lane_verify_inputs(
        position_row,
        active_row,
        inactive_row,
        row_positions,
        position_constants,
        chunk_constants,
        verify_constants,
        lane_constants,
        lane_verify,
        single_row=single_row,
    )
    return inputs, row_positions


@pytest.mark.parametrize("rows", (1, 4, 5))
@pytest.mark.parametrize("active", ((1, 1, 1, 1), (1, 0, 1, 1), (0, 1, 1, 0)))
@pytest.mark.parametrize("positions", LANE_POSITIONS)
def test_lane_verify_inputs_match_the_torch_reference_for_mixed_positions_and_active_lanes(
    expect_error, fake, positions, active, rows: int
) -> None:
    """R = 1 is the draft rows' ``single_row`` form: one real row per lane, the current KV block and compressed block
    alone (a lane's row never reaches the next block, which at the lane's last block is the next lane's first)."""

    lanes, single_row = len(positions), rows == 1
    derived, row_positions = _derive_lane_inputs(
        fake, positions, active, _lane_qsa_constants(rows, lanes), single_row=single_row
    )
    expected = qsa_module.emulate_qsa_lane_verify_inputs(
        positions, active, lanes=lanes, rows=rows, allocated_compressed_blocks=VERIFY_BLOCKS, single_row=single_row
    )
    assert derived.single_row == single_row
    assert torch.equal(row_positions.torch_shards()[0], expected["row_positions"])
    assert same_zero(derived.indexer_neg_mask.torch_shards()[0], expected["indexer_neg_mask"])
    for name in ("row_keep_bits", "row_fill", "kv_read_indices"):
        assert torch.equal(getattr(derived, name).torch_shards()[0], expected[name]), name
    assert len(derived.block_index_i32) == len(expected["block_index_i32"]) == (1 if single_row else 2)
    for index, wanted in zip(derived.block_index_i32, expected["block_index_i32"]):
        assert index.dtype is I32 and index.torch_shards()[0].tolist() == wanted.tolist()
    for name in ("kv_row_start_lanes", "kv_row_start_next_lanes"):  # no next block rows in the single_row form
        starts = [int(t.torch_shards()[0].reshape(-1)[0]) for t in getattr(derived, name)]
        assert starts == expected[name].tolist(), name
    for name in ("stage_keep", "stage_a_select", "stage_b_select", "pool_select"):
        actual, wanted = getattr(derived, name), expected[name]
        assert (actual is None) == (wanted is None) == (single_row and name == "stage_b_select"), name
        if wanted is not None:
            assert torch.equal(actual.torch_shards()[0].float(), wanted.float()), name
    # Every real column lands exactly once, on its own lane's batch and at row P_row % 32; pad columns never land;
    # an inactive lane's block rows are the scratch rows past the last lane.
    both = derived.stage_a_select.torch_shards()[0].float()[0]
    if not single_row:
        both = both + derived.stage_b_select.torch_shards()[0].float()[0]
    for c in range(32):
        assert both[:, :, c].sum(dim=1).tolist() == [
            float(c < lanes * rows and u == c // rows) for u in range(lanes)
        ], c
        if c < lanes * rows:
            u = c // rows
            (landing,) = both[u, :, c].nonzero().reshape(-1).tolist()
            assert landing == (positions[u] + c % rows) % 32
    for u in range(lanes):
        start = int(derived.kv_row_start_lanes[u].torch_shards()[0].reshape(-1)[0])
        assert start == (u * VERIFY_CONTEXT + (positions[u] & ~31) if active[u] else lanes * VERIFY_CONTEXT), u
        keep = derived.stage_keep.torch_shards()[0][0, u].reshape(-1).float().tolist()
        assert keep == [float(i < positions[u] % 32) for i in range(32)], u
        # The pool select reads lane u's window: row 0 the four positions of block P_u // 4, row 1 the next block.
        pool = derived.pool_select.torch_shards()[0][0, u].float()
        assert torch.count_nonzero(pool[2:]) == 0 and set(pool.unique().tolist()) <= {0.0, 0.25}
        for block_row, block in ((0, positions[u] // 4), (1, positions[u] // 4 + 1)):
            columns = pool[block_row].nonzero().reshape(-1).tolist()
            named = [positions[u] - 3 + w if w < 32 else positions[u] + w - 32 for w in columns]
            assert named == [p for p in range(4 * block, 4 * block + 4) if p < positions[u] + rows], (u, block_row)
    derived.deallocate()
    with expect_error(ValueError):
        qsa_module.emulate_qsa_lane_verify_inputs(
            positions[:2], active, lanes=lanes, rows=rows, allocated_compressed_blocks=VERIFY_BLOCKS
        )


# --------------------------------------------------------------------------- the per-lane rollback invariant


def _run_lane_pass(module, fake, streams, state, verify_state, positions, active, rows: int, lane_qsa):
    """The two lane verify cache writers on every lane's rows P_u .. P_u + R - 1. An inactive lane's KV writes land in
    the scratch rows and its pooled rows in blocks its next real pass rewrites before any row names them."""

    _, lane_constants, lane_verify = lane_qsa
    inputs, row_positions = _derive_lane_inputs(fake, positions, active, lane_qsa)
    raw = torch.zeros(1, 1, 32, 128, dtype=torch.bfloat16)
    keys = [torch.zeros(1, 1, 32, 256, dtype=torch.bfloat16) for _ in range(TP)]
    values = [torch.zeros(1, 1, 32, 256, dtype=torch.bfloat16) for _ in range(TP)]
    cos = torch.zeros(1, 1, 32, 64, dtype=torch.bfloat16)
    sin = torch.zeros(1, 1, 32, 64, dtype=torch.bfloat16)
    for lane, (stream, position) in enumerate(zip(streams, positions)):
        if active[lane]:
            raw[0, 0, lane * rows : (lane + 1) * rows] = stream.raw[position : position + rows]
            for device in range(TP):
                packed = stream.packed[device][position : position + rows]
                values[device][0, 0, lane * rows : (lane + 1) * rows] = packed[:, :256]
                keys[device][0, 0, lane * rows : (lane + 1) * rows] = packed[:, 256:]
        for block in range(2):
            start = min((position & ~3) + 4 * block, VERIFY_CONTEXT - 1)
            cos[0, 0, 2 * lane + block], sin[0, 0, 2 * lane + block] = stream.cos[start], stream.sin[start]
    module._write_compressed_index_lanes_verify(
        state,
        verify_state,
        replicated(raw, BF16),
        replicated(cos, BF16),
        replicated(sin, BF16),
        inputs,
        lane_constants,
        lane_verify,
    )
    module._write_packed_kv_lanes_verify(
        state, FakeTensor(keys, BF16, TILE, 1), FakeTensor(values, BF16, TILE, 1), inputs
    )
    inputs.deallocate()
    fake.ttnn.deallocate(row_positions)


@pytest.mark.parametrize("rows", (4, 5))
@pytest.mark.parametrize(
    "positions,accepted,active",
    (
        ((5, 31, 32, 100), (0, 3, 1, 2), (1, 1, 1, 1)),
        ((29, 30, 63, 60), (2, 0, 3, 1), (1, 1, 0, 1)),
        ((0, 61, 96, 33), (3, 1, 0, 2), (0, 1, 1, 1)),
        ((127, 5, 30, 31), (1, 2, 3, 0), (1, 0, 1, 0)),
    ),
)
def test_lane_kv_and_compressed_writes_commit_each_lanes_own_prefix(fake, positions, accepted, active, rows) -> None:
    """Pass, per-lane commit, pass: every active lane's KV rows, compressed blocks and raw history equal its own
    sequential stream and the B=1 verify path bitwise; an inactive lane's region is untouched."""

    accepted = tuple(min(a, rows - 1) for a in accepted)
    lanes = len(positions)
    lane_qsa = _lane_qsa_constants(rows, lanes)
    constants = lane_qsa[0]
    gdn_lane_constants = gdn_module.Qwen38TTNNGDNLaneRowsConstants.allocate(
        "mesh", FakeContract(), lanes=lanes, rows=rows
    )
    gdn_constants = gdn_module.Qwen38TTNNGDNRowsConstants.allocate("mesh", FakeContract(), rows=rows)
    module = make_qsa_module()
    streams = [KVStream(seed=200 + lane) for lane in range(lanes)]
    state = module.allocate_lane_state(lanes, kv_scratch_rows=KV_SCRATCH_ROWS)
    verify_state = module.allocate_lane_verify_state(lanes)
    assert state.packed_kv_cache.shape == (1, 1, lanes * VERIFY_CONTEXT + KV_SCRATCH_ROWS, 512)
    for lane, (stream, position) in enumerate(zip(streams, positions)):
        seed_qsa_caches(module, fake.ttnn, stream, state, verify_state, position, lane=lane)
    for target in state.packed_kv_cache.locals:
        target[0, 0, lanes * VERIFY_CONTEXT :] = 5.0
    _run_lane_pass(module, fake, streams, state, verify_state, positions, (1,) * lanes, rows, lane_qsa)
    # The B=1 verify path per active lane on the same stream (pass N before the scramble).
    references = {}
    for lane in range(lanes):
        if active[lane]:
            single = make_qsa_module()
            generic, single_verify = single.allocate_generic_state(), single.allocate_verify_state()
            seed_qsa_caches(single, fake.ttnn, streams[lane], generic, single_verify, positions[lane])
            run_qsa_verify_pass(single, streams[lane], generic, single_verify, positions[lane], rows, constants)
            references[lane] = (single, generic, single_verify)
    after_pass = [x.clone() for x in state.packed_kv_cache.locals]
    after_blocks = [x.clone() for x in state.compressed_index_cache.locals]
    next_positions = tuple(p + (a + 1) * flag for p, a, flag in zip(positions, accepted, active))
    for lane in range(lanes):
        if active[lane]:  # rows past the committed prefix are garbage from here on: scramble the stream there
            torch.manual_seed(500 + lane)
            streams[lane].raw[next_positions[lane] :] = bf16_randn(VERIFY_CONTEXT - next_positions[lane], 128)
            for device in range(TP):
                streams[lane].packed[device][next_positions[lane] :] = bf16_randn(
                    VERIFY_CONTEXT - next_positions[lane], 512
                )
    selectors = _lane_selectors(accepted, active, gdn_lane_constants)
    module.commit_verify_lanes(verify_state, selectors)
    history = verify_state.raw_history.torch_shards()[0]
    for lane in range(lanes):
        for back in range(1, 4):
            position = next_positions[lane] - back
            expected = streams[lane].raw[position] if position >= 0 else torch.zeros(128, dtype=torch.bfloat16)
            assert torch.equal(history[0, lane, 3 - back], expected), (lane, back)
        assert torch.count_nonzero(history[0, lane, 3:].float()) == 0
    _run_lane_pass(module, fake, streams, state, verify_state, next_positions, active, rows, lane_qsa)
    for lane in range(lanes):
        base = lane * VERIFY_CONTEXT
        if not active[lane]:
            for device in range(TP):
                region = state.packed_kv_cache.torch_shards()[device][0, 0, base : base + VERIFY_CONTEXT]
                assert torch.equal(region, after_pass[device][0, 0, base : base + VERIFY_CONTEXT]), (lane, device)
            blocks = state.compressed_index_cache.torch_shards()[0][lane, 0]
            complete = positions[lane] // 4  # its complete blocks stay; the two it rewrote hold finite junk
            assert torch.equal(blocks[:complete], after_blocks[0][lane, 0, :complete]), lane
            assert torch.equal(blocks[complete + 2 :], after_blocks[0][lane, 0, complete + 2 :]), lane
            assert torch.isfinite(blocks.float()).all()
            continue
        single, generic, single_verify = references[lane]
        single.commit_verify(single_verify, gdn_rows_selectors(accepted[lane], gdn_constants))
        run_qsa_verify_pass(single, streams[lane], generic, single_verify, next_positions[lane], rows, constants)
        visible = next_positions[lane] + rows
        for device in range(TP):
            region = state.packed_kv_cache.torch_shards()[device][0, 0, base : base + VERIFY_CONTEXT]
            assert torch.equal(region[:visible], streams[lane].packed[device][:visible]), (lane, device)
            assert torch.equal(region[:visible], generic.packed_kv_cache.torch_shards()[device][0, 0, :visible]), (
                lane,
                device,
            )
            # Rows past the two blocks the pass wrote are untouched.
            untouched = (next_positions[lane] & ~31) + 64
            assert torch.equal(
                region[untouched:], after_pass[device][0, 0, base + untouched : base + VERIFY_CONTEXT]
            ), lane
        blocks = state.compressed_index_cache.torch_shards()[0][lane, 0]
        for block in range(visible // 4):
            assert torch.equal(blocks[block], streams[lane].compressed_block(module, fake.ttnn, block)), (lane, block)
        assert torch.equal(
            blocks[: visible // 4], generic.compressed_index_cache.torch_shards()[0][0, 0, : visible // 4]
        )
        assert torch.equal(history[0, lane], single_verify.raw_history.torch_shards()[0][0, 0])
        assert torch.isfinite(blocks.float()).all()
    scratch = state.packed_kv_cache.torch_shards()[0][0, 0, lanes * VERIFY_CONTEXT :]
    assert scratch.shape[0] == KV_SCRATCH_ROWS and torch.isfinite(scratch.float()).all()
    if not all(active):
        assert not torch.equal(scratch, torch.full_like(scratch, 5.0))  # the redirected writes landed here
    selectors.deallocate()
    module.release_lane_verify_state(verify_state)
    module.release_lane_state(state)


# --------------------------------------------------------------------------- GDN lane rows


@pytest.mark.parametrize(
    "lanes,rows,accepted,active",
    (
        (4, 5, (0, 2, 4, 1), (1, 1, 0, 1)),
        (4, 5, (-1, 3, 4, 4), (1, 1, 1, 1)),  # lane 0 seeded (commits nothing), lanes 2-3 the stage-1 form
        (2, 4, (3, 0), (1, 1)),
    ),
)
def test_gdn_lane_rows_are_the_rows_path_per_lane_and_commit_lands_each_lanes_prefix(
    fake, gdn_weights, lanes: int, rows: int, accepted, active
) -> None:
    _, weights = gdn_weights
    module = make_gdn_module(weights)
    torch.manual_seed(31)
    hidden = [torch.randn(1, rows, 2560).to(torch.bfloat16) for _ in range(lanes)]
    constants = module.allocate_rows_constants(rows)
    lane_constants = module.allocate_lane_rows_constants(lanes, rows)
    states = [seed_gdn_state(module, 40 + lane) for lane in range(lanes)]
    lane_state = module.allocate_lane_state(lanes)
    lane_rows = module.allocate_lane_rows_state(constants, lane_constants)
    references = []
    for lane in range(lanes):
        rows_state = module.allocate_rows_state(constants)
        module.sync_rows_history_from_state(states[lane], rows_state)
        for device in range(TP):
            lane_state.recurrent.locals[device][lane] = states[lane].recurrent.locals[device][0]
            lane_rows.history.locals[device][0, lane] = rows_state.history.locals[device][0, 0]
        result = module.forward_rows(hidden_sharded(hidden[lane]), states[lane], rows_state, full_tile=True)
        references.append((rows_state, result))
    tile = torch.zeros(1, 32, 2560, dtype=torch.bfloat16)
    for lane in range(lanes):
        tile[:, lane * rows : (lane + 1) * rows] = hidden[lane]
    calls = len(fake.chunk.calls)
    result = module.forward_rows_lanes(hidden_sharded(tile), lane_state, lane_rows)
    assert len(fake.chunk.calls) - calls == lanes  # the rows path's recurrence, once per lane
    output = cat_shards(result.hidden_rows, 3)
    assert output.shape == (1, 1, 32, 2560) and torch.count_nonzero(output[:, :, lanes * rows :].float()) == 0
    final = cat_shards(result.final_state, 1)
    for lane in range(lanes):
        rows_state, reference = references[lane]
        assert torch.equal(
            output[:, :, lane * rows : (lane + 1) * rows], cat_shards(reference.hidden_rows, 3)[:, :, :rows]
        ), lane
        assert torch.equal(final[lane : lane + 1], cat_shards(reference.final_state, 1)), lane
        for name in ("q", "k"):
            assert torch.equal(
                cat_shards(getattr(lane_rows, name), 2)[lane : lane + 1], cat_shards(getattr(rows_state, name), 2)
            ), name
        for name in ("v", "beta", "g"):
            assert torch.equal(
                cat_shards(getattr(lane_rows, name), 3)[:, lane : lane + 1], cat_shards(getattr(rows_state, name), 3)
            ), name
    # The commit: c_u = (a_u + 1) * active_u rows per lane; the state select is [c_u >= 1].
    selectors = _lane_selectors(accepted, active, lane_constants)
    committed = [(a + 1) * flag for a, flag in zip(accepted, active)]
    for lane in range(lanes):
        onehots = [float(t.torch_shards()[0].reshape(-1)[lane]) for t in selectors.onehot_bf16]
        assert onehots == [float(j == committed[lane]) for j in range(rows + 1)], lane
        mask = selectors.committed_mask.torch_shards()[0][0, lane].reshape(-1).tolist()
        assert mask == [float(j < committed[lane]) for j in range(32)], lane
        assert float(selectors.commit_col.torch_shards()[0].reshape(-1)[lane]) == float(committed[lane] >= 1)
        assert float(selectors.keep_col.torch_shards()[0].reshape(-1)[lane]) == float(committed[lane] == 0)
    assert selectors.commit_col.dtype is FP32 and selectors.commit_col.shape == (lanes, 1, 1, 1)
    before_recurrent = [x.clone() for x in lane_state.recurrent.locals]
    before_history = [x.clone() for x in lane_rows.history.locals]
    calls = len(fake.chunk.calls)
    module.commit_rows_lanes(lane_state, lane_rows, selectors)
    assert len(fake.chunk.calls) - calls == lanes  # the masked rerun, once per lane
    for lane in range(lanes):
        rows_state, _ = references[lane]
        if committed[lane] == 0:
            for device in range(TP):
                assert torch.equal(lane_state.recurrent.locals[device][lane], before_recurrent[device][lane]), lane
                assert torch.equal(lane_rows.history.locals[device][0, lane], before_history[device][0, lane]), lane
            continue
        module.commit_rows(states[lane], rows_state, gdn_rows_selectors(accepted[lane], constants))
        for device in range(TP):
            assert same_zero(lane_state.recurrent.locals[device][lane], states[lane].recurrent.locals[device][0]), lane
            assert torch.equal(lane_rows.history.locals[device][0, lane], rows_state.history.locals[device][0, 0]), lane
    selectors.deallocate()
    # The selectors' columns are fresh tensors: releasing them leaves the persistent mask and counts alive.
    assert selectors.commit_col.alive is False and selectors.keep_col.alive is False


# --------------------------------------------------------------------------- PLE lane rows


def _seeded_ple_state(module, seed: int, context):
    state = module.allocate_state()
    torch.manual_seed(seed)
    for slot in state.conv:
        for local in slot.locals:
            local.copy_(torch.randn(1, 1, 4, 640).to(torch.bfloat16))
    state.token_context = None if context is None else torch.tensor([list(context)], dtype=torch.long)
    return state


@pytest.mark.parametrize(
    "lanes,rows,accepted,active",
    ((4, 5, (0, 2, 4, 1), (1, 1, 0, 1)), (3, 4, (3, -1, 0), (1, 1, 1)), (6, 5, (4, 4, 4, 4, 4, 4), (1, 1, 1, 1, 1, 1))),
)
def test_ple_lane_rows_are_the_rows_path_per_lane_and_commit_selects_each_lanes_history(
    fake, lanes: int, rows: int, accepted, active
) -> None:
    module = make_ple_module()
    torch.manual_seed(24)
    residuals = [torch.randn(rows, 4, 2560).to(torch.bfloat16) for _ in range(lanes)]
    tokens = [tuple(int(t) for t in torch.randint(0, 2000, (rows,))) for _ in range(lanes)]
    tokens[0] = (17, 15, 95859) + tokens[0][3:]
    lanes_state = module.allocate_lane_rows_state(lanes, rows)
    references, contexts = [], []
    for lane in range(lanes):
        rows_state = module.allocate_rows_state(rows)
        rows_state.load_from_state(_seeded_ple_state(module, 60 + lane, None if lane % 2 == 0 else (11 + lane, 7)))
        prepared = module.prepare_rows_input(tokens[lane], rows_state)
        result = module.forward_prepared_rows(residual_rows(residuals[lane]), prepared, rows_state)
        references.append((rows_state, prepared, cat_shards(result.residual_delta, 3)))
        contexts.append(rows_state.token_context)
        for device in range(TP):
            lanes_state.history.locals[device][lane] = rows_state.history.locals[device][0]
    lanes_state.token_contexts = tuple(contexts)
    lanes_state.validate()
    host_rows = torch.cat([module.host_rows(tokens[lane], contexts[lane])[0] for lane in range(lanes)], dim=2)
    for lane in range(lanes):
        assert torch.equal(
            host_rows[:, :, lane * rows : (lane + 1) * rows], cat_shards(references[lane][1].embedding_rows, 3)
        )
    prepared_lanes = ple_module.Qwen38TTNNPLERowsPreparedInput(
        sharded(host_rows, layout=ROW_MAJOR), tuple(token for lane_tokens in tokens for token in lane_tokens), ()
    )
    delta = cat_shards(
        module.forward_prepared_rows_lanes(residual_rows(torch.cat(residuals, dim=0)), prepared_lanes, lanes_state), 3
    )
    assert delta.shape == (1, lanes * rows, 4, 2560)
    for lane in range(lanes):
        assert torch.equal(delta[:, lane * rows : (lane + 1) * rows], references[lane][2]), lane
        for device in range(TP):
            assert torch.equal(
                lanes_state.normalized.locals[device][lane], references[lane][0].normalized.locals[device][0]
            )
    # The branch-major injection consumes the lane-major rows and equals the rows form per lane.
    branch_major = torch.cat(residuals, dim=0).permute(1, 0, 2).reshape(1, 4, lanes * rows, 2560)
    injected = cat_shards(module.inject_rows_lanes(sharded(branch_major), prepared_lanes, lanes_state), 3)
    for lane in range(lanes):
        single = residuals[lane].permute(1, 0, 2).reshape(1, 4, rows, 2560)
        reference = module.inject_rows(sharded(single), references[lane][1], references[lane][0])
        assert torch.equal(injected[:, :, lane * rows : (lane + 1) * rows], cat_shards(reference, 3)), lane
    gdn_constants = gdn_module.Qwen38TTNNGDNRowsConstants.allocate("mesh", FakeContract(), rows=rows)
    lane_constants = gdn_module.Qwen38TTNNGDNLaneRowsConstants.allocate("mesh", FakeContract(), lanes=lanes, rows=rows)
    selectors = _lane_selectors(accepted, active, lane_constants)
    before = [x.clone() for x in lanes_state.history.locals]
    module.commit_rows_lanes(lanes_state, selectors)
    for lane in range(lanes):
        committed = (accepted[lane] + 1) * active[lane]
        if committed == 0:
            for device in range(TP):
                assert torch.equal(lanes_state.history.locals[device][lane], before[device][lane]), lane
            continue
        module.commit_rows(references[lane][0], gdn_rows_selectors(accepted[lane], gdn_constants))
        for device in range(TP):
            assert same_zero(
                lanes_state.history.locals[device][lane], references[lane][0].history.locals[device][0]
            ), lane


# --------------------------------------------------------------------------- the lane accept, select and advance


def _lane_verify_stub(lanes: int, drafts: int, accepted, active, positions):
    rows = drafts + 1
    contract = FakeContract()
    constants = mtp_lanes.Qwen38TTNNMTPLaneConstants.build("mesh", contract, lanes=lanes, drafts=drafts)
    lane_verify = qsa_module.Qwen38TTNNQSALaneVerifyConstants.build(
        "mesh", contract, lanes=lanes, rows=rows, allocated_context=VERIFY_CONTEXT
    )
    return SimpleNamespace(
        lanes=lanes,
        drafts=drafts,
        rows=rows,
        constants=constants,
        qsa_lane_verify_constants=lane_verify,
        positions=mtp_lanes.Qwen38TTNNMTPLanePositions.allocate("mesh", contract, positions, lanes=lanes),
        accepted_row=replicated(mtp_lanes._accepted_row_host(accepted, lanes), FP32, ROW_MAJOR),
        active_row=replicated(mtp_lanes._mask_row_host(active, lanes), U32, ROW_MAJOR),
        k_eff_mask_row=replicated(torch.ones(1, 1, 1, 32), FP32, ROW_MAJOR),
        alignment=SimpleNamespace(
            residual=FakeTensor([torch.zeros(1, 4, lanes, 640, dtype=torch.bfloat16) for _ in range(TP)], BF16, TILE, 3)
        ),
    )


def _lane_row(values_by_lane, rows: int, *, width: int | None = None, pad: float = -1.0) -> torch.Tensor:
    row = torch.full((1, 1, 1, 32), pad)
    for lane, values in enumerate(values_by_lane):
        row[..., lane * rows : lane * rows + (width or len(values))] = torch.tensor(
            values[: width or len(values)], dtype=torch.float32
        )
    return row


@pytest.mark.parametrize("lanes,drafts", ((4, 4), (4, 3), (2, 5), (6, 4), (8, 3)))
def test_accept_rows_lanes_is_exact_per_lane_for_every_pattern_and_large_ids(fake, lanes: int, drafts: int) -> None:
    rows = drafts + 1
    stub = _lane_verify_stub(lanes, drafts, [rows - 1] * lanes, [1] * lanes, [0] * lanes)
    assert (stub.constants.sentinel_tail is None) == (lanes * rows == 32)
    patterns = list(itertools.product((0, 1), repeat=drafts))
    garbage = list(LARGE_IDS) * 8
    for start in range(0, len(patterns), lanes):
        chosen = [patterns[(start + lane) % len(patterns)] for lane in range(lanes)]
        for ids in (SMALL_IDS, LARGE_IDS, SMALL_IDS + LARGE_IDS):
            per_lane = [accept_pattern_rows(pattern, ids) for pattern in chosen]
            references = [
                accept_select(
                    torch.tensor(targets, dtype=torch.float32),
                    torch.tensor(draft_ids, dtype=torch.float32),
                    torch.tensor(alignment, dtype=torch.float32),
                )
                for targets, draft_ids, alignment in per_lane
            ]
            for pad_ids in (None, garbage):  # the body pads with the sentinel; real ids on the pad rows change nothing
                argmax = _lane_row([targets for targets, _, _ in per_lane], rows)
                alignment_row = _lane_row([alignment for _, _, alignment in per_lane], rows)
                if pad_ids is not None:
                    argmax[..., lanes * rows :] = torch.tensor(pad_ids[: 32 - lanes * rows], dtype=torch.float32)
                    alignment_row[..., lanes * rows :] = torch.tensor(pad_ids[: 32 - lanes * rows], dtype=torch.float32)
                drafts_row = _lane_row([draft_ids for _, draft_ids, _ in per_lane], rows, width=drafts)
                accepted_row = mtp_lanes.accept_rows_lanes(
                    replicated(argmax, FP32, ROW_MAJOR),
                    replicated(drafts_row, FP32, ROW_MAJOR),
                    stub.k_eff_mask_row,
                    stub.constants,
                )
                assert (
                    accepted_row.dtype is FP32
                    and accepted_row.layout == ROW_MAJOR
                    and accepted_row.shape == (1, 1, 1, lanes)
                )
                counts = [int(value) for value in accepted_row.torch_shards()[0].reshape(-1).tolist()]
                assert counts == [reference["accepted"] for reference in references], (chosen, ids)
                # The gathers at u*R + a_u out of the persistent accept row the body reads.
                for local in stub.accepted_row.locals:
                    local.copy_(mtp_lanes._accepted_row_host(counts, lanes))
                index = mtp_lanes._lane_gather_index(stub)
                assert index.dtype is U32 and index.torch_shards()[0].reshape(-1).tolist() == [
                    lane * rows + count for lane, count in enumerate(counts)
                ]
                next_token = fake.ttnn.gather(replicated(argmax, FP32, ROW_MAJOR), 3, index)
                first_draft = fake.ttnn.gather(replicated(alignment_row, FP32, ROW_MAJOR), 3, index)
                assert next_token.torch_shards()[0].reshape(-1).tolist() == [
                    float(r["next_token_gather"]) for r in references
                ]
                assert first_draft.torch_shards()[0].reshape(-1).tolist() == [
                    float(r["first_draft_gather"]) for r in references
                ]
    # k_eff < k on a lane: its columns past k_eff are masked, so its count is capped there.
    if drafts >= 3:
        targets = [[7, 8, 9, 10, 11, 12][:rows] for _ in range(lanes)]
        argmax = _lane_row(targets, rows)
        drafts_row = _lane_row([t[:drafts] for t in targets], rows, width=drafts)  # every draft matches
        mask = torch.ones(1, 1, 1, 32)
        mask[..., 2:drafts] = 0.0  # lane 0: k_eff = 2
        accepted_row = mtp_lanes.accept_rows_lanes(
            replicated(argmax, FP32, ROW_MAJOR),
            replicated(drafts_row, FP32, ROW_MAJOR),
            replicated(mask, FP32, ROW_MAJOR),
            stub.constants,
        )
        assert accepted_row.torch_shards()[0].reshape(-1).tolist() == [2.0] + [float(drafts)] * (lanes - 1)


@pytest.mark.parametrize("lanes,drafts", ((4, 4), (2, 3), (6, 4), (8, 3)))
def test_lane_residual_row_select_is_exact_for_every_accept(fake, lanes: int, drafts: int) -> None:
    rows = drafts + 1
    torch.manual_seed(71)
    residual = FakeTensor([bf16_randn(1, 4, 32, 640, scale=3.0) for _ in range(TP)], BF16, TILE, 3)
    for trial in range(rows):
        accepted = [(trial + lane) % rows for lane in range(lanes)]
        stub = _lane_verify_stub(lanes, drafts, accepted, [1] * lanes, [0] * lanes)
        mtp_lanes._select_lane_residual_rows(stub, residual)
        for device in range(TP):
            landed = stub.alignment.residual.torch_shards()[device]
            for lane in range(lanes):
                expected = residual.torch_shards()[device][:, :, lane * rows + accepted[lane]].contiguous()
                assert torch.equal(landed[:, :, lane].contiguous().view(torch.int16), expected.view(torch.int16)), (
                    lane,
                    accepted,
                )
        assert residual.alive


def test_lane_position_advance_adds_each_lanes_committed_rows_and_leaves_the_pad_lanes(expect_error, fake) -> None:
    positions, accepted, active = (5, 29, 60, 100), (-1, 0, 2, 4), (1, 1, 0, 1)
    stub = _lane_verify_stub(4, 4, accepted, active, positions)
    assert stub.positions.read() == list(positions) and stub.positions.positions == list(positions)
    mtp_lanes._advance_lane_positions(stub)
    assert stub.positions.read() == [5, 30, 60, 105]
    row = stub.positions.row.torch_shards()[0].reshape(-1).tolist()
    assert row[4:] == [5] * 28 and stub.positions.row.dtype is U32
    stub.positions.advance_mirror([(a + 1) * flag for a, flag in zip(accepted, active)])
    assert stub.positions.positions == [5, 30, 60, 105]
    stub.positions.write((7, 8, 9, 10))
    assert stub.positions.read() == [7, 8, 9, 10] and stub.positions.positions == [7, 8, 9, 10]
    for bad in ((1, 2, 3), (1, 2, 3, -1), (1, 2, 3, 2**32)):
        with expect_error(ValueError):
            stub.positions.write(bad)
    stub.positions.write_lane(2, 177)  # one lane moves
    assert stub.positions.read() == [7, 8, 177, 10] and stub.positions.positions == [7, 8, 177, 10]
    with expect_error(ValueError):
        stub.positions.write_lane(4, 1)
    mask = stub.positions.block_start_mask_row.torch_shards()[0].reshape(-1).tolist()
    assert mask == [0xFFFFFFFC] * 32
    stub.positions.deallocate()


def test_lane_readback_parses_per_lane_and_the_pass_fit_rule_follows_every_active_lane(expect_error) -> None:
    lanes, rows = 4, 5
    values = torch.tensor(
        [3, 1, 0, 4, 11, 12, 13, 14, 21, 22, 23, 24] + list(range(100, 132)) + list(range(200, 232)),
        dtype=torch.float32,
    )
    parsed = mtp_lanes.parse_lane_readback(values, lanes=lanes, rows=rows)
    assert parsed.accepted == (3, 1, 0, 4) and parsed.next_token == (11, 12, 13, 14)
    assert parsed.first_draft == (21, 22, 23, 24)
    assert parsed.argmaxes == tuple(tuple(range(100 + lane * rows, 100 + (lane + 1) * rows)) for lane in range(lanes))
    assert parsed.alignment_argmaxes == tuple(
        tuple(range(200 + lane * rows, 200 + (lane + 1) * rows)) for lane in range(lanes)
    )
    with expect_error(RuntimeError):
        mtp_lanes.parse_lane_readback(values[:-1], lanes=lanes, rows=rows)
    assert mtp_lanes.verify_pass_fits_lanes((0, 1984, 2015, 5), (1, 1, 1, 1), VERIFY_CONTEXT)
    assert not mtp_lanes.verify_pass_fits_lanes((0, 2016, 5, 5), (1, 1, 1, 1), VERIFY_CONTEXT)
    assert mtp_lanes.verify_pass_fits_lanes((0, 2016, 5, 5), (1, 0, 1, 1), VERIFY_CONTEXT)
    with expect_error(ValueError):
        mtp_lanes._mask_row_host((1, 2, 0, 1), 4)
    with expect_error(ValueError):
        mtp_lanes._accepted_row_host((1, 2), 4)


# --------------------------------------------------------------------------- source pins


LANE_BODY = (
    "forward_verify_lanes",
    "forward_commit_lanes",
    "_forward_layer_verify_lanes",
    "_forward_alignment_lanes",
    "accept_rows_lanes",
    "_select_lane_residual_rows",
    "_lane_gather_index",
    "_land_accept_rows",
    "_advance_lane_positions",
)
# The lane methods with no loop over the lanes (gdn.py's _chunk_rows_lanes runs the rows path's recurrence per lane).
LANE_METHODS = {
    GDN_SOURCE: (
        "forward_rows_lanes",
        "commit_rows_lanes",
        "_causal_conv_rows_lanes",
        "_make_chunk_inputs_lanes",
        "_gate_and_project_rows_lanes",
        "build_rows_selectors_lanes",
    ),
    PLE_SOURCE: ("forward_prepared_rows_lanes", "inject_rows_lanes", "_convolve_rows_lanes", "commit_rows_lanes"),
    QSA_SOURCE: (
        "forward_verify_lanes",
        "commit_verify_lanes",
        "_write_compressed_index_lanes_verify",
        "_score_blocks_lanes_verify",
        "_write_packed_kv_lanes_verify",
    ),
}


def _loops_over_lanes(node: ast.AST) -> list[str]:
    """``for``s and comprehensions iterating a ``range`` / ``enumerate`` / ``zip`` over the lanes (the tuples of
    argument checks are not loops over the lanes)."""

    found = []
    for loop in ast.walk(node):
        if isinstance(loop, ast.For):
            iterated = [loop.iter]
        elif isinstance(loop, (ast.ListComp, ast.GeneratorExp, ast.SetComp)):
            iterated = [generator.iter for generator in loop.generators]
        else:
            continue
        for iterated_node in iterated:
            source = ast.unparse(iterated_node)
            if isinstance(iterated_node, ast.Call) and ("lanes" in source or "count" in source):
                found.append(source)
    return found


def test_lane_bodies_create_no_host_tensors_loop_over_no_lane_and_advance_the_positions_last() -> None:
    functions = source_functions(MTP_LANES_SOURCE)
    for name in LANE_BODY:
        for called in ordered_calls(functions[name]):
            assert not called.startswith(HOST_TENSOR_CALLS), (name, called)
            assert "synchronize" not in called and ".item" not in called, (name, called)
        assert _loops_over_lanes(functions[name]) == [], name
    for source, names in LANE_METHODS.items():
        methods = source_functions(source)
        for name in names:
            assert _loops_over_lanes(methods[name]) == [], (source.name, name)
    source = source_segment(MTP_LANES_SOURCE, functions["forward_verify_lanes"])
    tail = source[source.index("_advance_lane_positions(verify)") :]
    assert "ttnn." not in tail
    advance = source_segment(MTP_LANES_SOURCE, functions["_advance_lane_positions"])
    tail = advance[advance.index("ttnn.copy(advanced, row)") :]
    assert "ttnn." not in tail.replace("ttnn.copy(advanced, row)", "").replace("_deallocate", "")
    # Counts by matmul against the block-diagonal constants, selects by gather; never a reduce over ids.
    accept = source_segment(MTP_LANES_SOURCE, functions["accept_rows_lanes"])
    assert "ttnn.matmul(flags_tile, constants.prefix_upper_block" in accept
    assert "ttnn.matmul(prefix, constants.block_sum" in accept and "ttnn.sum(" not in accept
    assert "ttnn.gather(argmax_row, 3, gather_index" in source
    alignment = source_segment(MTP_LANES_SOURCE, functions["_forward_alignment_lanes"])
    assert (
        "ttnn.gather(lanes_row, 3, gather_index" in alignment
        and "_select_lane_residual_rows(verify, residual)" in alignment
    )
    for body in (source, alignment):
        assert "rows=verify.lanes * verify.rows, sentinel_tail=verify.constants.sentinel_tail" in body
    # The QSA lane verify redirects an inactive lane's KV writes: the state needs the scratch rows.
    assert "kv_scratch_rows=kv_scratch_rows(layer.attention.allocated_context)" in source_segment(
        MTP_LANES_SOURCE, functions["_allocate_layer_lane_state"]
    )


def test_every_lane_layer_commits_the_previous_pass_before_its_new_rows() -> None:
    functions = source_functions(MTP_LANES_SOURCE)
    calls = ordered_calls(functions["_forward_layer_verify_lanes"])
    order = {name: calls.index(name) for name in calls}
    assert order["layer.ple.commit_rows_lanes"] < order["layer.ple.inject_rows_lanes"]
    assert order["layer.attention.commit_rows_lanes"] < order["layer.attention.forward_rows_lanes"]
    assert order["layer.attention.commit_verify_lanes"] < order["layer.attention.forward_verify_lanes"]
    assert (
        order["layer.attention_gr.read_rows"]
        < order["layer.attention.forward_rows_lanes"]
        < order["layer.attention_gr.write_rows"]
    )
    assert order["layer.mlp_gr.read_rows"] < order["layer_state.moe.forward"] < order["layer.mlp_gr.write_rows"]
    source = source_segment(MTP_LANES_SOURCE, functions["_forward_layer_verify_lanes"])
    assert (
        "_deallocate(attention_hidden, residual_owner" in source and "_deallocate(mlp_result.hidden_sharded" in source
    )
    assert "_deallocate(moe_hidden" not in source
    model_calls = ordered_calls(functions["forward_verify_lanes"])
    assert model_calls.index("gdn_module.build_rows_selectors_lanes") < model_calls.index(
        "qsa_module.derive_qsa_lane_verify_inputs"
    )
    assert (
        model_calls.index("_embed_rows")
        < model_calls.index("_forward_layer_verify_lanes")
        < model_calls.index("model.final_mixer.rows")
    )
    assert (
        model_calls.index("accept_rows_lanes")
        < model_calls.index("_forward_alignment_lanes")
        < model_calls.index("_advance_lane_positions")
    )
    commit_calls = ordered_calls(functions["forward_commit_lanes"])
    assert commit_calls.index("gdn_module.build_rows_selectors_lanes") < commit_calls.index(
        "layer.ple.commit_rows_lanes"
    )
    assert "alignment.layer.attention.commit_verify_lanes" in commit_calls
    # The commit selects: one derivation per pass, the state select flag is the committed-rows flag (a seeded or
    # inactive lane keeps its recurrent bitwise), and the columns are fresh tensors (never a view of the caller's mask).
    gdn = source_segment(GDN_SOURCE, source_functions(GDN_SOURCE)["build_rows_selectors_lanes"])
    assert "ttnn.ge(committed_col, 1.0" in gdn and "ttnn.experimental.view(active_lanes" not in gdn
    commit = source_segment(GDN_SOURCE, source_functions(GDN_SOURCE)["commit_rows_lanes"])
    assert "selectors.keep_col" in commit and "selectors.commit_col" in commit


# --------------------------------------------------------------------------- admission: the 1-lane image into lane u
def _pack(shards, dtype, layout, shard_dim=None) -> FakeTensor:
    return FakeTensor([s.clone() for s in shards], dtype, layout, shard_dim)


@pytest.mark.parametrize("context,expected", ((32768, 4096), (16384, 4096), (4096, 4096), (2048, 2048), (64, 64)))
def test_kv_scratch_rows_are_one_import_chunk(expect_error, context, expected):
    assert mtp_lanes.kv_scratch_rows(context) == expected and context % expected == 0
    for bad in (4096 + 32, 100, 32):  # not whole chunks / not two blocks
        with expect_error(ValueError):
            mtp_lanes.kv_scratch_rows(bad)


@pytest.mark.parametrize("lane,context", ((0, 8192), (2, 8192), (3, 16384), (1, 2048)))
def test_kv_slab_lands_in_the_lane_in_chunks_the_cache_rows_divide(fake, lane, context):
    """A 4-lane cache of ``4C + scratch`` rows takes a lane's C-row slab as scratch-sized writes (the fake refuses a
    write whose rows do not divide the cache rows, as the device op does); the other lanes and the scratch keep."""

    use_aliasing_view(fake.ttnn)
    lanes, scratch = 4, mtp_lanes.kv_scratch_rows(context)
    cache = FakeTensor(
        [torch.full((1, 1, lanes * context + scratch, 512), 3.0, dtype=torch.bfloat16) for _ in range(TP)],
        BF16,
        ROW_MAJOR,
        1,
    )
    torch.manual_seed(context + lane)
    slab = FakeTensor([bf16_randn(1, 1, context, 512) for _ in range(TP)], BF16, ROW_MAJOR, 1)
    mtp_lanes.write_kv_slab_lane(cache, slab, lane, context, label="test")
    for d in range(TP):
        rows = cache.torch_shards()[d][0, 0]
        assert torch.equal(rows[lane * context : (lane + 1) * context], slab.torch_shards()[d][0, 0])
        others = torch.cat([rows[: lane * context], rows[(lane + 1) * context :]])
        assert torch.equal(others, torch.full_like(others, 3.0))
    assert slab.alive and cache.alive


@pytest.mark.parametrize(
    "family,axis,shape",
    (("compressed", 0, (1, 1, 96, 128)), ("staging", 1, (1, 1, 32, 512)), ("recurrent", 0, (1, 12, 128, 128))),
)
def test_pager_lane_slice_of_a_one_lane_layout_is_a_packed_copy_not_a_view_of_the_state(fake, family, axis, shape):
    """At one lane the pack body's lane slice spans the whole tensor, which the device op returns as its input: the
    part must be an owned copy, so releasing it leaves the state allocated."""

    dtype = FP32 if family == "recurrent" else BF16
    state = FakeTensor([torch.full(shape, 2.5, dtype=dtype.torch) for _ in range(TP)], dtype, TILE)
    part = lanes_module.Qwen38TTNNLanePager._slice_lane(None, state, axis, 0)
    assert part is not state and all(torch.equal(a, b) for a, b in zip(part.torch_shards(), state.torch_shards()))
    fake.ttnn.deallocate(part)
    assert state.alive and not part.alive
    # Into a pack buffer the whole-tensor slice is a copy that lands in the buffer (the single-layer pack path).
    pack = FakeTensor([torch.zeros(shape, dtype=dtype.torch) for _ in range(TP)], dtype, TILE)
    landed = lanes_module.Qwen38TTNNLanePager._slice_lane(None, state, axis, 0, output_tensor=pack)
    assert landed is pack and all(torch.equal(a, b) for a, b in zip(pack.torch_shards(), state.torch_shards()))
    # A real lane slice of a 4-lane tensor is the lane's rows alone.
    wide = list(shape)
    wide[axis] = 4
    lanes4 = FakeTensor(
        [torch.arange(math.prod(wide), dtype=torch.float32).reshape(wide).to(dtype.torch) for _ in range(TP)],
        dtype,
        TILE,
    )
    lane2 = lanes_module.Qwen38TTNNLanePager._slice_lane(None, lanes4, axis, 2)
    assert all(torch.equal(s, w.narrow(axis, 2, 1)) for s, w in zip(lane2.torch_shards(), lanes4.torch_shards()))


def test_pack_slices_of_a_one_layer_family_are_owned_not_the_pack(fake):
    """The alignment layer's packs hold one layer each and a 1-lane layout's lane slice spans the whole tensor: the
    import's slice must be a copy the import may release, never the pack (or the state) itself."""

    pack = _pack([torch.full((1, 1, 32, 128), 5.0, dtype=torch.bfloat16) for _ in range(TP)], BF16, TILE)
    assert fake.ttnn.slice(pack, (0, 0, 0, 0), (1, 1, 32, 128)) is pack  # the device op's no-op path, modelled
    part = mtp_lanes._slice0(pack, 0)
    assert part is not pack and part.tensor_id != pack.tensor_id
    assert all(torch.equal(a, b) for a, b in zip(part.torch_shards(), pack.torch_shards()))
    fake.ttnn.deallocate(part)
    assert pack.alive and not part.alive
    two = _pack([torch.arange(2 * 32 * 128, dtype=torch.float32).reshape(2, 1, 32, 128) for _ in range(TP)], FP32, TILE)
    second = mtp_lanes._slice0(two, 1)
    assert all(torch.equal(s[0], t[1]) for s, t in zip(second.torch_shards(), two.torch_shards()))


@pytest.mark.parametrize("lane,phase", ((0, 0), (2, 1), (3, 3), (1, 2)))
def test_gdn_lane_import_lands_the_recurrent_state_and_the_ring_history_in_one_lane(fake, gdn_weights, lane, phase):
    use_aliasing_view(fake.ttnn)
    _, weights = gdn_weights
    module = make_gdn_module(weights)
    generic = seed_gdn_state(module, 70 + lane)
    generic.conv_phase = phase
    rows_state_b1 = module.allocate_rows_state(module.allocate_rows_constants(5))
    module.sync_rows_history_from_state(generic, rows_state_b1)  # the B=1 seed form
    # The 1-lane image as the pager packs it: recurrent [G,12,128,128], conv rows [1,4G,1,W] (slot s at row 4g + s).
    conv_rows = [
        torch.cat([slot.torch_shards()[d].reshape(1, 1, 1, -1) for slot in generic.conv], dim=1) for d in range(TP)
    ]
    pager = SimpleNamespace(
        packs={
            "recurrent": _pack([x.clone() for x in generic.recurrent.torch_shards()], FP32, TILE, 1),
            "conv": _pack(conv_rows, BF16, ROW_MAJOR, 3),
        }
    )
    conv_tiles = fake.ttnn.to_layout(pager.packs["conv"], TILE)
    lanes = 4
    constants = module.allocate_rows_constants(5)
    lane_state = module.allocate_lane_state(lanes)
    lane_rows = module.allocate_lane_rows_state(constants, module.allocate_lane_rows_constants(lanes, 5))
    for local in lane_state.recurrent.locals:
        local.fill_(7.0)  # the other lanes' states must survive the import untouched
    mtp_lanes.import_gdn_lane(lane_state, lane_rows, pager, conv_tiles, 0, phase=phase, lane=lane)
    for d in range(TP):
        assert torch.equal(lane_state.recurrent.locals[d][lane], generic.recurrent.locals[d][0])
        others = [u for u in range(lanes) if u != lane]
        assert torch.equal(lane_state.recurrent.locals[d][others], torch.full((lanes - 1, 12, 128, 128), 7.0))
        assert torch.equal(lane_rows.history.locals[d][0, lane], rows_state_b1.history.locals[d][0, 0]), lane
        assert torch.count_nonzero(lane_rows.history.locals[d][0, others].float()) == 0
        expected_rows = torch.cat([generic.conv_window()[i].torch_shards()[d].reshape(1, -1) for i in range(3)])
        assert torch.equal(lane_rows.history.locals[d][0, lane, :3], expected_rows)


@pytest.mark.parametrize("lane,position", ((0, 5), (3, 31), (1, 32), (2, 100)))
def test_qsa_lane_import_lands_the_kv_slab_the_blocks_and_the_raw_history_in_one_lane(fake, lane, position):
    use_aliasing_view(fake.ttnn)
    module = make_qsa_module()
    stream = KVStream(seed=300 + lane)
    generic, verify_b1 = module.allocate_generic_state(), module.allocate_verify_state()
    seed_qsa_caches(module, fake.ttnn, stream, generic, verify_b1, position)
    for target in generic.raw_key_ring.locals:
        torch.manual_seed(400 + lane)
        target.copy_(bf16_randn(1, 1, 32, 128))
    module.sync_verify_raw_history_from_ring(generic, verify_b1, position=position)  # the B=1 seed form
    slot = SimpleNamespace(tensors={"kv": (_pack(generic.packed_kv_cache.torch_shards(), BF16, ROW_MAJOR, 1),)})
    pager = SimpleNamespace(
        packs={
            "compressed": _pack(generic.compressed_index_cache.torch_shards(), BF16, TILE),
            "ring": _pack(generic.raw_key_ring.torch_shards(), BF16, TILE),
        },
        kv_stagings=tuple(
            FakeTensor(
                [torch.zeros(1, 1, VERIFY_CONTEXT, 512, dtype=torch.bfloat16) for _ in range(TP)], BF16, ROW_MAJOR, 1
            )
            for _ in range(2)
        ),
    )
    lanes = 4
    lane_state = module.allocate_lane_state(lanes, kv_scratch_rows=KV_SCRATCH_ROWS)
    lane_verify = module.allocate_lane_verify_state(lanes)
    for target in lane_state.packed_kv_cache.locals:
        target.fill_(3.0)
    mtp_lanes.import_qsa_lane(
        lane_state, lane_verify, pager, slot, 0, position=position, lane=lane, allocated_context=VERIFY_CONTEXT
    )
    for d in range(TP):
        cache = lane_state.packed_kv_cache.torch_shards()[d][0, 0]
        assert torch.equal(
            cache[lane * VERIFY_CONTEXT : (lane + 1) * VERIFY_CONTEXT], generic.packed_kv_cache.torch_shards()[d][0, 0]
        )
        untouched = torch.cat([cache[: lane * VERIFY_CONTEXT], cache[(lane + 1) * VERIFY_CONTEXT :]])
        assert torch.equal(untouched, torch.full_like(untouched, 3.0))
    assert torch.equal(
        lane_state.compressed_index_cache.torch_shards()[0][lane], generic.compressed_index_cache.torch_shards()[0][0]
    )
    others = [u for u in range(lanes) if u != lane]
    assert torch.count_nonzero(lane_state.compressed_index_cache.torch_shards()[0][others].float()) == 0
    history = lane_verify.raw_history.torch_shards()[0][0]
    assert torch.equal(history[lane], verify_b1.raw_history.torch_shards()[0][0, 0])
    assert torch.count_nonzero(history[others].float()) == 0


@pytest.mark.parametrize("lane,context", ((0, None), (2, (11, 7)), (3, (95859, 2))))
def test_ple_lane_import_lands_the_nine_slots_and_the_context_in_one_lane(fake, lane, context):
    use_aliasing_view(fake.ttnn)
    module = make_ple_module()
    generic = _seeded_ple_state(module, 80 + lane, context)
    rows_state_b1 = module.allocate_rows_state(5)
    rows_state_b1.load_from_state(generic)  # the B=1 seed form
    pager = SimpleNamespace(
        packs={
            "ple": _pack(
                [torch.cat([slot.torch_shards()[d] for slot in generic.conv], dim=0) for d in range(TP)], BF16, TILE, 3
            )
        }
    )
    lanes_state = module.allocate_lane_rows_state(4, 5)
    for target in lanes_state.history.locals:
        target.fill_(5.0)
    mtp_lanes.import_ple_lane(lanes_state, pager, lane=lane, context=context)
    for d in range(TP):
        assert torch.equal(lanes_state.history.locals[d][lane], rows_state_b1.history.locals[d][0])
        others = [u for u in range(4) if u != lane]
        assert torch.equal(lanes_state.history.locals[d][others], torch.full((3, 9, 4, 640), 5.0, dtype=torch.bfloat16))
    assert lanes_state.token_contexts[lane] == (None if context is None else tuple(context))
    assert all(lanes_state.token_contexts[u] is None for u in range(4) if u != lane)


def test_batched_ple_lane_lookup_is_the_per_lane_lookup_in_one_read(expect_error, monkeypatch) -> None:
    """``lookup_tokens_lanes`` chains every lane from its own context and reads every row in one batch, equal to the
    per-lane ``lookup_tokens`` payloads back to back."""

    lookup = object.__new__(ple_module.Qwen38ResidentPLELookup)
    lookup.vocab_size, lookup.eos_token_id = 1000, 2
    lookup.multipliers, lookup.head_vocab_sizes, lookup.head_offsets = (1,), (1,), (0,)
    reads: list[list[int]] = []

    def read_rows(hashed):
        reads.append(list(hashed))
        return bytearray(b"".join(int(value).to_bytes(4, "little") for value in hashed))

    lookup._read_rows = read_rows
    monkeypatch.setattr(
        ple_module,
        "ngram_token_ids_decode_step",
        lambda token, context, **kwargs: (
            [token * 16 + i + (0 if context is None else sum(context)) for i in range(16)],
            ((2, 2) if context is None else context)[1:] + (token,),
        ),
    )
    tokens_by_lane = [[17, 15, 16, 21, 12], [5, 6, 7, 8, 9], [999, 0, 1, 2, 3]]
    contexts = [None, (7, 11), (2, 2)]
    payload, lane_contexts = lookup.lookup_tokens_lanes(tokens_by_lane, contexts)
    assert len(reads) == 1  # one read for every lane
    expected = bytearray()
    for tokens, context in zip(tokens_by_lane, contexts):
        part, chain = lookup.lookup_tokens(tokens, context)
        expected += part
        assert lane_contexts[tokens_by_lane.index(tokens)] == chain
    assert bytes(payload) == bytes(expected)
    with expect_error(ValueError):
        lookup.lookup_tokens_lanes(tokens_by_lane, contexts[:2])


# --------------------------------------------------------------------------- stage 3: the draft body at B rows


@pytest.mark.parametrize("rows", (4, 5))
def test_commit_verify_lanes_into_the_draft_state_leaves_the_alignment_windows_and_advances_by_one(
    expect_error, fake, rows
) -> None:
    lanes = 4
    module = make_qsa_module()
    alignment_state, draft_state = module.allocate_lane_verify_state(lanes), module.allocate_lane_verify_state(lanes)
    gdn_lane_constants = gdn_module.Qwen38TTNNGDNLaneRowsConstants.allocate(
        "mesh", FakeContract(), lanes=lanes, rows=rows
    )
    torch.manual_seed(9)
    history, raw_rows = bf16_randn(lanes, 32, 128), bf16_randn(lanes, 32, 128)
    history[:, 3:] = 0
    accepted = [0, rows - 1, 2, 1][:lanes]
    for target in alignment_state.raw_history.locals:
        target[0] = history
    for target in alignment_state.raw_rows.locals:  # lane-major flat rows [1, 1, B*32, 128]
        target[0, 0] = raw_rows.reshape(lanes * 32, 128)
    for target in draft_state.raw_history.locals + draft_state.raw_rows.locals:
        target[...] = 5.0
    selectors = _lane_selectors(accepted, [1] * lanes, gdn_lane_constants)
    module.commit_verify_lanes(alignment_state, selectors, target=draft_state)
    for u in range(lanes):
        expected = draft_history(history[u], raw_rows[u], accepted[u], 0)
        assert same_zero(draft_state.raw_history.torch_shards()[0][0, u], expected), u
    assert torch.equal(alignment_state.raw_history.torch_shards()[0][0], history), "alignment histories touched"
    assert torch.equal(alignment_state.raw_rows.torch_shards()[0][0, 0], raw_rows.reshape(lanes * 32, 128)), "raw rows"
    assert torch.all(draft_state.raw_rows.torch_shards()[0] == 5.0)
    # The a = 0 advance per lane: [h1, h2, the lane's raw row 0].
    advance = _lane_selectors([0] * lanes, [1] * lanes, gdn_lane_constants)
    row0 = bf16_randn(lanes, 128)
    for target in draft_state.raw_rows.locals:
        target[...] = 0
        for u in range(lanes):
            target[0, 0, u * 32] = row0[u]
    before = draft_state.raw_history.torch_shards()[0][0].clone()
    module.commit_verify_lanes(draft_state, advance)
    for u in range(lanes):
        advanced = torch.zeros(32, 128, dtype=torch.bfloat16)
        advanced[:2] = before[u, 1:3]
        advanced[2] = row0[u]
        assert same_zero(draft_state.raw_history.torch_shards()[0][0, u], advanced), u
    with expect_error(ValueError):  # allow-pytest.raises: the target must hold the same lanes
        module.commit_verify_lanes(alignment_state, selectors, target=module.allocate_lane_verify_state(2))


class _FakeLaneMoE:
    def __init__(self, *args, rows: int, **kwargs) -> None:
        self.rows = rows

    def release_owned_buffers(self) -> None:
        return None


@pytest.mark.parametrize(
    "drafts,positions,accepted",
    (
        (3, (5, 31, 32, 100), (0, 2, 1, 3)),
        (4, (5, 31, 32, 100), (0, 2, 1, 3)),
        (3, (29, 30, 63, 60), (3, 0, 2, 1)),
        (4, (29, 30, 63, 60), (3, 0, 2, 1)),
        (3, (0, 3, 61, 127), (1, 1, 0, 0)),
        (4, (0, 3, 61, 127), (1, 1, 0, 0)),
        # eight lanes at k = 3 fill the verify tile (8 x 4 = 32 rows): the constants carry no sentinel tail and the
        # assembly concatenates the lane blocks alone
        (3, (5, 31, 32, 100, 29, 63, 60, 127), (0, 2, 1, 3, 3, 0, 2, 1)),
    ),
)
def test_forward_draft_lanes_is_the_eager_row_chain_per_lane_and_assembles_the_next_pass(
    fake, monkeypatch, drafts, positions, accepted
):
    """The lane draft body over the real QSA lane module, constants, positions and selectors, with the step-6
    stand-ins (embedding table, row-wise layer, vocabulary resolve) per lane."""

    lanes, rows = len(positions), drafts + 1
    accepted = [min(a, drafts) for a in accepted]
    world = DraftWorld()
    contract = FakeContract()
    qsa = make_qsa_module()
    (_, chunk_constants, _), lane_constants, _ = _lane_qsa_constants(rows, lanes)
    attention_state = qsa.allocate_lane_state(lanes, kv_scratch_rows=KV_SCRATCH_ROWS)
    alignment_rows = qsa.allocate_lane_verify_state(lanes)
    torch.manual_seed(23)
    for target in alignment_rows.raw_history.locals:
        target[0, :, :3] = bf16_randn(lanes, 3, 128)
    for target in alignment_rows.raw_rows.locals:  # lane-major flat rows [1, 1, B*32, 128]
        target[0, 0] = bf16_randn(lanes * 32, 128)
    mlp = SimpleNamespace(
        weights=object(),
        rows=1,
        mesh_device="mesh",
        mesh_contract=contract,
        tt_ccl=None,
        collective_topology=None,
        synchronization_policy=None,
    )
    layer = SimpleNamespace(
        namespace=Qwen38TTNNLayerNamespace.MTP, layer_index=0, attention=qsa, ple=None, mlp=mlp, mesh_contract=contract
    )
    alignment_layer_state = mtp_lanes.Qwen38TTNNMTPLaneLayerState(
        layer.namespace, 0, attention_state, alignment_rows, None, SimpleNamespace(rows=lanes * rows), None
    )
    final_mixer = SimpleNamespace(rows=lambda residual_rows, flat_views=False: world.final(residual_rows))
    alignment = mtp_lanes.Qwen38TTNNMTPLaneAlignment(
        layer,
        RowsMixer(world.mix),
        final_mixer,
        alignment_layer_state,
        sharded(bf16_randn(1, 4, lanes, DRAFT_WIDTH * TP)),
    )
    owner = object()
    verify = SimpleNamespace(
        lanes=lanes,
        drafts=drafts,
        rows=rows,
        alignment=alignment,
        qsa_chunk_constants=chunk_constants,
        qsa_lane_constants=lane_constants,
        gdn_lane_constants=gdn_module.Qwen38TTNNGDNLaneRowsConstants.allocate("mesh", contract, lanes=lanes, rows=rows),
        constants=mtp_lanes.Qwen38TTNNMTPLaneConstants.build("mesh", contract, lanes=lanes, drafts=drafts),
        positions=mtp_lanes.Qwen38TTNNMTPLanePositions.allocate("mesh", contract, positions, lanes=lanes),
        active_row=_u32_row(_pad32([1] * lanes, 1)),
        inactive_row=_u32_row(_pad32([0] * lanes, 0)),
        accepted_lanes=_lane_column(accepted),
        active_lanes=_lane_column([1] * lanes),
        token_row=replicated(torch.full((1, 1, 1, 32), -1.0), FP32),
        draft_lanes=fp32_lanes([]),
        _owner=owner,
    )

    def rows_chunk(index_rows, block_start_rows):
        world.rope_requests.append(index_rows.torch_shards()[0].reshape(-1).tolist())
        tile = replicated(torch.zeros(1, 1, 32, 64, dtype=torch.bfloat16), BF16)
        return SimpleNamespace(cos=tile, sin=tile, block_start_cos=tile, block_start_sin=tile, deallocate=lambda: None)

    model = draft_body_model(world, contract, owner, rows_chunk)
    seen: list[dict] = []

    def forward_layer_verify_lanes(
        layer_, residual_, layer_state, geometry, *, prepared_ple_rows, rope_rows, qsa_inputs, selectors
    ):
        assert layer_ is layer and layer_state.attention_rows is not alignment_rows and geometry.rows == 1
        assert layer_state.attention_state is attention_state and prepared_ple_rows is None and selectors is None
        assert qsa_inputs.single_row and geometry.qsa_lane_verify_constants.rows == 1
        assert residual_.shape == (1, 4, 32, DRAFT_WIDTH) and residual_.dtype is BF16
        seen.append(
            {
                "blocks": qsa_inputs.block_index_i32[0].torch_shards()[0].tolist(),
                "starts": [int(x.torch_shards()[0].reshape(-1)[0]) for x in qsa_inputs.kv_row_start_lanes],
                "remainders": qsa_inputs.stage_keep.torch_shards()[0].float().sum(dim=(2, 3)).reshape(-1).tolist(),
            }
        )
        out = world.layer(residual_)
        fake.ttnn.deallocate(residual_)
        return out

    def resolve_rows(model_, hidden, *, rows, sentinel_tail):
        ids = world.resolve(hidden).torch_shards()[0].reshape(-1).tolist()[:rows]
        assert sentinel_tail.shape == (1, 1, 1, 32 - rows)
        return fp32_lanes(ids)

    monkeypatch.setattr(mtp_lanes, "_validate_lane_verify_state", lambda model_, verify_: None)
    monkeypatch.setattr(mtp_lanes, "_forward_layer_verify_lanes", forward_layer_verify_lanes)
    monkeypatch.setattr(mtp_lanes, "_resolve_rows", resolve_rows)
    monkeypatch.setattr(mtp_lanes, "Qwen38TTNNMoE", _FakeLaneMoE)

    assert (verify.constants.sentinel_tail is None) == (lanes * rows == 32)
    draft = mtp_lanes.allocate_lane_draft_state(model, verify)
    assert draft.rows == 1 and draft.qsa_verify_constants.rows == 1 and draft.qsa_lane_verify_constants.rows == 1
    assert draft.layer_state.moe.rows == lanes and draft.layer_state.moe_input.shape == (1, 1, lanes, DRAFT_WIDTH)
    assert len(draft.step_offset_rows) == drafts - 1
    next_tokens, first_drafts = [17 + u for u in range(lanes)], [41 + 3 * u for u in range(lanes)]
    argmaxes, alignment_row = list(range(32)), list(range(100, 132))
    readback = mtp_v2.Qwen38TTNNVerifyOutput(
        fp32_lanes(
            [float(a) for a in accepted] + next_tokens + first_drafts + argmaxes + alignment_row,
            width=mtp_lanes.lane_readback_width(lanes),
        )
    )
    history_before = alignment_rows.raw_history.torch_shards()[0].clone()
    raw_rows_before = alignment_rows.raw_rows.torch_shards()[0].clone()
    mtp_lanes.forward_draft_lanes(model, verify, draft, readback)
    pass_readback, tokens = mtp_lanes.read_lane_pass_row(verify, draft)
    residual_all = cat_shards(verify.alignment.residual)
    expected_tokens = []
    for u in range(lanes):
        ids = eager_draft_rows(world, sharded(residual_all[:, :, u : u + 1]), [first_drafts[u]], drafts - 1)
        expected_tokens.append((next_tokens[u], first_drafts[u], *ids))
    assert tokens == tuple(expected_tokens), (drafts, positions, accepted)
    host_row = embedding_module.Qwen38TTNNTokenEmbedding.host_verify_token_rows([t for block in tokens for t in block])
    assert torch.equal(verify.token_row.torch_shards()[0], host_row) and verify.token_row.layout == TILE
    host_drafts = torch.full((1, 1, 1, 32), -1.0)
    for u in range(lanes):
        host_drafts[..., u * rows : u * rows + drafts] = torch.tensor(tokens[u][1:], dtype=torch.float32)
    assert torch.equal(verify.draft_lanes.torch_shards()[0], host_drafts) and verify.draft_lanes.layout == ROW_MAJOR
    pass_row = draft.pass_row.torch_shards()[0]
    split = mtp_lanes.lane_readback_width(lanes)
    assert torch.equal(pass_row[..., :split], readback.readback.torch_shards()[0])
    assert torch.equal(pass_row[..., split:], host_row)
    assert pass_readback.accepted == tuple(accepted) and pass_readback.next_token == tuple(next_tokens)
    assert pass_readback.first_draft == tuple(first_drafts)
    # Step i ran every lane at MTP position P_u + i (its KV block row start, compressed block and block row).
    assert len(seen) == drafts - 1
    for i, entry in enumerate(seen):
        assert entry["blocks"] == [(p + i) // 4 for p in positions], i
        assert entry["starts"] == [u * VERIFY_CONTEXT + ((p + i) & ~31) for u, p in enumerate(positions)], i
        assert entry["remainders"] == [float((p + i) % 32) for p in positions], i
    assert [req[:lanes] for req in world.rope_requests] == [[p + i for p in positions] for i in range(drafts - 1)]
    # The alignment windows are untouched; lane u's draft history started as window_u[a_u + 1 : a_u + 4] and advanced
    # once per row after the first.
    assert torch.equal(alignment_rows.raw_history.torch_shards()[0], history_before)
    assert torch.equal(alignment_rows.raw_rows.torch_shards()[0], raw_rows_before)
    for u in range(lanes):
        raw_rows = raw_rows_before[0, 0, u * 32 : (u + 1) * 32]
        expected_history = draft_history(history_before[0, u], raw_rows, accepted[u], drafts - 2)
        assert same_zero(draft.qsa_state.raw_history.torch_shards()[0][0, u], expected_history), u
    assert verify.positions.read() == list(positions)  # the verify body owns the positions
    mtp_lanes.release_lane_draft_state(model, verify, draft)


def _lane_chain_stub(monkeypatch) -> SimpleNamespace:
    """A 4-lane k = 4 chain's stand-ins: the readback of accepts (1, 4, 0, 2), its assembled next tokens, and a host
    commit that logs the committed counts and advances the positions."""

    lanes, drafts = 4, 4
    rows = drafts + 1
    owner = object()
    positions = mtp_lanes.Qwen38TTNNMTPLanePositions.__new__(mtp_lanes.Qwen38TTNNMTPLanePositions)
    positions.positions, positions.lanes = [10, 20, 30, 40], lanes
    accepted = (1, 4, 0, 2)
    argmaxes = tuple(tuple(100 * u + j for j in range(rows)) for u in range(lanes))
    readback = mtp_lanes.Qwen38TTNNLaneVerifyReadback(
        accepted=accepted,
        next_token=tuple(argmaxes[u][accepted[u]] for u in range(lanes)),
        first_draft=tuple(7 + u for u in range(lanes)),
        argmaxes=argmaxes,
        alignment_argmaxes=argmaxes,
    )
    stub = SimpleNamespace(
        model=SimpleNamespace(_state_owner=owner, allocated_context=VERIFY_CONTEXT),
        verify=SimpleNamespace(lanes=lanes, drafts=drafts, rows=rows, positions=positions, _owner=owner),
        draft=SimpleNamespace(_owner=owner),
        positions=positions,
        readback=readback,
        assembled=tuple(
            (readback.next_token[u], readback.first_draft[u], *range(50 + u, 50 + u + drafts - 1)) for u in range(lanes)
        ),
        counts=[],
    )

    def commit_host(model_, verify_, counts):
        stub.counts.append(list(counts))
        positions.positions = [p + c for p, c in zip(positions.positions, counts)]

    monkeypatch.setattr(mtp_lanes, "_validate_lane_draft_state", lambda model_, verify_, draft_: None)
    monkeypatch.setattr(mtp_lanes, "commit_lane_verify_host", commit_host)
    return stub


def test_lane_chain_steps_commit_then_ple_rows_then_verify_and_draft_and_reads_one_pass_row(
    expect_error, monkeypatch
) -> None:
    stub = _lane_chain_stub(monkeypatch)
    readback, assembled = stub.readback, stub.assembled
    calls: list[str] = []
    monkeypatch.setattr(
        mtp_lanes, "read_lane_pass_row", lambda verify_, draft_: (calls.append("readback"), (readback, assembled))[1]
    )
    monkeypatch.setattr(
        mtp_lanes, "write_lane_ple_rows", lambda model_, verify_, tokens: calls.append(f"ple_rows:{tokens[0][0]}")
    )
    monkeypatch.setattr(
        mtp_lanes, "write_lane_verify_inputs", lambda model_, verify_, tokens: calls.append("host_inputs")
    )
    traces = mtp_lanes.Qwen38TTNNMTPLaneTraces(verify_first=11, commit=12, draft=13)
    names = {11: "verify", 12: "commit", 13: "draft"}
    chain = mtp_lanes.Qwen38TTNNMTPLaneChain(
        stub.model,
        stub.verify,
        stub.draft,
        traces,
        replay=lambda tid: calls.append(f"replay:{names[tid]}"),
        enqueue=lambda tid: calls.append(f"enqueue:{names[tid]}"),
    )
    seed = [[1, 0, 0, 0, 0]] * 4
    record = chain.bootstrap(seed)
    assert calls == ["host_inputs", "enqueue:verify", "enqueue:draft", "readback"]
    assert record.accepted == readback.accepted
    assert record.committed == tuple(row[: a + 1] for row, a in zip(readback.argmaxes, readback.accepted))
    assert stub.counts == [[2, 5, 1, 3]] and record.positions == (10, 20, 30, 40) and chain.next_tokens == assembled
    assert set(record.segments_ns) == {"host_inputs", "verify_enqueue", "draft_enqueue", "readback"}
    calls.clear()
    record = chain.step()
    assert calls == ["enqueue:commit", f"ple_rows:{assembled[0][0]}", "enqueue:verify", "enqueue:draft", "readback"]
    assert record.index == 1 and record.tokens == assembled and record.positions == (12, 25, 31, 43)
    chain.enqueue = None  # the blocking (segment-measuring) form
    calls.clear()
    record = chain.step()
    assert calls == ["replay:commit", f"ple_rows:{assembled[0][0]}", "replay:verify", "replay:draft", "readback"]
    assert set(record.segments_ns) == {"commit_replay", "ple_rows", "verify_replay", "draft_replay", "readback"}
    with expect_error(RuntimeError):  # allow-pytest.raises: a chain that ran passes cannot bootstrap again
        chain.bootstrap(seed)
    # A chain continuing an eager pass takes its assembled tokens and refuses the bootstrap.
    continued = mtp_lanes.Qwen38TTNNMTPLaneChain(
        stub.model, stub.verify, stub.draft, traces, replay=lambda tid: None, next_tokens=assembled
    )
    with expect_error(RuntimeError):  # allow-pytest.raises: seeded chains step
        continued.bootstrap(seed)
    with expect_error(ValueError):  # allow-pytest.raises: the seed must be B blocks of R tokens
        mtp_lanes.Qwen38TTNNMTPLaneChain(
            stub.model, stub.verify, stub.draft, traces, replay=lambda tid: None, next_tokens=[[1, 2]]
        )


def test_lane_draft_body_moves_ids_by_copy_and_orders_history_rows_and_assembly() -> None:
    functions = source_functions(MTP_LANES_SOURCE)
    body = source_segment(MTP_LANES_SOURCE, functions["forward_draft_lanes"])
    history = source_segment(MTP_LANES_SOURCE, functions["forward_draft_history_lanes"])
    for forbidden in (
        "to_torch",
        "from_torch",
        "copy_host_to_device_tensor",
        "copy_device_to_host_tensor",
        "ttnn.matmul(",
    ):
        assert forbidden not in body and forbidden not in history, forbidden
    assert not _loops_over_lanes(functions["forward_draft_history_lanes"])
    # The history derivation lands in the draft state with this pass's accept counts and active flags.
    assert "commit_verify_lanes(alignment.layer_state.attention_rows, selectors, target=draft.qsa_state )" in history
    assert (
        "build_rows_selectors_lanes(verify.accepted_lanes, verify.active_lanes, verify.gdn_lane_constants )" in history
    )
    # The steps: the a = 0 advance between rows, one real row per lane, the alignment layer over the draft state.
    order = [
        "forward_draft_history_lanes(model, verify, draft)",
        "qsa.commit_verify_lanes(draft.qsa_state, draft.advance_selectors)",
        "single_row=True,",
        "embed_device_token_rows(token_tile)",
        "alignment.input_mixer.rows(embedding_rows, roots)",
        "_forward_layer_verify_lanes(alignment.layer, mixed, draft.layer_state, draft,",
        "alignment.final_mixer.rows(residual, flat_views=True)",
        "_resolve_rows(model, hidden, rows=lanes, sentinel_tail=draft.sentinel_tail)",
        '_land(pass_row, draft.pass_row, label="assembled lane pass row")',
        '_land(token_tile, verify.token_row, label="assembled lane verify token row")',
        '_land(draft_lanes, verify.draft_lanes, label="assembled lane draft lanes")',
    ]
    indices = [body.index(text) for text in order]
    assert indices == sorted(indices), order
    assert "selectors=None" in body and "positions.row" in body and "verify.positions.write" not in body
    # The chain: commit, PLE rows, verify, draft, one readback; the sweep's split order.
    tree = ast.parse(MTP_LANES_SOURCE.read_text(encoding="utf-8"))
    chain = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Qwen38TTNNMTPLaneChain")
    step = source_segment(MTP_LANES_SOURCE, chain)
    order = ('"commit_enqueue"', '"ple_rows"', "return self._finish_pass(tokens, segments)")
    indices = [step.index(text) for text in order]
    assert indices == sorted(indices), order
    finish = source_segment(
        MTP_LANES_SOURCE, source_methods(MTP_LANES_SOURCE, "Qwen38TTNNMTPLaneChain")["_finish_pass"]
    )
    order = (
        'f"verify_{form}"',
        'f"draft_{form}"',
        '"readback"',
        "commit_lane_verify_host(self.model, self.verify, [len(block) for block in committed])",
    )
    indices = [finish.index(text) for text in order]
    assert indices == sorted(indices), order


# --------------------------------------------------------------------------- stage 4: the lane image round trip


def _lane_tensors(verify, lane: int) -> dict[str, list[torch.Tensor]]:
    """Lane ``lane``'s slices of every family as per-device torch tensors (the family layouts of the lane states)."""

    qsa_layer, gdn_layer = verify.layers
    alignment = verify.alignment.layer_state
    context = VERIFY_CONTEXT
    out = {}
    for name, state in (("kv", qsa_layer.attention_state), ("alignment_kv", alignment.attention_state)):
        out[name] = [x[0, 0, lane * context : (lane + 1) * context].clone() for x in state.packed_kv_cache.locals]
    for name, state in (("compressed", qsa_layer.attention_state), ("alignment_compressed", alignment.attention_state)):
        out[name] = [x[lane].clone() for x in state.compressed_index_cache.locals]
    for name, rows_state in (
        ("raw_history", qsa_layer.attention_rows),
        ("alignment_raw_history", alignment.attention_rows),
    ):
        out[name] = [x[0, lane].clone() for x in rows_state.raw_history.locals]
    out["recurrent"] = [x[lane].clone() for x in gdn_layer.attention_state.recurrent.locals]
    out["gdn_history"] = [x[0, lane].clone() for x in gdn_layer.attention_rows.history.locals]
    out["ple_history"] = [x[lane].clone() for x in gdn_layer.ple.history.locals]
    return out


def _seed_lane(verify, lane: int, seed: int, position: int, context) -> None:
    """Random values into lane ``lane`` of every family (history rows past the three zero, as the bodies keep them)."""

    torch.manual_seed(seed)
    qsa_layer, gdn_layer = verify.layers
    alignment = verify.alignment.layer_state
    # The KV slabs are sharded (each device its own heads: distinct draws); the compressed caches and raw histories
    # are replicated (one draw for every device), as the lane states hold them.
    for state in (qsa_layer.attention_state, alignment.attention_state):
        for x in state.packed_kv_cache.locals:
            x[0, 0, lane * VERIFY_CONTEXT : lane * VERIFY_CONTEXT + position] = bf16_randn(position, 512)
        blocks = bf16_randn(1, *state.compressed_index_cache.locals[0].shape[2:])
        for x in state.compressed_index_cache.locals:
            x[lane] = blocks
    for rows_state in (qsa_layer.attention_rows, alignment.attention_rows):
        history = bf16_randn(3, 128)
        for x in rows_state.raw_history.locals:
            x[0, lane] = 0
            x[0, lane, :3] = history
    for x in gdn_layer.attention_state.recurrent.locals:
        x[lane] = torch.randn(*x.shape[1:], dtype=torch.float32)
    for x in gdn_layer.attention_rows.history.locals:
        x[0, lane] = 0
        x[0, lane, :3] = bf16_randn(3, x.shape[3])
    for x in gdn_layer.ple.history.locals:
        x[lane] = bf16_randn(*x.shape[1:])
    contexts = list(gdn_layer.ple.token_contexts)
    contexts[lane] = context
    gdn_layer.ple.token_contexts = tuple(contexts)
    verify.positions.write_lane(lane, position)


@pytest.mark.parametrize("lane,target,position", ((2, 2, 30), (2, 0, 30), (3, 1, 100), (0, 3, 1)))
def test_lane_image_round_trip_restores_every_family_bitwise_and_leaves_the_other_lanes(
    expect_error, fake, monkeypatch, gdn_weights, lane, target, position
):
    # A 4-lane MTP verify state of one QSA layer, one GDN layer with the PLE (checkpoint layer 1) and the alignment
    # layer's lane state, over the real lane modules on the fake.
    use_aliasing_view(fake.ttnn)
    lanes, rows = 4, 5
    qsa = make_qsa_module()
    gdn = make_gdn_module(gdn_weights[1])
    ple = make_ple_module()
    moe = SimpleNamespace(rows=lanes * rows)
    gdn_constants = gdn.allocate_rows_constants(rows)
    qsa_layer = mtp_lanes.Qwen38TTNNMTPLaneLayerState(
        "backbone",
        0,
        qsa.allocate_lane_state(lanes, kv_scratch_rows=KV_SCRATCH_ROWS),
        qsa.allocate_lane_verify_state(lanes),
        None,
        moe,
        None,
    )
    gdn_layer = mtp_lanes.Qwen38TTNNMTPLaneLayerState(
        "backbone",
        1,
        gdn.allocate_lane_state(lanes),
        gdn.allocate_lane_rows_state(gdn_constants, gdn.allocate_lane_rows_constants(lanes, rows)),
        ple.allocate_lane_rows_state(lanes, rows),
        moe,
        None,
    )
    alignment_layer = mtp_lanes.Qwen38TTNNMTPLaneLayerState(
        "mtp",
        0,
        qsa.allocate_lane_state(lanes, kv_scratch_rows=KV_SCRATCH_ROWS),
        qsa.allocate_lane_verify_state(lanes),
        None,
        moe,
        None,
    )
    verify = SimpleNamespace(
        lanes=lanes,
        rows=rows,
        drafts=rows - 1,
        layers=(qsa_layer, gdn_layer),
        alignment=SimpleNamespace(layer_state=alignment_layer),
        positions=mtp_lanes.Qwen38TTNNMTPLanePositions.allocate("mesh", FakeContract(), [10, 20, 30, 40], lanes=lanes),
        pass_contexts=[(None,) * (rows + 1) for _ in range(lanes)],
    )
    model = SimpleNamespace(mesh_device="mesh", allocated_context=VERIFY_CONTEXT)
    monkeypatch.setattr(mtp_lanes, "_validate_lane_verify_state", lambda model_, verify_: None)
    _seed_lane(verify, lane, 300 + lane, position, (7, 8 + lane))
    for other in range(4):
        if other != lane:
            _seed_lane(verify, other, 400 + other, 5 + other, (1, other))
    before = _lane_tensors(verify, lane)
    others_before = {u: _lane_tensors(verify, u) for u in range(4) if u not in (lane, target)}
    image = mtp_lanes.evict_lane(model, verify, lane, accepted=-1)
    assert image.position == position and image.ple_context == (7, 8 + lane) and image.accepted == -1
    assert image.kv_rows == mtp_lanes.kv_image_rows(position, VERIFY_CONTEXT) and image.kv_rows % KV_SCRATCH_ROWS == 0
    digest = image.digest()
    assert digest == mtp_lanes.evict_lane(model, verify, lane, accepted=-1).digest()  # the same state twice
    # Scramble the target lane (and the source when they differ), then re-admit the image into the target.
    _seed_lane(verify, target, 999, 3, (9, 9))
    if target != lane:
        _seed_lane(verify, lane, 998, 2, (8, 8))
    mtp_lanes.readmit_lane(model, verify, target, image)
    after = _lane_tensors(verify, target)
    for name in ("kv", "alignment_kv"):
        for a, b in zip(after[name], before[name]):
            assert torch.equal(a[:position], b[:position]), name
    for name in (
        "compressed",
        "alignment_compressed",
        "raw_history",
        "alignment_raw_history",
        "recurrent",
        "gdn_history",
        "ple_history",
    ):
        assert all(torch.equal(a, b) for a, b in zip(after[name], before[name])), name
    assert verify.positions.positions[target] == position and verify.positions.read()[target] == position
    assert verify.layers[1].ple.token_contexts[target] == (7, 8 + lane)
    assert verify.pass_contexts[target] == ((7, 8 + lane),) * (verify.rows + 1)
    assert mtp_lanes.evict_lane(model, verify, target, accepted=-1).digest() == digest  # the round trip is the identity
    for u, tensors in others_before.items():
        now = _lane_tensors(verify, u)
        assert all(torch.equal(a, b) for name in tensors for a, b in zip(now[name], tensors[name])), u
    with expect_error(ValueError):  # allow-pytest.raises: an image with ragged KV rows is refused
        mtp_lanes.readmit_lane(model, verify, target, dataclasses.replace(image, kv_rows=image.kv_rows + 1))


def test_lane_chain_active_mask_parks_and_admits_lanes_exactly(expect_error, monkeypatch) -> None:
    """A parked lane commits nothing and keeps its position; set_active writes the coming mask with the commit mask of
    the pass being committed; set_next_tokens rewrites the row the next step runs on."""

    stub = _lane_chain_stub(monkeypatch)
    calls: list = []
    monkeypatch.setattr(mtp_lanes, "read_lane_pass_row", lambda verify_, draft_: (stub.readback, stub.assembled))
    monkeypatch.setattr(mtp_lanes, "write_lane_ple_rows", lambda model_, verify_, tokens: None)
    monkeypatch.setattr(
        mtp_lanes,
        "write_lane_active",
        lambda model_, verify_, active, *, commit: calls.append(("active", list(active), list(commit))),
    )
    monkeypatch.setattr(
        mtp_lanes,
        "write_lane_tokens",
        lambda model_, verify_, tokens: calls.append(("tokens", [list(b) for b in tokens])),
    )
    chain = mtp_lanes.Qwen38TTNNMTPLaneChain(
        stub.model,
        stub.verify,
        stub.draft,
        mtp_lanes.Qwen38TTNNMTPLaneTraces(1, 2, 3),
        replay=lambda tid: None,
        next_tokens=stub.assembled,
    )
    chain.set_active([1, 0, 1, 1])  # park lane 1: the commit mask stays the previous pass's (all active)
    assert calls[-1] == ("active", [1, 0, 1, 1], [1, 1, 1, 1])
    record = chain.step()
    # The step's commit used the old mask; the refresh right behind it makes the next commit skip the parked lane.
    assert calls[-1] == ("active", [1, 0, 1, 1], [1, 0, 1, 1]) and "mask_refresh" in record.segments_ns
    assert (
        stub.counts[-1] == [2, 0, 1, 3] and record.committed[1] == () and stub.positions.positions == [12, 20, 31, 43]
    )
    calls.clear()
    chain.step()
    assert not any(call[0] == "active" for call in calls)  # no change: nothing rewritten
    chain.set_active([1, 1, 1, 1])  # unpark: the commit mask is the parked pass's mask (lane 1 commits nothing)
    assert calls[-1] == ("active", [1, 1, 1, 1], [1, 0, 1, 1])
    held = [list(b) for b in stub.assembled]
    held[1] = [5, 6, 7, 8, 9]
    chain.set_next_tokens(held)
    assert calls[-1] == ("tokens", held) and chain.next_tokens[1] == (5, 6, 7, 8, 9)
    record = chain.step()
    assert record.tokens[1] == (5, 6, 7, 8, 9) and stub.counts[-1] == [2, 5, 1, 3]
    with expect_error(ValueError):  # allow-pytest.raises: the mask needs B 0/1 flags
        chain.set_active([1, 2, 1, 1])
