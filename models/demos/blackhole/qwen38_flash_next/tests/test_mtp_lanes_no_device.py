# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""MTP lanes stage 1: the lane forms on the torch-backed ttnn of steps 4 and 5 (no device).

The step-5 fake is extended with the batched forms the lanes add (the chunk kernel over B lanes as B independent
single-lane calls -- the per-lane bitwise property the single-chip forms test measured -- the multi-user
``paged_update_cache`` and the ``(shape, padded_shape)`` view of row 0 of every lane's tile).  What is pinned:

* the accept / alignment constants are block-diagonal per lane and zero on the pad columns; the GDN and QSA lane
  selects expand, fold, indicate and pick exactly the rows they name and never a pad row;
* the per-pass QSA lane verify inputs against their torch reference for lanes at distinct positions (every residue,
  both KV-block cases) with inactive lanes redirected to the scratch rows;
* the per-lane rollback invariant: after a lane pass, a per-lane commit (a_u in 0..k, some lanes inactive) and the
  next pass, every active lane's KV rows, compressed blocks and raw history equal its own sequential 1-row stream
  and the B=1 verify path bitwise, and an inactive lane's region is untouched;
* the GDN lane rows are the rows path per lane bitwise (one batched chunk call), and the lane commit lands each
  lane's own prefix state and FIR history (an inactive or seeded lane keeps both bitwise);
* the PLE lane rows are the rows path per lane bitwise, and the lane commit selects each lane's own history;
* the device accept per lane is exact for every match pattern and large ids, the residual row select exact, the
  position advance exact per lane, and the readback parses per lane;
* source pins: no host tensor in the lane bodies, no Python loop over the lanes, every layer commits before its new
  rows, the position advance last.
"""

from __future__ import annotations

import ast
import dataclasses
import itertools
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.tests import test_mtp_v2_step6_draft_no_device as step6
from models.demos.blackhole.qwen38_flash_next.tests.test_mtp_v2_step4_rows_no_device import (
    BF16,
    FP32,
    HEAD_DIM,
    HEADS,
    ROW_MAJOR,
    TILE,
    TP,
    FakeChunk,
    FakeContract,
    FakeTensor,
    _bf16,
    _cat,
    _device_gdn_weights,
    _gdn_module,
    _gdn_oracle_weights,
    _hidden_sharded,
    _ple_module,
    _residual_rows,
    _seed_state,
    install_source_chunk_fake,
)
from models.demos.blackhole.qwen38_flash_next.tests.test_mtp_v2_step5_verify_no_device import (
    BLOCKS,
    CONTEXT,
    HOST_TENSOR_CALLS,
    I32,
    LARGE_IDS,
    SMALL_IDS,
    U32,
    _calls,
    _functions,
    _pattern_rows,
    _qsa_constants,
    _qsa_module,
    _replicated,
    _rope_rows,
    _run_pass,
    _same_zero,
    _seed_caches,
    _segment,
    _Stream,
    make_verify_fake,
)
from models.demos.blackhole.qwen38_flash_next.tools.mtp_v2_verify_reference import accept_select
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


# --------------------------------------------------------------------------- the extended fake


class FakeLaneChunk(FakeChunk):
    """The step-4 kernel model per lane: a batched head-major call is B independent single-lane calls (the property
    the single-chip forms test measured: bitwise per lane), so the lane orchestration around it is pinned bitwise
    against the rows path.  Token-major calls fall through to the step-4 model."""

    def chunk_gated_delta_rule(self, q, k, v, g, beta, *, output_head_major=False, **kwargs):
        if not output_head_major:
            return super().chunk_gated_delta_rule(q, k, v, g, beta, output_head_major=False, **kwargs)
        lanes, rows = q.shape[0], q.shape[1]
        initial_state, scale = kwargs["initial_state"], kwargs["scale"]
        assert kwargs["output_final_state"] and kwargs["chunk_size"] == 32 and rows == 32, (kwargs, q.shape)
        assert len(v.shape) == 3 and v.shape == (lanes, rows, HEADS * HEAD_DIM), v.shape
        assert q.dtype is BF16 and k.dtype is BF16 and v.dtype is BF16 and g.dtype is FP32 and beta.dtype is FP32
        assert initial_state.dtype is FP32 and initial_state.shape[0] == lanes, initial_state.shape
        self.calls.append({"chunk_size": 32, "rows": rows, "lanes": lanes, "flat_v": True, "output_head_major": True})
        outputs, states = [], []
        for index in range(TP):
            per_lane = [
                self.impl(
                    q.torch_shards()[index][lane : lane + 1],
                    k.torch_shards()[index][lane : lane + 1],
                    v.torch_shards()[index][lane : lane + 1].reshape(1, rows, HEADS, HEAD_DIM),
                    g.torch_shards()[index][lane : lane + 1],
                    beta.torch_shards()[index][lane : lane + 1],
                    scale,
                    initial_state.torch_shards()[index][lane : lane + 1],
                )
                for lane in range(lanes)
            ]
            outputs.append(torch.cat([o.permute(0, 2, 1, 3).reshape(HEADS, rows, HEAD_DIM) for o, _ in per_lane]))
            states.append(torch.cat([s for _, s in per_lane]))
        return FakeTensor(outputs, FP32, TILE), FakeTensor(states, FP32, TILE)


def make_lane_fake(chunk: FakeLaneChunk) -> SimpleNamespace:
    fake = make_verify_fake(chunk)
    base_reshape = fake.reshape

    def reshape(t, shape, padded_shape=None, *, pad_value=None, memory_config=None):
        shape = tuple(int(item) for item in shape)
        if padded_shape is not None and not isinstance(padded_shape, (int, float)):
            # The (logical, padded) view: the logical extents inside the padded tile grid (row 0 of every lane's tile
            # for [1,B,32,128] -> [1,B,1,128]; the step-5 single-lane case is the same rule).
            index = tuple(slice(0, size) for size in shape)
            return FakeTensor([x[index].clone() for x in t.torch_shards()], t.dtype, t.layout)
        return base_reshape(t, shape, pad_value=pad_value, memory_config=memory_config)

    def paged_update_cache(cache, row, *, update_idxs_tensor):
        for target, source, index in zip(cache.locals, row.torch_shards(), update_idxs_tensor.torch_shards()):
            users = target.shape[0]
            rows = source.reshape(users, -1, source.shape[-1])
            for user, position in enumerate(index.reshape(-1).tolist()[:users]):
                target[user, 0, int(position)] = rows[user, 0]
        return cache

    base_slice = fake.slice

    def slice_(tensor, start, end, *, memory_config=None, output_tensor=None):
        # The device op returns its input over the whole tensor (``slice.cpp``'s no-op path): the alias the owned
        # slice guards against.  Into an output tensor it copies.
        if output_tensor is None and all(v == 0 for v in start) and tuple(end) == tensor.shape:
            return tensor
        return base_slice(tensor, start, end, memory_config=memory_config, output_tensor=output_tensor)

    fake.slice = slice_
    fake.clone = lambda t, memory_config=None: FakeTensor([x.clone() for x in t.torch_shards()], t.dtype, t.layout)
    fake.reshape = reshape
    fake.experimental.paged_update_cache = paged_update_cache
    fake.num_cores_to_corerangeset = lambda count, grid, row_wise: ("cores", count)
    fake.CoreCoord = lambda x, y: (x, y)
    fake.create_sharded_memory_config = lambda *args, **kwargs: ("sharded", args[0])
    fake.ShardStrategy = SimpleNamespace(HEIGHT="HEIGHT")
    fake.ShardOrientation = SimpleNamespace(ROW_MAJOR="ROW_MAJOR")
    return fake


@pytest.fixture
def fake(monkeypatch):
    chunk = FakeLaneChunk()
    install_source_chunk_fake(monkeypatch, chunk)
    fake_ttnn = make_lane_fake(chunk)
    for module in (
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
    ):
        monkeypatch.setattr(module, "ttnn", fake_ttnn)
        monkeypatch.setattr(module, "replicate_tensor_2d_mesh_mapper", lambda device: "replicate", raising=False)
    monkeypatch.setattr(qsa_module, "apply_partial_rope_prefill", _rope_rows)
    return SimpleNamespace(ttnn=fake_ttnn, chunk=chunk)


def _u32_row(values) -> FakeTensor:
    return _replicated(torch.tensor([int(v) for v in values], dtype=torch.int64).reshape(1, 1, 1, 32), U32, ROW_MAJOR)


def _lane_column(values, dtype=FP32) -> FakeTensor:
    return _replicated(torch.tensor([float(v) for v in values]).reshape(1, len(values), 1, 1), dtype)


def _pad32(values, fill) -> list:
    return list(values) + [fill] * (32 - len(values))


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
    host = qsa_module.qsa_lane_verify_constant_rows(lanes, rows, CONTEXT)
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
    assert set(host["scratch_row"].reshape(-1).tolist()) == {lanes * CONTEXT}
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
            host["block_offsets_lanes"][0, 0, r] - offsets, torch.full_like(offsets, CONTEXT * lane_of[r])
        )
        assert torch.equal(
            host["arange_slots_rows"][0, 0, r], torch.arange(qsa_module.SPARSE_INDEX_CAPACITY) + CONTEXT * lane_of[r]
        )
    with expect_error(ValueError):
        qsa_module.qsa_lane_verify_constant_rows(lanes, 33 // lanes + 1, CONTEXT)


# --------------------------------------------------------------------------- QSA lane verify inputs


def _lane_qsa_constants(rows: int, lanes: int):
    contract = FakeContract()
    lane_constants = qsa_module.Qwen38TTNNQSALaneConstants.build(
        "mesh", contract, lanes=lanes, allocated_context=CONTEXT
    )
    lane_verify = qsa_module.Qwen38TTNNQSALaneVerifyConstants.build(
        "mesh", contract, lanes=lanes, rows=rows, allocated_context=CONTEXT
    )
    return _qsa_constants(rows), lane_constants, lane_verify


def _derive_lane_inputs(fake, positions, active, constants, lane_constants, lane_verify):
    position_constants, chunk_constants, verify_constants = constants
    lanes = len(positions)
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
    )
    return inputs, row_positions


@pytest.mark.parametrize("rows", (4, 5))
@pytest.mark.parametrize("active", ((1, 1, 1, 1), (1, 0, 1, 1), (0, 1, 1, 0)))
@pytest.mark.parametrize("positions", LANE_POSITIONS)
def test_lane_verify_inputs_match_the_torch_reference_for_mixed_positions_and_active_lanes(
    expect_error, fake, positions, active, rows: int
) -> None:
    lanes = len(positions)
    constants, lane_constants, lane_verify = _lane_qsa_constants(rows, lanes)
    derived, row_positions = _derive_lane_inputs(fake, positions, active, constants, lane_constants, lane_verify)
    expected = qsa_module.emulate_qsa_lane_verify_inputs(
        positions, active, lanes=lanes, rows=rows, allocated_compressed_blocks=BLOCKS
    )
    assert torch.equal(row_positions.torch_shards()[0], expected["row_positions"])
    assert _same_zero(derived.indexer_neg_mask.torch_shards()[0], expected["indexer_neg_mask"])
    for name in ("row_keep_bits", "row_fill", "kv_read_indices"):
        assert torch.equal(getattr(derived, name).torch_shards()[0], expected[name]), name
    for block in range(2):
        index = derived.block_index_i32[block]
        assert index.dtype is I32 and index.torch_shards()[0].tolist() == expected["block_index_i32"][block].tolist()
    for name in ("kv_row_start_lanes", "kv_row_start_next_lanes"):
        starts = [int(t.torch_shards()[0].reshape(-1)[0]) for t in getattr(derived, name)]
        assert starts == expected[name].tolist(), name
    for name in ("stage_keep", "stage_a_select", "stage_b_select", "pool_select"):
        assert torch.equal(getattr(derived, name).torch_shards()[0].float(), expected[name].float()), name
    # Every real column lands exactly once, on its own lane's batch and at row P_row % 32; pad columns never land;
    # an inactive lane's block rows are the scratch rows past the last lane.
    both = (derived.stage_a_select.torch_shards()[0] + derived.stage_b_select.torch_shards()[0]).float()[0]
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
        assert start == (u * CONTEXT + (positions[u] & ~31) if active[u] else lanes * CONTEXT), u
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
            positions[:2], active, lanes=lanes, rows=rows, allocated_compressed_blocks=BLOCKS
        )


# --------------------------------------------------------------------------- the per-lane rollback invariant


def _seed_lane_caches(module, ttnn_fake, streams, state, verify_state, positions) -> None:
    lanes = len(streams)
    for lane, (stream, position) in enumerate(zip(streams, positions)):
        base = lane * CONTEXT
        for target, source in zip(state.packed_kv_cache.locals, stream.packed):
            target[0, 0, base : base + position] = source[:position]
            target[0, 0, base + position : base + CONTEXT] = 7.0
        complete = position // 4
        for block in range(complete):
            row = stream.compressed_block(module, ttnn_fake, block)
            for target in state.compressed_index_cache.locals:
                target[lane, 0, block] = row
        for target in state.compressed_index_cache.locals:
            target[lane, 0, complete:] = 9.0
        history = torch.zeros(32, 128, dtype=torch.bfloat16)
        for back in range(1, 4):
            if position - back >= 0:
                history[3 - back] = stream.raw[position - back]
        for target in verify_state.raw_history.locals:
            target[0, lane] = history
    for target in state.packed_kv_cache.locals:
        target[0, 0, lanes * CONTEXT :] = 5.0


def _run_lane_pass(
    module, fake, streams, state, verify_state, positions, active, rows, constants, lane_constants, lane_verify
):
    """The two lane verify cache writers on every lane's rows P_u .. P_u + R - 1 (pad rows and an inactive lane's rows
    zero: whatever an inactive lane's rows hold, its KV writes land in the scratch rows and its pooled rows in blocks
    at or past P_u // 4 that its next real pass rewrites before any row can name them)."""

    lanes = len(streams)
    inputs, row_positions = _derive_lane_inputs(fake, positions, active, constants, lane_constants, lane_verify)
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
            start = min((position & ~3) + 4 * block, CONTEXT - 1)
            cos[0, 0, 2 * lane + block], sin[0, 0, 2 * lane + block] = stream.cos[start], stream.sin[start]
    module._write_compressed_index_lanes_verify(
        state,
        verify_state,
        _replicated(raw, BF16),
        _replicated(cos, BF16),
        _replicated(sin, BF16),
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
    """Pass N with every lane active, a per-lane commit (a_u drafts; inactive lanes commit nothing), pass N + 1 with
    the active mask: every active lane's visible KV rows and compressed blocks equal its own sequential stream and
    the B=1 verify path on that stream bitwise, its raw history holds P'_u - 3 .. P'_u - 1; an inactive lane's region
    and history are untouched (its writes landed in the scratch rows)."""

    accepted = tuple(min(a, rows - 1) for a in accepted)
    lanes = len(positions)
    constants, lane_constants, lane_verify = _lane_qsa_constants(rows, lanes)
    gdn_lane_constants = gdn_module.Qwen38TTNNGDNLaneRowsConstants.allocate(
        "mesh", FakeContract(), lanes=lanes, rows=rows
    )
    gdn_constants = gdn_module.Qwen38TTNNGDNRowsConstants.allocate("mesh", FakeContract(), rows=rows)
    module = _qsa_module(fake.ttnn)
    streams = [_Stream(seed=200 + lane) for lane in range(lanes)]
    state = module.allocate_lane_state(lanes, kv_scratch_rows=mtp_lanes.kv_scratch_rows(CONTEXT))
    verify_state = module.allocate_lane_verify_state(lanes)
    assert state.packed_kv_cache.shape == (1, 1, lanes * CONTEXT + mtp_lanes.kv_scratch_rows(CONTEXT), 512)
    _seed_lane_caches(module, fake.ttnn, streams, state, verify_state, positions)
    _run_lane_pass(
        module,
        fake,
        streams,
        state,
        verify_state,
        positions,
        (1,) * lanes,
        rows,
        constants,
        lane_constants,
        lane_verify,
    )
    # The B=1 verify path per active lane on the same stream (pass N before the scramble).
    references = {}
    for lane in range(lanes):
        if active[lane]:
            single = _qsa_module(fake.ttnn)
            generic, single_verify = single.allocate_generic_state(), single.allocate_verify_state()
            _seed_caches(single, fake.ttnn, streams[lane], generic, single_verify, positions[lane])
            _run_pass(single, fake.ttnn, streams[lane], generic, single_verify, positions[lane], rows, constants)
            references[lane] = (single, generic, single_verify)
    after_pass = [x.clone() for x in state.packed_kv_cache.locals]
    after_blocks = [x.clone() for x in state.compressed_index_cache.locals]
    next_positions = tuple(p + (a + 1) * flag for p, a, flag in zip(positions, accepted, active))
    for lane in range(lanes):
        if active[lane]:  # rows past the committed prefix are garbage from here on: scramble the stream there
            torch.manual_seed(500 + lane)
            streams[lane].raw[next_positions[lane] :] = _bf16(CONTEXT - next_positions[lane], 128)
            for device in range(TP):
                streams[lane].packed[device][next_positions[lane] :] = _bf16(CONTEXT - next_positions[lane], 512)
    selectors = gdn_module.build_rows_selectors_lanes(_lane_column(accepted), _lane_column(active), gdn_lane_constants)
    module.commit_verify_lanes(verify_state, selectors)
    history = verify_state.raw_history.torch_shards()[0]
    for lane in range(lanes):
        for back in range(1, 4):
            position = next_positions[lane] - back
            expected = streams[lane].raw[position] if position >= 0 else torch.zeros(128, dtype=torch.bfloat16)
            assert torch.equal(history[0, lane, 3 - back], expected), (lane, back)
        assert torch.count_nonzero(history[0, lane, 3:].float()) == 0
    _run_lane_pass(
        module, fake, streams, state, verify_state, next_positions, active, rows, constants, lane_constants, lane_verify
    )
    for lane in range(lanes):
        base = lane * CONTEXT
        if not active[lane]:
            for device in range(TP):
                region = state.packed_kv_cache.torch_shards()[device][0, 0, base : base + CONTEXT]
                assert torch.equal(region, after_pass[device][0, 0, base : base + CONTEXT]), (lane, device)
            blocks = state.compressed_index_cache.torch_shards()[0][lane, 0]
            complete = positions[lane] // 4  # its complete blocks stay; the two it rewrote hold finite junk
            assert torch.equal(blocks[:complete], after_blocks[0][lane, 0, :complete]), lane
            assert torch.equal(blocks[complete + 2 :], after_blocks[0][lane, 0, complete + 2 :]), lane
            assert torch.isfinite(blocks.float()).all()
            continue
        single, generic, single_verify = references[lane]
        single.commit_verify(
            single_verify,
            gdn_module.build_rows_selectors(
                _replicated(torch.full((1, 1, 1, 1), float(accepted[lane])), FP32), gdn_constants
            ),
        )
        _run_pass(single, fake.ttnn, streams[lane], generic, single_verify, next_positions[lane], rows, constants)
        visible = next_positions[lane] + rows
        for device in range(TP):
            region = state.packed_kv_cache.torch_shards()[device][0, 0, base : base + CONTEXT]
            assert torch.equal(region[:visible], streams[lane].packed[device][:visible]), (lane, device)
            assert torch.equal(region[:visible], generic.packed_kv_cache.torch_shards()[device][0, 0, :visible]), (
                lane,
                device,
            )
            # Rows past the two blocks the pass wrote are untouched.
            untouched = (next_positions[lane] & ~31) + 64
            assert torch.equal(region[untouched:], after_pass[device][0, 0, base + untouched : base + CONTEXT]), lane
        blocks = state.compressed_index_cache.torch_shards()[0][lane, 0]
        for block in range(visible // 4):
            assert torch.equal(blocks[block], streams[lane].compressed_block(module, fake.ttnn, block)), (lane, block)
        assert torch.equal(
            blocks[: visible // 4], generic.compressed_index_cache.torch_shards()[0][0, 0, : visible // 4]
        )
        assert torch.equal(history[0, lane], single_verify.raw_history.torch_shards()[0][0, 0])
        assert torch.isfinite(blocks.float()).all()
    scratch = state.packed_kv_cache.torch_shards()[0][0, 0, lanes * CONTEXT :]
    assert scratch.shape[0] == mtp_lanes.kv_scratch_rows(CONTEXT) and torch.isfinite(scratch.float()).all()
    if not all(active):
        assert not torch.equal(scratch, torch.full_like(scratch, 5.0))  # the redirected writes landed here
    selectors.deallocate()
    module.release_lane_verify_state(verify_state)
    module.release_lane_state(state)


# --------------------------------------------------------------------------- GDN lane rows


@pytest.fixture(scope="module")
def gdn_weights():
    oracle = _gdn_oracle_weights()
    return oracle, _device_gdn_weights(oracle)


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
    module = _gdn_module(weights)
    torch.manual_seed(31)
    hidden = [torch.randn(1, rows, 2560).to(torch.bfloat16) for _ in range(lanes)]
    constants = module.allocate_rows_constants(rows)
    lane_constants = module.allocate_lane_rows_constants(lanes, rows)
    states = [_seed_state(module, 40 + lane) for lane in range(lanes)]
    lane_state = module.allocate_lane_state(lanes)
    lane_rows = module.allocate_lane_rows_state(constants, lane_constants)
    references = []
    for lane in range(lanes):
        rows_state = module.allocate_rows_state(constants)
        module.sync_rows_history_from_state(states[lane], rows_state)
        for device in range(TP):
            lane_state.recurrent.locals[device][lane] = states[lane].recurrent.locals[device][0]
            lane_rows.history.locals[device][0, lane] = rows_state.history.locals[device][0, 0]
        result = module.forward_rows(_hidden_sharded(hidden[lane]), states[lane], rows_state, full_tile=True)
        references.append((rows_state, result))
    tile = torch.zeros(1, 32, 2560, dtype=torch.bfloat16)
    for lane in range(lanes):
        tile[:, lane * rows : (lane + 1) * rows] = hidden[lane]
    calls = len(fake.chunk.calls)
    result = module.forward_rows_lanes(_hidden_sharded(tile), lane_state, lane_rows)
    assert len(fake.chunk.calls) - calls == 1 and fake.chunk.calls[-1]["lanes"] == lanes
    output = _cat(result.hidden_rows, 3)
    assert output.shape == (1, 1, 32, 2560) and torch.count_nonzero(output[:, :, lanes * rows :].float()) == 0
    final = _cat(result.final_state, 1)
    for lane in range(lanes):
        rows_state, reference = references[lane]
        assert torch.equal(
            output[:, :, lane * rows : (lane + 1) * rows], _cat(reference.hidden_rows, 3)[:, :, :rows]
        ), lane
        assert torch.equal(final[lane : lane + 1], _cat(reference.final_state, 1)), lane
        for name in ("q", "k"):
            assert torch.equal(
                _cat(getattr(lane_rows, name), 2)[lane : lane + 1], _cat(getattr(rows_state, name), 2)
            ), name
        for name in ("v", "beta", "g"):
            assert torch.equal(
                _cat(getattr(lane_rows, name), 3)[:, lane : lane + 1], _cat(getattr(rows_state, name), 3)
            ), name
    # The commit: c_u = (a_u + 1) * active_u rows per lane; the state select is [c_u >= 1].
    selectors = gdn_module.build_rows_selectors_lanes(_lane_column(accepted), _lane_column(active), lane_constants)
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
    assert len(fake.chunk.calls) - calls == 1  # one masked rerun for every lane
    for lane in range(lanes):
        rows_state, _ = references[lane]
        if committed[lane] == 0:
            for device in range(TP):
                assert torch.equal(lane_state.recurrent.locals[device][lane], before_recurrent[device][lane]), lane
                assert torch.equal(lane_rows.history.locals[device][0, lane], before_history[device][0, lane]), lane
            continue
        module.commit_rows(
            states[lane],
            rows_state,
            gdn_module.build_rows_selectors(
                _replicated(torch.full((1, 1, 1, 1), float(accepted[lane])), FP32), constants
            ),
        )
        for device in range(TP):
            assert _same_zero(lane_state.recurrent.locals[device][lane], states[lane].recurrent.locals[device][0]), lane
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
    module = _ple_module()
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
        result = module.forward_prepared_rows(_residual_rows(residuals[lane]), prepared, rows_state)
        references.append((rows_state, prepared, _cat(result.residual_delta, 3)))
        contexts.append(rows_state.token_context)
        for device in range(TP):
            lanes_state.history.locals[device][lane] = rows_state.history.locals[device][0]
    lanes_state.token_contexts = tuple(contexts)
    lanes_state.validate()
    host_rows = torch.cat([module.host_rows(tokens[lane], contexts[lane])[0] for lane in range(lanes)], dim=2)
    for lane in range(lanes):
        assert torch.equal(
            host_rows[:, :, lane * rows : (lane + 1) * rows], _cat(references[lane][1].embedding_rows, 3)
        )
    prepared_lanes = ple_module.Qwen38TTNNPLERowsPreparedInput(
        FakeTensor([piece.clone() for piece in torch.chunk(host_rows, TP, dim=3)], BF16, ROW_MAJOR, 3),
        tuple(token for lane_tokens in tokens for token in lane_tokens),
        (),
    )
    delta = _cat(
        module.forward_prepared_rows_lanes(_residual_rows(torch.cat(residuals, dim=0)), prepared_lanes, lanes_state), 3
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
    injected = _cat(
        module.inject_rows_lanes(
            FakeTensor([piece.clone() for piece in torch.chunk(branch_major, TP, dim=3)], BF16, TILE, 3),
            prepared_lanes,
            lanes_state,
        ),
        3,
    )
    for lane in range(lanes):
        single = residuals[lane].permute(1, 0, 2).reshape(1, 4, rows, 2560)
        reference = module.inject_rows(
            FakeTensor([piece.clone() for piece in torch.chunk(single, TP, dim=3)], BF16, TILE, 3),
            references[lane][1],
            references[lane][0],
        )
        assert torch.equal(injected[:, :, lane * rows : (lane + 1) * rows], _cat(reference, 3)), lane
    gdn_constants = gdn_module.Qwen38TTNNGDNRowsConstants.allocate("mesh", FakeContract(), rows=rows)
    lane_constants = gdn_module.Qwen38TTNNGDNLaneRowsConstants.allocate("mesh", FakeContract(), lanes=lanes, rows=rows)
    selectors = gdn_module.build_rows_selectors_lanes(_lane_column(accepted), _lane_column(active), lane_constants)
    before = [x.clone() for x in lanes_state.history.locals]
    module.commit_rows_lanes(lanes_state, selectors)
    for lane in range(lanes):
        committed = (accepted[lane] + 1) * active[lane]
        if committed == 0:
            for device in range(TP):
                assert torch.equal(lanes_state.history.locals[device][lane], before[device][lane]), lane
            continue
        module.commit_rows(
            references[lane][0],
            gdn_module.build_rows_selectors(
                _replicated(torch.full((1, 1, 1, 1), float(accepted[lane])), FP32), gdn_constants
            ),
        )
        for device in range(TP):
            assert _same_zero(
                lanes_state.history.locals[device][lane], references[lane][0].history.locals[device][0]
            ), lane


# --------------------------------------------------------------------------- the lane accept, select and advance


def _lane_verify_stub(lanes: int, drafts: int, accepted, active, positions):
    rows = drafts + 1
    contract = FakeContract()
    constants = mtp_lanes.Qwen38TTNNMTPLaneConstants.build("mesh", contract, lanes=lanes, drafts=drafts)
    lane_verify = qsa_module.Qwen38TTNNQSALaneVerifyConstants.build(
        "mesh", contract, lanes=lanes, rows=rows, allocated_context=CONTEXT
    )
    return SimpleNamespace(
        lanes=lanes,
        drafts=drafts,
        rows=rows,
        constants=constants,
        qsa_lane_verify_constants=lane_verify,
        positions=mtp_lanes.Qwen38TTNNMTPLanePositions.allocate("mesh", contract, positions, lanes=lanes),
        accepted_row=_replicated(mtp_lanes._accepted_row_host(accepted, lanes), FP32, ROW_MAJOR),
        active_row=_replicated(mtp_lanes._mask_row_host(active, lanes), U32, ROW_MAJOR),
        k_eff_mask_row=_replicated(torch.ones(1, 1, 1, 32), FP32, ROW_MAJOR),
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
            per_lane = [_pattern_rows(pattern, ids) for pattern in chosen]
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
                    _replicated(argmax, FP32, ROW_MAJOR),
                    _replicated(drafts_row, FP32, ROW_MAJOR),
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
                next_token = fake.ttnn.gather(_replicated(argmax, FP32, ROW_MAJOR), 3, index)
                first_draft = fake.ttnn.gather(_replicated(alignment_row, FP32, ROW_MAJOR), 3, index)
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
            _replicated(argmax, FP32, ROW_MAJOR),
            _replicated(drafts_row, FP32, ROW_MAJOR),
            _replicated(mask, FP32, ROW_MAJOR),
            stub.constants,
        )
        assert accepted_row.torch_shards()[0].reshape(-1).tolist() == [2.0] + [float(drafts)] * (lanes - 1)


@pytest.mark.parametrize("lanes,drafts", ((4, 4), (2, 3), (6, 4), (8, 3)))
def test_lane_residual_row_select_is_exact_for_every_accept(fake, lanes: int, drafts: int) -> None:
    rows = drafts + 1
    torch.manual_seed(71)
    residual = FakeTensor([_bf16(1, 4, 32, 640, scale=3.0) for _ in range(TP)], BF16, TILE, 3)
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
    assert mtp_lanes.verify_pass_fits_lanes((0, 1984, 2015, 5), (1, 1, 1, 1), CONTEXT)
    assert not mtp_lanes.verify_pass_fits_lanes((0, 2016, 5, 5), (1, 1, 1, 1), CONTEXT)
    assert mtp_lanes.verify_pass_fits_lanes((0, 2016, 5, 5), (1, 0, 1, 1), CONTEXT)
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
LANE_METHODS = {
    GDN_SOURCE: (
        "forward_rows_lanes",
        "commit_rows_lanes",
        "_causal_conv_rows_lanes",
        "_make_chunk_inputs_lanes",
        "_chunk_rows_lanes",
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
    functions = _functions(MTP_LANES_SOURCE)
    for name in LANE_BODY:
        for called in _calls(functions[name]):
            assert not called.startswith(HOST_TENSOR_CALLS), (name, called)
            assert "synchronize" not in called and ".item" not in called, (name, called)
        assert _loops_over_lanes(functions[name]) == [], name
    for source, names in LANE_METHODS.items():
        methods = _functions(source)
        for name in names:
            assert _loops_over_lanes(methods[name]) == [], (source.name, name)
    source = _segment(MTP_LANES_SOURCE, functions["forward_verify_lanes"])
    tail = source[source.index("_advance_lane_positions(verify)") :]
    assert "ttnn." not in tail
    advance = _segment(MTP_LANES_SOURCE, functions["_advance_lane_positions"])
    tail = advance[advance.index("ttnn.copy(advanced, row)") :]
    assert "ttnn." not in tail.replace("ttnn.copy(advanced, row)", "").replace("_deallocate", "")
    # Counts by matmul against the block-diagonal constants, selects by gather; never a reduce over ids.
    accept = _segment(MTP_LANES_SOURCE, functions["accept_rows_lanes"])
    assert "ttnn.matmul(flags_tile, constants.prefix_upper_block" in accept
    assert "ttnn.matmul(prefix, constants.block_sum" in accept and "ttnn.sum(" not in accept
    assert "ttnn.gather(argmax_row, 3, gather_index" in source
    alignment = _segment(MTP_LANES_SOURCE, functions["_forward_alignment_lanes"])
    assert (
        "ttnn.gather(lanes_row, 3, gather_index" in alignment
        and "_select_lane_residual_rows(verify, residual)" in alignment
    )
    for body in (source, alignment):
        assert "rows=verify.lanes * verify.rows, sentinel_tail=verify.constants.sentinel_tail" in body
    # The QSA lane verify redirects an inactive lane's KV writes: the state needs the scratch rows.
    assert "kv_scratch_rows=kv_scratch_rows(layer.attention.allocated_context)" in _segment(
        MTP_LANES_SOURCE, functions["_allocate_layer_lane_state"]
    )


def test_every_lane_layer_commits_the_previous_pass_before_its_new_rows() -> None:
    functions = _functions(MTP_LANES_SOURCE)
    calls = _calls(functions["_forward_layer_verify_lanes"])
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
    source = _segment(MTP_LANES_SOURCE, functions["_forward_layer_verify_lanes"])
    assert (
        "_deallocate(attention_hidden, residual_owner" in source and "_deallocate(mlp_result.hidden_sharded" in source
    )
    assert "_deallocate(moe_hidden" not in source
    model_calls = _calls(functions["forward_verify_lanes"])
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
    commit_calls = _calls(functions["forward_commit_lanes"])
    assert commit_calls.index("gdn_module.build_rows_selectors_lanes") < commit_calls.index(
        "layer.ple.commit_rows_lanes"
    )
    assert "alignment.layer.attention.commit_verify_lanes" in commit_calls
    # The commit selects: one derivation per pass, the state select flag is the committed-rows flag (a seeded or
    # inactive lane keeps its recurrent bitwise), and the columns are fresh tensors (never a view of the caller's mask).
    gdn = _segment(GDN_SOURCE, _functions(GDN_SOURCE)["build_rows_selectors_lanes"])
    assert "ttnn.ge(committed_col, 1.0" in gdn and "ttnn.experimental.view(active_lanes" not in gdn
    commit = _segment(GDN_SOURCE, _functions(GDN_SOURCE)["commit_rows_lanes"])
    assert "selectors.keep_col" in commit and "selectors.commit_col" in commit


# --------------------------------------------------------------------------- admission: the 1-lane image into lane u


def _install_admission_ops(fake) -> None:
    """The ops the lane import adds to the fake: ``fill_cache`` (batch ``batch_idx`` of the cache <- the input), the
    two argument forms of ``update_padded_kv_cache`` (the pager's ints: the start row is ``kv_actual_global``; the
    verify writer's tensors), and a ``view`` that ALIASES its source whenever the tile-page reinterpretation is a plain
    reshape (a fill through the view then lands in the owner, as on device)."""

    from models.demos.blackhole.qwen38_flash_next.tests.test_mtp_v2_step4_rows_no_device import (
        _from_tile_pages,
        _tile_pages,
    )

    def fill_cache(cache, source, batch_idx: int):
        for target, part in zip(cache.locals, source.torch_shards()):
            assert part.shape[0] == 1 and tuple(part.shape[1:]) == tuple(target.shape[1:]), (part.shape, target.shape)
            target[batch_idx] = part[0]
        return cache

    def update_padded_kv_cache(cache, staging, slot_idx, a, b, c, axis, valid_global=None, tp_axis=None):
        for index, (target, source) in enumerate(zip(cache.locals, staging.torch_shards())):
            begin = int(c) if isinstance(slot_idx, int) else int(a.torch_shards()[index].reshape(-1)[0])
            rows = source.shape[2]
            assert begin % 32 == 0 and begin + rows <= target.shape[2], (begin, rows, target.shape)
            # The op's block-cyclic invariant: the cache rows are a multiple of the rows one call writes.
            assert target.shape[2] % rows == 0, (target.shape, rows)
            target[0, 0, begin : begin + rows] = source[0, 0]
        return cache

    def view(t, shape, memory_config=None):
        shape = tuple(int(item) for item in shape)
        faithful = [_from_tile_pages(_tile_pages(x), shape) for x in t.torch_shards()]
        if all(torch.equal(f, x.reshape(shape)) for f, x in zip(faithful, t.torch_shards())):
            return FakeTensor([x.reshape(shape) for x in t.torch_shards()], t.dtype, t.layout)  # the owner's storage
        return FakeTensor(faithful, t.dtype, t.layout)

    fake.ttnn.fill_cache = fill_cache
    fake.ttnn.experimental.deepseek_prefill.update_padded_kv_cache = update_padded_kv_cache
    fake.ttnn.experimental.view = view


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

    _install_admission_ops(fake)
    lanes, scratch = 4, mtp_lanes.kv_scratch_rows(context)
    cache = FakeTensor(
        [torch.full((1, 1, lanes * context + scratch, 512), 3.0, dtype=torch.bfloat16) for _ in range(TP)],
        BF16,
        ROW_MAJOR,
        1,
    )
    torch.manual_seed(context + lane)
    slab = FakeTensor([_bf16(1, 1, context, 512) for _ in range(TP)], BF16, ROW_MAJOR, 1)
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
    """The pager's pack body slices lane u of every state tensor and releases the parts after the concat; at one lane
    (the generic-state pager of the lane server and of the lanes sweep) the slice spans the whole tensor, which the
    device op returns as its input.  The part must be an owned copy: releasing it leaves the state allocated."""

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
        [
            torch.arange(int(torch.tensor(wide).prod()), dtype=torch.float32).reshape(wide).to(dtype.torch)
            for _ in range(TP)
        ],
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
    _install_admission_ops(fake)
    _, weights = gdn_weights
    module = _gdn_module(weights)
    generic = _seed_state(module, 70 + lane)
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
    _install_admission_ops(fake)
    module = _qsa_module(fake.ttnn)
    stream = _Stream(seed=300 + lane)
    generic, verify_b1 = module.allocate_generic_state(), module.allocate_verify_state()
    _seed_caches(module, fake.ttnn, stream, generic, verify_b1, position)
    for target in generic.raw_key_ring.locals:
        torch.manual_seed(400 + lane)
        target.copy_(_bf16(1, 1, 32, 128))
    module.sync_verify_raw_history_from_ring(generic, verify_b1, position=position)  # the B=1 seed form
    slot = SimpleNamespace(tensors={"kv": (_pack(generic.packed_kv_cache.torch_shards(), BF16, ROW_MAJOR, 1),)})
    pager = SimpleNamespace(
        packs={
            "compressed": _pack(generic.compressed_index_cache.torch_shards(), BF16, TILE),
            "ring": _pack(generic.raw_key_ring.torch_shards(), BF16, TILE),
        },
        kv_stagings=tuple(
            FakeTensor([torch.zeros(1, 1, CONTEXT, 512, dtype=torch.bfloat16) for _ in range(TP)], BF16, ROW_MAJOR, 1)
            for _ in range(2)
        ),
    )
    lanes = 4
    lane_state = module.allocate_lane_state(lanes, kv_scratch_rows=mtp_lanes.kv_scratch_rows(CONTEXT))
    lane_verify = module.allocate_lane_verify_state(lanes)
    for target in lane_state.packed_kv_cache.locals:
        target.fill_(3.0)
    mtp_lanes.import_qsa_lane(
        lane_state, lane_verify, pager, slot, 0, position=position, lane=lane, allocated_context=CONTEXT
    )
    for d in range(TP):
        cache = lane_state.packed_kv_cache.torch_shards()[d][0, 0]
        assert torch.equal(
            cache[lane * CONTEXT : (lane + 1) * CONTEXT], generic.packed_kv_cache.torch_shards()[d][0, 0]
        )
        untouched = torch.cat([cache[: lane * CONTEXT], cache[(lane + 1) * CONTEXT :]])
        assert torch.equal(untouched, torch.full_like(untouched, 3.0))
    assert torch.equal(
        lane_state.compressed_index_cache.torch_shards()[0][lane], generic.compressed_index_cache.torch_shards()[0][0]
    )
    assert (
        torch.count_nonzero(
            lane_state.compressed_index_cache.torch_shards()[0][[u for u in range(lanes) if u != lane]].float()
        )
        == 0
    )
    assert torch.equal(
        lane_verify.raw_history.torch_shards()[0][0, lane], verify_b1.raw_history.torch_shards()[0][0, 0]
    )
    assert (
        torch.count_nonzero(
            lane_verify.raw_history.torch_shards()[0][0, [u for u in range(lanes) if u != lane]].float()
        )
        == 0
    )


@pytest.mark.parametrize("lane,context", ((0, None), (2, (11, 7)), (3, (95859, 2))))
def test_ple_lane_import_lands_the_nine_slots_and_the_context_in_one_lane(fake, lane, context):
    _install_admission_ops(fake)
    module = _ple_module()
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


def test_lane_position_write_lane_moves_one_lane(expect_error, fake) -> None:
    stub = _lane_verify_stub(4, 4, [-1] * 4, [1] * 4, (10, 20, 30, 40))
    stub.positions.write_lane(2, 177)
    assert stub.positions.read() == [10, 20, 177, 40] and stub.positions.positions == [10, 20, 177, 40]
    with expect_error(ValueError):
        stub.positions.write_lane(4, 1)


def test_batched_ple_lane_lookup_is_the_per_lane_lookup_in_one_read(expect_error, monkeypatch) -> None:
    """``lookup_tokens_lanes`` chains every lane from its own context and reads every row in ONE batch: the payload is
    the per-lane ``lookup_tokens`` payloads back to back and the contexts are theirs (the hash and the row read are
    stand-ins: the chaining and batching are what is pinned)."""

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


@pytest.mark.parametrize("active", ((1, 1, 1, 1), (1, 0, 1, 1)))
@pytest.mark.parametrize("positions", LANE_POSITIONS)
def test_single_row_lane_verify_inputs_drop_the_next_block_and_match_the_reference(fake, positions, active) -> None:
    """The draft rows' form: one real row per lane, the current KV block and compressed block alone (a lane's row
    never reaches the next block, which at the lane's last block is the next lane's first)."""

    lanes = len(positions)
    constants, lane_constants, lane_verify = _lane_qsa_constants(1, lanes)
    position_constants, chunk_constants, verify_constants = constants
    position_row = _u32_row(_pad32(positions, positions[0]))
    active_row = _u32_row(_pad32(active, 1))
    inactive_row = _u32_row(_pad32([1 - flag for flag in active], 0))
    lane_rows = fake.ttnn.gather(position_row, 3, lane_verify.lane_of_row)
    row_positions = fake.ttnn.add(lane_rows, lane_verify.draft_of_row)
    derived = qsa_module.derive_qsa_lane_verify_inputs(
        position_row,
        active_row,
        inactive_row,
        row_positions,
        position_constants,
        chunk_constants,
        verify_constants,
        lane_constants,
        lane_verify,
        single_row=True,
    )
    expected = qsa_module.emulate_qsa_lane_verify_inputs(
        positions, active, lanes=lanes, rows=1, allocated_compressed_blocks=BLOCKS, single_row=True
    )
    assert derived.single_row and derived.stage_b_select is None and derived.kv_row_start_next_lanes == ()
    assert len(derived.block_index_i32) == 1 and expected["stage_b_select"] is None
    assert derived.block_index_i32[0].torch_shards()[0].tolist() == expected["block_index_i32"][0].tolist()
    for name in ("row_keep_bits", "row_fill", "kv_read_indices"):
        assert torch.equal(getattr(derived, name).torch_shards()[0], expected[name]), name
    for name in ("stage_keep", "stage_a_select", "pool_select"):
        assert torch.equal(getattr(derived, name).torch_shards()[0].float(), expected[name].float()), name
    starts = [int(x.torch_shards()[0].reshape(-1)[0]) for x in derived.kv_row_start_lanes]
    assert starts == [u * CONTEXT + (positions[u] & ~31) if active[u] else lanes * CONTEXT for u in range(lanes)]
    # Every lane's one row lands on its own batch at row P_u % 32 of the current block.
    stage_a = derived.stage_a_select.torch_shards()[0].float()[0]
    for u in range(lanes):
        (landing,) = stage_a[u, :, u].nonzero().reshape(-1).tolist()
        assert landing == positions[u] % 32 and stage_a[:, :, u].sum() == 1.0
    derived.deallocate()


@pytest.mark.parametrize("rows", (4, 5))
def test_commit_verify_lanes_into_the_draft_state_leaves_the_alignment_windows_and_advances_by_one(
    expect_error, fake, rows
) -> None:
    lanes = 4
    module = _qsa_module(fake.ttnn)
    alignment_state, draft_state = module.allocate_lane_verify_state(lanes), module.allocate_lane_verify_state(lanes)
    gdn_lane_constants = gdn_module.Qwen38TTNNGDNLaneRowsConstants.allocate(
        "mesh", FakeContract(), lanes=lanes, rows=rows
    )
    torch.manual_seed(9)
    history, raw_rows = _bf16(lanes, 32, 128), _bf16(lanes, 32, 128)
    history[:, 3:] = 0
    accepted = [0, rows - 1, 2, 1][:lanes]
    for target in alignment_state.raw_history.locals:
        target[0] = history
    for target in alignment_state.raw_rows.locals:  # lane-major flat rows [1, 1, B*32, 128]
        target[0, 0] = raw_rows.reshape(lanes * 32, 128)
    for target in draft_state.raw_history.locals + draft_state.raw_rows.locals:
        target[...] = 5.0
    selectors = gdn_module.build_rows_selectors_lanes(
        _lane_column(accepted), _lane_column([1] * lanes), gdn_lane_constants
    )
    module.commit_verify_lanes(alignment_state, selectors, target=draft_state)
    for u in range(lanes):
        window = torch.cat([history[u, :3], raw_rows[u]])
        expected = torch.zeros(32, 128, dtype=torch.bfloat16)
        expected[:3] = window[accepted[u] + 1 : accepted[u] + 4]
        assert _same_zero(draft_state.raw_history.torch_shards()[0][0, u], expected), u
    assert torch.equal(alignment_state.raw_history.torch_shards()[0][0], history), "alignment histories touched"
    assert torch.equal(alignment_state.raw_rows.torch_shards()[0][0, 0], raw_rows.reshape(lanes * 32, 128)), "raw rows"
    assert torch.all(draft_state.raw_rows.torch_shards()[0] == 5.0)
    # The a = 0 advance per lane: [h1, h2, the lane's raw row 0].
    advance = gdn_module.build_rows_selectors_lanes(
        _lane_column([0] * lanes), _lane_column([1] * lanes), gdn_lane_constants
    )
    row0 = _bf16(lanes, 128)
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
        assert _same_zero(draft_state.raw_history.torch_shards()[0][0, u], advanced), u
    with expect_error(ValueError):  # allow-pytest.raises: the target must hold the same lanes
        module.commit_verify_lanes(alignment_state, selectors, target=module.allocate_lane_verify_state(2))


class _FakeLaneMoE:
    def __init__(self, *args, rows: int, **kwargs) -> None:
        self.rows = rows

    def release_owned_buffers(self) -> None:
        return None


def _draft_world(fake, monkeypatch, *, drafts: int, positions, accepted):
    """The lane verify/draft objects the lane draft body reads, over the real QSA lane module, constants, positions
    and selectors, with the step-6 stand-ins (embedding table, row-wise layer, vocabulary resolve) per lane."""

    world = step6._World()
    lanes, rows = len(positions), drafts + 1
    contract = FakeContract()
    qsa = _qsa_module(fake.ttnn)
    constants, lane_constants, lane_verify = _lane_qsa_constants(rows, lanes)
    position_constants, chunk_constants, verify_constants = constants
    attention_state = qsa.allocate_lane_state(lanes, kv_scratch_rows=mtp_lanes.kv_scratch_rows(CONTEXT))
    alignment_rows = qsa.allocate_lane_verify_state(lanes)
    torch.manual_seed(23)
    for target in alignment_rows.raw_history.locals:
        target[0, :, :3] = _bf16(lanes, 3, 128)
    for target in alignment_rows.raw_rows.locals:  # lane-major flat rows [1, 1, B*32, 128]
        target[0, 0] = _bf16(lanes * 32, 128)
    gdn_lane_constants = gdn_module.Qwen38TTNNGDNLaneRowsConstants.allocate("mesh", contract, lanes=lanes, rows=rows)
    lane_accept_constants = mtp_lanes.Qwen38TTNNMTPLaneConstants.build("mesh", contract, lanes=lanes, drafts=drafts)
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
    residual = step6._sharded(_bf16(1, 4, lanes, step6.WIDTH * TP))
    final_mixer = SimpleNamespace(rows=lambda residual_rows, flat_views=False: world.final(residual_rows))
    alignment = mtp_lanes.Qwen38TTNNMTPLaneAlignment(
        layer, step6._Mixer(world.mix), final_mixer, alignment_layer_state, residual
    )
    owner = object()
    verify = SimpleNamespace(
        lanes=lanes,
        drafts=drafts,
        rows=rows,
        alignment=alignment,
        qsa_chunk_constants=chunk_constants,
        qsa_lane_constants=lane_constants,
        gdn_lane_constants=gdn_lane_constants,
        constants=lane_accept_constants,
        positions=mtp_lanes.Qwen38TTNNMTPLanePositions.allocate("mesh", contract, positions, lanes=lanes),
        active_row=_u32_row(_pad32([1] * lanes, 1)),
        inactive_row=_u32_row(_pad32([0] * lanes, 0)),
        accepted_lanes=_lane_column(accepted),
        active_lanes=_lane_column([1] * lanes),
        token_row=_replicated(torch.full((1, 1, 1, 32), -1.0), FP32),
        draft_lanes=step6._lanes([]),
        _owner=owner,
    )

    def rows_chunk(index_rows, block_start_rows):
        world.rope_requests.append(index_rows.torch_shards()[0].reshape(-1).tolist())
        tile = _replicated(torch.zeros(1, 1, 32, 64, dtype=torch.bfloat16), BF16)
        return SimpleNamespace(cos=tile, sin=tile, block_start_cos=tile, block_start_sin=tile, deallocate=lambda: None)

    embedding = SimpleNamespace(
        embed_device_token_rows=world.embed,
        validate_token_row=lambda token_row, label="": embedding_module._validate_token_row(
            token_row, mesh_contract=contract, label=label
        ),
    )
    model = SimpleNamespace(
        _state_owner=owner,
        poisoned=False,
        mesh_device="mesh",
        mesh_contract=contract,
        allocated_context=CONTEXT,
        model_io=SimpleNamespace(embedding=embedding),
        rope_table=SimpleNamespace(rows_chunk=rows_chunk),
        qsa_position_constants=position_constants,
    )

    def poisoned(operation, processed, error):
        raise error

    model._mark_poisoned = poisoned
    seen: list[dict] = []

    def forward_layer_verify_lanes(
        layer_, residual_, layer_state, geometry, *, prepared_ple_rows, rope_rows, qsa_inputs, selectors
    ):
        assert layer_ is layer and layer_state.attention_rows is not alignment_rows and geometry.rows == 1
        assert layer_state.attention_state is attention_state and prepared_ple_rows is None and selectors is None
        assert qsa_inputs.single_row and geometry.qsa_lane_verify_constants.rows == 1
        assert residual_.shape == (1, 4, 32, step6.WIDTH) and residual_.dtype is BF16
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
        return step6._lanes(ids)

    monkeypatch.setattr(mtp_lanes, "_validate_lane_verify_state", lambda model_, verify_: None)
    monkeypatch.setattr(mtp_lanes, "_forward_layer_verify_lanes", forward_layer_verify_lanes)
    monkeypatch.setattr(mtp_lanes, "_resolve_rows", resolve_rows)
    monkeypatch.setattr(mtp_lanes, "Qwen38TTNNMoE", _FakeLaneMoE)
    return world, model, verify, seen


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
        # assembly concatenates the lane blocks alone (the 2026-09-26 64K B=8 k=3 row poisoned the model here)
        (3, (5, 31, 32, 100, 29, 63, 60, 127), (0, 2, 1, 3, 3, 0, 2, 1)),
    ),
)
def test_forward_draft_lanes_is_the_eager_row_chain_per_lane_and_assembles_the_next_pass(
    fake, monkeypatch, drafts, positions, accepted
):
    lanes, rows = len(positions), drafts + 1
    accepted = [min(a, drafts) for a in accepted]
    world, model, verify, seen = _draft_world(fake, monkeypatch, drafts=drafts, positions=positions, accepted=accepted)
    assert (verify.constants.sentinel_tail is None) == (lanes * rows == 32)
    draft = mtp_lanes.allocate_lane_draft_state(model, verify)
    assert draft.rows == 1 and draft.qsa_verify_constants.rows == 1 and draft.qsa_lane_verify_constants.rows == 1
    assert draft.layer_state.moe.rows == lanes and draft.layer_state.moe_input.shape == (1, 1, lanes, step6.WIDTH)
    assert len(draft.step_offset_rows) == drafts - 1
    next_tokens, first_drafts = [17 + u for u in range(lanes)], [41 + 3 * u for u in range(lanes)]
    argmaxes, alignment_row = list(range(32)), list(range(100, 132))
    readback = mtp_v2.Qwen38TTNNVerifyOutput(
        step6._lanes(
            [float(a) for a in accepted] + next_tokens + first_drafts + argmaxes + alignment_row,
            width=mtp_lanes.lane_readback_width(lanes),
        )
    )
    alignment_rows = verify.alignment.layer_state.attention_rows
    history_before = alignment_rows.raw_history.torch_shards()[0].clone()
    raw_rows_before = alignment_rows.raw_rows.torch_shards()[0].clone()
    mtp_lanes.forward_draft_lanes(model, verify, draft, readback)
    pass_readback, tokens = mtp_lanes.read_lane_pass_row(verify, draft)
    residual_all = step6._cat(verify.alignment.residual)
    expected_tokens = []
    for u in range(lanes):
        lane_residual = step6._sharded(residual_all[:, :, u : u + 1])
        ids = step6._eager_rows(world, lane_residual, [first_drafts[u]], drafts - 1)
        expected_tokens.append((next_tokens[u], first_drafts[u], *ids))
    assert tokens == tuple(expected_tokens), (drafts, positions, accepted)
    flat = [t for block in tokens for t in block]
    host_row = embedding_module.Qwen38TTNNTokenEmbedding.host_verify_token_rows(flat)
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
        assert entry["starts"] == [u * CONTEXT + ((p + i) & ~31) for u, p in enumerate(positions)], i
        assert entry["remainders"] == [float((p + i) % 32) for p in positions], i
    assert [req[:lanes] for req in world.rope_requests] == [[p + i for p in positions] for i in range(drafts - 1)]
    # The alignment windows are untouched; lane u's draft history started as window_u[a_u + 1 : a_u + 4] and advanced
    # once per row after the first (the stub layer writes no raw rows: zeros shift in).
    assert torch.equal(alignment_rows.raw_history.torch_shards()[0], history_before)
    assert torch.equal(alignment_rows.raw_rows.torch_shards()[0], raw_rows_before)
    for u in range(lanes):
        window = torch.cat([history_before[0, u, :3], raw_rows_before[0, 0, u * 32 : (u + 1) * 32]])
        derived = torch.zeros(32, 128, dtype=torch.bfloat16)
        derived[:3] = window[accepted[u] + 1 : accepted[u] + 4]
        advances = drafts - 2
        expected_history = torch.zeros(32, 128, dtype=torch.bfloat16)
        expected_history[: 3 - advances] = derived[advances:3]
        assert _same_zero(draft.qsa_state.raw_history.torch_shards()[0][0, u], expected_history), u
    assert verify.positions.read() == list(positions)  # the verify body owns the positions
    mtp_lanes.release_lane_draft_state(model, verify, draft)


def test_lane_chain_steps_commit_then_ple_rows_then_verify_and_draft_and_reads_one_pass_row(
    expect_error, monkeypatch
) -> None:
    lanes, drafts = 4, 4
    rows = drafts + 1
    calls: list[str] = []
    owner = object()
    positions = mtp_lanes.Qwen38TTNNMTPLanePositions.__new__(mtp_lanes.Qwen38TTNNMTPLanePositions)
    positions.positions, positions.lanes = [10, 20, 30, 40], lanes
    verify = SimpleNamespace(lanes=lanes, drafts=drafts, rows=rows, positions=positions, _owner=owner)
    draft = SimpleNamespace(_owner=owner)
    model = SimpleNamespace(_state_owner=owner, allocated_context=CONTEXT)
    accepted = (1, 4, 0, 2)
    argmaxes = tuple(tuple(100 * u + j for j in range(rows)) for u in range(lanes))
    readback = mtp_lanes.Qwen38TTNNLaneVerifyReadback(
        accepted=accepted,
        next_token=tuple(argmaxes[u][accepted[u]] for u in range(lanes)),
        first_draft=tuple(7 + u for u in range(lanes)),
        argmaxes=argmaxes,
        alignment_argmaxes=argmaxes,
    )
    assembled = tuple(
        (readback.next_token[u], readback.first_draft[u], *range(50 + u, 50 + u + drafts - 1)) for u in range(lanes)
    )
    monkeypatch.setattr(mtp_lanes, "_validate_lane_draft_state", lambda model_, verify_, draft_: None)
    monkeypatch.setattr(
        mtp_lanes, "read_lane_pass_row", lambda verify_, draft_: (calls.append("readback"), (readback, assembled))[1]
    )
    monkeypatch.setattr(
        mtp_lanes, "write_lane_ple_rows", lambda model_, verify_, tokens: calls.append(f"ple_rows:{tokens[0][0]}")
    )
    monkeypatch.setattr(
        mtp_lanes, "write_lane_verify_inputs", lambda model_, verify_, tokens: calls.append("host_inputs")
    )
    committed_counts: list[list[int]] = []

    def commit_host(model_, verify_, counts):
        committed_counts.append(list(counts))
        positions.positions = [p + c for p, c in zip(positions.positions, counts)]

    monkeypatch.setattr(mtp_lanes, "commit_lane_verify_host", commit_host)
    traces = mtp_lanes.Qwen38TTNNMTPLaneTraces(verify_first=11, commit=12, draft=13)
    names = {11: "verify", 12: "commit", 13: "draft"}
    chain = mtp_lanes.Qwen38TTNNMTPLaneChain(
        model,
        verify,
        draft,
        traces,
        replay=lambda tid: calls.append(f"replay:{names[tid]}"),
        enqueue=lambda tid: calls.append(f"enqueue:{names[tid]}"),
    )
    record = chain.bootstrap([[1, 0, 0, 0, 0]] * lanes)
    assert calls == ["host_inputs", "enqueue:verify", "enqueue:draft", "readback"]
    assert record.accepted == accepted and record.committed == tuple(
        argmaxes[u][: accepted[u] + 1] for u in range(lanes)
    )
    assert (
        committed_counts == [[2, 5, 1, 3]] and record.positions == (10, 20, 30, 40) and chain.next_tokens == assembled
    )
    assert set(record.segments_ns) == {"host_inputs", "verify_enqueue", "draft_enqueue", "readback"}
    calls.clear()
    record = chain.step()
    assert calls == [f"enqueue:commit", f"ple_rows:{assembled[0][0]}", "enqueue:verify", "enqueue:draft", "readback"]
    assert record.index == 1 and record.tokens == assembled and record.positions == (12, 25, 31, 43)
    chain.enqueue = None  # the blocking (segment-measuring) form
    calls.clear()
    record = chain.step()
    assert calls[:2] == ["replay:commit", f"ple_rows:{assembled[0][0]}"] and calls[2:] == [
        "replay:verify",
        "replay:draft",
        "readback",
    ]
    assert set(record.segments_ns) == {"commit_replay", "ple_rows", "verify_replay", "draft_replay", "readback"}
    with expect_error(RuntimeError):  # allow-pytest.raises: a chain that ran passes cannot bootstrap again
        chain.bootstrap([[1, 0, 0, 0, 0]] * lanes)
    # A chain continuing an eager pass takes its assembled tokens and refuses the bootstrap.
    continued = mtp_lanes.Qwen38TTNNMTPLaneChain(
        model, verify, draft, traces, replay=lambda tid: None, next_tokens=assembled
    )
    with expect_error(RuntimeError):  # allow-pytest.raises: seeded chains step
        continued.bootstrap([[1, 0, 0, 0, 0]] * lanes)
    with expect_error(ValueError):  # allow-pytest.raises: the seed must be B blocks of R tokens
        mtp_lanes.Qwen38TTNNMTPLaneChain(model, verify, draft, traces, replay=lambda tid: None, next_tokens=[[1, 2]])


def test_lane_draft_body_moves_ids_by_copy_and_orders_history_rows_and_assembly() -> None:
    tree = ast.parse(MTP_LANES_SOURCE.read_text(encoding="utf-8"))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    body = _segment(MTP_LANES_SOURCE, functions["forward_draft_lanes"])
    history = _segment(MTP_LANES_SOURCE, functions["forward_draft_history_lanes"])
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
    step = _segment(MTP_LANES_SOURCE, classes["Qwen38TTNNMTPLaneChain"])
    order = ('"commit_enqueue"', '"ple_rows"', "return self._finish_pass(tokens, segments)")
    indices = [step.index(text) for text in order]
    assert indices == sorted(indices), order
    finish = _segment(
        MTP_LANES_SOURCE,
        next(
            n
            for n in classes["Qwen38TTNNMTPLaneChain"].body
            if isinstance(n, ast.FunctionDef) and n.name == "_finish_pass"
        ),
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


def _lifecycle_world(fake, monkeypatch, gdn_weights, positions=(10, 20, 30, 40)):
    """A 4-lane MTP verify state of one QSA layer, one GDN layer with the PLE (checkpoint layer 1) and the alignment
    layer's lane state, over the real lane modules on the fake."""

    _install_admission_ops(fake)
    lanes, rows = 4, 5
    scratch = mtp_lanes.kv_scratch_rows(CONTEXT)
    qsa = _qsa_module(fake.ttnn)
    gdn = _gdn_module(gdn_weights[1])
    ple = _ple_module()
    moe = SimpleNamespace(rows=lanes * rows)
    gdn_constants = gdn.allocate_rows_constants(rows)
    qsa_layer = mtp_lanes.Qwen38TTNNMTPLaneLayerState(
        "backbone",
        0,
        qsa.allocate_lane_state(lanes, kv_scratch_rows=scratch),
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
        qsa.allocate_lane_state(lanes, kv_scratch_rows=scratch),
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
        positions=mtp_lanes.Qwen38TTNNMTPLanePositions.allocate("mesh", FakeContract(), list(positions), lanes=lanes),
        pass_contexts=[(None,) * (rows + 1) for _ in range(lanes)],
    )
    model = SimpleNamespace(mesh_device="mesh", allocated_context=CONTEXT)
    monkeypatch.setattr(mtp_lanes, "_validate_lane_verify_state", lambda model_, verify_: None)
    return model, verify


def _lane_tensors(verify, lane: int) -> dict[str, list[torch.Tensor]]:
    """Lane ``lane``'s slices of every family as per-device torch tensors (the family layouts of the lane states)."""

    qsa_layer, gdn_layer = verify.layers
    alignment = verify.alignment.layer_state
    context = CONTEXT
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
            x[0, 0, lane * CONTEXT : lane * CONTEXT + position] = _bf16(position, 512)
        blocks = _bf16(1, *state.compressed_index_cache.locals[0].shape[2:])
        for x in state.compressed_index_cache.locals:
            x[lane] = blocks
    for rows_state in (qsa_layer.attention_rows, alignment.attention_rows):
        history = _bf16(3, 128)
        for x in rows_state.raw_history.locals:
            x[0, lane] = 0
            x[0, lane, :3] = history
    for x in gdn_layer.attention_state.recurrent.locals:
        x[lane] = torch.randn(*x.shape[1:], dtype=torch.float32)
    for x in gdn_layer.attention_rows.history.locals:
        x[0, lane] = 0
        x[0, lane, :3] = _bf16(3, x.shape[3])
    for x in gdn_layer.ple.history.locals:
        x[lane] = _bf16(*x.shape[1:])
    contexts = list(gdn_layer.ple.token_contexts)
    contexts[lane] = context
    gdn_layer.ple.token_contexts = tuple(contexts)
    verify.positions.write_lane(lane, position)


@pytest.mark.parametrize("lane,target,position", ((2, 2, 30), (2, 0, 30), (3, 1, 100), (0, 3, 1)))
def test_lane_image_round_trip_restores_every_family_bitwise_and_leaves_the_other_lanes(
    expect_error, fake, monkeypatch, gdn_weights, lane, target, position
):
    model, verify = _lifecycle_world(fake, monkeypatch, gdn_weights)
    _seed_lane(verify, lane, 300 + lane, position, (7, 8 + lane))
    for other in range(4):
        if other != lane:
            _seed_lane(verify, other, 400 + other, 5 + other, (1, other))
    before = _lane_tensors(verify, lane)
    others_before = {u: _lane_tensors(verify, u) for u in range(4) if u not in (lane, target)}
    image = mtp_lanes.evict_lane(model, verify, lane, accepted=-1)
    assert image.position == position and image.ple_context == (7, 8 + lane) and image.accepted == -1
    assert (
        image.kv_rows == mtp_lanes.kv_image_rows(position, CONTEXT)
        and image.kv_rows % mtp_lanes.kv_scratch_rows(CONTEXT) == 0
    )
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

    lanes, drafts = 4, 4
    rows = drafts + 1
    calls: list = []
    owner = object()
    positions = mtp_lanes.Qwen38TTNNMTPLanePositions.__new__(mtp_lanes.Qwen38TTNNMTPLanePositions)
    positions.positions, positions.lanes = [10, 20, 30, 40], lanes
    verify = SimpleNamespace(lanes=lanes, drafts=drafts, rows=rows, positions=positions, _owner=owner)
    draft = SimpleNamespace(_owner=owner)
    model = SimpleNamespace(_state_owner=owner, allocated_context=CONTEXT)
    accepted = (1, 4, 0, 2)
    argmaxes = tuple(tuple(100 * u + j for j in range(rows)) for u in range(lanes))
    readback = mtp_lanes.Qwen38TTNNLaneVerifyReadback(
        accepted=accepted,
        next_token=tuple(argmaxes[u][accepted[u]] for u in range(lanes)),
        first_draft=tuple(7 + u for u in range(lanes)),
        argmaxes=argmaxes,
        alignment_argmaxes=argmaxes,
    )
    assembled = tuple(
        (readback.next_token[u], readback.first_draft[u], *range(50 + u, 50 + u + drafts - 1)) for u in range(lanes)
    )
    monkeypatch.setattr(mtp_lanes, "_validate_lane_draft_state", lambda model_, verify_, draft_: None)
    monkeypatch.setattr(mtp_lanes, "read_lane_pass_row", lambda verify_, draft_: (readback, assembled))
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
    counts_seen: list[list[int]] = []

    def commit_host(model_, verify_, counts):
        counts_seen.append(list(counts))
        positions.positions = [p + c for p, c in zip(positions.positions, counts)]

    monkeypatch.setattr(mtp_lanes, "commit_lane_verify_host", commit_host)
    chain = mtp_lanes.Qwen38TTNNMTPLaneChain(
        model, verify, draft, mtp_lanes.Qwen38TTNNMTPLaneTraces(1, 2, 3), replay=lambda tid: None, next_tokens=assembled
    )
    chain.set_active([1, 0, 1, 1])  # park lane 1: the commit mask stays the previous pass's (all active)
    assert calls[-1] == ("active", [1, 0, 1, 1], [1, 1, 1, 1])
    record = chain.step()
    # The step's commit used the old mask; the refresh right behind it makes the next commit skip the parked lane.
    assert calls[-1] == ("active", [1, 0, 1, 1], [1, 0, 1, 1]) and "mask_refresh" in record.segments_ns
    assert counts_seen[-1] == [2, 0, 1, 3] and record.committed[1] == () and positions.positions == [12, 20, 31, 43]
    calls.clear()
    chain.step()
    assert not any(call[0] == "active" for call in calls)  # no change: nothing rewritten
    chain.set_active([1, 1, 1, 1])  # unpark: the commit mask is the parked pass's mask (lane 1 commits nothing)
    assert calls[-1] == ("active", [1, 1, 1, 1], [1, 0, 1, 1])
    held = [list(b) for b in assembled]
    held[1] = [5, 6, 7, 8, 9]
    chain.set_next_tokens(held)
    assert calls[-1] == ("tokens", held) and chain.next_tokens[1] == (5, 6, 7, 8, 9)
    record = chain.step()
    assert record.tokens[1] == (5, 6, 7, 8, 9) and counts_seen[-1] == [2, 5, 1, 3]
    with expect_error(ValueError):  # allow-pytest.raises: the mask needs B 0/1 flags
        chain.set_active([1, 2, 1, 1])
