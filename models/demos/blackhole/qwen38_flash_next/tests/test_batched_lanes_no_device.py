# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Batched decode lanes without a device: every lane result bitwise against the 1-row path on that lane's stream.

The lane contracts, the QSA lane derive, the RoPE rows and the greedy resolve run on a small torch-backed ttnn
modeling their exact UINT32/FP32 op surface. The GDN/PLE lanes are in test_batched_lanes_gdn_no_device, the GR rows
in test_gr_rows_no_device.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.tests.helpers.torch_ttnn import (
    BF16,
    FP32,
    I32,
    ROW_MAJOR,
    TILE,
    TP,
    U32,
    FakeContract,
    FakeTensor,
    function_calls,
    make_lanes_ttnn,
    patch_ttnn,
    use_tile_padding,
)
from models.demos.blackhole.qwen38_flash_next.ttnn import contracts as contracts_module
from models.demos.blackhole.qwen38_flash_next.ttnn import embedding as embedding_module
from models.demos.blackhole.qwen38_flash_next.ttnn import gdn as gdn_module
from models.demos.blackhole.qwen38_flash_next.ttnn import model as model_module
from models.demos.blackhole.qwen38_flash_next.ttnn import moe as moe_module
from models.demos.blackhole.qwen38_flash_next.ttnn import ple as ple_module
from models.demos.blackhole.qwen38_flash_next.ttnn import qsa as qsa_module
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import (
    GDN_RESIDUE_CLASSES,
    MAX_LANES,
    POSITION_INDEX_ROW_SHAPE,
    Qwen38TTNNDevicePositionRow,
    admission_wait_steps,
    require_lane_count,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.embedding import (
    LOCAL_VOCAB_SIZE,
    TOKEN_ROW_SHAPE,
    Qwen38GreedyCandidates,
    Qwen38TTNNTokenRowConstants,
    resolve_greedy_lanes_on_device,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.qsa import (
    COMPRESS_RATIO,
    SPARSE_INDEX_CAPACITY,
    TOKEN_BUDGET,
    Qwen38TTNNQSAChunkConstants,
    Qwen38TTNNQSALaneConstants,
    Qwen38TTNNQSAPositionConstants,
    derive_qsa_lane_inputs,
    emulate_qsa_lane_inputs,
    emulate_qsa_position_inputs,
    lane_kv_offsets,
    qsa_lane_constant_rows,
    qsa_selection_geometry,
)

LANE_BLOCKS = 512  # a 2,048-token allocation: the smallest legal QSA cache, 512 compressed blocks
LANE_CONTEXT = LANE_BLOCKS * COMPRESS_RATIO
# 32 lane positions of one residue class (3 mod 4) across block, tile and top-k boundaries.
LANE_POSITIONS = tuple(3 + 4 * k for k in (0, 1, 2, 7, 8, 15, 16, 31, 32, 63, 64, 100, 127, 128, 255, 256))
LANE_POSITIONS += tuple(3 + 4 * k for k in (300, 340, 400, 450, 500, 505, 506, 507, 508, 509, 510, 511, 5, 9, 13, 17))
assert len(LANE_POSITIONS) == MAX_LANES and all(p % 4 == 3 and p < LANE_CONTEXT for p in LANE_POSITIONS)


# --------------------------------------------------------------------------- the lane fake


@pytest.fixture
def lane_fake(monkeypatch):
    # the fused PLE lane body serves by default and the fake ttnn runs the chains only: switch it off here
    monkeypatch.setenv("QWEN38_FUSED_OFF", "ple")
    use_tile_padding(monkeypatch)
    fake_ttnn = make_lanes_ttnn()
    patch_ttnn(monkeypatch, fake_ttnn, contracts_module, qsa_module, model_module, embedding_module)
    return fake_ttnn


def _same_everywhere(tensor: FakeTensor) -> torch.Tensor:
    first = tensor.torch_shards()[0]
    for other in tensor.torch_shards()[1:]:
        assert torch.equal(first, other)
    return first


# --------------------------------------------------------------------------- lane count and admission


def test_lane_count_contract_names_the_range_and_the_value(expect_error) -> None:
    assert require_lane_count(1) == 1 and require_lane_count(MAX_LANES) == MAX_LANES == 32
    for bad in (0, 33, -1, True, 2.0, "8"):
        with pytest.raises(ValueError) as caught:  # allow-pytest.raises: pure contract test
            require_lane_count(bad, label="lanes under test")
        assert "lanes under test" in str(caught.value)
    with expect_error(RuntimeError, match=r"\[1,32\], got 40"):  # allow-pytest.raises: pure contract test
        require_lane_count(40, error_type=RuntimeError)


def test_admission_waits_until_the_resident_residue_is_the_lanes(expect_error) -> None:
    assert GDN_RESIDUE_CLASSES == gdn_module.CONV_KERNEL_SIZE == 4
    for resident_residue in range(4):
        for position in range(0, 40):
            wait = admission_wait_steps(resident_residue, position)
            assert 0 <= wait < 4
            assert (resident_residue + wait) % 4 == position % 4
    for bad in ((4, 0), (-1, 0), (0, -1), (0.0, 1), (True, 1)):
        with expect_error(ValueError):  # allow-pytest.raises: pure contract test
            admission_wait_steps(*bad)


def test_position_row_holds_a_position_per_lane_and_advances_every_lane(expect_error, lane_fake) -> None:
    row = Qwen38TTNNDevicePositionRow.allocate("mesh", FakeContract(), [4, 8, 12, 4], lanes=4)
    assert row.lanes == 4 and row.residue == 0 and row.positions == [4, 8, 12, 4] + [0] * 28
    assert row.read() == row.positions and row.row.shape == POSITION_INDEX_ROW_SHAPE and row.row.dtype is U32
    row.admit(2, 20)
    assert row.read()[2] == 20
    with expect_error(ValueError, match="residue 1, expected the row's residue 0; admit it 1 steps later"):
        row.admit(2, 21)
    with expect_error(ValueError, match=r"admitted lane must be in \[0,4\), got 4"):
        row.admit(4, 20)
    row.advance()
    assert row.positions[:4] == [5, 9, 21, 5] and row.read() == row.positions and row.residue == 1
    # A replayed body advanced the device row in-trace: the mirror follows without a device op or a host write
    # (the fake device stays; the residue the next admission is checked against is the mirror's).
    device_before = row.read()
    row.advance_replayed()
    assert row.positions[:4] == [6, 10, 22, 6] and row.residue == 2 and row.read() == device_before
    with expect_error(ValueError, match="residue 1, expected the row's residue 2; admit it 3 steps later"):
        row.admit(3, 9)
    row.admit(3, 10)  # the host write comes from the mirror
    assert row.positions[:4] == [6, 10, 22, 10] and row.read() == row.positions
    index = row.index_row()
    assert index is not row.row and _same_everywhere(index).reshape(-1).tolist() == row.positions
    block_start = row.block_start_index_row(index)
    assert _same_everywhere(block_start).reshape(-1).tolist() == [p & ~3 for p in row.positions]
    row.reset([7, 3, 11, 3])
    assert row.residue == 3 and row.read() == [7, 3, 11, 3] + [3] * 28
    with expect_error(ValueError, match="lane 1 position 6 has residue 2, expected the row's residue 3"):
        row.reset([7, 6, 11, 3])
    with expect_error(ValueError, match="got 3, expected 4"):
        row.reset([7, 3, 11])
    row.validate()
    with expect_error(ValueError, match=r"position row lanes must be in \[1,32\], got 33"):
        Qwen38TTNNDevicePositionRow.allocate("mesh", FakeContract(), list(range(0, 132, 4)), lanes=33)
    with expect_error(ValueError, match="lane 1 position 8 has residue 0, expected the row's residue 3"):
        Qwen38TTNNDevicePositionRow.allocate("mesh", FakeContract(), [2051, 8], lanes=2)


# --------------------------------------------------------------------------- QSA lane derive


def test_lane_inputs_emulation_is_the_per_position_emulation_lane_by_lane(expect_error) -> None:
    positions = list(LANE_POSITIONS[:16]) + [0, 1, 2, 5, 6, 30, 31, 32, 33, 34, 2044, 2045, 2046, 2047, 100, 1000]
    for count in (1, 3, 8, 32):
        lanes = emulate_qsa_lane_inputs(positions, allocated_compressed_blocks=LANE_BLOCKS, lanes=count)
        offsets = lane_kv_offsets(count, LANE_CONTEXT)
        assert offsets == [lane * LANE_CONTEXT if lane < count else 0 for lane in range(32)]
        assert lanes["kv_row_start_row"].shape == POSITION_INDEX_ROW_SHAPE
        assert lanes["block_index_i32"].shape == (count,)
        for name in ("kv_hit_tiles", "ring_hit_tiles"):
            assert lanes[name].shape == (1, count, 32, 32) and lanes[name].dtype == torch.bfloat16
        for name in ("kv_keep_col", "ring_keep_col"):
            assert lanes[name].shape == (1, count, 32, 1) and lanes[name].dtype == torch.bfloat16
        for lane, position in enumerate(positions):
            single = emulate_qsa_position_inputs(position, allocated_compressed_blocks=LANE_BLOCKS)
            assert int(lanes["kv_row_start_row"][0, 0, 0, lane]) == int(single["kv_block_start"]) + offsets[lane]
            for name in ("indexer_neg_mask", "row_keep_bits"):
                assert torch.equal(lanes[name][:, :, lane : lane + 1], single[name]), (name, lane)
            # The tail slots [lo, lo + tail_count) carry the lane offset; kept slots and sentinels are untouched.
            fill, want = lanes["row_fill"][0, 0, lane], single["row_fill"].reshape(-1)
            geometry = qsa_selection_geometry(position + 1)
            slots, lo = torch.arange(SPARSE_INDEX_CAPACITY), geometry.complete_token_count
            tail = (slots >= lo) & (slots < lo + geometry.tail_count)
            assert torch.equal(fill[tail], want[tail] + offsets[lane]) and torch.equal(fill[~tail], want[~tail]), lane
            if lane < count:
                assert int(lanes["block_index_i32"][lane]) == int(single["block_index_i32"][0])
                for tiles, col, one_hot in (
                    ("kv_hit_tiles", "kv_keep_col", "kv_row_hit"),
                    ("ring_hit_tiles", "ring_keep_col", "ring_hit"),
                ):
                    tile = lanes[tiles][0, lane]
                    assert torch.equal(tile[:, lane : lane + 1], single[one_hot][0, 0]), (tiles, lane)
                    assert torch.count_nonzero(tile.float()) == 1 and tile[:, lane].sum() == 1.0
                    assert torch.equal(
                        lanes[col][0, lane], torch.tensor(1.0, dtype=torch.bfloat16) - single[one_hot][0, 0]
                    )
    with expect_error(ValueError, match="needs 32 lane positions, got 3"):  # allow-pytest.raises: contract
        emulate_qsa_lane_inputs([0, 1, 2], allocated_compressed_blocks=LANE_BLOCKS, lanes=1)


def test_lane_constant_rows_carry_the_offsets_and_the_lane_indicator(expect_error) -> None:
    host = qsa_lane_constant_rows(4, LANE_CONTEXT)
    assert host["arange32_tile"].tolist() == [[[[r] * 32 for r in range(32)]]]
    assert host["lane_indicator"].shape == (1, 4, 32, 32)
    for lane in range(4):
        assert torch.equal(host["lane_indicator"][0, lane].sum(dim=1), torch.ones(32))
        assert torch.equal(host["lane_indicator"][0, lane][:, lane], torch.ones(32))
    assert host["kv_offsets_row"].reshape(-1).tolist() == [LANE_CONTEXT * u for u in range(4)] + [0] * 28
    assert torch.equal(
        host["block_offsets_lanes"][0, 0, 3], torch.arange(4).repeat(TOKEN_BUDGET // 4) + 3 * LANE_CONTEXT
    )
    assert torch.equal(host["block_offsets_lanes"][0, 0, 9], torch.arange(4).repeat(TOKEN_BUDGET // 4))
    assert torch.equal(host["arange_slots_lanes"][0, 0, 2], torch.arange(SPARSE_INDEX_CAPACITY) + 2 * LANE_CONTEXT)
    for bad in ((0, LANE_CONTEXT), (33, LANE_CONTEXT), (4, 100)):
        with expect_error(ValueError):  # allow-pytest.raises: pure contract test
            lane_kv_offsets(*bad)


def _lane_constants(contract, lanes: int = MAX_LANES):
    constants = Qwen38TTNNQSAPositionConstants.build("mesh", contract, LANE_BLOCKS)
    chunk = Qwen38TTNNQSAChunkConstants.build("mesh", contract, LANE_BLOCKS)
    lane_constants = Qwen38TTNNQSALaneConstants.build("mesh", contract, lanes=lanes, allocated_context=LANE_CONTEXT)
    return constants, chunk, lane_constants


def _assert_lane_inputs_match(inputs, expected: dict[str, torch.Tensor]) -> None:
    for lane, scalar in enumerate(inputs.kv_row_start_lanes):
        assert _same_everywhere(scalar).tolist() == [[[[int(expected["kv_row_start_row"][0, 0, 0, lane])]]]], lane
    for name, want in expected.items():
        got = _same_everywhere(getattr(inputs, name))
        assert tuple(got.shape) == tuple(want.shape), (name, tuple(got.shape), tuple(want.shape))
        if want.is_floating_point():
            assert got.dtype == want.dtype and torch.equal(got.view(torch.int16), want.view(torch.int16)), name
        else:
            assert torch.equal(got.to(torch.int64), want.to(torch.int64)), name


@pytest.mark.parametrize("count", [1, 2, 8, 32])
@pytest.mark.parametrize(
    "positions",
    [
        LANE_POSITIONS,
        tuple(range(32)),  # every residue: the derive itself is residue-agnostic
        tuple(2047 - k for k in range(32)),  # the top of the allocation, every top-k tail length
    ],
)
def test_lane_derive_on_the_fake_equals_the_emulation(expect_error, lane_fake, positions, count: int) -> None:
    contract = FakeContract()
    constants, chunk, lanes = _lane_constants(contract, count)
    assert lanes.lanes == count and lanes.allocated_context == LANE_CONTEXT and lanes.indexer_form == "wide"
    assert _same_everywhere(lanes.arange32_tile).tolist() == [[[[r] * 32 for r in range(32)]]]
    assert lanes.compressed_rows_memory_config == (
        "sharded",
        (32, 128),
        ("cores", count, ("core", 8, -(-count // 8))),
        "HEIGHT",
    )
    row = lane_fake.from_torch(
        torch.tensor(positions, dtype=torch.int64).reshape(POSITION_INDEX_ROW_SHAPE),
        dtype=U32,
        layout=ROW_MAJOR,
        mesh_mapper="replicate",
    )
    inputs = derive_qsa_lane_inputs(row, constants, chunk, lanes)
    _assert_lane_inputs_match(
        inputs, emulate_qsa_lane_inputs(positions, allocated_compressed_blocks=LANE_BLOCKS, lanes=count)
    )
    assert inputs.block_index_i32.dtype is I32 and inputs.kv_hit_tiles.layout == TILE
    assert inputs.kv_keep_col.shape == (1, count, 32, 1) and len(inputs.kv_row_start_lanes) == count
    assert inputs.indexer_neg_mask.layout == ROW_MAJOR and inputs.row_fill.shape == (1, 1, 32, SPARSE_INDEX_CAPACITY)
    inputs.deallocate()
    assert row.alive  # the column form is a view of the resident row: the row is never released by the derive
    with expect_error(ValueError, match="indexer form"):  # allow-pytest.raises: pure contract test
        Qwen38TTNNQSALaneConstants.build(
            "mesh", contract, lanes=count, allocated_context=LANE_CONTEXT, indexer_form="x"
        )


def test_lane_derive_from_the_position_row_follows_admission_and_advance(expect_error, lane_fake) -> None:
    contract = FakeContract()
    constants, chunk, lanes = _lane_constants(contract, 4)
    row = Qwen38TTNNDevicePositionRow.allocate("mesh", contract, [3, 7, 2047, 31], lanes=4)
    for step in range(3):
        inputs = derive_qsa_lane_inputs(row.row, constants, chunk, lanes)
        _assert_lane_inputs_match(
            inputs, emulate_qsa_lane_inputs(row.positions, allocated_compressed_blocks=LANE_BLOCKS, lanes=4)
        )
        inputs.deallocate()
        if step == 1:
            row.admit(3, 100 + row.residue - 100 % 4)  # the residue-aligned re-admission of lane 3
        row.advance()
    assert row.positions[:4] == [6, 10, 2050, 102] and row.read() == row.positions
    with expect_error(RuntimeError, match=r"QSA position row shape must be \[1, 1, 1, 32\]"):
        derive_qsa_lane_inputs(
            lane_fake.from_torch(torch.zeros(1, 1, 1, 1), dtype=U32, layout=ROW_MAJOR, mesh_mapper="replicate"),
            constants,
            chunk,
            lanes,
        )


# --------------------------------------------------------------------------- RoPE rows per lane


def test_rope_rows_chunk_reads_each_lanes_position_and_block_start(lane_fake) -> None:
    torch.manual_seed(3)
    context = 256
    cos_host, sin_host = (torch.randn(1, 1, context, 64).to(torch.bfloat16) for _ in range(2))
    table = model_module.Qwen38TTNNRoPETable(
        lane_fake.from_torch(cos_host, dtype=BF16, layout=ROW_MAJOR, mesh_mapper="replicate"),
        lane_fake.from_torch(sin_host, dtype=BF16, layout=ROW_MAJOR, mesh_mapper="replicate"),
        context,
        None,
        FakeContract(),
    )
    positions = [1 + 4 * k for k in range(32)]
    row = Qwen38TTNNDevicePositionRow.allocate("mesh", table.mesh_contract, positions, lanes=32)
    index = row.index_row()
    rope = table.rows_chunk(index, row.block_start_index_row(index))
    for name, host, lookup in (
        ("cos", cos_host, positions),
        ("sin", sin_host, positions),
        ("block_start_cos", cos_host, [p & ~3 for p in positions]),
        ("block_start_sin", sin_host, [p & ~3 for p in positions]),
    ):
        got = _same_everywhere(getattr(rope, name))
        assert tuple(got.shape) == (1, 1, 32, 64) and getattr(rope, name).layout == TILE
        for lane, position in enumerate(lookup):
            assert torch.equal(got[0, 0, lane].view(torch.int16), host[0, 0, position].view(torch.int16)), (name, lane)


# --------------------------------------------------------------------------- MoE rows 1..32


def test_moe_row_contract_admits_every_lane_count_and_keeps_the_one_row_shapes(expect_error) -> None:
    assert moe_module.ROWS5_HARDWARE_PROVEN is True and moe_module.ROWS32_HARDWARE_PROVEN is True
    one = moe_module.Qwen38TTNNMoERowContract(1)
    assert one.hidden_sharded == (1, 1, 1, 640) and one.moe_sparse_input == (1, 1, 1, 2560)
    assert one.local_combine == (10, 1, 2560) and one.fast_reduce_scores == (1, 1, 1, 10)
    for rows in (2, 5, 8, 16, 32):
        contract = moe_module.Qwen38TTNNMoERowContract(rows)
        assert contract.hidden_sharded == (1, 1, rows, 640) and contract.full_hidden == (1, 1, rows, 2560)
        assert contract.moe_sparse_input == (1, rows, 2560) and contract.moe_routing == (1, rows, 10)
        assert contract.local_combine == (10, rows, 2560) and contract.fast_reduce_input == (10, 1, rows, 2560)
        assert contract.fast_reduce_scores == (rows, 1, 1, 10) and contract.output_sharded == (1, 1, rows, 640)
    for bad in (0, 33, 5.0, True):
        with expect_error(ValueError, match=r"MoE rows must be exactly one of"):  # allow-pytest.raises: contract
            moe_module.Qwen38TTNNMoERowContract(bad)


# --------------------------------------------------------------------------- greedy resolve per lane


@pytest.mark.parametrize("rows", [1, 5, 8, 32])
def test_lane_greedy_resolve_puts_row_u_in_lane_u(expect_error, lane_fake, rows: int) -> None:
    contract = FakeContract()
    constants = Qwen38TTNNTokenRowConstants.build("mesh", contract)
    torch.manual_seed(rows)
    values = (torch.randn(TP, rows) * 8).to(torch.bfloat16)
    values[1, 0] = values[0, 0]  # a cross-owner tie in row 0: the lowest owner wins
    if rows > 2:
        values[3, 2] = values[2, 2] = values.max() + 1
    indices = torch.randint(0, LOCAL_VOCAB_SIZE, (TP, rows))
    candidates = Qwen38GreedyCandidates(
        local_indices=FakeTensor([indices[d].reshape(1, 1, rows).to(torch.int64) for d in range(TP)], U32, ROW_MAJOR),
        local_values=FakeTensor([values[d].reshape(1, 1, rows, 1).clone() for d in range(TP)], BF16, TILE),
        rows=rows,
        vocab_ranges=tuple((d * LOCAL_VOCAB_SIZE, (d + 1) * LOCAL_VOCAB_SIZE) for d in range(TP)),
    )
    token_row = resolve_greedy_lanes_on_device(
        candidates, constants=constants, mesh_contract=contract, collective_topology="linear"
    )
    assert token_row.shape == TOKEN_ROW_SHAPE and token_row.dtype is FP32 and token_row.layout == TILE
    owners = torch.argmax(values.float(), dim=0)  # resolve_greedy's rule: the first owner on ties
    expected = (
        torch.tensor([owner * LOCAL_VOCAB_SIZE for owner in owners.tolist()]) + indices[owners, torch.arange(rows)]
    )
    got = _same_everywhere(token_row).reshape(-1)
    assert got[:rows].to(torch.int64).tolist() == expected.tolist()
    assert got[rows:].eq(0).all() and torch.equal(got[:rows], got[:rows].to(torch.int64).to(torch.float32))
    too_many = Qwen38GreedyCandidates(local_indices=None, local_values=None, rows=33, vocab_ranges=())
    with expect_error(ValueError, match=r"greedy candidate rows must be in \[1,32\], got 33"):
        resolve_greedy_lanes_on_device(too_many, constants=None, mesh_contract=contract, collective_topology="linear")


# --------------------------------------------------------------------------- source pins


def test_one_row_paths_are_untouched_and_the_lane_derive_and_resolve_keep_their_ops() -> None:
    # The 1-row position class, derive, greedy resolve and PLE body keep their contracts (the lane bodies' host I/O
    # pins: test_batched_lanes_gdn_no_device).
    scalar = inspect.getsource(contracts_module.Qwen38TTNNDevicePosition)
    assert "lanes" not in scalar and "ttnn.add(self.scalar, 1, memory_config=ttnn.DRAM_MEMORY_CONFIG)" in scalar
    derive_one = inspect.getsource(qsa_module.derive_qsa_position_inputs)
    assert '_require_shape(position_scalar, (1, 1, 1, 1), "QSA position scalar")' in derive_one
    assert "lanes" not in derive_one and "ttnn.repeat" not in derive_one
    assert "candidates.rows != 1" in inspect.getsource(embedding_module.Qwen38TTNNLMHead.resolve_greedy_on_device)
    convolve_one = inspect.getsource(ple_module.Qwen38TTNNPLE._convolve)
    assert "rows = (state.conv[0], state.conv[3], state.conv[6], normalized)" in convolve_one
    assert "state.token_context = prepared.next_token_context" in inspect.getsource(
        ple_module.Qwen38TTNNPLE.forward_prepared
    )
    # The lane derive: exact integer / data-movement ops only.
    lane_derive = function_calls(qsa_module.derive_qsa_lane_inputs)
    allowed = {
        "ttnn.bitwise_and",
        "ttnn.bitwise_right_shift",
        "ttnn.bitwise_left_shift",
        "ttnn.bitwise_or",
        "ttnn.reshape",
        "ttnn.typecast",
        "ttnn.to_layout",
        "ttnn.eq",
        "ttnn.lt",
        "ttnn.ge",
        "ttnn.rsub",
        "ttnn.repeat",
        "ttnn.add",
        "ttnn.multiply",
        "ttnn.minimum",
        "ttnn.subtract",
        "ttnn.slice",
        "ttnn.sum",
    }
    assert {call for call in lane_derive if call.startswith("ttnn.")} <= allowed, lane_derive
    assert lane_derive.count("ttnn.repeat") == 2 and lane_derive.count("ttnn.reshape") == 2
    # The per-lane scalars are slices of the derived rows (one slab-row scalar per lane, the INT32 users), and the
    # selection tiles come from the lane indicator times the one-hot (batch broadcast), their keep columns from a sum.
    assert lane_derive.count("ttnn.slice") == 2 and lane_derive.count("ttnn.sum") == 1
    resolve = function_calls(embedding_module.resolve_greedy_lanes_on_device)
    assert (
        resolve.count("ttnn.gather") == 1 and resolve.count("ttnn.argmax") == 1 and resolve.count("ttnn.to_layout") == 6
    )
    assert "ttnn.repeat" not in resolve  # repeat of a 16-byte ROW_MAJOR row crashes the runtime
    assert "ttnn.pad" in resolve and "ttnn.multiply" not in resolve  # lane u only, no unit-column splat
    # The lanes PLE convolution shifts the per-lane slots exactly as the 1-row ring.
    convolve_lanes = inspect.getsource(ple_module.Qwen38TTNNPLE._convolve_lanes)
    assert "rows = (lanes_state.conv[0], lanes_state.conv[3], lanes_state.conv[6], normalized)" in convolve_lanes
    assert "_copy_inplace(lanes_state.conv[index + 1], lanes_state.conv[index]" in convolve_lanes
