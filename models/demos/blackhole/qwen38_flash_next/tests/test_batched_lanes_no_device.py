# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Batched decode lanes without a device: the rows contract at 1..32, the per-lane position row, the per-lane
QSA derive, RoPE rows and greedy resolve, the PLE lanes and the GDN/GR lane validators.

The integer ops (position row, QSA lane derive, RoPE lookup, greedy resolve) run on a small torch-backed ttnn
that models the exact UINT32/FP32 op surface they use; the PLE lanes and the GDN state run on the step-4 fake
(``test_mtp_v2_step4_rows_no_device``), the GR rows on the GR fake (``test_gr_rows_no_device``).  Every lane
result is compared bitwise to the 1-row path on that lane's stream.  The source pins hold the 1-row bodies
untouched and the lane derive free of host I/O.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from models.demos.blackhole.qwen38_flash_next.ttnn import contracts as contracts_module
from models.demos.blackhole.qwen38_flash_next.ttnn import embedding as embedding_module
from models.demos.blackhole.qwen38_flash_next.ttnn import gdn as gdn_module
from models.demos.blackhole.qwen38_flash_next.ttnn import gr as gr_module
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

TESTS = Path(__file__).resolve().parent
TP = 4
LANE_BLOCKS = 512  # a 2,048-token allocation: the smallest legal QSA cache, 512 compressed blocks
LANE_CONTEXT = LANE_BLOCKS * COMPRESS_RATIO
# 32 lane positions of one residue class (3 mod 4) across block, tile and top-k boundaries.
LANE_POSITIONS = tuple(3 + 4 * k for k in (0, 1, 2, 7, 8, 15, 16, 31, 32, 63, 64, 100, 127, 128, 255, 256))
LANE_POSITIONS += tuple(3 + 4 * k for k in (300, 340, 400, 450, 500, 505, 506, 507, 508, 509, 510, 511, 5, 9, 13, 17))
assert len(LANE_POSITIONS) == MAX_LANES and all(p % 4 == 3 and p < LANE_BLOCKS * COMPRESS_RATIO for p in LANE_POSITIONS)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, TESTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


step4 = _load("test_mtp_v2_step4_rows_no_device")
gr_test = _load("test_gr_rows_no_device")
BF16, FP32, TILE, ROW_MAJOR = step4.BF16, step4.FP32, step4.TILE, step4.ROW_MAJOR
FakeContract, FakeTensor = step4.FakeContract, step4.FakeTensor
U32 = step4._DType("uint32", torch.int64)  # UINT32 values as int64 in [0, 2**32)
I32 = step4._DType("int32", torch.int32)
U32_MASK = 0xFFFFFFFF


# --------------------------------------------------------------------------- integer / resolve fake


def _padded_shape(self) -> tuple[int, ...]:
    shape = self.shape
    if self.layout != TILE or len(shape) < 2:
        return shape
    return (*shape[:-2], -(-shape[-2] // 32) * 32, -(-shape[-1] // 32) * 32)


FakeTensor.padded_shape = property(_padded_shape)


def _host(host: torch.Tensor, dtype) -> torch.Tensor:
    return host.to(torch.int64) & U32_MASK if dtype is U32 else host.to(dtype.torch)


def _from_torch(host, *, dtype, layout, device=None, memory_config=None, mesh_mapper):
    host = _host(host, dtype)
    if mesh_mapper == "replicate":
        return FakeTensor([host.clone() for _ in range(TP)], dtype, layout)
    kind, dims = mesh_mapper
    assert kind == "shard"
    return FakeTensor([piece.clone() for piece in torch.chunk(host, TP, dim=dims[1])], dtype, layout, dims[1])


def _operand(b, index):
    return b.torch_shards()[index] if isinstance(b, FakeTensor) else b


def _arith(torch_op):
    def op(a, b, *, memory_config=None, dtype=None):
        results = []
        for index, x in enumerate(a.torch_shards()):
            y = _operand(b, index)
            if a.dtype is U32:
                result = torch_op(x, y) & U32_MASK
            else:
                y = y.to(torch.float32) if isinstance(y, torch.Tensor) else y
                result = torch_op(x.float(), y)
                if torch_op is torch.mul:
                    # binary_ng forces the product to +0.0 whenever an input is zero (the device's rule).
                    result = torch.where(x.float() == 0, torch.zeros_like(result), result)
                result = result.to(a.dtype.torch)
            results.append(result)
        return FakeTensor(results, a.dtype, a.layout)

    return op


def _compare(torch_op):
    def op(a, b, *, dtype, memory_config=None):
        assert dtype is U32
        return FakeTensor(
            [torch_op(x, _operand(b, index)).to(torch.int64) for index, x in enumerate(a.torch_shards())], U32, a.layout
        )

    return op


def _typecast(t, dtype, memory_config=None):
    if dtype is I32:
        assert all(int(x.max()) < 2**31 for x in t.torch_shards())
    return FakeTensor([x.to(dtype.torch) for x in t.torch_shards()], dtype, t.layout)


def _embedding(indices, table, *, layout, dtype, memory_config=None):
    assert indices.dtype is U32 and indices.shape == (1, 1, 32)
    return FakeTensor(
        [
            row_table[0, 0][index.reshape(-1)].reshape(1, 32, -1).clone()
            for index, row_table in zip(indices.torch_shards(), table.torch_shards())
        ],
        dtype,
        layout,
    )


def _all_gather(t, *, dim, cluster_axis, memory_config=None):
    full = torch.cat(t.torch_shards(), dim=dim)
    return FakeTensor([full.clone() for _ in range(TP)], t.dtype, t.layout)


def _argmax(t, dim, keepdim):
    return FakeTensor(
        [torch.argmax(x, dim=dim, keepdim=keepdim).to(torch.int64) for x in t.torch_shards()], U32, t.layout
    )


def _gather(t, dim, index, memory_config=None):
    return FakeTensor(
        [torch.gather(x, dim, i.to(torch.int64)) for x, i in zip(t.torch_shards(), index.torch_shards())],
        t.dtype,
        t.layout,
    )


def _pad(t, padding, value=0.0, *, memory_config=None):
    flat = [amount for pair in reversed(padding) for amount in pair]
    return FakeTensor([F.pad(x, flat, value=value) for x in t.torch_shards()], t.dtype, t.layout)


def _shift(torch_op):
    return lambda a, count, memory_config=None: FakeTensor(
        [torch_op(x, count) & U32_MASK for x in a.torch_shards()], U32, a.layout
    )


def make_lane_fake() -> SimpleNamespace:
    return SimpleNamespace(
        uint32=U32,
        int32=I32,
        bfloat16=BF16,
        float32=FP32,
        TILE_LAYOUT=TILE,
        ROW_MAJOR_LAYOUT=ROW_MAJOR,
        TILE_SIZE=32,
        DRAM_MEMORY_CONFIG="DRAM_MEMORY_CONFIG",
        Topology=SimpleNamespace(Linear="linear"),
        ShardTensor2dMesh=lambda device, mesh_shape, dims: ("shard", dims),
        from_torch=_from_torch,
        copy_host_to_device_tensor=lambda host, target: step4._copy(host, target),
        copy=step4._copy,
        deallocate=step4._deallocate,
        get_device_tensors=step4._get_device_tensors,
        to_torch=lambda t, mesh_composer=None: t.torch_shards()[0].clone(),
        add=_arith(torch.add),
        subtract=_arith(torch.sub),
        multiply=_arith(torch.mul),
        minimum=lambda a, b, memory_config=None: FakeTensor(
            [torch.clamp(x, max=b) for x in a.torch_shards()], a.dtype, a.layout
        ),
        rsub=lambda t, value, memory_config=None: FakeTensor(
            [(value - x.float()).to(t.dtype.torch) for x in t.torch_shards()], t.dtype, t.layout
        ),
        bitwise_and=_arith(torch.bitwise_and),
        bitwise_or=_arith(torch.bitwise_or),
        bitwise_left_shift=_shift(torch.bitwise_left_shift),
        bitwise_right_shift=_shift(torch.bitwise_right_shift),
        eq=_compare(torch.eq),
        lt=_compare(torch.lt),
        ge=_compare(torch.ge),
        typecast=_typecast,
        reshape=lambda t, shape, memory_config=None: FakeTensor(
            [x.reshape(tuple(shape)) for x in t.torch_shards()], t.dtype, t.layout
        ),
        repeat=lambda t, multipliers, memory_config=None: FakeTensor(
            [x.repeat(*multipliers) for x in t.torch_shards()], t.dtype, t.layout
        ),
        to_layout=lambda t, layout, memory_config=None: FakeTensor(
            [x.clone() for x in t.torch_shards()], t.dtype, layout
        ),
        unsqueeze_to_4D=lambda t: FakeTensor([x.reshape(1, *x.shape) for x in t.torch_shards()], t.dtype, t.layout),
        embedding=_embedding,
        all_gather=_all_gather,
        argmax=_argmax,
        gather=_gather,
        pad=_pad,
        slice=step4._slice,
        sum=_sum,
        CoreCoord=lambda x, y: ("core", x, y),
        num_cores_to_corerangeset=lambda cores, grid, row_wise: ("cores", cores, grid),
        ShardStrategy=SimpleNamespace(HEIGHT="HEIGHT"),
        ShardOrientation=SimpleNamespace(ROW_MAJOR="ROW_MAJOR"),
        create_sharded_memory_config=lambda shape, grid, strategy, orientation, use_height_and_width_as_shard_shape: (
            "sharded",
            tuple(shape),
            grid,
        ),
    )


def _sum(t, dim, keepdim, *, memory_config=None, compute_kernel_config=None, scalar=1.0):
    assert keepdim and dim in (2, 3)
    return FakeTensor(
        [(x.double().sum(dim=dim, keepdim=True) * scalar).to(t.dtype.torch) for x in t.torch_shards()],
        t.dtype,
        t.layout,
    )


@pytest.fixture
def lane_fake(monkeypatch):
    # the fused PLE lane body serves by default and the fake ttnn runs the chains only: switch it off here
    monkeypatch.setenv("QWEN38_FUSED_OFF", "ple")
    fake = make_lane_fake()
    for module in (contracts_module, qsa_module, model_module, embedding_module):
        monkeypatch.setattr(module, "ttnn", fake)
        monkeypatch.setattr(module, "replicate_tensor_2d_mesh_mapper", lambda device: "replicate", raising=False)
    return fake


def _values(tensor: FakeTensor, coordinate: int = 0) -> torch.Tensor:
    return tensor.torch_shards()[coordinate]


def _same_everywhere(tensor: FakeTensor) -> torch.Tensor:
    first = _values(tensor)
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


# --------------------------------------------------------------------------- QSA lane derive


def test_lane_inputs_emulation_is_the_per_position_emulation_lane_by_lane(expect_error) -> None:
    positions = list(LANE_POSITIONS[:16]) + [0, 1, 2, 5, 6, 30, 31, 32, 33, 34, 2044, 2045, 2046, 2047, 100, 1000]
    for count in (1, 3, 8, 32):
        lanes = emulate_qsa_lane_inputs(positions, allocated_compressed_blocks=LANE_BLOCKS, lanes=count)
        offsets = lane_kv_offsets(count, LANE_CONTEXT)
        assert offsets == [lane * LANE_CONTEXT if lane < count else 0 for lane in range(32)]
        assert lanes["kv_row_start_row"].shape == POSITION_INDEX_ROW_SHAPE and lanes["block_index_i32"].shape == (
            count,
        )
        for name in ("kv_hit_tiles", "ring_hit_tiles"):
            assert lanes[name].shape == (1, count, 32, 32) and lanes[name].dtype == torch.bfloat16
        for name in ("kv_keep_col", "ring_keep_col"):
            assert lanes[name].shape == (1, count, 32, 1) and lanes[name].dtype == torch.bfloat16
        for lane, position in enumerate(positions):
            single = emulate_qsa_position_inputs(position, allocated_compressed_blocks=LANE_BLOCKS)
            assert int(lanes["kv_row_start_row"][0, 0, 0, lane]) == int(single["kv_block_start"]) + offsets[lane]
            for name in ("indexer_neg_mask", "row_keep_bits"):
                assert torch.equal(lanes[name][:, :, lane : lane + 1], single[name]), (name, lane)
            # The tail slots [lo, lo + tail_count) carry the lane offset; the kept slots and the sentinels are untouched.
            fill, want = lanes["row_fill"][0, 0, lane], single["row_fill"].reshape(-1)
            geometry = qsa_selection_geometry(position + 1)
            slots = torch.arange(SPARSE_INDEX_CAPACITY)
            tail = (slots >= geometry.complete_token_count) & (
                slots < geometry.complete_token_count + geometry.tail_count
            )
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
    assert lanes.compressed_rows_memory_config == ("sharded", (32, 128), ("cores", count, ("core", 8, -(-count // 8))))
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
    assert moe_module.SUPPORTED_ROWS == (*range(1, MAX_LANES + 1), moe_module.LONG_PREFILL_CHUNK_ROWS)
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


# --------------------------------------------------------------------------- GDN lanes state


@pytest.fixture
def fake(monkeypatch):
    chunk = step4.FakeChunk()
    fake_ttnn = step4.make_fake_ttnn(chunk)
    step4.install_source_chunk_fake(monkeypatch, chunk)
    for module in (gdn_module, ple_module):
        monkeypatch.setattr(module, "ttnn", fake_ttnn)
        monkeypatch.setattr(module, "replicate_tensor_2d_mesh_mapper", lambda device: "replicate", raising=False)
    return fake_ttnn


def test_gdn_state_allocates_b_lanes_and_the_one_row_body_admits_batch_one_only(expect_error, fake) -> None:
    module = step4._gdn_module(step4._device_gdn_weights(step4._gdn_oracle_weights()))
    contract = module.mesh_contract
    state = gdn_module.Qwen38TTNNGDNState.allocate("mesh", contract, layer_index=0, batch_size=8)
    assert state.batch_size == 8 and state.recurrent.shape == (8, 12, 128, 128) and state.recurrent.dtype is FP32
    assert all(slot.shape == (1, 1, 8, 2560) and slot.dtype is BF16 for slot in state.conv)
    state.reset_inplace()
    assert state.conv_phase == 0 and len(state.conv_window()) == 4
    state.batch_size = 9
    with expect_error(
        RuntimeError,
        match=r"GDN recurrent state \(batch 9\) local shape must be \(9, 12, 128, 128\), got \(8, 12, 128, 128\)",
    ):
        state.validate()
    state.batch_size = 8
    with expect_error(ValueError, match="the 1-row GDN decode body admits a batch-1 state, got batch 8"):
        module._validate_state(state)
    one = module.allocate_state()
    assert one.batch_size == 1 and one.recurrent.shape == (1, 12, 128, 128) and one.conv[0].shape == (1, 1, 1, 2560)
    module._validate_state(one)
    for bad in (0, 33):
        with expect_error(ValueError, match=r"GDN state batch size must be in \[1,32\]"):  # allow-pytest.raises
            gdn_module.Qwen38TTNNGDNState.allocate("mesh", contract, layer_index=0, batch_size=bad)


# --------------------------------------------------------------------------- GR rows at a lane count


@pytest.fixture
def gr_fake(monkeypatch):
    fake_ttnn = gr_test._gr_fake()
    monkeypatch.setattr(gr_module, "ttnn", fake_ttnn)
    return fake_ttnn


def test_gr_rows_shapes_follow_the_row_count_and_keep_the_chunk_constants(expect_error) -> None:
    chunk = gr_module.gr_rows_shapes(32)
    assert chunk["residual"] == gr_module.RESIDUAL_ROWS_LOCAL_SHAPE and chunk["flat"] == gr_module.FLAT_ROWS_LOCAL_SHAPE
    assert chunk["block"] == gr_module.BLOCK_ROWS_LOCAL_SHAPE and chunk["injection"] == gr_module.INJECTION_ROWS_SHAPE
    assert chunk["partial"] == gr_module.PARTIAL_REDUCTION_ROWS_SHAPE and chunk["flat_padded"] == chunk["flat"]
    eight = gr_module.gr_rows_shapes((1, 4, 8, 640))
    assert eight["rows"] == 8 and eight["residual"] == (1, 4, 8, 640) and eight["token_major"] == (1, 8, 4, 640)
    assert eight["flat"] == (1, 1, 8, gr_module.FLAT_LOCAL_WIDTH) and eight["flat_padded"] == (
        1,
        1,
        32,
        gr_module.FLAT_LOCAL_WIDTH,
    )
    assert eight["partial"] == (1, 1, 8, gr_module.PARTIAL_WIDTH) and eight["partial_padded"] == (
        1,
        1,
        32,
        gr_module.PARTIAL_WIDTH,
    )
    with expect_error(ValueError, match=r"GR block rows must be in \[1,32\], got 0"):  # allow-pytest.raises: contract
        gr_module.gr_rows_shapes((1, 1, 0, 640), label="GR block rows")
    with expect_error(ValueError, match=r"must be rank 4 .*, got \[1, 4, 640\]"):  # allow-pytest.raises: contract
        gr_module.gr_rows_shapes((1, 4, 640))


@pytest.mark.parametrize("flat_views", [False, True])
@pytest.mark.parametrize("rows", [1, 8])
def test_gr_rows_at_a_lane_count_equal_the_one_row_read_and_write_row_for_row(
    expect_error, gr_fake, rows: int, flat_views: bool
) -> None:
    module = gr_test._gr_module(gr_fake)
    torch.manual_seed(11 + rows)
    shapes = gr_module.gr_rows_shapes(rows)
    # The GR test module loads its own copy of the step-4 fake: its dtype objects are the ones gr.py compares.
    residual_rows = gr_test.FakeTensor(
        [gr_test._bf16(*shapes["residual"]) for _ in range(TP)], gr_test.BF16, gr_test.TILE, 3
    )
    block_rows = gr_test.FakeTensor([gr_test._bf16(*shapes["block"]) for _ in range(TP)], gr_test.BF16, gr_test.TILE, 3)
    block_input, state = module.read_rows(residual_rows, flat_views=flat_views)
    written = module.write_rows(block_rows, state)
    assert block_input.shape == shapes["block"] and state.injection.shape == shapes["injection"]
    assert written.shape == shapes["residual"]
    one_row = [module.read(gr_test._rows(residual_rows, row)) for row in range(rows)]
    gr_test._equal(block_input, [block for block, _ in one_row], dim=2, label="GR lanes read block input")
    gr_test._equal(
        state.injection, [row_state.injection for _, row_state in one_row], dim=2, label="GR lanes injection"
    )
    gr_test._equal(
        written,
        [module.write(gr_test._rows(block_rows, row), one_row[row][1]) for row in range(rows)],
        dim=2,
        label="GR lanes write",
    )
    if rows == 8:
        # Four block rows against the eight-row state: the row count comes from the residual rows, the block rows must match.
        with expect_error(ValueError, match=r"GR block rows must be TILE BF16 \(1, 1, 8, 640\)"):
            module.write_rows(
                gr_test.FakeTensor([gr_test._bf16(1, 1, 4, 640) for _ in range(TP)], gr_test.BF16, gr_test.TILE, 3),
                state,
            )


# --------------------------------------------------------------------------- PLE lanes


class _LaneLookup(step4._FakeResidentLookup):
    """The step-4 table with the batched lane read: one payload of every lane's row, lane-major."""

    def lookup_lanes(self, tokens, contexts):
        payloads, next_contexts = zip(
            *(self.lookup_token(int(token), context) for token, context in zip(tokens, contexts))
        )
        return bytearray().join(payloads), tuple(next_contexts)


def _ple_lanes_module():
    module = step4._ple_module()
    module._resident_lookup = _LaneLookup()
    module.host_embedding = module._resident_lookup
    return module


def test_ple_lanes_equal_the_one_row_path_per_lane_with_independent_contexts(expect_error, fake) -> None:
    module = _ple_lanes_module()
    lanes, steps = 3, 3
    streams = ((17, 15, 16), (16, 95859, 17), (20, 17, 17))  # lane u's tokens, step by step
    torch.manual_seed(24)
    residual = torch.randn(steps, lanes, 4, 2560).to(torch.bfloat16)

    # 1-row references: one fresh state per lane, threaded through its nine slots and its host context.
    states = [module.allocate_state() for _ in range(lanes)]
    deltas = [[None] * lanes for _ in range(steps)]
    contexts = [[None] * lanes for _ in range(steps)]
    for step in range(steps):
        for lane, state in enumerate(states):
            prepared = module.prepare_decode_input(torch.tensor([[streams[lane][step]]], dtype=torch.long), state)
            result = module.forward_prepared(step4._residual_rows(residual[step, lane : lane + 1]), prepared, state)
            deltas[step][lane] = step4._cat(result.residual_delta, 3)
            contexts[step][lane] = tuple(int(v) for v in state.token_context[0])

    lanes_state = module.allocate_lanes_state(lanes)
    assert lanes_state.token_contexts == (None,) * lanes and lanes_state.conv[0].shape == (1, lanes, 4, 640)
    for step in range(steps):
        tokens = [streams[lane][step] for lane in range(lanes)]
        prepared = module.prepare_lanes_input(tokens, lanes_state)
        assert prepared.embedding_rows.shape == (1, 1, lanes, 640) and prepared.embedding_rows.layout == ROW_MAJOR
        delta = module.forward_prepared_lanes(step4._residual_rows(residual[step]), prepared, lanes_state)
        delta_lanes = step4._cat(delta, 3)
        assert delta_lanes.shape == (1, lanes, 4, 2560)
        for lane in range(lanes):
            assert torch.equal(delta_lanes[:, lane : lane + 1], deltas[step][lane]), (step, lane)
        assert lanes_state.token_contexts == tuple(contexts[step]), step
        prepared.release()
    for index in range(ple_module.CONV_STATE_LENGTH):
        lane_slots = step4._cat(lanes_state.conv[index], 3)
        for lane, state in enumerate(states):
            assert torch.equal(lane_slots[:, lane : lane + 1], step4._cat(state.conv[index], 3)), (index, lane)
    # The streams differ, so the per-lane comparison is not vacuous.
    assert not torch.equal(deltas[2][1], deltas[2][2])

    # Contracts: token count, stale prepared input, lane count.
    with expect_error(ValueError, match=r"needs 3 tokens \(one per lane\), got 2"):
        module.prepare_lanes_input([1, 2], lanes_state)
    stale = module.prepare_lanes_input([1, 2, 3], lanes_state)
    lanes_state.token_contexts = ((1, 1),) * lanes
    with expect_error(RuntimeError, match="prepared PLE lanes were looked up from contexts"):
        module.forward_prepared_lanes(step4._residual_rows(residual[0]), stale, lanes_state)
    with expect_error(ValueError, match=r"PLE lanes must be in \[1,32\], got 33"):  # allow-pytest.raises: contract
        module.allocate_lanes_state(33)


def test_ple_inject_lanes_adds_the_permuted_delta_to_the_branch_major_rows(expect_error, fake) -> None:
    module = _ple_lanes_module()
    lanes = 2
    torch.manual_seed(25)
    residual = torch.randn(lanes, 4, 2560).to(torch.bfloat16)
    tokens = [17, 20]
    lanes_state = module.allocate_lanes_state(lanes)
    delta = step4._cat(
        module.forward_prepared_lanes(
            step4._residual_rows(residual), module.prepare_lanes_input(tokens, lanes_state), lanes_state
        ),
        3,
    )
    fresh = module.allocate_lanes_state(lanes)
    branch_major = residual.reshape(1, lanes, 4, 2560).permute(0, 2, 1, 3).contiguous()
    injected = module.inject_lanes(
        FakeTensor([p.clone() for p in torch.chunk(branch_major, TP, dim=3)], BF16, TILE, 3),
        module.prepare_lanes_input(tokens, fresh),
        fresh,
    )
    assert injected.shape == (1, 4, lanes, 640)
    expected = (branch_major.float() + delta.permute(0, 2, 1, 3).float()).to(torch.bfloat16)
    assert torch.equal(step4._cat(injected, 3).view(torch.int16), expected.view(torch.int16))
    with expect_error(ValueError, match=r"lanes residual must be BF16 TILE \(1, 4, 2, 640\)"):
        module.inject_lanes(step4._residual_rows(residual), module.prepare_lanes_input(tokens, fresh), fresh)


# --------------------------------------------------------------------------- greedy resolve per lane


@pytest.mark.parametrize("rows", [1, 5, 8, 32])
def test_lane_greedy_resolve_puts_row_u_in_lane_u(lane_fake, rows: int) -> None:
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


def test_lane_greedy_resolve_rejects_more_rows_than_lanes(expect_error, lane_fake) -> None:
    candidates = Qwen38GreedyCandidates(local_indices=None, local_values=None, rows=33, vocab_ranges=())
    with expect_error(ValueError, match=r"greedy candidate rows must be in \[1,32\], got 33"):
        resolve_greedy_lanes_on_device(
            candidates, constants=None, mesh_contract=FakeContract(), collective_topology="linear"
        )


# --------------------------------------------------------------------------- source pins


def _ttnn_calls(function) -> list[str]:
    lines = inspect.getsource(function).splitlines()
    indent = len(lines[0]) - len(lines[0].lstrip())
    tree = ast.parse("\n".join(line[indent:] for line in lines))
    return [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]


def test_one_row_paths_are_untouched_and_the_lane_paths_touch_no_host() -> None:
    # The 1-row position class, derive, GDN body, greedy resolve and PLE body keep their contracts.
    scalar = inspect.getsource(contracts_module.Qwen38TTNNDevicePosition)
    assert "lanes" not in scalar and "ttnn.add(self.scalar, 1, memory_config=ttnn.DRAM_MEMORY_CONFIG)" in scalar
    derive_one = inspect.getsource(qsa_module.derive_qsa_position_inputs)
    assert '_require_shape(position_scalar, (1, 1, 1, 1), "QSA position scalar")' in derive_one
    assert "lanes" not in derive_one and "ttnn.repeat" not in derive_one
    assert "candidates.rows != 1" in inspect.getsource(embedding_module.Qwen38TTNNLMHead.resolve_greedy_on_device)
    assert "Advance one true-global-B1 token" in inspect.getsource(gdn_module.Qwen38TTNNGDN.forward_decode)
    assert "admits a batch-1 state" in inspect.getsource(gdn_module.Qwen38TTNNGDN._validate_state)
    convolve_one = inspect.getsource(ple_module.Qwen38TTNNPLE._convolve)
    assert "rows = (state.conv[0], state.conv[3], state.conv[6], normalized)" in convolve_one
    assert "state.token_context = prepared.next_token_context" in inspect.getsource(
        ple_module.Qwen38TTNNPLE.forward_prepared
    )
    # The lane bodies: exact integer / data-movement ops only, no host I/O, no per-lane host ints.
    lane_derive = _ttnn_calls(qsa_module.derive_qsa_lane_inputs)
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
    for function in (
        contracts_module.Qwen38TTNNDevicePositionRow.advance,
        contracts_module.Qwen38TTNNDevicePositionRow.index_row,
        embedding_module.resolve_greedy_lanes_on_device,
        ple_module.Qwen38TTNNPLE.forward_prepared_lanes,
        ple_module.Qwen38TTNNPLE._convolve_lanes,
    ):
        calls = _ttnn_calls(function)
        assert not any(
            call.startswith(("ttnn.from_torch", "ttnn.to_torch", "ttnn.copy_host_to_device_tensor", "torch."))
            for call in calls
        ), function.__qualname__
    resolve = _ttnn_calls(embedding_module.resolve_greedy_lanes_on_device)
    assert (
        resolve.count("ttnn.gather") == 1 and resolve.count("ttnn.argmax") == 1 and resolve.count("ttnn.to_layout") == 6
    )
    assert "ttnn.repeat" not in resolve  # repeat of a 16-byte ROW_MAJOR row crashed the pinned runtime (2026-09-03)
    assert "ttnn.pad" in resolve and "ttnn.multiply" not in resolve  # lane u only, no unit-column splat
    # The lanes PLE convolution shifts the per-lane slots exactly as the 1-row ring.
    convolve_lanes = inspect.getsource(ple_module.Qwen38TTNNPLE._convolve_lanes)
    assert "rows = (lanes_state.conv[0], lanes_state.conv[3], lanes_state.conv[6], normalized)" in convolve_lanes
    assert "_copy_inplace(lanes_state.conv[index + 1], lanes_state.conv[index]" in convolve_lanes
