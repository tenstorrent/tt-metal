# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Prefill chunks (32 and 128 rows) of the QSA layer: the chunk inputs, constants and RoPE rows, without a device.

The chunk inputs are the 1-row position-generic derivation on same-shape templates: the torch emulation of
row j must equal the 1-row emulation at P + j field for field, the 128-row emulation the four 32-row emulations.
The two bf16 select tiles (block-mean pooling and the per-block row pick) are checked under the device's matmul
arithmetic model.
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.tests.helpers.torch_ttnn import (
    BF16,
    I32,
    ROW_MAJOR,
    TILE,
    U32,
    FakeDType,
    deallocate,
)
from models.demos.blackhole.qwen38_flash_next.ttnn import contracts as contracts_module
from models.demos.blackhole.qwen38_flash_next.ttnn import qsa as qsa_module
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import CHUNK_ROWS, LONG_CHUNK_ROWS
from models.demos.blackhole.qwen38_flash_next.ttnn.qsa import (
    ALL_ONES_U32,
    BLOCK_TOPK,
    CHUNK_BLOCKS,
    COMPRESS_RATIO,
    SPARSE_INDEX_CAPACITY,
    TOKEN_BUDGET,
    Qwen38TTNNQSAChunkConstants,
    Qwen38TTNNQSAChunkInputs,
    emulate_qsa_chunk_inputs,
    emulate_qsa_position_inputs,
    qsa_chunk_constant_rows,
)

CHUNK_POSITIONS = (0, 32, 64, 2016, 2048, 2080, 8160, 32736)
CHUNK_INPUT_FIELDS = ("kv_block_start", "block_index_i32", "indexer_neg_mask", "row_keep_bits", "row_fill")
RESIDENT_BLOCKS = 32768 // COMPRESS_RATIO
CHUNK_UINT32_TEMPLATES = (
    "arange32_lanes",
    "block_start_lanes",
    "row_index_blocks",
    "arange_blocks_rows",
    "row_index_slots",
    "arange_slots_rows",
    "all_ones_rows",
)


# --- emulation: row j of the chunk is the 1-row derivation at P + j --------------------


@pytest.mark.parametrize("position", CHUNK_POSITIONS)
def test_chunk_emulation_rows_are_the_per_position_emulation(position: int) -> None:
    chunk = emulate_qsa_chunk_inputs(position, allocated_compressed_blocks=RESIDENT_BLOCKS)
    assert tuple(chunk) == CHUNK_INPUT_FIELDS
    assert chunk["kv_block_start"].tolist() == [[[[position]]]]  # P % 32 == 0: the slab starts at P itself
    assert [int(index) for index in chunk["block_index_i32"]] == [position // 4 + block for block in range(8)]
    assert all(index.dtype == torch.int32 and tuple(index.shape) == (1,) for index in chunk["block_index_i32"])
    assert chunk["indexer_neg_mask"].shape == (1, 1, CHUNK_ROWS, RESIDENT_BLOCKS)
    assert chunk["row_keep_bits"].shape == chunk["row_fill"].shape == (1, 1, CHUNK_ROWS, SPARSE_INDEX_CAPACITY)
    for row in range(CHUNK_ROWS):
        single = emulate_qsa_position_inputs(position + row, allocated_compressed_blocks=RESIDENT_BLOCKS)
        for name in ("indexer_neg_mask", "row_keep_bits", "row_fill"):
            assert torch.equal(chunk[name][:, :, row : row + 1], single[name]), (name, row)
    # The chunk's last row completes block P/4 + 7 and is the first row whose block P/4 + 7 is unmasked.
    mask = chunk["indexer_neg_mask"][0, 0]
    unmasked = (mask == 0).sum(dim=-1).tolist()
    assert unmasked == [(position + row + 1) // 4 for row in range(CHUNK_ROWS)]
    assert unmasked[-1] == position // 4 + 8 and mask.dtype == torch.bfloat16


def test_chunk_emulation_rejects_unaligned_positions(expect_error) -> None:
    for position in (1, 31, 33, -32, True, 2.0):
        with expect_error(ValueError):  # allow-pytest.raises: pure contract test
            emulate_qsa_chunk_inputs(position, allocated_compressed_blocks=RESIDENT_BLOCKS)


@pytest.mark.parametrize("position", (0, 128, 1920, 2016, 2048, 8064, 32640))
def test_long_chunk_emulation_is_the_four_chunk_emulations_stacked(position: int) -> None:
    long = emulate_qsa_chunk_inputs(position, allocated_compressed_blocks=RESIDENT_BLOCKS, rows=LONG_CHUNK_ROWS)
    short = [
        emulate_qsa_chunk_inputs(position + CHUNK_ROWS * tile, allocated_compressed_blocks=RESIDENT_BLOCKS)
        for tile in range(LONG_CHUNK_ROWS // CHUNK_ROWS)
    ]
    assert torch.equal(long["kv_block_start"], short[0]["kv_block_start"])
    # The four 32-row chunks write compressed blocks P/4 .. P/4 + 31 one by one; the 128-row chunk writes them as the
    # cache's tile P/128 in one page (no per-block indices).
    assert [t.item() for chunk in short for t in chunk["block_index_i32"]] == [position // 4 + b for b in range(32)]
    assert long["block_index_i32"] == () and long["compressed_tile_i32"].tolist() == [[position // 128]]
    assert long["compressed_tile_i32"].dtype == torch.int32 and "compressed_tile_i32" not in short[0]
    for name in ("indexer_neg_mask", "row_keep_bits", "row_fill"):
        assert torch.equal(long[name], torch.cat([chunk[name] for chunk in short], dim=2)), name


# --- constants: host images and the two select tiles under the matmul arithmetic model -----


def test_chunk_constant_rows_carry_the_row_index_templates_and_the_block_lanes() -> None:
    host = qsa_chunk_constant_rows(RESIDENT_BLOCKS)
    assert host["arange32_lanes"].reshape(-1).tolist() == list(range(32))
    assert host["block_start_lanes"].reshape(-1).tolist() == [4 * i for i in range(8)] + [0] * 24
    for name, width in (("row_index_blocks", RESIDENT_BLOCKS), ("row_index_slots", SPARSE_INDEX_CAPACITY)):
        assert host[name].shape == (1, 1, CHUNK_ROWS, width)
        assert torch.equal(host[name], torch.arange(CHUNK_ROWS).reshape(1, 1, CHUNK_ROWS, 1).expand_as(host[name]))
    for name, width in (("arange_blocks_rows", RESIDENT_BLOCKS), ("arange_slots_rows", SPARSE_INDEX_CAPACITY)):
        assert torch.equal(host[name], torch.arange(width).reshape(1, 1, 1, width).expand_as(host[name]))
    assert torch.equal(host["all_ones_rows"], torch.full((1, 1, CHUNK_ROWS, SPARSE_INDEX_CAPACITY), ALL_ONES_U32))
    for name in ("row_index_blocks", "arange_blocks_rows", "row_index_slots", "arange_slots_rows", "all_ones_rows"):
        assert host[name].dtype == torch.int64 and int(host[name].min()) >= 0 and int(host[name].max()) < 2**32
    assert host["block_offsets_rows"].shape == (1, 1, CHUNK_ROWS, TOKEN_BUDGET)
    assert torch.equal(host["block_offsets_rows"][0, 0, 5], torch.arange(4).repeat(BLOCK_TOPK))
    assert host["sentinel_pad_rows"].shape == (1, 1, CHUNK_ROWS, SPARSE_INDEX_CAPACITY - TOKEN_BUDGET)
    assert int(host["sentinel_pad_rows"].min()) == ALL_ONES_U32
    # Memory at the resident lane: two [32, 8192], three [32, 2080], one [32, 2048] and one [32, 32] UINT32
    # templates, 3.16 MB.
    total = sum(host[name].numel() * 4 for name in host if host[name].dtype == torch.int64)
    assert total == 4 * 32 * (2 * RESIDENT_BLOCKS + 3 * SPARSE_INDEX_CAPACITY + TOKEN_BUDGET + 32 + 2)
    # The 128-row chunk: 128 row lanes, 32 block starts, a [32, 128] pool and 32 row picks; its first rows are the
    # 32-row chunk's.
    long = qsa_chunk_constant_rows(RESIDENT_BLOCKS, LONG_CHUNK_ROWS)
    assert long["arange32_lanes"].shape == (1, 1, 4, 32)
    assert long["arange32_lanes"].reshape(-1).tolist() == list(range(128))
    assert long["block_start_lanes"].reshape(-1).tolist() == [4 * i for i in range(32)]
    assert long["pool_select"].shape == (1, 1, 32, 128) and long["row_selects"].shape == (32, 1, 1, 32, 32)
    for block in range(32):
        assert long["pool_select"][0, 0, block].nonzero().reshape(-1).tolist() == list(range(4 * block, 4 * block + 4))
        assert long["row_selects"][block, 0, 0, 0, block] == 1.0 and long["row_selects"][block].sum() == 1.0
    for name in ("row_index_blocks", "arange_blocks_rows", "row_index_slots", "arange_slots_rows", "all_ones_rows"):
        assert long[name].shape[2] == 128, name
    assert torch.equal(long["pool_select"][..., :8, :32], host["pool_select"][..., :8, :])
    assert torch.equal(long["row_index_slots"][:, :, :32], host["row_index_slots"])


def _select_matmul(select: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """HiFi4 model: exact bf16 x bf16 products, fp32 accumulation, one rounding to bf16."""

    products = select.to(torch.bfloat16).float().unsqueeze(-1) * rows.float().unsqueeze(-3)
    return products.sum(dim=-2).to(torch.bfloat16)


def test_pool_select_is_the_decode_ring_mean_bitwise_and_row_selects_pick_one_row() -> None:
    host = qsa_chunk_constant_rows(RESIDENT_BLOCKS)
    pool, selects = host["pool_select"][0, 0], host["row_selects"][:, 0, 0]
    assert pool.shape == (32, 32) and selects.shape == (8, 32, 32)
    assert torch.equal(pool.to(torch.bfloat16).float(), pool)  # 0.25 is exact in bf16
    assert torch.count_nonzero(pool[8:]) == 0 and (pool[:8] != 0).sum(dim=-1).eq(4).all()
    torch.manual_seed(5)
    raw_keys = (torch.randn(32, 128) * 3).to(torch.bfloat16)
    pooled = _select_matmul(pool, raw_keys)
    for block in range(CHUNK_BLOCKS):
        # The decode ring path: rows 4-31 zero, fp32 sum over 32 rows scaled by 1/4, one bf16 rounding.
        ring = torch.zeros(32, 128)
        ring[:4] = raw_keys[block * 4 : block * 4 + 4].float()
        expected = (ring.sum(dim=0) * (1.0 / COMPRESS_RATIO)).to(torch.bfloat16)
        assert torch.equal(pooled[block].view(torch.int16), expected.view(torch.int16)), block
    assert torch.count_nonzero(pooled[8:].float()) == 0
    rotated = (torch.randn(32, 128) * 2).to(torch.bfloat16)
    for block in range(CHUNK_BLOCKS):
        picked = _select_matmul(selects[block], rotated)
        assert torch.equal(picked[0].view(torch.int16), rotated[block].view(torch.int16)), block
        assert torch.count_nonzero(picked[1:].float()) == 0


def test_chunk_constants_and_inputs_declare_the_device_fields() -> None:
    assert tuple(Qwen38TTNNQSAChunkConstants.__dataclass_fields__) == (
        "allocated_compressed_blocks",
        "rows",
        "arange32_lanes",
        "block_start_lanes",
        "row_index_blocks",
        "arange_blocks_rows",
        "row_index_slots",
        "arange_slots_rows",
        "all_ones_rows",
        "block_offsets_rows",
        "sentinel_pad_rows",
        "pool_select",
        "row_selects",
        "zero_value_half_rows",
        # The slab's mask operands and page offsets (None for the chunk forms).
        "arange_blocks_row",
        "row_index_col",
        "page_offsets",
        # The slab's hoist decision for its block masks (None for the chunk forms).
        "hoist_masks",
    )
    # compressed_tile_i32: the 128-row chunk's / the slab's page table (None for the 32-row forms);
    # complete_blocks_col: the slab's mask column (None for the chunk forms); block_masks: the slab's hoisted
    # block masks when the hoist is admitted (empty otherwise).
    assert tuple(Qwen38TTNNQSAChunkInputs.__dataclass_fields__) == ("rows",) + CHUNK_INPUT_FIELDS + (
        "compressed_tile_i32",
        "complete_blocks_col",
        "block_masks",
    )
    assert CHUNK_BLOCKS == 8 and CHUNK_ROWS == 32 and CHUNK_ROWS % COMPRESS_RATIO == 0


def test_device_position_advance_by_refuses_a_count_that_is_not_a_positive_int(expect_error) -> None:
    shell = object.__new__(contracts_module.Qwen38TTNNDevicePosition)
    for count in (0, -1, True, 2.0, "32"):
        with expect_error(ValueError):  # allow-pytest.raises: pure contract test
            shell.advance_by(count)


# --- the device derivation after k chunk advances, on an exact-integer fake ------------
# The chain runs on a torch model of the integer ops after k in-trace ``advance_by(rows)`` calls, so every chunk start
# the model reaches by replaying (a tile crossing at P = 128, the top-k regime at 2048, the cache's last chunk) is
# pinned field for field; the 64k/128k/256k options at each context's top chunk and its 32k/64k/128k boundaries.

_int_ids = itertools.count(1)


class IntTensor:
    """One device tensor of the integer fake: int64 storage for UINT32/INT32, bf16 for the mask."""

    def __init__(self, value: torch.Tensor, dtype: FakeDType, layout: str) -> None:
        self.value, self.dtype, self.layout = value, dtype, layout
        self.tensor_id = next(_int_ids)
        self.alive = True

    @property
    def shape(self):
        return tuple(self.value.shape)

    def _read(self) -> torch.Tensor:
        assert self.alive, "read of a deallocated tensor"
        return self.value


class IntContract:
    def validate_tensor(self, tensor, *, placement, shard_dim=None, require_device=True) -> None:
        assert tensor.alive


def int_u32(values: torch.Tensor) -> IntTensor:
    return IntTensor(values.to(torch.int64).clone(), U32, ROW_MAJOR)


def _int_op(torch_op):
    def op(a, b, *, memory_config=None, dtype=None):
        result = torch_op(a._read(), b._read() if isinstance(b, IntTensor) else b)
        tag = dtype or a.dtype
        if tag is BF16:
            result = result.to(torch.bfloat16)
        elif result.dtype != torch.int64:
            result = result.to(torch.int64)
        return IntTensor(result, tag, a.layout)

    return op


def make_int_ttnn() -> SimpleNamespace:
    """The exact integer ops of the chunk-input derivation on single-device ``IntTensor``s."""

    def typecast(tensor, dtype, *, memory_config=None):
        value = tensor._read()
        return IntTensor(value.to(torch.bfloat16) if dtype is BF16 else value.clone(), dtype, tensor.layout)

    def from_torch(host, *, dtype, layout, device=None, memory_config=None, mesh_mapper=None):
        return IntTensor(host.to(torch.int64) if dtype is U32 else host.to(torch.bfloat16), dtype, layout)

    def copy_int(source, target):
        target._read().copy_(source._read())
        return target

    return SimpleNamespace(
        uint32=U32,
        int32=I32,
        bfloat16=BF16,
        ROW_MAJOR_LAYOUT=ROW_MAJOR,
        TILE_LAYOUT=TILE,
        TILE_SIZE=32,
        DRAM_MEMORY_CONFIG="DRAM",
        bitwise_and=_int_op(torch.bitwise_and),
        bitwise_or=_int_op(torch.bitwise_or),
        bitwise_right_shift=_int_op(torch.bitwise_right_shift),
        bitwise_left_shift=_int_op(torch.bitwise_left_shift),
        add=_int_op(torch.add),
        subtract=_int_op(torch.sub),
        multiply=_int_op(torch.mul),
        minimum=_int_op(lambda a, b: torch.clamp(a, max=b)),
        lt=_int_op(torch.lt),
        ge=_int_op(torch.ge),
        rsub=_int_op(lambda a, b: b - a),
        typecast=typecast,
        reshape=lambda t, shape, memory_config=None: IntTensor(t._read().reshape(tuple(shape)), t.dtype, t.layout),
        from_torch=from_torch,
        copy_host_to_device_tensor=copy_int,
        copy=copy_int,
        deallocate=deallocate,
    )


DERIVE_CASES = [(CHUNK_ROWS, 32_768, CHUNK_ROWS * k) for k in (1, 2, 4, 63, 64, 127, 254, 255, 1023)]
DERIVE_CASES += [(CHUNK_ROWS, 65_536, 32_768), (CHUNK_ROWS, 65_536, 65_504), (CHUNK_ROWS, 131_072, 65_536)]
DERIVE_CASES += [(CHUNK_ROWS, 131_072, 131_040), (CHUNK_ROWS, 262_144, 131_072), (CHUNK_ROWS, 262_144, 262_112)]
DERIVE_CASES += [(LONG_CHUNK_ROWS, 32_768, position) for position in (0, 2048, 32640)]


@pytest.fixture
def integer_fake(monkeypatch):
    fake = make_int_ttnn()
    monkeypatch.setattr(qsa_module, "ttnn", fake)
    monkeypatch.setattr(contracts_module, "ttnn", fake)
    monkeypatch.setattr(contracts_module, "replicate_tensor_2d_mesh_mapper", lambda device: "replicate")
    return fake


@pytest.mark.parametrize(("rows", "allocated_context", "position"), DERIVE_CASES)
def test_chunk_inputs_after_advancing_the_device_position_are_the_emulation(
    expect_error, integer_fake, rows: int, allocated_context: int, position: int
) -> None:
    blocks = allocated_context // COMPRESS_RATIO
    device_position = contracts_module.Qwen38TTNNDevicePosition(
        scalar=int_u32(torch.zeros(1, 1, 1, 1)),
        ones_row=int_u32(torch.ones(1, 1, 1, CHUNK_ROWS)),
        block_start_mask_row=int_u32(torch.full((1, 1, 1, CHUNK_ROWS), contracts_module.BLOCK_START_LANE_MASK)),
        mesh_device="mesh",
        mesh_contract=IntContract(),
    )
    for _ in range(position // rows):
        device_position.advance_by(rows)  # the chunk body's last op, once per replay
    assert device_position.scalar._read().reshape(-1).tolist() == [position]

    host = qsa_chunk_constant_rows(blocks, rows)
    constants = SimpleNamespace(
        allocated_compressed_blocks=blocks,
        high27_mask=int_u32(torch.full((1, 1, 1, 1), qsa_module.KV_BLOCK_START_MASK)),
    )
    chunk = SimpleNamespace(
        allocated_compressed_blocks=blocks, rows=rows, **{name: int_u32(host[name]) for name in CHUNK_UINT32_TEMPLATES}
    )
    inputs = qsa_module.derive_qsa_chunk_inputs(device_position.scalar, constants, chunk)
    expected = emulate_qsa_chunk_inputs(position, allocated_compressed_blocks=blocks, rows=rows)
    assert inputs.rows == rows
    assert inputs.kv_block_start._read().tolist() == expected["kv_block_start"].tolist() == [[[[position]]]]
    assert [(t.dtype, t._read().tolist()) for t in inputs.block_index_i32] == [
        (I32, e.tolist()) for e in expected["block_index_i32"]
    ]
    assert inputs.indexer_neg_mask.dtype is BF16 and inputs.indexer_neg_mask.shape == (1, 1, rows, blocks)
    assert torch.equal(inputs.indexer_neg_mask._read(), expected["indexer_neg_mask"])
    for name in ("row_keep_bits", "row_fill"):
        actual = getattr(inputs, name)
        assert actual.dtype is U32 and actual.shape == (1, 1, rows, SPARSE_INDEX_CAPACITY)
        assert torch.equal(actual._read(), expected[name]), name
    # The last row's block count: the chunk completes blocks up to (position + rows) / 4, inside the allocation.
    unmasked = (expected["indexer_neg_mask"][0, 0] == 0).sum(dim=-1).tolist()
    assert unmasked[-1] == (position + rows) // 4 <= blocks
    with expect_error(ValueError):  # allow-pytest.raises: pure contract test
        qsa_module.derive_qsa_chunk_inputs(
            device_position.scalar,
            SimpleNamespace(allocated_compressed_blocks=blocks // 2, high27_mask=constants.high27_mask),
            chunk,
        )
    if rows == LONG_CHUNK_ROWS:
        # The 128-row chunk writes its 32 compressed rows as one tile: the page table P / 128, no per-block indices.
        assert expected["block_index_i32"] == ()
        assert inputs.compressed_tile_i32._read().tolist() == expected["compressed_tile_i32"].tolist()
        assert expected["compressed_tile_i32"].tolist() == [[position // 128]]
        assert inputs.compressed_tile_i32.dtype is I32 and inputs.compressed_tile_i32.shape == (1, 1)
        # The layer's admission of the 128-row inputs: the page table, no per-block indices.
        layer = qsa_module.Qwen38TTNNQSA.__new__(qsa_module.Qwen38TTNNQSA)
        layer.allocated_compressed_blocks = blocks
        layer._validate_chunk_inputs(inputs)
        with expect_error(ValueError):  # allow-pytest.raises: the 128-row inputs without their page table
            layer._validate_chunk_inputs(
                Qwen38TTNNQSAChunkInputs(
                    rows, inputs.kv_block_start, (), inputs.indexer_neg_mask, inputs.row_keep_bits, inputs.row_fill
                )
            )
    # The RoPE index rows of the same replay: lane j = P + j; the block-start lanes P + 4i (i < rows / 4), P past them.
    index_row = device_position.index_row()
    index_rows = integer_fake.add(index_row, chunk.arange32_lanes)
    block_start_rows = integer_fake.add(index_row, chunk.block_start_lanes)
    assert index_rows._read().reshape(-1).tolist() == [position + j for j in range(rows)]
    assert block_start_rows._read().reshape(-1).tolist() == [
        position + (4 * i if i < rows // 4 else 0) for i in range(CHUNK_ROWS)
    ]
    # The hand-off's host reset lands any position, as finish_prefill does after a padded tail (5 pad rows here).
    device_position.reset(position + rows - 5)
    assert device_position.scalar._read().reshape(-1).tolist() == [position + rows - 5]
    assert inputs.kv_block_start.alive and inputs.row_fill.alive
