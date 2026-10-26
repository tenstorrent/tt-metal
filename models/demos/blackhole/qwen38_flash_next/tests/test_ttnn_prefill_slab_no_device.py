# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The prefill slab (``--prefill-slab ROWS``) without a device: the row-count contract, the QSA slab constants and
chunk-input emulation, the driver's plan (slabs, then 128-row chunks, then 32-row chunks and the tail) and the
2D-multicast matmul config.
"""

from __future__ import annotations

import contextlib
import itertools
import math
from types import SimpleNamespace

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.tools import qwen38_prefill_driver as driver_module
from models.demos.blackhole.qwen38_flash_next.ttnn import contracts as contracts_module
from models.demos.blackhole.qwen38_flash_next.ttnn import decode_matmul as decode_matmul_module
from models.demos.blackhole.qwen38_flash_next.ttnn import gdn as gdn_module
from models.demos.blackhole.qwen38_flash_next.ttnn import gr as gr_module
from models.demos.blackhole.qwen38_flash_next.ttnn import layer as layer_module
from models.demos.blackhole.qwen38_flash_next.ttnn import model as model_module
from models.demos.blackhole.qwen38_flash_next.ttnn import moe as moe_module
from models.demos.blackhole.qwen38_flash_next.ttnn import prefill_dense
from models.demos.blackhole.qwen38_flash_next.ttnn import qsa as qsa_module
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import (
    CHUNK_ROW_COUNTS,
    CHUNK_ROWS,
    LONG_CHUNK_ROWS,
    MAX_SLAB_ROWS,
    MIN_SLAB_ROWS,
    is_slab_rows,
)

SLAB = MAX_SLAB_ROWS
BLOCKS = 8192  # the 32k context's compressed blocks


# --------------------------------------------------------------------------- contracts


def test_slab_row_contract(expect_error) -> None:
    assert (MIN_SLAB_ROWS, MAX_SLAB_ROWS) == (256, 2048), (MIN_SLAB_ROWS, MAX_SLAB_ROWS)
    assert CHUNK_ROW_COUNTS == (32, 128)  # the chunk forms are untouched
    for rows in (256, 512, 1024, 2048):
        assert is_slab_rows(rows) and contracts_module.chunk_row_tiles(rows) == rows // 32
    for rows in (32, 128, 0, 31, 64, 96, 192, 2000, 2176, 4096, 8192, True, 2048.0, "2048"):
        assert not is_slab_rows(rows)
    for rows in (0, 31, 64, 96, 192, 4096, 8192):
        with expect_error(ValueError):  # allow-pytest.raises: pure contract test
            contracts_module.chunk_row_tiles(rows)
    assert gdn_module.rows_tile_count(SLAB) == SLAB and gdn_module.rows_tile_count(128) == 128
    assert gr_module.residual_rows_shape(SLAB) == (1, 4, SLAB, 640)
    assert moe_module.routed_tokens_per_call_for(SLAB) == SLAB and moe_module.routed_tokens_per_call_for(128) == 128
    assert moe_module.routed_tokens_per_call_for(32) == 32 and moe_module.routed_tokens_per_call_for(5) == 5
    contract = moe_module.Qwen38TTNNMoERowContract(SLAB, moe_module.SUPPORTED_ROWS + (SLAB,))
    assert contract.row_tiles == SLAB // 32 and contract.local_combine == (10, SLAB, 2560)
    # 1..32 = decode, the MTP verifier, the chunk and B lanes, 128 = the long chunk; the slab per instance, not globally
    assert moe_module.SUPPORTED_ROWS == (*range(1, 33), 128)
    with expect_error(ValueError):  # allow-pytest.raises: pure contract test
        moe_module.Qwen38TTNNMoERowContract(SLAB)


def test_slab_routes_its_rows_in_one_call_and_reduces_in_512_row_blocks() -> None:
    """A slab instance routes its rows in one moe_compute call on the local output path (height shard 1: nothing is
    staged) and reduces in 512-row blocks; the chunk forms are untouched."""
    per_call = (moe_module.routed_tokens_per_call_for(SLAB), moe_module.routed_tokens_per_call_for(256))
    assert per_call == (SLAB, 256), per_call
    assert moe_module.routed_tokens_per_call_for(128) == 128 and moe_module.routed_tokens_per_call_for(32) == 32
    for ring in (7, 8):
        assert moe_module.moe_compute_output_height_shard_dim(SLAB, matmul_ring_size=ring) == 1
    assert moe_module.SLAB_REDUCE_BLOCK_ROWS == 512 and SLAB % moe_module.SLAB_REDUCE_BLOCK_ROWS == 0
    # a slab whose rows the 512-row blocks do not divide is refused at construction, not at its first slice
    assert [rows for rows in range(256, 4097, 128) if rows % moe_module.SLAB_REDUCE_BLOCK_ROWS] == [
        256,
        384,
        640,
        768,
        896,
        1152,
        1280,
        1408,
        1664,
        1792,
        1920,
        2176,
        2304,
        2432,
        2688,
        2816,
        2944,
        3200,
        3328,
        3456,
        3712,
        3840,
        3968,
    ]


class _AliasingTensor:
    """A stand-in device tensor: ttnn.slice over the full extent and ttnn.concat of one tensor return their input."""

    def __init__(self, shape, name, dtype=None, layout=None):
        self.shape, self.name, self.dtype, self.layout = tuple(shape), name, dtype, layout


def _slab_reduce_fakes(monkeypatch, rows: int):
    """_weighted_reduce_slab_blocks on a fake ttnn whose slice/concat alias like the real ones (a full-extent slice and
    a one-tensor concat return their input); returns the deallocated names and the reduce's call count."""
    freed, reduced = [], []
    K, H = moe_module.TOP_K, moe_module.HIDDEN_SIZE

    def slice_(tensor, start, end, memory_config):
        if start == (0,) * len(start) and tuple(end) == tensor.shape:
            return tensor  # the alias the real op returns for a full-extent slice
        return _AliasingTensor(
            tuple(e - s for s, e in zip(start, end)), f"{tensor.name}[{start[-2] if len(start) > 2 else start[1]}]"
        )

    def concat(tensors, dim, memory_config):
        if len(tensors) == 1:
            return tensors[0]  # the alias the real op returns for one tensor
        shape = list(tensors[0].shape)
        shape[dim] = sum(t.shape[dim] for t in tensors)
        return _AliasingTensor(shape, "concat")

    def reduce_(stack, indices, mapping, **kwargs):
        reduced.append((stack.name, indices.name, kwargs["scores_tensor"].name))
        return [_AliasingTensor((1, 1, stack.shape[2], H), f"partial{len(reduced)}")]

    fake_ttnn = SimpleNamespace(
        DRAM_MEMORY_CONFIG="dram",
        TILE_LAYOUT="tile",
        slice=slice_,
        concat=concat,
        reshape=lambda tensor, shape: _AliasingTensor(shape, f"reshape({tensor.name})"),
        to_layout=lambda tensor, layout, memory_config, pad_value: _AliasingTensor(
            tensor.shape, f"tilized({tensor.name})"
        ),
        deallocate=lambda tensor: freed.append(tensor.name),
        experimental=SimpleNamespace(
            view=lambda tensor, shape: _AliasingTensor(shape, f"view({tensor.name})"),
            deepseek_moe_fast_reduce_nc_fused=reduce_,
        ),
    )
    monkeypatch.setattr(moe_module, "ttnn", fake_ttnn)
    combine = _AliasingTensor((K, rows, H), "combine")
    routing = SimpleNamespace(
        scores=_AliasingTensor((1, 1, rows, K), "scores"), indices=_AliasingTensor((1, 1, rows, K), "indices")
    )
    marked = []
    owner = SimpleNamespace(
        rows=rows,
        expert_mapping="mapping",
        compute_config="compute",
        mesh_contract=SimpleNamespace(mark_local_partial=lambda tensor, **kwargs: marked.append((tensor.name, kwargs))),
        row_contract=SimpleNamespace(full_hidden=(1, 1, rows, H)),
    )
    partial = moe_module.Qwen38TTNNMoE._weighted_reduce_slab_blocks(
        owner, combine, "full_hidden", routing, lambda phase: None
    )
    return SimpleNamespace(partial=partial, freed=freed, reduced=reduced, marked=marked)


@pytest.mark.parametrize("rows", [512, 1024, 2048])
def test_slab_reduce_never_frees_what_it_did_not_create(monkeypatch, rows: int) -> None:
    """The slab reduce frees only what it created: a full-extent slice and a one-tensor concat return their input."""
    result = _slab_reduce_fakes(monkeypatch, rows)
    blocks = rows // moe_module.SLAB_REDUCE_BLOCK_ROWS
    assert len(result.reduced) == blocks
    assert result.partial.shape == (1, 1, rows, moe_module.HIDDEN_SIZE)
    assert result.marked and result.marked[0][0] == result.partial.name
    for name in ("combine", "scores", "indices", result.partial.name):
        assert name not in result.freed, (name, result.freed)
    if blocks == 1:
        assert result.freed == ["view(tilized(reshape(combine)))"]  # only the tilized stack the method created
        assert result.partial.name == "partial1"
    else:
        assert result.freed.count("combine[0]") == 1 and f"combine[{rows - 512}]" in result.freed
        assert all(f"partial{i}" in result.freed for i in range(1, blocks + 1)) and result.partial.name == "concat"
        assert all(f"scores[{s}]" in result.freed and f"indices[{s}]" in result.freed for s in range(0, rows, 512))


@pytest.mark.parametrize("rows", [512, 2048])
def test_block_slab_routes_each_128_row_block_through_the_128_row_worker(monkeypatch, rows: int) -> None:
    """A slab the op refuses in one call routes each 128-row block through the 128-row worker; every slice is freed
    after its block and the concatenated partial is marked as the slab's local partial."""
    freed, slices, blocks, marked, phases = [], [], [], [], []
    K, H, block = moe_module.TOP_K, moe_module.HIDDEN_SIZE, LONG_CHUNK_ROWS

    def slice_(tensor, start, end, memory_config):
        slices.append((tensor.name, start[2], end[2]))
        return _AliasingTensor(tuple(e - s for s, e in zip(start, end)), f"{tensor.name}[{start[2]}]")

    def concat(tensors, dim, memory_config):
        assert dim == 2 and [t.name for t in tensors] == [f"partial[{s}]" for s in range(0, rows, block)]
        return _AliasingTensor((1, 1, sum(t.shape[2] for t in tensors), H), "concat")

    def block_partial(hidden, routing, packed_w0_w1, packed_w2):
        assert hidden.shape == (1, 1, block, H) and routing.scores.shape == routing.indices.shape == (1, 1, block, K)
        assert (packed_w0_w1, packed_w2) == ("w0_w1", "w2") and routing.tiles is None
        blocks.append((hidden.name, routing.scores.name, routing.indices.name))
        return _AliasingTensor((1, 1, block, H), f"partial[{hidden.name.partition('[')[2][:-1]}]")

    monkeypatch.setattr(
        moe_module,
        "ttnn",
        SimpleNamespace(
            DRAM_MEMORY_CONFIG="dram", slice=slice_, concat=concat, deallocate=lambda tensor: freed.append(tensor.name)
        ),
    )
    owner = object.__new__(moe_module.Qwen38TTNNMoE)
    owner.rows, owner.block_instance = rows, SimpleNamespace(_routed_partial=block_partial)
    owner.row_contract = SimpleNamespace(full_hidden=(1, 1, rows, H))
    owner.mesh_contract = SimpleNamespace(
        validate_tensor=lambda tensor, placement: None,
        mark_local_partial=lambda tensor, **kwargs: marked.append((tensor.name, kwargs["expected_shape"])),
    )
    routing = moe_module.Qwen38TTNNRouting(
        _AliasingTensor((1, 1, rows, K), "scores"), _AliasingTensor((1, 1, rows, K), "indices")
    )
    hidden = _AliasingTensor((1, 1, rows, H), "hidden")
    partial = owner._routed_partial_blocks(hidden, routing, "w0_w1", "w2", phases.append)
    starts = range(0, rows, block)
    assert blocks == [(f"hidden[{s}]", f"scores[{s}]", f"indices[{s}]") for s in starts], blocks
    assert slices == [(name, s, s + block) for s in starts for name in ("hidden", "scores", "indices")], slices
    assert freed == [f"{name}[{s}]" for s in starts for name in ("hidden", "scores", "indices")] + [
        f"partial[{s}]" for s in starts
    ], freed
    assert partial.name == "concat" and marked == [("concat", (1, 1, rows, H))], marked
    assert phases[:2] == ["before-routed-dispatch", "after-routed-dispatch"], phases


@pytest.mark.parametrize("case", ["admitted", "refused", "inside a capture", "model error", "128-row blocks"])
def test_slab_call_admission_asks_the_op_without_dispatching(monkeypatch, expect_error, case: str) -> None:
    """The slab's one moe_compute call is put to the op under a NO_DISPATCH graph capture before any slab work.

    True when the op takes the call, False on an op refusal (TT_FATAL / TT_THROW), each logged with the slab form it
    selects; any other error stays itself; inside another capture, or on a 128-row-block instance, it does not run."""
    capture, freed, calls, stamped, logged = [], [], [], [], []
    combine = _AliasingTensor((moe_module.TOP_K, SLAB, moe_module.HIDDEN_SIZE), "combine")
    combine.tensor_id, combine.buffer_address = 0, lambda: 4096
    captured_combine = _AliasingTensor(combine.shape, "combine, captured")
    captured_combine.tensor_id, captured_combine.buffer_address = 1, combine.buffer_address
    fatal = "Too many tokens in input, got: 2048 but expected max: 128"
    op_error = RuntimeError(f"TT_FATAL @ moe_compute_device_operation.cpp:140: cond\ninfo:\n{fatal}\nbacktrace:\n ---")

    def moe_compute(*args, **kwargs):
        calls.append((args, kwargs))
        if case == "refused":
            raise op_error
        if case == "model error":  # slot 5 is not the slab's combine buffer: the model's own check raises
            return [_AliasingTensor((1,), f"out{index}") for index in range(6)]
        return [_AliasingTensor((1,), f"out{index}") for index in range(5)] + [captured_combine]

    fake_ttnn = SimpleNamespace(
        Shape=tuple,
        bfloat16="bf16",
        uint16="u16",
        ROW_MAJOR_LAYOUT="rm",
        DRAM_MEMORY_CONFIG="dram",
        graph=SimpleNamespace(
            RunMode=SimpleNamespace(NO_DISPATCH="no-dispatch"),
            is_graph_capture_active=lambda: case == "inside a capture",
            begin_graph_capture=lambda mode: capture.append(mode),
            end_graph_capture=lambda: capture.append("end"),
        ),
        allocate_tensor_on_device=lambda shape, dtype, layout, mesh, memory_config: _AliasingTensor(
            shape, f"stand-in {dtype} {memory_config}", dtype, layout
        ),
        deallocate=freed.append,
        experimental=SimpleNamespace(moe_compute=moe_compute),
    )
    monkeypatch.setattr(moe_module, "ttnn", fake_ttnn)
    monkeypatch.setattr(moe_module.fused.program, "stamp_topology", lambda tensor, reference: stamped.append(reference))
    monkeypatch.setattr(moe_module, "logger", SimpleNamespace(info=logged.append))
    module = object.__new__(moe_module.Qwen38TTNNMoE)
    module.rows, module.routed_calls = SLAB, SLAB // LONG_CHUNK_ROWS if case == "128-row blocks" else 1
    module.row_contract = moe_module.Qwen38TTNNMoERowContract(SLAB, moe_module.SUPPORTED_ROWS + (SLAB,))
    module.mesh_device, module.expert_mapping, module.local_combine_output = "mesh", "mapping", combine
    module.routing_l1_memory_config, module.output_height_shard_dim = "drain-shard", 1
    if case in ("inside a capture", "128-row blocks"):
        error = RuntimeError if case == "inside a capture" else ValueError
        match = "cannot run inside another graph capture" if case == "inside a capture" else "one-call slab instance"
        with expect_error(error, match):
            module.admit_slab_call("w0_w1", "w2")
        assert capture == [] and calls == [], (capture, calls)
        return
    if case == "model error":
        with expect_error(RuntimeError, "did not return the persistent six-slot output"):
            module.admit_slab_call("w0_w1", "w2")
        assert capture == ["no-dispatch", "end"] and len(calls) == 1, (capture, len(calls))
        return
    admitted = module.admit_slab_call("w0_w1", "w2")
    assert admitted is (case == "admitted"), admitted
    assert capture == ["no-dispatch", "end"], capture
    ((args, kwargs),) = calls
    stand_ins = args[:3]
    assert [(stand_in.shape, stand_in.dtype) for stand_in in stand_ins] == [
        ((1, SLAB, moe_module.HIDDEN_SIZE), "bf16"),
        ((1, SLAB, moe_module.TOP_K), "u16"),
        ((1, SLAB, moe_module.TOP_K), "bf16"),
    ], [stand_in.name for stand_in in stand_ins]
    in_l1 = [stand_in.name.endswith("drain-shard") for stand_in in stand_ins]
    assert in_l1 == [False, True, True], in_l1
    assert args[3:] == ("mapping", "w0_w1", "w2") and stamped == [combine] * 3, (args[3:], stamped)
    assert kwargs["optional_output_tensor"] is combine and kwargs["output_height_shard_dim"] == 1, kwargs
    assert all(stand_in in freed for stand_in in stand_ins), [tensor.name for tensor in freed]
    assert combine not in freed and captured_combine not in freed, [tensor.name for tensor in freed]
    assert logged[0].startswith(f"asking moe_compute whether it takes the {SLAB}-row slab in one call"), logged
    if case == "refused":
        assert logged[1:] == [
            f"the {SLAB}-row prefill slab routes its experts in 16 moe_compute calls of 128 tokens per layer: "
            f"moe_compute refuses its one {SLAB}-token call ({fatal})"
        ], logged
    else:
        assert logged[1:] == [f"the {SLAB}-row prefill slab routes its experts in one moe_compute call per layer"]
        others = [tensor.name for tensor in freed if tensor not in stand_ins]
        assert others == ["out0", "out1", "out2", "out4"], others


# --------------------------------------------------------------------------- the 2D-multicast matmul config


def test_same_buffer_compares_buffer_addresses_and_every_freeing_module_shares_it() -> None:
    """``ttnn.reshape`` can return a view (new tensor id, same buffer), so freeing its source is decided by
    ``contracts.same_buffer``, never by tensor id."""
    from models.demos.blackhole.qwen38_flash_next.ttnn import contracts, final_mixer, gr, mtp, mtp_v2

    class _Tensor:
        def __init__(self, address: int, tensor_id: int) -> None:
            self._address, self.tensor_id = address, tensor_id

        def buffer_address(self) -> int:
            return self._address

        def device(self):
            return "mesh"

    view, source, copy = _Tensor(0x1000, 7), _Tensor(0x1000, 3), _Tensor(0x2000, 9)
    assert contracts.same_buffer(view, source) and not contracts.same_buffer(copy, source)
    assert contracts._tensor_key(view) != contracts._tensor_key(source)  # a view never shares the tensor id
    host_a, host_b = SimpleNamespace(tensor_id=4), SimpleNamespace(tensor_id=5)  # a host stand-in: no buffer to alias
    assert contracts.same_buffer(host_a, host_a) and not contracts.same_buffer(host_a, host_b)
    for module in (moe_module, final_mixer, gr, mtp, mtp_v2, qsa_module):
        assert module.same_buffer is contracts.same_buffer


def test_prefill_matmul_config_covers_the_rows_and_columns_with_whole_subblocks(expect_error) -> None:
    captured = {}

    class _Config:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    grid = SimpleNamespace(x=11, y=10)
    mesh = SimpleNamespace(compute_with_storage_grid_size=lambda: grid)
    original = decode_matmul_module.ttnn.MatmulMultiCoreReuseMultiCastProgramConfig
    decode_matmul_module.ttnn.MatmulMultiCoreReuseMultiCastProgramConfig = _Config
    try:
        for k, n in (
            (2560, 4160),
            (1536, 2560),
            (2560, 384),
            (384, 2560),
            (2560, 512),
            (2560, 160),
            (160, 2560),
            (2560, 1),
        ):
            captured.clear()
            decode_matmul_module.prefill_matmul_program_config(mesh, SLAB, k, n)
            cols, rows = captured["compute_with_storage_grid_size"]
            n_tiles, k_tiles = math.ceil(n / 32), k // 32
            assert 1 <= cols <= min(11, n_tiles) and rows == 10
            assert captured["per_core_M"] * rows >= SLAB // 32 and captured["per_core_N"] * cols >= n_tiles
            assert captured["out_subblock_h"] == 1 and captured["per_core_N"] % captured["out_subblock_w"] == 0
            assert 1 <= captured["out_subblock_w"] <= 4 and k_tiles % captured["in0_block_w"] == 0
            assert captured["in0_block_w"] <= 8 and captured["transpose_mcast"] is False
        with expect_error(ValueError):  # allow-pytest.raises: pure contract test
            decode_matmul_module.prefill_matmul_program_config(mesh, 2000, 2560, 512)
    finally:
        decode_matmul_module.ttnn.MatmulMultiCoreReuseMultiCastProgramConfig = original


# --------------------------------------------------------------------------- QSA slab constants and inputs


def test_qsa_slab_constants_pool_every_block_and_carry_no_row_templates() -> None:
    host = qsa_module.qsa_chunk_constant_rows(BLOCKS, SLAB)
    blocks = SLAB // 4
    tiles = -(-blocks // 32)
    assert "row_index_blocks" not in host and "arange_blocks_rows" not in host
    assert tuple(host["row_selects"].shape) == (0, 1, 1, 32, 32)
    assert tuple(host["block_start_lanes"].shape) == (1, 1, tiles, 32)
    assert host["block_start_lanes"].reshape(-1)[:blocks].tolist() == [4 * i for i in range(blocks)]
    pool = host["pool_select"]
    assert tuple(pool.shape) == (1, 1, tiles * 32, SLAB)
    expected = torch.zeros(tiles * 32, SLAB)
    for block in range(blocks):
        expected[block, 4 * block : 4 * block + 4] = 0.25
    assert torch.equal(pool[0, 0], expected)
    assert host["page_offsets"].reshape(-1).tolist() == list(range(tiles))
    assert tuple(host["arange_blocks_row"].shape) == (1, 1, 1, BLOCKS)
    assert host["row_index_col"].reshape(-1).tolist() == list(range(SLAB))
    assert tuple(host["row_keep_bits" if "row_keep_bits" in host else "row_index_slots"].shape) == (1, 1, SLAB, 2080)
    # The chunk forms are unchanged.
    short = qsa_module.qsa_chunk_constant_rows(BLOCKS, CHUNK_ROWS)
    assert tuple(short["row_index_blocks"].shape) == (1, 1, 32, BLOCKS) and len(short["row_selects"]) == 8
    assert tuple(short["block_start_lanes"].shape) == (1, 1, 1, 32)


@pytest.mark.parametrize("position", [0, 2048, 30720])
def test_qsa_slab_emulation_is_the_one_row_rule_per_row(position: int) -> None:
    inputs = qsa_module.emulate_qsa_chunk_inputs(position, allocated_compressed_blocks=BLOCKS, rows=SLAB)
    assert inputs["indexer_neg_mask"] is None and inputs["block_index_i32"] == ()
    tiles = SLAB // 128
    assert inputs["compressed_tile_i32"].tolist() == [[position // 128 + t for t in range(tiles)]]
    complete = inputs["complete_blocks_col"].reshape(-1)
    for row in (0, 1, 3, 4, 127, SLAB - 1):
        one = qsa_module.emulate_qsa_position_inputs(position + row, allocated_compressed_blocks=BLOCKS)
        visible = int((one["indexer_neg_mask"].reshape(-1) == 0).sum())
        assert int(complete[row]) == (position + row + 1) // 4 == visible
        assert torch.equal(inputs["row_keep_bits"][0, 0, row], one["row_keep_bits"].reshape(-1))
        assert torch.equal(inputs["row_fill"][0, 0, row], one["row_fill"].reshape(-1))


def _qsa_fake_ttnn(monkeypatch, blocks: int):
    """A fake ttnn for the QSA slab selection: slice/concat alias like the real ones and ops refuse a freed input."""
    freed, parts = [], []
    u32, rm = "uint32", "row_major"

    def live(*tensors):
        for tensor in tensors:
            if any(tensor is gone for gone in freed):
                raise RuntimeError(f"{tensor.name} is not device resident")

    def made(shape, name, *inputs, dtype=u32, layout=rm):
        live(*inputs)
        return _AliasingTensor(shape, name, dtype, layout)

    def slice_(tensor, start, end, memory_config):
        live(tensor)
        if not any(start) and tuple(end) == tensor.shape:
            return tensor  # the alias the real op returns for a full-extent slice
        shape = tuple(stop - begin for begin, stop in zip(start, end))
        return _AliasingTensor(shape, f"{tensor.name}[{start[2]}]", tensor.dtype, tensor.layout)

    def concat(tensors, dim, memory_config):
        live(*tensors)
        if len(tensors) == 1:
            return tensors[0]  # the alias the real op returns for one tensor
        shape = list(tensors[0].shape)
        shape[dim] = sum(part.shape[dim] for part in tensors)
        return _AliasingTensor(shape, "concat", tensors[0].dtype, tensors[0].layout)

    def topk_large_indices(masked, k):
        parts.append(made((1, 1, masked.shape[2], k), f"block_ids{len(parts)}", masked))
        return parts[-1]

    fake_ttnn = SimpleNamespace(
        DRAM_MEMORY_CONFIG="dram",
        ROW_MAJOR_LAYOUT=rm,
        uint32=u32,
        bfloat16="bfloat16",
        slice=slice_,
        concat=concat,
        all_reduce=lambda tensor, **kwargs: made(tensor.shape, "scores", tensor),
        add=lambda lhs, rhs, **kwargs: made(lhs.shape, "sum", lhs, rhs),
        ge=lambda lhs, rhs, **kwargs: made((1, 1, rhs.shape[2], lhs.shape[3]), "invalid_bits", lhs, rhs),
        typecast=lambda tensor, dtype, **kwargs: made(tensor.shape, "invalid", tensor),
        multiply=lambda tensor, value, **kwargs: made(tensor.shape, "mask_tiled", tensor),
        to_layout=lambda tensor, layout, **kwargs: made(tensor.shape, "mask", tensor),
        bitwise_left_shift=lambda tensor, bits, **kwargs: made(tensor.shape, "starts", tensor),
        repeat_interleave=lambda tensor, repeats, dim, **kwargs: made(
            (*tensor.shape[:3], tensor.shape[3] * repeats), "repeated", tensor
        ),
        bitwise_and=lambda lhs, rhs, **kwargs: made(lhs.shape, "kept", lhs, rhs),
        bitwise_or=lambda lhs, rhs, **kwargs: made(lhs.shape, "sparse_indices", lhs, rhs),
        deallocate=freed.append,
        experimental=SimpleNamespace(
            indexer_score_dsa=lambda query, cache, gate, **kwargs: made(
                (1, 1, CHUNK_ROWS, blocks + CHUNK_ROWS), "local_scores", query, cache
            ),
            topk_large_indices=topk_large_indices,
        ),
    )
    monkeypatch.setattr(qsa_module, "ttnn", fake_ttnn)
    return SimpleNamespace(freed=freed, parts=parts, live=live)


@pytest.mark.parametrize("rows", [512, 1024, 2048])
def test_qsa_slab_selection_never_frees_its_block_ids_before_use(monkeypatch, rows: int) -> None:
    """Each top-k block-id part is freed exactly once, after the expansion reads it; nothing else is freed."""
    blocks, block_rows = 64, qsa_module.SLAB_SCORE_BLOCK_ROWS
    fake = _qsa_fake_ttnn(monkeypatch, blocks)
    owner = SimpleNamespace(
        allocated_compressed_blocks=blocks,
        index_gate="gate",
        indexer_chunk_start=0,
        indexer_compute_config="compute",
        collective_topology="linear",
        mesh_contract=SimpleNamespace(
            mark_local_partial=lambda tensor, **kwargs: fake.live(tensor),
            validate_tensor=lambda tensor, **kwargs: fake.live(tensor),
        ),
    )
    capacity, budget = qsa_module.SPARSE_INDEX_CAPACITY, qsa_module.TOKEN_BUDGET
    inputs = {
        "index_query": _AliasingTensor((1, 1, rows, qsa_module.INDEX_HEAD_DIM), "index_query"),
        "cache": _AliasingTensor((1, 1, blocks, qsa_module.INDEX_HEAD_DIM), "cache"),
        "keep": _AliasingTensor((1, 1, rows, capacity), "keep"),
        "fill": _AliasingTensor((1, 1, rows, capacity), "fill"),
        "offsets": _AliasingTensor((1, 1, rows, budget), "offsets"),
        "pad": _AliasingTensor((1, 1, rows, capacity - budget), "pad"),
    }
    masks = tuple(_AliasingTensor((1, 1, block_rows, blocks), f"mask{i}") for i in range(rows // block_rows))
    result = qsa_module.Qwen38TTNNQSA._sparse_indices_slab(
        owner,
        inputs["index_query"],
        SimpleNamespace(compressed_index_cache=inputs["cache"]),
        SimpleNamespace(block_masks=masks, row_keep_bits=inputs["keep"], row_fill=inputs["fill"]),
        SimpleNamespace(rows=rows, block_offsets_rows=inputs["offsets"], sentinel_pad_rows=inputs["pad"]),
    )
    fake.live(result)
    assert result.shape == (1, 1, rows, capacity), result.shape
    assert len(fake.parts) == rows // block_rows, [part.name for part in fake.parts]
    for tensor in (*inputs.values(), *masks):
        assert all(tensor is not gone for gone in fake.freed), tensor.name
    for part in fake.parts:
        times = sum(part is gone for gone in fake.freed)
        assert times == 1, (part.name, times)


@pytest.mark.parametrize("rows", [512, 1024])
def test_slab_block_mask_never_frees_the_complete_block_column(monkeypatch, rows: int) -> None:
    """A one-block slab's mask must not free the complete-block column its full-extent slice returns: the next QSA
    layer reads it again."""
    blocks, block_rows = 64, qsa_module.SLAB_SCORE_BLOCK_ROWS
    fake = _qsa_fake_ttnn(monkeypatch, blocks)
    column = _AliasingTensor((1, 1, rows, 1), "complete_blocks_col")
    arange = _AliasingTensor((1, 1, 1, blocks), "arange_blocks_row")
    masks = [
        qsa_module.slab_block_mask(column, arange, start, qsa_module.INDEXER_MASK_VALUE)
        for start in range(0, rows, block_rows)
    ]
    fake.live(column, arange, *masks)
    assert all(mask.shape == (1, 1, block_rows, blocks) for mask in masks), [mask.shape for mask in masks]
    sliced = [gone.name for gone in fake.freed if gone.name.startswith("complete_blocks_col[")]
    expected = [f"complete_blocks_col[{start}]" for start in range(0, rows, block_rows)]
    assert sliced == ([] if rows == block_rows else expected), sliced


# --------------------------------------------------------------------------- the driver's plan


class _FakeModel:
    allocated_context = 70000

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def reset_chunk_state_inplace(self, state, chunk_state) -> None:
        self.calls.append(("reset", chunk_state.rows))

    def write_chunk_accepted(self, chunk_state, accepted: int) -> None:
        self.calls.append(("accepted", accepted))

    def prepare_chunk_inputs(self, chunk_state, token_ids, *, ple_context, positions=None, features=None):
        tokens = list(token_ids)
        assert len(tokens) == chunk_state.rows
        self.calls.append(("write", chunk_state.rows, tuple(tokens)))
        return SimpleNamespace(rows=chunk_state.rows, contexts=tuple([ple_context] * (len(tokens) + 1)))

    def upload_chunk_inputs(self, chunk_state, prepared) -> None:
        assert prepared.rows == chunk_state.rows
        self.calls.append(("upload", chunk_state.rows))

    def finish_prefill(self, state, chunk_state, prefilled: int, *, rope_shift: int = 0) -> None:
        self.calls.append(("finish", prefilled))

    def forward_prefill_chunk_generic(self, chunk_state, state, *, gdn_step_anchor: bool = False, mtp=None) -> None:
        self.calls.append(("eager", chunk_state.rows, gdn_step_anchor, mtp))


@pytest.fixture
def driver(monkeypatch):
    fake = SimpleNamespace(
        synchronize_device=lambda mesh: None,
        record_event=lambda mesh, cq_id: object(),
        event_synchronize=lambda event: None,
        _ttnn_execute_trace=lambda mesh, trace_id, cq_id, blocking: None,
    )
    monkeypatch.setattr(driver_module, "ttnn", fake)
    return fake


@pytest.mark.parametrize("start", [0, 5, 32])
@pytest.mark.parametrize("count", [100, 2048, 2048 + 128 + 40, 2 * 2048 + 3 * 128 + 32 + 7, 5000])
def test_driver_runs_slabs_then_long_then_short_chunks(driver, start: int, count: int) -> None:
    model = _FakeModel()
    states = {rows: SimpleNamespace(rows=rows) for rows in (32, 128, SLAB)}
    tokens = list(range(1000, 1000 + count))
    prefill = driver_module.Qwen38ChunkPrefill(
        model,
        object(),
        object(),
        states[32],
        None,
        forced_step=lambda token, context: context,
        long_chunk_state=states[128],
        slab_state=states[SLAB],
    )
    result = prefill.run(tokens, start_position=start, ple_context=None)
    aligned = driver_module.alignment_steps(start, count)
    remaining = count - aligned
    slabs = remaining // SLAB
    long = (remaining - slabs * SLAB) // 128
    short = remaining - slabs * SLAB - long * 128
    writes = [call for call in model.calls if call[0] == "write"]
    assert [rows for _, rows, _ in writes] == [SLAB] * slabs + [128] * long + [32] * math.ceil(short / 32)
    written = [token for _, _, chunk in writes for token in chunk]
    assert written[:remaining] == tokens[aligned:]
    assert result.position == start + count
    assert result.timing.slabs == slabs and result.timing.slab_rows == SLAB
    uploads = [call for call in model.calls if call[0] == "upload"]
    assert [rows for _, rows in uploads] == [rows for _, rows, _ in writes]
    timing = result.timing
    assert len(timing.slab_prepare_ms) == len(timing.slab_upload_ms) == len(timing.slab_wait_ms) == slabs
    assert all(ms >= 0.0 for ms in timing.slab_prepare_ms + timing.slab_upload_ms + timing.slab_wait_ms)
    assert result.timing.long_chunks == long and result.timing.chunks == math.ceil(short / 32)
    eager = [call for call in model.calls if call[0] == "eager"]
    assert [rows for _, rows, _, _ in eager] == [SLAB] * slabs + [128] * long + [32] * math.ceil(short / 32)
    assert all(not anchor and mtp is None for _, _, anchor, mtp in eager)
    resets = [call for call in model.calls if call[0] == "reset"]
    if remaining:
        assert resets[0] == ("reset", 32) and (slabs == 0 or ("reset", SLAB) in resets)
        assert model.calls[-1] == ("finish", start + count)


def test_driver_prepares_the_next_slab_while_the_current_one_replays(monkeypatch) -> None:
    """With slab k queued the host waits for slab k - 1's event, so slab k + 1's inputs are prepared while slab k
    runs; the chunks after keep the every-``event_interval`` sync."""

    log: list[tuple] = []
    events = itertools.count()

    def record_event(mesh, cq_id):
        event = next(events)
        log.append(("record", event))
        return event

    monkeypatch.setattr(
        driver_module,
        "ttnn",
        SimpleNamespace(
            synchronize_device=lambda mesh: log.append(("synchronize",)),
            record_event=record_event,
            event_synchronize=lambda event: log.append(("wait", event)),
            _ttnn_execute_trace=lambda mesh, trace_id, cq_id, blocking: log.append(("replay", trace_id, blocking)),
        ),
    )
    model = _FakeModel()
    model.calls = log
    states = {rows: SimpleNamespace(rows=rows) for rows in (32, 128, SLAB)}
    traces = {"slab": 3, "long": 2, "short": 1}
    count = 3 * SLAB + 5 * 128 + 32 + 8
    tokens = list(range(1000, 1000 + count))
    prefill = driver_module.Qwen38ChunkPrefill(
        model,
        object(),
        object(),
        states[32],
        traces["short"],
        forced_step=lambda token, context: context,
        verify_allocations=False,
        long_chunk_state=states[128],
        long_chunk_trace_id=traces["long"],
        slab_state=states[SLAB],
        slab_trace_id=traces["slab"],
    )
    result = prefill.run(tokens, start_position=0, ple_context=None)

    plan = [("slab", SLAB)] * 3 + [("long", 128)] * 5 + [("short", 32)] * 2
    expected: list[tuple] = [("synchronize",), ("reset", 32), ("reset", 128), ("reset", SLAB)]
    chunks = []
    offset = 0
    for kind, rows in plan:
        chunk = tokens[offset : offset + rows]
        offset += rows
        accepted = len(chunk) - 1 if len(chunk) < rows else None
        chunks.append((kind, rows, chunk + [driver_module.CHUNK_PAD_TOKEN_ID] * (rows - len(chunk)), accepted))
    pending = None
    event = 0
    # The host half of chunk i + 1 (its "write") is prepared right after chunk i's replay is queued, before the wait
    # for the previous slab's event or the chunk event: the host runs one chunk ahead of the device at every cadence.
    expected.append(("write", chunks[0][1], tuple(chunks[0][2])))
    for index, (kind, rows, chunk, accepted) in enumerate(chunks):
        if accepted is not None:
            expected.append(("accepted", accepted))
        expected += [("upload", rows), ("replay", traces[kind], False)]
        if index + 1 < len(chunks):
            expected.append(("write", chunks[index + 1][1], tuple(chunks[index + 1][2])))
        if kind == "slab":
            if pending is not None:
                expected.append(("wait", pending))
            expected.append(("record", event))
            pending, event = event, event + 1
        elif (index + 1) % driver_module.CHUNK_EVENT_INTERVAL == 0:
            expected += [("record", event), ("wait", event)]
            event += 1
    expected += [("synchronize",), ("finish", count), ("synchronize",)]
    assert log == expected
    assert result.position == count and result.timing.slabs == 3 and len(result.timing.slab_wait_ms) == 3
    # The blocking (timed) form keeps every replay blocking and records no event.
    log.clear()
    timed = prefill.run(tokens[: 2 * SLAB], start_position=0, ple_context=None, time_each_chunk=True)
    assert [entry for entry in log if entry[0] in ("replay", "record", "wait")] == [("replay", 3, True)] * 2
    assert len(timed.timing.slab_replay_ms) == 2 and timed.timing.slab_wait_ms == ()


def test_driver_rejects_a_slab_without_the_long_chunks_or_with_mtp(expect_error, driver) -> None:
    states = {rows: SimpleNamespace(rows=rows) for rows in (32, 128, SLAB)}
    with expect_error(ValueError):  # allow-pytest.raises: pure contract test
        driver_module.Qwen38ChunkPrefill(
            _FakeModel(), object(), object(), states[32], None, forced_step=lambda t, c: c, slab_state=states[SLAB]
        )
    with expect_error(ValueError):  # allow-pytest.raises: pure contract test
        driver_module.Qwen38ChunkPrefill(
            _FakeModel(),
            object(),
            object(),
            states[32],
            None,
            forced_step=lambda t, c: c,
            long_chunk_state=states[128],
            slab_state=SimpleNamespace(rows=200),
        )
    with expect_error(ValueError):  # allow-pytest.raises: pure contract test
        driver_module.Qwen38ChunkPrefill(
            _FakeModel(),
            object(),
            object(),
            states[32],
            None,
            forced_step=lambda t, c: c,
            long_chunk_state=states[128],
            slab_state=states[SLAB],
            mtp=object(),
        )
    # a slab with drafting needs the 128-row twin allocated with the same slab form
    with expect_error(ValueError):  # allow-pytest.raises: pure contract test
        driver_module.Qwen38ChunkPrefill(
            _FakeModel(),
            object(),
            object(),
            states[32],
            None,
            forced_step=lambda t, c: c,
            long_chunk_state=states[128],
            slab_state=states[SLAB],
            mtp=object(),
            long_mtp=SimpleNamespace(rows=128, slab_rows=SLAB + 128),
        )
    admitted = driver_module.Qwen38ChunkPrefill(
        _FakeModel(),
        object(),
        object(),
        states[32],
        None,
        forced_step=lambda t, c: c,
        long_chunk_state=states[128],
        slab_state=states[SLAB],
        mtp=object(),
        long_mtp=SimpleNamespace(rows=128, slab_rows=SLAB),
    )
    assert admitted._extension("slab") is admitted.long_mtp and admitted._extension("long") is admitted.long_mtp
    assert driver_module.slab_count(5000, slab_rows=SLAB) == 2 and driver_module.slab_count(5000, slab_rows=None) == 0


# --------------------------------------------------------------------------- the shared MoE combine buffer of a slab state


class _FakeBuffer:
    def __init__(self, shape):
        self.shape = tuple(int(item) for item in shape)


def _moe_buffer_fakes(monkeypatch):
    """moe.allocate_local_combine_output on a fake ttnn: the buffer keeps the requested shape, the mesh contract
    records what it validated."""

    validated = []
    fake_ttnn = SimpleNamespace(
        from_torch=lambda tensor, **kwargs: _FakeBuffer(tensor.shape),
        deallocate=lambda tensor: None,
        ROW_MAJOR_LAYOUT="row-major",
        TILE_LAYOUT="tile",
        TILE_SIZE=32,
        bfloat16="bf16",
        DRAM_MEMORY_CONFIG="dram",
        ShardTensor2dMesh=lambda device, *, mesh_shape, dims: "shard",
    )
    monkeypatch.setattr(moe_module, "ttnn", fake_ttnn)
    monkeypatch.setattr(moe_module, "replicate_tensor_2d_mesh_mapper", lambda device: "replicate", raising=False)
    contract = SimpleNamespace(
        validate_tensor=lambda tensor, placement, **kwargs: validated.append((tensor.shape, placement))
    )
    return SimpleNamespace(ttnn=fake_ttnn, contract=contract, validated=validated)


def test_slab_combine_buffer_admits_the_slab_rows(expect_error, monkeypatch) -> None:
    """The one-call slab writes its whole [10, rows, 2560] page: allocate_local_combine_output must admit the slab
    rows the way the slab's layer instances do; the chunk forms keep their rows and every other count stays refused."""
    fakes = _moe_buffer_fakes(monkeypatch)
    for rows in (CHUNK_ROWS, LONG_CHUNK_ROWS, MIN_SLAB_ROWS, MAX_SLAB_ROWS):
        buffer = moe_module.allocate_local_combine_output("mesh", fakes.contract, rows)
        assert buffer.shape == (moe_module.TOP_K, rows, moe_module.HIDDEN_SIZE)
    validated_rows = [shape[1] for shape, _ in fakes.validated]
    assert validated_rows == [CHUNK_ROWS, LONG_CHUNK_ROWS, MIN_SLAB_ROWS, MAX_SLAB_ROWS], validated_rows
    # counts no form admits: not a lane count (1..MAX_LANES), not a chunk, not a slab
    for rows in (100, LONG_CHUNK_ROWS + 1, 2050, 4096):
        with expect_error(ValueError):  # allow-pytest.raises: pure contract test
            moe_module.allocate_local_combine_output("mesh", fakes.contract, rows)


def _fake_text_model_for_chunk_state(monkeypatch, fakes, *, takes_one_call: bool = True):
    """A stand-in model (a GDN and a QSA layer) on which the real ``Qwen38TTNNTextModel.allocate_chunk_state`` and
    ``Qwen38TTNNDecoderLayer.admit_slab_call`` run; each probe MoE answers ``takes_one_call`` and records its
    construction, admissions and release in ``admitted``."""

    releasable = lambda: SimpleNamespace(deallocate=lambda: None, release=lambda: None, active=True)
    monkeypatch.setattr(
        qsa_module.Qwen38TTNNQSAChunkConstants,
        "build",
        classmethod(
            lambda cls, mesh, contract, blocks, *, rows: SimpleNamespace(
                deallocate=lambda: None, block_tiles=max(1, rows // LONG_CHUNK_ROWS)
            )
        ),
    )
    monkeypatch.setattr(model_module, "ttnn", fakes.ttnn)
    monkeypatch.setattr(model_module, "replicate_tensor_2d_mesh_mapper", lambda device: "replicate", raising=False)

    admitted = []

    def make_layer(layer_type, index):
        attention = SimpleNamespace(
            allocate_rows_constants=lambda rows: releasable(), allocated_compressed_blocks=BLOCKS
        )
        ple = SimpleNamespace(prepare_rows_input=lambda tokens, rows_state, *, rows: releasable())
        streamer = SimpleNamespace(
            layer=lambda layer_index, *, namespace: contextlib.nullcontext(
                (f"w0_w1[{layer_index}]", f"w2[{layer_index}]")
            )
        )
        layer = SimpleNamespace(
            layer_type=layer_type,
            attention=attention,
            ple=ple,
            layer_index=index,
            namespace=SimpleNamespace(value="backbone"),
            expert_streamer=streamer,
        )
        layer.admit_slab_call = lambda rows: layer_module.Qwen38TTNNDecoderLayer.admit_slab_call(layer, rows)

        def chunk_moe(rows, *, local_combine_output, routed_tokens_per_call):
            admitted.append(("probe", index, rows, local_combine_output, routed_tokens_per_call))
            return SimpleNamespace(
                admit_slab_call=lambda w0_w1, w2: admitted.append(("ask", index, w0_w1, w2)) or takes_one_call,
                release_owned_buffers=lambda: admitted.append(("release", index)),
            )

        layer._chunk_moe = chunk_moe
        layer.allocate_chunk_state = (
            lambda rows_constants, *, base, local_combine_output, gdn_body, routed_tokens_per_call: SimpleNamespace(
                attention=SimpleNamespace(),
                ple=SimpleNamespace(),
                local_combine_output=local_combine_output,
                routed_tokens_per_call=routed_tokens_per_call,
            )
        )
        layer.release_chunk_state = lambda state: None
        return layer

    owner = object()
    layers = [make_layer(layer_module.Qwen38TTNNLayerType.GDN, 0), make_layer(layer_module.Qwen38TTNNLayerType.QSA, 1)]
    assert layer_module.PLE_CHECKPOINT_LAYER == 1
    model = SimpleNamespace(
        layers=layers,
        mesh_device="mesh",
        mesh_contract=fakes.contract,
        rope_table=object(),
        qsa_position_constants=object(),
        model_io=SimpleNamespace(embedding=SimpleNamespace(upload_token_rows=lambda rows: _FakeBuffer((1, rows)))),
        _state_owner=owner,
        _validate_generic_state=lambda state: None,
        _validate_chunk_state=lambda state: None,
        # the chunk's host-written RoPE rows and vision feature rows (model.py allocates them beside the token rows)
        _allocate_chunk_rope_rows=lambda rows, block_tiles: SimpleNamespace(deallocate=lambda: None),
        _allocate_chunk_feature_rows=lambda rows: SimpleNamespace(
            tensor=_FakeBuffer((1, 1, rows, 640)), rows=rows, clean=True
        ),
    )
    base = SimpleNamespace(rows=CHUNK_ROWS, layers=(SimpleNamespace(), SimpleNamespace()))
    return model, base, admitted


@pytest.mark.parametrize("takes_one_call", [True, False], ids=["one call", "128-row blocks"])
def test_slab_chunk_state_routes_in_the_form_moe_compute_takes(monkeypatch, takes_one_call: bool) -> None:
    """moe_compute is asked once, through a layer-0 probe, whether it takes the slab in one call; the shared combine
    buffer and every layer's MoE follow the answer: the whole slab per call, or 128-row blocks."""
    fakes = _moe_buffer_fakes(monkeypatch)
    model, base, admitted = _fake_text_model_for_chunk_state(monkeypatch, fakes, takes_one_call=takes_one_call)
    generic_state = SimpleNamespace(position=SimpleNamespace(read=lambda: 0))
    state = model_module.Qwen38TTNNTextModel.allocate_chunk_state(model, generic_state, rows=SLAB, base=base)
    routed = SLAB if takes_one_call else LONG_CHUNK_ROWS
    assert moe_module.routed_tokens_per_call_for(SLAB) == SLAB  # the default, the probe's form
    assert state.rows == SLAB and state.local_combine_output.shape == (moe_module.TOP_K, routed, moe_module.HIDDEN_SIZE)
    assert all(layer_state.local_combine_output is state.local_combine_output for layer_state in state.layers)
    assert [layer_state.routed_tokens_per_call for layer_state in state.layers] == [routed, routed]
    # the combine buffer's validation (the chunk's RoPE rows and feature rows validate their own 4-D shapes too)
    assert [shape[1] for shape, _ in fakes.validated if len(shape) == 3] == [routed]
    assert admitted == [
        ("probe", 0, SLAB, None, None),
        ("ask", 0, "w0_w1[0]", "w2[0]"),
        ("release", 0),
    ], admitted


def test_a_slab_record_replays_only_on_a_server_with_a_prefill_slab() -> None:
    """A server without --prefill-slab skips the slab acceptance record by name; one with a slab replays them all."""

    from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_server as chat_server

    records = [
        {"prompt": "json", "prompt_token_ids": [1] * 85},
        {"prompt": "document", "prompt_token_ids": [1] * 2160},
        {"prompt": "sky", "prompt_token_ids": [1] * 37},
    ]
    assert chat_server.ACCEPTANCE_SLAB_RECORD_TOKENS == 2048
    assert [record["prompt"] for record in chat_server.slab_records_admitted(records, slab_rows=None)] == [
        "json",
        "sky",
    ]
    assert chat_server.slab_records_admitted(records, slab_rows=2048) == records


def test_the_resident_plan_is_the_fused_siblings_of_the_qsa_and_the_moe(expect_error) -> None:
    """A slab build holds one resident fused sibling per QSA layer ([k | v]) and per MoE layer ([gate | up]), in the
    module's dense weight format; the gated residuals and the GDN hold none."""

    for dtype in prefill_dense.SPEC_DTYPES:
        (kv,) = prefill_dense.resident_plan("qsa", dtype)
        (gate_up,) = prefill_dense.resident_plan("moe", dtype)
        assert (kv.name, kv.k, kv.n, kv.dtype, kv.columns) == ("kv", 2560, 512, dtype, prefill_dense.KV_COLUMNS), kv
        assert (gate_up.name, gate_up.k, gate_up.n, gate_up.dtype, gate_up.columns) == (
            "shared_gate_up",
            2560,
            320,
            dtype,
            prefill_dense.SHARED_COLUMNS,
        ), gate_up
        for module in ("gr", "gdn"):
            assert prefill_dense.resident_plan(module, dtype) == (), (module, dtype)
    # the bf8 siblings of the 12 QSA and 48 MoE layers
    bf8 = prefill_dense.plan_bytes_per_device({"qsa": "bf8", "moe": "bf8"})
    (kv,), (gate_up,) = prefill_dense.resident_plan("qsa", "bf8"), prefill_dense.resident_plan("moe", "bf8")
    assert bf8 == 12 * kv.bytes + 48 * gate_up.bytes == 58_490_880, f"{bf8} bytes"
    with expect_error(ValueError, match="prefill dense module must be one of"):
        prefill_dense.resident_plan("mlp")
    with expect_error(ValueError, match="dense weight dtype must be one of"):
        prefill_dense.resident_plan("qsa", "fp32")
