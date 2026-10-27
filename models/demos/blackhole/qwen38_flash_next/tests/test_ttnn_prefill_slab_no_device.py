# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The prefill slab (``--prefill-slab ROWS``) without a device: the row-count contract, the QSA slab constants and
chunk-input emulation, the driver's plan (slabs, then 128-row chunks, then 32-row chunks and the tail) and the
2D-multicast matmul config.
"""

from __future__ import annotations

import contextlib
import math
from types import SimpleNamespace

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.tests.helpers.prefill import prefill_context_after
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
from models.demos.blackhole.qwen38_flash_next.ttnn.builder import Qwen38ResidentContext
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
    assert [rows for rows in range(256, 4097, 128) if rows % moe_module.SLAB_REDUCE_BLOCK_ROWS] == sorted(
        set(range(256, 4097, 128)) - set(range(512, 4097, 512))
    )


class _AliasingTensor:
    """A stand-in device tensor: ttnn.slice over the full extent and ttnn.concat of one tensor return their input."""

    def __init__(self, shape, name, dtype=None, layout=None):
        self.shape, self.name, self.dtype, self.layout = tuple(shape), name, dtype, layout


@pytest.mark.parametrize("rows", [512, 1024, 2048])
def test_slab_reduce_never_frees_what_it_did_not_create(monkeypatch, rows: int) -> None:
    """The slab reduce frees only what it created: a full-extent slice and a one-tensor concat return their input
    (as the real ops do)."""
    freed, reduced, marked = [], [], []
    K, H = moe_module.TOP_K, moe_module.HIDDEN_SIZE

    def slice_(tensor, start, end, memory_config):
        if start == (0,) * len(start) and tuple(end) == tensor.shape:
            return tensor
        return _AliasingTensor(
            tuple(e - s for s, e in zip(start, end)), f"{tensor.name}[{start[-2] if len(start) > 2 else start[1]}]"
        )

    def concat(tensors, dim, memory_config):
        if len(tensors) == 1:
            return tensors[0]
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
    routing = SimpleNamespace(
        scores=_AliasingTensor((1, 1, rows, K), "scores"), indices=_AliasingTensor((1, 1, rows, K), "indices")
    )
    owner = SimpleNamespace(
        rows=rows,
        expert_mapping="mapping",
        compute_config="compute",
        mesh_contract=SimpleNamespace(mark_local_partial=lambda tensor, **kwargs: marked.append((tensor.name, kwargs))),
        row_contract=SimpleNamespace(full_hidden=(1, 1, rows, H)),
    )
    partial = moe_module.Qwen38TTNNMoE._weighted_reduce_slab_blocks(
        owner, _AliasingTensor((K, rows, H), "combine"), "full_hidden", routing, lambda phase: None
    )
    blocks = rows // moe_module.SLAB_REDUCE_BLOCK_ROWS
    assert len(reduced) == blocks
    assert partial.shape == (1, 1, rows, moe_module.HIDDEN_SIZE)
    assert marked and marked[0][0] == partial.name
    for name in ("combine", "scores", "indices", partial.name):
        assert name not in freed, (name, freed)
    if blocks == 1:
        assert freed == ["view(tilized(reshape(combine)))"]  # only the tilized stack the method created
        assert partial.name == "partial1"
    else:
        assert freed.count("combine[0]") == 1 and f"combine[{rows - 512}]" in freed
        assert all(f"partial{i}" in freed for i in range(1, blocks + 1)) and partial.name == "concat"
        assert all(f"scores[{s}]" in freed and f"indices[{s}]" in freed for s in range(0, rows, 512))


@pytest.mark.parametrize("rows", [512, 2048])
def test_block_slab_routes_each_128_row_block_through_the_128_row_worker(monkeypatch, rows: int) -> None:
    """A slab the op refuses in one call routes each 128-row block through the 128-row worker; every slice is freed
    after its block and the concatenated partial is marked as the slab's local partial."""
    freed, slices, blocks, marked, phases = [], [], [], [], []
    top_k, hidden_size, block = moe_module.TOP_K, moe_module.HIDDEN_SIZE, LONG_CHUNK_ROWS
    starts = range(0, rows, block)

    def slice_(tensor, start, end, memory_config):
        slices.append((tensor.name, start[2], end[2]))
        extent = tuple(stop - first for first, stop in zip(start, end))
        return _AliasingTensor(extent, f"{tensor.name}[{start[2]}]")

    def concat(tensors, dim, memory_config):
        names = [tensor.name for tensor in tensors]
        assert dim == 2 and names == [f"partial[{first}]" for first in starts], (dim, names)
        return _AliasingTensor((1, 1, sum(tensor.shape[2] for tensor in tensors), hidden_size), "concat")

    def block_partial(hidden, routing, packed_w0_w1, packed_w2):
        shapes = (hidden.shape, routing.scores.shape, routing.indices.shape)
        assert shapes == ((1, 1, block, hidden_size), (1, 1, block, top_k), (1, 1, block, top_k)), shapes
        assert (packed_w0_w1, packed_w2) == ("w0_w1", "w2"), (packed_w0_w1, packed_w2)
        assert routing.tiles is None, routing.tiles
        blocks.append((hidden.name, routing.scores.name, routing.indices.name))
        return _AliasingTensor((1, 1, block, hidden_size), f"partial[{hidden.name.partition('[')[2][:-1]}]")

    monkeypatch.setattr(
        moe_module,
        "ttnn",
        SimpleNamespace(
            DRAM_MEMORY_CONFIG="dram", slice=slice_, concat=concat, deallocate=lambda tensor: freed.append(tensor.name)
        ),
    )
    owner = object.__new__(moe_module.Qwen38TTNNMoE)
    owner.rows, owner.block_instance = rows, SimpleNamespace(_routed_partial=block_partial)
    owner.row_contract = SimpleNamespace(full_hidden=(1, 1, rows, hidden_size))
    owner.mesh_contract = SimpleNamespace(
        validate_tensor=lambda tensor, placement: None,
        mark_local_partial=lambda tensor, **kwargs: marked.append((tensor.name, kwargs["expected_shape"])),
    )
    routing = moe_module.Qwen38TTNNRouting(
        _AliasingTensor((1, 1, rows, top_k), "scores"), _AliasingTensor((1, 1, rows, top_k), "indices")
    )
    hidden = _AliasingTensor((1, 1, rows, hidden_size), "hidden")
    partial = owner._routed_partial_blocks(hidden, routing, "w0_w1", "w2", phases.append)
    names = ("hidden", "scores", "indices")
    assert blocks == [tuple(f"{name}[{first}]" for name in names) for first in starts], blocks
    assert slices == [(name, first, first + block) for first in starts for name in names], slices
    expected_freed = [f"{name}[{first}]" for first in starts for name in names] + [
        f"partial[{first}]" for first in starts
    ]
    assert freed == expected_freed, freed
    assert partial.name == "concat", partial.name
    assert marked == [("concat", (1, 1, rows, hidden_size))], marked
    assert phases[:2] == ["before-routed-dispatch", "after-routed-dispatch"], phases


class _Stub:
    """Any attribute and any call: the device configs the MoE constructor builds and never reads back here."""

    def __getattr__(self, name):
        return _Stub()

    def __call__(self, *args, **kwargs):
        return _Stub()


def _moe_constructor_fakes(monkeypatch, *, fail_free: str | None = None):
    """``Qwen38TTNNMoE.__init__`` on a fake ttnn: every buffer is an ``_AliasingTensor`` named for its owner,
    ``ttnn.deallocate`` records the names it frees (``fail_free`` fails once), the rest is ``_Stub``."""

    freed, failures = [], []
    built = iter(range(1, 100))

    def deallocate(tensor):
        if tensor.name == fail_free and not failures:
            failures.append(tensor.name)
            raise RuntimeError(f"cannot free {tensor.name}")
        freed.append(tensor.name)

    fake_ttnn = _Stub()
    for name, value in {
        "bfloat16": "bf16",
        "uint16": "u16",
        "ROW_MAJOR_LAYOUT": "rm",
        "DRAM_MEMORY_CONFIG": "dram",
        "from_torch": lambda tensor, **kwargs: _AliasingTensor(tensor.shape, f"expert_mapping{next(built)}"),
        "deallocate": deallocate,
    }.items():
        setattr(fake_ttnn, name, value)
    monkeypatch.setattr(moe_module, "ttnn", fake_ttnn)
    monkeypatch.setattr(moe_module, "effective_matmul_ring_size", lambda mesh: 8)
    for name in ("auto_output_width_shard_dim", "replicate_tensor_2d_mesh_mapper"):
        monkeypatch.setattr(moe_module, name, _Stub())
    monkeypatch.setattr(moe_module, "dram_sharded_matmul_configs", lambda *args, **kwargs: (_Stub(), _Stub()))
    monkeypatch.setattr(moe_module, "dense_math_fidelity_name", lambda dtype: "HiFi4")
    monkeypatch.setattr(
        moe_module.Qwen38TTNNPrefillDense, "resolve", classmethod(lambda cls, given, mesh: given or "prefill-dense")
    )
    monkeypatch.setattr(
        moe_module,
        "allocate_local_combine_output",
        lambda mesh, contract, rows: _AliasingTensor(
            (moe_module.TOP_K, rows, moe_module.HIDDEN_SIZE), f"own combine[{rows}]", "bf16", "rm"
        ),
    )

    def build(rows: int, routed_tokens_per_call: int | None = None, local_combine_output=None):
        return moe_module.Qwen38TTNNMoE(
            _Stub(),
            SimpleNamespace(validate_mesh=lambda mesh: None, validate_tensor=lambda tensor, **kwargs: None),
            SimpleNamespace(shared_dtype="bf16", shared_gate_up_scalar=None),
            tt_ccl="ccl",
            rows=rows,
            routed_tokens_per_call=routed_tokens_per_call,
            admitted_rows=moe_module.SUPPORTED_ROWS + ((rows,) if is_slab_rows(rows) else ()),
            local_combine_output=local_combine_output,
        )

    return SimpleNamespace(build=build, freed=freed, failures=failures)


@pytest.mark.parametrize(
    "rows, per_call, expected",
    [
        (512, None, 512),
        (2048, None, 2048),
        (512, 512, 512),
        (512, 128, 128),
        (2048, 128, 128),
        (512, 32, None),
        (2048, 32, None),
        (2048, 256, None),
        (2048, 1024, None),
        (128, None, 128),
        (128, 32, 32),
        (32, 128, None),
    ],
)
def test_a_slab_routes_its_rows_or_128_tokens_per_call_and_never_32(
    monkeypatch, expect_error, rows, per_call, expected
):
    """A slab's ``moe_compute`` call takes the slab's rows (one call) or 128 tokens (one call per 128-row block); 32
    per call, the 128-row chunk's oracle form, and every other count are refused at construction."""
    fakes = _moe_constructor_fakes(monkeypatch)
    if expected is None:
        with expect_error(ValueError, "routed tokens per call must be"):
            fakes.build(rows, per_call)
        return
    module = fakes.build(rows, per_call)
    assert (module.routed_tokens, module.routed_calls) == (expected, rows // expected)
    block_slab = is_slab_rows(rows) and expected == LONG_CHUNK_ROWS
    assert module.block_slab is block_slab and (module.block_instance is not None) is block_slab


def test_a_slab_off_the_512_row_reduce_blocks_is_refused_in_either_form(monkeypatch, expect_error) -> None:
    """768 rows is not a multiple of the 512-row reduce block: refused in one call and in 128-row blocks alike."""
    fakes = _moe_constructor_fakes(monkeypatch)
    for per_call in (None, LONG_CHUNK_ROWS):
        with expect_error(contracts_module.Qwen38SlabRefused, "a slab takes a multiple of 512 rows.* 768 rows is not"):
            fakes.build(768, per_call)


@pytest.mark.parametrize("owned", [False, True], ids=["shared buffer", "own buffer"])
def test_the_128_row_worker_borrows_the_combine_buffer_and_is_released_first(monkeypatch, owned: bool) -> None:
    """The block slab's 128-row worker writes the parent's [10, 128, 2560] combine buffer (shared by every layer's
    slab, or the parent's own) and borrows its weights and prefill dense policy; it owns only its expert mapping.
    ``release_owned_buffers`` releases the worker before the parent; the combine buffer is freed only by the instance
    that allocated it, once: never by the worker, never when it is the shared one."""
    fakes = _moe_constructor_fakes(monkeypatch)
    shared = None if owned else _AliasingTensor((moe_module.TOP_K, 128, moe_module.HIDDEN_SIZE), "shared", "bf16", "rm")
    parent = fakes.build(SLAB, LONG_CHUNK_ROWS, shared)
    worker = parent.block_instance
    assert parent.local_combine_output.name == ("own combine[128]" if owned else "shared")
    assert worker.local_combine_output is parent.local_combine_output
    assert (worker.rows, worker.routed_tokens, worker.block_instance) == (128, 128, None)
    assert worker.weights is parent.weights and worker.prefill_dense is parent.prefill_dense
    assert (parent._owns_local_combine_output, worker._owns_local_combine_output) == (owned, False)
    assert (parent.expert_mapping.name, worker.expert_mapping.name) == ("expert_mapping1", "expert_mapping2")
    parent.release_owned_buffers()
    assert fakes.freed == ["expert_mapping2", "expert_mapping1"] + (["own combine[128]"] if owned else [])
    assert parent.block_instance is None and parent._owned_buffers_released
    assert worker.local_combine_output is None and worker._owned_buffers_released
    one_call = fakes.build(SLAB)
    assert one_call.block_instance is None and one_call.local_combine_output.name == f"own combine[{SLAB}]"


def test_a_failed_worker_release_is_retried_and_the_parent_still_releases(monkeypatch, expect_error) -> None:
    """A worker whose release fails stays attached (named in the error) while the parent releases its own buffers;
    the next call releases only the worker."""
    fakes = _moe_constructor_fakes(monkeypatch, fail_free="expert_mapping2")
    parent = fakes.build(SLAB, LONG_CHUNK_ROWS)
    worker = parent.block_instance
    with expect_error(RuntimeError, "failed to release MoE instance-owned buffer.*block_instance"):
        parent.release_owned_buffers()
    assert fakes.failures == ["expert_mapping2"], fakes.failures
    assert fakes.freed == ["expert_mapping1", "own combine[128]"], fakes.freed
    assert parent.block_instance is worker, parent.block_instance
    parent.release_owned_buffers()
    assert fakes.freed == ["expert_mapping1", "own combine[128]", "expert_mapping2"], fakes.freed
    assert parent.block_instance is None, parent.block_instance


@pytest.mark.parametrize("case", ["admitted", "refused", "inside a capture", "model error", "128-row blocks"])
def test_slab_call_admission_asks_the_op_without_dispatching(monkeypatch, expect_error, case: str) -> None:
    """The slab's one moe_compute call is put to the op under a NO_DISPATCH graph capture before any slab work.

    True when the op takes the call, False on an op refusal (TT_FATAL / TT_THROW), each logged with the slab form it
    selects (a refusal, which falls back to the 128-row blocks, as a warning); any other error stays itself; inside
    another capture, or on a 128-row-block instance, it does not run."""
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
    monkeypatch.setattr(
        moe_module,
        "logger",
        SimpleNamespace(
            info=lambda message: logged.append(("info", message)),
            warning=lambda message: logged.append(("warning", message)),
        ),
    )
    module = object.__new__(moe_module.Qwen38TTNNMoE)
    module.routed_tokens = LONG_CHUNK_ROWS if case == "128-row blocks" else SLAB
    module.rows, module.routed_calls = SLAB, SLAB // module.routed_tokens
    module.row_contract = moe_module.Qwen38TTNNMoERowContract(SLAB, moe_module.SUPPORTED_ROWS + (SLAB,))
    module.mesh_device, module.expert_mapping, module.local_combine_output = "mesh", "mapping", combine
    module.routing_l1_memory_config, module.output_height_shard_dim = "drain-shard", 1
    if case in ("inside a capture", "128-row blocks"):
        error = RuntimeError if case == "inside a capture" else ValueError
        match = (
            "cannot run inside another graph capture"
            if case == "inside a capture"
            else f"one-call slab instance's, not a {SLAB}-row instance's of {LONG_CHUNK_ROWS} tokens per call"
        )
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
    level, message = logged[0]
    assert level == "info" and message.startswith(f"asking moe_compute whether it takes the {SLAB}-row slab"), logged
    if case == "refused":
        assert logged[1:] == [
            (
                "warning",
                f"the {SLAB}-row prefill slab routes its experts in 16 moe_compute calls of 128 tokens per layer: "
                f"moe_compute refuses its one {SLAB}-token call ({fatal})",
            )
        ], logged
    else:
        one_call = f"the {SLAB}-row prefill slab routes its experts in one moe_compute call per layer"
        assert logged[1:] == [("info", one_call)], logged
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


def test_prefill_matmul_config_covers_the_rows_and_columns_with_whole_subblocks(expect_error, monkeypatch) -> None:
    captured = {}
    mesh = SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=11, y=10))
    monkeypatch.setattr(
        decode_matmul_module.ttnn,
        "MatmulMultiCoreReuseMultiCastProgramConfig",
        lambda **kwargs: captured.update(kwargs),
    )
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


STATES = {rows: SimpleNamespace(rows=rows) for rows in (32, 128, SLAB)}  # the chunk states by rows


@pytest.mark.parametrize("start", [0, 5, 32])
@pytest.mark.parametrize("count", [100, 2048, 2048 + 128 + 40, 2 * 2048 + 3 * 128 + 32 + 7, 5000])
def test_driver_runs_slabs_then_long_then_short_chunks(prefill_fakes, start: int, count: int) -> None:
    calls = prefill_fakes.log
    tokens = list(range(1000, 1000 + count))
    prefill = driver_module.Qwen38ChunkPrefill(
        prefill_fakes.model,
        "mesh",
        "state",
        STATES[32],
        None,
        forced_step=prefill_fakes.forced_step,
        long_chunk_state=STATES[128],
        slab_state=STATES[SLAB],
    )
    result = prefill.run(tokens, start_position=start, ple_context=None)
    aligned = driver_module.alignment_steps(start, count)
    remaining = count - aligned
    slabs = remaining // SLAB
    long = (remaining - slabs * SLAB) // 128
    short = math.ceil((remaining - slabs * SLAB - long * 128) / 32)
    plan = [SLAB] * slabs + [128] * long + [32] * short
    writes = [call for call in calls if call[0] == "prepare"]
    assert [call[1] for call in writes] == plan
    assert [token for call in writes for token in call[2]][:remaining] == tokens[aligned:]
    assert [call[1] for call in calls if call[0] == "upload"] == plan
    assert result.position == start + count
    timing = result.timing
    assert (timing.slabs, timing.slab_rows, timing.long_chunks, timing.chunks) == (slabs, SLAB, long, short)
    assert len(timing.slab_prepare_ms) == len(timing.slab_upload_ms) == len(timing.slab_wait_ms) == slabs
    assert all(ms >= 0.0 for ms in timing.slab_prepare_ms + timing.slab_upload_ms + timing.slab_wait_ms)
    eager = [call for call in calls if call[0] == "eager"]
    assert [rows for _, rows, _, _ in eager] == plan
    assert all(not anchor and mtp is None for _, _, anchor, mtp in eager)
    resets = [call for call in calls if call[0] == "reset"]
    if remaining:
        assert resets[0] == ("reset", "state", 32) and (slabs == 0 or ("reset", "state", SLAB) in resets)
        assert calls[-2:] == [("finish", 32, start + count), ("synchronize",)]


def test_driver_prepares_the_next_slab_while_the_current_one_replays(prefill_fakes) -> None:
    """With slab k queued the host waits for slab k - 1's event, so slab k + 1's inputs are prepared while slab k
    runs; the chunks after keep the every-``event_interval`` sync."""

    log = prefill_fakes.log
    traces = {"slab": 3, "long": 2, "short": 1}
    count = 3 * SLAB + 5 * 128 + 32 + 8
    tokens = list(range(1000, 1000 + count))
    prefill = driver_module.Qwen38ChunkPrefill(
        prefill_fakes.model,
        "mesh",
        "state",
        STATES[32],
        traces["short"],
        forced_step=prefill_fakes.forced_step,
        verify_allocations=False,
        long_chunk_state=STATES[128],
        long_chunk_trace_id=traces["long"],
        slab_state=STATES[SLAB],
        slab_trace_id=traces["slab"],
    )
    result = prefill.run(tokens, start_position=0, ple_context=None)

    plan = [("slab", SLAB)] * 3 + [("long", 128)] * 5 + [("short", 32)] * 2
    prepares, chunks, offset = [], [], 0
    for kind, rows in plan:
        chunk = tokens[offset : offset + rows]
        padded = tuple(chunk + [driver_module.CHUNK_PAD_TOKEN_ID] * (rows - len(chunk)))
        prepares.append(("prepare", rows, padded, prefill_context_after(tokens[:offset], None)))
        chunks.append((kind, rows, padded, len(chunk) - 1 if len(chunk) < rows else None))
        offset += rows
    pending, event = None, 0
    # The host half of chunk i + 1 (its "prepare") is prepared right after chunk i's replay is queued, before the wait
    # for the previous slab's event or the chunk event: the host runs one chunk ahead of the device at every cadence.
    expected = [("synchronize",), *(("reset", "state", rows) for rows in (32, 128, SLAB)), prepares[0]]
    for index, (kind, rows, padded, accepted) in enumerate(chunks):
        if accepted is not None:
            expected.append(("accepted", 32, accepted))
        expected += [("upload", rows, padded), ("replay", traces[kind], False)]
        if index + 1 < len(chunks):
            expected.append(prepares[index + 1])
        if kind == "slab":
            if pending is not None:
                expected.append(("wait", pending))
            expected.append(("record", event))
            pending, event = event, event + 1
        elif (index + 1) % driver_module.CHUNK_EVENT_INTERVAL == 0:
            expected += [("record", event), ("wait", event)]
            event += 1
    expected += [("synchronize",), ("finish", 32, count), ("synchronize",)]
    assert log == expected
    assert result.position == count and result.timing.slabs == 3 and len(result.timing.slab_wait_ms) == 3
    # The blocking (timed) form keeps every replay blocking and records no event.
    log.clear()
    timed = prefill.run(tokens[: 2 * SLAB], start_position=0, ple_context=None, time_each_chunk=True)
    assert [entry for entry in log if entry[0] in ("replay", "record", "wait")] == [("replay", 3, True)] * 2
    assert len(timed.timing.slab_replay_ms) == 2 and timed.timing.slab_wait_ms == ()


def test_driver_rejects_a_slab_without_the_long_chunks_or_with_mtp(expect_error, prefill_fakes) -> None:
    def chunk_prefill(**kwargs):
        forms = {"long_chunk_state": STATES[128], "slab_state": STATES[SLAB]} | kwargs
        return driver_module.Qwen38ChunkPrefill(
            prefill_fakes.model, "mesh", "state", STATES[32], None, forced_step=prefill_fakes.forced_step, **forms
        )

    for refused in (
        {"long_chunk_state": None},
        {"slab_state": SimpleNamespace(rows=200)},
        {"mtp": object()},
        # a slab with drafting needs the 128-row twin allocated with the same slab form
        {"mtp": object(), "long_mtp": SimpleNamespace(rows=128, slab_rows=SLAB + 128)},
    ):
        with expect_error(ValueError):  # allow-pytest.raises: pure contract test
            chunk_prefill(**refused)
    admitted = chunk_prefill(mtp=object(), long_mtp=SimpleNamespace(rows=128, slab_rows=SLAB))
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


@pytest.mark.parametrize("takes_one_call", [True, False], ids=["one call", "128-row blocks"])
def test_slab_chunk_state_routes_in_the_form_moe_compute_takes(monkeypatch, takes_one_call: bool) -> None:
    """moe_compute is asked once, through a layer-0 probe, whether it takes the slab in one call; the shared combine
    buffer and every layer's MoE follow the answer: the whole slab per call, or 128-row blocks.  The real
    ``allocate_chunk_state`` and ``Qwen38TTNNDecoderLayer.admit_slab_call`` run on a stand-in model (a GDN and a QSA
    layer) whose probe MoEs record their construction, admissions and release in ``admitted``."""
    fakes = _moe_buffer_fakes(monkeypatch)
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
    generic_state = SimpleNamespace(position=SimpleNamespace(read=lambda: 0))
    state = model_module.Qwen38TTNNTextModel.allocate_chunk_state(model, generic_state, rows=SLAB, base=base)
    routed = SLAB if takes_one_call else LONG_CHUNK_ROWS
    assert moe_module.routed_tokens_per_call_for(SLAB) == SLAB  # the default, the probe's form
    assert state.rows == SLAB and state.local_combine_output.shape == (moe_module.TOP_K, routed, moe_module.HIDDEN_SIZE)
    assert all(layer_state.local_combine_output is state.local_combine_output for layer_state in state.layers)
    assert [layer_state.routed_tokens_per_call for layer_state in state.layers] == [routed, routed]
    assert state.routed_tokens_per_call == routed, state.routed_tokens_per_call  # READY's prefill_slab_routed_tokens
    # the combine buffer's validation (the chunk's RoPE rows and feature rows validate their own 4-D shapes too)
    assert [shape[1] for shape, _ in fakes.validated if len(shape) == 3] == [routed]
    assert admitted == [("probe", 0, SLAB, None, None), ("ask", 0, "w0_w1[0]", "w2[0]"), ("release", 0)], admitted


@pytest.mark.parametrize("admission", ["takes", "refuses", "raises"])
def test_a_failed_probe_release_keeps_the_admission_answer_or_error(admission: str) -> None:
    """The probe's release runs after the admission either way; when it fails, the cleanup error names the answer, or
    carries the admission's own error as its primary (never the release failure alone)."""
    admission_error = RuntimeError("the admission's own error")
    released = []

    def admit(packed_w0_w1, packed_w2):
        if admission == "raises":
            raise admission_error
        return admission == "takes"

    def release():
        released.append("probe")
        raise RuntimeError("cannot free the probe's combine buffer")

    probe = SimpleNamespace(admit_slab_call=admit, release_owned_buffers=release)
    layer = SimpleNamespace(
        layer_index=0,
        namespace=SimpleNamespace(value="backbone"),
        expert_streamer=SimpleNamespace(layer=lambda index, *, namespace: contextlib.nullcontext(("w0_w1", "w2"))),
        _chunk_moe=lambda rows, *, local_combine_output, routed_tokens_per_call: probe,
    )
    with pytest.raises(layer_module.Qwen38TTNNLayerCleanupError) as caught:  # allow-pytest.raises: reads .primary
        layer_module.Qwen38TTNNDecoderLayer.admit_slab_call(layer, SLAB)
    cleanup = caught.value
    assert released == ["probe"], released
    assert "cannot free the probe's combine buffer" in str(cleanup), str(cleanup)
    if admission == "raises":
        assert cleanup.primary is admission_error and cleanup.__cause__ is admission_error, cleanup.primary
    else:
        assert cleanup.primary is None and f"moe_compute {admission} the call" in str(cleanup), str(cleanup)


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


def test_the_slab_working_set_is_the_measured_band_less_the_32k_state_and_the_margin() -> None:
    """The working-set term: the measured free DRAM after the build at the band's upper end, less the 32k context's
    state the builder derives and the margin; 1,036,507,648 B (988.5 MiB) for every slab."""
    context_state = Qwen38ResidentContext(32768).context_state_bytes_per_device
    assert context_state == prefill_dense.SLAB_BAND_CONTEXT_STATE_BYTES == 436_797_440, context_state
    working_set = prefill_dense.SLAB_WORKING_SET_BYTES
    assert working_set == 2_010_176_000 - 436_797_440 - (512 << 20) == 1_036_507_648, working_set
    assert all(prefill_dense.slab_working_set_bytes(rows) == working_set for rows in (256, 512, 2048)), working_set
