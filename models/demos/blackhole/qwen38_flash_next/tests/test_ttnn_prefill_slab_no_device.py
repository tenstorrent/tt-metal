# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The prefill slab (``--prefill-slab ROWS``) without a device: the row-count contract, the QSA slab constants and
chunk-input emulation, the driver's plan (slabs, then 128-row chunks, then 32-row chunks and the tail), the 2D-multicast
matmul config, and the source pins that keep the slab's forms apart from the 32/128-row bodies (whose own pins hold).
"""

from __future__ import annotations

import inspect
import itertools
import math
import re
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
from models.demos.blackhole.qwen38_flash_next.ttnn import qsa as qsa_module
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import (
    CHUNK_ROW_COUNTS,
    CHUNK_ROWS,
    DEFAULT_SLAB_ROWS,
    LONG_CHUNK_ROWS,
    MAX_SLAB_ROWS,
    MIN_SLAB_ROWS,
    is_slab_rows,
)

SLAB = DEFAULT_SLAB_ROWS
BLOCKS = 8192  # the 32k context's compressed blocks


# --------------------------------------------------------------------------- contracts


def test_slab_row_contract(expect_error, monkeypatch) -> None:
    monkeypatch.setenv(
        moe_module.MOE_SLAB_ONE_CALL_ENV, "0"
    )  # the 128-row blocks' contract; the default is tested below
    assert (MIN_SLAB_ROWS, MAX_SLAB_ROWS, DEFAULT_SLAB_ROWS) == (256, 4096, 2048)
    assert CHUNK_ROW_COUNTS == (32, 128)  # the chunk forms are untouched
    for rows in (256, 512, 1024, 2048, 4096):
        assert is_slab_rows(rows) and contracts_module.chunk_row_tiles(rows) == rows // 32
    for rows in (32, 128, 0, 31, 64, 96, 192, 2000, 8192, True, 2048.0, "2048"):
        assert not is_slab_rows(rows)
    for rows in (0, 31, 64, 96, 192, 8192):
        with expect_error(ValueError):  # allow-pytest.raises: pure contract test
            contracts_module.chunk_row_tiles(rows)
    assert gdn_module.rows_tile_count(SLAB) == SLAB and gdn_module.rows_tile_count(128) == 128
    assert gr_module.residual_rows_shape(SLAB) == (1, 4, SLAB, 640)
    assert moe_module.routed_tokens_per_call_for(SLAB) == 128 == moe_module.routed_tokens_per_call_for(128)
    assert moe_module.routed_tokens_per_call_for(32) == 32 and moe_module.routed_tokens_per_call_for(5) == 5
    contract = moe_module.Qwen38TTNNMoERowContract(SLAB, moe_module.SUPPORTED_ROWS + (SLAB,))
    assert contract.row_tiles == SLAB // 32 and contract.local_combine == (10, SLAB, 2560)
    # 1..32 = decode, the MTP verifier, the chunk and B lanes, 128 = the long chunk; the slab per instance, not globally
    assert moe_module.SUPPORTED_ROWS == (*range(1, 33), 128)
    with expect_error(ValueError):  # allow-pytest.raises: pure contract test
        moe_module.Qwen38TTNNMoERowContract(SLAB)


def test_slab_one_call_switch(expect_error, monkeypatch) -> None:
    """QWEN38_MOE_SLAB_ONE_CALL (unset = 1) makes a slab instance route its rows in one moe_compute call on the local
    output path (height shard 1: nothing is staged) with the fill of unowned rows off, and reduce in 512-row blocks;
    0 restores the 16 x 128-row blocks; the chunk forms are untouched either way."""
    monkeypatch.delenv(moe_module.MOE_SLAB_ONE_CALL_ENV, raising=False)
    assert moe_module.moe_slab_one_call_enabled()
    assert moe_module.routed_tokens_per_call_for(SLAB) == SLAB and moe_module.routed_tokens_per_call_for(4096) == 4096
    assert moe_module.routed_tokens_per_call_for(128) == 128 and moe_module.routed_tokens_per_call_for(32) == 32
    for ring in (7, 8):
        assert moe_module.moe_compute_output_height_shard_dim(SLAB, matmul_ring_size=ring) == 1
    monkeypatch.setenv(moe_module.MOE_SLAB_ONE_CALL_ENV, "1")
    assert moe_module.moe_slab_one_call_enabled() and moe_module.routed_tokens_per_call_for(SLAB) == SLAB
    monkeypatch.setenv(moe_module.MOE_SLAB_ONE_CALL_ENV, "2")
    with expect_error(ValueError):  # allow-pytest.raises: pure contract test
        moe_module.moe_slab_one_call_enabled()
    monkeypatch.setenv(moe_module.MOE_SLAB_ONE_CALL_ENV, "0")
    assert not moe_module.moe_slab_one_call_enabled() and moe_module.routed_tokens_per_call_for(SLAB) == 128
    assert moe_module.routed_tokens_per_call_for(128) == 128 and moe_module.routed_tokens_per_call_for(32) == 32
    monkeypatch.delenv(moe_module.MOE_SLAB_ONE_CALL_ENV)
    # the one-call slab's ring mode: unset = 3 (three rings, the default since 2026-09-26), 0 / 1 / 2 admitted,
    # nothing refused, anything else refused as unknown
    monkeypatch.delenv(moe_module.MOE_SLAB_RINGS_ENV, raising=False)
    assert moe_module.moe_slab_prefill_rings() == 3 == moe_module.MOE_SLAB_RINGS_DEFAULT
    assert moe_module.MOE_SLAB_RINGS_ADMITTED == (0, 1, 2, 3)
    for value in ("0", "1", "2", "3"):
        monkeypatch.setenv(moe_module.MOE_SLAB_RINGS_ENV, value)
        assert moe_module.moe_slab_prefill_rings() == int(value)
    assert moe_module.MOE_SLAB_RINGS_REFUSED == {}
    monkeypatch.setenv(moe_module.MOE_SLAB_RINGS_ENV, "4")
    with expect_error(ValueError):  # allow-pytest.raises: pure contract test
        moe_module.moe_slab_prefill_rings()
    monkeypatch.delenv(moe_module.MOE_SLAB_RINGS_ENV)
    # the kwarg reaches the op from the one-call slab (its ring switch) and from the one-tile rows forms (their
    # QWEN38_MOE_ROWS_FORM table); the 128-row chunk passes None (the op's default)
    partial = inspect.getsource(moe_module.Qwen38TTNNMoE._routed_partial)
    assert "prefill_rings=self.prefill_rings," in partial
    prop = inspect.getsource(moe_module.Qwen38TTNNMoE.prefill_rings.fget)
    assert "if self.slab_one_call:" in prop and "rings = moe_slab_prefill_rings()" in prop
    assert "return rings if rings else None" in prop
    assert 'return MOE_ROWS_FORMS[getattr(self, "rows_form", "fulllocal")][1]' in prop
    assert moe_module.SLAB_REDUCE_BLOCK_ROWS == 512 and SLAB % moe_module.SLAB_REDUCE_BLOCK_ROWS == 0
    # a slab whose rows the 512-row blocks do not divide is refused at construction, not at its first slice
    init = inspect.getsource(moe_module.Qwen38TTNNMoE.__init__)
    assert "if self.slab_one_call and self.rows % SLAB_REDUCE_BLOCK_ROWS:" in init
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
    routed = inspect.getsource(moe_module.Qwen38TTNNMoE._routed_partial)
    assert "zero_fill_non_owned_rows=self.zero_fill_non_owned_rows" in routed
    assert "if self.slab and not self.slab_one_call:" in routed and "if self.slab_one_call:" in routed
    blocks = inspect.getsource(moe_module.Qwen38TTNNMoE._weighted_reduce_slab_blocks)
    assert blocks.count("deepseek_moe_fast_reduce_nc_fused(") == 1 and "range(0, self.rows, block)" in blocks
    assert "scores_tensor=ttnn.reshape(scores, (block, 1, 1, TOP_K))" in blocks
    assert (
        "one_block = self.rows == block" in blocks
        and "_deallocate(stack, *(() if one_block else (pages, scores, indices)))" in blocks
    )


class _AliasingTensor:
    """A stand-in device tensor: ttnn.slice over the full extent and ttnn.concat of one tensor return their input."""

    def __init__(self, shape, name):
        self.shape, self.name = tuple(shape), name


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
    """A 512-row slab is one 512-row block: ttnn.slice over the whole [10, 512, 2560] page returns the page itself
    and ttnn.concat of one partial returns that partial, so the reduce must not free the combine page, the routing's
    scores / indices or the partial it returns (it did before this fix: the persistent combine buffer was released
    under the next layer). Larger slabs slice proper sub-ranges and free every block's slices and partials."""
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


# --------------------------------------------------------------------------- the 2D-multicast matmul config


def test_slab_moe_defaults_when_nothing_is_set(monkeypatch) -> None:
    """The shipping frame: with neither switch set, a slab instance runs the one-call MoE (no zero fill of the rows
    it does not own) on two rings -- prefill_rings 2 on the instance -- and the 128-/32-row chunks keep their forms
    (prefill_rings None: the kwarg is not passed); QWEN38_MOE_SLAB_RINGS=0 restores one ring (None = the op's
    default stream)."""
    monkeypatch.delenv(moe_module.MOE_SLAB_ONE_CALL_ENV, raising=False)
    monkeypatch.delenv(moe_module.MOE_SLAB_RINGS_ENV, raising=False)
    assert moe_module.moe_slab_one_call_enabled() and moe_module.moe_slab_prefill_rings() == 3
    assert moe_module.routed_tokens_per_call_for(SLAB) == SLAB
    assert moe_module.routed_tokens_per_call_for(LONG_CHUNK_ROWS) == LONG_CHUNK_ROWS
    assert moe_module.routed_tokens_per_call_for(CHUNK_ROWS) == CHUNK_ROWS
    one_call = SimpleNamespace(slab_one_call=True)
    blocks = SimpleNamespace(slab_one_call=False)
    assert moe_module.Qwen38TTNNMoE.prefill_rings.fget(one_call) == 3
    assert moe_module.Qwen38TTNNMoE.prefill_rings.fget(blocks) is None  # only the one-call slab reads the switch
    assert moe_module.Qwen38TTNNMoE.zero_fill_non_owned_rows.fget(one_call) is False
    assert moe_module.Qwen38TTNNMoE.zero_fill_non_owned_rows.fget(blocks) is True
    monkeypatch.setenv(moe_module.MOE_SLAB_RINGS_ENV, "0")
    assert moe_module.Qwen38TTNNMoE.prefill_rings.fget(one_call) is None  # one ring = the op's default stream
    assert moe_module.Qwen38TTNNMoE.prefill_rings.fget(blocks) is None
    monkeypatch.setenv(moe_module.MOE_SLAB_RINGS_ENV, "1")
    assert moe_module.Qwen38TTNNMoE.prefill_rings.fget(one_call) == 1
    monkeypatch.setenv(moe_module.MOE_SLAB_RINGS_ENV, "2")
    assert moe_module.Qwen38TTNNMoE.prefill_rings.fget(one_call) == 2
    assert moe_module.Qwen38TTNNMoE.prefill_rings.fget(blocks) is None


def _reshape_is_view(layout: str, source: tuple[int, ...], target: tuple[int, ...]) -> bool:
    """``ttnn.reshape``'s view rule for an interleaved tensor whose buffer type does not change
    (reshape_view/reshape.cpp ``this_is_view``): the last dimension is kept and (ROW_MAJOR, or the second-last
    dimension is kept, or both second-last dimensions are tile multiples); a rank-1 target and an equal shape are views
    too.  Everything else copies."""

    if source == target or len(target) == 1:
        return True
    if source[-1] != target[-1]:
        return False
    return layout == "ROW_MAJOR" or source[-2] == target[-2] or (source[-2] % 32 == 0 and target[-2] % 32 == 0)


# Every reshape whose SOURCE the model frees afterwards: (module, function, source layout, source shape, target shape).
# The predicate at each site is ``same_buffer`` (buffer address), never ``_tensor_key`` (tensor ids, which a view never
# shares).  The view column is today's rule; the predicate keeps every site right if the rule moves.
RESHAPE_FREE_SITES = (
    ("moe", "_routed_partial", "ROW_MAJOR", (1, 1, 2048, 2560), (1, 2048, 2560), True),  # the 2026-09-26 defect
    ("moe", "_routed_local_sum", "ROW_MAJOR", (1, 1, 128, 2560), (1, 128, 2560), True),  # a view of the caller's rows
    ("final_mixer", "rows", "TILE", (1, 2048, 4, 640), (1, 1, 2048, 2560), False),
    ("final_mixer", "rows", "TILE", (1, 1, 2048, 2560), (1, 2048, 4, 640), False),
    ("gr", "read_rows", "TILE", (1, 2048, 4, 640), (1, 1, 2048, 2560), False),
    ("gr", "read_rows", "TILE", (1, 1, 2048, 2560), (1, 2048, 4, 640), False),
    ("mtp", "rows", "TILE", (1, 128, 4, 640), (1, 1, 128, 2560), False),
    ("mtp", "rows", "TILE", (1, 1, 128, 2560), (1, 128, 4, 640), False),
    ("mtp_v2", "forward_verify_head", "ROW_MAJOR", (1, 1, 5, 64), (1, 1, 1, 320), False),
    ("qsa", "_sparse_value_attention", "TILE", (1, 12, 1, 256), (1, 1, 1, 1536), False),
)


def test_a_reshape_source_is_freed_only_when_the_reshape_copied() -> None:
    """The 2026-09-26 slab defect: ``ttnn.reshape`` of the untilized rows is a zero-cost view (a new tensor id over
    the SAME buffer), so the old tensor-id test freed the buffer ``sparse_input`` still read and the op's packed
    token-list page landed on the input's first rows (tokens 0, 8, .., 40 of a fresh launch).  Pinned for every
    reshape-then-free site of the model: the predicate is ``same_buffer`` (by buffer address), the moe_compute source
    is freed before the call only when the reshape copied and never after it, no tensor-id test decides a free, and
    today's view/copy fact of each site under the reshape rule."""
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
    assert contracts._tensor_key(view) != contracts._tensor_key(source)  # the id test that misled the old code
    host_a, host_b = SimpleNamespace(tensor_id=4), SimpleNamespace(tensor_id=5)  # a host stand-in: no buffer to alias
    assert contracts.same_buffer(host_a, host_a) and not contracts.same_buffer(host_a, host_b)
    for module in (moe_module, final_mixer, gr, mtp, mtp_v2, qsa_module):
        assert module.same_buffer is contracts.same_buffer
    # the moe_compute site: reshape -> the guarded free of the source -> the call -> the rows freed once, after it
    routed = inspect.getsource(moe_module.Qwen38TTNNMoE._routed_partial)
    reshape = routed.index("sparse_input = ttnn.reshape(untilized_rows, self.row_contract.moe_sparse_input)")
    guard = routed.index("if not same_buffer(sparse_input, untilized_rows):")
    call = routed.index("outputs = ttnn.experimental.moe_compute(")
    release = routed.index(
        "_deallocate(outputs[0], outputs[1], outputs[2], outputs[4], sparse_input, indices_l1, scores_l1)"
    )
    assert reshape < guard < call < release
    assert (
        "_deallocate(untilized_rows)" in routed[guard : guard + 200]
        and routed.count("_deallocate(untilized_rows)") == 1
    )
    local_sum = inspect.getsource(moe_module.Qwen38TTNNMoE._routed_local_sum)
    assert "if sparse_rows is None or not same_buffer(sparse_input, sparse_rows):" in local_sum
    # every site's source text: same_buffer decides, no tensor-id test does
    functions = {
        ("moe", "_routed_partial"): moe_module.Qwen38TTNNMoE._routed_partial,
        ("moe", "_routed_local_sum"): moe_module.Qwen38TTNNMoE._routed_local_sum,
        ("final_mixer", "rows"): final_mixer.Qwen38TTNNFinalMixer.rows,
        ("gr", "read_rows"): gr.Qwen38TTNNGatedResidual.read_rows,
        ("gr", "read_rows"): gr.Qwen38TTNNGatedResidual.read_rows,
        ("mtp", "rows"): mtp.Qwen38TTNNMTPInput.rows,
        ("mtp_v2", "forward_verify_head"): mtp_v2.forward_verify_head,
        ("qsa", "_sparse_value_attention"): qsa_module.Qwen38TTNNQSA._sparse_value_attention,
    }
    for (module_name, function_name), function in functions.items():
        text = inspect.getsource(function)
        assert "not same_buffer(" in text, (module_name, function_name)
        assert not re.search(
            r"_tensor_key\((\w+)\) != _tensor_key\((\w+)\):\s*\n\s*(_deallocate|ttnn\.deallocate)\(\)", text
        ), (
            module_name,
            function_name,
        )
    # today's view/copy fact per site (the rule as mirrored above): only the two moe_compute sites are views
    for module_name, function_name, layout, source_shape, target_shape, expected_view in RESHAPE_FREE_SITES:
        assert (module_name, function_name) in functions
        assert _reshape_is_view(layout, source_shape, target_shape) is expected_view, (module_name, function_name)
    assert sum(1 for site in RESHAPE_FREE_SITES if site[5]) == 2


def test_slab_moe_switches_are_admitted_before_the_device(expect_error, monkeypatch) -> None:
    """A refused QWEN38_MOE_SLAB_RINGS ends the process at its start: the server admits both slab MoE switches right
    after its --prefill-slab argument checks and before the runtime admission opens a device, and every slab MoE
    instance re-admits them at construction (not at its first forward, 79 s into the warm pass)."""
    from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_server as server_module

    monkeypatch.delenv(moe_module.MOE_SLAB_ONE_CALL_ENV, raising=False)
    monkeypatch.delenv(moe_module.MOE_SLAB_RINGS_ENV, raising=False)
    assert moe_module.admit_slab_moe_switches() == (True, moe_module.MOE_SLAB_RINGS_DEFAULT)
    monkeypatch.setenv(moe_module.MOE_SLAB_RINGS_ENV, "0")
    assert moe_module.admit_slab_moe_switches() == (True, 0)
    monkeypatch.setenv(moe_module.MOE_SLAB_ONE_CALL_ENV, "0")
    monkeypatch.setenv(moe_module.MOE_SLAB_RINGS_ENV, "3")  # the blocks never read the ring switch
    assert moe_module.admit_slab_moe_switches() == (False, 0)
    monkeypatch.setenv(moe_module.MOE_SLAB_ONE_CALL_ENV, "1")
    assert moe_module.admit_slab_moe_switches() == (True, 3)
    monkeypatch.setenv(moe_module.MOE_SLAB_RINGS_ENV, "4")
    with expect_error(ValueError, match="must be one of"):  # allow-pytest.raises: pure contract test
        moe_module.admit_slab_moe_switches()
    main = inspect.getsource(server_module.main)
    admit = main.index("admit_slab_moe_switches()")
    assert "if args.prefill_slab is not None:" in main[:admit]
    assert admit < main.index("runtime_admission.admit_runtime(args)")
    assert "raise SystemExit(str(error)) from error" in main[admit : admit + 200]
    init = inspect.getsource(moe_module.Qwen38TTNNMoE.__init__)
    assert "if self.slab_one_call:\n            admit_slab_moe_switches()" in init


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
    assert tuple(host["row_index_row"].shape) == (1, 1, 1, SLAB) and host["row_index_row"].reshape(-1).tolist() == list(
        range(SLAB)
    )
    assert tuple(host["row_keep_bits" if "row_keep_bits" in host else "row_index_slots"].shape) == (1, 1, SLAB, 2080)
    # The chunk forms are as before.
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
    assert inputs["q_positions_row"].tolist() == [[[list(range(position, position + SLAB))]]]


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
    """The slab cadence: with slab k's replay queued the host waits for slab k - 1's event and records slab k's, so
    slab k + 1's inputs are prepared and their copies queued while slab k runs; the first slab waits for nothing;
    the chunks after the slabs keep the every-``event_interval`` sync; the hand-off synchronizes everything."""

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
    # a slab with drafting needs the 128-row twin allocated with the slab form: the twin of another slab row count
    # is refused, the matching one admitted (2026-09-26)
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


# --------------------------------------------------------------------------- source pins


def test_slab_source_pins() -> None:
    linear = inspect.getsource(decode_matmul_module.prefill_linear)
    # The 2D-multicast program misreads a DRAM-width-sharded in1: the helper always copies the weight interleaved
    # first and never hands the resident weight to the matmul.
    assert linear.index("ttnn.to_memory_config(weight, ttnn.DRAM_MEMORY_CONFIG)") < linear.index("ttnn.linear(")
    assert "weight_interleaved," in linear and "memory_config=ttnn.DRAM_MEMORY_CONFIG" in linear
    for module in (gdn_module, gr_module, moe_module, qsa_module):
        source = inspect.getsource(module)
        assert "prefill_linear(" in source and "prefill_matmul_program_config(" in source
        # Every slab linear goes through the helper: no direct ttnn.linear on a slab branch.
        assert "is_slab_rows" in source
    gdn = inspect.getsource(gdn_module.Qwen38TTNNGDN._causal_conv_rows)
    assert "if is_slab_rows(rows_state.constants.tile_rows):" in gdn and "_shifted_rows_slab" in gdn
    shifted = inspect.getsource(gdn_module.Qwen38TTNNGDN._shifted_rows_slab)
    assert "ttnn.ROW_MAJOR_LAYOUT" in shifted and "ttnn.concat([kept, new], dim=2" in shifted
    fold = inspect.getsource(gdn_module.Qwen38TTNNGDN._gate_and_project_rows)
    assert "elif is_slab_rows(tile_rows):" in fold and fold.index("if tile_rows == CHUNK_SIZE:") < fold.index(
        "elif is_slab_rows(tile_rows):"
    )
    moe_route = inspect.getsource(moe_module.Qwen38TTNNMoE._route)
    assert "self._slab_router_logits(full_hidden) if self.slab else logits_tiles[0]" in moe_route
    router = inspect.getsource(moe_module.Qwen38TTNNMoE._slab_router_logits)
    assert "prefill_linear(" in router and "ttnn.typecast(router_bf16, ttnn.float32" in router
    blocks = inspect.getsource(moe_module.Qwen38TTNNMoE._routed_partial_blocks)
    assert (
        "self.block_instance._routed_partial(rows, Qwen38TTNNRouting(scores, indices), packed_w0_w1, packed_w2)"
        in blocks
    )
    slab_select = inspect.getsource(qsa_module.Qwen38TTNNQSA._sparse_indices_slab)
    assert (
        "ttnn.all_reduce(" in slab_select
        and "ttnn.experimental.topk_large_indices(masked, k=BLOCK_TOPK)" in slab_select
    )
    # Re-pinned with the prefill glue forms: the broadcast comparison moved into slab_block_mask so that the per-slab
    # hoist (qsa_mask_hoist, a default) and the per-layer derivation (``today``, and the fallback when the hoist is
    # not admitted) run one function; the per-layer branch of _sparse_indices_slab still derives it per block after
    # the all-reduce.
    assert "slab_block_mask(chunk.complete_blocks_col, constants.arange_blocks_row, start, mask_value)" in slab_select
    block_mask = inspect.getsource(qsa_module.slab_block_mask)
    assert "ttnn.ge(arange_blocks_row, complete_col" in block_mask and "ttnn.multiply(invalid, value" in block_mask
    assert slab_select.index("ttnn.all_reduce(") < slab_select.rindex("slab_block_mask(")
    assert qsa_module.SLAB_SCORE_BLOCK_ROWS == 512
    forward = inspect.getsource(qsa_module.Qwen38TTNNQSA.forward_chunk_generic)
    assert (
        "if is_slab_rows(rows):" in forward
        and "self._sparse_indices_slab(index_query, state, chunk, constants, expand=not block_shared)" in forward
    )
    # The block-shared attention (QWEN38_FUSED=sparse_sdpa_tiled) ends the selection at the block ids and takes the
    # slab's positions row; the chain keeps the expansion.  Its admission is per slab on the shapes.
    assert "block_shared = self._slab_attention_admits(rows, state)" in forward
    assert "self._block_shared_attention_rows(" in forward and "chunk.q_positions_row" in forward
    attention = inspect.getsource(qsa_module.Qwen38TTNNQSA._block_shared_attention_rows)
    assert "self._slab_attention_fused(" in attention and "ttnn.to_layout(query, ttnn.ROW_MAJOR_LAYOUT" in attention
    select = inspect.getsource(qsa_module.Qwen38TTNNQSA._sparse_indices_slab)
    assert "if not expand:" in select and select.index("if not expand:") < select.index(
        "ttnn.bitwise_left_shift(block_ids"
    )
    # The 32-row and 128-row bodies keep their forms (their own pins hold): the per-tile loops are still there.
    assert "dram_sharded_row_tiles(full_hidden, self.in_proj_act_memory_config)" in inspect.getsource(gdn_module)
    assert "def _routed_partial_tiles" in inspect.getsource(moe_module)


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
    for rows in (CHUNK_ROWS, LONG_CHUNK_ROWS, SLAB, MAX_SLAB_ROWS):
        buffer = moe_module.allocate_local_combine_output("mesh", fakes.contract, rows)
        assert buffer.shape == (moe_module.TOP_K, rows, moe_module.HIDDEN_SIZE)
    assert [shape[1] for shape, _ in fakes.validated] == [CHUNK_ROWS, LONG_CHUNK_ROWS, SLAB, MAX_SLAB_ROWS]
    # counts no form admits: not a lane count (the lanes lineage admits 1..MAX_LANES), not a chunk, not a slab
    for rows in (100, LONG_CHUNK_ROWS + 1, 2050):
        with expect_error(ValueError):  # allow-pytest.raises: pure contract test
            moe_module.allocate_local_combine_output("mesh", fakes.contract, rows)


def _fake_text_model_for_chunk_state(monkeypatch, fakes):
    """A stand-in for Qwen38TTNNTextModel with the collaborators allocate_chunk_state touches (two layers: a GDN and
    a QSA, the PLE checkpoint layer at index 1), so the REAL Qwen38TTNNTextModel.allocate_chunk_state runs its own
    body, including the shared combine buffer allocation through moe.allocate_local_combine_output."""

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

    def make_layer(layer_type):
        attention = SimpleNamespace(
            allocate_rows_constants=lambda rows: releasable(), allocated_compressed_blocks=BLOCKS
        )
        ple = SimpleNamespace(prepare_rows_input=lambda tokens, rows_state, *, rows: releasable())
        layer = SimpleNamespace(layer_type=layer_type, attention=attention, ple=ple)
        layer.allocate_chunk_state = lambda rows_constants, *, base, local_combine_output, gdn_body: SimpleNamespace(
            attention=SimpleNamespace(), ple=SimpleNamespace(), local_combine_output=local_combine_output
        )
        layer.release_chunk_state = lambda state: None
        return layer

    owner = object()
    layers = [make_layer(layer_module.Qwen38TTNNLayerType.GDN), make_layer(layer_module.Qwen38TTNNLayerType.QSA)]
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
    return model, base


@pytest.mark.parametrize("one_call", [False, True], ids=["blocks", "one-call"])
def test_slab_chunk_state_allocates_the_combine_buffer_the_slab_writes(monkeypatch, one_call: bool) -> None:
    """The chain opens a slab state through Qwen38TTNNTextModel.allocate_chunk_state(state, rows=slab, base=chunk):
    the shared MoE combine buffer it allocates is sized by routed_tokens_per_call_for(rows) -- the whole slab by
    default (the one call's page), 128 rows under QWEN38_MOE_SLAB_ONE_CALL=0 (the 128-row blocks) -- and must be
    admitted by the row contract either way (the line gate of 2026-09-25 hit 'MoE rows must be exactly one of
    (1, 5, 32, 128), got 2048' here)."""
    if one_call:
        monkeypatch.delenv(moe_module.MOE_SLAB_ONE_CALL_ENV, raising=False)  # the default
    else:
        monkeypatch.setenv(moe_module.MOE_SLAB_ONE_CALL_ENV, "0")
    fakes = _moe_buffer_fakes(monkeypatch)
    model, base = _fake_text_model_for_chunk_state(monkeypatch, fakes)
    generic_state = SimpleNamespace(position=SimpleNamespace(read=lambda: 0))
    state = model_module.Qwen38TTNNTextModel.allocate_chunk_state(model, generic_state, rows=SLAB, base=base)
    expected_rows = SLAB if one_call else LONG_CHUNK_ROWS
    assert moe_module.routed_tokens_per_call_for(SLAB) == expected_rows
    assert state.rows == SLAB and state.local_combine_output.shape == (
        moe_module.TOP_K,
        expected_rows,
        moe_module.HIDDEN_SIZE,
    )
    assert all(layer_state.local_combine_output is state.local_combine_output for layer_state in state.layers)
    # the combine buffer's validation (the chunk's RoPE rows and feature rows validate their own 4-D shapes too)
    assert [shape[1] for shape, _ in fakes.validated if len(shape) == 3] == [expected_rows]
    # the model allocates that buffer with exactly this expression (the chain's path), and moe admits slab rows there
    allocate = inspect.getsource(model_module.Qwen38TTNNTextModel.allocate_chunk_state)
    assert (
        "moe_module.allocate_local_combine_output(\n"
        "                    self.mesh_device, self.mesh_contract, moe_module.routed_tokens_per_call_for(rows)\n"
        "                )" in allocate
    )
    combine = inspect.getsource(moe_module.allocate_local_combine_output)
    assert "admitted_rows=SUPPORTED_ROWS + ((rows,) if is_slab_rows(rows) else ())" in combine


def test_a_slab_record_replays_only_on_a_server_with_a_prefill_slab() -> None:
    """An acceptance record of 2048 or more prompt tokens (the slab record) exists to run a slab: a server without
    --prefill-slab skips it by name and keeps every other record in order; a server with a slab replays them all.
    main() applies the rule right after loading the records."""

    import inspect

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
    assert chat_server.slab_records_admitted(records, slab_rows=4096) == records
    source = inspect.getsource(chat_server.main)
    assert "records = slab_records_admitted(records, slab_rows=args.prefill_slab)" in source
