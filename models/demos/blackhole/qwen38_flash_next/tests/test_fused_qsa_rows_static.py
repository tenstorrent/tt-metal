# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The qsa_rows family's registry entry and its hooks in the QSA verify path (no device): program 1 (the score
merge over the rows) and program 2 (the main tail with the verify rows' KV stage)."""

from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
from pathlib import Path

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn import fused
from models.demos.blackhole.qwen38_flash_next.ttnn import qsa as qsa_module
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import qsa_block, qsa_rows

# the package rebinds its ``main_tail_rows`` attribute to the function; the module is reached through sys.modules
main_tail_rows_module = importlib.import_module(
    "models.demos.blackhole.qwen38_flash_next.ttnn.fused.qsa_rows.main_tail_rows"
)

MODEL_DIR = Path(__file__).resolve().parents[1]


def test_qsa_rows_is_registered_bitwise_and_default_on() -> None:
    kernel = fused.kernel("qsa_rows")
    assert kernel.tolerance == fused.BITWISE
    assert (
        kernel.default_on
    ), "qsa_rows serves by default since its pass pair, A3 pins and D1 band read clean (2026-09-26)"
    assert kernel.fused is qsa_rows.score_blocks_rows
    assert kernel.composed is qsa_rows.score_blocks_rows_composed
    assert "qsa_rows" in fused.__all__ and fused.qsa_rows is qsa_rows


def test_score_blocks_rows_composes_the_proven_merge() -> None:
    """Program 1 is the decode chain's qsa_score_merge over the tile's rows: the same page form and the same merge."""

    assert qsa_rows.PAGE == qsa_block.SCORE_CHUNK == qsa_module.SCORE_GATHER_PAGE == 1024
    assert qsa_rows.DEVICES == qsa_block.DEVICES == qsa_module.TP_SIZE == 4
    source = inspect.getsource(qsa_rows.score_blocks_rows)
    assert "paged = ttnn.reshape(score_rows, (1, 1, rows * pages, PAGE))" in source
    merge = inspect.getsource(qsa_rows._gather_and_merge)  # the gather + merge both program 1 forms share
    assert "ttnn.all_gather(paged, dim=2, cluster_axis=cluster_axis" in merge
    assert "qsa_block.score_merge(gathered, mask)" in merge
    composed = inspect.getsource(qsa_rows.score_blocks_rows_composed)
    assert "ttnn.all_reduce(score_rows, cluster_axis=cluster_axis" in composed
    assert "fast_and_approximate_mode=False" in composed


def test_the_verify_path_takes_program_one_on_the_single_tile_only() -> None:
    """forward_verify_generic's score step runs the fused form on the 32-row tile with the chunk's row mask; the slab
    (rows > 32, block masks instead of one mask) and the unfused module keep the chain."""

    source = inspect.getsource(qsa_module.Qwen38TTNNQSA._score_blocks_chunk)
    hook = "if self._rows_fused is not None and rows == CHUNK_ROWS and chunk.indexer_neg_mask is not None:"
    assert hook in source
    assert "self._rows_fused.score_blocks_rows(score_rows, chunk.indexer_neg_mask, cluster_axis=TP_AXIS)" in source
    assert source.index(hook) < source.index("scores = ttnn.all_reduce("), "the chain stays as the fallback"
    init = inspect.getsource(qsa_module.Qwen38TTNNQSA.__init__)
    assert 'if fused_kernels.enabled("qsa_rows"):' in init and "self._rows_fused = fused_kernels.qsa_rows" in init
    assert qsa_module.Qwen38TTNNQSA._rows_fused is None


def test_main_tail_rows_is_the_decode_program_with_the_kv_stage() -> None:
    """Program 2 reuses qsa_block.main_tail (its norm / RoPE / head-split / query kernels on the 32-row tile) with the
    staging cores replaced by the verify rows' KV core; the decode form keeps its default (kv_stage None)."""

    assert qsa_rows.main_tail_rows is main_tail_rows_module.main_tail_rows
    assert qsa_rows.main_tail_rows_composed is main_tail_rows_module.main_tail_rows_composed
    assert inspect.signature(qsa_block.main_tail).parameters["kv_stage"].default is None
    source = inspect.getsource(qsa_rows.main_tail_rows)
    assert "kv_stage(position, rows, kv_cache, single_row=single_row" in source and "kv_stage=stage" in source
    for name in ("rows_kv_cbs.h", "rows_kv_reader.cpp", "rows_kv_writer.cpp"):
        assert (MODEL_DIR / "ttnn/fused/qsa_rows/kernels" / name).is_file(), name
    reader = (MODEL_DIR / "ttnn/fused/qsa_rows/kernels/rows_kv_reader.cpp").read_text()
    assert "rows_read < tile_rows::TILE_ROWS ? rows_read : tile_rows::TILE_ROWS" in reader, "R is read on the core"
    assert "slot = (P & 31u) + rows; slot < tile_rows::TILE_ROWS" in reader, "the rows past the pass are zeroed"


def test_the_verify_path_takes_program_two_when_the_pass_carries_its_scalars() -> None:
    """forward_verify_generic runs the fused main tail when the verify inputs carry P (and the constants R); the
    chain stays as the fallback; the two scalars default to None on the inputs and the constants."""

    source = inspect.getsource(qsa_module.Qwen38TTNNQSA.forward_verify_generic)
    hook = "if self._rows_fused is not None and verify.position is not None:"
    assert hook in source and "self._main_tail_rows_step(full_hidden, cos, sin, state, verify, constants)" in source
    assert source.index(hook) < source.index("self._main_projection_rows(full_hidden, None, cos, sin, constants)")
    step = inspect.getsource(qsa_module.Qwen38TTNNQSA._main_tail_rows_step)
    assert "self._rows_fused.main_tail_rows(" in step and "rows=verify.rows_u32" in step
    assert "_retag_tensor(sparse_query, reference=state.packed_kv_cache, shard_dim=1)" in step
    fields = {f.name: f for f in dataclasses.fields(qsa_module.Qwen38TTNNQSAVerifyInputs)}
    assert fields["position"].default is None and fields["rows_u32"].default is None
    constants = {f.name: f for f in dataclasses.fields(qsa_module.Qwen38TTNNQSAVerifyConstants)}
    assert constants["rows_u32"].default is None
    host = qsa_module.qsa_verify_constant_rows(5, 512)
    assert tuple(host["rows_u32"].shape) == (1, 1, 1, 1) and int(host["rows_u32"]) == 5


def test_selection_rows_is_the_decode_selection_program_on_the_tile(expect_error) -> None:
    """Program 3 runs qsa_block.selection_row (device-proven per row at rows 1..32) on the verify tile's 32 rows; the
    verify path takes it after topk_large_indices when the family is on; the chain stays as the fallback."""

    source = inspect.getsource(qsa_rows.selection_rows)
    assert "qsa_block.selection_row(block_ids, sentinel_pad, block_offsets_rows, row_keep_bits, row_fill)" in source
    composed = inspect.getsource(qsa_rows.selection_rows_composed)
    for op in ("bitwise_left_shift", "repeat_interleave", "ttnn.add(", "ttnn.concat(", "bitwise_and", "bitwise_or"):
        assert op in composed, op
    from types import SimpleNamespace

    with expect_error(ValueError, match="32-row block ids"):  # the shape check runs before any device call
        qsa_rows.selection_rows(SimpleNamespace(shape=(1, 1, 5, 512)), None, None, None, None)
    hook = inspect.getsource(qsa_module.Qwen38TTNNQSA._materialize_rows_chunk)
    guard = "if self._rows_fused is not None and rows == CHUNK_ROWS:"
    assert guard in hook and "self._rows_fused.selection_rows(" in hook
    assert hook.index("topk_large_indices") < hook.index(guard) < hook.index("starts = ttnn.bitwise_left_shift(")
    assert "constants.block_offsets_rows, chunk.row_keep_bits, chunk.row_fill" in hook


def test_score_pages_is_a_default_bitwise_data_movement_program() -> None:
    """Program 4 (qsa_score_pages): the indexer's rows repaged into the gather's pages as one program, in place of the
    chain's slice to the resident blocks and the row-major reshape; default since its pass pair (2026-09-26)."""

    kernel = fused.kernel(qsa_rows.PAGES_NAME)
    assert qsa_rows.PAGES_NAME == "qsa_score_pages" and kernel.tolerance == fused.BITWISE
    assert kernel.default_on is True and qsa_rows.PAGES_NAME in fused.DEFAULT_ON
    assert kernel.fused is qsa_rows.score_pages and kernel.composed is qsa_rows.score_pages_composed
    source = inspect.getsource(qsa_rows.score_pages)
    assert "fp.allocate((1, 1, rows * chunks, PAGE), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, mesh)" in source
    assert (
        'fp.program_meta(\n        NAME,\n        "score_pages",' in source
        and "outputs=((out, local_scores),)," in source
    )
    composed = inspect.getsource(qsa_rows.score_pages_composed)
    assert "ttnn.slice(local_scores, (0, 0, 0, 0), (1, 1, rows, blocks), memory_config=dram)" in composed
    assert "ttnn.reshape(sliced, (1, 1, rows * (blocks // PAGE), PAGE))" in composed
    fed = inspect.getsource(qsa_rows.score_blocks_rows_from_scores)
    assert (
        "paged = pages(local_scores, blocks)" in fed
        and "_gather_and_merge(paged, rows, blocks // PAGE, mask, cluster_axis)" in fed
    )
    assert "_gather_and_merge(paged, rows, pages, mask, cluster_axis)" in inspect.getsource(qsa_rows.score_blocks_rows)
    kernel_source = (MODEL_DIR / "ttnn/fused/qsa_rows/kernels/score_pages.cpp").read_text(encoding="utf-8")
    assert (
        kernel_source.count("FUSED_ZONE(") == 2
        and "noc_async_read(src.get_noc_addr(row, chunk * PAGE_BYTES)" in kernel_source
    )
    assert "constexpr uint32_t PAGE_BYTES = 2048;" in kernel_source


def test_the_verify_path_takes_program_four_before_the_slice_when_switched_on() -> None:
    source = inspect.getsource(qsa_module.Qwen38TTNNQSA._score_blocks_chunk)
    hook = (
        "if (\n"
        "                self._score_pages is not None\n"
        "                and rows == CHUNK_ROWS\n"
        "                and chunk.indexer_neg_mask is not None\n"
        "                and self._score_pages_admits(local_scores, self.allocated_compressed_blocks)\n"
        "            ):"
    )
    assert hook in source
    assert (
        "self._rows_fused.score_blocks_rows_from_scores(\n"
        "                    local_scores, chunk.indexer_neg_mask, cluster_axis=TP_AXIS, pages=self._score_pages\n"
        "                )" in source
    )
    assert (
        source.index("indexer_score_dsa(") < source.index(hook) < source.index("score_tiles.append(")
    )  # before the slice
    init = inspect.getsource(qsa_module.Qwen38TTNNQSA.__init__)
    assert "if fused_kernels.enabled(fused_kernels.qsa_rows.PAGES_NAME):" in init
    assert "self._score_pages = fused_kernels.kernel(fused_kernels.qsa_rows.PAGES_NAME).fused" in init
    assert init.index('if fused_kernels.enabled("qsa_rows"):') < init.index("self._score_pages = ")  # inside the family
    assert qsa_module.Qwen38TTNNQSA._score_pages is None


def test_post_attention_rows_is_a_default_bitwise_program_on_48_cores() -> None:
    """Program 5 (qsa_rows_post_attention): the tile's post-attention glue (12 programs) as one program, one core per
    (head, tile column); the decode kernel's two ops per tile; default since its pass pair (2026-09-26)."""

    import importlib

    pa = importlib.import_module("models.demos.blackhole.qwen38_flash_next.ttnn.fused.qsa_rows.post_attention_rows")

    kernel = fused.kernel(qsa_rows.PA_NAME)
    assert qsa_rows.PA_NAME == "qsa_rows_post_attention" and kernel.tolerance == fused.BITWISE
    assert kernel.default_on is True and qsa_rows.PA_NAME in fused.DEFAULT_ON
    assert kernel.fused is qsa_rows.post_attention_rows  # the launch is the family's (its __init__)
    assert kernel.composed is qsa_rows.post_attention_rows_composed is pa.post_attention_rows_composed
    assert len(pa.ITEMS) == 48 == qsa_block.LOCAL_HEADS * qsa_block.HEAD_TILES
    source = inspect.getsource(qsa_rows.post_attention_rows)
    assert "if any(w.count != 1 for w in work):" in source  # one output tile per core
    assert "outputs=((out, 3),)," in source and '"post_attention_rows",' in source
    composed = inspect.getsource(pa.post_attention_rows_composed)
    for op in (
        "ttnn.to_memory_config(qg_ws, dram)",
        "ttnn.concat(gate_heads, dim=1",
        "ttnn.to_layout(local, ttnn.TILE_LAYOUT",
        "ttnn.sigmoid(gate, memory_config=dram)",
        "ttnn.mul(local_tiled, activated_gate",
        "ttnn.concat(heads, dim=3",
    ):
        assert op in composed, op
    kernels = MODEL_DIR / "ttnn/fused/qsa_rows/kernels"
    compute = (kernels / "pa_rows_compute.cpp").read_text(encoding="utf-8")
    decode = (MODEL_DIR / "ttnn/fused/qsa_block/kernels/post_attention_compute.cpp").read_text(encoding="utf-8")
    for op in (
        "sigmoid_tile<VectorMode::RC, false, false>(",
        "mul_binary_tile<false>(",
        "pack_reconfig_data_format(CB_SIG)",
    ):
        assert op in compute and op in decode, op  # the decode kernel's ops, one tile per core
    reader = (kernels / "pa_rows_reader.cpp").read_text(encoding="utf-8")
    assert "qg_first + 2 * HEAD_TILES * head + GATE_FIRST + column" in reader
    assert "tile_rows::chunk_offset(lane, half)" in reader
    writer = (kernels / "pa_rows_writer.cpp").read_text(encoding="utf-8")
    assert sum(k.count("FUSED_ZONE(") for k in (reader, compute, writer)) == 5


def test_the_verify_path_takes_program_five_around_sparse_sdpa_when_switched_on() -> None:
    tail = inspect.getsource(qsa_module.Qwen38TTNNQSA._main_tail_rows_step)
    assert "if self._post_attention_rows is not None:" in tail
    assert tail.index("return sparse_query, None, qg_ws") < tail.index(
        "gate = self._gate_from_qg(qg_ws, rows, full_hidden)"
    )
    attention = inspect.getsource(qsa_module.Qwen38TTNNQSA._sparse_value_attention_rows)
    hook = "if qg_ws is not None:"
    assert hook in attention
    assert (
        attention.index("ttnn.transformer.sparse_sdpa(")
        < attention.index(hook)
        < attention.index("local = ttnn.slice(")
    )
    assert "self._post_attention_rows(sparse_output, qg_ws, memory_config=self.out_act_memory_config)" in attention
    project = inspect.getsource(qsa_module.Qwen38TTNNQSA._project_output_rows)
    assert "in_shard = rows == CHUNK_ROWS and local_attention.memory_config() == self.out_act_memory_config" in project
    assert "if not in_shard:" in project and project.index("if not in_shard:") < project.index(
        "_deallocate(local_attention)"
    )
    init = inspect.getsource(qsa_module.Qwen38TTNNQSA.__init__)
    assert "self._post_attention_rows = fused_kernels.kernel(fused_kernels.qsa_rows.PA_NAME).fused" in init
    assert qsa_module.Qwen38TTNNQSA._post_attention_rows is None


class _Stub:
    """A host stand-in with the fields the admission predicates read (shape, padded shape, dtype, layout)."""

    def __init__(self, shape, padded=None, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT):
        self.shape, self.padded_shape, self.dtype, self.layout = tuple(shape), tuple(padded or shape), dtype, layout


def test_admission_of_programs_four_and_five_serves_the_row_tile_forms_and_leaves_the_rest_to_the_chain() -> None:
    """The standing gate rule (2026-09-26): every fused row program's admission is exercised with the verify-tile
    shapes AND the prefill chunk shapes (32-row, 128-row, slab) AND the one-row draft shape, asserting which form takes
    the program and which the chain.  The 32-row chunk body's six-branch gate (1, 6, 32, 256) is the shape that
    poisoned the warm pass when program 5 was keyed on the shared attention step instead of its own argument."""

    TILE_L = ttnn.TILE_LAYOUT
    qg_tile = _Stub((1, 1, 32, qsa_block.QG_WIDTH), layout=TILE_L)
    qg_one_row = _Stub((1, 1, 1, qsa_block.QG_WIDTH), padded=(1, 1, 32, qsa_block.QG_WIDTH), layout=TILE_L)
    att = lambda rows: _Stub((1, qsa_block.SPARSE_HEADS, rows, qsa_block.HEAD_DIM))  # noqa: E731
    admits = qsa_rows.post_attention_rows_admits
    assert admits(att(32), qg_tile) and admits(att(1), qg_one_row)  # the verify tile and the one-row draft
    chunk_gate = _Stub((1, 6, 32, qsa_block.HEAD_DIM), layout=TILE_L)  # the 32-row chunk body's gate
    assert not admits(att(32), chunk_gate)
    assert not admits(att(128), _Stub((1, 1, 128, qsa_block.QG_WIDTH), layout=TILE_L))  # the 128-row chunk
    assert not admits(att(2048), _Stub((1, 1, 2048, qsa_block.QG_WIDTH), layout=TILE_L))  # the slab
    assert not admits(_Stub((1, 6, 32, qsa_block.HEAD_DIM)), qg_tile)  # a wrong attention form
    assert not admits(att(32), _Stub((1, 1, 32, qsa_block.QG_WIDTH)))  # a ROW_MAJOR qg is not the linear's tile shard
    pages = qsa_rows.score_pages_admits
    assert pages(_Stub((1, 1, 32, 8192 + 32)), 8192) and pages(_Stub((1, 1, 1, 8224)), 8192)
    assert not pages(_Stub((1, 1, 128, 8224)), 8192)  # four tiles at once: the chain
    assert not pages(_Stub((1, 1, 32, 8224), layout=TILE_L), 8192) and not pages(_Stub((1, 1, 32, 8224)), 8000)
    assert not pages(_Stub((1, 1, 32, 8192)), 8224)  # blocks past the rows' width
    for name, predicate in (
        (qsa_rows.PAGES_NAME, qsa_rows.score_pages_admits),
        (qsa_rows.PA_NAME, qsa_rows.post_attention_rows_admits),
    ):
        assert fused.kernel(name).admits is predicate, name


def test_the_qg_shard_travels_as_its_own_argument_and_every_hook_asks_its_predicate() -> None:
    attention = inspect.getsource(qsa_module.Qwen38TTNNQSA._sparse_value_attention_rows)
    assert "qg_ws=None," in attention and "if qg_ws is not None:" in attention
    assert "self._post_attention_rows_admits(sparse_output, qg_ws)" in attention
    assert "gate = self._gate_from_qg(qg_ws, rows, state.packed_kv_cache)" in attention  # the chain's gate otherwise
    tail = inspect.getsource(qsa_module.Qwen38TTNNQSA._main_tail_rows_step)
    assert "return sparse_query, None, qg_ws" in tail and "return sparse_query, gate, None" in tail
    assert "gate = self._gate_from_qg(qg_ws, rows, full_hidden)" in tail
    helper = inspect.getsource(qsa_module.Qwen38TTNNQSA._gate_from_qg)
    assert "_retag_tensor(gate, reference=reference, shard_dim=1)" in helper  # the helper reads no caller's local
    assert "full_hidden" not in helper.split('"""')[-1]  # (the docstring may name it; the body may not)
    verify = inspect.getsource(qsa_module.Qwen38TTNNQSA.forward_verify_generic)
    assert "sparse_query, gate, qg_ws = self._main_tail_rows_step(" in verify
    assert "sparse_query=sparse_query, qg_ws=qg_ws" in verify
    # the chunk body, the lanes and the lane verify never hand a qg shard: their gate stays the chain's
    for method in ("forward_chunk_generic", "forward_decode_lanes", "forward_verify_lanes"):
        source = inspect.getsource(getattr(qsa_module.Qwen38TTNNQSA, method))
        assert "self._sparse_value_attention_rows(query, gate, sparse_indices, state, constants)" in source, method
        assert "qg_ws=" not in source, method
    score = inspect.getsource(qsa_module.Qwen38TTNNQSA._score_blocks_chunk)
    assert "self._score_pages_admits(local_scores, self.allocated_compressed_blocks)" in score


def test_every_fused_kernels_name_in_qsa_is_bound_in_its_scope_and_the_predicates_ride_on_self() -> None:
    """The lever-5 re-fire hold: the hooks named ``fused_kernels`` (a local import of ``__init__``) from other methods
    and raised NameError inside the layer at the first served verify pass.  Every reference to that name must sit in a
    function that imports it; the hooks read the predicates bound on ``self`` in ``__init__``."""

    source = Path(qsa_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for fn in (n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))):
        refs = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Name) and n.id == "fused_kernels"]
        binds = any(
            isinstance(n, ast.ImportFrom) and any((a.asname or a.name) == "fused_kernels" for a in n.names)
            for n in ast.walk(fn)
        )
        assert not refs or binds, f"{fn.name} reads fused_kernels without importing it (lines {refs[:3]})"
    init = inspect.getsource(qsa_module.Qwen38TTNNQSA.__init__)
    for name in ("_score_pages_admits", "_post_attention_rows_admits"):
        assert getattr(qsa_module.Qwen38TTNNQSA, name) is None
        assert f"self.{name} = fused_kernels.qsa_rows." in init, name
    # the switched hooks never call a predicate with the switch off: the handle check comes first
    score = inspect.getsource(qsa_module.Qwen38TTNNQSA._score_blocks_chunk)
    assert score.index("self._score_pages is not None") < score.index("self._score_pages_admits(")
    attention = inspect.getsource(qsa_module.Qwen38TTNNQSA._sparse_value_attention_rows)
    assert attention.index("self._post_attention_rows is not None") < attention.index(
        "self._post_attention_rows_admits("
    )
