# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-device audit for GR's retired rank-320 branch-axis repeat.

This boundary is intentionally different from the hidden-axis repeat whose
TILE-to-ROW_MAJOR/native-repeat/ROW_MAJOR-to-TILE composite stalled.  These
tests pin the TTNN router/kernel facts that made the rank-320 branch repeat a
direct RepeatCodegen operation, and that production no longer issues it: the
DRAM-sharded up matmul consumes the global low-rank row directly.
"""

import ast
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[5]
GR_SOURCE = REPO_ROOT / "models/demos/blackhole/qwen38_flash_next/ttnn/gr.py"
REPEAT_SOURCE = REPO_ROOT / "ttnn/cpp/ttnn/operations/data_movement/repeat/repeat.cpp"
REPEAT_HEADER = REPO_ROOT / "ttnn/cpp/ttnn/operations/data_movement/repeat/repeat.hpp"
CODEGEN_SUPPORT = REPO_ROOT / "ttnn/cpp/ttnn/operations/data_movement/repeat/codegen/repeat_codegen_supported.cpp"
CODEGEN_FACTORY = REPO_ROOT / "ttnn/cpp/ttnn/operations/data_movement/repeat/codegen/repeat_codegen_program_factory.cpp"
CODEGEN_FACTORY_HEADER = (
    REPO_ROOT / "ttnn/cpp/ttnn/operations/data_movement/repeat/codegen/repeat_codegen_program_factory.hpp"
)
REPEAT_INTERLEAVE_SOURCE = REPO_ROOT / "ttnn/cpp/ttnn/operations/data_movement/repeat_interleave/repeat_interleave.cpp"

PINNED_RUNTIME_SOURCE = "933c15b9213acba2bcc76bf7cfb633cbc7089506"
INPUT_LOGICAL_SHAPE = (1, 1, 1, 320)
INPUT_PADDED_SHAPE = (1, 1, 32, 320)
REPEAT_DIMS = (1, 4, 1, 1)
OUTPUT_LOGICAL_SHAPE = (1, 4, 1, 320)
OUTPUT_PADDED_SHAPE = (1, 4, 32, 320)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _method(source: str, name: str, next_name: str) -> str:
    return source.split(f"    def {name}(", 1)[1].split(f"\n    def {next_name}(", 1)[0]


def _tile_codegen_plan(shape: tuple[int, int, int, int], repeat_dim: int, repeats: int) -> tuple[int, int, int]:
    """Transcribe repeat_dim_codegen's TILE page arithmetic for this audit."""

    n, c, h, w = shape
    dim_pages = (n, c, (h + 31) // 32, (w + 31) // 32)
    lower_pages = 1
    for pages in dim_pages[repeat_dim + 1 :]:
        lower_pages *= pages
    rep_dim_pages = dim_pages[repeat_dim]
    total_out_pages = n * c * dim_pages[2] * dim_pages[3] * repeats
    return lower_pages, rep_dim_pages, total_out_pages


def test_production_read_retired_the_branch_repeat_for_the_stacked_up_weight() -> None:
    production = _method(_read(GR_SOURCE), "read", "write")

    # The up weight stacks its four output-branch matrices along N, so the
    # global rank-320 row feeds one DRAM-sharded matmul; the branch repeat
    # audited below is no longer issued.
    assert "ttnn.repeat(" not in production
    assert "low_rank_by_branch" not in production
    ordered = (
        "low_rank_ws = ttnn.silu(reduced_bf16, memory_config=self.up_act_memory_config)",
        "_deallocate(reduced_bf16)",
        "self.mesh_contract.validate_tensor(low_rank_ws, placement=TensorPlacement.REPLICATED)",
        "ttnn.linear(\n            low_rank_ws,\n            self.weights.up",
        "_deallocate(low_rank_ws)",
    )
    positions = [production.index(needle) for needle in ordered]
    assert positions == sorted(positions)


def test_auto_router_selects_codegen_for_rank4_interleaved_tile_upper_dimension() -> None:
    header = _read(REPEAT_HEADER)
    router = _read(REPEAT_SOURCE)
    support = _read(CODEGEN_SUPPORT)

    if 'const std::string& implementation = "auto"' not in header:
        # tt-metal 26b2b80074 ("repeat: drop the implementation selector from the public op", #53323) removed the
        # selector this audit pinned; the router facts below describe the pre-#53323 source (PINNED_RUNTIME_SOURCE).
        pytest.skip(
            "historical audit: the repeat implementation selector is not in this tree (dropped upstream, #53323)"
        )
    assert "const bool codegen_output_ok = !output_mem_config.is_sharded() && placement_matches;" in router
    assert "repeat_codegen::supported_by_codegen(working_tensor, working_repetition_vector)" in router
    assert "supported && !repeat_codegen::is_demoted(working_tensor, working_repetition_vector)" in router
    assert (
        "repeat_via_codegen(\n                    working_tensor, working_repetition_vector, output_mem_config)"
        in router
    )
    assert "if (layout == ttnn::TILE_LAYOUT)" in support
    assert "if (d == ndim - 1)" in support
    assert "if (d == ndim - 2)" in support
    assert "return true;" in support
    assert "bool is_demoted" in support
    assert "return false;" in support.split("bool is_demoted", 1)[1]

    # repeat_dim=1 is an upper (C) dimension, so neither H nor W alignment
    # rejection applies even though logical H is one row.
    repeat_dim = 1
    rank = len(INPUT_LOGICAL_SHAPE)
    assert repeat_dim not in (rank - 2, rank - 1)
    assert INPUT_LOGICAL_SHAPE == (1, 1, 1, 320)
    assert INPUT_PADDED_SHAPE == (1, 1, 32, 320)


def test_exact_tile_page_plan_and_output_shape_are_pinned() -> None:
    router = _read(REPEAT_SOURCE)

    assert "const std::array<uint32_t, 4> dim_pages = {shape4d[0], shape4d[1], Ht, Wt};" in router
    assert "lower_pages *= dim_pages[d];" in router
    assert "rep_dim_pages = dim_pages[rep_dim_4d];" in router
    assert "total_out_pages = dim_pages[0] * dim_pages[1] * dim_pages[2] * dim_pages[3] * repetitions;" in router
    assert "expected_shape[dim] *= repetitions;" in router

    assert _tile_codegen_plan(INPUT_LOGICAL_SHAPE, repeat_dim=1, repeats=4) == (10, 1, 40)
    logical = tuple(size * repeats for size, repeats in zip(INPUT_LOGICAL_SHAPE, REPEAT_DIMS))
    assert logical == OUTPUT_LOGICAL_SHAPE
    assert OUTPUT_PADDED_SHAPE == (1, 4, 32, 320)


def test_codegen_uses_single_repeat_sequencer_program_not_hidden_repeat_composite() -> None:
    factory = _read(CODEGEN_FACTORY)
    factory_header = _read(CODEGEN_FACTORY_HEADER)

    assert "constexpr uint32_t kSeqRepeat = 1;" in factory
    assert (
        '"ttnn/cpp/ttnn/operations/data_movement/common/kernels/codegen/reader_tile_interleaved_unified.cpp"' in factory
    )
    assert '"ttnn/cpp/ttnn/operations/data_movement/common/kernels/codegen/writer_interleaved.cpp"' in factory
    assert '{"seq_id", kSeqRepeat}' in factory
    assert ".total_size = kRepeatCbDepth * page_size" in factory
    assert "inline constexpr uint32_t kRepeatCbDepth = 8;" in factory_header
    assert "to_layout" not in factory
    assert "repeat_interleave" not in factory


def test_dim1_repeat_interleave_is_exact_but_uses_a_distinct_composite() -> None:
    source = _read(REPEAT_INTERLEAVE_SOURCE)
    values = torch.arange(320, dtype=torch.bfloat16).reshape(INPUT_LOGICAL_SHAPE)

    repeated = values.repeat(REPEAT_DIMS)
    interleaved = torch.repeat_interleave(values, repeats=4, dim=1)
    assert repeated.shape == OUTPUT_LOGICAL_SHAPE
    assert interleaved.shape == OUTPUT_LOGICAL_SHAPE
    assert torch.equal(repeated, interleaved)

    if "repeat_codegen" in source:
        # upstream moved repeat_interleave onto the codegen path after this audit; the distinct-composite pin below
        # describes the source at PINNED_RUNTIME_SOURCE.
        pytest.skip("historical audit: repeat_interleave uses the codegen path in this tree")
    assert "rm_input = ttnn::to_layout(rm_input, Layout::ROW_MAJOR);" in source
    assert "auto unsqueezed_tensor = ttnn::unsqueeze(rm_input, normalized_dim + 1);" in source
    assert "auto batch_concat = ttnn::concat(combined_tensors_batch, normalized_dim + 1);" in source
    assert "auto reshaped_tensor = ttnn::reshape(concatenated_tensor, ttnn::Shape(final_shape));" in source
    assert "auto original_layout = ttnn::to_layout(reshaped_tensor, input_a.layout());" in source
    assert "repeat_codegen" not in source


def test_audit_is_source_only_and_records_exact_runtime_provenance() -> None:
    assert PINNED_RUNTIME_SOURCE == "933c15b9213acba2bcc76bf7cfb633cbc7089506"
    tree = ast.parse(_read(Path(__file__)))
    imported_roots = {
        alias.name.split(".", 1)[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
    }
    imported_roots.update(
        node.module.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    called_names = {
        node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    called_names.update(
        node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    )

    assert imported_roots == {"ast", "pathlib", "pytest", "torch"}
    assert called_names.isdisjoint({"open_mesh_device", "MeshDevice", "synchronize_device", "reset"})
