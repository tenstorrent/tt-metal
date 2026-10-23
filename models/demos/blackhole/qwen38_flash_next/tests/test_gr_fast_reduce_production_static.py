# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-device contracts for the qualified production GR partial reduction."""

from __future__ import annotations

import ast
from pathlib import Path

QWEN_ROOT = Path(__file__).resolve().parents[1]
GR_SOURCE = QWEN_ROOT / "ttnn" / "gr.py"
BUILDER_SOURCE = QWEN_ROOT / "ttnn" / "builder.py"


def _attribute_name(node: ast.AST) -> str:
    names: list[str] = []
    while isinstance(node, ast.Attribute):
        names.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        names.append(node.id)
    return ".".join(reversed(names))


def _method(tree: ast.Module, class_name: str, method_name: str) -> ast.FunctionDef:
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name]
    assert len(classes) == 1
    methods = [node for node in classes[0].body if isinstance(node, ast.FunctionDef) and node.name == method_name]
    assert len(methods) == 1
    return methods[0]


def _calls(function: ast.FunctionDef) -> tuple[ast.Call, ...]:
    return tuple(node for node in ast.walk(function) if isinstance(node, ast.Call))


def _keywords(call: ast.Call) -> dict[str, ast.AST]:
    assert all(keyword.arg is not None for keyword in call.keywords)
    return {keyword.arg: keyword.value for keyword in call.keywords if keyword.arg is not None}


def _arguments(call: ast.Call) -> list[str]:
    return [ast.unparse(argument) for argument in call.args]


def test_partial_reduction_is_exactly_qualified_async_gather_then_fast_reduce() -> None:
    tree = ast.parse(GR_SOURCE.read_text(encoding="utf-8"), filename=str(GR_SOURCE))
    function = _method(tree, "Qwen38TTNNGatedResidual", "_all_reduce_partial")
    calls = _calls(function)
    names = [_attribute_name(call.func) for call in calls]

    assert names.count("ttnn.experimental.all_gather_async") == 1
    assert names.count("ttnn.experimental.fast_reduce_nc") == 1
    assert names.count("ttnn.reshape") == 1
    assert "ttnn.all_reduce" not in names

    gather = calls[names.index("ttnn.experimental.all_gather_async")]
    gather_keywords = _keywords(gather)
    assert set(gather_keywords) == {
        "persistent_output_buffer",
        "dim",
        "multi_device_global_semaphore",
        "num_links",
        "cluster_axis",
        "memory_config",
        "topology",
        "barrier_semaphore",
        "chunks_per_sync",
        "num_workers_per_link",
        "num_buffers_per_channel",
    }
    assert isinstance(gather_keywords["persistent_output_buffer"], ast.Constant)
    assert gather_keywords["persistent_output_buffer"].value is None
    assert ast.literal_eval(gather_keywords["dim"]) == 0
    assert _attribute_name(gather_keywords["cluster_axis"]) == "TP_AXIS"
    assert _attribute_name(gather_keywords["memory_config"]) == "ttnn.DRAM_MEMORY_CONFIG"
    assert _attribute_name(gather_keywords["topology"]) == "ttnn.Topology.Linear"
    assert _attribute_name(gather_keywords["num_links"]) == "num_links"
    assert ast.literal_eval(gather_keywords["chunks_per_sync"]) == 1
    assert ast.literal_eval(gather_keywords["num_workers_per_link"]) == 1
    assert ast.literal_eval(gather_keywords["num_buffers_per_channel"]) == 2

    for keyword, method in (
        ("multi_device_global_semaphore", "self.tt_ccl.get_and_cycle_ag_semaphore_handles"),
        ("barrier_semaphore", "self.tt_ccl.get_and_cycle_barrier_semaphore_handle"),
    ):
        semaphore_call = gather_keywords[keyword]
        assert isinstance(semaphore_call, ast.Call)
        assert _attribute_name(semaphore_call.func) == method
        assert len(semaphore_call.args) == 1
        assert _attribute_name(semaphore_call.args[0]) == "TP_AXIS"
        assert not semaphore_call.keywords

    reduce = calls[names.index("ttnn.experimental.fast_reduce_nc")]
    reduce_keywords = _keywords(reduce)
    assert set(reduce_keywords) == {"dims", "output", "compute_kernel_config", "memory_config"}
    assert ast.literal_eval(reduce_keywords["dims"]) == [0]
    assert isinstance(reduce_keywords["output"], ast.Constant)
    assert reduce_keywords["output"].value is None
    assert _attribute_name(reduce_keywords["compute_kernel_config"]) == "self.compute_config"
    assert _attribute_name(reduce_keywords["memory_config"]) == "ttnn.DRAM_MEMORY_CONFIG"


def test_partial_reduction_restores_only_logical_shape_and_releases_gather_after_consumer() -> None:
    tree = ast.parse(GR_SOURCE.read_text(encoding="utf-8"), filename=str(GR_SOURCE))
    function = _method(tree, "Qwen38TTNNGatedResidual", "_all_reduce_partial")
    calls = _calls(function)
    names = [_attribute_name(call.func) for call in calls]
    gather = calls[names.index("ttnn.experimental.all_gather_async")]
    reduce = calls[names.index("ttnn.experimental.fast_reduce_nc")]
    reshape = calls[names.index("ttnn.reshape")]
    releases = [call for call in calls if _attribute_name(call.func) == "_deallocate"]

    assert _arguments(reshape) == ["reduced", "original_shape", "reduced.padded_shape"]
    assert not reshape.keywords
    assert len(releases) == 1
    assert _arguments(releases[0]) == ["gathered"]
    assert gather.lineno < reduce.lineno < reshape.lineno < releases[0].lineno


def test_partial_reduction_is_fail_closed_to_the_single_fused_tile_row_contract() -> None:
    source = GR_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(GR_SOURCE))
    function = _method(tree, "Qwen38TTNNGatedResidual", "_all_reduce_partial")
    function_source = ast.get_source_segment(source, function)
    assert function_source is not None

    assert "expected_shape != PARTIAL_REDUCTION_SHAPE" in function_source
    assert "_padded_shape(tensor) != PARTIAL_REDUCTION_PADDED_SHAPE" in function_source
    assert "tensor.dtype != ttnn.float32" in function_source
    assert "tensor.layout != ttnn.TILE_LAYOUT" in function_source
    assert "tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG" in function_source
    assert "self.collective_topology != ttnn.Topology.Linear" in function_source
    assert "self.tt_ccl is None" in function_source
    assert "self.tt_ccl.get_num_links(TP_AXIS)" in function_source
    assert "type(num_links) is not int or num_links < 1" in function_source
    assert "placement=TensorPlacement.REPLICATED" in function_source
    assert "_padded_shape(reduced) != PARTIAL_REDUCTION_PADDED_SHAPE" in function_source
    assert "reduced.dtype != ttnn.float32" in function_source

    disallowed = {
        "ttnn.synchronize_device",
        "ttnn.ReadDeviceProfiler",
        "ttnn.tracy_message",
        "ttnn.to_torch",
        "ttnn.from_torch",
    }
    names = {_attribute_name(call.func) for call in _calls(function)}
    assert not names & disallowed

    module_source = source.split("class Qwen38TTNNGatedResidual:", 1)[0]
    assert "PARTIAL_WIDTH = RESIDUAL_RANK + 2 * 32" in module_source
    assert "PARTIAL_REDUCTION_SHAPE = (1, 1, 1, PARTIAL_WIDTH)" in module_source
    assert "PARTIAL_REDUCTION_PADDED_SHAPE = (1, 1, 32, PARTIAL_WIDTH)" in module_source
    # The fused weight's N is the partial row itself (the injection tile is its
    # eleventh tile column, the twelfth is zero) and the up weight's K is
    # zero-padded to the same row.
    assert '"down_inject": ((1, 1, FLAT_LOCAL_WIDTH, PARTIAL_WIDTH), ttnn.bfloat16, 2)' in module_source
    assert '"up": ((1, 1, PARTIAL_WIDTH, FLAT_LOCAL_WIDTH), ttnn.bfloat16, 3)' in module_source


def test_read_carries_both_partials_through_one_collective() -> None:
    tree = ast.parse(GR_SOURCE.read_text(encoding="utf-8"), filename=str(GR_SOURCE))
    function = _method(tree, "Qwen38TTNNGatedResidual", "read")
    calls = _calls(function)
    names = [_attribute_name(call.func) for call in calls]

    reductions = [call for call in calls if _attribute_name(call.func) == "self._all_reduce_partial"]
    assert len(reductions) == 1
    assert _arguments(reductions[0]) == ["partial", "PARTIAL_REDUCTION_SHAPE"]
    assert not reductions[0].keywords
    assert "ttnn.all_reduce" not in names
    assert names.count("ttnn.experimental.all_gather_async") == 0

    # One DRAM-sharded decode matmul on the flat (branch, local hidden) row
    # produces the fused FP32 row directly: the down and injection weights are
    # stacked along K and the injection tile is the eleventh N tile, so there
    # is no branch sum and no runtime concat.
    assert names.count("ttnn.sum") == 0
    assert names.count("ttnn.concat") == 0
    assert names.count("ttnn.repeat") == 0
    linears = [call for call in calls if _attribute_name(call.func) == "ttnn.linear"]
    assert [_arguments(call) for call in linears] == [
        ["normalized_ws", "self.weights.down_inject"],
        ["low_rank_ws", "self.weights.up"],
    ]
    fused, up = linears
    fused_keywords = _keywords(fused)
    assert set(fused_keywords) == {"memory_config", "program_config", "dtype", "compute_kernel_config"}
    assert _attribute_name(fused_keywords["memory_config"]) == "ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG"
    assert _attribute_name(fused_keywords["program_config"]) == "self.down_inject_program_config"
    assert _attribute_name(fused_keywords["dtype"]) == "ttnn.float32"
    assert _attribute_name(fused_keywords["compute_kernel_config"]) == "self.compute_config"
    up_keywords = _keywords(up)
    assert set(up_keywords) == {"memory_config", "program_config", "compute_kernel_config"}
    assert _attribute_name(up_keywords["memory_config"]) == "ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG"
    assert _attribute_name(up_keywords["program_config"]) == "self.up_program_config"
    assert _attribute_name(up_keywords["compute_kernel_config"]) == "self.compute_config"

    # Each matmul reads an activation its producer wrote in the width-sharded
    # config (the gamma multiply in _normalize, the SiLU below); the only
    # copies move the matmul outputs back to interleaved DRAM, where the
    # collective contract and the elementwise epilogues live.
    moves = [call for call in calls if _attribute_name(call.func) == "ttnn.to_memory_config"]
    assert [_arguments(call) for call in moves] == [
        ["partial_ws", "ttnn.DRAM_MEMORY_CONFIG"],
        ["up_ws", "ttnn.DRAM_MEMORY_CONFIG"],
    ]
    assert not any(call.keywords for call in moves)

    # One BF16 cast serves both means (the 1/4 lives in the folded gamma); the
    # injection tile is sliced off and the SiLU runs on the whole row, which
    # the K-padded up weight consumes without a slice.
    typecasts = [call for call in calls if _attribute_name(call.func) == "ttnn.typecast"]
    assert len(typecasts) == 1
    assert _arguments(typecasts[0]) == ["reduced", "ttnn.bfloat16"]
    scales = [
        call
        for call in calls
        if _attribute_name(call.func) == "ttnn.multiply"
        and len(call.args) == 2
        and ast.unparse(call.args[1]) == "1.0 / RESIDUAL_BRANCHES"
    ]
    assert scales == []
    slices = [call for call in calls if _attribute_name(call.func) == "ttnn.slice"]
    assert [_arguments(call) for call in slices] == [
        ["reduced_bf16", "(0, 0, 0, RESIDUAL_RANK)", "(1, 1, 1, RESIDUAL_RANK + RESIDUAL_BRANCHES)"],
    ]
    slice_keywords = _keywords(slices[0])
    assert set(slice_keywords) == {"memory_config"}
    assert _attribute_name(slice_keywords["memory_config"]) == "ttnn.DRAM_MEMORY_CONFIG"
    silus = [call for call in calls if _attribute_name(call.func) == "ttnn.silu"]
    assert [_arguments(call) for call in silus] == [["reduced_bf16"]]
    assert _attribute_name(_keywords(silus[0])["memory_config"]) == "self.up_act_memory_config"

    # The gate runs on the flat up row and multiplies the normalized shard in
    # place; the injection epilogue is one fused kernel: sigmoid, then x2.
    sigmoids = [call for call in calls if _attribute_name(call.func) == "ttnn.sigmoid"]
    assert [_arguments(call)[0] for call in sigmoids] == ["up_flat"]
    gates = [
        call
        for call in calls
        if _attribute_name(call.func) == "ttnn.multiply" and _arguments(call) == ["normalized_ws", "gate_flat"]
    ]
    assert len(gates) == 1
    assert _attribute_name(_keywords(gates[0])["memory_config"]) == "ttnn.DRAM_MEMORY_CONFIG"
    doubles = [
        call
        for call in calls
        if _attribute_name(call.func) == "ttnn.multiply" and len(call.args) == 2 and _arguments(call)[1] == "2.0"
    ]
    assert len(doubles) == 1
    assert _arguments(doubles[0]) == ["inject_row", "2.0"]
    double_keywords = _keywords(doubles[0])
    assert set(double_keywords) == {"input_tensor_a_activations", "memory_config"}
    assert ast.unparse(double_keywords["input_tensor_a_activations"]) == (
        "[ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID, 4.0, 0.0)]"
    )

    ordered = (
        fused,
        moves[0],
        reductions[0],
        typecasts[0],
        slices[0],
        silus[0],
        up,
        moves[1],
        sigmoids[0],
        gates[0],
        doubles[0],
    )
    assert [call.lineno for call in ordered] == sorted(call.lineno for call in ordered)

    # The normalized shard lives from _normalize to the gate multiply and is
    # released with the gate; gated_flat is released through its branch-major
    # view; every other operand right after its consumer.
    releases = [_arguments(call) for call in calls if _attribute_name(call.func) == "_deallocate"]
    assert ["gated_flat"] not in releases
    assert releases.count(["gate_flat", "normalized_ws"]) == 1
    for owner in (
        "partial_ws",
        "partial",
        "reduced",
        "reduced_bf16",
        "low_rank_ws",
        "up_ws",
        "up_flat",
        "gated",
        "inject_row",
    ):
        assert releases.count([owner]) == 1


def test_production_gr_uses_the_builder_owned_lazy_ccl_manager() -> None:
    source = BUILDER_SOURCE.read_text(encoding="utf-8")
    lazy = source.split("class _Qwen38LazyTTCCL:", 1)[1].split("\ndef _require_lower_hex", 1)[0]
    build_gr = source.split("    def _build_gr(", 1)[1].split("\n    def ", 1)[0]

    assert "def get_num_links(self, cluster_axis=None):" in lazy
    assert "def get_and_cycle_barrier_semaphore_handle(self, cluster_axis=None):" in lazy
    assert "def get_and_cycle_ag_semaphore_handles(self, cluster_axis=None):" in lazy
    assert "tt_ccl=self.tt_ccl" in build_gr
