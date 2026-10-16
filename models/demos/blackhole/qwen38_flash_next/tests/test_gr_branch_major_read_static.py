# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-device contracts for the permute-free branch-major GR read path."""

from __future__ import annotations

import ast
from pathlib import Path

QWEN_ROOT = Path(__file__).resolve().parents[1]
GR_SOURCE = QWEN_ROOT / "ttnn" / "gr.py"
FINAL_MIXER_SOURCE = QWEN_ROOT / "ttnn" / "final_mixer.py"
LAYER_SOURCE = QWEN_ROOT / "ttnn" / "layer.py"
MODEL_SOURCE = QWEN_ROOT / "ttnn" / "model.py"


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


def test_gr_read_is_permute_free_and_reduces_branches_with_fast_reduce_nc() -> None:
    tree = ast.parse(GR_SOURCE.read_text(encoding="utf-8"), filename=str(GR_SOURCE))
    function = _method(tree, "Qwen38TTNNGatedResidual", "read")
    calls = _calls(function)
    names = [_attribute_name(call.func) for call in calls]

    assert names.count("ttnn.permute") == 0
    assert names.count("ttnn.mean") == 0
    assert names.count("ttnn.experimental.fast_reduce_nc") == 1

    reduce = calls[names.index("ttnn.experimental.fast_reduce_nc")]
    reduce_keywords = _keywords(reduce)
    assert set(reduce_keywords) == {"dims", "output", "compute_kernel_config", "memory_config"}
    assert ast.literal_eval(reduce_keywords["dims"]) == [1]
    assert isinstance(reduce_keywords["output"], ast.Constant)
    assert reduce_keywords["output"].value is None
    assert _attribute_name(reduce_keywords["compute_kernel_config"]) == "self.compute_config"
    assert _attribute_name(reduce_keywords["memory_config"]) == "ttnn.DRAM_MEMORY_CONFIG"
    assert [ast.unparse(argument) for argument in reduce.args] == ["gated"]

    # fast_reduce_nc reports the 32-row tile padding as its logical row count;
    # the metadata-only reshape restores the logical shape and is the block
    # input itself: the 1/4 branch mean is folded into the RMS gamma, so no
    # scale follows the sum.  It is the only reshape in read.
    reshapes = [call for call in calls if _attribute_name(call.func) == "ttnn.reshape"]
    assert [ast.unparse(call.args[0]) for call in reshapes] == ["gated_sum"]
    restore = reshapes[0]
    assert [ast.unparse(argument) for argument in restore.args] == [
        "gated_sum",
        "BLOCK_LOCAL_SHAPE",
        "gated_sum.padded_shape",
    ]
    assert not restore.keywords
    assert reduce.lineno < restore.lineno
    scales = [
        call
        for call in calls
        if _attribute_name(call.func) == "ttnn.multiply"
        and len(call.args) == 2
        and ast.unparse(call.args[1]) == "1.0 / RESIDUAL_BRANCHES"
    ]
    assert scales == []

    # The gate multiplies the normalized residual in the down+inject activation
    # shard directly (flat row, no permute); the branch reduce reads the
    # product through its branch-major view.
    gate_multiplies = [
        call
        for call in calls
        if _attribute_name(call.func) == "ttnn.multiply"
        and [ast.unparse(argument) for argument in call.args] == ["normalized_ws", "gate_flat"]
    ]
    assert len(gate_multiplies) == 1

    # The decode matmuls read the same tiles as one (branch, local hidden) row
    # (the view into the row lives in _normalize): zero-copy views, never a
    # permute, a batched projection or a branch repeat.
    views = [call for call in calls if _attribute_name(call.func) == "ttnn.experimental.view"]
    assert [[ast.unparse(argument) for argument in call.args] for call in views] == [
        ["gated_flat", "RESIDUAL_LOCAL_SHAPE"],
    ]
    assert not any(call.keywords for call in views)
    linears = [call for call in calls if _attribute_name(call.func) == "ttnn.linear"]
    assert [[ast.unparse(argument) for argument in call.args] for call in linears] == [
        ["normalized_ws", "self.weights.down_inject"],
        ["low_rank_ws", "self.weights.up"],
    ]
    assert names.count("ttnn.repeat") == 0
    assert names.count("ttnn.sum") == 0


def test_gr_normalize_and_write_are_branch_major() -> None:
    source = GR_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(GR_SOURCE))

    normalize = _method(tree, "Qwen38TTNNGatedResidual", "_normalize")
    normalize_source = ast.get_source_segment(source, normalize)
    assert normalize_source is not None
    assert "ttnn.reshape(stats, (1, RESIDUAL_BRANCHES, 1, 32))" in normalize_source
    post_calls = [call for call in _calls(normalize) if _attribute_name(call.func) == "ttnn.rms_norm_post_all_gather"]
    assert len(post_calls) == 1
    # The per-branch scale must stay unfused: the fused gamma kernel broadcasts
    # one tile row-block over every branch block of a branch-major input.  It
    # runs on the flat row (the same tile pages) so that it can write the
    # down+inject activation shard, which is defined on that row.
    assert "weight" not in _keywords(post_calls[0])
    views = [call for call in _calls(normalize) if _attribute_name(call.func) == "ttnn.experimental.view"]
    assert [[ast.unparse(argument) for argument in call.args] for call in views] == [["unit", "FLAT_LOCAL_SHAPE"]]
    assert (
        "ttnn.multiply(\n            ttnn.experimental.view(unit, FLAT_LOCAL_SHAPE),\n            self.norm_scale_flat,"
        in normalize_source
    )

    write = _method(tree, "Qwen38TTNNGatedResidual", "write")
    write_source = ast.get_source_segment(source, write)
    assert write_source is not None
    assert "ttnn.reshape(state.injection, (1, RESIDUAL_BRANCHES, 1, 1))" in write_source
    assert "ttnn.permute(" not in write_source

    module_source = source.split("class Qwen38TTNNGatedResidualWeights", 1)[0]
    assert "RESIDUAL_LOCAL_SHAPE = (1, RESIDUAL_BRANCHES, 1, LOCAL_HIDDEN_SIZE)" in module_source


def test_final_mixer_repeats_the_permute_free_branch_major_read() -> None:
    source = FINAL_MIXER_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(FINAL_MIXER_SOURCE))

    call_method = _method(tree, "Qwen38TTNNFinalMixer", "__call__")
    calls = _calls(call_method)
    names = [_attribute_name(call.func) for call in calls]
    assert names.count("ttnn.permute") == 0
    assert names.count("ttnn.mean") == 0
    assert names.count("ttnn.experimental.fast_reduce_nc") == 1

    # Same padded-row contract as the GR read: restore the logical shape right
    # after the branch reduce; the 1/4 branch mean lives in the folded gamma,
    # so the restored sum is the output.
    reduce = calls[names.index("ttnn.experimental.fast_reduce_nc")]
    reshapes = [call for call in calls if _attribute_name(call.func) == "ttnn.reshape"]
    assert len(reshapes) == 1
    restore = reshapes[0]
    assert [ast.unparse(argument) for argument in restore.args] == [
        "gated_sum",
        "OUTPUT_LOCAL_SHAPE",
        "gated_sum.padded_shape",
    ]
    assert not restore.keywords
    assert reduce.lineno < restore.lineno
    scales = [
        call
        for call in calls
        if _attribute_name(call.func) == "ttnn.multiply"
        and len(call.args) == 2
        and ast.unparse(call.args[1]) == "1.0 / RESIDUAL_BRANCHES"
    ]
    assert scales == []

    # Same decode matmuls as the GR read: the gamma multiply writes the flat
    # row into the down activation shard (its view lives in _normalize), the
    # gate product comes back through the branch-major view; stacked weights,
    # no branch repeat and no branch sum.
    views = [call for call in calls if _attribute_name(call.func) == "ttnn.experimental.view"]
    assert [[ast.unparse(argument) for argument in call.args] for call in views] == [
        ["gated_flat", "RESIDUAL_LOCAL_SHAPE"],
    ]
    linears = [call for call in calls if _attribute_name(call.func) == "ttnn.linear"]
    assert [[ast.unparse(argument) for argument in call.args] for call in linears] == [
        ["normalized_ws", "self.weights.down"],
        ["low_rank_ws", "self.weights.up"],
    ]
    gate_multiplies = [
        call
        for call in calls
        if _attribute_name(call.func) == "ttnn.multiply"
        and [ast.unparse(argument) for argument in call.args] == ["normalized_ws", "gate_flat"]
    ]
    assert len(gate_multiplies) == 1
    assert names.count("ttnn.repeat") == 0
    assert names.count("ttnn.sum") == 0

    normalize = _method(tree, "Qwen38TTNNFinalMixer", "_normalize")
    normalize_source = ast.get_source_segment(source, normalize)
    assert normalize_source is not None
    assert "ttnn.reshape(stats, (1, RESIDUAL_BRANCHES, 1, 32))" in normalize_source
    post_calls = [call for call in _calls(normalize) if _attribute_name(call.func) == "ttnn.rms_norm_post_all_gather"]
    assert len(post_calls) == 1
    assert "weight" not in _keywords(post_calls[0])
    assert (
        "ttnn.multiply(\n            ttnn.experimental.view(unit, FLAT_LOCAL_SHAPE),\n            self.norm_scale_flat,"
        in normalize_source
    )

    assert "RESIDUAL_LOCAL_SHAPE = (1, RESIDUAL_BRANCHES, 1, LOCAL_HIDDEN_SIZE)" in source


def test_layer_permutes_only_at_the_once_per_token_ple_boundary() -> None:
    source = LAYER_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(LAYER_SOURCE))
    classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    permute_owners: list[str] = []
    for class_node in classes:
        for function in class_node.body:
            if not isinstance(function, ast.FunctionDef):
                continue
            for call in _calls(function):
                if _attribute_name(call.func) == "ttnn.permute":
                    permute_owners.append(f"{class_node.name}.{function.name}")
                    assert ast.literal_eval(call.args[1]) == (0, 2, 1, 3)
    assert permute_owners == ["Qwen38TTNNDecoderLayer._apply_ple", "Qwen38TTNNDecoderLayer._apply_ple"]
    assert "RESIDUAL_LOCAL_SHAPE = (1, RESIDUAL_BRANCHES, 1, LOCAL_HIDDEN_SIZE)" in source


def test_model_builds_the_initial_residual_branch_major() -> None:
    source = MODEL_SOURCE.read_text(encoding="utf-8")
    embed_residual = source.split("    def _embed_residual(", 1)[1].split("\n    def ", 1)[0]
    assert "residual = ttnn.repeat_interleave(" in embed_residual
    assert "repeats=RESIDUAL_BRANCHES" in embed_residual
    assert "dim=1" in embed_residual
    assert "dim=2" not in embed_residual
