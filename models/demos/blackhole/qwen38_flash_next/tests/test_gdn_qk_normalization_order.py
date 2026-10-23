# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""No-device source contracts for Qwen3.8 GDN Q/K normalization order."""

from __future__ import annotations

import ast
from pathlib import Path

QWEN38_GDN_SOURCE = Path(__file__).parents[1] / "ttnn" / "gdn.py"
SHARED_DELTA_RULE_SOURCE = (
    Path(__file__).parents[4] / "experimental" / "gated_attention_gated_deltanet" / "tt" / "ttnn_delta_rule_ops.py"
)


def _function(source_path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    return next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)


def _assigns_call(node: ast.AST, target: str, function: str) -> bool:
    return any(
        isinstance(child, ast.Assign)
        and len(child.targets) == 1
        and isinstance(child.targets[0], ast.Name)
        and child.targets[0].id == target
        and isinstance(child.value, ast.Call)
        and ast.unparse(child.value.func) == function
        for child in ast.walk(node)
    )


def test_shared_decode_defaults_to_legacy_promote_then_normalize_order() -> None:
    function = _function(SHARED_DELTA_RULE_SOURCE, "recurrent_gated_delta_rule_decode_ttnn")
    argument_names = [argument.arg for argument in function.args.args]
    flag_index = argument_names.index("normalize_qk_in_input_dtype")
    default_offset = len(argument_names) - len(function.args.defaults)
    flag_default = function.args.defaults[flag_index - default_offset]
    assert isinstance(flag_default, ast.Constant) and flag_default.value is False

    conditionals = [statement for statement in function.body if isinstance(statement, ast.If)]
    normalize_before = next(
        statement
        for statement in conditionals
        if isinstance(statement.test, ast.Name) and statement.test.id == "normalize_qk_in_input_dtype"
    )
    promote = next(
        statement
        for statement in conditionals
        if isinstance(statement.test, ast.Name) and statement.test.id == "high_precision"
    )
    normalize_after = next(
        statement
        for statement in conditionals
        if isinstance(statement.test, ast.UnaryOp)
        and isinstance(statement.test.op, ast.Not)
        and isinstance(statement.test.operand, ast.Name)
        and statement.test.operand.id == "normalize_qk_in_input_dtype"
    )

    assert function.body.index(normalize_before) < function.body.index(promote) < function.body.index(normalize_after)
    assert _assigns_call(promote, "q", "ttnn.typecast")
    assert _assigns_call(promote, "k", "ttnn.typecast")
    assert _assigns_call(normalize_before, "q", "l2_norm_ttnn")
    assert _assigns_call(normalize_before, "k", "l2_norm_ttnn")
    assert _assigns_call(normalize_after, "q", "l2_norm_ttnn")
    assert _assigns_call(normalize_after, "k", "l2_norm_ttnn")


def test_qwen38_decode_normalizes_qk_in_bf16_before_the_fp32_promotion() -> None:
    """The in-module step keeps the input-dtype order: l2 norm in BF16, then promote q and k only."""

    method = _function(QWEN38_GDN_SOURCE, "_recurrent_decode")
    statements = [statement for statement in method.body if isinstance(statement, ast.Assign)]

    def index_of(target: str, function: str) -> int:
        return next(index for index, statement in enumerate(statements) if _assigns_call(statement, target, function))

    q_norm, q_unit, q_fp32 = (
        index_of("q_normed", "ttnn.rms_norm"),
        index_of("q_unit", "ttnn.multiply"),
        index_of("q_fp32", "ttnn.typecast"),
    )
    k_norm, k_unit, k_row = (
        index_of("k_normed", "ttnn.rms_norm"),
        index_of("k_unit", "ttnn.multiply"),
        index_of("k_row", "ttnn.typecast"),
    )
    assert q_norm < q_unit < q_fp32 and k_norm < k_unit < k_row

    promotions = [
        statement
        for statement in statements
        if isinstance(statement.targets[0], ast.Name)
        and _assigns_call(statement, statement.targets[0].id, "ttnn.typecast")
    ]
    assert [statement.targets[0].id for statement in promotions] == ["q_fp32", "k_row"]
    for statement, normalized in ((statements[q_fp32], "q_unit"), (statements[k_row], "k_unit")):
        assert [ast.unparse(argument) for argument in statement.value.args] == [normalized, "ttnn.float32"]

    # v is widened by the FP32 subtract itself and the FP32 log decay is used as is.
    subtract = next(statement for statement in statements if _assigns_call(statement, "delta", "ttnn.subtract"))
    assert [ast.unparse(argument) for argument in subtract.value.args] == ["v", "v_read"]
    assert {keyword.arg: ast.unparse(keyword.value) for keyword in subtract.value.keywords}["dtype"] == "ttnn.float32"
