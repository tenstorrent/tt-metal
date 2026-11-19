# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU coverage of physical-slot selection, independent of TTNN execution."""

import ast
from pathlib import Path

import pytest


def selector(capacity, buckets):
    path = Path(__file__).resolve().parents[1] / "tt/generator.py"
    owner = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "QwenGenerator"
    )
    owner.bases = []
    owner.body = [
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name in ("_bucket_trace_key", "model_trace_key")
    ]
    namespace = {}
    exec(compile(ast.Module(body=[owner], type_ignores=[]), str(path), "exec"), namespace)
    result = namespace["QwenGenerator"]()
    result.max_batch_size, result.decode_buckets = capacity, buckets
    return result


@pytest.mark.parametrize(
    "active,expected",
    [((0,), "model_2"), ((1,), "model_2"), ((1, 0), "model_2"), ((17,), "model_fixed"), ((0, 2), "model_fixed")],
)
def test_selector_covers_physical_identity_not_active_count(active, expected):
    assert selector(32, (2, 32)).model_trace_key(active) == expected


def test_default_fixed_capacity_and_standalone_solo_are_preserved():
    assert selector(32, (32,)).model_trace_key((0,)) == "model_fixed"
    assert selector(1, (1,)).model_trace_key((0,)) == "model"


def test_out_of_capacity_cannot_select_a_partial_graph():
    with pytest.raises(ValueError, match="outside"):  # allow-pytest.raises: CPU-only validation, no device error state
        selector(32, (2, 32)).model_trace_key((32,))
