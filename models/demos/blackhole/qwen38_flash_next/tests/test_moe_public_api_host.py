# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exercise the model's actual call expressions against the public nanobind signature."""

import ast
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[5]
MODEL = ROOT / "models/demos/blackhole/qwen38_flash_next/ttnn/moe.py"
BINDING = ROOT / "ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/moe_compute_nanobind.cpp"
RETIRED = {"local_combine", "zero_fill_non_owned_rows", "prefill_rings", "enable_a2a_pipeline"}


def calls():
    return [
        node
        for node in ast.walk(ast.parse(MODEL.read_text()))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "moe_compute"
    ]


def public_signature():
    source = BINDING.read_text()
    return set(re.findall(r'nb::arg\("(\w+)"\)', source))


@pytest.mark.parametrize("call", calls(), ids=["partial", "local-sum", "tiles"])
def test_actual_launch_matches_public_signature(call):
    allowed = public_signature()
    captured = {}

    def operation(*args, **kwargs):
        assert len(args) == 6
        unknown = kwargs.keys() - allowed
        if unknown:
            raise TypeError(f"unsupported public kwargs: {sorted(unknown)}")
        captured.update(kwargs)
        return kwargs["optional_output_tensor"]

    values = {node.id: object() for node in ast.walk(call) if isinstance(node, ast.Name)}
    output = object()
    attrs = {
        node.attr: object()
        for node in ast.walk(call)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self"
    }
    attrs["local_combine_output"] = output
    values.update(
        self=SimpleNamespace(**attrs),
        ttnn=SimpleNamespace(experimental=SimpleNamespace(moe_compute=operation), DRAM_MEMORY_CONFIG=object()),
        MoEActivationFunction=SimpleNamespace(SILU=object()),
        LOCAL_COMBINE_AXIS=0,
        INTERMEDIATE_SIZE=640,
    )
    assert eval(compile(ast.Expression(call), str(MODEL), "eval"), values) is output
    assert captured["cluster_axis"] == 0
    assert captured["compute_only"] is False
    assert captured["num_shared_experts_per_device"] == 0
    assert not RETIRED & captured.keys()
    # The retired branch's actual incompatibility must be rejected, not silently ignored.
    for keyword in RETIRED:
        with pytest.raises(TypeError, match="unsupported public kwargs"):  # allow-pytest.raises: pure host contract
            operation(*([object()] * 6), **{keyword: True})


def test_removed_selector_names_have_no_live_python_callers():
    names = {"moe_rows_form", "moe_local_output_enabled", "moe_slab_prefill_rings", "admit_slab_moe_switches"}
    for path in MODEL.parents[1].rglob("*.py"):
        if "tests" in path.parts:
            continue
        tree = ast.parse(path.read_text())
        assert not names & {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}, path
        assert not names & {
            item.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) for item in node.names
        }, path
