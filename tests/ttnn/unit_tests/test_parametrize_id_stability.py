# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from tests.ttnn.utils_for_testing import merge_gate_subset


def _plain_function(x):
    return x


def test_parametrize_id_for_builtin_has_no_memory_address(request):
    """A parametrized torch builtin must reduce to a stable test ID.

    Its default repr is "<built-in method erfinv of type object at 0x...>", and that
    address differs per xdist worker process, so workers collect different IDs and the
    parallel session aborts on a collection mismatch.
    """
    made_id = request.config.hook.pytest_make_parametrize_id(
        config=request.config, val=torch.erfinv, argname="torch_op"
    )
    assert made_id == "torch_op=erfinv", made_id


def test_parametrize_id_for_python_function_has_no_memory_address(request):
    """Same for a plain Python function, whose repr is "<function _plain_function at 0x...>".

    torch.erfinv is a BuiltinFunctionType; a def'd helper passed through @pytest.mark.parametrize
    is a FunctionType and hits the other half of the isinstance check.
    """
    made_id = request.config.hook.pytest_make_parametrize_id(
        config=request.config, val=_plain_function, argname="torch_op"
    )
    assert made_id == "torch_op=_plain_function", made_id


def test_merge_gate_subset_ids_match_stacked_parametrize(request):
    """merge_gate_subset IDs must match stacked parametrize, and only keep() cases are marked."""
    hook = request.config.hook.pytest_make_parametrize_id
    axes = {"dim": [-2, 0], "w": [64, 22], "h": [15]}
    _, params = merge_gate_subset(axes, keep=lambda dim, **_: dim < 0)

    expected = [
        "-".join(
            hook(config=request.config, val=value, argname=name) for name, value in [("dim", dim), ("w", w), ("h", h)]
        )
        for dim in axes["dim"]
        for w in axes["w"]
        for h in axes["h"]
    ]
    assert [param.id for param in params] == expected
    assert [param.id for param in params if param.marks] == ["dim=-2-w=64-h=15", "dim=-2-w=22-h=15"]


def test_merge_gate_subset_ids_for_ttnn_op_have_no_memory_address(request):
    """merge_gate_subset sets id=, so it must use the same conversion as the hook.

    str(ttnn.std) embeds a process address. pytest-xdist -n 4 then aborts because
    workers collect different node IDs. Passing the op as a list (not a dict of
    hand-written IDs) is the case a future caller would write.
    """
    hook = request.config.hook.pytest_make_parametrize_id
    _, params = merge_gate_subset({"op": [ttnn.std, ttnn.var]}, keep=lambda **_: False)
    expected = [hook(config=request.config, val=op, argname="op") for op in (ttnn.std, ttnn.var)]
    assert [param.id for param in params] == expected
    assert expected == ["op=ttnn.std", "op=ttnn.var"]
