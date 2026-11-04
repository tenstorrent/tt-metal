# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch


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


class _Opaque:
    """Has the default object repr, "<module._Opaque object at 0x...>"."""


def _make_id(request, val, argname="cfg"):
    return request.config.hook.pytest_make_parametrize_id(config=request.config, val=val, argname=argname)


def test_parametrize_id_for_default_repr_object_has_no_memory_address(request):
    """A default-repr object keeps its type name without the address, the same for every instance."""
    first, second = _make_id(request, _Opaque()), _make_id(request, _Opaque())
    assert first == second, (first, second)
    assert "0x" not in first and first.startswith("cfg=<") and first.endswith("_Opaque object>"), first


def test_parametrize_id_for_local_class_has_no_memory_address(request):
    """A class defined in a function has a "<locals>" qualname, e.g. "<mod.f.<locals>.C object at 0x...>"."""

    class _Local:
        pass

    first, second = _make_id(request, _Local()), _make_id(request, _Local())
    assert first == second, (first, second)
    assert "0x" not in first and first.endswith(".<locals>._Local object>"), first


def test_parametrize_id_for_nested_default_repr_objects_has_no_memory_address(request):
    """Default-repr objects inside a dict or list, e.g. device_params, are normalized in place."""
    first = _make_id(request, {"router": _Opaque(), "configs": [_Opaque(), 1]}, argname="device_params")
    second = _make_id(request, {"router": _Opaque(), "configs": [_Opaque(), 1]}, argname="device_params")
    assert first == second, (first, second)
    assert "0x" not in first and first.count("_Opaque object>") == 2, first
    assert first.startswith("device_params={'router': <") and first.endswith(", 1]}"), first


def test_parametrize_id_without_addresses_is_unchanged(request):
    """Values with no embedded address keep their usual ID."""
    assert _make_id(request, {"num_links": 1, "dims": [0, 0x10]}) == "cfg={'num_links': 1, 'dims': [0, 16]}"
    assert _make_id(request, "0xdeadbeef") == "cfg=0xdeadbeef"
