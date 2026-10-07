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
