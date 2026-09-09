# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Generated in full from this op's coverage data and replaced whenever that data changes, so edits
# made here do not survive.
#
# Every one of these 0 cases is outside what the codegen path supports. `ttnn.reshape` decides
# internally which implementation serves a call, so each case must be served natively and must
# produce exactly what the native path produces. They are grouped by the condition that rejects
# them.

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_equal

_DTYPES = {
    "bfloat16": ttnn.bfloat16,
    "float32": ttnn.float32,
    "int32": ttnn.int32,
    "uint32": ttnn.uint32,
    "uint16": ttnn.uint16,
    "bfloat8_b": ttnn.bfloat8_b,
    "bfloat4_b": ttnn.bfloat4_b,
}
_LAYOUTS = {"row_major": ttnn.ROW_MAJOR_LAYOUT, "tile": ttnn.TILE_LAYOUT}


def _make_input(shape, dtype):
    if dtype == "uint16":
        return torch.randint(0, 100, shape).to(torch.int16)
    if dtype == "int32":
        return torch.randint(-(2**31), 2**31, shape, dtype=torch.int32)
    if dtype == "uint32":
        return torch.randint(0, 2**31, shape, dtype=torch.int32)
    return torch.rand(shape).bfloat16().float()


# (dtype, layout, shape, kwargs)
_ROUTING = []

_ROUTING_IDS = []


@pytest.mark.parametrize("dtype,layout,shape,kwargs", _ROUTING, ids=_ROUTING_IDS)
def test_reshape_codegen_routing(device, dtype, layout, shape, kwargs):
    torch_input = _make_input(shape, dtype)
    tt_input = ttnn.from_torch(torch_input, dtype=_DTYPES[dtype], layout=_LAYOUTS[layout], device=device)

    # The forced-native entry rather than `ttnn.reshape` itself: the public entry is the thing under
    # test here, so using it to produce the reference would compare it against itself and pass no
    # matter where it routed.
    golden = ttnn.to_torch(ttnn._ttnn.operations.data_movement.reshape_force_native(tt_input, **kwargs))
    # The native call above compiled and cached its program, so a correct fallback reuses it and
    # leaves the cache flat. Only a mis-route to codegen compiles something new. Taking the
    # snapshot after the native call is what makes the two distinguishable -- before it, both a
    # fallback and a mis-route add exactly one entry.
    entries_before = device.num_program_cache_entries()
    routed = ttnn.to_torch(ttnn.reshape(tt_input, **kwargs))

    assert_equal(golden, routed)
    assert (
        device.num_program_cache_entries() == entries_before
    ), "an unsupported case routed to the codegen path (the program cache grew); expected native"
