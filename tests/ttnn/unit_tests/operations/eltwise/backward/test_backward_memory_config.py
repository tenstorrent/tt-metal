# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""A gradient the caller did not preallocate still lands where the caller asked.

The backward ops allocate their outputs with empty_like(input), which inherits the
input's memory config and ignores the memory_config argument. div_bw went further:
with the request unmet, its second gradient came back as -inf.
"""

import pytest
import torch
import ttnn

SHAPE = (1, 1, 32, 32)

# op name, number of tensor arguments, trailing scalar arguments
OPS = [
    ("abs_bw", 2, []),
    ("exp_bw", 2, []),
    ("neg_bw", 2, []),
    ("sqrt_bw", 2, []),
    ("silu_bw", 2, []),
    ("rsqrt_bw", 2, []),
    ("frac_bw", 2, []),
    ("pow_bw", 2, [2.0]),
    ("add_bw", 3, []),
    ("sub_bw", 3, []),
    ("mul_bw", 3, []),
    ("div_bw", 3, []),
    ("rsub_bw", 3, []),
    ("assign_bw", 3, []),
    ("addalpha_bw", 3, [2.0]),
    ("subalpha_bw", 3, [2.0]),
    ("addcmul_bw", 4, [2.0]),
    ("addcdiv_bw", 4, [2.0]),
]

VALUES = (1.5, 2.5, 0.5, 3.0)


def _t(v, device, memory_config):
    return ttnn.from_torch(
        torch.full(SHAPE, v, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )


def _call(op, arity, scalars, device, memory_config):
    args = [_t(VALUES[i], device, ttnn.DRAM_MEMORY_CONFIG) for i in range(arity)]
    kwargs = {} if memory_config is None else {"memory_config": memory_config}
    out = getattr(ttnn, op)(*args, *scalars, **kwargs)
    out = out if isinstance(out, (list, tuple)) else [out]
    return [x for x in out if x is not None]


@pytest.mark.parametrize("op, arity, scalars", OPS)
def test_gradients_land_in_the_requested_memory(device, op, arity, scalars):
    for i, out in enumerate(_call(op, arity, scalars, device, ttnn.L1_MEMORY_CONFIG)):
        assert out.memory_config().buffer_type == ttnn.BufferType.L1, f"{op} output {i} is in {out.memory_config()}"


@pytest.mark.parametrize("op, arity, scalars", OPS)
def test_requesting_a_memory_config_does_not_move_any_value(device, op, arity, scalars):
    default = _call(op, arity, scalars, device, None)
    asked = _call(op, arity, scalars, device, ttnn.L1_MEMORY_CONFIG)
    for i, (a, b) in enumerate(zip(default, asked)):
        x, y = ttnn.to_torch(a).float(), ttnn.to_torch(b).float()
        moved = ((x != y) & ~(torch.isnan(x) & torch.isnan(y))).sum().item()
        assert moved == 0, f"{op} output {i}: {moved} of {x.numel()} values move when L1 is requested"
