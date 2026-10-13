# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Repro: ProgramSpecMeshWorkloadFactoryAdapter mis-binds aliased tensor arguments.

Any Metal 2.0 op whose factory declares two TensorParameters can be called with the
same tensor for both.  On that call, ProgramSpecMeshWorkloadFactoryAdapter::
resolve_bindings (ttnn/api/ttnn/mesh_device_operation_adapter.hpp) used to resolve both
parameters to the *first* matching slot in the io-tensor enumeration, losing the fact
that the second parameter tracked the second slot.  A tensor's program hash covers only
storage type and spec (ttnn/api/ttnn/tensor/tensor.hpp), so a later call with distinct
but same-spec tensors hits that cache entry and replays the collapsed index map --
silently feeding the first tensor to both parameters.

batch_norm is used here because running_mean and running_var are separate
TensorParameters of identical shape, so aliasing them is legal, and eval-mode
batch_norm makes the substitution visible in the output:

    out = (input - running_mean) / sqrt(running_var + eps)

With input=5, mean=1, var=4:  correct = (5-1)/sqrt(4) = 2.0
If var is fed the mean tensor:  buggy   = (5-1)/sqrt(1) = 4.0
"""

import pytest
import torch

import ttnn

pytestmark = pytest.mark.use_module_device


def _channel_tensor(value, channels, device):
    """Per-channel [1, C, 1, 1] tensor filled with `value`."""
    return ttnn.from_torch(
        torch.full((1, channels, 1, 1), value, dtype=torch.float32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )


def test_program_cache_aliased_then_distinct_tensor_args(device):
    channels = 2
    eps = 0.0

    input_tensor = ttnn.from_torch(
        torch.full((1, channels, 32, 32), 5.0, dtype=torch.float32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    mean = _channel_tensor(1.0, channels, device)
    var = _channel_tensor(4.0, channels, device)

    # Call 1 -- ALIASED: the same tensor is passed for running_mean and running_var.
    # This is the call that poisons the cache entry (running_var's binding collapses
    # onto running_mean's slot).
    aliased = ttnn.batch_norm(input_tensor, running_mean=mean, running_var=mean, training=False, eps=eps)
    aliased_out = ttnn.to_torch(aliased)[0, 0, 0, 0].item()
    # (5 - 1) / sqrt(1) = 4.0 -- correct for THIS call, since mean and var really are equal here.
    assert aliased_out == pytest.approx(4.0, abs=0.05), f"call 1 (aliased) itself is wrong: {aliased_out}"

    # Call 2 -- DISTINCT: same specs as call 1, so it hits the cache entry above.
    # running_var must be read from `var` (4.0), not from `mean` (1.0).
    distinct = ttnn.batch_norm(input_tensor, running_mean=mean, running_var=var, training=False, eps=eps)
    distinct_out = ttnn.to_torch(distinct)[0, 0, 0, 0].item()

    expected = 2.0  # (5 - 1) / sqrt(4)
    buggy = 4.0  # (5 - 1) / sqrt(1)  <- running_var silently bound to the mean tensor
    print(f"\ncall 1 (mean=var=1.0): {aliased_out}   [expected 4.0]")
    print(f"call 2 (mean=1.0, var=4.0): {distinct_out}   [expected {expected}, bug gives {buggy}]")

    assert distinct_out != pytest.approx(buggy, abs=0.05), (
        f"program-cache aliasing bug reproduced: call 2 returned {distinct_out}, which is "
        f"(input - mean) / sqrt(mean + eps) -- running_var was re-bound to the mean tensor "
        f"by the cache-hit path"
    )
    assert distinct_out == pytest.approx(expected, abs=0.05), f"call 2 returned {distinct_out}, expected {expected}"
