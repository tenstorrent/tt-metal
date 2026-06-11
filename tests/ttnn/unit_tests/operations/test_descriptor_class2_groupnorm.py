# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Class-2 descriptor cache-hit fast-path test for the groupnorm family (#46506).

The three groupnorm program factories (sharded / no_mcast / mcast) share one
GroupNormDeviceOperation. That device op declares get_dynamic_runtime_args(...),
which opts the op into the descriptor cache-hit FAST PATH (create_descriptor is
NOT re-run on a cache hit). Every per-dispatch address-derived runtime arg is
patched by the framework via a CB `.buffer` binding (sharded input/output and
reciprocals) or a patchable Buffer* runtime-arg binding (DRAM
input/output/gamma/beta/masks); all other args are hash/shape/grid-derived and
stable across hits. get_dynamic_runtime_args therefore returns {} for all three.

We set TT_METAL_FORBID_DESCRIPTOR_REBUILD_ON_CACHE_HIT=1 BEFORE importing ttnn:
if any groupnorm dispatch rebuilds its descriptor on a cache hit, the adapter
raises and the test fails. We run each config 3x (cache hit on calls 2 and 3)
and assert correctness via PCC vs torch.nn.functional.group_norm.

Routing note: ttnn.group_norm picks the factory from input layout / grid / batch.
- A sharded input routes to GroupNormShardedProgramFactory.
- A DRAM (non-sharded) input routes to GroupNormNoMcastProgramFactory when
  batch >= num_virtual_rows, else GroupNormMcastProgramFactory.
The DRAM-with-weight/bias case below exercises the writer Buffer* bindings that
this fix added to the mcast/no_mcast factories.
"""

import os

os.environ["TT_METAL_FORBID_DESCRIPTOR_REBUILD_ON_CACHE_HIT"] = "1"

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    dev.enable_program_cache()
    yield dev
    ttnn.close_device(dev)


def _run_thrice(fn, dev):
    """Run fn 3x (calls 2 and 3 are cache hits). Raises if the op rebuilds its
    descriptor on a cache hit (guard) or binds a Buffer* at a wrong arg slot
    (resolve_bindings validation on the first/miss call)."""
    out = None
    for _ in range(3):
        out = fn()
        ttnn.synchronize_device(dev)
    return out


def test_group_norm_no_weight_no_rebuild(device):
    # DRAM tilized input WITHOUT weight/bias. Same structure as the weight+bias case below (tilize +
    # input_mask + TILE output) but with weight=bias=None -> exercises the no-gamma/beta writer path.
    #
    # NOTE: the DRAM group_norm op needs a tilized input AND an input_mask for correct channel
    # grouping. Feeding a raw ROW_MAJOR input with no mask (as an earlier version of this test did)
    # produces an incorrectly grouped result (PCC ~0.24); it is NOT an op bug, just an invalid config.
    N, C, H, W, num_groups = 1, 64, 1, 64, 2

    torch_input = torch.rand((N, C, H, W), dtype=torch.bfloat16)

    torch_ref = torch.nn.functional.group_norm(torch_input.float(), num_groups, eps=1e-5)
    torch_ref = torch_ref.permute(0, 2, 3, 1).reshape(N, 1, W * H, C)

    input_tensor = torch_input.permute(0, 2, 3, 1).view(N, 1, W * H, C)
    input_rm = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    input_tilized = ttnn.tilize_with_zero_padding(input_rm, use_multicore=True)

    # We only need the input_mask here (no gamma/beta); dram_group_norm_params_from_torch is the
    # canonical way to build a correctly-shaped DRAM input mask.
    [_gamma_t, _beta_t], input_mask = ttnn.dram_group_norm_params_from_torch(
        [torch.ones((C,), dtype=torch.bfloat16), torch.zeros((C,), dtype=torch.bfloat16)],
        C,
        num_groups,
        device,
        return_mask=True,
    )

    out = _run_thrice(
        lambda: ttnn.group_norm(
            input_tilized,
            num_groups=num_groups,
            input_mask=input_mask,
            epsilon=1e-5,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_layout=ttnn.TILE_LAYOUT,
            inplace=False,
        ),
        device,
    )

    assert_with_pcc(torch_ref, ttnn.to_torch(out).float(), 0.999)


def test_group_norm_dram_weight_bias_no_rebuild(device):
    # DRAM tilized input WITH weight + bias + input_mask. Exercises the writer
    # gamma/beta/input_mask Buffer* runtime-arg bindings added by this fix.
    N, C, H, W, num_groups = 1, 64, 1, 64, 2

    torch_input = torch.rand((N, C, H, W), dtype=torch.bfloat16)
    torch_weight = torch.rand((C,), dtype=torch.bfloat16)
    torch_bias = torch.rand((C,), dtype=torch.bfloat16)

    torch_ref = torch.nn.functional.group_norm(
        torch_input.float(), num_groups, weight=torch_weight.float(), bias=torch_bias.float(), eps=1e-5
    )
    torch_ref = torch_ref.permute(0, 2, 3, 1).reshape(N, 1, W * H, C)

    input_tensor = torch_input.permute(0, 2, 3, 1).view(N, 1, W * H, C)
    input_rm = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    input_tilized = ttnn.tilize_with_zero_padding(input_rm, use_multicore=True)

    [gamma_t, beta_t], input_mask = ttnn.dram_group_norm_params_from_torch(
        [torch_weight, torch_bias], C, num_groups, device, return_mask=True
    )

    out = _run_thrice(
        lambda: ttnn.group_norm(
            input_tilized,
            num_groups=num_groups,
            input_mask=input_mask,
            weight=gamma_t,
            bias=beta_t,
            epsilon=1e-5,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_layout=ttnn.TILE_LAYOUT,
            inplace=False,
        ),
        device,
    )

    assert_with_pcc(torch_ref, ttnn.to_torch(out).float(), 0.999)
