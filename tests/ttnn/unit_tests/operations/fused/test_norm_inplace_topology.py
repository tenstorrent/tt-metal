# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""In-place softmax / layernorm keep the caller's tensor topology on a mesh.

The in-place variants (``softmax_in_place``, ``scale_mask_softmax_in_place`` and ``layer_norm`` with a
sharded program config and ``inplace=True``) hand the caller's own tensor back, so its distribution over
the mesh cannot change. Before the fix the framework relabelled that tensor with the union of ALL inputs,
so a replicated activation came back labelled ``Shard`` whenever the mask / weight was sharded across the
mesh. The out-of-place variants allocate a fresh tensor and keep the union label; the controls here
document that path for the replicated-everything case, where the union is the input's own label.
"""

import math

import pytest
import torch
import torch.nn.functional as F

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def _replicated(tensor, mesh_device, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        tensor,
        device=mesh_device,
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _sharded(tensor, mesh_device, dim, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        tensor,
        device=mesh_device,
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=dim),
    )


def _per_device(tensor):
    return [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(tensor.cpu())]


# ---------------------------------------------------------------------------------------------------------
# softmax
# ---------------------------------------------------------------------------------------------------------

SOFTMAX_SHAPE = (2, 4, 64, 128)


@pytest.mark.parametrize("mesh_device", [2], indirect=True)
@pytest.mark.parametrize("shard_dim", [None, 0], ids=["replicated", "sharded_dim0"])
def test_softmax_in_place_keeps_input_topology(mesh_device, shard_dim):
    """softmax_in_place has a single input, so its label was already correct; this pins the contract for
    both a replicated and a sharded activation and checks the returned handle aliases the input."""
    torch.manual_seed(0)
    num_devices = mesh_device.get_num_devices()
    if shard_dim is None:
        torch_input = torch.randn(SOFTMAX_SHAPE, dtype=torch.bfloat16)
        input_tensor = _replicated(torch_input, mesh_device)
        reference = _replicated(torch_input, mesh_device)
        expected = [torch.softmax(torch_input.float(), dim=-1)] * num_devices
    else:
        stacked = (SOFTMAX_SHAPE[0] * num_devices,) + SOFTMAX_SHAPE[1:]
        torch_input = torch.randn(stacked, dtype=torch.bfloat16)
        input_tensor = _sharded(torch_input, mesh_device, shard_dim)
        reference = _sharded(torch_input, mesh_device, shard_dim)
        expected = [torch.softmax(chunk.float(), dim=-1) for chunk in torch_input.chunk(num_devices, dim=shard_dim)]

    output = ttnn.softmax_in_place(input_tensor, numeric_stable=True)

    assert output.tensor_topology() == reference.tensor_topology()
    assert input_tensor.tensor_topology() == reference.tensor_topology()
    for got, want in zip(_per_device(output), expected):
        assert_with_pcc(want, got.float(), 0.99)


@pytest.mark.parametrize("mesh_device", [2], indirect=True)
def test_scale_mask_softmax_in_place_keeps_input_topology(mesh_device):
    """Replicated activation, mask sharded over the mesh on dim 0.

    Negative control: before the fix the returned (aliased) input carried the union label, i.e. exactly the
    mask's ``Shard(0)`` topology, so ``output.tensor_topology() == mask.tensor_topology()`` held and the
    ``!=`` assert below failed. The program cache must not see the topology at all: a second call with a
    replicated mask of the same shape must not add a cache entry.
    """
    torch.manual_seed(1)
    num_devices = mesh_device.get_num_devices()
    batch, _, _, width = SOFTMAX_SHAPE
    scale = 0.5

    torch_input = torch.randn(SOFTMAX_SHAPE, dtype=torch.bfloat16)
    # Non-causal mask: one tile row per batch, broadcast over heads and rows; the kernel reads its first row.
    mask_shape = (batch * num_devices, 1, 32, width)
    torch_mask = torch.where(torch.rand(mask_shape) > 0.5, 0.0, -10000.0).to(torch.bfloat16)

    input_tensor = _replicated(torch_input, mesh_device)
    reference = _replicated(torch_input, mesh_device)
    mask = _sharded(torch_mask, mesh_device, 0)
    assert mask.tensor_topology() != reference.tensor_topology()

    mesh_device.enable_program_cache()
    mesh_device.clear_program_cache()
    try:
        output = ttnn.scale_mask_softmax_in_place(input_tensor, scale, mask, numeric_stable=True)
        cache_entries = mesh_device.num_program_cache_entries()
        assert cache_entries > 0

        assert output.tensor_topology() == reference.tensor_topology()
        assert input_tensor.tensor_topology() == reference.tensor_topology()
        assert output.tensor_topology() != mask.tensor_topology()

        per_device_masks = torch_mask.chunk(num_devices, dim=0)
        for got, device_mask in zip(_per_device(output), per_device_masks):
            want = torch.softmax(torch_input.float() * scale + device_mask[:, :, :1, :].float(), dim=-1)
            assert_with_pcc(want, got.float(), 0.99)

        # Same shapes, replicated mask: a cache hit, and still the input's label.
        second_input = _replicated(torch_input, mesh_device)
        replicated_mask = _replicated(torch_mask[:batch], mesh_device)
        output = ttnn.scale_mask_softmax_in_place(second_input, scale, replicated_mask, numeric_stable=True)
        assert mesh_device.num_program_cache_entries() == cache_entries
        assert output.tensor_topology() == reference.tensor_topology()
    finally:
        mesh_device.disable_and_clear_program_cache()


@pytest.mark.parametrize("mesh_device", [2], indirect=True)
def test_scale_mask_softmax_out_of_place_control(mesh_device):
    """Out-of-place softmax returns an empty topology list and keeps the framework union: with a replicated
    input and a replicated mask that union is the input's own label."""
    torch.manual_seed(2)
    batch, _, _, width = SOFTMAX_SHAPE
    torch_input = torch.randn(SOFTMAX_SHAPE, dtype=torch.bfloat16)
    torch_mask = torch.where(torch.rand((batch, 1, 32, width)) > 0.5, 0.0, -10000.0).to(torch.bfloat16)

    input_tensor = _replicated(torch_input, mesh_device)
    mask = _replicated(torch_mask, mesh_device)

    output = ttnn.scale_mask_softmax(input_tensor, 0.5, mask, numeric_stable=True)

    assert output.tensor_topology() == input_tensor.tensor_topology()
    want = torch.softmax(torch_input.float() * 0.5 + torch_mask[:, :, :1, :].float(), dim=-1)
    for got in _per_device(output):
        assert_with_pcc(want, got.float(), 0.99)


# ---------------------------------------------------------------------------------------------------------
# layernorm
# ---------------------------------------------------------------------------------------------------------


def _layernorm_setup(mesh_device, inplace=True):
    """Mirrors tests/ttnn/nightly/unit_tests/operations/fused/test_layernorm_sharded.py: a block-sharded
    activation over the whole compute grid, one batch row per grid column, 4 tiles wide per core."""
    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    grid_size = [compute_grid_size.x, min(compute_grid_size.y, 8)]
    batch = grid_size[1]
    width = 128 * grid_size[1]
    in0_shape = (batch, 1, 32 * grid_size[0], width)
    M = in0_shape[2] * batch
    K = in0_shape[3]
    shard_shape = [M // grid_size[0], math.ceil(K / grid_size[1] / 32) * 32]
    program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=grid_size,
        subblock_w=4,
        block_h=batch,
        block_w=4,
        inplace=inplace,
    )
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=True, fp32_dest_acc_en=True
    )
    out_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1)
    return in0_shape, grid_size, shard_shape, program_config, compute_kernel_config, out_mem_config


def _block_sharded_replicated_input(torch_input, mesh_device, grid_size, shard_shape):
    interleaved = _replicated(torch_input, mesh_device)
    return ttnn.interleaved_to_sharded(
        interleaved,
        grid_size,
        shard_shape,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.ShardOrientation.COL_MAJOR,
    )


def _gamma_row_major(torch_gamma_rows):
    # Row-major gamma is laid out as [1, 1, K / 32, 32] (see the nightly sharded layernorm tests).
    return torch_gamma_rows.reshape(1, 1, -1, 32)


@pytest.mark.parametrize("mesh_device", [2], indirect=True)
def test_layer_norm_in_place_keeps_input_topology(mesh_device):
    """Replicated block-sharded activation, weight sharded over the mesh on dim 2 (one gamma per device).

    Negative control: before the fix the returned (aliased) input carried the union label, i.e. exactly the
    weight's ``Shard(2)`` topology, so ``output.tensor_topology() == weight.tensor_topology()`` held and the
    ``!=`` assert below failed. A second run with a replicated weight must hit the program cache.
    """
    torch.manual_seed(3)
    num_devices = mesh_device.get_num_devices()
    in0_shape, grid_size, shard_shape, program_config, compute_kernel_config, out_mem_config = _layernorm_setup(
        mesh_device
    )
    K = in0_shape[3]
    eps = 1e-2

    torch_input = (torch.rand(in0_shape) * 2 - 0.95).to(torch.bfloat16)
    torch_gamma = (torch.rand(K * num_devices) * 2 - 1).to(torch.bfloat16)
    gamma_stacked = _gamma_row_major(torch_gamma)  # [1, 1, num_devices * K / 32, 32]

    input_tensor = _block_sharded_replicated_input(torch_input, mesh_device, grid_size, shard_shape)
    reference = _replicated(torch_input, mesh_device)
    assert input_tensor.tensor_topology() == reference.tensor_topology()
    weight = _sharded(gamma_stacked, mesh_device, 2, layout=ttnn.ROW_MAJOR_LAYOUT)
    assert weight.tensor_topology() != reference.tensor_topology()

    mesh_device.enable_program_cache()
    mesh_device.clear_program_cache()
    try:
        output = ttnn.layer_norm(
            input_tensor,
            epsilon=eps,
            weight=weight,
            memory_config=out_mem_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )
        cache_entries = mesh_device.num_program_cache_entries()
        assert cache_entries > 0

        assert output.tensor_topology() == reference.tensor_topology()
        assert input_tensor.tensor_topology() == reference.tensor_topology()
        assert output.tensor_topology() != weight.tensor_topology()

        for got, device_gamma in zip(_per_device(output), torch_gamma.chunk(num_devices, dim=0)):
            want = F.layer_norm(torch_input.float(), (K,), device_gamma.float(), None, eps)
            assert_with_pcc(want, got.float(), 0.99)

        # Same shapes, replicated weight: a cache hit, and still the input's label.
        second_input = _block_sharded_replicated_input(torch_input, mesh_device, grid_size, shard_shape)
        replicated_weight = _replicated(_gamma_row_major(torch_gamma[:K]), mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT)
        output = ttnn.layer_norm(
            second_input,
            epsilon=eps,
            weight=replicated_weight,
            memory_config=out_mem_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )
        assert mesh_device.num_program_cache_entries() == cache_entries
        assert output.tensor_topology() == reference.tensor_topology()
    finally:
        mesh_device.disable_and_clear_program_cache()


@pytest.mark.parametrize("mesh_device", [2], indirect=True)
def test_layer_norm_out_of_place_control(mesh_device):
    """Out-of-place sharded layernorm returns an empty topology list and keeps the framework union: with a
    replicated input and a replicated weight that union is the input's own label."""
    torch.manual_seed(4)
    in0_shape, grid_size, shard_shape, program_config, compute_kernel_config, out_mem_config = _layernorm_setup(
        mesh_device, inplace=False
    )
    K = in0_shape[3]
    eps = 1e-2

    torch_input = (torch.rand(in0_shape) * 2 - 0.95).to(torch.bfloat16)
    torch_gamma = (torch.rand(K) * 2 - 1).to(torch.bfloat16)

    input_tensor = _block_sharded_replicated_input(torch_input, mesh_device, grid_size, shard_shape)
    weight = _replicated(_gamma_row_major(torch_gamma), mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT)

    output = ttnn.layer_norm(
        input_tensor,
        epsilon=eps,
        weight=weight,
        memory_config=out_mem_config,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )

    assert output.tensor_topology() == input_tensor.tensor_topology()
    want = F.layer_norm(torch_input.float(), (K,), torch_gamma.float(), None, eps)
    for got in _per_device(output):
        assert_with_pcc(want, got.float(), 0.99)
