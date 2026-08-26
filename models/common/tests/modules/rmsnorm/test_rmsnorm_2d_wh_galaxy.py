# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Real-hardware correctness tests for the common WH Galaxy RMSNorm2D."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

import ttnn
from models.common.models.galaxy import GalaxyCollectivePlan, GalaxyResourceKey, GalaxyResourcesConfig, GalaxyTensorSpec
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.rmsnorm.rmsnorm_2d import (
    RMSNorm2D,
    RMSNorm2DConfig,
    RMSNorm2DGeometry,
    RMSNorm2DResidualPolicy,
)
from models.common.tests.modules._wh_galaxy_hardware import (
    compose_2d_sharded_tensor,
    deallocate_module_weights,
    deallocate_tensor,
    exact_tensor_resource,
    galaxy_mode_plan,
    galaxy_prefetch_decode_mode_plan,
    require_galaxy_ccl_hardware_resources,
    require_galaxy_hardware_resources,
)
from models.common.utility_functions import comp_pcc

EPS = 1e-6


def _reference_norm(weight: torch.Tensor) -> torch.nn.RMSNorm:
    """Reference normalization: torch.nn.RMSNorm, as the 1D suite compares against,
    rather than a hand-written variance/rsqrt re-implementation."""
    reference = torch.nn.RMSNorm(weight.numel(), eps=EPS).to(torch.bfloat16)
    with torch.no_grad():
        reference.weight.copy_(weight)
    return reference


def _lazy(source: torch.Tensor, mesh_device: ttnn.MeshDevice) -> LazyWeight:
    return LazyWeight(source=source, device=mesh_device)


def _weight_lazy(source: torch.Tensor, mesh_device: ttnn.MeshDevice) -> LazyWeight:
    return LazyWeight(
        source=source.reshape(1, 1, source.numel() // 32, 32),
        device=mesh_device,
        mesh_mapper_config=ttnn.MeshMapperConfig(
            placements=[ttnn.PlacementReplicate(), ttnn.PlacementShard(2)],
            mesh_shape_override=ttnn.MeshShape(8, 4),
        ),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
    )


def _assert_pcc(expected: torch.Tensor, actual: torch.Tensor, *, case: str) -> None:
    passing, message = comp_pcc(expected.float(), actual.float(), 0.99)
    assert passing, f"{case} failed PCC>=0.99: {message}"


def _resources_config(mesh_device, dim):
    # The fused RMS stats circular buffer is created on the first core of the norm
    # input shard grid (x=2, y=0) and bound to this buffer's L1 address, so the
    # persistent stats shard has to live on that core - see RMSNorm2D's
    # _require_fused_stats_placement.
    decode_memcfg = ttnn.create_sharded_memory_config(
        shape=(1, 1, 32, 128),
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(2, 0))}),
        strategy=ttnn.ShardStrategy.WIDTH,
        use_height_and_width_as_shard_shape=True,
    )

    def plan(sequence, memory_config, *, semaphores_per_slot, topology=ttnn.Topology.Linear):
        input_shape = (1, 1, sequence, 32)
        return GalaxyCollectivePlan(
            key=GalaxyResourceKey("all_gather", 1, input_shape, sequence),
            topology=topology,
            num_links=1,
            semaphores_per_slot=semaphores_per_slot,
            persistent_output_specs=(
                GalaxyTensorSpec((1, 1, sequence, 128), ttnn.bfloat16, ttnn.TILE_LAYOUT, memory_config),
            ),
        )

    local_hidden_tiles = (dim // 4) // 32
    norm_grid_height = local_hidden_tiles // 8
    norm_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(3, norm_grid_height - 1))})
    decode = replace(
        galaxy_prefetch_decode_mode_plan(
            (plan(32, decode_memcfg, semaphores_per_slot=1, topology=ttnn.Topology.Ring),)
        ),
        semaphore_cores=norm_cores,
        # The fused RMS all-gather binds its semaphore to the norm grid it owns,
        # so it is the one collective for which narrowing is safe (D3).
        allow_narrow_semaphore_cores=True,
    )
    return GalaxyResourcesConfig(
        architecture=ttnn.device.Arch.WORMHOLE_B0,
        prefill=galaxy_mode_plan(
            "prefill",
            (
                plan(128, ttnn.DRAM_MEMORY_CONFIG, semaphores_per_slot=1),
                plan(2048, ttnn.DRAM_MEMORY_CONFIG, semaphores_per_slot=1),
            ),
            mesh_device,
        ),
        decode=decode,
    )


def _module(mesh_device, resources, weight, *, geometry, fused_decode=False):
    decode_context = resources.context("decode") if resources is not None else None
    prefill_context = resources.context("prefill") if resources is not None else None
    config = RMSNorm2DConfig(
        weight=weight if isinstance(weight, LazyWeight) else _lazy(weight, mesh_device),
        mesh_device=mesh_device,
        tt_ccl=resources.ccl if resources is not None else None,
        decode_ccl_context=decode_context,
        prefill_ccl_context=prefill_context,
        decode_prefetch_context=resources.prefetch_context("decode") if resources is not None else None,
        prefill_prefetch_context=resources.prefetch_context("prefill") if resources is not None else None,
        collective_resource_selector=exact_tensor_resource if resources is not None else None,
        geometry=geometry,
        residual_policy=RMSNorm2DResidualPolicy.FUSED_DECODE if fused_decode else RMSNorm2DResidualPolicy.NONE,
        eps=EPS,
    )
    return RMSNorm2D.from_config(config)


def _invoke(module, resources, mesh_device, x, *, mode, residual=None):
    if resources is not None:
        resources.activate(mode)
    lazy_x = _lazy(x, mesh_device)
    lazy_residual = _lazy(residual, mesh_device) if residual is not None else None
    result = module(lazy_x, mode=mode, residual=lazy_residual)
    outputs = result if isinstance(result, tuple) else (result,)
    try:
        if resources is not None:
            resources.synchronize(mode)
        else:
            ttnn.synchronize_device(mesh_device)
        composed = tuple(compose_2d_sharded_tensor(output, mesh_device) for output in outputs)
        if module.config.geometry is RMSNorm2DGeometry.HEAD_LOCAL:
            composed = tuple(output[..., : x.shape[-1]] for output in composed)
        return composed
    finally:
        input_tensors = tuple(
            tensor
            for tensor in (lazy_x._value, lazy_residual._value if lazy_residual is not None else None)
            if tensor is not None
        )
        for output in outputs:
            if all(output is not tensor for tensor in input_tensors):
                deallocate_tensor(output)
        for tensor in input_tensors:
            deallocate_tensor(tensor)


def _invoke_decode_repeat(module, resources, mesh_device, x, residual, *, count):
    resources.activate("decode")
    queued = []
    try:
        for _ in range(count):
            lazy_x = _lazy(x, mesh_device)
            lazy_residual = _lazy(residual, mesh_device)
            outputs = module(lazy_x, mode="decode", residual=lazy_residual)
            queued.append((outputs, lazy_x._value, lazy_residual._value))

        resources.synchronize("decode")
        return tuple(
            tuple(compose_2d_sharded_tensor(output, mesh_device) for output in outputs)
            for outputs, _input, _residual in queued
        )
    finally:
        for outputs, input_tensor, residual_tensor in queued:
            for output in outputs:
                if output is not input_tensor and output is not residual_tensor:
                    deallocate_tensor(output)
            deallocate_tensor(input_tensor)
            deallocate_tensor(residual_tensor)


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("dim", [8192, 5120], ids=["llama-final-8192", "qwen-final-5120"])
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
@torch.no_grad()
def test_rmsnorm_2d_wh_galaxy_final_norm_decode_batch_32_fused_residual_repeat(mesh_device, dim):
    torch.manual_seed(2)
    weight = torch.randn(dim, dtype=torch.bfloat16)
    reference = _reference_norm(weight)
    x = torch.randn(1, 1, 32, dim, dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    residual_sum = x + residual
    expected = reference(residual_sum)
    lazy_weight = _weight_lazy(weight, mesh_device)
    resources = require_galaxy_ccl_hardware_resources(
        mesh_device,
        config=_resources_config(mesh_device, dim),
    )
    module = None
    try:
        module = _module(
            mesh_device,
            resources,
            lazy_weight,
            geometry=RMSNorm2DGeometry.DISTRIBUTED,
            fused_decode=True,
        )
        results = _invoke_decode_repeat(module, resources, mesh_device, x, residual, count=2)
        for invocation, (actual, actual_residual) in enumerate(results):
            _assert_pcc(expected, actual, case=f"final norm invocation {invocation}")
            _assert_pcc(residual_sum, actual_residual, case=f"residual sum invocation {invocation}")
    finally:
        try:
            resources.cleanup()
        finally:
            deallocate_module_weights(module, "weight")


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("dim", [8192, 5120], ids=["llama-final-8192", "qwen-final-5120"])
@pytest.mark.parametrize("seq_len", [128, 2048], ids=["seq128", "seq2048"])
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": True}],
    indirect=True,
)
@torch.no_grad()
def test_rmsnorm_2d_wh_galaxy_final_norm_prefill_repeat(mesh_device, dim, seq_len):
    torch.manual_seed(3)
    weight = torch.randn(dim, dtype=torch.bfloat16)
    reference = _reference_norm(weight)
    lazy_weight = _weight_lazy(weight, mesh_device)
    device_weight = lazy_weight.get_device_weight()
    resources = require_galaxy_hardware_resources(
        mesh_device,
        config=_resources_config(mesh_device, dim),
        prefetch_weights=(("norm.weight", device_weight),),
    )
    module = None
    try:
        module = _module(mesh_device, resources, lazy_weight, geometry=RMSNorm2DGeometry.DISTRIBUTED)
        for invocation in range(2):
            x = torch.randn(1, 1, seq_len, dim, dtype=torch.bfloat16)
            residual = torch.randn_like(x)
            (actual,) = _invoke(module, resources, mesh_device, x, mode="prefill", residual=residual)
            _assert_pcc(
                reference(x + residual),
                actual,
                case=f"final norm prefill {seq_len} invocation {invocation}",
            )
    finally:
        try:
            resources.cleanup()
        finally:
            deallocate_module_weights(module, "weight")


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("projection", ["q_norm", "k_norm"])
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": True}],
    indirect=True,
)
@torch.no_grad()
def test_rmsnorm_2d_wh_galaxy_head_local_128_qk_decode_and_prefill_repeat(mesh_device, projection):
    torch.manual_seed(4 if projection == "q_norm" else 5)
    weight = torch.randn(128, dtype=torch.bfloat16)
    reference = _reference_norm(weight)
    resources = None
    module = None
    try:
        module = _module(mesh_device, resources, weight, geometry=RMSNorm2DGeometry.HEAD_LOCAL)
        cases = (("decode", 32), ("prefill", 128), ("prefill", 2048))
        for invocation in range(2):
            for mode, rows in cases:
                x = torch.randn(1, 1, rows, 128, dtype=torch.bfloat16)
                (actual,) = _invoke(module, resources, mesh_device, x, mode=mode)
                _assert_pcc(
                    reference(x),
                    actual,
                    case=f"{projection} {mode} {rows} invocation {invocation}",
                )
    finally:
        try:
            deallocate_module_weights(module, "weight")
        finally:
            if resources is not None:
                resources.cleanup()
