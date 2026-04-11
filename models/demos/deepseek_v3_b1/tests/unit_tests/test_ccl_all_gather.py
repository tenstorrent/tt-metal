# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
TTNN CCL All-Gather Test

Tests the all-gather operation on a 4-device torused row ring where:
1. Each device starts with a local shard on the gather core
2. After all-gather, every device holds the concatenation of all 4 shards
   in canonical rank order

Topology: 4x1 submesh (single ring of 4 devices along row dimension).
Tile: (1, 32) bf16.
Sharding: HEIGHT_SHARDED on the gather core for input/output,
          HEIGHT_SHARDED on the transport core for scratch.
"""

import os
from dataclasses import dataclass
from typing import Any

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.common.utility_functions import is_slow_dispatch
from models.demos.deepseek_v3_b1.micro_ops.ccl_all_gather.op import DeepseekMinimalAllGather
from models.demos.deepseek_v3_b1.tests.unit_tests.ccl_test_utils import create_fabric_router_config
from models.perf.benchmarking_utils import BenchmarkProfiler

GATHER_CORE = ttnn.CoreCoord(0, 0)
TRANSPORT_CORE = ttnn.CoreCoord(0, 1)

NUM_DEVICES = 4
TILE_W = 32

ENV_NUM_LINKS = "CCL_ALL_GATHER_NUM_LINKS"
ENV_MAX_PAYLOAD_SIZE = "CCL_ALL_GATHER_MAX_PAYLOAD_SIZE_BYTES"


def _parse_env_int(name: str, value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be > 0, got {parsed}")
    return parsed


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return _parse_env_int(name, value)


def _get_num_links_params(defaults: list[int]) -> list[int]:
    value = os.getenv(ENV_NUM_LINKS)
    if value is None or value.strip() == "":
        return defaults
    return [_parse_env_int(ENV_NUM_LINKS, value)]


MAX_PAYLOAD_SIZE = _get_env_int(ENV_MAX_PAYLOAD_SIZE, 15232)


@dataclass(frozen=True)
class AllGatherTestInputs:
    output_shape: list[int]
    input_tensors_per_device: list[torch.Tensor]
    torch_expected: torch.Tensor
    input_tensor_mesh: Any
    output_tensor_mesh: Any
    scratch_tensor_mesh: Any
    semaphores: list[Any]


def build_all_gather_test_inputs(
    *,
    mesh_device,
    output_shape,
    input_tensors_per_device=None,
):
    """Build test tensors for the 4-device all-gather.

    Args:
        output_shape: Full gathered output shape, e.g. [1, 896].
            output_shape[1] must be divisible by NUM_DEVICES * TILE_W.
        input_tensors_per_device: Optional list of NUM_DEVICES tensors.
            If None, random tensors are generated.
    """
    tile_h = output_shape[0]
    output_width = output_shape[1]
    slice_width = output_width // NUM_DEVICES

    if output_width % NUM_DEVICES != 0:
        raise ValueError(f"output_width={output_width} must be divisible by NUM_DEVICES={NUM_DEVICES}")
    if slice_width % TILE_W != 0:
        raise ValueError(f"slice_width={slice_width} must be divisible by TILE_W={TILE_W}")

    tile = ttnn.Tile((tile_h, TILE_W))

    if input_tensors_per_device is None:
        input_tensors_per_device = [torch.rand(tile_h, slice_width, dtype=torch.bfloat16) for _ in range(NUM_DEVICES)]

    input_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(GATHER_CORE, GATHER_CORE)})
    input_shard_spec = ttnn.ShardSpec(input_shard_grid, (tile_h, slice_width), ttnn.ShardOrientation.ROW_MAJOR)
    input_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, buffer_type=ttnn.BufferType.L1, shard_spec=input_shard_spec
    )

    input_tensor_mesh = ttnn.from_torch(
        torch.cat(input_tensors_per_device, dim=0),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        tile=tile,
        dtype=ttnn.bfloat16,
        memory_config=input_mem_config,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    output_shard_spec = ttnn.ShardSpec(input_shard_grid, (tile_h, output_width), ttnn.ShardOrientation.ROW_MAJOR)
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, buffer_type=ttnn.BufferType.L1, shard_spec=output_shard_spec
    )

    output_tensor_mesh = ttnn.from_torch(
        torch.zeros(tile_h * NUM_DEVICES, output_width, dtype=torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        tile=tile,
        dtype=ttnn.bfloat16,
        memory_config=output_mem_config,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    scratch_width = slice_width * 2
    scratch_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(TRANSPORT_CORE, TRANSPORT_CORE)})
    scratch_shard_spec = ttnn.ShardSpec(scratch_shard_grid, (tile_h, scratch_width), ttnn.ShardOrientation.ROW_MAJOR)
    scratch_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, buffer_type=ttnn.BufferType.L1, shard_spec=scratch_shard_spec
    )

    scratch_tensor_mesh = ttnn.from_torch(
        torch.zeros(tile_h * NUM_DEVICES, scratch_width, dtype=torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        tile=tile,
        dtype=ttnn.bfloat16,
        memory_config=scratch_mem_config,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    torch_expected = DeepseekMinimalAllGather.golden(input_tensors_per_device)

    semaphores = DeepseekMinimalAllGather.create_semaphores(mesh_device)
    ttnn.synchronize_device(mesh_device)

    return AllGatherTestInputs(
        output_shape=output_shape,
        input_tensors_per_device=input_tensors_per_device,
        torch_expected=torch_expected,
        input_tensor_mesh=input_tensor_mesh,
        output_tensor_mesh=output_tensor_mesh,
        scratch_tensor_mesh=scratch_tensor_mesh,
        semaphores=semaphores,
    )


# ---------------------------------------------------------------------------
# Verification helper
# ---------------------------------------------------------------------------


def _verify_all_gather_output(submesh, ttnn_result, inputs):
    logger.info("Verifying all-gather results...")
    tile_h = inputs.output_shape[0]

    output_torch = ttnn.to_torch(
        ttnn_result,
        mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0),
    )

    all_passed = True
    ref_device_output = output_torch[:tile_h, :]

    for device_idx in range(NUM_DEVICES):
        start_row = device_idx * tile_h
        received = output_torch[start_row : start_row + tile_h, :]

        assert received.shape == inputs.torch_expected.shape, (
            f"Shape mismatch at device {device_idx}: " f"expected {inputs.torch_expected.shape}, got {received.shape}"
        )

        if device_idx != 0:
            assert torch.equal(received, ref_device_output), f"Device {device_idx} output differs from device 0"

        if not torch.allclose(received, inputs.torch_expected, rtol=1e-2, atol=1e-2):
            logger.error(f"Output mismatch for device {device_idx}")
            logger.error(f"Expected:\n{inputs.torch_expected[:2, :8]}")
            logger.error(f"Received:\n{received[:2, :8]}")
            all_passed = False
        else:
            logger.info(f"Device {device_idx}: PASSED")

    assert all_passed, "Not all devices have the correct all-gathered data"
    logger.info("CCL all-gather test passed!")


def _verify_per_slot_identity(submesh, ttnn_result, inputs, expected_slot_tensors):
    """Verify that each slot on each device contains exactly the expected tensor."""
    tile_h = inputs.output_shape[0]
    slice_width = inputs.output_shape[1] // NUM_DEVICES

    output_torch = ttnn.to_torch(
        ttnn_result,
        mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0),
    )

    all_passed = True
    for device_idx in range(NUM_DEVICES):
        start_row = device_idx * tile_h
        device_output = output_torch[start_row : start_row + tile_h, :]

        for slot_idx in range(NUM_DEVICES):
            col_start = slot_idx * slice_width
            slot_data = device_output[:, col_start : col_start + slice_width]
            expected = expected_slot_tensors[slot_idx]

            if not torch.equal(slot_data, expected):
                expected_val = float(slot_idx + 1)
                actual_unique = slot_data.unique().tolist()
                logger.error(
                    f"Device {device_idx}, slot {slot_idx}: "
                    f"expected all {expected_val}, got unique values {actual_unique}"
                )
                all_passed = False
            else:
                logger.info(f"Device {device_idx}, slot {slot_idx}: PASSED (all {float(slot_idx + 1)})")

    assert all_passed, "Per-slot identity check failed"
    logger.info("Deterministic fill test passed!")


# ---------------------------------------------------------------------------
# Deterministic fill test (per-slot identity check)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("output_shape", [[1, 7168]])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_X,
            "fabric_router_config": create_fabric_router_config(15232),
            "trace_region_size": 573440,
        }
    ],
    indirect=True,
)
def test_ccl_all_gather_deterministic_fill(
    bh_2d_mesh_device,
    output_shape,
    num_links,
):
    """Each device's input is filled with (device_idx + 1.0).

    After all-gather, every device's output should have:
      slot 0 = all 1.0, slot 1 = all 2.0, slot 2 = all 3.0, slot 3 = all 4.0
    """
    total_devices = bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1]
    if total_devices < NUM_DEVICES:
        pytest.skip(f"Test requires {NUM_DEVICES} devices, only {total_devices} available")

    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((NUM_DEVICES, 1)))

    tile_h = output_shape[0]
    slice_width = output_shape[1] // NUM_DEVICES
    fill_tensors = [
        torch.full((tile_h, slice_width), fill_value=float(dev_idx + 1), dtype=torch.bfloat16)
        for dev_idx in range(NUM_DEVICES)
    ]

    inputs = build_all_gather_test_inputs(
        mesh_device=submesh,
        output_shape=output_shape,
        input_tensors_per_device=fill_tensors,
    )

    logger.info(f"Running deterministic fill all-gather: output_shape={output_shape}, num_links={num_links}")

    ttnn_result = DeepseekMinimalAllGather.op(
        inputs.input_tensor_mesh,
        inputs.output_tensor_mesh,
        inputs.scratch_tensor_mesh,
        inputs.semaphores,
        cluster_axis=0,
        num_links=num_links,
    )
    ttnn.synchronize_device(submesh)

    _verify_all_gather_output(submesh, ttnn_result, inputs)
    _verify_per_slot_identity(submesh, ttnn_result, inputs, fill_tensors)


# ---------------------------------------------------------------------------
# Main trace-based test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("output_shape", [[1, 7168]])
@pytest.mark.parametrize("num_links", _get_num_links_params([1]))
@pytest.mark.parametrize("num_iter, num_warmup_iter", [(30, 15)])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_X,
            "fabric_router_config": create_fabric_router_config(MAX_PAYLOAD_SIZE),
            "trace_region_size": 573440,
        }
    ],
    indirect=True,
)
def test_ccl_all_gather(
    bh_2d_mesh_device,
    output_shape,
    num_links,
    num_warmup_iter,
    num_iter,
):
    if is_slow_dispatch():
        pytest.skip("CCL all-gather trace test needs fast dispatch")

    total_devices = bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1]
    if total_devices < NUM_DEVICES:
        pytest.skip(f"Test requires {NUM_DEVICES} devices, only {total_devices} available")

    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((NUM_DEVICES, 1)))

    inputs = build_all_gather_test_inputs(mesh_device=submesh, output_shape=output_shape)

    logger.info(f"Running CCL all-gather: {NUM_DEVICES} devices, output_shape={output_shape}, num_links={num_links}")
    profiler = BenchmarkProfiler()

    logger.info("Compiling model")
    ttnn_result = DeepseekMinimalAllGather.op(
        inputs.input_tensor_mesh,
        inputs.output_tensor_mesh,
        inputs.scratch_tensor_mesh,
        inputs.semaphores,
        cluster_axis=0,
        num_links=num_links,
    )
    ttnn.synchronize_device(submesh)

    logger.info("Capturing warmup trace")
    trace_id_warmup = ttnn.begin_trace_capture(submesh, cq_id=0)
    for _ in range(num_warmup_iter):
        ttnn_result = DeepseekMinimalAllGather.op(
            inputs.input_tensor_mesh,
            inputs.output_tensor_mesh,
            inputs.scratch_tensor_mesh,
            inputs.semaphores,
            cluster_axis=0,
            num_links=num_links,
        )
    ttnn.end_trace_capture(submesh, trace_id_warmup, cq_id=0)
    ttnn.synchronize_device(submesh)

    logger.info("Capturing trace")
    trace_id = ttnn.begin_trace_capture(submesh, cq_id=0)
    for _ in range(num_iter):
        ttnn_result = DeepseekMinimalAllGather.op(
            inputs.input_tensor_mesh,
            inputs.output_tensor_mesh,
            inputs.scratch_tensor_mesh,
            inputs.semaphores,
            cluster_axis=0,
            num_links=num_links,
        )
    ttnn.end_trace_capture(submesh, trace_id, cq_id=0)
    ttnn.synchronize_device(submesh)

    logger.info("Executing warmup trace...")
    profiler.start("deepseek-all-gather-warmup")
    ttnn.execute_trace(submesh, trace_id_warmup, blocking=False)
    ttnn.release_trace(submesh, trace_id_warmup)
    ttnn.synchronize_device(submesh)
    profiler.end("deepseek-all-gather-warmup")

    logger.info("Starting Trace perf test...")
    signpost("start")
    profiler.start("deepseek-all-gather-trace")
    ttnn.execute_trace(submesh, trace_id, blocking=False)
    ttnn.release_trace(submesh, trace_id)
    ttnn.synchronize_device(submesh)
    profiler.end("deepseek-all-gather-trace")
    signpost("stop")

    _verify_all_gather_output(submesh, ttnn_result, inputs)


# ---------------------------------------------------------------------------
# Feature-matrix test (shape, num_links, chunk sizing)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("output_shape", [[1, 896], [1, 7168]])
@pytest.mark.parametrize("num_links", [1, 2])
@pytest.mark.parametrize(
    "max_chunk_size_bytes",
    [
        None,
        2048,
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_X,
            "fabric_router_config": create_fabric_router_config(15232),
            "trace_region_size": 573440,
        }
    ],
    indirect=True,
)
def test_ccl_all_gather_chunk_matrix(
    bh_2d_mesh_device,
    output_shape,
    num_links,
    max_chunk_size_bytes,
):
    total_devices = bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1]
    if total_devices < NUM_DEVICES:
        pytest.skip(f"Test requires {NUM_DEVICES} devices, only {total_devices} available")

    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((NUM_DEVICES, 1)))

    inputs = build_all_gather_test_inputs(mesh_device=submesh, output_shape=output_shape)

    logger.info(
        f"Running all-gather feature matrix: output_shape={output_shape}, "
        f"num_links={num_links}, max_chunk_size_bytes={max_chunk_size_bytes}"
    )
    ttnn_result = DeepseekMinimalAllGather.op(
        inputs.input_tensor_mesh,
        inputs.output_tensor_mesh,
        inputs.scratch_tensor_mesh,
        inputs.semaphores,
        cluster_axis=0,
        num_links=num_links,
        max_chunk_size_bytes=max_chunk_size_bytes,
    )
    ttnn.synchronize_device(submesh)

    _verify_all_gather_output(submesh, ttnn_result, inputs)
