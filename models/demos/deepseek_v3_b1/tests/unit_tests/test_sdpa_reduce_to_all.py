# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_slow_dispatch, skip_for_wormhole_b0
from models.demos.deepseek_v3_b1.metadata.metadata import DeepseekMetadata, create_metadata_tensor
from models.demos.deepseek_v3_b1.micro_ops.sdpa_reduce_to_all.config import round_up
from models.demos.deepseek_v3_b1.micro_ops.sdpa_reduce_to_all.op import SdpaReduceToAll, compute_forwarder_scratch_size
from models.demos.deepseek_v3_b1.tests.unit_tests.ccl_test_utils import (
    create_fabric_router_config,
    get_env_int,
    get_optional_env_int,
    run_trace_benchmark,
)
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.box.nightly.test_all_gather_nightly import validate_test

NUM_DEVICES = 4
NUM_CORES = 8
L_WIDTH = 512
MS_WIDTH = 32
BATCH_SIZE = 8
SCALE_VALUE = 1.0
PER_DEVICE_CHUNK_SIZE = 1024

TRACE_POSITION_ID = 3500
TRACE_SCATTER_ENABLED = False

ENV_MAX_PAYLOAD_SIZE = "SDPA_REDUCE_TO_ALL_MAX_PAYLOAD_SIZE_BYTES"
TRACE_MAX_PAYLOAD_SIZE = get_env_int(ENV_MAX_PAYLOAD_SIZE, 15232)
ENV_NUM_L_CHUNKS = "SDPA_REDUCE_TO_ALL_NUM_L_CHUNKS"
TRACE_NUM_L_CHUNKS = get_optional_env_int(ENV_NUM_L_CHUNKS)
ENV_COMPUTE_BLOCK_SIZE = "SDPA_REDUCE_TO_ALL_COMPUTE_BLOCK_SIZE"
TRACE_COMPUTE_BLOCK_SIZE = get_optional_env_int(ENV_COMPUTE_BLOCK_SIZE)


@dataclass(frozen=True)
class SdpaReduceToAllTestInputs:
    submesh_device: Any
    num_devices: int
    num_cores: int
    batch_size: int
    l_width: int
    scale_value: float
    per_device_chunk_size: int
    num_l_chunks_override: int | None
    compute_block_size_override: int | None
    forwarder_cores: list[Any]
    input_l_mesh: Any
    input_ms_mesh: Any
    output_l_mesh: Any
    interm_recv_mesh: Any
    forwarder_scratch_mesh: Any
    scatter_dest_mesh: Any
    scatter_grid: Any
    position_id_tensor_mesh: Any
    semaphores: list[Any]
    ref_l: torch.Tensor
    max_diff_check: float


def build_sdpa_reduce_to_all_test_inputs(
    bh_2d_mesh_device,
    *,
    scatter_enabled: bool,
    position_id: int,
    max_payload_size_bytes: int | None = None,
    num_l_chunks_override: int | None = None,
    compute_block_size_override: int | None = None,
) -> SdpaReduceToAllTestInputs:
    l_shape = [BATCH_SIZE, L_WIDTH * NUM_CORES]
    ms_shape = [BATCH_SIZE, MS_WIDTH * NUM_CORES]
    intermediate_shape = [2, BATCH_SIZE, (L_WIDTH + MS_WIDTH) * NUM_CORES]

    topology = ttnn.Topology.Torus
    validate_test(NUM_DEVICES, topology, bh_2d_mesh_device.shape, 0)
    submesh_device = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((NUM_DEVICES, 1)))

    dtype = ttnn.bfloat16
    layout = ttnn.TILE_LAYOUT
    final_output_layout = ttnn.ROW_MAJOR_LAYOUT
    tile = ttnn.Tile((8, 32))

    forwarder_cores = [ttnn.CoreCoord(6, 8), ttnn.CoreCoord(6, 9)]

    shard_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(ttnn.CoreCoord(2, 8), ttnn.CoreCoord(5, 8)),
            ttnn.CoreRange(ttnn.CoreCoord(2, 9), ttnn.CoreCoord(5, 9)),
        }
    )
    shard_spec_l = ttnn.ShardSpec(shard_grid, [BATCH_SIZE, L_WIDTH], ttnn.ShardOrientation.ROW_MAJOR)
    shard_spec_ms = ttnn.ShardSpec(shard_grid, [BATCH_SIZE, MS_WIDTH], ttnn.ShardOrientation.ROW_MAJOR)
    shard_spec_int = ttnn.ShardSpec(shard_grid, [2 * BATCH_SIZE, (L_WIDTH + MS_WIDTH)], ttnn.ShardOrientation.ROW_MAJOR)

    mem_config_l = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.WIDTH_SHARDED, ttnn.types.BufferType.L1, shard_spec_l
    )
    mem_config_ms = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.WIDTH_SHARDED, ttnn.types.BufferType.L1, shard_spec_ms
    )
    mem_config_int = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.WIDTH_SHARDED, ttnn.types.BufferType.L1, shard_spec_int
    )

    mesh_mapper_config = ttnn.MeshMapperConfig(
        [ttnn.PlacementShard(0), ttnn.PlacementReplicate()], submesh_device.shape
    )
    mesh_mapper = ttnn.create_mesh_mapper(submesh_device, mesh_mapper_config)

    mesh_mapper_config2 = ttnn.MeshMapperConfig(
        [ttnn.PlacementReplicate(), ttnn.PlacementReplicate()], submesh_device.shape
    )
    mesh_mapper2 = ttnn.create_mesh_mapper(submesh_device, mesh_mapper_config2)

    torch.manual_seed(42)
    l_data_per_device = [torch.randn(l_shape, dtype=torch.float32).to(torch.bfloat16) for _ in range(NUM_DEVICES)]
    ms_data_per_device = [torch.randn(ms_shape, dtype=torch.float32).to(torch.bfloat16) for _ in range(NUM_DEVICES)]

    position_mask = torch.tensor(
        [1.0 if position_id >= d * PER_DEVICE_CHUNK_SIZE else 0.0 for d in range(NUM_DEVICES)],
        dtype=torch.bfloat16,
    )
    m_data_per_device = []
    s_data_per_device = []
    for device_idx in range(NUM_DEVICES):
        ms_device = ms_data_per_device[device_idx]
        m_device = torch.zeros((ms_shape[0], NUM_CORES), dtype=torch.bfloat16)
        s_device = torch.zeros((ms_shape[0], NUM_CORES), dtype=torch.bfloat16)
        for core_idx in range(NUM_CORES):
            m_device[:, core_idx] = torch.randn(ms_shape[0], dtype=torch.bfloat16) * 0.5 - 1.0
            s_device[:, core_idx] = torch.abs(torch.randn(ms_shape[0], dtype=torch.bfloat16)) + 0.1
            ms_device[:, core_idx * MS_WIDTH + 0] = m_device[:, core_idx]
            ms_device[:, core_idx * MS_WIDTH + 1] = s_device[:, core_idx]
        m_data_per_device.append(m_device)
        s_data_per_device.append(s_device)

    l_data_f32 = [tensor.float() for tensor in l_data_per_device]
    s_data_f32 = [tensor.float() for tensor in s_data_per_device]
    m_data_f32 = [tensor.float() for tensor in m_data_per_device]

    ref_l, _, _ = SdpaReduceToAll.golden(
        l_data_f32,
        s_data_f32,
        m_data_f32,
        NUM_CORES,
        SCALE_VALUE,
        position_id,
        PER_DEVICE_CHUNK_SIZE,
    )
    ref_l = ref_l.to(torch.bfloat16)

    l_data_all = torch.stack(l_data_per_device, dim=0)
    ms_data_all = torch.stack(ms_data_per_device, dim=0)

    input_l_mesh = ttnn.from_torch(
        l_data_all,
        device=submesh_device,
        layout=layout,
        tile=tile,
        dtype=dtype,
        memory_config=mem_config_l,
        mesh_mapper=mesh_mapper,
    )
    input_ms_mesh = ttnn.from_torch(
        ms_data_all,
        device=submesh_device,
        layout=layout,
        tile=tile,
        dtype=dtype,
        memory_config=mem_config_ms,
        mesh_mapper=mesh_mapper,
    )

    output_l_mesh = ttnn.from_torch(
        torch.zeros_like(l_data_all),
        device=submesh_device,
        layout=final_output_layout,
        tile=tile,
        dtype=dtype,
        memory_config=mem_config_l,
        mesh_mapper=mesh_mapper,
    )

    interm_recv_mesh = ttnn.from_torch(
        torch.zeros(intermediate_shape, dtype=torch.bfloat16),
        device=submesh_device,
        layout=layout,
        tile=tile,
        dtype=dtype,
        memory_config=mem_config_int,
        mesh_mapper=mesh_mapper2,
    )

    forwarder_buffer_size_bytes = compute_forwarder_scratch_size(
        batch_size=BATCH_SIZE,
        l_width=L_WIDTH,
        num_cores=NUM_CORES,
        tile_height=8,
        tile_width=32,
        bytes_per_element=2,
        max_payload_size_bytes=max_payload_size_bytes,
        num_l_chunks_override=num_l_chunks_override,
        compute_block_size_override=compute_block_size_override,
    )

    num_forwarder_cores = 2
    forwarder_shard_width_elements = forwarder_buffer_size_bytes // (tile.tile_shape[0] * 2)
    forwarder_shard_width_elements = round_up(forwarder_shard_width_elements, tile.tile_shape[1])

    forwarder_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(forwarder_cores[0], forwarder_cores[1])})
    forwarder_shard_spec = ttnn.ShardSpec(
        forwarder_shard_grid, [tile.tile_shape[0], forwarder_shard_width_elements], ttnn.ShardOrientation.ROW_MAJOR
    )
    forwarder_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.WIDTH_SHARDED, ttnn.types.BufferType.L1, forwarder_shard_spec
    )
    forwarder_scratch_shape = [tile.tile_shape[0], forwarder_shard_width_elements * num_forwarder_cores]
    forwarder_scratch_mesh = ttnn.from_torch(
        torch.zeros(forwarder_scratch_shape, dtype=torch.bfloat16),
        device=submesh_device,
        layout=layout,
        tile=tile,
        dtype=dtype,
        memory_config=forwarder_mem_config,
        mesh_mapper=mesh_mapper2,
    )

    scatter_dest_mesh = None
    scatter_grid = None
    num_scatter_cores = NUM_CORES * BATCH_SIZE
    if scatter_enabled:
        scatter_tile = ttnn.Tile((1, 32))
        scatter_grid = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(NUM_CORES - 1, BATCH_SIZE - 1))}
        )
        scatter_shard_shape = [1, L_WIDTH]
        scatter_shard_spec = ttnn.ShardSpec(scatter_grid, scatter_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
        scatter_mem_config = ttnn.MemoryConfig(
            ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, scatter_shard_spec
        )
        scatter_dest_mesh = ttnn.from_torch(
            torch.zeros([num_scatter_cores, L_WIDTH], dtype=torch.bfloat16),
            device=submesh_device,
            layout=layout,
            tile=scatter_tile,
            dtype=dtype,
            memory_config=scatter_mem_config,
            mesh_mapper=mesh_mapper2,
        )

    # ========================================================================
    # Metadata tensor: HEIGHT_SHARDED, uint32, replicated across mesh
    # Each worker core gets [1, 2] shard containing DeepseekMetadata fields.
    # ========================================================================
    metadata = DeepseekMetadata(position_id=position_id)
    position_id_tensor_mesh = create_metadata_tensor(submesh_device, shard_grid, metadata)

    semaphores = [ttnn.create_global_semaphore(submesh_device, shard_grid, 0) for _ in range(2)]
    ttnn.synchronize_device(submesh_device)

    max_diff_check = 0.07 if (position_mask.sum() > 1.0).item() else 0.13
    return SdpaReduceToAllTestInputs(
        submesh_device=submesh_device,
        num_devices=NUM_DEVICES,
        num_cores=NUM_CORES,
        batch_size=BATCH_SIZE,
        l_width=L_WIDTH,
        scale_value=SCALE_VALUE,
        per_device_chunk_size=PER_DEVICE_CHUNK_SIZE,
        num_l_chunks_override=num_l_chunks_override,
        compute_block_size_override=compute_block_size_override,
        forwarder_cores=forwarder_cores,
        input_l_mesh=input_l_mesh,
        input_ms_mesh=input_ms_mesh,
        output_l_mesh=output_l_mesh,
        interm_recv_mesh=interm_recv_mesh,
        forwarder_scratch_mesh=forwarder_scratch_mesh,
        scatter_dest_mesh=scatter_dest_mesh,
        scatter_grid=scatter_grid,
        position_id_tensor_mesh=position_id_tensor_mesh,
        semaphores=semaphores,
        ref_l=ref_l,
        max_diff_check=max_diff_check,
    )


def run_sdpa_reduce_to_all(inputs: SdpaReduceToAllTestInputs):
    return SdpaReduceToAll.op(
        inputs.input_l_mesh,
        inputs.input_ms_mesh,
        inputs.output_l_mesh,
        inputs.interm_recv_mesh,
        inputs.forwarder_scratch_mesh,
        inputs.semaphores,
        scale_fp32=inputs.scale_value,
        cluster_axis=0,
        input_forwarder_cores=inputs.forwarder_cores,
        scatter_dest_tensor_mesh=inputs.scatter_dest_mesh,
        scatter_dest_grid=inputs.scatter_grid,
        position_id_tensor_mesh=inputs.position_id_tensor_mesh,
        per_device_chunk_size=inputs.per_device_chunk_size,
        num_l_chunks_override=inputs.num_l_chunks_override,
        compute_block_size_override=inputs.compute_block_size_override,
    )


def verify_sdpa_reduce_to_all_output(
    inputs: SdpaReduceToAllTestInputs,
    output_mesh,
    *,
    scatter_enabled: bool,
) -> None:
    output_l_torch = ttnn.to_torch(output_mesh, mesh_composer=ttnn.ConcatMeshToTensor(inputs.submesh_device, dim=0))
    out_l_root = output_l_torch[0]

    max_diff = torch.max(torch.abs(out_l_root.flatten().float() - inputs.ref_l.flatten().float())).item()
    match = max_diff < inputs.max_diff_check

    logger.info(f"L tensor match: {match}, max_diff: {max_diff:.4f}")
    assert match, f"L tensor mismatch! Max diff: {max_diff}"

    for device_idx in range(1, inputs.num_devices):
        dev_eq = torch.equal(output_l_torch[device_idx], out_l_root)
        assert dev_eq, f"L tensor mismatch on device {device_idx}"

    if scatter_enabled:
        num_scatter_cores = inputs.num_cores * inputs.batch_size
        scatter_out_torch = ttnn.to_torch(
            inputs.scatter_dest_mesh,
            mesh_composer=ttnn.ConcatMeshToTensor(inputs.submesh_device, dim=0),
        )
        scatter_out_root = scatter_out_torch[:num_scatter_cores, :]

        scatter_max_diff = 0.0
        for batch_idx in range(inputs.batch_size):
            for core_idx in range(inputs.num_cores):
                row_idx = batch_idx * inputs.num_cores + core_idx
                expected = inputs.ref_l[batch_idx, core_idx * inputs.l_width : (core_idx + 1) * inputs.l_width].float()
                actual = scatter_out_root[row_idx, :].float()
                diff = torch.max(torch.abs(actual - expected)).item()
                scatter_max_diff = max(scatter_max_diff, diff)

        scatter_match = scatter_max_diff < inputs.max_diff_check
        logger.info(f"Scatter output match: {scatter_match}, max_diff: {scatter_max_diff:.4f}")
        assert scatter_match, f"Scatter output mismatch! Max diff: {scatter_max_diff}"


@skip_for_wormhole_b0("This test is for blackhole")
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_X}],
    indirect=["device_params"],
)
@pytest.mark.parametrize("scatter_enabled", [False, True], ids=["reduce_only", "reduce_and_scatter"])
@pytest.mark.parametrize("position_id", [500, 1500, 2500, 3500], ids=["pos500", "pos1500", "pos2500", "pos3500"])
def test_sdpa_reduce_to_all(bh_2d_mesh_device, scatter_enabled, position_id):
    inputs = build_sdpa_reduce_to_all_test_inputs(
        bh_2d_mesh_device,
        scatter_enabled=scatter_enabled,
        position_id=position_id,
    )

    logger.info(f"Running SDPA reduce-to-all (scatter={'enabled' if scatter_enabled else 'disabled'})...")
    output_mesh = run_sdpa_reduce_to_all(inputs)
    ttnn.synchronize_device(inputs.submesh_device)

    verify_sdpa_reduce_to_all_output(inputs, output_mesh, scatter_enabled=scatter_enabled)


@skip_for_wormhole_b0("This test is for blackhole")
@pytest.mark.parametrize("num_iter, num_warmup_iter", [(100, 25)])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_X,
            "fabric_router_config": create_fabric_router_config(TRACE_MAX_PAYLOAD_SIZE),
            "trace_region_size": 2965504,
        }
    ],
    indirect=["device_params"],
)
def test_sdpa_reduce_to_all_trace(
    bh_2d_mesh_device,
    num_warmup_iter,
    num_iter,
):
    if is_slow_dispatch():
        pytest.skip("SDPA reduce-to-all trace test needs fast dispatch")

    inputs = build_sdpa_reduce_to_all_test_inputs(
        bh_2d_mesh_device,
        scatter_enabled=TRACE_SCATTER_ENABLED,
        position_id=TRACE_POSITION_ID,
        max_payload_size_bytes=TRACE_MAX_PAYLOAD_SIZE,
        num_l_chunks_override=TRACE_NUM_L_CHUNKS,
        compute_block_size_override=TRACE_COMPUTE_BLOCK_SIZE,
    )

    logger.info(
        "Running SDPA reduce-to-all trace: "
        f"forwarder_cores=2, scatter={'enabled' if TRACE_SCATTER_ENABLED else 'disabled'}, "
        f"position_id={TRACE_POSITION_ID}, max_payload_size_bytes={TRACE_MAX_PAYLOAD_SIZE}, "
        f"num_l_chunks={'default' if TRACE_NUM_L_CHUNKS is None else TRACE_NUM_L_CHUNKS}, "
        f"compute_block_size={'default' if TRACE_COMPUTE_BLOCK_SIZE is None else TRACE_COMPUTE_BLOCK_SIZE}"
    )

    output_mesh = run_trace_benchmark(
        inputs.submesh_device,
        lambda: run_sdpa_reduce_to_all(inputs),
        num_warmup_iter=num_warmup_iter,
        num_iter=num_iter,
        profiler_name="deepseek-sdpa-reduce-to-all",
    )

    verify_sdpa_reduce_to_all_output(inputs, output_mesh, scatter_enabled=TRACE_SCATTER_ENABLED)
