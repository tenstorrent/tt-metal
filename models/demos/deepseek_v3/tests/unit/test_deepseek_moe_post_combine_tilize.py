# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc


@pytest.mark.requires_device(["TG", "DUAL", "QUAD"])
@pytest.mark.parametrize("iterations", [10])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": 135168,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
        }
    ],
    indirect=True,
)
def test_deepseek_moe_post_combine_tilize(mesh_device, iterations):
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape((1, 1)))

    deepseek_moe_post_combine_tilize_output_memory_config = ttnn.MemoryConfig(
        buffer_type=ttnn.BufferType.L1,
        nd_shard_spec=ttnn.NdShardSpec(
            shard_shape=[32, 1024],
            grid=ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(6, 7)),
                }
            ),
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    tt_token_inputs = []
    tt_scores_inputs = []
    goldens = []
    for i in range(iterations):
        torch_token_input = torch.rand((8, 1, 32, 7168), dtype=torch.bfloat16)
        tt_token_input = ttnn.from_torch(
            torch_token_input,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            device=submesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(submesh_device),
        )
        tt_token_inputs.append(tt_token_input)

        torch_scores_input = torch.rand((8, 1, 32, 1), dtype=torch.bfloat16)
        tt_scores_input = ttnn.from_torch(
            torch_scores_input,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            device=submesh_device,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(submesh_device),
        )
        tt_scores_inputs.append(tt_scores_input)

        golden = torch_token_input * torch_scores_input
        goldens.append(golden)

    def run_test(i):
        tt_tilized_token_input = ttnn.experimental.deepseek_moe_post_combine_tilize(
            tt_token_inputs[i],
            output_memory_config=deepseek_moe_post_combine_tilize_output_memory_config,
        )

        tt_mul = ttnn.mul(tt_tilized_token_input, tt_scores_inputs[i], memory_config=ttnn.L1_MEMORY_CONFIG)

        return tt_mul

    # compile
    for i in range(iterations):
        run_test(i)

    # capture
    tt_outputs = []
    trace_id = ttnn.begin_trace_capture(submesh_device, cq_id=0)
    for i in range(iterations):
        tt_output = run_test(i)
        tt_outputs.append(tt_output)
    ttnn.end_trace_capture(submesh_device, trace_id, cq_id=0)

    # execute
    ttnn.execute_trace(submesh_device, trace_id, cq_id=0, blocking=False)

    all_iterations_passed = True
    for i in range(iterations):
        torch_out = ttnn.to_torch(
            tt_outputs[i], dtype=torch.bfloat16, mesh_composer=ttnn.ConcatMeshToTensor(submesh_device, dim=0)
        )
        golden = goldens[i]

        pcc_passed, pcc_output = comp_pcc(torch_out, golden)
        logger.info(f"Iteration: {i} - PCC: {pcc_output}")
        if not pcc_passed:
            logger.warning(f"FAILED Iteration: {i} - PCC: {pcc_output}")
            all_iterations_passed = False

    assert all_iterations_passed, "deepseek_moe_post_combine_tilize Verification Failed!"

    ttnn.ReadDeviceProfiler(submesh_device)
