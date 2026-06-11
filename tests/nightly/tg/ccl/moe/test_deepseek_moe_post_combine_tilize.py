# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Unit test for ``deepseek_moe_post_combine_tilize`` — the fused kernel that tilizes
the ROW_MAJOR post-combine token activation before it is scaled by the per-token
scores. The op is a purely local per-device operation (no inter-chip fabric), so
the device argument may be a full mesh or a single-device submesh; the kernel runs
independently and identically on every device it is given.

The reusable core (``run_post_combine_tilize_test``) is shared with the original
single-device unit test at
``models/demos/deepseek_v3/tests/unit/test_deepseek_moe_post_combine_tilize.py``,
which imports it from here (same pattern as ``test_combine_tg.py`` reusing
``test_selective_combine_6U``). This file adds a (1, 8) Blackhole mesh case that
exercises the op replicated across all eight devices.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, is_blackhole


def make_post_combine_tilize_output_memory_config(grid_end=(6, 7), shard_shape=(32, 1024)):
    """L1 ND-sharded output config for the tilized token activation.

    The default grid (0,0)-(6,7) = 56 cores fits both the Wormhole 8x8 compute grid
    and the larger Blackhole grid, so the same config is reused across architectures.
    """
    return ttnn.MemoryConfig(
        buffer_type=ttnn.BufferType.L1,
        nd_shard_spec=ttnn.NdShardSpec(
            shard_shape=list(shard_shape),
            grid=ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(*grid_end)),
                }
            ),
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )


def run_post_combine_tilize_test(
    target_device,
    iterations,
    *,
    token_shape=(8, 1, 32, 7168),
    output_memory_config=None,
):
    """Run the fused ``deepseek_moe_post_combine_tilize`` → ``mul`` pipeline under a
    captured trace and verify each device's output against the host golden.

    Builds ``iterations`` independent (token, scores) input pairs replicated onto
    ``target_device``, captures one trace covering all iterations, executes it, then
    checks every device tensor of every iteration against ``token * scores``.

    ``target_device`` may be a full mesh or a submesh — the op runs replicated across
    whatever devices it contains, and each device's result is verified independently
    (so this works for the original (1, 1) submesh and a (1, 8) replicated mesh alike).

    Asserts that every (iteration, device) output passes PCC.
    """
    if output_memory_config is None:
        output_memory_config = make_post_combine_tilize_output_memory_config()

    scores_shape = (token_shape[0], token_shape[1], token_shape[2], 1)

    tt_token_inputs = []
    tt_scores_inputs = []
    goldens = []
    for _ in range(iterations):
        torch_token_input = torch.rand(token_shape, dtype=torch.bfloat16)
        tt_token_input = ttnn.from_torch(
            torch_token_input,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            device=target_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(target_device),
        )
        tt_token_inputs.append(tt_token_input)

        torch_scores_input = torch.rand(scores_shape, dtype=torch.bfloat16)
        tt_scores_input = ttnn.from_torch(
            torch_scores_input,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            device=target_device,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(target_device),
        )
        tt_scores_inputs.append(tt_scores_input)

        goldens.append(torch_token_input * torch_scores_input)

    def run_iter(i):
        tt_tilized_token_input = ttnn.experimental.deepseek_moe_post_combine_tilize(
            tt_token_inputs[i],
            output_memory_config=output_memory_config,
        )
        return ttnn.mul(tt_tilized_token_input, tt_scores_inputs[i], memory_config=ttnn.L1_MEMORY_CONFIG)

    # compile
    for i in range(iterations):
        run_iter(i)

    # capture
    tt_outputs = []
    trace_id = ttnn.begin_trace_capture(target_device, cq_id=0)
    for i in range(iterations):
        tt_outputs.append(run_iter(i))
    ttnn.end_trace_capture(target_device, trace_id, cq_id=0)

    # execute
    ttnn.execute_trace(target_device, trace_id, cq_id=0, blocking=False)

    all_iterations_passed = True
    for i in range(iterations):
        golden = goldens[i]
        for didx, dev_tensor in enumerate(ttnn.get_device_tensors(tt_outputs[i])):
            torch_out = ttnn.to_torch(dev_tensor, dtype=torch.bfloat16)
            pcc_passed, pcc_output = comp_pcc(torch_out, golden)
            logger.info(f"Iteration: {i} device: {didx} - PCC: {pcc_output}")
            if not pcc_passed:
                logger.warning(f"FAILED Iteration: {i} device: {didx} - PCC: {pcc_output}")
                all_iterations_passed = False

    assert all_iterations_passed, "deepseek_moe_post_combine_tilize Verification Failed!"

    ttnn.ReadDeviceProfiler(target_device)


@pytest.mark.skipif(not is_blackhole(), reason="1x8 mesh case targets a Blackhole 1x8 machine")
@pytest.mark.parametrize("iterations", [10])
@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 500000, "dispatch_core_axis": ttnn.DispatchCoreAxis.COL}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [pytest.param((1, 8), id="1x8_grid_bh")],
    indirect=["mesh_device"],
)
def test_deepseek_moe_post_combine_tilize_bh(mesh_device, iterations):
    """Blackhole 1x8 case: run the op replicated across all eight devices."""
    run_post_combine_tilize_test(mesh_device, iterations)
