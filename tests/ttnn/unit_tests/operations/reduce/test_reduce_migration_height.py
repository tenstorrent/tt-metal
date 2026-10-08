# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn


@pytest.mark.parametrize("placement", ["one_core", "two_groups", "width_sharded"])
@pytest.mark.parametrize("full_sync", [False, True])
@pytest.mark.parametrize(
    "operation,dtype,fp32_dest",
    [
        ("sum", "bf16", False),
        ("mean", "bf16", True),
        ("max", "bf16", False),
        ("min", "bf16", False),
        ("min", "bf16", True),
        ("mean", "fp32", True),
        ("max", "int32", True),
    ],
)
def test_reduce_height_column_groups(device, placement, full_sync, operation, dtype, fp32_dest):
    # More columns than DEST slots, a partial H tile, and a short last group.
    # Two interleaved cores receive different widths; sharding repeats the
    # rectangle for two batches and exercises the negate reader's small FIFO.
    torch.manual_seed(714)
    batches, width_tiles = (2, 34) if placement == "width_sharded" else (1, 35 if placement == "two_groups" else 17)
    # Sharded fill-pad currently supports one batch only, so keep repeated
    # sharded batches tile-aligned and exercise partial H on interleaved inputs.
    shape = (batches, 1, 96 if placement == "width_sharded" else 95, width_tiles * 32)
    torch_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32, "int32": torch.int32}[dtype]
    source = (
        torch.randint(-100, 100, shape, dtype=torch.int32) if dtype == "int32" else torch.randn(shape).to(torch_dtype)
    )
    if operation == "sum":
        # Exact BF16 addends isolate stream ordering from cancellation error.
        source = (torch.randint(-4, 5, shape).float() / 8).to(torch_dtype)
    tt_dtype = {"bf16": ttnn.bfloat16, "fp32": ttnn.float32, "int32": ttnn.int32}[dtype]
    cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0 if placement == "one_core" else 1, 0))}
    )
    input_memory = output_memory = ttnn.DRAM_MEMORY_CONFIG
    if placement == "width_sharded":
        input_memory = ttnn.create_sharded_memory_config(
            (batches * 96, 17 * 32),
            cores,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        output_memory = ttnn.create_sharded_memory_config(
            (batches * 32, 17 * 32),
            cores,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
    value = ttnn.from_torch(source, dtype=tt_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=input_memory)
    config = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=fp32_dest, dst_full_sync_en=full_sync
    )
    actual = ttnn.to_torch(
        getattr(ttnn, operation)(
            value,
            dim=-2,
            keepdim=True,
            memory_config=output_memory,
            compute_kernel_config=config,
            sub_core_grids=cores,
        )
    )
    reference = ({"max": torch.amax, "min": torch.amin}.get(operation, getattr(torch, operation)))(
        source.float() if operation in ("sum", "mean") else source,
        dim=-2,
        keepdim=True,
    )
    if operation in ("max", "min"):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0, check_dtype=False)
    else:
        torch.testing.assert_close(
            actual.float(), reference.float(), rtol=0.03, atol=0.04 if operation == "sum" else 0.001
        )
