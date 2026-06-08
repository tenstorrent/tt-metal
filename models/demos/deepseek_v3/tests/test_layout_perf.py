# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Device-perf test for the gate INPUT reformat into the multi-tile (option-1) gate layout.

Source = the gate-proj matmul result: ``(1,1,32,N)`` bf16, TILE, **L1 interleaved**
(32 = batch = 32 tokens, one core per token).

Target (option 1, what the >256 gate design needs) = each token's N experts split into
ceil(N/256) blocks of 256, **each block padded into face0 of its own 32x32 tile**, height-sharded
one token per core (so each core holds ``num_blocks`` tiles, block b in face0 of tile b).

Getting "separate tiles, each block in face0" from the *contiguous* (32,N) needs an explicit pad
(256 -> 32x32 tile-padded), so the real reformat is reshape -> pad -> to_memory_config. Each op's
device time is measured (trace + signposts). bias/indices/host prep are out of scope.
"""

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn

BATCH = 32  # tokens, one per core


@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 7000000}],
    indirect=True,
)
@pytest.mark.parametrize("num_experts", [512])
@pytest.mark.parametrize("mode", ["two_tiles", "two_faces", "slice_two_tiles"])
@pytest.mark.parametrize("warmup_iters, num_iters", [(5, 10)])
def test_layout_perf(device, num_experts, mode, warmup_iters, num_iters, device_params):
    """Capture a trace of the logits reformat. Two candidate target layouts:
    - two_tiles (option 1): reshape -> PAD (256->32x32 per block) -> reshard. Each block in face0 of
      its own tile (num_blocks tiles/core). Clean kernel, but the pad is expensive.
    - two_faces (option 2): reshape -> reshard, NO pad. The N experts fill faces of one tile per core
      ((16,32) for 512). Cheap reformat, but the kernel must process across faces.
    """
    num_blocks = num_experts // 256  # 512 -> 2

    src = torch.rand((1, 1, BATCH, num_experts), dtype=torch.bfloat16)
    tt_src = ttnn.from_torch(
        src,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,  # interleaved L1 (matmul output)
    )

    grid = device.compute_with_storage_grid_size()
    core_grid = ttnn.num_cores_to_corerangeset(BATCH, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)

    if mode == "two_tiles":
        shard_shape = (num_blocks * 32, 32)  # num_blocks tiles stacked per core
    else:  # two_faces / slice_two_tiles: one (32,32) tile per core (per block for the slice variant)
        shard_shape = (32, 32)
    sharded_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
    )

    def run_reformat():
        if mode == "two_tiles":
            # (1,1,32,N) -> (32, num_blocks, 16, 16); PAD each (16,16) block -> (32,32) face0; reshard.
            r = ttnn.reshape(tt_src, (BATCH, num_blocks, 16, 16))
            p = ttnn.pad(r, [(0, 0), (0, 0), (0, 16), (0, 16)], value=0.0)
            return ttnn.to_memory_config(p, memory_config=sharded_mem_config)
        elif mode == "two_faces":
            # (1,1,32,N) -> (32, 16, N//16): N fills faces of one tile; NO pad; reshard.
            r = ttnn.reshape(tt_src, (BATCH, 16, num_experts // 16))
            return ttnn.to_memory_config(r, memory_config=sharded_mem_config)
        else:  # slice_two_tiles (the proposed approach): keep the 2 blocks as SEPARATE tensors, so each
            # (16,16) gets its own (32,32) tile via IMPLICIT shard padding (face0) -- NO explicit pad op.
            outs = []
            for b in range(num_blocks):
                blk = ttnn.slice(tt_src, [0, 0, 0, b * 256], [1, 1, BATCH, (b + 1) * 256])  # (1,1,32,256)
                r = ttnn.reshape(blk, (BATCH, 16, 16))  # implicit tile -> face0
                outs.append(ttnn.to_memory_config(r, memory_config=sharded_mem_config))
            return outs

    # Compile
    run_reformat()
    ttnn.synchronize_device(device)

    # Warmup trace
    trace_id_warmup = ttnn.begin_trace_capture(device, cq_id=0)
    for _ in range(warmup_iters):
        run_reformat()
    ttnn.end_trace_capture(device, trace_id_warmup, cq_id=0)
    ttnn.synchronize_device(device)

    # Main trace
    logger.info(f"Capturing main trace (reformat, {num_experts} experts, {num_blocks} tiles/core, {num_iters} iters)")
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    for _ in range(num_iters):
        run_reformat()
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.synchronize_device(device)

    ttnn.execute_trace(device, trace_id_warmup, blocking=False)
    ttnn.release_trace(device, trace_id_warmup)

    signpost("start")
    ttnn.execute_trace(device, trace_id, blocking=False)
    ttnn.release_trace(device, trace_id)
    signpost("stop")

    # Sanity: one eager call.
    out = run_reformat()
    ttnn.synchronize_device(device)
    shapes = [tuple(t.shape) for t in out] if isinstance(out, list) else tuple(out.shape)
    logger.info(f"reformat out shape={shapes} (mode={mode})")

    ttnn.deallocate(tt_src)


if __name__ == "__main__":
    pytest.main([__file__])
