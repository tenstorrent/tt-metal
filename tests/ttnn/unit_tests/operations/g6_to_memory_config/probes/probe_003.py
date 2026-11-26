import ttnn, torch
from ttnn.operations.g6_to_memory_config.g6_to_memory_config_program_descriptor import (
    Geometry,
    Placement,
    owner_shards,
    shard_owner_cores,
    num_shards,
)

print("is_tensor_storage_on_device", hasattr(ttnn, "is_tensor_storage_on_device"))
k = ttnn.KernelDescriptor(
    kernel_source="x.cpp",
    core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]),
    compile_time_args=[],
    runtime_args=ttnn.RuntimeArgs(),
    common_runtime_args=[0, 0],
    config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
)
pd = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[])
for kk in pd.kernels:
    kk.common_runtime_args = [7, 9]
print("after mutate via iteration:", list(pd.kernels[0].common_runtime_args))
pd.kernels[0].common_runtime_args = [11, 13]
print("after mutate via index:", list(pd.kernels[0].common_runtime_args))
pd.custom_program_hash = 2**63 + 5
print("hash", pd.custom_program_hash)


def check(name, shape, scheme, ranges, shard_shape):
    import math

    lead = math.prod(shape[:-2])
    ttr = lead * math.ceil(shape[-2] / 32)
    ttc = math.ceil(shape[-1] / 32)
    g = Geometry(32, 32, ttr, ttc, 2048)
    p = Placement("l1", scheme, tuple(ranges), tuple(shard_shape), "row_major")
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(a, b), ttnn.CoreCoord(c, d)) for a, b, c, d in ranges])
    strat = (
        ttnn.ShardDistributionStrategy.GRID_2D
        if scheme == "block_sharded"
        else ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D
    )
    padded = [lead * math.ceil(shape[-2] / 32) * 32, ttc * 32]
    ref = ttnn._ttnn.bfp_utils.compute_shard_page_mapping(
        padded, list(shard_shape), 32, 32, crs, ttnn.ShardOrientation.ROW_MAJOR, strat
    )
    refmap = {(c.x, c.y): list(pages) for c, pages in ref if len(pages)}
    mine = {}
    for o in owner_shards(p, g):
        mine[o.core] = [(o.r0 + r) * ttc + o.c0 + c for r in range(o.core_rows) for c in range(o.core_cols)]
    print(
        name,
        "OK" if mine == refmap else f"MISMATCH mine={list(mine.items())[:3]} ref={list(refmap.items())[:3]}",
        "shards",
        num_shards(p, g),
    )


check(
    "2058",
    [1, 16, 1, 256],
    "height_sharded",
    [
        [0, 2, 0, 2],
        [1, 2, 1, 2],
        [2, 2, 2, 2],
        [3, 2, 3, 2],
        [4, 2, 4, 2],
        [5, 1, 5, 1],
        [5, 2, 5, 2],
        [6, 1, 6, 1],
        [6, 2, 6, 2],
        [7, 1, 7, 1],
        [7, 2, 7, 2],
        [8, 1, 8, 1],
        [8, 2, 8, 2],
        [9, 1, 9, 1],
        [9, 2, 9, 2],
        [10, 1, 10, 1],
    ],
    [32, 256],
)
check("2065", [1, 1, 16, 8704], "width_sharded", [[0, 0, 10, 2], [0, 3, 2, 3]], [32, 256])
check("2071", [1, 1, 16, 12928], "width_sharded", [[0, 0, 7, 0]], [32, 1664])
check("2062", [1, 16, 6, 256], "height_sharded", [[0, 0, 10, 0], [0, 1, 4, 1]], [32, 256])
check("block-2158", [1, 1, 6, 256], "block_sharded", [[0, 0, 0, 0]], [32, 256])
check("block-uneven", [1, 1, 224, 256], "block_sharded", [[1, 1, 4, 3]], [96, 64])
check("height-multi", [2, 3, 70, 128], "height_sharded", [[0, 0, 3, 3]], [96, 128])
