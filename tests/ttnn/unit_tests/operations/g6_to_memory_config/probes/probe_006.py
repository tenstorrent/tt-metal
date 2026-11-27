import torch, ttnn
import ttnn.reports

device = ttnn.open_device(device_id=0)
info = ttnn.reports.get_device_info(device)
print("l1_num_banks", info.l1_num_banks, "num_compute_cores", info.num_compute_cores)
g = device.compute_with_storage_grid_size()
print("grid", g.x, g.y, "dram grid", device.dram_grid_size())
print("dram align", ttnn.get_dram_alignment(), "l1 align", ttnn.get_l1_alignment())


def mc(scheme, buf, ranges, shape):
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(a, b), ttnn.CoreCoord(c, d)) for a, b, c, d in ranges])
    return ttnn.MemoryConfig(scheme, buf, ttnn.ShardSpec(grid, tuple(shape), ttnn.ShardOrientation.ROW_MAJOR))


H = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
W = ttnn.TensorMemoryLayout.WIDTH_SHARDED
B = ttnn.TensorMemoryLayout.BLOCK_SHARDED
L1 = ttnn.BufferType.L1
cfg = mc(
    H,
    L1,
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
print("ranges order", [(r.start.x, r.start.y) for r in cfg.shard_spec.grid.ranges()])
t = ttnn.from_torch(
    torch.randn([1, 16, 1, 256]).bfloat16(),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=cfg,
)
print("owners", [(c.x, c.y) for c in ttnn.get_optimal_worker_cores_for_sharded_tensor(t)])
print(
    "page",
    t.buffer_page_size(),
    t.buffer_aligned_page_size(),
    "npages",
    t.buffer_num_pages(),
    "padded",
    t.padded_shape,
    "addr",
    t.buffer_address(),
)
ct = ttnn.TensorAccessorArgs(t).get_compile_time_args()
print("ct len", len(ct), list(ct)[:12])
cfg2 = mc(W, L1, [[0, 0, 10, 2], [0, 3, 2, 3]], [32, 256])
t2 = ttnn.from_torch(
    torch.randn([1, 1, 16, 8704]).bfloat16(),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=cfg2,
)
print(
    "owners2",
    len(ttnn.get_optimal_worker_cores_for_sharded_tensor(t2)),
    [(c.x, c.y) for c in ttnn.get_optimal_worker_cores_for_sharded_tensor(t2)][:14],
)
print("npages2", t2.buffer_num_pages(), "ct len", len(ttnn.TensorAccessorArgs(t2).get_compile_time_args()))
t3 = ttnn.from_torch(
    torch.randn([16, 128, 5120]).bfloat16(),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
print(
    "interleaved ct",
    list(ttnn.TensorAccessorArgs(t3).get_compile_time_args()),
    t3.memory_config().buffer_type == ttnn.BufferType.DRAM,
    t3.is_sharded(),
)
r = ttnn.split_work_to_cores(g, 1000, row_wise=True)
print("split", r[0], r[4], r[5], r[1])
print(dir(t3.memory_config()))
print(type(t3.memory_config().shard_spec))
ttnn.close_device(device)
