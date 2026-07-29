import torch, ttnn

dev = ttnn.open_device(device_id=0)
print("dram_grid_size:", dev.dram_grid_size(), " compute grid:", dev.compute_with_storage_grid_size())
for n in [4, 8, 12]:
    try:
        g = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(n - 1, 0))})
        mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(g, (32, 64), ttnn.ShardOrientation.ROW_MAJOR),
        )
        t = ttnn.from_torch(
            torch.randn(32, 64 * n).bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=dev,
            memory_config=mc,
        )
        args = ttnn.TensorAccessorArgs(t).get_compile_time_args()
        print(
            f"  n={n}: DRAM-sharded TensorAccessorArgs OK, {len(args)} CT args; page={t.buffer_page_size()} npages={t.buffer_num_pages()}"
        )
    except Exception as e:
        print(f"  n={n}: FAIL {type(e).__name__}: {str(e)[:110]}")
ttnn.close_device(dev)
