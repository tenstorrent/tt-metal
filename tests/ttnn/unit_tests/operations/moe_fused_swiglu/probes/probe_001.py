import torch, ttnn

TILE = 32
GRID_X = 11  # the routed-expert op's N-parallel grid width (fixed in its PF)
EMB, HIDDEN, CAPACITY = 7168, 2048, 5120
NUM_GLOBAL, NUM_LOCAL, LOCAL_ID, GLOBAL_ID = 256, 8, 3, 137
COUNT = 512

device = ttnn.open_device(device_id=0)
try:

    def nd_mc(n_dim):
        n_t = n_dim // TILE
        per_core_n = (n_t + GRID_X - 1) // GRID_X
        d = device.dram_grid_size()
        return ttnn.MemoryConfig(
            buffer_type=ttnn.BufferType.DRAM,
            nd_shard_spec=ttnn.NdShardSpec(
                shard_shape=ttnn.Shape([TILE, per_core_n * TILE]),
                grid=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(d.x - 1, d.y - 1))]),
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

    print("gate/up nd shard:", nd_mc(HIDDEN), flush=True)
    print("down    nd shard:", nd_mc(EMB), flush=True)

    torch.manual_seed(42)
    x = ttnn.from_torch(
        torch.randn((1, 1, CAPACITY, EMB), dtype=torch.float32).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    counts = torch.zeros(NUM_GLOBAL, dtype=torch.int32)
    counts[GLOBAL_ID] = COUNT
    tt_counts = ttnn.from_torch(counts, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    idx = torch.tensor([(11 + 37 * i) % NUM_GLOBAL for i in range(NUM_LOCAL)], dtype=torch.int32)
    idx[LOCAL_ID] = GLOBAL_ID
    tt_idx = ttnn.from_torch(idx, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    for place in ("interleaved", "nd_shard"):
        gu_mc = ttnn.DRAM_MEMORY_CONFIG if place == "interleaved" else nd_mc(HIDDEN)
        d_mc = ttnn.DRAM_MEMORY_CONFIG if place == "interleaved" else nd_mc(EMB)
        w = [
            ttnn.from_torch(
                torch.randn(s, dtype=torch.bfloat16),
                dtype=ttnn.bfloat4_b,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=mc,
            )
            for s, mc in (((EMB, HIDDEN), gu_mc), ((EMB, HIDDEN), gu_mc), ((HIDDEN, EMB), d_mc))
        ]
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, CAPACITY, EMB]), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        )
        r = ttnn.experimental.deepseek_prefill.unified_routed_expert_ffn(
            x,
            w[0],
            w[1],
            w[2],
            tt_counts,
            tt_idx,
            LOCAL_ID,
            output=out,
            x_is_row_major=True,
        )
        print(f"OK {place}: shape={list(r.shape)} dtype={r.dtype} layout={r.layout}", flush=True)
        for t in (*w, out):
            ttnn.deallocate(t)
finally:
    ttnn.close_device(device)
