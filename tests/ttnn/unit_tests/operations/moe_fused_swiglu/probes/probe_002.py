import torch, ttnn
import torch.nn.functional as F

TILE, GRID_X = 32, 11
EMB, HIDDEN, CAPACITY, COUNT = 7168, 2048, 512, 128
NUM_GLOBAL, NUM_LOCAL, LOCAL_ID, GLOBAL_ID = 256, 8, 3, 137


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


device = ttnn.open_device(device_id=0)
try:

    def nd_mc(n_dim):
        per_core_n = (n_dim // TILE + GRID_X - 1) // GRID_X
        d = device.dram_grid_size()
        return ttnn.MemoryConfig(
            buffer_type=ttnn.BufferType.DRAM,
            nd_shard_spec=ttnn.NdShardSpec(
                shard_shape=ttnn.Shape([TILE, per_core_n * TILE]),
                grid=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(d.x - 1, d.y - 1))]),
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

    torch.manual_seed(0)
    # gate/up are (emb, hidden); down is (hidden, emb) — the op's K-major orientation.
    wg = torch.randn(EMB, HIDDEN, dtype=torch.bfloat16) * 0.02
    wu = torch.randn(EMB, HIDDEN, dtype=torch.bfloat16) * 0.02
    wd = torch.randn(HIDDEN, EMB, dtype=torch.bfloat16) * 0.02
    xt = torch.zeros(CAPACITY, EMB, dtype=torch.bfloat16)
    xt[:COUNT] = torch.randn(COUNT, EMB, dtype=torch.bfloat16)

    ref = (F.silu(xt[:COUNT].float() @ wg.float()) * (xt[:COUNT].float() @ wu.float())) @ wd.float()

    x = ttnn.from_torch(
        xt.reshape(1, 1, CAPACITY, EMB),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    c = torch.zeros(NUM_GLOBAL, dtype=torch.int32)
    c[GLOBAL_ID] = COUNT
    tt_counts = ttnn.from_torch(c, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ix = torch.tensor([(11 + 37 * i) % NUM_GLOBAL for i in range(NUM_LOCAL)], dtype=torch.int32)
    ix[LOCAL_ID] = GLOBAL_ID
    tt_idx = ttnn.from_torch(ix, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    for place in ("interleaved", "nd_shard"):
        gu = ttnn.DRAM_MEMORY_CONFIG if place == "interleaved" else nd_mc(HIDDEN)
        dm = ttnn.DRAM_MEMORY_CONFIG if place == "interleaved" else nd_mc(EMB)
        w = [
            ttnn.from_torch(t, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
            for t, mc in ((wg, gu), (wu, gu), (wd, dm))
        ]
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, CAPACITY, EMB]), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        )
        r = ttnn.experimental.deepseek_prefill.unified_routed_expert_ffn(
            x, w[0], w[1], w[2], tt_counts, tt_idx, LOCAL_ID, output=out, x_is_row_major=True
        )
        got = ttnn.to_torch(r)[0, 0, :COUNT]
        print(f"{place:12s} PCC={pcc(ref, got):.6f}  nan={torch.isnan(got).any().item()}", flush=True)
        for t in (*w, out):
            ttnn.deallocate(t)
finally:
    ttnn.close_device(device)
