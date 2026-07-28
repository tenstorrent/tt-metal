import torch, ttnn
from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd
from eval.sharding import auto_shard_config, shard_config

device = ttnn.open_device(device_id=0)
ML = ttnn.TensorMemoryLayout
grid = device.compute_with_storage_grid_size()

# Every sharded geometry the suites exercise. Does the two-budget model change
# the blocking (old = shard charged against the CB budget) for any of them?
PINNED = [
    ((1, 1, 32, 1024), ML.WIDTH_SHARDED, [32, 128], (8, 1)),
    ((1, 1, 32, 2304), ML.WIDTH_SHARDED, [32, 256], (9, 1)),
    ((1, 1, 32, 5120), ML.WIDTH_SHARDED, [32, 160], (8, 4)),
    ((1, 1, 32, 7168), ML.WIDTH_SHARDED, [32, 256], (7, 4)),
    ((1, 1, 8192, 1024), ML.BLOCK_SHARDED, [1024, 128], (8, 8)),
    ((1, 1, 256, 512), ML.BLOCK_SHARDED, [32, 64], (8, 8)),
]
AUTO = [
    ((1, 1, 32, 2048), ML.WIDTH_SHARDED),
    ((1, 1, 32, 8192), ML.WIDTH_SHARDED),
    ((1, 1, 64, 17), ML.WIDTH_SHARDED),
    ((4, 8, 47, 256), ML.BLOCK_SHARDED),
    ((1, 1, 32, 50), ML.BLOCK_SHARDED),
    ((1, 1, 256, 512), ML.HEIGHT_SHARDED),
    ((1, 1, 8192, 1024), ML.HEIGHT_SHARDED),
    ((4, 8, 32, 256), ML.HEIGHT_SHARDED),
]


def report(tag, shape, mc):
    x = ttnn.from_torch(
        torch.zeros(shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mc,
    )
    g = ttnn.from_torch(
        torch.zeros(1, 1, 1, shape[-1], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    ht, wt = pd._tile_geometry(x)
    p = pd._select_placement(device, grid, x, ht, wt, True)
    out = []
    for name, budget in (("old", pd.L1_CB_BUDGET_BYTES), ("new", pd._l1_total_budget(device))):
        b = pd._derive_blocking(x, g, grid.x * grid.y, p, sharded_in=True, sharded_out=True, l1_total_budget=budget)
        out.append(
            (b.wt_chunk, b.nw, b.ht_block, b.x_res_depth, b.gamma_resident, b.program_cb_bytes, b.resident_shard_bytes)
        )
    flag = "  <-- CHANGED" if out[0][:5] != out[1][:5] else ""
    print(
        f"{tag:6s} {str(shape):18s} cores={p.num_cores:3d} old(chunk,nw,htb)={out[0][:3]} new={out[1][:3]} "
        f"prog={out[1][5]} shard={out[1][6]}{flag}",
        flush=True,
    )
    ttnn.deallocate(x)
    ttnn.deallocate(g)


for shape, ml, ss, cg in PINNED:
    report("pinned", shape, shard_config(ss, cg, ml, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device))
for shape, ml in AUTO:
    report(
        "auto", shape, auto_shard_config(list(shape), ml, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    )
ttnn.close_device(device)
