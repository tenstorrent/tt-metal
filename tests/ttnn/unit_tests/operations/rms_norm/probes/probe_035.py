import torch, ttnn
from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd
from eval.sharding import shard_config

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
print(f"grid {grid.x}x{grid.y} = {grid.x*grid.y}")
print("L1 bank:", ttnn.get_memory_view(device, ttnn.BufferType.L1).total_bytes_per_bank)

hdr = f"{'case':44s} {'Wt':>4} {'C':>4} {'NW':>4} {'H':>4} {'rowsmax':>8} {'CW':>4} {'CW1':>4} {'CW2':>4} {'cores':>6} {'xdep':>5} {'gres':>5} {'rdbat':>6} {'progKB':>7} {'shardKB':>8}"
print(hdr)


def report(label, tt_x, tt_g, in_sharded):
    ht_total, wt_global = pd._tile_geometry(tt_x)
    p = pd._select_placement(device, grid, tt_x, ht_total, wt_global, in_sharded)
    blk = pd._derive_blocking(
        tt_x,
        tt_g,
        grid.x * grid.y,
        p,
        sharded_in=in_sharded,
        sharded_out=in_sharded,
        l1_total_budget=pd._l1_total_budget(device),
    )
    print(
        f"{label:44s} {blk.Wt:4d} {blk.wt_chunk:4d} {blk.nw:4d} {blk.ht_block:4d} {blk._rows_core_max:8d} "
        f"{p.cw:4d} {p.cw1:4d} {p.cw2:4d} {p.num_cores:6d} {blk.x_res_depth:5d} {int(blk.gamma_resident):5d} "
        f"{pd._x_read_chunks(blk):6d} {blk.program_cb_bytes//1024:7d} {blk.resident_shard_bytes//1024:8d}"
    )


# interleaved prefill + decode
for shape in [(1, 1, 8192, 1024), (1, 1, 8192, 2304), (1, 1, 8192, 5120), (1, 1, 8192, 7168), (1, 1, 32, 7168)]:
    x = ttnn.from_torch(
        torch.zeros(shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    g = ttnn.from_torch(
        torch.zeros(1, 1, 1, shape[-1], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    report(f"IL {shape}", x, g, False)
    ttnn.deallocate(x)
    ttnn.deallocate(g)

SHARDED = [
    ((1, 1, 32, 1024), "WIDTH", (32, 128), (8, 1)),
    ((1, 1, 32, 2304), "WIDTH", (32, 256), (9, 1)),
    ((1, 1, 32, 5120), "WIDTH", (32, 160), (8, 4)),
    ((1, 1, 32, 7168), "WIDTH", (32, 256), (7, 4)),
    ((1, 1, 8192, 1024), "BLOCK", (1024, 128), (8, 8)),
]
for shape, kind, ss, cg in SHARDED:
    ml = getattr(ttnn.TensorMemoryLayout, f"{kind}_SHARDED")
    mc = shard_config(list(ss), cg, ml, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
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
    report(f"{kind} {shape} {ss}{cg}", x, g, True)
    ttnn.deallocate(x)
    ttnn.deallocate(g)

ttnn.close_device(device)
