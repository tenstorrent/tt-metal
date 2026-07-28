import ttnn
from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd


def test_plan(device):
    grid = device.compute_with_storage_grid_size()
    print(f"grid {grid.x}x{grid.y}")
    print(
        f"{'case':34s}{'cw':>4s}{'cw1':>4s}{'cw2':>4s}{'htb':>5s}{'nw':>4s}{'colpk':>6s}{'bf16':>6s}{'g_ht':>6s}{'g_B':>6s}"
    )
    cases = [
        # (shape, kind, shard, grid, fp32_dest)
        ((1, 1, 8192, 1024), "BLOCK", [1024, 128], (8, 8), False),
        ((1, 1, 8192, 1024), "BLOCK", [1024, 128], (8, 8), True),
        ((1, 1, 32, 1024), "WIDTH", [32, 128], (8, 1), False),
        ((1, 1, 32, 7168), "WIDTH", [32, 256], (7, 4), False),
        ((1, 1, 32, 5120), None, None, None, False),
        ((1, 1, 32, 7168), None, None, None, False),
        ((1, 1, 8192, 1024), None, None, None, False),
        ((1, 1, 8192, 7168), None, None, None, False),
    ]
    from eval.sharding import shard_config

    for shape, kind, shard, cg, fp32d in cases:
        import torch

        mc = None
        if kind is not None:
            ml = getattr(ttnn.TensorMemoryLayout, f"{kind}_SHARDED")
            mc = shard_config(list(shard), cg, ml, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
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
        ht_total, wt_global = pd._tile_geometry(x)
        in_sh = mc is not None
        p = pd._select_placement(device, grid, x, ht_total, wt_global, in_sh)
        blk = pd._derive_blocking(
            x,
            g,
            grid.x * grid.y,
            p,
            sharded_in=in_sh,
            sharded_out=in_sh,
            l1_total_budget=pd._l1_total_budget(device),
            fp32_dest_acc_en=fp32d,
        )
        name = f"{'x'.join(map(str,shape))}-{kind or 'interleaved'}-fp32d{int(fp32d)}"
        print(
            f"{name:34s}{p.cw:4d}{p.cw1:4d}{p.cw2:4d}{blk.ht_block:5d}{blk.nw:4d}"
            f"{int(blk.colpack):6d}{int(blk.partial_bf16):6d}{blk.gather_ht:6d}{blk.gather_tile_bytes:6d}"
        )
        ttnn.deallocate(x)
        ttnn.deallocate(g)
