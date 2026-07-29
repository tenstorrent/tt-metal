# Refinement 3 first light: one-sided alias bit-exactness + plan structure.
import torch, ttnn
from ttnn.operations.tilize import tilize
from ttnn.operations.tilize import tilize_program_descriptor as tpd

_L1 = ttnn.BufferType.L1
ROW = ttnn.ShardOrientation.ROW_MAJOR
COL = ttnn.ShardOrientation.COL_MAJOR
H = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
W = ttnn.TensorMemoryLayout.WIDTH_SHARDED
B = ttnn.TensorMemoryLayout.BLOCK_SHARDED
DRAM = ttnn.DRAM_MEMORY_CONFIG


def crs(ex, ey):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(ex, ey))})


def sh(scheme, grid, shape, orient=ROW):
    return ttnn.MemoryConfig(scheme, _L1, ttnn.ShardSpec(grid, shape, orient))


device = ttnn.open_device(device_id=0)
try:
    cases = [
        ("g_out_HEIGHT_small", (1, 1, 128, 64), DRAM, sh(H, crs(3, 0), (32, 64))),
        ("g_in_HEIGHT_small", (1, 1, 128, 64), sh(H, crs(3, 0), (32, 64)), DRAM),
        ("g_out_BLOCK_8x8", (1, 1, 2048, 512), DRAM, sh(B, crs(7, 7), (256, 64))),
        ("g_in_BLOCK_8x8", (1, 1, 2048, 512), sh(B, crs(7, 7), (256, 64)), DRAM),
        ("g_out_WIDTH", (1, 1, 64, 512), DRAM, sh(W, crs(3, 0), (64, 128))),
        ("g_in_WIDTH", (1, 1, 64, 512), sh(W, crs(3, 0), (64, 128)), DRAM),
        ("g_out_BLOCK_col", (1, 1, 128, 128), DRAM, sh(B, crs(1, 1), (64, 64), COL)),
        ("g_in_BLOCK_col", (1, 1, 128, 128), sh(B, crs(1, 1), (64, 64), COL), DRAM),
        ("g_out_wideH_chunked", (1, 1, 128, 2048), DRAM, sh(H, crs(3, 0), (32, 2048))),
        ("g_in_wideH", (1, 1, 128, 2048), sh(H, crs(3, 0), (32, 2048)), DRAM),
        ("reshard_cross_spec", (1, 1, 128, 64), sh(H, crs(3, 0), (32, 64)), sh(H, crs(1, 0), (64, 64))),
        ("same_spec_pathB", (1, 1, 512, 64), sh(H, crs(3, 0), (128, 64)), sh(H, crs(3, 0), (128, 64))),
    ]
    for name, shape, in_cfg, out_cfg in cases:
        n = 1
        for d in shape:
            n *= d
        t = (torch.arange(n, dtype=torch.float32).reshape(shape) % 4096).bfloat16()
        tt_in = ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_cfg
        )
        probe_out = ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, out_cfg
        )
        plan = tpd.build_plan(tt_in, probe_out, device, use_multicore=True, use_double_buffer=None)
        out = tilize(tt_in, out_cfg)
        got = ttnn.to_torch(out)
        ok = torch.equal(got.float(), t.float())
        print(
            f"{name:22s} path={plan['path']:10s} cores={plan['ncores']:3d} chk={plan['chunk_wt']:3d} "
            f"d={plan['depth']} blk={plan['blocks_per_core']:2d} chunks={plan['chunks_per_core']} "
            f"coal={plan['coalesce_rows']} c7={plan['split_read']} b13={plan['stateful_read']} "
            f"b8={plan['prefetch_blocks']} cbB={plan['cb_bytes_per_core']:6d} aliasB={plan['alias_cb_bytes']:6d} "
            f"EXACT={ok}"
        )
        if not ok:
            bad = got.float() != t.float()
            print("   mismatch count:", int(bad.sum()), "of", n)
finally:
    ttnn.close_device(device)
