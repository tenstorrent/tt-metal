import torch, ttnn
from ttnn.operations.tilize import tilize
import ttnn.operations.tilize.tilize_program_descriptor as pd

device = ttnn.open_device(device_id=0)


def run(shape, tile_h, *, pad=None, low_l1=False, out_mc=None, in_mc=None, note=""):
    n = 1
    for d in shape:
        n *= d
    t = (torch.arange(n) % 977).reshape(shape).to(torch.bfloat16)
    imc = in_mc or ttnn.DRAM_MEMORY_CONFIG
    omc = out_mc or ttnn.DRAM_MEMORY_CONFIG
    tt = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=imc)
    kw = {}
    if pad is not None:
        kw["pad_value"] = pad
    try:
        out = tilize(tt, memory_config=omc, dtype=ttnn.bfloat16, tile=ttnn.Tile([tile_h, 32]), low_l1=low_l1, **kw)
        rb = ttnn.to_torch(out)
        ok = torch.equal(rb.float(), t.float())
        plan = pd.derive_plan(tt, out, low_l1=low_l1, grid=device.compute_with_storage_grid_size(), pad_value=pad)
        print(
            f"CASE {note} shape={shape} tile_h={tile_h} pad={pad} low_l1={low_l1} "
            f"bw={plan.block_width_tiles} wch={plan.num_w_chunks} rg={plan.num_row_groups} "
            f"blocks={plan.num_blocks_total} l1={plan.l1_per_core_bytes} ok={ok}",
            flush=True,
        )
        if not ok:
            print("   maxdiff", (rb.float() - t.float()).abs().max().item(), flush=True)
    except Exception as e:
        print(f"CASE {note} shape={shape} tile_h={tile_h} pad={pad} EXC {type(e).__name__}: {e}", flush=True)


# multi-core / wide, at every tiny height
for th in (16, 8, 4, 2, 1):
    run([1, 1, 2048, 64], th, note="tall_narrow")
    run([1, 1, 32, 2048], th, note="short_wide")
run([1, 1, 512, 512], 1, note="tile1_grid")
run([1, 1, 32, 8192], 16, low_l1=True, note="low_l1_wide")

# alignment re-partition: H=48 with tile_h=16 is THREE whole tile-rows, no H tail
run([1, 1, 48, 64], 16, note="H48_tile16_aligned")
run([1, 1, 48, 64], 32, pad=0.0, note="H48_tile32_padded")
# H tail against a tiny tile
run([1, 1, 40, 64], 16, pad=0.0, note="H40_tile16_htail")
run([1, 1, 33, 50], 16, pad=-3.0, note="hw_tail_tile16")
run([1, 1, 3, 50], 4, pad=1.5, note="hw_tail_tile4")
run([1, 1, 5, 32], 2, pad=0.0, note="htail_tile2")
run([1, 1, 7, 40], 1, pad=0.0, note="wtail_tile1")

# rank fold
run([3, 2, 64, 64], 8, note="rank4_fold")
run([8, 1, 249, 256], 16, pad=0.0, note="fold_htail_tile16")

# L1 / sharded
run([1, 1, 64, 128], 8, in_mc=ttnn.L1_MEMORY_CONFIG, out_mc=ttnn.L1_MEMORY_CONFIG, note="l1_to_l1")
grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
run(
    [1, 1, 64, 64],
    16,
    out_mc=ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [32, 64], ttnn.ShardOrientation.ROW_MAJOR),
    ),
    note="height_sharded_out_tile16",
)
