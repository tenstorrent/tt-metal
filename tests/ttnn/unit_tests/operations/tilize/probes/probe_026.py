import torch, ttnn
from ttnn.operations.tilize import tilize
import ttnn.operations.tilize.tilize_program_descriptor as pd

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
HEIGHTS = (1, 2, 4, 8, 16, 32)
fails = []


def run(shape, hi, ho, note="", imc=None, omc=None, quiet=False):
    n = 1
    for d in shape:
        n *= d
    t = (torch.arange(n) % 4093).reshape(shape).to(torch.bfloat16)
    imc = imc or ttnn.DRAM_MEMORY_CONFIG
    omc = omc or ttnn.DRAM_MEMORY_CONFIG
    tt = ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, tile=ttnn.Tile([hi, 32]), device=device, memory_config=imc
    )
    tag = f"{note} {shape} {hi}->{ho}"
    try:
        out = tilize(tt, memory_config=omc, dtype=ttnn.bfloat16, tile=ttnn.Tile([ho, 32]))
        rb = ttnn.to_torch(out)
        ok = torch.equal(rb.float(), t.float())
        p = pd.derive_plan(tt, out, low_l1=False, grid=grid, pad_value=None)
        info = f"unit={pd.retile_copy_unit(hi,ho)} bw={p.block_width_tiles} wch={p.num_w_chunks} rg={p.num_row_groups} blocks={p.num_blocks_total}"
        if not ok:
            fails.append((tag, "MISMATCH"))
            print(f"FAIL {tag} {info}", flush=True)
        elif not quiet:
            print(f"ok   {tag} {info}", flush=True)
    except Exception as e:
        fails.append((tag, f"{type(e).__name__}: {e}"))
        print(f"EXC  {tag} {type(e).__name__}: {e}", flush=True)


print("--- all 36 (in,out) height pairs on [1,1,32,64] and [2,1,64,96] ---", flush=True)
for hi in HEIGHTS:
    for ho in HEIGHTS:
        run([1, 1, 32, 64], hi, ho, "pair", quiet=True)
        run([2, 1, 64, 96], hi, ho, "fold", quiet=True)
print("  done", flush=True)

print("--- multi-core / wide, block_width > 1, several blocks per core ---", flush=True)
run([1, 1, 2048, 64], 32, 16, "tall")
run([1, 1, 32, 4096], 32, 16, "wide")
run([1, 1, 512, 512], 16, 32, "square")
run([1, 1, 512, 512], 8, 1, "square_tiny")
run([1, 1, 2048, 64], 1, 32, "tall_1to32")

print("--- H not a multiple of in_tile_h: per-image split ---", flush=True)
run([1, 1, 20, 64], 8, 4, "h20")
run([2, 3, 20, 64], 8, 4, "h20_multi_image")
run([3, 1, 12, 96], 8, 2, "h12_multi_image")
run([1, 1, 40, 64], 16, 8, "h40")

print("--- buffer transitions ---", flush=True)
run([1, 1, 64, 128], 32, 16, "dram_to_l1", omc=ttnn.L1_MEMORY_CONFIG)
run([1, 1, 64, 128], 32, 16, "l1_to_l1", imc=ttnn.L1_MEMORY_CONFIG, omc=ttnn.L1_MEMORY_CONFIG)
run([1, 1, 64, 128], 8, 4, "l1_to_dram", imc=ttnn.L1_MEMORY_CONFIG)

print("--- ranks ---", flush=True)
run([32, 64], 32, 8, "rank2")
run([2, 2, 2, 32, 64], 16, 4, "rank5")

print(f"=== FAILS: {len(fails)}", flush=True)
for f in fails:
    print("   ", f, flush=True)
