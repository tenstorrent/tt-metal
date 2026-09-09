import torch, ttnn
from ttnn.operations.tilize import tilize
import ttnn.operations.tilize.tilize_program_descriptor as pd

device = ttnn.open_device(device_id=0)


def run(shape, hi, ho, note=""):
    n = 1
    for d in shape:
        n *= d
    t = (torch.arange(n) % 4093).reshape(shape).to(torch.bfloat16)
    tt = ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        tile=ttnn.Tile([hi, 32]),
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    try:
        out = tilize(tt, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16, tile=ttnn.Tile([ho, 32]))
        rb = ttnn.to_torch(out)
        ok = torch.equal(rb.float(), t.float())
        plan = pd.derive_plan(tt, out, low_l1=False, grid=device.compute_with_storage_grid_size(), pad_value=None)
        print(
            f"RETILE {note} {shape} {hi}->{ho} unit={pd.retile_copy_unit(hi, ho)} "
            f"bw={plan.block_width_tiles} wch={plan.num_w_chunks} rg={plan.num_row_groups} "
            f"tile={out.tile.tile_shape} ok={ok}",
            flush=True,
        )
        if not ok:
            d = (rb.float() - t.float()).abs()
            print("   maxdiff", d.max().item(), "nbad", int((d > 0).sum()), flush=True)
    except Exception as e:
        print(f"RETILE {note} {shape} {hi}->{ho} EXC {type(e).__name__}: {e}", flush=True)


# the six golden retile cases
run([1, 1, 32, 64], 32, 16, "g1")
run([1, 1, 32, 64], 16, 32, "g2")
run([1, 1, 64, 128], 8, 4, "g3")
run([1, 1, 32, 64], 4, 2, "g4")
run([1, 1, 32, 64], 2, 1, "g5")
run([1, 1, 32, 64], 1, 32, "g6")
