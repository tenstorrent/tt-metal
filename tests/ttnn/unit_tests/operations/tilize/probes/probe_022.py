import torch, ttnn
from ttnn.operations.tilize import tilize

for tile_h in (16, 8, 4, 2, 1):
    for shape in ([1, 1, 32, 64], [1, 1, 64, 128]):
        t = torch.arange(torch.tensor(shape).prod().item()).reshape(shape).to(torch.bfloat16)
        tt = ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        out = tilize(tt, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16, tile=ttnn.Tile([tile_h, 32]))
        rb = ttnn.to_torch(out)
        ok = torch.equal(rb.float(), t.float())
        print(f"tile_h={tile_h} shape={shape} layout={out.layout} tile={out.tile.tile_shape} ok={ok}", flush=True)
        if not ok:
            print("  maxdiff", (rb.float() - t.float()).abs().max().item())
