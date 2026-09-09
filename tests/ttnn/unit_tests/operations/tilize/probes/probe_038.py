"""Block-float OUTPUT x tiny tile: which (tile_h, out_dtype) survive?"""
import importlib, torch, ttnn

M = importlib.import_module("ttnn.operations.tilize.tilize")
import ttnn.operations.tilize.tilize_program_descriptor as pd

device = ttnn.open_device(device_id=0)
device.disable_and_clear_program_cache()
grid = device.compute_with_storage_grid_size()


def pcc(a, b):
    a = a.flatten().to(torch.float64)
    b = b.flatten().to(torch.float64)
    va, vb = a - a.mean(), b - b.mean()
    d = va.norm() * vb.norm()
    return float((va * vb).sum() / d) if d > 0 else 1.0


torch.manual_seed(0)
shape = (1, 1, 64, 128)
x = torch.randn(shape).bfloat16()
for out in (ttnn.bfloat8_b, ttnn.bfloat4_b):
    for th in (32, 16, 8, 4, 2, 1):
        tt = ttnn.from_torch(
            x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        o = M.tilize(tt, dtype=out, tile=ttnn.Tile([th, 32]))
        p = pd.derive_plan(tt, o, low_l1=False, grid=grid)
        got = ttnn.to_torch(o).float()
        print(
            f"{str(out)[9:]:10s} tile_h={th:2d} out_page={o.buffer_page_size():5d} "
            f"bw={p.block_width_tiles} pcc={pcc(got, x.float()):.6f} "
            f"max_abs={float((got - x.float()).abs().max()):.5g}",
            flush=True,
        )
# does the same tiny-tile bf8b geometry work at other shapes / widths?
for shape2 in ((1, 1, 32, 32), (1, 1, 32, 64), (1, 1, 128, 256)):
    x2 = torch.randn(shape2).bfloat16()
    for th in (32, 16, 8):
        tt = ttnn.from_torch(
            x2, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        o = M.tilize(tt, dtype=ttnn.bfloat8_b, tile=ttnn.Tile([th, 32]))
        got = ttnn.to_torch(o).float()
        print(f"  bf8b {shape2} tile_h={th}: pcc={pcc(got, x2.float()):.6f}", flush=True)
ttnn.close_device(device)
