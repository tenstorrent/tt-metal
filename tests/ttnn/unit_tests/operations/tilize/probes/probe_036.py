"""Refinement 5 pass 2: fp32->fp32 lossless + int32<->uint32 readback dtype."""
import importlib, torch, ttnn

M = importlib.import_module("ttnn.operations.tilize.tilize")
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc

device = ttnn.open_device(device_id=0)
device.disable_and_clear_program_cache()


def run(x, i, o, shape, tile_h=32):
    tt = ttnn.from_torch(x, dtype=i, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    kw = {} if tile_h == 32 else {"tile": ttnn.Tile([tile_h, 32])}
    return ttnn.to_torch(M.tilize(tt, dtype=o, **kw))


torch.manual_seed(0)
for shape in [(1, 1, 64, 128), (1, 1, 32, 32), (2, 3, 128, 256), (1, 1, 1024, 1024)]:
    x = torch.randn(shape, dtype=torch.float32)
    got = run(x, ttnn.float32, ttnn.float32, shape)
    print(f"fp32->fp32 {shape}: bit_exact={torch.equal(got, x)} max_abs={float((got-x).abs().max()):.6g}", flush=True)

# tiny tile fp32 lossless
x = torch.randn((1, 1, 64, 128), dtype=torch.float32)
for th in (16, 8, 1):
    got = run(x, ttnn.float32, ttnn.float32, (1, 1, 64, 128), th)
    print(f"fp32->fp32 tile_h={th}: bit_exact={torch.equal(got, x)}", flush=True)

# int32 <-> uint32 readback semantics (the golden oracle's exact comparison)
xi = torch.randint(-1000, 1000, (1, 1, 64, 128), dtype=torch.int32)
for i, o in ((ttnn.int32, ttnn.uint32), (ttnn.uint32, ttnn.int32), (ttnn.int32, ttnn.int32)):
    src = xi if i == ttnn.int32 else torch.randint(0, 100, (1, 1, 64, 128), dtype=torch.int32)
    got = run(src, i, o, (1, 1, 64, 128))
    exp = src.to(torch.int32)
    print(f"{str(i)[9:]}->{str(o)[9:]}: readback_dtype={got.dtype} comp_equal={comp_equal(exp, got)}", flush=True)

# uint8 across shapes / tile heights
for shape in [(1, 1, 64, 128), (1, 1, 32, 32), (2, 3, 128, 256), (1, 1, 512, 512)]:
    xu = torch.randint(0, 256, shape, dtype=torch.uint8)
    got = run(xu, ttnn.uint8, ttnn.uint8, shape)
    print(f"uint8 {shape}: dtype={got.dtype} exact={torch.equal(got, xu)}", flush=True)
xu = torch.randint(0, 256, (1, 1, 64, 128), dtype=torch.uint8)
for th in (16, 4, 1):
    got = run(xu, ttnn.uint8, ttnn.uint8, (1, 1, 64, 128), th)
    print(f"uint8 tile_h={th}: exact={torch.equal(got, xu)}", flush=True)
ttnn.close_device(device)
