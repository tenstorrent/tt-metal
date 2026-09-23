import torch, ttnn
from ttnn.operations.tilize import tilize
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

dev = ttnn.open_device(device_id=0)
torch.manual_seed(0)
# 1) tiny tile bfp
x = torch.randn(1, 1, 32, 64).bfloat16()
for bf in [ttnn.bfloat8_b, ttnn.bfloat4_b]:
    for th in [32, 16]:
        try:
            h = ttnn.from_torch(x, dtype=bf, layout=ttnn.TILE_LAYOUT, tile=ttnn.Tile([th, 32]))
            hr = ttnn.to_torch(h)
            print("RES host roundtrip", bf, th, comp_pcc(x, hr, 0.99)[1][:60])
        except Exception as e:
            print("RES host roundtrip EXC", bf, th, str(e)[:200])
        try:
            h = ttnn.from_torch(x, dtype=bf, layout=ttnn.TILE_LAYOUT, tile=ttnn.Tile([th, 32]), device=dev)
            hr = ttnn.to_torch(h)
            print("RES host->dev roundtrip", bf, th, comp_pcc(x, hr, 0.99)[1][:60])
        except Exception as e:
            print("RES host->dev EXC", bf, th, str(e)[:200])
        t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        y = tilize(t, dtype=bf, tile=ttnn.Tile([th, 32]) if th != 32 else None)
        print("RES device", bf, th, y.tile.tile_shape, comp_pcc(x, ttnn.to_torch(y), 0.99)[1][:60])
# 2) bfp4 rank 1 [64]
x = torch.randn(64).bfloat16()
t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
y = tilize(t, dtype=ttnn.bfloat4_b, pad_value=0.0)
dv = ttnn.to_torch(y)
hp = torch.nn.functional.pad(x.reshape(1, 64), (0, 0, 0, 31))
h = ttnn.to_torch(ttnn.from_torch(hp, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT))[0]
print("RES bfp4 rank1 device pcc", comp_pcc(x, dv.flatten()[:64], 0.98)[1][:80])
print("RES bfp4 rank1 host   pcc", comp_pcc(x, h.flatten()[:64], 0.98)[1][:80])
print("RES device==host", torch.equal(dv.flatten()[:64], h.flatten()[:64]))
x = torch.randn(64).bfloat16()
hp = torch.nn.functional.pad(x.reshape(1, 64), (0, 0, 0, 31))
h = ttnn.to_torch(ttnn.from_torch(hp, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT))[0]
print("RES bfp4 rank1 host pcc seed2", comp_pcc(x, h.flatten()[:64], 0.98)[1][:80])
ttnn.close_device(dev)
