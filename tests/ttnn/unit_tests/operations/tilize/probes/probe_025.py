import sys, torch, ttnn

sys.path.insert(0, "tests/ttnn/unit_tests/operations/tilize")
import test_tilize_numeric_formats as T

dev = ttnn.open_device(device_id=0)
e, g = T._run(dev, (32, 32), ttnn.uint8, ttnn.uint8)
print("RES", e.dtype, g.dtype, e.shape, g.shape, e.flatten()[:4].tolist(), g.flatten()[:4].tolist(), torch.equal(e, g))
ed, gd = e.double(), g.double()
print("RES", ed.flatten()[:4].tolist(), gd.flatten()[:4].tolist(), (gd - ed).abs().max().item())
print("RES", T._metrics(e, g))
ttnn.close_device(dev)
