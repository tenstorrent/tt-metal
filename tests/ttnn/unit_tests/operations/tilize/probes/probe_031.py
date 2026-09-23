import sys, torch, ttnn

sys.path.insert(0, "/tmp/r7pkg")
from tilize_r7 import tilize as t7
from ttnn.operations.tilize import tilize as tnow

device = ttnn.open_device(device_id=0)
same = 0
n = 0
diffs = []
for seed in range(20):
    for shape, pv in [((1, 1, 50, 50), -2.5), ((1, 1, 128, 64), None), ((1, 1, 32, 2048), None)]:
        torch.manual_seed(seed)
        x = torch.randn(shape, dtype=torch.float32)
        t = ttnn.from_torch(
            x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        kw = dict(dtype=ttnn.bfloat4_b)
        if pv is not None:
            kw["pad_value"] = pv
        a = t7(t, **kw)
        b = tnow(t, **kw)
        ra = a.cpu().to_torch_with_padded_shape()
        rb = b.cpu().to_torch_with_padded_shape()
        n += 1
        if torch.equal(ra, rb):
            same += 1
        else:
            diffs.append((seed, shape))
print("RES bit-identical", same, "of", n, "diffs", diffs[:5])
ttnn.close_device(device)
