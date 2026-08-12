import torch, ttnn
import torch.nn.functional as F
from ttnn.operations.tilize import tilize

device = ttnn.open_device(device_id=0)
crs = lambda a, b: ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(*a), ttnn.CoreCoord(*b))})
ROW = ttnn.ShardOrientation.ROW_MAJOR
L1, DRAM = ttnn.BufferType.L1, ttnn.BufferType.DRAM
HEIGHT = ttnn.TensorMemoryLayout.HEIGHT_SHARDED


def nd(grid, shape):
    return ttnn.MemoryConfig(L1, ttnn.NdShardSpec(ttnn.Shape(shape), grid, ROW))


def legacy(grid, shape, scheme=HEIGHT):
    return ttnn.MemoryConfig(scheme, L1, ttnn.ShardSpec(grid, shape, ROW))


def check(name, shape, padded, pad_value, in_mc, out_mc):
    torch.manual_seed(0)
    x = torch.randn(shape).bfloat16()
    tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_mc)
    try:
        out = tilize(tt, memory_config=out_mc, output_padded_shape=padded, pad_value=pad_value)
    except Exception as e:
        print(f"{name}: RAISED {type(e).__name__}: {e}")
        return
    got = out.cpu().to_torch_with_padded_shape().to(torch.bfloat16)
    xe = x.to(torch.bfloat16)
    pads = tuple(j for i in reversed(range(xe.dim())) for j in (0, padded[i] - xe.shape[i]))
    exp = F.pad(xe, pads, value=float(pad_value))
    ok = list(got.shape) == list(exp.shape) and torch.equal(got, exp)
    print(f"{name}: equal={ok} shape={list(got.shape)} logical={list(out.shape)}")
    if not ok and list(got.shape) == list(exp.shape):
        idx = (got != exp).nonzero()
        print("   nmismatch", idx.shape[0], "first", idx[:4].tolist())


try:
    g2 = crs((0, 0), (1, 0))
    g2c = crs((0, 0), (0, 1))
    check("nd->nd", [3, 50, 96], [3, 64, 96], 10.2, nd(g2, (2, 50, 96)), nd(g2, (1, 64, 96)))
    check("nd->il", [3, 50, 96], [3, 64, 96], 10.2, nd(g2, (2, 50, 96)), ttnn.L1_MEMORY_CONFIG)
    check("il->nd", [3, 50, 96], [3, 64, 96], 0.0, ttnn.DRAM_MEMORY_CONFIG, nd(g2, (1, 64, 96)))
    check("legacyH->il", [3, 100, 128], [3, 128, 128], 10.2, legacy(g2, (150, 128)), ttnn.L1_MEMORY_CONFIG)
    check("il->legacyH", [3, 100, 64], [3, 128, 64], 10.2, ttnn.DRAM_MEMORY_CONFIG, legacy(g2c, (192, 64)))
    check("nd->legacyH", [3, 50, 64], [3, 64, 64], 10.2, nd(g2c, (2, 50, 64)), legacy(g2c, (96, 64)))
    check("legacyH->nd", [3, 100, 128], [3, 128, 128], 10.2, legacy(g2, (150, 128)), nd(g2, (3, 96, 96)))
finally:
    ttnn.close_device(device)
