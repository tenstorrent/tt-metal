import torch, ttnn
from ttnn.operations.tilize import tilize

device = ttnn.open_device(device_id=0)
CRS = lambda ex, ey: ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(ex, ey))})


def check(label, x, out, pad_value, logical_shape):
    lg = ttnn.to_torch(out)
    ok_logical = list(lg.shape) == list(logical_shape) and torch.equal(lg, x)
    pad = out.cpu().to_torch_with_padded_shape()
    xs = x.reshape((1,) * (len(list(pad.shape)) - x.dim()) + tuple(x.shape)) if x.dim() < len(list(pad.shape)) else x
    pads = tuple(j for i in reversed(range(xs.dim())) for j in (0, list(pad.shape)[i] - xs.shape[i]))
    exp = torch.nn.functional.pad(xs, pads, value=pad_value)
    ok = torch.equal(pad, exp)
    print(
        f"{label}: padded={list(pad.shape)} logical_ok={ok_logical} pad_ok={ok}"
        + ("" if ok else f"  ndiff={(pad != exp).sum().item()}/{exp.numel()}")
    )


def run(label, shape, pad_value=0.0, target=None, in_mc=None, out_mc=None, low_l1=False):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16)
    ti = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=in_mc or ttnn.DRAM_MEMORY_CONFIG,
    )
    kw = {"pad_value": pad_value, "low_l1": low_l1}
    if target is not None:
        kw["output_padded_shape"] = target
    out = tilize(ti, out_mc or ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16, **kw)
    check(label, x, out, pad_value, shape)


try:
    run("rank5", [1, 2, 1, 50, 50], -1.0)
    run("padded_l1_to_l1", [1, 1, 32, 50], 0.0, in_mc=ttnn.L1_MEMORY_CONFIG, out_mc=ttnn.L1_MEMORY_CONFIG)
    run("padded_low_l1", [1, 1, 50, 50], 0.0, low_l1=True)
    # padded -> height sharded out
    hs = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(CRS(1, 0), (32, 64), ttnn.ShardOrientation.ROW_MAJOR),
    )
    run("padded_to_height_sharded", [1, 1, 50, 64], 0.0, out_mc=hs)
    run("leading_fold_h_tail", [8, 1, 249, 2048], 0.0)
    # nd sharded input, width-cut page (strided + pad)
    nd_in = ttnn.MemoryConfig(
        ttnn.BufferType.L1, ttnn.NdShardSpec(ttnn.Shape([2, 64, 96]), CRS(1, 0), ttnn.ShardOrientation.ROW_MAJOR)
    )
    run("nd_in_widthcut_pad", [3, 100, 158], 10.2, target=[3, 128, 160], in_mc=nd_in)
    nd_in2 = ttnn.MemoryConfig(
        ttnn.BufferType.L1, ttnn.NdShardSpec(ttnn.Shape([2, 50, 96]), CRS(1, 0), ttnn.ShardOrientation.ROW_MAJOR)
    )
    run("nd_in_h_tail_pad", [3, 50, 96], 10.2, target=[3, 64, 96], in_mc=nd_in2)
    # explicit padded to nd sharded out
    nd_out = ttnn.MemoryConfig(
        ttnn.BufferType.L1, ttnn.NdShardSpec(ttnn.Shape([1, 64, 96]), CRS(1, 0), ttnn.ShardOrientation.ROW_MAJOR)
    )
    run("pad_to_nd_sharded_out", [3, 50, 96], 10.2, target=[3, 64, 96], out_mc=nd_out)
finally:
    ttnn.close_device(device)
