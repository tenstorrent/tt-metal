import torch, ttnn
from ttnn.operations.conv2d_3x3 import conv2d_3x3, prepare_conv2d_bias, prepare_conv2d_weights

device = ttnn.open_device(device_id=0)


def run(
    name,
    H,
    W,
    c_in,
    c_out,
    p,
    s,
    d,
    layout,
    dtype,
    wdtype,
    shard,
    grid,
    fid=ttnn.MathFidelity.HiFi4,
    bias=True,
    relu=True,
):
    torch.manual_seed(0)
    mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid[0] - 1, grid[1] - 1))]),
            shard,
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    kw = dict(
        stride=s,
        padding=p,
        dilation=d,
        input_dtype=dtype,
        input_layout=layout,
        weight_dtype=wdtype,
        input_memory_config=mc,
        device=device,
    )
    w = torch.randn(c_out, c_in, 3, 3).to(torch.bfloat16)
    b = torch.randn(c_out).to(torch.bfloat16) if bias else None
    tw = prepare_conv2d_weights(w, **kw)
    tb = prepare_conv2d_bias(b, **kw) if bias else None
    x = torch.randn(1, 1, H * W, c_in).to(torch.bfloat16)
    tx = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=mc)
    xr = ttnn.to_torch(tx).float()  # the quantized input the device sees
    y = conv2d_3x3(
        tx,
        tw,
        input_height=H,
        input_width=W,
        bias=tb,
        padding=p,
        stride=s,
        dilation=d,
        activation="relu" if relu else None,
        compute_kernel_config=ttnn.ComputeConfigDescriptor(
            math_fidelity=fid, fp32_dest_acc_en=True, math_approx_mode=False
        ),
    )
    got = ttnn.to_torch(y).float().flatten()
    ref = torch.nn.functional.conv2d(
        x.float().reshape(1, H, W, c_in).permute(0, 3, 1, 2),
        w.float(),
        None if b is None else b.float(),
        stride=s,
        padding=p,
        dilation=d,
    )
    if relu:
        ref = torch.relu(ref)
    ref = ref.permute(0, 2, 3, 1).flatten()
    pcc = torch.corrcoef(torch.stack([got, ref]))[0, 1].item()
    rr = ((got - ref).pow(2).mean().sqrt() / ref.std()).item()
    print(f"ACC {name}: pcc={pcc:.6f} rel_rms={rr:.4f}")


try:
    run(
        "rm_bf16 16x16x64->128 (4 slices) bf16 w",
        16,
        16,
        64,
        128,
        1,
        1,
        1,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.bfloat16,
        ttnn.bfloat16,
        [40, 16],
        (4, 7),
    )
    run(
        "tile_bf16 8x16x256->256 (8 slices) bf16 w",
        8,
        16,
        256,
        256,
        1,
        1,
        1,
        ttnn.TILE_LAYOUT,
        ttnn.bfloat16,
        ttnn.bfloat16,
        [32, 32],
        (8, 4),
    )
    run(
        "res4 HiFi4 bf8b",
        32,
        64,
        256,
        256,
        1,
        1,
        1,
        ttnn.TILE_LAYOUT,
        ttnn.bfloat8_b,
        ttnn.bfloat8_b,
        [256, 32],
        (8, 8),
    )
    run(
        "res4 LoFi bf8b",
        32,
        64,
        256,
        256,
        1,
        1,
        1,
        ttnn.TILE_LAYOUT,
        ttnn.bfloat8_b,
        ttnn.bfloat8_b,
        [256, 32],
        (8, 8),
        fid=ttnn.MathFidelity.LoFi,
    )
    run(
        "aspp.convs.1 bf8b",
        32,
        64,
        2048,
        256,
        6,
        1,
        6,
        ttnn.TILE_LAYOUT,
        ttnn.bfloat8_b,
        ttnn.bfloat8_b,
        [256, 192],
        (11, 8),
    )
finally:
    ttnn.close_device(device)
