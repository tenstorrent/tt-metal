import torch, ttnn
import ttnn.operations.conv2d_nhwc.conv2d_nhwc as m
from ttnn.operations.conv2d_nhwc import conv2d_nhwc, prepare_conv2d_weights, prepare_conv2d_bias

m.SUPPORTED["stride_mode"] = ["unit", "strided"]
dev = ttnn.open_device(device_id=0)
cases = [
    ((1, 56, 56, 64), 128, 3, 1, 2),
    ((1, 28, 28, 128), 256, 3, 1, 2),
    ((1, 14, 14, 256), 512, 3, 1, 2),
    ((1, 56, 56, 256), 512, 1, 0, 2),
    ((1, 32, 32, 32), 32, 5, 2, 2),
    ((1, 17, 17, 32), 64, 3, 1, 2),
    ((1, 32, 32, 64), 64, 1, 0, 2),
    ((2, 28, 28, 64), 64, 3, 1, 3),
]
for shape, co, k, p, s in cases:
    torch.manual_seed(0)
    N, H, W, C = shape
    x = torch.randn(shape)
    w = torch.randn(co, C, k, k) / (C * k * k) ** 0.5
    b = torch.randn(co)
    ref = torch.nn.functional.conv2d(x.permute(0, 3, 1, 2), w, b, stride=s, padding=p).permute(0, 2, 3, 1)
    tx = ttnn.from_torch(
        x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    c = dict(
        kernel_size=k,
        stride=s,
        padding=p,
        input_dtype=ttnn.float32,
        weight_dtype=ttnn.float32,
        input_memory_config=tx.memory_config(),
        device=dev,
    )
    o = ttnn.to_torch(
        conv2d_nhwc(tx, prepare_conv2d_weights(w, **c), k, bias=prepare_conv2d_bias(b, **c), padding=p, stride=s)
    ).float()
    pcc = torch.corrcoef(torch.stack([o.flatten(), ref.flatten()]))[0, 1].item()
    print("RES", shape, co, k, p, s, tuple(o.shape) == tuple(ref.shape), f"{pcc:.6f}")
ttnn.close_device(dev)
