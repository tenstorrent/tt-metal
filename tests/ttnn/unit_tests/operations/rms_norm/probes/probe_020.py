import torch, ttnn
from ttnn.operations.rms_norm import rms_norm

dev = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(0)
    W = 1016
    tx = torch.randn((1, 1, 32, W)).to(torch.bfloat16)
    x = ttnn.from_torch(tx, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    c = ttnn.ComputeConfigDescriptor()
    c.fp32_dest_acc_en = False
    out = ttnn.to_torch(rms_norm(x, compute_kernel_config=c)).float()
    print("RESULT ok", out.shape)
finally:
    ttnn.close_device(dev)
