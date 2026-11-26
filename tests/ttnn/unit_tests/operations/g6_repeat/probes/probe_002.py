import torch, ttnn
from ttnn.operations.g6_repeat import g6_repeat

device = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(42)
    W = 256
    full = torch.randn((1, 1, 32, W), dtype=torch.float32).to(torch.bfloat16)
    bits = full.view(torch.int16).clone()
    bits[..., 1:, :] = torch.tensor(0x7FC1, dtype=torch.int32).to(torch.int16)  # NaN with payload
    full = bits.view(torch.bfloat16)
    t32 = ttnn.from_torch(
        full, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tx = ttnn.reshape(t32, ttnn.Shape([1, 1, 1, W]), ttnn.Shape([1, 1, 32, W]))
    print("PROBE shape", list(tx.shape), "same buffer", tx.buffer_address() == t32.buffer_address())
    padded = tx.cpu().to_torch_with_padded_shape()
    pb = padded[..., 1:, :].contiguous().view(torch.int16).reshape(-1)
    print("PROBE padding bits", [hex(int(u) & 0xFFFF) for u in torch.unique(pb)])
    out = g6_repeat(tx, [1, 1, 16, 1])
    got = ttnn.to_torch(out).view(torch.int16)
    want = full[..., :1, :].repeat(1, 1, 16, 1).view(torch.int16)
    print("PROBE equal", torch.equal(got, want))
finally:
    ttnn.close_device(device)
