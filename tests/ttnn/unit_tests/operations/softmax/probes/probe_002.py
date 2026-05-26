"""Quick probe — softmax with non-aligned W."""
import torch
import ttnn

device = ttnn.open_device(device_id=0)
try:
    from ttnn.operations.softmax import softmax, SUPPORTED, validate

    print("SUPPORTED[alignment] =", SUPPORTED["alignment"])

    # Smallest non-aligned shape: 1x1x32x33 (W partial=1)
    torch.manual_seed(0)
    shape = (1, 1, 32, 33)
    x = torch.randn(shape, dtype=torch.float32)
    expected = torch.softmax(x, dim=-1)

    tt_x = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    print(f"ttnn shape: {tt_x.shape}")
    tt_y = softmax(tt_x, dim=-1)
    print(f"output shape: {tt_y.shape}")
    out = ttnn.to_torch(tt_y)
    diff = (out - expected).abs().max().item()
    print(f"max abs diff: {diff:.4e}")
    # quick PCC sanity
    o_flat = out.flatten().float()
    e_flat = expected.flatten().float()
    pcc = (
        ((o_flat - o_flat.mean()) * (e_flat - e_flat.mean())).sum()
        / ((o_flat - o_flat.mean()).norm() * (e_flat - e_flat.mean()).norm())
    ).item()
    print(f"PCC: {pcc:.6f}")
finally:
    ttnn.close_device(device)
