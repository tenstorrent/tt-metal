import torch, ttnn

device = ttnn.open_device(device_id=0)
try:
    # Check zero-padding of TILE tensors for non-aligned shapes
    t = torch.randn(1, 1, 47, 50, dtype=torch.bfloat16)
    tt = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    print("logical shape:", tt.shape)
    print("padded shape:", tt.padded_shape if hasattr(tt, "padded_shape") else "n/a")
    # read back full padded
    back = ttnn.to_torch(tt)
    print("readback shape:", back.shape)
    # Now read with padding using to_torch on a tensor that exposes padding?
    # Check the physical: convert to row-major to see padding
    rm = ttnn.to_torch(tt)
    print("readback matches logical region:", torch.allclose(rm.float(), t.float(), atol=0.01))
finally:
    ttnn.close_device(device)
