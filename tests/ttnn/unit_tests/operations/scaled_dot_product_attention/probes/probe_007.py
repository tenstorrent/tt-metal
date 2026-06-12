import torch, ttnn

device = ttnn.open_device(device_id=0)
try:
    # Make logical region a known constant, see what padding holds after tilize->untilize roundtrip via device
    t = torch.full((1, 1, 47, 50), 7.0, dtype=torch.bfloat16)
    tt = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    # Convert to row-major on device then read back the PADDED shape using to_torch with logical...
    # Use tt.cpu() then untilize to see padded region:
    cpu = ttnn.to_torch(tt.cpu().to(ttnn.ROW_MAJOR_LAYOUT))
    print("cpu rm shape:", cpu.shape)  # likely logical (strips padding)
    # Try reading padded via internal: reshape
    import numpy as np

    # Use ttnn.to_torch with no strip? Check pad value by tilize of a tensor whose pad we set
finally:
    ttnn.close_device(device)
