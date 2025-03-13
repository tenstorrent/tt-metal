import torch
import ttnn

with ttnn.manage_device(device_id=0) as device:
    x_torch = torch.ones((2, 4, 32, 32), dtype=torch.bfloat16)
    gamma_torch = torch.randn((1, 1, 1, 32), dtype=torch.bfloat16)

    x = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    gamma = ttnn.from_torch(gamma_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    x_original = ttnn.rms_norm(x, weight=gamma, epsilon=1e-5)
    x_new, rms = ttnn.experimental.rmsnorm_fw(x, gamma, True, 1e-5)
    print(x_original.shape)
    print(x_new.shape)
    print(rms.shape)

    # x = ttnn.experimental.dropout(x, 0.5, 0.5, 1337)
    # print(x.shape)
