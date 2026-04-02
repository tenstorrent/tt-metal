#!/usr/bin/env python3
import torch
import ttnn
import numpy as np

device = ttnn.open_device(device_id=0)

torch.manual_seed(213919)
torch_input = torch.rand((1, 1, 32, 32), dtype=torch.bfloat16) * 9.0 + 1.0
print(f"Input range: [{torch_input.min().item()}, {torch_input.max().item()}]")
print(f"Any input non-finite? {not torch.isfinite(torch_input).all()}")

input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

for k in [1, 5]:
    output = ttnn.polygamma(input_tensor, k)
    output_torch = ttnn.to_torch(output)
    ttnn.deallocate(output)

    non_finite_mask = ~torch.isfinite(output_torch)
    num_non_finite = non_finite_mask.sum().item()
    print(f"\nn={k}: non-finite count = {num_non_finite} / {output_torch.numel()}")

    if num_non_finite > 0:
        indices = torch.nonzero(non_finite_mask)
        print(f"  Non-finite positions (first 10):")
        for idx in indices[:10]:
            pos = tuple(idx.tolist())
            print(f"    {pos}: input={torch_input[pos].item():.6f}, output={output_torch[pos].item()}")

    print(
        f"  Output range (finite only): [{output_torch[torch.isfinite(output_torch)].min().item():.6f}, "
        f"{output_torch[torch.isfinite(output_torch)].max().item():.6f}]"
    )
    print(f"  Output shape: {output_torch.shape}")

ttnn.deallocate(input_tensor)
ttnn.close_device(device)
