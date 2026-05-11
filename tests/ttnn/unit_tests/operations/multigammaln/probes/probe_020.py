import torch, ttnn, sys

torch.manual_seed(42)
shape = (1, 1, 32, 64)
half = torch.rand(shape, dtype=torch.float32) * 1.5  # [0, 1.5]
other_half = 1.6 + 3.0 * torch.rand(shape, dtype=torch.float32)  # (1.6, 4.6)
mask = torch.rand(shape) < 0.5
torch_input = torch.where(mask, half, other_half)
device = ttnn.open_device(device_id=0)
ttnn_input = ttnn.from_torch(
    torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
)
from ttnn.operations.multigammaln import multigammaln

ttnn_output = multigammaln(ttnn_input)
actual = ttnn.to_torch(ttnn_output).float()
expected = torch.special.multigammaln(torch_input.float(), 4)
nan_actual = torch.isnan(actual)
nan_expected = torch.isnan(expected)
print("NaN match:", torch.equal(nan_actual, nan_expected))
# Inf positions in actual
inf_actual = torch.isinf(actual) & ~nan_actual
print("Inf positions in actual:", inf_actual.sum().item())
if inf_actual.any():
    idxs = inf_actual.nonzero()[:5]
    for idx in idxs:
        i = tuple(idx.tolist())
        print(
            f"  pos={i}, input={torch_input[i].item():.6f}, actual={actual[i].item()}, expected={expected[i].item():.6f}"
        )
ttnn.close_device(device)
