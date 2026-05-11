import torch, ttnn
from ttnn.operations.multigammaln import multigammaln

torch.manual_seed(42)
shape = (1, 1, 32, 64)
half = torch.rand(shape, dtype=torch.float32) * 1.5
other_half = 1.6 + 3.0 * torch.rand(shape, dtype=torch.float32)
mask = torch.rand(shape) < 0.5
torch_input = torch.where(mask, half, other_half)

torch_expected = torch.special.multigammaln(torch_input.float(), 4)

device = ttnn.open_device(device_id=0)
ttnn_input = ttnn.from_torch(
    torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
)
ttnn_output = multigammaln(ttnn_input)
actual = ttnn.to_torch(ttnn_output).float()
ttnn.close_device(device)

nan_a = torch.isnan(actual)
nan_e = torch.isnan(torch_expected)
print("NaN positions match:", torch.equal(nan_a, nan_e))
print("Actual: #NaN =", nan_a.sum().item(), "#inf =", torch.isinf(actual).sum().item())
print("Expected: #NaN =", nan_e.sum().item(), "#inf =", torch.isinf(torch_expected).sum().item())

# Find positions where actual is inf but expected is finite
bad = torch.isinf(actual) & ~torch.isinf(torch_expected) & ~torch.isnan(torch_expected)
print("Actual inf, expected finite:", bad.sum().item())
if bad.any():
    idx = bad.flatten().nonzero().flatten()[:5]
    flat_input = torch_input.flatten()
    flat_actual = actual.flatten()
    flat_exp = torch_expected.flatten()
    for i in idx.tolist():
        print(
            f"  i={i}: input={flat_input[i].item():.6f}, actual={flat_actual[i].item()}, expected={flat_exp[i].item():.6f}"
        )

bad2 = torch.isinf(torch_expected) & ~torch.isinf(actual) & ~torch.isnan(actual)
print("Expected inf, actual finite:", bad2.sum().item())
if bad2.any():
    idx = bad2.flatten().nonzero().flatten()[:5]
    flat_input = torch_input.flatten()
    flat_actual = actual.flatten()
    flat_exp = torch_expected.flatten()
    for i in idx.tolist():
        print(
            f"  i={i}: input={flat_input[i].item():.6f}, actual={flat_actual[i].item():.6f}, expected={flat_exp[i].item()}"
        )
