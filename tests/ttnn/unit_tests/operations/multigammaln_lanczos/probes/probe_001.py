import torch, ttnn
from ttnn.operations.multigammaln_lanczos import multigammaln_lanczos

torch.manual_seed(42)
shape = (1, 1, 32, 64)
half = 0.1 + 1.4 * torch.rand(shape, dtype=torch.float32)
other_half = 2.0 + 3.0 * torch.rand(shape, dtype=torch.float32)
mask = torch.rand(shape) < 0.5
torch_input = torch.where(mask, half, other_half)

torch_expected = torch.special.multigammaln(torch_input.float(), 4)

device = ttnn.open_device(device_id=0)
ttnn_input = ttnn.from_torch(
    torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
)
ttnn_output = multigammaln_lanczos(ttnn_input)
actual = ttnn.to_torch(ttnn_output).float()

# Inspect: for IN-DOMAIN positions (a > 1.5), is kernel finite?
in_domain_mask = torch_input > 1.5
ood_mask = torch_input <= 1.5
print(f"In-domain count: {in_domain_mask.sum().item()} / {in_domain_mask.numel()}")
print(f"OOD count: {ood_mask.sum().item()}")
print(f"Kernel non-finite at in-domain positions: {(~torch.isfinite(actual) & in_domain_mask).sum().item()}")
print(f"Kernel non-finite at OOD positions: {(~torch.isfinite(actual) & ood_mask).sum().item()}")
print(f"Torch NaN at OOD positions: {(torch.isnan(torch_expected) & ood_mask).sum().item()}")
print(f"Torch finite at OOD positions: {(torch.isfinite(torch_expected) & ood_mask).sum().item()}")

# In-domain accuracy check:
in_act = actual[in_domain_mask]
in_exp = torch_expected[in_domain_mask]
print(f"In-domain max abs diff: {(in_act - in_exp).abs().max().item():.6f}")
print(f"In-domain mean abs diff: {(in_act - in_exp).abs().mean().item():.6f}")
ttnn.close_device(device)
