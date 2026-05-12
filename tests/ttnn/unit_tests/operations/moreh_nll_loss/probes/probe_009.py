import torch
import ttnn

torch.manual_seed(0)
shape = [5, 10]
C = shape[1]
torch_input = torch.rand(shape, dtype=torch.float32).requires_grad_()
torch_target = torch.randint(0, C, [5], dtype=torch.long)
torch_weight = torch.rand(C, dtype=torch.float32)
torch_divisor = torch.tensor([0], dtype=torch.float32)
torch_output = torch.tensor([0], dtype=torch.float32)

nll_loss = torch.nn.NLLLoss(weight=torch_weight, ignore_index=1, reduction="mean")
torch_loss = torch.tensor([nll_loss(torch_input, torch_target)])
print(f"torch_loss={torch_loss.item():.6f}")

device = ttnn.open_device(device_id=0)


def t(x, dtype=ttnn.bfloat16):
    return ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


tt_input = t(torch_input)
tt_target = ttnn.from_torch(torch_target, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
tt_weight = t(torch_weight)
tt_divisor = t(torch_divisor)
tt_output = t(torch_output)

# Call the prim directly to inspect step2 output
step2_out = ttnn.prim.moreh_nll_loss_step2(
    tt_input,
    tt_target,
    "mean",
    weight_tensor=tt_weight,
    divisor_tensor=tt_divisor,
    output_tensor=tt_output,
    ignore_index=1,
)
print(f"step2_out shape={step2_out.shape}, padded={step2_out.padded_shape}")
print(f"divisor_t after step1/sum: {ttnn.to_torch(tt_divisor).reshape([1]).item():.6f}")
step2_full = ttnn.to_torch(step2_out)
print(f"step2_out values: {step2_full}")
print(f"step2 sum = {step2_full.sum().item():.6f}")
ttnn.close_device(device)
