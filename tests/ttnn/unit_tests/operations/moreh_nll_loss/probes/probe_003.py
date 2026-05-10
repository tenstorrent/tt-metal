import torch
import ttnn

torch.manual_seed(0)
shape = [5, 10]
C = shape[1]
target_shape = shape[:1] + shape[2:]

torch_input = torch.rand(shape, dtype=torch.float32)
torch_target = torch.randint(0, C, target_shape, dtype=torch.long)
torch_weight = torch.rand(C, dtype=torch.float32)
torch_divisor = torch.tensor([0], dtype=torch.float32)
torch_output = torch.tensor([0], dtype=torch.float32)

device = ttnn.open_device(device_id=0)


def t(x, dtype=ttnn.bfloat16):
    return ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


tt_input = t(torch_input)
tt_target = ttnn.from_torch(torch_target, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
tt_weight = t(torch_weight)
tt_divisor = t(torch_divisor)
tt_output = t(torch_output)

# Manually run step1 (which calls moreh_nll_loss_step1 + moreh_sum)
from ttnn.operations.moreh import moreh_nll_loss_step1

print(f">>> Calling step1...")
step1_out = ttnn.operations.moreh.moreh_nll_loss_step1(
    tt_target, weight_tensor=tt_weight, ignore_index=1, reduction="mean", output_dtype=tt_input.dtype, c=C
)
print(f"step1_out shape={step1_out.shape}, dtype={step1_out.dtype}")
print(f"step1_out torch:")
print(ttnn.to_torch(step1_out))

# Now moreh_sum reduces step1_out into divisor
print(f">>> Calling moreh_sum -> divisor...")
divisor_filled = ttnn.operations.moreh.moreh_sum(step1_out, output_tensor=tt_divisor)
print(f"divisor_filled shape={divisor_filled.shape}")
print(f"divisor_filled torch={ttnn.to_torch(divisor_filled)}")
print(f"expected divisor (sum-of-weights for valid) = 1.1335")
ttnn.close_device(device)
