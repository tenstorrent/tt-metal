import torch
import ttnn
from models.common.utility_functions import comp_allclose_and_pcc

torch.manual_seed(0)
shape = [5, 10]
C = shape[1]
target_shape = shape[:1] + shape[2:]

torch_input = torch.rand(shape, dtype=torch.float32).requires_grad_()
torch_target = torch.randint(0, C, target_shape, dtype=torch.long)
torch_weight = torch.rand(C, dtype=torch.float32)
torch_divisor = torch.tensor([0], dtype=torch.float32)
torch_output = torch.tensor([0], dtype=torch.float32)

nll_loss = torch.nn.NLLLoss(weight=torch_weight, ignore_index=1, reduction="mean")
torch_loss = torch.tensor([nll_loss(torch_input, torch_target)])

device = ttnn.open_device(device_id=0)


def t(x, dtype=ttnn.bfloat16):
    return ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


tt_input = t(torch_input)
tt_target = ttnn.from_torch(torch_target, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
tt_weight = t(torch_weight)
tt_divisor = t(torch_divisor)
tt_output = t(torch_output)

tt_loss = ttnn.operations.moreh.nll_loss(
    tt_input,
    tt_target,
    "mean",
    weight_tensor=tt_weight,
    divisor_tensor=tt_divisor,
    output_tensor=tt_output,
    ignore_index=1,
)

tt_loss = ttnn.to_torch(tt_loss).reshape([1])

print(f"\n>>> Torch loss = {torch_loss}")
print(f">>> TTNN  loss = {tt_loss}")
print(f">>> Diff       = {(tt_loss - torch_loss).abs()}")
print(f">>> PCC/allclose: {comp_allclose_and_pcc(torch_loss, tt_loss, pcc=0.99, rtol=0.05, atol=0.05)}")
ttnn.close_device(device)
