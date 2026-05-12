import torch

torch.manual_seed(0)
shape = [5, 10]
torch_input = torch.rand(shape, dtype=torch.float32)
torch_target = torch.randint(0, 10, [5], dtype=torch.long)
torch_weight = torch.rand(10, dtype=torch.float32)
print(f"input={torch_input}")
print(f"target={torch_target}")
print(f"weight={torch_weight}")
