import torch

torch.manual_seed(0)
shape = [5, 10]
torch_input = torch.rand(shape, dtype=torch.float32)
torch_target = torch.randint(0, 10, [5], dtype=torch.long)
torch_weight = torch.rand(10, dtype=torch.float32)
print(f"target={torch_target}")
print(f"weight={torch_weight}")
print(f"input rows for n=0..4 cols=target value: input[0,{torch_target[0]}]={torch_input[0, torch_target[0]]:.4f}")
