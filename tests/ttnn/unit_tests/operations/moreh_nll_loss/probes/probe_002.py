import torch
import ttnn

torch.manual_seed(0)
shape = [5, 10]
C = shape[1]
target_shape = shape[:1] + shape[2:]

torch_input = torch.rand(shape, dtype=torch.float32)
torch_target = torch.randint(0, C, target_shape, dtype=torch.long)
torch_weight = torch.rand(C, dtype=torch.float32)

print(f"torch_input.shape={torch_input.shape}")
print(f"torch_target={torch_target}")
print(f"torch_weight={torch_weight}")
print(f"input at target positions:")
for n, t in enumerate(torch_target):
    print(f"  n={n} target={t.item()} input[n,t]={torch_input[n,t].item():.4f} weight[t]={torch_weight[t].item():.4f}")

valid = torch_target != 1
print(f"valid mask = {valid}")
print(f"sum-of-weights for valid = {torch_weight[torch_target[valid]].sum().item():.4f}")
print(
    f"sum -input*weight for valid = {(-torch_input[range(5), torch_target] * torch_weight[torch_target])[valid].sum().item():.4f}"
)
expected_loss = (-torch_input[range(5), torch_target] * torch_weight[torch_target])[valid].sum().item() / torch_weight[
    torch_target[valid]
].sum().item()
print(f"expected loss = {expected_loss:.4f}")

nll_loss_obj = torch.nn.NLLLoss(weight=torch_weight, ignore_index=1, reduction="mean")
print(f"torch.NLLLoss(mean) = {nll_loss_obj(torch_input, torch_target).item():.4f}")
