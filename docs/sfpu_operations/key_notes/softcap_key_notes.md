# softcap

## Formula
softcap(x, cap) = cap * tanh(x / cap)

## Parameters
- cap: float, positive scalar (default = 50.0)

## Precision Requirements
- Use extended Taylor series for tanh to maximize fp32 accuracy
- ULP-based accuracy metric is critical — minimize ULP error
- Consider range reduction: x/cap maps input to tanh's domain
- For small u = x/cap, tanh(u) ≈ u - u³/3 + 2u⁵/15 - 17u⁷/315 + ...
- For large |u|, tanh(u) → sign(u), so softcap → ±cap

## PyTorch Reference
cap * torch.tanh(x / cap)

**Note: NEW - to be implemented.**
