# elu

## Formula
x if x > 0, alpha * (exp(x) - 1) if x <= 0

## Parameters & Common Values
- alpha: default=1.0, common range=[0.1, 3.0]

## Training vs Evaluation Mode
N/A - deterministic, mode-independent.

## PyTorch Reference
[torch.nn.ELU](https://pytorch.org/docs/stable/generated/torch.nn.ELU.html)
