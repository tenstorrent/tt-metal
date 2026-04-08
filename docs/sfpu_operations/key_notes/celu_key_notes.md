# celu

## Formula
max(0, x) + min(0, alpha * (exp(x / alpha) - 1))

## Parameters & Common Values
- alpha: default=1.0, common range=[0.5, 2.0]

## Training vs Evaluation Mode
N/A - deterministic, mode-independent.

## PyTorch Reference
[torch.nn.CELU](https://pytorch.org/docs/stable/generated/torch.nn.CELU.html)
