# hardswish

## Formula
x * hardsigmoid(x) = x * clamp(x / 6 + 0.5, 0, 1)

## Parameters & Common Values
No parameters.

## Training vs Evaluation Mode
N/A - deterministic, mode-independent.

## PyTorch Reference
[torch.nn.Hardswish](https://pytorch.org/docs/stable/generated/torch.nn.Hardswish.html)
