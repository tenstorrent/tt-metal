# selu

## Formula
scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))

## Parameters & Common Values
- alpha: 1.6732632423543772 (fixed constant)
- scale: 1.0507009873554805 (fixed constant)

## Training vs Evaluation Mode
N/A - deterministic, mode-independent. Designed for self-normalizing networks (SNNs).

## PyTorch Reference
[torch.nn.SELU](https://pytorch.org/docs/stable/generated/torch.nn.SELU.html)
