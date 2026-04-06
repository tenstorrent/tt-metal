# relu_max

## Formula
min(max(0, x), upper_limit)

## Parameters & Common Values
- upper_limit: user-specified ceiling for the ReLU output

## Training vs Evaluation Mode
N/A - deterministic, mode-independent.

## PyTorch Reference
Generalization of ReLU6; no direct single PyTorch equivalent. See [torch.clamp](https://pytorch.org/docs/stable/generated/torch.clamp.html).
