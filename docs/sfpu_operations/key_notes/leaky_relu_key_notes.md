# leaky_relu

## Formula
max(0, x) + negative_slope * min(0, x)

## Parameters & Common Values
- negative_slope: default=0.01, common range=[0.001, 0.3]

## Training vs Evaluation Mode
N/A - deterministic, mode-independent.

## PyTorch Reference
[torch.nn.LeakyReLU](https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html)
