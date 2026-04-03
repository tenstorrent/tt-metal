# softshrink

## Formula
x - lambda if x > lambda, x + lambda if x < -lambda, 0 otherwise

## Parameters & Common Values
- lambda: default=0.5, common range=[0.1, 1.0]

## Training vs Evaluation Mode
N/A - deterministic, mode-independent.

## PyTorch Reference
[torch.nn.Softshrink](https://pytorch.org/docs/stable/generated/torch.nn.Softshrink.html)
