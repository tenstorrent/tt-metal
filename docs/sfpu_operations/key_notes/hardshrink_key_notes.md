# hardshrink

## Formula
x if |x| > lambda, 0 otherwise

## Parameters & Common Values
- lambda: default=0.5, common range=[0.1, 1.0]

## Training vs Evaluation Mode
N/A - deterministic, mode-independent.

## PyTorch Reference
[torch.nn.Hardshrink](https://pytorch.org/docs/stable/generated/torch.nn.Hardshrink.html)
