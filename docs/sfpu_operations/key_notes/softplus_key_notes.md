# softplus

## Formula
(1 / beta) * log(1 + exp(beta * x))

Reverts to linear function (x) when beta * x > threshold for numerical stability.

## Parameters & Common Values
- beta: default=1.0, common range=[0.5, 5.0]
- threshold: default=20.0

## Training vs Evaluation Mode
N/A - deterministic, mode-independent.

## PyTorch Reference
[torch.nn.Softplus](https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html)
