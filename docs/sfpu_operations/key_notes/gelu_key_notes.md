# gelu

## Formula
x * Phi(x), where Phi is the standard normal CDF.

Exact: x * 0.5 * (1 + erf(x / sqrt(2)))

Tanh approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

## Parameters & Common Values
- approximate: 'none' (exact) or 'tanh', common='none'

## Training vs Evaluation Mode
N/A - deterministic, mode-independent.

## PyTorch Reference
[torch.nn.GELU](https://pytorch.org/docs/stable/generated/torch.nn.GELU.html)
