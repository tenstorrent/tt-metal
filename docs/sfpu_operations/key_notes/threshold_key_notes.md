# threshold

## Formula
x if x > threshold, value otherwise

## Parameters & Common Values
- threshold: comparison value (required, no default)
- value: replacement value when x <= threshold (required, no default)

## Training vs Evaluation Mode
N/A - deterministic, mode-independent.

## PyTorch Reference
[torch.nn.Threshold](https://pytorch.org/docs/stable/generated/torch.nn.Threshold.html)
