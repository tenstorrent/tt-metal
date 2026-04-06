# relu_min

## Formula
max(relu(x), lower_limit)

## Parameters & Common Values
- lower_limit: user-specified floor for the ReLU output

## Training vs Evaluation Mode
N/A - deterministic, mode-independent.

## PyTorch Reference
No direct PyTorch equivalent. Composite of ReLU + clamp_min.
