# rrelu

## Formula
x if x >= 0, a * x if x < 0

## Parameters & Common Values
- lower: default=1/8 (0.125)
- upper: default=1/3 (0.333...)

## Training vs Evaluation Mode
- Training: a is sampled from Uniform(lower, upper) for each forward pass.
- Evaluation: a is fixed at (lower + upper) / 2.

## PyTorch Reference
[torch.nn.RReLU](https://pytorch.org/docs/stable/generated/torch.nn.RReLU.html)

**Note: NEW - to be implemented.**
