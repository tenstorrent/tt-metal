# prelu_sfpu

## Formula
max(0, x) + weight * min(0, x)

## Parameters & Common Values
- weight: learnable parameter, default init=0.25

## Training vs Evaluation Mode
- Training: weight is learned via backpropagation.
- Evaluation: weight is fixed at its learned value.

## PyTorch Reference
[torch.nn.PReLU](https://pytorch.org/docs/stable/generated/torch.nn.PReLU.html)
