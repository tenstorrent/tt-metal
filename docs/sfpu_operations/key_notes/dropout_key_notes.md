# dropout

## Formula
Randomly zeroes elements with probability p, scales remaining elements by 1 / (1 - p).

## Parameters & Common Values
- p: default=0.5, common range=[0.1, 0.5]

## Training vs Evaluation Mode
- Training: stochastic dropout applied (elements randomly zeroed).
- Evaluation: identity function (no dropout, tensor passed through unchanged).

## PyTorch Reference
[torch.nn.Dropout](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html)

**Note: EXCLUDED from nuke (infrastructure operation).**
