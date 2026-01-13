# ResBlock

This package contains a low-latency fused kernel implementation of an example residual block.

Note that this code is still under construction.

### Overview

The residual block can be described using the following pseudocode:

```python
def resblock(activation, weight0, weight1) {
    x = relu(activation @ weight0)
    return (x @ weight1) + activation
}
```

This block is fused into a single operation in [op.py](./op.py). At a high-level, the operation is broken down into two stages:

1. Matmul+Relu: Accumulate matmul result into `dst` register and then apply activation function before packing
2. Matmul+Add: Accumulate matmul result into `dst` register and add residual in-place before packing

The operation consumes replicated input activations of shape `[B, K]`, where `B` is the batch size (which is typically small) and  `K` is the hidden dimension size.

The input activations are replicated across each core and the weights are width sharded across the cores. The weights and activations are both assumed to be resident in L1.

When operating on inputs sharded over more than one core, we must also multicast to gather the result of each matmul across all cores.
