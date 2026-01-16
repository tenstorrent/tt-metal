# ResBlock

This package contains a low-latency fused kernel implementation of an example residual block.

## Overview

The residual block can be described using the following pseudocode:

```python
def resblock(activation, weight0, weight1):
    x = relu(activation @ weight0)
    return (x @ weight1) + activation
```

This block is fused into a single operation in [op.py](./op.py). At a high-level, the operation is broken down into two stages:

1. **Matmul+Relu**: Accumulate matmul result into `dst` register and then apply activation function before packing
2. **Matmul+Add**: Accumulate matmul result into `dst` register and add residual in-place before packing

## Architecture

### Data Layout and Sharding

- **Input activations**: Replicated across all matmul cores with shape `[B, K]` where `B` is the batch size (typically small) and `K` is the hidden dimension size. Inputs are height-sharded.
- **Weights**: Width-sharded across matmul cores. Each weight matrix has shape `[K, K]` globally, sharded to `[K, K/num_cores]` per core.
- **Output**: Width-sharded across matmul cores with shape `[B, K]` globally, `[B, K/num_cores]` per core.
- **Memory**: All tensors must be sharded and resident in L1.

### Core Allocation

- **Matmul cores**: Determined by the weight sharding grid. These cores perform the matrix multiplications.
- **Mcast core**: Fixed at `(7, 4)`. Used for gathering intermediate results when operating across multiple cores. The matmul cores must not overlap with this core.

### Multicast

A dedicated multicast core gathers intermediate results from all matmul cores. Each matmul core will produce one tile of output after the first matmul stage but requires the fully replicated input to proceed to the next stage.

### Kernels

The operation uses five kernels:

1. **reader.cpp**: Reads activations and weights into CBs, handles multicast coordination
2. **compute.cpp**: Performs the fused matmul+relu and matmul+add operations
3. **writer.cpp**: Writes the final output
4. **mcast_reader.cpp**: Reads gathered data on the multicast core
5. **mcast_writer.cpp**: Writes data to be gathered on the multicast core

### Computation Flow

1. Reader kernel loads activations and weights into CBs
2. Compute kernel performs:
   - First matmul: `activation @ weight0` → stored in `INTERMEDIATE_PREGATHER_CB`
   - ReLU activation applied in-place
   - Second matmul: `intermediate @ weight1` → accumulated with residual
   - Result stored in `OUT_CB`
3. For multi-core operations, intermediate results are gathered via multicast
4. Writer kernel outputs the final result
