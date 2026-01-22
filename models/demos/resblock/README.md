# ResBlock

This package contains a low-latency fused kernel implementation of an example residual block.

## Overview

The residual block can be described using the following pseudocode:

```python
def resblock(activation, weight0, weight1):
    x = relu(activation @ weight0)
    return (x @ weight1) + activation
```

This block is fused into a single operation in [op.py](./op.py) and supports multiple layers with per-layer weights. At a high-level, each layer performs two stages:

1. **Matmul+Relu**: Accumulate matmul result into `dst` register and then apply activation function before packing
2. **Matmul+Add**: Accumulate matmul result into `dst` register and add residual in-place before packing

The operation processes `num_layers` sequentially, with each layer using its own unique `weight0` and `weight1` matrices.

Tests can be run using the following command:

```sh
pytest models/demos/resblock/tests/test_op.py
```

## Architecture

### Data Layout and Sharding

- **Input activations**: Replicated across all matmul cores with shape `[B, K]` where `B` is the batch size (typically small) and `K` is the hidden dimension size. Inputs are height-sharded.
- **Weights**: Width-sharded across matmul cores. Weights are stacked per layer: `weight0` and `weight1` have shape `[num_layers * K, K]` globally, sharded to `[num_layers * K, K/num_cores]` per core. Each layer's `K×K` weights are contiguous in the stacked tensor. The compute kernel pops `K` tiles per layer to advance to the next layer's weights.
- **Output**: Width-sharded across matmul cores with shape `[B, K]` globally, `[B, K/num_cores]` per core.
- **Memory**: All tensors must be sharded and resident in L1.

### Core Allocation

- **Matmul cores**: Determined by the weight sharding grid. These cores perform the matrix multiplications.
- **Mcast core**: Fixed at `(7, 7)`. Used for gathering intermediate results when operating across multiple cores. The matmul cores must not overlap with this core.

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

For each of `num_layers` layers:

1. Reader kernel loads activations and stacked weights (all layers) into CBs at startup
2. Compute kernel performs per layer:
   - First matmul: `activation @ weight0[layer]` → stored in `INTERMEDIATE_PREGATHER_CB`
   - ReLU activation applied in-place
   - Second matmul: `intermediate @ weight1[layer]` → accumulated with residual
   - Result stored in `MM1_FULL_CB` (ping-pong pattern)
   - Weight CBs are popped by `num_tiles_k` to advance to next layer's weights
3. For multi-core operations, intermediate results are gathered via multicast after each matmul stage
4. After all layers, writer kernel outputs the final result

### Weight Stacking

Weights are stacked in PyTorch before conversion to TTNN:
- Create per-layer weights: `weights = [(w0_i, w1_i) for i in range(num_layers)]`
- Stack along row dimension: `stacked_weight0 = torch.cat([w0_i for w0_i, _ in weights], dim=0)` → shape `[num_layers * K, K]`
- Stack similarly for `weight1`
- Convert stacked weights to TTNN width-sharded tensors
- The golden reference function accepts a list of `(weight0, weight1)` tuples and applies them sequentially
