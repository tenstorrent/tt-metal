# LLMs in TT-NN
Authors:
## Contents
- [LLMs in TT-NN](#llms-in-tt-nn)
  - [Contents](#contents)
  - [1. Overview](#1-overview)
  - [2. Modules](#2-modules)
    - [2.1 Embedding](#21-embedding)
    - [2.2 RoPE](#22-rope)
    - [2.3 Norm](#23-norm)
    - [2.4 Attention](#24-attention)
    - [2.5 MLP](#25-mlp)
    - [2.6 Decoder](#26-decoder)
    - [2.7 LM Head](#27-lm-head)
  - [3. Features](#3-features)
    - [3.1 Generative Decoding](#31-generative-decoding)
    - [3.2 Prefill and Decode](#32-prefill-and-decode)
    - [3.3 Multi-Device](#33-multi-device)
    - [3.4 Continuous Batching](#34-continuous-batching)
    - [3.5 vLLM Integration](#34-vllm-integration)
  - [4. Best Practices and Optimizations](#4-best-practices-and-optimizations)
    - [4.1 Tracing](#41-tracing)
    - [4.2 Async Mode](#42-async-mode)
    - [4.3 Multiple CQs](#43-multiple-cqs)
    - [4.4 Op Configs](#44-op-configs)
    - [4.5 Accuracy](#45-accuracy)
    - [4.6 Performance Analysis](#46-performance-analysis)
    - [4.7 Misc. Performance Optimizations](#47-misc-performance-optimizations)
    - [4.8 Module Tests](#48-module-tests)
    - [4.9 Performance Testing](#49-performance-testing)
    - [4.10 Common Pitfalls](#410-common-pitfalls)
      - [4.10.1 Error Messages](#4101-error-messages)
      - [4.10.2 Shard Spec Mismatches](#4102-shard-spec-mismatches)
      - [4.10.3 Ethernet Dispatch Cores](#4103-ethernet-dispatch-cores)
      - [4.10.4 Hangs](#4104-hangs)
        - [4.10.4.1 Tracing](#41041-tracing)
        - [4.10.4.2 Large Matmuls](#41042-large-matmuls)

## 1. Overview
## 2. Modules
### 2.1 Embedding
### 2.2 RoPE
  - Iterative update system
  - When to use our fused op
### 2.3 Norm

Normalization is a critical operation in Large Language Models (LLMs), ensuring stable training and efficient inference. Two widely adopted normalization techniques in modern LLMs, **LayerNorm** and **RMSNorm**, are fully supported in TT-NN.

#### Implementations of Normalization Operations

TT-NN includes two primary implementations of normalization operations to handle diverse activation layouts efficiently:

1. **Non-Distributed Norm**
2. **Distributed Norm**


#### 1. Non-Distributed Norm

This implementation supports both sharded and interleaved inputs. It is employed in the following scenarios:
- **Single-Device Activations**: When the entire embedding resides on a single device.
- **Multi-Device Replicated Activations**: When activation data is replicated across devices in a data-parallel setup.



**Example: RMSNorm on Single Device (Decode Scenario)**

```python
import torch
import ttnn

def torch_rms_norm(x, gamma, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * gamma

batch, seq_len, embedding_dim = 32, 1, 8192
torch_input = torch.randn((batch, seq_len, embedding_dim))
torch_gamma = torch.randn((embedding_dim))
torch_output = torch_rms_norm(torch_input, torch_gamma, eps=1e-5)

# Reshape inputs/weights to 4D tensors
torch_input = torch_input.view(1, 1, batch, embedding_dim)  # seq_len = 1 for decode
torch_gamma = torch_gamma.view(1, 1, 1, embedding_dim)

# Convert tensors to TT-NN tensors
ttnn_input = ttnn.as_tensor(
    torch_input,
    device=device,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG
)
ttnn_gamma = ttnn.as_tensor(
    torch_gamma,
    device=device,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG
)

# Perform RMSNorm
ttnn_output = ttnn.rms_norm(ttnn_input, epsilon=1e-5, weight=ttnn_gamma)
```

**Optimization for Efficient Weight Reads from DRAM**

In above example, weights were traditionally pushed to device in **TILE layout**. But in this case, padding is required to match the TILE_HEIGHT. This padding increased memory footprint and reduced DRAM access efficiency. To address this, weights are now wrapped into **TILE_WIDTH** sticks and converted to **ROW_MAJOR_LAYOUT** without requiring any padding.

```python
# Optimized Weight Layout for DRAM
torch_gamma = torch_gamma.view(1, 1, embedding_dim // TILE_WIDTH, TILE_WIDTH)
ttnn_gamma_rm = ttnn.as_tensor(
    torch_gamma,
    device=device,
    dtype=ttnn.bfloat16,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG
)
```




#### 2. Distributed Norm

The distributed implementation is designed for cases where activations are **sharded along the embedding dimension** across multiple devices. It ensures the correct computation of mean and variance across shards by leveraging cross-device communication. We provide support for both interleaved and width-sharded inputs.

#### Steps to Perform Distributed Normalization on TT-Devices

1. **Compute Local Statistics**
   Each device computes the required statistics (e.g., \(E[x]\), \(E[x^2]\)) locally on its shard of the input tensor.
   - For **RMSNorm**, only \(E[x^2]\) is required.
   - For **LayerNorm**, both \(E[x]\) and \(E[x^2]\) are computed.

   ```python
   tt_distributed_stats = ttnn.rms_norm_pre_all_gather(tt_distributed_input_tensor)
   ```

   - **Output**: A `stats` tensor of shape `[1, 1, batch, TILE_WIDTH * num_stats]`.
   - **Note**:
     - `num_stats=1` for RMSNorm.
     - `num_stats=2` for LayerNorm.
     - Only the first column of the stats tile contains meaningful data; the rest are padding.

2. **Gather Statistics Across Devices**
   The statistics are gathered from all devices along the specified dimension (`dim=3`) and replicated across the device mesh.

   ```python
   tt_gathered_stats = ttnn.all_gather(
       tt_distributed_stats,
       dim=3,
       num_links=1,
       cluster_axis=1,
       mesh_device=mesh_device,
       memory_config=ttnn.DRAM_MEMORY_CONFIG,
       topology=ttnn.Topology.Linear,
   )
   ```

   - **Output**: A tensor of shape `[1, 1, batch, TILE_WIDTH * num_stats * num_devices]`.

3. **Global Normalization**
   The gathered statistics are used to compute the global mean and variance, and normalization is performed on the sharded input.

   ```python
   tt_distributed_output_tensor = ttnn.rms_norm_post_all_gather(
       tt_distributed_input_tensor,
       epsilon=eps,
       weight=tt_distributed_weights,
       program_config=sharded_program_config,
       memory_config=ttnn.DRAM_MEMORY_CONFIG,
       stats=tt_gathered_stats,
   )
   ```
   - **Output**: A tensor of shape `[1, 1, batch, embedding_dim // num_devices]`.


#### Key Notes (Valid for Both Implementations):

- **Interleaved Inputs**:
  For interleaved inputs, the kernel parallelizes work across the sequence length (`seq_len`).
  This makes it highly **optimal for prefill cases**, where the sequence length is large.

- **Width-Sharded Inputs**:
  For width-sharded inputs, the kernel splits the work across the embedding dimension.
  This design is more **optimal for decode cases**, where the sequence length is typically `seq_len=1`.


#### References
- Non-Distributed Norm Op Code [[1]](https://github.com/tenstorrent/tt-metal/tree/main/ttnn/cpp/ttnn/operations/normalization/layernorm) [[2]](https://github.com/tenstorrent/tt-metal/tree/main/ttnn/cpp/ttnn/operations/normalization/rmsnorm)
- Distributed Norm Op Code [[3]](https://github.com/tenstorrent/tt-metal/tree/main/ttnn/cpp/ttnn/operations/normalization/layernorm_distributed) [[4]](https://github.com/tenstorrent/tt-metal/tree/main/ttnn/cpp/ttnn/operations/normalization/rmsnorm_distributed)
- Non-Distributed Norms Unit Tests [[5]](https://github.com/tenstorrent/tt-metal/blob/main/tests/tt_eager/python_api_testing/unit_testing/misc/test_layernorm_sharded.py) [[6]](https://github.com/tenstorrent/tt-metal/blob/main/tests/tt_eager/python_api_testing/unit_testing/misc/test_layernorm.py)
- Distributed Norms Unit Tests [[7]](https://github.com/tenstorrent/tt-metal/blob/main/tests/ttnn/unit_tests/operations/test_distributed_layernorm.py) [[8]](https://github.com/tenstorrent/tt-metal/blob/main/tests/ttnn/unit_tests/operations/test_distributed_layernorm_sharded.py)
- Distributed Norm in LLama3 [[9]](https://github.com/tenstorrent/tt-metal/blob/main/models/demos/llama3/tt/distributed_norm.py)

### 2.4 Attention
  - Flash Attention and Flash Decode
    - general description
    - limitations
    - which dims are parallelized
### 2.5 MLP
### 2.6 Decoder
### 2.7 LM Head
## 3. Features
### 3.1 Generative Decoding
### 3.2 Prefill and Decode
  - submodules, tests
  - how to combine prefill and decode,
  - slicing prefill to fit in L1
### 3.3 Multi-Device
  - device mesh
  - column parallel followed by row parallel
  - sharding, CCL ops, reducing CCL overheads, etc.
### 3.4 Continuous Batching
  - quick intro and how it is implemented in demos.
### 3.5 vLLM Integration
  - Our vLLM repo and what's needed to integrate with it.
## 4. Best Practices and Optimizations
### 4.1 Tracing
  - link to existing doc, why it helps decode more
### 4.2 Async Mode
### 4.3 Multiple CQs
  - how to feed back output to input and read output asyncronously
### 4.4 Op Configs
  - Writing correct program configs and shard specs
  - Deciding how many cores to run an op on
    - Why did we use 16 cores for MLP
  - Which matmul to use when @Colman Glagovich
    - 1d, 2d, dram-sharded, ...
  - Implicitly padding weights in program config for matmuls
### 4.5 Accuracy
  - How we measure it (PCC, perplexity, top-1/top-5, end-user tests, benchmarking)
  - How much PCC is enough? Rules of thumb.
  - Accuracy tests
  - Debugging PCC issues
### 4.6 Performance Analysis
  - Performance tooling, tracy
### 4.7 Misc. Performance Optimizations
  - Which dim to shard matmuls on
  - DRAM-sharding
  - Avoiding sharded to interleaved calls
### 4.8 Module Tests
### 4.9 Performance Testing
### 4.10 Common Pitfalls
#### 4.10.1 Error Messages
  - Running out of L1
  - Shard spec and program config mismatches
  - For some TTNN ops (e.g. ttnn.all_gather) it's not supported to pass -1 in the dim argument.
    - You'll see an error related to op invocation where the arguments don't match
#### 4.10.2 Shard Spec Mismatches
#### 4.10.3 Ethernet Dispatch Cores
  - link to any other description, and mention it is needed for N300 and T3K
#### 4.10.4 Hangs
##### 4.10.4.1 Tracing
  - Host communications cause tracing to hang
  - Running without async mode enabled causes tracing to hang
  - Careful with print in tracing
##### 4.10.4.2 Large Matmuls
  - Large matmuls hanging? Link to appropriate ticket with workaround
  - Issue is being investigated with a workaround of setting the output subblock to 1,1 and grid size to 8x7
