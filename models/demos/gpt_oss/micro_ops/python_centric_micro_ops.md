# Python-Centric Micro Ops Design Pattern

This document describes the Python-centric micro ops design pattern used in `models/demos/deepseek_v3_b1/` and how it differs from the standard ttnn operation pattern. This approach enables rapid prototyping of custom operations with full Python control over kernel orchestration.

## Overview

The Python-centric micro ops pattern provides a way to define custom operations entirely from Python using `ttnn.generic_op()`. Instead of writing C++ operation classes with pybind/nanobind bindings, the entire operation orchestration is defined in Python, with only the kernel code written in C++.

### When to Use This Pattern

- **Rapid prototyping** of new operations before committing to full C++ implementation
- **Fused operations** that combine multiple compute steps in a single program
- **Model-specific optimizations** where standard ops don't meet performance requirements
- **Single-core or specialized multi-core** data movement patterns
- **Custom sharding strategies** that differ from standard ttnn layouts

## Architecture Comparison

### Standard TTNN Operation Pattern

```
Python call (ttnn.relu)
    ↓
Nanobind binding (unary_nanobind.cpp)
    ↓
C++ Executor Template (ExecuteUnary<RELU>::invoke)
    ↓
Device Operation (UnaryDeviceOperation)
    ↓
Program Factory (UnaryProgramFactory::create)
    ↓
Kernels (reader, compute, writer)
    ↓
Hardware Execution
```

**Characteristics:**
- Configuration split across Python bindings, C++ ops, and program factories
- Heavy use of C++ templates for type specialization
- Program caching based on hash computation
- Multiple files per operation (headers, cpp, nanobind, device operation, program factory, kernels)

### Python-Centric Micro Ops Pattern

```
Python call (RMSNormSingleCore.op)
    ↓
Python op() method - constructs ProgramDescriptor
    ↓
ttnn.generic_op(io_tensors, program_descriptor)
    ↓
Kernels (reader, compute, writer)
    ↓
Hardware Execution
```

**Characteristics:**
- All orchestration logic in a single Python file
- Explicit descriptor-based design (kernels, CBs, semaphores)
- Pre-allocated output tensors passed as arguments
- Minimal kernel code (kernels are highly parameterized via compile-time args)

## Core Components

### 1. Operation Class Structure

Each micro op is implemented as a **static class** with two key methods:

```python
class MyOpSingleCore:
    @staticmethod
    def golden(input_tensor, ...):
        """PyTorch reference implementation for validation."""
        # Return expected output using PyTorch
        return torch.some_operation(input_tensor)

    @staticmethod
    def op(input_tensor, output_tensor, ...):
        """Execute operation using generic_op."""
        # 1. Analyze input tensors
        # 2. Create CB descriptors
        # 3. Create kernel descriptors
        # 4. Create program descriptor
        # 5. Call ttnn.generic_op()
        return output
```

**Key Design Points:**
- **Static methods**: No instance state, purely functional
- **Pre-allocated output**: Output tensor is created externally and passed in
- **Golden function**: Provides PyTorch reference for testing

### 2. Circular Buffer (CB) Descriptors

CBs manage data flow between kernels. There are two ways to create them:

```python
# From sharded tensor (input/output backed by L1 memory)
cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(cb_index, tensor)

# Manual creation (for intermediate buffers)
cb_format = ttnn.CBFormatDescriptor(
    buffer_index=cb_index,
    data_format=dtype,
    page_size=tile_size,
    tile=tile_descriptor,
)
cb_descriptor = ttnn.CBDescriptor(
    total_size=num_tiles * tile_size,
    core_ranges=core_grid,
    format_descriptors=[cb_format],
)
```

### 3. Kernel Descriptors

Each kernel (reader, writer, compute) needs a descriptor:

```python
kernel_descriptor = ttnn.KernelDescriptor(
    kernel_source="path/to/kernel.cpp",
    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
    core_ranges=core_grid,
    compile_time_args=[arg1, arg2, ...],  # Fixed at compile time
    runtime_args=[[rt_args]],              # Dynamic per-execution
    config=ttnn.ComputeConfigDescriptor(   # Or ReaderConfigDescriptor/WriterConfigDescriptor
        math_fidelity=ttnn.MathFidelity.LoFi,
        fp32_dest_acc_en=False,
    ),
)
```

**Kernel Types:**
- `ttnn.ReaderConfigDescriptor()` - Data movement kernel (reads data into CBs)
- `ttnn.WriterConfigDescriptor()` - Data movement kernel (writes data from CBs)
- `ttnn.ComputeConfigDescriptor()` - Compute kernel (SFPU/math operations)
- `ttnn.DataMovementConfigDescriptor()` - Advanced data movement (NOC operations)

### 4. Semaphore Descriptors (for multi-core coordination)

```python
semaphore_descriptor = ttnn.SemaphoreDescriptor(
    id=0,  # Unique ID on each core
    core_ranges=all_cores,
    initial_value=0,
)
```

### 5. Program Descriptor and Execution

```python
program_descriptor = ttnn.ProgramDescriptor(
    kernels=[reader_kernel, writer_kernel, compute_kernel],
    cbs=[input_cb, output_cb, intermediate_cb],
    semaphores=[sem1, sem2],  # Optional
)

io_tensors = [input_tensor, output_tensor]
output = ttnn.generic_op(io_tensors, program_descriptor)
```

## Kernel Implementation Patterns

### Reader Kernel (Sharded Input)

For sharded tensors, the reader just signals that data is ready:

```cpp
void kernel_main() {
    constexpr uint32_t input_cb = get_compile_time_arg_val(0);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(1);

    // Signal that input buffer is ready (backed by L1 shard)
    cb_reserve_back(input_cb, num_tiles);
    cb_push_back(input_cb, num_tiles);
}
```

### Reader Kernel (Generating Scalars)

```cpp
void kernel_main() {
    constexpr uint32_t scalars_cb = get_compile_time_arg_val(0);
    uint32_t scalar_value = get_arg_val<uint32_t>(0);  // Runtime arg

    cb_reserve_back(scalars_cb, 1);
    volatile tt_l1_ptr uint16_t* ptr =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(scalars_cb));
    ptr[0] = scalar_value;
    cb_push_back(scalars_cb, 1);
}
```

### Compute Kernel

```cpp
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/eltwise_binary.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t input_cb = get_compile_time_arg_val(0);
    constexpr uint32_t output_cb = get_compile_time_arg_val(1);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(2);

    // Initialize compute APIs
    binary_op_init_common(input_cb, input_cb, output_cb);

    // Wait for input data
    cb_wait_front(input_cb, num_tiles);
    cb_reserve_back(output_cb, num_tiles);

    // Perform computation
    tile_regs_acquire();
    for (uint32_t i = 0; i < num_tiles; i++) {
        // ... compute operations ...
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile_block(0, output_cb, num_tiles);
    tile_regs_release();

    // Signal output ready
    cb_pop_front(input_cb, num_tiles);
    cb_push_back(output_cb, num_tiles);
}
}
```

### Writer Kernel (Sharded Output)

For sharded outputs, the writer just waits for compute to finish:

```cpp
void kernel_main() {
    constexpr uint32_t output_cb = get_compile_time_arg_val(0);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(1);

    // Wait for output to be ready, then signal completion
    cb_wait_front(output_cb, num_tiles);
}
```

## Data Movement Operations

### Gather (Multi-Core to Single-Core)

```python
class GatherSingleCore:
    @staticmethod
    def op(input_tensor, output_tensor, noc=None):
        # Get core grids
        input_core_grid = input_tensor.memory_config().shard_spec.grid
        output_core_grid = output_tensor.memory_config().shard_spec.grid
        gather_core = output_core_grid.ranges()[0].start

        # Optimize NOC routing per core
        for core in input_cores:
            noc0_hop = device.get_worker_noc_hop_distance(core, gather_core, ttnn.NOC.NOC_0)
            noc1_hop = device.get_worker_noc_hop_distance(core, gather_core, ttnn.NOC.NOC_1)
            # Use shorter path

        # Create sender kernels (one per NOC group)
        # Create receiver kernel
        # Use semaphores for synchronization
```

### Multicast (Single-Core to Multi-Core)

```python
class McastSingleCore:
    @staticmethod
    def op(input_tensor, output_tensor, noc=ttnn.NOC.NOC_1):
        # Named compile-time args for clarity
        sender_named_compile_args = [
            ("mcast_dest_noc_start_x", start_x),
            ("mcast_dest_noc_start_y", start_y),
            ("mcast_dest_noc_end_x", end_x),
            ("mcast_dest_noc_end_y", end_y),
            ("mcast_num_cores", num_cores),
            ("mcast_loopback", 1 if loopback else 0),
            # ...
        ]

        kernel_descriptor = ttnn.KernelDescriptor(
            named_compile_time_args=sender_named_compile_args,
            # ...
        )
```

## Testing Pattern

```python
import pytest
import torch
import ttnn
from models.common.utility_functions import comp_pcc

@pytest.mark.parametrize("width", [7168, 1536, 512])
@pytest.mark.parametrize("use_fp32", [True, False])
def test_my_op(device, width, use_fp32):
    # 1. Create PyTorch reference
    torch_input = torch.randn((1, width), dtype=torch.bfloat16)
    torch_expected = MyOpSingleCore.golden(torch_input)

    # 2. Create sharded TTNN tensors
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        (1, width),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        shard_spec
    )

    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16,
                                  layout=ttnn.TILE_LAYOUT, device=device,
                                  memory_config=mem_config)
    ttnn_output = ttnn.from_torch(torch.zeros_like(torch_input), ...)

    # 3. Execute op
    result = MyOpSingleCore.op(ttnn_input, ttnn_output, fp32_dest_acc_en=use_fp32)

    # 4. Verify with PCC (Pearson Correlation Coefficient)
    output_torch = ttnn.to_torch(result)
    passing, pcc_message = comp_pcc(torch_expected, output_torch, 0.98)
    assert passing, pcc_message
```

## Fused Operations Design

Fused ops combine multiple compute steps in a single program, avoiding intermediate memory writes.

### Example: RMSNorm Fusion

The RMSNorm kernel fuses these steps:
1. Square input: `x^2`
2. Reduce (sum): `sum(x^2)`
3. Divide by count: `mean = sum / N`
4. Add epsilon: `mean + eps`
5. Rsqrt: `1 / sqrt(mean + eps)`
6. Multiply input: `x * rsqrt`
7. Multiply by gamma: `result * gamma`

```cpp
// All operations in single compute kernel, using intermediate CBs
template <...>
void compute_rmsnorm() {
    // Step 1-2: Square and reduce
    mul_tiles_init(input_cb, input_cb);
    // ... square tiles to interm_cb ...
    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_SCALAR, fp32_acc>(...);

    // Step 3-5: Add epsilon and rsqrt
    binary_dest_reuse_tiles<ELWADD, ...>(scalars_cb, epsilon_index, 0);
    rsqrt_tile<false, rsqrt_fast_approx>(0);

    // Step 6: Multiply by normalized factor
    mul_tiles_bcast_scalar(...);

    // Step 7: Multiply by gamma
    binary_dest_reuse_tiles<ELWMUL, ...>(weight_cb, i, i);
}
```

### Tips for Fused Op Design

1. **Plan CB allocation**: Each intermediate result needs a CB
2. **Use `dest_reuse`**: Avoid spilling intermediate results when possible
3. **Minimize CB memory**: Share CBs between non-overlapping stages
4. **Profile tile-by-tile**: Some operations can be pipelined

## File Structure

```
micro_ops/
├── my_op/
│   ├── op.py              # Python op definition
│   └── kernels/
│       ├── my_op_reader.cpp
│       ├── my_op_compute.cpp
│       └── my_op_writer.cpp
tests/
├── test_my_op.py          # Pytest tests
```

## Key Differences from Standard TTNN Ops

| Aspect | Standard TTNN | Python-Centric Micro Ops |
|--------|---------------|-------------------------|
| Configuration | Split across C++/Python | All in Python |
| Kernel parameterization | Preprocessor defines | Compile-time args |
| Output tensor | Created by operation | Pre-allocated, passed in |
| Program caching | Automatic hash-based | Manual (if needed) |
| Code location | ttnn/cpp/ttnn/operations/ | models/demos/.../micro_ops/ |
| Binding layer | Nanobind templates | None (direct ttnn.generic_op) |
| Flexibility | High for standard patterns | Maximum for custom patterns |
| Development speed | Slower (C++ compile) | Faster (Python iteration) |

## Migration Path

Once a micro op is validated and stable:
1. Profile performance characteristics
2. If performance-critical, consider migrating to standard C++ op pattern
3. The kernel code can often be reused directly
4. Add proper program caching and validation

## Reference Implementation

See `models/demos/deepseek_v3_b1/micro_ops/` for complete examples:
- `rmsnorm/`: Fused normalization with compute kernel
- `matmul/`: Single-core matrix multiplication
- `gather/`: Multi-core to single-core data collection
- `mcast/`: Single-core to multi-core broadcast
