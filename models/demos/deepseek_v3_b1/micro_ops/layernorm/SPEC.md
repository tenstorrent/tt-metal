# LayerNorm Generic Op Specification

## Overview

This document specifies the implementation of a **row-wise Layer Normalization** operation using the generic_op infrastructure. The operation takes row-major interleaved tensors and performs standardization followed by affine transformation.

### Mathematical Definition

For an input tensor `x` with final dimension `W`:

```
# For each row independently:
mean = sum(x) / W
variance = sum((x - mean)^2) / W
stddev_inv = 1 / sqrt(variance + epsilon)

# Standardization + Affine transform:
output = (x - mean) * stddev_inv * gamma + beta
```

Where:
- `gamma` and `beta` are learnable parameters with shape `[1, 1, ..., W]`
- `epsilon` is a small constant (default: 1e-6) to prevent division by zero

### Requirements

| Requirement | Description |
|-------------|-------------|
| Input tensor | Row-major, interleaved, DRAM-resident |
| Gamma tensor | Row-major, shape `[..., W]` where W = input final dimension |
| Beta tensor | Row-major, shape `[..., W]` where W = input final dimension |
| Output tensor | Row-major, same shape as input |
| Core utilization | Single-core implementation |
| Memory layout | DRAM-to-DRAM |
| Infrastructure | Generic op (Python program setup, C++ kernels) |
| Tilization | Gamma/beta tilized once by kernels |

---

## High-Level Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                            SINGLE TENSIX CORE                               │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐     ┌─────────────────────────────────────────────────┐   │
│  │   READER    │     │                    COMPUTE                      │   │
│  │  (NCRISC)   │     │                    (TRISC)                      │   │
│  │             │     │                                                 │   │
│  │ 1. Read X   │     │ 1. Tilize X (row-major → tiles)                 │   │
│  │    (sticks) │────►│ 2. Compute mean per row                         │   │
│  │             │     │ 3. Compute variance per row                     │   │
│  │ 2. Read γ   │────►│ 4. Compute rsqrt(var + ε)                       │   │
│  │    (sticks) │     │ 5. Standardize: (X - mean) * rsqrt              │   │
│  │             │     │ 6. Tilize γ, β (once)                           │   │
│  │ 3. Read β   │────►│ 7. Apply affine: result * γ + β                 │   │
│  │    (sticks) │     │ 8. Untilize output (tiles → row-major)          │   │
│  │             │     │                                                 │   │
│  └─────────────┘     └────────────────────────────────────┬────────────┘   │
│                                                           │                 │
│                      ┌─────────────┐                      │                 │
│                      │   WRITER    │◄─────────────────────┘                 │
│                      │  (BRISC)    │                                        │
│                      │             │                                        │
│                      │ Write Y     │                                        │
│                      │ (sticks)    │                                        │
│                      └─────────────┘                                        │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Stages

### Stage 1: Generic Op Python Setup

**Goal**: Create the Python-side program descriptor infrastructure.

#### Step 1.1: Basic Infrastructure Setup
- [ ] Create directory structure: `models/demos/deepseek_v3_b1/micro_ops/layernorm/`
- [ ] Create `__init__.py`
- [ ] Create `op.py` with `LayerNormSingleCore` class skeleton

**Test**: Import succeeds, class instantiates without errors.

#### Step 1.2: Golden Reference Implementation
- [ ] Implement `LayerNormSingleCore.golden()` method
- [ ] Takes PyTorch tensors, returns expected output

**Test**: Compare golden output with `torch.nn.functional.layer_norm()`.

#### Step 1.3: Circular Buffer Configuration
- [ ] Calculate page sizes for row-major data (stick size = W * sizeof(dtype))
- [ ] Define CB indices:
  - `cb_input_rm`: Input row-major data (CB 0)
  - `cb_input_tiled`: Input after tilization (CB 1)
  - `cb_gamma_rm`: Gamma row-major (CB 2)
  - `cb_gamma_tiled`: Gamma after tilization (CB 3)
  - `cb_beta_rm`: Beta row-major (CB 4)
  - `cb_beta_tiled`: Beta after tilization (CB 5)
  - `cb_scalars`: Scalar values (mean, variance, etc.) (CB 6)
  - `cb_interm`: Intermediate computations (CB 7)
  - `cb_output_tiled`: Output in tile format (CB 16)
  - `cb_output_rm`: Output row-major (CB 17)
- [ ] Create `CBDescriptor` objects for each buffer

**Test**: CBDescriptors created without errors, sizes are correct.

#### Step 1.4: Kernel Descriptors
- [ ] Create reader kernel descriptor (NCRISC)
- [ ] Create compute kernel descriptor (TRISC)
- [ ] Create writer kernel descriptor (BRISC)
- [ ] Set up compile-time and runtime args

**Test**: KernelDescriptors created, program descriptor builds.

#### Step 1.5: Program Descriptor Assembly
- [ ] Combine kernels and CBs into `ProgramDescriptor`
- [ ] Handle `TensorAccessorArgs` for all tensors

**Test**: `ttnn.generic_op()` executes without kernel compilation errors.

---

### Stage 2: Kernel Implementation

**Goal**: Implement the three kernels (reader, compute, writer).

#### Step 2.1: Reader Kernel (NCRISC)

**Purpose**: Read row-major data from DRAM into circular buffers.

```
Reader flow:
1. Read input tensor (row-major sticks) → cb_input_rm
2. Read gamma tensor (row-major sticks) → cb_gamma_rm (ONCE)
3. Read beta tensor (row-major sticks) → cb_beta_rm (ONCE)
4. Generate scalar constants (epsilon, 1/W) → cb_scalars
```

**Sub-steps**:
- [ ] 2.1.1: Basic reader that reads input sticks to CB
- [ ] 2.1.2: Add gamma/beta reading (single pass)
- [ ] 2.1.3: Add scalar generation (epsilon, reduction scalar)

**Test 2.1.1**: Reader produces data in CB, writer can see it (passthrough test).
**Test 2.1.2**: Gamma/beta data correctly in CBs.
**Test 2.1.3**: Scalars available for compute.

#### Step 2.2: Compute Kernel (TRISC)

**Purpose**: Perform tilization, normalization, and untilization.

```
Compute flow:
1. Wait for input sticks → Tilize → cb_input_tiled
2. Wait for gamma sticks → Tilize → cb_gamma_tiled (ONCE)
3. Wait for beta sticks → Tilize → cb_beta_tiled (ONCE)
4. For each tile row:
   a. Compute row sum → mean = sum / W
   b. Compute (x - mean)^2 → sum → variance = sum / W
   c. Compute rsqrt(variance + epsilon)
   d. Compute (x - mean) * rsqrt
   e. Apply gamma * result + beta
   f. Untilize → cb_output_rm
5. Push output to writer
```

**Sub-steps**:
- [ ] 2.2.1: Passthrough - tilize input, untilize output (no computation)
- [ ] 2.2.2: Add mean computation (reduce sum, scale)
- [ ] 2.2.3: Add variance computation
- [ ] 2.2.4: Add rsqrt(var + eps) computation
- [ ] 2.2.5: Add standardization (x - mean) * rsqrt
- [ ] 2.2.6: Add gamma multiplication
- [ ] 2.2.7: Add beta addition

**Test 2.2.1**: Tilize → untilize produces identical output.
**Test 2.2.2**: Mean computation matches torch.mean().
**Test 2.2.3**: Variance computation matches torch.var(unbiased=False).
**Test 2.2.4**: rsqrt output matches torch.rsqrt().
**Test 2.2.5**: Standardized output matches (x - mean) / std.
**Test 2.2.6**: With gamma=1, output unchanged from 2.2.5.
**Test 2.2.7**: Full layer_norm matches golden.

#### Step 2.3: Writer Kernel (BRISC)

**Purpose**: Write row-major output from CB to DRAM.

```
Writer flow:
1. Wait for output sticks in cb_output_rm
2. Write to output tensor DRAM location
3. Pop consumed data
```

**Sub-steps**:
- [ ] 2.3.1: Basic writer that writes sticks to DRAM

**Test 2.3.1**: Data written to DRAM matches CB contents.

---

## Circular Buffer Layout

### Memory Layout (Single Core)

```
L1 SRAM Layout:
┌─────────────────────────────────────────────────────────────┐
│ CB 0  (cb_input_rm)     │ Input sticks (row-major)         │
│       │ Size: 2 * stick_size (double buffer)               │
├─────────────────────────────────────────────────────────────┤
│ CB 1  (cb_input_tiled)  │ Tilized input                    │
│       │ Size: tiles_per_row * tile_size                    │
├─────────────────────────────────────────────────────────────┤
│ CB 2  (cb_gamma_rm)     │ Gamma sticks (row-major)         │
│       │ Size: W * sizeof(dtype) (single buffer)            │
├─────────────────────────────────────────────────────────────┤
│ CB 3  (cb_gamma_tiled)  │ Tilized gamma                    │
│       │ Size: tiles_per_row * tile_size                    │
├─────────────────────────────────────────────────────────────┤
│ CB 4  (cb_beta_rm)      │ Beta sticks (row-major)          │
│       │ Size: W * sizeof(dtype) (single buffer)            │
├─────────────────────────────────────────────────────────────┤
│ CB 5  (cb_beta_tiled)   │ Tilized beta                     │
│       │ Size: tiles_per_row * tile_size                    │
├─────────────────────────────────────────────────────────────┤
│ CB 6  (cb_scalars)      │ Scalar tile (eps, 1/W)           │
│       │ Size: 1 * tile_size                                │
├─────────────────────────────────────────────────────────────┤
│ CB 7  (cb_interm)       │ Intermediate results             │
│       │ Size: tiles_per_row * tile_size                    │
├─────────────────────────────────────────────────────────────┤
│ CB 16 (cb_output_tiled) │ Output tiles (before untilize)   │
│       │ Size: tiles_per_row * tile_size                    │
├─────────────────────────────────────────────────────────────┤
│ CB 17 (cb_output_rm)    │ Output sticks (row-major)        │
│       │ Size: 2 * stick_size (double buffer)               │
└─────────────────────────────────────────────────────────────┘
```

### Size Calculations

```python
# Dimensions
W = input_shape[-1]  # Final dimension (normalization dimension)
num_rows = prod(input_shape[:-1])  # Total rows to normalize
tiles_per_row = (W + 31) // 32  # Tiles per normalization row

# Sizes
stick_size = W * element_size  # Row-major stick size (align to 32 bytes)
tile_size = 32 * 32 * element_size  # 32x32 tile
```

---

## Data Flow Diagram

```
                            DRAM
                              │
        ┌─────────────────────┴─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
   ┌─────────┐          ┌─────────┐          ┌─────────┐
   │ Input X │          │ Gamma γ │          │ Beta β  │
   │(RM,Intl)│          │(RM,Intl)│          │(RM,Intl)│
   └────┬────┘          └────┬────┘          └────┬────┘
        │                    │                    │
        │ Reader (NCRISC)    │                    │
        │ noc_async_read     │                    │
        ▼                    ▼                    ▼
   ┌─────────┐          ┌─────────┐          ┌─────────┐
   │cb_in_rm │          │cb_γ_rm  │          │cb_β_rm  │
   │(sticks) │          │(sticks) │          │(sticks) │
   └────┬────┘          └────┬────┘          └────┬────┘
        │                    │                    │
        │ Compute (TRISC)    │ (once)             │ (once)
        │ tilize_block()     │ tilize_block()     │ tilize_block()
        ▼                    ▼                    ▼
   ┌─────────┐          ┌─────────┐          ┌─────────┐
   │cb_in_ti │          │cb_γ_ti  │          │cb_β_ti  │
   │(tiles)  │          │(tiles)  │          │(tiles)  │
   └────┬────┘          └────┬────┘          └────┬────┘
        │                    │                    │
        │    ┌───────────────┴────────────────────┘
        │    │
        ▼    ▼
   ┌───────────────────────────────────────┐
   │         LAYER NORM COMPUTE            │
   │  1. mean = reduce_sum(x) * (1/W)      │
   │  2. var = reduce_sum((x-mean)²) * 1/W │
   │  3. rstd = rsqrt(var + ε)             │
   │  4. y = (x - mean) * rstd             │
   │  5. out = y * γ + β                   │
   └───────────────┬───────────────────────┘
                   │
                   ▼
            ┌─────────────┐
            │cb_out_tiled │
            │  (tiles)    │
            └──────┬──────┘
                   │
                   │ Compute (TRISC)
                   │ untilize_block()
                   ▼
            ┌─────────────┐
            │ cb_out_rm   │
            │ (sticks)    │
            └──────┬──────┘
                   │
                   │ Writer (BRISC)
                   │ noc_async_write
                   ▼
              ┌─────────┐
              │Output Y │
              │(RM,Intl)│
              └─────────┘
                   │
                   ▼
                 DRAM
```

---

## Testing Strategy

### Unit Tests (Incremental)

| Test ID | Stage | Description | Pass Criteria |
|---------|-------|-------------|---------------|
| T1.1 | 1.1 | Import and instantiate | No errors |
| T1.2 | 1.2 | Golden vs torch.layer_norm | PCC > 0.9999 |
| T1.3 | 1.3 | CB descriptors valid | No errors |
| T1.4 | 1.4 | Kernel descriptors valid | No errors |
| T1.5 | 1.5 | Program executes | No kernel errors |
| T2.1.1 | 2.1.1 | Reader passthrough | Data in CB |
| T2.2.1 | 2.2.1 | Tilize → untilize | Identity |
| T2.2.2 | 2.2.2 | Mean computation | PCC > 0.999 |
| T2.2.3 | 2.2.3 | Variance computation | PCC > 0.999 |
| T2.2.4 | 2.2.4 | rsqrt computation | PCC > 0.99 |
| T2.2.5 | 2.2.5 | Standardization | PCC > 0.99 |
| T2.2.6 | 2.2.6 | Gamma multiply | PCC > 0.99 |
| T2.2.7 | 2.2.7 | Full layer_norm | PCC > 0.99 |

### Test Shapes

```python
# Minimal test cases
test_shapes = [
    ([1, 32], "Single row, single tile width"),
    ([1, 64], "Single row, two tiles"),
    ([32, 32], "32 rows, single tile width"),
    ([4, 128], "4 rows, 4 tiles"),
    ([1, 1, 256], "3D input"),
    ([2, 2, 512], "Batch of 4 sequences"),
]
```

---

## File Structure

```
models/demos/deepseek_v3_b1/micro_ops/layernorm/
├── __init__.py
├── op.py                           # Python program descriptor
├── SPEC.md                         # This specification
├── kernels/
│   └── layernorm_kernel.cpp        # Unified kernel (reader/compute/writer)
└── tests/
    └── test_layernorm.py           # Unit tests
```

---

## Key APIs Used

### Python Side (generic_op)

```python
# Core APIs
ttnn.ProgramDescriptor(kernels, cbs, semaphores)
ttnn.KernelDescriptor(kernel_source, core_ranges, compile_time_args, runtime_args, config)
ttnn.CBDescriptor(total_size, core_ranges, format_descriptors)
ttnn.CBFormatDescriptor(buffer_index, data_format, page_size)
ttnn.RuntimeArgs()
ttnn.TensorAccessorArgs(tensor).get_compile_time_args()
ttnn.generic_op(io_tensors, program_descriptor)
ttnn.allocate_tensor_on_device(shape, dtype, layout, device, memory_config)
```

### C++ Kernel Side

```cpp
// Dataflow APIs (reader/writer)
noc_async_read_page(page_id, tensor_accessor, l1_addr)
noc_async_write_page(page_id, tensor_accessor, l1_addr)
noc_async_read_barrier()
noc_async_write_barrier()
cb_reserve_back(cb_id, num_pages)
cb_push_back(cb_id, num_pages)
cb_wait_front(cb_id, num_pages)
cb_pop_front(cb_id, num_pages)

// Compute APIs
tilize_init(in_cb, tiles_per_row, out_cb)
tilize_block(in_cb, tiles_per_row, out_cb)
tilize_uninit(in_cb, out_cb)
untilize_init(in_cb)
untilize_block(in_cb, tiles_per_row, out_cb)
untilize_uninit(in_cb)

// Reduction APIs
reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(in_cb, scalar_cb)
reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(in_cb, scalar_cb, tile_idx, dst_idx)

// Math APIs
mul_tiles(cb_a, cb_b, a_idx, b_idx, dst_idx)
add_tiles(cb_a, cb_b, a_idx, b_idx, dst_idx)
sub_tiles(cb_a, cb_b, a_idx, b_idx, dst_idx)
rsqrt_tile(dst_idx)
bcast_scalar_tile(cb, tile_idx, dst_idx)
```

---

## Dependencies

- `ttnn.generic_op` infrastructure
- Kernel helper library:
  - `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp`
  - `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp`
- Existing reader/writer patterns from `ttnn/cpp/ttnn/operations/data_movement/`

---

## Notes

1. **Single-core limitation**: This initial implementation uses a single core. Multi-core parallelization over rows can be added later.

2. **Row-major constraint**: Input/output must be row-major. The tilize/untilize operations handle format conversion internally.

3. **Gamma/beta caching**: These tensors are read and tilized once at the start, then reused for all rows.

4. **Memory pressure**: For large W, intermediate buffers may need streaming rather than holding all tiles simultaneously.

5. **Precision**: Consider using FP32 accumulation for reduce operations to maintain numerical stability.
