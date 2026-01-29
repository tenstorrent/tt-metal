# LayerNorm Implementation Plan

## Overview

This is a step-by-step implementation checklist for the LayerNorm generic op. Each step must be completed and verified before moving to the next. Check off items only when fully complete.

---

## Stage 1: Generic Op Python Setup

### Step 1.1: Basic Infrastructure Setup

- [x] **1.1.1** Create directory structure
  - Create `models/demos/deepseek_v3_b1/micro_ops/layernorm/`
  - Create `models/demos/deepseek_v3_b1/micro_ops/layernorm/kernels/`
  - Create `models/demos/deepseek_v3_b1/micro_ops/layernorm/tests/`

- [x] **1.1.2** Create `__init__.py`
  - File: `models/demos/deepseek_v3_b1/micro_ops/layernorm/__init__.py`
  - Content: Empty or minimal exports

- [x] **1.1.3** Create `op.py` with class skeleton
  - File: `models/demos/deepseek_v3_b1/micro_ops/layernorm/op.py`
  - Requirements:
    - Define `LayerNormSingleCore` class
    - Add `@staticmethod golden(input_tensor, gamma_tensor, beta_tensor, epsilon=1e-6)` stub
    - Add `@staticmethod op(input_tensor, gamma_tensor, beta_tensor, output_tensor, epsilon=1e-6)` stub
  - Verification: `from models.demos.deepseek_v3_b1.micro_ops.layernorm.op import LayerNormSingleCore` succeeds

**Step 1.1 Complete:** [x]

---

### Step 1.2: Golden Reference Implementation

- [x] **1.2.1** Implement `golden()` method
  - File: `models/demos/deepseek_v3_b1/micro_ops/layernorm/op.py`
  - Requirements:
    - Input: `input_tensor` (torch.Tensor), `gamma_tensor` (torch.Tensor), `beta_tensor` (torch.Tensor), `epsilon` (float)
    - Compute: `mean = input.mean(dim=-1, keepdim=True)`
    - Compute: `var = input.var(dim=-1, unbiased=False, keepdim=True)`
    - Compute: `normalized = (input - mean) / sqrt(var + epsilon)`
    - Compute: `output = normalized * gamma + beta`
    - Return: output tensor

- [x] **1.2.2** Create golden validation test
  - File: `models/demos/deepseek_v3_b1/micro_ops/layernorm/tests/test_layernorm.py`
  - Requirements:
    - Test function: `test_golden_vs_torch()`
    - Compare `LayerNormSingleCore.golden()` output with `torch.nn.functional.layer_norm()`
    - Test shapes: `[1, 32]`, `[1, 64]`, `[4, 128]`
    - Pass criteria: `torch.allclose(golden, torch_ref, rtol=1e-4, atol=1e-4)` for all shapes
  - Verification: `pytest tests/test_layernorm.py::test_golden_vs_torch -v` passes

**Step 1.2 Complete:** [x]

---

### Step 1.3: Circular Buffer Configuration

- [x] **1.3.1** Define CB indices as constants
  - File: `models/demos/deepseek_v3_b1/micro_ops/layernorm/op.py`
  - Requirements:
    ```python
    # CB indices
    CB_INPUT_RM = 0       # Input row-major sticks from DRAM
    CB_INPUT_TILED = 1    # Input after tilization
    CB_GAMMA_RM = 2       # Gamma row-major sticks
    CB_GAMMA_TILED = 3    # Gamma after tilization
    CB_BETA_RM = 4        # Beta row-major sticks
    CB_BETA_TILED = 5     # Beta after tilization
    CB_SCALARS = 6        # Scalar values (epsilon, 1/W)
    CB_INTERM = 7         # Intermediate computation results
    CB_OUTPUT_TILED = 16  # Output in tile format
    CB_OUTPUT_RM = 17     # Output row-major sticks
    ```

- [x] **1.3.2** Implement size calculation helper function
  - File: `models/demos/deepseek_v3_b1/micro_ops/layernorm/op.py`
  - Requirements:
    - Function: `_calculate_sizes(input_shape, dtype)`
    - Calculate `W` (final dimension)
    - Calculate `num_rows` (product of all dims except last)
    - Calculate `tiles_per_row = (W + 31) // 32`
    - Calculate `stick_size = W * element_size` (aligned to 32 bytes)
    - Calculate `tile_size` based on dtype (e.g., 2048 for bfloat16)
    - Return dict with all sizes

- [x] **1.3.3** Implement CB descriptor creation function
  - File: `models/demos/deepseek_v3_b1/micro_ops/layernorm/op.py`
  - Requirements:
    - Function: `_create_cb_descriptors(core_grid, sizes, dtype)`
    - Create `CBDescriptor` for each of the 10 CBs
    - CB_INPUT_RM: `total_size = 2 * stick_size` (double buffer)
    - CB_INPUT_TILED: `total_size = tiles_per_row * tile_size`
    - CB_GAMMA_RM: `total_size = stick_size` (single buffer, read once)
    - CB_GAMMA_TILED: `total_size = tiles_per_row * tile_size`
    - CB_BETA_RM: `total_size = stick_size` (single buffer, read once)
    - CB_BETA_TILED: `total_size = tiles_per_row * tile_size`
    - CB_SCALARS: `total_size = tile_size` (1 tile)
    - CB_INTERM: `total_size = tiles_per_row * tile_size`
    - CB_OUTPUT_TILED: `total_size = tiles_per_row * tile_size`
    - CB_OUTPUT_RM: `total_size = 2 * stick_size` (double buffer)
    - Return list of all CB descriptors

- [x] **1.3.4** Add CB configuration unit test
  - File: `models/demos/deepseek_v3_b1/micro_ops/layernorm/tests/test_layernorm.py`
  - Requirements:
    - Test function: `test_cb_configuration()`
    - Verify all CBs are created without errors
    - Verify sizes are positive and properly aligned
  - Verification: `pytest tests/test_layernorm.py::test_cb_configuration -v` passes

**Step 1.3 Complete:** [x]

---

### Step 1.4: Kernel Descriptors

- [ ] **1.4.1** Create placeholder kernel files
  - File: `models/demos/deepseek_v3_b1/micro_ops/layernorm/kernels/reader.cpp`
    - Minimal reader that does nothing (just `void kernel_main() {}`)
  - File: `models/demos/deepseek_v3_b1/micro_ops/layernorm/kernels/compute.cpp`
    - Minimal compute that does nothing
  - File: `models/demos/deepseek_v3_b1/micro_ops/layernorm/kernels/writer.cpp`
    - Minimal writer that does nothing

- [ ] **1.4.2** Implement reader kernel descriptor creation
  - File: `models/demos/deepseek_v3_b1/micro_ops/layernorm/op.py`
  - Requirements:
    - Function: `_create_reader_descriptor(core_grid, input_tensor, gamma_tensor, beta_tensor, sizes)`
    - Compile-time args: CB indices, page sizes, TensorAccessorArgs for all 3 input tensors
    - Runtime args: buffer addresses, num_pages, start_ids for each tensor
    - Config: `ttnn.ReaderConfigDescriptor()`
    - Return `KernelDescriptor`

- [ ] **1.4.3** Implement compute kernel descriptor creation
  - File: `models/demos/deepseek_v3_b1/micro_ops/layernorm/op.py`
  - Requirements:
    - Function: `_create_compute_descriptor(core_grid, sizes, epsilon)`
    - Compile-time args: CB indices, tiles_per_row, num_rows
    - Runtime args: epsilon (packed as uint32)
    - Config: `ttnn.ComputeConfigDescriptor()`
    - Return `KernelDescriptor`

- [ ] **1.4.4** Implement writer kernel descriptor creation
  - File: `models/demos/deepseek_v3_b1/micro_ops/layernorm/op.py`
  - Requirements:
    - Function: `_create_writer_descriptor(core_grid, output_tensor, sizes)`
    - Compile-time args: CB indices, page sizes, TensorAccessorArgs for output
    - Runtime args: buffer address, num_pages, start_id
    - Config: `ttnn.WriterConfigDescriptor()`
    - Return `KernelDescriptor`

- [ ] **1.4.5** Add kernel descriptor unit test
  - File: `models/demos/deepseek_v3_b1/micro_ops/layernorm/tests/test_layernorm.py`
  - Requirements:
    - Test function: `test_kernel_descriptors(device)`
    - Create dummy tensors on device
    - Verify all kernel descriptors are created without errors
    - Verify compile-time and runtime args are populated
  - Verification: `pytest tests/test_layernorm.py::test_kernel_descriptors -v` passes

**Step 1.4 Complete:** [ ]

---

### Step 1.5: Program Descriptor Assembly

- [ ] **1.5.1** Implement full `op()` method
  - File: `models/demos/deepseek_v3_b1/micro_ops/layernorm/op.py`
  - Requirements:
    - Validate input shapes (gamma/beta shape matches input[-1])
    - Calculate all sizes using `_calculate_sizes()`
    - Create core grid (single core: `CoreCoord(0, 0)`)
    - Create all CB descriptors using `_create_cb_descriptors()`
    - Create all kernel descriptors
    - Assemble `ProgramDescriptor(kernels=[reader, compute, writer], cbs=[...])`
    - Call `ttnn.generic_op([input_tensor, gamma_tensor, beta_tensor, output_tensor], program_descriptor)`
    - Return output tensor

- [ ] **1.5.2** Add program execution test (placeholder kernels)
  - File: `models/demos/deepseek_v3_b1/micro_ops/layernorm/tests/test_layernorm.py`
  - Requirements:
    - Test function: `test_program_executes(device)`
    - Create input, gamma, beta tensors on device (row-major, DRAM)
    - Allocate output tensor on device
    - Call `LayerNormSingleCore.op()`
    - Pass criteria: No exceptions, no kernel compilation errors
  - Verification: `pytest tests/test_layernorm.py::test_program_executes -v` passes

**Step 1.5 Complete:** [ ]

---

## Stage 2: Kernel Implementation

### Step 2.1: Reader Kernel (NCRISC)

- [ ] **2.1.1** Implement basic input reading
  - File: `models/demos/deepseek_v3_b1/micro_ops/layernorm/kernels/reader.cpp`
  - Requirements:
    - Read compile-time args: CB indices, page sizes, TensorAccessorArgs
    - Read runtime args: input buffer address, num_sticks, start_stick_id
    - Loop over sticks:
      - `cb_reserve_back(CB_INPUT_RM, 1)`
      - `noc_async_read_page(stick_id, tensor_accessor, l1_addr)`
      - `noc_async_read_barrier()`
      - `cb_push_back(CB_INPUT_RM, 1)`
  - Test: Create passthrough test where reader reads data, writer writes same data
  - Verification: Input data appears in output buffer

- [ ] **2.1.2** Add gamma reading (single pass)
  - File: `models/demos/deepseek_v3_b1/micro_ops/layernorm/kernels/reader.cpp`
  - Requirements:
    - Read gamma tensor once at kernel start
    - Read all gamma sticks into CB_GAMMA_RM
    - Push all gamma pages
    - Do NOT pop (compute will use and pop)
  - Test: Verify gamma data is in CB_GAMMA_RM
  - Verification: Gamma data accessible by compute kernel

- [ ] **2.1.3** Add beta reading (single pass)
  - File: `models/demos/deepseek_v3_b1/micro_ops/layernorm/kernels/reader.cpp`
  - Requirements:
    - Read beta tensor once at kernel start (after gamma)
    - Read all beta sticks into CB_BETA_RM
    - Push all beta pages
  - Test: Verify beta data is in CB_BETA_RM
  - Verification: Beta data accessible by compute kernel

- [ ] **2.1.4** Add scalar generation
  - File: `models/demos/deepseek_v3_b1/micro_ops/layernorm/kernels/reader.cpp`
  - Requirements:
    - Generate scalar tile with `1/W` value (for mean/variance calculation)
    - Use `generate_reduce_scaler()` or manual tile fill
    - Push to CB_SCALARS
  - Verification: Scalar tile available for compute reduction

**Step 2.1 Complete:** [ ]

---

### Step 2.2: Compute Kernel (TRISC)

- [ ] **2.2.1** Implement tilize/untilize passthrough
  - File: `models/demos/deepseek_v3_b1/micro_ops/layernorm/kernels/compute.cpp`
  - Requirements:
    - Include tilize/untilize helpers
    - Call `compute_kernel_hw_startup()`
    - Wait for input sticks in CB_INPUT_RM
    - Tilize: `tilize<TilizeConfig<InputCB<CB_INPUT_RM>, OutputCB<CB_INPUT_TILED>>>(tiles_per_row, 1)`
    - Copy tiles from CB_INPUT_TILED to CB_OUTPUT_TILED (simple copy)
    - Untilize: `untilize<UntilizeConfig<WidthInTiles<tiles_per_row>, InputCB<CB_OUTPUT_TILED>, OutputCB<CB_OUTPUT_RM>>>(1)`
    - Push to CB_OUTPUT_RM
  - Test function: `test_tilize_untilize_passthrough(device)`
  - Pass criteria: `torch.allclose(input, output, rtol=1e-3, atol=1e-3)`
  - Verification: `pytest tests/test_layernorm.py::test_tilize_untilize_passthrough -v` passes

- [ ] **2.2.2** Implement mean computation
  - File: `models/demos/deepseek_v3_b1/micro_ops/layernorm/kernels/compute.cpp`
  - Requirements:
    - After tilize, compute row-wise sum using `reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>`
    - Multiply by scalar (1/W) to get mean
    - Store mean in intermediate buffer
  - Test function: `test_mean_computation(device)`
  - Pass criteria: Compare computed mean with `torch.mean(input, dim=-1)`
  - Verification: `pytest tests/test_layernorm.py::test_mean_computation -v` passes with PCC > 0.999

- [ ] **2.2.3** Implement variance computation
  - File: `models/demos/deepseek_v3_b1/micro_ops/layernorm/kernels/compute.cpp`
  - Requirements:
    - Compute `(x - mean)` using `sub_tiles_bcast`
    - Square the result using `mul_tiles`
    - Reduce sum using `reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>`
    - Multiply by scalar (1/W) to get variance
  - Test function: `test_variance_computation(device)`
  - Pass criteria: Compare with `torch.var(input, dim=-1, unbiased=False)`
  - Verification: `pytest tests/test_layernorm.py::test_variance_computation -v` passes with PCC > 0.999

- [ ] **2.2.4** Implement rsqrt(var + epsilon)
  - File: `models/demos/deepseek_v3_b1/micro_ops/layernorm/kernels/compute.cpp`
  - Requirements:
    - Add epsilon to variance
    - Compute rsqrt using `rsqrt_tile()` or `add_rsqrt_tile()`
    - Store inverse standard deviation
  - Test function: `test_rsqrt_computation(device)`
  - Pass criteria: Compare with `torch.rsqrt(var + epsilon)`
  - Verification: `pytest tests/test_layernorm.py::test_rsqrt_computation -v` passes with PCC > 0.99

- [ ] **2.2.5** Implement standardization
  - File: `models/demos/deepseek_v3_b1/micro_ops/layernorm/kernels/compute.cpp`
  - Requirements:
    - Compute `(x - mean)` for each tile
    - Multiply by rsqrt (broadcast scalar across row)
    - Result: standardized values with zero mean, unit variance
  - Test function: `test_standardization(device)`
  - Pass criteria: Compare with `(input - mean) / std`
  - Verification: `pytest tests/test_layernorm.py::test_standardization -v` passes with PCC > 0.99

- [ ] **2.2.6** Add gamma/beta tilization (once)
  - File: `models/demos/deepseek_v3_b1/micro_ops/layernorm/kernels/compute.cpp`
  - Requirements:
    - At kernel start, tilize gamma from CB_GAMMA_RM to CB_GAMMA_TILED
    - Tilize beta from CB_BETA_RM to CB_BETA_TILED
    - Do NOT pop gamma/beta tiled buffers (reuse for all rows)
  - Verification: Gamma/beta tiles available for affine transform

- [ ] **2.2.7** Implement gamma multiplication
  - File: `models/demos/deepseek_v3_b1/micro_ops/layernorm/kernels/compute.cpp`
  - Requirements:
    - Multiply standardized output by gamma tiles
    - Use `mul_tiles()` element-wise
  - Test function: `test_gamma_multiply(device)`
  - Pass criteria: With gamma=torch.ones, output unchanged from 2.2.5
  - Verification: `pytest tests/test_layernorm.py::test_gamma_multiply -v` passes

- [ ] **2.2.8** Implement beta addition
  - File: `models/demos/deepseek_v3_b1/micro_ops/layernorm/kernels/compute.cpp`
  - Requirements:
    - Add beta tiles to result
    - Use `add_tiles()` element-wise
    - Untilize final result to CB_OUTPUT_RM
  - Test function: `test_full_layernorm(device)`
  - Pass criteria: Compare with `LayerNormSingleCore.golden()` and `torch.nn.functional.layer_norm()`
  - Verification: `pytest tests/test_layernorm.py::test_full_layernorm -v` passes with PCC > 0.99

**Step 2.2 Complete:** [ ]

---

### Step 2.3: Writer Kernel (BRISC)

- [ ] **2.3.1** Implement basic stick writing
  - File: `models/demos/deepseek_v3_b1/micro_ops/layernorm/kernels/writer.cpp`
  - Requirements:
    - Read compile-time args: CB indices, page sizes, TensorAccessorArgs
    - Read runtime args: output buffer address, num_sticks, start_stick_id
    - Loop over sticks:
      - `cb_wait_front(CB_OUTPUT_RM, 1)`
      - `noc_async_write_page(stick_id, tensor_accessor, l1_addr)`
      - `noc_async_write_barrier()`
      - `cb_pop_front(CB_OUTPUT_RM, 1)`
  - Verification: Output data written to DRAM matches CB contents

**Step 2.3 Complete:** [ ]

---

## Stage 3: Integration and Final Testing

### Step 3.1: Full Integration Test

- [ ] **3.1.1** Test with various shapes
  - File: `models/demos/deepseek_v3_b1/micro_ops/layernorm/tests/test_layernorm.py`
  - Requirements:
    - Test function: `test_layernorm_shapes(device, shape)`
    - Parametrize with shapes:
      - `[1, 32]` - Single row, single tile
      - `[1, 64]` - Single row, two tiles
      - `[32, 32]` - 32 rows, single tile width
      - `[4, 128]` - 4 rows, 4 tiles
      - `[1, 1, 256]` - 3D input
      - `[2, 2, 512]` - Batch of sequences
    - Pass criteria: PCC > 0.99 for all shapes
  - Verification: `pytest tests/test_layernorm.py::test_layernorm_shapes -v` passes all

- [ ] **3.1.2** Test with various gamma/beta values
  - File: `models/demos/deepseek_v3_b1/micro_ops/layernorm/tests/test_layernorm.py`
  - Requirements:
    - Test function: `test_layernorm_weights(device)`
    - Test with gamma=1, beta=0 (identity affine)
    - Test with gamma=2, beta=1 (scaling + shift)
    - Test with random gamma, beta
    - Pass criteria: PCC > 0.99 for all cases
  - Verification: `pytest tests/test_layernorm.py::test_layernorm_weights -v` passes all

- [ ] **3.1.3** Test numerical edge cases
  - File: `models/demos/deepseek_v3_b1/micro_ops/layernorm/tests/test_layernorm.py`
  - Requirements:
    - Test function: `test_layernorm_edge_cases(device)`
    - Test with near-zero variance (constant input)
    - Test with large values
    - Test with small epsilon values
    - Pass criteria: No NaN/Inf, PCC > 0.99
  - Verification: `pytest tests/test_layernorm.py::test_layernorm_edge_cases -v` passes all

**Step 3.1 Complete:** [ ]

---

## Completion Checklist

- [ ] Stage 1: Generic Op Python Setup Complete
- [ ] Stage 2: Kernel Implementation Complete
- [ ] Stage 3: Integration and Final Testing Complete
- [ ] All tests pass: `pytest models/demos/deepseek_v3_b1/micro_ops/layernorm/tests/ -v`
- [ ] Code reviewed and cleaned up
- [ ] Documentation updated (SPEC.md reflects final implementation)

---

## Notes

- Check off items ONLY when fully complete and verified
- Run device reset before each test session: `tt-smi -r 0`
- Use timeout for tests: `timeout 30 pytest ...`
- Kill hung processes: `pkill -9 -f pytest`
- If a step fails, do not proceed to the next step until fixed
