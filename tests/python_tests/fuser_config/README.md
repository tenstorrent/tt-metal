# Fused Test YAML Configuration

## Table of Contents
- [Overview](#overview)
- [YAML Structure](#yaml-structure)
- [Global Settings](#global-settings)
- [Operation Fields](#operation-fields)
- [Operation Chaining](#operation-chaining)
- [Running and Debugging Tests](#running-and-debugging-tests)
- [Pipeline Validation](#pipeline-validation)
- [Reference](#reference)

---

## Overview

This document explains how to write YAML configuration files for testing fused LLK operations. A fused operation pipeline allows multiple computational steps to execute sequentially, where each operation processes data through three stages:

1. **Unpacker** - Reads data from L1 memory and prepares it for computation
2. **Math Unit** - Performs FPU and optionally SFPU operations
3. **Packer** - Writes results back to L1 memory

In a fused pipeline, intermediate results from one operation are written to L1 memory and can be directly used as inputs for subsequent operations, allowing efficient chaining without manual data management.

**Test Discovery:** Any YAML file placed in the `fuser_config/` directory automatically becomes a test case. The test framework discovers all `.yaml` files and executes each as a separate test.

---

## YAML Structure

Each configuration file contains global settings followed by an `operations` list:

```yaml
# Global Settings (apply to all operations)
dest_acc: "Yes"              # 16-bit or 32-bit dest mode
profiler_enabled: true       # Enable performance profiling
loop_factor: 16              # Loop factor for performance tests

operations:
  - # Input/Output Configuration
    src_a: "input_A"           # First input operand
    src_b: "input_B"           # Second input operand
    output: "result"           # Output operand name
    src_a_dims: [64, 64]       # Dimensions of src_a
    src_b_dims: [64, 64]       # Dimensions of src_b
    input_format: "Float16_b"  # Input data format
    output_format: "Float16_b" # Output data format

    # Operation Configuration
    unpacker: "UnpackerAB"     # Which unpacker to use
    math:                      # Math operations
      fpu: "Elwadd"            # FPU operation
      sfpu:                    # Optional SFPU chain
        - type: "UnarySfpu"
          operation: "Gelu"
          approximation_mode: "No"
          iterations: 128
    packer: "Packer"          # Which packer to use

    # Hardware Settings
    batch_size: 4             # Number of tiles per batch
    math_fidelity: "HiFi3"
    dest_sync: "Full"
    unpack_transpose_within_face: "Yes"
    unpack_transpose_faces: "Yes"
```

---

## Global Settings

Global settings are specified at the top level of the YAML file. These settings apply to all operations in the pipeline.

### `dest_acc` (string, required)
Controls whether dest operates in 32-bit accumulation mode for all operations in the pipeline. When set to `"Yes"`, dest operates in 32-bit format. When set to `"No"`, dest operates in 16-bit format.

**Important:** The `dest_acc` setting cannot be changed in the middle of kernel execution. All operations in a fused pipeline must use the same `dest_acc` setting.

**When to use `dest_acc: "Yes"`:**

1. **Any 32‑bit formats (Float32)**
   If any operation in the fused pipeline uses a 32‑bit data format (`Float32`) as `input_format` or `output_format`, `dest_acc` should be set to `"Yes"` for all operations so that dest can store 32‑bit values.

2. **8‑bit exponent → Float16 conversions (`Float16_b` / `Bfp8_b` → `Float16`)**
   When the input format has an 8‑bit exponent (`Float16_b`, `Bfp8_b`) and the `output_format` is `Float16` (5‑bit exponent), a 32‑bit intermediate in dest is required. In this case `dest_acc` must be set to `"Yes"`.

3. **Improved numerical precision (optional)**
   For all other supported format combinations where both `dest_acc: "No"` and `"Yes"` are allowed, you can enable `dest_acc: "Yes"` to accumulate in 32‑bit for better numerical accuracy, at the cost of reduced dest tile capacity.

**Dest Tile Capacity:**

The available dest tile capacity depends on both `dest_acc` and the per-operation `dest_sync` setting:

| Dest Capacity                 | `dest_acc: "No"` (16-bit) | `dest_acc: "Yes"` (32-bit) |
|-------------------------------|-----------------------------|----------------------------|
| `dest_sync: "Half"` (default) | 8 tiles                     | 4 tiles                    |
| `dest_sync: "Full"`           | 16 tiles                    | 8 tiles                    |

### `profiler_enabled` (boolean, optional)
Enables performance profiling for the fused kernel. When set to `true`, the test framework measures and reports execution time and throughput metrics. When set to `false` or omitted (default), performance profiling is disabled and the test only validates correctness.

### `loop_factor` (integer, optional)
Specifies the number of times to repeat the operation sequence when performance profiling is enabled. This allows measuring sustained performance over multiple iterations. The value must be a positive integer. This setting is only used when `profiler_enabled` is `true`.

---

## Operation Fields

### Operand Name

#### `src_a` (string, required)
The name of the first input operand. This can be either an input operand (e.g., `"input_A"`, `"input_B"`) or an output from a previous operation (e.g., `"result1"`).

**Important:** If the operand is not an output from a previous operation, data will be automatically generated for it. For outputs from previous operations, the data comes from that operation's result.

#### `src_b` (string, required)
The name of the second input operand. Follows the same rules as `src_a`.

**Note:** Even if the unpacker only uses one input (e.g., `UnpackerA`), both `src_a` and `src_b` must be specified.

#### `output` (string, required)
The name of the output operand where results will be stored. This output can be used as input in subsequent operations. The output name must be unique and cannot match any previously defined operand names (neither inputs nor outputs from previous operations).

---

### Operand Dimensions

#### `src_a_dims` (array of 2 integers, required)
Dimensions of the src_a operand in format `[height, width]`. Values must be multiples of 32 (tile size is 32x32). For example, `[64, 64]` represents a 2x2 tile matrix, while `[32, 128]` represents a 1x4 tile matrix.

#### `src_b_dims` (array of 2 integers, required)
Dimensions of the src_b operand in format `[height, width]`. Same rules as `src_a_dims`.

#### `output_pack_dims` (array of 2 integers, optional)
Override dimensions of the output operand in format `[height, width]`. Same rules as `src_a_dims`.

---

### Operand Format

#### `input_format` (string, required)
The data format of input operands. Available options: `"Float16_b"`, `"Float16"`, `"Float32"`, and `"Bfp8_b"`.

**Important:** When chaining operations, this must match the `output_format` of the previous operation that produced the input operand. See [Format Handling in Chains](#format-handling-in-chains) for details.

#### `output_format` (string, required)
The data format for the output operand. Options are the same as `input_format`.

**Important:** Can be different from `input_format` (allows format conversion within an operation). When this output is used in subsequent operations, their `input_format` must match this value. See [Format Handling in Chains](#format-handling-in-chains) for details.

### Operand Const Values

#### `src_a_const_value` (float, optional)
A constant value used to initialize the src_a operand. When specified, no random tensor is generated for src_a and the operand is fully initialized with this constant value. This field should not be set when src_a refers to the output of a previous operation.

#### `src_b_const_value` (float, optional)
A constant value used to initialize the src_b operand. Follows the same rules as `src_a`.

**Important:** When using the ReduceFpu operation, `src_b_const_value` must be set to 1.0.

---

### Unpacker Configuration

#### `unpacker` (string, required)
Specifies which unpacker to use. Options:

- **`"UnpackerA"`** - Unpacks only the `src_a` operand. Use this for unary operations that only need one input. It's compatible with `Datacopy`.

- **`"UnpackerAB"`** - Unpacks both `src_a` and `src_b` operands. Use this for binary FPU operations (e.g., `Elwadd`, `Elwmul`, `Elwsub`). It's required for element-wise FPU operations.

- **`"UnpackerTilizeA"`** - Unpacks and tilizes the `src_a` operand by converting row-major data to tilized format during unpacking. It's compatible with `Datacopy`.

- **`"MatmulUnpacker"`** - Specialized unpacker for matrix multiplication that unpacks both operands with matmul specific layout. This is required for the `Matmul` FPU operation and is not compatible with `Datacopy` or element-wise FPU operations.

**Compatibility Matrix:**

| Unpacker        | Datacopy | Elwadd/Elwmul/Elwsub | Matmul | Reduce(Scalar/Row/Column) | SFPU |
|-----------------|----------|----------------------|--------|---------------------------|------|
| UnpackerA       | ✓        | ✗ (needs 2 inputs)   | ✗      | ✗                         | ✓    |
| UnpackerAB      | ✓        | ✓                    | ✗      | ✓                         | ✓    |
| UnpackerTilizeA | ✓        | ✗ (needs 2 inputs)   | ✗      | ✗                         | ✓    |
| MatmulUnpacker  | ✗        | ✗                    | ✓      | ✗                         | ✓    |

---

### Math Configuration

#### `math` (object, required)
Defines the mathematical operations to perform. Contains:

##### `fpu` (string, required)
Available FPU operations:

- **`"Datacopy"`** - Simple data copy without transformation that copies data from input to output. This is useful for testing unpacker/packer combinations.

- **`"Elwadd"`** - Element-wise addition (A + B) that requires `UnpackerAB` since it needs both operands. This operation adds corresponding elements from src_a and src_b, producing Output[i] = src_a[i] + src_b[i].

- **`"Elwmul"`** - Element-wise multiplication (A * B) that requires `UnpackerAB`. This operation multiplies corresponding elements from src_a and src_b, producing Output[i] = src_a[i] * src_b[i].

- **`"Elwsub"`** - Element-wise subtraction (A - B) that requires `UnpackerAB`. This operation subtracts src_b from src_a element-wise, producing Output[i] = src_a[i] - src_b[i].

- **`"Matmul"`** - Matrix multiplication (A × B) that requires `MatmulUnpacker`. This performs standard matrix multiplication where src_a dimensions must be [M, K], src_b must be [K, N], and output dimensions will be [M, N].

- **`"ReduceScalar"`** - Reduces all elements of src_a within each tile into a single scalar value. The result for each tile is written to output coordinate (0, 0) of that tile, while all other output elements are set to zero by the packer.

- **`"ReduceRow"`** - Performs a row-wise reduction of src_a within each tile, producing one reduced value per row. Each result is written to the first column of the corresponding row in that tile, while all other output elements are set to zero by the packer.

- **`"ReduceColumn"`** - Performs a column-wise reduction of src_a within each tile, producing one reduced value per column. Each result is written to the first row of the corresponding column in that tile, while all other output elements are set to zero by the packer.

#### `reduce_pool` (string, required for Reduce operations)
Specifies the reduction method to use for the Reduce operation. Available options are: `"Sum"`, `"Average"` and `"Max"`

##### `sfpu` (array of objects, optional)
A list of Special Function Unit operations to execute after the FPU operation. SFPU operations execute **in the order specified** in the array.

**SFPU Chain Execution Model:**
1. FPU operation completes and writes to dest
2. First SFPU operation reads from dest, processes, writes back
3. Second SFPU operation reads from dest, processes, writes back
4. ... continues for all SFPU operations in the chain
5. Packer reads final result from dest

**UnarySfpu Configuration:**
```yaml
sfpu:
  - type: "UnarySfpu"
    operation: "Exp"              # Operation name (see below)
    approximation_mode: "No"      # "Yes" or "No"
    iterations: 128               # Number of iterations (see below)
```

**UnarySfpu Fields:**

The `type` field must be set to `"UnarySfpu"`.

The `operation` field specifies which unary SFPU operation to perform. Available operations include:
  - **Activation Functions**: `"Gelu"`, `"ReluMax"`, `"ReluMin"`, `"Silu"`, `"Elu"`, `"Celu"`, `"Hardsigmoid"`
  - **Trigonometric**: `"Sin"`, `"Cos"`
  - **Hyperbolic**: `"Asinh"`, `"Acosh"`, `"Atanh"`
  - **Exponential/Logarithmic**: `"Exp"`, `"Exp2"`, `"Log"`
  - **Power/Root**: `"Sqrt"`, `"Rsqrt"`, `"Square"`, `"Reciprocal"`
  - **Other**: `"Abs"`, `"Neg"`, `"Fill"`, `"Threshold"`

The `approximation_mode` controls the precision/speed tradeoff. Set it to `"Yes"` to use faster approximations with lower accuracy, or `"No"` for full precision calculations. Approximation mode is particularly useful for complex functions like `Gelu` or `Exp` where hardware approximations can significantly affect the precision of the results.

The `iterations` field determines how many datums to process. Each iteration processes 32 datums (one row of a face), meaning 32 iterations process one full tile (1024 elements). The number of iterations must be calculated based on your input dimensions. For example, a 64×64 matrix contains 4 tiles (4096 elements), requiring 128 iterations to process all data. The value must be at least 1 and cannot exceed the total number of elements divided by 32, where total elements equals `src_a_dims[0] * src_a_dims[1]`.

The `dst_dest_tile_index` specifies the starting tile index within the dest register where the operation begins (default: 0). See **Understanding Tile Indices** for details on tile indexing.

The `fill_const_value` sets the constant value that each element will be set to when using the `Fill` operation (default: 1.0).

**BinarySfpu Configuration:**
```yaml
sfpu:
  - type: "BinarySfpu"
    operation: "SfpuElwadd"           # Operation name
    approximation_mode: "No"          # "Yes" or "No"
    iterations: 128                   # Number of iterations
    src1_dest_tile_index: 0           # First source tile index in destination
    src2_dest_tile_index: 1           # Second source tile index in destination
    dst_dest_tile_index: 1            # Destination tile index for result
```

**BinarySfpu Fields:**

The `type` field must be set to `"BinarySfpu"`.

The `operation` field specifies which binary SFPU operation to perform. Available operations include:
  - **Arithmetic**: `"SfpuElwadd"`, `"SfpuElwmul"`, `"SfpuElwsub"`
  - **Bitwise Shifts**: `"SfpuElwLeftShift"`, `"SfpuElwRightShift"`, `"SfpuElwLogicalRightShift"`
  - **Special**: `"SfpuXlogy"` (x * log(y)), `"SfpuAddTopRow"`

The `approximation_mode` and `iterations` fields work the same as in UnarySfpu, where each iteration processes 32 elements.

The three tile index fields specify where in dest to read operands and write results. All indices are 0-based:
- `src1_dest_tile_index`: Tile index for the first source operand
- `src2_dest_tile_index`: Tile index for the second source operand
- `dst_dest_tile_index`: Tile index where the result will be written

**Understanding Tile Indices:**

Tile indices must stay within valid bounds. The number of tiles in dest is calculated as `(height × width) / 1024`. For a 64×64 input (4096 elements), there are 4 tiles numbered 0, 1, 2, and 3. Each tile index points to a starting position of `tile_index × 1024` elements.

The system validates that memory accesses won't exceed tensor boundaries. For each tile index and iteration count, it checks that `(tile_index × 1024) + (iterations × 32) ≤ total_elements`. This ensures that operations starting at a given tile don't read or write beyond the available data.

BinarySfpu supports in-place operations where the result overwrites one of the source tiles. For example, setting `src1_dest_tile_index: 0` and `dst_dest_tile_index: 0` will overwrite tile 0 with the operation result.

**SFPU Chain Example:**
```yaml
math:
  fpu: "Datacopy"
  sfpu:
    - type: "UnarySfpu"
      operation: "Exp"        # First: apply exp(x)
      approximation_mode: "No"
      dest_idx: 0
      iterations: 128
    - type: "UnarySfpu"
      operation: "Neg"        # Second: apply -x (so result is -exp(x))
      approximation_mode: "No"
      iterations: 128
      dest_idx: 0
    - type: "BinarySfpu"
      operation: "SfpuElwadd" # Third: add two tiles together
      approximation_mode: "No"
      iterations: 32
      src1_dest_tile_index: 0
      src2_dest_tile_index: 1
      dst_dest_tile_index: 1
```

**Important:** When using SFPU operations after a Reduce operation, ensure iteration counts do not access elements outside the reduced region, as they may contain residual values before the packer zeroes them.

---

### Packer Configuration

#### `packer` (string, required)
Specifies which packer to use. Options:

- **`"Packer"`** - Standard packer that packs results from dest to L1 memory. It works with all FPU and SFPU operations and is the default choice for most operations.

---

### Hardware Configuration

#### `batch_size` (integer, optional)
Controls how many output tiles are processed together in a single batch. The system automatically determines the optimal batch size based on dest capacity, but you can manually override it when needed.
See the dest tile capacity table in the [Global Settings](#global-settings) section.

**Automatic Batch Size Determination:** When `batch_size` is not specified or set to a value that exceeds dest capacity, the system automatically adjusts it based on dest capacity.

**Special Case for Matmul:** When using `Matmul` FPU and the output exceeds dest capacity, `batch_size` is automatically set to 1, meaning one output tile is computed at a time.

**Manual Override:** You can manually specify `batch_size` in the YAML configuration. This is useful for performance tuning or when you want explicit control over batching behavior. However, the system will still enforce that `batch_size` does not exceed dest capacity or total output tile count.

#### `dest_sync` (string, optional)
Controls the synchronization mode between the math unit and packer. When set to `"Half"` (the default), dest operates in half synchronization mode (double buffering), where the math unit and packer can work on different halves of dest simultaneously. When set to `"Full"`, dest operates in full synchronization mode (single buffering), where the math unit and packer share the full dest space without overlap.

**Important:** Due to the synchronization between unpacker and packer (the unpacker waits for the packer from the previous operation to finish), `dest_sync` does not impact overall pipeline performance. It only affects dest capacity. See the dest tile capacity table in the [Global Settings](#global-settings) section for how `dest_sync` and `dest_acc` interact.

#### `math_fidelity` (string, required)
Controls the precision/speed tradeoff for math operations. Available settings are `"LoFi"`, `"HiFi2"`, `"HiFi3"`, and `"HiFi4"`. Higher fidelity settings provide greater precision at the cost of slower execution. The actual impact depends on the specific operation.

#### `unpack_transpose_within_face` (string, optional)
Controls whether to transpose data within each 16x16 face during unpacking. Set to `"Yes"` to enable transpose within faces, or `"No"` to disable (default).

**Understanding the operation:** Each 32x32 tile is composed of four 16x16 faces arranged as [Top-Left, Top-Right, Bottom-Left, Bottom-Right]. When this option is enabled, the hardware transposes rows and columns within each face independently, leaving the arrangement of faces unchanged. This means element `[i, j]` within a face becomes `[j, i]` within that same face.

#### `unpack_transpose_faces` (string, optional)
Controls whether to transpose the layout of faces within tiles during unpacking. Set to `"Yes"` to enable face transposition, or `"No"` to disable (default).

**Understanding the operation:** This rearranges the four 16x16 faces within each 32x32 tile. The standard layout [Top-Left, Top-Right, Bottom-Left, Bottom-Right] becomes [Top-Left, Bottom-Left, Top-Right, Bottom-Right] after transposition. This effectively swaps the positions of the top-right and bottom-left faces.

**Combining transpose operations:** These two transpose options can be used together. When both are set to `"Yes"`, the hardware first transposes the face layout, then transposes data within each face. This combination produces a complete matrix transpose:

```yaml
unpack_transpose_faces: "Yes"        # Step 1: Transpose face layout
unpack_transpose_within_face: "Yes"  # Step 2: Transpose within each face
```

**Unpacker compatibility:** `UnpackerA` supports independent values for these two transpose options. `UnpackerAB` and `MatmulUnpacker` require both transpose options to have the same value. `UnpackerTilizeA` does not support transpose operations.

---

## Operation Chaining

### Basic Chaining Concept

Operations are chained by using the `output` of one operation as the `src_a` or `src_b` of a subsequent operation. Each operation's output is packed to L1 memory by the packer and can then be unpacked as input for the next operation in the chain. See the [Complex Chain Example](#complex-chain-example) below for a complete demonstration.

### Critical Rules for Chaining

1. **Output names must be unique** - Each operation must have a unique output name
2. **Operations execute in order** - Operations are processed in the order they appear in the YAML
3. **No forward references** - An operation can only reference outputs from operations that appear before it
4. **Dimension consistency** - Output dimensions from operation N must match input dimensions for operation N+1 when that output is used
5. **Format consistency** - Next operation's `input_format` must match previous operation's `output_format` at boundaries

### Format Handling in Chains

Format changes are allowed within operations, but chain boundaries must maintain format consistency.

**Key Rules:**
1. Within an operation: `input_format` and `output_format` can differ
2. Between operations: next operation's `input_format` must match previous operation's `output_format`
3. When using an output as input: formats must align exactly

**Correct Practice:**
```yaml
operations:
  - output: "result1"
    input_format: "Float16_b"
    output_format: "Float32"  # Format change is allowed
    # ...

  - src_a: "result1"
    input_format: "Float32"   # Must match previous output_format
    output_format: "Float32"
    # ...
```

**Invalid Example:**
```yaml
operations:
  - output: "result1"
    output_format: "Float32"
    # ...

  - src_a: "result1"
    input_format: "Float16_b"     # ERROR: format mismatch!
```

### Complex Chain Example

```yaml
operations:
  # Step 1: Apply exponential to input
  - src_a: "input_A"
    src_b: "input_A"  # Dummy, not used
    output: "exp_result"
    unpacker: "UnpackerA"
    math:
      fpu: "Datacopy"
      sfpu:
        - type: "UnarySfpu"
          operation: "Exp"
          approximation_mode: "No"
          iterations: 128
    packer: "Packer"
    # ...

  # Step 2: Multiply exp result with another input
  - src_a: "exp_result"
    src_b: "input_B"
    output: "scaled_result"
    unpacker: "UnpackerAB"
    math:
      fpu: "Elwmul"
    packer: "Packer"
    # ...

  # Step 3: Matrix multiply scaled result with weight matrix
  - src_a: "scaled_result"
    src_b: "input_C"
    output: "final_output"
    unpacker: "MatmulUnpacker"
    math:
      fpu: "Matmul"
    packer: "Packer"
    # ...
```

---

## Running and Debugging Tests

### Running All Tests

To run all tests, execute the following command from the `tests/python_tests/` directory:

```bash
pytest fused_test.py
```

This command runs all YAML configuration files from the `tests/python_tests/fuser_config/` directory. For each test, it generates corresponding C++ code in `tests/sources/fused_tests/`.

### Running a Specific Test

To run a specific test, use the test name (YAML filename without extension) as a parameter:

```bash
pytest fused_test.py::test_fused[example]
```

This command runs only `example.yaml` from the `tests/python_tests/fuser_config/` directory and generates `tests/sources/fused_tests/example.cpp`.

### Debugging with Manual C++ Edits

By default, pytest regenerates C++ code for each test run. To skip code generation and manually edit the generated C++ file, use the `--skip-codegen` flag:

```bash
pytest fused_test.py::test_fused[example] --skip-codegen
```

This is useful when debugging specific C++ code issues or testing manual optimizations without modifying the YAML configuration.

---

## Pipeline Validation

### How Validation Works

When you run a test, the pipeline goes through the following stages:

1. **YAML Parsing** - The YAML file is loaded and parsed into operation objects
2. **Data Generation** - Random input data is generated for all operands
3. **C++ Code Generation** - C++ code is generated that implements the operations using LLK
4. **Hardware Execution** - The generated C++ code is compiled and executed on hardware
5. **Result Collection** - Output data is read from hardware L1 memory
6. **Golden Comparison** - Results are compared against two golden references

### The Two Golden Tests

Each operation's output is validated against **two independent golden references**:

#### 1. L1 Golden Test
This golden uses actual hardware outputs from L1 memory as operands when operations are chained. After each operation completes on hardware, results are read from L1 memory. For chained operations, these L1 results become inputs to the golden computation of the next operation. For standalone operations (not chained), L1 golden is identical to Master golden.

This validates that each operation produces correct results given its actual inputs from hardware. It checks whether the operation itself is implemented correctly, independent of whether its inputs are correct. In chains, if a previous operation produces wrong output, L1 golden will use that wrong output as input, so the test will pass as long as the current operation processes it correctly.

This test can fail when the operation itself has an error—incorrect FPU/SFPU implementation or wrong unpacker/packer behavior.

#### 2. Master Golden Test
This is a high-level golden computed using standard mathematical operations (typically PyTorch) without any hardware-specific transformations. It uses PyTorch or NumPy operations directly, operates on untiled row-major data with no format conversions (using native floating point), and serves as a pure mathematical reference.

This validates that the overall mathematical operation is correct, that the FPU and SFPU operations produce correct results, that the operation chain produces the expected output, and that there are no numerical accuracy issues.

This test can fail due to incorrect operation chaining, numerical precision issues (especially with `LoFi` fidelity), or unsupported reconfiguration in LLK.

**Key Insight:** When operations are chained, L1 golden uses real hardware outputs as inputs, while Master golden computes the entire chain mathematically. This distinction helps identify where errors occur in a chain.

### Tolerance in Comparisons

The comparison allows for small numerical differences due to floating point precision limitations, different execution order (hardware vs. CPU), approximation modes in SFPU operations, and math fidelity settings.

Tolerance levels depend on the data format (Float32 has tighter tolerance than Float16), math fidelity (LoFi has looser tolerance than HiFi4), and SFPU approximation mode.

---

## Reference

For working examples, see [example.yaml](example.yaml).

---
