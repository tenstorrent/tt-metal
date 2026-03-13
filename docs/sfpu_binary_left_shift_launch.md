# How the SFPU Binary Left Shift Launches — Full Call Chain

## 1. Python Entry Point → `ExecuteBitwiseLeftShift::invoke`

When you call `ttnn.bitwise_left_shift(tensor_a, tensor_b)`, it dispatches to `ExecuteBitwiseLeftShift::invoke`.

**File:** `ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_composite_op.cpp:1150`

This method checks whether to use the **legacy** path or the **binary_ng** path:

```cpp
if (not(use_legacy ? *use_legacy : binary::is_legacy_only(...))) {
    return BinaryOperation<BinaryOpType::LEFT_SHIFT>::invoke(...);  // binary_ng path
}
return BinaryOperationSfpu<BinaryOpType::LEFT_SHIFT>::invoke(...);  // SFPU path (legacy)
```

When `is_legacy_only()` returns true (e.g. sharded inputs, certain memory configs), it falls through to the **SFPU path**.

The struct is declared in:

**File:** `ttnn/cpp/ttnn/operations/eltwise/binary/binary_composite.hpp:437`

And registered as a TTNN operation at line 524:

```cpp
constexpr auto bitwise_left_shift =
    ttnn::register_operation<"ttnn::bitwise_left_shift", operations::binary::ExecuteBitwiseLeftShift>();
```

## 2. `BinaryOperationSfpu::invoke` → `invoke_binary_ng`

**File:** `ttnn/cpp/ttnn/operations/eltwise/binary/binary.cpp:896`

This just calls `detail::invoke_binary_ng(...)`. Inside `invoke_binary_ng` (`binary.cpp:290`), it again checks `is_legacy_only`. Since we're in the SFPU path, it calls:

```cpp
return ttnn::prim::binary(a, b, binary_op_type, dtype, memory_config, output, activations, lhs_activation);
```

This dispatches into the `BinaryDeviceOperation` which selects the **`ElementWiseMultiCoreSfpu`** program factory.

The `BinaryOperationSfpu` template is declared in:

**File:** `ttnn/cpp/ttnn/operations/eltwise/binary/binary.hpp:249`

## 3. Program Factory: `ElementWiseMultiCoreSfpu::create`

**File:** `ttnn/cpp/ttnn/operations/eltwise/binary/device/element_wise_multi_core_sfpu_pgm_factory.cpp:19`

This factory builds the full device program with **3 kernels**.

### Circular Buffers (lines 82–132)

| CB | Index | Purpose |
|----|-------|---------|
| `c_0` | src0 | Input tensor A |
| `c_1` | src1 | Input tensor B |
| `c_2` | output | Result |
| `c_3` | interim0 | Pre-scaled input A (only if `SFPU_OP_INIT_PRE_IN0_0` defined) |
| `c_4` | interim1 | Pre-scaled input B (only if `SFPU_OP_INIT_PRE_IN1_0` defined) |

For left shift, neither pre-scaling CB is needed.

### Compile-time Defines via `get_defines_fp32`

**File:** `ttnn/cpp/ttnn/operations/eltwise/binary/common/binary_op_utils.cpp:307`

For `BinaryOpType::LEFT_SHIFT`:

```cpp
new_defines.insert({"SHIFT_INIT", "binary_shift_tile_init();"});
op_name = "binary_left_shift_tile<DataFormat::Int32>";  // or UInt32/UInt16
// ...
new_defines.insert({"BINARY_SFPU_OP", "binary_left_shift_tile<DataFormat::Int32>(i*2+1, i*2, i*2+1);"});
```

The data format is chosen based on input dtypes (Int32, UInt32, or UInt16).

The final define at line 535:

```cpp
new_defines.insert({"BINARY_SFPU_OP", fmt::format("{}({}, {}, {});", op_name, idst1, idst2, idst1)});
```

### Three Kernels Created (lines 154–197)

1. **Reader** (`ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp`): Reads both input tensors from DRAM into CBs `c_0` and `c_1`
2. **Writer** (`ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`): Writes output from CB `c_2` to DRAM
3. **Compute** (`ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_sfpu_kernel.cpp`): The SFPU compute kernel with defines injected. `fp32_dest_acc_en = true` and `UnpackToDestFp32` mode for all CBs (since integer data requires 32-bit dest).

## 4. Compute Kernel Execution

**File:** `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_sfpu_kernel.cpp`

For each block of tiles:

1. **Wait** for both inputs: `cb_wait_front(cb_inp0, ...)`, `cb_wait_front(cb_inp1, ...)`
2. **Reserve** output space: `cb_reserve_back(cb_out0, ...)`
3. **Acquire** DST registers: `tile_regs_acquire()` + `tile_regs_wait()`
4. **Copy input A** tiles into DST at even positions (`i*2`):
   ```cpp
   copy_tile(cb_inp0, i, i * 2);
   ```
5. **Copy input B** tiles into DST at odd positions (`i*2+1`):
   ```cpp
   copy_tile(cb_inp1, i, i * 2 + 1);
   ```
6. **Init + Execute** the SFPU op (via the injected defines):
   ```cpp
   SHIFT_INIT       →  binary_shift_tile_init();
   BINARY_SFPU_OP   →  binary_left_shift_tile<DataFormat::Int32>(i*2+1, i*2, i*2+1);
   ```
   Note: `idst0=i*2+1` is input B (shift amount), `idst1=i*2` is input A (value), `odst=i*2+1`
7. **Pack** result from DST to output CB: `pack_tile(i * 2, cb_out0)`
8. **Pop/push** CBs to free inputs and publish output

## 5. Compute API Layer

**File:** `tt_metal/hw/inc/api/compute/binary_shift.h:37`

```cpp
template <DataFormat data_format>
ALWI void binary_left_shift_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((llk_math_eltwise_binary_sfpu_left_shift<APPROX, data_format>(idst0, idst1, odst)));
}
```

The init function at line 98:

```cpp
ALWI void binary_shift_tile_init() {
    MATH((llk_math_eltwise_binary_sfpu_shift_init<APPROX>()));
}
```

## 6. LLK Layer

**File:** `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_shift.h:19`

```cpp
template <bool APPROXIMATE, DataFormat DATA_FORMAT, bool SIGN_MAGNITUDE_FORMAT = false>
inline void llk_math_eltwise_binary_sfpu_left_shift(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = VectorMode::RC) {
    static_assert(
        DATA_FORMAT == DataFormat::Int32 || DATA_FORMAT == DataFormat::UInt32 || DATA_FORMAT == DataFormat::UInt16);
    constexpr InstrModLoadStore INSTRUCTION_MODE =
        (DATA_FORMAT == DataFormat::UInt16) ? InstrModLoadStore::LO16 : InstrModLoadStore::INT32;
    _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_binary_left_shift<APPROXIMATE, 8, INSTRUCTION_MODE, SIGN_MAGNITUDE_FORMAT>,
        dst_index0, dst_index1, odst, vector_mode);
}
```

The `_llk_math_eltwise_binary_sfpu_params_` helper handles iterating over sub-tile rows and calls the SFPU function.

## 7. SFPU Microcode

**File:** `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_shift.h:20`

The actual hardware instructions:

```cpp
for (int d = 0; d < 8; d++) {  // 8 iterations = 8 rows of 4 elements = 32 elements per face
    SFPLOAD LREG0 ← DST[in0]   // load value to shift
    SFPLOAD LREG1 ← DST[in1]   // load shift amount
    // Bounds check: if shift_amount < 0 or >= 32, result = 0
    SFPSETCC(LREG1 < 0)
    SFPIADD(-32, LREG1, LREG2)  // LREG2 = shift_amount - 32
    SFPCOMPC                      // complement condition
    SFPMOV(0 → LREG0)           // zero out result for out-of-range
    SFPENCC                       // end conditional
    // Actual shift
    SFPSHFT(LREG1, LREG0)       // LREG0 = LREG0 << LREG1
    // Store result
    SFPSTORE LREG0 → DST[out]
    dst_reg++                     // advance to next row
}
```

The SFPU processes **4 elements per cycle** (vector width), and iterates 8 times to cover all 32 rows of a tile face (32x32 tile = 4 faces x 16 rows x 16 cols, processed as 64 rows x 16 cols with the SFPU seeing groups of 4).

The Wormhole B0 implementation is at:

**File:** `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_shift.h`

## Summary Flow

```
ttnn.bitwise_left_shift(A, B)
  → ExecuteBitwiseLeftShift::invoke()
    → BinaryOperationSfpu<LEFT_SHIFT>::invoke()
      → invoke_binary_ng() → ttnn::prim::binary()
        → BinaryDeviceOperation → ElementWiseMultiCoreSfpu::create()
          ├── Creates CBs: c_0 (A), c_1 (B), c_2 (output)
          ├── Defines: SHIFT_INIT, BINARY_SFPU_OP
          ├── Reader kernel: reads A,B from DRAM → c_0, c_1
          ├── Compute kernel: eltwise_binary_sfpu_kernel.cpp
          │     copy tiles to DST → binary_left_shift_tile<Int32>()
          │       → llk_math_eltwise_binary_sfpu_left_shift()
          │         → _llk_math_eltwise_binary_sfpu_params_()
          │           → calculate_binary_left_shift()
          │             → SFPLOAD, SFPSETCC, SFPSHFT, SFPSTORE (hardware SFPU instructions)
          └── Writer kernel: writes c_2 → DRAM
```

## Key File Index

| Layer | File |
|-------|------|
| TTNN entry point | `ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_composite_op.cpp` |
| TTNN registration | `ttnn/cpp/ttnn/operations/eltwise/binary/binary_composite.hpp` |
| BinaryOperationSfpu | `ttnn/cpp/ttnn/operations/eltwise/binary/binary.hpp` / `binary.cpp` |
| Program factory | `ttnn/cpp/ttnn/operations/eltwise/binary/device/element_wise_multi_core_sfpu_pgm_factory.cpp` |
| Defines generation | `ttnn/cpp/ttnn/operations/eltwise/binary/common/binary_op_utils.cpp` |
| Compute kernel | `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_sfpu_kernel.cpp` |
| Reader kernel | `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp` |
| Writer kernel | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` |
| Compute API | `tt_metal/hw/inc/api/compute/binary_shift.h` |
| LLK (Blackhole) | `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_shift.h` |
| SFPU microcode (Blackhole) | `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_shift.h` |
| SFPU microcode (Wormhole) | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_shift.h` |
