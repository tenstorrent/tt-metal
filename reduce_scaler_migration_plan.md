# Plan: Replace Legacy `generate_reduce_scaler_legacy` Calls

## Progress

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | moreh_sum_h (proof of concept) | Done |
| 1 | Remaining moreh ops (3 kernels, 2 host files) | Done |
| 2 | Generic reduce ops (3 kernels, 3 host factories) | Pending |
| 3 | SDPA + MOE + Sampling (7 files) | Pending |
| 4 | Softmax (5 files) | Done |
| 5 | Layernorm (9 files) | Pending |
| 6 | GroupNorm + Experimental + Tests (10 files) | Pending |
| Post | Cleanup: remove legacy API + dead code | Pending |

## Context

The codebase has 38 kernel files calling `dataflow_kernel_lib::generate_reduce_scaler_legacy(cb_id, packed_scaler)`, which takes a pre-packed bf16 uint32_t. Two new APIs in `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp` replace it:

- **`generate_reduce_scaler<cb_id, pool_type, reduce_dim, reduce_factor>()`** — Auto-computes the scaler (1.0 for SUM/MAX, 1/N for AVG). Use when the scaler is a known constant.
- **`prepare_reduce_scaler<cb_id>(float scaler_f)`** — Fills CB with a caller-provided float. Use when the scaler is computed on the host and varies per invocation.

Both auto-deduce data format (bf16/f32) and tile shape (half/full) from the CB.

### Migration patterns

**Pattern A: Fixed scaler (SUM/MAX with scaler=1.0)** — kernel-only, no host args needed:
```cpp
// OLD:
constexpr uint32_t scaler = get_compile_time_arg_val(N);
dataflow_kernel_lib::generate_reduce_scaler_legacy(cb_id, scaler);

// NEW (remove scaler CT arg from host entirely):
dataflow_kernel_lib::generate_reduce_scaler<cb_id, PoolType::SUM, ReduceDim::REDUCE_COL>();
```

**Pattern B: AVG with known reduce_factor** — pass reduce_factor as CT arg instead of packed scaler:
```cpp
// HOST: replace packed_scaler_value with origin_H (the reduce dimension size)
reader_compile_time_args.push_back(origin_H);

// KERNEL:
constexpr uint32_t reduce_factor = get_compile_time_arg_val(N);
dataflow_kernel_lib::generate_reduce_scaler<cb_id, PoolType::AVG, ReduceDim::REDUCE_COL, reduce_factor>();
```

**Pattern C: Variable scaler from ttnn API** — pass reduce_factor + input_scaler as CT args:
```cpp
// HOST:
reader_compile_time_args.push_back(1);  // reduce_factor (1 for SUM/MAX)
reader_compile_time_args.push_back(std::bit_cast<uint32_t>(operation_attributes.scaler));  // input_scaler

// KERNEL:
constexpr uint32_t reduce_factor = get_compile_time_arg_val(N);
constexpr uint32_t input_scaler_bits = get_compile_time_arg_val(N + 1);
float input_scaler = __builtin_bit_cast(float, input_scaler_bits);
dataflow_kernel_lib::generate_reduce_scaler<cb_id, REDUCE_OP, REDUCE_DIM, reduce_factor>(input_scaler);
```

**Host cleanup** — remove packed scaler computation (`bfloat16::truncate` + `pack_two_bfloat16_into_uint32`). If the packed arg is still used by other calls (e.g. `generate_bcast_col_scalar`), keep it; otherwise remove it and shift CT/RT arg indices.

---

## Phase 0: moreh_sum_h (proof of concept) → `generate_reduce_scaler` Pattern A

Single kernel migration. SUM with constant scaler 1.0.

### Kernel: `ttnn/cpp/ttnn/operations/moreh/moreh_sum/device/moreh_sum_h_impl_kernels/reader_moreh_sum_h.cpp`

**Remove** (line 29-30):
```cpp
constexpr uint32_t scaler = get_compile_time_arg_val(src_args.next_compile_time_args_offset());
dataflow_kernel_lib::generate_reduce_scaler_legacy(cb_id_in2, scaler);
```

**Replace with:**
```cpp
dataflow_kernel_lib::generate_reduce_scaler<cb_id_in2, PoolType::SUM, ReduceDim::REDUCE_COL>();
```

### Host: `ttnn/cpp/ttnn/operations/moreh/moreh_sum/device/moreh_sum_h_program_factory.cpp`

1. Remove scaler computation (lines 121-122): `bfloat16` + `pack_two_bfloat16_into_uint32`
2. Remove `reader_compile_time_args.push_back(packed_scaler_value)` (line 125)
3. Can also remove `float scaler = 1.0f;` (line 29) and `#include <tt-metalium/bfloat16.hpp>` if unused

### Testing
```bash
pytest tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_sum.py -v -k "test_moreh_sum_dims"
```

---

## Phase 1: Remaining moreh ops (3 kernels, 2 host files) → `generate_reduce_scaler`

### 1a. moreh_linear_backward — Pattern A (SUM, constant 1.0)

**Kernel:** `ttnn/cpp/ttnn/operations/moreh/moreh_linear_backward/device/kernels/reader_moreh_bias_backward_h.cpp`

- Remove `constexpr uint32_t scaler = get_compile_time_arg_val(0);` (line 19)
- Change `constexpr auto src0_args = TensorAccessorArgs<1>();` → `TensorAccessorArgs<0>();` (CT arg indices shift)
- Replace line 25: `generate_reduce_scaler_legacy(cb_id_scaler, scaler)` → `generate_reduce_scaler<cb_id_scaler, PoolType::SUM, ReduceDim::REDUCE_COL>()`

**Host:** `ttnn/cpp/ttnn/operations/moreh/moreh_linear_backward/device/moreh_linear_backward_multi_core_program_factory.cpp`

- Remove scaler computation (lines 89-90)
- Change `reader_compile_time_args{packed_scaler_value}` → `reader_compile_time_args{}` (line 91)
- Remove `#include <tt-metalium/bfloat16.hpp>` if unused

### 1b. moreh_mean_h — Pattern B (AVG with reduce_factor)

**Kernel:** `ttnn/cpp/ttnn/operations/moreh/moreh_mean/device/kernels/reader_moreh_mean_h.cpp`

- Change `constexpr uint32_t scaler = ...` (line 30) → `constexpr uint32_t reduce_factor = get_compile_time_arg_val(src_args.next_compile_time_args_offset());`
- Replace line 31: `generate_reduce_scaler_legacy(cb_id_in2, scaler)` →
  ```cpp
  dataflow_kernel_lib::generate_reduce_scaler<cb_id_in2, PoolType::AVG, ReduceDim::REDUCE_COL, reduce_factor>();
  ```

**Host:** `ttnn/cpp/ttnn/operations/moreh/moreh_mean/device/moreh_mean_h_program_factory.cpp`

- Remove scaler computation (lines 78-80: `float scaler`, `bfloat16`, `pack_two_bfloat16`)
- Change `reader_compile_time_args.push_back(packed_scaler_value)` (line 83) → `reader_compile_time_args.push_back(origin_H);`
- Remove `#include <tt-metalium/bfloat16.hpp>` if unused

### 1c. moreh_sum_nc — Pattern A (SUM with zero scaler, kernel-only)

**Kernel:** `ttnn/cpp/ttnn/operations/moreh/moreh_sum/device/moreh_sum_nc_impl_kernels/reader_moreh_sum_nc.cpp`

- Remove `constexpr uint32_t scaler = 0;` (line 31)
- Replace line 32: `generate_reduce_scaler_legacy(cb_id_in1, scaler)` →
  ```cpp
  dataflow_kernel_lib::generate_reduce_scaler<cb_id_in1, PoolType::SUM, ReduceDim::REDUCE_COL>(0.0f);
  ```
- No host changes needed (scaler=0 was hardcoded in kernel)

### Testing
```bash
pytest tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_linear.py -v
pytest tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_mean.py -v
pytest tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_sum.py -v
```

---

## Phase 2: Generic reduce ops (3 kernels, 3 host factories) → `generate_reduce_scaler` Pattern C

`REDUCE_OP`/`REDUCE_DIM` from `get_defines()` (always SUM or MAX, never AVG).
`reduce_factor` as CT arg (uint32_t, constexpr → template param). Currently always 1 for generic reduce.
`input_scaler` as CT arg (uint32_t bits → `__builtin_bit_cast(float, ...)` in kernel). This is `operation_attributes.scaler` from the ttnn reduce API.

### 2a. reader_unary_reduce_universal_start_id.cpp (W + HW factories)

Used by both `reduce_op_multi_core_w_program_factory` and `reduce_op_single_core_hw_program_factory`.

**Kernel changes:**
- Change CT arg 0 from packed scaler to two args:
  ```cpp
  constexpr uint32_t reduce_factor = get_compile_time_arg_val(0);
  constexpr uint32_t input_scaler_bits = get_compile_time_arg_val(1);
  constexpr auto tensor_args = TensorAccessorArgs<2>();  // shifted from 1
  ```
- Replace `#ifndef REDUCE_ROW_SUM_VIA_MM` block (lines 21-22):
  ```cpp
  float input_scaler = __builtin_bit_cast(float, input_scaler_bits);
  dataflow_kernel_lib::generate_reduce_scaler<cb_id_in2, REDUCE_OP, REDUCE_DIM, reduce_factor>(input_scaler);
  ```

**Host W factory** (`reduce_op_multi_core_w_program_factory.cpp`):
- Already passes `reduce_defines` to reader (line 96) — REDUCE_OP/REDUCE_DIM available
- Remove packed bf16 computation (lines 80-81)
- CT args (line 83): `{packed_scaler_value}` → `{1, std::bit_cast<uint32_t>(operation_attributes.scaler)}`

**Host HW factory** (`reduce_op_single_core_hw_program_factory.cpp`):
- Currently passes NO defines to reader — add reduce_defines to `ReaderDataMovementConfig` (line 92)
- Remove packed bf16 computation (lines 79-80), keep `sqrt()` (line 31)
- CT args (line 81): `{packed_scaler_value}` → `{1, std::bit_cast<uint32_t>(scaler)}` (scaler already has sqrt applied)

### 2b. reader_unary_transpose_wh_universal_input_cols_partitioned.cpp (H factory non-sharded)

**Kernel changes:**
- Change CT arg 3 from packed scaler to two args:
  ```cpp
  constexpr uint32_t reduce_factor = get_compile_time_arg_val(3);
  constexpr uint32_t input_scaler_bits = get_compile_time_arg_val(4);
  constexpr auto tensor_args = TensorAccessorArgs<5>();  // shifted from 4
  ```
- Replace line 32: `generate_reduce_scaler_legacy(cb_id_in2, scalar)` →
  ```cpp
  float input_scaler = __builtin_bit_cast(float, input_scaler_bits);
  dataflow_kernel_lib::generate_reduce_scaler<cb_id_in2, REDUCE_OP, REDUCE_DIM, reduce_factor>(input_scaler);
  ```

**Host** (`reduce_op_multi_core_h_program_factory.cpp`, non-sharded path):
- Remove packed bf16 computation (lines 125-126)
- CT args (line 140): `{Ht, Wt, HtWt, packed_scaler_value}` → `{Ht, Wt, HtWt, 1, std::bit_cast<uint32_t>(operation_attributes.scaler)}`
- Merge `reduce_defines` into `reader_defines` (around line 144)

### 2c. reader_unary_transpose_wh_interleaved_input_cols_partitioned_sharded.cpp (H factory sharded)

**Kernel changes:**
- Add new CT args and remove RT arg:
  ```cpp
  constexpr uint32_t reduce_factor = get_compile_time_arg_val(3);  // new CT arg after cb_id args
  constexpr uint32_t input_scaler_bits = get_compile_time_arg_val(4);
  ```
- Remove `uint32_t scalar = get_arg_val<uint32_t>(6);` (line 22)
- Replace line 23: `generate_reduce_scaler_legacy(cb_id_in2, scalar)` →
  ```cpp
  float input_scaler = __builtin_bit_cast(float, input_scaler_bits);
  dataflow_kernel_lib::generate_reduce_scaler<cb_id_in2, REDUCE_OP, REDUCE_DIM, reduce_factor>(input_scaler);
  ```

**Host** (`reduce_op_multi_core_h_program_factory.cpp`, sharded path):
- Remove `packed_scaler_value` from RT args (line 233): remove last element
- Add `1, std::bit_cast<uint32_t>(operation_attributes.scaler)` to sharded `reader_compile_time_args` (line 129)
- Merge `reduce_defines` into `reader_defines` for sharded path (around line 130-131)

### Testing
```bash
pytest tests/ttnn/unit_tests/operations/reduce/test_sum.py -v
pytest tests/ttnn/unit_tests/operations/reduce/test_reduction_mean.py -v
pytest tests/ttnn/unit_tests/operations/reduce/test_max.py -v
pytest tests/ttnn/unit_tests/operations/reduce/test_reduction.py -v
```

---

## Phase 3: SDPA + MOE + Sampling (7 files) → `generate_reduce_scaler`

All use `identity_scalar_packed = 1.0` for SUM reduce. The scaler is a fixed constant — `generate_reduce_scaler` eliminates the need for host-side scaler computation entirely.

### Kernel files
1. `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/writer_interleaved.cpp`
2. `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/joint_writer.cpp`
3. `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/ring_joint_writer.cpp`
4. `ttnn/cpp/ttnn/operations/transformer/sdpa_windowed/device/kernels/dataflow/writer_windowed.cpp`
5. `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/writer_decode_all.cpp`
6. `ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/writer_unary_interleaved.cpp`
7. `ttnn/cpp/ttnn/operations/reduction/sampling/device/kernels/dataflow/writer_interleaved.cpp`

### Host cleanup
- **SDPA** (`sdpa_program_factory.cpp` line 341-342, `sdpa_decode_program_factory.cpp` line 664-665, `sdpa_windowed_program_factory.cpp` line 280-281): Remove `packed_identity_scalar` computation. **BUT** `identity_scalar_packed` is also used by `generate_bcast_col_scalar()` in the same kernels → keep the host arg, just remove the kernel's usage of it for reduce.
- **MOE** (`moe_program_factory.cpp` lines 167-169): Remove `packed_identity_scalar` computation and CT arg slot. Shift subsequent CT arg indices.
- **Sampling** (`sampling_program_factory.cpp` lines 239-240, 262): Remove `packed_identity_scalar` computation and CT arg slot. Shift subsequent CT arg indices.

### Special case
`writer_decode_all.cpp` has an **additional** call: `generate_reduce_scaler_legacy(cb_zero_in, zero_scalar_packed)` (scaler = 0.0). Replace this with `prepare_reduce_scaler<cb_zero_in>(0.0f)`.

### Testing
**Sanity check:**
```bash
pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_scaled_dot_product_attention.py::test_sdpa_tt[1-8-1-256-128-128-128-BFLOAT16-False-DRAM] -v
```

**Full test suite:**
```bash
pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_scaled_dot_product_attention.py -v
pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_scaled_dot_product_attention_decode.py::test_sdpa_decode -v
pytest tests/ttnn/unit_tests/operations/reduce/test_moe.py -v
pytest tests/ttnn/unit_tests/operations/eltwise/test_sampling.py -v
```

---

## Phase 4: Softmax (5 files) → `generate_reduce_scaler`

The reduce scaler in softmax is always 1.0 (hardcoded as `0x3f803f80` on the host). `generate_reduce_scaler` with `PoolType::SUM` auto-produces 1.0.

### Kernel files
1. `ttnn/cpp/ttnn/operations/normalization/softmax/device/kernels/attention/dataflow/reader_unary_interleaved_sm.cpp`
2. `ttnn/cpp/ttnn/operations/normalization/softmax/device/kernels/attention/dataflow/reader_unary_interleaved_sm_large_tensor.cpp`
3. `ttnn/cpp/ttnn/operations/normalization/softmax/device/kernels/attention/dataflow/reader_unary_sharded_sm.cpp` (4 calls)
4. `ttnn/cpp/ttnn/operations/normalization/softmax/device/kernels/attention/dataflow/reader_unary_sharded_sm_causal_mask_hw_dims.cpp`
5. `ttnn/cpp/ttnn/operations/normalization/softmax/device/kernels/attention/dataflow/reader_unary_sharded_sm_rm_mask.cpp`

### Host cleanup
- `softmax_program_factory_attention_optimized_sharded.cpp` (lines 294, 334): Remove hardcoded `0x3f803f80` RT arg at position 0. Shift subsequent RT arg indices (position 1 `s.u` becomes position 0, etc.).
- `softmax_program_factory_attention_optimized.cpp` (lines 293-295, 334, 353): Same — remove `0x3f803f80` from reader args.
- Update kernel `get_arg_val` indices to match shifted args.

### Testing
**Sanity check:**
```bash
pytest tests/ttnn/unit_tests/operations/fused/test_softmax.py::test_softmax[1-32-32--1] -v
```

**Full test suite:**
```bash
pytest tests/ttnn/unit_tests/operations/fused/test_softmax.py -v
```

---

## Phase 5: Layernorm (9 files) → `prepare_reduce_scaler`

Mixed scalers: interleaved variants use 1.0, sharded variants use variable scalers (`1/block_w`, `1/num_blocks`). Using `prepare_reduce_scaler` for all since sharded variants need host-passed values.

### Kernel files
1. `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/reader_unary_interleaved_ln.cpp`
2. `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/reader_unary_interleaved_ln_large_tensor.cpp`
3. `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/reader_unary_interleaved_ln_rm_gb.cpp`
4. `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/writer_unary_sharded_ln.cpp` (2 calls: scalar_w, scalar_c)
5. `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/writer_unary_sharded_ln_pre_all_gather.cpp` (2 calls)
6. `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/writer_unary_sharded_ln_rm_gb.cpp` (2 calls: scalar_w, scalar_c)
7. `ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/dataflow/reader_layernorm_preallgather_2d.cpp` (2 calls)
8. `ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/dataflow/reader_unary_interleaved_ln_rm_gb_post_allgather.cpp`
9. `ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/dataflow/reader_unary_interleaved_ln_rm_gb_pre_allgather.cpp`

### Host files to change
- `ttnn/cpp/ttnn/operations/normalization/layernorm/device/layernorm_op_multi_core.cpp` (line 505: `packed_one_value` → pass float 1.0 bits)
- `ttnn/cpp/ttnn/operations/normalization/layernorm/device/layernorm_op_multi_core_sharded.cpp` (lines 1068-1076: `packed_winv_value`, `packed_cinv_value` → pass float bits)
- `ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/layernorm_pre_all_gather_program_factory.cpp` (lines 225-227: `packed_winv_value` → float bits)
- `ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/layernorm_post_all_gather_program_factory.cpp` (lines 433-435: `packed_winv_value` → float bits)

### Testing
**Sanity check:**
```bash
pytest tests/ttnn/unit_tests/operations/fused/test_layer_norm.py::test_layer_norm[32-64-True-BFLOAT16] -v
```

**Full test suite:**
```bash
pytest tests/ttnn/unit_tests/operations/fused/test_layer_norm.py -v
pytest tests/ttnn/unit_tests/operations/fused/test_distributed_layernorm.py -v
pytest tests/ttnn/unit_tests/operations/fused/test_distributed_layernorm_sharded.py -v
```

---

## Phase 6: GroupNorm + Experimental + Tests (10 files) → `generate_reduce_scaler`

All scalers are uniform across cores. GroupNorm uses `1/sqrt(N)` which maps to `PoolType::AVG, ReduceDim::REDUCE_SCALAR, reduce_factor=N`. Most others are SUM with scaler=1.0 (Pattern A).

### Kernel files & migration strategy
1. `ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/writer_unary_gn_rm_gb.cpp` (2 calls: scalar_w, scalar_c)
   - scalar_w = `1/sqrt(num_rows * num_channels_per_group)` → `AVG, REDUCE_SCALAR, reduce_factor=num_rows*num_channels_per_group`
   - scalar_c = `1/sqrt(num_cores_per_batch * num_cores_per_group)` → `AVG, REDUCE_SCALAR, reduce_factor=num_cores_per_batch*num_cores_per_group`
   - Move from RT args to CT args (reduce_factor for each)
2. `ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/dataflow/writer_unary_sharded_gn_rm_gb_v2.cpp` (2 calls) — same as above
3. `ttnn/cpp/ttnn/operations/experimental/reduction/fast_reduce_nc/device/kernels/reader_reduce_nc.cpp`
   - scaler=0 hardcoded → `generate_reduce_scaler<cb, PoolType::SUM, ReduceDim::REDUCE_COL>(0.0f)` (kernel-only)
4. `ttnn/cpp/ttnn/operations/experimental/ssm/hc_sum_reduce/device/kernels/reader_ssm_1d_sum_reduce.cpp`
   - scaler=1.0 CT arg → Pattern A: `generate_reduce_scaler<cb, PoolType::SUM, ReduceDim::REDUCE_ROW>()`, remove scaler CT arg
5. `ttnn/cpp/ttnn/operations/experimental/transformer/dit_layernorm_pre_all_gather/device/kernels/dataflow/reader_layernorm_preallgather_dit.cpp`
   - scaler=1.0 RT arg (uniform) → Pattern A: `generate_reduce_scaler<cb, PoolType::SUM, ReduceDim::REDUCE_ROW>()`, remove scaler RT arg
6. `ttnn/cpp/ttnn/operations/experimental/transformer/fused_distributed_rmsnorm/device/kernels/dataflow/rms_post_allgather_reader.cpp`
   - scaler=1/(W*num_devices) CT arg → `AVG, REDUCE_ROW, reduce_factor=W*num_devices`
7. `ttnn/cpp/ttnn/operations/experimental/transformer/fused_distributed_rmsnorm/device/kernels/dataflow/rms_pre_allgather_reader.cpp`
   - scaler=1.0 CT arg → Pattern A: `generate_reduce_scaler<cb, PoolType::SUM, ReduceDim::REDUCE_ROW>()`, remove scaler CT arg
8. `tests/ttnn/unit_tests/gtests/udm/reduction/interleaved/kernels/dataflow_reduce.cpp`
   - scaler=1.0 CT arg → Pattern A
9. `tests/ttnn/unit_tests/gtests/udm/reduction/sharded/kernels/reader_receiver_unary_sharded_reduce.cpp`
   - scaler=1.0 CT arg → Pattern A
10. `tests/ttnn/unit_tests/gtests/udm/reduction/sharded/kernels/reader_sender_unary_sharded_reduce.cpp`
    - scaler=1.0 CT arg → Pattern A

### Host files to change
- `ttnn/cpp/ttnn/operations/normalization/groupnorm/device/groupnorm_mcast_program_factory.cpp`: Replace packed scalers with reduce_factor CT args
- `ttnn/cpp/ttnn/operations/experimental/ssm/hc_sum_reduce/device/hc_sum_reduce_program_factory.cpp`: Remove scaler CT arg
- `ttnn/cpp/ttnn/operations/experimental/reduction/fast_reduce_nc/device/fast_reduce_nc_program_factory.cpp`: No host change (scaler hardcoded in kernel)
- `ttnn/cpp/ttnn/operations/experimental/transformer/dit_layernorm_pre_all_gather/device/`: Remove scaler RT arg
- `ttnn/cpp/ttnn/operations/experimental/transformer/fused_distributed_rmsnorm/device/`: Replace packed scaler with reduce_factor CT arg
- UDM test host setup files in `tests/ttnn/unit_tests/gtests/udm/reduction/`: Remove scaler CT args

### Testing
**Sanity check:**
```bash
pytest tests/ttnn/unit_tests/operations/fused/test_group_norm.py::test_group_norm_with_height_sharded[1-320-32-32-16-True] -v
```

**Full test suite:**
```bash
pytest tests/ttnn/unit_tests/operations/fused/test_group_norm.py -v
pytest tests/ttnn/unit_tests/operations/reduce/test_fast_reduce_nc.py -v
pytest tests/ttnn/nightly/unit_tests/operations/ssm/test_ssm_1d_sum_reduce.py -v
pytest tests/ttnn/nightly/unit_tests/operations/transformers/test_distributed_dit_layernorm.py -v
pytest tests/ttnn/nightly/unit_tests/operations/transformers/test_distributed_fused_rmsnorm.py -v
```
UDM gtests (requires multi-device):
```bash
./build/test/ttnn/unit_tests_ttnn_udm --gtest_filter="*TestWidthReduction*"
```

---

## Post-Migration Cleanup

1. Remove `generate_reduce_scaler_legacy` from `reduce_helpers_dataflow.hpp` and `.inl`
2. Remove local `pack_two_bfloat16_into_uint32` from `layernorm_op_multi_core.cpp` (line 42) if no longer used
3. Grep for any remaining references to ensure nothing was missed
4. Remove `#include "ttnn/kernel/dataflow/generate_reduce_scaler.hpp"` from files that used it only for the old API
