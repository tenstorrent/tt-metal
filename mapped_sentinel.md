# Compute Kernel API Sentinel Mapping

This document maps all functions listed in `compute_kernel_api_reference.md` to their corresponding `init` and `uninit` calls. Each operation in the compute API has associated initialization and uninitialization functions that need to be identified and connected.

## Mapping Overview

Operations fall into three categories:
1. **Operations with both init and uninit**: These operations modify hardware state that persists and must be cleaned up (e.g., `tilize`, `untilize`, `reduce`, `pack_untilize`, `pack_rows`)
2. **Operations with only init**: Most tile operations only require initialization (e.g., `tanh_tile`, `sigmoid_tile`, `relu_tile`)
3. **Operations with neither init nor uninit**: Utility functions, debug functions, and simple operations that don't require initialization

---

## Main API Files

### `compute_kernel_api.h`

#### Mathematical Functions

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `sigmoid_tile<vec_mode, fast_and_approx>(uint32_t idst)` | `sigmoid_tile_init<fast_and_approx>()` | None | |
| `log_tile<fast_and_approx>(uint32_t idst)` | `log_tile_init<fast_and_approx>()` | None | |
| `log_with_base_tile<fast_and_approx>(uint32_t idst, uint32_t base_scale)` | `log_with_base_tile_init<fast_and_approx>()` | None | |
| `tanh_tile<fast_and_approx>(uint32_t idst)` | `tanh_tile_init<fast_and_approx>()` | None | |
| `exp2_tile(uint32_t idst)` | `exp2_tile_init()` | None | |
| `expm1_tile<approx>(uint32_t idst)` | `expm1_tile_init<approx>()` | None | |

#### Sign and Absolute Value

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `signbit_tile(uint32_t idst)` | `signbit_tile_init()` | None | |
| `signbit_tile_int32(uint32_t idst)` | `signbit_tile_init()` | None | Uses same init |
| `abs_tile(uint32_t idst)` | `abs_tile_init()` | None | |
| `abs_tile_int32(uint32_t idst)` | `abs_tile_init()` | None | Uses same init |
| `sign_tile(uint32_t idst)` | `sign_tile_init()` | None | |

#### Power and Square

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `square_tile(uint32_t idst)` | `square_tile_init()` | None | |
| `power_tile(uint32_t idst, uint32_t param0)` | `power_tile_init()` | None | |
| `tiled_prod_tile(uint32_t idst)` | `tiled_prod_tile_init()` | None | |

#### Max/Min Operations

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `max_tile(uint32_t idst0, uint32_t idst1, int vector_mode)` | `max_tile_init()` | None | |
| `unary_max_tile(uint32_t idst, uint32_t param0)` | `unary_max_tile_init()` | None | |
| `unary_max_int32_tile(uint32_t idst, uint32_t param0)` | `unary_max_tile_init()` | None | Uses same init |
| `unary_min_tile(uint32_t idst, uint32_t param0)` | `unary_min_tile_init()` | None | |
| `unary_min_int32_tile(uint32_t idst, uint32_t param0)` | `unary_min_tile_init()` | None | Uses same init |

#### Special Functions

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `heaviside_tile(uint32_t idst, uint32_t param0)` | `heaviside_tile_init()` | None | |
| `silu_tile(uint32_t idst)` | `silu_tile_init()` | None | |
| `alt_complex_rotate90_tile(uint32_t idst)` | `alt_complex_rotate90_tile_init()` | None | |

#### TopK Operations

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `topk_local_sort<stable_sort>(...)` | `topk_tile_init()` | None | |
| `topk_merge<idir, stable_sort>(...)` | `topk_tile_init()` | None | Uses same init |
| `topk_rebuild<stable_sort>(...)` | `topk_tile_init()` | None | Uses same init |

#### Reduction Operations

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `max_reduce_with_indices<num_rows, layout, ITERATIONS>(...)` | `max_reduce_with_indices_init<layout>()` | None | |
| `sfpu_reduce<pool_type, format, reduce_dim>(uint32_t idst)` | `sfpu_reduce_init<pool_type, format>()` | None | |
| `sfpu_add_top_row<format>(...)` | `sfpu_add_top_row_init()` | None | |

#### Debug Functions

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `dbg_halt()` | None | None | Debug function, no init/uninit |
| `dbg_unhalt()` | None | None | Debug function, no init/uninit |
| `dbg_read_dest_acc_row(...)` | None | None | Debug function, no init/uninit |

#### Special Value Flags

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `get_compute_special_value_flags()` | None | None | Utility function |
| `get_compute_special_value_flags_fpu(...)` | None | None | Utility function |
| `get_compute_special_value_flags_sfpu(...)` | None | None | Utility function |
| `clear_compute_special_value_flags()` | None | None | Utility function |
| `store_compute_special_value_flags_to_l1(...)` | None | None | Utility function |

### `add_int_sfpu.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `add_int32_tile(...)` | `add_int_tile_init()` | None | |
| `add_uint16_tile(...)` | `add_int_tile_init()` | None | Uses same init |
| `add_uint32_tile(...)` | `add_int_tile_init()` | None | Uses same init |

### `bcast.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `unary_bcast<BroadcastType bcast_type>(...)` | `unary_bcast_init<BroadcastType bcast_type>(...)` | None | |
| `reconfigure_unary_bcast<old_bcast_type, new_bcast_type>(...)` | None | None | Reconfiguration function |
| `sub_tiles_bcast_cols(...)` | `init_bcast<EltwiseBinaryType::ELWSUB, BroadcastType::COL>(...)` | None | |
| `sub_tiles_bcast_scalar(...)` | `init_bcast<EltwiseBinaryType::ELWSUB, BroadcastType::SCALAR>(...)` or `sub_bcast_scalar_init_short(...)` | None | |
| `mul_tiles_bcast_cols(...)` | `init_bcast<EltwiseBinaryType::ELWMUL, BroadcastType::COL>(...)` or `mul_bcast_cols_init_short(...)` | None | |
| `mul_tiles_bcast_rows(...)` | `init_bcast<EltwiseBinaryType::ELWMUL, BroadcastType::ROW>(...)` or `mul_bcast_rows_init_short(...)` | None | |
| `add_tiles_bcast_rows(...)` | `init_bcast<EltwiseBinaryType::ELWADD, BroadcastType::ROW>(...)` or `add_bcast_rows_init_short(...)` | None | |
| `add_tiles_bcast_cols(...)` | `init_bcast<EltwiseBinaryType::ELWADD, BroadcastType::COL>(...)` or `add_bcast_cols_init_short(...)` | None | |
| `add_tiles_bcast_scalar(...)` | `init_bcast<EltwiseBinaryType::ELWADD, BroadcastType::SCALAR>(...)` or `add_bcast_scalar_init_short(...)` | None | |
| `mul_tiles_bcast_scalar(...)` | `init_bcast<EltwiseBinaryType::ELWMUL, BroadcastType::SCALAR>(...)` or `mul_tiles_bcast_scalar_init_short(...)` | None | |

### `binary_bitwise_sfpu.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `bitwise_and_binary_tile(...)` | `binary_bitwise_tile_init()` | None | |
| `bitwise_and_uint32_binary_tile(...)` | `binary_bitwise_tile_init()` | None | Uses same init |
| `bitwise_and_uint16_binary_tile(...)` | `binary_bitwise_tile_init()` | None | Uses same init |
| `bitwise_or_binary_tile(...)` | `binary_bitwise_tile_init()` | None | |
| `bitwise_or_uint32_binary_tile(...)` | `binary_bitwise_tile_init()` | None | Uses same init |
| `bitwise_or_uint16_binary_tile(...)` | `binary_bitwise_tile_init()` | None | Uses same init |
| `bitwise_xor_binary_tile(...)` | `binary_bitwise_tile_init()` | None | |
| `bitwise_xor_uint32_binary_tile(...)` | `binary_bitwise_tile_init()` | None | Uses same init |
| `bitwise_xor_uint16_binary_tile(...)` | `binary_bitwise_tile_init()` | None | Uses same init |

### `binary_comp.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `lt_int32_tile(...)` | `lt_int32_tile_init()` | None | |
| `gt_int32_tile(...)` | `gt_int32_tile_init()` | None | |
| `ge_int32_tile(...)` | `ge_int32_tile_init()` | None | |
| `le_int32_tile(...)` | `le_int32_tile_init()` | None | |

### `binary_max_min.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `binary_max_int32_tile(...)` | `binary_max_tile_init()` | None | |
| `binary_max_tile(...)` | `binary_max_tile_init()` | None | |
| `binary_min_int32_tile(...)` | `binary_min_tile_init()` | None | |
| `binary_min_tile(...)` | `binary_min_tile_init()` | None | |

### `binary_shift.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `binary_left_shift_tile(...)` | `binary_shift_tile_init()` | None | |
| `binary_left_shift_uint32_tile<...>(...)` | `binary_shift_tile_init()` | None | Uses same init |
| `binary_left_shift_int32_tile(...)` | `binary_shift_tile_init()` | None | Uses same init |
| `binary_right_shift_tile(...)` | `binary_shift_tile_init()` | None | |
| `binary_right_shift_uint32_tile<...>(...)` | `binary_shift_tile_init()` | None | Uses same init |
| `binary_right_shift_int32_tile(...)` | `binary_shift_tile_init()` | None | Uses same init |
| `binary_logical_right_shift_tile(...)` | `binary_shift_tile_init()` | None | |
| `binary_logical_right_shift_uint32_tile<...>(...)` | `binary_shift_tile_init()` | None | Uses same init |
| `binary_logical_right_shift_int32_tile(...)` | `binary_shift_tile_init()` | None | Uses same init |

### `cb_api.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `cb_wait_front(...)` | None | None | CB utility function |
| `cb_pop_front(...)` | None | None | CB utility function |
| `cb_reserve_back(...)` | None | None | CB utility function |
| `cb_push_back(...)` | None | None | CB utility function |
| `get_tile_address(...)` | None | None | CB utility function |
| `read_tile_value(...)` | None | None | CB utility function |

### `common.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `get_arg_addr(...)` | None | None | Utility function |
| `get_common_arg_addr(...)` | None | None | Utility function |
| `get_arg_val<T>(...)` | None | None | Utility function |
| `get_common_arg_val<T>(...)` | None | None | Utility function |
| `get_absolute_logical_x()` | None | None | Utility function |
| `get_absolute_logical_y()` | None | None | Utility function |
| `get_relative_logical_x()` | None | None | Utility function |
| `get_relative_logical_y()` | None | None | Utility function |

### `compute_kernel_hw_startup.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `compute_kernel_hw_startup(uint32_t icb0, uint32_t icb1, uint32_t ocb)` | None | None | Hardware startup, no uninit needed |
| `compute_kernel_hw_startup(uint32_t icb0, uint32_t ocb)` | None | None | Hardware startup, no uninit needed |

### `copy_dest_values.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `copy_dest_values(uint32_t idst0, uint32_t idst1)` | `copy_dest_values_init()` | None | |

### `cumsum.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `cumsum_tile(uint32_t idst, bool first = true)` | `cumsum_tile_init()` | None | |

### `div_int32_floor.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `div_int32_floor_tile(...)` | `div_int32_floor_tile_init()` | None | |
| `div_int32_trunc_tile(...)` | `div_int32_trunc_tile_init()` | None | |

### `div_int32_sfpu.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `div_int32_tile(...)` | `div_int32_tile_init()` | None | |

### `eltwise_binary_sfpu.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `add_binary_tile(...)` | `add_binary_tile_init()` | None | |
| `sub_binary_tile(...)` | `sub_binary_tile_init()` | None | |
| `mul_binary_tile(...)` | `mul_binary_tile_init()` | None | |
| `div_binary_tile(...)` | `div_binary_tile_init()` | None | |
| `rsub_binary_tile(...)` | `rsub_binary_tile_init()` | None | |
| `power_binary_tile(...)` | `power_binary_tile_init()` | None | |

### `eltwise_binary.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `binary_op_init_common(...)` | None | None | Common init helper |
| `binary_tiles_init<...>(...)` | None | None | Template init function |
| `mul_tiles_init(...)` | None | None | Init function |
| `add_tiles_init(...)` | None | None | Init function |
| `sub_tiles_init(...)` | None | None | Init function |
| `mul_tiles(...)` | `mul_tiles_init(...)` | None | |
| `add_tiles(...)` | `add_tiles_init(...)` | None | |
| `sub_tiles(...)` | `sub_tiles_init(...)` | None | |
| `binary_dest_reuse_tiles<...>(...)` | `binary_dest_reuse_tiles_init<...>(...)` | None | |

### `ema.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `ema_tile(uint32_t input_dst_index)` | `ema_init(uint32_t alpha, uint32_t beta)` | None | |
| `ema_clear_previous_output()` | None | None | Clear function, not uninit |

### `gcd.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `gcd_tile(...)` | `gcd_tile_init()` | None | |

### `lcm.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `lcm_tile(...)` | `lcm_tile_init()` | None | |

### `logsigmoid.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `logsigmoid_tile(...)` | `logsigmoid_tile_init()` | None | |

### `mask.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `mask_tile(...)` | `mask_tile_init()` | None | |
| `mask_posinf_tile(...)` | `mask_tile_init()` | None | Uses same init |

### `matmul.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `matmul_tiles(...)` | `mm_init(...)` or `mm_init_short(...)` or `mm_init_short_with_dt(...)` | None | |
| `matmul_tiles_math<num_faces>(...)` | `mm_init(...)` or `mm_init_short(...)` | None | Uses same init |
| `matmul_block(...)` | `mm_block_init(...)` or `mm_block_init_short(...)` or `mm_block_init_short_with_dt(...)` or `mm_block_init_short_with_both_dt(...)` | None | |
| `matmul_block_math_dynamic_throttle(...)` | `mm_block_init(...)` or variants | None | Uses same init |

### `mul_int_sfpu.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `mul_uint16_tile(...)` | `mul_int_tile_init()` | None | |

### `mul_int32_sfpu.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `mul_int32_tile(...)` | `mul_int32_tile_init()` | None | |
| `mul_uint32_tile(...)` | `mul_int32_tile_init()` | None | Uses same init |

### `pack_untilize.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `pack_untilize_block<...>(...)` | `pack_untilize_init<...>(...)` or `pack_untilize_dest_init<...>(...)` | `pack_untilize_uninit(uint32_t ocb)` | **Has uninit** |
| `pack_untilize_dest<...>(...)` | `pack_untilize_dest_init<...>(...)` | `pack_untilize_uninit(uint32_t ocb)` | **Has uninit** |

### `pack.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `pack_tile<...>(...)` | None | None | Standard pack operation |
| `pack_tile_block(...)` | None | None | Standard pack operation |
| `pack_reconfig_data_format(...)` | None | None | Reconfiguration function |
| `pack_reconfig_l1_acc(...)` | None | None | Reconfiguration function |
| `pack_rows(...)` | `pack_rows_init(uint32_t num_rows)` | `pack_rows_uninit()` | **Has uninit** |

### `quantization.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `quant_tile(...)` | `quant_tile_init(uint32_t zero_point)` | None | |
| `requant_tile(...)` | `requant_tile_init(uint32_t zero_point)` | None | |
| `dequant_tile(...)` | `dequant_tile_init(uint32_t zero_point)` | None | |

### `reconfig_data_format.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `reconfig_data_format<...>(...)` | None | None | Reconfiguration function |
| `reconfig_data_format_srca<...>(...)` | None | None | Reconfiguration function |
| `reconfig_data_format_srcb<...>(...)` | None | None | Reconfiguration function |

### `reduce_custom.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `reduce_block_max_row<block_ct_dim>(...)` | `reduce_block_max_row_init<block_ct_dim>()` | `reduce_block_max_row_uninit<clear_fp32_accumulation>()` | **Has uninit** |

### `reduce.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `reduce_tile<...>(...)` | `reduce_init<reduce_type, reduce_dim, enforce_fp32_accumulation>(...)` | `reduce_uninit<enforce_fp32_accumulation>()` | **Has uninit** |
| `reduce_tile_math<...>(...)` | `reduce_init<...>(...)` | `reduce_uninit<enforce_fp32_accumulation>()` | **Has uninit**, uses same init/uninit |

### `reg_api.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `acquire_dst()` | None | None | Deprecated, use tile_regs_acquire |
| `tile_regs_acquire()` | None | None | Register management |
| `tile_regs_wait()` | None | None | Register management |
| `release_dst()` | None | None | Deprecated, use tile_regs_commit/release |
| `tile_regs_commit()` | None | None | Register management |
| `tile_regs_release()` | None | None | Register management |

### `reshuffle.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `reshuffle_rows_tile(...)` | `reshuffle_rows_tile_init()` | None | |

### `sub_int_sfpu.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `sub_int32_tile(...)` | `sub_int_tile_init()` | None | |
| `sub_uint32_tile(...)` | `sub_int_tile_init()` | None | Uses same init |
| `sub_uint16_tile(...)` | `sub_int_tile_init()` | None | Uses same init |
| `rsub_int32_tile(...)` | `rsub_int_tile_init()` | None | |
| `rsub_uint32_tile(...)` | `rsub_int_tile_init()` | None | Uses same init |
| `rsub_uint16_tile(...)` | `rsub_int_tile_init()` | None | Uses same init |

### `tile_move_copy.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `copy_tile(...)` | `copy_tile_init(uint32_t cbid)` or `copy_tile_to_dst_init_short(...)` or `copy_tile_to_dst_init_short_with_dt(...)` | None | |
| `copy_block_matmul_partials(...)` | `copy_tile_init(...)` | None | Uses same init |

### `tilize.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `tilize_block(...)` | `tilize_init(...)` or `tilize_init_no_pack(...)` or `tilize_init_short_with_dt(...)` or `tilize_init_short_with_dt_no_pack(...)` | `tilize_uninit(...)` or `tilize_uninit_no_pack(...)` or `tilize_uninit_with_dt(...)` or `tilize_uninit_with_dt_no_pack(...)` | **Has uninit** |
| `tilize_block_no_pack(...)` | `tilize_init_no_pack(...)` or `tilize_init_short_with_dt_no_pack(...)` | `tilize_uninit_no_pack(...)` or `tilize_uninit_with_dt_no_pack(...)` | **Has uninit** |
| `unpack_tilizeA_B_block<...>(...)` | `tilizeA_B_reduce_init<...>(...)` | `unpack_tilizeA_B_uninit(uint32_t icb)` | **Has uninit** |
| `fast_tilize_block(...)` | `fast_tilize_init(...)` or `fast_tilize_init_with_dt(...)` | `fast_tilize_uninit(...)` | **Has uninit** |

### `transpose_wh_dest.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `transpose_wh_dest<is_32bit>(...)` | `transpose_wh_dest_init_short<is_32bit>()` | None | |

### `transpose_wh.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `transpose_wh_tile(...)` | `transpose_wh_init(...)` or `transpose_wh_init_short(...)` | None | |

### `untilize.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `untilize_block<block_ct_dim>(...)` | `untilize_init(uint32_t icb)` | `untilize_uninit(uint32_t icb)` | **Has uninit** |

### `welford.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `welford_update<reciprocal_size>(...)` | `welford_init()` | None | Uses `welford_clear()` to reset state |
| `welford_update_rows<reciprocal_size>(...)` | `welford_init()` | None | Uses `welford_clear()` to reset state |
| `welford_save_state(...)` | `welford_init()` | None | Uses same init |
| `welford_restore_state(...)` | `welford_init()` | None | Uses same init |
| `welford_finalize_to_row<reciprocal_size>(...)` | `welford_init()` | None | Uses same init |
| `welford_finalize_to_face<reciprocal_size>(...)` | `welford_init()` | None | Uses same init |
| `welford_clear()` | None | None | Clear function, not uninit |

### `xlogy.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `xlogy_binary_tile(...)` | `xlogy_binary_tile_init()` | None | |

---

## Eltwise Unary Operations

### `eltwise_unary/eltwise_unary.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `unary_op_init_common(...)` | None | None | Common init helper |
| `unary_op_init_common_no_pack(...)` | None | None | Common init helper |
| `init_sfpu(...)` | None | None | SFPU init helper, used by many unary ops |

### `eltwise_unary/activations.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `hardsigmoid_tile(...)` | `hardsigmoid_tile_init()` | None | |
| `softsign_tile(...)` | `softsign_tile_init()` | None | |
| `celu_tile(...)` | `celu_tile_init()` | None | |
| `softshrink_tile(...)` | `softshrink_tile_init()` | None | |

### `eltwise_unary/binop_with_scalar.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `add_unary_tile(...)` | `binop_with_scalar_tile_init()` | None | |
| `sub_unary_tile(...)` | `binop_with_scalar_tile_init()` | None | Uses same init |
| `mul_unary_tile(...)` | `binop_with_scalar_tile_init()` | None | Uses same init |
| `div_unary_tile(...)` | `binop_with_scalar_tile_init()` | None | Uses same init |
| `rsub_unary_tile(...)` | `binop_with_scalar_tile_init()` | None | Uses same init |
| `add_unary_tile_int32(...)` | `binop_with_scalar_tile_init()` | None | Uses same init |
| `sub_unary_tile_int32(...)` | `binop_with_scalar_tile_init()` | None | Uses same init |

### `eltwise_unary/exp.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `exp_tile<...>(...)` | `exp_tile_init<approx, fast_and_approx, scale>()` | None | |

### `eltwise_unary/recip.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `recip_tile<legacy_compat>(...)` | `recip_tile_init<legacy_compat>()` | None | |

### `eltwise_unary/sqrt.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `sqrt_tile<FAST_APPROX>(...)` | `sqrt_tile_init()` | None | |

### `eltwise_unary/rsqrt.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `rsqrt_tile<legacy_compat, FAST_APPROX>(...)` | `rsqrt_tile_init<legacy_compat>()` | None | |

### `eltwise_unary/trigonometry.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `sin_tile(...)` | `sin_tile_init()` | None | |
| `cos_tile(...)` | `cos_tile_init()` | None | |
| `tan_tile(...)` | `tan_tile_init()` | None | |
| `asin_tile(...)` | `asin_tile_init()` | None | |
| `acos_tile(...)` | `acos_tile_init()` | None | |
| `atan_tile(...)` | `atan_tile_init()` | None | |
| `sinh_tile(...)` | `sinh_tile_init()` | None | |
| `cosh_tile(...)` | `cosh_tile_init()` | None | |
| `asinh_tile(...)` | `asinh_tile_init()` | None | |
| `acosh_tile(...)` | `acosh_tile_init()` | None | |
| `atanh_tile(...)` | `atanh_tile_init()` | None | |

### `eltwise_unary/gelu.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `gelu_tile<fast_and_approx>(...)` | `gelu_tile_init<fast_and_approx>()` | None | |

### `eltwise_unary/relu.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `relu_tile(...)` | `relu_tile_init()` | None | |
| `relu_tile_int32(...)` | `relu_tile_init()` | None | Uses same init |
| `relu_max_tile(...)` | `relu_max_tile_init()` | None | |
| `relu_max_tile_int32(...)` | `relu_max_tile_init()` | None | Uses same init |
| `relu_min_tile(...)` | `relu_min_tile_init()` | None | |
| `relu_min_tile_int32(...)` | `relu_min_tile_init()` | None | Uses same init |
| `leaky_relu_tile(...)` | `leaky_relu_tile_init()` | None | |

### `eltwise_unary/fill.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `fill_tile(...)` | `fill_tile_init()` | None | |
| `fill_tile_int(...)` | `fill_tile_init()` | None | Uses same init |
| `fill_tile_bitcast(...)` | `fill_tile_init()` | None | Uses same init |

### `eltwise_unary/typecast.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `typecast_tile<IN_DTYPE, OUT_DTYPE>(...)` | `typecast_tile_init<IN_DTYPE, OUT_DTYPE>()` | None | |

### `eltwise_unary/comp.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `unary_eq_tile(...)` | `unary_eq_tile_init()` | None | |
| `unary_eq_tile_int32(...)` | `unary_eq_tile_init()` | None | Uses same init |
| `unary_ne_tile(...)` | `unary_ne_tile_init()` | None | |
| `unary_ne_tile_int32(...)` | `unary_ne_tile_init()` | None | Uses same init |
| `unary_gt_tile(...)` | `unary_gt_tile_init()` | None | |
| `unary_gt_tile_int32(...)` | `unary_gt_tile_init()` | None | Uses same init |
| `unary_ge_tile(...)` | `unary_ge_tile_init()` | None | |
| `unary_ge_tile_int32(...)` | `unary_ge_tile_init()` | None | Uses same init |
| `unary_lt_tile(...)` | `unary_lt_tile_init()` | None | |
| `unary_lt_tile_int32(...)` | `unary_lt_tile_init()` | None | Uses same init |
| `unary_le_tile(...)` | `unary_le_tile_init()` | None | |
| `unary_le_tile_int32(...)` | `unary_le_tile_init()` | None | Uses same init |
| `gtz_tile(...)` | `gtz_tile_init()` | None | |
| `gtz_tile_int32(...)` | `gtz_tile_init()` | None | Uses same init |
| `gez_tile(...)` | `gez_tile_init()` | None | |
| `gez_tile_int32(...)` | `gez_tile_init()` | None | Uses same init |
| `ltz_tile(...)` | `ltz_tile_init()` | None | |
| `ltz_tile_int32(...)` | `ltz_tile_init()` | None | Uses same init |
| `lez_tile(...)` | `lez_tile_init()` | None | |
| `lez_tile_int32(...)` | `lez_tile_init()` | None | Uses same init |
| `eqz_tile(...)` | `eqz_tile_init()` | None | |
| `eqz_tile_int32(...)` | `eqz_tile_init()` | None | Uses same init |
| `eqz_tile_uint16(...)` | `eqz_tile_init()` | None | Uses same init |
| `eqz_tile_uint32(...)` | `eqz_tile_init()` | None | Uses same init |
| `nez_tile(...)` | `nez_tile_init()` | None | |
| `nez_tile_int32(...)` | `nez_tile_init()` | None | Uses same init |
| `nez_tile_uint16(...)` | `nez_tile_init()` | None | Uses same init |
| `nez_tile_uint32(...)` | `nez_tile_init()` | None | Uses same init |

### `eltwise_unary/bitwise_*.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `bitwise_and_tile(...)` | `bitwise_and_tile_init()` | None | |
| `bitwise_not_tile(...)` | `bitwise_not_tile_init()` | None | |
| `bitwise_or_tile(...)` | `bitwise_or_tile_init()` | None | |
| `bitwise_xor_tile(...)` | `bitwise_xor_tile_init()` | None | |
| *(Additional typed variants)* | *(Corresponding init functions)* | None | |

### `eltwise_unary/where.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `where_tile(...)` | `where_tile_init()` | None | |
| `where_fp32_tile(...)` | `where_tile_init()` | None | Uses same init |
| `where_int32_tile(...)` | `where_tile_init()` | None | Uses same init |
| `where_uint32_tile(...)` | `where_tile_init()` | None | Uses same init |

### `eltwise_unary/rsub.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `rsub_tile()` | `rsub_tile_init()` | None | |
| `rsub_unary_int32_tile(...)` | `rsub_unary_int32_tile_init()` | None | |

### `eltwise_unary/rdiv.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `rdiv_tile<rounding_mode>(...)` | `rdiv_tile_init()` | None | |

### `eltwise_unary/rounding.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `ceil_tile()` | `rounding_op_tile_init()` | None | |
| `floor_tile()` | `rounding_op_tile_init()` | None | Uses same init |
| `trunc_tile()` | `rounding_op_tile_init()` | None | Uses same init |
| `round_tile()` | `rounding_op_tile_init()` | None | Uses same init |
| `round_tile(uint32_t idst, int32_t decimals)` | `rounding_op_tile_init()` | None | Uses same init |
| `stochastic_round_tile()` | `rounding_op_tile_init()` | None | Uses same init |
| `frac_tile()` | `rounding_op_tile_init()` | None | Uses same init |

### `eltwise_unary/softplus.h`

| Operation Function | Init Function | Uninit Function | Notes |
|-------------------|---------------|-----------------|-------|
| `softplus_tile()` | `softplus_tile_init()` | None | |
| `softplus_tile(uint32_t idst, ...)` | `softplus_tile_init()` | None | Uses same init |

### Other `eltwise_unary/` Functions

| File | Operation Function | Init Function | Uninit Function | Notes |
|------|-------------------|---------------|-----------------|-------|
| `isinf_isnan.h` | `isinf_tile()`, `isposinf_tile()`, `isneginf_tile()`, `isnan_tile()`, `isfinite_tile()` | `isinf_tile_init()`, `isposinf_tile_init()`, `isneginf_tile_init()`, `isnan_tile_init()`, `isfinite_tile_init()` | None | |
| `identity.h` | `identity_tile()`, `identity_tile_uint32()` | `identity_tile_init()` | None | |
| `negative.h` | `negative_tile()`, `negative_tile_int32()` | `negative_tile_init()` | None | |
| `logical_not_noti.h` | `logical_not_unary_tile()`, various typed variants | `logical_not_unary_tile_init()` | None | |
| `log1p.h` | `log1p_tile<fast_and_approx>()` | `log1p_tile_init<fast_and_approx>()` | None | |
| `left_shift.h` | `left_shift_tile()` | `left_shift_tile_init()` | None | |
| `right_shift.h` | `right_shift_tile()` | `right_shift_tile_init()` | None | |
| `remainder.h` | `remainder_tile()` | `remainder_tile_init()` | None | |
| `fmod.h` | `fmod_tile()` | `fmod_tile_init()` | None | |
| `prelu.h` | `prelu_tile()` | `prelu_tile_init()` | None | |
| `rand.h` | `rand_tile()` | `rand_tile_init()` | None | |
| `i0.h` | `i0_tile()` | `i0_tile_init()` | None | |
| `i1.h` | `i1_tile()` | `i1_tile_init()` | None | |
| `erfinv.h` | `erfinv_tile()` | `erfinv_tile_init()` | None | |
| `erf_erfc.h` | `erf_tile()`, `erfc_tile()` | `erf_tile_init()`, `erfc_tile_init()` | None | |
| `elu.h` | `elu_tile()` | `elu_tile_init()` | None | |
| `dropout.h` | `dropout_tile()` | `dropout_kernel_init()` | None | |
| `clamp.h` | `clamp_tile()` | `clamp_tile_init()` | None | |
| `hardtanh.h` | `hardtanh_tile()` | `hardtanh_tile_init()` | None | |
| `hardmish.h` | `hardmish_tile()` | `hardmish_tile_init()` | None | |
| `rpow.h` | `rpow_tile()` | `rpow_tile_init()` | None | |
| `threshold.h` | `threshold_tile()` | `threshold_tile_init()` | None | |
| `selu.h` | `selu_tile()` | `selu_tile_init()` | None | |
| `cbrt.h` | `cbrt_tile()` | `cbrt_tile_init()` | None | |
| `reverseops.h` | Various reverse operations | Corresponding init functions | None | |
| `sfpu_int_sum.h` | `sfpu_sum_int_col()`, `sfpu_sum_int_row()`, `sfpu_add_int()` | `sfpu_sum_int_init()` | None | |

---

## Summary

### Operations with Both Init and Uninit

These operations modify hardware state that persists and must be cleaned up:

1. **tilize operations**: `tilize_init()` / `tilize_uninit()` and variants
2. **untilize operations**: `untilize_init()` / `untilize_uninit()`
3. **pack_untilize operations**: `pack_untilize_init()` / `pack_untilize_uninit()`
4. **reduce operations**: `reduce_init()` / `reduce_uninit()`
5. **reduce_block_max_row**: `reduce_block_max_row_init()` / `reduce_block_max_row_uninit()`
6. **pack_rows**: `pack_rows_init()` / `pack_rows_uninit()`

### Operations with Only Init

Most tile operations only require initialization (no uninit needed):

- All `*_tile()` operations with corresponding `*_tile_init()` functions
- Binary operations with `*_binary_tile_init()` or `*_tiles_init()` functions
- Broadcast operations with `*_bcast_init()` or `*_init_short()` functions
- Matmul operations with `mm_init()` or variants
- Copy operations with `copy_tile_init()` or variants

### Operations with Neither Init nor Uninit

- Utility functions (`get_arg_*`, `get_*_logical_*`, etc.)
- Debug functions (`dbg_*`)
- CB utility functions (`cb_*`)
- Register management functions (`tile_regs_*`)
- Reconfiguration functions (`*_reconfig_*`)
- Hardware startup (`compute_kernel_hw_startup`)
- Welford clear function (`welford_clear()` - not an uninit, but a state reset)

### Special Cases

- **Welford operations**: Use `welford_init()` for initialization, but use `welford_clear()` (not `welford_uninit()`) to reset state between operations
- **EMA operations**: Use `ema_init()` for initialization, and `ema_clear_previous_output()` (not `ema_uninit()`) to clear state
- **SFPU operations**: Many unary operations use `init_sfpu()` as a common initialization helper

---

## Notes

- Operations marked with **Has uninit** require explicit cleanup to restore hardware state
- Many operations share the same init function (e.g., int32 variants use the same init as their float counterparts)
- Template parameters in init functions must match those used in the operation function
- Some operations have multiple init variants (e.g., `_init()`, `_init_short()`, `_init_with_dt()`) - choose based on use case
- Always call uninit functions after completing operations that have them, before initializing other operations
- The order of init/uninit calls matters - follow the pattern: init → operation → uninit
