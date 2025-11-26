# Compute Kernel API Reference

This document lists all functions available in the `tt_metal/include/compute_kernel_api/` directory.

---

## Table of Contents

1. [Main API Files](#main-api-files)
2. [Eltwise Unary Operations](#eltwise-unary-operations)

---

## Main API Files

### `compute_kernel_api.h`

This is the main API file that includes many core mathematical and utility functions.

#### Mathematical Functions

| Function | Description |
|----------|-------------|
| `sigmoid_tile_init<fast_and_approx>()` | Init sigmoid operation |
| `sigmoid_tile<vec_mode, fast_and_approx>(uint32_t idst)` | Sigmoid activation |
| `log_tile_init<fast_and_approx>()` | Init logarithm operation |
| `log_tile<fast_and_approx>(uint32_t idst)` | Natural logarithm |
| `log_with_base_tile_init<fast_and_approx>()` | Init logarithm with base |
| `log_with_base_tile<fast_and_approx>(uint32_t idst, uint32_t base_scale)` | Logarithm with specified base |
| `tanh_tile_init<fast_and_approx>()` | Init hyperbolic tangent |
| `tanh_tile<fast_and_approx>(uint32_t idst)` | Hyperbolic tangent |
| `exp2_tile_init()` | Init 2^x operation |
| `exp2_tile(uint32_t idst)` | Compute 2^x |
| `expm1_tile_init<approx>()` | Init exp(x)-1 operation |
| `expm1_tile<approx>(uint32_t idst)` | Compute exp(x)-1 |

#### Sign and Absolute Value

| Function | Description |
|----------|-------------|
| `signbit_tile_init()` | Init sign bit operation |
| `signbit_tile(uint32_t idst)` | Set sign bit of tile |
| `signbit_tile_int32(uint32_t idst)` | Set sign bit for int32 |
| `abs_tile_init()` | Init absolute value |
| `abs_tile(uint32_t idst)` | Absolute value |
| `abs_tile_int32(uint32_t idst)` | Absolute value for int32 |
| `sign_tile_init()` | Init signum operation |
| `sign_tile(uint32_t idst)` | Signum function |

#### Power and Square

| Function | Description |
|----------|-------------|
| `square_tile_init()` | Init square operation |
| `square_tile(uint32_t idst)` | Square each element |
| `power_tile_init()` | Init power operation |
| `power_tile(uint32_t idst, uint32_t param0)` | Power operation x^param0 |
| `tiled_prod_tile_init()` | Init tiled product |
| `tiled_prod_tile(uint32_t idst)` | Element-wise multiplication on each row |

#### Max/Min Operations

| Function | Description |
|----------|-------------|
| `max_tile_init()` | Init max operation |
| `max_tile(uint32_t idst0, uint32_t idst1, int vector_mode)` | Element-wise max |
| `unary_max_tile_init()` | Init unary max |
| `unary_max_tile(uint32_t idst, uint32_t param0)` | Unary max: x if x > value, else value |
| `unary_max_int32_tile(uint32_t idst, uint32_t param0)` | Unary max for int32 |
| `unary_min_tile_init()` | Init unary min |
| `unary_min_tile(uint32_t idst, uint32_t param0)` | Unary min: x if x < value, else value |
| `unary_min_int32_tile(uint32_t idst, uint32_t param0)` | Unary min for int32 |

#### Special Functions

| Function | Description |
|----------|-------------|
| `heaviside_tile_init()` | Init Heaviside step function |
| `heaviside_tile(uint32_t idst, uint32_t param0)` | Heaviside step function |
| `silu_tile_init()` | Init SiLU (Swish) activation |
| `silu_tile(uint32_t idst)` | SiLU activation: x * sigmoid(x) |
| `alt_complex_rotate90_tile_init()` | Init complex rotation |
| `alt_complex_rotate90_tile(uint32_t idst)` | Rotate complex numbers 90 degrees |

#### TopK Operations

| Function | Description |
|----------|-------------|
| `topk_tile_init()` | Init TopK algorithm |
| `topk_local_sort<stable_sort>(uint32_t idst, int idir, int i_end_phase, ...)` | TopK local sort stage |
| `topk_merge<idir, stable_sort>(uint32_t idst, int m_iter, int k)` | TopK merge stage |
| `topk_rebuild<stable_sort>(uint32_t idst, bool idir, int m_iter, int k, int logk, int skip_second)` | TopK rebuild stage |

#### Reduction Operations

| Function | Description |
|----------|-------------|
| `max_reduce_with_indices_init<layout>()` | Init max reduction with indices |
| `max_reduce_with_indices<num_rows, layout, ITERATIONS>(uint32_t idst, uint32_t idst_idx)` | MaxPool with indices |
| `sfpu_reduce_init<pool_type, format>()` | Init SFPU reduce |
| `sfpu_reduce<pool_type, format, reduce_dim>(uint32_t idst)` | SFPU reduce operation |
| `sfpu_add_top_row_init()` | Init add top row |
| `sfpu_add_top_row<format>(uint32_t dst_tile_0, uint32_t dst_tile_1, uint32_t dst_tile_out)` | Add top rows of two tiles |

#### Debug Functions

| Function | Description |
|----------|-------------|
| `dbg_halt()` | Pause cores for debug inspection |
| `dbg_unhalt()` | Resume execution after debug halt |
| `dbg_read_dest_acc_row(int row_addr, uint32_t* rd_data)` | Read destination register row |

#### Special Value Flags

| Function | Description |
|----------|-------------|
| `get_compute_special_value_flags()` | Get special value flags |
| `get_compute_special_value_flags_fpu(uint32_t special_value_flags_reg)` | Get FPU special value flags |
| `get_compute_special_value_flags_sfpu(uint32_t special_value_flags_reg)` | Get SFPU special value flags |
| `clear_compute_special_value_flags()` | Clear special value flags |
| `store_compute_special_value_flags_to_l1(uint32_t l1_addr)` | Store special value flags to L1 |

### `add_int_sfpu.h`

| Function | Description |
|----------|-------------|
| `add_int32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` | Elementwise add operation with two integer inputs (int32) |
| `add_uint16_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` | Elementwise add operation with two uint16 inputs |
| `add_uint32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` | Elementwise add operation with two uint32 inputs |
| `add_int_tile_init()` | Initialization for add_int operations |

### `bcast.h`

| Function | Description |
|----------|-------------|
| `unary_bcast_init<BroadcastType bcast_type>(uint32_t icb, uint32_t ocb)` | Init function for unary broadcast operations |
| `unary_bcast<BroadcastType bcast_type>(uint32_t icb, uint32_t in_tile_index, uint32_t dst_tile_index)` | Unary broadcast operation |
| `reconfigure_unary_bcast<old_bcast_type, new_bcast_type>(uint32_t old_icb, uint32_t new_icb, uint32_t old_ocb, uint32_t new_ocb)` | Reconfigure unary broadcast |
| `sub_tiles_bcast_cols(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)` | Subtract tiles with column broadcast |
| `sub_tiles_bcast_scalar(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)` | Subtract tiles with scalar broadcast |
| `mul_tiles_bcast_cols(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)` | Multiply tiles with column broadcast |
| `mul_tiles_bcast_rows(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst, uint32_t bcast_row_idx)` | Multiply tiles with row broadcast |
| `add_tiles_bcast_rows(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst, uint32_t bcast_row_idx)` | Add tiles with row broadcast |
| `add_tiles_bcast_cols(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)` | Add tiles with column broadcast |
| `add_tiles_bcast_scalar(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)` | Add tiles with scalar broadcast |
| `init_bcast<EltwiseBinaryType tBcastOp, BroadcastType tBcastDim>(uint32_t icb0, uint32_t icb1, uint32_t ocb)` | Init function for broadcast operations |
| `any_tiles_bcast<EltwiseBinaryType, BroadcastType>(...)` | Internal helper for broadcast ops |
| `add_tiles_bcast<BroadcastType tBcastDim>(...)` | Add tiles with specified broadcast dimension |
| `sub_tiles_bcast<BroadcastType tBcastDim>(...)` | Subtract tiles with specified broadcast dimension |
| `mul_tiles_bcast<BroadcastType tBcastDim>(...)` | Multiply tiles with specified broadcast dimension |
| `add_bcast_rows_init_short(uint32_t icb0, uint32_t icb1)` | Short init for add broadcast rows |
| `add_bcast_cols_init_short(uint32_t icb0, uint32_t icb1)` | Short init for add broadcast cols |
| `add_bcast_scalar_init_short(uint32_t icb0, uint32_t icb1)` | Short init for add broadcast scalar |
| `mul_tiles_bcast_scalar_init_short(uint32_t icb0, uint32_t icb1)` | Short init for mul broadcast scalar |
| `mul_tiles_bcast_scalar(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)` | Multiply tiles with scalar broadcast |
| `mul_bcast_cols_init_short(uint32_t icb0, uint32_t icb1)` | Short init for mul broadcast cols |
| `mul_bcast_rows_init_short(uint32_t icb0, uint32_t icb1)` | Short init for mul broadcast rows |
| `sub_bcast_cols_init_short(uint32_t icb0, uint32_t icb1)` | Short init for sub broadcast cols |
| `sub_tiles_bcast_scalar_init_short(uint32_t icb0, uint32_t icb1)` | Short init for sub broadcast scalar |

### `binary_bitwise_sfpu.h`

| Function | Description |
|----------|-------------|
| `bitwise_and_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` | Bitwise AND operation on two tiles |
| `bitwise_and_uint32_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` | Bitwise AND for uint32 |
| `bitwise_and_uint16_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` | Bitwise AND for uint16 |
| `bitwise_or_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` | Bitwise OR operation |
| `bitwise_or_uint32_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` | Bitwise OR for uint32 |
| `bitwise_or_uint16_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` | Bitwise OR for uint16 |
| `bitwise_xor_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` | Bitwise XOR operation |
| `bitwise_xor_uint32_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` | Bitwise XOR for uint32 |
| `bitwise_xor_uint16_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` | Bitwise XOR for uint16 |
| `binary_bitwise_tile_init()` | Initialization for binary bitwise operations |

### `binary_comp.h`

| Function | Description |
|----------|-------------|
| `lt_int32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` | Less than comparison for int32 |
| `gt_int32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` | Greater than comparison for int32 |
| `ge_int32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` | Greater than or equal comparison for int32 |
| `le_int32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` | Less than or equal comparison for int32 |
| `lt_int32_tile_init()` | Init for less than comparison |
| `gt_int32_tile_init()` | Init for greater than comparison |
| `ge_int32_tile_init()` | Init for greater than or equal comparison |
| `le_int32_tile_init()` | Init for less than or equal comparison |

### `binary_max_min.h`

| Function | Description |
|----------|-------------|
| `binary_max_int32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` | Maximum operation on int32 tiles |
| `binary_max_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` | Maximum operation on tiles |
| `binary_max_tile_init()` | Init for binary max operation |
| `binary_min_int32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` | Minimum operation on int32 tiles |
| `binary_min_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` | Minimum operation on tiles |
| `binary_min_tile_init()` | Init for binary min operation |

### `binary_shift.h`

| Function | Description |
|----------|-------------|
| `binary_left_shift_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` | Left shift operation |
| `binary_left_shift_uint32_tile<sign_magnitude_format>(uint32_t idst0, uint32_t idst1, uint32_t odst)` | Left shift for uint32 |
| `binary_left_shift_int32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` | Left shift for int32 |
| `binary_right_shift_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` | Right shift operation |
| `binary_right_shift_uint32_tile<sign_magnitude_format>(...)` | Right shift for uint32 |
| `binary_right_shift_int32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` | Right shift for int32 |
| `binary_logical_right_shift_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` | Logical right shift |
| `binary_logical_right_shift_uint32_tile<sign_magnitude_format>(...)` | Logical right shift for uint32 |
| `binary_logical_right_shift_int32_tile(...)` | Logical right shift for int32 |
| `binary_shift_tile_init()` | Init for binary shift operations |

### `cb_api.h`

| Function | Description |
|----------|-------------|
| `cb_wait_front(uint32_t cbid, uint32_t ntiles)` | Blocking wait for tiles in CB |
| `cb_pop_front(uint32_t cbid, uint32_t ntiles)` | Pop tiles from front of CB |
| `cb_reserve_back(uint32_t cbid, uint32_t ntiles)` | Reserve space in back of CB |
| `cb_push_back(uint32_t cbid, uint32_t ntiles)` | Push tiles to back of CB |
| `get_tile_address(uint32_t cb_id, uint32_t tile_index)` | Get L1 address of tile in CB |
| `read_tile_value(uint32_t cb_id, uint32_t tile_index, uint32_t element_offset)` | Read value from tile in CB |

### `common.h`

| Function | Description |
|----------|-------------|
| `get_arg_addr(int arg_idx)` | Get L1 address for unique runtime argument |
| `get_common_arg_addr(int arg_idx)` | Get L1 address for common runtime argument |
| `get_arg_val<T>(int arg_idx)` | Get value at runtime argument index |
| `get_common_arg_val<T>(int arg_idx)` | Get value at common runtime argument index |
| `get_absolute_logical_x()` | Get absolute logical X coordinate |
| `get_absolute_logical_y()` | Get absolute logical Y coordinate |
| `get_relative_logical_x()` | Get relative logical X coordinate |
| `get_relative_logical_y()` | Get relative logical Y coordinate |

### `compute_kernel_hw_startup.h`

| Function | Description |
|----------|-------------|
| `compute_kernel_hw_startup(uint32_t icb0, uint32_t icb1, uint32_t ocb)` | Hardware initialization for compute kernel |
| `compute_kernel_hw_startup(uint32_t icb0, uint32_t ocb)` | Hardware initialization (single input CB version) |

### `copy_dest_values.h`

| Function | Description |
|----------|-------------|
| `copy_dest_values(uint32_t idst0, uint32_t idst1)` | Copy values from one DST tile to another |
| `copy_dest_values_init()` | Init for copy_dest_values |

### `cumsum.h`

| Function | Description |
|----------|-------------|
| `cumsum_tile(uint32_t idst, bool first = true)` | Columnwise cumulative sum |
| `cumsum_tile_init()` | Init for cumsum operation |

### `div_int32_floor.h`

| Function | Description |
|----------|-------------|
| `div_int32_floor_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` | Floor division for int32 |
| `div_int32_trunc_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` | Truncated division for int32 |
| `div_int32_floor_tile_init()` | Init for floor division |
| `div_int32_trunc_tile_init()` | Init for truncated division |

### `div_int32_sfpu.h`

| Function | Description |
|----------|-------------|
| `div_int32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` | Division operation for int32 |
| `div_int32_tile_init()` | Init for int32 division |

### `eltwise_binary_sfpu.h`

| Function | Description |
|----------|-------------|
| `add_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` | Elementwise add for float |
| `sub_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` | Elementwise subtract for float |
| `mul_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` | Elementwise multiply for float |
| `div_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` | Elementwise divide for float |
| `rsub_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` | Reverse subtract for float |
| `power_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` | Power operation |
| `add_binary_tile_init()` | Init for add |
| `sub_binary_tile_init()` | Init for subtract |
| `mul_binary_tile_init()` | Init for multiply |
| `div_binary_tile_init()` | Init for divide |
| `rsub_binary_tile_init()` | Init for reverse subtract |
| `power_binary_tile_init()` | Init for power |

### `eltwise_binary.h`

| Function | Description |
|----------|-------------|
| `binary_op_init_common(uint32_t icb0, uint32_t icb1, uint32_t ocb)` | Common init for binary ops |
| `binary_tiles_init<full_init, eltwise_binary_type>(uint32_t icb0, uint32_t icb1, bool acc_to_dest)` | Template init for binary tiles |
| `mul_tiles_init(uint32_t icb0, uint32_t icb1)` | Init for multiply tiles |
| `add_tiles_init(uint32_t icb0, uint32_t icb1, bool acc_to_dest)` | Init for add tiles |
| `sub_tiles_init(uint32_t icb0, uint32_t icb1, bool acc_to_dest)` | Init for subtract tiles |
| `mul_tiles(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)` | Elementwise multiply C=A*B |
| `add_tiles(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)` | Elementwise add C=A+B |
| `sub_tiles(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)` | Elementwise subtract C=A-B |
| `binary_dest_reuse_tiles_init<eltwise_binary_type, binary_reuse_dest>(uint32_t icb0)` | Init for dest reuse binary ops |
| `binary_dest_reuse_tiles<eltwise_binary_type, binary_reuse_dest>(uint32_t in_cb_id, uint32_t in_tile_index, uint32_t dst_tile_index)` | Binary op with dest reuse |

### `ema.h`

| Function | Description |
|----------|-------------|
| `ema_init(uint32_t alpha, uint32_t beta)` | Init for exponential moving average |
| `ema_clear_previous_output()` | Clear previous EMA output |
| `ema_tile(uint32_t input_dst_index)` | Compute EMA on tile |

### `gcd.h`

| Function | Description |
|----------|-------------|
| `gcd_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` | Greatest common divisor operation |
| `gcd_tile_init()` | Init for GCD |

### `lcm.h`

| Function | Description |
|----------|-------------|
| `lcm_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` | Least common multiple operation |
| `lcm_tile_init()` | Init for LCM |

### `logsigmoid.h`

| Function | Description |
|----------|-------------|
| `logsigmoid_tile(uint32_t idst_in0, uint32_t idst_in1, uint32_t idst_out)` | Log-sigmoid operation |
| `logsigmoid_tile_init()` | Init for log-sigmoid |

### `mask.h`

| Function | Description |
|----------|-------------|
| `mask_tile_init()` | Init for mask operation |
| `mask_tile(uint32_t idst_data, uint32_t idst2_mask, DataFormat data_format)` | Mask operation |
| `mask_posinf_tile(uint32_t idst_data, uint32_t idst2_mask)` | Mask with positive infinity |

### `matmul.h`

| Function | Description |
|----------|-------------|
| `mm_init(uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t out_cb_id, uint32_t transpose)` | Init for matmul |
| `matmul_tiles(uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t in0_tile_index, uint32_t in1_tile_index, uint32_t idst)` | Tile-sized matrix multiplication C=A*B |
| `matmul_tiles_math<num_faces>(uint32_t idst)` | Math-only matmul on tiles in SRC registers |
| `mm_init_short(uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t transpose)` | Short init for matmul |
| `mm_init_short_with_dt(uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t c_in_old_srca, uint32_t transpose)` | Short init with data format reconfig |
| `mm_block_init(uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t out_cb_id, uint32_t transpose, uint32_t ct_dim, uint32_t rt_dim, uint32_t kt_dim)` | Init for block matmul |
| `matmul_block(...)` | Block-sized matrix multiplication |
| `mm_block_init_short(...)` | Short init for block matmul |
| `mm_block_init_short_with_dt(...)` | Short init with data type reconfig |
| `mm_block_init_short_with_both_dt(...)` | Short init with both data type reconfigs |
| `matmul_block_math_dynamic_throttle(...)` | Block matmul with dynamic throttling (Blackhole only) |

### `mul_int_sfpu.h`

| Function | Description |
|----------|-------------|
| `mul_uint16_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` | Multiply uint16 tiles |
| `mul_int_tile_init()` | Init for integer multiply |

### `mul_int32_sfpu.h`

| Function | Description |
|----------|-------------|
| `mul_int32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` | Multiply int32 tiles |
| `mul_uint32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` | Multiply uint32 tiles |
| `mul_int32_tile_init()` | Init for int32 multiply |

### `pack_untilize.h`

| Function | Description |
|----------|-------------|
| `pack_untilize_dest_init<block_ct_dim, full_ct_dim, narrow_row, row_num_datums>(uint32_t ocb, uint32_t face_r_dim, uint32_t num_faces)` | Init pack untilize with DEST input |
| `pack_untilize_init<block_ct_dim, full_ct_dim>(uint32_t icb, uint32_t ocb)` | Init pack untilize operation |
| `pack_untilize_block<block_ct_dim, full_ct_dim>(uint32_t icb, uint32_t block_rt_dim, uint32_t ocb, uint32_t block_c_index)` | Untilize a block of tiles |
| `pack_untilize_dest<...>(uint32_t ocb, uint32_t block_rt_dim, ...)` | Pack untilize from DEST register |
| `pack_untilize_uninit(uint32_t ocb)` | Uninit pack untilize |

### `pack.h`

| Function | Description |
|----------|-------------|
| `pack_tile<out_of_order_output>(uint32_t ifrom_dst, uint32_t icb, uint32_t output_tile_index)` | Copy tile from DEST to CB |
| `pack_tile_block(uint32_t ifrom_dst, uint32_t icb, uint32_t ntiles)` | Pack block of tiles from DEST to CB |
| `pack_reconfig_data_format(uint32_t new_cb_id)` | Reconfigure packer data format |
| `pack_reconfig_data_format(uint32_t old_cb_id, uint32_t new_cb_id)` | Reconfigure packer with old/new comparison |
| `pack_reconfig_l1_acc(uint32_t l1_acc_en)` | Reconfigure L1 accumulation flag |
| `pack_rows_init(uint32_t num_rows)` | Init pack rows operation |
| `pack_rows(uint32_t idst, uint32_t ocb, uint32_t output_index)` | Pack rows from DEST to CB |
| `pack_rows_uninit()` | Uninit pack rows |

### `quantization.h`

| Function | Description |
|----------|-------------|
| `quant_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` | Per-tensor affine quantization |
| `requant_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` | Per-tensor affine re-quantization |
| `dequant_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` | Per-tensor affine de-quantization |
| `quant_tile_init(uint32_t zero_point)` | Init quantization |
| `requant_tile_init(uint32_t zero_point)` | Init re-quantization |
| `dequant_tile_init(uint32_t zero_point)` | Init de-quantization |

### `reconfig_data_format.h`

| Function | Description |
|----------|-------------|
| `reconfig_data_format<to_from_int8>(uint32_t srca_new_operand, uint32_t srcb_new_operand)` | Reconfigure srca and srcb data formats |
| `reconfig_data_format<to_from_int8>(uint32_t srca_old, uint32_t srca_new, uint32_t srcb_old, uint32_t srcb_new)` | Reconfigure with old/new comparison |
| `reconfig_data_format_srca<to_from_int8>(uint32_t srca_new_operand)` | Reconfigure srca data format |
| `reconfig_data_format_srca<to_from_int8>(uint32_t srca_old_operand, uint32_t srca_new_operand)` | Reconfigure srca with comparison |
| `reconfig_data_format_srcb<to_from_int8>(uint32_t srcb_new_operand)` | Reconfigure srcb data format |
| `reconfig_data_format_srcb<to_from_int8>(uint32_t srcb_old_operand, uint32_t srcb_new_operand)` | Reconfigure srcb with comparison |

### `reduce_custom.h`

| Function | Description |
|----------|-------------|
| `reduce_block_max_row_init<block_ct_dim>()` | Init block-based max row reduction |
| `reduce_block_max_row<block_ct_dim>(uint32_t icb, uint32_t icb_scaler, uint32_t row_start_index, uint32_t idst)` | Block-based max row reduction |
| `reduce_block_max_row_uninit<clear_fp32_accumulation>()` | Uninit block max row reduction |

### `reduce.h`

| Function | Description |
|----------|-------------|
| `reduce_init<reduce_type, reduce_dim, enforce_fp32_accumulation>(uint32_t icb, uint32_t icb_scaler, uint32_t ocb)` | Init reduce operation |
| `reduce_uninit<enforce_fp32_accumulation>()` | Uninit reduce operation |
| `reduce_tile<reduce_type, reduce_dim, enforce_fp32_accumulation>(uint32_t icb, uint32_t icb_scaler, uint32_t itile, uint32_t itile_scaler, uint32_t idst)` | Reduce tile operation |
| `reduce_tile_math<reduce_type, reduce_dim, enforce_fp32_accumulation>(uint32_t idst, uint32_t num_faces)` | Math-only reduce tile |

### `reg_api.h`

| Function | Description |
|----------|-------------|
| `acquire_dst()` | *(deprecated)* Acquire DST register lock |
| `tile_regs_acquire()` | Acquire DST register lock for MATH thread |
| `tile_regs_wait()` | Acquire DST register lock for PACK thread |
| `release_dst()` | *(deprecated)* Release DST register lock |
| `tile_regs_commit()` | Release DST lock by MATH thread |
| `tile_regs_release()` | Release DST lock by PACK thread |

### `reshuffle.h`

| Function | Description |
|----------|-------------|
| `reshuffle_rows_tile(uint32_t idst, uint32_t idx_addr)` | Reshuffle rows of tile |
| `reshuffle_rows_tile_init()` | Init for row reshuffle |

### `sub_int_sfpu.h`

| Function | Description |
|----------|-------------|
| `sub_int32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` | Subtract int32 tiles |
| `sub_uint32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` | Subtract uint32 tiles |
| `sub_uint16_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` | Subtract uint16 tiles |
| `rsub_int32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` | Reverse subtract int32 |
| `rsub_uint32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` | Reverse subtract uint32 |
| `rsub_uint16_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` | Reverse subtract uint16 |
| `sub_int_tile_init()` | Init for integer subtract |
| `rsub_int_tile_init()` | Init for reverse subtract |

### `tile_move_copy.h`

| Function | Description |
|----------|-------------|
| `copy_tile_to_dst_init_short(uint32_t cbid, uint32_t transpose, uint32_t transpose_within_16x16_face)` | Short init for tile copy |
| `copy_tile_init(uint32_t cbid)` | Init for copy tile |
| `copy_tile_to_dst_init_short_with_dt(uint32_t old_cbid, uint32_t new_cbid, uint32_t transpose)` | Short init with data format reconfig |
| `copy_tile(uint32_t in_cb_id, uint32_t in_tile_index, uint32_t dst_tile_index)` | Copy tile from CB to DST |
| `copy_block_matmul_partials(uint32_t in_cb_id, uint32_t start_in_tile_index, uint32_t start_dst_tile_index, uint32_t ntiles)` | Copy block of matmul partials |

### `tilize.h`

| Function | Description |
|----------|-------------|
| `tilize_init(uint32_t icb, uint32_t block, uint32_t ocb)` | Init tilize operation |
| `tilize_init_no_pack(uint32_t icb, uint32_t block)` | Init tilize without pack |
| `tilizeA_B_reduce_init<neginf_srcA, zero_srcA_reduce>(...)` | Init tilize with reduction |
| `tilize_init_short_with_dt(uint32_t old_icb, uint32_t new_icb, uint32_t block, uint32_t ocb)` | Short init with data type reconfig |
| `tilize_init_short_with_dt_no_pack(uint32_t old_icb, uint32_t new_icb, uint32_t block)` | Short init no pack variant |
| `tilize_block(uint32_t icb, uint32_t block, uint32_t ocb, uint32_t input_tile_index, uint32_t output_tile_index)` | Tilize a block |
| `tilize_block_no_pack(uint32_t icb, uint32_t block, uint32_t dst_idx, uint32_t input_tile_index)` | Tilize block without pack |
| `unpack_tilizeA_B_block<...>(...)` | Unpack and tilize from two CBs |
| `tilize_uninit(uint32_t icb, uint32_t ocb)` | Uninit tilize |
| `tilize_uninit_no_pack(uint32_t icb)` | Uninit tilize no pack variant |
| `tilize_uninit_with_dt(uint32_t old_icb, uint32_t new_icb, uint32_t ocb)` | Uninit with data type reconfig |
| `tilize_uninit_with_dt_no_pack(uint32_t old_icb, uint32_t new_icb)` | Uninit no pack with data type |
| `fast_tilize_init(uint32_t icb, uint32_t full_dim, uint32_t ocb)` | Init fast tilize |
| `fast_tilize_init_with_dt(uint32_t icb, uint32_t full_dim, uint32_t ocb)` | Fast tilize init with data type |
| `fast_tilize_uninit(uint32_t icb, uint32_t ocb)` | Uninit fast tilize |
| `fast_tilize_block(uint32_t icb, uint32_t block, uint32_t ocb, uint32_t input_tile_index, uint32_t output_tile_index)` | Fast tilize a block |
| `unpack_tilizeA_B_uninit(uint32_t icb)` | Uninit unpack tilizeA_B |

### `transpose_wh_dest.h`

| Function | Description |
|----------|-------------|
| `transpose_wh_dest_init_short<is_32bit>()` | Short init for in-place transpose |
| `transpose_wh_dest<is_32bit>(uint32_t idst)` | In-place 32x32 transpose in DST |

### `transpose_wh.h`

| Function | Description |
|----------|-------------|
| `transpose_wh_init(uint32_t icb, uint32_t ocb)` | Init for WH transpose |
| `transpose_wh_init_short(uint32_t icb)` | Short init for WH transpose |
| `transpose_wh_tile(uint32_t icb, uint32_t itile, uint32_t idst)` | Transpose tile B[w,h] = A[h,w] |

### `untilize.h`

| Function | Description |
|----------|-------------|
| `untilize_init(uint32_t icb)` | Init untilize operation |
| `untilize_block<block_ct_dim>(uint32_t icb, uint32_t full_ct_dim, uint32_t ocb)` | Untilize a block |
| `untilize_uninit(uint32_t icb)` | Uninit untilize operation |

### `welford.h`

| Function | Description |
|----------|-------------|
| `welford_init()` | Init Welford's algorithm |
| `welford_clear()` | Clear stale mean and m2 values |
| `welford_update<reciprocal_size>(uint32_t input_dst_idx, uint32_t start_idx, const std::array<uint32_t, reciprocal_size>& reciprocal_lut)` | Welford update for mean/m2 |
| `welford_update_rows<reciprocal_size>(uint32_t input_dst_idx, uint32_t start_idx, uint32_t start_row, uint32_t num_rows, const std::array<...>& reciprocal_lut)` | Welford update for subset of rows |
| `welford_save_state(uint32_t mean_dst_idx)` | Save mean/m2 to DST |
| `welford_save_state(uint32_t mean_dst_idx, uint32_t group_id)` | Save state with group_id |
| `welford_restore_state(uint32_t mean_dst_idx)` | Restore mean/m2 from DST |
| `welford_restore_state(uint32_t mean_dst_idx, uint32_t group_id)` | Restore state with group_id |
| `welford_finalize_to_row<reciprocal_size>(uint32_t mean_dst_idx, uint32_t scale_idx, const std::array<...>& reciprocal_lut)` | Finalize and store mean/variance to row |
| `welford_finalize_to_face<reciprocal_size>(uint32_t mean_dst_idx, uint32_t scale_idx, const std::array<...>& reciprocal_lut)` | Finalize and store in raw format |
| `welford_finalize_to_face<reciprocal_size>(uint32_t mean_dst_idx, uint32_t group_id, uint32_t scale_idx, const std::array<...>& reciprocal_lut)` | Finalize with group_id |

### `xlogy.h`

| Function | Description |
|----------|-------------|
| `xlogy_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` | Element-wise xlogy operation |
| `xlogy_binary_tile_init()` | Init for xlogy |

---

## Eltwise Unary Operations

### `eltwise_unary/eltwise_unary.h`

| Function | Description |
|----------|-------------|
| `unary_op_init_common(uint32_t icb, uint32_t ocb)` | Common init for unary ops |
| `unary_op_init_common_no_pack(uint32_t icb)` | Common init without pack config |
| `init_sfpu(uint32_t icb, uint32_t ocb)` | Init SFPU |

### `eltwise_unary/activations.h`

| Function | Description |
|----------|-------------|
| `hardsigmoid_tile(uint32_t idst)` | Hardsigmoid activation |
| `hardsigmoid_tile_init()` | Init hardsigmoid |
| `softsign_tile(uint32_t idst)` | Softsign activation |
| `softsign_tile_init()` | Init softsign |
| `celu_tile(uint32_t idst, uint32_t alpha, uint32_t alpha_recip)` | CELU activation |
| `celu_tile_init()` | Init CELU |
| `softshrink_tile(uint32_t idst, uint32_t param0)` | Softshrink activation |
| `softshrink_tile_init()` | Init softshrink |

### `eltwise_unary/binop_with_scalar.h`

| Function | Description |
|----------|-------------|
| `add_unary_tile(uint32_t idst, uint32_t param1)` | Add scalar to tile |
| `sub_unary_tile(uint32_t idst, uint32_t param1)` | Subtract scalar from tile |
| `mul_unary_tile(uint32_t idst, uint32_t param1)` | Multiply tile by scalar |
| `div_unary_tile(uint32_t idst, uint32_t param1)` | Divide tile by scalar |
| `rsub_unary_tile(uint32_t idst, uint32_t param1)` | Reverse subtract (scalar - tile) |
| `add_unary_tile_int32(uint32_t idst, uint32_t param1)` | Add int32 scalar |
| `sub_unary_tile_int32(uint32_t idst, uint32_t param1)` | Subtract int32 scalar |
| `binop_with_scalar_tile_init()` | Init for scalar binops |

### `eltwise_unary/exp.h`

| Function | Description |
|----------|-------------|
| `exp_tile_init<approx, fast_and_approx, scale>()` | Init exponential operation |
| `exp_tile<approx, fast_and_approx, scale_en, skip_positive_check, iterations>(uint32_t idst, int vector_mode, uint16_t scale)` | Exponential on tile |

### `eltwise_unary/recip.h`

| Function | Description |
|----------|-------------|
| `recip_tile_init<legacy_compat>()` | Init reciprocal |
| `recip_tile<legacy_compat>(uint32_t idst, int vector_mode)` | Reciprocal of tile |

### `eltwise_unary/sqrt.h`

| Function | Description |
|----------|-------------|
| `sqrt_tile_init()` | Init square root |
| `sqrt_tile<FAST_APPROX>(uint32_t idst)` | Square root of tile |

### `eltwise_unary/rsqrt.h`

| Function | Description |
|----------|-------------|
| `rsqrt_tile_init<legacy_compat>()` | Init reciprocal square root |
| `rsqrt_tile<legacy_compat, FAST_APPROX>(uint32_t idst)` | Reciprocal square root of tile |

### `eltwise_unary/trigonometry.h`

| Function | Description |
|----------|-------------|
| `sin_tile_init()` | Init sine |
| `sin_tile(uint32_t idst)` | Sine of tile |
| `cos_tile_init()` | Init cosine |
| `cos_tile(uint32_t idst)` | Cosine of tile |
| `tan_tile_init()` | Init tangent |
| `tan_tile(uint32_t idst)` | Tangent of tile |
| `asin_tile_init()` | Init arcsine |
| `asin_tile(uint32_t idst)` | Arcsine of tile |
| `acos_tile_init()` | Init arccosine |
| `acos_tile(uint32_t idst)` | Arccosine of tile |
| `atan_tile_init()` | Init arctangent |
| `atan_tile(uint32_t idst)` | Arctangent of tile |
| `sinh_tile_init()` | Init hyperbolic sine |
| `sinh_tile(uint32_t idst)` | Hyperbolic sine of tile |
| `cosh_tile_init()` | Init hyperbolic cosine |
| `cosh_tile(uint32_t idst)` | Hyperbolic cosine of tile |
| `asinh_tile_init()` | Init inverse hyperbolic sine |
| `asinh_tile(uint32_t idst)` | Inverse hyperbolic sine |
| `acosh_tile_init()` | Init inverse hyperbolic cosine |
| `acosh_tile(uint32_t idst)` | Inverse hyperbolic cosine |
| `atanh_tile_init()` | Init inverse hyperbolic tangent |
| `atanh_tile(uint32_t idst)` | Inverse hyperbolic tangent |

### `eltwise_unary/gelu.h`

| Function | Description |
|----------|-------------|
| `gelu_tile_init<fast_and_approx>()` | Init GELU |
| `gelu_tile<fast_and_approx>(uint32_t idst)` | GELU activation |

### `eltwise_unary/relu.h`

| Function | Description |
|----------|-------------|
| `relu_tile(uint32_t idst)` | ReLU activation |
| `relu_tile_int32(uint32_t idst)` | ReLU for int32 |
| `relu_tile_init()` | Init ReLU |
| `relu_max_tile(uint32_t idst, uint32_t param0)` | ReLU with max limit |
| `relu_max_tile_int32(uint32_t idst, uint32_t param0)` | ReLU max for int32 |
| `relu_max_tile_init()` | Init ReLU max |
| `relu_min_tile(uint32_t idst, uint32_t param0)` | ReLU with min limit |
| `relu_min_tile_int32(uint32_t idst, uint32_t param0)` | ReLU min for int32 |
| `relu_min_tile_init()` | Init ReLU min |
| `leaky_relu_tile(uint32_t idst, uint32_t slope)` | Leaky ReLU activation |
| `leaky_relu_tile_init()` | Init leaky ReLU |

### `eltwise_unary/fill.h`

| Function | Description |
|----------|-------------|
| `fill_tile(uint32_t idst, float param0)` | Fill tile with float value |
| `fill_tile_int(uint32_t idst, uint param0)` | Fill tile with integer value |
| `fill_tile_bitcast(uint32_t idst, uint32_t param0)` | Fill tile with bitcast value |
| `fill_tile_init()` | Init fill operation |

### `eltwise_unary/typecast.h`

| Function | Description |
|----------|-------------|
| `typecast_tile<IN_DTYPE, OUT_DTYPE>(uint32_t idst)` | Typecast tile |
| `typecast_tile_init<IN_DTYPE, OUT_DTYPE>()` | Init typecast |

### `eltwise_unary/comp.h`

| Function | Description |
|----------|-------------|
| `unary_eq_tile_init()` | Init unary equal comparison |
| `unary_eq_tile(uint32_t idst, uint32_t param0)` | Unary equal: 1.0 if x==value, else 0.0 |
| `unary_eq_tile_int32(uint32_t idst, uint32_t param0)` | Unary equal for int32: 1 if x==value, else 0 |
| `unary_ne_tile_init()` | Init unary not equal comparison |
| `unary_ne_tile(uint32_t idst, uint32_t param0)` | Unary not equal: 1.0 if x!=value, else 0.0 |
| `unary_ne_tile_int32(uint32_t idst, uint32_t param0)` | Unary not equal for int32: 1 if x!=value, else 0 |
| `unary_gt_tile_init()` | Init unary greater than |
| `unary_gt_tile(uint32_t idst, uint32_t param0)` | Unary greater than: 1.0 if x>value, else 0.0 |
| `unary_gt_tile_int32(uint32_t idst, uint32_t param0)` | Unary greater than for int32 |
| `unary_ge_tile_init()` | Init unary greater than or equal |
| `unary_ge_tile(uint32_t idst, uint32_t param0)` | Unary greater than or equal: 1.0 if x>=value, else 0.0 |
| `unary_ge_tile_int32(uint32_t idst, uint32_t param0)` | Unary greater than or equal for int32 |
| `unary_lt_tile_init()` | Init unary less than |
| `unary_lt_tile(uint32_t idst, uint32_t param0)` | Unary less than: 1.0 if x<value, else 0.0 |
| `unary_lt_tile_int32(uint32_t idst, uint32_t param0)` | Unary less than for int32 |
| `unary_le_tile_init()` | Init unary less than or equal |
| `unary_le_tile(uint32_t idst, uint32_t param0)` | Unary less than or equal: 1.0 if x<=value, else 0.0 |
| `unary_le_tile_int32(uint32_t idst, uint32_t param0)` | Unary less than or equal for int32 |
| `gtz_tile_init()` | Init greater than zero |
| `gtz_tile(uint32_t idst)` | Greater than zero check |
| `gtz_tile_int32(uint32_t idst)` | Greater than zero for int32 |
| `gez_tile_init()` | Init greater than or equal to zero |
| `gez_tile(uint32_t idst)` | Greater than or equal to zero check |
| `gez_tile_int32(uint32_t idst)` | Greater than or equal to zero for int32 |
| `ltz_tile_init()` | Init less than zero |
| `ltz_tile(uint32_t idst)` | Less than zero check |
| `ltz_tile_int32(uint32_t idst)` | Less than zero for int32 |
| `lez_tile_init()` | Init less than or equal to zero |
| `lez_tile(uint32_t idst)` | Less than or equal to zero check |
| `lez_tile_int32(uint32_t idst)` | Less than or equal to zero for int32 |
| `eqz_tile_init()` | Init equal to zero |
| `eqz_tile(uint32_t idst)` | Equal to zero check |
| `eqz_tile_int32(uint32_t idst)` | Equal to zero for int32 |
| `eqz_tile_uint16(uint32_t idst)` | Equal to zero for uint16 |
| `eqz_tile_uint32(uint32_t idst)` | Equal to zero for uint32 |
| `nez_tile_init()` | Init not equal to zero |
| `nez_tile(uint32_t idst)` | Not equal to zero check |
| `nez_tile_int32(uint32_t idst)` | Not equal to zero for int32 |
| `nez_tile_uint16(uint32_t idst)` | Not equal to zero for uint16 |
| `nez_tile_uint32(uint32_t idst)` | Not equal to zero for uint32 |

### `eltwise_unary/bitwise_*.h`

| Function | Description |
|----------|-------------|
| `bitwise_and_tile_init()` | Init bitwise AND |
| `bitwise_and_tile(uint32_t idst, uint32_t param0)` | Bitwise AND with scalar |
| `bitwise_not_tile_init()` | Init bitwise NOT |
| `bitwise_not_tile(uint32_t idst)` | Bitwise NOT operation |
| `bitwise_or_tile(uint32_t idst, uint32_t param0)` | Bitwise OR with scalar |
| `bitwise_or_tile_init()` | Init bitwise OR |
| `bitwise_xor_tile(uint32_t idst, uint32_t param0)` | Bitwise XOR with scalar |
| `bitwise_xor_tile_init()` | Init bitwise XOR |
| *(Additional typed variants with init functions)* | |

### `eltwise_unary/where.h`

| Function | Description |
|----------|-------------|
| `where_tile(uint32_t idst0, uint32_t idst1, uint32_t idst2, uint32_t odst)` | Where operation (generic, Float16_b) |
| `where_tile_init()` | Init where operation |
| `where_fp32_tile(uint32_t idst0, uint32_t idst1, uint32_t idst2, uint32_t odst)` | Where operation for float32 |
| `where_int32_tile(uint32_t idst0, uint32_t idst1, uint32_t idst2, uint32_t odst)` | Where operation for int32 |
| `where_uint32_tile(uint32_t idst0, uint32_t idst1, uint32_t idst2, uint32_t odst)` | Where operation for uint32 |

### `eltwise_unary/rsub.h`

| Function | Description |
|----------|-------------|
| `rsub_tile()` | Reverse subtract (scalar - tile) |
| `rsub_tile_init()` | Init reverse subtract |
| `rsub_unary_int32_tile_init()` | Init reverse subtract for int32 |
| `rsub_unary_int32_tile(uint32_t idst, uint32_t scalar)` | Reverse subtract for int32: scalar - x |

### `eltwise_unary/rdiv.h`

| Function | Description |
|----------|-------------|
| `rdiv_tile_init()` | Init reverse divide |
| `rdiv_tile<rounding_mode>(uint32_t dst_index, uint32_t value, int vector_mode)` | Reverse divide: value / x |

### `eltwise_unary/rounding.h`

| Function | Description |
|----------|-------------|
| `rounding_op_tile_init()` | Init rounding operations |
| `ceil_tile()` | Ceiling function |
| `floor_tile()` | Floor function |
| `trunc_tile()` | Truncate function |
| `round_tile()` | Round to nearest integer |
| `round_tile(uint32_t idst, int32_t decimals)` | Round to specified decimal places |
| `stochastic_round_tile()` | Stochastic rounding |
| `frac_tile()` | Fractional part |

### `eltwise_unary/softplus.h`

| Function | Description |
|----------|-------------|
| `softplus_tile()` | Softplus activation |
| `softplus_tile_init()` | Init softplus |
| `softplus_tile(uint32_t idst, uint32_t beta, uint32_t beta_reciprocal, uint32_t threshold)` | Softplus with parameters |

### Other `eltwise_unary/` Functions

| File | Functions |
|------|-----------|
| `isinf_isnan.h` | `isinf_tile()`, `isposinf_tile()`, `isneginf_tile()`, `isnan_tile()`, `isfinite_tile()` + their `_init()` variants |
| `identity.h` | `identity_tile()`, `identity_tile_uint32()`, `identity_tile_init()` |
| `negative.h` | `negative_tile()`, `negative_tile_int32()`, `negative_tile_init()` |
| `logical_not_noti.h` | `logical_not_unary_tile()`, various typed variants, `logical_not_unary_tile_init()` |
| `log1p.h` | `log1p_tile<fast_and_approx>()`, `log1p_tile_init<fast_and_approx>()` |
| `left_shift.h` | `left_shift_tile()`, `left_shift_tile_init()` |
| `right_shift.h` | `right_shift_tile()`, `right_shift_tile_init()` |
| `remainder.h` | `remainder_tile()`, `remainder_tile_init()` |
| `fmod.h` | `fmod_tile()`, `fmod_tile_init()` |
| `prelu.h` | `prelu_tile()`, `prelu_tile_init()` |
| `rand.h` | `rand_tile()`, `rand_tile_init()` |
| `i0.h` | `i0_tile()`, `i0_tile_init()` |
| `i1.h` | `i1_tile()`, `i1_tile_init()` |
| `erfinv.h` | `erfinv_tile()`, `erfinv_tile_init()` |
| `erf_erfc.h` | `erf_tile()`, `erfc_tile()` + init variants |
| `elu.h` | `elu_tile()`, `elu_tile_init()` |
| `dropout.h` | `dropout_tile()`, `dropout_tile_init()` |
| `clamp.h` | `clamp_tile()`, `clamp_tile_init()` |
| `hardtanh.h` | `hardtanh_tile()`, `hardtanh_tile_init()` |
| `hardmish.h` | `hardmish_tile()`, `hardmish_tile_init()` |
| `rpow.h` | `rpow_tile()`, `rpow_tile_init()` |
| `threshold.h` | `threshold_tile()`, `threshold_tile_init()` |
| `selu.h` | `selu_tile()`, `selu_tile_init()` |
| `cbrt.h` | `cbrt_tile()`, `cbrt_tile_init()` |
| `reverseops.h` | Additional reverse operations |
| `sfpu_int_sum.h` | `sfpu_sum_int_init()`, `sfpu_sum_int_col()`, `sfpu_sum_int_row()`, `sfpu_add_int()` |

---

## Notes

- All functions are in the `ckernel` namespace
- Functions marked with `ALWI` are always inlined
- Template parameters are shown with angle brackets `<>`
- Many functions have corresponding `_init()` functions that must be called before use
- DST register must typically be acquired via `tile_regs_acquire()` before operations
- CB = Circular Buffer
- DST = Destination register
- SFPU = Scalar Floating Point Unit
- `tanhshrink` is not a direct API function - it's implemented as a composite operation using `tanh_tile()` and `sub_binary_tile()`
- Many functions have template parameters that are not fully documented
- Some functions have multiple overloads/variants (e.g., int32, uint32, uint16) that may not all be documented
