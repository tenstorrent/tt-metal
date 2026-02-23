# Layernorm Scaler Migration Analysis (commit 5d6b63f)

## Summary

In commit 5d6b63f, all layernorm kernels were migrated from `generate_reduce_scaler_legacy`
to `prepare_reduce_scaler`. This was mostly correct, but several kernels could use
`calculate_and_prepare_reduce_scaler` instead.

## Cases where `prepare_reduce_scaler` is required

| Kernel | Reason |
|--------|--------|
| `writer_unary_sharded_ln*.cpp` | `scalar_w` and `scalar_c` differ per core group (runtime, per-core) |
| `writer_unary_sharded_ln_pre_all_gather.cpp` | Same — per-core-group scalers |

## Cases where `calculate_and_prepare_reduce_scaler` could be used

### 1. Interleaved layernorm kernels — move W to compile-time

These kernels use scaler = `1/W`, but W is uniform across all cores (derived from `logical_shape[-1]`).
W can be moved from a runtime arg to a compile-time arg, enabling `calculate_and_prepare_reduce_scaler<cb, AVG, REDUCE_ROW, W>()`.

- `reader_unary_interleaved_ln.cpp`
- `reader_unary_interleaved_ln_rm_gb.cpp`
- `reader_unary_interleaved_ln_large_tensor.cpp`

**Kernel changes:**
- Remove runtime args for scaler (arg 4) and W (arg 9)
- Add W as a compile-time arg
- Replace `prepare_reduce_scaler` with `calculate_and_prepare_reduce_scaler<cb_in_2, AVG, REDUCE_ROW, W>()`
- Partial-tile logic (`W % TILE_WIDTH`) becomes `constexpr if`

**Factory changes (`layernorm_op_multi_core.cpp`):**
- Move W from runtime args to `reader_compile_time_args`
- Remove `packed_one_value` from runtime args (arg index 4)
- Shift remaining runtime arg indices accordingly

**Sanity test:**
```bash
pytest "tests/ttnn/unit_tests/operations/fused/test_layer_norm.py::test_layer_norm[dtype=torch.bfloat16-use_welford=True-w=64-h=32]" -v -s
```

**Full test suite:**
```bash
pytest tests/ttnn/unit_tests/operations/fused/test_layer_norm.py -v -s
```

### 2. Post-allgather distributed kernel — move W*num_devices to compile-time

`reader_unary_interleaved_ln_rm_gb_post_allgather.cpp` uses scaler = `1/(W * num_devices)`.
Both W (`shape[-1]`) and num_devices (`stats_tiles_cols / tile_cols_per_device`) are uniform across all cores.

**Kernel changes:**
- Remove runtime scaler arg
- Add `W * num_devices` (or W and num_devices separately) as compile-time args
- Replace `prepare_reduce_scaler` with `calculate_and_prepare_reduce_scaler<cb_reduce, AVG, REDUCE_ROW, W * num_devices>()`

**Factory changes (`layernorm_post_all_gather_program_factory.cpp`):**
- Move `W * num_devices` to compile-time args
- Remove `winv_bits` from runtime args

**Sanity test:**
```bash
pytest "tests/ttnn/unit_tests/operations/fused/test_distributed_layernorm_sharded.py::test_post_allgather_layernorm[core_grid=(8, 2)-mean=0-std=1-weights_df=DataType.BFLOAT8_B-output_df=DataType.BFLOAT8_B-input_df=DataType.BFLOAT8_B-num_devices=4-input_width=2048-min_pcc=0.9997-max_atol=0.45-eps=1e-06-seed=0-is_rmsnorm=True]" -v -s
```

**Full test suite:**
```bash
pytest tests/ttnn/unit_tests/operations/fused/test_distributed_layernorm_sharded.py::test_post_allgather_layernorm -v -s
pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_distributed_layernorm_post_allgather.py -v -s
```

### 3. Pre-allgather interleaved kernels — already constant 1.0f (SUM)

These always receive `1.0f` from their factory (pre-allgather computes partial sums, not yet averaging):

- `reader_unary_interleaved_ln_rm_gb_pre_allgather.cpp`
- `reader_layernorm_preallgather_2d.cpp` (main scaler only; `cb_zero = 0.0f` stays as `prepare_reduce_scaler`)

**Kernel changes:**
- Remove runtime scaler arg
- Replace with `calculate_and_prepare_reduce_scaler<cb_reduce, SUM, REDUCE_ROW>()`

**Factory changes:**
- Remove scaler from runtime args

**Sanity test:**
```bash
pytest "tests/ttnn/unit_tests/operations/fused/test_distributed_layernorm_sharded.py::test_pre_allgather_layernorm[fuse_residual=False-max_atol_ex2=0.04-min_pcc_ex2=0.982-min_pcc_residual_add=0.997-min_pcc_ex=0.9997-max_atol_ex=0.01-core_grid=(8, 4)-mean=0-std=1-input_df=DataType.BFLOAT8_B-num_devices=4-input_width=2048-seed=0-is_rmsnorm=True]" -v -s
```

**Full test suite:**
```bash
pytest tests/ttnn/unit_tests/operations/fused/test_distributed_layernorm_sharded.py::test_pre_allgather_layernorm -v -s
pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_distributed_layernorm_pre_allgather.py -v -s
```

## Key distinction between the two APIs

- `calculate_and_prepare_reduce_scaler<cb, pool_type, reduce_dim, reduce_volume>()` — all compile-time template params; computes 1/N for AVG, 1.0 for SUM/MAX
- `prepare_reduce_scaler<cb>(float scaler_f)` — takes a runtime float; use when scaler is computed on host or varies per core
