# Pre-compiled Test Binaries

Binaries for `test_generic_op.cpp::TestGenericOpMatmulFromBinary`

Sources:
- `ttnn/.../matmul/device/kernels/dataflow/reader_bmm_8bank_output_tiles_partitioned.cpp`
- `ttnn/.../eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- `ttnn/.../matmul/device/kernels/compute/bmm.cpp`

Note: `bmm` has 2 hash variants (different compile-time args for different core groups)

Regenerate: `python3 tests/scripts/regenerate_precompiled_binaries.py`
