Prod — Program Cache Review

Reviewed files:
- `ttnn/cpp/ttnn/operations/reduction/prod/device/prod_all_program_factory.cpp`
- `ttnn/cpp/ttnn/operations/reduction/prod/device/prod_nc_program_factory.cpp`
- `ttnn/cpp/ttnn/operations/reduction/prod/device/prod_op_all.cpp`
- `ttnn/cpp/ttnn/operations/reduction/prod/device/prod_nc_op.cpp`

Findings:
- Uses the old type-erased infra with override callbacks in both single-core (all) and N/C reduction paths.
- Override updates runtime-only buffer base addresses on cache hits:
  - Reader: input buffer base.
  - Writer: output buffer base.
- Other runtime-only values are not required; compile-time args cover dims, tile counts, and grid splits derived from shapes and dim, which participate in default hashing via op attributes and tensor args.

No program-cache issues identified.

Suggested optional tests:
- Two-run cache test for `prod_all` reallocating input/output to verify override correctness and single cache entry.
- Two-run cache test for `prod_nc` for dim in {0,1}, reallocating buffers while keeping shapes/dtypes constant.
