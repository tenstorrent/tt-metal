Accumulation (cumsum/cumprod) — Program Cache Review

Reviewed files:
- `ttnn/cpp/ttnn/operations/reduction/accumulation/device/accumulation_device_operation.hpp`
- `ttnn/cpp/ttnn/operations/reduction/accumulation/device/accumulation_program_factory.cpp`

Findings:
- Uses the new typed program-factory infra with explicit `override_runtime_arguments`.
- Override updates runtime-only buffer base addresses for reader and writer kernels across all active cores.
- Program hash uses default typed hashing of operation attributes (`dim`, `flip`, `op`) and tensor args; compile-time args (tiles per row, offsets, grid split) are derived from these and stay constant per cache entry.

No program-cache issues identified.

Suggested optional tests:
- Two-run cache test for cumsum and cumprod reusing same shapes/attrs but reallocating input/output; expect a single cache entry and identical results.
