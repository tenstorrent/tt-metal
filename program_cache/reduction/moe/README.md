MOE Reduction — Program Cache Review

Reviewed files:
- `ttnn/cpp/ttnn/operations/reduction/moe/device/moe_op.hpp`
- `ttnn/cpp/ttnn/operations/reduction/moe/device/moe_op.cpp`
- `ttnn/cpp/ttnn/operations/reduction/moe/device/moe_program_factory.cpp`

Findings:
- Uses the old type-erased infra; override callback updates reader input base address and writer output base address.
- Compile-time args cover index/value CB setup and per-core work split derived from shapes and attributes; default hashing includes these determinants via attributes and tensor args.

No program-cache issues identified.

Suggested optional tests:
- Two-run cache test reallocating inputs/outputs while keeping shapes/attrs constant; assert single cache entry and correctness on cache hit.
