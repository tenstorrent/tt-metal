## Program cache review — ccl/all_to_all_combine

Status: Reviewed — potential program cache over-keying detected (efficiency issue).

Findings
- Hashing: Uses default program hash for new-infra mesh adapter (no custom `compute_program_hash`). This includes all fields in `operation_attributes_t` and `tensor_args_t`.
  - `operation_attributes_t` includes `cross_device_semaphore` and `init_semaphore` which carry device-global addresses and can vary between runs even when compiled program is identical.
  - Reference locations:
    - Attributes definition: `ttnn/cpp/ttnn/operations/ccl/all_to_all_combine/device/all_to_all_combine_device_operation.hpp:L22-L44`.
    - Mesh hash path uses op’s program hash and appends coordinates: see mesh adapter excerpt in `PROGRAM_CACHE_REVIEW_PROMPT.md`.
- Effect: Different semaphore addresses will produce different program hashes, preventing cache hits across runs and fragmenting the cache. This is an efficiency problem (not correctness), since overrides correctly set semaphore addresses at runtime.
- Overrides: Cache-hit path properly updates all runtime-only base addresses and semaphores per core/coordinate.
  - Reader updates: mapping, metadata, and input base addresses
  - Writer updates: output base address and both semaphore addresses
  - Reference: `ttnn/cpp/ttnn/operations/ccl/all_to_all_combine/device/all_to_all_combine_program_factory.cpp:L302-L329`.
- Create-time runtime-arg ordering and per-core iteration match: writer/reader args are pushed in `create_at(...)` with indices later updated in override. Token ranges (`start/end`) and fabric routing args remain compile-time for a given hashed config and are not overridden, which is correct.

Suspected root cause
- Over-keying from including `cross_device_semaphore` and `init_semaphore` in the default hash for `operation_attributes_t`. These should be runtime-only and excluded from the hash.

Suggested fix
- Implement a custom `compute_program_hash(operation_attributes_t, tensor_args_t)` that:
  - Clones `operation_attributes_t` and zeroes or strips `cross_device_semaphore` and `init_semaphore` before hashing.
  - Hashes only determinants of codegen: tensor shapes/layouts/dtypes/memory configs, `num_links`, `topology`, `axis`, `subdevice_id`, and any values that affect CB sizing, kernel selection, and routing defines.
  - Let mesh adapter append tensor coordinates as usual.

Optional test idea (cache-hit efficiency)
- Two-run test on same shapes/layouts/configs but with different global semaphore handles should reuse the same program cache entry. Currently expected to MISS and increment cache entries on the second run; assert this to expose over-keying.
