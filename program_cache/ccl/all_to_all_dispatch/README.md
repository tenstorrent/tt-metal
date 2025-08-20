## Program cache review — ccl/all_to_all_dispatch

Status: Reviewed — potential program cache over-keying detected (efficiency issue).

Findings
- New-infra mesh op with default hash; attributes include `cross_device_semaphore` and `init_semaphore` which are runtime-only and vary between runs.
- Overrides correctly update runtime-only values on cache-hit:
  - Reader updates: input, indices, mapping, output, metadata buffer base addresses; semaphore address
  - Writer updates: same five buffer addresses; semaphore and init semaphore addresses
  - Reference: `ttnn/cpp/ttnn/operations/ccl/all_to_all_dispatch/device/all_to_all_dispatch_program_factory.cpp:L458-L489`.
- Core iteration in override matches creation; token range indices are compile-time for a given hashed config.

Suspected root cause
- Over-keying due to hashing semaphore handles via default struct hashing of `operation_attributes_t`.

Suggested fix
- Provide a custom `compute_program_hash` that ignores `cross_device_semaphore` and `init_semaphore` while hashing codegen determinants: shapes/layouts/dtypes/mem configs, `num_links`, `topology`, `axis`, `impl`, `worker_core_range_set`.

Optional test
- Two-run test reusing same shapes/configs but with different semaphore handles; expect a single program cache entry across both runs after fix. Current behavior likely creates two entries.
