## Program cache review: data_movement/sharded

Scope reviewed
- This directory contains shared kernels and helpers (e.g., `device/kernels`, `sharded_common.*`) used by concrete sharded data-movement operations such as `interleaved_to_sharded`, `sharded_to_interleaved`, and `reshard`.
- There is no standalone OP entrypoint or program factory defined at the root `data_movement/sharded/` directory.

Findings
- Program cache behavior for the sharded data-movement flows is implemented and exercised within the specific OPs that consume these helpers:
  - `data_movement/sharded/interleaved_to_sharded` [already reviewed]
  - `data_movement/sharded/sharded_to_interleaved` [already reviewed]
  - `data_movement/sharded/reshard` [already reviewed]
- The common utilities do not own a program factory nor `override_runtime_arguments(...)`; therefore, there is no separate cache-hit override path to validate at this directory level.

Conclusion
- No standalone OP to test here. Caching correctness is covered by the individual sharded OPs listed above, which were reviewed separately.
