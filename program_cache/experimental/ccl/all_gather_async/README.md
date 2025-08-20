## Program cache review: experimental/ccl/all_gather_async

OP reviewed
- `ttnn::operations::experimental::ccl::all_gather_async` (old infra; `ProgramWithCallbacks`). Multiple variants: minimal_default and llama_sharded.

Creation path summary
- Sets up worker cores per link and generates reader/writer command streams with run-time args emitters.
- Runtime args include input/output buffer addresses, per-core ranges, semaphore addresses, and fabric connections.

Override behavior (cache-hit path)
- Minimal default variant updates:
  - Reader: address overrides via `reader_rt_args_overrider_map`.
  - Writer: output address and semaphore via `writer_rt_args_overrider_map` and explicit index updates.
- Llama sharded variant updates:
  - Reader `[0] = input.addr`; Writer `[0] = output.addr; [1] = semaphore.addr` for each sender core.

Findings
- Addresses/semaphores refreshed for all active cores; per-core index/count parameters recomputed consistently.
- Hash computed in `AllGatherAsync::compute_program_hash` includes op hyperparams and input tensor properties.

Conclusion
- No program cache issues found. No failing cache test required.
