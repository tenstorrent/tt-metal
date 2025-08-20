## Program cache review: experimental/ccl/all_to_all_async

OP reviewed
- `ttnn::operations::experimental::ccl::all_to_all_async` (old infra; `ProgramWithCallbacks`).

Creation path summary
- Builds one sender core per link and multiple receiver cores; sets per-core slice, stride, and offset args.
- Writer args include intermediate DRAM address, final output base, semaphore and fabric connections.

Override behavior (cache-hit path)
- Sender reader `[0] = input.addr`; sender writer `[0] = intermediate.addr; [1] = output.addr; [2] = semaphore.addr`.
- Receiver writer `[0] = output.addr`; receiver reader `[0] = intermediate.addr; [1] = input.addr; [2] = semaphore.addr`.

Findings
- All dynamic addresses updated; per-core ranges recomputed from shapes (hashed properties).

Conclusion
- No program cache issues found. No failing cache test required.
