## Program cache review: experimental/ccl/all_reduce_async

OP reviewed
- `ttnn::operations::experimental::ccl::all_reduce_async` (old infra; `ProgramWithCallbacks`).

Creation path summary
- Sets sender and reduction cores; builds reader/writer for senders and receiver/compute for reducers.
- Uses globally allocated CBs for output and reduction buffers; writer args contain semaphore and fabric connections.

Override behavior (cache-hit path)
- Sender cores: reader `[0] = input.addr`; writer `[1] = semaphore.addr`.
- Updates dynamic CB addresses for output and reduction CBs to new buffers.
- Reduction readers per output core refresh semaphore address.

Findings
- All runtime-only addresses (input/output/semaphore/CB) updated; indices/counts derived from shapes remain stable.

Conclusion
- No program cache issues found. No failing cache test required.
