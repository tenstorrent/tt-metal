# Runtime-argument advice discriminator

The advice-enabled target report (`tracy_postloop_target/stack32/decode_perf_report.txt`, lines84–89) reports a9us gap before GenericOp115392 and suggests tracing or moving fixed runtime arguments to compile time, estimating3us potential savings. The path already uses a warmed trace. This candidate tests the second suggestion on the accepted fast-zero reader.

Reader head stride/count become the first two compile-time words; TensorAccessorArgs starts at offset2 instead of0. Writer heads/stride become compile-time words3/4; its accessors start at5 instead of3. Per-core first head and buffer addresses stay runtime values. Compute source, arithmetic, CB capacities, loop scheduling, output layout and persistent state are unchanged. This removes four runtime words per active core (1664bytes at104cores), without claiming that all9us gap is argument traffic.

Host scheduling enumeration covers batches1..32 and waves7/64/80/86/96/104/110: every head appears exactly once. Hardware commands are in `../compile_args_jobs.json`: B6/B32 short,4K/1024-answer, and unchanged independent direct watcher. Results are pending at preparation. Compare with fast_zero, not with the earlier scalar-clear kernel; do not attribute its larger zero-fill gain to this argument change. Final profiling must still reconcile any remaining gap.
