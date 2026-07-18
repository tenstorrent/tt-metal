# AutoFix outcome

Round-1 `$stage-review` returned three required findings. `$autofix` tested
each hypothesis in isolation, kept only measured improvements, and preserved
the rejected candidates as reproducible evidence.

## Decode collective

Hypothesis: persistent-semaphore reduce-scatter plus all-gather could replace
the two decode ring all-reduces.

Result: correct and trace-safe, but refuted on performance. Explicit
2,880->2,944 padding is required for four 736-wide tile-aligned pieces. Both
layer kinds pass, while 500-replay decode is 0.638599 ms versus 0.598641 ms
for current all-reduce. The candidate profiler contains real
`ReduceScatterMinimalAsync` and `AllGatherAsync` rows. Production remains
all-reduce. Fused matmul + minimal RS is separately blocked on Blackhole by
GPT-OSS race #46181.

## EP4 prefill

Hypothesis: the sparse programs and dtype/layout boundaries caused avoidable
prefill time.

Result: proven and fixed. A legal grid/chunk/rewrite sweep selected 9x10,
subblock 1 for both gate/up and down, chunk 128, with BF16 immediately after
each sparse result and BFP8 restored only for the final local-partial
collective. S=128 prefill improves from 26.6769 to 22.6186 ms (15.2%) with
real sliding/full PCC, exact-route, and non-aligned synthetic tests passing.

## Watcher

Hypothesis: full Ethernet watcher might run the selected path or expose a
scoped incompatibility.

Result: the incompatibility is scoped and retained. With Ethernet watcher
enabled, both parametrizations fail before model execution because the
instrumented ACTIVE_ETH program is 27,920 bytes versus a physical 25,600-byte
kernel-config buffer. Devices were reset and the 1x4 ring reopened. The
maximal legal worker/Tensix watcher run passes both layer kinds on the promoted
path with no watcher/NoC/assert/hang/timeout.

## Final disposition

`$autofix` succeeded; no fallback to a failed-stop state was needed. Exact
commands, timings, PCC, profile tables, recovery steps, and artifacts are in
`work_log.md`, `perf/perf_report.md`, and the `logs/` and `perf/` directories.
