# Native cache hits must be faster than misses

Required performance evidence before calling a migrated operation PR ready:
profile the **same C++ operation's hot and cold program-cache paths**. This is
separate from Python-versus-C++ performance and cannot establish that migration
itself is faster than Python.

Status: benchmark specification and review requirement. The current driver does
not run this benchmark or verify its statistical result. A `not_measured` review
receipt remains behavior-only completion, not PR-ready performance evidence.

## Define hot and cold precisely

| Case | In-memory program cache | Compiled kernel cache | What it measures |
| --- | --- | --- | --- |
| Cold path, primary comparison | Enabled; selected program absent | Warm | Hash/lookup, miss validation, descriptor/program construction and dispatch |
| Hot path, primary comparison | Enabled; selected program present | Warm | Hash/lookup, hit validation, runtime refresh and dispatch |
| Truly cold first use, separate diagnostic | Enabled; selected program absent | Cold, isolated cache | First use including kernel compilation; not the hot-path acceptance comparison |

Keep caching **enabled in both primary cases**. Disabling it skips hash lookup
in `launch_operation_with_adapter` and changes the path being compared. Merely
resetting `CacheEntriesCounter` resets its measurement, not the program cache.
Do not change shape, dtype, attributes or factory to manufacture the miss.
Do not delete shared disk kernel caches or reset the device between timed calls.

## Prepare a comparable experiment

- Use the final registered native entry, with generic-op fallback forbidden.
  Choose representative non-no-op cases for every supported factory/planner
  branch, including small calls where host overhead matters.
- Use one verified Release performance configuration for both paths, with
  descriptor-patching parity OFF. Verify its CMake setting and actual factory
  compile flags, loaded library identity and relevant debug/profiler settings.
  Watcher and other intrusive diagnostics must be off for normal timing.
  Preserve the parity-enabled correctness evidence separately; do not invalidate
  a resumable correctness build in place. Recheck correctness on the performance
  build before interpreting its timings.
- Initialize the process/device and compile the exact kernels before sampling.
  Prepare identical inputs outside timing; keep shape, placement, allocation
  policy and output lifetime consistent. Restore mutable inputs between calls.
- Own an isolated benchmark process/device session under the safe runner's
  cooperative lock. Synchronize before clearing its in-memory program cache.
  Use the evaluated branch's supported clear API, then explicitly enable the
  cache if that API disables it. Never clear another session's cache.
- Recreate the requested cache state before each sample/block, outside timing.
  Verify misses create the expected selected-operation entries and hits reuse
  them; exclude setup/readback operations from those observations. A zero delta
  alone can also mean a no-op or disabled cache and is not enough.

For a miss sample, clear the owned program cache and keep compiled kernels warm.
For a hit sample, populate the exact key with an untimed call, synchronize, then
time the repeated call. Alternate which sample comes first across blocks, setting
up each state independently. This avoids always timing a hit immediately after
its miss. Record first-call/JIT state rather than silently mixing it into samples.

## Measure the path, not device execution alone

Use paired native hit/miss host traces covering hash/lookup, validation,
miss construction or hit refresh, and enqueue. Verify actual zones/branch events
in the target revision; do not assume function names automatically create Tracy
zones. If those boundaries are unavailable, report path attribution as missing.

Also collect minimally instrumented public-call latency: synchronize before the
start timestamp, call the native entry, take the return timestamp, then
synchronize and validate the result outside that interval. Report synchronized
call-to-completion separately. Public-call time includes binding and output
allocation; label it honestly rather than calling it refresh-hook time. Device
kernel duration alone cannot show that the host cache-hit path is faster.

Run profiling separately from the low-overhead timing pass. The parity checker
deliberately reconstructs programs on hits; its timings are invalid here. Record
profiler overhead and preserve underlying correctness exit status: the safe
runner's `--profile` wrapper can mask pytest failures.

## Decision and evidence

Before measuring, freeze representative cases, timing boundaries, warmup/sample
counts, confidence method and a bounded measurement budget. Reuse the paired-block
method in [COMPARISON_GATE.md](COMPARISON_GATE.md#measurement-procedure), with
native-hit/native-miss replacing Python/native. Do not resample until a pass.

For each required case, report raw samples, median, p95, absolute savings and
`hot / cold` ratios. The primary speed check is the paired median host-path
difference `hot - cold`: its predeclared confidence interval must lie below zero.
If a minimum meaningful saving is required, declare it before measurement.
Report public-call results alongside path timings and investigate discrepancies;
apply any additional predeclared tail-latency/non-regression budgets separately.

- **Pass:** a repeatable hot-path improvement on each required case, correct
  outputs, verified hits/misses and complete configuration/trace evidence.
- **Fail:** evidence shows the hot path is not faster, or a required correctness
  or declared performance budget fails. Investigate before PR readiness.
- **Inconclusive:** noise overlaps zero, too few samples, unavailable path zones,
  ambiguous cache state or missing evidence. This is not a pass.

Keep logs, raw samples, traces, cache observations, build identities and analysis
under the performance worktree's `generated/generic_op_to_factory/`. The
independent review must inspect these artifacts. No fixed universal microsecond
threshold or speedup is claimed by this document.
