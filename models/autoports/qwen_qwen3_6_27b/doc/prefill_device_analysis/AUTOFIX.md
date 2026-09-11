# AutoFix Report

## Starting Evidence

The source and triage diagnosis is [AUTOTRIAGE.md](AUTOTRIAGE.md). Its proposed
ownership control is now verified for the reduced S128 experiment described
below; its exact reshape-map victim hypothesis remains unproven.

The original `traced` candidate retained stable recurrence inputs and outputs,
but released intermediate tensor storage that later trace replay could reuse.
It also first warmed input-copy programs after capture. The original artifacts
are [traced_s128.json](artifacts/traced_s128.json),
`/tmp/qwen_real_prefill_traced.log`, `/tmp/qwen_trace_debug.log`,
`/tmp/qwen_trace_triage.txt` and `/tmp/qwen_trace_triage.log`.

The S128 run produced hidden-state errors up to `3.78e25` and wrong logits
top-1. The S32 debug run hung in the tail reshape after a recurrent-cache copy.
The original 20.651 ms linear-layer timing is **rejected** because correctness
failed. No production source or precision policy was changed.

## Hypothesis Experiments

### 1. Trace lifetime crosses later allocations

- **Hypothesis:** recurrence replay can overwrite live allocations created after
  capture, including program-owned buffers and surrounding eager activations.
- **Experiment:** the coordinating agent reran the original candidate with
  startup `TT_METAL_TRACE_ALLOC_TRACKING=1` and
  `TT_METAL_TRACE_ALLOC_TRACEBACKS=1`, retaining program-cache reporting.
- **Result:** `/tmp/qwen_trace_tracker.log` rejects the first replay because
  three live `CopyDeviceOperation` program buffers were allocated after
  capture, directly from the six-input refresh loop. IDs 8931/8941 have BF16
  contexts; 8951 has a BFP8 context. The run exits cleanly before the hang.
- **Verdict:** the allocation-lifetime exposure is verified. The conservative
  tracker does not prove exact buffer-address overlap or identify the later
  overwritten reshape map.

### 2. Preserve ownership across replay

- **Hypothesis:** warming persistent buffers before capture and retaining
  recurrence intermediates prevents surrounding eager work from reusing the
  captured intermediate addresses.
- **Experiment:** the coordinating agent ran `probe.py` with candidate
  `traced_keepalive`, sequence 128, batch 1, layers `[0,3]`, and three measured
  iterations. The recorded run is `/tmp/qwen_trace_keepalive.log`; parameters,
  precision and results are in
  [traced_keepalive_s128.json](artifacts/traced_keepalive_s128.json).
- **Fix under test:** run the entire eager prefill once before enabling trace;
  warm all six exact input-copy variants before capture; wrap capture with
  [preserve_trace_tensors](trace_keepalive.py). The helper retains Python-visible
  operation outputs and intercepts explicit Python `ttnn.deallocate` while
  capture is open. It temporarily disables fast runtime so post-operation
  hooks execute, then restores the runtime setting and original function.
  `probe.py` keeps the yielded list alive alongside the trace until release.
- **Result:** one shared recurrence trace retained 420 Python tensor objects.
  Captured layer 0 and layer 3 outputs, logits, layer 0 conv/recurrent caches,
  and layer 3 key/value caches all have `max_abs=0` and `relative_l2=0` against
  the eager baseline. Logits top-1 is equal. Three further timed prefills finish,
  and the mesh closes cleanly.
- **Verdict:** the combined ownership-preserving setup fixes the observed
  reduced S128 corruption/hang and preserves exact measured reference values.
  Warmup and retention were tested together; their individual contributions
  have not been isolated by ablation.

The correctness captures come from iteration 1, the first traced prefill after
the eager warmup. S128 uses four S32 chunks, so this checks a trace reused with
successive chunk inputs and recurrent state. The three timing iterations
establish repeated completion, but the probe does not independently save and
compare their hidden/cache/logit tensors. These are distinct evidence claims.

### 3. Specific reshape mapping victim

- **Hypothesis:** recurrence replay overwrites the tail reshape's cached segment
  mapping, whose offsets and lengths then cause invalid local data movement.
- **Evidence:** the tail reshape matches triage's `[384,1,128]` BF16 input and
  48 workers/device; its DRAM mapping was first allocated after capture. All
  192 readers wait for input FIFO capacity. The map drives writer copy offsets
  and lengths. See the producer/consumer ledger in `AUTOTRIAGE.md`.
- **Verdict:** still uncertain. No captured map bytes or allocation-overlap
  measurement identify this as the exact victim. The passing ownership fix
  supports the broader lifetime diagnosis without proving this particular
  mechanism. No reshape kernel change was made.

## Final Status

**Fixed for the reduced S128 experimental candidate.** The coordinating agent
performed all device runs; this report's author only inspected their artifacts.
Full-model and watcher validation remain with the coordinating agent and are
outside the evidence reported here.

| Warmed median | Eager baseline | Protected recurrence trace | Change |
| --- | ---: | ---: | ---: |
| Linear layer 0 | 51.119 ms | 20.689 ms | 2.47× faster; 59.5% lower |
| Full-attention layer 3 | 2.193 ms | 2.411 ms | 10.0% higher in this sample |
| Reduced generator | 62.937 ms | 33.175 ms | 1.90× faster; 47.3% lower |

These are three-sample medians from real Qwen3.8 revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, TP4 batch 1, chunk size 32, with
the unchanged selected precision policy. Per-layer measurements synchronize
the mesh. Layer 3 receives layer 0's output directly in this reduced stack.
The measurements are not full-model latency or serving TTFT.

Remaining limits:

- 420 retained objects are not 420 unique allocations; views can share storage.
  This artifact does not yet record retained bytes. Deduplicate buffer addresses
  by memory type and account for tile-packed storage before reporting memory.
- The helper cannot retain temporaries created and freed wholly inside C++
  composite operations. Passing this shape does not prove coverage for other
  sequences, program variants, batch allocations or serving lifetimes.
- The runtime warning can remain because its tracker conservatively marks
  post-capture allocations. Neither suppressing that warning nor excluding
  program-cache buffers establishes correctness.
- `TRACE_DEBUG` in the original experiment computed its eager oracle after
  replay using potentially overwritten source tensors. That comparison alone
  did not prove a recurrence arithmetic error. The accepted result instead
  compares model-visible values with an independent eager baseline artifact.
- The helper is experimental, changes process-wide behavior during capture,
  and is used only in serialized probe setup. Production integration requires
  explicit ownership with suitable lifetime and memory accounting.

Validation performed by this report's author: host-only JSON inspection and
arithmetic checks, plus `python3` AST parsing of the helper without importing
TTNN. `pre-commit` is unavailable. Python/docs-only changes require no C++ build.

## Coordinating-agent follow-up

Full 64-layer S128 prefill subsequently passed exact captured comparisons and
measured 2537.329 → 998.780 ms with this protected experiment. Packed tensor
accounting now records approximately 75.6 MiB/device across 424 unique buffers,
including persistent inputs/outputs (not peak allocator usage). S33 with 32
allocated slots and only slot 17 active also passed exact reduced-stack checks.
Watcher coverage remains unavailable: ACTIVE_ETH instrumented program size
exceeds the configuration buffer during mesh open, before model execution.
See the main README and retained startup log for these later runs. The initial
independent report above remains scoped to its original evidence.
