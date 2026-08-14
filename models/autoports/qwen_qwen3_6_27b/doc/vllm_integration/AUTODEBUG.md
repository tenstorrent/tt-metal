# AutoDebug Report: reduced-target pinned-memory alignment message

## Scope and observations

- Target: Qwen/Qwen3.6-27B reduced four-layer TP4 vLLM adapter bring-up.
- Reported observation: after mesh open, the process printed
  `Pinned source memory start address ... must be aligned 64 B` and was still
  alive roughly 35 seconds later while model weights were being loaded. It was
  then terminated manually.
- No TT fatal, assertion, exception, traceback, non-zero process exit, hang
  backtrace, or timeout artifact accompanies the message.
- Inspection was source-only. No TT device was opened and no hardware command
  was run.
- The requested fresh `.agents/scripts/autodebug.sh` session was attempted,
  but its isolated command launcher could not execute even `pwd` because its
  sandbox wrapper was unavailable. The report below was therefore verified
  directly against the checked-out source rather than fabricated from that
  blocked run.

## Headline finding

### The alignment line is an informational rejected-optimization diagnostic, not a failure

In `tt_metal/impl/buffers/dispatch.cpp:1154-1193`,
`write_to_device_buffer` considers a direct transfer from pinned host memory.
For an unsharded, unpadded device buffer it computes
`src_region_start = src_ptr + buffer.root_buffer_region().offset`. If that
address is not aligned to the L1 read alignment, lines 1175-1180 call
`log_info`; they do not throw, assert, return, or mark the command failed.
`use_pinned_transfer` remains `false`.

The same function subsequently executes the ordinary non-pinned dispatch path
when `use_pinned_transfer == false`. Therefore this line means only that one
host-to-device write could not use the direct pinned-memory fast path. It does
not establish that initialization or prefill failed.

This explains all available observations with no model-code assumption:

1. A host buffer was pinned and a source region within it started at an address
   not divisible by 64.
2. Dispatch logged that direct pinned transfer was ineligible.
3. The process remained alive because it continued initialization/weight
   loading through the fallback path.
4. Manual termination after about 35 seconds created the appearance of a
   failed attempt, but no failure boundary was observed.

Verdict: **the reported run is inconclusive and most likely was terminated
prematurely; there is currently no verified alignment bug to fix.**

## Can the exact host tensor be identified statically?

No. The log contains only the computed address, not the TTNN operation, tensor
shape, Python call site, root-buffer offset, or dispatch command identity.
Initialization performs many host-to-device transfers through helpers such as
`_replicate`, `_shard`, and `_shard_decode_weight`. Any pinned host allocation
whose dispatched root region has a non-64-byte offset can emit the same line.
The new `attention_cache_blocks=1` path also allocates and transfers attention
cache tensors, but nothing in the source ties this particular message to those
tensors; naming one of them as the cause would be speculation.

The stage diff does not introduce a custom pinned-memory API, change dispatch,
or convert this info log into an error. Its initialization changes are limited
to reducing temporary attention-cache block allocation and adding the vLLM
adapter. None supplies evidence for an alignment-induced stop.

## Focused verify/refute experiments

Run these serially under the TT-device-usage rules. Do not change tensor
alignment before proving an actual failure.

### Experiment 1: bounded unchanged rerun (decisive first control)

- Run the exact reduced four-layer TP4 script unchanged, preserving a complete
  timestamped console log and allowing a realistic model-load timeout (for
  example 10 minutes rather than 35 seconds).
- Add coarse existing-script progress markers around model construction,
  adapter construction, KV allocation, prefill entry, and prefill return; do
  not change tensor creation.
- Prediction if this report is correct: the alignment info may recur, but
  execution advances beyond it and reaches a later marker or completes.
- Verified bug criterion: a bounded timeout with no progress, a traceback,
  process exit, or a TT fatal/assert distinct from this `log_info` line.

### Experiment 2: locate a real stall only if Experiment 1 times out

- Before killing the live timed-out process, capture Python/native stacks and
  `tt-triage` evidence.
- If stacks are in weight deserialization, conversion, or normal H2D traffic,
  increase the load bound and treat the run as slow initialization.
- If stacks repeatedly remain in
  `FDMeshCommandQueue::finish_nolock`/`enqueue_write_mesh_buffer`, follow the
  device recovery and AutoTriage path; the alignment info still is not proof
  of causality.

### Experiment 3: identify the emitting transfer only if it materially matters

- Temporarily enable dispatch/operation correlation or add narrowly scoped
  diagnostics at the `log_info` site recording buffer size, region offset,
  page size, layout, and whether fallback completes. This is preferable to
  guessing from Python tensors.
- Prediction: one transfer has `src_ptr + region.offset` modulo 64 nonzero and
  completes via the ordinary path.
- Only if fallback itself fails should the correlated TTNN call be reduced to a
  minimal host-to-device write and its host/root-region alignment repaired.

### Experiment 4: adapter-stage continuation

- Once reduced initialization completes, rerun the intended non-aligned 65/63
  prompt check and stale token/current-position/page-table checks.
- Treat failures there by their actual exception/output/timeout evidence. Do
  not attribute them retrospectively to the earlier alignment info line.

## Other potential issues (not causes of this observation)

- `generator_vllm.py` performs many initialization transfers and its full
  serving contracts still require runtime validation. This report does not
  validate cache ownership, sampling, async overlap, or non-aligned prompt
  correctness.
- Previous stage notes describe genuine command-queue stalls that happened to
  print the same informational line. Their later stack and recovery evidence,
  not this shared log message, is what established those stalls. Reusing the
  message alone as a failure signature would conflate unrelated events.

## Recommended action

Make no implementation change for host alignment. Correct the work log's
“failed during initialization” wording to say the attempt was manually stopped
while still loading after an informational pinned-transfer fallback, then run
Experiment 1. Escalate to AutoTriage/AutoFix only if that bounded run supplies
an actual failure or stall signature.
