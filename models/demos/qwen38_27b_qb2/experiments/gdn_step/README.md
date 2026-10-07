# Single-token GDN candidate

This experiment implements the P1 recurrence from the Galaxy plan. It is **not
imported by the qualified model**. The first TP4 hardware run passes compilation
and accuracy, including 4,096 changing-input steps, but misses the P1 latency
target at every tested batch. CPU tests alone do not qualify the kernel.

Measured on `10.228.203.98` on 2026-10-07, five samples of 100 warm trace
replays (recurrence and dispatch only):

| Users per TP4 | Median call (us) | P1 target (us) |
| ---: | ---: | ---: |
| 1 | 68.67 | 13.07 |
| 8 | 92.51 | 34.58 |
| 16 | 157.21 | 59.15 |
| 64 | 454.64 | 206.61 |

All four ranks pass input immutability, allocation rebinding, cancellation,
per-head accuracy and near-identity decay checks. At step 4,096 the maximum
relative RMS error is 3.05e-7 for state and 4.48e-7 for output. Raw results and
kernel hashes are in `../../galaxy-evidence/gdn-step-candidate-v1/`. This is
accuracy evidence for the standalone recurrence, not a model speedup or
reference-evaluation result. Integration and further latency work remain.

The next isolated candidate supports `value_splits=1/2/4`, with the one-partition
baseline retained. Each work item owns a disjoint 128-by-128, 128-by-64, or
128-by-32 section of one user/value-head state matrix. At batch 1 this exposes
12/24/48 independent workers per chip; higher batches use multiple waves.
Every output section is a disjoint 128-byte-aligned part of its compact row.
There is no cross-core reduction: each worker owns the entire key dimension.
Q/K/gate vectors are replicated across partitions; state traffic remains one
read/write per element. Hardware speed and accuracy for split variants are
pending. Thirty-two CPU tests pass for complete/disjoint coverage, uneven waves,
invalid partitions, and the per-head accuracy gate.

Q and K arrive normalized,
Q scaled by `128**-0.5`, and gates already transformed to decay and beta. V is
unscaled. Compact FP32 row-major vectors are expanded into broadcast tiles in
L1. The FP32 tiled state is read from DRAM once and written once to the same
address. All recurrent arithmetic uses direct FP32 unpack and SFPU operations,
including the output reduction. There is no padded time dimension or temporary
DRAM state output. Convolution, normalization, and output gating are outside this
experiment's timing.

The reader and writer assign work item `first + item * stride`, with
`head = work_item / value_splits`. Each core can process multiple independent items.
This covers batches beyond one item per core and uneven final waves without
separate host dispatches. CB capacities and tile counts shrink with the value
partition and participate in the program cache key.

| CB | Content | Producer | Consumer/release |
| --- | --- | --- | --- |
| 0–2 | Q columns, K columns, V rows | Reader | Compute |
| 3–4 | Scalar decay, beta | Reader | Compute |
| 5 | Old FP32 state | Reader | Compute |
| 6 | FP32 delta | Compute | Compute |
| 7 | Updated FP32 state | Compute | Writer |
| 8 | Output rows | Compute | Writer |
| 9 | Compact input scratch | Reader only | Reader only |
| 10 | Compact output scratch | Writer only | Writer only |

Compute reads CB7 before producing CB8. The writer starts the state DRAM write
as soon as CB7 is ready, overlapping it with the output reduction; both only
read these L1 pages. The writer waits for CB8 **and** the NoC write barrier
before reclaiming CB7, so it cannot free state while either reader still uses it.
Only the writer pops CB7/8. Scratch CB9/10 are private storage and carry no tokens.
Every work item waits for its complete old state before its writeback can begin.
Separate work items own disjoint state, and consecutive invocations are ordered
on the same command queue.

`generic_op` hashes kernel source text, accessor metadata, CB geometry, and core
placement. The pinned runtime's descriptor adapter copies raw runtime arguments
on cache hits. The wrapper rebuilds the Python descriptor for every invocation;
it does not cache buffer addresses. The device test alternates simultaneously
live allocations to verify rebinding before tracing.

Run under the shared device lock with the bounded launcher:

```bash
bash models/demos/qwen38_27b_qb2/demo/run_gdn_step_candidate.sh \
  "$QWEN_TASK_ROOT" "$QWEN_TASK_ROOT/gdn-step-candidate-v1"
```

The test now covers TP4 batches 1/8/16/32/64 at every selected partition count,
all four ranks, cancellation-sensitive
inputs, input immutability, buffer rebinding, 4,096 changing-input steps, and 64
near-identity decay-only steps. Every head must meet PCC ≥ 0.999 and relative RMS
error ≤ 0.005. Timing uses five warm samples of 100 trace replays and includes
dispatch. The separate P1 latency target is state read/write bytes divided by
512 GB/s plus 10 microseconds; the report does not treat accuracy as a latency
pass. `QWEN_GDN_VALUE_SPLITS=1,2,4` selects the partition sweep (the default);
each variant gets its own 4,096-step accuracy run. Full-model integration,
the fused gated-RMSNorm epilogue, and unchanged reference evals remain required.

`run_galaxy_serving.sh` optionally accepts `QWEN_GDN_STEP_EXPERIMENT_DIR` to run
this experiment before the baseline server occupies the device lock. A failed
candidate is recorded without promotion and does not prevent baseline GPQA.
The baseline still requires its passing G0 receipt and matching model hashes;
the existing reset-under-lock path handles a dirty device after a failed test.
