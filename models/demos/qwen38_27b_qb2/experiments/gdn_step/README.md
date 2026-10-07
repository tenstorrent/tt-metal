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

Each work item owns one user/value-head state matrix. Q and K arrive normalized,
Q scaled by `128**-0.5`, and gates already transformed to decay and beta. V is
unscaled. Compact FP32 row-major vectors are expanded into broadcast tiles in
L1. The FP32 tiled state is read from DRAM once and written once to the same
address. All recurrent arithmetic uses direct FP32 unpack and SFPU operations,
including the output reduction. There is no padded time dimension or temporary
DRAM state output. Convolution, normalization, and output gating are outside this
experiment's timing.

The reader and writer assign head `first + item * stride`; each core can process
multiple independent heads. This covers batches beyond one head per core and
uneven final waves without separate host dispatches.

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

Compute reads CB7 before producing CB8. The writer waits for CB8 **before**
reclaiming CB7, so it cannot free state while the output reduction still reads it.
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

The test covers TP4 batches 1/8/16/64, all four ranks, cancellation-sensitive
inputs, input immutability, buffer rebinding, 4,096 changing-input steps, and 64
near-identity decay-only steps. Every head must meet PCC ≥ 0.999 and relative RMS
error ≤ 0.005. Timing uses five warm samples of 100 trace replays and includes
dispatch. The separate P1 latency target is state read/write bytes divided by
512 GB/s plus 10 microseconds; the report does not treat accuracy as a latency
pass. Full-model integration and unchanged reference evals remain required.

`run_galaxy_serving.sh` optionally accepts `QWEN_GDN_STEP_EXPERIMENT_DIR` to run
this experiment before the baseline server occupies the device lock. A failed
candidate is recorded without promotion and does not prevent baseline GPQA.
The baseline still requires its passing G0 receipt and matching model hashes;
the existing reset-under-lock path handles a dirty device after a failed test.
