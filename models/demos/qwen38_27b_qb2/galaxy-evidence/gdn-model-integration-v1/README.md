# Opt-in GDN model integration, 2026-10-07

Branch: `anatarajan/qwen38-gdn-integration-20261007`.
All hardware evidence comes from one TP4 submesh on the allocated Galaxy
`10.228.203.98`, using the pinned local checkpoint and existing native runtime.
No firmware, installed library, NFS or original checkpoint was modified.

## Real-weight layer result

`gdn-layer-integration-v4/layer.json` passes the original per-head PCC >= 0.999
and relative RMS <= 0.005 checks at batches 1, 8, 16 and 32 on all four ranks.
The test loads layer 0's actual weights. Its trace includes packed QKV/gate
projection, packed convolution, input preparation, recurrence, gated output
normalization and TP output projection. It excludes input normalization, MLP,
residual additions and other layers. The existing TP decoder enables packed
decode convolution for supported batches; both timing paths use that policy.
Five samples of 30 warm replays include trace dispatch and synchronization.

| Users per TP4 | Candidate us | Native us | Native / candidate |
|---:|---:|---:|---:|
| 1 | 321.41 | 271.23 | 0.84x |
| 8 | 545.95 | 536.82 | 0.98x |
| 16 | 730.55 | 1030.74 | 1.41x |
| 32 | 1091.87 | 1837.16 | 1.68x |

At B16/B32 these are 29.1%/40.6% latency reductions for this block. B1/B8
regress. They are not full-model or long-context speedups. The native path is
a timing control; the acceptance reference is FP32 recurrence evaluated on
actual per-rank convolution outputs and gates. After 64 traced updates, worst
state relative RMS is 1.12e-6 and worst output relative RMS is 1.90e-4. The
projected layer output is checked for finiteness; the strict FP32 comparison
is at recurrent state and pre-epilogue output. Persistent state and scratch
addresses must remain stable throughout trace replay.

## Numerical diagnosis and change

The first real-weight attempt (`gdn-layer-integration-v1`) passed B1/B8/B16,
but B32 failed: rank 2's worst head reached 1.094% output relative RMS against
the unchanged 0.5% limit. Diagnostic v2 compares two references using the same
initial state: ideal FP32 input preparation and the adapter's actual operands.
The recurrence matches the latter with worst B32 output error below 0.0076%,
while Q/K preparation differs by up to 0.086%. This isolates the observed
failure to the preparation boundary; the operator was not losing that much
precision given its actual inputs.

The candidate now performs Q/K sum-of-squares, reciprocal square root and
scaling inside the recurrence kernel, with direct FP32 unpack and SFPU math.
The generic RMSNorm-plus-scaling sequence is removed from the adapter. The
same real-weight case then passes without loosening any tolerance.
An intermediate v3 compile failed because scalar operations use the shared
`binop_with_scalar_tile_init` API; v4 corrects the API name. All host logs and
source snapshots remain in `/home/ttuser/qwen38-artifacts-20261007`.

Raw v1/v2/v4 receipts and compressed JUnit XML are preserved here.
`reproduction-sources.json` records the historical implementation and test
sources for each attempt, including the failed implementation. The current
runtime defaults remain native; the candidate requires explicit selection.

## Integration contract and remaining gates

- `tt/gdn_step/workspace.py` allocates output scratch during cache setup and
  retains each batch's addresses. Decode and trace capture only look up buffers.
- `decode_recurrence=single_step` selects the candidate only from decode;
  one-token and longer prefill retain the native recurrence.
- Q/K normalization is fused; the gated RMSNorm-times-z epilogue remains
  external. P1/P2 fusion and latency targets are not complete.
- B64 component recurrence is supported, but the real layer's DRAM-sharded
  projection rejects two tile rows (`M == 1`). B64 full-model support remains
  outstanding; do not report its potential capacity as measured throughput.
- One-replica full-model repeatability now passes. Eight-replica G0 and
  unchanged online reference evaluations still need to qualify this policy
  before serving promotion.

The persistent `qwen38-gdn-kernel-validation-v1-20261007.service` completed
successfully: 187 CPU tests plus 40 subtests, 4096 changing-input steps with
fused normalization, then repeatable one-replica full-model generation.
The final long-horizon state/output relative RMS values are 3.11e-7/4.91e-7.
Near-identity decay, input immutability and multiwave allocation rebinding also
pass. The fused normalization-plus-recurrence still misses every P1 latency
target: B1/B8/B16/B32/B64 take 36.72/108.75/175.24/308.58/596.43 us.

The 64-layer model loaded in 301.16 seconds. Warm B1 generation measured
34.97 tokens/s and 69.66 ms TTFT for the 63-token prompt; both greedy runs
returned the same fixed 128-token output. This is a short-context smoke test,
not a reference-evaluation pass. It deliberately continues beyond EOS for
timing; the raw text includes those trailing tokens.
The launcher is `demo/run_gdn_kernel_validation.sh`; its three arguments are
the isolated runtime, pinned source checkout and a new results directory.
`MODEL_WEIGHTS_DIR` must point to the verified host-local checkpoint. It has a
two-hour systemd deadline and uses the shared `/tmp/tt-device.lock` for hardware.

## Requested next sweep

Following the passing checks, `qwen38-gdn-perf-sweeps-v1-20261007.service`
was launched persistently to compare the native and candidate recurrence
under the same attention precision and other runtime settings, first on one
TP4 replica, then with actual eight-replica execution. Sweep supported batch
sizes against input lengths including 8K, 32K, 128K and near 256K, reserving
space for the fixed output budget. Record direct prefill input tokens/s,
decode output tokens/s, per-user decode speed, TTFT and output throughput over
the entire request. The TP4 sequence runs native then candidate, prioritizing
128K and near-256K, then 32K, 8K and 128 input tokens. Batch axes are
1/2/4/8/16/32/64; B64 is marked as an implementation guard. Larger configurations
that exceed the current token-pool guard are also explicitly unmeasured.
The launcher verifies the passing source/policy and 4096-step receipt before
starting. Its exact systemd command and source hashes are in
`launch-gdn-perf-sweeps-v1.json`. Full-Galaxy scaling has not been launched yet.
The report now plots the separate input and output rates
and exports JSON/CSV/PNG/SVG/PDF/HTML. Unsupported and capacity-guard cells
remain labeled and are never substituted with predicted values.

The native harness currently submits replica prefills sequentially. Its
aggregate input rate reflects that schedule; it is not a parallel-prefill
capacity estimate. The existing 1,179,648-token per-replica allocation guard
is a configuration limit, not a measured physical OOM boundary.

## Sweep allocation failure and recovery

The v1 native sweep completed 12 cells and stopped at B32 / 32K after 5,229.73
seconds. Prefill's MLP down projection could not allocate its 1.34 GB output:
each DRAM bank needed 160 MiB contiguously but the largest free block was
120 MiB, despite roughly 321 MiB total free per bank. A fresh-process retry
also failed during prefill, this time requesting a 2.68 GB buffer. It loaded
the model in 297.34 seconds and exited after 357.93 seconds. Both failures
closed devices cleanly. Prior sweep allocations are therefore not necessary
to trigger the failure; this does not establish a decode-kernel defect or
the maximum physical KV capacity.

The preserved measurements include:

| ISL | Batch per TP4 | Input tokens/s during prefill | Output tokens/s during decode | Output tokens/s/user |
|---:|---:|---:|---:|---:|
| 32,768 | 16 | 6,565.49 | 209.28 | 13.08 |
| 131,072 | 8 | 4,875.90 | 123.58 | 15.45 |
| 262,016 | 4 | 3,677.10 | 65.68 | 16.42 |

These are the matched native control, not measurements of the candidate.
`../gdn-sweep-recovery-v1/baseline-preview/index.html` provides all 12 points
with raw data and exportable graphs.

`run_gdn_perf_sweeps.sh` now accepts an optional fifth argument, the previous
sweep root. It copies valid completed cells into new attempt receipts after
checking model and precision hashes, runtime settings, prompt tokens, raw
timings and repeatability. Original receipts remain intact. An allocator
OOM closes the current process; only after confirmed device cleanup does
`run_sweep_attempts.py` start another process for the remaining cells. Such
cells retain `oom` status, and the final sweep becomes `completed_with_oom`.
Accuracy failures, changed source/settings, dispatch failures, timeouts and
unclean shutdown stop the controller. A completed measurement loop does not
mean every configuration passed or fits in memory.

Recovery validation passed 204 CPU tests and 40 subtests, including the real
tokenizer check. A first regression-test attempt exposed a missing required error-message
argument to the repository's `expect_error` fixture; that test-only issue was
corrected before launch. The v2 launch then stopped before hardware because
the copied snapshot omitted the tokenizer test's baseline JSON fixture. The
v3 snapshot restores that unchanged fixture, and all CPU tests pass with the
pinned tokenizer present. Its hardware collection then exposed another
omitted snapshot file, `pytest.ini`: discovery wandered into an unused stale
weights mount. Snapshot v9 restores the working configuration; the controller
also passes explicit pytest root/config paths. Collection through the actual
safe runner passes before the next launch. The recovery service is
`qwen38-gdn-perf-sweeps-v4-20261007.service`, using immutable source snapshot
`gdn-integration-source-v9`, the same native runtime, a 14-hour outer limit,
six hours per variant and the shared device lock. The 12 completed native
measurements were verified reusable before launch. The native control has
now completed: 24 measured cells, one recorded OOM at B32 / 32K, plus ten
unattempted capacity/implementation guards. Automatic recovery closed the
failed process cleanly and finished the other cells in a second process.
The complete baseline graphs and raw attempts are in
[`../gdn-native-sweep-v4/`](../gdn-native-sweep-v4/README.md). The candidate
sweep has started at 128K / B8; full-Galaxy scaling remains pending.

## Fixed-precision long-context bandwidth work

The user prioritizes output throughput at total contexts 32K, 128K and 256K,
targeting 2,680 output tokens/s per Galaxy. Lower precision is explicitly
deferred. Preserve BFP4 projection weights, BFP8 KV, BF16 activations and FP32
recurrent state, plus the existing numerical tolerances. The measured TP4
rates at B16/32K, B8/128K and B4/near256K are 209.28, 123.58 and 65.68 output
tokens/s. Eight-replica projections are not measured Galaxy throughput.

The running matched sweep varies workload geometry and recurrence variant;
it does not vary DRAM reader placement. Prior accurate-attention sweeps tried
KV chunks 128/256/512 and core caps 16/32. With 110 available cores and one
local KV head, B8 gets 13 cores/user under either cap; that comparison did
not change its allocation. At B4, the caps select different allocations.
Chunk/core-count tests therefore do not establish that bank-to-core mapping,
NoC routes or read pipelining have been optimized.

Relevant references, present unchanged in the pinned native runtime:

- `models/demos/deepseek_v3_b1/micro_ops/flash_mla/op.py`: explicit NOC0
  core groups near their assigned DRAM banks, matching bank ordering and tree
  reductions. MLA's shared K/V representation is not Qwen's attention
  contract; use the placement/dataflow ideas, not a model-architecture change.
- `models/demos/deepseek_v3_b1/micro_ops/dram_streaming_matmul/op.py` and
  `unified_kernels/dram_streaming_matmul.hpp`: bank-local worker assignment,
  contiguous K sticks, tensor-backed circular buffers, triple-buffered reads
  and transaction-specific barriers to keep another block in flight.
- `tech_reports/Saturating_DRAM_bandwidth/Saturating_DRAM_bandwidth.md`:
  bank-local readers, route/virtual-channel balance and pipelining. Its
  Wormhole/Grayskull utilization results are reference evidence, not measured
  Blackhole or Qwen model throughput.

Next comparisons should hold precision and mathematical work fixed, testing
bank-aware reader placement, bank/NoC distribution, buffering depth and read
granularity. Record useful bytes/time per operation, recurrence/projection/
attention/collective latency and full-model decode throughput separately.
Existing long-context standalone attention timings imply about 70--74% of
the plan's 512 GB/s/device in useful KV bytes/time; the roughly 40% full-model
roofline fraction is not a DRAM-counter measurement and cannot all be blamed
on attention's reader. Projection, GDN and inter-operation time still need
stage-level attribution.

The separate prefill token-budget change reduces transient activation rows
to test more resident decode users. `QWEN_PREFILL_MAX_BATCH_TOKENS=32768`
uses 4,096/2,048/1,024-token chunks per user at B8/B16/B32 respectively. The
default 131,072-row budget preserves all existing chunk sizes. Cache capacity,
precision and decode batch remain unchanged. It is a capacity experiment,
not proof of increased bandwidth, and has not yet been hardware-qualified.

CPU validation passes 228 tests plus 40 subtests, including preservation of
every prompt token/slot/nonzero prefix, aligned chunk budgets and persistent
queue behavior. All pre-commit checks pass. The raw compressed JUnit receipt
and exact launch command are in [`../long-context-capacity-v1/`](../long-context-capacity-v1/).

`qwen38-long-context-capacity-v1-20261007.service` is running persistently on
the allocated host, waiting for `qwen38-gdn-perf-sweeps-v4-20261007.service` to
finish. Source is the isolated `long-context-source-v1` snapshot; existing
runtime libraries, checkpoint and JIT cache are reused. No installed native
library or precision file changes. The launcher has a 14-hour overall bound,
six hours for its sweep and the shared safe runner's device lock. A live
dependency PID always keeps it waiting, even if a report says completed.
A terminal/missing dependency must also have both clean terminal receipts;
failed service results stop the queue. Source hashes are checked again before
execution. Each hardware allocation failure remains explicit and only restarts
the remaining cells after confirmed device cleanup.

The hardware plan is B16/32K as a control, B32/32K, B16/128K and B8/near256K.
Its explicit 2,359,296-token pool budget overrides only this experiment's
planning guard. Allocator snapshots are recorded after model load, before and
after KV allocation, after warmup and after measurement; timing windows omit
the snapshots. Numerical repeatability and fixed-precision configuration checks
do not replace future reference-evaluation qualification.

Reproduce with the pinned source on `PYTHONPATH` and the isolated runtime's
Python environment:

```sh
python -m models.demos.qwen38_27b_qb2.demo.run_long_context_capacity \
  --task /home/ttuser/kimi-prefill.Ubx2wY/runtime/qwen38-27b-20261006 \
  --source /home/ttuser/qwen38-artifacts-20261007/long-context-source-v1 \
  --weights /home/ttuser/qwen38-artifacts-20261007/checkpoint-pinned-1d4bf0f2 \
  --results /home/ttuser/qwen38-artifacts-20261007/long-context-capacity-NEW \
  --after-unit qwen38-gdn-perf-sweeps-v4-20261007.service \
  --after-results /home/ttuser/qwen38-artifacts-20261007/gdn-perf-sweeps-v4
```

Use a new results directory and launch through systemd using the saved command
for persistence. Inspect `queue.json` for waiting/running state and
`capacity/attempt-*/sweep.json` for live per-cell progress. To cancel only this
experiment, stop `qwen38-long-context-capacity-v1-20261007.service` with
`systemctl --user stop`; this does not remove the runtime or weights.

## Attention placement diagnostic

Branch: `anatarajan/qwen38-long-context-throughput-20261007`.
The persistent `qwen38-attention-placement-v1-20261007.service` is queued
after the capacity experiment. At 2026-10-08 00:40 UTC both controllers were
waiting on their live dependencies; the GDN full-model sweep was running.
There are no placement hardware results or model promotion yet. CPU validation
passed 236 tests plus 40 subtests, and the opt-in hardware test collects.
The launch manifest and compressed validation receipt are in
[`../attention-placement-v1/`](../attention-placement-v1/).

On one TP4 replica the diagnostic measures these six context/batch pairs:
32,768/B16, 131,072/B8, 262,016/B4, 32,768/B32, 131,072/B16 and 262,016/B8.
Each runs the native layout, row-major 64 cores, 64 cores at DeepSeek FlashMLA's
bank-proximity locations, row-major 80 cores, 80 cores in the outer columns,
then the native layout again to detect drift. The two equal-core-count pairs
isolate placement from available core count. Active cores per user are recorded
explicitly because the native split does not always use all 110 cores.

KV remains interleaved across DRAM banks. The 64-core candidate borrows
FlashMLA's physical distribution, not its bank ownership or KV representation.
Native SDPA requires sharded input or output for an explicit core subgrid;
these candidates use height-sharded output followed by conversion to the same
DRAM output layout as the native path. That conversion is included in timing.
The comparison against native therefore includes this layout cost; the matched
64/80-core pairs share it. No native library, precision or checkpoint changes.

All cases use 256-token KV chunks, BF16 Q, BFP8 KV, HiFi4 with FP32 accumulation
and accurate exponential. Inputs, page tables and numerical tolerances are held
fixed. Five samples of 100 trace replays exclude allocation and host readback.
Numerically failing candidates remain visible and are excluded from selection;
baseline drift above 3% disqualifies the cross-layout timing comparison. Useful
KV bytes divided by time is reported as effective bandwidth, not as a hardware
DRAM-counter measurement. A completed diagnostic does not establish a full-model
speedup or an evaluation pass.

The isolated `attention-placement-source-v1` snapshot and pinned native runtime
are reused. The controller verifies its source hashes after waiting for the
capacity service and its clean terminal receipt. It then repeats CPU validation
and runs hardware through the shared device lock with a 90-minute timeout;
the outer service has a 14-hour bound. Inspect `queue.json` for controller state
and `placement.json` for individual cases and paired comparisons.

```sh
python -m models.demos.qwen38_27b_qb2.demo.run_attention_placement \
  --task /home/ttuser/kimi-prefill.Ubx2wY/runtime/qwen38-27b-20261006 \
  --source /home/ttuser/qwen38-artifacts-20261007/attention-placement-source-v1 \
  --weights /home/ttuser/qwen38-artifacts-20261007/checkpoint-pinned-1d4bf0f2 \
  --results /home/ttuser/qwen38-artifacts-20261007/attention-placement-NEW \
  --after-unit qwen38-long-context-capacity-v1-20261007.service \
  --after-results /home/ttuser/qwen38-artifacts-20261007/long-context-capacity-v1
```

Use a new results directory and the saved systemd launch command for persistence.
To cancel this diagnostic alone, stop its named user service; leave the running
GDN and capacity services intact. Bank-local KV ownership, read-buffer depth,
transaction-specific barriers and NoC channel scheduling remain follow-up kernel
experiments, rather than knobs implemented by this placement diagnostic.
