# GDN recurrence/epilogue fusion: correct, slower, not selected

October 11, 2026. The first dedicated-consumer prototype passed native build,
exact B16/B32 correctness, trace replay and distinct-address rebinding checks.
It **does not improve performance** and is not connected to the model or serving
policy. Current measured model performance remains 20.884562 TSU at B16/32K/TP4.

## What changed

`tt/gdn_fused/op.py` groups four register-resident recurrence producers with
one epilogue consumer in a single program. Two 512-byte L1 mailboxes per group
gather the four 128-byte value partitions. Each partition has its own cumulative
ready counter; consumed credits prevent overwrite until the full head is copied
into consumer-owned storage. Payload/state write barriers precede readiness.
No counter is reset inside a replay; program dispatch initializes semaphores.

The consumer reuses the qualified epilogue's gate/weight loading, full-head
RMS reduction and BF16 rounding. Separate compute cores preserve recurrence's
`math_approx_mode=False` and epilogue's `True`. Canonical FP32 state remains in
DRAM. The benchmark omits the diagnostic raw-output DRAM write/read. An optional
raw write supports direct validation. Existing model defaults are unchanged.

The descriptor uses a single mailbox CB over the union of both core roles,
ensuring identical local L1 offsets. Other CBs are role-local. Core schedules
cover every head/partition once, including partial final waves. This initial
fixed ratio trades recurrence parallelism for overlap: 24 groups use 96
recurrence cores and 24 epilogue cores, compared with 120 cores available to
each sequential operation in the unfused path.

## Validation and measured result

- 16 CPU tests passed, including skewed producers, repeated mailbox wrap,
  partial waves and a regression showing why one aggregate ready count cannot
  replace four independent partition counters.
- Native C++ JIT compilation and six B16/B32 smoke cases passed on physical
  ranks `[0,4,12,8]`. Four updates per case exactly matched state/raw/final
  output. Read-only inputs remained unchanged; output padding was zero and
  final values finite. Both raw-capture modes passed.
- The follow-up retains two independent live instances and asserts different
  addresses before exercising the cached descriptor. This strengthens the
  original smoke, which allocated sequential instances without proving their
  addresses differed.
- Traced comparison captures 16 complete recurrence-plus-epilogue calls per
  replay, then runs 2 warm replays plus 5 samples of 32 replays: 2592 state
  updates per arm. Before, three candidates and after have exact matching final
  state/output hashes. This is repeated-input trace validation, not a 4096-step
  changing-input or model-eval qualification.

| Batch | Unfused before / after | Fused 24 groups | Fused 20 groups | Fused 16 groups |
|---|---:|---:|---:|---:|
| 16 | 112.673 / 112.462 us | 118.769 us | 136.953 us | 155.026 us |
| 32 | 194.263 / 194.847 us | 198.529 us | 236.800 us | 275.975 us |

These are fenced host timings of **traced complete component boundaries**,
including synchronization and delivery. They exclude projection/preparation
and are not full-model token times. Each table entry is the median of five
samples. Before/after control drift is about 0.19% at B16 and 0.30% at B32.
The best fused variant is 5.51% slower at B16 and 2.04% slower at B32 than the
mean control. Applying the B16 delta to 48 layers projects about 0.30 ms of
extra work, not a saving. No full-model run or GPQA is justified for this loser.

The result is consistent with lost core parallelism and handoff overhead
outweighing the saved boundary; it does not individually prove their shares.
Fewer groups regress further. The original 1-3-ms fusion range remains an
unverified target for a broader preparation/dataflow redesign, not a gain from
this implementation. Do not add it to performance projections as achieved.

## Follow-up and limitations

Keep this opt-in prototype as a tested handoff mechanism. Prioritize removing
operand-formatting work or delivering prepared operands directly before
spending further effort on this fixed producer/consumer split. A same-core
alternative requires explicit preservation of the two approximation modes and
the ordered full-head reduction. More consumer cores alone would further
reduce recurrence capacity.

Both services completed with exit 0 and clean teardown; no reset was needed.
They ran independently of SSH under `/tmp/tt-device.lock`, with a 30-minute
systemd limit and shorter pytest/controller limits. The v2 log contains a native
trace-allocation warning despite matching outputs; allocation-tracker checks
remain required before any model integration. Other warnings were deprecated
tile accessors and process-priority requests. No claim of production trace
safety or combined feature qualification follows from this component test.

Pinned native Metal is `a08819ddbe23077f8037d3802303939064868ff6`.
Frozen task-local source directories are `gdn-fused-source-v1` and
`gdn-fused-source-v2` under `/home/ttuser/qwen38-artifacts-20261007`.
The task-local safe wrapper uses the allocated Galaxy's `tt-smi -glx_reset`
spelling if recovery is needed; installed software was unchanged.
`raw/gdn-fused-publication-v1/manifest.json` contains original hashes/lengths
for 21 collected files; XML/logs are losslessly compressed locally. Controller
scripts, source manifests, exact launch commands and terminal receipts are
retained. [summary.json](summary.json) records the comparison arithmetic.
