# AutoDebug report

Fresh source-only investigator: `/root/autodebug_fast`. No hardware was opened
and no implementation files were modified during diagnosis.

## Starting evidence

- Formal stage review: `stage_review_round1.md`, verdict
  `more-work-needed`.
- Selected decoder: two bare `ttnn.all_reduce` calls per decode; constructed
  `CCLManager` is not used by them.
- Final wall timing: EP4 prefill 26.6769 ms versus 13.4083 ms single-chip.
- Final watcher disables Ethernet; same-firmware prior TP2 evidence records an
  ACTIVE_ETH heartbeat failure during full-watcher fabric teardown.

## Hypothesis 1: persistent decode collectives

Hypothesis: repeated synchronous collective setup contributes materially to
decode; the GPT-OSS persistent-semaphore reduce-scatter + all-gather family may
reduce warmed latency without changing numerics.

Focused A/B: current `ttnn.all_reduce` versus `MeshConfig.allreduce` at only
the two existing reduction call sites, padding hidden width 2,880 to 2,944 so
each of four ranks receives 736 tile-aligned values, then slicing to 2,880.
Hold dtype, residual contract, weights, and all other programs fixed. Compare
PCC, trace replay, latency, and collective profile rows. An API/alignment
failure or unchanged/slower latency refutes selection of the candidate.

Source blocker: GPT-OSS explicitly disables fused minimal matmul +
reduce-scatter on Blackhole because of race #46181. This closes only that fused
family and must not be treated as a blocker for persistent RS+AG.

Minimal fix boundary: collective policy/config, two call sites, internal
padding/slicing, and candidate test/measurement only.

## Hypothesis 2: EP4 prefill expert grid

Hypothesis: the 53.04% sparse prefill category is constrained by the selected
3x5 gate/up grid. First compare only a legal 5x6 gate/up grid while keeping the
5x6 down grid, chunk 128, DRAM placement, dtype/fidelity, routing, and all
other programs fixed.

Compare identical route counts, PCC, total prefill latency, and sparse-op time.
Flat/slower timing or a program/L1 failure refutes the grid hypothesis. Then
isolate chunk-size and DRAM-materialization candidates one at a time; do not
bundle them with precision changes.

Minimal fix boundary: EP prefill program configuration and its candidate
parser/tests.

## Hypothesis 3: full Ethernet watcher

Hypothesis: on firmware 19.8.0, full Ethernet watcher permits the selected EP4
correctness bodies to pass and then reports a stopped ACTIVE_ETH heartbeat
during fabric teardown, matching the retained TP2 control.

Run the selected EP4 canonical sliding/full correctness path with full watcher
in a profiler-free process. Retain ordered body PCC/pass output, watcher
failure, exit status, firmware/device identity, and teardown point. If it
passes fully, use the full-watcher run. If the body passes but teardown aborts,
classify only with the exact retained failure and same-firmware prior control;
do not call the command a pass. A body-time error is a decoder/fabric failure.

After any heartbeat/watcher failure, stop all experiments, capture bounded
health evidence, reset all four devices using the approved recovery sequence,
and prove mesh reopen before continuing.

## Repair order

1. Persistent CCL focused A/B and trace/correctness verification.
2. EP4 prefill grid, then isolated chunk/memory candidates if needed.
3. Full Ethernet watcher last, because its expected teardown failure may
   require device reset/recovery.
