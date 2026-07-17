# Independent stage review: initial verdict

Date: 2026-07-17

Verdict: `more-work-needed`

The fresh read-only `$stage-review` subagent inspected the original goal,
`$multichip`, `$optimize`, `$tt-device-usage`, implementation/tests, every
stage XML, context contract, watcher log, and raw/processed profiler CSVs.  It
ran no hardware commands and changed no files.

## Required work

1. P1, trace/performance: decode capture bakes the Python `current_pos` into
   position-dependent slices, and the ten replays overwrite one cache slot
   with an unchanged input.  Headline latency and profiler evidence are eager,
   not traced.  Required: persistent refreshable input/position/RoPE state,
   advancing-position correctness with changed inputs, and like-for-like
   traced single/TP4 latency plus device/end-to-end accounting.
2. P1, topology: flat TP4 only measures replicated residual plus two immediate
   all-reduces against the five-collective compiler 2D path.  Required:
   coherent lower-movement residual, reduce-scatter/delayed-gather, fused CCL+
   matmul, persistent-buffer, and CCL-dtype candidates, or adapted minimal
   blockers.
3. P1, geometry: all four decode projection rows are `SLOW`; QKV/output/
   gate-up/down use `in0_block_w` 16/2/4/7 with missing output-subblock data
   and 40--54% DRAM utilization.  Required: BFP4/LoFi local-shape sweeps across
   grids, shards, block widths, `per_core_N`, and exposed subblocks.
4. P2, context: the 13.52-GB headroom is calculated before activations, CCL,
   trace, constants, and allocator reserve.  Required: conservative peak-live
   per-device allocation plan proving 131072 tokens still fit.

## Other concerns and controlled anomalies

- The candidate `flat_tp4_1x4_ring_correctness.xml` retains an old `2x2`
  testcase identifier; current `final_correctness.xml` is an explicit 1x4
  control.
- Final source/test formatting postdates the evidence and must be rebound on
  remediation.
- Length-7 prefill proves nonalignment but not long-context prefill cost.
- No actual two-layer stack or batch-greater-than-one hardware test exists.
- Ethernet watcher checks were disabled because the instrumented fabric router
  cannot fit its active-ETH kernel buffer.  Tensix watcher on all four devices
  is a documented, clean control.

This verdict is the input to the `$autofix` repair loop.  It does not satisfy
stage completion; a later fresh `clean-pass` review is required.
