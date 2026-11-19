# Collector audit before timed attempt

CPU-only source review of `tools/benchmark_roofline.py`,
`tools/benchmark_accounting.py`, `tt/benchmark_phases.py`, adapter integration,
and installed benchmark_stage 0.1.24 `roofline.py`. No device import or execution.

## Findings requiring correction or an explicit checked invariant

1. **Final host processing can be omitted.** `partition()` extends the final CQ
   event only by that event's `host_complete_ns`. It never checks host completion
   timestamps on earlier events. Async readback processing need not be assumed
   to finish in the same order as prefill queue markers. A CPU counterexample
   with decode completion 100 / host completion 500, followed by prefill
   completion 200 / host completion 250 yields a partition ending at 250,
   omitting the final 250 ns. Require every selected decode ticket's host
   completion, end at the maximum selected host/completion timestamp, and retain
   which phase owns any final host tail. Alternatively validate and document the
   stronger runtime ordering invariant before using the current shortcut.

2. **Work coverage checks are weaker than the claims.** `select_events()` checks
   only maximum prefill end=4096 and decode count>=127. Missing prefill ranges,
   duplicate decode positions, or position holes can pass those tests. Validate
   per-request scheduler prefill ranges cover [0,4096) without holes, distinguish
   internal rollback replays, and require consecutive decode positions beginning
   at 4096. Retain extra queued steps separately as executed work beyond the
   127 steps needed for 128 emitted tokens. Also reject missing request identity
   and missing host-processing completion for nonempty selected decode events.

3. **Mixed-phase observation uncertainty has no per-phase bound.** FIFO event
   completion intervals are a defensible exclusive CQ timeline convention;
   they are not exact device intervals. The observer can find several events
   completed in one poll and assign nearly all time to the first phase, even
   when later phases consumed part of it. Raw completion brackets and maximum
   poll gap are preserved, which is good. Derive/retain each phase's possible
   duration range from the brackets (or reject unresolved transition uncertainty)
   and identify the headline as host-observed FIFO attribution. A single global
   maximum uncertainty does not show whether the reported prefill/decode split
   is resolved. Do not describe the scalar estimates as exact completion times.

4. **Readiness checks only configuration.** The `check` action validates runtime
   precision and exports a recorder snapshot but does not establish that both
   phases emitted completed events. The requested pre-accuracy smoke must prove
   prefill and decode completion, host readback association, and usable accounting
   before the timed attempt. This can be a separate setup smoke; runner check
   happens before its own warmup.

## Validated structure and qualifications

- The written roofline object matches the installed validator's schema:
  numeric-string batch keys; performance file hash; canonical parsed server
  digest; raw request/input/output counts; positive work/time/peak fields;
  required scope, method/source strings and retained evidence hashes. The
  validator additionally requires each phase duration <= raw performance
  duration. It does not independently verify the scientific work ledger or
  request-to-event coverage, so the checks above remain collector obligations.
- Existing asynchronous decode readback events are reused. Host materialization
  remains after the plugin's original `event_synchronize`, with a separate host
  marker after token validation. Prefill adds CQ0 record-event commands, but
  neither recorder nor collector invokes new event/device waits or a profiler.
  Added event dispatch/notification and observer scheduling overhead still need
  disclosure. Buffered export occurs after the server is idle.
- Event streams use common Linux monotonic clock request-start boundaries and
  contiguous CQ order; probe/warmup requests are excluded by their first prefill
  identity. Distinct hashes currently prove distinct retained initial token
  buffers, not independent hash equality with each client prompt. Keep that
  exact mapping limitation explicit unless client token hashes are also saved.
- Accounting preserves all48 layers, selected constructed dtypes/fidelities,
  BFP tile exponents and bank padding, TP4/EP4 replication, per-slot
  attention/head work, shared bucket MoE, actual prefill chunks and live decode
  positions. The B2 expert-union [10,20] sensitivity interval is honestly
  retained; its lower endpoint must be labeled as the headline estimate rather
  than an observed union. It is not a bound on total physical DRAM traffic.
- The byte ledger explicitly omits some collective staging, sampler scratch,
  intermediate/layout and repeated-read traffic. These are modeled payload and
  selected-materialization estimates, not complete transactions. Collective
  staging is real DRAM traffic (`Collectives.gather(...memory_config=DRAM)`);
  do not call the byte total exhaustive or imply those omitted categories have
  been measured. Consider quantifying major CCL intermediates, whose shapes are
  already documented in the topology audit.
- Prefill peak uses a work-weighted matrix-engine fidelity reference. SFPU exact
  projections/recurrence are normalized by divisor4; the implementation properly
  states that this is not an achievable SFPU ceiling. Keep that qualification in
  the final report, alongside the model's arithmetic/traffic estimates.

The host counterexample above was run by importing only the CPU collector module
in the client environment. No implementation edits were made by this audit.
