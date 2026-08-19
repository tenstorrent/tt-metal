# Fused AGMM Candidate Triage

Candidate: `all_gather_minimal_matmul_async` plus output `all_gather_async` for
Qwen/Qwen3.6-35B-A3B TP-axis row-parallel decode projection.

This was an optimization candidate only. The shipped multichip decoder default
remains public `ttnn.all_reduce`, BF16 payload, two-link Ring topology.

## Trigger

The first fused AGMM probe exposed a program-factory constraint rather than a
valid rejection:

- target mesh: `2x2` Blackhole p300c
- target axis: TP `cluster_axis=1`
- input shape: `[1,1,32,2048]`
- output-sharded adapted weight shape: `[1,1,2048,2048]`
- requested `num_links=2`
- initial failure: sender-axis grouping required exactly two groups of workers

The probe was adapted to the legal grouping with
`num_workers_per_link=4`, then retried.

## Evidence

Persistent fused retry:

- log: `logs/candidate_fused_agmm_persistent_rsag_probe.log`
- triage: `triage/fused_agmm_persistent/tt-triage.txt`
- result: after adapting past the initial worker-grouping error, the candidate
  hung in fabric/router state and required reset/recovery.

Non-persistent fused retry:

- command log: `logs/candidate_fused_agmm_bf16_nonpersistent_probe.log`
- triage: `triage/fused_agmm_bf16_nonpersistent/tt-triage.txt`
- recovery: `logs/tt_smi_after_fused_agmm_bf16_nonpersistent_reset.log` and
  `logs/mesh_smoke_after_fused_agmm_bf16_nonpersistent_reset.log`
- result: the probe reached `cluster_axis=1`, `force_transpose=True`,
  `num_links=2`, `num_workers_per_link=4`, BF16 output-sharded weights, and
  no persistent buffers before it stopped making forward progress.

Useful triage anchors:

- `check_noc_status.py` reports mismatched ETH NOC counters beginning at line
  `331` in the non-persistent triage output.
- `fabric_erisc_router` callstacks begin at line `350` in the non-persistent
  triage output.
- `dump_fast_dispatch.py` reports dispatch-symbol read failures beginning at
  line `290`.

## Decision

The fused AGMM family is rejected for this stage. The rejection is not based on
the first TTNN/API error: shape/layout/worker grouping was adapted and retried,
both persistent and non-persistent variants reached runtime, and the adapted
runtime path hung in fabric/router state. The final path stays on public
two-link all-reduce, which has clean correctness, fallback, watcher, and Tracy
evidence.
