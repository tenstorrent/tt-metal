# Gemma4 multichip decoder — blocked attempt

Stage 04 for `google/gemma-4-26B-A4B-it` starts from the completed
[optimized decoder](../optimized_decoder/README.md). The TP4 candidate in
[`multichip_decoder.py`](../../tt/multichip_decoder.py) runs on all four connected
Blackhole ASICs as a 1x4 FABRIC_1D mesh. It is **not stage-complete**. AutoFix exhausted the source-supported model-scoped
workaround for a fabric teardown failure; see [AUTOFIX_watcher.md](AUTOFIX_watcher.md).

[Mesh plan](mesh_plan.md), [context contract](../context_contract.json),
[memory plan](memory_capacity_plan.json), and [work log](work_log.md) record the
contracts and evidence. Context remains 262144; multichip maximum-context
execution is still pending. No full-model or vLLM implementation is included.

## Current evidence

The paired runner uses identical real checkpoint weights and recorded layer
inputs, comparing directly with `OptimizedDecoder`. Public short/non-aligned
length 65 and the sliding 4096/128 target pass. Full attention 4096/128 also passes
on v1. Each replicated device output is compared exactly; refreshed-position
trace replays repeat exactly. Device-only guards reject host torch work and
host tensor boundaries during forward passes.

| Candidate | Sliding 4096 prefill host µs | Sliding traced decode host µs | Minimum PCC |
| --- | ---: | ---: | ---: |
| v0 TP4 |819988|938.33|.998766 |
| v1 expert geometry |225144|898.61|.998766 |
| v2 projection geometry |224246|864.36|.999855 |
| paired single-chip baseline (v2 run) |221219|824.99|reference |

These are medians of warmed host intervals, not device telemetry. The candidate
is still slower than the baseline and is not accepted as optimized. Different
layer kinds are never averaged. The hidden-sharded residual control carries
shards through every norm/residual boundary and passes, but was slower at 65
full-attention tokens; it remains an optimization candidate, not an earned
headline-workload rejection.

`profile_v0/whole_layer.json` derives whole-layer device windows from native
firmware timestamps, taking the maximum complete per-device span across 4
chips. It includes every op and internal gap. That profile used 4096 input
and **one** traced decode, so it does not populate the required4096/128
telemetry performance fields. Native `decode_table.txt`, `prefill_table.txt`
and report CSVs identify sparse gate/down geometry, QKV, shared projections,
CCL and data movement. Gate/up prefill K-block 1 and down eight-core defaults
motivated v1; direct QKV HiFi4/K2 motivated v2. Source snapshots preserve both
older configurations. EP4 is being investigated using AutoFix because TP4's
192-wide experts leave only 12 gate/up output tiles/cores per device.

Profiler close appeared slow, but completed without intervention. The
[AutoTriage report](AUTOTRIAGE.md) refutes a persistent fabric/decoder hang;
its triage capture lacked Inspector data and is not a device-health result.

## Open gates

Final topology/geometry selection; native target profiles and rooflines;
explicit local KV/page ownership, request reuse, batching and stack tests;
maximum/non-aligned context validation; watcher; final fallback/stress audit;
independent clean-pass stage review; local checkpoint commits. Performance,
correctness and capacity claims above are limited to their named runs.

## Final attempt evidence and blocker

EP4 v2 paired runs cover real weights, 4096 input tokens and 128 traced decode
steps, batch/concurrency 1. Both layer kinds pass PCC and all-rank local KV
checks, with deterministic replay and device-only runtime guards:

| Layer kind | Minimum output PCC | Minimum cache PCC | TP1/TP4 prefill host µs | TP1/TP4 decode host µs | Prefill/decode speedup | Prefill/decode efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sliding_attention | .999850850 | .999999999 | 221224 / 100370 | 825.53 / 936.39 | 2.204 / .882 | .551 / .220 |
| full_attention | .997917284 | .999971502 | 186598 / 85972 | 876.82 / 935.38 | 2.170 / .937 | .543 / .234 |

Sources: `sliding_ep_v2.json`, `full_ep_v2.json`, `candidate_summary.json`.
These are warmed **host-wall** timings. EP improves prefill but loses decode;
no final topology winner or best-possible performance claim is made. A dual
EP-prefill/TP-decode layout is only a memory-planned future candidate.

All-check Watcher fails after correct work. A model-free reduce-scatter control
also fails during Ethernet teardown; the supported single-ERISC fallback does
not fix it. Firmware 19.9.0 exceeds the diagnostic minimum 18.10.0.
[AUTOTRIAGE_watcher.md](AUTOTRIAGE_watcher.md) and
[AUTOFIX_watcher.md](AUTOFIX_watcher.md) distinguish the observed tag assertion
from the controls' subsequent heartbeat timeouts. This is not waived as a false
positive. The source-supported repair belongs to C++ fabric teardown, outside
this goal's model/tests/docs scope. No infrastructure code was changed.

After both failures, bounded reset/list passed; the final normal four-chip
fabric mesh smoke exited 0. Device health recovery does not clear the Watcher
gate. Python syntax checks and all applicable pre-commit hooks passed.
`source_provenance.json` ties the formatted source to the measured
version. No C++ build was needed for these Python/docs-only changes.

Batch/prefix, two-layer stack and fused-CCL test harnesses are prepared but
**not hardware-validated**. Maximum-context execution, final topology tuning,
final target native profiling, rooflines and clean-pass stage review remain
open. No advertised capability was reduced. The telemetry packet at
`bringup/artifacts/multigoal-runs/20260925T171711Z/telemetry/packets/
cd88dda8-3baa-459f-9ff7-6beb4847565d.json` records actual accuracy and leaves
missing target device performance unknown. No full-model/vLLM work or push.
