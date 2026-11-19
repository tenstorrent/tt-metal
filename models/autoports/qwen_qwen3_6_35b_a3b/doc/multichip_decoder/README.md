# Qwen3.6 multichip decoder

Target: four Blackhole p300c chips, mesh 1x4, FABRIC_1D and Linear collectives.
Single-chip numerical/timing baseline: `tt/optimized_decoder.py` at 2901d8a5
(implementation 17810c63). All 27 acceptance gates pass, including final profiler,
correctness, stress, watcher and trace-allocation rejection evidence. Independent [round-2 review](stage_review_round2.md) returned **clean-pass**.
The local checkpoint is recorded in [the work log](work_log.md).

## Selected implementation

`MultichipDecoder` subclasses the optimized decoder. Full/linear attention and
the shared expert use TP4. Each full-attention rank owns four query heads and
one KV head; rank pairs (0,1) and (2,3) duplicate the two global KV heads. Linear
attention owns four key heads and eight value/recurrent heads per rank.

Routed experts use indexed TP4 for batch 1 and token-grouped EP4 for prefill
and batched decode: 64 complete experts per rank, only gate-selected token/expert
pairs executed. Local dispatch drops remote experts before computing compact
offsets. Indexed GU uses DRAM input, eight cores and K block 64; down uses
64 cores and K block 4. Precision remains BFP8/LoFi for routed experts.

The residual is replicated `[1,S,2048]` per rank. Singleton decode uses direct
all-reduce in L1 width shards and local RMSNorm. FP32 linear-attention partials
remain FP32 through reduction before BF16 output. Multirow/prefill uses RS+AG.
Bounded concat keeps logical lengths independent of native input-count limits.
Padding stays internal; it never extends attention/cache positions.

[Mesh plan](mesh_plan.md) records per-device tensors, shard specs, padding,
collectives and memory arithmetic. Hidden-sharded residuals, distributed norms,
fused output MM-RS and adapted Linear AG-MM were measured through their next
consumers. [Geometry audit](local_geometry_audit.md) records precision-locked
core/block/subblock and input-placement controls. The selected local dispatch
improves B32 decode; linear prefill was 0.79% slower in its matched screening.

## Warmed latency

Fresh serialized controls, same weights/inputs/positions, no watcher, profiler
or allocation tracking. Decode is the median blocking warmed trace replay for
one complete batch. Prefill is the median of three warmed passes. Decode uses
200 samples at B1 and 60 at B32. Speedup is single-chip/mesh; efficiency is
speedup/4. Values describe standalone decoder layers, not model throughput.

| Kind | Batch | Phase | Single chip, ms | Mesh, ms | Speedup | Efficiency |
|---|---:|---|---:|---:|---:|---:|
| linear | 1 | prefill | 103.163 | 63.693 | 1.62× | 40.5% |
| linear | 1 | decode | 1.349 | 1.016 | 1.33× | 33.2% |
| linear | 32 | prefill | 3211.274 | 1340.356 | 2.40× | 59.9% |
| linear | 32 | decode | 42.491 | 21.727 | 1.96× | 48.9% |
| full | 1 | prefill | 3.803 | 2.031 | 1.87× | 46.8% |
| full | 1 | decode | 0.433 | 0.394 | 1.10× | 27.5% |
| full | 32 | prefill | 87.386 | 44.025 | 1.98× | 49.6% |
| full | 32 | decode | 13.496 | 6.453 | 2.09× | 52.3% |

Exact values: [latency CSV](latency_comparison.csv), `final_l{0,3}_b{1,32}.json`.

[B1 profiler findings](profile_findings_b1.md) and
[B32 profiler findings](profile_findings.md) inspect actual precision, geometry,
CCL, compute and movement. MoE contributes 23.08%/55.55% of linear/full B1
kernel work and 4.67%/14.45% at B32. These are kernel-work fractions, not clean
elapsed-time fractions. B32 output splitting costs about 200 us per layer;
attention remains per request before the grouped expert path.

## Stack, paging and trace contract

Share one `CollectiveWorkspace(mesh)` across serial layer instances; retain it,
weights and states until all using traces are released. It owns EP buffers,
BF16/FP32 direct-AR scratch and semaphores. Concurrent forwards sharing this
workspace are unsupported. Prepare capacities, prefill plans, batch signatures
and position tensors before capture, then warm them. Input and output layouts
agree between linear and full layers; no layer-boundary reshards are needed.

`mixed_stack_selected.json` validates a linear→full stack, nonaligned resumed
prefill, heterogeneous current positions, changing/permuted page tables, and
one shared physical KV pool against independent caches. Untouched future cells
and guard pages remain unchanged. All four output replicas are bit-equal.
Natural B32 tests also poison expert workspace and force every selected expert
onto one rank: the three empty ranks return zeros without stale output.

Both layer kinds pass whole-output PCC ≥0.995 against the optimized TTNN
baseline. Natural B32 minimum decode PCC is 0.999902546 (linear) and
0.999854728 (full). Per-row PCC is diagnostic. Natural activations come from
32 distinct checkpoint-tokenized prompts; these are decoder activation checks,
not generation-quality evaluations. Eager versus trace outputs match bitwise
at every tested step; changing-input replay and live-trace prefill also match.
Runtime forward/replay guards prohibit Torch and host tensor conversions.

## Context and capacity

Advertised context remains 262144. The pool supports 262144 aggregate cached
tokens and up to 32 recurrent request slots; one request can use the whole
context. This is not a claim that 32 simultaneous full-size contexts fit.
[Capacity plan](memory_capacity_plan.json) includes both EP/indexed expert
representations, all 40 layers, KV pool and reserved activation/trace space.
It budgets approximately 27.92 GiB per chip against 31.77 GiB allocator capacity,
leaving 3.85 GiB. Full-stack weight memory is extrapolated from constructor
censuses; no full model is loaded in this decoder-only stage.

Both 32769-token context smokes pass with capacity 65536 and two subsequent
decode positions. The changed local KV layout also passes the single maximum
check: 262142-token prefill plus two decodes reaches 262144, with prefill PCC
0.999976139 and minimum decode PCC 0.999976423. Cache ownership, local heads
and untouched cells pass. These long-context checks are eager correctness
evidence, with no trace or latency claim. See `context_l{0,3}_32k.json`,
`context_fullmax.json` and [context contract](../context_contract.json).

## Evidence and reproduction

Start at repository root with `source python_env/bin/activate` and
`source ../lane-env.sh`. Hardware commands run serially under bounded timeouts
after ownership checks. Script paths below are relative to this autoport.

- `tests/run_multichip_acceptance.sh all`: mixed stack, natural B32 and context.
- `tests/run_multichip_validation.sh clean|boundary|stress|watcher`: paired
  timings, lengths 1/31/65/257, 128-step replay and separate watcher runs.
- `tests/collect_multichip_perf.sh <layer> <fresh-directory>`: B1 Tracy capture.
- `tests/collect_multichip_profile.sh <layer> <fresh-directory>`: narrow B32
  capture from paired post-prefill state seeds, plus one representative request
  prefill. Both collectors produce tt-perf-report human tables and CSVs.
- `tests/summarize_multichip_perf.py`: raw four-rank accounting.
- `tests/summarize_multichip_moe_share.py`: source-boundary MoE kernel-work share.
- `tests/summarize_multichip_validation.py`: fail-closed gate manifest.

[Artifact manifest](artifact_manifest.md) identifies exact evidence paths,
provenance and historical controls. [Work log](work_log.md) records commands,
recoveries and review/commit receipts. [AutoFix record](AUTOFIX.md) records
repaired failures. Raw tensors/captures remain on local disk.
