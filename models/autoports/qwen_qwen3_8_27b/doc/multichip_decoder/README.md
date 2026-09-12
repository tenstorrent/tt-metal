# Qwen3.8-27B multichip decoder

Stage state: **multichip-decoder complete**.

Correctness, full-context capacity, trace, stress, worker/NoC watcher, and
profiler checks pass for the final default. Independent
[stage review](stage_review.md) returned **clean-pass**. Local checkpoint SHAs
are recorded in [work_log.md](work_log.md).

The decoder uses four Blackhole p300c devices as a 1x4 tensor-parallel mesh,
with a Ring fabric and two CCL links. It subclasses the completed
`OptimizedDecoder`; single-chip attention, delta recurrence, logical-tail
handling, and paged-cache operations execute with local head dimensions.
The frozen baseline is commit `1477a0eecf4f6654cebb4010aa89b852d2ca75fb`.
This stage adds no full-model or serving implementation.

## Selected layer contract

Public input and output are BF16 TILE `[B,S,5120]`, replicated over the four
chips, with `1 <= B <= 32`. Decode uses `S=1`. B1 carries compact L1 tensors
between layers. For B>1, public TILE tensors use DRAM because their physical
height expands from32 packed rows to B*32 rows; internal norms, matmuls, and
direct all-reduce still use compact L1 layouts. All layer kinds accept this same contract, without a mesh
conversion at the boundary. Setup packs each rank's complete head groups;
attention and recurrent state never cross rank ownership.

| Projection | Local weight `[K,N]` | DRAM bank shard | Decode input cores / block K tiles |
| --- | --- | --- | --- |
| Linear packed Q/K/V/Z/B/A | 5120,4160 | 5120,544 | 10 / 16 |
| Full packed Q/K/V/gate | 5120,3584 | 5120,448 | 10 / 16 |
| Attention output | 1536,5120 | 1536,640 | 8 / 6 |
| Packed MLP gate/up | 5120,8704 | 5120,1088 | 10 / 16 |
| MLP down | 4352,5120 | 4352,640 | 8 / 17 |

There are eight DRAM bank shards and one reader per bank. Decode keeps BFP4
weights, BF16 activations, LoFi projections, and FP32 accumulation; recurrence
uses FP32. Prefill preserves optimized minimal-matmul selection and internal
2048-token chunks. Twelve local linear B/A gates are padded to32 at load time;
logical head widths are sliced before use. No public sequence alignment is
required. Separate gate/up weight copies are absent in the selected packed path.

Each full-attention rank owns six Q heads and one KV head of width256. K/V are
BFP8 TILE `[physical_pages,1,32,256]`; page tables and absolute current positions
are replicated. Each linear-attention rank owns four key heads and twelve
value heads of width128, FP32 recurrence `[B,12,128,128]`, and row-major BF16
convolution history `[B,3,2560]`. The model is dense; there is no expert/router
path. Caller-provided RoPE follows the optimized decoder's partial-RoPE contract.

Decode row-parallel output/down projections use direct all-reduce. Prefill
uses reduce-scatter followed by all-gather (the one-token case uses decode
projection geometry).
The measured alternative retains hidden1280 residuals through distributed norm
and residual add, then gathers normalized projection inputs; it is slower on
this hardware. Shared `TT_CCL` manages cycling semaphores and one persistent
all-reduce workspace for the entire layer stack: BF16 `[1,1,32,20480]`,
width-sharded over80 cores with `[32,256]` shards (16KiB/core). Reuse is limited
to one ordered CQ0 execution stream; concurrent models/traces must not share
this context. Outputs are independent native allocations. See
[AUTOFIX_shared_ccl.md](AUTOFIX_shared_ccl.md) for the source audit and model
precedent. State allocation also prepares fallback RS/AG buffers before capture.
Caller inputs, positions, page tables, RoPE, and state must keep their device
addresses during trace replay. Restore state after warmup/capture as appropriate.

## Strategy and measurements

[mesh_plan.md](mesh_plan.md) records the plan made before implementation,
including tensor ownership, padding, collective volume, and rejected TP/DP/2D
choices. [candidate_summary.json](candidate_summary.json) indexes measurements.
The runner uses real checkpoint weights and recorded HF layer inputs; the
single-chip TTNN baseline is the direct numerical reference.

The selected family combines packed MLP, larger precision-matched DRAM blocks,
carried L1 residuals, direct all-reduce with a shared persistent workspace,
and two links. Measured
alternatives include Linear/Ring, direct all-reduce, hidden-sharded residuals,
AGMM and MMRS with their downstream distributed norms, BF16/BFP8 CCL payloads,
projection geometries, gate epilogue, and higher math fidelity. A coherent
packed direct-all-reduce BFP8 candidate passed PCC but lost both paired50-replay
measurements: BF16 medians0.4685/0.4664ms versus BFP8 0.4789/0.4747ms
([datatype comparison](direct_ar_dtype_comparison.json)). Fused family
measurements include corrected native dtype contracts; see
[AUTOFIX_fused_ccl_dtype.md](AUTOFIX_fused_ccl_dtype.md).

Native multi-reader bank assignment requires a unit mesh in this checkout.
[AUTODEBUG_dram_mesh.md](AUTODEBUG_dram_mesh.md) and
[AUTOFIX_dram_mesh.md](AUTOFIX_dram_mesh.md) document the source proof and
one-reader hardware controls. This restricts native DRAM projection execution;
all-core minimal-matmul and fused alternatives were measured, not assumed faster.
Optional policy overrides retain experiment paths; the stage-supported contract
and capacity accounting refer to the default policy. Historical experiments
record their complete effective policy and source SHA; source snapshots are in
`sources/`. Reproducing an old override against newer defaults is not equivalent.

## Final warmed latency

B1, logical prefill128, same recorded input/checkpoint and harness. Prefill is
median of20 warmed executions; decode is median of50 warmed trace replays.
Host tensor reads, state restoration, and reference comparisons are outside the
measured decode window. Efficiency is speedup divided by four devices.

| Layer kind | Phase | Single-chip ms | TP4 ms | Speedup | Efficiency |
| --- | --- | --- | --- | --- | --- |
| Linear attention | Prefill | 2.0416 | 1.2971 | 1.57x | 39.3% |
| Linear attention | Decode | 0.8229 | 0.4654 | 1.77x | 44.2% |
| Full attention | Prefill | 1.7158 | 1.2290 | 1.40x | 34.9% |
| Full attention | Decode | 0.6619 | 0.3506 | 1.89x | 47.2% |

Exact source, policy, PCC, and timings are in `final_benchmark_l{0,3}.json`
and matching `_baseline.json` artifacts, indexed by
[performance_summary.json](performance_summary.json). This is decoder timing,
not full-model generation throughput. Prefill host timing has dispatch variance;
these are final default measurements, not the minimum earlier candidate times.

## Validation commands and artifacts

Run from the repository root, serially. The experiment wrapper resolves the
installed model-bringup environment, records command/source provenance, and
closes the mesh after each process. Do not overlap watcher, profiler, or other
hardware jobs.

```bash
python_env/bin/python -m pytest models/autoports/qwen_qwen3_8_27b/tests/test_multichip_decoder.py --confcutdir=models/autoports/qwen_qwen3_8_27b/tests -x -q
python_env/bin/python models/autoports/qwen_qwen3_8_27b/tests/run_multichip_capacity.py
# Fixed stack: linear layer0 followed by full-attention layer3:
bash models/autoports/qwen_qwen3_8_27b/tests/run_multichip_experiment.sh stress_stack_batch32_baseline --layer 0 --batch 32 --length 257 --stack --stress-iterations 100 --baseline
bash models/autoports/qwen_qwen3_8_27b/tests/run_multichip_experiment.sh stress_stack_batch32 --layer 0 --batch 32 --length 257 --stack --stress-iterations 100
python_env/bin/python models/autoports/qwen_qwen3_8_27b/tests/watch_multichip_decoder.py
# Profile both baseline and TP4 for each kind (layers0 and3):
bash models/autoports/qwen_qwen3_8_27b/tests/profile_multichip_decoder.sh final_profile_l0 --layer 0 --repeats 10
python_env/bin/python models/autoports/qwen_qwen3_8_27b/tests/multichip_profile_tables.py models/autoports/qwen_qwen3_8_27b/doc/multichip_decoder/tracy/final_profile_l0
python_env/bin/python models/autoports/qwen_qwen3_8_27b/tests/summarize_multichip_validation.py
python_env/bin/python models/autoports/qwen_qwen3_8_27b/tests/multichip_memory_plan.py
python_env/bin/python models/autoports/qwen_qwen3_8_27b/tests/summarize_multichip_performance.py
```

The31-case regression suite covers both layer kinds, lengths1/31/32/33/2047/2048/2049/
4097, continuation at31, batches1/3/8/32, and a two-layer linear/full stack at batches1/2/3/8/16.
It compares complete outputs and states, changes inputs/RoPE/page tables/current
positions before replay, checks untouched cache rows and unowned pages, and
requires bitwise eager/replay output and state equality. A B32 two-layer
stress case queues100 replays without host barriers between calls and compares
evolving state and output against100 eager invocations. Cross-baseline cache
PCC covers valid logical rows; native final-page padding has no equality
contract, while raw-state replay and unchanged-row checks remain exact. Every forward runs
under a Torch-operation and host-conversion prohibition.

The context remains262144. [memory_capacity_plan.json](memory_capacity_plan.json)
accounts for64 layers, both projection layouts, all16 local KV caches, all48
recurrent states, untied BF16 embedding/head reserves, constants, and18GiB for
trace/activations. The current planned total is29,939,351,552 bytes/device.
Capacity runs hold12,759,482,368 additional bytes/device for all planned
persistent weights/state plus2GiB trace/CCL while real full-length activations
execute. The activation estimate is not double-counted as empty storage. Large-context inputs repeat
recorded activations; this is TTNN parity/capacity evidence, not a fresh long
context HF accuracy oracle. Batch32 and full context are separate coverage
points; the memory budget is for one full-context user.

The overall minimum output/state PCC is0.9998265382 across31 regressions,
four watcher cases, five capacity cases, and the queued B32 stack stress.
[validation_summary.json](validation_summary.json) indexes the exact reports.
Watcher ETH instrumentation is excluded because the native instrumented fabric
program exceeds its configuration buffer (29072 versus26624 bytes); worker
and NoC checks run on all four devices. The failed instrumented-fabric control
and successful scoped watcher runs are retained.

[profiler_interpretation.md](profiler_interpretation.md) connects final raw
rows and human-readable tables to projection geometry, DRAM/compute utilization,
collective choice, and data movement. Large raw files are archived with SHA256
provenance in [raw_archive_manifest.json](raw_archive_manifest.json); compact
CSV/gzip files and readable tables remain in this directory.

See [work_log.md](work_log.md) for commands, failures, recoveries, and exact
artifact paths. The independent review and local checkpoint SHAs are recorded there.

## Caller setup and continuation

Configure `ttnn.FabricConfig.FABRIC_1D_RING` before opening `MeshShape(1,4)`.
Create one `TT_CCL(mesh_device)` and pass `ccl=shared_ccl` to every layer
constructed with `MultichipDecoder.from_state_dict(..., hf_config=...,
layer_idx=..., mesh_device=..., ccl=shared_ccl)`. A mismatched mesh is rejected.
The global HF config is copied before local head dimensions are derived. Allocate state and all persistent trace inputs
before capture. `allocate_state(batch_size=B, num_pages=P)` requires pages for
full attention; the caller owns page-table construction and request isolation.

`prefill_forward` accepts logical `[B,S,5120]` and `start_pos`; a continuation
starting inside a page supplies device INT32 `positions[S,B]`. Requests in a
prefill batch share length and prefix position. Decode uses device INT32
`current_pos[B]`, allowing distinct initialized histories/positions per user.
Only valid logical cache rows are meaningful; causal attention masks future
rows written as physical padding by native prefill. Restore or replace request
state explicitly before independent runs. State/collective allocation belongs
to setup; recapture traces after changing tensor allocations or batch geometry.
