# GPT-OSS-20B full-mesh multichip decoder

Status: complete. Independent stage review returned `clean-pass`; local
checkpoint SHAs are recorded in `work_log.md`.

## Result

`tt/multichip_decoder.py` is a real four-chip decoder specialized for the four
Blackhole P300c devices on this host. It subclasses the archived TP2 adapter so
that `tt/optimized_decoder.py` remains its single-chip implementation baseline,
then replaces the mesh policy with TP4 attention and EP4 routed MoE on a fixed
`1x4` `FABRIC_1D_RING`.

The production policy is:

- replicated BF16 residual and norm stream;
- 16 query heads and 2 KV heads per rank;
- local paged KV caches, local SDPA, row-parallel O, and one ring all-reduce;
- eight whole experts per rank, with exactly the gate-selected global top-4
  routes executed through three sparse matmuls and one expert-result
  all-reduce; and
- decode QKV geometry `(input_cores=10, in0_block_w=9,
  output_tiles_per_core=2, out_subblock_w=2)`; plus
- 9x10/subblock-1 gate/up and down prefill programs, with BF16 materialized
  immediately after each sparse result and BFP8 restored only for the final
  expert collective.

The fixed mesh is deliberate. This stage does not promise smaller or alternate
mesh support.

## Baseline and pre-implementation mesh decision

The optimized single-chip reference is
`models/autoports/openai_gpt_oss_20b/tt/optimized_decoder.py` at optimized-stage
commit `9949cb70f3f`. The archived `tt/tp2_multichip_decoder.py` at starting HEAD
`51757f775a6` supplied API/correctness scaffolding only.

Bounded discovery found four local Blackhole P300c devices, and a serialized
`MeshShape(1,4)` ring smoke opened and closed successfully. The compiler prior
in `doc/functional_decoder/multichip_provenance.json` independently specifies
the same mesh, TP degree 4, 16/2 Q/KV heads per rank, local packed QKV width
1,280, local O input width 1,024, and eight whole experts per rank. The target
was therefore selected before the final implementation: all four devices,
cluster axis 1, one link, ring collectives.

Two active-expert strategies were implemented and measured:

1. TP4 experts fracture every expert's 2,880 intermediate width to logical
   720/rank, physically padded to 736 (23 tiles). This balances every route but
   communicates its result.
2. EP4 experts keep eight complete experts/rank. A token can select zero to
   four experts on a given rank; sparse matmul infers the rank-local nonzero
   count and the weighted local sums are ring-reduced.

EP4 became the default because it was faster in full-layer prefill and traced
decode. Dense all-expert execution was never used. A 2D TP/EP split is not
available on a 1x4 mesh, and sequence parallelism is a poor batch-one decode
fit.

## Tensor, activation, cache, and shard contract

| Tensor / boundary | Global logical shape | Mapping | Per-device logical shape | Physical/on-chip detail |
| --- | --- | --- | --- | --- |
| layer input/output | `[1,1,S,2880]` | replicated | unchanged | decode M pads internally to 32; public S is unchanged |
| input/post-attention norm | `[2880]` | replicated | unchanged | 10-core block-sharded decode norm over `[32,2880]` |
| packed QKV weight | `[2880,5120]` | column/head TP4, reordered `Q_r,K_r,V_r` | `[2880,1280]` | Q=1,024, K=128, V=128 |
| packed QKV bias | `[5120]` | same head map | `[1280]` | no padding |
| decode QKV input/output | `[32,2880]` / `[32,1280]` physical | input replicated, output head-local | input full K; local N=1,280 | input width-sharded over 10 cores at `[32,288]`; output over 20 cores at `[32,64]`; 11x2 program |
| query heads | `64x64` | head TP4 | `16x64` | local decode head rows use height sharding |
| K/V heads | `8x64` | head TP4 | `2x64` | never gathered before cache/SDPA |
| one K or V cache/layer | `[2048,8,64,64]` global view | KV-head TP4 | `[2048,2,64,64]` | 32 MiB/device; 64-token pages |
| page table / position | `[1,2048]` / `[1]` | replicated INT32 | unchanged | arbitrary physical-page permutation |
| RoPE cos/sin | each `[1,1,131072,64]` | replicated | unchanged | BF16 tiled prefill caches plus row-major decode matrices; conservatively 64 MiB/layer/device including both retained layouts |
| attention sinks | `[64]` | query-head TP4 | `[16]` | scalar row padded to 32 for decode |
| O weight | `[4096,2880]` | row/K TP4 | `[1024,2880]` | 90 output shards `[32,32]`, 11x9 program |
| O bias | `[2880]` | rank selective | real on rank 0, zero on ranks 1-3 | applied exactly once before all-reduce |
| router | weights `[2880,32]`, global scores `[S,32]` | weight/projection replicated; EP scores partition expert axis | local sparse scores `[S,8]` | FP32 projection/top-k boundary; row-major partition before BF16 sparse metadata |
| selected EP4 experts | 32 complete experts | expert-axis EP4 | 8 complete `2880x2880` gate/up/down matrices | BFP8 weights; BF16 biases; 9x10/subblock-1 prefill grid; sparse results immediately BF16; no dimension padding |
| TP4 alternative | 32 fractured experts | intermediate TP4 | logical 720/rank | load-time padding to 736 for 23 tiles |
| expert result | `[1,1,S,2880]` | rank-local partial | unchanged partial | BFP8 ring all-reduce, then replicated residual add |

Non-aligned logical prefill lengths remain valid. The runtime pads/chunks
internally, fills only logical cache positions, and slices output back to S.
S=17 and S=33 are hardware-validated.

The stacked-decoder boundary is replicated BF16 `[1,1,S,2880]` in and out.
The direct decoder-to-decoder probe validates that this layout is accepted by
the next layer without a host conversion or hidden layout repair.

## Context and memory capacity

The advertised 131,072-token context is retained. At page size 64 there are
2,048 blocks. One local K or V cache is
`2048*2*64*64*2 = 33,554,432` bytes, so K+V consume 64 MiB/layer and 1.5
GiB/device for 24 layers.

Eight local BFP8 experts plus attention/router/norm weights are approximately
215 MiB/device/layer, or 5.04 GiB for 24 layers. The retained replicated tiled
prefill RoPE pair plus row-major decode pair are conservatively 64 MiB/layer,
or 1.5 GiB for 24 layers. Adding 1.2 GiB for embedding and head weights, 1.5
GiB KV, and a conservative 4.0 GiB trace/activation/fragmentation reserve gives
about 13.24 GiB/device versus the 31.875 GiB DRAM view. Decode key-position
vectors are small and covered by the reserve. No capability reduction is
needed.

`full_context_131072.junit.xml` allocates both `[2048,2,64,64]` caches per rank
on the selected EP4 default and decodes at logical position 131,071 through a
reversed page table. Physical page 0, offset 63 is nonzero in both K and V on
all four ranks.

## Correctness, cache, and trace evidence

All comparisons use the isolated single-chip TTNN optimized decoder, not a
PyTorch shortcut. The acceptance threshold is PCC 0.99 unless noted.

| Layer kind / path | Prefill PCC | Decode attention PCC, positions 128-130 | Final decode PCC, positions 128-130 |
| --- | ---: | --- | --- |
| sliding attention, selected EP4 | 0.99921279 | 0.99999379, 0.99999441, 0.99999431 | 0.99945879, 0.99940098, 0.99942912 |
| full attention, selected EP4 | 0.99781508 | 0.99990365, 0.99993454, 0.99994090 | 0.99923630, 0.99909284, 0.99912406 |

Routing PCCs are 0.99997479 or higher for the canonical seeds. TP4 also passes
both layer kinds and is retained as a measured alternative.

Additional gates:

- S=17/S=33 synthetic TP4 and EP4 prefill PCC is at least 0.99999941; every
  rank's arbitrary-page local K-cache comparison is PCC 1.0. Value-cache
  behavior is covered separately by bit-identical eager/trace page checks and
  nonzero final-position writes on every rank.
- Controlled S=17 instrumentation records exactly `4*S=68` global sparse
  routes for gate, up, and down in both strategies. EP4 uses `nnz=None`
  because its per-rank selected count legitimately varies from zero to four.
- Warmed trace capture/replay passes for sliding and full attention. Mutable
  positions 128-130 are PCC 1.0 against eager, reverse-page-table physical K/V
  pages are bit-identical, and five repeated replays are deterministic. Page
  table contents are capture-static in this TTNN op: a valid future-block
  allocation swap is applied on stable tensors, then release/warm/recapture at
  position 192 passes eager PCC 1.0 and writes the newly selected page.
- Seed 20260718 intentionally stresses a near-tied fourth/fifth router score.
  TP attention has PCC 0.99992784 but can swap that discontinuous route; the
  exact baseline-attention control restores routing PCC 1.0 and final PCC
  0.99938182. This is recorded stress coverage, while canonical seeds pass the
  end-to-end gate.
- The consolidated suite reports 17 passed and 6 opt-in invocations skipped.
  The two CCL, one context, one topology, and two timing skips have separate
  passing artifacts.

## Performance and topology selection

All accepted wall timings are isolated, warmed runs at S=128: 20 prefill
iterations and 500 traced decode replays. The single-chip baseline is measured
separately from the multichip process.

| Path | Prefill ms/layer | Traced decode ms/layer | Decode speedup | Decode efficiency | Prefill speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| optimized single chip | 13.4083 | 0.975834 | 1.000x | 100.0% | 1.000x |
| selected TP4-attention + EP4-experts | 22.6186 | 0.598719 | **1.630x** | **40.75%** | 0.593x |

Prefill is a measured regression, not hidden: the selected four-chip path is
1.687x slower than one chip at S=128. The remediation sweep improved the
original 26.6769 ms multichip path by 15.2%; this decoder stage remains
decode-focused.

| Full-layer candidate (10 prefill / 100 trace replays) | Prefill ms | Decode ms | Disposition |
| --- | ---: | ---: | --- |
| TP4 experts, QKV `(30,3,2,2)` | 39.8186 | 0.656294 | rejected: slower |
| EP4, QKV `(30,3,2,2)` | 26.7193 | 0.604859 | EP selected |
| EP4, QKV `(30,3,1,1)` | 26.7182 | 0.609944 | rejected |
| EP4, QKV `(30,3,4,4)` | 26.7303 | 0.609892 | rejected |
| EP4, QKV `(18,5,2,2)` | 26.7117 | 0.599635 | close, slower decode |
| EP4, QKV `(10,9,2,2)` | 26.7505 | **0.598815** | selected geometry; reconfirmed at 20/500 |
| EP4, QKV `(45,2,2,2)` | 26.7347 | 0.618945 | rejected |

The independent-review remediation held QKV10 fixed and swept the dominant
EP4 prefill rows. The production 9x10/subblock-1 gate/up and down grids plus
post-sparse BF16 measured 22.6186 ms at 20 iterations. Isolated 10-iteration
results were: 5x6 gate 24.9871 ms; down 5x9 26.7416 ms; down 9x10 26.4462 ms;
5x6+9x10 24.6387 ms; chunks 64/32 27.0177/27.2026 ms; post-sparse BF16 alone
24.8080 ms; 5x6+9x10+BF16 22.8116 ms; 5x9+9x10+BF16 22.9150 ms; and selected
9x10+9x10+BF16 22.6662 ms. Real sliding/full correctness passes after
promotion.

The residual topology probe measures the complete O projection -> residual ->
distributed norm -> router -> expert-input boundary. Replicated O all-reduce
takes 0.608773 ms versus 1.271655 ms for reduce-scatter, width-sharded
residual, distributed norm, row-sharded router, and full-hidden gather.
Replicated is 2.089x faster and remains the stack contract. Candidate norm and
router PCCs are 0.99999258 and 0.99999750.

The decode collective audit also implemented a trace-safe persistent-semaphore
candidate: explicit 2,880->2,944 padding, minimal-async reduce-scatter, async
all-gather, then slice. Both layer kinds pass correctness and mutable trace
replay, but its 500-replay decode is 0.638599 ms versus 0.598641 ms for the
current ring all-reduce (6.7% slower). Its profile contains six
`ReduceScatterMinimalAsync` rows (152.28 us) and six `AllGatherAsync` rows
(88.28 us), plus padding/slicing. It is rejected. Fused matmul + minimal
reduce-scatter is unavailable on Blackhole because the GPT-OSS source disables
that family for race #46181. Gathered-input/local-output O would require a
second all-gather to restore the replicated stack boundary; the broader
sharded-stream family above is already 2.089x slower.

See `perf/perf_report.md` for the `tt-perf-report` tables and communication,
DRAM, compute, CCL, and data-movement audit.

## Reliability and runtime audit

- Static runtime-method audit forbids `torch`, `from_torch`, `to_torch`,
  `get_device_tensors`, and `cpu`; it passes. Host conversion is load/test
  setup only.
- A worker/Tensix-watcher canonical EP4 run passes both meaningful layer kinds.
  No watcher/NoC assertion, deadlock, illegal access, hang, or timeout appears.
  A separately retained full-Ethernet-watcher attempt deterministically fails
  before model execution: instrumentation makes the ACTIVE_ETH program 27,920
  bytes, exceeding the physical 25,600-byte kernel-config buffer for both
  parametrizations. The exact console/generated-watcher logs are retained;
  all four devices were reset and the 1x4 ring smoke passed before continuing.
- Watcher and profiler were run in separate processes as required.
- Final `tt-smi -ls --local` shows all four P300c devices present and resettable.
- Known non-fatal host warnings: only ~17.5 MiB `/dev/shm` remained for a 16
  MiB MPI segment, motherboard `B850M-C` is not in discovery metadata, and the
  inspector cannot replace a permission-owned generated YAML. None caused a
  failed final gate.

## Reproduction commands

```bash
# Consolidated correctness/stress/trace/default audit
pytest -q models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py \
  --junitxml=models/autoports/openai_gpt_oss_20b/doc/multichip_decoder/logs/final_consolidated_suite.junit.xml

# Full advertised-context capacity and final-position cache update
RUN_MULTICHIP_CONTEXT=1 pytest -q \
  models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_full_context_cache_allocation_and_last_page_update

# Selected warmed timing (capture the single-chip reference in an isolated 1x1
# run first, then run the 1x4 test)
RUN_MULTICHIP_DECODER_PERF=1 MULTICHIP_DECODER_PREFILL_REPEATS=20 \
MULTICHIP_DECODER_TRACE_REPLAYS=500 pytest -q \
  models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_multichip_decoder_perf

# Profiler: no watcher in this process
RUN_MULTICHIP_DECODER_PERF=1 MULTICHIP_DECODER_PREFILL_REPEATS=1 \
MULTICHIP_DECODER_TRACE_REPLAYS=3 \
MULTICHIP_PERF_RESULT_PATH=models/autoports/openai_gpt_oss_20b/doc/multichip_decoder/logs/profile_perf_final_autofix.json \
python -m tracy -r -p \
  -o models/autoports/openai_gpt_oss_20b/doc/multichip_decoder/perf/tracy_autofix \
  -n gpt_oss_20b_ep4_qkv10_autofix -m pytest -q \
  models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_multichip_decoder_perf

# Watcher: no profiler in this process
TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 pytest -q \
  models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_real_weight_prefill_decode_matches_single_chip_optimized \
  -k ep
```

## Exact artifacts

- Correctness: `logs/final_ep9x10_bf16_correctness.junit.xml`
- Consolidated suite: `logs/final_consolidated_suite.junit.xml`
- Active-expert audit: `logs/active_expert_tp4_ep4.junit.xml`
- Trace replay: `logs/final_trace_replay_page_table_recapture.junit.xml`
- Context: `logs/full_context_131072.junit.xml`
- Near-tie stress: `logs/near_tie_baseline.junit.xml`,
  `logs/near_tie_tp4_stress.junit.xml`
- Topology: `logs/residual_topology_candidate.json`,
  `logs/residual_topology_probe.junit.xml`
- Final performance: `logs/single_chip_perf_reference_seq128.json`,
  `logs/final_perf_ep4_qkv10_seq128.json`
- Profiler provenance: `logs/profile_perf_final_autofix.json`,
  `perf/cpp_device_perf_report_ep4_qkv10_autofix.csv`,
  `perf/decode_report_autofix.csv`, `perf/prefill_report_autofix.csv`, and the
  matching summary CSV/PNG and human-readable `tt_perf_report_*` logs
- Collective candidate: `logs/autofix_ccl_rs_ag_correctness.junit.xml`,
  `logs/autofix_ccl_rs_ag_trace.junit.xml`,
  `logs/autofix_perf_ccl_rs_ag_pad64_seq128.json`,
  `perf/cpp_device_perf_report_ccl_rs_ag_pad64.csv`, and its decode
  report/summary
- Watcher: `logs/watcher_final_ep9x10_bf16.log` and generated-watcher log;
  full-Ethernet incompatibility in `logs/watcher_full_eth_ep4_qkv10.log` and
  `logs/watcher_full_eth_generated_watcher.log`
- Compiler prior: `doc/functional_decoder/multichip_provenance.json`

The generated unmerged op dumps remain locally at
`perf/ops_perf_results_ep4_qkv10_autofix.csv` and
`perf/ops_perf_results_ccl_rs_ag_pad64.csv`. At 4.48 and 4.63 MiB they exceed
the repository's 500 KiB evidence-file limit, so they are intentionally not
checkpointed. The checkpointed `decode_report*`/`prefill_report*` op tables,
device CSVs, summaries, provenance JSON, and exact commands reproduce their
accepted conclusions.

Historical failing XMLs are retained and explicitly marked as superseded in
`work_log.md`; this includes
`logs/final_trace_replay_mutable_page_table.junit.xml`, whose two failures
establish that replay without recapture is unsupported, and its passing
`final_trace_replay_page_table_recapture.junit.xml` successor. The final
artifacts above are the acceptance records.

## Limitations

- Only this host's fixed 1x4 P300c ring and batch one are supported.
- Capacity is proven by physical cache allocation and final-position update;
  a single 131,072-token prefill compute invocation was not run.
- S=128 prefill remains slower than the optimized single-chip baseline after a
  15.2% multichip-path improvement and is dominated by sparse expert compute
  and associated data movement.
- Per-rank EP active count is data dependent, so forcing `nnz=4` is incorrect;
  sparse matmul must infer it at runtime.
- Page-table tensor contents are fixed for a captured trace on this target.
  Mutable positions replay without recapture; a valid future page allocation
  requires releasing and recapturing the warmed decode trace.
- The near-tied-router stress demonstrates that very small TP attention
  rounding can change a discontinuous fourth-place route. Canonical layer
  seeds remain above the full-decoder PCC gate.
