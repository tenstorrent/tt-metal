# GPT-OSS-20B multichip decoder

This stage implements a real tensor-parallel decoder layer for the complete
mesh available on the bring-up host: two Blackhole P300 chips in a fixed `1x2`
ring.  It subclasses `OptimizedDecoder` from optimized-decoder commit
`9949cb70f3f` and preserves that baseline's BF16 attention, BFP8 routed sparse
experts, and batch-one contract.  Full-model assembly and vLLM are deliberately
out of scope.

## Chosen mesh and parallel plan

`tt-smi -ls --local` on 2026-07-17 found exactly two Blackhole `p300c`
devices.  The selected plan is therefore:

- physical/logical mesh: `1x2`, TP axis 1, TP=2, EP=SP=1;
- fabric/collective topology: `FABRIC_1D_RING`, `ttnn.Topology.Ring`, one link;
- residual boundary: replicated BF16 `[1,1,S,2880]` on both ranks;
- attention: Q/K/V head parallel, row-parallel O, one O all-reduce;
- KV cache: four of eight KV heads per rank in local 64-token pages;
- MoE: both ranks execute only the gate-selected top-4 experts, with every
  selected expert fractured over its 2,880-wide intermediate dimension;
- gate/up: column parallel; down: row parallel plus one BFP8 all-reduce;
- router, norms, rotary tables, and logical page table: replicated;
- non-aligned logical sequence lengths: padded only inside TTNN operations and
  sliced back to the requested logical length.

A 2D mesh is physically unavailable.  Whole-expert EP=2 was rejected because
an arbitrary top-4 route can select zero through four experts on either rank;
it neither balances a one-token decode nor halves each active expert's matrix
bandwidth.  Dense all-expert execution is not used.

## Per-device tensor, activation, and cache shapes

| Tensor / boundary | Global logical shape | TP mapping | Per-device logical shape | Padding / device shard |
| --- | --- | --- | --- | --- |
| residual / layer I/O | `[1,1,S,2880]` | replicated | unchanged | decode M physically pads to 32; logical S is preserved |
| input and post-attention norm weight | `[2880]` | replicated | `[2880]` | row-major weights; both decode norms use the advisor's 10-core L1 layout and return the required boundary layout |
| packed QKV weight | `[2880,5120]` | column, reordered `[Q_r,K_r,V_r]` | `[2880,2560]` | Q=2048, K=256, V=256; 80 output tiles |
| packed QKV bias | `[5120]` | same reordered column map | `[2560]` | no extra padding |
| query heads | `64 x 64` | head ownership | `32 x 64` | rank `r` owns heads `[32r,32r+31]` |
| key/value heads | `8 x 64` | head ownership | `4 x 64` | rank `r` owns heads `[4r,4r+3]` |
| one paged K or V cache | `[2048,8,64,64]` global | KV-head local | `[2048,4,64,64]` | 64-token pages; 64 MiB/cache/device/layer |
| cosine / sine rotary cache | each `[1,1,131072,64]` | replicated | unchanged | BF16 DRAM; logical position selects a row |
| page table / current position | `[1,2048]` / `[1]` | replicated | unchanged | INT32; arbitrary logical-to-physical page permutation |
| attention sinks | `[64]` | query-head local | `[32]` | decode form pads each scalar to a 32-wide row |
| O weight | `[4096,2880]` | row/K | `[2048,2880]` | no dimension padding |
| O bias | `[2880]` | rank 0 real, rank 1 zero | `[2880]` | applied exactly once before reduction |
| router weight/bias/scores | `[2880,32]`, `[32]`, `[S,32]` | replicated | unchanged | FP32 router path, active top-4 |
| expert gate/up weight | each `[32,2880,2880]` | column/intermediate | each `[32,2880,1440]` | BFP8, 45 output tiles, active experts only |
| expert gate/up bias | each `[32,2880]` global | column | each `[32,1440]` | no padding |
| expert down weight | `[32,2880,2880]` | row/intermediate | `[32,1440,2880]` | BFP8, 45 K tiles, 90 N tiles |
| expert down bias | `[32,2880]` | rank 0 real, rank 1 zero | `[32,2880]` | applied once before reduction |
| layer output | `[1,1,S,2880]` | replicated | unchanged | directly consumable by the next decoder layer |

Prefill keeps the shape-efficient 45-core input / 80-core output QKV path over
an `11x8` physical extent.  Decode uses the measured 30-core input map with
`[32,96]` shards, 40-core output map with `[32,64]` shards, `in0_block_w=3`,
`per_core_N=2`, `out_subblock_w=2`, and an `11x4` program extent.  O uses 90
cores and an `11x9` extent.  The compiler prior retained under
`shard_advise/` independently selected the prefill QKV/O extents and both
10-core decode norm layouts; the narrower decode QKV map was chosen by a
hardware A/B sweep described below.

## Exact active-expert prefill contract

Prefill does not fall back to the optimized decoder's generic dense-group
expert path.  It pads the logical sequence to a multiple of 32 internally,
chunks at 128 tokens, and gives each token its own sparse batch
`[S,1,1,2880]`.  Router sparsity is token-specific: gate and up consume
`[S,1,1,32]`, while down consumes `[1,1,S,32]`.  Intermediate elementwise work
is compacted to `[S,32,1440]` per device and down produces `[S,32,2880]` before
route weighting, expert summation, one TP all-reduce, and slicing to logical
`S`.  A controlled S=17 test records exactly 68 nonzero entries in each of the
three sparse matmuls, rather than the 1,024 entries of a padded dense
32-expert group.

This is exact gate-selected top-4 execution, but the current sparse-matmul
contract selects a batch, not a row within its M tile.  Every token/expert
route therefore occupies one physical 32-row tile.  At S=128 the profiler
reports `active=512/4096` for each projection: only selected routes execute,
but each route retains the hardware tile cost.  This is the measured reason
exact multichip prefill is slower than the optimized single-chip baseline.
The selected expert program is mode-specific: exact prefill uses 15 gate/up
cores and 30 down cores with valid `1x3` output subblocks, while decode keeps
45 gate/up and 90 down cores with `1x1` subblocks.  This hybrid was measured
against both uniform alternatives and preserves the fastest result per mode.

AutoFix tested the existing fused `moe_compute` kernels before accepting that
tradeoff.  `compute_only` exposes only a rolling two-expert double buffer and
overwrites earlier active-expert outputs; full mode cannot use the singleton
dispatch mesh (Linear has no neighbour and Ring rejects a self-neighbour); and
`moe_gpt` hardcodes the unfractured 2,880 intermediate over 12 cores, so it
cannot consume the TP-local 1,440 intermediate.  A correct local-combine
special case would require C++ changes outside this decoder-only stage.  Its
packed weights would also add about 5.54 GiB/device for 24 layers.  Dense
all-expert execution remains rejected: it violates the active-expert contract
and no evidence showed it faster for decode.

## Residual and collective topology decision

The selected layer has two local-result boundaries: attention O and expert
down.  A BF16 decode hidden vector is 5,760 bytes; a local BFP8 expert result is
about 3,060 bytes including block exponents.  The selected all-reduces appear
as reduce-scatter plus all-gather in the profiler.

The replicated contract was not accepted on convenience alone.  The retained
shape-faithful probe measures the real O matmul, residual add, the next
post-attention norm, and router.  Its alternative reduce-scatters O to a local
`[1,1,1,1440]` residual, consumes that layout with distributed RMSNorm and a
row-sharded router, all-reduces only 32 FP32 logits, then gathers the normalized
hidden only when sparse gate/up requires the full dimension.

| Contract through O -> norm -> router | Result | Warmed latency, 20 repeats | Decision |
| --- | --- | ---: | --- |
| O row matmul -> all-reduce -> replicated add -> local RMSNorm -> router | reference | 0.529831 ms | selected |
| O row matmul -> reduce-scatter -> sharded add -> distributed RMSNorm -> row-sharded router/logit all-reduce -> gate/up input gather | norm PCC 1.0, router PCC 1.0 | 0.733042 ms | rejected; 38.4% slower |

The sharded path saves the all-gather half of O only until the next consumer.
It then pays a stats collective, a small 32-logit all-reduce, and a BF16 hidden
all-gather because the column-parallel sparse gate/up weights consume all 2,880
hidden elements.  Keeping the residual fractured across the whole layer
also changes the expert boundary from a cheap BFP8 all-gather into a BF16
hidden gather before the next layer's QKV.  It does not reduce steady-state
bytes for this geometry.

Other topology alternatives were checked as follows:

| Alternative | Contract / expected bytes per rank | Evidence and disposition |
| --- | --- | --- |
| reduce-scatter then immediate all-gather | 2,880 B RS + 2,880 B AG at O | exactly reconstructs the selected all-reduce; not treated as an optimization |
| fused all-gather + O matmul | would gather local 2,048 head features before a differently sharded O | reusable module requires a different WO ownership contract; column-O and replicated-O correctness controls were refuted during AutoFix |
| fused O matmul + reduce-scatter | 2,880 B O output | no generic fused API accepts this GPT-OSS `[32,2048] x [2048,2880]` row-parallel projection; separate RS is the measured candidate |
| full residual-sharded stack | O RS + distributed norm stats + full-hidden gathers for sparse gate/up and next QKV + expert RS | more bytes than two selected reductions because expert output CCL is BFP8 while required hidden gathers are BF16; measured O-to-router slice is already slower |
| whole-expert EP=2 | route-dependent expert traffic and output collective | rejected for one-token top-4 load imbalance and no per-expert bandwidth split |

The generic GPT-OSS `CCLManager` supplies persistent ping-pong semaphores.  The
public high-level all-reduce used here does not expose persistent output-buffer
arguments, but trace capture includes both collectives.  Since a representative
final decode replay spends only about 48 us in the four RS/AG operations, replacing
this path with a custom persistent collective was not the highest-value
bottleneck.

## Context and stack capacity

The public multichip cache contract is the HF-advertised 131,072 tokens.  Each
key or value cache is `[2048,4,64,64]` BF16 per device and layer (64 MiB), so K
plus V use 128 MiB/device/layer and exactly 3.0 GiB/device for 24 layers.

TP2 BFP8 expert weights are about 422 MB/device/layer; TP attention/router/norm
weights add about 26.8 MB/device/layer.  The estimated 24-layer weights are
10.03 GiB/device.  Adding 3.0 GiB KV cache, 1.2 GiB for TP embedding/head, and
a conservative 4.0 GiB runtime/trace/activation/fragmentation reserve gives
about 18.23 GiB/device, below the measured 31.875 GiB DRAM view.

`RUN_MULTICHIP_CONTEXT=1` physically allocates both caches and executes decode
at position 131,071.  It validates `[2048,4,64,64]` on each rank and the final
page update.  It does not claim that a single 131,072-token prefill computation
was benchmarked.  `doc/context_contract.json` records the distinction.

## Correctness and runtime evidence

The optimized references are captured in isolated `1x1` device sessions.
Opening a parent mesh while a single-chip submesh handle was live reproduced a
runtime deadlock, so baseline and TP execution intentionally never overlap
device handles.

| Gate | Sliding attention | Full attention |
| --- | ---: | ---: |
| real-weight prefill PCC, S=128 | 0.99936591 | 0.99938529 |
| minimum attention PCC, positions 128-130 | 0.99999608 | 0.99990406 |
| exact-baseline-attention routing control, positions 128-130 | >=0.999998 | 1.0 |
| minimum final decode PCC, positions 128-130 | 0.99909930 | 0.99887053 |
| warmed trace replay PCC, positions 128-130 | 1.0 | 1.0 |

Additional contracts:

- synthetic non-aligned prefill S=17 and S=33: PCC 0.99999965 and
  0.99999970; local rank cache slices match the single-chip KV-head slices at
  PCC 1.0 through a reversed physical page table;
- trace: hidden and position inputs are mutable, eager and trace caches are
  independent, five repeated replays are bit-identical, and the eager/trace
  physical K/V page is bit-identical on both TP ranks;
- stack boundary: one layer's replicated decode output is passed directly to a
  second layer call with no host conversion or gather;
- controlled active prefill, S=17: exactly 68 (=17x4) token-specific entries
  reach each of gate, up, and down, with no dense-group expert execution;
- near-tie stress, seed 3301 position 129: TP attention PCC 0.99999619; the
  fourth/fifth expert swap is deterministic, exactly four experts remain
  active, and feeding exact baseline attention through the same TP router/MoE
  gives routing PCC 1.0 and active-expert output PCC 0.99956265;
- runtime fallback audit: no `torch`, host conversion, CPU, or `from_torch` /
  `to_torch` call is present in runtime forward methods;
- watcher: nine gates pass with worker/dataflow watcher enabled on both chips.
  `TT_METAL_WATCHER_DISABLE_ETH=1` is required because firmware 19.8.0 leaves
  an ACTIVE_ETH heartbeat stopped during watcher fabric teardown; without that
  scoped setting test bodies pass but runtime teardown aborts.  The final
  worker watcher log contains no error/assert/hang signature.

## Warmed performance

Wall-clock runs use the same real layer, S=128, ten warmed prefill repeats, and
100 warmed trace replays.  Efficiency is speedup divided by two devices.

| Path | Optimized 1x1 | TP2 1x2 | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: |
| prefill | 13.453845 ms | 49.858989 ms | 0.269838x | 13.492% |
| traced decode | 0.975634 ms | 0.690026 ms | 1.413908x | 70.695% |

The exact token-specific active-prefill repair replaced an earlier fast but
incorrect generic grouping result; the 49.86 ms value is the honest final
path.  Sparse gate/up/down consume 27.197 ms (54.65%) and physical tile
padding plus compact/uncompact data movement accounts for most of the rest.
Decode retains the generic exact top-4 path and improves materially: both
norms use 10 cores, and a precision-locked QKV geometry sweep reduced that
kernel from about 60.5 us on 80 output cores to 40-41 us on 40 output cores.

All QKV candidates used BF16 inputs/weights/outputs, HiFi4, and FP32
accumulator destination.  Short 10-replay results were 0.712398 ms for the old
default, 0.694127 for K3, 0.691108 for K5, 0.691703 for K10, 0.697210 for O2,
0.695538 for O4, 0.690411 for K3O2, and 0.690858 for K5O4.  The final 100-replay
confirmation was 0.690263 ms for K3O2, 0.690447 for K5O4, and 0.690927 for K5;
K3O2 was selected.  Canonical sliding/full correctness and trace replay both
passed after the change.

The final-policy expert A/B uses the same BFP8/LoFi weights and exact active
routes for every row:

| Expert program | S=128 prefill | traced decode | Decision |
| --- | ---: | ---: | --- |
| 45/90 cores, `1x1` for both modes | 50.696530 ms | 0.690262 ms | rejected: slower prefill |
| 15/30 cores, `1x3` for both modes | 49.854757 ms | 0.693085 ms | rejected: slower decode |
| 15/30 `1x3` prefill + 45/90 `1x1` decode | 49.858989 ms | 0.690026 ms | selected; fastest of each mode within run noise |

All three runs passed their execution gate; selected real-weight prefill PCC
remains 0.99936591 sliding and 0.99938529 full, and the complete selected-path
correctness/trace/context suite is 9/9.

### `tt-perf-report` findings

| Profile range | Device ops | Device time | Op-to-op gap | Overall DRAM | DRAM roofline |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1x1 prefill | 57 | 13,355 us | 714 us | 190 GB/s | 37.2% |
| 1x2 prefill | 72 | 49,770 us | 738 us | 135 GB/s | 26.3% |
| 1x1 decode | 89 | 952 us | 83 us | 162 GB/s | 31.7% |
| 1x2 decode, 3 trace replays | 282 | 1,984 us | 434 us | 118 GB/s | 23.1% |

The final decode signpost contains three trace executions, or about 660.7 us
of device work per replay.  A representative replay spends 12+12 us on the
BF16 attention RS/AG and 14+10 us on the BFP8 expert RS/AG, about 48 us total.
Main kernels are QKV 40-41 us, O 31-32 us, router 24-25 us, gate/up 73-74 us,
expert down 67-68 us, and both norms 7-8 us on 10 cores.  Communication is not
the dominant decode cost.

Prefill attention CCL is 28+27 us and expert CCL is 43+23 us, 121 us total;
CCL is negligible beside exact active-expert compute and data movement.  The
three sparse projections take 9,139, 9,141, and 8,917 us.  Typecasts consume
8.431 ms, reshapes 7.059 ms, unary work 3.948 ms, and fill/pad 2.037 ms.
Thus the accepted prefill limitation is neither fabric nor host dispatch: it
is the sparse op's 32-row physical granularity plus compact/uncompact traffic.
Human-readable operation/advice tables, CSV summaries, stacked breakdowns,
and raw Tracy provenance are retained under `perf/`.

## Reproduction commands

Run baseline capture before TP correctness because device sessions must not
overlap:

```bash
pytest -q models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_capture_synthetic_single_chip_optimized_reference
pytest -q models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_capture_real_weight_single_chip_optimized_reference
MULTICHIP_REAL_SEED=3301 pytest -q 'models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_capture_real_weight_single_chip_optimized_reference[blackhole-sliding_attention-mesh_device0-device_params0]'
pytest -q models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py -k 'synthetic_non_aligned or real_weight_prefill_decode or near_tied_router or warmed_trace'
RUN_MULTICHIP_CONTEXT=1 pytest -q models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_full_context_cache_allocation_and_last_page_update
```

Final wall-clock and topology runs:

```bash
RUN_MULTICHIP_DECODER_PERF=1 MULTICHIP_DECODER_PERF_SEQ=128 MULTICHIP_DECODER_PREFILL_REPEATS=10 MULTICHIP_DECODER_TRACE_REPLAYS=100 pytest -q models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_capture_single_chip_optimized_perf_reference
RUN_MULTICHIP_DECODER_PERF=1 MULTICHIP_DECODER_PERF_SEQ=128 MULTICHIP_DECODER_PREFILL_REPEATS=10 MULTICHIP_DECODER_TRACE_REPLAYS=100 pytest -q models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_multichip_decoder_perf
RUN_MULTICHIP_DECODER_PERF=1 MULTICHIP_DECODER_PERF_SEQ=128 MULTICHIP_DECODER_PREFILL_REPEATS=1 MULTICHIP_DECODER_TRACE_REPLAYS=3 python -m tracy -r -p -o /tmp/gpt_oss_multichip_final -m pytest -- -q models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_multichip_decoder_perf
RUN_MULTICHIP_TOPOLOGY_PROBE=1 MULTICHIP_TOPOLOGY_REPEATS=20 pytest -q models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_sharded_residual_topology_candidate
```

Watcher is intentionally separate from profiling:

```bash
TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 RUN_MULTICHIP_CONTEXT=1 pytest -q models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py -k 'runtime_contract_and_fallback_audit or synthetic_non_aligned_prefill_matches_optimized_and_cache_is_head_local or real_weight_prefill_decode_matches_single_chip_optimized or near_tied_router_isolated_to_tp_attention_rounding or warmed_trace_replay_updates_hidden_position_and_paged_cache or full_context_cache_allocation_and_last_page_update'
```

Profiler CSV conversion:

```bash
tt-perf-report perf/single_chip/ops.csv --start-signpost PERF_PREFILL_SINGLE_CHIP --end-signpost PERF_PREFILL_SINGLE_CHIP_END --active-experts 4 --csv perf/single_chip/prefill_report.csv --no-color
tt-perf-report perf/single_chip/ops.csv --start-signpost PERF_DECODE_SINGLE_CHIP --end-signpost PERF_DECODE_SINGLE_CHIP_END --active-experts 4 --csv perf/single_chip/decode_report.csv --no-color
tt-perf-report perf/multichip/ops.csv --start-signpost PERF_PREFILL_MULTICHIP --end-signpost PERF_PREFILL_MULTICHIP_END --active-experts 4 --csv perf/multichip/prefill_report.csv --no-color
tt-perf-report perf/multichip/ops.csv --start-signpost PERF_DECODE_MULTICHIP --end-signpost PERF_DECODE_MULTICHIP_END --active-experts 4 --csv perf/multichip/decode_report.csv --no-color
tt-perf-report perf/multichip/ops.csv --start-signpost PERF_PREFILL_MULTICHIP --end-signpost PERF_PREFILL_MULTICHIP_END --active-experts 4 --no-summary --no-color > perf/multichip/prefill_report.txt
tt-perf-report perf/multichip/ops.csv --start-signpost PERF_DECODE_MULTICHIP --end-signpost PERF_DECODE_MULTICHIP_END --active-experts 4 --no-summary --no-color > perf/multichip/decode_report.txt
```

## Retained artifacts and limitations

- final wall timing: `logs/single_chip_perf_reference_seq128.json`,
  `logs/multichip_perf_result_seq128.json`, and `logs/perf_final_*.junit.xml`;
- correctness/trace/context/watcher:
  `logs/final_correctness_trace_context.junit.xml`,
  `logs/watcher_clean_workers_final.junit.xml`, and
  `logs/watcher_workers_final.log`;
- active-expert control and QKV AutoFix evidence:
  `logs/active_prefill_sparse_entries.junit.xml`,
  `logs/autofix_*.junit.xml`, and the final QKV geometry embedded in
  `logs/multichip_perf_result_seq128.json`;
- final expert geometry A/B:
  `logs/multichip_perf_expert_prefill_width1_seq128.json`,
  `logs/multichip_perf_expert_subblock_candidate_seq128.json`, matching JUnit
  files, and the selected result JSON;
- post-recovery controls: `logs/device_recovery_final.txt`,
  `logs/device_recovery_mesh_open_close.junit.xml`, and
  `logs/device_recovery_tt_smi_snapshot.json`;
- residual topology: `logs/residual_topology_candidate.json` and matching
  JUnit XML;
- profiler provenance: `perf/{single_chip,multichip}/ops.csv`, plus each
  `prefill_report.{txt,csv}` and `decode_report.{txt,csv}`;
- shard prior and rerun failure: `shard_advise/baseline_report.json`,
  `shard_advise/baseline_final_ir.mlir`, and `shard_advise/rerun_stderr.log`;
- serialized overlapping-handle hang evidence: `triage/` and
  `triage/serialized-overlap-repro.md`.

This implementation intentionally supports only the current 1x2 P300 mesh and
batch one.  Full-length prefill, a complete 24-layer stack, generation, and
vLLM serving belong to later pipeline stages.  Exact active prefill is correct
but has only 0.270x speedup at S=128 because token-specific sparse batches cost
one physical 32-row tile per active route; fixing the TTNN fused local-combine
kernel is outside this decoder-only stage.  The host also emits a low
`/dev/shm` warning (about 17.5 MiB free for a 16 MiB MPI segment) and an unknown
`B850M-C` motherboard fallback warning; neither changed correctness, profiler,
or final wall-clock outcomes.
