# Falcon3-7B-Base optimized multichip decoder

This is the optimized-multichip-decoder stage for
`tiiuae/Falcon3-7B-Base`. It optimizes the completed TP4 decoder layer in
place on the target `1x4` Blackhole p300c mesh. It does not add embeddings,
the 28-layer stack, generation, full-model code, or vLLM integration.

The selected decode path is genuinely tensor-parallel: QKV/gate/up remain
column-parallel, O/down remain row-parallel, KV heads and cache are rank-local,
and both material reductions run over mesh axis 1. There is no replicated or
single-chip fallback in the measured path.

## Selected implementation

- Decode O/down reductions use persistent asynchronous Ring all-reduce with
  two links and BFP8 CCL payloads. Prefill deliberately keeps the standard
  BF16 collective because the persistent decode buffers are sized for the
  fixed decode rows.
- Two preallocated CCL buffers and semaphores are reusable across all decoder
  layers through `persistent_ccl_resources`. They total 835,584 L1 bytes per
  device: 417,792 bytes per buffer, or 26,112 bytes per participating core.
- Decode returns a replicated-mesh BF16 TILE tensor that is already 32-core L1
  width-sharded inside each device. This is the inter-layer residual contract;
  the next decoder consumes it without a gather, reshard, all-reduce, or
  layout conversion. `return_sharded_decode_output=false` is an explicit
  true-public-boundary opt-out.
- The selected local matmuls retain BFP4/LoFi and DRAM-sharded decode weights.
  Attention and MLP decode projection outputs remain BFP8 through their
  local projection/activation and BFP8 CCL boundary, then typecast once for
  the BF16 residual add. The attention-only, MLP-only, and combined policies
  were measured independently; combined BFP8 is the selected default.
  An exact per-role geometry sweep changes QKV/O/gate-up/down from 8/8/24/8
  to 4/4/24/8 cores. QKV remains packed. Separate gate and up projections
  beat the measured rank-correct packed alternative.
- Paged K/V remain rank-local BFP8 DRAM tensors. Public sequence lengths and
  cache positions remain logical; internal padding, masking, and slicing own
  all tile and page alignment.

Falcon3-7B-Base is dense, not MoE, so active-expert execution is not
applicable.

## Before and after

All numbers use real layer-14 weights and recorded layer-14 activations. Each
decode value is the median of five samples of 100 warmed trace replays. Each
prefill value is the median of five warmed executions. The before artifacts
were captured from the completed multichip decoder at the start of this pass;
the after artifacts carry implementation SHA256
`1bb774f48c3dd19e9c4ba0550e6eb279809973ff0fc856552cd06f16d2b1a199`.

| Batch / phase | Before | Final default | Change |
| --- | ---: | ---: | ---: |
| 1 warmed prefill, seq 17 | 0.822900 ms | 0.902849 ms | 9.72% slower |
| 1 traced warmed decode | 0.356704 ms | **0.286597 ms** | **19.65% faster** |
| 32 warmed prefill, seq 17 | 3.125765 ms | **3.047774 ms** | **2.50% faster** |
| 32 traced warmed decode | 0.578790 ms | **0.491488 ms** | **15.08% faster** |

The selected changes do not alter prefill execution; its official single-call
samples are noisy at this duration. A controlled paired A/B used the final
4/4/24/8 geometry, BFP8 decode activations, and 50 synchronized prefill
executions per sample. Two persistent-resource runs measured
0.906946/0.857275 ms; two standard/no-buffer runs measured
0.861957/0.887350 ms. That overlap rules out persistent-buffer
allocation as the source of the old-to-new batch-1 difference. The table
still uses the latest exact-source default run, not a favorable repeat. Both
direct profiler-advice candidates lost: grid-x 10/subblock measured 0.924963
ms and L1 inputs measured 0.979226 ms.

The final default is also 2.2472x faster than the selected single-chip decoder
at batch 1 and 1.5636x faster at batch 32. These speedups are supporting
evidence only; the before/after comparison above is against the multichip
decoder required by this stage.

## Correctness and contract gates

Falcon3 has one meaningful dense decoder-layer kind. The representative
layer-14 path remains above the accepted real-weight PCC threshold throughout
prefill, sequential decode, and cache updates.

| Check | Final result |
| --- | ---: |
| TP4 prefill vs selected single-chip decoder, batch 32 / seq 17 | 0.999999758 |
| TP4 decode vs selected single-chip decoder, batch 32 / position 17 | 0.999999802 |
| TP4 K / V vs selected single-chip decoder | 0.999994839 / 0.999998595 |
| TP4 prefill vs HF, logical seq 31 | 0.999934822 |
| TP4 decode vs HF, positions 31 / 32 | 0.999113356 / 0.998935752 |
| TP4 prefill K / V vs HF | 0.996775280 / 0.995555068 |
| TP4 decode K / V vs HF, minimum | 0.996767833 / 0.995551142 |
| Logical seq 1,025, two internal chunks: output / K / V | 0.999944377 / 0.996792529 / 0.995554319 |
| Heterogeneous positions 17 / 31 vs independent batch-1 | 1.000000000 / 0.999976281 |
| Full logical seq 32,768 plus decode position 32,767 | passed; sampled K 0.996441066, V 0.995348166 |

The non-aligned gate uses a permuted page table, prefill length 31, and decode
positions 31 and 32 across a page boundary. Mesh-rank outputs are bitwise
equal, the program-cache count is stable, and eight trace replays are bitwise
deterministic. A second decode consumes the final output directly in the
documented L1 width-sharded layout.

`doc/context_contract.json` preserves the 32,768-token batch-1 contract and
records the additional shared L1 CCL storage plus BFP8 decode activation
policy. The activation change reduces decode transient traffic; it does not
change BF16 inter-layer layout or prefill/context capacity. No context
reduction was made.

## Operation-topology audit

The audit was performed before local tuning and measured candidates through
their material downstream consumer.

| Topology item | Finding and action | Evidence |
| --- | --- | --- |
| Repeated same-input matmuls | Gate and up share the normalized input. A rank-correct packed projection was implemented; 0.631922 ms decode lost to the 0.578790-ms starting split path, so split remains. | `results/candidates/packed_mlp/final_batch32.json` |
| Fused/packed QKV | QKV was already packed and rank-grouped in the completed TP4 path; no separate Q/K/V regression was introduced. | implementation and final profiler |
| Material collectives | O and down each produce a full-width row-parallel partial. Persistent async BF16 reduced batch-32 decode to 0.533160 ms; BFP8 reduced it to 0.529814 ms and retained accepted PCC. | `results/candidates/persistent_*` |
| CCL link count | The two-link Ring default beats the adapted one-link persistent candidate, 0.533160 vs 0.551102 ms under BF16 CCL. | `results/candidates/persistent_{async_ccl,links1}` |
| Persistent-buffer ownership | A real owner and borrower share both buffers and semaphores. Sequential execution, three borrower trace replays, borrower release, and owner reuse all pass at PCC 1.0; the pair is allocated once, not per layer. | `results/final_correctness/persistent_resource_sharing.json` |
| Boundary layout conversions | The old public output converted the replicated L1-sharded residual through interleaved L1 and DRAM. Returning the directly stackable residual lowers batch-32 decode to 0.504766 ms; the geometry-only 4/4/24/8 candidate is 0.503439 ms, and the final exact BFP8-activation default is 0.491488 ms. | `results/candidates/sharded_interlayer/`, `results/candidates/geometry/`, `results/final/` |
| Lower-movement fractured residual | The selected-policy family is measured as persistent-buffer BFP8 RS, local add, distributed RMSNorm, BFP8 statistics AG, BFP8 hidden AG, and the selected real QKV. It is 0.108096 ms versus 0.043906 ms for persistent BFP8 all-reduce plus the same residual/norm/QKV boundary: 2.462x slower at PCC 0.999851. Ring bytes sent per rank are 159,936 versus 156,672 because the next QKV still needs gathered normalization input. Two earlier clean BF16 repeats reach the same decision. It was not rejected using an immediate restore. | `results/final_correctness/fractured_selected_policy_boundary.json`, `results/candidates/fractured_repeat{1,2}/` |
| Gathered-input/local-output O/down | A real-weight BFP8 probe changes both material row-parallel boundaries to all-gathered input and local-output BFP4 weights, adds the fractured BF16 residual locally, and never restores replication. Separate async AG+matmul totals 0.173024 ms; fused AGMM improves it to 0.149221 ms, but selected row-parallel matmul+persistent AR totals 0.089642 ms. O is nearly tied (0.033311 vs 0.032602 ms); padded down loses decisively (0.115910 vs 0.057040 ms). Fused PCC is 0.999879/0.999154 and trace replay is deterministic. The first rank-2 weight API error was adapted to rank-4 local-output weights and rerun. | `results/candidates/o_down_agmm/o_down_agmm_decomposition.json` |
| Fused all-gather + next-QKV | The prior completed stage's real next-QKV AGMM family is 0.110681 ms versus 0.097550 ms replicated at PCC 0.999859. The new O/down family above crosses the selected BFP8 fractured contract and closes the distinct output-projection decomposition. | `../multichip_decoder/results/fused_agmm_boundary.json`, `results/candidates/o_down_agmm/` |
| Fused matmul + reduce-scatter | Exact O and padded global down shapes were retried on Blackhole with current persistent buffer API. Eager and trace pass. Fused O is 38.614 us vs 41.146 us separate; fused down is 114.594 us vs 115.865 us separate. The 1.3--2.5-us primitive gain cannot recover the complete fractured family's 12.5-us boundary loss. | `tracy/mmrs_{fused,non_fused}/` |
| Residual/activation sharding | A fresh shard-advisor graph proposed alternate QKV/O/MLP and 96-core residual layouts. Both coherent report and legacy-residual variants lose decode at 0.644761/0.641023 ms. | `shard_advise/`, `results/candidates/advisor_*` |
| DRAM-sharded decode matmuls | Selected. The advisor's L1/interleaved alternatives, including its exact activation reshards, lose as a coherent family. | same advisor artifacts |
| Per-role decode geometry | QKV 4/8/16, O 4/8/12/16/24/48, gate/up 8/12/24/48, and down 8/12/24/48 were executed with exact physical shapes, L1 input/output shards, eight-bank DRAM weights, HF PCC, and repeat evidence. O16 maps to the same 6x2/12-core physical program as O12; O48 maps to O24's 8x3 program. O12 narrowly edged O4 under BF16 activations, so both were re-crossed with the winning BFP8 activation family: O4 wins at batch 1 (0.286602 vs 0.288198 ms) and batch 32 (0.491443 vs 0.495796 ms). Final geometry is 4/4/24/8. The program class exposes `in0_block_w` and per-core M/N but no output-subblock fields. | `results/candidates/geometry/`, `results/candidates/activation_selected/o{4,12}_*`, `geometry_repeats/` |
| Activation/CCL dtype | Under the selected topology and all-BFP4/LoFi weights, BF16/BF16 decode is about 0.3026 ms, attention-only BFP8 is 0.295602 ms, MLP-only BFP8 is 0.293329 ms, and combined BFP8 is 0.286602 ms at PCC 0.999998147. Combined BFP8 is selected and reproduced by the final default at 0.286597 ms. CCL remains BFP8. | `results/candidates/activation_selected/`, `results/final/` |
| Precision/fidelity | BF16/HiFi4 initially exceeded L1; adapting CCL/layout and grids made it execute at 0.887704 ms batch 32. On the selected topology at batch 1, BFP8 HiFi2/LoFi are 0.401050/0.320416 ms; attention-BFP8/LoFi and MLP-BFP8/LoFi are 0.395538/0.325894 ms. Holding BFP4 fixed, attention HiFi2 is 0.308197 ms and MLP HiFi2 is 0.389421 ms. All-BFP4/LoFi retains accepted PCC and wins. | `results/candidates/precision_*`, `results/candidates/precision_selected/` |
| Prefill profiler advice | Grid-x 10/subblock and short-input L1 residency were both implemented and measured; both lose. Prefill trace caching is not a public-path replacement because shape-changing non-aligned logical lengths are part of the contract. | `results/candidates/prefill_*`, `tracy/final_default/prefill_perf_report.*` |

The asynchronous CCL API used here does not expose a `use_noc1_only` control,
and `use_optimal_ccl_for_llama` is not a Falcon model path. They are therefore
not model-applicable switches. All applicable async, buffer, layout, sharding,
matmul, CCL placement, link, dtype, and fidelity families were executed; none
is deferred.

## Profiler and safety evidence

The final same-source Tracy capture is delimited by
`MULTICHIP_PREFILL[_END]` and `MULTICHIP_DECODE[_END]`. The human-readable
`tt-perf-report` tables and corresponding CSVs are under
`tracy/final_default/`. The decode report is the selected path: it contains
the BFP4/LoFi DRAM-sharded matmuls and two BFP8
`AllReduceAsyncDeviceOperation` rows per replay. Its merged report totals
825 us of device work plus 144 us of visible gaps for three trace replays; the
reported overall modeled DRAM roofline is 22.3% (114 GB/s). Prefill totals
441 us of device work plus 1,005 us of host-dispatch gaps and reports 14.1%
(72 GB/s). Decode rows explicitly show BFP8 projection outputs, BFP8 gated
MLP input to down, and BFP8 async all-reduce inputs/outputs.

The runtime fallback audit inspects all hot methods and finds no Torch,
host-transfer, or single-chip decoder call. The serialized final Watcher run
passes 10 tests with 2 manual performance/profile tests skipped in 80.84 s;
the raw device
log has no error, fatal, exception, hang, timeout, assert, NoC-failure, or
stuck-waypoint match. Watcher and profiler were run separately, as required.
Post-run `tt-smi -s` at 2026-07-18T19:43:05Z reports four p300c devices with healthy DRAM and zero
corrected or uncorrected GDDR errors.

Firmware 19.8 is newer than the latest fully tested 19.5 bundle and the host
has only about 17.5 MB free in `/dev/shm`; both warnings are recorded. Neither
caused a correctness, Watcher, profiler, or final health failure.

## Artifacts

- `results/baseline/` and `results/final/`: exact before/after wall timings.
  `results/final_prefill_control/` contains the exact-source 50-execution
  paired prefill controls.
- `results/final_correctness/`: final PCC, non-aligned, heterogeneous,
  maximum-context, resource-sharing, topology-family, and direct-baseline
  gates. `results/watcher/` is the separately instrumented current-source run.
- `results/candidate_summary.csv`: selected/rejected family index.
- `results/candidates/activation_selected/`, `precision_selected/`, and
  `o_down_agmm/`: independent activation/fidelity matrices and the coherent
  gathered-input/local-output projection decomposition.
- `shard_advise/`: capture, report, IR, decision trace, and pipeline log.
- `tracy/final_default/`: final raw ops CSV, TTNN/API tables and CSV reports.
- `tracy/mmrs_{fused,non_fused}/`: exact fused matmul-reduce-scatter closure.
- `watcher_final.log`, `watcher_raw.log`, `watcher_clean.json`: Watcher proof.
- `device_health.json`: final target-mesh health summary.
- `work_log.md`: exact commands, chronology, provenance, and gate checklist.

The historical completed-multichip documentation remains under
`doc/multichip_decoder/`; this directory supersedes its runtime defaults and
its earlier fused-MMRS applicability assessment.
