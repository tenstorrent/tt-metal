# GPT-OSS 20B optimized decoder work log

Date: 2026-07-22 UTC

## Scope and starting point

This stage starts from `tt/fused_decoder.py` at commit
`0da25aafeac04e2ff21cb39c4687565dc28f7f98` and is limited to
`tt/optimized_decoder.py`, optimized-decoder tests/drivers, and
`doc/optimized_decoder/` plus the context-contract documentation if the final
memory policy changes capacity. It does not start multichip, full-model, or
vLLM work.

Hardware was healthy before the first stage run:

```text
timeout 60 tt-smi -ls --local
PASS: four local Blackhole P300c devices visible.

TT_VISIBLE_DEVICES=0
TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
python: open/close MeshShape(1, 1), trace_region_size=0
PASS: MESH_SMOKE_OK.
```

All hardware-facing commands are serialized. Watcher and profiler runs are
kept separate.

## Operation-topology audit before tuning

The audit uses the current fused source and the retained final fused reports
`doc/fused_decoder/perf/{prefill,decode}_tt_perf_report.csv`. The fused
baseline is the strongest correct path at stage entry: 7.158369 ms warmed
prefill and 5.891381 ms traced warmed decode in its same-process gate.

| Region / current sequence | Repeated input or movement | Candidate | Precision/layout constraints | Initial action |
| --- | --- | --- | --- | --- |
| input RMSNorm -> packed QKV linear -> create heads -> RoPE -> cache update -> SDPA | Q/K/V are already one packed matmul; decode SDPA requires interleaved output for GQA and concat-heads requires a height shard, leaving one measured 0.584 us conversion | preserve packed QKV and composite SDPA; later seed norm/QKV/O layouts from the mandatory advisor and compare DRAM-sharded attention | norms/residual BF16; attention weights start BF16, then BFP8/BFP4 and LoFi/HiFi trials; cache begins BF16 | preserve fused topology; advisor/tuning pending |
| concat heads -> O linear -> residual add | concat output is width-sharded, but O output and residual return to DRAM interleaved | advisor L1 residual chain; DRAM-sharded O alternative | GQA head concat and residual shard specifications must remain compatible; bias may require a separate add after sharded matmul | advisor/tuning pending |
| post-attention RMSNorm -> repeat token over all 32 experts -> packed or split dense gate/up -> exact clipped SwiGLU -> dense down -> route weight -> reduce | same normalized token is expanded to every expert; dense gate/up and down read all expert weights; many DRAM tilize/untilize/slice rows | routed active-expert `Experts` module using `ttnn.sparse_matmul`; retain exact router, bias, clipping, score weighting, and reduction | real weights required for precision decisions; BFP8/LoFi baseline, mandatory BFP4/LoFi trials; `nnz=None` because BF16 routing weights may flush and a wrong static count can deadlock | highest-priority rewrite; implement first |
| router FP32 linear -> BF16 top-k -> stable softmax -> scatter | router is shared by dense and sparse expert families; one small 1-core matmul | keep exact router semantics; advisor one-core layout is a candidate after sparse rewrite | router remains FP32/BF16 boundary because it is part of the emitted correctness contract | preserve, then measure |
| prefill dense expert path | split gate and up avoid strided slices but perform two 1.946 ms all-expert matmuls; down adds another 1.946 ms | sparse expert prefill with internal tile padding and output slicing | public logical sequence length must remain arbitrary; only the internal expert call may be padded to 32 | implement internal padding, never a public divisibility restriction |

### Baseline op accounting

| Path | Device time | Ops | Material rows |
| --- | ---: | ---: | --- |
| fused decode | 5,855.639 us | 56 | dense packed gate/up 2,597.105 us; dense down 1,945.872 us; QKV/O 73.750/71.421 us; two norms 44.999/44.978 us; SDPA 13.586 us |
| fused prefill S=17 | 7,028.254 us | 52 | dense split gate/up/down 1,945.734/1,945.627/1,945.620 us; QKV/O 74.021/89.170 us; two norms 45.016/44.867 us; SDPA 9.896 us |

The graph-fusing scan confirms that packed QKV, SDPA, rotary embedding,
create/concat-head kernels, TopK, cache-update composites, fused linear bias,
and the exact sigmoid-in-multiply rewrite are already present in the fused
starting point. The new applicable dedicated replacement is the repo-native
routed GPT-OSS expert module. Residual-plus-RMSNorm cannot be folded because
the residual sum is also the later skip input. There are no collectives in
this one-device stage.

## Candidate and command ledger

Results are appended here as candidates are run. A first API error is never
treated as a final rejection.

### Initial sparse-expert rewrite

The first cumulative candidate keeps the fused BF16 attention graph and
replaces the batch-one dense all-expert MoE with the shared routed
`Experts` implementation. It uses BFP8_B expert weights, LoFi sparse
matmuls, a 9x10 gate/up and down grid, and `in0_block_w=45`. The inherited
dense expert tensors are explicitly deallocated after the sparse tensors load,
so the selected batch-one path cannot fall back to the dense implementation.
Logical prefill lengths are padded only inside the expert module and sliced
back to their public extent.

The first non-aligned run exposed an output-buffer dtype mismatch: the shared
expert returns BFP8_B while the residual contract is BF16. Reusing that tensor
as the BF16 residual-add destination was rejected by TTNN. The repaired path
allocates the BF16 residual result explicitly; this was a real boundary fix,
not a rejected optimization.

```text
pytest .../test_optimized_decoder.py::test_optimized_decoder_non_aligned_and_deterministic -q -s
S=3:  output PCC 0.99988498, K 0.99994542, V 0.99995132
S=17: output PCC 0.99984944, K 0.99994559, V 0.99995043
S=33: output PCC 0.99984022, K 0.99994468, V 0.99995011
Repeated S=17 output and both caches: bitwise deterministic.
```

Fresh real-checkpoint coverage includes both the sliding-attention layer kind
(layer 12) and full-attention layer kind (layer 13):

| Layer/kind | Prefill output PCC | Prefill K/V PCC | Decode output PCC | Decode K/V PCC |
| --- | ---: | ---: | ---: | ---: |
| 12 / sliding | 0.99024635 | 0.99994780 / 0.99995241 | 0.99732448 | 0.99994649 / 0.99994865 |
| 13 / full | 0.99308115 | 0.99995099 / 0.99995162 | 0.99706825 | 0.99995790 / 0.99995405 |

Relative to the fused BF16 path, the initial BFP8 sparse expert policy changes
the layer-12 prefill output PCC by -0.002795 and layer-13 by -0.002348. Both
remain above the functional 0.99 bar; lower-precision candidates still require
real-weight qualification before selection.

The first same-process performance gate (S=17, 100 traced decode replays)
establishes the candidate baseline:

| Path | Warmed prefill mean / min | Traced decode mean / min |
| --- | ---: | ---: |
| fused baseline | 7.349289 / 7.172097 ms | 5.895582 / 5.887387 ms |
| sparse candidate | 3.829130 / 3.768075 ms | 0.952120 / 0.948899 ms |

This is a 47.9% prefill reduction and an 83.9% traced-decode reduction. It is
an intermediate candidate, not the final result; the mandatory advisor and
precision/geometry/layout sweeps follow.

## Mandatory shard-advisor gate (OPT-015)

The advisor was run fresh during this pass on the rewritten dense
attention-plus-MLP decode graph. The routed sparse expert op cannot be lowered
by the current advisor, so `shard_advise/advise_gpt_oss.py:decode` presents the
same attention/router/exact clipped-SwiGLU math with dense expert linears, as
required by OPT-015. Setup was performed in a separate shell as prescribed by
`shard-advise/SETUP.md` Part B:

```bash
export TTMLIR_ADVISOR_HOME=/home/mvasiljevic/tt-mlir
source "$TTMLIR_ADVISOR_HOME/scripts/bootstrap.sh"
export PYTHONPATH="$TTMLIR_ADVISOR_HOME/build_Release/python_packages:$PYTHONPATH"
export LD_LIBRARY_PATH="$TTMLIR_ADVISOR_HOME/build_Release/lib:$LD_LIBRARY_PATH"
ttnn-advise capture \
  models.autoports.openai_gpt_oss_20b.doc.optimized_decoder.shard_advise.advise_gpt_oss:decode \
  --tracer interception \
  --out models/autoports/openai_gpt_oss_20b/doc/optimized_decoder/shard_advise
```

The successful report covers 43 graph ops, records 40 final choices, ran the
spill pass with zero spills, and has no unfixable ops. Required fresh artifacts:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `shard_advise/report.json` | 13,710 | `2d41e8b909b8e86f7f8295b80bcaa8f48adb548dfd675da77d19480a5cfa6013` |
| `shard_advise/final_ir.mlir` | 29,317 | `331abe4bfe5d34117e78074d5102cf376662527baa996ed115d5f2ec5c290b8e` |

### Advisor recommendation disposition

Every recommended family was materialized as a candidate, including the full
dense-MoE chain. An initial sharded `typecast` API mismatch was adapted by
providing the output memory config; the first error was not used as a rejection.

| Advisor ops / recommendation | Action | Evidence and final disposition |
| --- | --- | --- |
| 4: input RMSNorm, L1 block shard on 10 cores | Implemented | Kept for batch-one decode; appears as a 10-core L1 block-sharded norm in the final profiler. |
| 5: packed QKV, width-sharded output, 11x8 1D matmul | Implemented | Kept with `in0_block_w=2`, `per_core_N=2`, output subblock 1x2. The final row is 72.664 us on 80 output cores. |
| 6-16: head transforms and SDPA boundaries | Adapted | Kept the legal L1/interleaved head boundaries, but cache slices/updates and SDPA remain DRAM where required by the paged-cache APIs. Explicit SDPA 8x8/k32 was then swept and kept. |
| 17-18: O projection plus residual in a width shard | Implemented | Kept 11x9 1D matmul, `in0_block_w=8`, 90-core width-sharded residual. The final O row is 58.319 us. |
| 20-33 and 41-44: post norm, all-expert repeat, dense gate/up/down, activation, weighting, reduction, residual | Implemented as one chain | Real layer-12 PCC passed (prefill 0.99304056, decode 0.99849431), and decode improved from dense fused 5.90123 to 5.70996 ms. Rejected as the final graph because routed sparse decode is 0.84683 ms and avoids reading all 32 experts. |
| 34-40: FP32 router, one-core height-sharded linear, top-k/softmax/scatter | Implemented | Kept for batch-one decode. Router precision remains FP32 through the linear and BF16 for top-k, matching the functional semantics. |
| Suggested continuous L1 expert boundary | Adapted and measured | Moving sparse expert input/routing weights to L1 preserved real layer-12 PCC but measured 3.97947/0.85112 ms versus the selected 4.10713/0.84683 final reproduction (and 3.83104/0.84725 in the immediately adjacent tuning run). Rejected for decode latency. |

The advisor is therefore the seed for the selected attention/residual/router
layouts, while its dense-MoE layout is retained as an opt-in diagnostic variant
and not as the shipped path. This is an evidence-based rejection, not a decision
to ignore the report.

## Precision, fidelity, sharding, and program search

All correctness decisions below used real `openai/gpt-oss-20b` checkpoint
weights for both meaningful layer kinds unless explicitly identified as a
performance-only geometry trial. Times are same-process S=17 warmed prefill and
100-replay traced decode means in milliseconds.

### Expert dtype and sparse geometry

| Candidate | Layer-12 / layer-13 PCC result | Prefill / decode ms | Decision |
| --- | --- | ---: | --- |
| BFP4_B/LoFi, 9x10, block 45 | 0.96961295 / 0.97277828 prefill | not promoted | Reject: both fail the 0.99 functional bar. |
| BFP4_B/LoFi, 5x6 | same material failure family | not promoted | Reject: smaller geometry cannot repair BFP4 accuracy. |
| BF16/LoFi, 9x10 | L12 0.990258/0.995283; L13 0.993081/0.993949 prefill/decode | 5.76075 / 1.06843 | Reject: correct but materially slower than BFP8_B. |
| BFP8_B/LoFi, 9x10, block 45 | L12 0.990246/0.995763; L13 0.993081/0.994192 | **4.10713 / 0.84683 final** | Keep. |

The BFP8_B geometry search was precision-locked and tuned gate/up and down
separately:

| Gate/up cores; down cores; block/subblock | Prefill / decode ms | Result |
| --- | ---: | --- |
| 5x6; 5x6; 45/1 | 4.42867 / 0.94234 | slower |
| 3x4; 5x6; 45/1 | 6.11874 / 1.16223 | slower |
| 8x8 first attempt | n/a | Illegal because 90 output tiles do not divide the 48-core rectangle; adapted to 9x5. |
| 9x5; 9x5; 45/1 | 5.11114 / 0.96295 | legal adapted candidate, slower |
| 9x10; 9x10; 30/1 | 3.89040 / 0.87462 | slower decode |
| gate/up block 30 only | 3.83309 / 0.86612 | slower decode |
| down block 30 only | 3.82777 / 0.85883 | slower decode |
| 9x10; 9x10; 90/1 | 3.91926 / 0.85493 | slower decode |
| 5x6; 5x6; 45/3 | 4.34516 / 0.88558 | legal wider-subblock retry, slower |
| **9x10; 9x10; 45/1** | **selected** | fastest correct decode family |

The profiler's subblock and L1-input advice was therefore acted on. A wider
legal subblock and an L1 sparse input both lost; the final 1x1 subblock and DRAM
input are deliberate. `nnz=None` is retained because a fixed count derived from
BF16 routing scores can disagree after device-side flush and deadlock the sparse
kernel. Gate/up and down are separate routed sparse matmuls, use the exact
1.703125 clipped SwiGLU coefficient, multiply by routing scores, and reduce the
active experts on device.

### Attention, cache, SDPA, and large-prefill choices

| Candidate | Real-weight correctness | Prefill / decode ms | Decision |
| --- | --- | ---: | --- |
| Advisor BF16/LoFi attention layouts | L12 decode 0.995459; L13 0.993901 | 3.88248 / 0.84943 in combined trial | Keep as the layout seed/default. |
| Non-DRAM BF16/LoFi forced | L12 0.995459; L13 0.993901 | 3.85912 / 0.84950 | Correct; auto already resolves to LoFi and is marginally faster in the final cumulative path. |
| Non-DRAM BF16/HiFi2 | L12 0.997232; L13 0.997001 | decode 0.88809 | Reject: higher PCC but slower; LoFi remains above 0.99. |
| DRAM-sharded BFP4_B/LoFi, max cores | L12 0.772627; L13 0.770887 | not promoted | Reject on real weights. |
| DRAM-sharded BFP4_B/LoFi, 64-core cap | L12 0.769485; L13 0.771220 | not promoted | Adapted smaller-core geometry also fails correctness. |
| DRAM-sharded BFP8_B/HiFi2 | L12 0.997015; L13 0.996686 | 3.83867 / 0.92138 | Reject: correct, slower decode. |
| DRAM-sharded BF16/HiFi4 | L12 0.997154; L13 0.997141 | 3.83274 / 0.99585 | Reject: correct, slower decode. |
| BF16 KV cache | functional baseline | 3.93331 / comparable pre-SDPA baseline | Replaced after BFP8 qualification. |
| BFP8_B KV cache | L12 decode/cache 0.995551/>0.99980; L13 0.994343/>0.99981 | 3.93331 / 0.84968 | Keep for persistent capacity; current-token K/V stay BF16. |
| SDPA default | correct | decode about 0.8495 | Replaced by explicit legal configuration. |
| SDPA 8x8, k=64 | same PCC as k=32 | 3.83547 / 0.84682 | Reject: no benefit over k=32. |
| **SDPA 8x8, k=32 + BFP8 cache** | L12 decode 0.995763; L13 0.994192 | 3.83104 / 0.84725 adjacent trial | Keep; final reproduced decode 0.84683 ms. |

For S=128, automatic QKV/O configs measured 13.05792 ms, explicit 8x4
measured 13.11501 ms, and explicit 10x4 measured 13.04211 ms. The 10x4 2D
configs are selected only at S=128. This narrow winner remains slower than the
dense fused S=128 prefill (9.56244 ms) because a 128-token group activates all
32 experts; this is a named workload limitation, not hidden by the S=17 result.
The public path accepts arbitrary valid logical lengths and uses internal
padding/slicing; there is no `seq_len % chunk == 0` restriction.

## Final correctness and semantic coverage

The final rerun uses the selected default, BFP8_B cache, and real checkpoint
weights. All entries exceed the functional 0.99 PCC bar.

| Layer / kind | Prefill output | Prefill K / V | Decode output | Decode K / V |
| --- | ---: | ---: | ---: | ---: |
| 12 / sliding | 0.99024635 | 0.99991794 / 0.99992703 | 0.99576329 | 0.99981871 / 0.99980306 |
| 13 / full | 0.99308115 | 0.99990794 / 0.99992701 | 0.99419248 | 0.99982619 / 0.99981078 |

Against the fused path, output PCC changes are -0.002795/-0.003535 for layer
12 prefill/decode and -0.002348/-0.005158 for layer 13. The delta is explained
by BFP8_B routed expert weights and the persistent BFP8_B cache; BF16 and HiFi2
controls confirm the expected precision tradeoff. No lower-precision real-weight
candidate that passed the bar was rejected in favor of a slower policy.

Additional final coverage:

- S=3/17/33 non-aligned prefill passes; S=17 output and full caches repeat
  bitwise-identically.
- S=128 prefill passes with output PCC 0.99903004 and cache PCC
  0.99984336/0.99984904, advancing the validated context boundary to 128.
- Batch-2 prefill/decode passes at 0.99988596/0.99981636. The shared sparse
  module is batch-one-only, so batch 2 intentionally retains the exact fused
  dense expert compatibility branch; dense tensors are absent and fallback is
  impossible in the measured batch-one optimized path.
- Paged decode positions 3 through 18 pass output and full-prefix cache PCC;
  untouched cache tails remain bitwise unchanged.
- Ten traced replays are bitwise deterministic for output and the entire K/V
  cache, and match the HF reference above 0.99.

`run_20260722_final_correctness_rerun.log` records 8 passed and 2 opt-in skips.
The first full run exposed the batch-2 use of a batch-one advisor shard; adding
the explicit batch-one guard fixed it, and the complete suite was rerun.

## Final performance and profiler evidence

The final same-process gate uses S=17, ten warmed prefill iterations, and 100
traced warmed decode replays:

| Path | Prefill mean / min | Traced decode mean / min |
| --- | ---: | ---: |
| fused starting path | 7.331439 / 7.093966 ms | 5.904240 / 5.885844 ms |
| **final optimized default** | **4.107129 / 3.978260 ms** | **0.846833 / 0.844108 ms** |

The final default is 44.0% faster for S=17 prefill and 85.7% faster for traced
decode. It also beats the earlier optimized artifact at commit
`9949cb70f3f82cde84cd725d864bdb092c97ea62` (0.928144 ms decode) by 8.8%.
The authoritative final evidence is `run_20260722_final_perf.log`; earlier
candidate logs are not substituted for this reproduced number.

Fresh Tracy/device-profiler collection was run separately from watcher with
real layer-12 weights and bounded `PERF_PREFILL`/`PERF_DECODE` signposts.
Compact reports are under `tracy/final/`; raw Tracy CSVs are intentionally not
stage artifacts.

| Profile window | Device ops | Op gaps | Total | Modeled DRAM roofline |
| --- | ---: | ---: | ---: | ---: |
| S=17 warmed prefill | 3,665.898 us | 608.621 us | 4,274.519 us | 46.6%, 239 GB/s |
| traced warmed decode | 860.461 us | 80.932 us | 941.393 us | 35.0%, 179 GB/s |

The profile build/instrumentation makes its one-replay total 0.0946 ms slower
than the separate 100-replay wall-clock benchmark; the benchmark is the latency
gate and the profile is the op attribution. A simple selected-path traffic
estimate is about 159.2 MB/token, or 0.311 ms at the nominal 512 GB/s peak; the
gap to the profiled device total is consistent with sparse gather efficiency,
non-modeled composite work, and launch gaps rather than a hidden host path.

Decode sparse gate/up/down rows take 132.351/131.467/128.541 us (45.6% of the
profile total). The report proves BFP8 weights and LoFi math reached all three
dominant rows. Packed QKV and O are BF16/LoFi at 72.664/58.319 us. SDPA is
9.773 us. Prefill sparse expert rows total 2,334.332 us (63.7%); internal
padding makes all experts active for this small group. The `untilize`/`tilize`
rows around routing are the required TopK/sparse-matmul row-major sparsity
contract in the shared expert module, not host conversions. Decode layout
conversions are explicit advisor/head/cache boundaries and total only a few
microseconds; there is no Python `torch`, `from_torch`, `to_torch`, host golden
fallback, collective, or implicit functional forward in either measured window.

## Final commands and device safety

All hardware commands used device 0 and the one-chip mesh descriptor and were
serialized. Profiler and watcher were separate processes.

```bash
TT_VISIBLE_DEVICES=0 \
TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
pytest -q models/autoports/openai_gpt_oss_20b/tests/test_optimized_decoder.py -s

TT_VISIBLE_DEVICES=0 \
TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
RUN_OPTIMIZED_DECODER_PERF=1 OPTIMIZED_DECODER_PREFILL_REPEATS=10 \
OPTIMIZED_DECODER_TRACE_REPLAYS=100 pytest -q \
models/autoports/openai_gpt_oss_20b/tests/test_optimized_decoder.py::test_optimized_beats_fused_warmed_prefill_and_traced_decode -s

TT_VISIBLE_DEVICES=0 \
TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}' pytest -q \
models/autoports/openai_gpt_oss_20b/tests/test_optimized_decoder.py \
-k 'real_weight_layer_kind or repeated_paged_decode_stress or traced_decode_output_and_full_cache_integrity' -s
```

The final watcher run is recorded in `run_20260722_watcher.log`: both real
layer kinds, repeated paged decode, and traced replay pass with no watcher
assert, illegal NoC, hang, or corruption report. Firmware 19.8 is newer than
the latest fully tested 19.5 and emits the standard warning; it caused no run
failure. The post-run device closed normally and required no reset.

## Optimize checklist disposition

- [x] Owned prefill/decode methods execute the optimized batch-one path; static
  audit and trace tests rule out inherited functional forward or host fallback.
- [x] Decode attention/norm/residual uses advisor-seeded L1 shards. The full
  dense L1 MLP chain and sparse-input L1 chain were both measured; sparse DRAM
  input is retained because it is faster with the shared active-expert kernel.
- [x] Prefill remains DRAM interleaved, with 10x4 2D QKV/O configs at S=128.
- [x] Operation-topology and `$graph-fusing` audits are recorded above; packed
  QKV, composite SDPA/cache/head ops, and routed `sparse_matmul` are kept.
- [x] `$shard-advise` ran fresh; required artifacts, hashes, applied entries,
  adapted retries, and measured rejections are recorded above.
- [x] Multi-device collectives, lower-movement collective residuals, fused CCL
  matmuls, and persistent CCL buffers are not applicable to this one-device
  stage. No collective appears in source or the measured reports.
- [x] Final default beats the strongest current fused baseline, the historical
  optimized artifact, and the material correct candidates; it was reproduced
  under final code.
- [x] Profiler rows prove final expert/attention dtype and fidelity policies.
- [x] Composite SDPA and packed QKV are used. GPT-OSS gate/up weights are stored
  packed, while the routed kernel consumes separate gate/up projections; the
  all-expert packed and split dense controls are both much slower.
- [x] Important norms, projections, sparse matmuls, SDPA, cache, and residuals
  have explicit memory/program/compute policies where their APIs accept them.
- [x] Dominant expert core grids, `in0_block_w`, subblock widths, input memory,
  dtype, and fidelity were swept under the selected BFP8/LoFi policy.
- [x] Attention precision/fidelity was swept independently, including two real
  BFP4/LoFi DRAM-sharded geometries and correct BFP8/BF16 controls.
- [x] Dense/expert BFP4/LoFi gate/up and down were tried before lower-priority
  prefill tuning and rejected on real-weight PCC.
- [x] DRAM-sharded attention geometries were tried and rejected by real PCC or
  traced latency; the advisor L1 1D configs win the target decode workload.
- [x] Routed active-expert semantics, sparse input conversion, routing-score
  weighting, expert reduction, and dynamic `nnz` are validated. The measured
  batch-one path has no dense expert fallback.
- [x] LM head, sampling, full-model token feedback, multichip, and vLLM items are
  out of scope for this decoder-layer-only stage.
- [x] Roofline, device-op time, gaps, and separate end-to-end traced latency are
  reconciled above. A full-model `perf_summary.json` is not applicable.
- [x] Batch-one is the latency target; batch 2 is the largest compatibility case
  exercised because the shared routed expert implementation is batch-one-only.
- [x] Non-aligned, cache, deterministic, repeated, real-weight, trace, profiler,
  and watcher-clean evidence all exist.

## Limitations

- S=128 sparse prefill is correct but slower than dense fused prefill because
  the tile-padded token group activates all experts. S=17 remains faster.
- The shared sparse expert implementation currently supports batch one; batch
  two preserves semantics through the fused dense compatibility branch and is
  not the measured optimized latency path.
- The configured cache/mask extent is 128. The HF-advertised 131072 context is
  incompatible with the inherited dense-mask design on one device; this stage
  increases validated capability to 128 and does not reduce it.
