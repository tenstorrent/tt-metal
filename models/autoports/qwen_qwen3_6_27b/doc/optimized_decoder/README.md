# Qwen3.6-27B optimized decoder

This stage adds the single-device `OptimizedDecoder` for representative linear
attention layer 0 and full-attention layer 3. It does not start multichip,
full-model, or vLLM work.

## Result

The final runtime owns its public prefill and decode entry points. Full
attention and both decode token mixers are optimized overrides. Linear
prefill intentionally reuses the proven functional mixer's implementation as
a method on the optimized object; it does not construct or call a fallback
`FunctionalDecoder`. Decode keeps the
residual/norm/MLP stream width-sharded in L1, uses DRAM-width-sharded weights
and explicit decode programs, and uses explicit sharded RMSNorm and SDPA
configs. Prefill uses DRAM-interleaved activations and explicit 8x10 2-D
matmul programs. No measured window contains a Torch conversion, host fallback,
or functional-decoder object.

Prompts above 32K use an optimized, memory-bounded two-pass path: K/V are
projected and filled into their paged cache in aligned chunks, then Q/gate,
chunked SDPA, output projection, and MLP are evaluated in bounded chunks. The
packed projection remains the faster serving-prompt path. A capacity-only run
completed at the functional contract's single-pass limit of 192,511 tokens
with nonzero output and populated cache.

The selected policy is layer-kind-aware:

| Group | Linear attention | Full attention |
|---|---|---|
| Attention/token-mixer weights | packed BFP4/LoFi input and BFP4/LoFi output DRAM-sharded projections | BF16/HiFi2 projections; HiFi4 SDPA |
| MLP gate/up/down | BFP4/LoFi | BFP4/LoFi |
| Persistent recurrent state | BFP8 (BF16 recurrent math) | N/A |
| Full-attention KV cache | N/A | BFP8 |
| Decode matmul | DRAM-sharded packed input/output and MLP | DRAM-sharded QKV/O/MLP |
| MLP projection form | split gate/up with fused SiLU | split gate/up with fused SiLU |
| QKV projection form | packed `qkv/z/b/a` | packed Q/K/V/gate |

Official weights determined the full-attention exception. BFP8 attention was
fast but reached only PCC 0.898606 against the completed functional output.
BF16 attention plus BFP4 MLP preserved the intended HF layer at PCC 0.997612
at B1 and 0.998095 at B32.
Changing that passing policy's cache from BF16 to BFP8 retained PCC 0.997073
against the functional control and improved B32 decode from 2.060 to 2.045 ms.
The context contract is not reduced; BFP8 KV halves persistent cache bytes.
For linear attention, official-weight prefill-to-decode transitions selected
BFP8 persistent recurrent state: minimum PCC was 0.997950 for FP32, 0.997950
for BF16, and 0.997965 for BFP8. BFP4 fell to 0.993340, below the 0.995 bar,
and was rejected. With BFP8 state fixed, independently reducing the packed
input and output projections to BFP4/LoFi retained minimum official transition
PCC 0.997175 and improved traced decode from 1.925025/16.198853 ms to
1.710466/15.986414 ms at B1/B32. Crossing that precision policy with legal
geometry selected packed-input width 5 and output width 12, reaching
1.670179/15.949088 ms while retaining official transition PCC 0.997167.

## Operation-topology audit

| Path | Current functional topology | Candidate | Action | Evidence |
|---|---|---|---|---|
| Full Q/K/V/gate | q+gate, K, and V are three same-input matmuls followed by concat | repack `[Q,K,V,gate]` into one projection | kept | one DRAM-sharded matmul; official QKV projection PCC 0.999841 |
| QKV heads | width-sharded packed output fed directly to head creation | add a narrow L1-interleaved consumer boundary | kept as required movement | V-head PCC changed from -0.020282 to 0.999837 |
| Q/gate order | functional flat-splits q_proj | repack HF per-head `[q,gate]` order at load time | kept | final optimized official HF layer PCC 0.997612; functional HF control was 0.687928 |
| MLP gate/up | two same-input matmuls | packed gate/up then split | rejected | packed BFP4: 1.156 ms B1 / 1.357 ms B32; split: 1.103 / 1.301 ms in the controlled full-layer candidate |
| Decode residual | DRAM interleaved around norms, residuals, and MLP | coherent L1 width-sharded chain | kept | final traced improvements at B1 and B32; no fallback |
| Decode dense matmuls | interleaved weights/default programs | DRAM-width-sharded weights with `per_core_M=1` | kept | final profiler rows show `DRAM Sharded=True`; B1 is legal because M is tile-padded to 32 |
| Prefill dense matmuls | framework-selected programs | explicit large 8x10 2-D configs | kept | final B1/B32 prefill improvements; block-width and grid sweeps below |
| Long prefill live set | full-sequence packed Q/K/V/gate and MLP intermediates | page-aligned K/V fill plus bounded Q/gate/O and MLP chunks | kept above 32K | S=192511 completed; initial unbounded attempt hit a hard 295,698,432-byte DRAM allocation failure at 4,092,087,616 bytes/bank allocated |
| Full attention | paged decode SDPA/default config | explicit 8x8 paged SDPA and explicit 8x8 causal prefill SDPA | kept | paged traced decode and non-aligned prefill pass |
| Linear recurrent update | `keyᵀ @ delta`, a tile-padded K=1 batched matmul | broadcast outer product | kept | isolated B1 2.110 ms; final projection geometry reaches 1.670/15.956 ms B1/B32 and official PCC 0.998717 |
| Linear same-input projections | `in_qkv`, z, b, and a are four matmuls | pack all into one independently typed DRAM-sharded projection | BFP4/LoFi width 5 kept | precision baseline width 2 was 1.710/15.986 ms; width 5 plus selected output reaches 1.670/15.956 ms |
| Linear output projection | DRAM-interleaved weight/default program | independently typed DRAM-width-sharded weight | BFP4/LoFi width 12 kept | widths 1/2/3/4/6/8/12/24 swept at B1/B32; cumulative width 12 is fastest traced B1 and best cumulative B32 |
| Linear DRAM-sharded subblock | profiler cannot recover output subblock | expose/sweep subblock | unavailable by API contract | config exposes only `in0_block_w`, `per_core_M/N`, and fused activation; factory internally selects 1x8 input and 1x7 output |
| Linear recurrent matmuls | bare default matmuls over the Tile32 state composite | explicit 1-D programs, block widths 1/2/4, 1x2 subblock, HiFi2/HiFi4 | width-4 HiFi2 kept | under final BFP8 state, B32 recurrent row mean fell from 4.261 ms at width 1 to 1.717 ms at width 4; whole layer fell from 21.268 to 16.190 ms |
| Linear persistent state | FP32 `[B,32,128,128]` | BF16, BFP8, and BFP4 physical storage with explicit BF16 math boundary | BFP8 kept | real transition min PCC FP32/BF16/BFP8 = 0.997950/0.997950/0.997965; BFP4 = 0.993340 |
| Linear recurrence layout | interleaved 32x32-tile state composite | L1-sharded 1-D or DRAM-height-sharded Tile(4,32) family | blocked by exact contracts | sharded mcast A requires fused batch while recurrent B has batch 48/1536; B32 Tile32 A is 3 MiB/core on four cores; Tile(4,32) requires a producer retile not exposed at this boundary |
| Host/layout crossings | loader uses Torch and `from_torch`; linear composite has required interleaved boundaries | remove runtime host crossings and avoid isolated reshards | kept | all eight final4 decode/prefill `perf.csv` files contain zero Torch/fallback rows |

There are no collectives in this single-device stage. CCL, fused matmul-CCL,
persistent CCL buffers, MoE, LM-head, and sampling checklist items are not
applicable.

## Correctness

All checks use full model dimensions and `throw_exception_on_fallback=True`.

| Check | Linear attention | Full attention |
|---|---:|---:|
| Official-weight decode | 0.998717 vs HF | 0.997612 B1 / 0.998095 B32 vs HF |
| Synthetic prefill, non-aligned | 0.999996 at S=5; 0.999996 at S=65 | 0.999994 at S=33 |
| Traced decode B1, 10 steps | 0.999987–0.999997 | 0.999009–0.999977 |
| Traced decode B32, 10 steps | 0.999968–0.999998 | 0.999560–0.999979 |
| Exact fresh-run determinism | decode and prefill bit-exact | decode and prefill bit-exact |
| Stateful transition | S=513 prefill + 16 decode steps passes; repeated B32 transition bit-exact | paged transition covered by traced decode |
| 10-step B32 watcher stress | 0.999968–0.999998, watcher clean | 0.999557–0.999979, watcher clean |
| Capacity-only single-pass prefill | S=192511; 985,656,229 output and 129,830 BFP8-state nonzeros | S=192511, nonzero output/cache |

The synthetic full-attention fixture uses diagonal projections and originally
masked two real-weight bugs. The durable official-weight test and the QKV/V-head
probe prevent that regression. The optimized code corrects HF's per-head
q/gate convention without editing the completed functional stage.

## Performance

Warmed latency is wall-clock trace replay for decode and synchronized warmed
full-layer execution for prefill. Units are milliseconds.

| Layer kind / phase | Batch | Functional | Optimized | Change |
|---|---:|---:|---:|---:|
| Full decode, traced | 1 | 2.324382 | 1.268103 | -45.4% |
| Full decode, traced | 32 | 2.529912 | 1.453556 | -42.5% |
| Linear decode, traced | 1 | 3.006078 | 1.670179 | -44.4% |
| Linear decode, traced | 32 | 21.440020 | 15.949088 | -25.6% |
| Full prefill, S=33 | 1 | 3.820037 | 3.255676 | -14.8% |
| Full prefill, S=33 | 32 | 68.680565 | 16.559895 | -75.9% |
| Linear prefill, S=5 | 1 | 11.663751 | 11.086341 | -5.0% |
| Linear prefill, S=5 | 32 | 313.205223 | 275.375946 | -12.1% |

The final default is slower than the aggressive, real-weight-invalid BFP4
full-attention candidate (1.10 ms B1), but it is the fastest candidate that
passes the official-weight bar. Per-role geometry and fidelity tuning closes
most of that gap while retaining HiFi4 SDPA. It beats the strongest correct
functional baseline at the primary B1 target and improves B32.
Linear prefill remains faster than functional at both batches, although the
final BFP8 persistent-state boundary adds per-chunk FP32/BFP8 conversion and
therefore gives up part of the earlier FP32-state prefill gain.

### Candidate evidence

The machine-readable `artifacts/candidate_matrix.csv` records both batches,
resolved dtype/fidelity, grid/shard geometry, block/subblock choice, profiler
row time, whole traced-layer time, correctness, and keep/reject reason.
Every candidate console result is retained under `artifacts/candidates/`.
`artifacts/program_contracts.json` records the exact TTNN API/validator
boundaries for the non-expressible alternatives.

| Candidate | B1/B32 decode or prefill result | Decision |
|---|---|---|
| Decode `in0_block_w=20` | packed-QKV CB 2.243 MB > 1.573 MB L1 | adapt, not reject |
| Decode block width 5 | QKV fits; packed MLP CB 1.585 MB > 1.573 MB | adapt |
| Decode block width 4 | CB overlaps persistent residual at byte 1,212,416 | adapt |
| Decode block width 2 | both layer kinds pass synthetic PCC | legal geometry |
| BFP8/HiFi2 packed policy, block width 1 | persistent L1/CB clash remains; attention-only BFP8 control later runs | do not use packed all-BFP8 |
| BFP8 attention + BFP4 MLP after layout fix | PCC 0.898606 vs functional official-weight control | reject for accuracy |
| BF16 attention + BFP4 MLP + BF16 KV | PCC 0.997086; 1.858/2.060 ms | correct |
| Same policy + BFP8 KV | PCC 0.997073; 1.858/2.045 ms | selected |
| Full cumulative baseline | 1.265899/1.455179 ms | selected topology |
| Cumulative QKV width 4 | CB 1,618,688 > 1,572,864 bytes at both batches | reject; width 2 retained |
| Cumulative O widths 4/6/8 | 1.266152/1.454871; 1.269347/1.455031; 1.268025/1.456613 ms | width 3 retained; alternatives are neutral/slower at B32 |
| Cumulative O width 12 | CB 1,678,080 > 1,572,864 bytes at both batches | reject |
| Cumulative gate width 2/4 | 1.291910/1.482203; 1.271421/1.456364 ms | width 5 baseline retained |
| Cumulative up width 2/4 | 1.290507/1.477614; 1.270679/1.455912 ms | width 5 baseline retained |
| Full gate/up widths 10 and 20 | width 10 collides with persistent L1; width 20 requires 2,690,816 B CB vs 1,572,864 B L1 at both batches | reject after device attempts |
| Full coherent four-core storage grid | exact L1/CB collision at B1 and B32 | reject after adapted device attempt |
| Full MLP LoFi vs same-dtype HiFi2 | 1.268/1.454 vs 1.674/1.859 ms B1/B32; identical official PCC | LoFi retained |
| Cumulative down width 4 | 1.298532/1.481572 ms | reject for latency |
| Cumulative down widths 34/68 | L1 overlap / CB 2,782,976 bytes at both batches | reject |
| QKV-only / O-only HiFi2 | 1.647 / 1.771 ms | both kept; SDPA remains HiFi4 |
| Linear packed input/output | 2.046/20.412 ms | kept |
| Linear packed width 4 | CB 1,872,640 > 1,572,864 bytes | reject; larger widths monotonic |
| Linear broadcast outer update | 2.110 ms B1 isolated | kept cumulatively |
| Linear recurrent w1 / w2 / w4 under BFP8 state | 2.084/21.268; 1.976/17.728; 1.927/16.190 ms | width 4 selected at both batches |
| Linear recurrent 1x2 subblock | 1.999/18.591 ms | reject |
| Linear recurrent HiFi4 | 1.938/16.468 ms | reject; HiFi2 retained |
| Linear recurrent state FP32/BF16/BFP8 | real transition min PCC 0.997950/0.997950/0.997965; BFP8 traced 1.930/16.198 ms | BFP8 selected |
| Linear recurrent state BFP4 | real transition min PCC 0.993340 | reject below 0.995 |
| Linear projection BF16/HiFi2 baseline | 1.925/16.199 ms | correct control |
| Linear packed-input BF16/LoFi; output BF16/LoFi; both | 1.925/16.172; 1.928/16.188; 1.923/16.180 ms | reject: smaller/noisy gains |
| Linear packed-input BFP8/HiFi2 and BFP8/LoFi | 1.896/16.164 and 1.772/16.045 ms | BFP8/LoFi passes official transition PCC 0.997965; BFP4 is faster |
| Linear output BFP8/HiFi2 and BFP8/LoFi | 1.919/16.191 and 1.872/16.139 ms | BFP4 is faster |
| Linear packed-input BFP4/LoFi only | 1.766/16.028 ms; official transition min PCC 0.997432 | independently valid |
| Linear output BFP4/LoFi only | 1.873/16.138 ms | independently valid |
| Linear both projections BFP4/LoFi at input w2 / output w3 | 1.710/15.986 ms; official transition min PCC 0.997175 | precision baseline |
| Linear packed-input widths 1/4/5 | 1.834/16.101; 1.691/15.974; 1.684/15.960 ms | width 5 selected |
| Linear packed-input widths 10/20 | width 10 overlaps L1 at 1,357,824 vs CB end 1,422,976; width 20 needs 2,587,136 B CB vs 1,572,864 B L1 | reject exact blockers at both batches |
| Linear output widths 1/2/4/6/8/12/24 | B1 1.777/1.720/1.704/1.699/1.698/1.696/1.695 ms; all pass B32 | cross leading widths cumulatively |
| Linear cumulative input w5 + output w8/w12/w24 | 1.674/15.958; 1.670/15.943; 1.672/15.944 ms | width 5/12 selected |
| Linear four-core storage | exact L1/CB collision at B1 and B32 | reject |
| Linear cumulative final | 1.670/15.956 ms; official decode PCC 0.998717; transition min 0.997167 | selected |
| All BF16/HiFi4 | correct control; 3.008/3.207 ms | reject for latency |
| Packed gate/up BFP4 | 1.156/1.357 ms | reject |
| Split gate/up BFP4 | 1.103/1.301 ms | selected topology |
| Prefill K block 2 | 3.271/13.350 ms under BFP4 candidate | slower than block 4 there; BF16 attention requires block 2 at B32 |
| Prefill 8x8 grid | 3.027 ms B1; B32 CB 1.670 MB > 1.573 MB | reject; 8x10 selected |

### Profiler accounting

The compact reports are under `artifacts/tracy/`. Final reports use the
`final4_{full,linear}_{decode,prefill}_{b1,b32}` prefixes. Each directory
contains `profile_run.json`, which records phase, sequence, iteration count,
exact Tracy and `tt-perf-report` argv, exit status, resolved policy, source
raw-report path, and filtered outputs. Raw Tracy reports were removed after
compact evidence was generated; the retained wrapper provenance makes them
reproducible.

| Window | Device time / iteration | Wall time | DRAM roofline |
|---|---:|---:|---:|
| Full decode B1 | 1.200407 ms | 1.268103 ms | 55.9%, 286 GB/s |
| Full decode B32 | 1.278195 ms | 1.453556 ms | 52.5%, 269 GB/s |
| Full prefill B1 | 3.054999 ms | 3.255676 ms | 22.8%, 117 GB/s |
| Full prefill B32 | 16.310691 ms | 16.559895 ms | 8.9%, 46 GB/s |
| Linear decode B1 | 1.521271 ms | 1.687251 ms | 25.2%, 129 GB/s |
| Linear decode B32 | 15.893577 ms | 15.975878 ms | 4.2%, 22 GB/s |
| Linear prefill B1 | 10.517650 ms | 11.086341 ms | 10.7%, 55 GB/s |
| Linear prefill B32 | 275.038348 ms | 275.375946 ms | 5.0%, 26 GB/s |

The small wall/device gaps are dispatch/synchronization and measurement
variance, not host compute. Final full-decode rows verify BF16/HiFi2
DRAM-sharded QKV/O, HiFi4 SDPA, and BFP4/LoFi DRAM-sharded MLP. Dense matmuls
fell from 1.552 to 0.964 ms per replay. Linear B32 now has two rather than
three recurrent matmuls per replay. Under the final BFP8 state, explicit
width 4 reduces the recurrent row mean from 4.261 ms at width 1 and 2.486 ms
at width 2 to 1.717 ms. The final BFP8 state further reduces device time, while
its remaining interleaved recurrence is still dominated by
matmul, untilize, permute, binary, slice, and tilize movement.
Final linear decode rows verify BFP4/LoFi DRAM-sharded packed-input and output
matmuls. The independent profiler controls measured 1.776752/16.146869 ms for
BF16/HiFi2, 1.615684/15.989006 ms for input-only BFP4/LoFi,
1.719812/16.089277 ms for output-only BFP4/LoFi, and
1.562287/15.933102 ms for the precision-only cumulative policy. The geometry
sweep then measured every passing contender with compact profiles; selected
width 5/12 reports 1.521726/15.890707 ms, while the final-default reproduction
reports 1.521271/15.893577 ms. `program_contracts.json` records why output
subblock and worker-grid controls cannot be swept in this DRAM-sharded family:
the Python config has no such fields and the factory internally selects 1x8
and 1x7 for the packed-input and output shapes. Final linear prefill visibly
contains one `BFP8 => FP32` state expansion and one `FP32 => BFP8` writeback
per measured iteration. Together they cost
0.022020 ms/iteration at B1 and 0.685789 ms/iteration at B32; the selected
physical state policy is therefore measured rather than inferred.

## Optimization checklist

- [x] Decoder path traced with no host fallback.
- [x] Decode residual/norm/MLP activations remain width-sharded in L1; the
  QKV-head helper has one proven-required L1-interleaved boundary.
- [x] Prefill uses DRAM-interleaved activations and explicit large 2-D matmuls.
- [x] Operation-topology audit recorded above.
- [x] Per-op L1 layouts and program configs derived and swept for both B1/B32.
- [x] Multi-device/collective topology, fused CCL, and persistent CCL buffers:
  N/A for the required 1x1 stage.
- [x] Best correct candidate and final default compared at B1 and B32.
- [x] Final default performance reproduced after official-weight fixes.
- [x] Runtime rows verify selected dtype, fidelity, and DRAM sharding.
- [x] Explicit paged SDPA and causal prefill SDPA used.
- [x] Q/K/V/gate packing kept; MLP packing tested and rejected with timings.
- [x] Important memory, program, and compute-kernel configs are explicit.
- [x] Decode block widths, prefill block widths/grids, precision, fidelity,
  cache dtype, and packed/split projection forms swept.
- [x] Dominant linear recurrent programs swept at B1/B32 across block widths,
  output subblock, and HiFi2/HiFi4; exact L1/tile blockers retained.
- [x] Persistent linear recurrent state swept independently across
  FP32/BF16/BFP8/BFP4 with physical-dtype checks, synthetic B1/B32 and
  official-weight transitions; BFP8 selected and BFP4 rejected on PCC.
- [x] Packed-input and output projection dtype/fidelity policies are
  independent. BF16/HiFi2, BF16/LoFi, BFP8/HiFi2, BFP8/LoFi, and legal
  BFP4/LoFi candidates were swept at B1/B32; the cumulative BFP4/LoFi winner
  has focused profiler and official-transition evidence.
- [x] Final BFP4/LoFi projection geometry was independently swept at B1/B32:
  packed-input widths 1/4/5/10/20, output widths 1/2/3/4/6/8/12/24,
  cumulative leading crosses, and a four-core storage control. Every passing
  contender has compact profiler evidence; blockers retain exact failure JSON.
- [x] DRAM-sharded output-subblock review completed: the public program config
  exposes no subblock or worker-grid control, factory-selected 1x8/1x7
  subblocks are source-recorded, and the negative API contract is statically
  tested.
- [x] Precision-locked full gate/up larger divisors, MLP fidelity, and a
  coherent smaller storage-grid candidate attempted at both batches.
- [x] Every full cumulative QKV/O/gate/up/down one-role geometry candidate was
  attempted at B1 and B32; feasible timings and exact L1/CB blockers retained.
- [x] Attention BFP4/BFP8/BF16 and MLP gate/up/down BFP4 trials have
  official-weight or focused real-weight evidence.
- [x] Legal DRAM-sharded decode matmuls include B1 `per_core_M=1`.
- [x] Avoidable runtime Torch/device crossings removed; required layout
  boundaries are identified.
- [x] MoE, LM head, sampling, and LM-head DRAM sharding: N/A to decoder layer.
- [x] Roofline, device time, and wall time reconciled from signposted runs.
- [x] B1 primary target wins; B32 correctness and performance preserved.
- [x] Paged cache, non-aligned prefill, trace replay, exact determinism,
  repeated stress, fallback audit, and watcher checks pass.
- [x] Final BFP8-state S=513 plus 16-step transition, repeated B32 transition,
  B32 prefill/decode determinism, watcher stress, and S=192511 capacity pass.
- [x] Final post-AutoFix decode and prefill profiles were regenerated through
  the provenance-retaining wrapper in all eight `final4_*` directories.
- [x] Memory-bounded long prefill crosses the 32K SDPA boundary and completes
  at the advertised S=192511 single-pass limit without reducing the contract.

## Limitations

The linear-attention B32 path remains dominated by the model's exact
gated-delta recurrent composite after removing its K=1 matmul. A future
Tile(4,32)-producing transpose/reshape path or framework-level fused
gated-delta op could eliminate more untilize/permute movement. No model
capability or public sequence-alignment restriction was introduced.
