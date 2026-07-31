# Optimized decoder work log

Original tagged optimization: 2026-07-29 UTC at `9c405211e7f`

Current fused-derived integration: 2026-07-31 UTC, starting HEAD
`9d3247ee51c`. The tagged topology was ported onto the current completed
`FusedDecoder`, then fully revalidated and reprofiled. The first review of the
slower preliminary implementation returned `more-work-needed`; its findings
triggered discovery and integration of the tagged best-correct topology.

## Operation-topology audit

| Path | Current topology | Candidates | Final action and evidence |
| --- | --- | --- | --- |
| Attention projections | packed QKV; head ops/SDPA; O | advisor 1D, local/DRAM-sharded, BFP8/BFP4 | sliding QKV local w1 and adapted L1-interleaved O applied; full QKV keeps its correct default; advisor QKV w2 rejected at 0.993473 PCC and valid DRAM O rejected at 3.007 ms |
| Dense MLP | two same-input gate/up projections, GELU/mul, down | load-time packing, advisor 1D, DRAM-sharded, reduced precision | gate/up packed once at load and projected once at runtime; dense down BF16/LoFi w3 selected; separate advisor projections superseded |
| Router | norms/static scales, FP32 matmul, TopK/softmax/scatter | fuse static scales, lower precision, composite replacements | static scales fused at load; FP32 router retained for correctness; row/tile movement retained for TopK/scatter → sparse row-major contract |
| Sparse MoE | gate/up/down sparse matmuls, exact `nnz=8` | exact grid, K-block, output/input placement, BFP8/BFP4, gate fidelity | BFP8; exact 11x2 gate/up grid with w11 per serialized routing row; down w1/L1; gate HiFi4 retained over sub-noise LoFi candidate |
| Residual stream | O→RMSNorm, dense residual, expert reduction/final norm | persistent L1 width/interleaved and exact-width sharded norms | L1-interleaved O retained; 88-core HiFi4/FP32 sharded hidden norms ran but failed PCC, including the isolated post-attention norm |
| Host boundaries | no host ops in forwards | fallback and trace audits | retained; final signposted report has 0 host ops |

## Required shard-advisor gate

The advisor was run this pass after rewriting decode attention to expose its
layout contract. Setup was performed in a separate shell:

```bash
export TTMLIR_ADVISOR_HOME=/home/mvasiljevic/tt-mlir
source .agents/skills/shard-advise/scripts/bootstrap.sh
ttnn-advise capture \
  --output-dir models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/shard_advise \
  models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/shard_advise/advise_gemma4.py:decode
```

Advisor checkout: `618cd4e75dae69d334bb9d8cdeff314816ccf214`.
Capture scope: batch-1 sliding layer 0, dense attention plus dense MLP, real
model shapes. Sparse matmul is not representable in the captured TTIR.

```text
[ttnn-advise] ops=22 final_choices=19 spill.ran=True total_spills=2
```

Required artifacts:

- `report.json`, SHA-256
  `1adacb14d5b0354cb2c4d016b1d993caeb7b46bbd452ce7a9ab90976a2466e4f`
- `final_ir.mlir`, SHA-256
  `518fc043632a2f16944a6d34ad304ac713c97dc291657bad2ecca85a44ac535c`

Recommendations and disposition:

- QKV 11x8, `in0_block_w=2`, `per_core_N=3`: rejected, decode PCC
  0.993473 < 0.995.
- O 11x8, `in0_block_w=8`, `per_core_N=1`: applied.
- dense gate/up 11x6, `in0_block_w=8`, `per_core_N=1`: validated as
  advisor seed, then superseded by the faster correct packed projection.
- dense down 11x8, `in0_block_w=2`, `per_core_N=1`: validated as advisor
  seed, then superseded by the final BF16/LoFi `in0_block_w=3` sweep winner.
- L1 width outputs and IR reverts: applied at head split and residual
  boundaries. The persistent fully-sharded chain was rejected after the IR's
  required reverts proved necessary.

## Candidate ledger

| Candidate | Correctness | Performance | Decision |
| --- | --- | --- | --- |
| functional | sliding 0.998617/0.999655; full 0.997773/0.999861 | sliding decode 3.038 ms | baseline |
| advisor QKV | decode 0.993473 | completed after explicit revert | reject |
| advisor O | decode 0.999626 | improves selected path | apply |
| advisor gate/up | decode 0.999654 | 31/31 us | apply |
| advisor down | decode 0.999654 | 45 us | apply |
| advisor O+dense | decode 0.999564 | 2.962 ms | apply |
| sparse all-wide | decode 0.999655 | gate/up/down 558/557/234 us | reject up/down |
| sparse gate w1/2/4/8/11 | unchanged | host 2.407/2.148/2.033/1.964/1.956 ms | apply w11 |
| expert BFP8 | final PCC table in README | prefill -1.6%; batch-32 decode -50% | apply |
| expert BFP4 | sliding 0.996457/0.997452; full 0.995313/0.996917 | prefill 659.752 ms | reject: weak margin |
| dense BFP8 | 32-token pass; boundary 0.994220 | faster candidate | reject |
| dense BFP4 | 0.886020/0.929458 | no further timing | reject |
| DRAM-sharded O | valid after explicit input/output shards | 3.007 ms | reject |
| packed dense | decode 0.999605 | 1.653410 ms whole decode | apply |
| persistent O, L1-width | RMSNorm contract error | no valid timing | adapt, not reject |
| persistent O, L1-interleaved | passing | 1.646468 ms whole decode | apply |
| fused router static scales | sliding decode 0.999521 | 1.617621 ms candidate | apply |
| expert up w11 | unchanged final bar | 1.713520 ms vs 1.885219 ms w1 whole decode | apply |
| expert down w1/L1 | unchanged final bar | 1.885219 ms; all tested block/DRAM alternatives slower | apply |
| expert up portable 2-core → exact 11x2 | selected-grid PCC unchanged | 1.621431 → 1.391426 ms batch-1 decode | apply exact 11x2 |
| expert BFP4 on selected 11x2 grid | 0.996457 prefill / 0.997280 decode | 1.391919 ms vs BFP8 1.393212 ms | reject: negligible speedup, inadequate accuracy margin |
| batch-32 expert up w1/w11/w22/w44/w88 | invalid experiment: each live sparse call has M=1 after routing-row serialization | reported 34.495/19.598/18.989/18.534/18.471 ms did not exercise the named M=32 branches | invalidate; live exact-grid w11 per row |
| batch-32 dense TTNN auto | passing | 19.579731 ms | apply |
| batch-32 packed dense | passing | 19.561017 ms | reject: within noise |
| batch-32 packed+dense-down | passing | 19.560304 ms | reject: within noise |
| seq-256 prefill packed dense auto/w8/w11 | passing | 168.636/171.481/171.703 ms | retain auto; both 2D candidates regress |
| final retained artifacts | sliding 0.998634/0.999521; full 0.998006/0.999824 | sliding/full 1.391/1.572 ms batch-1; 19.580/19.392 ms batch-32 | apply |

Rows above the final retained-artifacts row are candidate-time observations,
not the final policy. In particular, the earlier 1.885/2.051 ms “final”
result and the 0.997771 full-prefill value are superseded by the rerun
artifacts named in the final row.

The first DRAM-sharded attempt failed at an unspecified generic shard config;
the second exposed a non-divisible 11-core input width. The third used an
explicit 8-core input and 11-core output and ran successfully. The candidate
is rejected by that successful measurement, not by its initial API errors.

## Validation commands and results

All runtime runs used `GEMMA4_RANGE_DOWNLOAD=1` and
`TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}'`.

```bash
python -m pytest -q \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py::test_optimized_real_weights_prefill_decode
# 3 passed

python -m pytest -q \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py::test_optimized_paged_prefill_logical_boundaries
# full passed; sliding exposed dense-BFP8 failure, then passed after rejection

GEMMA4_FUNCTIONAL_DECODER_CONTEXT=1 python -m pytest -q \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py::test_optimized_advertised_context
# 2 passed

python -m pytest -q \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py::test_optimized_traced_decode_batch_contract
# 4 passed: batch 1/32 x sliding/full, including repeat replay

GEMMA4_FUNCTIONAL_DECODER_PERF=1 python -m pytest -q \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py::test_optimized_perf_profile
# 4 passed

TT_METAL_WATCHER=1 python -m pytest -q \
  'models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py::test_optimized_real_weights_prefill_decode[blackhole-sliding_attention-device_params0-mesh_device0]'
# passed; no watcher assert/error
```

Final decode Tracy used the decode-only harness. The retained reports contain
80 device ops and 0 host ops: sliding totals 1.343 ms device time; full totals
1.524 ms. The earlier 1.570/1.751 ms decode reports are superseded by the exact
11x2 expert-up grid rerun.

Prefill Tracy initially used seq-1024, which overflowed the device profiler.
That failure was adapted rather than treated as a program rejection: bounded
seq-256 retries for sliding and full attention produced valid 172-op compact
reports with 168.120/168.469 ms summed device time and 168.262/168.761 ms
warmed host time. The focused large-prefill dense sweep retained TTNN auto
(168.636 ms); 11x8 2D packed configs at `in0_block_w=8/11` regressed to
171.481/171.703 ms.

### Final conversion and residual ledger

| Conversion family | Sliding | Full | Disposition |
| --- | ---: | ---: | --- |
| sharded/interleaved | 12 ops, 10.248 us | 9 ops, 9.905 us | retained for QKV/head, rotary/cache/SDPA, concat, packed slice, and normalization contracts |
| untilize/tilize | 5 ops, 12.687 us | 5 ops, 12.792 us | retained at TopK/scatter and sparse row-major routing boundary |
| typecast | 3 ops, 6.002 us | 3 ops, 6.064 us | retained for FP32 router and BF16 sparse inputs |
| device copy | 4 ops, 6.966 us | 3 ops, 5.812 us | retained to materialize selected consumer layouts; no host transfer |

Final rows 344–351 show expert output reshape/reduction, L1 RMSNorm, then
DRAM residual/norm operations. There is no layout conversion between expert
reduction and final RMSNorm. These are device contract boundaries, not
`torch`/`from_torch`/`to_torch`, host fallback, or unexplained reshards.

## Hardware recovery and AutoFix

The first broad advisor seed hit `TT_FATAL: Sharded inputs require sharded
outputs` and wedged PCIe access. Triage was attempted before intervention.
Only the exact stale pytest was terminated; bounded `tt-smi` reset recovered
all four P300s and a 1x1 mesh smoke printed `MESH_SMOKE_OK`.

The next whole-layer retry hung. `$autofix` produced `AUTOTRIAGE.md`; role
isolation found the missing QKV output revert from `final_ir.mlir`. Adding that
revert removed the hang. Advisor roles were then accepted or rejected
individually. The installed triage reader's incompatible `noc_read` signature
is recorded under `triage/` and was not treated as model evidence.

## Optimize checklist

- [x] operation topology audited before edits
- [x] profiler bottleneck identified and final `tt-perf-report` retained
- [x] batch-1 and batch-32 warmed/trace measurements recorded
- [x] shard-advisor hard gate and both artifacts complete
- [x] precision/fidelity: BF16, BFP8, BFP4 and LoFi/HiFi paths evaluated
- [x] sharded layouts and DRAM-sharded matmul evaluated beyond first error
- [x] repeated same-input projections and packing opportunity evaluated
- [x] SDPA/composite and movement opportunities evaluated
- [x] no host fallback or runtime torch conversion
- [x] real-weight PCC across layer kinds and cache kinds
- [x] non-aligned logical lengths and advertised context
- [x] traced repeat determinism at batch 1 and 32
- [x] watcher-clean correctness run
- [x] raw profiler dumps removed; compact evidence retained
- [x] independent `$stage-review` clean pass after current remediation
- [x] local stage commit and SHA

The fresh independent remediation rereview returned `clean-pass` with no
required work, material concerns, or acceptance-affecting hard-check gaps.

## First stage-review remediation

The first independent review returned `more-work-needed`. Every required-work
item was treated as implementation or measurement work:

1. Sparse up/down were swept independently under final BFP8. Up tested K
   blocks 1/2/4/8/11/22/44 and selected 11, reducing its device row from
   553 us to 317 us. Down tested 1/2/11/22, DRAM output, and the 88-core grid;
   block 1/L1/8-core remained fastest at 164 us. For both projections
   `per_core_N=11`; TTNN subblocks must divide it and have area <=4, making
   width 1 the only legal output subblock.
   A subsequent exact-grid sweep replaced the portable 2-core expert-up
   program with 11x2: 4x6 and 8x3 were blocked by 22 active cores versus 24
   receivers, while exact 11x2 passed PCC and reduced whole decode from
   1.621431 to 1.391426 ms. BFP4 on that same grid was only 0.001293 ms faster
   and reduced PCC to 0.996457/0.997280, so BFP8 remains selected.
2. Dense gate/up was packed at load time and split on device. It passed
   real-weight PCC and beat tuned separate projections. A persistent L1-width
   O output failed the next RMSNorm contract; the adapted L1-interleaved
   O→RMSNorm path ran and won. The two static router input scales were fused at
   load time and passed PCC.
3. Dense down tested K blocks 2/3/6/11/22/33 under BF16/LoFi and selected 3.
   Sliding QKV tested 1/2/4/8/11; block 1 passed PCC and won. Full-attention
   QKV retains its local correct default because its 10,240-wide output needs
   107 cores under the sliding advisor geometry, exceeding the 88-core grid.
4. Real batch-32 seq-1024 prefill was added and measured:
   sliding functional/optimized 21793.675/21440.032 ms; full
   21831.195/21478.793 ms.
5. A three-repetition trace stress now covers batch 1/32 x sliding/full. All
   four cases pass, with repeat-replay PCC 1.0 in every repetition. The final
   watcher run covers sliding, full-natural, and full-shared HMA; all pass with
   no watcher assert/error.
6. Timing artifacts now replace stale functional provenance with optimized
   source/test hashes and commands. The advisor report points to retained
   stage-relative artifacts instead of `/tmp`.

Machine-readable sweep results are in `candidate_matrix.json`. The earlier
batch-32 expert-up K-block values were invalidated during rereview: the sparse
expert contract serializes independent routing masks and dynamically calls the
optimized single-row implementation, so an `M == 32` branch was unreachable.
The dead role was removed and a unit test now pins dynamic dispatch to the
optimized sparse kernel. Batch-32 dense auto/packed/packed-plus-down measured
19.579731/19.561017/19.560304 ms; the packed deltas are within noise, so auto
is retained. Final profiler
rows for both attention kinds are under
`tracy/final_{sliding,full}_attention_batch1/`. Final device totals are
1.343 ms sliding and 1.524 ms full, both with 0 host ops. Final warmed host
results are:

| Layer | Batch | Prefill ms | Decode ms |
| --- | ---: | ---: | ---: |
| sliding | 1 | 671.330 | 1.369 |
| full | 1 | 672.472 | 1.539 |
| sliding | 32 | 21471.650 | 19.540 |
| full | 32 | 21509.164 | 19.369 |

The final policy roles are `qkv_local_w1` (sliding only),
`persistent_o_proj`, `packed_dense`, `dense_down_w3`,
`expert_gate_grid_w11`, `expert_up_grid_11x2`, `expert_up_w11`, and
`fused_router_scale`.

## Current fused-port review remediation

The 2026-07-31 independent review rejected the preliminary 2.9--3.0 ms path.
It identified the unclassified tagged baseline, dominant untuned sparse rows,
incomplete per-role geometry and attention precision, stale evidence, and an
unmeasured residual/layout dismissal. Work performed:

1. Inspected tag `9c405211e7f`, restored its stage-owned topology/evidence, and
   adapted `OptimizedDecoder` to inherit the current `FusedDecoder`. An
   explicit `_moe_decode` override preserves the selected sparse single-user
   implementation instead of the fused stage's deliberate functional bypass.
2. Reproduced the port on current hardware. Real-weight PCC passes for
   sliding, full-natural, and full-shared HMA. The current fused port improves
   the tag to 1.369/1.539 ms batch 1 and 19.540/19.369 ms batch 32.
3. Added attention-only dtype/fidelity controls. BFP4/LoFi fails real weights;
   BF16/LoFi passes but is mixed within noise versus the runtime-selected
   fidelity. The selected profiler rows prove sliding QKV/O LoFi and full QKV
   HiFi2/O LoFi under BF16.
4. Retested BFP4 experts under the exact selected grid. Ordinary layer tests
   pass and latency improves, so it was temporarily promoted. The mandatory
   non-aligned sliding length-31 run then failed at PCC 0.994344; BFP8 was
   restored for an exact model-visible reason. `candidate_matrix.json` records
   the corrected decision.
5. Added bounded-tail and mutable stable-buffer wrappers. Refreshed ten-length
   non-aligned boundaries, advertised position 262143, four three-repetition
   trace stresses, real PCC, batch contracts, and nine watcher cases on the
   frozen source. `final_watcher.log` has no error/assert/hang match.
6. Recollected current-source decode and bounded seq-256 prefill Tracy windows.
   `tt-perf-report` now shows exact 22-core BFP8 sparse gate/up, K block 11,
   and the retained 8-core down K block 1; sparse projections no longer carry
   the review's untuned two-core/K-block-1 gate/up defect.
7. Reran current batch-32 prefill controls and the default suite. The JUnit
   files retain exact case counts, failures, skips, and timings.

Representative final commands:

```bash
GEMMA4_RANGE_DOWNLOAD=1 GEMMA4_FUNCTIONAL_DECODER_PERF=1 GEMMA4_DECODER_PERF_REPEATS=50 pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py -k optimized_perf_profile
GEMMA4_RANGE_DOWNLOAD=1 GEMMA4_FUNCTIONAL_DECODER_CONTEXT=1 pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py -k 'optimized_real_weights_prefill_decode or optimized_paged_prefill_logical_boundaries or optimized_advertised_context or optimized_bounded_tail_cache_integrity or optimized_trace_mutable_stable_buffers or optimized_traced_decode_batch_contract'
GEMMA4_RANGE_DOWNLOAD=1 GEMMA4_OPTIMIZED_STRESS=1 pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py -k optimized_repeated_trace_stress
GEMMA4_RANGE_DOWNLOAD=1 GEMMA4_OPTIMIZED_BATCH32_PREFILL=1 pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py -k batch32_prefill_profile
GEMMA4_RANGE_DOWNLOAD=1 TT_METAL_WATCHER=10 pytest -q --junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/final_watcher_tests.xml models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py -k 'optimized_real_weights_prefill_decode or optimized_traced_decode_batch_contract or optimized_trace_mutable_stable_buffers'
GEMMA4_RANGE_DOWNLOAD=1 pytest -q --junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/final_suite.xml models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py
```

Profiler commands use `python -m tracy -r -p` with the dedicated
`test_optimized_{decode,prefill}_only_profile` cases. Prefill uses
`GEMMA4_FUNCTIONAL_DECODER_SEQ_LEN=256` after the documented seq-1024 device
profiler overflow. Compact current reports are in
`tracy/current_fused_final/`.

## Second stage-review remediation

This section is a historical checkpoint. Its final-policy numbers were
superseded by the AutoFix remediation immediately below.

The next independent rereview found two live policy-wiring defects and three
evidence gaps. The repaired source/test hashes are
`65322c9b39b3ec45c05c7a25582f07ffa9eb01eb007e74a1c5740e04a604cec8` and
`e517cd0079303c4deb0a48bd76df90c9a4a027d1e42506436ae793abae246f97`.

1. The optimized attention override had replaced the fused stage's cache
   update selection with two independent updates. Native, non-modulo batch-1
   and sliding batch-32 decode now use `paged_fused_update_cache`; full
   attention batch 32 retains the measured separate-update winner. Bounded and
   modulo cache geometry still uses separate updates to preserve logical-tail
   semantics. Current Tracy proves one fused row for both batch-1 layer kinds.
2. The advertised `expert_up_b32_w88` branch was dead because the expert batch
   dimension represents 128 experts, while decode users carry independent
   sparsity rows and must be serialized. The role and claims were removed;
   live batch 32 dynamically dispatches the optimized exact-grid w11 kernel for
   every device-sliced row. `test_moe_batch_orchestration_dispatches_to_optimized_sparse_kernel`
   pins this contract.
3. An 88-core width-sharded RMSNorm was implemented and run rather than rejected
   at its initial prefill-layout error. With HiFi4/FP32 accumulation, all
   hidden-width decode norms produced sliding/full PCC 0.993390/0.990832; an
   isolated post-attention norm produced 0.993870 sliding. Both fail 0.995.
4. Gate fidelity was screened on the selected exact 11x2 geometry. LoFi passes
   at sliding/full decode PCC 0.999507/0.999806, but versus HiFi4 its 20-replay
   delta is below 0.001 ms at batch 1 and roughly 0.04/0.03 ms at batch 32,
   while accuracy margin declines. Gate HiFi4 is retained; up/down are LoFi.
5. Every frozen-source artifact was regenerated: 14/14 combined correctness,
   4/4 stress, 4/4 50-replay performance, 4/4 batch-32 prefill, 9/9 watcher,
   and the default suite (18 pass, 18 explicit opt-in skips). Current device
   totals are 1.323/1.500 ms decode and 168.335/168.909 ms seq-256 prefill.
   Modeled DRAM roofline values are 23.2%/25.1% decode and 8.0% prefill.

## AutoFix remediation and frozen result

The subsequent review found three remaining issues. Fresh-context AutoDebug
traced seq-256 prefill's 96.5% sparse-matmul share to the inherited portable
prefill helper, confirmed that QKV/packed/down had no per-role DRAM candidate
surface, and found that dedicated full-attention profiles used shared cache
geometry. `AUTOFIX.md` retains the diagnosis.

The optimized source now owns an exact-grid sparse-prefill tile path. At
seq-256, the frozen-source portable baseline was 168.958800 ms. Gate w11/L1
was 108.494765 ms, gate+up was 48.633498 ms, and gate+up+down w11/L1 was
32.093236 ms. Chunk64 regressed to 54.814527 ms. Chunk128 first hit an L1
capacity validation; the adapted DRAM version ran correctly at about 103 ms,
so the option is rejected by measured latency.

Per-role DRAM experiments covered QKV, packed dense, and dense down at batch 1
and 32. Global QKV policies exposed real accuracy asymmetry: w1 fails full at
0.983679 and w2 fails sliding at 0.993311. The selected layer-specific w1/w2
policy passes both and improves both. Batch-32 QKV w2 fails full at 0.984152;
packed-dense w4 passes and wins; dense-down w3 is correct but mixed/sub-noise.
Batch-1 packed widths 8/4 and down widths 3/6 regress.

Frozen 50-replay baseline -> final medians are 1.369657 -> 1.349989 ms
sliding batch 1, 1.538101 -> 1.502324 ms full batch 1, 19.531627 ->
19.473742 ms sliding batch 32, and 19.374607 -> 19.330319 ms full batch 32.
Seq-1024 batch-1 prefill is 125.210932/126.288810 ms; real batch-32 prefill is
4003.866818/4039.915158 ms versus 21779.879575/21817.087552 ms functional.

Natural-cache Tracy plus `tt-perf-report` gives device+gap totals of
1.382908/1.535627 ms decode and 32.078922/32.280459 ms seq-256 prefill.
Same-run host values are 1.408000/1.560049 and 32.203006/32.400070 ms.
Modeled DRAM roofline is 23.5%/25.6% decode and 40.5%/40.2% prefill. The full
profiles record non-shared natural cache shapes; invocation-counter assertions
prove the optimized entry points ran.

Frozen source/test SHA256:
`7cbcbd4f775db2ed932690b9ff3c7a76e8f3628d637e747d3f059caaf804ff11` /
`34adb78536d4e6cb94519dc8ca0a88ca24cb98df1c23285de445fea00cf1c364`.

Final gate results: `final_correctness.xml` 14/14, `final_stress.xml` 4/4,
`final_perf.xml` 4/4, `final_batch32_prefill.xml` 4/4,
`final_watcher_tests.xml` 9/9, and `final_suite.xml` 23 pass/18 opt-in skip.
The watcher log has no error/assert/hang match and all four P300c boards remain
discoverable after profiling and watcher runs.

Correct Tracy commands use the required opt-in environment, for example:

```bash
GEMMA4_RANGE_DOWNLOAD=1 GEMMA4_OPTIMIZED_DECODE_PROFILE=1 \
  python -m tracy -r -p -m pytest -q '<exact decode profiler node id>'
GEMMA4_RANGE_DOWNLOAD=1 GEMMA4_OPTIMIZED_PREFILL_PROFILE=1 \
  GEMMA4_FUNCTIONAL_DECODER_SEQ_LEN=256 \
  python -m tracy -r -p -m pytest -q '<exact prefill profiler node id>'
```

The frozen optimize checklist is complete except for the local stage commit.

## Final-review AutoFix remediation

The independent review returned `more-work-needed` for packed sparse
gate/up, layer-specific batch-32 QKV, immutable final performance JSON, and
profiler accounting. All four findings were treated as work:

1. Implemented an opt-in packed all-expert prefill projection from the HF
   fused `[1,128,2816,1408]` weight. One exact 11x4/w11 sparse matmul replaces
   the same-input 704-wide gate and up operations, followed by tile-aligned
   slices and unchanged GeGLU/down math. Seq-256 improved from 32.093236 to
   21.490080 ms; real PCC, non-aligned boundaries, stress, watcher, and context
   gates pass. The packed path is selected.
2. Added batch-32 layer-kind QKV roles. Sliding QKV w2 plus packed dense w4
   passes trace PCC 0.999451 and measures 19.450342 ms over 50 candidate
   replays. The final default measures 19.444872 ms. Full attention does not
   select QKV w2 because its retained PCC is 0.984152; packed-only full is
   19.342580 ms. A current-source 50-replay control without layer-specific
   QKV records 19.477006/19.343033 ms.
3. Regenerated the final default performance last, after every candidate run.
   Unsuffixed JSONs now contain 50 samples, frozen hashes, prefill
   82.472863/83.621333 ms, batch-1 decode 1.351698/1.503703 ms, and batch-32
   decode 19.444872/19.342580 ms.
4. Added `profiler_accounting.md`. It reports modeled bytes, the explicit
   `bytes / 512 GB/s` theoretical times, same-run host/device/gaps, readable
   per-op report filenames, and required tilize/untilize counts and costs.

Final frozen source/test hashes are
`608da0656b1d4f0b8c3b3c812b032cfdcb6cd99631a32f1f3bb7cfa58a53a747` /
`cc62897949aba36ec7019313ed81372bbff514d0cee3a4ca8322336a8267a5e6`.
Frozen gates were regenerated: correctness 14/14, stress 4/4, performance
4/4, batch-32 prefill 4/4, watcher 9/9, and default suite 23 pass/18 opt-in
skip. A fresh independent rereview inspected the frozen code, tests, PCC,
performance, profiler accounting, boundary/context, stress, watcher, and JUnit
artifacts and returned `clean-pass` with no required work or material concerns.

## Local checkpoint

The optimized decoder implementation, tests, compact profiler reports, and
frozen evidence were committed locally as `e7a39a9e19d` (`Add optimized Gemma
4 26B decoder`) on top of starting checkpoint `9d3247ee51c`. Nothing was
pushed. The four generated raw Tracy CSV captures exceed the repository's
500-KB artifact policy and remain local/ignored; their compact per-op reports,
stacked summaries, plots, and reconciled accounting are included in the
checkpoint.
