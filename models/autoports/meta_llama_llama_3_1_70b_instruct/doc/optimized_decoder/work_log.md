# Optimized decoder work log

## Scope, starting point, and hardware

- Model: `meta-llama/Llama-3.1-70B-Instruct`
- Autoport: `models/autoports/meta_llama_llama_3_1_70b_instruct`
- Representative layer: dense layer 39. The model has one meaningful decoder-layer kind.
- Hardware: one Blackhole p300c from two visible local ASICs; device work was serialized through `/tmp/tt-device.lock` or `run_safe_pytest.sh`.
- Functional checkpoint: `0d020f9ea7221dc89ae32d205421e6bdf4bc5493`.
- Scope: `tt/optimized_decoder.py`, `tests/test_optimized_decoder.py`, and `doc/optimized_decoder/` only. No multichip decoder, full model, generator, LM head, sampling, or vLLM code was begun.

Health checks on 2026-07-16:

```text
timeout 60 tt-smi -ls --local       -> two Blackhole p300c ASICs visible
open_mesh_device(MeshShape(1,1))    -> MESH_SMOKE_OK
```

The functional watcher gate was rerun with real layer-39 weights and passed 3/3 tests. Baseline PCC:

| Case | PCC |
| --- | ---: |
| synthetic prefill seq 4 / 18 | 0.997370 / 0.996538 |
| synthetic decode pos 18 | 0.995945 |
| real prefill seq 18 | 0.9999927 |
| real decode pos 18 | 0.9999942 |
| real decode K / V append | 0.9998497 / 0.9998579 |

## Operation-topology audit

The audit and `$graph-rewrite` pass preceded program-config tuning. Repo exploration covered TTNN transformer, normalization, matmul, cache, and elementwise ops plus `models/common/modules/{attention,mlp,rmsnorm}` and TT-Transformers config patterns.

| Region | Functional/current sequence | Candidate or issue | Action | Evidence/decision |
| --- | --- | --- | --- | --- |
| input/norm/residual | DRAM/interleaved residual and default RMSNorm | keep decode residual/norm in width-sharded L1 | implemented 32-core sharded norm/residual; advisor-exact 86/11-core chain also measured | simple advisor chain 2.454 ms; exact residual chain 2.545 ms, so extra block/width reshards rejected |
| QKV | one packed QKV matmul, then three slices and three reshapes | dedicated head-creation op | replaced manual split topology with `nlp_create_qkv_heads_decode` | real PCC and trace pass; profiler shows the selected BFP4/LoFi QKV reached hardware |
| RoPE/cache | two rotary ops and two paged cache updates | fuse K/V update | adapted head-aligned, nonoverlapping QKV split was tested | default overlap failed validator; adapted 64-core output then triggered a watcher NoC sanitizer fault in `reader_tm_tile_layout_nlp_create_qkv_heads_decode.cpp`; devices reset/recovered, split paged updates retained |
| attention | additive mask, SDPA decode, concat heads | composite SDPA and explicit SDPA configs | retained `scaled_dot_product_attention[_decode]`; tested implicit and explicit 8x8/10x10/11x10 configs | explicit program config 2.602 ms on DRAM family and 2.285 ms on all-BFP4 advisor family, both slower than corresponding implicit controls |
| O projection | concat, layout conversion, O matmul, residual add | L1 input and sharded 1-D O output | applied advisor 11x8, `in0_block_w=2`, `per_core_N=3`, subblock 1x3 | retained; remaining concat/SDPA conversions are required by those op contracts |
| MLP | separate same-input gate/up, SiLU, multiply, down | packed gate/up; fuse SiLU into multiply | folded SiLU into `ttnn.mul(...SILU...)`; built full-phase and decode-only packed variants | full-phase packing needs 2,263,296 B CB vs 1,572,864 B L1; three decode-only geometries pass watcher correctness, but the best is 2.1220 ms vs 2.1201 ms split over 500 replays |
| prefill projections | default matmuls | explicit large 2-D configs and lower precision | 11x10, fuse-batch, phase-specific interleaved weights, explicit per-role blocks | final 7.969 ms; 10x10 final-family prefill is slower, and DRAM-sharded prefill weights silently corrupt when worker-N and eight-bank shards disagree |
| host/runtime boundaries | no host conversion in functional forward | independent optimized forward without fallback | source audit asserts optimized method ownership and forbids `torch`, `from_torch`, `to_torch`, or functional `super()` calls | pass; setup-time weight conversion only, none in measured forwards |

The final traced decode report contains about 43 device ops per replay, versus about 42 for the functional path. The win is therefore not an op-count claim: it comes from low-precision, high-utilization matmuls and coherent L1 layouts. All remaining reshards/layout changes sit at incompatible contracts (height-sharded head ops, DRAM SDPA/cache, 1-D projection output, or the public output). The more aggressive exact advisor residual chain was the targeted lower-movement comparison and lost by 3.7%.

## Mandatory shard-advisor gate

The advisor was run this pass after the topology rewrite. Setup followed `.agents/skills/shard-advise/SETUP.md` Part B in a separate shell: `TTMLIR_ADVISOR_HOME=/home/mvasiljevic/tt-mlir`, change into that checkout before sourcing `scripts/bootstrap.sh`, return to this checkout, then invoke capture.

```bash
(
  export TTMLIR_ADVISOR_HOME=/home/mvasiljevic/tt-mlir
  cd "$TTMLIR_ADVISOR_HOME"
  source scripts/bootstrap.sh
  cd /home/mvasiljevic/tt-metal
  ttnn-advise capture \
    models/autoports/meta_llama_llama_3_1_70b_instruct/doc/optimized_decoder/shard_advise/advise_llama70b.py \
    --output-dir models/autoports/meta_llama_llama_3_1_70b_instruct/doc/optimized_decoder/shard_advise
)
```

Result: `ops=28`, `final_choices=25`, spill pass ran with one spill, and one unfixable `nlp_concat_heads_decode` query because the advisor left SDPA output DRAM while concat-heads requires sharded input. Required artifacts:

- `shard_advise/report.json`
- `shard_advise/final_ir.mlir`

The 45 MB auxiliary search decision trace was pruned after the two required artifacts and `report.txt` were validated; `report.json` records that retention decision instead of carrying a stale path.

Recommendations and dispositions:

| Advisor recommendation | Disposition | Evidence |
| --- | --- | --- |
| QKV 1-D matmul 11x10, `in0=2`, `per_core_N=3`, subblock 1x3, 107-core width-sharded output | applied | advisor simple real PCC 0.9999794 prefill / 0.9999808 decode; faster than DRAM family |
| O 1-D matmul 11x8, `in0=2`, `per_core_N=3`, subblock 1x3, 86-core output | applied | retained in the 2.114 ms final traced decode |
| gate/up 1-D matmuls 11x10, `in0=2`, `per_core_N=9`, subblock 1x3, 100-core outputs | applied | final-family block/grid sweep rejected block 4/8/16 and grids 11x9/11x8; profiler reports ~427/426 us |
| down 1-D matmul 11x8, `in0=2`, `per_core_N=3`, subblock 1x3, 86-core output | applied as seed, then tuned | real-weight sweep selected 11x6, `in0=4`, `per_core_N=4`, subblock 1x4, 64-core output; 2.1140 vs 2.1203 ms over 500 replays; profiler down row falls to ~423 us |
| block-sharded 11-core norms plus exact 86-core residual chain | tested, rejected | correct at 0.9999794/0.9999797 but 9.926 ms prefill / 2.545 ms decode vs simple advisor 9.761/2.454 ms |
| advisor L1 mask layouts and DRAM SDPA output | partially rejected | composite SDPA contract remains DRAM; explicit SDPA configs were slower; concat requires a targeted sharding conversion |
| advisor split gate/up | kept | full-phase packed adaptation exceeds L1 by 690,432 B per affected core; decode-only adaptation is correct but loses the 500-replay latency comparison |

The advisor tracer did not model fused cache update, so the capture uses the final split-update graph. The capture script now freezes the exact saved-IR precision policy (BFP8 attention/down, BFP4 gate/up, down `in0=2`) rather than inheriting the later live default. The advisor IR is the geometry seed; precision and final down geometry were swept afterward on hardware. A bounded external TTMLIR tracer handler was needed for `where`; no external toolchain file is part of this repo commit.

## AutoFix and correctness recovery

A fresh-context AutoDebug report is saved in `AUTODEBUG.md`. It identified the catastrophic prefill result (PCC 0.000324 with infinities) as a storage/program mismatch, not ordinary BFP4 loss: eight-bank DRAM-width-sharded weights were consumed by an 11-column 2-D program with a different `per_core_N`. In-tree common modules explicitly warn this can silently produce bad PCC.

The proven fix keeps DRAM-interleaved prefill weight copies for the DRAM-sharded experiment family while decode weights remain sharded. The final advisor topology uses shared interleaved weights and therefore avoids duplicate persistent copies. Further isolated L1 fixes set the DRAM-family gate/up `in0_block_w=4` (8 required 1,921,792 B) and down `in0_block_w=7` (auto 14 required 1,696,512 B). The repaired DRAM default reached 0.9999796 prefill / 0.9999811 decode before advisor selection.

## Candidate sweeps

All correctness decisions use real weights. Timing rows use batch 32 and seq 18. Intermediate rows use three warmed prefill repetitions and 50 traced replays; final and close-call adjudication rows use five prefill repetitions and 500 traced replays.

### Cumulative topology and precision

| Candidate | Prefill ms | Decode ms | Accuracy/resource result | Decision |
| --- | ---: | ---: | --- | --- |
| functional BF16 | 143.845 | 142.906 | functional bar | before baseline |
| repaired 32-core DRAM-sharded decode; BFP8 attention/down, BFP4/LoFi gate/up | 9.817 | 2.600 | 0.9999796 / 0.9999811 | correct baseline, slower |
| advisor 1-D, same precision | 9.761 | 2.454 | 0.9999794 / 0.9999808 | strongest correct pre-final baseline |
| advisor exact residual/layout chain | 9.926 | 2.545 | 0.9999794 / 0.9999797 | reject extra reshards |
| advisor + BFP4/LoFi down, BFP8 attention | 8.692 | 2.360 | prefill/decode 0.9999731/0.9999745; K/V 0.9998454/0.9998538 | correct pre-final baseline |
| advisor + BFP4/HiFi2 down | 9.808 | 2.360 | real PCC passes | reject: same decode, slower prefill |
| advisor all BFP4/LoFi, seed down geometry | 7.954 | 2.281 | output 0.9999021/0.9999540; K/V 0.992796/0.993537; eight recurrent decode steps all pass 0.99 | correct precision checkpoint |
| all BFP4/LoFi, down 11x8/block 4 | 7.979 | 2.120 | same PCC; 500 replay adjudication | correct immediate baseline |
| final all BFP4/LoFi, down 11x6/block 4 | 7.969 | 2.114 | eight-step watcher result matches the control; 500 replay adjudication passes | **selected final: fastest correct BF16-cache policy** |
| final + BFP8 cache | 8.025 | 2.079 | decode PCC 0.9999542; BFP8 K/V append 0.9928525/0.9935404 | accepted faster batch-32 throughput policy; loses the primary batch-1 comparison and cache dtype remains caller-owned |

Attention was isolated from MLP: BFP8 attention LoFi is 2.450 ms; BFP4 attention LoFi/HiFi2 is 2.374/2.375 ms. Gate/up BFP4 HiFi2 is 2.463 ms; BFP8 HiFi2 is 2.987 ms. Down BFP8 LoFi is 2.435 ms; down BFP4 LoFi/HiFi2 is 2.360/2.360 ms. BFP4 attention causes the material cache PCC delta above, so it was not promoted until an eight-step recurrent real-weight check showed every output and K/V append still met the 0.99 functional contract. With that evidence, all projection roles use BFP4/LoFi because it is the fastest correct policy.

### Program, geometry, composite, and packing candidates

| Family | Values tried | Result/decision |
| --- | --- | --- |
| DRAM core geometry | coherent 32, 16, 8-core QKV/O/gate/up/down families, repeated with the then-selected BFP8-attention/BFP4-MLP precision | both 16- and 8-core candidates reach prefill PCC 0.9999731, then the decode program needs 2,347,776 B CB vs 1,572,864 B L1; 32 retained for DRAM controls |
| DRAM QKV/O block | auto 8, 4, 2 | 2.600, 2.608, ~2.630 ms neighborhood; auto retained in DRAM control |
| DRAM gate/up block | 8, 4, 2, 1 | 8 fails at 1,921,792 B; 4 control; 2 is 2.654 ms, 1 is 3.233 ms |
| DRAM down block | auto 14, 7, 4, 2, 1 | 14 fails at 1,696,512 B; 7 control; 4/2/1 are 2.632/2.708/2.867 ms |
| advisor gate/up subblock | 3, 1 | 3 retained; 1 slower |
| final gate/up block/grid | blocks 2/4/8/16; grids 11x10/11x9/11x8 | advisor block 2 and 11x10 retained; alternatives measure 2.123--2.451 ms versus the final-family control |
| final down block | 2, 4, 7, 8, 14, 16 | block 4 retained; 2 is 2.281 ms and 7--16 are 2.135--2.171 ms |
| final down grid | 11x8/perN3/subblock3, 11x6/perN4/subblock4, 11x5/perN5/subblock1 | 11x6 wins at 2.1140 ms over 500 replays vs 2.1203 for 11x8; 11x5 is 2.169 ms |
| prefill grid | 11x10, 10x10; earlier 8x10 | final-family 11x10 wins; 10x10 all-BFP4 8.381 ms vs 7.954; 8x10 gate CB 2,192,128 B |
| SDPA config | implicit, explicit 8x8/10x10/11x10, explicit compute kernel | implicit fastest; all retain composite SDPA semantics |
| packed gate/up | full-phase 10/11-column programs, then decode-only 100-core blocks 2/1 and 106-core block 2 | full phase fails exact 2,263,296 B CB limit; all decode-only variants pass watcher PCC, best packed is 2.1220 ms vs split 2.1201 over 500 replays, so split kept |
| cache update | split, default-overlap fused, adapted nonoverlap fused | fused validator failure followed by adapted NoC-sanitizer failure; split kept |

The independent review's initial evidence findings were addressed before the
final rereview:

- the advisor-family gate/up and down program geometries were swept on the
  final all-BFP4 topology, including a 500-replay 11x8-versus-11x6 down-grid
  adjudication;
- BF16 and BFP8 cache policies were both rerun on the final default. Cache
  allocation is caller-owned, so BF16 compatibility remains while BFP8 is the
  documented faster option rather than an implicit contract change;
- full-phase packing received a retained hard-resource failure, then three
  adapted decode-only packed programs received watcher correctness and timing;
- `candidate_evidence/*.xml` retains machine-readable pass/failure output and
  numeric JUnit properties, indexed by `candidate_evidence/README.md`.

## Final correctness and contract evidence

| Case | Functional PCC | Final PCC | Material delta |
| --- | ---: | ---: | --- |
| real prefill seq 18 | 0.9999927 | 0.9999023 | -0.0000904; all-BFP4 projection quantization, above 0.99 acceptance bar |
| real decode pos 18 | 0.9999942 | 0.9999545 | -0.0000397; same policy |
| K append | 0.9998497 | 0.9929778 | -0.0068719; material BFP4-attention delta, above 0.99 bar |
| V append | 0.9998579 | 0.9933865 | -0.0064714; material BFP4-attention delta, above 0.99 bar |
| trace replay | n/a | 0.9999546 | five bitwise-equal replays; K/V 0.9927492/0.9934069 |
| refreshed trace | n/a | 0.9999543 | input refresh changes output and updates K/V correctly |
| nonaligned seq=7, batch=13 | n/a | 0.9998395 / 0.9999273 | three bitwise-equal prefill/decode repetitions |
| batch=1 decode/K/V | n/a | 0.9996235 / 0.9924039 / 0.9941090 | primary latency shape preserved |

The static audit proves both forwards and MLP helpers are owned by `optimized_decoder.py`, not a functional fallback. It also asserts two `paged_update_cache` calls and composite decode SDPA are present. There is no `torch`, `from_torch`, `to_torch`, explicit untilize/tilize, or host fallback inside measured forwards. Tracy does expose small tilize/untilize operations required inside the paged-cache and SDPA input contracts; they are not host boundaries, and the attempted fused-cache rewrite failed only after validator adaptation and a watcher-sanitized hardware trial.

The eight-step recurrent all-BFP4 test covered positions 18 through 25. Output PCC rose from 0.9999540 to 0.9999599; K stayed approximately 0.99280--0.99303 and V approximately 0.99334--0.99354. Thus the material cache delta does not accumulate below the public 0.99 bar over the tested recurrence.

`doc/context_contract.json` is unchanged: the public tensor/cache contract, maximum validated seq 18, batch 32, and cache length 128 are unchanged. Logical seq 7 is accepted without a public alignment restriction. The final advisor topology does not add the phase-duplicated weights used only by the DRAM experiment, so no hard capacity reduction exists.

## Performance and profiler accounting

Primary single-user reproduction:

| Workload | Warmed prefill | Traced warmed decode |
| --- | ---: | ---: |
| functional, batch 1, BF16 cache, 5/500 repetitions | 5.059 ms | 4.917 ms |
| final optimized, batch 1, BF16 cache, 5/500 repetitions | 3.179 ms | 1.846 ms |
| final optimized, batch 1, BFP8 cache, 5/500 repetitions | 3.204 ms | 1.846 ms |

The final path improves the primary batch-1 target by 1.59x in prefill and 2.66x in traced decode. This is the cumulative contract comparison: final all-BFP4/LoFi projections, the 11x6 down grid, batch 1, logical sequence 18, unchanged cache layout/page contract, and the same 5/500 timing harness. BF16 is selected for the primary cache policy because BFP8 is 0.78% slower in prefill and 0.015% slower in decode at batch 1. The decoder consumes caller-allocated cache tensors, so BFP8 remains supported and is the faster batch-32 throughput policy rather than an implicit API default.

Batch-32 scaling reproduction:

| Workload | Warmed prefill | Traced warmed decode |
| --- | ---: | ---: |
| batch 32, BF16 cache, 5/500 repetitions | 7.969 ms | 2.114 ms |
| batch 32, BFP8 cache, 5/500 repetitions | 8.025 ms | 2.079 ms |

The final batch-32 path is 18.1x faster in prefill and 67.6x faster in traced decode than functional. It beats the BFP8-attention/BFP4-MLP 2.360 ms baseline by 10.4%, the first correct all-BFP4 2.281 ms checkpoint by 7.3%, and the immediately preceding correct 11x8/block-4 2.1203 ms topology by 0.30%.

Reduced-layer Tracy and advice-enabled `tt-perf-report` were collected separately from watcher runs. Summed final device time is 7.514 ms prefill and 10.413 ms for five decode replays, or 2.083 ms/replay. Compared with the canonical 2.114 ms traced host latency, the roughly 0.031 ms gap is trace launch/synchronization accounting, not a host tensor fallback.

Roofline conclusions:

- functional prefill/decode: 2.7% modeled DRAM roofline, 14 GB/s;
- final prefill: 18.9%, 97 GB/s;
- final decode: 40.1%, 205 GB/s;
- final decode device categories: 80.66% compute, 14.04% other, 2.84% data movement, 2.45% tensor manipulation;
- dominant final rows verify the policy reached hardware: QKV/O/gate/up/down are all `LoFi BF16 x BFP4 => BF16`;
- per replay QKV/O are about 194/169 us, gate/up about 427/426 us, and the tuned down projection about 423 us. MLP remains dominant, but the down row fell from roughly 589 us at the advisor seed. The report recommends HiFi for accuracy and DRAM-sharded attention; both were explicitly swept. HiFi loses prefill/perf, while the coherent DRAM-sharded family is slower than advisor 1-D. Prefill's L1-input suggestion is not kept because the 2-D DRAM-interleaved path preserves arbitrary logical sequence padding and is already the measured winner.

Artifacts are under `tracy/`; raw `ops_perf_results_*.csv.gz` files are retained with lossless compression to satisfy the repository's 500 KB file limit, alongside the processed advice/report CSVs.

## Validation commands

Real-weight path and opt-in candidate/perf tests use:

```bash
export LLAMA_31_70B_REAL_WEIGHT_FILE=/home/mvasiljevic/hf-cache/hub/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b/model-00015-of-00030.safetensors

python -m pytest \
  models/autoports/meta_llama_llama_3_1_70b_instruct/tests/test_optimized_decoder.py -q -s

RUN_OPTIMIZED_DECODER_CANDIDATES=1 \
OPTIMIZED_DECODER_CANDIDATE_VARIANT=optimized \
OPTIMIZED_DECODER_CANDIDATE_DECODE_STEPS=8 \
python -m pytest \
  models/autoports/meta_llama_llama_3_1_70b_instruct/tests/test_optimized_decoder.py \
  -k candidate_executes_real_optimized_path -q -s

RUN_OPTIMIZED_DECODER_PERF=1 \
OPTIMIZED_DECODER_TRACE_REPLAYS=50 \
OPTIMIZED_DECODER_PREFILL_REPEATS=3 \
python -m pytest \
  models/autoports/meta_llama_llama_3_1_70b_instruct/tests/test_optimized_decoder.py \
  -k optimized_decoder_perf -q -s

RUN_OPTIMIZED_DECODER_PERF=1 \
OPTIMIZED_DECODER_BATCH=1 \
OPTIMIZED_DECODER_TRACE_REPLAYS=500 \
OPTIMIZED_DECODER_PREFILL_REPEATS=5 \
OPTIMIZED_DECODER_PERF_VARIANT=optimized \
python -m pytest \
  models/autoports/meta_llama_llama_3_1_70b_instruct/tests/test_optimized_decoder.py \
  -k optimized_decoder_perf -q -s
```

Exact final watcher gate (asserts enabled, separate from profiler):

```bash
flock /tmp/tt-device.lock env \
  TT_METAL_WATCHER=10 \
  TT_METAL_OPERATION_TIMEOUT_SECONDS=5 \
  LLAMA_31_70B_REAL_WEIGHT_FILE="$LLAMA_31_70B_REAL_WEIGHT_FILE" \
  python -m pytest \
    models/autoports/meta_llama_llama_3_1_70b_instruct/tests/test_optimized_decoder.py -q -s
```

Result: 6 passed, 2 expected opt-in skips in 62.94 s; watcher initialized with `disabled features: None`; no watcher error/sanitizer/assert entry. The runner report and watcher output are preserved as `watcher_test_report.xml` and `watcher.log`.

Profiler command shape:

```bash
RUN_OPTIMIZED_DECODER_PERF=1 OPTIMIZED_DECODER_PERF_VARIANT=optimized \
OPTIMIZED_DECODER_TRACE_REPLAYS=5 OPTIMIZED_DECODER_PREFILL_REPEATS=1 \
python -m tracy -r -p -v --check-exit-code \
  -o models/autoports/meta_llama_llama_3_1_70b_instruct/doc/optimized_decoder/tracy/final \
  -n final -m pytest \
  models/autoports/meta_llama_llama_3_1_70b_instruct/tests/test_optimized_decoder.py \
  -k optimized_decoder_perf -q -s
```

`uv pip install tt-perf-report` confirmed the package in `python_env`; reports were run without `--no-advice`, delimited by `PERF_PREFILL[_END]` and `PERF_DECODE[_END]`.

## `$optimize` checklist

- [x] Functional checks pass on the independent optimized path.
- [x] Prefill/decode and paged-cache PCC remain at the functional acceptance bar; deltas are quantified above.
- [x] Decode is fully traced with no host fallback; five replays and input refresh pass.
- [x] Decode activations are generally L1 sharded across norm, projection, residual, and MLP boundaries; head ops use required height sharding and SDPA/cache use required DRAM layouts.
- [x] Prefill uses DRAM-interleaved activations and explicit 2-D matmul configs.
- [x] Operation-topology audit completed before knob tuning.
- [x] `$graph-rewrite` applied: dedicated QKV-head creation, composite SDPA, and fused SiLU/multiply; each kept rewrite has real PCC evidence.
- [x] `$shard-advise` run this pass; required artifacts exist; all material recommendations have applied/rejected evidence.
- [x] Lower-movement exact advisor residual chain measured coherently and rejected on traced latency.
- [x] Strongest baseline and material candidates compared; final default reproduces the selected correct winner.
- [x] Primary batch-1 functional/final and BF16/BFP8-cache comparisons reproduce with the final 11x6 code under the 5/500 harness.
- [x] Final dtype/fidelity is verified in profiler runtime rows.
- [x] QKV is packed; adapted gate/up packing and fused cache-update candidates have exact post-adaptation resource/runtime blockers.
- [x] Important memory, program, and compute-kernel configs are explicit.
- [x] Dominant roles have separate core/block/subblock/memory/fidelity/dtype sweeps with real-weight and traced evidence.
- [x] Attention and MLP precision were swept separately; BFP4 attention was promoted only after real K/V PCC and an eight-step recurrent decode established contract compliance.
- [x] MLP gate/up/down BFP4/LoFi trials completed and selected.
- [x] Legal tile-dividing shard/core grids are enforced; valid final-family grids were swept and the measured 11x6 down winner is used.
- [x] DRAM-sharded decode matmuls were implemented and swept across 32/16/8-core families and per-role blocks; advisor 1-D interleaved weights win and are final.
- [x] Avoidable layout changes were removed or justified by the exact-chain comparison and incompatible op contracts.
- [x] Stress/repeated-run and batch-1/batch-13/batch-32 coverage pass.
- [x] Advice-enabled `tt-perf-report`, roofline, device-time, and end-to-end accounting are recorded.
- [x] Exact `TT_METAL_WATCHER=10` run is clean and separate from profiler collection.
- [x] Multi-device collectives, fused matmul-CCL, persistent CCL buffers: not applicable to this single-device decoder-layer scope.
- [x] MoE sparse expert path: not applicable; this is a dense layer.
- [x] LM head/sampling/token feedback: not applicable; explicitly out of decoder-layer scope.
- [x] vLLM serving and full-model qualitative gates: not applicable and not started.

## Review and commits

Independent `$stage-review`: **clean-pass**. The final rereview found no required work after the precision-locked geometry sweep, cache-policy adjudication, decode-only packed gate/up comparison, machine-readable candidate evidence, advisor-provenance freeze, and batch-1 primary-target reproduction. The full verdict and anomaly ledger are retained in `STAGE_REVIEW.md`.

Stage implementation and evidence commit: `35ccb90250cab6c538413f239245212335840c63` (`Optimize Llama 3.1 70B decoder`). This work-log SHA update is a follow-up local metadata commit. No push was performed.
