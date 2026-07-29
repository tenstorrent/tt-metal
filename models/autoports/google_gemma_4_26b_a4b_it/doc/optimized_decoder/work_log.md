# Optimized decoder work log

Date: 2026-07-29 UTC
Base checkout: `dca6a039ed1`

## Operation-topology audit

| Path | Current topology | Candidates | Final action and evidence |
| --- | --- | --- | --- |
| Attention projections | packed QKV; head ops/SDPA; O | advisor 1D, local/DRAM-sharded, BFP8/BFP4 | sliding QKV local w1 and adapted L1-interleaved O applied; full QKV keeps its correct default; advisor QKV w2 rejected at 0.993473 PCC and valid DRAM O rejected at 3.007 ms |
| Dense MLP | two same-input gate/up projections, GELU/mul, down | load-time packing, advisor 1D, DRAM-sharded, reduced precision | gate/up packed once at load and projected once at runtime; dense down BF16/LoFi w3 selected; separate advisor projections superseded |
| Router | norms/static scales, FP32 matmul, TopK/softmax/scatter | fuse static scales, lower precision, composite replacements | static scales fused at load; FP32 router retained for correctness; row/tile movement retained for TopK/scatter → sparse row-major contract |
| Sparse MoE | gate/up/down sparse matmuls, exact `nnz=8` | exact grid, K-block, output placement, BFP8/BFP4 | BFP8; exact 11x2 gate/up grid selected, up w11 at batch 1 and w88 at batch 32; down w1/L1 retained after block/grid/DRAM sweeps regressed |
| Residual stream | O→RMSNorm, dense residual, expert reduction/final norm | persistent L1 width/interleaved chains | L1-width O failed the RMSNorm contract; adapted L1-interleaved O→RMSNorm won; expert reduction feeds final RMSNorm in L1 before DRAM residual updates |
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
| batch-32 expert up w1/w11/w22/w44/w88 | unchanged contract | 34.495/19.598/18.989/18.534/18.471 ms on exact 11x2; portable w1 baseline 26.837 ms | apply w88 |
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
- [x] independent `$stage-review` clean pass
- [x] local stage commit and SHA

Fresh final stage review returned `clean-pass`: no required work and no
hard-check gaps. The reviewer independently checked the current source/test
hashes, advisor artifacts, PCC, trace/stress, batch-32 timing/prefill evidence,
candidate matrix, and all four compact profiler reports.

Local implementation checkpoint: `9c405211e7f` (`Optimize Gemma 4 decoder`).
No push was performed. The compact profiler CSVs and this SHA record are added
in the documentation follow-up checkpoint.

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

Machine-readable sweep results are in `candidate_matrix.json`. Batch-32 expert
up tested exact-grid K blocks 1/11/22/44/88 at
34.495/19.598/18.989/18.534/18.471 ms (portable w1 baseline 26.837 ms) and
selected 88. Batch-32 dense auto/packed/packed-plus-down measured
19.579731/19.561017/19.560304 ms; the packed deltas are within noise, so auto
is retained. Final profiler
rows for both attention kinds are under
`tracy/final_{sliding,full}_attention_batch1/`. Final device totals are
1.343 ms sliding and 1.524 ms full, both with 0 host ops. Final warmed host
results are:

| Layer | Batch | Prefill ms | Decode ms |
| --- | ---: | ---: | ---: |
| sliding | 1 | 670.522 | 1.391 |
| full | 1 | 671.529 | 1.572 |
| sliding | 32 | 21440.032 | 19.580 |
| full | 32 | 21478.793 | 19.392 |

The final policy roles are `qkv_local_w1` (sliding only),
`persistent_o_proj`, `packed_dense`, `dense_down_w3`,
`expert_gate_grid_w11`, `expert_up_grid_11x2`, `expert_up_w11`,
`expert_up_batch32_w88`, and `fused_router_scale`.
