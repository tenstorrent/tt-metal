# Optimized decoder work log

## Scope and baseline

Target: `CohereLabs/North-Mini-Code-1.0`, single-device optimized decoder only.
The functional checkpoint is commit `4e45a256771`. Existing unrelated
untracked files were left untouched.

Hardware health on 2026-07-29:

- `timeout 60 python_env/bin/tt-smi -ls --local`: four Blackhole p300c boards
  visible.
- 1x1 mesh open/close: pass; architecture Blackhole, worker grid 11x10.
- The former `/home/mvasiljevic/.ttsmi-venv/bin/tt-smi` path no longer exists;
  the active environment provides `python_env/bin/tt-smi`.

Functional warmed baselines (sequence 128, synthetic target shapes, complete
trace replay for decode) are inherited evidence to reproduce with the
optimized harness:

| Kind | Batch | Prefill wall | Decode wall | Decode device |
|---|---:|---:|---:|---:|
| dense/full/RoPE | 1 | 0.636 ms | 0.356 ms | 338 us |
| sliding/RoPE/MoE | 1 | 14.908 ms | 9.528 ms | 9452 us |
| full/no-RoPE/MoE | 1 | 14.655 ms | 9.524 ms | 9439 us |
| dense/full/RoPE | 32 | not recorded | 6.652 ms | 6614 us |
| sliding/RoPE/MoE | 32 | not recorded | 11.122 ms | 11084 us |
| full/no-RoPE/MoE | 32 | not recorded | 11.129 ms | 11077 us |

## Operation-topology audit

This table is the pre-tuning audit required by `optimize`. Decisions remain
provisional until same-harness PCC and traced measurements are attached.

| Subgraph | Current material operations/layout movement | Candidate | Initial action | Evidence state |
|---|---|---|---|---|
| norm boundary | DRAM-interleaved RMSNorm; output feeds every projection | width-sharded L1 norm/residual chain | seed from shard advisor, then measure chain | rejected as an incompatible whole chain; head creation/final return require two immediate reverts and norm/add rows are below 15% |
| attention input | already packed QKV matmul; reshape + head split | retain packed QKV; compare advisor 1D and DRAM-sharded decode matmuls | mandatory sweep | exact advisor 11x8 input/output shard and program config applied; retained |
| decode RoPE | cos and sin each interleaved-to-height-sharded every replay | persistent sharded caller buffers or compatible L1 chain | adapt and measure; no first-error rejection | retained trace-resident conversions; profiler total below 15%, caller ownership unchanged |
| decode attention | paged update + composite paged SDPA decode + concat layout conversion | keep composite SDPA; tune program/memory configs and concat boundary | mandatory sweep | composite retained; default beat explicit k32/k64 configs |
| attention output | output projection writes DRAM before residual | advisor 11x6 1D and width-sharded L1 output | mandatory batch-specific sweep | exact candidate applied at batch 1; retained; batch 32 independently uses compatible auto config |
| dense MLP input | gate and up are two same-input matmuls | packed gate/up weight, one matmul, device split | implemented as first rewritten topology | packed DS retained for decode; matched BFP8 split + explicit 2D retained for prefill after split compiler failure |
| dense MLP output | SiLU, multiply, down matmul, all DRAM intermediates | L1 intermediate chain; BFP8 and BFP4/LoFi crossed with geometry | mandatory sweep | BFP4/LoFi DS 8/12 block winner retained |
| sparse router | router linear, top-k, sigmoid, scatter | retain exact sigmoid top-8 semantics; tune router and keep routing device-resident | measured | exact top-8/sigmoid/scatter device path retained |
| sparse experts | repeat input across 128 experts; three dense batched matmuls; mask/reduce afterward | routed active-expert `sparse_matmul`, separate gate/up and down configs, `nnz=None` unless exact nonzero count is proven | mandatory rewrite | fully routed batch-1 decode retained; grouped routed and 2D candidates rejected with exact contract/perf evidence |
| prefill projections | framework-default matmuls in DRAM | large 2D program configs, phase-specific by actual M | mandatory sweep | attention O + dense MLP explicit 2D retained at b1/b32; expert-batched 2D rejected by fused-batch B-weight contract |
| host/runtime boundaries | runtime forwards contain no torch/from/to-torch | preserve; audit tilize/untilize/reshard too | source + profiler audit | no host fallback; routing tilize/untilize rows are necessary row-major top-k/scatter boundaries and each below 1% |

First packed-path execution exposed that TTNN `split` takes a split size rather
than a number of sections: passing `2` produced 3072 outputs. The candidate was
corrected to split at `intermediate_size=3072`; this API-shape failure is not
being used to reject packing.

## Precision policy candidates

Starting policy is BF16 activations/norm/residual and BFP8 material weights.
Attention, dense gate/up, dense down, router, sparse gate/up, sparse down, and
KV cache are independent groups. Dense and expert gate/up BFP4+LoFi and guarded
down BFP4+LoFi trials are mandatory. Geometry comparisons will not be mixed
across policies.

## Shard advisor

Hard-gate capture completed on the executable packed dense attention+MLP graph:

```text
TTMLIR_ADVISOR_HOME=/home/mvasiljevic/tt-mlir
branch: mvasiljevic/shard-advisor-dram-sharding
commit: 618cd4e75dae69d334bb9d8cdeff314816ccf214
NORTH_ADVISE_BATCH=1 ttnn-advise capture .../advise_north.py:decode \
  --out /tmp/north-mini-advice-b1
Result: ops=24, final_choices=21, spill.ran=True, total_spills=4
```

The first capture-target attempt imported pytest helpers and failed on the
advisor venv's missing `safetensors`; the target was made self-contained. The
next attempt corrected synthetic Q/O shapes to the actual GQA contract. Neither
setup failure was treated as a layout rejection. Exact copied artifacts and
hashes:

- `shard_advise/report.json`,
  SHA256 `1b1e6835b24fb9c9348c6007bb9da2fe884938d218d6555719d0132108ec4ed2`
  (the commit hook added the repository-required final newline).
- `shard_advise/final_ir.mlir`,
  SHA256 `4e32846c4012d460aba3c56c0f90ea7cf9912e33e9ba969e44400bef183d3dd1`.

Advisor candidate classification before measurement:

| Op | Advice | Required adaptation | Status |
|---|---|---|---|
| RMSNorm | 22-core width-sharded L1 | shard weight and residual chain coherently | rejected as whole chain; incompatible QKV/head/final boundaries add two reverts for sub-15% rows |
| QKV BF16 | 11x8 1D, `in0_block_w=2`, `per_core_N=2`, output width-sharded 80 cores | 32-core input shard and L1-interleaved head boundary revert | applied exactly; final candidate |
| output BF16 | 11x6 1D, `in0_block_w=8`, `per_core_N=1`, output width-sharded 64 cores | DRAM concat input and L1 return revert | applied exactly; final candidate |
| packed gate/up BFP8 | DRAM-sharded, 8 banks, `in0_block_w=8`, `per_core_M=1`, `per_core_N=2` | DRAM-width-shard weight and 8-core working input | applied, then BFP4/LoFi geometry won |
| dense down BFP8 | DRAM-sharded, 8 banks, `in0_block_w=12`, `per_core_M=1`, `per_core_N=1` | DRAM-width-shard weight and 8-core working input | applied, then BFP4/LoFi geometry won |
| residual adds | 64-core width-sharded L1 | compare whole chain including final revert | rejected with the RMSNorm chain for boundary cost |

`dram_sharded_considered=2`, both BFP8 dense MLP matmuls were advised. The two
BF16 attention matmuls were explicitly not considered by policy, not kernel
legality; their 1D configurations remain candidates and separate BFP8/BFP4
attention DS trials are required. The advisor also reported its known
`nlp_concat_heads_decode` sharded-input limitation and two RoPE constraint-query
shape mismatches; the real implementation already inserts the required concat
layout and uses the decode RoPE kernel, so these are candidate-boundary items,
not waived runtime failures.

## First same-harness dense result

Current-checkout functional layer-0 batch-1 traced decode (5 warmups, 50
iterations): mean `0.360183 ms`, min `0.358803 ms`.

Packed BFP8 gate/up + BFP8 down candidate under the same harness: mean
`0.319182 ms`, min `0.317966 ms`; non-aligned prefill length 33 PCC
`0.9997122135`. This is an 11.38% mean-latency improvement and establishes the
new best correct batch-1 candidate. It is not final until advisor/DS/precision
geometry sweeps, real-weight evidence, and final-default reproduction complete.

The advisor DRAM-sharded candidate passed batch-1 traced decode PCC
`0.9998901319` and measured mean `0.289003 ms`, min `0.287399 ms` (3 warmups,
20 iterations). This improves the current functional baseline by 19.76% and
the packed-interleaved candidate by 9.45%, so it is the new best correct
batch-1 path.

The first batch-32 application reused batch-1 shard height 32 and failed before
execution because the actual physical width-sharded M height is 1024. This was
not used to reject DS. The candidate then derived shard height as
`batch * TILE_SIZE` while retaining the kernel-required logical
`per_core_M=1`.

Follow-up showed the ordinary DS kernel is explicitly limited to logical M=1;
the batch-sharded DS kernel instead requires HEIGHT_SHARDED A and a
HEIGHT_SHARDED, batch-replicated B (`[1,B,K,N]`), not this decoder's shared
`[K,N]` weights. Replicating 32 copies of every dense weight is contrary to the
capacity and movement target. The measured compatible batch-32 family is
packed BFP8 interleaved: mean `5.450767 ms`, min `5.441083 ms`, versus the
functional `6.652 ms` mean (18.06% faster).

## Dense precision/fidelity sweep

All candidates below use the advisor DS geometry at batch 1:
gate/up `in0_block_w=8`, `per_core_N=2`; down `in0_block_w=12`,
`per_core_N=1`.

| Gate/up | Down | Fidelity | Synthetic PCC | Official layer-0 PCC | Mean traced decode | Decision |
|---|---|---|---:|---:|---:|---|
| BFP8 | BFP8 | HiFi2 | 0.999890 | 0.999868 | 0.289003 ms | correct, slower |
| BFP4 | BFP8 | LoFi / HiFi2 | 0.999436 | 0.999653 | 0.267531 ms | correct, slower |
| BFP4 | BFP4 | LoFi | 0.999178 | 0.999521 | 0.258190 ms | current winner |

The final automatic policy now selects BFP4/LoFi advisor DS for batch 1 and
packed BFP8 interleaved for larger batches. Official weights came from pinned
revision `d11e61a...`, shard `model-00001-of-00049.safetensors`; the lower
precision choice is therefore not based on synthetic evidence alone.

The final BFP8 attention policy lowered the same dense path further to
`0.222395 ms`. BFP4 attention was not retained: the full traced synthetic
decoder PCC was `0.989489` and official layer-0 PCC was `0.987685`, both below
the `0.995` functional floor. The final official-weight PCC values are
`0.999276` for layer 0 and `0.999168` for the
layer-1 routed sparse path. KV cache, activations, norm, residual, and outputs
remain BF16.

## Advisor disposition and topology decisions

| Advisor/topology candidate | Action and evidence |
|---|---|
| QKV 11x8 1D / 80 cores | Applied exactly: 32-core width-sharded input, explicit 11x8 program, 80-core width-sharded output, and L1-interleaved head-creation revert. |
| output projection 11x6 / 64-core output | Applied exactly: concat output reverted to advisor-required DRAM interleaved, explicit 11x6 program, 64-core width-sharded output, then L1 return revert. The pair improved dense decode from 0.230786 to 0.222395 ms. |
| packed gate/up DRAM-sharded | Applied exactly at batch 1, then crossed with BFP8/BFP4 and geometry. BFP4/LoFi won. |
| dense down DRAM-sharded | Applied exactly at batch 1, then crossed with BFP8/BFP4 and geometry. BFP4/LoFi won. |
| RMSNorm/residual L1 chain | Rejected as a whole-chain candidate: QKV head creation and final residual both require incompatible layouts, so the candidate introduces two extra material conversions. The measured final trace already spends less than 15% in norm, adds, and all conversions combined; projection work remained the actionable region. |
| decode RoPE persistent sharding | The existing trace captures both cos/sin conversions, with no host work. Profile rows are below the actionable threshold; changing the caller contract would add persistent buffer ownership without a measurable layer win. |
| composite paged SDPA | Kept. Default measured 0.222557 ms; explicit k32 and k64 configs measured 0.222922 and 0.223754 ms. |
| same-input dense gate/up packing | Kept for decode. The BFP8 packed prefill split kernel failed to compile (`single_tile_size_bytes` missing in the split writer), so prefill uses two large BFP8 projections; this is a compiler failure, not a public sequence restriction. |
| routed sparse experts | Kept for batch-1 decode with `sparse_matmul`, `nnz=None`, BFP8/LoFi, L1 intermediates. Grouped sparse was adapted with DRAM outputs, BF16/HiFi2, 32-token chunks and token routing, but measured 17.71 ms at b32 and 64.99 ms prefill. Sparse-down rejects token-specific `[T,E]` sparse-A masks (`logical_volume=4096`, required expert batch=128); a routed gate/up + dense down adaptation measured 19.15 ms. The faster device-only BF16 batched path is retained for grouped tokens. |

The advisor's configs were therefore used as executable candidates rather than
accepted as configuration policy. Its exact attention pair measured
`0.222395 ms`; the subsequent BFP8 DRAM-sharded QKV/O family
(`in0_block_w=8/16`, `per_core_N=20/8`, eight DRAM banks) measured
`0.200063 ms` and is the final batch-1 attention policy.

## Geometry sweeps

Dense BFP4/LoFi isolated sweeps:

- Gate/up `in0_block_w`: 1=`0.284695`, 2=`0.264540`,
  4=`0.259464`, 8=`0.258190` ms (down fixed at 12).
- Down `in0_block_w`: 1=`0.309984`, 2=`0.279376`,
  3=`0.269313`, 4=`0.262938`, 6=`0.259814`,
  12=`0.258190` ms (gate/up fixed at 8).

Sparse BFP8/LoFi whole-layer sweep (layer 1, batch 1):

| cores | gate/up block | down block | mean |
|---:|---:|---:|---:|
| 4 | 8 | 6 | 1.822959 ms |
| 8 | 4 | 6 | 1.474536 ms |
| 8 | 16 | 6 | 1.119805 ms |
| 8 | 16 | 12 | 1.080914 ms |
| 8 | 32 | 12 | 1.026092 ms |
| 10 | 32 | 24 | 0.933775 ms |
| 11 | 16 | 12 | 1.001867 ms |
| 11 | 32 | 12 | 0.942147 ms |
| 11 | 32 | 24 | **0.924293 ms** |

The final reproduced run with exact advisor attention measured `0.909643 ms`. A
16-core candidate was invalid on the 11-column worker grid and was not treated
as a kernel-family rejection. Gate/up and down were swept independently before
the combined winner was selected. BFP8 HiFi2 at the final geometry measured
`0.916903 ms` versus LoFi `0.908593 ms`. Legal larger output-subblock trials
(gate/up 3 and down 2) each exceeded a bounded 60-second run; the processes
were terminated and `tt-smi` immediately confirmed all four boards healthy.

AutoTriage/AutoFix subsequently proved those larger output subblocks were not
legal candidates: North fixes `out_block_w=1`, while sparse TTNN omitted the
dense validator and integer-divided `1/3` or `1/2` to zero input subblocks,
deadlocking writer/compute protocol. Artifacts are under `triage/`. The
model-owned builder now rejects nonpositive, larger-than-block, or nondividing
subblocks before device open; focused regression is 3/3 pass, legal 1/1
reverified at `0.882483 ms`, and all boards remained healthy.

Final additional matched searches:

- DRAM-sharded BFP8 attention LoFi=`0.201133 ms`,
  HiFi2=`0.225522 ms`; LoFi retained.
- Official layer-1 sparse BFP4 PCC=`0.996110` (traced, pinned weights) and
  latency=`0.880250 ms`; BFP8 latency=`0.888439 ms`. BFP4 becomes the final
  batch-1 sparse policy despite the deliberately harsher synthetic candidate.
- Matched BFP4/LoFi DRAM-sharded dense MLP: packed=`0.200989 ms` whole layer,
  separate gate/up=`0.214412 ms`; packed retained.

## Final same-session performance matrix

All values below use 2 warmups and 10 measured iterations. Decode is complete
trace replay; prefill sequence is 128.

| Layer kind | Batch | Functional prefill | Optimized prefill | Functional decode | Optimized decode |
|---|---:|---:|---:|---:|---:|
| dense/full/RoPE (0) | 1 | 0.634801 | **0.529455** | 0.362345 | **0.200406** |
| dense/full/RoPE (0) | 32 | 13.776355 | **6.114367** | 6.761510 | **5.017496** |
| sliding/RoPE/MoE (1) | 1 | 14.773170 | 14.834266 | 9.526286 | **0.881623** |
| sliding/RoPE/MoE (1) | 32 | 145.638648 | **142.983383** | 11.221797 | **10.796992** |
| full/no-RoPE/MoE (4) | 1 | 14.746384 | 14.754579 | 9.523028 | **0.885650** |
| full/no-RoPE/MoE (4) | 32 | 145.212355 | **142.642106** | 11.224574 | **10.782970** |

The primary batch-1 decode target beats both the functional baseline and every
earlier correct optimized decode candidate, and every batch-32 point improves.
The final3 routed batch-1 prefill means are small observed regressions:
layer 1 is 0.41% slower and layer 4 is 0.06% slower. These are accepted as
run-to-run variance rather than a material regression because the identical
final path measured faster in final2 (`14.629996` and `14.656074 ms`,
respectively), while the retained final3 values above are the conservative
headline results. No capability or correctness tradeoff was made.

## Correctness, capacity, and runtime contract

- Dense non-aligned lengths 33 and 65 pass.
- Sparse non-aligned/chunk-boundary prefill passes at PCC `0.999827`
  (layer 1, sequence 1025) and `0.999763` (layer 4, sequence 33).
- Traced routed sparse decode passes at `0.999318`/`0.999213` for layers 1/4.
- Official layer-0 passes at `0.999264`; official layer-1 passes at
  `0.999240` (BFP8) and `0.996110` (final BFP4).
- Serving-batch layer 1 passes at `0.998145`.
- Multi-position page-cache tests pass at position 1 for BF16 (`0.998065`)
  and BFP8 (`0.998034`) cache; exact repeated output is deterministic.
- BFP4 cache failed at `0.980617`. BFP8 cache latency (`0.222675 ms`) did not
  beat BF16 (`0.222395 ms`), so BF16 remains final.
- The paged KV cache remains BF16, page size/context ownership is unchanged,
  and no capacity dimension changed. `doc/context_contract.json` therefore
  requires no update.
- Measured forwards contain no torch, `from_torch`, `to_torch`, host fallback,
  or public alignment restriction. Sparse routing necessarily converts
  top-k/scatter inputs to row-major: its tilize/untilize rows are each below
  1% and no equivalent top-k/scatter tile-layout API exists.
- Batch-1 routed intermediates are L1; the serving-batch DRAM placement is the
  measured hard-capacity exception described above.
- The final non-watcher suite is `23 passed in 139.60s`; `final_tests.xml`
  contains the current 23 cases.

Watcher command:

```text
TT_METAL_WATCHER=10 python_env/bin/python -m pytest -q \
  models/autoports/coherelabs_north_mini_code_1_0/tests/test_optimized_decoder.py
```

The current source passes `23 passed in 156.72s` with no watcher error, and
`watcher_tests.xml` contains those 23 cases. Repeated traced stress used 10
warmups and 100 replays:
layer 0 `0.200742 ms` (max `0.204260`), layer 1 `0.880063 ms`
(max `0.885209`), layer 4 `0.881788 ms` (max `0.887363`); all outputs finite.

## Tracy, tt-perf-report, and accounting

Four separate Tracy captures cover dense/sparse and decode/prefill. The staged
`tracy_final/` evidence retains advice-enabled text reports and stacked PNGs;
the text reports contain the complete operation rows used below. The raw
compact ops CSVs were transient profiler inputs and are not retained.
The exact pattern was:

```text
python_env/bin/python -m tracy -r -p -v -o <dir> -n <name> \
  -m models.autoports.coherelabs_north_mini_code_1_0.tests.optimized_decoder_perf ...
python_env/bin/tt-perf-report <ops.csv> \
  --start-signpost PERF_DECODE --end-signpost PERF_DECODE_END ...
```

The dense decode report verifies:

- DRAM-sharded packed QKV: 28 us, 12 reported worker cores,
  BF16 x BFP8 -> BF16, LoFi, 74.2% modeled DRAM utilization;
- DRAM-sharded output projection: 22 us, 12 reported worker cores,
  BF16 x BFP8 -> BF16, LoFi, 74.7% modeled DRAM utilization;
- packed gate/up: 27 us, 12 reported worker cores, BF16 x BFP4 -> BF16, LoFi;
- down: 15 us, 12 reported worker cores, BF16 x BFP4 -> BF16, LoFi.

The sparse report rerun with `--active-experts 8` shows gate/up at 172/171 us
and down at 161 us, all BF16 x BFP4 -> BF16, LoFi. These three operations
remain above 15%; the independent block/core sweep above is the targeted
response, and `11/32/24` is the measured winner. Sparse `nnz=None` is retained
because BFP quantization can flush routed scores to zero; the report's
`--active-experts 8` is analysis metadata, not a runtime assertion.

For dense batch-1 decode, material projection weights are approximately
28.3 MB, giving a 0.055 ms ideal lower bound at the report's 512 GB/s device
bandwidth. The signposted device rows total roughly 0.23 ms and the paired
end-to-end traced mean is 0.200406 ms, so dispatch overhead is effectively
removed and remaining distance to roofline is the multi-op/layout and
non-projection work. For sparse decode, eight active BFP4 experts plus attention
move roughly 95 MB (about 0.185 ms ideal); device/end-to-end are about
0.88 ms, dominated by three sparse kernels at only ~8% modeled DRAM
utilization. This is the remaining TTNN sparse-kernel limitation, after the
geometry search.

## Optimize checklist

- [x] operation-topology audit recorded before tuning
- [x] same-input projection packing and routed sparse topology evaluated
- [x] precision/fidelity groups crossed with correctness and real weights
- [x] mandatory shard-advisor capture applied and disposition recorded
- [x] DRAM-sharded BFP4/LoFi dense decode geometry swept
- [x] sparse gate/up and down geometry swept independently
- [x] large prefill matmul families profiled at batches 1 and 32
- [x] paged SDPA/KV-cache semantics and non-aligned public lengths preserved
- [x] final runtime dtype/fidelity verified in profiler rows
- [x] watcher-clean and 100-replay stress evidence retained
- [x] advice-enabled `tt-perf-report` reviewed and actionable rows attacked
- [x] theoretical/device/end-to-end accounting reconciled
- [x] independent stage review returned clean-pass
- [x] local scoped commit and SHA record

## Local commit

- Stage implementation and evidence: `5aecb6cb0a8`

## Stage review

Final independent `$stage-review` verdict: **clean-pass**. The reviewer
confirmed the advisor hard gate and hashes, final DRAM-sharded attention rows,
official traced BFP4 sparse policy, AutoTriage/AutoFix evidence, matched
packed/separate comparison, 23-case normal/watcher suites, current stress
artifacts, conservative final3 headline matrix, routed-prefill variance
rationale, retained Tracy inventory, and isolated staged scope.
