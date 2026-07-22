# GPT-OSS 20B optimized decoder work log

Date: 2026-07-22 UTC

## Scope, provenance, and device discipline

This stage starts from `tt/fused_decoder.py` at
`0da25aafeac04e2ff21cb39c4687565dc28f7f98`. It changes only the model's
optimized decoder, tests, and documentation. Multichip, full-model, and vLLM
work are out of scope.

The initial stage-owned commit is `d3b34a9bf87` (`Optimize GPT-OSS 20B
decoder`). The independent review of that commit returned more-work-needed;
the remediation and rereview are recorded below. Nothing was pushed.

`tt-smi -s` showed four healthy local Blackhole P300c devices, DRAM status
true, and no uncorrectable GDDR errors. Hardware commands were serialized.
Profiler and watcher were never enabled in the same process.

## Operation-topology audit

The retained fused profiler reports were audited before local tuning.

| Region | Starting topology and cost | Candidate/action | Result |
| --- | --- | --- | --- |
| input norm -> QKV -> heads -> RoPE -> cache -> SDPA | QKV was already packed; fused decode QKV/O were 73.750/71.421 us and SDPA 13.586 us | Preserve packed QKV and composites; try advisor L1 and DRAM-sharded attention layouts | Advisor QKV/O and legal L1 boundaries kept; DRAM variants lost |
| post-attention norm -> all experts -> activation -> down -> weighting | Dense fused decode read all experts: packed gate/up 2,597.105 us and down 1,945.872 us | Replace with routed active-expert `ttnn.sparse_matmul` | Kept; largest speedup |
| router FP32 -> BF16 TopK/softmax/scatter | Small shared router with required precision boundary | Preserve semantics and evaluate advisor router shard | Kept |
| clipped SwiGLU | Separate clamp/sigmoid/multiply sequence | Fuse sigmoid into multiply; evaluate packed gate/up | Fused multiply kept; packed routed projection rejected by latency |
| residual/layout boundaries | DRAM returns between attention, norm, router, and experts | Evaluate complete advisor L1 chain, not just an isolated conversion | Corrected and kept |
| prefill S=128 | Automatic projection configs and sparse expert padding | Try 8x4/10x4 2D configs and profile selected auto path | Explicit configs rejected on real PCC; auto kept |

Fused baseline op accounting was 5,855.639 us across 56 decode ops and
7,028.254 us across 52 S=17 prefill ops. No collective exists in this
single-device graph. Residual-plus-norm cannot be folded because the residual
is also consumed by the later skip connection.

## Sparse expert rewrite and topology candidates

The selected batch-one graph loads BFP8_B routed expert weights, releases the
inherited dense tensors, and executes device-side routing, sparse gate/up/down,
clipped SwiGLU, routing-score multiplication, reduction, and residual. Public
prefill lengths remain arbitrary; padding and slicing are internal.

An early residual-add output-buffer dtype mismatch was repaired by allocating
the BF16 result rather than reusing a BFP8 expert output. It was not treated as
a rejected candidate.

### Expert precision and geometry

Real checkpoint weights were used for correctness decisions.

| Candidate | Correctness / performance | Decision |
| --- | --- | --- |
| BFP4_B/LoFi 9x10 | real S=17 prefill PCC 0.969613 / 0.972778 for layers 12/13 | reject |
| BFP4_B/LoFi 5x6 | same material failure family after geometry retry | reject |
| BF16/LoFi 9x10 | correct; 5.76075 ms prefill / 1.06843 ms decode | reject: slower |
| BFP8_B/LoFi 9x10, block 45, subblock 1 | correct; final 4.04530 / 0.83421 ms | keep |
| BFP8_B/HiFi2 with final layouts | identical final layer-kind PCC; 0.834096 ms adjacent decode trial | reject: no accuracy gain and no speed win over LoFi |

Precision-locked BFP8 geometry trials:

| Gate/up; down; block/subblock | Prefill / decode mean | Disposition |
| --- | ---: | --- |
| 5x6; 5x6; 45/1 | 4.42867 / 0.94234 ms | slower |
| 3x4; 5x6; 45/1 | 6.11874 / 1.16223 ms | slower |
| 8x8 initial | illegal 90-tile division | retried as 9x5; first error not rejection |
| 9x5; 9x5; 45/1 | 5.11114 / 0.96295 ms | slower |
| 9x10; 9x10; 30/1 | 3.89040 / 0.87462 ms | slower decode |
| gate block 30 only | 3.83309 / 0.86612 ms | slower decode |
| down block 30 only | 3.82777 / 0.85883 ms | slower decode |
| 9x10; 9x10; 90/1 | 3.91926 / 0.85493 ms | slower decode |
| 5x6; 5x6; 45/3 | 4.34516 / 0.88558 ms | legal retry, slower |
| **9x10; 9x10; 45/1** | **selected** | fastest correct family |

`nnz=None` is intentional: deriving a fixed count from host-visible BF16
routing scores can disagree after device-side flush and deadlock the sparse
kernel.

### Projection packing and fused SwiGLU

The selected split gate/up path now uses a device-side fused sigmoid multiply
inside exact clipped SwiGLU. A routed packed `[gate, up]` projection was also
implemented and measured, not dismissed at its first error:

- attempt 1 failed because BFP8 strided slice is unsupported;
- retry typecast the packed output to BF16 on device, then sliced gate/up;
- real layer-12 prefill/decode PCC passed at 0.990246/0.995631;
- traced decode was 1.149508 ms versus 0.873282 ms for split projections with
  fused SwiGLU, and the packed candidate retained about 0.56 GB of extra
  weights.

The packed routed candidate is therefore available only as an opt-in test
variant and is rejected by measured latency.

## Mandatory shard advisor (OPT-015 hard gate)

The advisor was rerun after the final graph rewrite on the dense
attention+MLP version of the block. The current advisor cannot lower
`ttnn.sparse_matmul`, so sparse grids were swept on device as required by the
skill.

The selected BFP8 cache is represented by metadata-only tracer inputs. Direct
host BFP8 conversion incorrectly switched this advisor build away from its
mock target, producing an all-unfixable report; that attempt was rejected.
This branch also traces but cannot lower paged-cache mutation. The harness
threads the cache mutation as an identity because it has no layout choice;
the downstream SDPA retains BFP8 cache argument types, and runtime cache
mutation is validated independently on device.

The successful command was run in a separate shell per SETUP Part B:

```bash
export TTMLIR_ADVISOR_HOME=/home/mvasiljevic/tt-mlir
cd "$TTMLIR_ADVISOR_HOME"
source /home/mvasiljevic/tt-metal/.agents/skills/shard-advise/scripts/bootstrap.sh >/dev/null 2>&1
export PYTHONPATH=/home/mvasiljevic:/home/mvasiljevic/tt-metal:${PYTHONPATH}
export LD_LIBRARY_PATH=/home/mvasiljevic/tt-mlir/third_party/tt-metal/src/tt-metal/build_Release/lib:${LD_LIBRARY_PATH}
ttnn-advise capture \
  /home/mvasiljevic/tt-metal/models/autoports/openai_gpt_oss_20b/doc/optimized_decoder/shard_advise/advise_gpt_oss.py:decode \
  --tracer interception \
  --out /home/mvasiljevic/tt-metal/models/autoports/openai_gpt_oss_20b/doc/optimized_decoder/shard_advise
```

Result: 39 modeled ops, 38 final choices, 24 reshards, spill pass run with
zero spills, and zero unfixable ops. `report.json` contains 39 summary entries
while its `ops` array contains 45 records because six multi-result/structural
records are retained separately; this is report schema accounting, not a
partial capture.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `shard_advise/report.json` | 12,432 | `942251261b4846fe977abaf19d78c1ee6d4d78bc20d61add455ae5b60d490f53` |
| `shard_advise/final_ir.mlir` | 27,874 | `7d42d2aa40e92b8eb086465b4531a339ee7a045cd300841d3bfd57fc9d3fe97e` |

### Recommendation disposition

| Recommendation family | Action and evidence |
| --- | --- |
| 10-core block-sharded input/post-attention RMSNorm | applied for batch-one decode |
| 45-core QKV input, 80-core QKV output, 11x8 1D program | applied; final QKV row 72.363 us |
| head/cache/SDPA boundaries | adapted exactly at API-required L1/interleaved/DRAM boundaries; 8x8/k32 SDPA selected |
| 90-core O/residual width shard, 11x9 1D program | applied; final O row 58.256 us |
| one-core router input/output and routing-weight layouts | applied while retaining FP32 linear -> BF16 TopK semantics |
| complete dense all-expert L1 MLP chain | implemented and PCC-qualified; 5.70996 ms decode, rejected against sparse |
| continuous L1 boundary into sparse experts | first retry fixed block-sharded-to-interleaved typecast; real PCC passed; selected after 0.833735 ms adjacent trial versus 0.873282 ms DRAM-boundary path |

## Attention, cache, fidelity, and prefill program search

| Candidate | Real correctness | Mean prefill / traced decode | Disposition |
| --- | --- | ---: | --- |
| advisor BF16 attention layouts | both layer kinds pass | 3.88248 / 0.84943 ms intermediate | layout seed kept |
| non-DRAM BF16 HiFi2 | higher decode PCC | decode 0.88809 ms | reject: slower |
| DRAM-sharded BFP4_B/LoFi, max cores | L12/L13 0.772627/0.770887 | not promoted | reject PCC |
| DRAM-sharded BFP4_B/LoFi, 64 cores | L12/L13 0.769485/0.771220 | not promoted | legal retry, reject PCC |
| DRAM-sharded BFP8_B/HiFi2 | L12/L13 0.997015/0.996686 | 3.83867 / 0.92138 ms | reject slower |
| DRAM-sharded BF16/HiFi4 | L12/L13 0.997154/0.997141 | 3.83274 / 0.99585 ms | reject slower |
| DRAM-sharded BF16/LoFi with final L1 experts | L12/L13 decode 0.995016/0.994374 | 4.01471 / 0.851718 ms | reject slower than advisor attention |
| BF16 cache | functional baseline | slower/capacity-neutral control | replace |
| BFP8_B cache | both layer kinds and cache >0.99977 | selected | keep |
| default SDPA | correct | about 0.8495 ms | replace |
| 8x8/k64 SDPA | same PCC as k32 | 0.84682 ms adjacent | no benefit |
| **8x8/k32 SDPA** | correct | selected | keep |

At S=128, explicit 8x4 and 10x4 QKV/O programs produced real layer-12/13
output PCC around 0.97469/0.97190. Automatic projection selection produced
0.98922/0.98700 and was materially better, so `prefill_matmul_config="auto"`
is selected. Dense+auto controls reached 0.99205/0.98983, confirming that the
remaining long-prefill delta is dominated by the shared sparse prefill
precision rather than projection layout. Internal chunk-by-32 retry produced
the same PCC and was removed. Synthetic S=128 remains above the stage's 0.99
test bar; S=128 real output is reported as a known precision boundary, not
silently described as qualified. The public API has no divisibility
restriction.

## Semantic corrections from independent review

The first independent stage review found that all layers were using a
128-token sliding window and that the original tests did not cross that
boundary. Remediation:

- derive `attention_window` from `hf_config.layer_types[layer_idx]`;
- use 128 for `sliding_attention` and `None` for `full_attention` in both
  prefill and decode SDPA;
- remove the unused eager dense attention mask;
- extend RoPE/position tables and K/V allocation for caller-selected
  `max_cache_len`;
- add a real-weight, 256-entry-cache test that prefills 128 and decodes at
  position 128 for both layer kinds.

Boundary results are 0.996112 output PCC for layer 12 and 0.994792 for layer
13, with K/V PCC above 0.99982. An AutoFix isolation pass verified that RoPE
tables are bitwise identical at the boundary and that the full-attention SDPA
window semantics are correct; the lower real S=128 prefill output PCC is the
shared sparse prefill precision boundary described above.

## Final correctness

Real S=17 checkpoint coverage under the selected default:

| Layer / kind | Prefill output | Prefill K / V | Decode output | Decode K / V |
| --- | ---: | ---: | ---: | ---: |
| 12 / sliding | 0.99024635 | 0.99991794 / 0.99992703 | 0.99563603 | 0.99981871 / 0.99980306 |
| 13 / full | 0.99308115 | 0.99990794 / 0.99992701 | 0.99410810 | 0.99982619 / 0.99981078 |

Additional coverage:

- non-aligned S=3/17/33 and bitwise deterministic S=17 output/cache rerun;
- synthetic S=128 output and cache PCC above 0.99;
- real sliding/full decode at position 128 using a 256-entry BFP8 cache;
- batch-two compatibility prefill/decode;
- repeated paged decode positions 3-18 with untouched cache tails;
- ten trace replays with bitwise-identical output and complete K/V cache.

## Final performance and profiler evidence

The final same-process real-weight gate uses ten warmed S=17 prefill runs and
100 traced decode replays:

| Path | Prefill mean / min | Decode mean / min |
| --- | ---: | ---: |
| fused | 7.193785 / 7.098952 ms | 6.050208 / 6.033769 ms |
| **selected optimized** | **4.045302 / 3.934355 ms** | **0.834213 / 0.826016 ms** |

The selected path reduces mean prefill by 43.8% and mean traced decode by
86.2%. It also beats the best previous correct selected-path reproduction
(0.846833 ms) and the historical optimized artifact (0.928144 ms).

Fresh Tracy was collected with real layer-12 weights and five bounded windows:

| Window | Ops | Device | Gaps | Total |
| --- | ---: | ---: | ---: | ---: |
| fused S=17 prefill | 52 | 7,026.773 us | 236.111 us | 7,262.884 us |
| optimized S=17 prefill | 63 | 3,671.552 us | 612.177 us | 4,283.729 us |
| fused traced decode | 56 | 5,857.295 us | 49.365 us | 5,906.660 us |
| optimized traced decode | 75 | 768.311 us | 77.188 us | 845.499 us |
| optimized S=128 prefill | 57 | 12,866.761 us | 375.417 us | 13,242.178 us |

Optimized decode sparse rows are 115.665/110.876/109.761 us (336.302 us
total). QKV/O are 72.363/58.256 us, and SDPA is about 10 us. Reports visibly
show BFP8 cache writes, BFP8/LoFi sparse matmuls, and the selected sharded
norm/residual chain. Required routing layout changes are device operations;
there is no `torch`, `from_torch`, `to_torch`, implicit functional forward,
host golden fallback, or collective in the measured runtime methods.

Human-readable tables and CSVs are under `tracy/final/` for fused prefill,
fused decode, optimized prefill/decode, and optimized S=128 prefill. Raw Tracy
artifacts are intentionally not stage-owned.

## Reproducible commands

```bash
pytest -q models/autoports/openai_gpt_oss_20b/tests/test_optimized_decoder.py -s

RUN_OPTIMIZED_DECODER_PERF=1 \
OPTIMIZED_DECODER_PREFILL_REPEATS=10 \
OPTIMIZED_DECODER_TRACE_REPLAYS=100 \
pytest -q models/autoports/openai_gpt_oss_20b/tests/test_optimized_decoder.py::test_optimized_beats_fused_warmed_prefill_and_traced_decode -s

TT_VISIBLE_DEVICES=0 \
TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
RUN_OPTIMIZED_DECODER_PROFILE=1 python -m tracy -r -p \
-o models/autoports/openai_gpt_oss_20b/doc/optimized_decoder/tracy/final \
-m pytest -q models/autoports/openai_gpt_oss_20b/tests/test_optimized_decoder.py \
-k test_profile_optimized_warmed_windows -s

TT_VISIBLE_DEVICES=0 \
TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}' pytest -q \
models/autoports/openai_gpt_oss_20b/tests/test_optimized_decoder.py \
-k 'real_weight_layer_kind or repeated_paged_decode_stress or traced_decode_output_and_full_cache_integrity' -s
```

Final gate results after all runtime changes:

- full suite: 10 passed, 2 intentional opt-in skips in 71.22 s;
- watcher/fallback suite: 6 passed, 6 deselected in 77.87 s;
- watcher scan: no device, NoC, kernel, or uninitialized-memory assertion;
- context runner: target=131072, supported=131072, full HF context;
- optimized-decoder runner: fresh advisor JSON parses and authoritative IR
  contains the rewritten dense block matmuls;
- post-run `tt-smi -s`: all four devices report DRAM healthy and live
  heartbeats.

The exact outputs are retained in
`logs/run_20260722_review_final_correctness.log`, its JUnit XML,
`logs/run_20260722_review_watcher.log`, and its JUnit XML. The final real-weight
performance result is in `logs/run_20260722_review_final_perf.log`; the fresh
advisor command and result are in `logs/run_20260722_review_shard_advise.log`.

## Optimize checklist

- [x] Owned prefill/decode paths; batch-one dense tensors released; static
  fallback/host-conversion audit in tests.
- [x] Operation-topology and graph-fusing audit recorded before tuning.
- [x] Fresh shard advisor run; valid report/IR saved; every recommendation
  family applied, adapted, or rejected with evidence.
- [x] Advisor attention/norm/router/residual layouts and complete sparse L1
  chain evaluated; winning chain selected.
- [x] Expert dtype/fidelity, core grid, block width, subblock, and L1/DRAM
  boundaries swept with real PCC and traced latency.
- [x] Attention dtype/fidelity and DRAM-sharded BFP4/BFP8/BF16 geometries
  swept; first failures retried where adaptation was possible.
- [x] BFP8 cache and explicit SDPA program selected with cache-integrity tests.
- [x] Packed QKV/composite attention retained; packed routed gate/up and fused
  SwiGLU explicitly evaluated.
- [x] Large-prefill program configs evaluated; PCC-losing 2D configs rejected.
- [x] Non-aligned, repeated, batch, layer-kind, long-boundary, trace, profiler,
  and watcher coverage present.
- [x] Before/after warmed prefill and 100-replay traced decode are comparable
  and final selected code beats the strongest correct baseline.
- [x] No host fallback, unnecessary host tensor conversion, or collective in
  the measured single-device graph.

## Limitations

- Real S=128 sparse prefill output is below the 0.99 qualification bar even
  though synthetic S=128 and its cache pass. Decode through position 128 is
  real-weight qualified for both layer kinds. This is documented in the
  context contract and is not used to claim longer real prefill qualification.
- The shared sparse expert implementation supports batch one. Batch two uses
  the exact fused dense compatibility branch and is not a measured optimized
  latency path.
- The default test cache extent remains 128, while callers may request a larger
  cache. Removing the quadratic dense mask and dynamically extending RoPE
  preserves the HF-advertised 131072-token logical contract. A 256-entry cache
  is hardware-validated through decode position 128; longer-context accuracy
  and performance remain unmeasured in this decoder-layer stage and are not
  confused with a public context cap.
