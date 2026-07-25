# Optimized decoder work log

## 2026-07-16 — Stage scope and baseline

- Stage root: `models/autoports/openai_gpt_oss_20b` on branch
  `mvasiljevic/model/openai-gpt-oss-20b`, starting commit `9e5c6451d00`.
- Scope is limited to `tt/optimized_decoder.py`, optimized-decoder tests, and
  model-local documentation/artifacts.  No multichip, full-model, generator,
  or vLLM work is part of this stage.
- The worktree was clean at entry.  `timeout 60 tt-smi -ls --local` found the
  reserved second P300 pair as local UMD devices 0 and 1 (physical PCI devices
  2 and 3).  A serialized `MeshShape(1, 1)` open/close passed on Blackhole with
  compute grid `11x10` and DRAM grid `8x1`.
- Re-ran the functional floor into `functional_baseline_results.xml`: 6/6
  passed in 7.11 s.  Exact PCC was unchanged: synthetic prefill 17
  `0.9999945678956855`, 128 `0.9999974798163773`, 256
  `0.9999979815485808`; official real-weight prefill 17
  `0.9997057201235178`; official real-weight decode position 17
  `0.9996046254703057`.

## Operation-topology and graph-rewrite audit

This audit was completed before program-config or precision tuning, from the
functional runtime and the closest GPT-OSS/common-module implementations.

| Path | Current operation sequence / movement | Candidate | Dtype/fidelity constraint | Initial action |
| --- | --- | --- | --- | --- |
| attention norm | DRAM hidden -> dedicated `rms_norm` -> DRAM | sharded L1 residual/norm chain | BF16 norm; preserve real-weight routing sensitivity | retain dedicated op; measure sharded chain after advisor seed |
| QKV | one packed same-input QKV `linear` with fused bias | split Q/K/V is strictly worse topology; retain packed | sweep attention BFP8/BFP4 and HiFi2/LoFi independently | packed topology retained |
| head creation | packed QKV -> dedicated prefill split-heads / decode create-heads | none lower-op-count found | output layout must remain legal for RoPE/cache | retain dedicated ops |
| RoPE | two dedicated rotary ops plus logical slice for non-aligned prefill | folding decode layout into token-index rotary is already used | BF16 query/key required by cache update contract | retain; preserve logical slice |
| cache + attention | fill/update cache -> dedicated prefill SDPA / decode SDPA | explicit SDPA program config; BFP8 cache | prefill fill operands cast to cache dtype; decode updates remain BF16 | measure explicit config and reduced cache |
| decode concat heads | decode SDPA DRAM output -> raw reshape -> output projection | DRAM -> height-sharded L1 -> `nlp_concat_heads_decode` | preserve 64 heads x 64 head dim and BF16 | first PCC/perf graph-rewrite candidate |
| attention output | output `linear` + bias -> DRAM residual add | DRAM-sharded decode weight and sharded residual output | attention precision independent of MoE | sweep after advisor seed |
| router | BF16 norm -> FP32 typecast -> FP32 `linear` -> BF16 -> topk/softmax/scatter | keep routing boundary; L1 decode logits | FP32 boundary is required by real-weight expert selection | retain math, move small decode intermediates to L1 if PCC holds |
| MoE gate/up | repeat normalized input across 32 experts -> one dense BF16 batched matmul -> interleaved slices | routed `sparse_matmul` gate/up with BFP4/LoFi; compare packed dense capture seed only | real-weight PCC decides; static `nnz=4` is unsafe on Blackhole because routing weights can flush to zero | replace dense runtime with routed sparse family; omit `nnz` unless exact count is proven |
| SwiGLU | clamp -> multiply -> sigmoid -> multiply -> add -> multiply | in-place/fused binary activation forms where legal | preserve BF16-rounded alpha `1.703125` and clamp order | reuse GPT-OSS sparse-expert in-place chain and profile elementwise rows |
| MoE down/reduce | dense BF16 down -> bias -> routing multiply -> sum | sparse down with `is_input_a_sparse=True`, L1 decode intermediates, fast reduce where legal | BFP4/LoFi cross geometry; real-weight PCC | replace dense runtime and sweep gate/up/down roles separately |
| host/layout audit | no runtime torch/from/to-torch; prefill DRAM; decode has DRAM/L1 transitions | trace decode; eliminate avoidable transitions | preserve public logical batch=1 and non-aligned lengths | keep runtime host-free and account every remaining transition |

Dedicated-op repository audit covered `ttnn/cpp/ttnn/operations`,
`models/common/modules/{attention,mlp,rmsnorm}`, and
`models/demos/gpt_oss/tt/{attention,experts,topk.py}`.  The functional graph
already uses packed QKV, RMSNorm, rotary, SDPA, prefill concatenate-heads,
TopK, numeric-stable softmax, and scatter.  The decode concat-head rewrite and
routed sparse experts are the material remaining graph changes.

The first concat-head candidate reached the output projection but initially
failed its final reshape because `nlp_concat_heads_decode` intentionally
returned 32 tile-padded logical rows.  The adapted candidate now slices that
output to the one active logical user before projection; this is the established
decode contract, not a fallback to the old reshape path.

## Commands and artifacts

```bash
timeout 60 tt-smi -ls --local
timeout 60 python - <<'PY'  # open/close MeshShape(1,1)
...
PY

pytest -q --capture=tee-sys -o junit_logging=all \
  --junitxml=models/autoports/openai_gpt_oss_20b/doc/optimized_decoder/functional_baseline_results.xml \
  models/autoports/openai_gpt_oss_20b/tests/test_functional_decoder.py
```

The full console log is `logs/functional_baseline.log`.

## Graph rewrite and mandatory shard-advisor gate

`$graph-rewrite` retained the already packed QKV and gate/up projections and
replaced the decode head flattening with
`ttnn.experimental.nlp_concat_heads_decode`. A logical one-user slice after
the tile-padded concat output is required before the packed O projection. The
isolated dense-control PCC is `0.999869`; graph-rewrite-only latency is
7.707365 ms prefill and 6.124800 ms traced decode versus 7.709439/6.139372 ms
functional. Fewer operations alone was therefore not accepted as stage
completion.

The required advisor was run this pass on the rewritten packed-attention plus
dense-MLP decode capture. It used Part B's separate shell/bootstrap flow:

```bash
export TTMLIR_ADVISOR_HOME=/home/mvasiljevic/tt-mlir
cd "$TTMLIR_ADVISOR_HOME"
source /home/mvasiljevic/tt-metal/.agents/skills/shard-advise/scripts/bootstrap.sh >/dev/null 2>&1
export LD_LIBRARY_PATH="$TTMLIR_ADVISOR_HOME/third_party/tt-metal/src/tt-metal/build/lib:$LD_LIBRARY_PATH"
ttnn-advise capture \
  /home/mvasiljevic/tt-metal/models/autoports/openai_gpt_oss_20b/doc/optimized_decoder/shard_advise/advise_gpt_oss.py:decode \
  --out /home/mvasiljevic/tt-metal/models/autoports/openai_gpt_oss_20b/doc/optimized_decoder/shard_advise
```

Hard-gate artifacts:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `shard_advise/report.json` | 13,792 | `0729a78b0d41ed1b2e87ba23a10dd28da6f529ebeb391a14719c5e0e96f9962d` |
| `shard_advise/final_ir.mlir` | 26,624 | `27e130f869d16646f898cef4521f4d8745e536d3ec5b3c29565c797d09416bd1` |

`report.json` retains the advisor's provenance path for the generated
`decision_trace/decode_decision_trace.json`. That large greedy-phase debug
trace was consumed to render `report.json`/`report.txt` and then deliberately
pruned; it is non-gating retained evidence. The authoritative optimized graph
is `final_ir.mlir`, and the required retained hard-gate pair is the JSON/MLIR
listed above. Neither generated hard-gate artifact was rewritten.

Advisor recommendation disposition:

| Recommendation | Application this pass | Decision and evidence |
| --- | --- | --- |
| decode norm block-sharded on 10 cores, then QKV input/output width-sharded on 45/80 cores with 11x8 1D mcast | exact memory/program configs and block-to-width conversion implemented | **applied**: both layer kinds and boundary 128--130 pass; sparse end-to-end timing improves from 0.983908 to 0.928144 ms |
| O input L1 interleaved, 11x9 1D mcast, 90-core width-sharded O/residual, final revert | exact captured chain implemented; selected sparse boundary reverts to DRAM before MoE | **applied**: same correctness evidence; final O row is 62 us on 90 cores |
| post-attention norm, 1-core router, and dense-MoE 90-core layout/revert chain | full `final_ir.mlir` chain implemented behind `advisor`, including tiled width-90 norm weights, post-softmax interleaved scatter input, one-core block scatter output, repeated-input width-90/L1 transitions, routing broadcast, reduction, and residual revert | **correct but rejected on performance**: complete real-weight prefill/decode PCC is `0.99958--0.99973`; 7.727175/5.909603 ms is slower than selected 3.874835/0.928144 ms |
| dense expert matmuls and L1 elementwise chain | captured dense matmul outputs remain DRAM before the exact width-shard/add/L1 sequence; final sum and residual revert are represented | correct in the complete A/B above, but routed sparse experts are 6.37x faster on traced decode; selected `sparse_matmul` falls outside the dense advisor capture |
| DRAM cache and SDPA | retained | applied: persistent K/V remain DRAM BF16 and the dedicated SDPA composite remains in use |

The report was used as the required first sharding seed, not treated as a
mandate. Unlike the initial two-config experiment, the final A/B code follows
all 30 layout/revert operations in the dense captured graph. The first-bad
tensor was localized to router scatter: the initial translation omitted the
IR's post-softmax L1-interleaved conversion. Explicit repeat and routing
broadcast transitions were also missing, and a direct decode reshape had to
remain a prefill permutation when token count exceeds one. After those
repairs, the complete chain passes. Logs are
`shard_advisor_full_chain_attempt{1..4}.log`,
`shard_advisor_moe_only_correctness_exact_layouts_attempt{2,3}.log`,
`shard_advisor_full_chain_correctness_final.{log,junit.xml}`,
`perf_shard_advisor_full_chain_100_final.log`, and
`perf_shard_advisor_attention_sparse_100.log`.

## Candidate matrix

All accuracy decisions below use real checkpoint weights. Synthetic coverage
was used for shape/non-alignment and crash detection, never as the sole
precision veto.

| Family | Candidate | PCC / failure | Prefill / traced decode | Decision |
| --- | --- | --- | ---: | --- |
| expert dtype | BFP4/LoFi sparse matmul, 9x10 and 5x6, cumulative selected advisor-attention topology | ordinary prefill `0.976127`; boundary prefill `0.973667` for both layer kinds and both grids | not timed as valid | reject below 0.99; trace is not entered after the mandatory prefill gate fails; `final_context_sparse_bfp4_{9x10,5x6}.{log,junit.xml}` |
| expert dtype | BFP8 | real prefill/decode initially `0.998096/0.998476` | selected below | keep |
| expert grid | inherited 3x4 gate/up, 5x6 down | pass | 6.322259 / 1.257649 ms | reject |
| expert grid | 5x6 both roles | pass | 4.436848 / 1.003853 ms | reject |
| expert grid | 8x8 | pass | recorded in `perf_sparse_bfp8_8x8.log` | reject versus 9x10 |
| expert grid | 9x10 both roles | pass | 3.973196 / 0.948142 ms before boundary fix | keep grid |
| gate/down block | 30/10 | pass | 3.919284 / 0.965641 ms | reject |
| gate block | 45/10 | pass | 3.880909 / 0.959733 ms | improve gate only |
| down block | 30/45 | pass | 3.895346 / 0.950561 ms | improve down only |
| gate/down block | 45/45 | pass | 3.915300 / 0.944606 ms before boundary fix | keep block widths |
| gate/down block | 90/90 | pass | 3.944548 / 0.951316 ms | reject |
| output subblock | 5x6, width 3 | first shared-builder config hung; AutoTriage found invalid `out_block_w=1` | adapted valid config 4.351867 / 0.961806 ms | local builder fixed to validate/divide output blocks; reject slower candidate |
| sparse input placement | force decode input to L1 | pass | 3.887633 / 0.968375 ms | reject slower than DRAM/interleaved module path |
| attention fidelity | global LoFi attention policy | decode PCC `0.982951` | not valid | reject below bar |
| KV cache | BFP8 retested with final explicit mask | full boundary passes; sliding position 129 PCC `0.863325` | not timed as valid | reject on final-context traced correctness |
| SDPA config | explicit 8x8, K chunk 32 | final boundary passes both layer kinds | 3.874835 / 0.928144 ms selected default | keep explicit contract |
| SDPA config | automatic | final boundary passes both layer kinds | 3.879351 / 0.928017 ms | reject as timing-equivalent, less explicit |
| SDPA config | explicit K chunk 64 | final boundary passes both layer kinds | 3.875034 / 0.928008 ms | reject as timing-equivalent, larger chunk |
| attention DRAM-sharded | corrected BFP4/LoFi separate-bias path | ordinary decode PCC `0.747--0.750` | invalid | reject |
| attention DRAM-sharded | corrected BFP8/HiFi2 separate-bias path | ordinary PCC `0.9957--0.9968`; full/sliding boundary fails | 3.881326 / 0.957894 ms | reject boundary correctness |
| attention DRAM-sharded | corrected BF16/HiFi4 separate-bias path | ordinary PCC `0.9967--0.9977`; sliding position 130 PCC `0.91306` in eager and trace | 3.877699 / 1.042835 ms | reject boundary correctness |
| advisor attention layouts + sparse MoE | real ordinary and boundary PCC pass | 3.875976 / 0.928019 ms candidate, **3.874835 / 0.928144 ms** post-review final rerun | **select** |
| large prefill | automatic sequence-128 programs | synthetic PCC `0.999998`; real perf path passes | 13.466493 ms | keep |
| large prefill | explicit 2D 8x4 QKV / 6x4 O | legal runtime path | 13.545396 ms | reject 0.59% slower |
| large prefill | explicit 2D 10x4 QKV / 10x4 O | legal runtime path | 13.525096 ms | reject 0.44% slower |

The sparse expert profiler rows confirm the intended BFP8/LoFi policy reached
the runtime. Gate/up/down are 117/116/113 us on 90 cores. The shared GPT-OSS
expert module internally selects LoFi for these BFP8 sparse kernels; the
attention and router compute policy remains separately controlled.

## Sequential positions, sliding-window boundary, and AutoFix

An early real-weight extension to positions 17/18/19 exposed PCC `0.892` at
the second decode. `$autofix` was invoked as required. The generated
`AUTODEBUG.md` hypotheses were tested rather than copied as conclusions:

- the HF `DynamicCache` reference matches full recomputation to `1.19e-7`;
- dense and sparse controls share the failure, refuting MoE as root cause;
- cache rows remain stable, refuting accidental overwrite;
- source inspection found that legacy host scalar RoPE position was absent
  from the operation program hash.

The selected fix gathers cos/sin on device from a persistent row-major table
using the mutable position tensor and applies `rotary_embedding_hf`. Positions
17/18/19 then pass for both layer kinds, same-position execution is bitwise
deterministic, and fixed-position trace replay is stable.

The new 128/129/130 boundary test then exposed a separate native sliding-mask
bug: sliding PCC fell to `0.913280` at position 130 while full attention stayed
`0.997193`. AutoFix source analysis localized the first failing window start
to the partial sliding-window tile path that combines a packed uint32 pair with
an odd uint16 tail. The following independent adaptations were tried:

| Adaptation | Result | Disposition |
| --- | --- | --- |
| native sliding, explicit K chunk 32 | position 128 PCC `0.1828` | reject |
| bounded paged circular cache/composite | positions 128/129/130 `0.997125/0.995865/0.912223` | reject; same native mask bug |
| elementwise ring cache | 128/129 `0.997334/0.863629` | reject |
| explicit mask, FP32 destination accumulation | position 128 PCC `0.279216` | reject |
| explicit mask with q-chunk, BF16 mask, no sink, DRAM query, finite mask, host-uploaded mask, and virtual sink controls | PCC `0.279-0.311` | these controls refute geometry, dtype, sink, placement, special values, and mask construction as root cause |
| explicit mask, BF16 destination accumulation | positions 128/129/130 `0.997271/0.996218/0.997132` | keep |

An upstream random-mask SDPA test and an exact
batch=1/QH=64/KVH=8/S=256/D=64 control both pass (`0.999831`). The failure is
specific to explicit-mask decode with FP32 destination accumulation. The final
stage-local workaround uses the supported BF16-destination SDPA configuration;
it preserves the native attention sink and adds no query reshard.

## Independent review and AutoFix remediation

The first independent `$stage-review` returned more-work-needed with four
specific findings: the advisor seed stopped at two matmul configs; the
DRAM-sharded attention failure was unexplained; BFP8 cache/SDPA evidence
predated the final mask; and explicit large-prefill configs plus a theoretical
roofline were missing. All four were reopened.

`$autofix` requested a fresh `$autodebug` investigation. The mandated
`.agents/scripts/autodebug.sh --agent codex` runner started three isolated
explorers, but all failed before filesystem access because the isolated Codex
sandbox could not initialize bubblewrap. It produced no `AUTODEBUG.md` and no
edits. The serial AutoFix fallback then continued with source controls and
hardware evidence rather than treating this infrastructure failure as a code
conclusion.

For DRAM-sharded attention, comparison with
`models/common/modules/attention/attention_1d.py` found that bias must be added
after the sharded QKV/O matmul. That repair restores ordinary-position BF16
and BFP8 PCC. It does not repair the real sliding boundary: BF16 fails position
130 in eager and traced execution even though the device-built mask is exact;
BFP8 also fails full-attention position 129. Automatic/K=64 SDPA, query in
DRAM, early deallocation, and mask probes are independent negative controls.
These logs begin with `autofix_dram_separate_bias_` and
`autofix_boundary_`.

The full advisor chain initially required four concrete adaptations:
CoreRangeSet shard shapes, decode-only application of the captured graph,
dense matmul DRAM outputs before the captured width-shard conversions, and an
explicit DRAM boundary for the attention-only control. The second independent
review correctly rejected the then-unlocalized dense-MoE PCC collapse and the
pre-advisor BFP4 veto. AutoFix localized the earliest dense divergence to the
router softmax/scatter boundary, then made every missing norm-weight, repeated
expert input, routing broadcast, reduction, and residual layout explicit.
The complete dense chain now passes both layer kinds at PCC
`0.99958--0.99973`; it is rejected only after a valid 100-replay measurement
of 7.727175/5.909603 ms. BFP4 was rerun cumulatively at 9x10 and 5x6 and still
fails ordinary/boundary prefill at `0.976127/0.973667`. The failing historical
`boundary_trace_correctness.junit.xml` was removed; its console log remains
negative pre-fix evidence and is superseded by `boundary_trace_final.junit.xml`
and the current passing `test_results.xml`. Final-context BFP8 cache,
automatic SDPA, and K=64 SDPA boundary logs are `final_context_boundary_*`.
Large-prefill sweep logs are `perf_prefill_{auto,2d_*}_seq128*`.

## Final correctness, performance, and profile

Final default results:

- `test_results.xml`: 8 passed, one opt-in perf test skipped;
- non-aligned synthetic prefill 17/33/128 passes;
- real positions 17/18/19 and traced boundary positions 128/129/130 pass for
  both sliding and full attention;
- 20 fixed-position trace replays per layer kind pass and same-position direct
  execution is bitwise deterministic;
- `TT_METAL_WATCHER=10` real-weight stress: 2 passed, no watcher assert/error;
- final sequence-17 wall timing: 3.874835 ms warmed prefill and 0.928144 ms
  traced decode over 10/100 repetitions;
- functional timing: 7.709439/6.139372 ms, so final speedups are 1.99x/6.61x;
- final sequence-128 warmed prefill: 13.466493 ms;
- final Tracy after advisor AutoFix: decode 267 device ops/three replays, 0
  host ops, 2.715058 ms device time and 253.808 us gaps; prefill 126 device
  ops/two repeats, 0 host ops, 7.602124 ms device time.

The final profile contains no runtime `torch`, `from_torch`, `to_torch`, or
host fallback. One block-to-width `ReshardDeviceOperation` per replay is the
advisor QKV producer/consumer contract; QKV width-to-interleaved, head concat,
O input/output, residual, and sparse-module DRAM revert are likewise required
layout boundaries. Gathered cos/sin are placed directly into RoPE's
height-sharded input. Mask tilize/untilize rows are internal to TTNN
elementwise/typecast/repeat implementations, not explicit source or host
transitions.

Advice-backed reports are:

- `tracy/sliding_attention/decode_perf_report.txt`;
- `tracy/sliding_attention/prefill_perf_report.txt`;
- CSV companions generated with `--active-experts 4`.

Aggregated decode device time is 38.36% sparse matmul, 17.95% dense matmul, and
5.09% norm; individual active expert rows are about 4% each. Advice to
DRAM-shard QKV/O is closed by the corrected ordinary-position and boundary
experiments above.
Advice to use HiFi2/LoFi on attention is closed by the LoFi PCC failure. Advice
to place expert input in L1 is closed by the measured 0.968375 ms regression.
Advice about larger output subblocks is closed by the valid width-3 adapted
candidate. Prefill reports host gaps because prefill is intentionally warmed
but not traced; decode is fully traced.

Theoretical roofline accounting uses 159,364,864 bytes/token of dominant
weights, biases, norms, and KV reads. At 512 GB/s this is a 0.311259 ms/token
floor. The profiled 0.999758 ms wall timing is 31.1% bandwidth equivalent and
the 0.905019 ms device timing is 34.4%; `tt-perf-report` independently reports
33.4% / 171 GB/s. The remaining distance is consistent with dispatch gaps,
layout/composite work, and imperfect per-matmul bandwidth rather than a single
unexamined dominant row.

## Device recovery record

One experiment orchestration mistake launched two sequence-128 perf jobs at
once. The second was stopped before measurement; the surviving compile was
terminated after six minutes and left PCI device 3 reporting `NOC0 is hung`.
No result from either run is used. Recovery followed `$tt-device-usage`:

```bash
timeout 60 tt-smi -ls --local   # NOC0 hung on PCI device 3
timeout 180 tt-smi -r           # warm reset devices 2 and 3
timeout 60 tt-smi -ls --local   # both P300c devices visible
python - <<'PY'                 # serialized 1x1 mesh open/close
import ttnn
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
ttnn.close_mesh_device(mesh)
print("MESH_SMOKE_OK")
PY
```

The serialized selected 9x10 sequence-128 run then passed in 11.22 seconds.
Subsequent final profile, correctness, and watcher runs also passed.

## Context contract

The optimized path keeps the inherited `SUPPORTED_CONTEXT=21248` and BF16
linear cache, so advertised capability is preserved. Sparse experts reduce
working memory, but raising the constructor contract requires editing and
revalidating the completed functional decoder, outside this stage's authorized
files. `doc/context_contract.json` records this explicitly; it does not claim
an unmeasured larger context.

## Optimization checklist

- [x] Decoder path fully traced with no host fallback.
- [x] Decode width-sharded norm/residual opportunity assessed as a coherent
  family. All 30 advisor layout/revert operations are represented in the full
  dense A/B candidate. The full chain passes PCC `0.99958--0.99973`; its
  attention half is selected with sparse MoE, while dense MoE is rejected on
  measured 5.909603 ms traced-decode latency.
- [x] Prefill is primarily DRAM interleaved. Explicit legal 8x4/6x4 and
  10x4/10x4 2D QKV/O configs were swept at sequence 128 and rejected against
  the faster automatic programs; the 90-core sparse expert grid is retained.
- [x] Operation-topology audit completed before tuning.
- [x] `$graph-rewrite` completed and PCC verified.
- [x] `$shard-advise` run this pass; required JSON/MLIR saved and every
  applicable recommendation applied or rejected with evidence.
- [x] Multi-device collective/CCL items are not applicable to this single-chip
  decoder stage.
- [x] Best-candidate comparison uses the final boundary-correct default; the
  faster native-mask number is explicitly excluded as incorrect.
- [x] Final default performance reproduced after review fixes: 3.874835 ms
  prefill and 0.928144 ms traced decode over 10/100 repetitions.
- [x] Final dtypes/fidelity verified in current profiler rows.
- [x] Dedicated SDPA/RoPE/head/TopK/sparse-matmul composites used.
- [x] Packed same-input QKV and packed gate/up retained; dedicated split-heads
  and concat-heads used.
- [x] Important memory, program, and compute-kernel configs are explicit.
- [x] Dominant gate/up and down configs swept separately across grids,
  `in0_block_w`, subblocks, and input placement.
- [x] Attention fidelity and expert precision/fidelity swept independently.
- [x] BFP4 gate/up and down tried across two legal geometries and rejected by
  cumulative selected-topology real-weight ordinary/boundary PCC.
- [x] Large clean core grids/divisibility selected: 9x10 and block width 45.
- [x] DRAM-sharded attention matmuls tried in BFP4/BFP8/BF16 policies. Bias
  placement was repaired; BFP8/BF16 ordinary-position PCC then passed, and
  selection was rejected only after eager/traced boundary failures. Sparse
  expert weights use the module's DRAM-backed 90-core path.
- [x] No collectives/CCLs exist in this single-chip path.
- [x] Routed active-expert sparse path selected; no dense runtime fallback.
- [x] LM head/sampling items are not part of a decoder-layer stage.
- [x] Reduced precision experiments used real weights; broad full-model
  datatype selection remains outside this stage.
- [x] Device time, wall time, gaps, profiler overhead, and a theoretical
  159.365 MB/token / 0.311259 ms roofline floor reconciled.
- [x] Batch-1 capability preserved. Larger batch is not applicable because the
  completed emitted functional contract rejects any batch other than one; the
  optimized stage does not hard-code an additional restriction.

The advisor chain is no longer an unchecked aspiration: every captured layout
and revert is implemented. The attention portion is now part of the selected
runtime; the complete dense chain remains available as a correct,
evidence-bearing A/B path and is rejected because it is much slower than the
selected sparse expert implementation.

## Command ledger

```bash
# Correct full dense advisor A/B
OPTIMIZED_DECODER_CORRECTNESS_VARIANT=advisor pytest -q -s \
  models/autoports/openai_gpt_oss_20b/tests/test_optimized_decoder.py \
  -k real_weight_optimized_prefill_decode_pcc
RUN_OPTIMIZED_DECODER_PERF=1 OPTIMIZED_DECODER_PERF_VARIANT=advisor \
OPTIMIZED_DECODER_TRACE_REPLAYS=100 OPTIMIZED_DECODER_PREFILL_REPEATS=5 \
pytest -q -s models/autoports/openai_gpt_oss_20b/tests/test_optimized_decoder.py \
  -k test_optimized_decoder_perf

# Cumulative selected-topology BFP4 veto; repeated with sparse_bfp4_5x6
OPTIMIZED_DECODER_CORRECTNESS_VARIANT=sparse_bfp4 \
OPTIMIZED_DECODER_BOUNDARY_VARIANT=sparse_bfp4 pytest -q -s \
  models/autoports/openai_gpt_oss_20b/tests/test_optimized_decoder.py \
  -k 'real_weight_optimized_prefill_decode_pcc or traced_decode_updates_position_across_sliding_window_boundary'

# Final suite
pytest -q -s models/autoports/openai_gpt_oss_20b/tests/test_optimized_decoder.py \
  --junitxml=models/autoports/openai_gpt_oss_20b/doc/optimized_decoder/test_results.xml

# Final default performance
RUN_OPTIMIZED_DECODER_PERF=1 OPTIMIZED_DECODER_PERF_VARIANT=optimized \
OPTIMIZED_DECODER_TRACE_REPLAYS=100 OPTIMIZED_DECODER_PREFILL_REPEATS=10 \
pytest -q -s models/autoports/openai_gpt_oss_20b/tests/test_optimized_decoder.py::test_optimized_decoder_perf

# Final sequence-128 large-prefill check
RUN_OPTIMIZED_DECODER_PERF=1 OPTIMIZED_DECODER_PERF_VARIANT=optimized \
OPTIMIZED_DECODER_PERF_SEQ_LEN=128 OPTIMIZED_DECODER_TRACE_REPLAYS=1 \
OPTIMIZED_DECODER_PREFILL_REPEATS=3 \
pytest -q -s models/autoports/openai_gpt_oss_20b/tests/test_optimized_decoder.py::test_optimized_decoder_perf

# Final profile, followed by advice-backed reports
RUN_OPTIMIZED_DECODER_PERF=1 OPTIMIZED_DECODER_PERF_VARIANT=optimized \
OPTIMIZED_DECODER_TRACE_REPLAYS=3 OPTIMIZED_DECODER_PREFILL_REPEATS=2 \
python -m tracy -r -p -v -m pytest \
  models/autoports/openai_gpt_oss_20b/tests/test_optimized_decoder.py \
  -k test_optimized_decoder_perf

tt-perf-report tracy/sliding_attention/ops.csv --active-experts 4 \
  --start-signpost PERF_DECODE --end-signpost PERF_DECODE_END --no-summary
tt-perf-report tracy/sliding_attention/ops.csv --active-experts 4 \
  --start-signpost PERF_PREFILL --end-signpost PERF_PREFILL_END --no-summary

# Watcher, deliberately separate from profiler
TT_METAL_WATCHER=10 pytest -q -s \
  models/autoports/openai_gpt_oss_20b/tests/test_optimized_decoder.py \
  -k real_weight_optimized_prefill_decode_pcc
```

Round three of the independent `$stage-review` returned **CLEAN PASS** after
verifying the dense advisor chain operation-by-operation against
`final_ir.mlir`, the cumulative BFP4 veto, mandatory artifact hashes, current
JUnits, selected timing, watcher output, and final profiler reports. No
blocking finding remained. The stage implementation/evidence commit SHA is
`9949cb70f3f82cde84cd725d864bdb092c97ea62`. The commit passed all applicable
repository hooks; the end-of-file normalizer alone was skipped to preserve the
byte-exact generated shard-advisor report and its documented SHA-256.
