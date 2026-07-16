# Optimized-decoder work log

## Scope and starting point

- Stage: single-device optimized decoder only. No multichip, full-model,
  LM-head, sampling, or vLLM changes were made.
- Functional checkpoint: `79e40abfccf` (`4 passed` in 25.95 s before this
  stage); parent checkpoint `6154b41e1f8`.
- Real weights: HF snapshot `9527884be6e5616bdd54de542f9ae13384489724`.
- Representative layer: 20. All 40 decoder layers have the same dense Mistral
  attention+SwiGLU kind; there is no MoE, SSM, or alternating layer kind.
- Hardware: one logical Blackhole p300c device backed by physical pair
  `TT_VISIBLE_DEVICES=2,3`, compute grid 11x10, DRAM grid 1x8.
- Device handling: all hardware jobs were serialized. Physical pair 0/1 hit a
  fixed-address sysmem conflict; no owner was visible, a bounded `tt-smi -r`
  completed, and pair 2/3 passed a 1x1 mesh open/close smoke. Watcher and Tracy
  were always run separately.

## Operation-topology audit

The audit preceded local config tuning. The functional decode path had packed
QKV but executed every dense op as 32 separately tile-padded M=1 jobs. It used
BF16 interleaved weights, DRAM residuals and norms, split gate/up matmuls,
standalone SiLU, and default dense configs. The first functional Tracy profile
reported 31 decode ops and 93.298 ms device time; QKV was 5.221 ms, O 3.459 ms,
gate 27.947 ms, up 27.950 ms, and down 27.261 ms. MLP was over 89% of the layer.

| Region | Functional topology / movement | Candidates | Final action and evidence |
| --- | --- | --- | --- |
| logical tokens | `[1,32,1,H]` decode and `[1,32,S,H]` prefill left batch outside M, so each user paid tile padding | flatten batch/token axes without changing cache users | decode reshapes to `[1,1,32,H]`; prefill permutes then reshapes to M=`32*S`; this caused the decisive 93 ms to 6 ms step while preserving 32 logical users |
| input norm | BF16 DRAM RMSNorm | advisor 11-core block-sharded norm and residual chain | kept; final norm config is 11x1, shard 32x480, block_w=15, subblock_w=3; final profile rows are 8 and 9 us |
| QKV | one packed BF16 interleaved matmul | keep packing; BFP8/BFP4, fidelity, DRAM sharding, geometry | kept packed; final BFP4/LoFi DRAM-sharded row is 59 us at 268 GB/s |
| heads / RoPE | QKV head creation, sliced rotary tables, L1->DRAM QKV conversion | retain dedicated composite ops; test full-table `token_index` and direct sharded edge | retained TTNN head creation and rotary; full rotary tables plus `token_index=current_pos` remove both untilize/slice/tilize triplets, and explicit last-use deallocation makes direct sharded QKV legal |
| cache / attention | paged updates and decode SDPA, BF16 cache | BFP8 cache, explicit 8x8 SDPA config | BFP8 kept; two updates are 11-12 us, SDPA is 43 us; repeated cache-consuming decode passes |
| O projection | BF16 interleaved | BFP4/LoFi DRAM sharding, block 2/4/8/16 geometries, direct concat-to-O reshard | block 16 kept; one L1 reshard replaces L1->DRAM->L1 and the final row is 41 us at 258 GB/s |
| residuals | DRAM add -> DRAM norm -> DRAM MLP -> DRAM add | advisor coherent L1 chain | kept width-sharded adds and block-sharded norms; advisor candidate improved traced decode 1.529 to 1.377 ms |
| gate/up | two same-input BF16 matmuls | packed gate+up versus separate DRAM-sharded BFP4 | packed prefill kept; separate decode kept because current-source full packed decode was 1.857 ms versus the final 1.288 ms |
| activation | standalone SiLU then multiply | fuse SiLU into multiply | kept; 80x64 geometry improved 1.826 to 1.723 ms |
| down | BF16 interleaved | BFP4/LoFi DRAM sharding, geometry/block sweep | kept; final row 286 us at 293 GB/s, block 16 |
| host / collectives | no collectives; no runtime torch crossings | keep absent | final profile: 31 device ops, 0 host ops, no collectives |

Dedicated/composite operations retained after the `$graph-rewrite` audit are
packed QKV head creation, rotary embedding, prefill causal SDPA, paged decode
SDPA, head concatenation, and RMSNorm. Structural rewrites flatten token axes
and fuse SiLU into `ttnn.multiply`. The paired gate/up pack is beneficial for
large-M prefill but loses in decode because slicing and wide-output movement
outweigh one saved launch/read.

## Cumulative selected contract

| Contract | Selected value |
| --- | --- |
| active batch / tile rows | 32 logical users / one M=32 tile; padded rows are never cache users |
| attention topology | one packed QKV, TTNN head create/rotary, paged cache, decode SDPA, head concat, O |
| attention weights / compute | BFP4 / LoFi; BF16 activations and outputs |
| KV cache | BFP8 DRAM interleaved, shape `[32,8,max_cache_len,128]`; BF16 update input |
| decode SDPA | 8x8, q_chunk=0, k_chunk=0, exp approximation disabled, HiFi4 SDPA kernel |
| prefill SDPA | 8x8, q_chunk=64, k_chunk=64, causal, HiFi4 SDPA kernel |
| decode residual/norm | width-sharded residual adds; 11-core block-sharded RMSNorm |
| QKV geometry | input 10 cores, output 12 cores, in0_block_w=16, per_core_M=1, per_core_N=16 |
| O geometry | input 8 cores, output 10 cores, in0_block_w=16, per_core_M=1, per_core_N=16 |
| MLP topology | separate gate/up, SiLU fused into multiply, down; packed gate/up only for prefill |
| MLP weights / compute | gate, up, down all BFP4 / LoFi |
| MLP geometry | hidden input/output 40 cores, intermediate 32 cores; gate/up block 4, down block 16 |
| prefill | DRAM interleaved activations; 2D 8x9, 10x9, 11x10, 10x9 dense grids inside the <=576-token tuning envelope |
| long prefill | general QKV/O configs above 576 flattened tokens; internal MLP chunks of at most 576; no public alignment restriction |

## Candidate ledger

All latency comparisons use the same real-weight batch-32 layer-20 harness,
prefill length 18, warmed prefill, and warmed traced decode unless stated.

| Candidate | Real correctness | Prefill ms | Decode ms | Decision |
| --- | ---: | ---: | ---: | --- |
| functional BF16 | prefill 0.9999699; decode 0.9999697 | 94.153 | 93.315 | starting baseline |
| packed gate/up BF16, old layout | functional bar passed | 93.96 | 93.10 | packing alone cannot fix padded topology |
| packed BFP4/LoFi, old layout | about 0.999847 | 52.37 | 51.59 | precision helps but topology remains dominant |
| flattened + DRAM MLP, 80x64 | passed | 6.30 | 1.826 | retained flatten/sharding |
| fused SiLU multiply, 80x64 | passed | 6.357 | 1.723 | retained fusion |
| MLP geometry 40x32 | passed | 6.313 | 1.696 | retained |
| MLP geometry 20x16 | not run after exact allocation failure | n/a | n/a | rejected: 2,175,744 B static L1 CB > 1,572,864 B; 80 and 40 candidates already adapted and measured |
| full BFP4 attention, block 2 | passed | 6.365 | 1.567 | retained precision/sharding, continue tuning |
| BFP8 KV cache | layer pass; raw cache about 0.993 | 6.371 | 1.529 | retained after cache-consuming repeated decode |
| explicit prefill configs | prefill 0.9997972; decode 0.9998396 | 5.347 | 1.529 | retained for <=576 flattened tokens |
| advisor residual/norm layout, attention block 2 | prefill 0.9997972; decode 0.9998257 | 5.370 | 1.377 | retained |
| packed gate/up final-family comparison | prefill 0.9997972; decode 0.9998302 | 5.377842 | 1.857137 | rejected; separate DRAM-sharded MLP wins decode by 569.040 us |
| BFP4 HiFi2, same final family | prefill 0.9997972; decode 0.9998257 | 6.265 | 2.228 | rejected; PCC identical at printed precision, much slower |
| attention block 4 (40/48,32/40) | decode 0.9998253 | 5.348 | 1.353 | valid, superseded |
| attention block 8 (20/24,16/20) | decode 0.9998252 | 5.377 | 1.344 | valid, superseded |
| attention block 16 (10/12,8/10) | decode 0.9998262 | 5.543 | 1.335 | fastest decode; promoted |
| full-table RoPE + direct QKV/concat edges | prefill 0.9997972; decode 0.9998262 | 5.387 | 1.288 | retained; 31 versus 39 final-profile ops |
| exact advisor 1D seed | prefill 0.9997972; decode 0.9998337 | 5.436 | 1.788 | rejected; exact 11x9 QKV, 11x8 O/down, separate 11x10 gate/up measured |
| gate/up block 16, input10/output32 | prefill passes; decode program allocation fails | n/a | n/a | rejected: 3,977,984 B static CB > 1,572,864 B L1 |
| gate/up block 16, input10/output64 | prefill passes; same decode failure | n/a | n/a | widest divisibility-legal adaptation; same exact 3,977,984 B blocker |
| final selected default reproduction | eager/trace PCC 1.0; replay/cache bitwise stable | 5.386907 | 1.288097 | final |

Precision/fidelity controls:

| Policy | Real prefill / decode PCC | Prefill / decode ms | Decision |
| --- | ---: | ---: | --- |
| full BFP8 HiFi2, 80x64 | 0.9999656 / 0.9999792 | 7.992 / 2.420 | correct, slower |
| full BFP8 LoFi, 80x64 | 0.9999347 / 0.9999530 | 7.106 / 1.723 | correct, slower |
| full BFP4 HiFi2 | 0.9997972 / 0.9998257 | 6.265 / 2.228 | correct, slower |
| full BFP4 LoFi | 0.9997972 / 0.9998262 | 5.387 / 1.288 | selected |

The final PCC deltas versus functional are -0.0001727 prefill and -0.0001436
decode, comfortably above the inherited 0.99 bar. Raw BFP8 cache PCC is lower
because the cache itself is quantized; layer output and five consecutive
cache-consuming decode positions pass, so it is not a model-visible failure.

## Mandatory shard advisor (OPT-015)

The advisor was run this pass after the dense topology rewrite. Setup followed
`.agents/skills/shard-advise/SETUP.md` Part B in a separate shell:

```bash
export TTMLIR_ADVISOR_HOME=/home/mvasiljevic/tt-mlir
cd /home/mvasiljevic/tt-mlir
source /home/mvasiljevic/tt-metal/.agents/skills/shard-advise/scripts/bootstrap.sh
export LD_LIBRARY_PATH=/home/mvasiljevic/tt-mlir/third_party/tt-metal/src/tt-metal/build_Release/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${PYTHONPATH:-}:/home/mvasiljevic/tt-metal/python_env/lib/python3.12/site-packages
TT_VISIBLE_DEVICES=2,3 ttnn-advise capture /home/mvasiljevic/tt-metal/models/autoports/mistralai_mistral_small_24b_instruct_2501/doc/optimized_decoder/shard_advise/advise_mistral.py:decode --out /home/mvasiljevic/tt-metal/models/autoports/mistralai_mistral_small_24b_instruct_2501/doc/optimized_decoder/shard_advise
```

The tracer initially could not emit `ttnn.repeat` because it passed a tuple to
`ShapeAttr.get`. The repo-external advisor environment was adapted in both its
source and installed `ttnn_emit_tracer.py` to convert repeat dims to a list;
the capture was then rerun, rather than rejecting the advisor on its first
error. A later current-graph capture exposed a traced
`hidden_states.memory_config()` query; the already-static MLP output shard was
made an explicit decoder config and capture was rerun. The final capture uses
BFP4 weights, BFP8 K/V caches, separate gate/up, full-table indexed RoPE, and
the selected dense graph: 23 ops, 20 choices, one reported concat-head spill.
Required artifacts are `shard_advise/report.json` and
`shard_advise/final_ir.mlir`.

| Advisor recommendation | Applied / rejected with evidence |
| --- | --- |
| 11-core block-sharded input/post RMSNorm | applied exactly in principle: 32x480 shard, block_w=15, subblock_w=3; both final rows 8-9 us |
| keep attention and MLP residual adds width-sharded | applied as a coherent candidate; traced decode improved 1.529 to 1.377 ms |
| width-sharded QKV/O and dense boundaries | applied; direct QKV-to-heads and concat-to-O edges are retained, locally retuned to DRAM-sharded BFP4 weights and block-16 activation/output geometries |
| exact 1D interleaved-weight QKV/O/down and separate gate/up | implemented as `advisor_1d` exactly from the IR: QKV 11x9, O/down 11x8, gate/up 11x10 with prescribed padded width shards; PCC passed and 100-replay decode was 1.788112 ms, rejected versus 1.288097 ms selected |
| separate gate/up with 103-core padded output | applied exactly in the advisor candidate; retained topology but rejected interleaved 1D implementation on the measured full-layer latency above |
| advisor down/output width shard | applied in the residual chain, locally retuned to 40 output cores and block 16 |
| L1 interleaved return then DRAM output revert | the public output contract requires DRAM; retained only at the external boundary; final return conversion is 2 us plus the required public-shape reshape |
| full-table indexed rotary | applied; removes two table slices and both untilize/slice/tilize triplets while preserving non-aligned `current_pos` |
| concat-head placement | report marks one unfixable spill because its abstract concat input is DRAM; runtime repairs the SDPA result to the helper-required height shard, concatenates, and performs one direct L1 reshard into O, which is correctness/performance verified |

The optional 44 MB advisor decision trace, 3.2 MB transient pipeline log, and
Python bytecode cache were pruned after extracting the choices; the mandatory
report/IR, bounded capture, and human report remain.

## Final profiler and performance accounting

Final advice-enabled artifacts are under `tracy/final/`. Decode has 31 device
ops and 0 host ops. The frozen-source profile sums to 1.281497 ms of device
kernel time and 0.019264 ms of intra-window op-to-op gaps after excluding the
initial 0.189379 ms replay-entry gap. Its profiled host signpost span is
1.478328 ms and modeled DRAM roofline is 217 GB/s / 42.4% of the 512 GB/s
Blackhole aggregate ceiling. Prefill has 154 device ops, 0 host ops,
4.934697 ms device-kernel time, 0.502682 ms intra-window gaps, a 5.860990 ms
profiled host span, and modeled 86 GB/s / 16.9%. The prefill gaps largely come
from the 32 per-user cache-fill slices/updates; prefill trace work belongs to a
later generator/full-model stage.

The decode weight payload is 555,745,280 BFP4 elements, or about 277.9 MB at
four logical bits per element. At 512 GB/s, weights alone have a 0.543 ms
idealized lower bound; the position-18 BFP8 K/V read adds about 1.25 MB. The
measured 1.281497 ms device-kernel sum is consistent with the tool's 42.4%
modeled roofline. The non-profiled 100-replay result is 1.288097 ms. It is a
separate run and is not subtracted from the profiled kernel/gap values; Tracy's
instrumentation expands the same-run host signpost window to 1.478328 ms.

Material final rows:

| Role / shape | Device time | BW | Runtime policy |
| --- | ---: | ---: | --- |
| packed QKV, 32x5120x6144 | 58.646 us | 268 GB/s | BF16 x BFP4 -> BF16, LoFi, DRAM-sharded, block 16 |
| O, 32x4096x5120 | 40.698 us | 258 GB/s | BF16 x BFP4 -> BF16, LoFi, DRAM-sharded, block 16 |
| gate, 32x5120x32768 | 292.396 us | 287 GB/s | BF16 x BFP4 -> BF16, LoFi, DRAM-sharded, block 4 |
| up, 32x5120x32768 | 294.627 us | 285 GB/s | BF16 x BFP4 -> BF16, LoFi, DRAM-sharded, block 4 |
| fused SiLU multiply | 53.697 us | n/a | BF16 width-sharded gate/up, activation fused into input A |
| down, 32x32768x5120 | 286.259 us | 293 GB/s | BF16 x BFP4 -> BF16, LoFi, DRAM-sharded, block 16 |
| decode SDPA | 43.114 us | n/a | BF16 query, BFP8 cache, explicit 8x8 config |

`tt-perf-report` labels the matmuls `SLOW` relative to architectural ceilings,
but the precision-locked geometry sweep tried blocks 2, 4, 8, and 16 for both
attention roles and blocks 2/4/8/16 across the MLP geometries. In addition to
the earlier 20x16 failure, input10/output32 and the widest legal output64
block-16 gate/up variants both require 3,977,984 B of static CB versus
1,572,864 B L1. `evidence/mlp_block16_10x32_runner.log` and
`mlp_block16_10x64_runner.log` preserve both real-weight attempts, including
the frozen hashes, BFP4/LoFi policy, geometry, passing prefill PCC, exact
exception, expected exit, and healthy post-failure inventory. The selected
legal configurations are the fastest full-layer results.

The final measured runtime contains no `torch`, `from_torch`, `to_torch`, host
fallback, or collectives. Its small reshards are intentional contract crossings
between the 11-core norm, role-specific DRAM matmul shards, head helpers, and
residual adds; each is 1-3 us. Full-table indexed RoPE and direct QKV/concat
edges removed the prior untilize/slice/tilize pairs and DRAM round trips. The
only decode slice is the one-element INT32 position index used by cache update
and SDPA. No extra conversion repairs an avoidable host or functional path.

## Contract, stress, watcher, and capacity evidence

- Selected synthetic path: logical prefill lengths 1, 17, 33, and 18; output
  and BFP8 cache PCC pass. Length 33 exercises internal MLP chunking.
- Decode: positions 18 through 22 pass against the HF layer while consuming
  the updated cache.
- Trace: eager versus first replay PCC 1.0; output and position-18 key/value
  cache slices are bitwise identical after three warmups and 100 measured
  replays.
- Runtime ownership static test checks `_mlp_forward`, `prefill_forward`, and
  `decode_forward` are defined by `optimized_decoder` and have no runtime host
  conversion tokens.
- Watcher: the exact final-default path with `TT_METAL_WATCHER=10` passed in
  23.89 s. `evidence/watcher_runner.log` preserves the environment, test
  output, watcher attach/detach for logical devices 0/1, and a zero-match
  error/assert/hang/NoC signature scan; `watcher_device.log` is the underlying
  device dump.
- Capacity: the first 3,584 attempt exposed a 33,018,624 B explicit-QKV CB
  versus 1,572,864 B L1. It was adapted by bounding tuned configs to their
  measured <=576-token envelope. Fresh-process 3,584 and 4,096 probes then
  passed and copied the full output to host. `doc/context_contract.json` now
  advertises 4,096 as tested, not as a hard maximum.
- Paged-cache scope is preserved from the functional decoder: batch-major
  caches are updated with `ttnn.experimental.paged_update_cache`,
  `share_cache=False`, and `page_table=None`; the five-position test consumes
  those writes. External page-table remapping was not part of the functional
  interface and is not newly claimed here.
- Frozen-snapshot JUnit logs are under `evidence/`. The runtime SHA-256 is
  `c4d3eb3c7df806c67160b20a2d5d22118dc8acae4227b885abcc5adbef85429e`;
  `evidence/README.md` maps every gate and exact advisor comparison to its
  captured XML or runner log. `runner_evidence.json` provides a
  machine-readable index for watcher and expected-failure evidence.

## Independent review remediation

The first independent review rejected the snapshot for an unmeasured exact
advisor seed, stale mixed-run profiler evidence, incomplete dominant MLP
block-16 probing, avoidable rotary/QKV/concat movement, and prose-only final
test claims. The runtime subsequently implemented and measured the exact
advisor 1D configs, tried both 10x32 and 10x64 block-16 MLP adaptations,
removed the identified movement, and regenerated current-source JUnit and
Tracy evidence.

The next fresh review found only two evidence-preservation gaps: watcher
attach/detach and signature-scan output, plus runner output for the two
expected MLP allocation failures. The current `watcher_runner.log`,
`watcher_device.log`, both `mlp_block16_*_runner.log` files, and
`runner_evidence.json` close those gaps. It also requested human-readable
advice tables and pruning optional advisor bulk output; both final
`tt-perf-report` text files now include their advice sections, and the optional
decision trace, pipeline log, and bytecode cache are absent.

## Reproduction commands

Set the real snapshot once:

```bash
export MISTRAL_SMALL_24B_REAL_WEIGHT_DIR=/home/mvasiljevic/hf-cache/hub/models--mistralai--Mistral-Small-24B-Instruct-2501/snapshots/9527884be6e5616bdd54de542f9ae13384489724
```

```bash
# Default selected synthetic/non-aligned/repeated-decode suite
TT_VISIBLE_DEVICES=2,3 pytest -q -s models/autoports/mistralai_mistral_small_24b_instruct_2501/tests/test_optimized_decoder.py

# Real PCC
TT_VISIBLE_DEVICES=2,3 MISTRAL_SMALL_24B_OPT_POLICY=selected pytest -q -s \
  models/autoports/mistralai_mistral_small_24b_instruct_2501/tests/test_optimized_decoder.py::test_real_weight_optimized_policy_pcc

# Final 100-iteration performance and deterministic trace check
TT_VISIBLE_DEVICES=2,3 MISTRAL_SMALL_24B_OPT_POLICY=selected \
  MISTRAL_SMALL_24B_PERF_IMPL=optimized MISTRAL_SMALL_24B_PERF_ITERS=100 \
  pytest -q -s models/autoports/mistralai_mistral_small_24b_instruct_2501/tests/test_optimized_decoder.py::test_warmed_prefill_and_traced_decode_perf

# Watcher, separate from profiler
TT_VISIBLE_DEVICES=2,3 TT_METAL_WATCHER=10 pytest -q -s \
  models/autoports/mistralai_mistral_small_24b_instruct_2501/tests/test_optimized_decoder.py::test_graph_rewrite_prefill_decode_pcc

# Precision-locked dominant MLP block-16 probes (expected exact L1 failures)
TT_VISIBLE_DEVICES=2,3 MISTRAL_SMALL_24B_OPT_POLICY=selected_mlp_10x32 pytest -q -s \
  models/autoports/mistralai_mistral_small_24b_instruct_2501/tests/test_optimized_decoder.py::test_real_weight_optimized_policy_pcc
TT_VISIBLE_DEVICES=2,3 MISTRAL_SMALL_24B_OPT_POLICY=selected_mlp_10x64 pytest -q -s \
  models/autoports/mistralai_mistral_small_24b_instruct_2501/tests/test_optimized_decoder.py::test_real_weight_optimized_policy_pcc

# Isolated optimized capacity
TT_VISIBLE_DEVICES=2,3 MISTRAL_SMALL_24B_OPT_CONTEXT_PROBE_LEN=4096 pytest -q -s \
  models/autoports/mistralai_mistral_small_24b_instruct_2501/tests/test_optimized_decoder.py::test_optimized_batch32_context_capacity_probe

# Final Tracy capture (run without watcher)
TT_VISIBLE_DEVICES=2,3 MISTRAL_SMALL_24B_OPT_POLICY=selected \
  MISTRAL_SMALL_24B_PERF_IMPL=optimized MISTRAL_SMALL_24B_PERF_ITERS=1 \
  python -m tracy -r -p -v \
  -o models/autoports/mistralai_mistral_small_24b_instruct_2501/doc/optimized_decoder/tracy/final \
  -m pytest -q -s \
  models/autoports/mistralai_mistral_small_24b_instruct_2501/tests/test_optimized_decoder.py::test_warmed_prefill_and_traced_decode_perf
```

The `tt-perf-report` commands used `PERF_DECODE`/`PERF_DECODE_END` and
`PERF_PREFILL`/`PERF_PREFILL_END` on the generated `tracy/final/ops.csv`, with
an advice-enabled CSV/table and summary CSV/PNG for each window. The exact raw
CSV is checked in as `ops.csv.gz` to stay below the repository artifact-size
gate; `gzip -dk ops.csv.gz` restores the report input.

## Optimize checklist

- [x] Functional checks pass against the optimized path; real PCC remains
  above the functional 0.99 acceptance bar.
- [x] Prefill/decode semantics, paged KV cache, BFP8 fills/updates, trace
  replay, and determinism are checked.
- [x] Runtime fallback audit is clean; final profile has 0 host ops.
- [x] Stress/repeated-run and watcher-clean evidence exist.
- [x] Warmed before/after latency and final default reproduction are recorded.
- [x] Advice-enabled `tt-perf-report` artifacts and conclusions exist.
- [x] Decode residuals generally remain L1 sharded across both norms, both
  adds, attention, and MLP; helper/dense boundaries are quantified.
- [x] Prefill activations are DRAM interleaved with explicit large 2D configs
  inside their measured envelope.
- [x] Operation-topology audit records current ops, repeated-input candidates,
  conversions, dtype constraints, and decisions.
- [x] `$graph-rewrite` ran before knob tuning; dedicated ops, token flattening,
  packed prefill gate/up, and fused SiLU multiply are PCC-verified.
- [x] `$shard-advise` ran this pass on the rewritten dense block; mandatory
  report/IR exist and every material recommendation has apply/reject evidence.
- [x] Best-candidate comparisons use cumulative real-weight contracts; the
  final default wins traced decode.
- [x] Final profiler rows prove BFP4/LoFi reached QKV, O, gate, up, and down;
  BFP8 reached cache/SDPA.
- [x] Composite SDPA/head/rotary ops replace hand-built attention.
- [x] Packed QKV is kept; packed versus separate MLP is measured and separate
  decode wins after slice/movement costs.
- [x] Important memory, program, SDPA, norm, and compute kernel configs are
  explicit.
- [x] Dominant program configs were swept by role: attention block 2/4/8/16;
  MLP 80x64, 40x32, 20x16, and input10/output32/output64 block-16 attempts;
  the latter two preserve the exact 3,977,984 B versus 1,572,864 B blocker.
- [x] LoFi/HiFi2 and BFP4/BFP8 were crossed with geometry using real weights.
- [x] Attention precision was swept independently; BFP4/LoFi wins.
- [x] Gate, up, and down all had BFP4/LoFi trials before prefill-only tuning.
- [x] Shard specs divide tiled dimensions; selected core counts unlock block 16
  where legal.
- [x] All five decode matmuls use DRAM-sharded BFP4 weights.
- [x] Collective topology is minimized: this single-device stage has no
  collectives. Multi-device CCL families, fused matmul-CCL, and persistent CCL
  buffers are not applicable.
- [x] MoE/sparse-matmul requirements are not applicable to this dense model.
- [x] LM-head/sampling requirements are outside this decoder-only stage.
- [x] Reduced precision/fidelity evidence is module-level and real-weight;
  full-model top-k selection remains for the later datatype stage.
- [x] Roofline, summed device-kernel time, intra-window gaps, profiled host
  span, and separate non-profiled end-to-end time are reported without
  cross-run subtraction. `perf_summary.json` is not required for a decoder
  module.
- [x] Batch capability is preserved: batch 32 is the emitted and tested path;
  logical users remain distinct from tile padding.

## Limitations

- This is a single layer, not a full autoregressive model or serving result.
- 4,096 is the largest optimized context tested in this stage, not a measured
  OOM boundary and not the HF-advertised 32,768 context.
- The prefill cache-fill loop remains one device update per user and leaves
  profiler-visible host gaps. Removing that orchestration requires a compatible
  batched fill API or a later traced generator integration; it does not affect
  the selected traced decode target.
- Role-specific 1-3 us L1 reshards remain between the norm, dense, head-helper,
  and residual layouts. The prior rotary slice conversions and attention DRAM
  round trips were removed.

## Checkpoint

The final fresh-context `$stage-review` returned `clean-pass` with no required
work after independently recomputing the profiler totals and checking the
advisor, correctness, capacity, watcher, and expected-failure artifacts.

- Optimized implementation/artifact checkpoint:
  `92f5a3cbf5c64708212a26938a33c7af00c55480`.
- This documentation-only follow-up records that review and checkpoint; its
  SHA is reported at handoff. No push was performed.
