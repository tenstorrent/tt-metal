# Optimized decoder work log

## Scope and provenance

- Model: `tiiuae/Falcon3-10B-Base`.
- Starting commit: `f5f421b08c386602772fad6584e961b260a9e588`.
- Stage-owned code: `tt/optimized_decoder.py`, `tests/test_optimized_decoder.py`,
  `doc/optimized_decoder/`, and the necessary `doc/context_contract.json` update.
- Hardware: one Blackhole P300c selected through the node's existing
  `TT_VISIBLE_DEVICES=2,3` reservation and a `1x1` TTNN mesh.
- Representative layer: 20. All 40 layers are the same dense Falcon3 decoder kind.
- Structural precedent: the repo's earlier Falcon3-7B optimized-decoder pattern at
  commit `36f41b593c4` was adapted to the 10B shapes, then retuned and validated
  independently with the evidence below.
- Independent stage review: the first fresh review returned `more-work-needed`
  for smaller DRAM geometry, fidelity-on-selected-topology, human-readable
  profiler-table, and theoretical-roofline gaps. Those findings were remediated
  below. A second fresh, read-only rereview returned `clean-pass` with no required
  work, no hard-check gaps, and no blocking concerns.

## Functional baseline

Command:

```bash
python -m pytest -q models/autoports/tiiuae_falcon3_10b_base/tests/test_functional_decoder.py -s \
  --junitxml=models/autoports/tiiuae_falcon3_10b_base/doc/optimized_decoder/logs/functional_baseline_pytest.xml
```

Three tests passed. Synthetic PCCs were 0.99893845 at prefill length 17,
0.99897713 at length 128, 0.99991646/0.99991996 for K/V cache,
0.99991428 for the decode query, 0.99991628/0.99991987 for post-decode K/V,
and 0.99911663 for decode output. Genuine layer-20 PCCs were 0.99879820
prefill and 0.99851787 decode. The acceptance bar inherited from the functional
stage is 0.99 for meaningful real-weight checks.

The functional topology used BF16 interleaved/default layouts, packed QKV,
separate gate/up matmuls with SiLU fused into the multiply, composite SDPA,
an explicit decode mask, and BF16 KV caches.

## Operation-topology audit and graph rewrite

The audit preceded precision and local program-config tuning. There are no
collectives because this stage is single-device.

| Path region | Current/repeated work | Candidate | Action and evidence |
|---|---|---|---|
| Input norm -> QKV | Interleaved residual, norm, one packed same-input QKV matmul | Advisor L1 block/width-sharded chain and 11x8 1D matmul | Applied as the first executable candidate. Full report-layout variants took 0.816962-0.819488 ms decode versus 0.793291 ms for selected DRAM-sharded decode. |
| Q/K/V projection | Already one packed projection | Split Q/K/V would add two weight reads and launches | Retained packed QKV; no structural split was introduced. |
| RoPE/cache/attention | Head transform, exact RoPE-row slices, paged cache updates, SDPA | Composite causal SDPA without material decode mask | Removed the explicit mask. Masked advisor candidate was 2.622968 ms versus 0.803918 ms maskless. Paged semantics stayed correct. |
| Head concat -> O -> residual | Head-layout conversion, O matmul, residual add | Advisor width-sharded output/residual chain; DRAM-sharded O | Both tried. Selected DRAM-sharded O as part of the whole-layer winner; its boundary reshard is required by the head helper and residual grid. |
| Post-attention norm -> gate/up | Two matmuls read the same normalized activation | Packed gate+up with on-device split versus separate matmuls with fused SiLU multiply | Both families were made runnable. Packed DRAM BFP4 took 0.968070 ms whole-layer versus 0.799328 ms separate; packed advisor took 0.855626 ms versus 0.803918 ms. Kept separate. |
| Gate product -> down | Product grid differed from down input | Align product/down shard or reshard | Aligned working shard removed the extra conversion and improved 0.804331 to 0.799328 ms in the 48-core control. Retained. |
| MLP -> residual -> output | MLP and residual grids differ; public output is DRAM interleaved | Keep full advisor residual chain or phase-specific L1 grids | Phase-specific 24-core MLP won. Four decode reshards remain and are required at head/O, O/residual, residual/MLP, and MLP/residual boundaries. |
| Large prefill M | Explicit 2D configs could over-allocate L1; dense MLP intermediates grow with sequence | Bounded automatic attention configs and on-device MLP chunks | First 6,528 run failed with a 41.7 MB requested QKV circular buffer. The revised path passes 6,528 without exposing a sequence-alignment restriction. |

Graph-rewrite work therefore retained packed QKV and composite SDPA, removed the
material decode mask, kept fused SiLU+multiply, aligned the gate-product/down
working layout, and compared the legal packed MLP rewrite against the best split
path. Each retained rewrite is covered by the real-weight PCC and repeat tests.

## Mandatory shard advisor gate

The advisor was run once this pass on the rewritten dense decode block. Required
bootstrap and capture command:

```bash
export TTMLIR_ADVISOR_HOME=/home/mvasiljevic/tt-mlir
source /home/mvasiljevic/tt-metal/.agents/skills/shard-advise/scripts/bootstrap.sh
export LD_LIBRARY_PATH=/home/mvasiljevic/tt-mlir/third_party/tt-metal/src/tt-metal/build_Release/lib:/home/mvasiljevic/tt-mlir/third_party/tt-metal/src/tt-metal/build/lib:${LD_LIBRARY_PATH:-}
export FALCON3_TT_METAL_ROOT=/home/mvasiljevic/tt-metal
export PYTHONPATH=/home/mvasiljevic/tt-metal:${PYTHONPATH:-}
ttnn-advise capture /home/mvasiljevic/tt-metal/models/autoports/tiiuae_falcon3_10b_base/doc/optimized_decoder/shard_advise/advise_falcon3_10b.py:decode \
  --out /tmp/falcon3-10b-advice
```

This ran from `/home/mvasiljevic/tt-mlir` in a separate shell. The report contains
24 operations, 21 final choices, one spill, and one unfixable
`nlp_concat_heads_decode` constraint because that op requires sharded input.

| Advisor recommendation | Applied first candidate | Final decision |
|---|---|---|
| QKV 11x8 1D, width-sharded output | Yes | Executable advisor path retained for regression coverage; DRAM-sharded selected. |
| O 11x9 1D, width-sharded output/residual | Yes | Whole report residual candidate measured 0.816962 ms; DRAM-sharded selected. |
| Gate/up 11x10 1D, width-sharded outputs | Yes | Exact BFP4 advisor candidate measured 0.803918 ms; selected 24-core DRAM path measured 0.793291 ms. |
| Down 11x9 1D, width-sharded output | Yes | Same comparison; DRAM-sharded selected. |
| Norm block-sharded on 11 cores; matmul inputs width-sharded on 48 cores | Yes | Full `report_sharded_inputs` chain measured 0.819488 ms; phase-specific DRAM-matmul input grids selected. |
| Residual width-sharded on 96 cores; gate output 103 and down input 90 cores | Yes | Report chain measured; selected 24-core MLP grid plus aligned down input was faster. |
| DRAM/interleaved SDPA output and concat-head constraint | Yes | Kept composite SDPA and adapted the required boundary around concat heads. |

The exact advisor BFP8 smoke passed with prefill/decode PCC
0.99863828/0.99930066. The advisor all-BFP4 path passed at 0.804132 ms on the
final rerun. Running the advisor was mandatory; retaining its DRAM-interleaved
weight strategy was rejected only after the executable, whole-layer comparison.

Initial advisor attempts were not treated as rejection evidence: the advisor
environment first missed the runtime library containing `moe_compute`, then lacked
`transformers`, and the tracer could not introspect a dynamic `memory_config`.
The library path was corrected, the capture helper was made independent of
Transformers, and a capture-only subclass supplied tracer-safe norm/movement
boundaries. Production code was not weakened for capture.

Artifacts and SHA256:

- `shard_advise/report.json`: `6bdc96ce3794968161a2d4fbdb225a35531f95093900c3793bd97e2f26ff16af`.
- `shard_advise/final_ir.mlir`: `659a5de831386e1e06fd5a48dbc6319aaaff15f7b15ee0b50ee303cd040bc736`.

## Precision and fidelity frontier

The real-weight length-31 frontier compared attention and MLP groups separately.
All candidates passed the 0.99 output/cache bar.

| Policy | Prefill PCC | Decode-31 PCC | Prefill ms | Traced decode ms | Decision |
|---|---:|---:|---:|---:|---|
| MLP BFP4, attention BFP8 HiFi2 | 0.99995804 | 0.99946462 | 4.262034 | 0.814243 | Reject: slower than LoFi. |
| MLP BFP4, attention BFP8 LoFi | 0.99995798 | 0.99946744 | 4.138178 | 0.812733 | Reject: slower than all BFP4. |
| Attention BFP4, MLP BFP8 | 0.99998389 | 0.99980810 | 6.915683 | 0.999123 | Reject: MLP precision dominates latency. |
| All projections BFP4 LoFi | 0.99994827 | 0.99932758 | 4.156834 | 0.804375 | Keep precision/fidelity policy. |
| All BFP4, attention HiFi2 | 0.99994793 | 0.99934014 | 4.288855 | 0.805026 | Reject: no accuracy need and slower. |

The profiler rows, rather than only constructor names, show BFP4_B weights and
LoFi fidelity for QKV, O, gate, up, and down. KV cache remains BFP8_B because the
cache PCC is above the functional bar while reducing persistent storage.

## Program, layout, and topology candidates

All listed decode numbers are traced warmed whole-layer latency on real layer-20
weights at batch 32 unless stated otherwise.

The 24/48-core labels below are the requested L1 activation-shard grid targets
used to derive legal `in0_block_w`/`per_core_N` geometry. Blackhole's
DRAM-sharded matmul program has a fixed worker mapping and Tracy reports 12
workers for each selected projection row; the labels do not claim 24/48 active
matmul workers.

| Candidate | Prefill ms | Decode ms | Decision/evidence |
|---|---:|---:|---|
| Advisor BFP8 | 5.390806 | 1.008006 | Correct, slower precision. |
| Advisor all BFP4 | 3.278730 | 0.803918 | Correct seed, beaten by DRAM path. |
| Advisor with explicit mask | 4.964503 | 2.622968 | Correct, mask movement dominates. |
| Advisor wider MLP blocks | 3.322293 | 0.831261 | Correct, slower. |
| Advisor report residual | 3.380935 | 0.816962 | Correct, slower full chain. |
| Advisor report-sharded inputs | 3.306724 | 0.819488 | Correct, extra boundaries lose. |
| Packed advisor MLP | 5.098298 | 0.855626 | Correct, split/layout cost loses. |
| Packed DRAM BFP4 MLP | 5.197918 | 0.968070 | Correct, loses to separate family. |
| DRAM MLP-only BFP4, 48-core target | 3.345298 | 0.838302 | Attention remained slower. |
| DRAM all-BFP4, 48-core target, unaligned down | 3.337532 | 0.804331 | Correct; alignment improvement retained. |
| DRAM all-BFP4, 48-core target | 3.364323 | 0.799328 | Strong control. |
| DRAM all-BFP4, 16-core target | 3.253775 | 0.809437 | Correct with gate/up `in0_block_w=6`, but slower. |
| DRAM all-BFP4, 12-core target | N/A | N/A | Gate/up `in0_block_w=8` requested 1,584,896 L1 bytes; only 1,572,864 available. |
| DRAM all-BFP4, 6-core target | N/A | N/A | Gate/up `in0_block_w=16` requested 2,861,824 L1 bytes; only 1,572,864 available. |
| DRAM all-BFP4, 24-core target | 3.322473 | 0.793291 | Selected. |
| Prefill grid 8 | 3.664826 | 0.812759 | Grid 11 family wins prefill. |
| Prefill `in0_block_w=1` | 4.058847 | 0.812885 | Reject. |
| Prefill `in0_block_w=8` | 3.775377 | 0.812851 | Keep for measured normal prompts. |

A 200-replay A/B reconfirmed 24 cores at 0.793096 ms versus 48 cores at
0.799102 ms. After first review, the legal smaller-target decode search added
16 cores (runnable but 2.06% slower), 12 cores (`in0_block_w=8`, exact L1
blocker), and 6 cores (`in0_block_w=16`, larger exact L1 blocker). Thus the
selected gate/up `in0_block_w=4` is the largest runnable tested DRAM-sharded
geometry; down already uses `in0_block_w=30`. Separately, prefill
`in0_block_w=16` failed at grid 11 with a 1,729,280-byte circular
buffer request versus 1,572,864 available. It was retried at grid 8 and failed
with 2,269,952 bytes requested; this is the precise L1 blocker, not a first-error
rejection. Unpacked DRAM BFP8 with the BFP4-oriented 24-core geometry also failed
repeatably because of an L1 circular-buffer clash at the preserved context.
The production default is precision-aware: BFP4 MLP weights select 24 cores,
while BFP8/BF16 select 48; advisor BFP8 and batch-1 DRAM BFP8 controls pass.

### Review-closure fidelity and final-default rerun

The first review also found that earlier fidelity controls used the advisor
topology. Real weights and recorded activations were therefore rerun on the
selected DRAM-sharded 24-core-target geometry for 200 trace replays:

| Selected DRAM topology | Prefill PCC | Decode PCC | Prefill ms | Decode ms | Decision |
|---|---:|---:|---:|---:|---|
| All BFP4, attention HiFi2, MLP LoFi | 0.99998254 | 0.99999666 | 3.271609 | 0.830321 | Reject: 4.64% slower decode than selected. |
| All BFP4, attention LoFi, MLP HiFi2 | 0.99998229 | 0.99999667 | 3.508560 | 1.116101 | Reject: 40.66% slower decode than selected. |
| All BFP4, attention/MLP LoFi (selected) | 0.99998229 | 0.99999667 | 3.278021 | 0.793496 | Keep. |

The final gate was then rerun with 11 timed prefill samples and 100 traced decode
replays per candidate:

```bash
FALCON3_RUN_FINAL_PERF=1 \
FALCON3_RESULTS_DIR=models/autoports/tiiuae_falcon3_10b_base/doc/optimized_decoder/results/final \
python -m pytest -q \
  models/autoports/tiiuae_falcon3_10b_base/tests/test_optimized_decoder.py::test_warmed_prefill_and_traced_decode_candidates \
  models/autoports/tiiuae_falcon3_10b_base/tests/test_optimized_decoder.py::test_batch1_traced_decode_candidates -s \
  --junitxml=models/autoports/tiiuae_falcon3_10b_base/doc/optimized_decoder/logs/final_review_rerun.xml
```

Both tests passed. At batch 32, final functional/selected values are
4.192523/3.278021 ms prefill and 4.201479/0.793496 ms traced decode. Selected
also beats advisor BFP4 (0.804016), 48-core target (0.799102), 16-core target
(0.809589), attention HiFi2 (0.830651), and MLP HiFi2 (1.116212). At batch 1,
selected is 0.668364 ms versus 0.675394 (48), 0.684719 (16), 0.706141
(attention HiFi2), and 0.991401 ms (MLP HiFi2).

## Correctness, capacity, and determinism

The post-review final ordinary module command produced 5 passes and 10 intentional
manual skips in 42.89 seconds:

```bash
python -m pytest -q models/autoports/tiiuae_falcon3_10b_base/tests/test_optimized_decoder.py -s \
  --junitxml=models/autoports/tiiuae_falcon3_10b_base/doc/optimized_decoder/pytest_results.xml
```

The passing default tests prove implementation ownership/no runtime host fallback,
a BF16 layout control, the exact advisor candidate, selected non-aligned/cache and
eight-repeat semantics, and genuine layer-20 prefill/decode. Manual evidence tests
cover capacity, candidate sweeps, precision, final batch-32 and batch-1 trace
latency, diagnostics, and Tracy.

The optimized capacity command was:

```bash
FALCON3_RUN_CAPACITY=1 python -m pytest -q \
  models/autoports/tiiuae_falcon3_10b_base/tests/test_optimized_decoder.py::test_selected_decoder_context_capacity -s \
  --junitxml=models/autoports/tiiuae_falcon3_10b_base/doc/optimized_decoder/logs/context_capacity_6528_retry.xml
```

It passes at batch 32 and logical sequence length 6,528 after the bounded large-M
fix. Lengths 17, 31, and 128 prove public non-aligned and aligned semantics; no
`seq_len % chunk == 0` precondition exists.

## Tracy and tt-perf-report

Profiler and watcher were separate hardware runs. Tracy command:

```bash
FALCON3_RUN_PROFILE=1 \
FALCON3_RESULTS_DIR=models/autoports/tiiuae_falcon3_10b_base/doc/optimized_decoder/results/final \
timeout 1800 python -m tracy -r -p -v \
  -o models/autoports/tiiuae_falcon3_10b_base/doc/optimized_decoder/tracy/dense_layer \
  -m pytest models/autoports/tiiuae_falcon3_10b_base/tests/test_optimized_decoder.py::test_profile_selected_decoder
```

The stable, line-ending-normalized `ops.csv` SHA256 is
`40702d1a10f3e022e3889d8b29a66386708df48a1c516abc3a03f47246cdfbc3`.
The review remediation regenerated actual human-readable, advice-enabled tables
from the preserved raw capture (not stdout from `--csv` mode):

```bash
tt-perf-report tracy/dense_layer/ops.csv \
  --start-signpost PERF_PREFILL --end-signpost PERF_PREFILL_END \
  --tracing-mode --no-color --no-summary > tracy/dense_layer/prefill_perf_report.txt
tt-perf-report tracy/dense_layer/ops.csv \
  --start-signpost PERF_DECODE --end-signpost PERF_DECODE_END \
  --tracing-mode --no-color --no-summary > tracy/dense_layer/decode_perf_report.txt
```

Their normalized SHA256 values are
`c323e593010766cdaa6803ae8e82b6a150dc0efc9a68b8e9dafdbd1115ea3261`
and `937f50b8a3b0ec9a46f176da1260e7c9fc7d8b820c5882b9705cfe801ce3973b`.
The selected default's hot path and configs did not change during review
remediation—only additional legal constructor policies/geometries and their tests
were added—so the capture remains the final selected runtime path.

Phase reports use the `PERF_PREFILL` and `PERF_DECODE` signposts. Prefill contains
215 rows/iteration: 2.6860885 ms kernels,
1.1974355 ms gaps, 3.883524 ms modeled total, and 4.0239859 ms wall. Decode has
42 rows/iteration: 0.7818333 ms kernels, 0.0327433 ms gaps, 0.8145767 ms modeled,
and 0.8261750 ms wall. The small wall-to-modeled decode difference is residual
TTNN trace-replay accounting, not a host tensor fallback.

The theoretical batch-32 layer-decode floor is calculated directly from mandatory
bytes. The five projection matrices contain 237,502,464 elements. BFP4 tiles use
576 bytes for 1,024 elements, so weights occupy 133,595,136 bytes. SDPA reads K
and V for 32 users, four KV heads, 32 tile-padded positions, and head dimension
256: 2,097,152 BFP8 elements at 1,088 bytes/tile, or 2,228,224 bytes. The total
135,823,360 bytes divided by Blackhole P300c's 512 GB/s aggregate DRAM bandwidth
is 0.26528 ms. That theoretical floor is 32.1% of 0.826175 ms wall, consistent
with the report's 29.7% modeled overall result once non-matmul rows are included.

The report's repeated prefill advice was to try L1 input and noted that
`in0_block_w=8` / output subblock `1x3` were reasonable; prefill uses interleaved
inputs because the full advisor-sharded chain lost, and the grid/block sweep is
recorded above. Its HiFi advice was evaluated in the precision frontier. Decode
reported no larger output subblock for 15 rows and suggested HiFi for accuracy;
LoFi/HiFi2 were compared at identical precision, and LoFi passed with lower
latency. Dominant decode matmuls reached 33-56% modeled DRAM bandwidth individually.

Final runtime audit: no `torch`, `from_torch`, `to_torch`, or CPU fallback occurs
inside measured `prefill_forward`/`decode_forward`. Prefill uses DRAM interleaved
activations and on-device slice/concat for long prompts. Decode uses L1 sharded
working tensors. Four necessary reshards remain, and RoPE uses exact one-row
untilize/slice/tilize operations; no avoidable tilize/untilize or reshard was found.

## Watcher and device health

Watcher command:

```bash
TT_METAL_WATCHER=10 python -m pytest -q \
  models/autoports/tiiuae_falcon3_10b_base/tests/test_optimized_decoder.py::test_selected_decoder_semantics_cache_and_repeated_decode -s \
  --junitxml=models/autoports/tiiuae_falcon3_10b_base/doc/optimized_decoder/logs/watcher_selected_semantics.xml
```

The post-review rerun passed in 25.49 seconds. `watcher.log` contains normal attach/detach data and no
error, assert, fatal, hang, or timeout signature; SHA256 is
`53c0fba0ee9a082ad6fbfaae37e207daba0d35908650adab1cc8ac3ab884464b`.
The allocated devices were healthy afterward.

An early health-check attempt accidentally replaced the inherited reservation
with devices 0 and 1, causing mesh-mapping failures. Bounded reset/lock cleanup did
not make that unallocated board usable; restoring the original device visibility
immediately passed. No profiler/watcher retries were performed on the wrong board.

## Optimize checklist

- [x] Functional checks pass against implementation-owned optimized methods; no functional fallback.
- [x] Prefill/decode PCC, paged cache behavior, non-aligned lengths, and eight-repeat determinism pass at the functional bar.
- [x] Warmed prefill and traced warmed decode before/after are recorded for final code at batch 32 and batch 1.
- [x] Runtime fallback audit is clean; measured methods contain no host tensor conversions.
- [x] Separate Tracy/tt-perf-report and `TT_METAL_WATCHER=10` runs are preserved.
- [x] Decode uses width-sharded L1 working activations across material sub-blocks; phase boundaries are justified by whole-chain measurements.
- [x] Prefill uses DRAM-interleaved activations, tuned 2D configs, and bounded large-M handling.
- [x] Operation-topology audit and graph rewrite were completed before knob tuning.
- [x] Shard advisor ran this pass; required artifacts exist; every recommendation was seeded and applied/rejected above with evidence.
- [x] Final code was compared against functional, advisor, packed, residual-chain, explicit-mask, precision, core-count, grid/block, and strongest correct candidates.
- [x] Final default was rerun after review with 11 prefill samples and expanded controls; headline values use the stable 3.278021/0.793496 ms selected run.
- [x] Profiler rows prove BFP4_B/LoFi reached QKV, O, gate, up, and down.
- [x] Composite SDPA and packed QKV are retained; explicit decode mask was removed.
- [x] Packed and split MLP families were compared including split/layout/elementwise costs; separate wins.
- [x] Important ops have explicit phase-specific memory, program, and compute-kernel configs.
- [x] Dominant matmul roles were swept across advisor/DRAM layouts, 6/12/16/24/48-core-target geometries, legal block sizes, BFP4/BFP8/BF16, and LoFi/HiFi2; smaller failures have exact L1 evidence.
- [x] Attention precision was swept separately from MLP precision; all-BFP4 LoFi wins and passes real-weight PCC.
- [x] Gate/up/down BFP4 LoFi was tried before lower-priority prefill tuning and retained.
- [x] Shard grids divide tiled tensor dimensions; the 24-core target is deliberately selected over runnable 16/48 controls and exact 6/12 L1 blockers.
- [x] Dominant decode matmuls use DRAM-sharded weights; advisor DRAM-interleaved controls were measured.
- [x] Avoidable reshard/layout work was removed; remaining single-device reshards are named and measured.
- [x] Reduced precision/fidelity experiments use genuine weights and HF-produced layer-20 activations.
- [x] Performance accounting reconciles a 0.26528 ms bytes-derived roofline, device kernel/gap time, modeled total, and wall time.
- [x] Batch capability is preserved: batch 1 is optimized and batch 32 correctness/performance/capacity pass.
- [x] Context contract was updated for BFP8 KV-cache/layout changes without reducing 6,528-token capability.
- [N/A] Multi-device collective families, lower-movement collective residuals, fused matmul-CCL, and persistent CCL buffers: single-device stage, no collective rows.
- [N/A] MoE routed sparse path: Falcon3-10B-Base is dense.
- [N/A] LM head, sampling, token feedback, qualitative generation, full-model `perf_summary.json`, and vLLM checks: explicitly outside decoder-layer scope.

## Failure and anomaly ledger

- BF16 batch-1 initially used a physical one-row shard, which violated tile shard
  constraints. Padding the physical decode head shard to 32 rows fixed it; logical
  batch remains 1 and the retry passes.
- Explicit prefill configs initially exceeded L1 at sequence 6,528. Automatic
  bounded large-M attention config selection plus existing on-device MLP chunking
  passes the preserved capacity.
- `in0_block_w=16` was retried on two grids and rejected only after both exact L1
  failures, as described above.
- First review correctly found that those were prefill failures, not decode
  blockers. Decoder-specific 12- and 6-core-target attempts then exercised
  gate/up `in0_block_w=8` and `16` and failed with 1,584,896 and 2,861,824 bytes
  requested versus 1,572,864 available; the intermediate legal 16-core target
  passed but was slower than 24.
- First review also found that fidelity had only been crossed on the advisor
  topology and that text reports contained `--csv` chatter. Selected-DRAM
  attention/MLP HiFi2 controls now pass but lose, and actual rendered advice
  tables plus theoretical bytes accounting replace the incomplete artifacts.
- A BFP8 DRAM test incorrectly reused the BFP4 24-core geometry and hit a repeatable
  L1 clash. Precision-aware automatic geometry now uses 48 cores for BFP8/BF16;
  the controls pass.
- The final ordinary pytest process exited successfully after reporting nanobind
  reference-leak diagnostics during interpreter teardown. All TT devices closed
  normally and subsequent health checks passed; no tensor/runtime failure occurred.
- The advisor environment/capture failures were fixed until the mandatory capture
  completed; none was used to reject its layouts.
- No unresolved correctness, performance, watcher, profiler, advisor, or capacity
  gate remains. `$autofix` was not needed because each discovered defect was
  isolated and repaired within this pass.

## Local commits

- Starting point: `f5f421b08c386602772fad6584e961b260a9e588`.
- Optimized decoder implementation, tests, and evidence:
  `50aeb2605ce28fd427d9d8584701a201e56e2d42`.
- A docs-only follow-up records the implementation commit SHA. Nothing was pushed.

## Artifact hashes

- HF activation tensor: `31fec043b5fc5a822611ac5398f439878640d0f2f5e8a87a3c47b729ea72128d`.
- HF activation metadata: `d07a65310355a9e44de7ab02b93ca420c36a7a58d2f232215b93b3018116c57f`.
- Final batch-32 JSON: `c10a020fa4b5af4face7630efcb4e6cc7d2a8e22b3710ea7fd2a6af47d9bb79c`.
- Final batch-1 JSON: `6c7b7f5d7430a04620963ab32589b1d01493a47c8cca2a98bd0aa3d380ef88f8`.
- Extended 16-core-target JSON: `2331f40799826a32bfe8ade1e9d14e15bc91d8fdd6ba574b707a7ec11ebe8769`.
- Selected-DRAM fidelity JSON: `a49846c52ff44a1e7dd96e426ca46c6092379a050fcc82b0d76c200f54d9b4b6`.
- Profile summary including theoretical roofline: `2831e22f86f3ecd53f82c5060ed54517f55a667556d0549d8444c6f855a1a0d2`.
- Watcher: `53c0fba0ee9a082ad6fbfaae37e207daba0d35908650adab1cc8ac3ab884464b`.
- Tracy operations: `40702d1a10f3e022e3889d8b29a66386708df48a1c516abc3a03f47246cdfbc3`.

## Limitations

This is a single optimized decoder layer. It does not establish stacked-model
resource behavior, multichip layouts, end-to-end generation quality, full-model
accuracy, serving throughput, LM-head/sampling performance, or vLLM readiness.
Those are deliberately deferred to their later stages, not silently implemented
or claimed here.
