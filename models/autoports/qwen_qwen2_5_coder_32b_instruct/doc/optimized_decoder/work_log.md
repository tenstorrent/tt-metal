# Optimized-decoder work log

## Scope and provenance

- Repository: `/home/mvasiljevic/tt-metal`
- Branch: `mvasiljevic/model/qwen-qwen2.5-coder-32b-instruct`
- Starting HEAD: `9e173e1c11bc`
- Model: `Qwen/Qwen2.5-Coder-32B-Instruct`
- Hardware: one device from a paired Blackhole p300c board, 11x10 worker grid,
  eight DRAM banks, firmware bundle 19.8.0
- Stage-owned paths: `tt/optimized_decoder.py`,
  `tests/test_optimized_decoder.py`, and `doc/optimized_decoder/**`
- Out of scope: multichip decoder, full model, embeddings, final norm, LM
  head, sampling, generator, vLLM, and serving
- Final stage checkpoint SHA: recorded in the final provenance commit below

The functional decoder is the semantic and accuracy floor. It supports one
dense layer on a 1x1 mesh, batch 1-32, arbitrary positive logical prefill
lengths up to the cache bound, persistent paged KV caches, and single-token
decode. All 64 source blocks are the same dense `Qwen2DecoderLayer` kind;
layer 32 is the representative middle layer used for real-weight evidence.

## Operation-topology audit

This audit was completed before local program tuning. The “before” column is
the measured functional path; “action” is the final graph-rewrite decision.

| Region | Before/current ops | Opportunity | Action and evidence |
| --- | --- | --- | --- |
| input norm | DRAM RMSNorm | carry decode residual in sharded L1 | Applied 11-core block-sharded RMSNorm with HiFi2/FP32 accumulation. |
| QKV | one packed `linear` plus bias | retain same-input Q/K/V fusion; tune placement | Kept packed QKV. Bias remains a separate device add because including bias in the BFP8 linear materially damaged isolated Q/K/V PCC; separate add gives whole-QKV PCC 0.999995. |
| heads/RoPE | head creation, RoPE, logical Q/K/V slices | remove movement without changing padded-head semantics | Retained dedicated head/RoPE ops. The 40-Q/8-KV logical slices are required after tile padding for 5:1 GQA. |
| KV and attention | two cache updates, SDPA | fused cache update, explicit composite program | Applied `paged_fused_update_cache` and explicit prefill/decode SDPA programs. Cache is BFP8; update inputs remain BF16. |
| concat/output | concat plus output matmul and residual add | sharded output/residual chain | Applied advisor 1D output projection into 80-core width-sharded L1, then sharded residual add. Dedicated concat remains because its input-layout contract is specialized. |
| post-attention norm | DRAM add then DRAM RMSNorm | keep residual/norm on device in L1 | Applied width-sharded residual to 11-core block-sharded RMSNorm. |
| MLP gate/up | two same-input matmuls, standalone SiLU, multiply | packed projection and adjacent-op fold | Applied packed gate/up plus on-device split and fused `mul(..., SILU)`; compared against tuned split DRAM-sharded family. |
| MLP down | DRAM intermediate and default matmul | phase-specific shard and explicit 1D program | Applied 87-core down-input shard, 80-core output, explicit program and HiFi2 compute config. |
| final residual | DRAM add | preserve sharded boundary until return | Applied 80-core width-sharded add; only the model-visible return converts to interleaved DRAM. |
| prefill | default large matmuls, DRAM activations | explicit 2D programs, internal padding | Applied sequence-aware 2D programs at the measured seq-17 geometry. Above 32 tiled M rows the explicit fused-batch form exceeded its per-core output-CB contract, so TTNN's multi-core auto program is used; seq 33 is correct. Public non-aligned lengths remain valid. |
| host/device crossings | no runtime host fallback | preserve | Static test forbids runtime `torch`, `from_torch`, `to_torch`, tilize/untilize, and functional-decoder calls. Profiler shows 44 device ops and no host ops in decode. |
| collectives | none: dense single-device stage | no applicable CCL/fusion/persistence work | Not applicable. No collective is present or introduced. |

Graph-rewrite classes completed before knob tuning:

1. Dedicated/composite ops: fused paged cache update and prefill/decode SDPA.
2. Structural merge: packed gate/up projection with on-device split.
3. Adjacent fold: SiLU folded into the gate/up binary multiply.

The kept graph is accepted on whole-path PCC and traced latency, not its lower
op count alone.

## Functional baseline

Real Qwen checkpoint layer 32, deterministic recorded activation:

| Path | Output PCC |
| --- | ---: |
| prefill, seq 17 | 0.9987814814 |
| decode, position 17 | 0.9989285759 |

Batch 32, seq-17 prefill, 50 warmed iterations on the same device/harness:

| Measurement | Functional |
| --- | ---: |
| warmed prefill | 83.2725586 ms |
| traced decode mean | 82.3735146 ms |
| traced decode minimum | 82.3412738 ms |

The functional path is therefore the PCC bar and the before-performance row.

## Correctness and semantic preservation

### Isolated QKV diagnosis

| Check | PCC |
| --- | ---: |
| sharded RMSNorm | 0.9999927 |
| Q projection | 0.9999956 |
| K projection | 0.9999945 |
| V projection | 0.9999893 |
| fused QKV result | 0.9999950 |

This isolated test caught the initially incorrect bias placement. The proven
fix—BFP8 matmul followed by BF16 device bias add—is retained in the final
runtime and its regression test.

### Final default results

| Weights / geometry | Path | Output PCC | K-cache PCC | V-cache PCC |
| --- | --- | ---: | ---: | ---: |
| real layer 32, batch 1 | prefill seq 17 | 0.9992013 | 0.9998955 | 0.9998086 |
| real layer 32, batch 1 | decode step 0 | 0.9992965 | preserved | preserved |
| real layer 32, batch 1 | decode step 1 | 0.9992799 | preserved | preserved |
| real layer 32, batch 1 | decode step 2 | 0.9993356 | preserved | preserved |
| real layer 32, batch 32 user 0 | prefill seq 17 | 0.9992139 | 0.9998958 | 0.9998122 |
| real layer 32, batch 32 user 0 | decode position 17 | 0.9995762 | preserved | preserved |
| synthetic, batch 32 | prefill seq 17 | 0.9997515 | 0.9998068 | 0.9998099 |
| synthetic, batch 32 | prefill seq 33 | 0.9998470 | 0.9998575 | 0.9998630 |
| synthetic, batch 32 | decode position 17 | 0.9998394 | 0.9998501 | 0.9998534 |

Every meaningful layer kind clears the functional 0.99878 prefill and 0.99892
decode bars. The optimized results are slightly higher because the selected
program/layout family changes accumulation grouping while retaining HiFi2 and
BF16 outputs. Three real-weight decode steps and a duplicate replay are
bitwise deterministic.

The public contract has no `seq_len % chunk == 0` restriction. Sequence 17
and 33 coverage proves non-aligned logical lengths; padding and slicing are
internal. Cache fill and append are compared against Hugging Face rotated K/V.

`doc/context_contract.json` is unchanged. The selected implementation does
not add a larger activation or public alignment constraint and reduces weight
and cache storage versus the BF16 functional stage, so it does not reduce the
recorded one-device batch-32 capability of 3,999 tokens.

## Precision, fidelity, packing, and geometry search

Candidate PCC is from real layer 32 unless noted. A candidate below either
functional output bar is rejected before performance ranking.

| Candidate | Prefill PCC | Decode PCC | Decision / evidence |
| --- | ---: | ---: | --- |
| BF16 weights/cache, HiFi4, unfused control | 0.999854 | 0.999892 | Correct control; slower/higher-memory policy, not selected. |
| BFP8 weights/cache, HiFi2 | 0.999227 | 0.999379 | Correct selected precision family. |
| BFP8 weights/cache, LoFi | 0.997911 | 0.998687 | Rejected below both functional bars. |
| BFP4 MLP, LoFi, 32 cores | 0.993511 | 0.992756 | Rejected on real weights. |
| BFP4 gate/up, LoFi, 32 cores | 0.995697 | 0.994972 | Rejected on real weights. |
| BFP4 attention, LoFi, 32 cores | 0.994837 | 0.996778 | Rejected on real weights. |
| packed BFP4 gate/up, LoFi, 32 cores | 0.995697 | 0.994972 | Rejected; packing does not recover precision. |
| BFP4 MLP, LoFi, 16 cores | 0.993511 | 0.992756 | Required cross-geometry repeat; same failing result. |
| BFP8, HiFi2, 16-core DRAM-sharded | 0.999227 | 0.999379 | Correct; 2.1929 ms batch-1 traced decode is slower than the selected 40-core packed path at 2.1503 ms. |

Attention and MLP policies were independently changed, so the attention BFP4
failure is not inferred from the MLP failure. LoFi and HiFi2 were compared at
BFP8; only HiFi2 passes the real-weight accuracy bar. BFP4/LoFi was repeated
at 16 and 32-core material geometries as required.

### Whole-layer topology candidates

| Family | Key configuration | Traced decode | Correctness | Decision |
| --- | --- | ---: | --- | --- |
| functional | BF16/default DRAM | 82.3735 ms | functional bar | baseline |
| split gate/up DRAM-sharded | BFP8/HiFi2, 32 cores | 2.3240 ms | 0.999227 / 0.999379 prefill/decode | correct, slower |
| packed gate/up DRAM-sharded | BFP8/HiFi2, gate/up 40 cores | 2.2906 ms | 0.999214 / 0.999361 batch-32 prefill/decode | strongest prior correct candidate |
| first advisor split/block chain | 88-core block family | 11.748 ms | correct | rejected on whole-layer latency |
| final advisor packed family | explicit 1D programs below | **1.9410 ms** retained final rerun | 0.999214 / 0.999576 real batch32 | selected for batch 32 |

Packed wins after accounting for its sharded-to-interleaved spill, split,
fused activation/multiply, reshard into down, and down projection. The smaller
batch default remains the 40-core DRAM-sharded packed family because the
advisor capture geometry is specifically batch 32. The paired batch-1 results
confirm that decision:

| Batch-1 family | Prefill | Traced decode mean | Traced minimum | Decision |
| --- | ---: | ---: | ---: | --- |
| functional BF16 | 3.1169 ms | 2.9028 ms | 2.8954 ms | before baseline |
| BFP8/HiFi2 split, 16 cores | 2.9485 ms | 2.1929 ms | 2.1901 ms | correct, 1.98% slower decode than final |
| BFP8/HiFi2 packed, 32 cores | 3.0088 ms | 2.1519 ms | 2.1501 ms | correct, marginally slower than final |
| BFP8/HiFi2 packed, gate/up 40 cores | **2.9489 ms** | **2.1503 ms** | **2.1423 ms** | selected default; 1.35x faster than functional decode |

### Dominant matmul program search

Final profiler percentages are percentages of 1.908 ms device decode time.

| Role | Time / share | Final program | Search and result |
| --- | ---: | --- | --- |
| QKV | 110 us / 5.7% | 11x7, 75 output cores, `in0_block_w=2`, `per_core_N=3`, subblock 1x3 | DRAM-sharded whole family measured at 2.291-2.324 ms layer latency; advisor interleaved family wins. Input shard is 2 tiles, so 2 is maximal. |
| output | 102 us / 5.3%, profiler `SLOW` | 11x8, 80 cores, block 2, `per_core_N=2`, subblock 1x2 | Larger block attempts share the 2-tile input-shard limit; block 2 is maximal. Sharded L1 output is retained into residual/norm. |
| packed gate/up | 821 us / 42.4% | 11x10, 108 cores, block 2, `per_core_N=16`, subblock 1x8 | DRAM 32/40-core split and packed families measured; advisor wins. Blocks 4, 8, 16, and non-power 10 were each run and rejected because the phase input shard is exactly 2 tiles. |
| down | 583 us / 30.1%, profiler `SLOW` | 11x8, 87-core input to 80-core output, block 2, `per_core_N=2`, subblock 1x2 | Blocks 4/8/16 fail shard divisibility; non-power 5 and 10 fail total tiled K=864 divisibility. The intersection of shard-width (10 tiles) and total-K divisors is {1,2}; 2 is exhaustive maximum. |

The larger-block experiments were continued past the first QKV validation
error by isolating the gate/up and down roles. No non-power-of-two value is
legal for the final dominant shards. A wider working-shard family was already
measured as the 32/40-core DRAM-sharded candidates; its extra layout/weight
movement yields 2.291-2.324 ms versus 1.941 ms. Padding the 27,648-channel
down K would add semantically inert channels, weight storage, and slicing but
cannot remove the phase reshard; it was rejected after the legal wider-shard
family lost end to end.

Exact rejected-program outputs are retained under `validation/geometry/`,
rather than treating the first TTNN validation error as the conclusion.

## Mandatory shard-advisor hard gate

The advisor was run during this pass on the rewritten packed dense decode
graph, after graph rewrite and before the final local program search.

Setup and capture were executed in a separate shell as required by SETUP Part
B:

```bash
export TTMLIR_ADVISOR_HOME=/home/mvasiljevic/tt-mlir
source .agents/skills/shard-advise/scripts/bootstrap.sh
export LD_LIBRARY_PATH=/home/mvasiljevic/tt-mlir/third_party/tt-metal/src/tt-metal/build_Release/lib:/home/mvasiljevic/tt-metal/build_Release/lib
export PYTHONPATH="$PYTHONPATH:/home/mvasiljevic/tt-metal/python_env/lib/python3.12/site-packages:/home/mvasiljevic/tt-metal"
ttnn-advise capture \
  models/autoports/qwen_qwen2_5_coder_32b_instruct/doc/optimized_decoder/shard_advise/advise_qwen2_5_coder_32b.py:decode \
  --out /tmp/qwen25-coder32-advice-final
```

Advisor checkout: `/home/mvasiljevic/tt-mlir`, branch
`ttnn-jit-shard-advisor`, HEAD `3f8b9c0a2`. The capture reports 28 ops, 25
final choices, and one spill. The tracer did not understand `ttnn.split`, so
bounded split-as-static-slices handlers were added to the external advisor
checkout's interception and emit tracers; these are advisor tooling changes,
not stage-owned repository edits. Fused cache update was disabled only in the
capture script because the tracer lacks its wrapper; the shipping runtime
retains the faster, tested fused update.

Required artifacts and hashes:

| Artifact | SHA256 |
| --- | --- |
| `shard_advise/report.json` | `ba79ab0ab572c410ba8c1e07e63cc07fa872d0abefa21c115fdf424f15f0fa0f` |
| `shard_advise/final_ir.mlir` | `a6a1116faa4c947c63229d0f41d2d5a5fef50628beaf2c526119561fd4ab9c11` |
| `shard_advise/report.txt` | `68eeaac306bfcd2769a476e24a87bec255a2d61375fee3609ba2a5a6ccdbaf73` |

### Applied versus rejected advisor recommendations

| Recommendation | Result |
| --- | --- |
| 11-core block-sharded RMSNorm and phase-specific width reshards | Applied. |
| QKV 1D program on 11x7 with 75-core output | Applied exactly. |
| output projection on 11x8 with 80-core sharded residual | Applied exactly. |
| packed gate/up on 11x10 with 108-core output | Applied exactly. |
| spill packed gate/up output to interleaved L1 before static split consumers | Applied exactly. Omitting this spill ran but produced synthetic decode PCC 0.000351; applying it restored 0.999839. |
| multiply on 108 cores and down input reshard to 87 cores | Applied exactly. |
| down/output on 11x8 / 80 cores | Applied exactly. |
| advisor output revert to DRAM | Applied only at the model-visible decoder return; no immediate old-contract restore inside attention/MLP. |
| interleaved/DRAM attention helpers and unfixable concat | Retained where dedicated TTNN op contracts require them. Advisor itself reports concat unfixable because its constraint query requires sharded input; the shipping dedicated head helper is correct and the measured whole path wins. |
| keep every advisor field without local search | Rejected as a rule: 4/8/16 and non-power block trials, DRAM-sharded alternatives, and precision/fidelity candidates were measured. Advisor block 2 remains because it is both fastest and the exhaustive legal maximum. |

The first advisor-derived split chain was correct but slow (11.748 ms), so the
advisor was rerun on the rewritten packed graph. The final report's exact
spill and programs became the first candidate, passed PCC, and won latency.

## Trace correctness and hardware recovery

The first trace capture used warm position 17 and capture position 18. That
caused a program-cache miss inside capture and an unsupported write; aborting
the partially built trace then stalled device close. Following
`$tt-enable-tracing`, `$autotriage`, and `$tt-device-usage`, the run was
bounded, evidence was captured with `tt-triage`, stale evidence processes were
terminated, and the device was reset. GDB placed the stall in
`clear_expected_num_workers_completed`, confirming cleanup rather than model
math as the failure site. Evidence is in `AUTOTRIAGE.md` and `triage/`; the
raw 1.2 MB `tt-triage` capture is retained losslessly as
`triage/tt-triage.txt.gz` to satisfy the repository artifact-size gate.

The fix warms and captures the exact same position/program and refreshes the
same device input tensor before replay. Final trace checks prove:

- same input produces bitwise-identical replay output;
- refreshed device input changes the output;
- cache updates and output remain correct;
- host reads occur only after the timed replay window.

Final performance and watcher runs were serialized and kept separate. No
`TT_VISIBLE_DEVICES` override was used.

## Final performance and profiler evidence

Batch 32, seq-17 prefill, layer 32, 50 warmed repetitions:

| Path | Functional | Final default | Speedup |
| --- | ---: | ---: | ---: |
| warmed prefill | 83.2726 ms | 9.9975 ms | 8.33x |
| traced decode mean | 82.3735 ms | 1.9410 ms | 42.44x |
| traced decode minimum | 82.3413 ms | 1.9347 ms | 42.56x |

The final code path reproduces the selected candidate and beats the strongest
earlier correct traced-decode result, 2.2906 ms, by 15.26%.

Profiler collection command (separate from watcher):

```bash
QWEN2_5_CODER_32B_REAL_WEIGHT_DIR=<snapshot> \
QWEN2_5_CODER_32B_OPT_PERF=1 \
QWEN2_5_CODER_32B_PERF_DECODER=optimized \
QWEN2_5_CODER_32B_PERF_BATCH=32 \
QWEN2_5_CODER_32B_PERF_REPS=1 \
python_env/bin/python -m tracy -r -p -v \
  -o models/autoports/qwen_qwen2_5_coder_32b_instruct/doc/optimized_decoder/tracy/layer32 \
  -n final_advisor -m pytest \
  models/autoports/qwen_qwen2_5_coder_32b_instruct/tests/test_optimized_decoder.py::test_warmed_prefill_and_traced_decode_perf -q -s
```

Compact reports were generated with advice enabled and signpost windows:

```bash
python_env/bin/tt-perf-report <raw-ops-csv> \
  --start-signpost PERF_DECODE --end-signpost PERF_DECODE_END \
  --no-host-ops --csv decode_perf_report.csv
python_env/bin/tt-perf-report <raw-ops-csv> \
  --start-signpost PERF_PREFILL --end-signpost PERF_PREFILL_END \
  --no-host-ops --csv prefill_perf_report.csv
```

The compact artifacts are under `tracy/layer32/`; raw Tracy databases and raw
ops CSVs were removed after reduction.

Profiler conclusions:

- the profiled decode contains 44 device ops and 1.908 ms device time versus
  1.941 ms traced wall time in that collection run; the 0.033 ms (1.7%) gap is dispatch/replay accounting, not
  a host fallback;
- modeled overall DRAM roofline is 49.9%; packed gate/up and down dominate;
- measured rows prove BF16 x BFP8 -> BF16 at HiFi2 for every dominant matmul;
- QKV is 110 us at 333 GB/s (65.0% modeled DRAM), packed gate/up is 821 us at
  345 GB/s (67.4%), output is 102 us at 258 GB/s (50.3%), and down is 583 us
  at 243 GB/s (47.4%);
- advice to try DRAM-sharded weight programs was addressed by the 32/40-core
  split/packed families (2.324/2.291 ms), both slower than final 1.941 ms;
- `SLOW` output/down rows triggered the exhaustive block and working-shard
  search documented above; block 2 is their legal maximum under the winning
  layout.

### Cumulative decode contract

The profiler row interpretation is tied to the exact traced test contract,
not inferred from padded tensor shapes:

| Contract item | Final traced value | Evidence |
| --- | --- | --- |
| logical users / tile rows | 32 / 32 | `PERF_RESULT.logical_users` and `tile_padded_rows`; no padding-only users at batch 32 |
| KV cache | `[32, 8, 128, 128]`, BFP8, persistent DRAM-interleaved | allocation/test contract plus fused-update row 680 (`BFP8, BF16 => BFP8`, 21.407 us) |
| decode SDPA | 8x8 grid, `q_chunk_size=0`, `k_chunk_size=0`, no exp approximation | explicit `SDPAProgramConfig`; row 681 is 49.514 us, BF16 query and BFP8 cache to BF16 output |
| RMSNorm | 11x1 grid, block-sharded L1, block 1x15, subblock width 3, HiFi2 with FP32 accumulation | rows 658/688 are 7.780/7.696 us on 11 cores and L1 block-sharded input/output |
| attention residual | 80-core width-sharded L1 through output projection and add | output row 685 is 101.801 us; residual add row 686 is 1.288 us without a DRAM round trip |
| MLP residual | 80-core width-sharded L1 through final add | down row 696 is 583.156 us; final residual row 697 is 1.333 us before the model-visible return |
| dominant dtype/fidelity | BF16 x BFP8 -> BF16, HiFi2 | rows 660, 685, 690, and 696 |

This makes the per-user/cache/SDPA/norm/residual assumptions cumulative and
explicit. Batch 1 uses the same logical-user accounting with 32 tile-padded
rows, as recorded in its retained `PERF_RESULT` logs.

No avoidable torch/device crossing or host op is present in the measured
windows. The report does contain small implicit tilize/untilize rows generated
inside the position-table slices around rotary embedding (each below 0.4%);
there is no explicit runtime tilize/untilize call, and retaining the logical
Q/K slices is required for the padded 40-Q/8-KV GQA contract. Remaining
reshards correspond to explicit TTNN contracts: norm phase shards, head
creation/RoPE/concat, packed split spill, down phase shard, and model-visible
return.

## Validation commands

Default optimized correctness, non-aligned lengths, representative layer,
repeated decode, determinism, and real batch-32 advisor path:

```bash
QWEN2_5_CODER_32B_REAL_WEIGHT_DIR=/home/mvasiljevic/hf-cache/hub/models--Qwen--Qwen2.5-Coder-32B-Instruct/snapshots/381fc969f78efac66bc87ff7ddeadb7e73c218a7 \
python_env/bin/pytest \
  models/autoports/qwen_qwen2_5_coder_32b_instruct/tests/test_optimized_decoder.py -q -s
```

Watcher-clean run, intentionally separate from profiler:

```bash
TT_METAL_WATCHER=10 \
QWEN2_5_CODER_32B_REAL_WEIGHT_DIR=/home/mvasiljevic/hf-cache/hub/models--Qwen--Qwen2.5-Coder-32B-Instruct/snapshots/381fc969f78efac66bc87ff7ddeadb7e73c218a7 \
python_env/bin/pytest \
  models/autoports/qwen_qwen2_5_coder_32b_instruct/tests/test_optimized_decoder.py -q -s
```

Result: `5 passed, 2 skipped in 61.82s`. The skipped tests are the explicit
candidate and serialized performance opt-ins. `generated/watcher/watcher.log`
contains 556 lines and no assert, error, hang, timeout, illegal-access, or
corruption signature.

The final evidence-preservation rerun completed in 42.63s with the same five
passes and PCC values. Its complete output and device watcher dump are retained
as `validation/watcher_correctness.log` and `validation/watcher_device.log`.
The paired 50-repetition endpoints are retained as
`validation/functional_perf.log` and `validation/optimized_perf.log`; their
machine-readable `PERF_RESULT` lines are the final numbers reported above.
Batch-1 functional/final timing and the correct 16/32-core alternatives are
retained beside them. `validation/candidate_sweep.log` contains the real-layer
precision/fidelity PCC sweep, `validation/packed40_batch32_correctness.log`
ties the strongest prior performance result to batch-32 PCC, and
`validation/README.md` maps every retained primary artifact to its command.

Final 50-repetition optimized timing:

```bash
QWEN2_5_CODER_32B_REAL_WEIGHT_DIR=<snapshot> \
QWEN2_5_CODER_32B_OPT_PERF=1 \
QWEN2_5_CODER_32B_PERF_DECODER=optimized \
QWEN2_5_CODER_32B_PERF_BATCH=32 \
QWEN2_5_CODER_32B_PERF_REPS=50 \
python_env/bin/pytest \
  models/autoports/qwen_qwen2_5_coder_32b_instruct/tests/test_optimized_decoder.py::test_warmed_prefill_and_traced_decode_perf -q -s
```

## Optimize checklist

- [x] Functional checks pass against the independent optimized path.
- [x] Prefill/decode/cache PCC clears the functional bar for the only
  meaningful layer kind; material deltas are recorded.
- [x] Paged cache semantics, repeated decode, determinism, and trace input
  refresh/replay are verified.
- [x] Runtime fallback audit is clean; no functional fallback is imported or
  called.
- [x] Warmed prefill and traced decode before/after are reported; final default
  beats the best earlier correct candidate.
- [x] Operation-topology audit precedes tuning and records current ops,
  movement, projection packing, composite candidates, actions, and evidence.
- [x] `$graph-rewrite` applied dedicated/composite, structural, and adjacent
  folds with PCC and whole-layer latency evidence.
- [x] `$shard-advise` ran this pass on the rewritten packed dense graph;
  `report.json` and `final_ir.mlir` are saved and every applied/rejected
  recommendation is documented.
- [x] Decode activations are generally width/block-sharded in L1; prefill is
  DRAM-interleaved with explicit large 2D matmul programs.
- [x] Lower-movement residual chain is measured without immediate contract
  restoration; final attention/MLP residuals remain sharded until return.
- [x] Packed and split gate/up are compared including split/layout/binary/down
  costs; packed wins whole-layer latency.
- [x] Explicit memory, program, SDPA, and compute-kernel configs are used for
  material ops.
- [x] Dominant matmul core grids, DRAM/interleaved weight families, working
  shards, legal block widths, output subblocks, and precision/fidelity are
  searched with exact blockers for rejected values.
- [x] BFP8 LoFi/HiFi2 and independent attention/MLP BFP4 LoFi trials were run;
  BFP4 was crossed with 16/32-core material geometries.
- [x] Profiler rows prove final BF16 x BFP8 -> BF16 HiFi2 policy; compact
  advice-enabled artifacts are retained.
- [x] SDPA and specialized head/cache composite ops are used.
- [x] DRAM-sharded decode matmul families were measured and rejected only
  after correct whole-layer comparisons; advisor-interleaved wins batch 32.
- [x] Avoidable conversions are removed; remaining movements are tied to
  measured phase/op contracts.
- [x] Batch capability is preserved and batch 32 correctness is exercised;
  smaller batches have their own faster correct default.
- [x] Watcher-enabled correctness is clean and separate from Tracy.
- [x] Multi-device topology, CCL fusion, persistent CCL buffers, MoE sparse
  matmul, LM head, sampling, full-model `perf_summary.json`, and vLLM
  qualitative/serving metrics are not applicable to this single-device dense
  decoder-only stage.
- [ ] Fresh `$stage-review` clean-pass verdict and local stage checkpoint SHA
  are appended after independent review.

## Limitations

- Evidence is one representative layer, not an assembled 64-layer model; that
  is the explicit optimized-decoder stage boundary.
- The advisor capture script uses an unfused cache update only because the
  tracer lacks the fused wrapper; shipping correctness/performance uses the
  fused update.
- The paired p300c host reports a low `/dev/shm` warning at process startup,
  but device tests, trace, Tracy, and watcher all complete. No device fault is
  present after the final run.
- Full-model datatype frontier selection remains for `$datatype-sweep`; this
  stage selects the fastest precision policy that clears its decoder PCC bar.

## Review and commits

- Independent stage review: pending.
- Stage implementation/evidence commit: pending.
- Provenance/checklist follow-up commit: pending.
- No push is performed.
