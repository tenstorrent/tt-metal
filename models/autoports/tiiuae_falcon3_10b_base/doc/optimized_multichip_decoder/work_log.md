# Optimized multichip decoder work log

## Scope and provenance

- Model: `tiiuae/Falcon3-10B-Base`, representative dense layer 20.
- Stage start HEAD: `3c69719127e668656ec77396bb042b024ff37659`.
- Completed input checkpoint: `64e3199158e765e2558115bbfcc1ed2e4edcd68a`.
- Target: fixed 1x4 Blackhole p300c TP mesh, Ring topology, two links.
- Input-stage historical best: 2.771582920 ms warmed prefill and 0.576824090
  ms 100-replay traced decode at batch 32/sequence 17.
- Fresh stage-entry rerun: 3.003479447 ms prefill and 0.576603180 ms
  decode. This is the same source/configuration timed in this pass.
- Input PCC versus the optimized single-chip layer: prefill 0.999999505,
  decode 0.999999934, K 0.999995759, V 0.999998539.
- Preserved unrelated pre-existing edit:
  `.agents/skills/forge-functional-decoder-from-ir/SKILL.md`.
- Scope excludes embeddings, layer-stack assembly, final norm, LM head,
  generation, full-model work, serving, and vLLM.

Functional and benchmark TT jobs were serialized through
`scripts/run_safe_pytest.sh`; Tracy used the same `/tmp/tt-device.lock` through
`flock`. Watcher and Tracy profiling were run in separate processes. All
results use real layer-20 weights unless explicitly labeled synthetic.

## 2026-07-18: graph-first audit

Read `$optimize`, `$tt-device-usage`, `$graph-rewrite`, and `$shard-advise` in
full. Read the completed multichip source, profiler tables, accepted candidate
matrix, context contract, and prior AutoFix reports. `topology_audit.md` was
written before local program tuning.

The audit identified:

- same-input gate/up `[3072,6144]` rank-local matmuls that could be packed;
- explicit decode rotate-half operations that could use a dedicated RoPE op;
- two row-parallel collective boundaries where allocation/setup could be made
  persistent;
- an exit L1-to-DRAM plus next-entry DRAM-to-L1 pair that could be removed by a
  stack-native residual API;
- packed QKV already present and correctly rank-grouped;
- material reshards at phase changes, but no collective required between layers;
- prior fused matmul-CCL/lower-movement attempts already closed by source,
  hardware, and failed `$autofix` evidence.

The prior reduce-scatter/distributed-residual boundary was not dismissed on its
immediate API behavior: standalone and integrated variants, one/two links,
explicit synchronization, sub-device cleanup, and post-exit health were tested.
It reached PCC 0.999794 and a 1.00765x isolated speedup, but poisoned Ethernet
heartbeats after exit. Fused matmul-reduce-scatter hung; standalone matmul-RS
passed. Fused all-gather-matmul has TP4 receiver accounting hardcoded for four
transfers. `doc/multichip_decoder/AUTOFIX.md`, `AUTOTRIAGE.md`, and
`results/graph_rewrite_*` retain the repair loop and the failed AutoFix result.

## 2026-07-18: fresh shard advice and matmul retries

Ran `shard_advise/advise_falcon3_10b_tp_local.py` through the repository's
tt-mlir/tt-metal bootstrap. `pipeline.log` preserves the full compiler command,
environment, and mock validation; `report.json`, `report.txt`, and
`final_ir.mlir` preserve the result.

The exact advice used QKV 11x4/block 8, O 11x9/block 8, gate/up 11x9/block 2,
and down 11x9/block 8. It passed the real PCC gate but measured 2.931702 ms
prefill and 0.641203 ms decode, slower than the 0.576603 ms control. An O-only
hybrid measured 0.580691 ms. A wider O retry first failed because its requested
logical grid did not match the legal physical grid; after adapting it to exact
8x6 placement, it measured 0.577288 ms and still lost. These candidates remain
opt-in modes (`shard_advisor`, `advisor_o`, `advisor_o_48core`) for provenance;
the default stays `dram_sharded`.

The input stage had already swept exact-divisor core families: QKV 2/8/16, O
2/6/8/12/16/24/48, gate/up 8/12/16/24/32/48, and down 4/8/12/24. The selected
targets remain 4/2/24/8. The final profiler's matmul advice was therefore acted
on through fresh compiler programs, not merely recorded.

## 2026-07-18: packed projection family

Added rank-local packed gate/up weights and a single `[3072,12288]` projection.
The first complete candidate split in DRAM, passed real PCC, and measured
3.087969/0.578894 ms prefill/decode. Because the split/layout movement was
material, adapted it to L1-sharded unpack and retried; PCC passed and timing was
3.078781/0.584399 ms. Both lose to separate projections, so the default retains
separate gate/up matmuls. Packed QKV remains selected from the input stage.

## 2026-07-18: persistent collective family

Added `DecodeAllReduceResources`, containing one mesh-wide sub-device manager,
two global semaphores, and a persistent L1 intermediate buffer. Decode BF16
row-parallel reductions call `ttnn.experimental.all_reduce_async`; prefill and
BFP8 controls retain the legal non-persistent route.

An isolated five-by-100-replay trace measured 0.056970 ms default versus
0.022101 ms persistent, a 2.5777x speedup at PCC 0.9999974. A fabric smoke after
teardown passed. The first complete decoder integration measured 2.966838 ms
prefill and 0.528248 ms decode; the production-lifetime version reproduced
0.528400 ms decode and passed real PCC. Persistent resources became default.

## 2026-07-18: dedicated RoPE and output graph rewrites

Replaced the batch-32 transpose/repeat/slice/neg/concat/mul/add rotate-half
cluster with `ttnn.experimental.rotary_embedding`, preserving independent
per-user positions. Real PCC passed and traced decode reached 0.404839 ms.

The first heterogeneous batch-2 run showed that the dedicated output cannot be
placed directly in the sub-tile height-sharded head cache. This was adapted,
not rejected: tile-aligned batches use the dedicated op, while other valid
logical batches use the existing internally padded explicit route. The batch-2
heterogeneous retry passed. An exact final-graph explicit-RoPE control measured
0.514621 ms, proving the selected dedicated path is a 23.97% whole-layer win.

Removed the final L1 staging copy for unpadded public decode output. The first
small-batch broad-suite attempt exposed a physical/logical row mismatch; the
implementation now writes direct DRAM only when rows match and owns slice/L1
staging otherwise. Batch-32 PCC passed and decode reached 0.391468 ms; the full
unaligned/heterogeneous suite then passed.

## 2026-07-18: inter-layer residual contract

Added `decode_forward_to_residual`, `decode_forward_from_residual`, and
`materialize_decode_output`. The layer-stack contract is replicated values with
per-device BF16 L1 width-sharded `[1,1,32,3072]` on 32 residual cores. The
logical batch remains separate from internal tile padding.

A real two-layer trace, using two independently materialized layer-20 decoder
instances that share one persistent all-reduce pool, compares the entire old
DRAM boundary with the carried residual boundary:

| Boundary | Median, 2 layers/replay | PCC | Inter-layer collectives |
|---|---:|---:|---:|
| Public DRAM materialization/re-entry | 0.778980660 ms | control | 0 |
| Stack-native L1 residual | 0.710055172 ms | 1.0 | 0 |

Each value is the median of five samples of 100 trace replays. The sharded
boundary is 1.097132x faster, and does not restore the old DRAM contract inside
the measurement window. Full-model bringup must preserve this contract.

## 2026-07-18: dtype, fidelity, topology, and CCL families

All rows below use the final graph except the named variable. Five warmed
prefill samples and five 100-replay decode samples were captured per row.

| Candidate | Prefill ms | Decode ms | Outcome |
|---|---:|---:|---|
| BFP4/LoFi weights, BF16 CCL, Ring 2-link, persistent | 3.028828 | **0.391257** | selected |
| BFP4 attention HiFi2 | 2.915361 | 0.400512 | reject |
| BFP4 MLP HiFi2 | 3.023798 | 0.478435 | reject |
| BFP8/HiFi2 weights | 3.010753 | 0.493697 | reject |
| BF16/HiFi4 weights, adapted down=24 cores | 2.774082 | 0.689089 | reject |
| BFP8 CCL | 2.944967 | 0.449578 | reject |
| Linear BF16, 2 links | 3.044918 | 0.397344 | reject |
| Ring BF16, 1 link | 3.054426 | 0.408873 | reject |
| Ring BF16, 2 links, no persistent resources | 2.873771 | 0.437456 | reject |

The first BF16/HiFi4 down-matmul attempt exceeded L1: 2,003,712 bytes required
versus 1,572,864 available at eight cores. It was adapted to 24 cores and
remeasured successfully before rejection. Final PCC validates the selected
BFP4/LoFi policy. KV cache remains BFP8_B.

## Final reproduction commands

The environment variables below select the final defaults explicitly. Omitting
them yields the same default path.

```bash
stage_dir=models/autoports/tiiuae_falcon3_10b_base/doc/optimized_multichip_decoder
test_file=models/autoports/tiiuae_falcon3_10b_base/tests/test_multichip_decoder.py

FALCON3_RUN_MULTICHIP_PERF=1 \
FALCON3_MULTICHIP_RESULTS_DIR=$stage_dir/results \
FALCON3_MULTICHIP_PERF_FILENAME=final_default_perf.json \
FALCON3_MULTICHIP_PRECISION_POLICY=all_bfp4_lofi \
FALCON3_MULTICHIP_DECODE_MATMUL_MODE=dram_sharded \
FALCON3_MULTICHIP_PACKED_MLP=0 \
FALCON3_MULTICHIP_DECODE_ROPE_MODE=dedicated \
FALCON3_MULTICHIP_DECODE_OUTPUT_MODE=direct_dram \
FALCON3_MULTICHIP_PERSISTENT_DECODE_AR=1 \
FALCON3_MULTICHIP_CCL_DTYPE=bf16 \
FALCON3_MULTICHIP_TOPOLOGY=ring \
FALCON3_MULTICHIP_NUM_LINKS=2 \
scripts/run_safe_pytest.sh \
  $test_file::test_warmed_multichip_trace_performance -s -q \
  --junitxml=$stage_dir/logs/final_default_perf_exact.xml

FALCON3_RUN_MULTICHIP_BASELINE=1 \
FALCON3_MULTICHIP_RESULTS_DIR=$stage_dir/results \
FALCON3_MULTICHIP_PCC_FILENAME=final_default_pcc.json \
scripts/run_safe_pytest.sh \
  $test_file::test_multichip_directly_matches_single_chip_optimized_baseline \
  -s -q --junitxml=$stage_dir/logs/final_default_pcc_exact.xml

FALCON3_RUN_MULTICHIP_STACK_CONTRACT=1 \
FALCON3_MULTICHIP_RESULTS_DIR=$stage_dir/results \
scripts/run_safe_pytest.sh \
  $test_file::test_warmed_two_layer_residual_contract -s -q \
  --junitxml=$stage_dir/logs/final_two_layer_residual_contract_exact.xml

FALCON3_RUN_MULTICHIP_MAX_CONTEXT=1 \
FALCON3_MULTICHIP_RESULTS_DIR=$stage_dir/results \
scripts/run_safe_pytest.sh \
  $test_file::test_batch1_advertised_context_paged_cache_and_last_position \
  -s -q --junitxml=$stage_dir/logs/final_max_context_batch1.xml
```

Candidate rows used the same performance command plus one or more of:

```text
FALCON3_MULTICHIP_DECODE_MATMUL_MODE=shard_advisor|advisor_o|advisor_o_48core
FALCON3_MULTICHIP_PACKED_MLP=1
FALCON3_MULTICHIP_PACKED_UNPACK=dram|l1_sharded
FALCON3_MULTICHIP_DECODE_ROPE_MODE=explicit|dedicated
FALCON3_MULTICHIP_PERSISTENT_DECODE_AR=0|1
FALCON3_MULTICHIP_PRECISION_POLICY=all_bfp4_attention_hifi2|all_bfp4_mlp_hifi2|bfp8_hifi2|bf16_hifi4
FALCON3_MULTICHIP_CCL_DTYPE=bf16|bfp8
FALCON3_MULTICHIP_TOPOLOGY=ring|linear
FALCON3_MULTICHIP_NUM_LINKS=1|2
```

## Profiler and tt-perf-report

Watcher was disabled during profiling.

```bash
FALCON3_RUN_MULTICHIP_PROFILE=1 \
FALCON3_MULTICHIP_RESULTS_DIR=$stage_dir/results \
FALCON3_MULTICHIP_PROFILE_FILENAME=final_default_profile_wall_post_advisor.json \
timeout 1800 flock /tmp/tt-device.lock python_env/bin/python -m tracy -r -p -v \
  -o $stage_dir/tracy/final_default_post_advisor \
  -m pytest $test_file::test_profile_selected_multichip_decoder \
  --junitxml=$stage_dir/logs/final_default_profile_post_review.xml

raw_csv=$stage_dir/tracy/final_default_post_advisor/reports/2026_07_18_22_20_27/ops_perf_results_2026_07_18_22_20_27.csv
tt-perf-report $raw_csv \
  --start-signpost MULTICHIP_PREFILL --end-signpost MULTICHIP_PREFILL_END \
  --tracing-mode --no-color --csv $stage_dir/tracy/final_default_post_advisor/derived/prefill_perf_report.csv \
  --summary-file $stage_dir/tracy/final_default_post_advisor/derived/prefill_summary \
  > $stage_dir/tracy/final_default_post_advisor/derived/prefill_perf_report.console.log
tt-perf-report $raw_csv \
  --start-signpost MULTICHIP_DECODE --end-signpost MULTICHIP_DECODE_END \
  --tracing-mode --no-color --csv $stage_dir/tracy/final_default_post_advisor/derived/decode_perf_report.csv \
  --summary-file $stage_dir/tracy/final_default_post_advisor/derived/decode_summary \
  > $stage_dir/tracy/final_default_post_advisor/derived/decode_perf_report.console.log
```

Final decode has 45 device operations/replay and 381.793 us summed device time,
down from 68 and 518.003 us. Excluding one documented cross-iteration profiler
gap, op gaps fall from 82.751 to 35.600 us. Matmuls remain effectively flat
(136.863 us/replay); material CCL falls from 76.97 us RS+AG to 27.991 us async
all-reduce. DRAM reshape views are the remaining non-matmul group above 15%; the
direct output and residual-carry candidates are the attempted movement fixes.

## Final functional, fallback, watcher, and health gates

```bash
FALCON3_MULTICHIP_RESULTS_DIR=$stage_dir/results \
scripts/run_safe_pytest.sh $test_file -s -q \
  --junitxml=$stage_dir/logs/final_functional_suite_clean.xml

TTNN_CONFIG_OVERRIDES='{"enable_fast_runtime_mode": false, "throw_exception_on_fallback": true}' \
scripts/run_safe_pytest.sh \
  $test_file::test_synthetic_tp4_prefill_decode_smoke \
  $test_file::test_real_layer_paged_non_aligned_prefill_decode_cache_and_trace \
  -s -q --junitxml=$stage_dir/logs/final_no_fallback.xml

TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 \
scripts/run_safe_pytest.sh \
  $test_file::test_real_layer_paged_non_aligned_prefill_decode_cache_and_trace \
  -s -q --junitxml=$stage_dir/logs/final_watcher.xml \
  > $stage_dir/logs/final_watcher.log 2>&1

tt-smi -s > $stage_dir/logs/final_tt_smi.json
```

Results:

- broad suite: 5 passed, 8 intentional manual skips; covers static fallback
  source audit, synthetic TP4, real paged seq31 and traced positions 31/32,
  heterogeneous batch-2 positions, and seq1025 chunking;
- strict fallback: 2 passed;
- watcher: real paged non-aligned trace passed, with no watcher/kernel
  error/fatal/assert/sanitizer/hang/heartbeat marker;
- health: all four devices report live heartbeat and zero GDDR errors;
- maximum context: actual 32,768-token prefill and last-position decode passed;
- Python compilation and `git diff --check`: passed before review.

ETH watcher instrumentation is disabled because the platform's active-fabric
watcher configuration exceeds its kernel config buffer; Ring CCL itself remains
enabled. Profiler and watcher were not combined.

`doc/context_contract.json` is updated with the exact persistent L1 and stack
residual contracts. The BFP8 paged KV contract and 32,768-token maximum are
unchanged; the persistent L1 resources do not consume DRAM KV capacity. Public
non-aligned sequence lengths and heterogeneous positions continue to own
padding, masking, and slicing internally.

## Final state and artifacts

Final default hashes:

```text
implementation 52d49bcf238ccd76ab2e763d9b5cd3c028ac12d40764eed84272aff5813f90ea
test           d0f4bed85fc6b33a922184f3b2a0a374258ba99634f4058b97e7f9e6896b8642
```

Final batch-32 timing is 3.030676860 ms warmed prefill and 0.391425113 ms
traced warmed decode. Final PCC is 0.999999505 prefill, 0.999999931 decode,
0.999995749 K, and 0.999998539 V. These come from the current default path, not
an earlier candidate.

Raw and derived Tracy CSVs, candidate JSONs, JUnit logs, watcher output,
tt-smi health, compiler advice, and provenance logs are retained under this
directory. Generated duplicate profiler runtime streams were removed after the
raw ops CSV and derived tables were verified; they are reproducible by rerunning
the profiler command above.

The initial independent `$stage-review` and first rereview returned
`more-work-needed`; their findings and remediations are recorded in
`stage_review_initial.md`, `stage_review_second.md`, and the sections below.
Final independent rereview returned `clean-pass`; its evidence is recorded in
`stage_review_final.md`. The local checkpoint commit is pending. No push will
be performed. The stage-owned checkpoint SHA will be appended below.

## 2026-07-18: initial review remediation

The reviewer requested a final-graph shard capture, internally consistent
profile/roofline accounting, and an exact persistent-resource context contract.
It also requested two real decoder instances for the stack boundary.

The first final advisor capture command is preserved in
`shard_advise/final_graph_pipeline.log`. It described 25 ops, 22 choices, and
three spills, but first rereview found that its cosine/sine capture shape was
`[1,batch,1,head_dim]` instead of production's `[1,1,batch,head_dim]`. Its
claimed rotary blockers and hardware measurements are retained as invalidated
historical evidence, not used for the final decision.

Every feasible choice was applied together in the opt-in
`final_shard_advisor` mode: 11-core block-sharded norms, 11x4 QKV, 11x9
O/gate/up/down, 40/48/96-core phase layouts, a 96-core carried residual, and a
matching persistent TP4 all-reduce buffer. Commands used the normal baseline,
performance, and stack tests with
`FALCON3_MULTICHIP_DECODE_MATMUL_MODE=final_shard_advisor`. Results:

| Malformed-capture historical evidence | Result |
|---|---:|
| Prefill | 2.827070188 ms |
| Traced decode | 0.458579878 ms |
| Decode PCC vs optimized single-chip | 0.999999037 |
| Two-layer DRAM boundary | 0.916460031 ms |
| Two-layer 96-core carried boundary | 0.841418738 ms |
| Two-layer PCC / inter-layer collectives | 1.0 / 0 |

This row is superseded by the corrected evidence below. Packed gate/up was also rerun on the current
dedicated-RoPE/direct-output/persistent-CCL graph: DRAM unpack is 0.392699768 ms
and L1-sharded unpack is 0.397105170 ms, versus the final 0.391376750 ms
separate-projection default. Packed DRAM PCC passes.

The final current-hash Tracy capture is
`tracy/final_default_post_advisor/reports/2026_07_18_22_20_27/ops_perf_results_2026_07_18_22_20_27.csv`.
Its `tt-perf-report` tables and summaries are under
`tracy/final_default_post_advisor/derived/`. One replay has 45 ops and
381.793333 us device time. Excluding one 90,574.710 us cross-iteration marker
gap, gaps are 35.600333 us/replay. The same-run 431.963243 us profile wall time
leaves 14.569576 us residual. Matmul is 136.863000 us and async all-reduce is
27.991333 us.

`results/roofline_accounting.json` enumerates the 31,457,280 B/rank projection
payload, known activation and KV bytes, and 589,824 Ring wire bytes/rank across
the two collectives. At the report's 512 GB/s/device model, projection weights
have a 61.44 us ideal time and achieve 82.38 GB/s, or 16.09% of roofline. The
raw CSV's `CORE COUNT=80` is profiler launch metadata; derived `Cores=12` is the
report tool's analytical override for every recognized DRAM-sharded matmul, not
a measured worker grid.

`doc/context_contract.json` now records the 196,608-byte/device carried
residual, 786,432-byte/device persistent intermediate (24,576 bytes on each of
32 residual cores), two 440-byte/device global semaphores across 110 cores, one
shared manager/pool, owner/non-owner lifetime, and cleanup order. The pool is
L1-only, so the physically retested 32,768-token paged-KV context is unchanged.

Post-review gates from the final hashes: real PCC passed; default timing passed;
batch-1 final timing refreshed at 0.917850/0.318242 ms prefill/decode;
two-instance stack passed; broad suite 5 passed/8 manual skips; strict fallback
2 passed; full 32,768 context passed; watcher passed; and Python compilation,
Black formatting, JSON parsing, and `git diff --check` passed.

The first post-review default timing repeat produced 4.041276 ms prefill while
decode remained 0.391420 ms. It is retained, not discarded, as
`results/final_default_perf_post_review_repeat1.json` with its JUnit/log. An
immediate complete five-sample repeat from that default/source hash produced
3.008326/0.391377 ms. This controls the observed prefill variance. After the
corrected advisor candidate changed the source hash, a new complete default run
produced the final 3.030677/0.391425 ms; this newest artifact is the final
default result, not an earlier candidate.

## 2026-07-18: first rereview remediation

First rereview found one defect: the capture script's RoPE cos/sin shape did not
match production. The script now uses `[1,1,batch,head_dim]`, and the exact
successful advisor command is preserved in
`shard_advise/final_graph_corrected_pipeline.log`. The corrected
`shard_advise/final_graph_corrected/report.json` has 25 ops, 22 choices, two
spills, and one unfixable op: concat requires a sharded input. Both rotary ops
are valid.

The opt-in `final_shard_advisor` family was adapted to apply the advised query
and key transpose/block/height layouts, L1 rotary inputs and outputs, and every
other feasible choice. Real TP4 PCC passed: 0.999999505 prefill, 0.999999037
decode, 0.999996100 K, and 0.999998343 V. Five-sample warmed timing was
2.892734017 ms prefill and 0.451891636 ms traced decode. Two independently
materialized decoder instances with the matching 96-core residual measured
0.888365922 ms through the old DRAM boundary and 0.819200180 ms carrying the
candidate residual, PCC 1.0 and zero inter-layer collectives. This proves the
family's lower-movement contract helps, but the full candidate remains 15.45%
slower than the 0.391425113 ms default and is rejected.

The candidate implementation changed the measured source hash to
`19385bd701b70ad6072266fdcd68aeba2c9d5d9dbd4841915dc18f3ce83bd174`.
Default PCC, batch-32/batch-1 timing, two-layer carry, full 32,768 context,
broad suite, strict fallback, watcher, Tracy/tt-perf-report, and final `tt-smi`
were therefore all regenerated from that exact hash. The current gates remain
clean: 5 passed/8 manual skips broad, 2 passed strict fallback, watcher pass,
and four live devices with zero corrected or uncorrected GDDR errors.

The commit hooks subsequently applied only Black line wrapping and isort
import ordering. The final repository hashes are
`52d49bcf238ccd76ab2e763d9b5cd3c028ac12d40764eed84272aff5813f90ea`
for the implementation and
`d0f4bed85fc6b33a922184f3b2a0a374258ba99634f4058b97e7f9e6896b8642`
for the test. The measured hashes remain in the device result artifacts for
honest provenance; no executable expression or configuration changed.

Final independent `$stage-review` returned `clean-pass` with no required work.
It verified the corrected RoPE capture/IR, complete coherent advisor-family
measurement, current-hash default artifacts, same-run profiler accounting,
context, fallback, watcher, health, and the absence of deferred optimization
items. The full verdict is `stage_review_final.md`.
