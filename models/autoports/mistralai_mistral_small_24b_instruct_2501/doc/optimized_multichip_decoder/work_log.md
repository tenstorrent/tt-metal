# Optimized multichip decoder work log

## Scope and provenance

- Pass dates: 2026-07-18 through 2026-07-19 UTC.
- Branch: `mvasiljevic/model/mistralai-mistral-small-24b-instruct-2501`.
- Starting commit: `a0f6df651c8` (`Record Mistral multichip stage checkpoint`).
- Target: real Mistral-Small-24B layer 20, TP4, logical `1x4` Blackhole p300c mesh.
- Applied `$optimize`, `$tt-device-usage`, `$graph-rewrite`, `$shard-advise`, and `$stage-review`.
- Hardware commands were serialized. Watcher and profiler were never enabled together.
- The unrelated pre-existing `.agents/skills/forge-functional-decoder-from-ir/SKILL.md` edit is preserved and excluded from stage commits.
- No full-model, generator, vLLM, or serving work was started.

Retained logs preserve complete pytest/TTNN output; exact environments and invocations are recorded in this log, and candidate result lines record the selected policies. In the copy-paste commands below:

```bash
MODEL_DIR=models/autoports/mistralai_mistral_small_24b_instruct_2501
STAGE_DIR=$MODEL_DIR/doc/optimized_multichip_decoder
SNAPSHOT=/home/mvasiljevic/hf-cache/hub/models--mistralai--Mistral-Small-24B-Instruct-2501/snapshots/9527884be6e5616bdd54de542f9ae13384489724
```

## Health, baseline, and topology audit

`tt-smi -ls --local` found all four boards visible/resettable. A fresh-process smoke opened and closed `MeshShape(1,4)` with logical device IDs `[3,2,1,0]`. The completed-stage starting run was retained as `evidence/baseline_perf.xml`: one-layer warmed prefill 3.741118 ms and traced decode 0.580079 ms. Because that run is a one-layer public API measurement, final performance claims instead use reproduced two-layer before/after contracts.

Before tuning, source plus the completed-stage per-device report established:

- packed QKV plus repeated same-input separate gate/up matmuls;
- two material TP reductions per layer;
- public DRAM restore/reprepare at both decode and prefill inter-layer boundaries;
- row-parallel WO/down partials followed by collective and residual/layout work;
- available persistent async all-reduce, fused MRS, and AGMM APIs;
- DRAM-sharded dense matmuls, BF16 activations/CCLs, BFP4 LoFi weights/kernels;
- opportunities in residual layout, collective placement, activation sharding, packed gate/up, fused CCL+matmul, precision, and persistent buffers.

The action/evidence table is in `README.md`; no candidate was deferred.

## Stack residual rewrite

Decode first moved public conversion outside the stack. The earliest real two-layer candidate changed repeated-public 1.144611 ms to internal L1 0.899734 ms with eager/traced PCC 1.0 (`stack_layout_candidate.xml`). Prefill changed repeated-public 5.168495 ms to flattened internal 4.593508 ms at PCC 1.0 (`prefill_stack_layout_candidate.xml`).

Final APIs are `prepare_*_residual`, `finish_*_residual`, and `*_forward_stacked`. Decode carries `[1,1,B,H]` BF16 on 11 L1 cores. Prefill carries `[1,1,B·S,H]` BF16 DRAM with explicit logical `S`. Strict input checks prevent a silent public-layout restore between layers.

## Shard advisor and matmul geometry

The repo-local advisor was adapted until it consumed a real rank-local TP4 decode graph. It emitted `shard_advise/report.json`, `report.txt`, and `final_ir.mlir`. The exact advised 1D qkv/WO/gate/up/down family passed at PCC 0.9999844 but regressed decode from 0.579367 to 0.755150 ms (`advisor_1d_control.xml`), so the tuned DRAM-sharded family remains default.

The final-stack geometry controls then tested attention `(20,24,8,16,20,2)` at 0.826607 ms and `(40,48,4,32,40,1)` at 0.864758 ms, plus MLP `(20,16,20,8,16)` at 0.832353 ms and `(80,64,80,2,8)` at 0.902445 ms. The final `(10,12,16,8,10,4)` attention and `(10,32,40,16,16)` MLP geometry reproduces 0.806169 ms in the adjacent control.

## Collective and projection families

The first explicit persistent BF16 all-reduce consumer chain improved 0.091877 to 0.055707 ms. The first full-decoder attempt exposed an output-memory mismatch; adapting result/workspace layout passed at 0.533762 vs 0.579367 ms. The two-layer family reached about 0.404 ms/layer, then one workspace/semaphore set was shared across layers.

Final paired precision selection used the same real weights, geometry artifact, and 300 replays:

```bash
timeout 900 env TT_LOGGER_LEVEL=fatal MISTRAL_SMALL_24B_REAL_WEIGHT_DIR=$SNAPSHOT \
MISTRAL_SMALL_24B_MULTICHIP_PERF_ITERS=300 \
MISTRAL_SMALL_24B_MULTICHIP_COLLECTIVE_DTYPE=bf16 \
MISTRAL_SMALL_24B_MULTICHIP_GEOMETRY_ARTIFACT=/tmp/mistral_multichip_final_stack.pt \
MISTRAL_SMALL_24B_MULTICHIP_GEOMETRY_ARTIFACT_MODE=compare \
pytest -q -s --junitxml=$STAGE_DIR/evidence/paired_collective_bf16_stack_300.xml \
$MODEL_DIR/tests/test_multichip_decoder.py::test_multichip_two_layer_stacked_decode_layout_perf

timeout 900 env TT_LOGGER_LEVEL=fatal MISTRAL_SMALL_24B_REAL_WEIGHT_DIR=$SNAPSHOT \
MISTRAL_SMALL_24B_MULTICHIP_PERF_ITERS=300 \
MISTRAL_SMALL_24B_MULTICHIP_COLLECTIVE_DTYPE=bfp8 \
MISTRAL_SMALL_24B_MULTICHIP_GEOMETRY_ARTIFACT=/tmp/mistral_multichip_final_stack.pt \
MISTRAL_SMALL_24B_MULTICHIP_GEOMETRY_ARTIFACT_MODE=compare \
pytest -q -s --junitxml=$STAGE_DIR/evidence/paired_collective_bfp8_stack_300.xml \
$MODEL_DIR/tests/test_multichip_decoder.py::test_multichip_two_layer_stacked_decode_layout_perf
```

BF16 is 0.807669 ms and BFP8 is 0.806020 ms with BFP8/BF16 PCC 0.999990825. A separate 100-replay pair is 0.807788/0.806169 ms. BFP8 is the final default. The final real optimized TP1 comparison is 0.999994280 prefill, 0.999988553 decode, and 1.0 K/V, above the existing acceptance contract.

Packed gate/up first exceeded L1 at block 16. The adapted earlier block-8 micro-family passed at PCC 0.99999418 but was slower than its control. Repeating on the final two-layer stack gives 0.811856 ms packed vs 0.807788 ms separate, PCC 0.999983112, so separate stays default.

## Fused matmul-CCL family

The fused chain was not rejected at its first API error:

1. Down output subblock 4 was incompatible with output block 10; changed to 2.
2. QKV weight rank 2 was invalid; adapted to required rank 4.
3. Three semaphores were supplied where two are required; corrected.
4. The real-weight 2D-interleaved Ring MRS → distributed RMSNorm → AGMM consumer chain then passed at rank PCC 0.999676–0.999697 and improved 0.393561 to 0.362035 ms over 100 iterations (`fused_ccl_interleaved_real_chain_100.*`).
5. Exact final WO and down DRAM-sharded program configs were retried separately with real weights. Both hit the same hard operator validation: `MatmulReduceScatterAsync ... Needs to be 2D Multicast` (`fused_ccl_tuned_dram_sharded_{wo,down}_blocker_real.*`).

The fused family is beneficial within its coherent 2D-interleaved topology, but that topology cannot retain the final 1D DRAM-sharded matmul family. The persistent all-reduce family wins the complete final decoder path.

## Prefill collective and activation placement

Persistent prefill CCL received five adaptations rather than a first-error rejection:

1. DRAM output failed because the CCL requires block-height sharded storage.
2. Host tilize into L1 exceeded CB capacity (2,732,800 > 1,572,864 bytes).
3. Exact L1 block sharding was rejected because the operator requires width sharding.
4. Width-sharded 40-core and 80-core attempts both exceeded CB capacity (3,061,248 bytes).
5. The implementation tile-chunked to 32 logical rows, reduced in supported L1 width-sharded storage, explicitly converted each chunk to DRAM, and concatenated. It passed at PCC 1.0 but measured 12.818221 ms internal vs 12.750085 ms public and 4.448526 ms for the final general BF16 path (`prefill_persistent_bf16_chunk32_dram_retry.*`).

Prefill L1 input advice also received an adapted retry: width sharding hit the kernel’s block/height-only constraint, while exact 5×9 block sharding passed at 4.110194 ms vs 3.809161 ms DRAM. General BF16 CCL and DRAM activation remain default.

## Precision and fidelity sweep

All final-stack candidate commands use the geometry artifact in compare mode and retain logs/JUnit files:

- attention BFP8 activation: 0.828778 ms, PCC 0.99999165; slower;
- MLP BFP8 activation: 0.824331 ms, PCC 0.9999744; slower;
- attention BFP8 weights: 0.813754 ms, PCC 0.9999749; slower;
- MLP BFP8 weights: block 16 exceeds L1 at 1,913,600 > 1,572,864 bytes; adapted block 8 runs at 0.905056 ms but PCC 0.9993978 fails the 0.9999 candidate gate;
- attention HiFi2: 0.845966 ms; slower;
- MLP HiFi2: 1.192333 ms; slower.

Final policy is BFP4 LoFi dense weights/kernels, BF16 attention/MLP activations, BFP8 decode collective output, BF16 workspace, and BF16 general prefill collective.

## Final wall timing

```bash
timeout 900 env TT_LOGGER_LEVEL=fatal MISTRAL_SMALL_24B_REAL_WEIGHT_DIR=$SNAPSHOT \
MISTRAL_SMALL_24B_MULTICHIP_PERF_ITERS=300 \
MISTRAL_SMALL_24B_MULTICHIP_GEOMETRY_ARTIFACT=/tmp/mistral_multichip_final_stack_bfp8.pt \
MISTRAL_SMALL_24B_MULTICHIP_GEOMETRY_ARTIFACT_MODE=write \
pytest -q -s --junitxml=$STAGE_DIR/evidence/final_default_stack_decode_bfp8_300.xml \
$MODEL_DIR/tests/test_multichip_decoder.py::test_multichip_two_layer_stacked_decode_layout_perf

timeout 900 env TT_LOGGER_LEVEL=fatal MISTRAL_SMALL_24B_REAL_WEIGHT_DIR=$SNAPSHOT \
MISTRAL_SMALL_24B_MULTICHIP_PERF_ITERS=20 \
pytest -q -s --junitxml=$STAGE_DIR/evidence/final_default_stack_prefill_20.xml \
$MODEL_DIR/tests/test_multichip_decoder.py::test_multichip_two_layer_stacked_prefill_layout_perf
```

Final decode is 1.055324 ms public and 0.805981 ms internal for two layers (0.402990 ms/layer). Final prefill is 5.107562 ms public and 4.448526 ms internal (2.224263 ms/layer). Both public/internal outputs are PCC 1.0; decode public/stacked trace is PCC 1.0.

## Profiler and tt-perf-report

The final captures use dedicated actual-stack tests and signposts:

```bash
timeout 1200 env TT_LOGGER_LEVEL=fatal MISTRAL_SMALL_24B_REAL_WEIGHT_DIR=$SNAPSHOT \
MISTRAL_SMALL_24B_MULTICHIP_PERF_ITERS=1 \
python -m tracy -r -p -v -o $STAGE_DIR/tracy/final_stacked_decode_bfp8 -m \
pytest -q -s --junitxml=$STAGE_DIR/evidence/final_stacked_decode_bfp8_tracy.xml \
$MODEL_DIR/tests/test_multichip_decoder.py::test_multichip_two_layer_stacked_decode_layout_perf

timeout 1200 env TT_LOGGER_LEVEL=fatal MISTRAL_SMALL_24B_REAL_WEIGHT_DIR=$SNAPSHOT \
MISTRAL_SMALL_24B_MULTICHIP_PROFILE_STACKED_PREFILL=1 \
MISTRAL_SMALL_24B_MULTICHIP_PERF_ITERS=1 \
python -m tracy -r -p -v -o $STAGE_DIR/tracy/final_stacked_prefill_bfp8 -m \
pytest -q -s --junitxml=$STAGE_DIR/evidence/final_stacked_prefill_tracy.xml \
$MODEL_DIR/tests/test_multichip_decoder.py::test_multichip_profile_internal_stacked_prefill
```

`tt-perf-report` was run in merged and `--no-merge-devices` modes over `MULTICHIP_INTERNAL_STACK_DECODE`/`_END` and `MULTICHIP_INTERNAL_STACK_PREFILL`/`_END`. `script` was used to retain the human-readable tables in `perf_report_table.txt`.

Corrected decode accounting uses each positive signposted session’s summed `DEVICE KERNEL DURATION`, divided by two layers: 398.166–400.827 µs/layer (median 399.426) vs 402.990 µs wall, with PM ideal 279.810 µs. The one-session prefill capture spans 2,078.558–2,135.854 µs/layer across devices vs 2,224.263 µs wall, with PM ideal 578.956 µs. Full details are in `README.md` and `evidence/profile_accounting.csv`.

The earlier combined and split-preload captures passed their model tests but failed the profiler join because a device-3 host op was missing. Dedicated decode and dedicated actual stacked-prefill capture fixed the provenance gap. Compact processed ops CSV/report outputs remain; four duplicate generated raw `.logs`/`reports` directories totaling about 1.45 GB were removed before commit.

## Correctness, context, fallback, and watcher

Final real TP1 export/compare commands use the same `/tmp/mistral_multichip_final_real_tp1.pt` artifact and `MISTRAL_SMALL_24B_MULTICHIP_BASELINE_REAL=1`, first with `BASELINE_IMPL=optimized`, then `multichip`. The result is 0.999994280 prefill, 0.999988553 decode, 1.0 K, and 1.0 V.

The final consolidated gate command is recorded verbatim in `evidence/final_full_gate_bfp8.log`. It enables the 32K capacity test and selects seven tests: static runtime/fallback ownership, synthetic S=17/18/32 PCC, reversed paged cache, mutable nonuniform traced positions, two-layer decode, two-layer prefill, and 40-layer capacity. Result: seven passed in 148.66 s.

The capacity test keeps 40 layers, 40 K/V cache pairs, TP embedding/head/final norm, shared RoPE, shared CCL workspace, and an allocated 4 GiB reserve resident while paged decode executes at position 32,767. Context remains 32,768.

Final watcher command, run separately from profiling after a health check:

```bash
timeout 900 env TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 \
TT_METAL_WATCHER_DISABLE_ETH=1 TT_LOGGER_LEVEL=info \
MISTRAL_SMALL_24B_REAL_WEIGHT_DIR=$SNAPSHOT \
MISTRAL_SMALL_24B_MULTICHIP_PERF_ITERS=100 \
pytest -q -s --junitxml=$STAGE_DIR/evidence/watcher_final_two_layer_shared_bfp8.xml \
$MODEL_DIR/tests/test_multichip_decoder.py::test_multichip_two_layer_stacked_decode_layout_perf
```

The actual two-layer real-weight stack, shared persistent buffers, and trace replay pass. Watcher attaches devices 0–3. `watcher_error_audit_bfp8.txt` is empty; `post_watcher_health_bfp8.log` shows all four p300c boards visible/resettable. Ethernet watcher stays disabled because the completed stage established the platform firmware limitation.

## Review and commit log

- First independent `$stage-review`: `more-work-needed`. Findings were incomplete command transcripts, invalid one-layer/synthetic prefill profiler provenance, missing coherent final-stack family comparisons, incorrect profiler normalization, and insufficient shared-workspace watcher coverage.
- Remediation: retained full transcripts; added actual two-layer stacked prefill capture; reran all dominant families on the final two-layer path; corrected per-session/per-layer accounting plus PM ideal; ran actual two-layer shared-workspace watcher and post-health audit.
- Final independent `$stage-review`: `clean-pass`; no required work and no hard-check gaps. The fresh read-only reviewer audited the source, current-source gates, TP1 PCC, primary profiler/accounting, candidate adaptations, watcher/health evidence, artifact references, and dirty-scope isolation.
- Stage-owned implementation/evidence commit: `29dd518771f` (`Optimize Mistral Small 24B multichip decoder`). This SHA contains only the model source, tests, context contract, and optimized-stage artifacts; the unrelated pre-existing skill edit is excluded. A documentation-only follow-up records this SHA.
- Push: prohibited and not performed.

## Final-review remediation: exact fused boundaries and primary batch 1

The second independent review found four material reporting/evidence gaps: the fused family had not been compared at both exact final boundaries, batch 1 was not the primary regime, batch-1 prefill could select decode CCL policy by shape, and the documentation incorrectly implied that async CCL wrote the 11-core residual layout directly. `$autofix` ran after a fresh `$autodebug` report. The report is retained as `AUTODEBUG.md`; the resulting gated test captures real HF activations and compares the exact final control against a coherent Ring MRS → fractured residual add → distributed norm → AGMM candidate.

```bash
timeout 1200 env TT_LOGGER_LEVEL=fatal \
MISTRAL_SMALL_24B_REAL_WEIGHT_DIR=$SNAPSHOT \
MISTRAL_SMALL_24B_MULTICHIP_PERF_ITERS=300 \
MISTRAL_SMALL_24B_MULTICHIP_COLLECTIVE_CANDIDATE=fused_exact_boundaries \
pytest -q -s --junitxml=$STAGE_DIR/evidence/fused_exact_boundaries_real_bfp8.xml \
$MODEL_DIR/tests/test_multichip_decoder.py::test_fused_matmul_ccl_exact_final_boundaries
```

Exact boundary A (WO → reduction → residual add → post-attention norm → separate gate/up) is 0.209183 ms control and 0.480580 ms fused. Boundary B (down → reduction → residual add → next input norm → QKV) is 0.137867 ms control and 0.361885 ms fused. The direct sum is 0.347050 vs 0.842464 ms, so the coherent fused family is 2.427499× slower. Gate/up/QKV rank PCC is 0.999603–0.999626. The `<=1%` combined-trace trigger is false; this is a complete measured rejection, not a deferral.

The decoder collective helper now requires an explicit `mode="prefill"|"decode"`. This prevents batch-1 prefill (`M=18`, physically tile padded) from silently selecting the decode persistent-BFP8 path. Decode keeps persistent BFP8; prefill keeps the selected general BF16 family.

The first batch-1 decode return exposed physical tile rows (`finish_decode_residual` expected logical 1 but saw 32). An explicit input padding retry was invalid because it changed the QKV core target from one active user to 32. The accepted fix keeps the logical batch throughout and slices the padded L1 output to `self.batch` on device before the next layer/public boundary. The failed attempts remain in `batch1_decode_padding_first.*` and `batch1_decode_explicit_pad_retry.*`.

Primary performance commands use real weights and the final default:

```bash
timeout 900 env TT_LOGGER_LEVEL=fatal MISTRAL_SMALL_24B_REAL_WEIGHT_DIR=$SNAPSHOT \
MISTRAL_SMALL_24B_MULTICHIP_ACTIVE_BATCH=1 \
MISTRAL_SMALL_24B_MULTICHIP_PERF_ITERS=100 \
pytest -q -s --junitxml=$STAGE_DIR/evidence/final_batch1_stack_decode_100.xml \
$MODEL_DIR/tests/test_multichip_decoder.py::test_multichip_two_layer_stacked_decode_layout_perf

timeout 900 env TT_LOGGER_LEVEL=fatal MISTRAL_SMALL_24B_REAL_WEIGHT_DIR=$SNAPSHOT \
MISTRAL_SMALL_24B_MULTICHIP_ACTIVE_BATCH=1 \
MISTRAL_SMALL_24B_MULTICHIP_PERF_ITERS=20 \
pytest -q -s --junitxml=$STAGE_DIR/evidence/final_batch1_stack_prefill_20.xml \
$MODEL_DIR/tests/test_multichip_decoder.py::test_multichip_two_layer_stacked_prefill_layout_perf
```

Batch-1 prefill is 2.256544 ms public vs 2.038562 ms internal for two layers (1.019281 ms/layer, 9.66% reduction). The general-BF16 decode control is 0.915775 ms public / 0.907943 ms internal; final persistent-BFP8 decode is 0.841827 ms public / 0.829644 ms internal (0.414822 ms/layer). Completed public general → final internal reduction is 9.41%. Public/stacked and traced public/stacked PCC are 1.0.

The inherited optimized TP1 implementation itself cannot return native batch 1: it produces the emitted 32 physical rows and fails its final reshape. The first native failure is retained in `batch1_tp1_native_return_first.*`. For the real reference only, the test runs TP1 at its established batch 32 with the active user first and zero independent inactive users, snapshots that active row/cache, then compares against true TP4 batch 1. Result: prefill 0.999985284, decode 0.999989612, K/V 1.0 in `final_batch1_real_tp1_{export,compare}.*`.

## Primary profiler and watcher commands

Decode capture:

```bash
timeout 1200 env TT_LOGGER_LEVEL=fatal MISTRAL_SMALL_24B_REAL_WEIGHT_DIR=$SNAPSHOT \
MISTRAL_SMALL_24B_MULTICHIP_ACTIVE_BATCH=1 \
MISTRAL_SMALL_24B_MULTICHIP_PERF_ITERS=1 \
python -m tracy -r -p -v -o $STAGE_DIR/tracy/final_batch1_stacked_decode_bfp8 -m \
pytest -q -s --junitxml=$STAGE_DIR/evidence/final_batch1_stacked_decode_bfp8_tracy.xml \
$MODEL_DIR/tests/test_multichip_decoder.py::test_multichip_two_layer_stacked_decode_layout_perf
```

The first ordinary prefill capture used a one-iteration public/internal timing assertion and failed from profiler noise (3.646491 vs 3.361037 ms) after producing data. The profile-only test was generalized to active batch; its first retry exposed missing `batch=active_batch` decoder/cache construction, which was fixed rather than dismissed. The passing capture is:

```bash
timeout 1200 env TT_LOGGER_LEVEL=fatal MISTRAL_SMALL_24B_REAL_WEIGHT_DIR=$SNAPSHOT \
MISTRAL_SMALL_24B_MULTICHIP_ACTIVE_BATCH=1 \
MISTRAL_SMALL_24B_MULTICHIP_PERF_ITERS=1 \
MISTRAL_SMALL_24B_MULTICHIP_PROFILE_STACKED_PREFILL=1 \
python -m tracy -r -p -v -o $STAGE_DIR/tracy/final_batch1_profile_stacked_prefill_bf16_pass -m \
pytest -q -s --junitxml=$STAGE_DIR/evidence/final_batch1_profile_stacked_prefill_bf16_pass_tracy.xml \
$MODEL_DIR/tests/test_multichip_decoder.py::test_multichip_profile_internal_stacked_prefill
```

Each source CSV was processed in merged and per-device modes and retained as CSV, summary, plot, stdout, and human table:

```bash
tt-perf-report $OPS_CSV --start-signpost $START --end-signpost $END \
--no-color --csv $PROCESSED_CSV --summary-file $SUMMARY
tt-perf-report $OPS_CSV --start-signpost $START --end-signpost $END \
--no-merge-devices --no-color --csv $PER_DEVICE_CSV --summary-file $PER_DEVICE_SUMMARY
script -q -c "tt-perf-report '$OPS_CSV' --start-signpost $START --end-signpost $END --no-color --no-summary" \
$REPORT_DIR/perf_report_table.txt
```

Primary decode accounting is matmul 63.46%, async all-reduce 10.82%, residual binary 7.93%, norm 4.07%, reshard 2.65%, and SDPA 1.90%; modeled DRAM roofline is 32.4% merged / 33.4% per-device. Primary prefill is matmul 42.95%, norm 24.66%, reduce-scatter + all-gather 11.60%, reshape 5.04%, and SDPA 1.17%; modeled roofline is 19.8% / 19.9%.

Final watcher is separate from profiling:

```bash
timeout 900 env TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 \
TT_METAL_WATCHER_DISABLE_ETH=1 TT_LOGGER_LEVEL=info \
MISTRAL_SMALL_24B_REAL_WEIGHT_DIR=$SNAPSHOT \
MISTRAL_SMALL_24B_MULTICHIP_ACTIVE_BATCH=1 \
MISTRAL_SMALL_24B_MULTICHIP_PERF_ITERS=100 \
pytest -q -s --junitxml=$STAGE_DIR/evidence/watcher_final_batch1_two_layer_shared_bfp8.xml \
$MODEL_DIR/tests/test_multichip_decoder.py::test_multichip_two_layer_stacked_decode_layout_perf
```

The watcher attaches/detaches devices 0–3 and passes. `watcher_error_audit_batch1_bfp8.txt` is empty. A post-run `tt-smi -s` in `post_watcher_health_batch1_bfp8.log` reports four p300c boards.

## Final-source regression reproduction

The seven-test batch-32 gate was rerun again after all candidate hooks and trace evidence. It passes in 147.98 s in `final_current_source_gate_bfp8.*`; the exact final default is 0.806937 ms/two-layer decode and 4.480664 ms/two-layer prefill, and the 40-layer position-32767 capacity probe passes with the 4 GiB reserve resident. A fresh real TP1 export/compare reproduces 0.999994280 prefill, 0.999988553 decode, and 1.0 K/V in `final_current_real_tp1_{export,compare}.*`.

The final residual topology is recorded precisely: each async all-reduce writes BFP8 to the 40-core L1 width-sharded CCL layout, then an intra-layer conversion produces the 11-core L1 block-sharded norm/residual family before the BF16 add. This happens twice per layer. The C++ async-all-reduce validator requires width-sharded input, workspace, and output; therefore it cannot emit the block-sharded residual config directly. There is still no gather, reshard, or all-reduce between decoder layers.

## Final rereview remediation: primary accounting and prefill norms

The next independent rereview found two remaining optimize-contract gaps: primary batch-1 decode lacked explicit theoretical/device/wall accounting, and prefill LayerNorm (24.66% of the report) had not been attacked with a downstream-ready sharded family. Both findings were treated as work.

Primary decode accounting uses the retained final profiler. Its positive signposted sessions total 4,182.453333 µs of device kernels across six executions of two layers, or 348.538 µs/layer. Per-rank stored weights plus norms are 78,807,040 bytes; position-18 K/V reads occupy one physical 32-position tile row, 17,408 bytes/rank. The aggregate layer payload is therefore 315,297,792 bytes, and the report model's 512 GB/s/chip × four chips gives a 153.954 µs/layer theoretical bandwidth floor. Device time is 2.264× that floor. Canonical 100-replay wall is 414.822 µs/layer, 66.284 µs or 15.98% beyond kernel time; the same profiler capture's single-replay wall is 456.946 µs/layer. `evidence/profile_accounting.csv` records all three values, byte provenance, and both wall protocols.

For prefill RMSNorm, the first candidate reused decode's 11-core block config and failed because the QKV shard width was 15 tiles while `in0_block_w=8`. The block-5 adaptation ran but produced PCC 0.999761602 and 2.221102 ms/two layers versus 2.084208 ms. A five-core no-padding adaptation improved PCC to 0.999898482 but lost at 2.265096 versus the longer 1.896023 control. These are retained in `prefill_sharded_norm_candidate_batch1_{first,block5_retry,block5_timing,5core_retry,5core_100}.*`.

The final adaptation uses downstream-ready layouts rather than immediately restoring DRAM: input norm feeds QKV in an 8×1 L1 block shard (`[32,640]`), while post-attention norm feeds MLP in its native 10×1 L1 width shard (`[32,512]`). Each path was isolated and the coherent pair was measured:

| Candidate | Candidate two-layer ms | Adjacent default ms | PCC | Result |
| --- | ---: | ---: | ---: | --- |
| QKV sharded norm only, 20 iterations | 2.167228 | 2.084208 | 0.999976375 | 3.98% slower |
| MLP sharded norm only, 20 iterations | 2.119506 | 2.084208 | 0.999897230 | 1.69% slower |
| both consumer-ready norms, 100 iterations | 2.254134 | 1.896023 | 0.999888951 | 18.89% slower |

The accepted final remains DRAM norm. The final-default 100-iteration control is 2.101789 ms public and 1.896023 ms internal for two layers (0.948012 ms/layer, 9.79% boundary reduction). Candidate support remains gated by `MISTRAL_SMALL_24B_MULTICHIP_PREFILL_ACTIVATION_FAMILY`; the default is unchanged and arbitrary logical lengths above the one-tile candidate case fall back internally to the general DRAM implementation.

The prefill table also advised tracing to remove 679 µs / 29.4% of op gaps. A decoder-local gated trace test was added and run without any full-model orchestration:

```bash
timeout 900 env TT_LOGGER_LEVEL=fatal MISTRAL_SMALL_24B_REAL_WEIGHT_DIR=$SNAPSHOT \
MISTRAL_SMALL_24B_MULTICHIP_TRACE_STACKED_PREFILL=1 \
MISTRAL_SMALL_24B_MULTICHIP_ACTIVE_BATCH=1 \
MISTRAL_SMALL_24B_MULTICHIP_PERF_ITERS=100 \
pytest -q -s --junitxml=$STAGE_DIR/evidence/prefill_trace_batch1_100.xml \
$MODEL_DIR/tests/test_multichip_decoder.py::test_multichip_two_layer_stacked_prefill_trace_perf
```

`prefill_trace_batch1_100.*` passes at eager/traced PCC 1.0 and 1.401168 ms/two layers (0.700584 ms/layer), 26.10% faster than the adjacent 1.896023 ms warmed eager default. This closes the actionable report advice and proves trace safety. The required final prefill number remains warmed eager; trace capture/replay is an optional caller-owned boundary and no full-model work was started.

## Commit packaging

The repository hooks passed for `29dd518771f`. The commit force-includes the otherwise globally ignored stage `.log` and `.csv` provenance, so the README's final-primary artifact references survive a fresh checkout. Three superseded raw artifacts exceed the repository's 500 KiB hook limit and remain local only: the 2.9 MiB shard-advisor console dump and two 3.0/5.7 MiB early profiler source CSVs. Their compact advisor JSON/text/MLIR, processed profiler reports, and every final-primary raw/processed CSV, table, log, and JUnit artifact are committed. No final claim depends on an excluded raw file.
