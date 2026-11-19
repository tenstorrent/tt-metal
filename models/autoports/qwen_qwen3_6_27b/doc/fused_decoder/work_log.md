# Fused decoder work log

Stage 2; Qwen/Qwen3.6-27B, pinned revision 8f33b98937c6031bb49e713dee3e8e185aafd26f.
Starting commit 60c3c9e0. Clean initial worktree. Only fused implementation, tests and docs are stage-owned.
Installed enabled tt-model-bringup 0.1.23 and tt-autodebug 0.1.8 verified by `codex plugin list`.
Environment: source python_env/bin/activate; source /home/ttuser/qwen36-27b/lane-env.sh.
All jobs serial on physical chip 0 (P300 Blackhole); `fuser /dev/tenstorrent/*` found no owners.
`timeout 60 tt-smi -ls --local` succeeded; device_list.log. No reset authorized or performed.
Performance uses allocation tracking/tracebacks off, skip-program-cache=0; correctness uses tracking=1.
No functional runtime file is modified. Shared test selector explicitly selects FusedDecoder under QWEN_DECODER=fused and records source hash.

## Initial operation topology

All tensors DRAM interleaved unless specified; BF16 activations/weights, FP32 norms and recurrent arithmetic; HiFi4 FP32 accumulation unchanged.
Prior-stage raw reports: ../functional_decoder/perf/{linear_attention,full_attention}/.

| Block | Ordered operations / inputs | Movement and fusion candidates |
|---|---|---|
| Entry norm | x -> FP32 cast -> rms_norm(weight, eps) -> BF16 cast | Dedicated RMSNorm already used; residual+norm candidate |
| GDN projections | normalized x -> qkv, a, b, z linear | Shared LHS packing; scalar projections padded to tile |
| GDN convolution | history+qkv concat -> four slices/multiply -> three adds -> SiLU; copy history | Dedicated causal conv+SiLU; history layout contract |
| GDN normalize | q/k squared -> sum -> epsilon -> rsqrt -> multiply; head repeat | Dedicated chunk GDN flat input can fold L2 and head expansion |
| GDN decode | exp(g), state decay, K prediction, delta update, Q contraction | Dedicated fused recurrent GDN; inspect wrapper conversions |
| GDN prefill | 32-row chunk delta-rule inverse, matmuls, exponentials, state copy | Dedicated chunk GDN with explicit persistent constants |
| GDN output | RMSNorm -> SiLU(z) multiply -> reshape -> projection | Fuse SiLU into multiply |
| Full projections | x -> query, key, value, output-gate linears | Pack QKV / gate; Q/K norm before partial RoPE |
| Full head/RoPE | reshape+permute -> Q/K rmsnorm -> slice64 -> rotate-half multiply/add -> concat192 | Dedicated head ops and partial rotary |
| Full KV / attention | K/V permute -> height shard -> paged update; Q permute -> paged SDPA; prefill pad+fill+chunked SDPA | SDPA already fused; preserve cache dtype/pages/position |
| Full output | permute -> reshape -> sigmoid(gate) multiply -> output linear | concatenate_heads; fuse sigmoid into multiply |
| MLP | residual add -> norm -> gate/up linear -> SiLU -> multiply -> down linear -> residual add | Packed gate/up; linear activation vs binary input activation |

## Measurement commands

All run results record argv, source hashes, dependency versions, regime and environment. Console logs share each JSON stem.
`timeout 900 python -m models.autoports.qwen_qwen3_6_27b.tests.run_decoder --layer {0,3} --length 512 --repeats 20 --output ...`
Baseline QWEN_DECODER unset; fused QWEN_DECODER=fused. wait_for bounds and records each job.
First candidate device job started 2026-11-12 17:50 UTC (recurrent_0).

## Initial candidate ledger

- Baseline real layer 0: prefill PCC .999967342, decode PCC .999997185, traced median 2.778 ms; warmed prefill wall 18.240 ms.
- Baseline real layer 3: prefill PCC .999979640, decode PCC .999985743, traced median 2.382 ms; warmed prefill wall 7.158 ms.
- Dedicated recurrent kernel: completed; results below.

## Candidate observations through 18:04 UTC

Successful candidates use real layer weights and HF >=0.995, repeated bit equality and refreshed trace inputs. From chunk_tiled onward, run_decoder also enforces functional output/contribution/state PCC >=0.995. The initial rejected recurrent_0 probe predates that added control and carries only HF evidence; it is not the delivered path. Failed runs are retained, not accepted.

- Native chunk default output is ROW_MAJOR; changed to explicit head-major TILE output, preserving downstream norm layout. `chunk_tiled_0`: 9.453 ms warmed prefill, 2.639 ms traced decode.
- Native rotary returns padded sequence shape. Initial concat failed at decode. Explicit logical output slicing repairs it. `rope_sliced_3`: 2.349 ms decode, 7.052 ms prefill.
- Dedicated concatenate_heads and binary input activation fusion pass; `binary_3`: 2.334 ms / 6.589 ms. Identical binary results preserve the BF16 rounding boundary in this kernel.
- Native flat chunk inputs fold L2, head expansion and query scaling. Initial Python Shape slice typo fixed to explicit indices. Zero padding is internal (beta=g=0); logical lengths are sliced after native output. `flat_direct_0`: 2.564 ms / 8.048 ms. Chunk decode outperforms native recurrent (2.622 vs 2.634 before removing head round trips) and supports batch-head waves. Recurrent implementation has a one-core-per-head limit in its factory; it is not the final batch path.
- Packed QKV + dedicated split heads: 2.218 ms / 6.115 ms; kept. Packed MLP gate/up: 2.570 ms / 8.289 ms for GDN and 2.340 ms / 6.926 ms for full; rejected versus separate gate/up with binary SiLU.
- Matmul activation without explicit grid is trailing activation (`matmul.cpp`); adapted to core_grid 8x8 to actually fuse it. Grid fused 2.637 ms vs matched grid/binary control 2.632 ms; both slower than 2.564 ms default. Rejected after supported adaptation, not an API failure.
- Packed GDN QKV/Z/B/A with tile padding for 48-wide gates: 2.475 ms / 7.934 ms; kept, removes three projections.
- Dedicated decode heads with L1 output then compatible norms: 2.264 ms vs packed generic split-head path 2.218 ms; rejected. Decode concatenate needed slicing its fixed 32-user output; adapted path 2.353 ms vs generic concat 2.334 ms; rejected. Decode token_index RoPE 2.348 ms vs 2.334 ms; rejected.
- GDN linear+bias: 2.563 ms / 8.019 ms vs 2.564 ms / 8.048 ms; no material gain and changes FP32 bias placement, rejected.
- Dedicated causal convolution first rejected packer_l1_acc=True. Adapted only that op to packer_l1_acc=False (kernel owns L1 accumulation); PCC passes, 2.494 ms / 6.733 ms. Combined with packed GDN: 2.392 ms / 6.514 ms.
- Native convolution returned carry removes history conversion and context concat. Row-major conv state, persistent exchange buffers/semaphores and all 1..512 logical end tensors prepared at setup; no per-forward host construction. `conv_carry_0`: 2.356 ms / 6.306 ms. Head-major norm/gating plus concatenate_heads: 2.356 ms / 6.097 ms; kept for prefill.
- Native GDN factory comparison: phased 2.354 ms / 6.322 ms, fused 2.358 ms / 6.067 ms. Automatic factory selection retained (phased decode, fused prefill; to confirm with profiler). Tiny decode differences are below measurement significance.

Candidate source snapshots are in candidates/. Successful result JSON includes the exact source hash and runner hash. `QWEN_FUSION_CANDIDATE=<stem>` selects a snapshot only in test code; final default selection has this unset.

### Scheduling observation

The batch-32 timing driver was launched while the pytest wrapper had not yet reported completion. Logs show the batch runner began driver discovery at 18:08:46.083 while pytest was finishing driver teardown (completed 18:08:46.449); the new mesh warning appears at 18:08:46.702 and measured prefill began 18:08:49.447. Driver lifecycles briefly overlapped, so that first timing row will be remeasured under the new lock before acceptance. No process was killed and no reset occurred. Subsequent chains are started only after an explicit completed-session result and use a persistent flock around each bounded hardware job.

## Optimization scope and stopping assessment

The named graph-fusing checklist is assessed in pattern_assessment.md, but that alone does not satisfy this goal's explicit stopping condition. Current B1 traced decode achieves roughly 64%/66% of the weight/KV roofline. Neither 70% plus two sub-3% kept gains nor eight hours of device optimization has been reached. Stage remains in progress. AutoDebug/AutoFix was invoked for the remaining performance gate; this supersedes the earlier checklist-only stopping interpretation.

AutoDebug launcher first stopped before starting a model because this host prohibits unprivileged bubblewrap namespaces. Retried explicitly with AUTODEBUG_SKIP_CHILD_SANDBOX=1 under the existing unrestricted execution and user-authorized skill delegation. Child remains inspection-only. No device reset or process termination.

### Not tried (stop rule) / next-stage handoff

Every named pattern has an outcome or no-matching-subgraph decision. AutoDebug identified additional native opportunities, subsequently measured below. Dedicated native GDN mono is a benchmark-only factory that does not consume the selected flat/norm-fused layout; the compatible phased/fused families are measured. Wider matmul geometry, DRAM-sharded weights/activations, residual sharding, KV precision and BF16/BFP8/BFP4 policy searches belong to the explicitly excluded optimized-decoder/datatype stages. They may improve the dominant dense weight streaming; no gain is claimed without measurement. Multi-device CCL, sampling, LM-head and serving checklist items are inapplicable to a single decoder-layer stage.

### Correctness runs on the selected implementation

After hooks_final_code_repeat.log passed, the delivered source passed contract_0/3 (20 cases each), batch32_0/3, run_trace_reuse_0/3 (retained trace across prepared requests), run_mixed_decode_0/3 and synthetic_pytest.log (4 passed). Allocation tracking enabled, program-cache exclusion disabled. Source hashes in all result JSON tie these to the delivered independent FusedDecoder, with no test candidate selected.

Batch-32 same-harness timing initially measured functional/fused decode 73.506/66.327 ms for GDN and 63.961/61.116 ms for full attention. The first functional row is remeasured due to the scheduling observation above; the other rows have clean serial driver lifetimes.

## Full-context validation

`QWEN_DECODER=fused TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 flock doc/fused_decoder/device.lock timeout 1800 python -m models.autoports.qwen_qwen3_6_27b.tests.run_long_context --layer {0,3} --length {262143,262144} --output doc/fused_decoder/context_<layer>_<length>.json`

All four runs exited zero (80/78/105/104 seconds wall including HF setup). No diagnostic chunk override, no tail-only mode, no injected chunk drains. Context 262143 includes the traced follow-on decode at context 262144; the 262144 prefill case has no out-of-contract follow-on token. Updated context_contract.json adds fused evidence and records the convolution history layout change without changing prior functional evidence or advertised capacity.

## Additional structural batch rewrite

Profiler follow-up identified repeated shared-weight reads across `[batch,sequence,width]` matmul batch dimensions. `_linear` now collapses batch/sequence into one logical M dimension when batch>1, then restores the public shape. This is an algebraically identical shared-weight matrix operation; batch=1 executes the previous path unchanged. `batch_fold_0/3` at batch32,length33 passes direct functional output/contribution/state PCC and HF checks. Decode improves 66.327 -> 8.924 ms (GDN) and 61.116 -> 5.704 ms (full). Prefill improves 73.081 -> 28.906 ms and 65.163 -> 23.161 ms. Kept. The affected boundary cases are batch2 lengths511/513 (including physical512 chunk + logical1 tail), batch32 length33, mixed-position batch4 and synthetic batch2. They are rerun on the final default. Prior max-context batch1 evidence remains applicable because only the batch>1 branch changed; its exact prior source is archived by hash.

## Watcher and anomaly classification

All four final Watcher1 runs exited zero: watcher_contract_0/3 and watcher_batch32_0/3. The contract runs cover awkward tails and continuation; batch32 runs cover the new shared-weight batch fold, trace replay, input refresh, determinism and allocation-tracking positive control. Raw watcher logs are compressed as watcher_<mode>_<layer>_raw.log.gz; watcher_audit.json records uncompressed hashes and clean status. An initial broad `hang` text search matched the kernel registry filename `history_exchange`; the audit records this benign match explicitly.

- HF optional FLA/causal-conv warning: CPU reference implementation only. TT hot paths are separately guarded against from/to-torch and audited in Python/native source. Resolution: controlled.
- Unknown motherboard B850M-C tray-id fallback / opening subset of MMIO devices: UMD topology labeling/performance advisory. This stage opens only physical chip0; no remote-chip execution or reset. All device and watcher checks pass. Resolution: controlled single-chip scope.
- Deliberate trace allocation error: run_decoder creates an unsafe live tensor and requires replay rejection under tracking=1, then releases it and verifies replay succeeds. This is a positive safety control, not accepted corrupted execution.
- Tracy optional WASM-viewer copy warning: host `.tracy` copy uses the default generated path while collection uses a custom output directory. Raw CSV generation and signpost reports succeed. No dropped-marker lines exist. The raw CSV is preserved compressed; the optional viewer copy is not used as timing evidence.
- Profiler histogram overflow counters are histogram bucket overflow, not dropped profiler markers. Durations are sane and reconcile with traced host wall time.
- pandas DtypeWarning in raw mixed-op CSV parsing and unclassified reshape category: reporting metadata only; numeric device-time columns and signposts are explicitly parsed and checked.
- Native-op first-attempt failures (chunk ROW_MAJOR norm, rotary padded concat, Python Shape slicing, convolution packer flag, decode concat padded batch) were corrected and rerun; they are retained as historical rejected attempts. No device hang or reset occurred.

## Exact command families

From `/home/ttuser/qwen36-27b/tt-metal`:

```bash
source python_env/bin/activate
source /home/ttuser/qwen36-27b/lane-env.sh
export QWEN_DECODER=fused
unset QWEN_FUSION_CANDIDATE
FUSED_DOC=models/autoports/qwen_qwen3_6_27b/doc/fused_decoder
# Correctness (LAYER=0 or3; LENGTH=262143 or262144):
TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
flock "$FUSED_DOC/device.lock" timeout 1800 python -m models.autoports.qwen_qwen3_6_27b.tests.run_long_context \
  --layer "$LAYER" --length "$LENGTH" --output "$FUSED_DOC/context_${LAYER}_${LENGTH}.json"
# Watcher (contract or run_decoder --batch 32 --length 33 in separate runs):
TT_METAL_WATCHER=1 TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
TT_METAL_LOGS_PATH="$FUSED_DOC/watcher_contract_$LAYER" \
flock "$FUSED_DOC/device.lock" timeout 1200 python -m models.autoports.qwen_qwen3_6_27b.tests.run_contract \
  --layer "$LAYER" --output "$FUSED_DOC/watcher_contract_$LAYER.json"
# Matched timing: QWEN_DECODER=functional/fused, layers0/3, lengths128/511/512/513/4097.
# 100 replays at512, 10 at the other lengths; batch32,length33 uses20.
TT_METAL_TRACE_ALLOC_TRACKING=0 TT_METAL_TRACE_ALLOC_TRACEBACKS=0 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
flock "$FUSED_DOC/device.lock" timeout 900 python -m models.autoports.qwen_qwen3_6_27b.tests.run_decoder \
  --layer "$LAYER" --length "$LENGTH" --repeats "$REPEATS" --output "$RESULT"
# Profiling (Watcher unset, capacity5000, drains outside signposts):
TT_METAL_TRACE_ALLOC_TRACKING=0 TT_METAL_TRACE_ALLOC_TRACEBACKS=0 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=5000 flock "$FUSED_DOC/device.lock" timeout 900 \
python -m tracy -r -p -v --no-web-server -o "$FUSED_DOC/tracy/$TAG" \
-m models.autoports.qwen_qwen3_6_27b.tests.run_decoder --layer "$LAYER" --length 512 \
--profile-drain --output "$FUSED_DOC/$TAG.json"
python -m models.autoports.qwen_qwen3_6_27b.tests.render_perf --advice \
--tracy-root "$FUSED_DOC/tracy/$TAG" --result "$FUSED_DOC/$TAG.json" --output "$FUSED_DOC/perf/$TAG"
```

Result JSON records the expanded Python argv. Console logs use the same stem. Outer wait_for timeout is job timeout+40 seconds and emits the exit status; all final hardware jobs hold the persistent lock. Timing driver checks STOP_AFTER_JOB between jobs so it can stop without interrupting dispatch.

## AutoFix performance remediation (18:31 onward)

AutoDebug completed, archived as AUTODEBUG.md. Its old-default roofline calculation is confirmed: 1.509888 ms GDN and 1.458184 ms full at B1/context513, from active BF16 weights plus mandatory state/KV traffic over512GB/s. No denominator change or idle-time accounting is used.

Additional candidates and exact source snapshots are preserved. norm_width uses an 8-core native width-sharded RMSNorm at B1/one-token width5120; norm_cast_only folds both casts into its required data movement and retains FP32 intermediates/gamma. The user explicitly permits fusing-induced layout/sharding changes; this native normalization adapter remains within fused_decoder.py. No optimized-decoder implementation or precision sweep was started. A 40-core control showed no meaningful gain and is rejected. Pure BF16-input cast absorption (norm_absorb) passes but is slower for decode; its prefill form is a candidate for combination.

The native SwiGLU epilogue passed PCC but lost: default2.648/2.513ms (GDN/full); adapted GDN Kblocks8/16/20,Nblock10,subblock1x2 remained2.692/2.643/2.639ms versus2.205ms control. These keep BF16/HiFi4. Rejected based on runtime, not API error.

GDN softplus+multiply fusion and mixed-input FP32 bias addition pass; initial UnaryWithParam list syntax was corrected to two scalar parameters. Retaining padded native conv Q/K/V (with original actual_end) removes slice/re-pad; keeping output logical shape via explicit logical/padded reshape removes a copy. Flat-Z gating moves concatenate_heads before the elementwise gate, eliminating gate reshape/transpose. Each has real-weight HF and direct functional state/output/contribution evidence. Finally logical_gates uses native prep+scan only for B1/s1, matching the public phased decode topology; its gate reader zeros invalid rows. Other batches/lengths retain the native public chunk path, not a functional fallback. Independent supporting agent audited exact adapter signatures, flags, gate masking and the phased head/core ceiling before use. logical_gates_0 median2.132001ms passes70% (1.509888/2.132001=70.82%). Last two incremental kept gains flat_z_view and logical_gates are <3% each. Remaining candidate integration and final gates are required before completion.

### Final selection after AutoFix

The delivered source combines logical_gates, fused cache updates, shape-preserving HF RoPE, and general native RMSNorm cast absorption. Short real-weight combined_final_0/3 passes: prefill PCC0.999970580/0.999979522; decode PCC0.999997493/0.999985391; traced2.131844/2.041396ms; prefill5.594715/5.771844ms. Final default validation uses new delivered_* artifacts, all tied to the formatted source hash; older evidence remains historical.

Named dedicated primitives and all listed graph patterns were assessed. Newly discovered native SwiGLU, sigmoid-gated norm with external Z, gate masking, flat-Z ordering, fused KV update, HF RoPE and cast absorption have measured outcomes in AUTOFIX.md and candidate_results.csv. Candidate raw source hashes are indexed in measured_sources.json; pre-format sources are preserved under source_archives. Large fused_profile_0 console output is preserved as fused_profile_0.log.gz.

### Not tried after reaching the explicit stop condition / hand-off

- Native GDN final-state output aliasing, or recurrent in-place ring mode: potential copy removal, requires alias/lifecycle and batch-wave validation; no performance rejection is claimed.
- Native matmul residual addcmul epilogue: possible small dispatch saving, but uses a different dense factory and changes projection rounding; not tested after reaching the stop condition.
- DRAM-sharded dense matmul and broad program-geometry/precision searches remain next-stage work. All current dense rows retain BF16/HiFi4; no lower precision result is implied.
- Shared convolution scratch is serialized; concurrent same-layer calls would need separate scratch ownership.

These hand-offs do not waive correctness/context/cache gates. No native implementation or shared runtime source was modified.

## Full-context L1 lifetime repair

Observed anomaly: final full-attention262143 prefill passed PCC0.999887014, but high-position continuation failed before SDPA dispatch with static CB/L1 overlap (allocated1540096, static region end1546304).
Evidence: before_lifetime_fix_delivered_context_3_262143.log/.json, before_lifetime_fix_driver.log; exact source dde4540e9af3999c0f9f712e4e11e267f1600c74b4e103cff2fcbc9039a4b4b8 archived.
Affected path: _attention_decode after native fused K/V update, including non-aligned continuation and final decode at large context.
Control/comparison: previous separate cache-update implementation passed full context. The fused implementation retained both L1 input tensors in the local inputs list through SDPA, whereas those inputs are no longer needed after the ordered cache-update command.
Likely subsystem: local tensor lifetime and native SDPA L1 circular-buffer reservation, not DRAM capacity or arithmetic.
Investigation: traceback identifies SDPA compile/allocate; the local two-input list remains live at that statement. Native update returns caches, so releasing the temporary list after update preserves the persistent caches and queued operation order.
Repair: del inputs immediately after paged_fused_update_cache. No context/key-block/cache-dtype reduction, no reset, no process kill. Preserved all initial delivered_* evidence under before_lifetime_fix_*; targeted262143 reproduction runs before the full final suite.
Resolution: fixed. lifetime_fix_context_3_262143 exits0, including high-position continuation and eager/traced decode at context262144. The full final suite is rerun on the repaired source.

## Final-source gate closure (2026-11-12 19:10 UTC)

Final source SHA256: `6e2429f7440438719c1c3bd42e8dc12d34c22281018602d2eb00a291ac739a31`. `bash models/autoports/qwen_qwen3_6_27b/doc/fused_decoder/validate_final.sh > models/autoports/qwen_qwen3_6_27b/doc/fused_decoder/validate_final_driver.log 2>&1` completed with exit0. All 34 final result JSONs identify FusedDecoder, no candidate override, and this same source hash. `python -m models.autoports.qwen_qwen3_6_27b.tests.summarize_fused` prints FUSED_EVIDENCE_PASS34. Independent review and checkpoint commits follow; no stage completion is claimed before those steps.

All final short/continuation, batch2/32, mixed-position, retained-trace, full-context and Watcher jobs passed. Synthetic pytest:4 passed. Full-context262143 and262144 run freshly for both kinds; context_contract.json now references these final-source results. Repeated outputs are bit-equal; changed trace inputs change outputs; allocation positive controls run with tracking1 and program-cache skipping0. Final raw Watcher logs are compressed and hashed in delivered_watcher_audit.json, with no error/assert/corruption lines.

### Matched final performance

| Kind | Prefill functional / fused ms | Decode functional / fused ms | Decode speedup | Roofline |
|---|---:|---:|---:|---:|
| GDN | 17.848329 / 5.564706 | 2.777220 / 2.132236 | 1.3025 | 70.8124% |
| Full attention | 7.087370 / 5.784396 | 2.382522 / 2.041422 | 1.1671 | 71.4298% |

The before/after PCCs, all tested timing lengths and batches are in README.md, performance_comparison.json/.csv and batch32_performance.json. Timing is per-layer, real weights, seeded BF16 activations, physical chip0, BF16/HiFi4, warmed prefill and100-replay traced decode at length512. Functional controls use the same run_decoder harness hash. Final medians reproduce combined_final within0.02%/0.002%, so no gain is claimed for measurement noise.

The strict stop rule is now satisfied: both kinds exceed70% nominal memory roofline, and the last two incremental kept decode gains per kind are below3% (GDN flat_z_view0.28%, logical_gates1.40%; full cache_fused0.29%, rope_hf0.13%, followed by combined general norm absorption below3%). Roofline uses only active BF16 weights and mandatory recurrent/KV traffic over512GB/s; no idle time, unused allocations or cache-capacity inflation. See roofline.json for byte counts and independent floor.

### Final tt-perf-report reconciliation

| Kind / regime | Functional kernels ms (+gaps ms) / ops | Fused kernels ms (+gaps ms) / ops |
|---|---:|---:|
| GDN prefill | 16.071698 (18.245061) / 1021 | 4.680895 (4.826828) / 26 |
| GDN decode | 2.732902 (2.796303) / 103 | 2.084840 (2.129975) / 31 |
| Full prefill | 6.320050 (6.357805) / 57 | 4.973014 (5.036797) / 27 |
| Full decode | 2.329634 (2.374351) / 63 | 2.016956 (2.041606) / 36 |

Profiler tables/CSV use microseconds; this table converts to milliseconds. Same-profile host medians are2.173755/2.071810ms versus unprofiled2.132236/2.041422ms, so instrumented roofline fractions are explicitly separate. Raw ops.csv.gz hashes match metrics.json. Final profiles contain no tilize/untilize/reshard and all dense MatmulDeviceOperation rows are HiFi4 BF16 x BF16 => BF16. Required norm cast/transfers and fused KV input shards are listed in profiler_audit.json and justified in native_runtime_audit.md. Runtime audit passes13 methods/37 TTNN operations.

tt-perf-report --advice: dominant decode matmuls achieve roughly79–82% nominal DRAM bandwidth and recommend DRAM-sharded configs; broad dense sharding/precision is a hand-off after this fusion stop condition. The GDN scan gap advice suggests tracing, but this measurement is already traced; it is not evidence of host eager dispatch. Native phased/fused alternatives were measured. Prefill reports and their advice remain beside decode tables. No dropped marker warnings appear; histogram bucket overflow and optional viewer-copy/CSV dtype warnings are classified in the anomaly ledger.

### Remaining limitations and review hand-off

No context or acceptance-bar reduction. The long-context GDN decode PCC changes from0.999986067 to0.999797103; its selected contribution minimum is0.999236844, and full-attention minimum is0.997757338, all above0.995. Native fused arithmetic/order changes are covered by short direct functional state/output/contribution controls; the exact accumulated long-context delta was not isolated to one fusion. This is recorded, not hidden by residual-output PCC. Full context was tested at batch1 and short requests at batch32; no batch32 full-context claim. Shared convolution scratch is serialized; private native GDN binding compatibility is checkout-specific.

All original failed experiments and the repaired L1 lifetime failure remain historical evidence. Final delivered_* files supersede before_lifetime_fix_*. The repair passed targeted reproduction and the entire fresh suite. Untried post-stop ideas remain the hand-off list above. Stage-owned scope is fused_decoder.py, tests and docs only; functional runtime and future pipeline stages are unchanged.

### Host checks before independent review

Repository pre-commit passes in hooks_final_3.log. The first pass normalized trailing whitespace/endlines in rendered tables and two historical console logs; raw ops CSVs remain byte-exact in gzip with their original hashes. CSV report line endings were normalized to LF for git diff --check; values are unchanged. diff_check.log is empty. Final source SHA is unchanged. All staged paths are inside this model’s fused implementation, tests or docs; raw Tracy directories and device locks are excluded. The fresh round1 reviewer runs only read-only artifact/source analysis and may write stage_review_1.md; no hardware job remains active.

## Independent stage review

Fresh xhigh reviewer `/root/fused_stage_review_1`, round 1, returned **clean-pass**, no Required Work. Report: stage_review_1.md. The reviewer independently checked all 34 final-source/harness artifacts, 35 zero-exit driver jobs, numerical/trace/context gates, raw Watcher/profiler hashes and totals, matched baselines, candidate decisions and the explicit stop criterion. No hardware was run in review.

Review notes retained for later stages: shared convolution scratch must remain serialized; native GDN internal bindings need revalidation after upstream changes; long-context HF comparison samples output rows while advancing all state/cache tokens; full-model generation accuracy is not established here. Prefill wall timing is one warmed sample, while the main decode comparison uses 100-replay medians. The summary script delegates detailed numerical assertions to the fail-fast runners; the AST audit does not enumerate the local native alias/C++ bodies, covered here by native source inspection, runtime sentinels and profiler evidence. These are disclosed coverage limits, not missing or waived gates. Small AICLK settling differences in secondary rows/profile reinforce that sub-percent candidate differences are not claimed as resolved gains.

The reviewer authorizes proceeding to the required local checkpoint step under the user’s existing instruction. Only this tt-metal checkout has stage-owned changes; never push.

## Local checkpoint and completion

- Repository: `/home/ttuser/qwen36-27b/tt-metal`
- Branch: `main`
- Implementation, tests, artifacts and clean-pass review commit: `e759ed7e37818386b3b880af1cd191eb72bea27c`
- Parent functional stage state: `60c3c9e00911429d1ab6d4a519e8ae0c79be5c4a`
- Commit message: `Add validated fused Qwen3.6-27B decoder and stage evidence`
- Push state: **never pushed**. No other repository was modified by this stage.

The checkpoint commit's own pre-commit hooks passed. This following documentation-only commit records its SHA and marks acceptance complete; implementation and measured artifacts are unchanged. All requested fused-decoder gates are complete. Optimized-decoder, multichip-decoder, full-model and vLLM work has not started.
