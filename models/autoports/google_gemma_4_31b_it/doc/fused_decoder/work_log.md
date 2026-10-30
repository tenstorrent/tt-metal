# Fused decoder stage work log

Stage 02 complete; independent stage-review returned clean-pass. Scope: fused_decoder.py, tests, docs.
Pinned HF revision 842da3794eaa0b77d5f08bae87a17459d91ff475.
Functional checkpoint 802858ebfe12eda34e85e76c0d333ff90172c40c; functional files unchanged.

Startup read bringup/run_contract.json and bringup/agentic_evals/inputs.json.
Enabled packages are the supplied tt-model-bringup 0.1.22 and tt-autodebug 0.1.8.
`python <installed model-bringup>/scripts/environment.py` verified dependency exports.
Local hardware only, serialized commands. Process inspection found no running device job.
`timeout 60 tt-smi -ls --local` failed because executable is absent, as in stage01.
`python_env/bin/python` TTNN open/close MeshShape(1,1) passed; UMD discovered four
Blackhole PCIe chips, firmware19.17.0, KMD2.10.0. Selected one ASIC, grid11x10.
No reset, lock cleanup, or remote publication performed.

Initial candidate: dedicated rotate-half RoPE (prefill) and sharded HF RoPE (decode),
GELU_TANH folded into gate/up multiplication. All other math/config unchanged.
`TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0
python_env/bin/python -m models.autoports.google_gemma_4_31b_it.tests.run_fused_decoder
--seq 33 --result models/autoports/google_gemma_4_31b_it/doc/fused_decoder/initial_sliding.json`
passed real weights, changed position/page rows and repeated traced replay;
initial_sliding.log/json record PCC. No tracker exemptions.

`PATH="$PWD/python_env/bin:$PATH" python_env/bin/python -m
models.autoports.google_gemma_4_31b_it.tests.run_fused_profile_queue` collects initial
candidate profiles in B1,32,16,8 order for both kinds. See profile_queue.log and
per-collection manifests. Profiler runs have watcher and tracker disabled; all
whole-layer windows include device gaps and all operations.

Candidate sweep commands/results: candidate_queue.log and candidates/manifest.json.
Full attention first sharded-RoPE candidate collided with SDPA static L1 buffers.
Adapted path releases K/V update temporaries before SDPA and returns full-attention
Q to DRAM (the long-context SDPA requires ~1.46MB static L1). Retaining Q still
collided after lifetime fix (full_lifetime_retry2.log), so the Q transfer has an
observed capacity reason. K stays sharded until its cache write.
Full head concat at D512 exceeded L1 (2,208,896 bytes vs1,572,864). Adapted two
16-head groups fit. Decode concat outputs32 padded batch rows; metadata-only
sequence view restores logical B (first reshape failure and passing repair in
full_lifetime_retry3/4 logs). Final adapted real-weight full-attention check passes
prefill0.999983506 and decode0.999990444, including changed positions/pages.
No op/library source edits. Extended candidate sweep tests layout-adapted paths
and shared full-attention K/V norm reuse, with exact commands in
candidates_adapted/manifest.json. Host timings rank candidates only.

Combined candidate matrix: candidates_combined/manifest.json and combined_queue.log.
The strongest B1 candidate before cos/sin sharing is sliding split-unpacked+
scalar/GELU+fused-cache at~2999us host-traced vs functional~3031us; full packed-QK+
shared-KV-norm+fused-cache at~3258us vs functional control. Host timings are ranking
measurements, not device-time telemetry. Full decode specialized QKV split was
adapted by duplicating K channels before the op (because its API has no tied-KV
argument); it passed but was slower~3304us. No duplicate K weights are retained.
MLP packing and matmul-GELU candidates pass PCC but are slower; they are not selected.

Final source being validated folds the selected graph directly into standalone
FusedDecoder and additionally shares the decode cos/sin reshard between Q and K.
It has no import or runtime call to FunctionalDecoder. Experimental candidates
now use tests/fusion_baseline.py, a preserved copy of the initial stage02 candidate,
so changing final source does not silently change candidate baselines.

Roofline calculation now estimates ordinary physical DRAM tensor I/O from each
operation in a complete replay and handles active KV/page/tile updates explicitly.
BinaryNg hash-cached profiler shapes retain the first prefill shape (1024 rows)
even for decode: current source's shape-independent program reuse and the actual
B<=32 runtime input geometry require replacing those cached dimensions in the
estimate. This does not change timestamp denominators. The earlier inherited
logical-intermediate model is superseded by summarize_fused_profile.py.

Batch matrix found a real crossover: sliding B32 split4387us vs separate4468us;
B16 split3628us vs3633us; B8 split3250us vs3215us. Therefore larger-batch splitting
must not be enabled at B8 merely because B32 benefits. Full B32 split6370us vs
separate6485us and B16 split4754us vs4776us; its B8 comparison is evaluated separately.
These host timings include shared cos/sin transfers and preserve the same BF16/HiFi4
policy. Direct-sharded split outputs consistently hit the RMSNorm height-shard
validation; adapted DRAM norm inputs pass. No library changes are required.

Last coherent combination: sliding_b32_packed_split_correctness.json passes;
sliding_b32_packed_split_timing.json shows prefill2466.75ms and decode4390.13us,
versus unpacked split prefill2525.28ms/decode4387.11us. To keep both gains, final
sliding B32 prefill uses an additional packed QKV setup weight; decode stays
separate. Extra persistent bytes =5376*(8192+4096+4096)*2 =176160768 (168MiB).
Copy reason: measured58.5ms B32 prefill reduction while preserving fastest decode.
All configured variants are prepared at construction; no late weight packing.
Final maximum-context validation includes this additional allocation. Full Q/K
packing retains no duplicate weight. Cache bytes/layout/dtype remain unchanged.

Full B8 dedicated decode split3963.2us loses to separate3945.1us. Both layer kinds
therefore use separate reshapes below B16, dedicated splitting at B16/B32.
Final full split path omits the unused duplicate-K DRAM conversion; normalized V
already supplies the shared raw-K/V reduction.

Final acceptance command: `python_env/bin/python -m
models.autoports.google_gemma_4_31b_it.tests.run_fused_acceptance`.
It serializes final smoke, per-batch timing, direct equivalence, context/boundary/
batch/watcher checks, synthetic pytest and8whole-layer profiles. Individual exact
commands, source hashes and exits are in acceptance_manifest.json and nested queues.

Compatibility guards added after the final configured-batch timing runs: dedicated
head splitting is limited to16<=B<=32 (the op's supported batch range), and fused
cache update falls back to two individual cache ops if two disjoint B-core grids
cannot fit. All other fusions remain active; this never calls FunctionalDecoder.
These guards preserve the pre-existing interface outside the tuned1/8/16/32 set.
The selected device graph for every configured batch is unchanged. Sliding maximum-
context run had already imported the previous source; its B1 path is identical.
Final later profiles and remaining validation import the guarded source. Additional
boundary checks cover B15/17,31/33 and55/56, including both core-capacity branches.

Independent stage review identified the batched paged_fill_cache interface as an
untried applicable rewrite. Prepared a real-weight candidate and a serialized
correctness/timing queue; no exhaustion claim until its measured disposition.
The profiler audit now also compares every replay op-ID set to the complete
TT_DNN capture body and TT_METAL_TRACE_ENQUEUE_PROGRAM count from host trace
metadata. This strengthens validation without changing the measured device graph.

Final context and validation queue passed all12 jobs. Both kinds preserve262144;
non-aligned262143 and exactmaximum PCC are in *_context_final.json. Full attention
exactmaximum PCC0.99819170 versus functional0.99818020: no material degradation.
Both watcher runs passed, including continuation, changed page rows and10 trace
replays with allocation tracking and program-cache tracking enabled. The local
watcher_audit.json records22 retained files and zero failure-keyword matches.
Synthetic pytest passed all4 tests. Final profiles are being collected separately
with watcher disabled and profiler capacity4096.

Batched-fill review finding: real-weight B32/S1025 correctness passed for sliding
and full attention, including repeated traced decode and updated inputs. Initial
single prefill samples were inconclusive for full attention, so compare_batched_fill
ran a controlled ABBA comparison with shared weights/inputs/cache and four samples
per path. Per-row -> batched median host prefill ms:
sliding B8 618.920->616.378, B16 1251.293->1247.326, B32 2475.338->2464.077;
full B8 856.321->855.064, B16 1731.676->1729.760, B32 3493.762->3488.440.
All six favor batched fills. Selected implementation prepares110 row indices at
setup, slices one device index vector per prompt, then uses two batched fills per
chunk. B1 retains its existing scalar fill path; decode arithmetic is unchanged.
The tiny index payload is440 bytes, separate from the168MiB packed weight copy.
Batches beyond the prepared grid capacity retain per-row fills, preserving the
functional interface rather than adding a public batch restriction.

run_batched_fill_acceptance initially requested S1025 with --extended, whose
existing harness contract requires S>=2049. The explicit assertion stopped the
queue after prefill; device teardown was clean. Fixed the queue argument to2049,
preserved input_error_* evidence, and reused only passed jobs with identical
commands and source hashes. No numerical check was weakened. New B32 watcher,
equivalence, boundary and affected-profile checks are underway.

Additional compatibility testing exposed full B55 static-L1 overflow. Functional
B55 control fails in its individual update too. B36/B37 controls and source buffer
accounting then isolated lower K placement after fused RoPE. The full B>=34 path
now relocates K through DRAM with an explicit original-shard deallocation before
resharding. B36 passes both implementations; B37 fails the same inherited physical
limit, with repaired K address moved1409024->1507328. This preserves the functional
boundary for the measured S33 geometry. Exact artifacts and arithmetic: AUTOFIX.md.
No B<=32 measured graph changes, so the successful configured acceptance/timings
remain valid. Final profile collection uses source cc1b358f90a2fadab24cbb3dd9826ffe32015fab46ccefac25c0c6c11c658149.
The previously failed boundary manifest/log is preserved as inherited_l1_*;
final checks replace that invalid extrapolation with supported B33/B34/B36 checks,
a watched B36 relocation run, explicit controlled physical-limit cases, B110/111
prefill-only equivalence, and the six affected batched profiles.

## Final evidence reconciliation

Final production SHA256: cc1b358f90a2fadab24cbb3dd9826ffe32015fab46ccefac25c0c6c11c658149. All eight measurements are reconciled in performance_summary.json and README.md, including before/after PCC, whole-device latency, profiler conclusions and candidate timing variation. Six affected batched profiles were recollected; B1 retains the unchanged device graph capture. Final checks passed in batched_acceptance_manifest.json, batch_boundary_manifest.json, prefill_index_boundary_manifest.json and full_batch_capacity_manifest.json. No hardware task remains active. The final context-contract check and independent review are the remaining closure checks.

Final closure checks: strict fused-decoder context-contract checker passed with target=supported=262144. All staged-file pre-commit hooks passed after adding missing terminal newlines to generated JSON (no measurement or source changes). Telemetry target workload and both whole-layer roofline formulas validated; packet written at the exact requested continuation path.

Independent final stage-review returned **clean-pass**, with no required work. All earlier findings were repaired and rereviewed; see stage_review.md. Stage-owned source/tests/docs are checkpointed locally below; no push or later-stage work performed.
