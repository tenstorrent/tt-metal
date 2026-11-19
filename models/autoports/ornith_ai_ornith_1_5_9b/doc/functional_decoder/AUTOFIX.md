# AUTOFIX

Contract: full advertised-context chunked prefill must preserve causal history and support trace-safe preparation without compiling a kernel family per absolute chunk offset.

Observed apparent stall was controlled: original full-context run ultimately passed; live triage and host stack establish a host compilation wait. Source inspection identifies repeated specialization; the evidence does not attribute all elapsed time to compilation. See AUTOTRIAGE.md and triage artifacts.

Hypotheses: (1) scalar SDPA offset changes compiled key extent — confirmed; (2) shrinking fill page-table tail changes compiled stick size — confirmed by isolated hypothesis agent; (3) TILE slice offset alone forces new binaries — refuted for equal-size aligned chunks (program-cache entries still vary).

Minimum repair in `tt/functional_decoder.py`: use exact chunk page window and device-valued SDPA start. No shared code changed. Full context remains 262144. Validation passed below.

AutoDebug launcher initially failed its bubblewrap preflight because this host disables unprivileged namespaces. Retried with AUTODEBUG_SKIP_CHILD_SANDBOX=1 under the existing unrestricted execution and explicitly authorized skill-subagent scope. Child completed read-only inspection; the report is archived as AUTODEBUG.md.

Focused dynamic-offset probe initially exposed an exact op contract: flexible SDPA requires page-table stick bytes divisible by 32. The public API now pads only the attention view to 8 int32 entries, using device-side pad with valid physical page 0 in the unused entries. Causal reads remain bounded by the actual start and physical query length; the fill view still contains only actual chunk pages. This accepts arbitrary table widths without reducing context or exposing an alignment restriction.

`full_fix_probe_r2.json` / `.log` passed: lengths 32 and 1025, real weights, physical KV-page readback, changed-input/page/position traced replay, repeated-input equality, unchunked control, prefix continuation per-row contribution PCC, dropped-history negative controls and traced batch 2, with allocation tracking and runtime fallback guard. No unsafe allocation or program-cache suppression.

Advertised-context revalidation: `full_long_fixed.log` / `.json` passed all three long prefills and final-position deterministic traced decode. Both final batteries also passed on the corrected source. The earlier linear long-context evidence remains applicable: the repair changes only the full-attention path.

No speedup factor is claimed for this repair. The source eliminates two position-dependent binary-geometry families; performance is measured separately in warmed signposted runs. The original and corrected sampled PCC values agree. Requests must remain within their actual allocated cache and unpadded page-map extent, as documented in README.md.

## Watcher kernel-text capacity repair

Failing contract: both kinds must execute the real decoder with watcher checks enabled. `linear_watcher_inline_failed.log` reports GDN prep total program71856 > buffer70688; `linear_watcher_noinline_failed.log` reports71840 >70688. Both fail before dispatch and close normally. Supported noinline alone was tested and refuted as a sufficient fix; no checker is disabled.

Independent source-only hypothesis audit found `chunk_gdn_math.hpp` documents the Ct=2 (64-token) kernel near the kernel-config size limit. `chunk_gdn_compute_config.hpp` selects a smaller SFPU FP32 forward-substitution solve for internal chunks32 on Blackhole. The same default selects Horner at64. This is a supported algorithm/configuration, not an assertion suppression.

Minimal model-local correction: explicitly pass internal `chunk_size=32` for prefill GDN and build matching32x32 eye/tril/ones constants in setup. Masks remain32x96. Public1024-token prefill chunks, recurrence state, full attention and decode implementation are unchanged. Arithmetic stays HiFi4/FP32. Both normal and watcher runs use the same32-token configuration. No environment-dependent alternate model path or shared-source edit.

Discriminating test: watcher-enabled real-weight lengths1,33,1025 plus resume and traced batch2 (`gdn32_watcher_probe.json/log`). Because this changes floating-point recurrence evaluation, final linear correctness, full-context, determinism, allocation tracking, synthetic CI, watcher and profiling were rerun and all passed. See linear_final.json, linear_long.json, synthetic_pytest.log, watcher_audit.json and the accepted profiler artifacts. Old64-token evidence is retained separately.

## Profiler source-ID collision

The original profile's decoder comparisons and both signposted passes completed, but close/export failed: `populateZoneSrcLocations` found two distinct locations with FNV-fold16 ID1831. `profiler_collision_diagnosis.json` names `rd_kick` in reader_chunk_gdn_prep.cpp:281 and `pp_kdec` in chunk_gdn_math.hpp:1004. This is metadata ambiguity, so no timings from that capture are accepted. Evidence: `linear_profile_hash_collision.log` and `tracy/linear_hash_collision/`.

A source-only hypothesis audit confirmed that `TT_METAL_KERNEL_PATH` can provide an unchanged kernel under a distinct absolute source identity, with the resolved include directory included in the kernel binary hash. `tests/run_profiles.sh` creates a model-local relative symlink overlay for that reader and its four sibling headers, verifies byte equality, runs from the model directory (so CWD does not override the overlay), and uses a fresh profiler output directory. No profiler assertion is suppressed and no shared source is edited. Header symlinks are necessary for quoted sibling includes; the first reader-only attempt failed to compile and is retained as `linear_profile_overlay_missing_headers.log` and `tracy/linear_overlay_missing_headers/`.

The Tracy wrapper exported a partial CSV and exited zero even after that target compile failure. The runner now requires the target JSON, an actual VALIDATION_PASS marker, and absence of Traceback/TT_FATAL/TT_THROW before marking a capture passed. Partial reports are excluded. Successful captures must also contain both signposts and nonempty device timing rows. This prevents accepting wrapper exit status as target success.

Final profiler resolution: accepted linear/full captures have complete signpost windows, real target PASS markers, collision-free source metadata and traced decode rows. profiler_collision_resolution.json records all unchanged source hashes and symlink targets. performance_summary.json and functional_decoder.md contain the independently reconciled timings. Stage review round 1 returned clean-pass.
