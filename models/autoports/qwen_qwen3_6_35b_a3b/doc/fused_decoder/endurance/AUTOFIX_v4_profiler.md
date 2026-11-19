# V4 profiler collection recovery

Failing contract: complete per-operation prefill/decode CSVs with16 identical traced replays, correct current shapes, no marker drops, clean close. Original v4 capture aborts at close with an unmatched TRISC-FW end and therefore supplies no accepted device profile. Exact failed capture and log are `profile_v4_failed_01/`.

[AutoDebug diagnosis](AUTODEBUG_v4_profiler.md) identifies a source-level unsafe live-high/low clock read and a distinctive low-word4 rollover signature. Root verified `kernel_profiler.hpp:253–257` versus the sampled-high reader in `risc_common.h:235–240`. The individual failed event remains an inference because the matching start record is absent. Ordinary buffer overflow is not established: no drop warning, guaranteed FW prefix handling, per-replay drains and20000 support count were already present.

Prioritized discriminator: one unchanged fresh-directory collection, retaining count20000, no-op-info-cache,16 measured replays and all completeness checks. Native edits, synthetic marker repairs and assertion suppression are excluded. If the same failure recurs, preserve its timestamp and investigate a timing-only outside-window contrast or diagnostic raw marker dumping; diagnostic weakened processing never substitutes for an accepted CSV.

The failed process exited, fuser found no holder, bounded tt-smi listed all4 chips and physicalchip0 mesh/add/readback/close passed. No reset/locks/process kill was used. Source and harness are unchanged.

Retry command: `QWEN_FUSED_ARTIFACT_DIR=models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/profile_v4_retry_01 bash models/autoports/qwen_qwen3_6_35b_a3b/tests/collect_fused_perf.sh`, after clean watcher/tracking/profiler environment and sourced lane setup. New directory existence was rejected before launch. Outcome pending. A passing retry would mitigate collection only, not repair the native clock primitive.


Retry completed cleanly for both layers. All16 measured replay sequences are identical;171 linear and68 full operations per replay, no dropped markers, current source b163. Aggregate checker passes after installing fresh reports. Raw retry remains under `../profile_v4_retry_01/`; qualified profiles preserved under `../profile_v4/`. This resolves the required collection gate through a validated retry; the native clock race remains an unmodified upstream limitation, not claimed fixed.
