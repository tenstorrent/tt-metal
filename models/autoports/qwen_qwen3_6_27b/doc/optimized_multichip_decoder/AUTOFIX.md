# AutoFix: compact prefill representation

Diagnosis: debug_flat_prefill/AUTODEBUG.md. Failing performance contract:
independent-row residual/norm/MLP operations repeatedly expand and compact
physical rows at B32/S33, costing roughly 4ms across the entire prefill.
No arithmetic correctness defect is alleged.

Priority 1: keep the post-attention residual/MLP flat. Discriminating check:
matched whole-layer prefill timing and PCC, then profile reshape count/time.
Priority 2: combine GDN corrected projections before BF16 temporal expansion.
Priority 3: retain compact representation through attention column projections
and the surrounding chunk, with explicit temporal bridges.

The first candidate adds policy flat_finish, preserving logical batch for
norm route selection. It excludes one-token tails so their existing correction,
conv and norm dispatch remains unchanged. Source edits are in the TP4 subclass;
the frozen single-chip baseline remains unchanged. Validation pending.

## Persistent Linear RS intermediate
AutoDebug identified half-sized staging; parent verified native compute_output_specs doubles padded_shape[0] for Linear. Minimal repair doubles only that allocation, leaving shared-role reuse, barriers, precision and placement unchanged. Exact original failure is retained in family3_persistent_l0_b32_s33.log. Discriminating rerun uses persistent_fix1.json; outcome pending.

Persistent repair validation: all eight TP4 B1/S512 and B32/S33 cases pass, including native and fused-MMRS paths for both layer kinds. Same shared-role buffer cache and barrier behavior retained. Pure persistent B32 decode: GDN .707717→.694858 ms; attention .501684→.488853 ms. Fixture retains four family3_mmrs_persistent_* tags that had not run before the repair. Larger stress, watcher and resource-ownership gates remain outstanding.

## Native matmul-reduce-scatter NoC ownership repair

Diagnosis independently verified in mmrs_noc_candidate.md against unchanged factory. The dedicated NCRISC reader issued W reads on the BRISC writer’s NoC1, changing shared counters without updating that RISC’s private exit accounting. Kernel-only routing to reader NoC0 passes direct-native M1056 attention and MLP under watcher, including 50 queued trace replays. MLP native plan is resident, M-on-X grid11x8, so this exercises the formerly unsafe branch rather than the baseline128-row avoidance. Existing failing control: ../multichip_decoder/debug_narrow_watcher/.

Canonical repair changes native factory W_READ_NOC to0 and consumes/asserts both operand compile arguments in the reader. Writer checks/barriers unchanged. Build: `cmake --build build_Release --target ttnn -j 12`. First canonical attempt compiled against the old installed runtime and failed the new ownership static assertion before native launch. Corrected installation using `cmake --install build_Release --component ttnn-runtime` and `--component tt_pybinds`; exact installed binaries/source hashes recorded in mmrs_canonical_build.json, build/install logs retained. Installed direct-native MLP M1056 watcher probe passes with PCC .9999927946, first/last replay identical. No reset or counter suppression. Watcher timings are not performance evidence. Full-layer larger-chunk sweep mmrs2 follows; keep/reject pending those results, resource pressure and final watcher coverage.

Provenance correction: lane-env.sh overrides TT_METAL_CACHE/TT_METAL_CACHE_DIR inside run_job.sh. Earlier MMRS probes actually used shared /home/ttuser/qwen36-27b/tt-metal-cache (sidecars correctly record it), despite caller cache requests. Canonical static assertion failure with oldhost and success afterinstall distinguish the repair, but these are not fresh-cache evidence. Explicit env overrides after wrapper setup are required and a fresh canonical watcher rerun is scheduled.

## Compact L1 output during non-aligned prefill resume

`cache_owner_b32_v2` passes initial PCC, full-cache equality, exposed/alias/replacement/fresh-table checks, then fails on the suffix after prefix1: reduce-scatter static CBs end at309184 while an L1 buffer begins at197888. The selected L1 MLP policy also keeps the attention output projection in L1. `_local_compact_attention_decode(..., decode=False)` expanded that private compact output to `[B,1,5120]`, adding per-user physical tile padding before the prefill reduce-scatter. The ordinary compact decode contract does not require that expansion.

Minimal repair moves the private projection to DRAM before legacy/public-layout expansion, while actual compact decode returns its previous L1 output. Same B32/S33, qknorm9 policy, oracle, watcher10 and tracking1/skip0 rerun `cache_owner_b32_l1resume_v3` passes all22checks: initial oracle, complete output/cache equality, aliases/remapping, prefixes1/31/32/33, ordered capture and changed-input/table replays. The failed result remains unchanged. This is an internal layout adaptation, not a restriction to aligned public lengths. Final cumulative/default gates remain necessary.

## Conventional Linear all-gather/matmul

Diagnosis: `debug_conventional_agmm/AUTODEBUG.md`. Verified repair specification and CPU tests are in `linear_agmm_repair_notes.md` and `linear_agmm_admission_notes.md`. Explicit Linear receiver scheduling and nine-word parsing are paired with persistent-scratch startup admission that preserves future barrier credits. The support envelope is bounded to the model's TP4 interleaved layouts; Ring behavior remains separate. Native buildv1 exposed a C++ span comparison error in new host validation; buildv2 uses ranges equality and passes. Runtime/bindings installed only after devices closed; hashes and commands in `linear_agmm_canonical_build_v2.json`. First fresh-cache command used a relative cache path, causing firmware dependency-file creation errors before operation execution. It is preserved and retried with an absolute fresh-cache path. No device result is claimed yet for the fused repair.


Native fused repair validation now passes fresh-cache 1D/2D smokes, nine unsupported-contract checks, all TP4 K4/K8 and B1/B32 pattern cases, and both factories' 50 queued plus 50 three-call trace replays with deliberate rank skew. Exact receipts: `agmm_linear_followup10_summary.json`, `agmm_linear_2d_b32_ownership_skew_v6.json`, and fresh-cache v2/v3. Independent CPU review `linear_agmm_regression_review.md` found two remaining host-admission gaps: persistent scratch logical shape and complete mesh coordinates. Negative tests are prepared; source guards/build and fresh device verification remain pending. No supported Ring behavior change was found by the review.


Both scratch admission guards are now built/installed (`linear_agmm_canonical_build_v3.json`) and independently rereviewed. `agmm_linear_guard11_1d_b32.json` and `agmm_linear_guard11_2d_b1.json` each pass15 expected native rejection checks plus supported numerical calls under watcher10/tracking1/skip0. The logical M1/M32 mismatches and same-mesh coordinate subset reject both before and after a valid call. These are cold/warm cache-environment checks; actual cache-hit validation delegates to the identical native validator by source. Expected TT_FATAL messages in these negative-test logs are deliberate and accompanied by exact-message checks, not failed positive runs. Devices close normally.

## Zero-interval watcher timing harness repair
Source diagnosis: tt_metal/llrt/rtoptions.cpp1467 enables watcher by variable
presence, even value0; get_watcher_interval becomes0. The accidental perf15
command stalled startup~94s under tight watcher polling, then passed numerics
and closed normally. Repaired launcher rejects watcher=0; newfocused probes
classify any present watcher as instrumentation. Perf16 unsets watcher/profiler.
No native change/reset/termination. Evidence: zero_watcher_command_audit.json,
norm_gate_perf15_invalidation.json, bounded watcher capture and originallog.

## Resume17: actual poison upload and native recurrence trace ownership

QK/RoPE probe v1–v4 failed its fixture before candidate admission. Initial
hypotheses were slice/reshape erasure or CPU read masking. v4 localized loss to
full physical upload. Source inspection found `from_torch` defaults
`preserve_nan_values=False`, permitting device tilization/typecast. v5 records
the decisive observation on all4 ranks: intended BF16 NaN0x7fc0 became
infinity0x7f80. Explicit `preserve_nan_values=True` plus raw full-tile setup copy
preserves exact NaN bits at upload, sharding, logical destination and live
input. `qkrope_smoke_v5` passes candidate/reference equality, changed inputs,
rebound addresses, trace allocation positive control, and watcher10 with
tracking1/skip0. No numerical threshold was relaxed. The raw-copy fixture
owns the logical-versus-physical padding distinction without relying on
reshape implementation details.

Native recurrence v1 compiled and passed eager checks but retained a new native
output allocated during a later capture. Replaying an earlier capture correctly
failed the allocation tracker for Buffer1031. This was not suppressed. v2 moves
output/reset-source ownership before all captures, copies final native output
into stable storage within the measured operation, releases capture temporaries,
and adds an intentional unsafe-allocation positive control that checks rejection
without executing unsafe replay. v2 hardware result is pending; extra output
copy is counted in timing and memory.
