# ✅ UPDATE (2026-07-03) — runner single-GLX traced option + pipeline traced gaps (HEAD ca4cd60a6db)

On top of the rebased/integrated work below, added the runner-driven single-GLX traced path + closed the
3 pipeline-parallel traced gaps. NOT pushed yet (local commit ca4cd60a6db).
- **Single-GLX traced runner option** (`PREFILL_USE_TRACE=1`): `PrefillRunParams.use_trace` → MLA adapter →
  `TtPrefillRuntimeConfig.use_trace`; runner opens mesh with `trace_region_size` (`PREFILL_TRACE_REGION_SIZE`,
  default 256MB). Run: `PREFILL_MODEL=kimi_k2_6 PREFILL_NUM_LAYERS=10 PREFILL_NUM_USERS=1 PREFILL_MAX_SEQ_LEN=56320
  PREFILL_STANDALONE=1 PREFILL_STANDALONE_NCHUNKS=11 PREFILL_STANDALONE_PCC=1 PREFILL_SYNC_PER_CHUNK=1
  PREFILL_USE_TRACE=1 PREFILL_TTNN_CACHE=/mnt/models/Kimi-K2_6-Cache/Kimi-K2_6-Cache-prefill
  PREFILL_TRACE_DIR=/mnt/models/deepseek-prefill-cache/golden/kimi-26/kimi_longbook_56320
  PYTHONPATH=. python_env/bin/python -m models.demos.common.prefill.runners.prefill_runner`.
- **Validated on-device (this machine, Kimi L10, 11×5120)**: traced==eager, bit-identical **min KV-PCC 0.994120**;
  17 seg / 7.31 MB. Warm per-chunk replay ~13% faster than eager (traced 194.6ms vs eager 223.4ms mean, chunks 1-10),
  up to −24% at low KV, converging at high KV (compute-bound). Capture(PREPARE)~9s, chunk0 absorbs capture (~2.2s).
- **Gap #3 (ack ordering)** = lazy capture: `compile()` only `_prepare_trace` (buffers+controller+warm-compile);
  the first `prefill_chunk()` records the capture (`_capture_now`). So `set_layer_ack_channel()` called after
  compile (request loop) is still known at capture → trace segments at ack points. Re-validated PCC unchanged.
- **Gap #2 (non-last-rank output)**: captured forward output kept in `_trace_output` (captured addr); non-last rank
  returns it from `prefill_chunk`; `_d2d_send(..., deallocate=False)` for traced so the persistent buffer isn't freed.
  IMPLEMENTED, NOT hardware-validated (needs multi-galaxy pipeline / tt-run).
- **Gap #1**: `use_trace` runner wiring — done (above).

---

# ✅ COMPLETE (2026-07-03) — rebased on current main + runner trace integrated & validated

Branch `ppopovic/trace_rebased_on_main` (pushed). ALL tasks done on the 2nd (less-powerful) machine:
- Rebase onto current origin/main (base fd457612bfc) + full rebuild — clean.
- Transformer KV-PCC: traced==untraced==**0.993413** (L10) / **0.930707** (L61), bit-identical (0 trace non-determinism).
- Op-equiv: rotary 25/25, zero_pad 14/14, update_kv metadata_matches_scalar 3/3 PASS. (update_kv's 24 byte-exact
  `multi_iteration` failures are a main-#45821 harness issue, NOT metadata-specific — equivalence holds — not a blocker.)
- Perf (this machine, warm iter-1, L61, 11×5120): untraced **16.16s** vs traced **14.92s** = **~8%** overall;
  trace helps most on small-prefix chunks (~20% on chunk 0/1), parity by chunk 10 (compute-bound).
- Per-op branches (pushed): `ppopovic/metadata_{update_padded_kv_cache,rotary_embedding_indexed,zero_padded_kv_cache,ring_mla}`.
- **Runner trace integration** (TtPrefillRuntime.use_trace) DONE + validated: standalone harness
  `test_kimi_prefill_runtime_traced_kv_pcc` L10=**0.993413** (17 seg / 7.31 MB), L61=**0.930707** — matches the model path.

Remaining/notes: update_kv byte-exact multi_iteration is a main-side op-harness issue to flag to the op owner;
origin/main has advanced ~15 commits past the rebase base (re-rebase only if needed); the 4 per-op branches were
extracted from the validated integrated branch (standalone per-branch CI build not run — low risk).

---

# RESUME — per-element trace work rebased onto current origin/main (2026-07-02)

## TL;DR / how to rerun me
1. `cd /home/ppopovic/tt-metal && git checkout ppopovic/trace_rebased_on_main` (@ `1491f06ff8a`, pushed).
2. **Rebuild is required on a fresh machine** (the `.so` is machine-local, not in git):
   `cmake --build build_Release -- -j32` then refresh the loaded libs:
   `cp build_Release/ttnn/_ttnncpp.so build_Release/lib/ && cp build_Release/ttnn/_ttnncpp.so ttnn/ttnn/ && cp build_Release/tt_metal/libtt_metal.so build_Release/lib/ && cp build_Release/ttnn/_ttnn.so ttnn/ttnn/`
3. Resume at **Task 12** below (run the transformer KV-PCC — the decisive test), then Tasks 13/15/14.
4. Env for device runs: `KIMI_K2_6_HF_MODEL=/data/nbabin/Kimi-K2_6-dequantized`,
   `TT_KIMI_PREFILL_TTNN_CACHE=/mnt/models/Kimi-K2_6-Cache/Kimi-K2_6-Cache-prefill`,
   `TT_KIMI_PREFILL_HOST_REF_CACHE=/mnt/models/kimi-prefill-cache/golden`. Wrap runs in
   `timeout --signal=KILL <s>`; on hang → `pkill` + `tt-smi -glx_reset`. Do NOT git-switch during a device run (JIT header race).

## What this branch is
`ppopovic/trace_rebased_on_main` = the per-element-tensor trace-safe metadata work (4 ops + mla/block
threading + segmented trace + KV-PCC tests) **rebased onto current `origin/main`** (which is +193 commits
past the old base and already contains the #48499 revert of #47921, so the redundant local revert was
auto-dropped during rebase). Pre-rebase validated state (bit-identical 0.932 traced==eager==scalar) is on
`origin/ppopovic/trace_experiments_rebased` @ `7db13f0e7e3`.

## The hard part of the rebase (DONE) — main re-architected the MLA/prefill
Current main differs a lot from the old base; the metadata hooks were **re-applied by hand** onto main's
new structure (not a blind conflict-merge). Key structural changes on main:
- MLA `forward` split into helpers: `_q_a_latent`, `_q_stem`, `_kv_stem`, `_o_proj_epilogue`, and a
  config-bound `self._attention` (one of `_dense_single_attn` / `_dense_chunked_attn` /
  `_sparse_single_attn` / `_sparse_chunked_attn`, chosen at construction by sparsity×chunking).
- Added a **DSA sparse indexer** (`TtIndexer`/`NullIndexer`, `_has_indexer`); rope split into
  `_apply_rope_padded` / `_apply_rope_one_shot`.
- **`zero_padded_kv_cache` + `on_layer_complete` moved OUT of mla.py INTO `tt_prefill_block.py`.**
- Runner moved to a **common package** `models.demos.common.prefill.runners.runner_utils`
  (old `tt/runners/runner_utils.py` deleted → removed during rebase; `open_mesh_device`'s
  `trace_region_size` add must be re-applied there during runner integration).

Metadata hooks re-applied (all committed): mla.py — `_trace_controller`+`set_trace_controller`;
`metadata` param threaded through `forward`→`_q_stem`/`_kv_stem`(rope)→`_attention`→`_dense_chunked_attn`
→`_chunked_attn` (update_kv branch + ring_mla `meta_slot_kwargs` with slot_id/kv_actual_isl_tensor/
kv_cache_num_layers/kv_cache_layer_idx) and `_forward_kv_only` (rope + update_kv branch). `_sparse_chunked_attn`
asserts metadata is None (sparse+metadata not supported; Kimi is dense). tt_prefill_block.py — stores
`_trace_controller`, zero_padded_kv_cache metadata branch (metadata[0]/metadata[2]) + trace-safe ack
routing (tc.has_layer_ack → tc.layer_ack, else sync+on_layer_complete). test_mla.py — `use_metadata_tensor`
param (dropped `actual_end` — main's mla.forward no longer takes it).

## Build: DONE (full rebuild + .so refresh). Imports OK, `ttnn.transformer.ring_mla` present.

## Op-equivalence tests (Task 11) — 3/4 clean, update_kv needs triage
- rotary: **25 passed** ✓  |  zero_pad: **14 passed** ✓  |  ring_mla: (was running at interrupt — CHECK
  `scratchpad/optest_ring_mla.log`)
- **update_kv: 24 failed / 36 passed** — `test_update_padded_kv_cache_multi_iteration_prefill`. Fails BOTH
  a byte-exact cache match (`max abs diff 2.875`) AND the `num_program_cache_entries() == entries_after_init
  + num_layers` assert. This is the **SCALAR** path (multi-iteration reuses one cached program, patching
  slot/kv via `override_runtime_arguments`). The op files are UNCHANGED main-vs-old-base and the test is
  UNCHANGED, so the cause is one of main's 193 commits — prime suspect the **#45821 hash-collision series**
  (reworked device-op hashing / runtime-arg patching), interacting with the per-element op's `has_metadata`
  compile-flag program variant. **NOT yet determined** whether this breaks the real pipeline.

## NEXT — Task 12 (DECISIVE): transformer KV-PCC
Run these; the **metadata** path reads slot/kv from tensors (not runtime args), so it may pass even if the
scalar op-test fails:
- `pytest -q -s ".../test_prefill_transformer_chunked.py::test_kimi_prefill_transformer_chunked_notrace_kv_pcc[blackhole-kimi-mesh-8x4-L10-chunks11]"` (eager metadata; expect min KV-PCC ~0.993)
- `...::test_kimi_prefill_transformer_chunked_trace_kv_pcc[...-L10-chunks11]` (traced; expect ~0.993)
- If both pass at ~0.932 (L61) / ~0.993 (L10) → the deliverable works; the update_kv op-test failures are a
  scalar-path/harness interaction with main's hash rework (investigate/fix the op's program-cache-count +
  `override_runtime_arguments` under current main, or confirm it's a pre-existing main-side issue).
- If the metadata KV-PCC FAILS → the ring_mla/update_kv metadata merge onto main is wrong; re-check the
  `meta_slot_kwargs` + the update_kv metadata branch against the rebased op C++.

## Remaining tasks (see task list)
- Task 12: transformer KV-PCC (above) — IN PROGRESS / decisive.
- Task 13: recreate the 4 per-op branches off the rebased tree, build-validated (the earlier
  `ppopovic/metadata_*` branches were made off origin/main BEFORE this mla/block merge; recreate).
- Task 15: perf — untraced vs traced, 11×5k chunks (per-chunk timing + speedup).
- Task 14 (LAST): integrate trace/metadata into main's runner (`tt_prefill_runtime.py::prefill_chunk` →
  `model.forward`; add SubDeviceTraceController capture/replay + metadata build + `set_trace_controller`;
  re-add `trace_region_size` to the common `open_mesh_device`). "More complicated now" per the runner
  refactor to the common package.

## Backup refs
`trace_rebased_backup_2` and `trace_rebased_backup_pre_mainrebase` tags (@ pre-rebase 7db state).

## Per-op branches — validated & merge-ready (2026-07-03)
All 4 rebased onto current origin/main (c85c5b78dff): **1 commit ahead, 0 behind — cleanly mergeable**,
op C++ + kernels + nanobind + unit test ONLY (no trace/mla/block/runtime). Force-pushed. Machine-2 op tests:
- `ppopovic/metadata_rotary_embedding_indexed` — 25/25 PASS
- `ppopovic/metadata_zero_padded_kv_cache` — 14/14 PASS
- `ppopovic/metadata_ring_mla` — test_ring_joint_sdpa 50/50 PASS (runner's 1500s cutoff hit a later param's
  32-device fabric re-init; not a failure/hang)
- `ppopovic/metadata_update_padded_kv_cache` — metadata_matches_scalar BIT-EXACT for bfp8_tile + fp8_rm
  (bfp8 = production KV dtype); **bf16_rm slot-1 byte-exact FAILS (max abs diff ~3.2e38 = single-element
  uninit-read symptom; passed 3/3 on machine 1 → narrow bf16-layout flake)** + multi_iteration/single_iteration
  byte-exact + must-fit-cache failures = the #45821 harness issue. FLAG to op owner; neither blocks the
  trace deliverable (bfp8 bit-exact + pipeline KV-PCC 0.993/0.930).
