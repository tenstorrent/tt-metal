# t409 — VSA port onto t400 (FastH3 Turbo)

Branch ttp/t409-port-vsa-sparse-attention-into-the-fasth, base a15eb37ec6a (t400), code commit d39d16e5c9b.

## Done
- C++ ops from fasth3-opt: vsa_sdpa, vsa_ring_sdpa, vsa_kv_gather (+ kernels, nanobind, cmake), unit tests under tests/ttnn/unit_tests/operations/sdpa/test_vsa_*.
- Python: vsa_geometry.py, vsa_stages_minimax_h3.py (MiniMaxH3VSAConfig, MiniMaxH3VSACoarseStage), vsa_reference/, attention VSA branch (agmm/to_out calls adapted to t400 signatures), transformer `vsa_config` + `set_vsa_stage`.
- Pipeline: opt-in `MINIMAX_H3_VSA_SPARSITY=<0..1)` (unset/empty = dense, code path unchanged). VSA tile order is realized only by permuting host index tables: assembly indices, video/audio out indices (via untile_index), rope position ids and AdaLN/timestep row indices (permute_metadata; pad slots repeat a valid row). rung = geometry.padded_len under VSA. Stages cached per geometry and never freed (traces replay their tensors). DiT weight cache subfolder gets a `_vsa` suffix. t2va/fl2va only.
- Host tests: test_vsa_geometry + test_vsa_oracle: 24 passed on g15blx02 (no device).
- test_vsa_attention: `_run_tt` now passes logical_n (t400 forward signature).

## Not done yet
1. C++ compile check (never built against the t400 base; ops written against Aug-31 main).
2. blx01 setup under /var/tmp/fasth3/t409: worktree + incremental C++ build (ionice -c3, nice).
3. Device: test_vsa_attention_minimax_h3.py::test_vsa_sparsity0_matches_ring_path (4x8 full mesh, PCC>=0.9995 vs dense ring).
4. After #408: JIT-fill job, then dense vs VSA-0.9 A/B at 10 s seed 0 in one job (both arms), via device-lint.sh. Template: tt-project/t400/run400.sh (drop its setsid re-exec; use timeout --foreground -k 10).

## t416 run 1565 (2026-10-11 22:36 PDT)
- Steps 1+2 merged: fresh Release build of the VSA code (d39d16e5c9b) on blx01 at /var/tmp/fasth3/t409/b
  (worktree of /var/tmp/fasth3/t48's repo, submodules from t284), driver tt-project/t409/build409.sh, run as
  `ttp detach --remote g15blx01 --dir /var/tmp/fasth3/t409 build409` (ionice -c3, nice 10, 24 jobs).
  Log /var/tmp/fasth3/t409/build409.log, marker build409.marker (rc=0 step=done = compile check passed and
  ttnn.transformer.vsa_sdpa / vsa_ring_sdpa import). An incremental build was not possible: ninja paths are absolute.
- Cherry-picked #408's test change (bff7d82040c -> 65689931ab0: 6 s duration + MINIMAX_H3_TURBO_SAVE_DIR) so the
  A/B uses exactly the #408 test. Python only: blx01 tree can be moved to it with `git checkout` (no rebuild).
  blx01 needs a bundle with 65689931ab0 first (t409.bundle has only up to a2dee6756f1).
- Step 3: tt-project/t409/chain416a.sh detached on g15blx02 (run dir state/runs/1565/chain416a.{log,rc}): waits
  for build409.rc, submits spec attn.txt (run409a.sh, -t 600 unmeasured) via tt-project/t288/blx01-submit.sh,
  retries once if not completed and not a test failure. Ends with "T416A_DONE <rc> <status>".
  Result: /var/tmp/fasth3/t409/out_attn/run.log on blx01 (PCC line from assert_quality).
- Step 4 script ready: tt-project/t409/run409b.sh <tag> dense vsa0.9 (one pytest per arm, 90 s watchdog, warm
  deadline 400, per-arm pytest timeout T409_PY_TIMEOUT). Size it from #408's 10 s job wall time. If the two arms
  do not fit in 600 s (+50% rule), run one arm per job. Before the VSA arm: one cold fill job (new `_vsa` DiT weight
  cache + VSA kernels).
- Note: I moved ttp/t409-... to d39d16e5c9b by mistake with `git branch -f` and restored it to a2dee6756f1 at once
  (update-ref); the t409 worktree was clean.

## t416 run 1597 (2026-10-12 00:05 PDT)
- Build409 finished 2026-10-12 05:42Z rc=0: the VSA C++ ops compile on the t400 base (compile check passed).
- chain416a (run 1565) never submitted: device-lint refused run409a.sh for `setsid`. Both run409a.sh and
  run409b.sh now run pytest in the broker's process group under `timeout --foreground -k 10` (b: watchdog stops
  $PG and its children by pid). Both pass device-lint and killscan; copied to blx01 /var/tmp/fasth3/t409/.
- chain416a restarted from run 1597: log $RUN/chain416a.out (state/runs/1597), rc chain416a.rc.
  blx01 is busy with #408/#421/#430 jobs; the chain waits for 2 clear minutes, then submits.
- Next: read /var/tmp/fasth3/t409/out_attn/run.log (PCC). Then step 4 once #408 is done.

## t416 run 1611 (2026-10-12 01:10 PDT)
- chain416a (run 1597) ended T416A_DONE 5 submit-rc2: device-lint refused because /var/tmp/fasth3 on blx01 was
  455 GB (cap 450). Freed ~2 GB: removed the done #423/#427 worktrees (t423/b, t427/b; clean, commits on
  g15blx02 branches) and JIT dirs from 10-06/07 in home/.cache/tt-metal-cache. Now 453 GB. Not removed (used by
  open tasks or a coordinator call): models/ltx-2.5 (67 GB, idle since 10-10), dit-h3hf transformer_precomputed_adaln
  (38 GB, #421), t408/b (#422 off arm), t428 builds.
- Code 53f7b6b707b: VSA builds `to_gate_compress` only with MiniMaxH3VSAConfig(gate_compress=True). The default VSA
  model has the dense weight set, so the pipeline reuses the dense DiT cache (no `_vsa` suffix): saves a 63 GB cache
  on blx01 (it would also push / past lint's 88% cap) and the cold weight fill. A nonzero gate in a checkpoint
  without gate_compress raises. Host tests 24/24. blx01 t409/b checked out at 53f7b6b707b (Python only, no rebuild).
- chain416a now waits (<=12 h) for blx01 /var/tmp/fasth3 <= 449 GB before the broker-clear wait. Restarted detached
  from run 1611 (state/runs/1611/chain416a.out).
- Step 4 no longer needs a separate weight-cache fill for VSA; one JIT warm-up of the VSA kernels is still needed.

## t416 run 1613 (2026-10-12 01:10 PDT)
- Host reboot killed chain416a (run 1611). Restarted unchanged from run 1613 (state/runs/1613/chain416a.out). blx01 /var/tmp/fasth3 still 453 GB, so it waits for the cap. Branch push of 502ae371602 landed.

## t416 run 1647 (2026-10-12 02:50 PDT)
- chain416a (run 1613) passed the cap (430 GB at 08:14Z), then ended T416A_DONE 5 submit-rc2 at 09:46Z. Cause: attn.txt
  had no ENV line, so blx01-submit.sh ran `run-bg ... -e -t 600` and the broker CLI refused (rc 2). Lint passed.
- Fix: tt-project/t409/env409.yaml (t400 env with t409/b paths), copied to blx01 /var/tmp/fasth3/t409/; attn.txt ENV= it.
- Direct submit at 09:50Z got 75 (another smarton job on blx01). chain416a restarted detached from run 1647.

## t416 run 1655 (2026-10-12 04:55 PDT)
- chain416a (run 1647) submitted blx01 job 939 at ~11:29Z: failed at collection, `No module named
  'diffusers.models.transformers.transformer_minimax_h3'` (t48 env has diffusers 0.38; the test needs the pinned
  commit abc5e9bf from MiniMaxH3.md). No device time used, no drop. Then the host reboot killed the chain.
- Fix: pinned diffusers installed with uv --no-deps --target /var/tmp/fasth3/t409/pydeps on blx01 (30 MB; shared
  t48 env untouched); env409.yaml PYTHONPATH gets it. `pytest --collect-only` on blx01: 4 tests collected.
- chain416a restarted detached from run 1655 (state/runs/1655/chain416a.out). Same next step as before.

## t416 run 1664 (2026-10-12 05:50 PDT)
- chain416a (run 1655) submitted blx01 job 958 at ~12:45Z: same collection error (`No module named
  'diffusers.models.transformers.transformer_minimax_h3'`). Cause: run409a.sh line 11 re-exports PYTHONPATH and
  dropped env409.yaml's pydeps entry. No device compute, no drop.
- Fix: run409a.sh PYTHONPATH adds $T/pydeps (only run409a; run409b keeps diffusers 0.38 so the dense arm matches #408).
  Copied to blx01 via tmp+mv; import check with that PYTHONPATH resolves pydeps/.../transformer_minimax_h3.py.
- chain416a restarted detached from run 1664 (state/runs/1664/chain416a.out).
