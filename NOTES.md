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
