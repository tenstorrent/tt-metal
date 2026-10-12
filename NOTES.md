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
