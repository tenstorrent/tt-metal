# XiaomiMiMo/MiMo-V2.6-Flash-RL bring-up on mesh 2x2: breadcrumbs

Prior bring-up: mimo_v2_6_d_p (mesh 1x4); goldens and CPU reference shared. Append-only log, one section per task attempt: what was done, decisions and why, gotchas, the re-run command, the verdict.

## PL.1 plan (attempt 1), 2026-09-28

- Wrote `plan.yaml`, `plan.md`, `components.yaml`; re-planned from the 1x4 prior's plan and its final modules
  (`models/demos/mimo_v2_6_d_p/tt`, which use the bring-up forks: rms_norm, sdpa with V 128, unified_routed_expert_ffn
  high_precision, dispatch/combine/offset_cumsum). tasks.yaml unchanged.
- Attention and dense MLP: TP=4 over the flattened mesh, chip d = 2*row + col holds checkpoint TP rank d (ShardTensorToMesh).
  The reduce is `ttnn.all_reduce(cluster_axis=None)`: on a 2x2 mesh it runs axis 1 then axis 0 (all_reduce.cpp). Rejected
  TP=2 x SP=2: 4 KV heads / 4 TP ranks map 1:1, SP on causal chunked prefill needs KV gather or ring SDPA.
- Experts: the prior's local 1-chip dispatch needed a size-1 mesh axis; 2x2 has none. Chose the DeepSeek 2D layout:
  dispatch axis 0 (DGS 2, fabric), one dispatch group per column (128 experts), chip (r, c) holds experts 128c + 64r .. +63
  (ExpertMapping col-major). `ttnn.mesh_partition` of x/idx/wts by row at the MoE entry (S/2 per row), after combine
  all_reduce axis 1 + all_gather axis 0 -> replicated [S, 4096]; component boundaries identical to 1x4. The forks
  behave as the source ops on a 2-device axis, so no op change planned. DeepSeek's conftest runs (2, 2) with FABRIC_2D.
  Fallback (in plan.md): a default-off local-dispatch option in the forks.
- V planned at 128 (sdpa fork), so the KV state is 0.06 GiB smaller than the prior's plan. Per chip 14.59 of 27.20 GiB.
- Gate (no device): plan_fits 1, 0 unplaced, 0 plan/component/ledger errors; plan approved False until a person approves.
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup.plan.check_plan`

## C.full_dense.attn_norm.test.1 (test role)
- Replaced the rendered test with the prior's frozen 1x4 test (`mimo_v2_6_d_p/tests/bringup/test_c_full_dense_attn_norm.py`), which uses the same golden. Only the docstring changed. It gates on PCC >= 0.99, and also asserts that the output is finite, rel L2 <= 0.03 and the per-token norm ratio is in [0.97, 1.03]. PCC alone misses sum-instead-of-mean, a wrong eps, and `1 + w` (see known_issues "PCC alone does not gate a component").
- BRINGUP_IMPL=reference: PCC 0.999999, rel_l2 0.001597, ratio [0.9980, 1.0023], passes. BRINGUP_IMPL=stub: PCC 0, fails.
- Default gate (the device mesh 2x2) currently fails with NotImplementedError from hooks.device_component. That is expected until the implement step.
- The `FAIL pcc=0.000000` line printed before the real result comes from the precompile collect pass (stubbed device results). Ignore it.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p_2x2/tests/bringup/test_c_full_dense_attn_norm.py`

## C.full_dense.attn_norm.implement.1 (implement role), 2026-09-28
- New `tt/`: `rms_norm.py` copied from the 1x4 prior's `tt/rms_norm.py`. Only the docstring changed: replication through
  `ReplicateTensorToMesh` works on a 2x2 mesh, and the norm needs no CCL. `TtRMSNorm` runs `ttnn.bringup.rms_norm`
  at HiFi4 with fp32 acc, using the plain w. `MIMO_NORM_IMPL=native` selects `ttnn.rms_norm`. `fused_add`
  (return_residual_sum) is kept for the later layers.
  `model.py` holds only `NORM_WEIGHTS` and `build_norm` so far. The hooks and the future all-device model will share these builders.
- `hooks.py`: `device_component` handles the norm steps (attn_norm and ffn_norm) through `_host_fn`, which does host->replicated
  device->chip-0 read-back at the harness boundary. `device_model` returns a `HybridDeviceModel`: the CPU reference
  with `DEVICE_STEPS` (currently only `full_dense: {attn_norm}`) swapped in. It stays that way until the assemble step.
  Weights load through the prior's `reference/weights.py:WeightLoader`. The CPU side is shared, and nothing is imported from the prior's `tt/`.
- Gate: pcc_attn_norm_L00 0.999999, rel_l2 0.001738, row_norm_ratio [0.9956, 1.0027], PASS. The first `FAIL pcc=0.000000`
  line comes from the precompile collect pass.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p_2x2/tests/bringup/test_c_full_dense_attn_norm.py`

## S.full_dense.01.test.1 (test role), 2026-09-28
- Replaced the rendered swap test with the 1x4 prior's frozen `mimo_v2_6_d_p/tests/bringup/test_swap_full_dense_01_attn_norm.py`
  (same golden); only the docstring changed. Gate stays pcc_swap_out >= 0.98. Extra asserted checks: block out finite and rel L2 <= 0.01;
  the swapped attn_norm output is finite, has PCC >= 0.99 (the component threshold) and rel L2 <= 0.03, and its per-token norm ratio is in [0.97, 1.03].
  Why: a zero stub alone reaches PCC 0.988 on out and passes the 0.98 gate, because the residual dominates out at layer 0.
- BRINGUP_IMPL=reference: out PCC 0.999999, rel L2 0.00168, step rel L2 0.0016, ratio [0.998, 1.002], PASS.
  BRINGUP_IMPL=stub: out rel L2 0.155 and step PCC 0, which fails the extra checks.
- Device gate (2x2, attn_norm module already implemented): out PCC 0.999999, rel L2 0.00169, step rel L2 0.00174,
  ratio [0.9956, 1.0027], PASS. The first block of pcc=0 lines comes from the precompile collect pass.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p_2x2/tests/bringup/test_swap_full_dense_01_attn_norm.py`
