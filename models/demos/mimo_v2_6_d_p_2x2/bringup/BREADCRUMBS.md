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

## C.full_dense.attention.test.1 (test role), 2026-09-28
- Replaced the rendered test with the 1x4 prior's frozen `mimo_v2_6_d_p/tests/bringup/test_c_full_dense_attention.py`
  (same golden, s4096 chunk 1 with a 2048-row KV prefix). Only the docstring changed. The gate is pcc_attention_L00 >= 0.99.
  The test also asserts: output finite, whole-chunk rel L2 <= 0.015, rel L2 on the first 128 rows <= 0.015, and the per-token norm ratio in
  [0.97, 1.03]. Why: RoPE positions from 0, a non-causal mask, a 1/sqrt(128) scale and a missing KV prefix all pass PCC 0.99
  (measurements are in the docstring). It also asserts that the chunk starts after 0.
- BRINGUP_IMPL=reference: PCC 0.999999, rel 0.00170, first 128 rows 0.00172, ratio [0.9992, 1.0008], PASS.
  BRINGUP_IMPL=stub: PCC 0, FAIL.
- Default gate (2x2 device): NotImplementedError from hooks.device_component, which is expected until the implement step.
  The implement step must keep device rel L2 well under 0.015. Gemma device attention measured 0.005-0.008.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p_2x2/tests/bringup/test_c_full_dense_attention.py`

## C.full_dense.attention.implement.1 (implement role), 2026-09-28
- New `tt/attention.py`: `TtFullAttention` + `TtKVCacheFull` copied from the 1x4 prior `mimo_v2_6_d_p/tt/attention.py`.
  The sliding class is not copied yet; it has its own task. Changes for 2x2:
  - The o_proj reduce is `ttnn.all_reduce(cluster_axis=None)`. all_reduce.cpp runs axis 1 then axis 0 on a non-line mesh.
  - `ShardTensorToMesh(dim)` on the 2x2 mesh shards over the flattened mesh in row-major order, so chip d = 2*row + col
    holds TP rank d (Q heads 16d..16d+15, KV head d). The weight and cache code is unchanged from 1x4.
- SDPA precision: the owner rule says every matmul runs at HiFi4, and preset A is HiFi2. I added preset `A4` (A's streaming kernel,
  fp32 dest off, approx exp, q512/k128, at HiFi4) and made it the full-layer default (`FULL_SDPA_DEFAULT`).
  `MIMO_SDPA_CFG=A` selects the prior's HiFi2 preset. The perf step should measure the cost (per known_issues, HiFi4 at q512/k128
  costs about 1.6x SDPA time at 51k on 1x4).
- `tt/model.py`: new `build_attention` (full layers only; sliding raises NotImplementedError) and `new_kv_cache`.
- `hooks.py`: `device_component("attention")` builds a fresh device cache per call and loads the golden prefix, as the prior does.
  `DEVICE_STEPS.full_dense` now contains `attention`. The hybrid state keeps device KV caches for layers whose attention runs on device
  (`ctx.extra["dev_cache"]`).
- Gate: pcc_attention_L00 0.999988, rel_l2 0.008036, first 128 rows 0.007031, row_norm_ratio [1.0018, 1.0115], PASS.
  The first `FAIL pcc=0.000000` line comes from the precompile collect pass.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p_2x2/tests/bringup/test_c_full_dense_attention.py`

## S.full_dense.02.test.1 (test role), 2026-09-28
- Replaced the rendered swap test with the 1x4 prior's frozen `mimo_v2_6_d_p/tests/bringup/test_swap_full_dense_02_attention.py`
  (same golden, s4096 chunk 1, KV prefix 2048). Only the docstring changed. Gate: pcc_swap_out >= 0.98. Extra asserted checks:
  block out finite, rel L2 <= 0.01 (whole chunk and first 128 rows); each swapped step finite, PCC >= 0.99, norm ratio in [0.97, 1.03],
  rel L2 <= 0.03 (attn_norm) / <= 0.015 (attention, also on the first 128 rows).
- BRINGUP_IMPL=reference: out 0.999999 / rel 0.0017, attention rel 0.0017, PASS. BRINGUP_IMPL=stub: every check fails (out PCC 0, rel 1.60).
- Device gate (2x2): pcc_swap_out 0.999995, out rel 0.0036, attn_norm rel 0.0017, attention rel 0.0079 / first rows 0.0070,
  ratio [1.0018, 1.0112], PASS. Attention headroom is ~1.9x (the 1x4 prior had ~3x at rel 0.0052); if a perf change pushes attention rel toward 0.015,
  look at the SDPA preset first. The first `FAIL pcc=0` block comes from the precompile collect pass.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p_2x2/tests/bringup/test_swap_full_dense_02_attention.py`

## C.full_dense.attn_residual.test.1 (test role), 2026-09-28
- Replaced the rendered test with the 1x4 prior's frozen `mimo_v2_6_d_p/tests/bringup/test_c_full_dense_attn_residual.py`
  (same golden, [2048, 4096]). Gate: pcc_attn_residual_L00 >= 0.99. Asserted extras: output size, finite, rel L2 <= 0.01, per-token norm
  ratio in [0.99, 1.01]. Added the prior sliding_moe test's addend check on delta = out - in: coef in [0.95, 1.05], attn rel L2 <= 0.3.
  Why: PCC passes 2x, zeroed rows or columns, and dropped tail rows (measurements are in the docstring).
- BRINGUP_IMPL=reference: PCC ~1.0, rel 0.00207, ratio [0.9991, 1.0007], coef 1.0000, attn rel 0.0000, PASS. BRINGUP_IMPL=stub: PCC 0, FAIL.
- Default gate (2x2): NotImplementedError from hooks.device_component, which is expected until the implement step. The add must stay bf16 or better
  (the bf16 add measured rel 0.0023 on 1x4). `ttnn.bringup.rms_norm(return_residual_sum=...)` can fuse this add into ffn_norm later.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p_2x2/tests/bringup/test_c_full_dense_attn_residual.py`

## C.full_dense.attn_residual.implement.1 (implement role), 2026-09-28
- `tt/residual.py`: `TtResidualAdd`, copied from the 1x4 prior unchanged: replicated bf16 `ttnn.add` into DRAM, with no collective, because
  attention's all_reduce over both axes leaves attn_out replicated on every chip. It is not yet fused into ffn_norm; `ttnn.bringup.rms_norm(return_residual_sum=...)`
  is still available for the assemble/perf step.
- `hooks.py`: added `_RESIDUAL_STEPS` (attn/mlp/ffn residual) and `_residual_host_fn`. `device_component` serves every residual step. The hybrid
  model adds residual overrides for the steps listed in DEVICE_STEPS. `DEVICE_STEPS.full_dense` now contains `attn_residual`.
- Gate: pcc_attn_residual_L00 0.999997, rel_l2 0.002393, row_norm_ratio [0.9996, 1.0012], coef 1.0005, attn rel 0.0024, PASS.
  The first `FAIL pcc=0` line comes from the precompile collect pass.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p_2x2/tests/bringup/test_c_full_dense_attn_residual.py`

## S.full_dense.03.test.1 (test role), 2026-09-28
- Replaced the rendered swap test with the 1x4 prior's frozen `mimo_v2_6_d_p/tests/bringup/test_swap_full_dense_03_attn_residual.py`
  (same golden, s4096 chunk 1). Added one check, taken from this run's component test: the addend on delta = h_mid - golden in against
  the attn_out the residual received, coef in [0.95, 1.05] and rel <= 0.3 (metrics attn_coef_swap_h_mid, attn_rel_l2_swap_h_mid).
  The other limits are unchanged from the prior: block out rel <= 0.01; per step PCC >= 0.99; rel 0.03 / 0.015 / 0.01; ratio [0.97, 1.03] for
  attn_norm and attention, [0.99, 1.01] for attn_residual.
- BRINGUP_IMPL=reference: out 0.999999 / rel 0.0017, coef 1.0000, PASS. BRINGUP_IMPL=stub: every check fails.
- Device gate (2x2): pcc_swap_out 0.999994, out rel 0.0039, attention rel 0.0079 ratio [1.0018, 1.0112], h_mid rel 0.0064 ratio
  [1.0014, 1.0084], coef 1.0005, PASS. h_mid headroom is small (ratio max 1.0084 vs 1.01) because the SDPA preset biases attention norms
  upward. A perf change to SDPA must re-run this test.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p_2x2/tests/bringup/test_swap_full_dense_03_attn_residual.py`

## C.full_dense.ffn_norm.test.1 (test role), 2026-09-28
- Replaced the rendered test with the 1x4 prior's frozen `mimo_v2_6_d_p/tests/bringup/test_c_full_dense_ffn_norm.py`
  (same golden, h_mid [2048, 4096] -> ffn_norm). Only the docstring changed. Gate: pcc_ffn_norm_L00 >= 0.99. Asserted extras: output finite,
  rel L2 <= 0.03, per-token norm ratio in [0.97, 1.03]. Why: PCC passes sum-instead-of-mean and eps 1e-2 (the docstring has the measurements).
- BRINGUP_IMPL=reference: PCC 0.999997, rel 0.00233, ratio [0.9953, 1.0045], PASS. BRINGUP_IMPL=stub: PCC 0, FAIL.
- Default gate (2x2): the device module already serves ffn_norm (the norm path attn_norm registered). PCC 0.999996, rel 0.00290,
  ratio [0.9933, 1.0054], PASS. The first `FAIL pcc=0` line comes from the precompile collect pass.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p_2x2/tests/bringup/test_c_full_dense_ffn_norm.py`

## S.full_dense.04.test.1 (test role), 2026-09-28
- Replaced the rendered swap test with this run's reviewed swap 03 (the prior's swap 04 body plus the attn_residual addend check) and added
  ffn_norm: rel L2 <= 0.03, per-token norm ratio [0.97, 1.03] (the ffn_norm component test's limits). The other limits are unchanged:
  block out rel <= 0.01; per step PCC >= 0.99; attention 0.015, attn_residual 0.01 / [0.99, 1.01]; addend coef [0.95, 1.05], rel <= 0.3.
- BRINGUP_IMPL=reference: out 0.999999 / rel 0.0017, PASS. BRINGUP_IMPL=stub: every check fails.
- Device gate (2x2): pcc_swap_out 0.999994, out rel 0.0039, h_mid rel 0.0064 ratio [1.0014, 1.0084], ffn_norm rel 0.0051 ratio
  [0.9942, 1.0106], coef 1.0005, PASS. The first `FAIL pcc=0` block comes from the precompile collect pass.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p_2x2/tests/bringup/test_swap_full_dense_04_ffn_norm.py`

## C.full_dense.mlp.test.1 (test role), 2026-09-28
- Replaced the rendered test with the 1x4 prior's frozen `mimo_v2_6_d_p/tests/bringup/test_c_full_dense_mlp.py` (same golden, ffn_norm
  [2048, 4096] -> mlp_out). Only the docstring changed. Gate: pcc_mlp_L00 >= 0.99. Asserted extras: output finite, rel L2 <= 0.015, per-token
  norm ratio in [0.98, 1.02], worst per-token rel L2 <= 0.05. Why: PCC passes a 2x output, an all_reduce counted 4x, zeroed rows and a missing
  TP shard (the docstring lists the prior's measurements). HiFi2-like weight truncation gives rel 0.0141, so the 0.015 limit leaves little room
  above HiFi2. HiFi4 (owner rule) is expected near 0.003-0.006.
- BRINGUP_IMPL=reference: PCC 0.999998, rel 0.001745, ratio [0.9986, 1.0015], worst row 0.0024, PASS. BRINGUP_IMPL=stub: PCC 0, FAIL.
- Default gate (2x2) fails with NotImplementedError ("no device module for mlp yet"). The implement step adds that module.
- On the 2x2 mesh the down projection's reduction has to cover all 4 chips (both axes). A reduction over one axis only leaves out half the TP
  shards (the prior measured PCC 0.9958 / rel 0.26 with one shard missing), and the rel check catches that.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p_2x2/tests/bringup/test_c_full_dense_mlp.py`

## C.full_dense.mlp.implement.1 (implement role), 2026-09-28
- `tt/mlp.py`: `TtDenseMLP`, copied from the 1x4 prior. The only code change is the down reduce, `ttnn.all_reduce(cluster_axis=None)`
  (axis 1 then axis 0, all 4 TP ranks). Sharding needed no change: `ShardTensorToMesh` over the row-major 2x2 order gives chip d = 2*row + col
  intermediate columns [4096d, 4096d+4096) for gate/up (dim -1) and the matching down rows (dim -2). Weights are fp8 + 128x128 block scale,
  dequantized to bf16 at load. Every matmul runs HiFi4 + fp32 acc, and silu is fused into the gate linear.
- `tt/model.py`: added `build_mlp` (the same loader path as the prior: `reference.weights.fp8_weight`). `hooks.py`: added `_mlp_module`.
  `device_component` serves `mlp`, the hybrid model adds an `mlp` override, and `DEVICE_STEPS.full_dense` now contains `mlp`.
- Gate: pcc_mlp_L00 0.999995, rel_l2 0.004004, row_norm_ratio [1.0007, 1.0050], worst row 0.0056, PASS. Every row's norm ratio is slightly
  above 1, a small upward bias well inside [0.98, 1.02]. The first `FAIL pcc=0` line comes from the precompile collect pass.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p_2x2/tests/bringup/test_c_full_dense_mlp.py`

## S.full_dense.05.test.1 (test role), 2026-09-28
- Replaced the rendered swap test with this run's reviewed swap 04 (attn_residual addend check included) and added mlp as the 1x4
  prior's frozen swap 05 does: mlp vs golden rel L2 <= 0.015, per-token norm ratio [0.98, 1.02]; mlp vs the CPU mlp on the same device
  ffn_norm output, rel <= 0.015, ratio [0.98, 1.02], worst row <= 0.05 (catches a one-axis down reduce on 2x2, which block out misses).
- BRINGUP_IMPL=reference: out 0.999999 / rel 0.0017, PASS. BRINGUP_IMPL=stub: FAIL.
- Device gate (2x2): pcc_swap_out 0.999994, out rel 0.0048, mlp rel 0.0037 ratio [0.9968, 1.0076], mlp vs CPU rel 0.0034 worst row
  0.0048, PASS.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p_2x2/tests/bringup/test_swap_full_dense_05_mlp.py`

## C.full_dense.mlp_residual.test.1 (test role), 2026-09-28
- Replaced the rendered test with the 1x4 prior's frozen `mimo_v2_6_d_p/tests/bringup/test_c_full_dense_mlp_residual.py` (same golden,
  out = h_mid + mlp_out). Its checks: PCC >= 0.99, output size, finite, rel L2 <= 0.01, per-token norm ratio [0.99, 1.01].
  Added the addend check from this run's attn_residual test: on delta = out - h_mid, coefficient <delta, mlp_out>/||mlp_out||^2 must be
  in [0.95, 1.05] and ||delta - mlp_out||/||mlp_out|| <= 0.3. The docstring records the prior's mutation measurements (2x, zeroed rows,
  h_mid + 0.5 mlp_out 0.987, and others).
- BRINGUP_IMPL=reference: PCC 0.999998, rel 0.0021, ratio [0.9991, 1.0009], coef 1.0000, PASS. BRINGUP_IMPL=stub: PCC 0, FAIL.
- The default gate (2x2) already passes because the device residual module exists: PCC 0.999997, rel 0.0028, ratio [1.0003, 1.0023], coef 1.0020,
  mlp term rel 0.0038. The first `FAIL pcc=0` line comes from the precompile collect pass.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p_2x2/tests/bringup/test_c_full_dense_mlp_residual.py`

## S.full_dense.06.test.1 (test role), 2026-09-28
- Replaced the rendered swap test with this run's reviewed swap 05 plus mlp_residual, following the 1x4 prior's frozen swap 06:
  mlp_residual vs golden rel <= 0.01, ratio [0.99, 1.01], first 128 rows; vs CPU on the same device h_mid/mlp_out rel <= 0.01, ratio
  [0.99, 1.01], worst row <= 0.02. The addend check now runs for both residuals: attn_out in h_mid - in and mlp_out in out - h_mid,
  coef [0.95, 1.05], rel <= 0.3. Metric names changed from attn_coef_swap_h_mid to addend_coef_swap_{h_mid,out}. These are informational
  only.
- BRINGUP_IMPL=reference: out 0.999999 / rel 0.0017, PASS. BRINGUP_IMPL=stub: FAIL.
- Device gate (2x2): pcc_swap_out 0.999992, out rel 0.0059, ratio [1.0017, 1.0085] (limit 1.01), mlp_residual vs CPU rel 0.0019,
  mlp_out coef 1.0020, PASS. The device error accumulates as a small upward norm bias: h_mid max 1.0084, out max 1.0085.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p_2x2/tests/bringup/test_swap_full_dense_06_mlp_residual.py`

## C.sliding_moe.attn_norm.test.1 (test role), 2026-09-28
- Replaced the rendered test with the 1x4 prior's frozen `mimo_v2_6_d_p/tests/bringup/test_c_sliding_moe_attn_norm.py` (same golden,
  layer 1). Checks: PCC >= 0.99, finite, rel L2 <= 0.03, per-token norm ratio [0.97, 1.03]; the docstring keeps the prior's mutation
  measurements (sum-for-mean, eps, `1 + w`, zeroed rows).
- BRINGUP_IMPL=reference: PCC 0.999997, rel 0.0024, ratio [0.9954, 1.0043], PASS. BRINGUP_IMPL=stub: PCC 0, FAIL.
- The default gate (2x2) already passes, because the device norm module covers every norm step: PCC 0.999996, rel 0.0029, ratio [0.9935, 1.0061].
  The first `FAIL pcc=0` line comes from the precompile collect pass.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p_2x2/tests/bringup/test_c_sliding_moe_attn_norm.py`

## S.sliding_moe.01.test.1 (test role), 2026-09-28
- Replaced the rendered swap test with the 1x4 prior's frozen `mimo_v2_6_d_p/tests/bringup/test_swap_sliding_moe_01_attn_norm.py` (same golden,
  layer 1); only the docstring differs. Checks: pcc_swap_out >= 0.98 (gated); block out finite, rel L2 <= 0.01; the attn_norm output vs golden:
  PCC >= component threshold, rel L2 <= 0.03, per-token norm ratio [0.97, 1.03].
- BRINGUP_IMPL=reference: out 0.999996 / rel 0.0030, step rel 0.0024 / ratio [0.9954, 1.0043], PASS. BRINGUP_IMPL=stub: out rel 16.9, FAIL.
- Default gate (2x2) already passes because the device norm module covers every norm step: out 0.999995 / rel 0.0031, step rel 0.0029 / ratio
  [0.9935, 1.0061]. The first `FAIL pcc=0` lines come from the precompile collect pass.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p_2x2/tests/bringup/test_swap_sliding_moe_01_attn_norm.py`

## C.sliding_moe.attention.test.1 (test role), 2026-09-28
- Replaced the rendered test with the 1x4 prior's frozen `mimo_v2_6_d_p/tests/bringup/test_c_sliding_moe_attention.py` (same golden,
  s4096 chunk 1, layer 1). Only the docstring changed. The gate is pcc_attention_L01 >= 0.99. The test also asserts: output finite, rel L2 <= 0.02 over
  the whole chunk and over the first 128 rows, per-token norm ratio in [0.95, 1.05], and worst per-token rel L2 <= 0.08. It also asserts that the output
  is closer to the CPU reference at window 128 than at 127 or 129 (off-by-one windows fall inside the device noise for every size check).
  The mutation measurements are in the docstring.
- BRINGUP_IMPL=reference: PCC 0.999989, rel 0.0047, first 128 rows 0.0042, ratio [0.9825, 1.0147], worst row 0.0176,
  vs window 127 0.0085 and vs window 129 0.0093, PASS. BRINGUP_IMPL=stub: PCC 0, FAIL.
- Default gate (2x2): NotImplementedError from `tt/model.py` ("sliding attention not ported to 2x2 yet"). This is expected until the implement step.
  Implement note: follow the prior's `mimo_v2_6_d_p/tt/attention.py:TtSlidingAttention`. It uses a power-of-two SDPA scale with the true scale
  folded into Q, a sink pre-divided by the same power, HiFi4 (see known issues), and a norm ratio limit of 1.05. P.2 on 1x4 reached 1.046.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p_2x2/tests/bringup/test_c_sliding_moe_attention.py`

## C.sliding_moe.attention implement (attempt 1)
- Ported `TtKVCacheSliding` + `TtSlidingAttention` from the prior's `tt/attention.py` into `tt/attention.py` (appended; `import math` restored). Only change: o_proj reduce `ttnn.all_reduce(cluster_axis=None)` (axis 1 then axis 0). Mappers unchanged (`ShardTensorToMesh` dim over the 2x2 mesh, row-major: chip d = 2*row + col holds Q heads 16d..16d+15, KV heads 2d, 2d+1, sink 16d..16d+15).
- Carried over: SDPA scale 2^-4 with true_scale/2^-4 folded into the Q rows, sink pre-divided by 2^-4 (bf16-exact), window tail = last 128 cache rows via `ttnn.slice` + q_pad concat, V at 128 through `ttnn.bringup.scaled_dot_product_attention`, preset S (HiFi4, fp32 dest off, exact exp).
- `tt/model.py`: `build_attention` builds TtSlidingAttention for sliding layers (swa_rope_theta, sink bias); `new_kv_cache` returns TtKVCacheSliding for them. `hooks.py`: `DEVICE_STEPS["sliding_moe"] = {"attention"}`.
- Gate: pcc 0.999912, rel 0.0134, first-128 rel 0.0116, norm ratio [0.9635, 1.0457], worst row 0.046; closer to window 128 (0.0128) than 127 (0.0161) / 129 (0.0146). Same numbers as the 1x4 prior's preset S.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p_2x2/tests/bringup/test_c_sliding_moe_attention.py`
