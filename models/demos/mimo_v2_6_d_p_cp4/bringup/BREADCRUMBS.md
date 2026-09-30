# XiaomiMiMo/MiMo-V2.6-Flash-RL bring-up on mesh 1x4: breadcrumbs

Prior bring-up: mimo_v2_6_d_p (mesh 1x4); goldens and CPU reference shared. Append-only log, one section per task attempt: what was done, decisions and why, gotchas, the re-run command, the verdict.

## PL.1 plan (attempt 1, 2026-09-30)

- Wrote `plan.yaml`, `plan.md` and `components.yaml` for the owner's CP=4 scheme on 1x4. The residual and the KV cache
  are split by sequence (chip c holds slice c of each chunk, in a chunk-major ring cache). Attention is TP=1 (whole
  qkv/o_proj, all heads, no all_reduce) through `ttnn.transformer.ring_joint_scaled_dot_product_attention`: causal
  ring for the full layers, window 128 + sink + compact halo for the sliding layers. The dense MLP and the router run
  locally. The experts are EP=4 with one fabric dispatch group of 4 on axis 1 (`ttnn.bringup.dispatch/combine/offset_cumsum`,
  then `post_combine_reduce`, no reduce after). The LM head is vocab-sharded after a [32, 4096] last-row all_gather.
- The gate prints 15.95 GiB per chip of the 27.20 budget. plan_fits 1, 0 unplaced, 0 plan / component / ledger errors.
  plan_approved 0 until a person approves (`approvals.yaml`).
- Forced by the ring op's validation (`ring_joint_sdpa_device_operation.cpp`): V is padded 128 -> 192 (tensor-V mode
  needs VDH == DH), the sliding K/V cache is bfp8 (the sliding path requires BFP8_B), and the sink needs streaming
  compute (fp32 dest off, as preset S). Full-layer caches stay bf16.
- Risks for the implement steps: MiMo's 64Q:8KV at D192 on the ring sliding path is unexercised (documented for the
  GPT-OSS 8Q:1KV D64 shape), so probe it at s4096 first. A bfp8 sliding cache may cost sink-layer accuracy. Both have
  a fallback: a `ttnn.bringup` ring_joint fork behind a default-off option. Axis-1 dispatch with DGS 4 on the forks is
  new in this repo.
- Gotcha: this checkpoint map lists `model.mtp.*` (48 tensors), so the plan skips them explicitly. In the plan.yaml
  flow mappings, a `group:` with a comma must be quoted, or YAML splits it into another key.
- tasks.yaml is unchanged (same block graphs and components).
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup.plan.check_plan`

## C.full_dense.attn_norm.test.1 (test review)

- Started from the prior's frozen `mimo_v2_6_d_p/tests/bringup/test_c_full_dense_attn_norm.py` (same golden: s4096 chunk 1, [2048, 4096]). It keeps the PCC gate and adds finite output, rel L2 <= 0.03 and a per-token norm ratio in [0.97, 1.03] (PCC is scale-invariant: sum instead of mean and `1 + w` get past it).
- Added: worst-row rel L2 vs golden <= 0.01 (reference 0.0029). This catches a wrong CP slice or a few zeroed rows.
- Added an eps check (known issue "A wrong RMSNorm eps is invisible..."). The golden's smallest row mean square is 3.95e-5 = 40x eps 1e-6, so eps 0 and 2e-6 score rel 0.006 on the golden and pass. The test also runs the module on input x 0.1 against the CPU step on that same input: rel <= 0.02 and worst row <= 0.04. bf16 in/out scores 0.0022 / 0.0044; eps 0, 1e-7, 2e-6 and 1e-5 score rel 0.40, 0.33, 0.17 and 0.54. The module must accept any [S, 4096] float input and be callable twice.
- Verified: BRINGUP_IMPL=reference passes (pcc 0.999999, rel 0.0016, ratio [0.9980, 1.0023]); BRINGUP_IMPL=stub fails (pcc 0). The device gate currently fails with NotImplementedError (no device module yet; the implement step adds it).
- The FAIL line printed before the real pass comes from the wrapper's precompile collect pass (its output is discarded). Ignore it.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p_cp4/tests/bringup/test_c_full_dense_attn_norm.py`

## C.full_dense.attn_norm.implement.1 (implement)

- `tt/rms_norm.py`: copied the prior's `TtRMSNorm` unchanged (`ttnn.bringup.rms_norm`, HiFi4, fp32 dest acc, plain
  `w`, eps 1e-6, replicated bf16 weight; `fused_add` kept for the residual fusion; `MIMO_NORM_IMPL=native` still
  selects `ttnn.rms_norm`). Rows are independent, so the CP=4 version needs no CCL. The only change is in the harness
  helpers: `to_device_cp` shards host [S, H] on dim 2 (`ShardTensorToMesh`, chip c gets rows [c*S/4, (c+1)*S/4), S
  must split into tile-aligned slices), and `cp_to_host` concatenates in chip order (`ConcatMeshToTensor` dim 2).
- `hooks.py`: `_NORM_WEIGHTS`, `_norm_module` (loads only that layer's weight through the prior's
  `reference.weights.WeightLoader`; nothing is imported from the prior's `tt/`), `_cp_host_fn`, and
  `device_component` for `attn_norm` and `ffn_norm` (the same module; ffn_norm has not been gated yet).
  `device_model` is a hybrid (`HybridDeviceModel`, CPU reference + `DEVICE_STEPS` through `run_block`, host in and
  host out per step). `DEVICE_STEPS` lists only `full_dense: attn_norm` so far. The assemble step replaces it with the
  all-device model.
- Gate: pcc 0.999999, rel L2 0.0011, row ratio [0.9966, 1.0029], worst row 0.0047. On input x0.1: rel 0.0023,
  worst row 0.0048.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p_cp4/tests/bringup/test_c_full_dense_attn_norm.py`
  (the `FAIL pcc=0` line before the real pass comes from the precompile collect pass, so ignore it).
