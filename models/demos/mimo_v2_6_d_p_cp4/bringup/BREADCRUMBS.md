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

## C.full_dense.attention.test.1 (test review)

- Started from the prior's frozen `mimo_v2_6_d_p/tests/bringup/test_c_full_dense_attention.py` (same golden: s4096 chunk 1, start 2048, [2048, 4096] attn_norm -> attn_out). Kept its checks: PCC >= 0.99 (gated), finite output, whole-chunk rel L2 <= 0.015, first 128 rows <= 0.015, and a per-token norm ratio in [0.97, 1.03].
- Added: rel L2 per CP slice (4 x 512 rows) <= 0.012, recorded as `rel_l2_cp_slice{r}_attention_L00`. CPU measurements against the golden (script `/tmp/cp4m/measure.py`, not kept), as PCC / whole rel / worst slice rel: reference 1.0 / 0.0017 / 0.0017; ring of one hop only 0.99992 / 0.0129 / 0.0216 (the prior checks all miss it); RoPE positions restarting per slice 0.99985 / 0.0165 / 0.0256; no ring 0.99968 / 0.0254 / 0.0374; non-causal across slices 0.99967 / 0.0258 / 0.0392; causal only within a slice 0.99976 / 0.0217 / 0.0251.
- Verified: BRINGUP_IMPL=reference passes (pcc 1.0, rel 0.0017, slices 0.0017 each, ratio [0.9992, 1.0008]). BRINGUP_IMPL=stub fails (pcc 0). The device gate currently fails with NotImplementedError (no device module yet).
- The device module must return the full chunk's attn_out in chunk row order (slice r = rows r*512..), in the same shape as the golden.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p_cp4/tests/bringup/test_c_full_dense_attention.py`

## C.full_dense.attention.implement.1 (implement)

- `tt/ccl.py`: `RingCCL`, from `gpt_oss_d_p/tt/ccl.py`. It holds the 3 ring semaphores, puts the CCL workers in the
  last compute column (offset (grid.x-1, 0)) and gives the SDPA grid (grid.x-1, grid.y). The persistent gather buffers
  [1, nkv, seq, W] are replicated, zeroed once, and keyed by shape. Linear topology, 1 link, cp_axis 1.
- `tt/attention.py`: `TtFullAttention` (CP=4, TP=1) + `TtKVCacheRing`, copied from the prior's TtFullAttention.
  Every chip has the whole weights. Q is [4096, 64*192]. KV is [4096, 4*(192+192)], per head [k_h | v_h*0.707
  zero-padded to 192]. The o_proj is unpadded [8192, 4096]. Two HiFi4 matmuls feed nlp_create_q_heads_split (the Q
  split gives the RoPE split). Partial RoPE uses `rotary_embedding` + concat. `update_padded_kv_cache`
  (kv_actual_global=start, cluster_axis=1) writes K and V. Then `ring_joint_scaled_dot_product_attention` runs
  (causal, chunked, kv_cache_batch_idx 0, logical_n start+S, q64/k256, HiFi4 + fp32 dest, exact exp, scale fp32(192^-0.5)).
  The output is sliced to V 128, then nlp_concat_heads, then o_proj. No CCL after.
- The RoPE tables are chunk-major per chip, built once at load for every chunk size in the spec (ladder + target:
  2048, 5120, 8192) up to max seq 56320. Per chunk they are sliced on device at local row start/4, the same on every
  chip. A chunk size that is not in the spec asserts.
- The ring cache is per chip [1, 4, max_seq/4, 192] bf16, interleaved DRAM. Its layout depends on the chunk size
  C (local row n*L+j on chip r = global n*C + r*L + j), so the cache is bound to one chunk (`bind_chunk`). The harness
  host fn binds it to ctx.length. A prefix loaded before that is kept on the host and written at bind time (harness
  boundary, not the module forward). `to_torch` inverts the layout and returns V[..., :128].
- Gotchas (proposed in known_issues): passing `kv_actual_isl` needs fp32 dest off (streaming kernel), so it is
  omitted. Starts are chunk-aligned, so the plain chunked path is exact. `nlp_concat_heads` on 64x192 overflows L1,
  hence the slice to 128 before the concat (instead of the prior's zero-row o_proj).
- hooks: `_attention_module`, `_attention_host_fn`, `device_component("attention")` (a fresh ring cache per call,
  holding the golden prefix, built for the golden chunk). `DEVICE_STEPS["full_dense"]` now has `attention`.
  `_HybridState` keeps device ring caches for the device-attention layers, and the hybrid shares one RingCCL.
- Gate: pcc 0.999997, rel L2 0.0028, first 128 rows 0.0031, ratio [1.0000, 1.0031], per CP slice
  [0.0028, 0.0029, 0.0028, 0.0027].
- Ad-hoc probe (not kept): chunk 0 and chunk 1 of s4096 run on one cache scored PCC 0.999998 / 0.999997
  (rel 0.0021 / 0.0027). The state read-back vs golden was rel 0.0029 for K and V over [0, 4096).
- Not done: the narrow-V ring path. Validation relaxes VDH == DH for causal/chunked calls, but no one has tried it
  in the kernel. Sliding layers still raise NotImplementedError.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p_cp4/tests/bringup/test_c_full_dense_attention.py`

## C.full_dense.attn_residual.test.1 (test review)
- Ported the prior's frozen test (`mimo_v2_6_d_p/tests/bringup/test_c_full_dense_attn_residual.py`, same golden:
  s4096 chunk 1, [2048, 4096]). Kept COMPARE/THRESHOLD None (PCC >= 0.99), plus asserted: output size, finite,
  rel L2 <= 0.01, per-token norm ratio in [0.99, 1.01]. New for CP=4: rel L2 <= 0.01 per CP slice (rows
  [r*512, (r+1)*512)), to catch slice-order or gather bugs. Metric `rel_l2_slice_max_attn_residual_L00` recorded.
- Measured: reference PCC 0.999998, rel 0.00207, ratio [0.9991, 1.0007], slices [0.00207, 0.00206, 0.00208,
  0.00207]. bf16 control passes (rel 0.00234). Stub fails (PCC). mutate:scale1.02 fails (rel 0.0201). rowshift and
  quarterzero fail (PCC).
- Device mode currently fails with NotImplementedError (no device module yet). The implement step supplies it.
- The "FAIL pcc=0.000000" line printed during collection is the precompile collect pass (known issue). Ignore it.
- Re-run: `PYTHONPATH=$PWD BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p_cp4/tests/bringup/test_c_full_dense_attn_residual.py` (and `BRINGUP_IMPL=stub`).

## C.full_dense.attn_residual implement (attempt 1)
- Copied `mimo_v2_6_d_p/tt/residual.py` to `tt/residual.py` (TtResidualAdd: `ttnn.add`, bf16 out, DRAM). It works unchanged on CP slices: both operands hold the same rows on each chip, so no CCL is needed.
- hooks.py: added `_RESIDUAL_STEPS` (attn/mlp/ffn residual) and `_residual_host_fn`, which does a CP split in via `to_device_cp` and a concat out via `cp_to_host`. It is registered in `device_component`, and the hybrid model builds residual overrides for any residual step in DEVICE_STEPS. `attn_residual` was added to DEVICE_STEPS["full_dense"].
- Gate: pcc 0.999997, rel_l2 0.0024, per-slice rel ~0.0024. The "FAIL pcc=0" line in the log comes from the precompile collect pass and is not the real result.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p_cp4/tests/bringup/test_c_full_dense_attn_residual.py`
