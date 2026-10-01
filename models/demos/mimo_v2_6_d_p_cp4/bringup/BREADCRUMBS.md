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

## C.full_dense.ffn_norm.test.1 (test review)
- Replaced the rendered one-liner with the cp4 attn_norm test's checks (same golden as the prior: s4096 chunk 1, h_mid [2048, 4096] -> ffn_norm). PCC >= 0.99 (gated), finite, rel L2 <= 0.03, per-token norm ratio [0.97, 1.03], worst row rel <= 0.015 (the golden is bf16, so the fp32 reference already scores 0.0050, which is why the limit is looser than attn_norm's 0.01), rel L2 per CP slice <= 0.01, and the eps check (module on input x0.1 vs the CPU step: rel <= 0.02, worst row <= 0.04).
- CPU measurements (script /tmp/cp4ffn/m.py, not kept): the golden's minimum row mean square is 1.8e-4 (180x eps 1e-6). Eps 0 / 1e-7 / 2e-6 score rel 0.003 on the golden (they pass). On the x0.1 input they score rel 0.18 / 0.16 / 0.11. Sum instead of mean gives rel 0.98; `1 + w` gives rel 7.4; CP slices 1/2 swapped give PCC 0.988 and rel 0.15; the last 32 rows zeroed give worst row 1.0; bf16 math gives rel 0.0028 / worst row 0.0062.
- Verified: reference passes (pcc 0.999997, rel 0.0023, slices 0.0023 each); stub fails (pcc 0). The device gate already passes with the existing TtRMSNorm registration: pcc 0.999996, rel 0.0028, ratio [0.9941, 1.0057], worst row 0.0062, slices 0.0027-0.0030, x0.1 rel 0.0024 / worst row 0.0056.
- The first `FAIL pcc=0` line is the precompile collect pass. Ignore it.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p_cp4/tests/bringup/test_c_full_dense_ffn_norm.py`

## C.full_dense.mlp.test.1 (test review)
- Ported the prior's frozen test (`mimo_v2_6_d_p/tests/bringup/test_c_full_dense_mlp.py`, same golden: s4096 chunk 1, ffn_norm [2048, 4096] -> mlp_out). It gates on PCC >= 0.99 and also asserts: finite output, output size, rel L2 <= 0.015, per-token norm ratio in [0.98, 1.02], and worst per-token rel L2 <= 0.05 (the prior's mutation table is in the docstring). New for CP=4: rel L2 <= 0.015 per CP slice (rows [r*512, (r+1)*512)). Metric: `rel_l2_slice_max_mlp_L00`.
- The reference MiMo dense MLP has no clamp (grep of `mimo_v2_6_d_p/reference`), so the Hy4 x30 clamp rerun is not needed. gelu_tanh already fails PCC here (0.911).
- Measured: reference PCC 0.999998, rel 0.00175, ratio [0.9986, 1.0015], worst row 0.0024, slices 0.00174-0.00175. Stub fails (PCC 0).
- Device mode fails with NotImplementedError (no device module yet). The implement step supplies it. The "FAIL pcc=0" line comes from the precompile collect pass. Ignore it.
- Re-run: `PYTHONPATH=$PWD BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p_cp4/tests/bringup/test_c_full_dense_mlp.py` (and `BRINGUP_IMPL=stub`, and without BRINGUP_IMPL for the device).

## C.full_dense.mlp implement (attempt 1)
- `tt/mlp.py`: `TtDenseMLP` + `build_mlp`, copied from the prior's `tt/mlp.py` / `tt/model.py:build_mlp`, now TP=1. Every chip holds the whole gate/up [4096, 16384] and down [16384, 4096] (bf16, replicated, 0.38 GiB per chip). Each chip runs silu-fused gate linear, up linear, mul, down linear on its own CP slice [1, 1, S/4, 4096]. The prior's all_reduce is gone. All matmuls are HiFi4 + fp32 acc, DRAM interleaved, default program configs. Weights are fp8 + 128x128 block scale, dequantized in fp32 at load and then cast to bf16.
- hooks.py: `_mlp_module`, `device_component("mlp")` via `_cp_host_fn` (CP split in, concat out), and the hybrid builds an `mlp` override. `mlp` added to `DEVICE_STEPS["full_dense"]`.
- Gate: pcc 0.999998, rel L2 0.0019, row norm ratio [0.9978, 1.0027], worst row 0.0040, per-slice rel [0.00196, 0.00193, 0.00193, 0.00196]. The first "FAIL pcc=0" line is the precompile collect pass.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p_cp4/tests/bringup/test_c_full_dense_mlp.py`

## C.full_dense.mlp_residual.test.1
- Replaced the rendered PCC-only test with the frozen cp4 attn_residual pattern (the prior mimo_v2_6_d_p mlp_residual limits, same golden): PCC >= spec 0.99 (gated), plus output size, finite, rel L2 <= 0.01, per-token norm ratio in [0.99, 1.01], and rel L2 <= 0.01 per CP=4 row slice. Records rel_l2 / row_norm_ratio_{min,max} / rel_l2_slice_max as informational metrics.
- Why: PCC misses scale (2x scores 0.999998) and zeroed rows (0.9998); the prior's CPU mutation table is in the test docstring.
- Results: BRINGUP_IMPL=reference passes (pcc 0.999998, rel 0.00214, ratio [0.9991, 1.0009]); BRINGUP_IMPL=stub fails on PCC; the device mode (already registered) passes (pcc 0.999997, rel 0.00277, ratio [1.0003, 1.0023], slice rel 0.0027-0.0028).
- The leading `FAIL ... pcc=0.000000` line in each run is the precompile pass's comp_pcc stub (known issue), not the test.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p_cp4/tests/bringup/test_c_full_dense_mlp_residual.py`

## C.sliding_moe.attn_norm.test.1 (test review)
- Replaced the rendered one-liner with the cp4 full_dense norm checks (from `test_c_full_dense_ffn_norm.py`), set to layer 1 / attn_norm. The golden is the same as the prior's: s4096 chunk 1, in [2048, 4096] -> attn_norm. Checks: PCC >= 0.99 (gated), finite, rel L2 <= 0.03, per-token norm ratio [0.97, 1.03], worst row rel <= 0.015, rel L2 per CP slice <= 0.01, and the eps check (module on input x0.1 vs the CPU step: rel <= 0.02, worst row <= 0.04).
- CPU measurements (script /tmp/cp4an1/m.py, not kept): the smallest row mean square is 4.4e-4 (440x eps 1e-6), so eps 0 / 1e-7 / 2e-6 pass on the golden (rel 0.0024). On x0.1 they score rel 0.071 / 0.063 / 0.058. Other cases: sum instead of mean rel 0.98; `1 + w` PCC 0.78; no weight PCC 0.46; last 32 rows zeroed worst row 1.0; CP slices 1/2 swapped PCC 0.989 / rel 0.15; bf16 math rel 0.0027 / worst row 0.0070.
- Verified: reference passes (pcc 0.999997, rel 0.0024, worst row 0.0053). Stub fails (pcc 0). The device gate already passes with the existing TtRMSNorm registration: pcc 0.999996, rel 0.0027, ratio [0.9937, 1.0062], worst row 0.0070, slices 0.0025-0.0029, x0.1 rel 0.0025 / worst row 0.0057.
- The first `FAIL pcc=0` line is the precompile collect pass. Ignore it.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p_cp4/tests/bringup/test_c_sliding_moe_attn_norm.py` (with `BRINGUP_IMPL=reference` / `stub` for the freeze checks).

## C.sliding_moe.attention.test.1 (test review)
- Ported the prior's frozen `mimo_v2_6_d_p/tests/bringup/test_c_sliding_moe_attention.py` (same golden: s4096 chunk 1, start 2048, attn_norm [2048, 4096] -> attn_out). It keeps all of the prior's checks: PCC >= 0.99 (gated), finite, rel L2 whole and first 128 rows <= 0.02, per-token norm ratio [0.95, 1.05], worst row rel <= 0.08, and output closer to the CPU reference at window 128 than at 127 / 129.
- Added for CP=4: rel L2 per CP slice <= 0.02 (`rel_l2_cp_slice{r}_attention_L01`) and rel L2 over the halo rows (first 128 rows of slices 1..3) <= 0.02 (`rel_l2_halo_rows_attention_L01`).
- CPU mutation measurements (script /tmp/cp4sa/m.py, not kept; numbers in the test docstring): a 96-row halo passes the whole-chunk rel (0.0197) but fails halo rows (0.043) and the ratio. A 120-row halo is caught by the ratio (0.937). No halo is loud (PCC 0.988). RoPE restarting per slice fails PCC (the sink-dominated softmax explodes on wrong relative positions).
- Verified: BRINGUP_IMPL=reference passes (pcc 0.999989, rel 0.0047, slices 0.0044-0.0051, halo rows 0.0048, window check 0.0085 / 0.0093 vs 0). Stub fails (pcc 0). The device gate fails with NotImplementedError (no sliding module yet; hooks.py:115).
- The device module must return the full chunk's attn_out in chunk row order (slice r = rows r*512..).
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p_cp4/tests/bringup/test_c_sliding_moe_attention.py` (with `BRINGUP_IMPL=reference` / `stub` for the freeze checks).

## C.sliding_moe.attention.implement.1 (implement)
- `tt/attention.py`: `TtSlidingAttention(TtFullAttention)`, CP=4, TP=1 (all 64 Q / 8 KV heads, whole qkv / o_proj per chip, no CCL after o_proj). I split `TtFullAttention.__call__` into `_qkv_rope` / `_write_cache` / `_attend` / `_o_proj` (same ops, full-layer gate unchanged: pcc 0.999997, rel 0.0028) and added `q_scale` (folded into the Q rows). Sliding: RoPE theta 1e4 (`swa_rope_theta`), SDPA scale 2^-4 with 192^-0.5 / 2^-4 folded into Q, sink / 2^-4 replicated [1, 64, 1, 1] bf16, V x 0.707 zero-padded to 192.
- Departure from the plan (components.yaml says ring_joint): the ring op's sliding path is not used, for two reasons.
  (1) `topology=Linear` hangs on this FABRIC_2D 1x4 box: the halo writer is stuck on a fabric write slot in the chip 3 -> chip 0 wrap (triage in the proposed known issue). `Topology.Ring` runs.
  (2) With Ring it scores rel 0.032, ratio [0.927, 1.038] (fails). Plain SDPA on the same device Q/K/V scores 0.0133 (bf16 K/V, preset S), 0.0165 (bfp8 K/V), and 0.0087 / 0.0094 with fp32 dest. bfp8 K/V costs nothing on the CPU (0.0047 -> 0.0049), so the gap comes from the ring kernel.
- What runs instead (all TTNN, no host work in the forward):
  - halo = `all_gather` (dim 2, axis 1) of each chip's [this slice's last 128 K|V rows; the previous chunk's last 128 rows from its own chunk-major cache].
  - An identical reorder on every chip ([B3 A0 A1 A2]), then `mesh_partition`, so chip r gets its predecessor's tail (chip 0 gets chip 3's tail of the previous chunk).
  - `ttnn.transformer.scaled_dot_product_attention(concat(q[:128], q), concat(halo, k), concat(halo, v), is_causal, sliding_window_size=128, attention_sink)`, then output rows [128, 128+L), then V slice 128 -> concat_heads -> o_proj.
  - Chunk 0 (start == 0): non-causal with a per-chip attn_mask built at load per chunk size (window, and on chip 0 no halo column), shared across layers through `RingCCL.constant`.
  - Preset S (HiFi4, fp32 dest off, exact exp, q128/k128), as the owner rule says. `MIMO_SLIDING_SDPA_CFG=base` turns fp32 dest on.
- Sliding ring cache: bf16, no ring gather buffers (`gather_seq=0`), not the plan's bfp8. That is about +1.7 GB per chip over the plan for the 39 sliding layers at 56320, still well inside the budget (plan 15.95 of 27.2 GiB). A plan update should record this.
- hooks: `_attention_module` builds the sliding module (sink, window, swa theta). `DEVICE_STEPS["sliding_moe"]` = {attention}. attn_norm is not listed there yet: it passed C/S.01 through `device_component`, but it was never added to the hybrid.
- Gate: pcc 0.999913, rel 0.0133, first 128 rows 0.0114, ratio [0.9629, 1.0450], worst row 0.045, slices 0.012-0.014, halo rows 0.0137. Window check: 0.01258 vs 0.01579 (127) and 0.01471 (129).
- Probe (not kept): chunk 0 then chunk 1 on one device cache: chunk 0 rel 0.0143, ratio [0.9535, 1.0515] (the max is 0.0015 over the 1.05 limit; chunk 0 is not gated here, but a ladder check on chunk-0 rows may be). Chunk 1 rel 0.0133. The row norm ratio is the margin to watch; `base` would tighten it, but the owner's preset is S.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p_cp4/tests/bringup/test_c_sliding_moe_attention.py`

## S.sliding_moe.02.implement.1 (implement, attn_norm + attention on device)
- Attempt 1 failed on the harness's step check: attention vs the CPU attention on the same device input, row norm ratio [0.973, 1.047] against 1 +- 0.02 (`SWAP_STEP_DEFAULTS.swap_step_ratio`). pcc_swap_out was already 0.999995.
- Diagnosis (probe test in `tt/`, since deleted; CPU emulation scripts in /tmp/cp4s02, not kept), per-row ratio vs the fp32 CPU step:
  - CPU attention on the device bf16 Q/K/V: [0.979, 1.012], median -0.54%. Device Q/K norms were +0.08% / +0.05% high (V was fine), which points to the bf16 RoPE.
  - The SDPA kernel on its own, vs CPU attention on the same device Q/K/V: preset S [0.972, 1.049], base (fp32 dest) [0.996, 1.006]. S cannot pass the check at all.
  - With fp32 projections and fp32 RoPE: [0.992, 1.019], median +0.45%. CPU emulation shows that +0.45% comes from bf16 rounding of the q/k weights on attn_norm's massive channels (column mean |x| up to 34). Exact weights for the top 32 channels remove it, but the set changes per layer. bf16 rounding of Q contributes [0.989, 1.010]; bf16 rounding of K is small.
- What changed in `tt/attention.py` (TtSlidingAttention only; TtFullAttention's forward is unchanged):
  - `_qkv_rope` computes x @ W + x @ W_lo in fp32, with W_lo = W - bf16(W) stored as bfp8 (HiFi4).
  - Heads are formed by reshape+permute in fp32.
  - RoPE runs in fp32: tables built at load with fp32 angles (the reference/HF formula), sin sign-folded, rotate-half done as tile-aligned half swaps.
  - Q goes in as [bf16 hi | bf16 lo] (head dim 384), K' = [k | k], V' = [v | v]. The output is sliced to 128 before the o_proj, as before.
  - The cache still stores bf16 K/V of width 192.
  - Sliding SDPA default preset is now "base" (fp32 dest).
  - Env switches for comparison: `MIMO_SLIDING_SDPA_CFG=S`, `MIMO_SLIDING_QK=fp32|bf16`, `MIMO_SLIDING_WLO=0`.
- **Owner-rule departure:** the owner asked to keep preset "S". S fails this gate's frozen per-row check (above), so the default is fp32 dest. The owner should confirm or revisit this. Cost at chunk 2048 (512 rows per chip, one layer, host-timed, before W_lo was added): S/bf16 3.7 ms, base/bf16 4.0 ms, base/split 8.7 ms. The W_lo matmuls add to that (not measured). This is a target for the perf step: fusing the fp32 elementwise RoPE/split ops.
- Memory: W_lo (bfp8) is about 64 MB per sliding layer (about 2.5 GB per chip for 39 layers) on top of the plan. The fp32 RoPE tables are small. A plan update should record both.
- hooks: `DEVICE_STEPS["sliding_moe"]` = {attn_norm, attention} (attn_norm was missing from the hybrid).
- Gate: PASS. pcc_swap_out 0.999996. Attention vs CPU on the same input: rel 0.0032, worst row 0.0084, ratio [0.9927, 1.0082], bias +0.0001. Block out vs CPU attention: 0.00022.
- Regressions re-run, all PASS:
  - C.sliding_moe.attention: pcc 0.999985, rel 0.0056, ratio [0.984, 1.017]. Window check 0.0032 vs 0.0095 / 0.0095 (was 0.0126 vs 0.0158 / 0.0147).
  - C.full_dense.attention: unchanged, 0.999997.
  - S.full_dense.02: PASS.
- Probe (not kept): chunk 0 (first-mask path), then chunk 1 on the same cache, ran the split path at rows [0.996, 1.013], before W_lo was added.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p_cp4/tests/bringup/test_swap_sliding_moe_02_attention.py` (the first `FAIL pcc=0` lines come from the precompile collect pass).

## C.sliding_moe.attn_residual.test.1 (test review)
- Ported the prior mimo_v2_6_d_p reviewed test (same golden, s4096 chunk 1, [2048, 4096]): PCC gate plus rel L2 <= 0.01, per-token norm ratio [0.99, 1.01], and attn-term checks on delta = out - in (coef in [0.95, 1.05], rel <= 0.3). These are needed because the sink keeps attn_out at ~1% of h_mid.
- Added for CP=4: every check is also asserted per CP slice (rows [r*S/4, (r+1)*S/4)). CPU study on the golden, per slice (rel / coef / attn rel):
  - bf16 add: 0.0029 / 0.998-0.999 / 0.139-0.166.
  - Slices 1 and 2 of attn_out swapped: 0.0067 / 0.83-0.86 / 0.57-0.59.
  - Slice 0 written to slice 1: 0.0064 / 0.79 / 0.54.
  - Slice 3 dropped: 0.0123 / 0 / 1.0.
  - The per-slice rel L2 alone misses misplaced slices (both cases pass it). The per-slice attn-term checks catch them.
- Results: reference PASS (rel 0.0024, attn rel 0). Stub FAIL (pcc 0). Gate on device PASS: pcc 0.999996, rel 0.0029, ratio [0.9990, 1.0015], coef 1.020, attn rel 0.155, per-slice attn rel 0.142-0.171.
- The first `FAIL pcc=0` line in each run comes from the precompile collect pass.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p_cp4/tests/bringup/test_c_sliding_moe_attn_residual.py`

## C.sliding_moe.ffn_norm.test.1 (test review)
- Replaced the rendered one-liner with the checks from the cp4 `test_c_full_dense_ffn_norm.py`, set to layer 1. The golden is the same as the prior's: s4096 chunk 1, h_mid [2048, 4096] -> ffn_norm, w in [-0.012, 2.33]. Checks: PCC >= 0.99 (gated), finite, rel L2 <= 0.03, per-token norm ratio [0.97, 1.03], worst row rel <= 0.015, rel L2 per CP slice <= 0.01, and the eps check (module on input x0.1 vs the CPU step: rel <= 0.02, worst row <= 0.04). The limits are unchanged from the earlier cp4 norm tests.
- CPU measurements (script /tmp/cp4fn1/m.py, not kept). The smallest row mean square is 4.5e-4 (450x eps). Eps 0 / 1e-7 / 2e-6 pass on the golden (rel 0.0025). On x0.1 they score rel 0.069 / 0.061 / 0.056, so they are caught. Other cases: sum instead of mean rel 0.98; eps 1e-3 rel 0.35; `1 + w` PCC 0.61; no weight PCC 0.44; CP slices 1/2 swapped PCC 0.95 / rel 0.31; last 32 rows zeroed worst row 1.0; bf16 math rel 0.0028 / worst row 0.0061.
- Verified: reference passes (pcc 0.999997, rel 0.0024, worst row 0.0047). Stub fails (pcc 0). The device gate already passes with the existing TtRMSNorm registration: pcc 0.999996, rel 0.0029, ratio [0.9962, 1.0038], worst row 0.0061, slices 0.0027-0.0030, x0.1 rel 0.0024 / worst row 0.0045.
- The first `FAIL pcc=0` line comes from the precompile collect pass. Ignore it.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p_cp4/tests/bringup/test_c_sliding_moe_ffn_norm.py`
