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

## C.sliding_moe.router.test.1
- Replaced the rendered test with the prior bring-up's frozen test (`models/demos/mimo_v2_6_d_p/tests/bringup/test_c_sliding_moe_router.py`): same golden (rung s4096, chunk 2048, layer 1 [2048, 256]), same reference, so its limits still apply.
- Added the guards `run_component_test` has: in device mode the test fails if the step is deferred to op-gen or if `device_component` returns a CPU bridge.
- Checks: PCC >= 0.99 (gated), exactly 8 nonzeros per row, weights >= 0, mean selection overlap >= 0.985, matched-row weight rel L2 <= 0.005, row sums 1 +- 0.01.
- Reference: PCC 0.999328, overlap 0.99878, matched 2028/2048, rel L2 0.00159, row sums 1.0. Stub: PCC 0, fails.
- Device gate: currently fails with NotImplementedError (no router module yet; that is the implement step).
- Implementer: compute the logits in fp32, and do the top-k selection on fp32 (or bias-recentred) sigmoid + bias. Rounding the bias or the choice score to bf16 fails the gate. The weights are the unbiased sigmoid, renormalized. Under CP=4 the output must be the full [S, 256], with the 4 slices in order.
- Re-run: `BRINGUP_IMPL=reference|stub scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p_cp4/tests/bringup/test_c_sliding_moe_router.py` (PYTHONPATH=$PWD).

## C.sliding_moe.router.implement.1
- Copied the prior `mimo_v2_6_d_p/tt/router.py:TtRouter` into `tt/router.py`. The math is unchanged: fp32 weight, HiFi4 + fp32 acc linear with fp32 output, SFPU sigmoid, fp32 bias add, `ttnn.topk(8)`, gather of the unbiased sigmoid, sum/div, and a bf16 `ttnn.scatter` into a zeros table to get the dense boundary.
- CP=4 changes: the weight and bias are replicated, and each chip routes only its own S/4 rows. There is no CCL. The zeros (and the bias in fused mode) are built once at load for max_rows = max chunk / 4 (1280) and sliced on the device for each chunk.
- `MIMO_ROUTER_MODE=fused` (moe_grouped_topk, TF32 keys) can still be selected for comparison.
- hooks changes:
  - Added `_max_chunk`, `_router_module` and `_router_host_fn`. The host fn takes the CP slices in and concatenates the per-chip [S/4, 256] outputs.
  - Added `device_component` "router".
  - Added "router" to `DEVICE_STEPS["sliding_moe"]` and to the hybrid overrides.
  - full_moe is not touched; its router is not gated yet.
- Gate: PASS. pcc_router_L01 0.999129, nnz 8/row, selection overlap 0.99841, matched 2022/2048, matched rel L2 0.00102, row sums [0.9976, 1.0020]. The first `FAIL pcc=0` line comes from the precompile collect pass.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p_cp4/tests/bringup/test_c_sliding_moe_router.py`

## C.sliding_moe.experts.test.1 (test review)
- Replaced the rendered one-liner with the prior bring-up's frozen test (`mimo_v2_6_d_p/tests/bringup/test_c_sliding_moe_experts.py`). It uses the same golden (s4096 chunk 1, layer 1, ffn_norm [2048, 4096] + router [2048, 256] -> experts_out) and the same reference, so its limits and mutation study still apply.
- Checks: PCC >= 0.99 (gated), finite output, rel L2 <= 0.03, per-token norm ratio [0.97, 1.03], worst row rel L2 <= 0.1.
- Added: the deferred / CPU-bridge guards of `run_component_test`, and rel L2 per CP slice <= 0.03 (rows [r S/4, (r+1) S/4)).
- CPU study (script /tmp/cp4ex/m.py, not kept): I zeroed the routing of one source slice to one chip's 64 experts, for each of the 16 (slice, chip) pairs, to model a lost EP dispatch / combine pair.
  - Every case fails the existing checks: PCC 0.954-0.990, rel 0.14-0.30, worst row 0.92-0.98, that slice's rel 0.29-0.58.
  - Slice 3 x1.03 fails the ratio check (1.033), and its slice rel is 0.0304.
  - So the per-slice check is a backstop. On this golden it caught nothing the per-row checks miss.
- Results:
  - Reference: PASS. pcc 0.999997, rel 0.00234, ratio [0.9954, 1.0037], worst row 0.0050, slices 0.00233-0.00235.
  - Stub: FAIL (pcc 0).
  - Device gate: FAIL with NotImplementedError (no experts module yet; that is the implement step).
- Implementer: the device output must be the full [S, 4096], with the 4 slices in order. Per the known issues, use `ttnn.bringup.unified_routed_expert_moe(high_precision=True)` at HiFi4 + fp32 dest with bf16 activations. The kernel's default LoFi Silu path fails the norm-ratio check.
- The first `FAIL pcc=0` line comes from the precompile collect pass.
- Re-run: `BRINGUP_IMPL=reference|stub PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p_cp4/tests/bringup/test_c_sliding_moe_experts.py`

## C.sliding_moe.experts.implement.1
- Copied `mimo_v2_6_d_p_2x2/tt/experts.py` into `tt/experts.py` and adapted it to CP=4.
  - `DISPATCH_AXIS = 1`, one dispatch group of 4 chips (dgs 4, groups 1), 64 complete experts per chip (chip c holds experts 64c..64c+63).
  - Each chip takes its own S/4 rows: dense routing -> `ttnn.topk` -> masked_bincount -> `ttnn.bringup.offset_cumsum(cluster_axis=1)` -> `ttnn.bringup.dispatch` -> `ttnn.bringup.unified_routed_expert_moe(high_precision=True)` (HiFi4 + fp32 dest, bf16 ROW_MAJOR x, bfp8 weights) -> `ttnn.bringup.combine` -> `post_combine_reduce`.
  - The output is [1, 1, S/4, H] on chip c. There is no all_reduce and no all_gather after it.
  - The per-expert cap is the max chunk (8192), with capacity factor 8.
  - `build_experts` lives in `tt/experts.py` (this model has no `tt/model.py` yet).
  - `MIMO_EXPERTS_MODE=loop|unified_lofi` are kept for comparison.
- Global-expert-idx table: built from `create_global_expert_idx_table(64, dgs=4, groups=1)` with the mapper `ShardTensor2dMesh(dims=(0, 1))`. `get_ep_mesh_mapper` shards the wrong dims on 1x4 with one group.
- The weight layout is the one `TtRoutedExpert` derives from the mesh shape, so chip c holds experts 64c.. (same as the 1x4 prior). The weights are cached under `generated/mimo_v2_6_d_p_cp4/tt_cache/experts`; the first run dequantizes the mxfp4 experts.
- Fork change:
  - `ttnn.bringup.dispatch` and `ttnn.bringup.combine` rejected `cluster_axis=1` with a host TT_FATAL. Their factories and kernels already support axis 1.
  - Added the opt-in `allow_cluster_axis_1` (default False) to both forks: a host-check change only, so the program with the option off is unchanged. Rebuilt with `./build_metal.sh`.
  - New tests: `dispatch/tests/unit/test_dispatch_cluster_axis_1.py` and `combine/tests/unit/test_combine_cluster_axis_1.py` (exact checks plus a refusal check). Each fails when its expected output is corrupted (checked once by hand).
  - Regression with the option off: model cases 12/12 pass; `fork_source` dispatch 82 tests and combine 117 tests, 0 regressions each.
  - CHANGELOG entries and INDEX rows are updated.
- hooks: added `_experts_module` / `_experts_host_fn` (the CP slices of x and the dense router in, the per-chip outputs concatenated). Added `device_component` "experts", added "experts" to `DEVICE_STEPS["sliding_moe"]`, and added it to the hybrid overrides.
- Gate: PASS. pcc_experts_L01 0.999980, rel L2 0.0063, row norm ratio [0.9928, 1.0064], worst row 0.0152, slice rel L2 0.0063-0.0064. The first `FAIL pcc=0` line comes from the precompile collect pass.
- Gotcha: fork test files with the same basename in two forks collide under pytest (`tests.unit.<name>`). Name them per fork.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p_cp4/tests/bringup/test_c_sliding_moe_experts.py`

## C.sliding_moe.ffn_residual.test.1 (test review)
- Replaced the rendered one-line test with the prior bring-up's frozen test (`models/demos/mimo_v2_6_d_p/tests/bringup/test_c_sliding_moe_ffn_residual.py`), which uses the same golden. Kept its limits: PCC >= 0.99 (spec), rel L2 <= 0.01, per-token norm ratio in [0.99, 1.01], experts term on out - h_mid: coef in [0.97, 1.03], rel <= 0.1. Why: PCC alone passes a dropped experts_out (0.9887), 0.5 x experts_out and 2 x (h_mid + experts_out).
- Results: reference PCC 0.999997 / rel 0.0024 / ratio [0.9991, 1.0010] / coef 1.0 / rel 0, which passes. Zero stub PCC 0.0, which fails. Device (default impl) PCC 0.999996 / rel 0.0029 / ratio [0.9993, 1.0016] / coef 1.0014 / experts rel 0.0114, which passes.
- The first `FAIL pcc=0.000000` line in each run comes from the precompile collect pass, not the real pass.
- Re-run: `PYTHONPATH=$PWD [BRINGUP_IMPL=reference|stub] scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p_cp4/tests/bringup/test_c_sliding_moe_ffn_residual.py`

## C.full_moe.attn_norm.test.1 (test review)
- Replaced the rendered one-liner with the frozen cp4 `test_c_sliding_moe_attn_norm.py` checks, set to layer 5. The step is the same RMSNorm (plain `w`, eps 1e-6), and the golden is the prior's: in [2048, 4096] -> attn_norm.
- Limits are unchanged from the sibling test:
  - PCC >= 0.99 (gated), plus finite output.
  - rel L2 <= 0.03.
  - Per-token norm ratio in [0.97, 1.03].
  - Worst row rel L2 <= 0.015.
  - rel L2 per CP slice <= 0.01.
  - eps check: the module on input x0.1 vs the CPU step, rel <= 0.02 and worst row <= 0.04.
- CPU measurements (script /tmp/cp4fan/m.py, not kept; numbers are in the test docstring):
  - The smallest row mean square is 7.5e-4 (750x eps), so a wrong eps passes on the golden itself.
  - On the x0.1 input, eps 0 / 1e-7 / 2e-6 score rel 0.031 / 0.028 / 0.028 and fail. bf16 math scores rel 0.0023.
  - Sum instead of mean: rel 0.98. `1 + w`: PCC 0.41. Last 32 rows zeroed: worst row 1.0. CP slices 1/2 swapped: PCC 0.65.
- Results:
  - Reference: PASS (pcc 0.999997, rel 0.0024, worst row 0.0045).
  - Stub: FAIL (pcc 0).
  - Device gate (the existing TtRMSNorm registration): PASS. pcc 0.999996, rel 0.0028, ratio [0.9969, 1.0027], worst row 0.0050, slices 0.0028-0.0029, x0.1 rel 0.0023 / worst row 0.0038.
- The first `FAIL pcc=0` line comes from the precompile collect pass.
- Re-run: `PYTHONPATH=$PWD [BRINGUP_IMPL=reference|stub] scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p_cp4/tests/bringup/test_c_full_moe_attn_norm.py`

## C.full_moe.attention.test.1 (test review)
- Replaced the rendered one-liner with the prior's frozen `mimo_v2_6_d_p/tests/bringup/test_c_full_moe_attention.py` (same golden: s4096 chunk 1, attn_norm [2048, 4096] -> attn_out, layer 5).
  - Kept its checks: PCC >= 0.99 (gated), finite output, rel L2 <= 0.015, first 128 rows <= 0.015, per-token norm ratio in [0.97, 1.03], and worst row rel L2 <= 0.06.
  - Kept its asserts that layer 5 is full attention and has no sink.
- Added the cp4 full_dense check: rel L2 per CP slice (4 x 512 rows) <= 0.012, recorded as `rel_l2_cp_slice{r}_attention_L05`.
- CPU measurements on layer 5 (script /tmp/cp4fma/m.py, not kept; the layer-0 /tmp/cp4m/measure.py with L=5). Values are PCC / rel / worst slice rel:
  - Reference: 1.0 / 0.0017 / 0.0017.
  - Ring of one hop only: 0.99995 / 0.0113 / 0.0192, worst row 0.037. Only the per-slice check catches it.
  - RoPE positions restarting per slice: 0.9997 / 0.0258 / 0.0358.
  - No ring: 0.99961 / 0.0296 / 0.0412.
  - Non-causal across slices: 0.99965 / 0.0275 / 0.0407.
  - Slices 0 and 1 swapped: PCC 0.983.
- Results:
  - Reference: PASS (pcc 0.999999, slices 0.0017).
  - Stub: FAIL (pcc 0).
  - Device gate: PASS with the existing attention registration. pcc 0.999996, rel 0.0030, first 128 rows 0.0031, ratio [0.9998, 1.0031], worst row 0.0044, slices 0.0030-0.0031.
- The first `FAIL pcc=0` line comes from the precompile collect pass.
- Re-run: `PYTHONPATH=$PWD [BRINGUP_IMPL=reference|stub] scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p_cp4/tests/bringup/test_c_full_moe_attention.py`

## C.full_moe.attn_residual.test.1 (test review)
- Replaced the rendered one-liner with the cp4 `test_c_sliding_moe_attn_residual.py`, set to layer 5. Those checks are run on the whole chunk and on each CP slice. The attention-term limits come from the prior's frozen full_moe test:
  - coef in [0.98, 1.02].
  - attn rel <= 0.05.
  - These are tighter than layer 1's [0.95, 1.05] / 0.3. Layer 5 has full attention and no sink, so ||attn_out|| is 78.9 against ||in|| 121.4.
- Other limits are unchanged:
  - PCC >= 0.99 (gated).
  - rel L2 <= 0.01.
  - Per-token norm ratio in [0.99, 1.01].
  - Finite output and output size.
- CPU measurements (script /tmp/cp4far/m.py, not kept; numbers are in the test docstring). Values are per-slice rel L2 / coef / attn rel:
  - bf16 add: 0.0030 / 1.0 / 0.0028.
  - Attn_out slices 1 and 2 swapped: 0.163 / 0.96-0.97 / 0.26.
  - Slice 3's attn_out dropped: 0.626 / 0 / 1.0.
  - attn_out shifted one row: 0.165 on every slice.
  - Unlike layer 1, the whole-chunk rel L2 already catches slice misplacement here (0.08-0.31).
- Results:
  - Reference: PASS (pcc 0.999997, rel 0.0025).
  - Stub: FAIL (pcc 0).
  - Device gate (existing attn_residual registration): PASS. pcc 0.999996, rel 0.0030, ratio [0.9998, 1.0018], slices 0.0030, coef 1.0007, attn rel 0.0028.
- The first `FAIL pcc=0` line comes from the precompile collect pass.
- Re-run: `PYTHONPATH=$PWD [BRINGUP_IMPL=reference|stub] scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p_cp4/tests/bringup/test_c_full_moe_attn_residual.py`

## C.full_moe.ffn_norm.test.1 (test review)
- Replaced the rendered one-liner with the cp4 `test_c_sliding_moe_ffn_norm.py`, set to layer 5. Limits unchanged: PCC >= 0.99 (gated), finite, rel L2 <= 0.03, per-token norm ratio [0.97, 1.03], worst row rel <= 0.015, rel L2 per CP slice <= 0.01, eps check (module on input x0.1 vs the CPU step: rel <= 0.02, worst row <= 0.04). This is a superset of the prior's frozen full_moe test (PCC, rel 0.03, ratio).
- CPU measurements on layer 5 (script /tmp/cp4fmn/m.py, not kept; numbers in the test docstring). w in [-1.06, 8.75], smallest row mean square 8.1e-4. Every mutation is caught: sum-for-mean, eps 1e-3, `1 + w`, no weight, slices swapped, last 32 rows zeroed. Wrong eps (0 / 1e-7 / 2e-6) is caught only by the x0.1 check: rel 0.028-0.032 against the 0.02 limit. That margin is thinner than layer 1's, but bf16 scores 0.0023.
- Results:
  - Reference: PASS (pcc 0.999997, rel 0.0024).
  - Stub: FAIL (pcc 0).
  - Device gate: PASS with the existing TtRMSNorm registration. pcc 0.999996, rel 0.0028, ratio [0.9967, 1.0031], worst row 0.0054, slices 0.0028-0.0029, x0.1 rel 0.0023 / worst row 0.0040.
- The first `FAIL pcc=0` line comes from the precompile collect pass.
- Re-run: `PYTHONPATH=$PWD [BRINGUP_IMPL=reference|stub] scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p_cp4/tests/bringup/test_c_full_moe_ffn_norm.py`

## C.full_moe.router.test.1
- Replaced the rendered test with the prior bring-up's frozen test (`models/demos/mimo_v2_6_d_p/tests/bringup/test_c_full_moe_router.py`), set to layer 5. It uses the same golden ([2048, 256]) and reference, so its limits and mutation study still apply. Added the same deferred / CPU-bridge guards as the cp4 sliding router test.
- Checks: PCC >= 0.99 (gated), exactly 8 nonzeros per row, weights >= 0, mean selection overlap >= 0.985, matched-row weight rel L2 <= 0.005, row sums 1 +- 0.01. Layer 5 needs these extra checks: a bf16 correction bias still passes PCC (0.9937) but fails overlap (0.952).
- Reference: PCC 0.998186, overlap 0.99573, matched 1978/2048, rel L2 0.00154, row sums 1.0. Stub: PCC 0, fails.
- Device gate: already PASSES with the existing `tt/router.py` registration: pcc_router_L05 0.998111, overlap 0.99536, matched 1973/2048, rel L2 0.00107, row sums [0.9976, 1.0023]. The first `FAIL pcc=0` line comes from the precompile collect pass.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p_cp4/tests/bringup/test_c_full_moe_router.py` (or with BRINGUP_IMPL=reference|stub).

## C.full_moe.experts.test.1 (test review)
- Replaced the rendered one-liner with the cp4 `test_c_sliding_moe_experts.py` body, set to layer 5. The docstring is the prior bring-up's frozen full_moe experts review, which covers the same golden and reference: bf16 x is required because of outlier channels, there are tiny routing weights, and the mutation study.
- Limits are unchanged from the prior test and the cp4 layer-1 test:
  - PCC >= 0.99 (gated).
  - Finite output.
  - rel L2 <= 0.03.
  - Per-token norm ratio [0.97, 1.03].
  - Worst row rel L2 <= 0.1.
  - rel L2 per CP slice <= 0.03.
  - Deferred / CPU-bridge guards.
- Host-only CPU check on the golden (script not kept), whole-chunk rel / per-slice rel:
  - Slices 1 and 2 swapped: 0.97 / 1.48, 1.30.
  - Slice 3 zeroed: 0.55 / 1.0.
  - Slice 3 x1.05: 0.0275 / 0.050. It passes the whole-chunk rel and is caught by the slice and ratio checks.
- Results:
  - Reference: PASS (pcc 0.999987, rel 0.0050, slices 0.0047-0.0057).
  - Stub: FAIL (pcc 0).
  - Device gate (existing experts registration): PASS. pcc 0.999974, rel 0.0073, ratio [0.9859, 1.0208], worst row 0.021 (row 12), slices 0.0069-0.0079.
- The first `FAIL pcc=0` line comes from the precompile collect pass.
- Re-run: `PYTHONPATH=$PWD [BRINGUP_IMPL=reference|stub] scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p_cp4/tests/bringup/test_c_full_moe_experts.py`

## C.full_moe.ffn_residual.test.1 (test review)
- Replaced the rendered one-liner with the prior's frozen full_moe ffn_residual test (same golden, [2048, 4096]), restructured like the cp4 `test_c_full_moe_attn_residual.py`: every check also runs per CP slice.
- Limits (unchanged from the prior and cp4 layer 1): PCC >= 0.99 (gated), output size, finite, rel L2 <= 0.01 (whole and per slice), per-token norm ratio [0.99, 1.01], experts term on delta = out - h_mid: coef in [0.97, 1.03] and rel <= 0.1 (whole and per slice).
- At layer 5 the experts term is only ~6.7% of ||out||, so PCC misses most experts_out bugs. The experts-term checks catch them.
- CPU measurements (script /tmp/cp4ffr/m.py, not kept; numbers in the test docstring):
  - bf16 add: per-slice experts rel 0.017-0.021.
  - Golden out itself: 0.026-0.032.
  - Slices 1 and 2 swapped: slice rel 0.09, experts rel 1.3-1.5.
  - Slice 3 dropped: slice rel 0.074, coef 0.
  - Slice 3 halved: coef 0.5.
  - Shifted one row: experts rel 1.39.
  - All caught.
- Results:
  - Reference: PASS (pcc 0.999998, rel 0.0019).
  - Stub: FAIL (pcc 0).
  - Device gate (existing ffn_residual registration): PASS. pcc 0.999997, rel 0.0023, ratio [0.9988, 1.0014], slices 0.0023, coef 1.0016, experts rel 0.0194 (slices 0.0175-0.0213).
- The first `FAIL pcc=0` line comes from the precompile collect pass.
- Re-run: `PYTHONPATH=$PWD [BRINGUP_IMPL=reference|stub] scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p_cp4/tests/bringup/test_c_full_moe_ffn_residual.py`

## M.1 assemble.1
- Added `tt/model.py`, a port of the prior's `tt/model.py` to CP=4:
  - `TtEmbedding`: replicated bf16 table, cached at `generated/mimo_v2_6_d_p_cp4/tt_cache/embed_bf16*` (1.2 GB). The ids arrive as chip c's CP slice [1, 1, S/4] uint32 (`ids_to_device`, ShardTensorToMesh dim 2).
  - `TtMiMoBlock`: runs the reference DENSE_GRAPH / MOE_GRAPH through `run_block`. Each intermediate is freed after its last reader.
  - `TtMiMoModel`: one shared `RingCCL`, `new_caches`, `prefill_chunk`.
- The builders mirror the hooks' component path. They use the same cp4 modules and arguments: RoPE max_seq = longest rung or target seq, the chunk sizes of every rung, router rows = max_chunk / 4, and `experts.build_experts` with max_chunk. The hooks' component functions are unchanged.
- The router hands (idx, wts) straight to the experts (`experts(x, idx=, wts=)`), as in the prior, with no dense -> topk round. The dense output is freed.
- Residual add + the next norm are fused (`TtRMSNorm.fused_add`, stash). This is on by default, as in the prior; `MIMO_FUSE_RESIDUAL_NORM=0` turns it off.
- The experts' dispatch / combine modules are prebuilt at load for every chunk size (`_seq_modules`), so nothing is built per chunk.
- `bringup/hooks.py`:
  - `MiMoCPDeviceModel` and `_DeviceState` (ring caches from `TtMiMoModel.new_caches`).
  - `layer()` calls `cache.bind_chunk(S)`, which is a no-op after the first chunk. It writes any golden prefix loaded before the chunk size was known, so the host write happens on a cold call only.
  - `device_model` returns the all-device model by default; `BRINGUP_HYBRID=1` selects the old `HybridDeviceModel`.
  - `to_host` / `from_host` concatenate / shard the CP slices on the sequence dim.
- Gate (s4096, layers 0-5): PASS.
  - Worst layer PCC 0.999969, worst state PCC 0.999926.
  - Host transfers per layer (warm) 0.
  - Chunk 0 took 39.7 s (cold, kernel compiles); chunk 1 took 0.20 s.
  - Load: roughly 45 s with the expert cache warm (estimated as the 86 s test call minus the chunks; not printed on its own).
- The orchestrator's earlier run (on the hybrid) died writing the junit xml with "No space left on device", because the 32 GB cp4 expert cache was being written. The cache is now complete and 26 GB is free (known issue proposed).
- Not run here: final norm and logits (the layer 0-5 subset does not end at the last layer). They follow the prior.
- Re-run: `PYTHONPATH=$PWD BRINGUP_RUNG=s4096 scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/common/bringup/tests/test_ladder.py` (`BRINGUP_HYBRID=1` for the hybrid).

## K.1 contract.1
- Added `tt/runners/` (`adapter.py`, `kv_contract.py`). Registered `mimo_v2_6_d_p_cp4` in `common/prefill/adapter.py:ADAPTER_PATHS`. This is a port of the 1x4 TP prior's runners to CP=4.
- Migratable KV layout (`MiMoContractKVCP`):
  - Per chip, one K slab and one V slab: bf8 [users * layers, 1, max_seq/4, 1536], DRAM NdShard [1, 1, 32, 1536] ROUND_ROBIN_1D.
  - All KV heads sit side by side (nlp_concat_heads order). Full layers hold 4 heads followed by 768 zero columns; V heads keep the attention's zero pad at columns 128..191.
  - Positions are chunk-major across the chips, the same as the attention's ring cache. It is written with `update_padded_kv_cache(cluster_axis=1, kv_actual_global=start)`.
  - Address table: config 0 = K, config 1 = V. Position p lives on chip (p % C) // (C/4) through a one-chip device group at MeshCoordinate(0, chip).
- Why a separate slab: the ring cache is DRAM interleaved with no slot dim, so a table entry cannot address it (known issue proposed). A 1536-wide slab means one write per K/V per layer instead of one per head.
- Hook: `TtKVCacheRing.kv_sink` (default None, so the ladder and hybrid paths are unchanged). `TtFullAttention._write_cache` calls it before the ring write; the sliding layers use the same method. The runtime sets the sink per (layer, chunk, slot) and clears it afterwards.
- KV caches: `allocate_kv_cache` builds per-slot `TtKVCacheRing`s bound to the served chunk. Their ring-gather buffers come from the adapter's own `RingCCL`; the model's `RingCCL` supplies the semaphores.
- Engine input: the whole chunk arrives replicated on every chip as [1, 1, chunk]. `ttnn.mesh_partition(dim=-1, cluster_axis=1)` gives each chip its CP slice, then `ttnn.minimum` (in TILE) clamps the 0xFFFFFFFF pad to V-1. The expert dispatch buffers are sized for the worst case, so pad rows cannot displace real tokens.
- Acks: an event sync, then `sink(global layer, request_id)` after each block.
- Gate (s4096, chunk 2048, layers 0-5, slot 1): PASS.
  - contract_checks_failed 0, acks_early 0 (48 blocks checked).
  - pcc_producer_kv_k 0.99996, pcc_producer_kv_v 0.99990.
  - Test time 78 s.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/common/bringup/tests/test_contract.py`
