# google/gemma-4-26B-A4B-it bring-up: breadcrumbs

Append-only log, one section per task attempt: what was done, decisions and why, gotchas, the re-run command, the verdict.

## R.2 reference (attempt 1)
- Wrote `reference/gemma4_ref.py` (standalone, safetensors direct, fp32, all layers resident ~104 GB) and the
  `reference` + `hf_layers` hooks. Attempt 0 failed only because `hf_layers` was missing (HF oracle is
  Gemma4ForConditionalGeneration; layers at `model.model.language_model.layers`).
- Block graph (same for sliding and global; attention type chosen in `chunk_context` / `_attention`), 14 steps:
  attn_norm, attention [stateful], post_attn_norm, attn_residual -> h_mid, ffn_norm, mlp, post_mlp_norm,
  router (on h_mid), moe_norm, experts(moe_norm, router), post_moe_norm, ffn_combine, post_ffn_norm,
  ffn_residual (`(h_mid + ffn_out) * layer_scalar`) -> out.
- Decision: the `router` boundary is a dense [S, 128] routing matrix (top-8 weights after renorm and
  per_expert_scale, zeros elsewhere), so one tensor carries both selection and weights; `experts` groups tokens
  per expert in ascending expert id (HF accumulation order).
- State: key/value [Hkv, max_seq, D] per layer; K post k_norm + RoPE, V = unscaled RMS(raw proj). On global layers
  V comes from k_proj (no v_proj) but is NOT equal to K.
- Attention: scale 1.0; sliding window keys (q-1024, q]; per 1024-query block the key range is bounded by the
  window, attend against the state (no prefix recompute). Global RoPE: proportional, 64 of 256 freqs non-zero.
- Logits: tied embedding + softcap 30*tanh(x/30). Embed scale sqrt(2816) in fp32.
- Gate: all 30 layers pcc=1.0000000 (maxabs <= 1.3e-3), logits pcc=1.0000000, top1_match=1.0000, ~1 min.
- Sanity (not a gate): 3000 tokens in 1000-token chunks vs one-shot on layers 0,4,5: state maxabs <= 8e-6; graph
  replay of layers 0 and 5 from recorded boundaries exact (0.0).
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup.reference.check_hf --seq 512`

## R.3 reference (attempt 1)
- Attempt 0 failed on chunked vs one-shot (hidden pcc 0.99983, state 0.99986); graph replay was already exact.
- Diagnosis: comparing every boundary of chunk 2 (tokens 2048-4095) between one-shot and 2x2048, everything was
  bit-exact until `L0.experts_out`. After that, near-tie top-8 flips (1 token at L14, 31 at L29) amplified the difference.
  Cause: MKL sgemm output for a row depends on M when M is small or odd, and the per-expert group size differs
  between chunked and one-shot.
- Fix: `experts_forward` zero-pads each expert's token group to a multiple of `EXPERT_ROW_BLOCK = 32` rows and
  slices the pad off. The math is unchanged. Chunked == one-shot now bit-exact (final_norm and all K/V maxabs 0.0).
- Gate: hidden pcc=1.000000000, state min pcc=1.000000000, sliding/global graph replay max abs 0.0, 0 missing.
- check_hf was not re-run (padding rows are discarded, so the math is the same as R.2).
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup.reference.check_reference --seq 4096 --chunk 2048`

## R.2 coverage check past the sliding window (2026-09-25, by hand)
- R.2 ran HF parity at 512 tokens, below the 1024-token sliding window. Checked by hand on the canonical prompt:
  HF fp32 vs the s4096 golden, top-1 agreement 100% at positions 0-1024, 1024-2048 and 2048-4096; next-token
  accuracy on the text 78.3% / 55.8% / 45.8% for both. The fall with position is the model's own (header, then the
  novel recited from memory). The framework now requires hf.parity_seq > sliding window at spec validation; this
  spec keeps 512 for run1 (changing R.2 would reset the goldens), with this check as the evidence.

## PL.1 plan (attempt 1)
- Wrote `plan.yaml`, `plan.md` and `components.yaml` (14 steps x 2 block types + embed/final_norm/lm_head). tasks.yaml unchanged.
- Scheme: residual replicated; attention TP=4 by head (sliding: 4 Q + 2 KV heads per chip; global: 4 Q heads + KV head
  c//2, so each of the 2 global KV heads lives on 2 chips); dense MLP TP=4 (528 per chip, padded to 544); experts EP=4
  (32 per chip, bf16); router replicated in fp32; embedding replicated; tied LM head as a vocab-sharded copy (extra).
- 3 all_reduces per layer (o_proj, MLP, experts): post_mlp_norm and post_moe_norm are separate nonlinear norms,
  so the MLP and MoE partials cannot share one all_reduce as ERNIE's routed and shared outputs do.
- State: full-length K and V (contract), sliding 2x256 per chip, global 1x512 per chip (`kv_heads: 2` in the entry); plus
  a bfp8 contract-copy estimate (1.84 GiB) as on ERNIE.
- Gate numbers: 21.52 GiB per chip of a 27.20 GiB budget, experts 10.63 GiB. Using bfp8 experts would free 4.6 GiB.
- Global k_proj layers are listed by explicit layer index and counted as replicated. The schema has no divide-by-2
  placement, so this over-counts by about 6 MB per layer.
- Gotcha: `unified_routed_expert_moe` has no GeGLU activation (see the known_issues proposal). The experts component is
  planned as dispatch plus per-expert matmul with apply_geglu; a fused GeluTanh activation is a perf follow-up.
- Sliding SDPA: chunked SDPA takes no `sliding_window_size`. The plan is to concatenate the previous 1024 K/V (read from the
  cache) with the chunk and run causal `scaled_dot_product_attention(sliding_window_size=1024)`, as in gemma4/tt/attention/prefill.py.
- All metrics pass except plan_approved (needs a person's approval in approvals.yaml).
- Re-run: `PYTHONPATH=$PWD python -m models.demos.common.bringup.plan.check_plan`

## C.sliding.attn_norm test (attempt 1)
- Reviewed the rendered test. Golden: s4096 rung, chunk 1 (start 2048), layer 0, `in` -> `attn_norm` [2048, 2816] bf16. PCC is the right
  mode (float output), threshold stays at the spec's 0.99.
- Gap: PCC ignores scale. On this golden, `rms(x) * (1 + w)` scores PCC 0.9977 (passes), and sum-instead-of-mean scores about 1.0.
  Added scale checks after the gated PCC: relative L2 <= 0.03 and per-token ||got||/||want|| in [0.97, 1.03]. They are recorded as
  informational metrics (`rel_l2_attn_norm_L00`, `row_norm_ratio_{min,max}_attn_norm_L00`) and asserted in the test.
- Measured (CPU, from the golden): reference rel 0.0024, ratio [0.9985, 1.0012]; bf16 in/out rel 0.0029; `1 + w` rel 0.151, ratio >= 1.10;
  LayerNorm-instead-of-RMS rel 0.0115 (not caught; mean is about 0 on this data, so harmless).
- The test now builds the harness pipeline inline (`component_golden`, `_step`, `module_under_test`, `compare`), because
  `run_component_test` returns only a bool. The metric name and threshold are unchanged.
- Verified: BRINGUP_IMPL=reference PASS (pcc 0.999997); BRINGUP_IMPL=stub FAIL (pcc 0.0). Device gate fails for now with
  `NotImplementedError: no device module for attn_norm` (the implement step adds it).
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_c_sliding_attn_norm.py`

## C.sliding.attn_norm implement (attempt 1)
- New `tt/rms_norm.py`: `TtRMSNorm` (replicated `ttnn.rms_norm`, TILE gamma [1,1,1,2816] bf16, HiFi4 + fp32 dest acc, eps 1e-6),
  adapted from the prefill path of `gemma4/tt/rms_norm.py`. The checkpoint weight is used as is (Gemma-4 is `x * w`, no `1 + w`).
  Helpers `to_device_replicated` (host [S,H] -> replicated [1,1,S,H] TILE bf16) and `replicated_to_host` (chip 0's copy).
- `hooks.py`: `DEVICE_STEPS = {"sliding": {"attn_norm"}, "global": set()}` and `_NORM_WEIGHTS` (step -> checkpoint weight name).
  `device_component` builds the norm for any step in `_NORM_WEIGHTS` and loads only that one weight. `device_model` is now a
  `HybridDeviceModel`: the CPU reference with the DEVICE_STEPS of each layer's block type swapped in through
  `run_block(overrides=...)`, with host tensors between steps. Later implement steps add their step to DEVICE_STEPS (and to
  `_NORM_WEIGHTS` for norms). Once more of the block runs on the device, it should keep activations on the device.
- The device output (bf16) is cast back to the input dtype, so the CPU steps downstream still see fp32.
- Gate: pcc_attn_norm_L00 = 0.999996, rel_l2 0.0031, row-norm ratio [0.9972, 1.0006]. PASS.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_c_sliding_attn_norm.py`

## S.sliding.01 test (attempt 1): swap attn_norm into the sliding block
- Reviewed the rendered swap test (layer 0, golden s4096 chunk 1). The gated metric is still `pcc_swap_out` >= 0.98 (spec block threshold).
- Gap: block-out PCC is weak for this swap. CPU numbers (block out PCC / rel L2): reference 0.999996 / 0.0027; `1 + w` 0.99959 / 0.0287;
  no weight 0.9804 / 0.197 (passes 0.98); 5% noise on attn_norm 0.99988 / 0.0154; x2 and sum-instead-of-mean are identical to the
  reference (the q/k/v norms absorb uniform scale, so they are harmless inside the block). Zero stub: 0.708.
- Added asserted extra checks (recorded as informational metrics `rel_l2_swap_out`, `rel_l2_swap_attn_norm`): each swapped float step's
  own output PCC >= component threshold (0.99) and rel L2 <= 0.03; block out rel L2 <= 0.01. The test now inlines `run_swap_test`
  (it returns only a bool). The gated metric name and threshold are unchanged.
- Verified: BRINGUP_IMPL=reference PASS (pcc 0.999996, rel 0.0027); BRINGUP_IMPL=stub FAIL (pcc 0.708). Device gate PASS:
  pcc_swap_out 0.999996, rel_l2_swap_out 0.0028, attn_norm pcc 0.999996 / rel 0.0031.
- Note: the first "pcc=0.000000" lines in the output come from run_safe_pytest's precompile pass (comp_pcc stub); ignore them.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_swap_sliding_01_attn_norm.py`

## C.sliding.attention test (attempt 1)
- Reviewed the rendered test. Golden: s4096 chunk 1 (start 2048), layer 0, `attn_norm` -> `attn_out` [2048, 2816] bf16, KV prefix
  [0, 2048) from the golden state (`device_ctx.extra["state_prefix"]`, `prefix_len`). PCC mode is right; threshold stays at 0.99.
- Gap: PCC barely sees stateful bugs. CPU (PCC / rel L2 whole / rel L2 first 1024 rows): reference 0.999998 / 0.0019 / 0.0019;
  no prefix 0.9979 / 0.064 / 0.090; RoPE positions from 0 0.9959 / 0.091 / 0.128; no window 0.9872 / 0.16; window 992 0.99998 / 0.0065
  (not caught, harmless); rotate-half swapped for interleaved rel 0.38; no v norm rel 16.8; `1 + w` in q/k norm PCC 0.876.
  Error budget: bf16 everywhere 0.0028, bf16 + bfp8 q/k/v/o weights 0.0055, 1% noise 0.010.
- Added asserted checks after the gated PCC: rel L2 <= 0.03 whole chunk and over the first `sliding_window` rows, plus finite output.
  Recorded as informational metrics `rel_l2_attention_L00`, `rel_l2_prefix_rows_attention_L00`. The test inlines the harness pipeline
  as the attn_norm test does. Also asserts the component chunk starts after 0.
- Not covered here: the K/V the step writes into the state (the device fn returns only attn_out); the state metrics cover it.
- Verified: BRINGUP_IMPL=reference PASS (pcc 0.999998, rel 0.0019 / 0.0019); BRINGUP_IMPL=stub FAIL (pcc 0.0). Device gate fails for
  now with `NotImplementedError: implement step: no device module for attention yet`.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_c_sliding_attention.py`

## C.sliding.attention implement (attempt 1)
- New `tt/attention.py`:
  - `TtSlidingAttention`, TP by head: chip c holds Q heads 4c..4c+3 and KV heads 2c, 2c+1. Fused per-chip QKV weight [2816, 2048] bf16
    (ERNIE layout), `nlp_create_qkv_heads(4, 2)`. Per-head `rms_norm` (reshape to [1,1,h*S,256]) with q_norm / k_norm, unscaled for V,
    HiFi4 + fp32 acc. `experimental.rotary_embedding` with host-built fp32->bf16 cos/sin tables for exactly [start, start+S).
    `fill_cache(update_idx=start)` for K and V. SDPA runs over the square [prev-window tail | chunk]: the tail is `ttnn.slice` of the cache at
    [start-hist, start) with hist = min(1024, start), and Q is front-padded with its own first hist rows (outputs dropped), as in
    gemma4/tt/attention/prefill.py. `scaled_dot_product_attention(is_causal, sliding_window_size=1024, scale 1.0)`, q256/k128, full grid,
    HiFi4 + fp32 acc. Then `nlp_concat_heads`, row-parallel o_proj ([1024, 2816] per chip) and `ttnn.all_reduce(cluster_axis=1)`.
  - `TtKVCacheSliding`: full-length [1, 8, max_seq, 256] bf16 sharded by head (dim 1), with `load_prefix` and `to_torch`.
- `hooks.py`: `DEVICE_STEPS["sliding"]` now has `attention`. `_attention_module` loads only the six self_attn tensors of a layer (not the
  experts). `device_component("attention")` builds a fresh device cache from `ctx.extra` (`state_prefix`, `prefix_len`, `max_seq`) on
  each call. In `HybridDeviceModel`, sliding layers keep K/V in a device `TtKVCacheSliding` held by `_HybridState.dev`.
  `load_prefix` and `to_torch` go to the device cache for those layers and to the CPU state for the others. `layer()` passes the cache to
  the override as `ctx.extra["dev_cache"]`. Global layers raise NotImplementedError from `_attention_module` and stay on the CPU.
- Gate: pcc_attention_L00 0.999988, rel_l2 0.0051 (whole chunk) and 0.0051 (first 1024 rows). PASS.
- tt-probe check of the hybrid `device_model` on layer 0, s4096 chunk 1: block out PCC 0.99997; the key/value state read back from the device
  scores PCC 0.99999 against the golden (whole state and the new chunk).
- Not done yet (perf): SDPA uses HiFi4 + fp32 acc, which turns off streaming SDPA (see known issues); ERNIE's preset A is the candidate.
  QKV and o_proj use HiFi2 + fp32 acc. Hidden states still go to the host between steps.
- Gotcha: tt-probe saves its scripts under tests/ttnn/unit_tests/operations/<name>/probes (outside the allowed paths). I deleted them.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_c_sliding_attention.py`

## S.sliding.02 test (attempt 1): swap attn_norm + attention into the sliding block
- Reviewed the rendered swap test (layer 0, golden s4096 chunk 1, start 2048). The gated metric stays `pcc_swap_out` >= 0.98.
- Gap: block-out PCC misses stateful attention bugs. CPU (block out PCC / rel L2 whole / first 1024 rows): reference 0.999996 / 0.0027 / 0.0033;
  no KV prefix 0.99897 / 0.045 / 0.064; zeroed prefix 0.99890 / 0.047 / 0.066; RoPE from 0 0.99776 / 0.067 / 0.094 (all pass 0.98);
  1% noise on attn 0.99998 / 0.0071; zero stub 0.604.
- Added asserted extra checks (informational metrics `rel_l2_swap_out`, `rel_l2_prefix_rows_swap_out`, `rel_l2_swap_<step output>`,
  `rel_l2_prefix_rows_swap_attn_out`): each swapped step's own output PCC >= 0.99 and rel L2 <= 0.03; attn_out rel L2 over the first
  `sliding_window` rows <= 0.03; block out rel L2 <= 0.02 (whole chunk and first window rows); finite outputs; chunk start > 0.
- Decision: block-out limit 0.02, not swap 1's 0.01. The device run scores 0.0074 (attn_out 0.0052 grows through router flips, router PCC 0.99969),
  and a faster SDPA preset later would leave no margin. The bugs still score more than 2x over the limit.
- Verified: BRINGUP_IMPL=reference PASS (rel 0.0027); stub FAIL (pcc 0.604). Device gate PASS: pcc_swap_out 0.999973, block out rel 0.0074 / 0.0062,
  attn_norm 0.999996 / 0.0031, attention 0.999988 / 0.0052 / first rows 0.0052.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_swap_sliding_02_attention.py`

## C.sliding.post_attn_norm test (attempt 1)
- Reviewed the rendered test (layer 0, golden [2048, 2816] attn_out -> attn_post_norm). The gated metric stays `pcc_post_attn_norm_L00` >= 0.99.
- Gap: the same one as attn_norm. PCC ignores scale. post_attention_layernorm w is in [0.014, 16.5]. On CPU (PCC / rel L2): `1 + w` 0.9954 / 0.135
  (passes PCC), sum instead of mean ~1.0 / 0.98, no weight 0.51 / 0.94, 1% noise ~1.0 / 0.010, bf16 in/out rel 0.0028.
- Decision: copied the asserted scale checks from test_c_sliding_attn_norm.py (rel L2 <= 0.03, per-token norm ratio in [0.97, 1.03], finite output;
  informational metrics `rel_l2_post_attn_norm_L00`, `row_norm_ratio_{min,max}_post_attn_norm_L00`). The reference ratio is [0.9954, 1.0042], so the margin is fine.
- Verified: BRINGUP_IMPL=reference PASS (pcc 0.999997, rel 0.0023); BRINGUP_IMPL=stub FAIL (pcc 0.0). The device gate fails for now with
  `NotImplementedError: implement step: no device module for post_attn_norm yet`.
- Gotcha: the reference run's log also prints a "FAIL pcc=0.000000" line from the precompile pass; the real run's line follows it.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_c_sliding_post_attn_norm.py`

## C.sliding.post_attn_norm implement (attempt 1)
- Reused the existing `tt/rms_norm.py` TtRMSNorm (ttnn.rms_norm, plain `w`, HiFi4 + fp32 acc, replicated [1, 1, S, 2816]). No new module.
- hooks.py: added `post_attn_norm -> post_attention_layernorm.weight` to `_NORM_WEIGHTS` (so `device_component` serves it) and
  `post_attn_norm` to `DEVICE_STEPS["sliding"]` (so the ladder's `device_model` swaps it in). Swap tests use `device_component` per step, so they are unaffected.
- Gate PASS: pcc_post_attn_norm_L00 0.999996, rel L2 0.0029, row-norm ratio [0.9941, 1.0052]. (The first "FAIL pcc=0.000000" line is the precompile pass.)
- Hidden states still round-trip through the host between steps.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_c_sliding_post_attn_norm.py`

## S.sliding.03 test (attempt 1): swap attn_norm + attention + post_attn_norm into the sliding block
- Reviewed the rendered swap test (layer 0, golden s4096 chunk 1, start 2048). The gated metric stays `pcc_swap_out` >= 0.98.
- Replaced the one-line template body with swap 2's checks (SWAPPED extended): step PCC >= 0.99 and rel L2 <= 0.03 per swapped step,
  attention first-window rows rel L2 <= 0.03, block out rel L2 <= 0.02 (whole chunk and first window rows), finite outputs, chunk start > 0.
- Added: for every swapped `norm` step, compare with the CPU norm run on the exact input the device step saw (rel L2 <= 0.03, per-token
  norm ratio in [0.97, 1.03]; metrics `rel_l2_iso_swap_<out>`, `row_norm_ratio_{min,max}_swap_<out>`). Why: post_attn_norm's input is the
  device attention output here, so a golden-based norm-ratio check would also count attention error. Scale bugs (`1 + w` rel 0.135, sum vs mean
  0.98, from the component test) already fail the step rel L2 check against the golden.
- Verified: BRINGUP_IMPL=reference PASS (pcc 0.999996, block rel 0.0027); stub FAIL (all checks). Device gate PASS: pcc_swap_out 0.999973,
  block out rel 0.0074 / 0.0060, attn_norm 0.0031 (iso 0.0020), attention 0.0052 / 0.0052, post_attn_norm 0.0052 (iso 0.0019, ratio [0.9958, 1.0024]).
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_swap_sliding_03_post_attn_norm.py`

## C.sliding.attn_residual test (attempt 1)
- Reviewed the rendered test (layer 0, golden [2048, 2816]: h_mid = in + attn_post_norm). The gated metric stays `pcc_attn_residual_L00` >= 0.99.
- Gap: PCC misses scale bugs and bad rows. On CPU vs golden (PCC / rel L2 / per-token norm ratio): reference 0.999997 / 0.0023 / [0.9958, 1.0041];
  bf16 add 0.999996 / 0.0028 / [0.9945, 1.0060]; `2 * (a + b)` 0.999997 / 1.0 / 2.0; row 0 zeroed 0.9998 / 0.020 / min 0; last 32 rows zeroed
  0.993 / 0.118; attn_post_norm only 0.979 / 0.205; `in` only 0.18.
- Decision: same asserted extras as the norm tests: rel L2 <= 0.03, per-token norm ratio in [0.97, 1.03], finite, element count matches
  (informational metrics `rel_l2_attn_residual_L00`, `row_norm_ratio_{min,max}_attn_residual_L00`).
- Verified: BRINGUP_IMPL=reference PASS (pcc 0.999997, rel 0.0023); BRINGUP_IMPL=stub FAIL (pcc 0.0). The device gate fails for now with
  `NotImplementedError: implement step: no device module for attn_residual yet`.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_c_sliding_attn_residual.py`

## C.sliding.attn_residual test (attempt 2)
- Attempt 1 failed the step because of a direct `python` call on device code, not because of the test. The test file is unchanged from attempt 1.
- Checked again with the safe runner only: BRINGUP_IMPL=reference PASS (pcc 0.999997, rel L2 0.0023, ratio [0.9958, 1.0041]);
  BRINGUP_IMPL=stub FAIL (pcc 0.0). The device gate fails as expected with `NotImplementedError: implement step: no device module for attn_residual yet`.
- Gotcha: run CPU-only exploration (e.g. mutation scoring) inside the test or through `scripts/tt-probe.sh`, never with `python -`, even with no device use.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_c_sliding_attn_residual.py`

## C.sliding.attn_residual test (run1 brief, attempt 1)
- The test file already had the review from the earlier attempts (rel L2 <= 0.03, per-token norm ratio in [0.97, 1.03], finite, element count). I left it unchanged.
- Re-checked with the safe runner only: BRINGUP_IMPL=reference PASS (pcc 0.999997, rel L2 0.002281, ratio [0.9958, 1.0041]); BRINGUP_IMPL=stub
  FAIL (pcc 0.0). The device gate fails as expected until the implement step: `NotImplementedError: implement step: no device module for attn_residual yet`.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_c_sliding_attn_residual.py`

## C.sliding.attn_residual implement (run1, attempt 1)
- Added `tt/residual.py:TtResidualAdd` (`ttnn.add` of two replicated [1, 1, S, H] bf16 TILE tensors, DRAM; optional scalar multiply for
  the later `ffn_residual` with `layer_scalar`). No collective: both operands are replicated.
- hooks.py: `_RESIDUAL_STEPS = {"attn_residual"}`, `_residual_host_fn` (host a, b -> device -> add -> chip 0 copy to host);
  `device_component` returns it for residual steps; `attn_residual` added to `DEVICE_STEPS["sliding"]`, so `HybridDeviceModel` (ladder) swaps it too.
- Gate PASS: pcc_attn_residual_L00 0.999996, rel L2 0.00282, per-token norm ratio [0.9947, 1.0060] (matches the bf16-add estimate in the test review).
- Gotcha: the log shows a `FAIL pcc=0.000000` line first; that is the precompile collect pass (stubbed ops), the real pass line follows.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_c_sliding_attn_residual.py`

## S.sliding.04 test (run1, attempt 1): swap attn_norm + attention + post_attn_norm + attn_residual into the sliding block
- Reviewed the rendered swap test (layer 0, golden s4096 chunk 1, start 2048). The gated metric stays `pcc_swap_out` >= 0.98.
- Replaced the one-line template body with swap 3's checks (SWAPPED extended). Changed one thing: the check against the CPU step on the device step's own inputs
  now covers `residual` steps as well as `norm` steps (`ISO_STEP_KINDS`). It asserts rel L2 <= 0.03 and per-token norm ratio in [0.97, 1.03]. Reason: PCC misses `2 * (a + b)`
  and zeroed rows (test_c_sliding_attn_residual.py). Checking against the CPU add on the same inputs keeps upstream attention error out of the result.
- Verified: BRINGUP_IMPL=reference PASS (pcc 0.999996, block rel 0.0027); stub FAIL (all checks). Device gate PASS: pcc_swap_out 0.999969,
  block out rel 0.0079 / 0.0064, attention 0.0052 / 0.0052, post_attn_norm 0.0052 (iso 0.0019), attn_residual 0.0054 (iso 0.0017, ratio [0.9977, 1.0031]).
- Block-out rel 0.0079 is within the 0.02 limit, but it creeps up with each added device step (0.0074 at swap 3). Watch the margin as more steps move to the device.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_swap_sliding_04_attn_residual.py`

## C.sliding.ffn_norm test (run1, attempt 1)
- Replaced the one-line template body with the attn_norm/post_attn_norm norm checks: PCC >= 0.99 (gated), plus asserted rel L2 <= 0.03,
  per-token output-norm ratio in [0.97, 1.03], finite output (informational metrics `rel_l2_ffn_norm_L00`, `row_norm_ratio_{min,max}_ffn_norm_L00`).
- Scored mutations on this golden (h_mid -> ffn_norm, [2048, 2816], pre_feedforward_layernorm w in [0.011, 22.9]) once, inside the test under
  BRINGUP_IMPL=reference, then removed the scoring code: reference PCC 0.999997 / rel 0.0024 / ratio [0.9975, 1.0024]; bf16 rel 0.0028; 1% noise 0.010;
  `1 + w` PCC 0.966 (fails here); no weight PCC 0.838; sum instead of mean PCC ~1.0 but rel 0.98. PCC alone misses the sum bug, and the rel L2 check catches it.
- Verified: BRINGUP_IMPL=reference PASS; BRINGUP_IMPL=stub FAIL (pcc 0.0). The device gate fails as expected until the implement step:
  `NotImplementedError: implement step: no device module for ffn_norm yet`.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_c_sliding_ffn_norm.py`

## C.sliding.ffn_norm implement (run1, attempt 1)
- No new module: reused `tt/rms_norm.py:TtRMSNorm` (ttnn.rms_norm, HiFi4 + fp32 acc, TILE [1,1,1,H] bf16 gamma, plain `w`).
- hooks.py: `_NORM_WEIGHTS["ffn_norm"] = "pre_feedforward_layernorm.weight"` (so `device_component` handles it), and `ffn_norm` added to
  `DEVICE_STEPS["sliding"]`, so `HybridDeviceModel` (ladder) swaps it too.
- Gate PASS: pcc_ffn_norm_L00 0.999996, rel L2 0.00294, per-token norm ratio [0.9953, 1.0015]. bf16 gamma is fine for w up to 22.9.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_c_sliding_ffn_norm.py`

## S.sliding.05 test (run1, attempt 1): swap attn_norm .. ffn_norm into the sliding block
- Replaced the one-line template body with swap 4's checks, SWAPPED extended with `ffn_norm`. No new check was needed: ffn_norm is a `norm` step, so the existing
  check against the CPU norm on the device step's own inputs already covers it (rel L2 <= 0.03, ratio in [0.97, 1.03]; catches `1 + w` and sum-instead-of-mean).
  The gated metric stays `pcc_swap_out` >= 0.98.
- Verified: BRINGUP_IMPL=reference PASS (pcc 0.999996, block rel 0.0027); stub FAIL (every check). Device gate PASS: pcc_swap_out 0.999969, block out rel
  0.0079 / 0.0064 (same as swap 4), ffn_norm 0.0060 vs golden (iso 0.0019, ratio [0.9971, 1.0013]).
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_swap_sliding_05_ffn_norm.py`

## C.sliding.mlp test (run1, attempt 1)
- Replaced the one-line template body with the ffn_norm-style checks: PCC >= 0.99 (gated), plus asserted element count, finite output, rel L2 <= 0.03,
  per-token output-norm ratio in [0.97, 1.03] (informational metrics `rel_l2_mlp_L00`, `row_norm_ratio_{min,max}_mlp_L00`).
- Scored mutations on this golden (ffn_norm -> mlp_out, [2048, 2816], row norms 385-4579) once, inside the test under BRINGUP_IMPL=reference, then
  removed the scoring code. PCC / rel: reference 0.999998 / 0.0018; emulated bfp8 weights 0.999985 / 0.0056; bfp8 weights and activations 0.999976 / 0.0069
  (ratio [0.992, 1.010]); silu instead of gelu_tanh 0.9986 (passes PCC) / 0.055; last row zeroed 0.9996 / 0.027, caught by the ratio (min 0); 2x caught by rel.
  Exact gelu vs tanh gelu makes no measurable difference, so the test can't tell them apart.
- Verified: BRINGUP_IMPL=reference PASS (pcc 0.999998, rel 0.0018, ratio [0.9979, 1.0023]); BRINGUP_IMPL=stub FAIL (pcc 0.0). The device gate fails as
  expected until the implement step: `NotImplementedError: implement step: no device module for mlp yet`.
- The implementation has a lot of margin: bfp8 weights should land near rel 0.006, and the limit is 0.03.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_c_sliding_mlp.py`

## C.sliding.mlp implement (run1, attempt 1)
- Added `tt/mlp.py:TtDenseMLP`: fused column-parallel gate/up matmul (per chip `[up_c | gate_c]`, [H, 2 * 544]), `ttnn.slice` into up/gate,
  `gemma4/tt/experts/operations.py:apply_geglu` (Accurate gelu), row-parallel down [544, H], `ttnn.all_reduce(cluster_axis=1)`. The output is replicated,
  and the all-reduce comes before post_mlp_norm. 2112 / 4 = 528 is padded to 544 per chip: zero gate/up columns and zero down rows, so the pad contributes exactly 0.
  Weights bf16 (plan.yaml).
- hooks.py: `_mlp_module` (loads only `layers.<i>.mlp.{gate,up,down}_proj.weight`). `device_component` returns `_host_fn(mlp)` for step `mlp`,
  `mlp` is in `DEVICE_STEPS["sliding"]`, and `HybridDeviceModel` swaps it in.
- Decision: HiFi4 + fp32 acc matmuls. HiFi2 passed (PCC 0.999993) but gave rel L2 0.0083 and a per-token norm ratio of [0.9863, 0.9975], always < 1,
  which is a systematic shrink. HiFi4 gives rel L2 0.00327 and ratio [0.9973, 1.0047]. Added a known-issues proposal for this.
- The attention (`tt/attention.py`) still uses HiFi2 for QKV and o_proj. Its rel L2 of 0.0052 may partly come from the same bias; this is a possible cheap win if the block-out margin gets tight.
- Gate PASS (HiFi4): pcc_mlp_L00 0.999995, rel L2 0.00327, ratio [0.9973, 1.0047].
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_c_sliding_mlp.py`

## S.sliding.06 test (run1, attempt 1): swap attn_norm .. mlp into the sliding block
- Replaced the one-line template body with swap 5's checks. SWAPPED is extended with `mlp`, and `mlp` is added to `ISO_STEP_KINDS`, so the device mlp is checked
  against the CPU mlp on the same device ffn_norm input (rel L2 <= 0.03, per-token norm ratio in [0.97, 1.03]). Reason: post_mlp_norm (CPU) comes next and undoes
  any per-row scale on mlp_out, so block out can't see a scaled or zeroed-row MLP. PCC also misses silu-for-gelu_tanh (test_c_sliding_mlp.py).
  The gated metric stays `pcc_swap_out` >= 0.98.
- Verified: BRINGUP_IMPL=reference PASS (block rel 0.0027); stub FAIL (every check). Device gate PASS: pcc_swap_out 0.999968, block out rel 0.0080 / 0.0066
  (0.0079 at swap 5), mlp 0.0048 vs golden (iso 0.0028, ratio [0.9975, 1.0039]).
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_swap_sliding_06_mlp.py`

## C.sliding.post_mlp_norm test (run1, attempt 1)
- Replaced the one-line template body with the ffn_norm/post_attn_norm norm checks: PCC >= 0.99 (gated), plus asserted finite output, rel L2 <= 0.03 and
  per-token output-norm ratio in [0.97, 1.03] (informational metrics `rel_l2_post_mlp_norm_L00`, `row_norm_ratio_{min,max}_post_mlp_norm_L00`).
- Scored mutations on this golden once (mlp_out -> mlp_post_norm, [2048, 2816]) inside the test under BRINGUP_IMPL=reference, then removed the scoring code.
  The recovered post_feedforward_layernorm_1 weight spans about [-2.4, 135], a wider range than the earlier norms. PCC / rel: reference 0.999997 / 0.0023
  (ratio [0.9971, 1.0033]); bf16 in/out 0.999995 / 0.0033; 1% noise 0.010; `1 + w` 0.9984 (passes PCC) / 0.059; no weight 0.27; sum instead of mean ~1.0 / 0.98;
  last row zeroed 0.9997 / 0.025, caught by the ratio (min 0). No new check was needed.
- Verified: BRINGUP_IMPL=reference PASS (pcc 0.999997, rel 0.0023); BRINGUP_IMPL=stub FAIL (pcc 0.0). The device gate fails as expected until the implement step:
  `NotImplementedError: implement step: no device module for post_mlp_norm yet`.
- For the implement step: reuse `tt/rms_norm.py:TtRMSNorm` through `_NORM_WEIGHTS["post_mlp_norm"] = "post_feedforward_layernorm_1.weight"` (the `_1` one; plain `post_feedforward_layernorm` is the later ffn_out norm, `_2` the MoE one). bf16 gamma up to 135
  adds only about 0.001 rel (bf16 emulation 0.0033).
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_c_sliding_post_mlp_norm.py`

## C.sliding.post_mlp_norm implement (run1, attempt 1)
- No new module. `tt/rms_norm.py:TtRMSNorm` is reused through `_NORM_WEIGHTS["post_mlp_norm"] = "post_feedforward_layernorm_1.weight"` in hooks.py, and
  `post_mlp_norm` is added to `DEVICE_STEPS["sliding"]`, so `HybridDeviceModel` swaps it in for the ladder.
- Gate PASS: pcc_post_mlp_norm_L00 0.999996, rel L2 0.00294, ratio [0.9950, 1.0033]. The log also has a `FAIL ... pcc=0.000000` line. It comes from the
  precompile collect pass (stubbed outputs), not from the real pass.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_c_sliding_post_mlp_norm.py`

## S.sliding.07 test (run1, attempt 1): swap attn_norm .. post_mlp_norm into the sliding block
- Replaced the one-line template body with swap 6's checks and added `post_mlp_norm` to SWAPPED. No new check was needed. post_mlp_norm is a `norm` step, so
  the existing check against the CPU norm on the device step's own inputs covers it (rel L2 <= 0.03, ratio in [0.97, 1.03]). That check catches `1 + w` and a
  zeroed row, which PCC misses, and which ffn_combine followed by the CPU post_ffn_norm would partly hide at block out. The gated metric stays `pcc_swap_out` >= 0.98.
- Verified: BRINGUP_IMPL=reference PASS (pcc 0.999996, block rel 0.0027); stub FAIL (every check). Device gate PASS: pcc_swap_out 0.999967, block out rel
  0.0081 / 0.0067 (0.0080 at swap 6), post_mlp_norm 0.0041 vs golden (iso 0.0019, ratio [0.9971, 1.0014]).
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_swap_sliding_07_post_mlp_norm.py`

## C.sliding.router test (run1, attempt 1)
- The golden `router` is the dense [S, 128] bf16 routing matrix, with exactly 8 nonzeros per row and row sums in [0.987, 1.011] (per_expert_scale is in [0.980, 1.023]).
  PCC stays the gated metric. The test also asserts: a finite output with the right element count, exactly 8 nonzeros per row, non-negative weights, mean top-8 selection
  overlap >= 0.995, weight rel L2 <= 0.005 on rows whose selected set matches the golden, and a per-row sum ratio in [0.99, 1.01]. The informational metrics are
  `selection_overlap_router_L00`, `matched_rel_l2_router_L00`, `row_sum_ratio_{min,max}_router_L00`.
- Scored mutations once in a temporary test under BRINGUP_IMPL=reference, then removed that code (numbers are in the test docstring). Even the CPU reference
  does not match the golden exactly: 6 of 2048 rows pick a different expert (near ties from the bf16 golden input), so overlap is 0.99963. bf16 h plus bf16 proj weight
  gives overlap 0.99921 and matched rel 0.0021. PCC alone misses: no per_expert_scale, per_expert_scale by rank, top-7/9, zeroed rows, and 1-3% logit noise.
- Known gap: renormalizing after per_expert_scale instead of before scores matched rel 0.0048 (passes). Only the row-sum check catches it ([0.9887, 1.0129]).
- For implement: keep h and the proj in fp32 (plan.yaml) to keep near-tie flips down. The overlap threshold allows about 80 flipped selections in 16384.
  The output must have exact zeros off the top-8 (scatter into zeros), not small values from a mask multiply.
- Verified: BRINGUP_IMPL=reference PASS (pcc 0.999965); BRINGUP_IMPL=stub FAIL (pcc 0.0). The device gate fails as expected until the implement step:
  `NotImplementedError: implement step: no device module for router yet`.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_c_sliding_router.py`

## C.sliding.router implement (run1, attempt 1)
- Added `tt/router.py:TtRouter`, replicated, all fp32 up to the scatter. Steps: fp32 `rms_norm` (no weight), `linear` with W' = (proj * router.scale * 2816^-0.5)^T
  (scale folded into the weight on the host, same math), `softmax(numeric_stable)`, fp32 `topk(8)`, then `gather`. The weights are
  gather(probs * per_expert_scale, idx) / sum(gather(probs, idx)), which equals renorm-then-per_expert_scale[id] without a per-row index lookup.
  The bf16 `scatter` goes into zeros, so off-top-8 entries are exactly 0 (ttnn.scatter has no fp32 TILE path). HiFi4 + fp32 acc throughout.
  `__call__` returns `(dense [1,1,S,128] bf16, idx, weights fp32)`, and the experts step should consume idx/weights.
- hooks.py: added `_router_module`, which loads only `router.{proj.weight, scale, per_expert_scale}`, and `_router_host_fn`, which uploads h_mid as **fp32**
  (not `to_device_replicated`, which rounds to bf16) and returns the dense matrix. `router` is in `DEVICE_STEPS["sliding"]`, `device_component` handles it,
  and `HybridDeviceModel` swaps it in.
- Gate PASS: pcc_router_L00 0.999815; nnz 8/row; selection overlap 0.99841 (CPU ref 0.99963, limit 0.995); matched rows 2022/2048; matched rel L2 0.00267;
  row-sum ratio [0.9956, 1.0028].
- The selection overlap is below the CPU reference (about 26 rows differ vs 6). The likely cause is fp32 matmul/softmax on device (the TILE fp32 matmul is not
  bit-exact fp32). There is a lot of margin, and I did not tune it further.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_c_sliding_router.py`

## S.sliding.08 test (run1, attempt 1): swap attn_norm .. router into the sliding block
- Replaced the one-line template body with swap 7's checks and added `router` to SWAPPED. The router step (kind `router`) is exempt from the generic
  step rel L2 <= 0.03: on device its whole-matrix rel L2 is 0.027, most of it near-tie selection flips caused by upstream h_mid error. It gets router checks
  instead: exactly 8 nonzeros per row, no negative weights, and, vs golden / vs the CPU router on the same device h_mid (iso), selection overlap >= 0.99 / 0.995,
  matched-row rel L2 <= 0.01 / 0.005, row-sum ratio in [0.99, 1.01]. The iso limits match the component test. The golden matched-rel limit is 0.01 because the
  device scores 0.0044 vs golden, close to 0.005. The per_expert_scale bugs (0.0105, 0.0168 in test_c_sliding_router.py) are still caught by the iso check.
  The gated metric stays `pcc_swap_out` >= 0.98.
- Verified: BRINGUP_IMPL=reference PASS (router overlap 0.99969 golden / 1.0 iso); stub FAIL (every check). Device gate PASS: pcc_swap_out 0.999967,
  block out rel 0.0081 / 0.0070; router pcc 0.99961, overlap 0.99701 / 0.99854, matched rel 0.00442 / 0.00224, row sums [0.9958, 1.0044] / [0.9978, 1.0024].
- Block out did not change from swap 7 (0.008135). The dense router output feeds the CPU experts, and the flips are near ties, so their effect is small.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_swap_sliding_08_router.py`

## C.sliding.moe_norm test (run1, attempt 1)
- Replaced the one-line template body with the reviewed norm-test body (same as test_c_sliding_post_mlp_norm.py / ffn_norm). PCC >= 0.99 stays the gated
  metric. Asserted extras: finite output, rel L2 <= 0.03, per-token norm ratio in [0.97, 1.03]. Input h_mid, output moe_norm (pre_feedforward_layernorm_2).
- Scored mutations once in a temporary block under BRINGUP_IMPL=reference, then removed it. The numbers are in the docstring. The weight is small (w in [-0.0125, 2.27]),
  so `1 + w`, no weight and the wrong norm weight all already fail PCC (0.46-0.64). PCC misses sum-instead-of-mean (PCC ~1.0, rel 0.98) and zeroed rows
  (last row 0.9996, last 32 rows 0.992). The rel L2 and ratio checks catch both. The CPU reference scores rel 0.0025, ratio [0.9976, 1.0025], and bf16 in/out rel 0.0029.
- Verified: BRINGUP_IMPL=reference PASS (pcc 0.999997); stub FAIL (pcc 0.0). The device gate fails as expected until the implement step:
  `NotImplementedError: implement step: no device module for moe_norm yet`.
- For implement: reuse `tt/rms_norm.py:TtRMSNorm` with `_NORM_WEIGHTS["moe_norm"] = "pre_feedforward_layernorm_2.weight"` and add `moe_norm` to `DEVICE_STEPS["sliding"]`.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_c_sliding_moe_norm.py`

## C.sliding.moe_norm implement (run1, attempt 1)
- No new module. hooks.py maps `_NORM_WEIGHTS["moe_norm"] = "pre_feedforward_layernorm_2.weight"`, so it reuses `tt/rms_norm.py:TtRMSNorm` (replicated, `x * w`, eps 1e-6),
  and `moe_norm` is now in `DEVICE_STEPS["sliding"]`, so `device_component` and `HybridDeviceModel` pick it up through the existing norm path.
- Gate PASS: pcc_moe_norm_L00 0.999996, rel_l2 0.00307, row_norm_ratio [0.9965, 1.0021].
- The log also has a `FAIL pcc_moe_norm_L00: pcc=0.000000` line. It comes from the precompile collect pass (the plugin stubs comp_pcc), not from the real run.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_c_sliding_moe_norm.py`

## S.sliding.09 test (run1, attempt 1): swap attn_norm .. moe_norm into the sliding block
- Replaced the one-line template body with swap 8's checks and added `moe_norm` to SWAPPED. No new check was needed. moe_norm is a `norm` step
  (h_mid -> pre_feedforward_layernorm_2), so it gets the existing checks: its own output vs golden (PCC >= 0.99, rel L2 <= 0.03) and vs the CPU norm on the same
  device h_mid (rel L2 <= 0.03, per-token norm ratio in [0.97, 1.03]). These catch sum-instead-of-mean and zeroed rows, which PCC misses. The gated metric stays `pcc_swap_out` >= 0.98.
- Verified: BRINGUP_IMPL=reference PASS (pcc 0.999996, block rel 0.0027); stub FAIL (every check). Device gate PASS (moe_norm is already implemented):
  pcc_swap_out 0.999967, block out rel 0.0081 / 0.0070 (no change from swap 8), moe_norm pcc 0.99998, rel 0.0065 vs golden, iso 0.0019, ratio [0.9984, 1.0002].
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_swap_sliding_09_moe_norm.py`

## C.sliding.experts test (run1, attempt 1)
- Replaced the one-line template body with the reviewed body. Inputs are moe_norm [2048, 2816] and the dense router [2048, 128] (bf16 golden); the output is experts_out.
  PCC >= 0.99 stays the gated metric. Asserted extras: finite output, rel L2 <= 0.03, per-token norm ratio in [0.97, 1.03], and a new check,
  worst per-token rel L2 <= 0.1, which catches a single dropped (token, expert) pair.
- Scored mutations once in a temporary test under BRINGUP_IMPL=reference, then deleted it. The numbers are in the docstring. PCC misses a dropped
  expert (0.9973), a dropped pair and zeroed rows, and 2x. The extra checks catch all of them.
- Expected device quality from emulated bfp8 activations and weights (the fused kernel packs to bfp8): rel 0.0113, ratio [0.990, 1.011], worst row 0.020.
  So there is margin, but HiFi2 norm shrink (known issue) would eat into the ratio budget.
- Routing is very skewed on this golden: per-expert token counts are 0..1255, and expert 47 takes 1255 of 2048 tokens. For implement: size the dispatch
  buffers for the worst case (every token to one expert), not the mean of 128. A capacity of 512 fails (PCC 0.978). Several experts get 0 tokens.
- Known gap: renormalizing the routing weights inside the module (dropping per_expert_scale) passes (rel 0.0051). The router test owns that bug.
- Verified: BRINGUP_IMPL=reference PASS (pcc 0.999997, rel 0.0023, ratio [0.9963, 1.0034], worst row 0.0043); stub FAIL (pcc 0.0). The device gate fails as
  expected until the implement step: `NotImplementedError: implement step: no device module for experts yet`.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_c_sliding_experts.py`

## C.sliding.experts implement (run1, attempt 1): gate NOT verified (board went down)
- ttnn (needs `./build_metal.sh`, done): `RoutedExpertActivation::GeluTanh = 4` (types.hpp, nanobind). The program factory sets the compute define
  `ROUTED_GELU_TANH`. The name `GELU_TANH` collides with the `KernelActivation::GELU_TANH` enumerator and breaks the trisc build.
  fused_swiglu.cpp: the unary gate activation is now the macros `GATE_ACT_INIT/GATE_ACT_TILE` = `gelu_tanh_tile_init/gelu_tanh_tile`
  under ROUTED_GELU_TANH, else silu. It uses the silu path (gate on dst, bf8 gate_intermed, then multiply_phase).
- ttnn accuracy hack, GeluTanh only (the other variants are unchanged): the factory honours compute_kernel_config.math_fidelity and fp32_dest_acc_en.
  Everything else stays hard-coded LoFi + bf16 dst. compute_kernel_config was added to the program-cache key (attribute_values).
- `tt/experts.py:TtExperts`: an adaptation of ernie `moe_unified.routed_partial`. It takes dense routing [1,1,S,128] and runs `ttnn.topk(8)` on device, then
  masked_bincount + offset_cumsum, dispatch (capacity factor 8 = top_k, the worst case with all 8 experts on one chip), TtRoutedExpert(GeluTanh,
  HiFi2, fp32 dest, bf16 weights, per-expert cap max_tokens = the largest chunk in the spec, 8192), combine(init_zeros), reduce, then `all_reduce(cluster_axis=1)`.
  The weight cache is `generated/gemma4_a4b_d_p/tt_cache/experts`. hooks.py: `_experts_module`, `_experts_host_fn`, `experts` in DEVICE_STEPS["sliding"],
  device_component, HybridDeviceModel.
- Measured on the gate golden: first attempt (LoFi, bf16 dst): pcc 0.99970, rel 0.045, ratio [0.927, 1.007] (FAIL rel/ratio). HiFi4 + bf16 dst:
  pcc 0.99981, rel 0.039, ratio [0.967, 1.052] (FAIL). Both are uniform output scales (per-column scale 1.040 +- 0.002), not dropped work.
  One-hot routing to expert 47 vs fp32 CPU: LoFi 0.96 (Silu 0.96 too), HiFi2/4 bf16 dst 1.036 (bfp8 weights 1.034, TILE-input path 1.033),
  **HiFi2 fp32 dst 1.011, rel 0.022**, HiFi4 fp32 dst 1.014. A CPU emulation of the kernel's bf8 roundings gives 1.000 / rel 0.017.
- The final config (HiFi2 + fp32 dest) has NOT been run through the gate. The `--dev` debug run that measured fp32 dest passed, but at teardown
  ethernet core 31-25 stopped responding. Every later run fails at mesh open (all 4 chips), and run_safe_pytest does not reset after a
  failed open. **The board needs a reset (`tt-smi -r` by a person), then re-run the gate.** If the ratio check still fails, try HiFi4 fp32 dest
  (the remaining +1% overshoot is probably from the bf16 partials / bf8 intermediates).
- Also note: a host-only analysis script that imported ttnn opened the UMD driver twice (00:40). Later device runs worked, so it is not the cause.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_c_sliding_experts.py`

## C.sliding.experts implement (run1, attempt 2): blocked, the board is still down
- No code changes. The attempt-1 implementation (GeluTanh kernel variant, HiFi2 + fp32 dest, `tt/experts.py:TtExperts`, hooks) is unchanged and still not verified on the gate.
- I ran the gate at 01:02. It failed again at mesh open on all 4 chips ("Timed out while waiting for active ethernet core 31-25"), and the test never ran.
  The wrapper resets only when its dirty flag is set by a hang, so it did not reset after the failed open. Agents may not run `tt-smi -r`, and I did not touch the dirty flag to force one.
- **A person must reset the board (`tt-smi -r`), then re-run the gate.** If the ratio check fails after that, the next thing to try is HiFi4 + fp32 dest in `tt/experts.py`.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_c_sliding_experts.py`

## S.sliding.10 test (run1, attempt 1): swap attn_norm .. experts into the sliding block
- Replaced the one-line template body with swap 9's checks, added `experts` to SWAPPED, and added a `moe` branch for experts. post_moe_norm runs on the CPU
  right after experts and hides per-row scale errors, and upstream router flips move whole rows vs golden. So experts is checked vs golden with PCC >= 0.99
  and rel L2 <= 0.05, and vs the CPU experts on the same device moe_norm and router with rel L2 <= 0.03, norm ratio in [0.97, 1.03], and worst per-token rel L2 <= 0.1
  (the component limits). The gated metric stays `pcc_swap_out` >= 0.98.
- Why 0.05 vs golden: the device scores 0.0266 (too close to 0.03). Most of that comes from router flips (45 rows mismatched vs golden, 16 vs iso). The iso check
  isolates the experts module (device 0.0183). A dropped expert still scores >= 0.07 vs golden.
- Verified: BRINGUP_IMPL=reference PASS (pcc 0.999996, experts rel 0.0031 / iso 0.0); stub FAIL (every check). Device gate PASS: pcc_swap_out 0.999954,
  block out rel 0.0096 / 0.0087 (limit 0.02), experts pcc 0.99969, rel 0.0266 vs golden, iso rel 0.0183, ratio [0.9905, 1.0203], worst row 0.0273.
- So the experts implementation (HiFi2 + fp32 dest GeluTanh kernel) now works on device. The board is back up after the ethernet-core outage.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_swap_sliding_10_experts.py`

## C.sliding.post_moe_norm test (run1, attempt 1)
- Replaced the one-line template body with the post_mlp_norm test's checks (STEP = post_moe_norm): the gated metric stays `pcc_post_moe_norm_L00` >= 0.99, and it also
  asserts rel L2 <= 0.03 and a per-token norm ratio in [0.97, 1.03]. These are recorded as informational metrics `rel_l2_*` and `row_norm_ratio_{min,max}_*`.
- Why: on this golden (experts_out -> moe_post_norm, w in [-3.7, 89.5]) PCC passes `1 + w` (0.9969), sum instead of mean (~1.0), 2x (~1.0), a zeroed last row (0.9998)
  and the last 32 rows zeroed (0.9916). rel L2 / ratio catch all of them (0.099 / 0.98 / 1.0 / ratio min 0 / 0.13). CPU reference: rel 0.0024, ratio [0.9981, 1.0017].
  bf16 in/out: rel 0.0028.
- Verified: BRINGUP_IMPL=reference PASS (pcc 0.999997, rel 0.0024); stub FAIL (PCC). The gate fails with NotImplementedError (there is no device module yet; that is the implement step).
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_c_sliding_post_moe_norm.py`

## C.sliding.post_moe_norm implement (run1, attempt 1)
- Reused `tt/rms_norm.py:TtRMSNorm` (replicated, HiFi4 + fp32 acc, plain `x * w`). The only change is in hooks.py: `"post_moe_norm": "post_feedforward_layernorm_2.weight"`
  in `_NORM_WEIGHTS` (so device_component and HybridDeviceModel pick it up), and `post_moe_norm` in DEVICE_STEPS["sliding"].
- Gate PASS: pcc_post_moe_norm_L00 0.999996, rel L2 0.0030, row norm ratio [0.9970, 1.0015]. The `FAIL ... pcc=0.000000` line in the log comes from the precompile
  collect pass (the comp_pcc stub), not from the real pass.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_c_sliding_post_moe_norm.py`

## S.sliding.11 test (run1, attempt 1): swap attn_norm .. post_moe_norm into the sliding block
- Replaced the one-line template body with swap 10's checks, and added `post_moe_norm` to SWAPPED. post_moe_norm is a `norm` step, so it gets the existing norm checks:
  vs golden PCC >= 0.99 and rel L2 <= 0.03, and vs the CPU norm on the same device experts_out rel L2 <= 0.03 and a per-token norm ratio in [0.97, 1.03]. No new limits.
  The gated metric stays `pcc_swap_out` >= 0.98.
- Why no looser golden limit here: post_moe_norm vs golden scores 0.0195 on device. That is mostly the experts error carried through (experts 0.0266 vs golden), and it leaves margin under 0.03.
  The iso check (device 0.0019) is the one that catches `1 + w`, sum instead of mean and zeroed rows. ffn_combine + post_ffn_norm partly hide those at block out.
- Verified: BRINGUP_IMPL=reference PASS (pcc 0.999996, post_moe_norm rel 0.0043 / iso 0.0); stub FAIL (every check). Device gate PASS: pcc_swap_out 0.999953,
  block out rel 0.0096 / 0.0087 (limit 0.02), post_moe_norm pcc 0.99981, rel 0.0195, iso 0.0019, ratio [0.9975, 1.0009]; experts iso 0.0183, ratio [0.9905, 1.0203].
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_swap_sliding_11_post_moe_norm.py`

## C.sliding.ffn_combine test (run1, attempt 1)
- Replaced the one-line template body with the attn_residual test's checks (STEP = ffn_combine, ffn_sum = mlp_post_norm + moe_post_norm). The gated metric stays
  `pcc_ffn_combine_L00` >= 0.99. The test also asserts rel L2 <= 0.03, a per-token norm ratio in [0.97, 1.03] and a finite output, and records `rel_l2_*` and
  `row_norm_ratio_{min,max}_*` as informational metrics.
- Why: on this golden PCC passes 2x (0.999999), 0.5x (0.999999), row 0 zeroed (0.99984), the last row zeroed (0.99969) and the last 32 rows zeroed (0.9917).
  rel L2 / ratio catch all of them. The last row zeroed scores rel 0.025, so only the ratio check (min 0) catches it. CPU reference: rel 0.0022, ratio [0.9977, 1.0020];
  a bf16 add scores rel 0.0027. Dropping an operand already fails PCC (mlp only 0.918, moe only 0.741).
- Verified: BRINGUP_IMPL=reference PASS (pcc 0.999998, rel 0.0022); stub FAIL (PCC). The gate fails with NotImplementedError because there is no device module yet
  (that is the implement step). `tt/` may already hold an add module to reuse from attn_residual.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_c_sliding_ffn_combine.py`

## C.sliding.ffn_combine implement (run1, attempt 1)
- Reused `tt/residual.py:TtResidualAdd` (replicated `ttnn.add`, no collective), the same module as attn_residual. The only change is in hooks.py: `ffn_combine` added to
  `_RESIDUAL_STEPS` (so device_component and HybridDeviceModel route it through `_residual_host_fn`), and to DEVICE_STEPS["sliding"].
- Gate PASS: pcc_ffn_combine_L00 0.999997, rel L2 0.0027, row norm ratio [0.9973, 1.0039]. The `FAIL ... pcc=0.000000` line comes from the precompile collect pass
  (comp_pcc stub), not the real pass.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_c_sliding_ffn_combine.py`

## S.sliding.12 test (run1, attempt 1): swap attn_norm .. ffn_combine into the sliding block
- Replaced the one-line template body with swap 11's checks, and added `ffn_combine` to SWAPPED. ffn_combine (ffn_sum = mlp_post_norm + moe_post_norm) is a `residual` step,
  so it gets the existing residual checks: vs golden PCC >= 0.99 and rel L2 <= 0.03, and vs the CPU add on the same device inputs rel L2 <= 0.03 and a per-token norm
  ratio in [0.97, 1.03]. No new limits. The gated metric stays `pcc_swap_out` >= 0.98.
- Why the iso check matters here: post_ffn_norm (CPU) normalizes every row right after ffn_combine, which hides a 2x / 0.5x scale or zeroed rows at block out.
- Verified: BRINGUP_IMPL=reference PASS (pcc 0.999996, ffn_combine rel 0.0024 / iso 0.0); stub FAIL (every check). Device gate PASS: pcc_swap_out 0.999953,
  block out rel 0.0097 / 0.0088 (limit 0.02), ffn_combine pcc 0.99996, rel 0.0093 vs golden, iso 0.0018, ratio [0.9992, 1.0022].
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_swap_sliding_12_ffn_combine.py`

## C.sliding.post_ffn_norm test (run1, attempt 1)
- Replaced the one-line template body with the post_moe_norm test's checks (STEP = post_ffn_norm, ffn_out = rms(ffn_sum) * post_feedforward_layernorm.weight).
  The gated metric stays `pcc_post_ffn_norm_L00` >= 0.99. The test also asserts rel L2 <= 0.03, a per-token norm ratio in [0.97, 1.03] and a finite output. It records
  `rel_l2_*` and `row_norm_ratio_{min,max}_*` as informational metrics.
- Why: on this golden (w in [0.02, 23]) PCC passes `1 + w` (0.99988), sum instead of mean, 2x, layer_scalar folded into the norm (all ~1.0), the last row zeroed (0.9997)
  and the last 32 rows zeroed (0.9922). rel L2 / ratio catch all of them (`1 + w` rel 0.047, ratio >= 1.042). CPU reference: rel 0.0023, ratio [0.9978, 1.0017];
  bf16 in/out: rel 0.0027. layer_scalar belongs to ffn_residual, not to this step.
- Verified: BRINGUP_IMPL=reference PASS (pcc 0.999997, rel 0.0023); stub FAIL (PCC). The gate fails with NotImplementedError because there is no device module yet
  (that is the implement step). The norm module used for the other post_*_norm steps should be reusable.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_c_sliding_post_ffn_norm.py`

## C.sliding.post_ffn_norm implement (run1, attempt 1)
- Reused `tt/rms_norm.py:TtRMSNorm` (replicated `ttnn.rms_norm`, `x * w`, eps 1e-6), the same module as the other norms. The only change is in hooks.py:
  `post_ffn_norm -> post_feedforward_layernorm.weight` added to `_NORM_WEIGHTS` (so device_component and HybridDeviceModel route it through `_norm_module`), and
  `post_ffn_norm` added to DEVICE_STEPS["sliding"].
- Gate PASS: pcc_post_ffn_norm_L00 0.999996, rel L2 0.0028, row norm ratio [0.9960, 1.0022]. The `FAIL ... pcc=0.000000` line comes from the precompile collect pass
  (comp_pcc stub), not the real pass.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_c_sliding_post_ffn_norm.py`

## S.sliding.13 test (run1, attempt 1): swap attn_norm .. post_ffn_norm into the sliding block
- Replaced the one-line template body with swap 12's checks, and added `post_ffn_norm` to SWAPPED. post_ffn_norm (ffn_out = rms(ffn_sum) * post_feedforward_layernorm.weight) is a
  `norm` step, so it gets the existing norm checks: vs golden PCC >= 0.99 and rel L2 <= 0.03, and vs the CPU norm on the same device ffn_sum rel L2 <= 0.03 and a
  per-token norm ratio in [0.97, 1.03]. No new limits. The gated metric stays `pcc_swap_out` >= 0.98.
- Why the iso check still matters: only ffn_residual ((h_mid + ffn_out) * layer_scalar, CPU) follows, and the h_mid residual dilutes a post_ffn_norm error at block out.
  The iso check catches `1 + w`, sum instead of mean, 2x and zeroed rows (see test_c_sliding_post_ffn_norm.py).
- Verified: BRINGUP_IMPL=reference PASS (pcc 0.999996, post_ffn_norm rel 0.0029 / iso 0.0); stub FAIL (every check). Device gate PASS (the device module was already
  registered in C.sliding.post_ffn_norm.implement): pcc_swap_out 0.999951, block out rel 0.0099 / 0.0090 (limit 0.02), post_ffn_norm pcc 0.99994, rel 0.0113 vs golden,
  iso 0.0019, ratio [0.9974, 1.0013].
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_swap_sliding_13_post_ffn_norm.py`

## C.sliding.ffn_residual test (run1, attempt 1)
- Replaced the one-line template body with the attn_residual test's checks (STEP = ffn_residual, out = (h_mid + ffn_out) * layer_scalar, layer 0 scalar 0.0703).
  The gated metric stays `pcc_ffn_residual_L00` >= 0.99. The test also asserts rel L2 <= 0.03, a per-token norm ratio in [0.97, 1.03] and a finite output. It records
  `rel_l2_*` and `row_norm_ratio_{min,max}_*` as informational metrics.
- Why: on this golden PCC passes a dropped layer_scalar (0.999997; rel 13.2, ratio 14.2), 2x (~1.0), row 0 zeroed (0.99976), the last row zeroed (0.99981) and the
  last 32 rows zeroed (0.9926). rel L2 / ratio catch all of them. CPU reference: rel 0.0023, ratio [0.9968, 1.0030]; bf16 add and scale: rel 0.0033, ratio [0.995, 1.004].
- Verified: BRINGUP_IMPL=reference PASS (pcc 0.999997, rel 0.0023); stub FAIL (PCC). The gate fails with NotImplementedError because there is no device module yet
  (that is the implement step). Note for implement: `tt/residual.py:TtResidualAdd` is a plain add. ffn_residual also needs the `* layer_scalar` (weight `layer_scalar`, shape [1]).
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_c_sliding_ffn_residual.py`

## C.sliding.ffn_residual implement (run1, attempt 1)
- Reused `tt/residual.py:TtResidualAdd(mesh, scale=...)` (replicated `ttnn.add` then `ttnn.multiply` by a Python float, no collective). No new tt/ file.
- hooks.py: new `_layer_scalar(spec, layer, loader)` reads `model.language_model.layers.<i>.layer_scalar` ([1]) on the host as a float; `_residual_host_fn` takes an
  optional `scale`. `device_component` routes `ffn_residual` to it with the layer's scalar, `HybridDeviceModel` does the same per layer, and `ffn_residual` is in
  DEVICE_STEPS["sliding"]. It is kept out of `_RESIDUAL_STEPS` because those steps have no scalar.
- Gate PASS: pcc_ffn_residual_L00 0.999995, rel L2 0.0033, row norm ratio [0.9958, 1.0046]. The `FAIL ... pcc=0.000000` line comes from the precompile collect pass
  (comp_pcc stub), not the real pass.
- With this step every sliding step is on the device, but each step still goes host -> device -> host (HybridDeviceModel).
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_c_sliding_ffn_residual.py`

## S.sliding.14 test (run1, attempt 1): swap attn_norm .. ffn_residual into the sliding block (whole block on device)
- Replaced the one-line template body with swap 13's checks and added `ffn_residual` to SWAPPED. ffn_residual
  (out = (h_mid + ffn_out) * layer_scalar) is a `residual` step whose output is block `out`, so the generic loop gives it vs golden
  PCC >= 0.99 / rel L2 <= 0.03 and vs the CPU residual on the same device h_mid and ffn_out rel L2 <= 0.03, per-token norm ratio in
  [0.97, 1.03]. These catch a dropped layer_scalar (ratio 14.2), 2x and zeroed rows. Block out rel L2 <= 0.02 still applies. No new limits.
- Measured: reference pcc_swap_out 0.999996 (pass); stub fails every check; device pcc_swap_out 0.999948, block out rel 0.0102 /
  first 1024 rows 0.0092, ffn_residual iso rel 0.0024, ratio [0.9972, 1.0042].
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_swap_sliding_14_ffn_residual.py`
  (BRINGUP_IMPL=reference / stub for the freeze checks).

## C.global.attn_norm test (run1, attempt 1)
- Replaced the one-line template body with the sliding attn_norm test's checks (STEP = attn_norm, LAYER = 5, input_layernorm.weight). The gated metric stays
  `pcc_attn_norm_L05` >= 0.99. The test also asserts rel L2 <= 0.03, a per-token norm ratio in [0.97, 1.03] and a finite output (informational metrics `rel_l2_*`,
  `row_norm_ratio_{min,max}_*`).
- Why: on the layer-5 golden (s4096 chunk 1, w in [-0.003, 136]) PCC passes sum instead of mean (0.99998), 2x (~1.0), the last row zeroed (0.9998) and the last
  32 rows zeroed (0.9928). rel L2 / ratio catch all of them. `1 + w` fails PCC here (0.82) because the weights are larger than on layer 0. CPU reference: rel 0.0024,
  ratio [0.9985, 1.0025]; bf16 in/out rel 0.0028.
- Verified: BRINGUP_IMPL=reference PASS (pcc 0.999997, rel 0.0024); stub FAIL (PCC). The device gate already PASSES: pcc 0.999996, rel 0.0030, ratio [0.9963, 1.0036],
  because hooks.py routes `attn_norm` to the shared `TtRMSNorm` for either block type. The implement step may only need to add attn_norm to DEVICE_STEPS["global"]
  (check). The `FAIL ... pcc=0.000000` line comes from the precompile collect pass (comp_pcc stub).
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_c_global_attn_norm.py`
- Re-issued brief (same attempt): the test body above was already in place. The only change was the docstring's template line, "layer 0" -> "layer 5". Re-verified:
  reference PASS (pcc 0.999997, rel 0.0024, ratio [0.9985, 1.0025]); stub FAIL (PCC 0.0); device gate PASS (pcc 0.999996, rel 0.0030, ratio [0.9963, 1.0036]).

## S.global.01 test (run1, attempt 1)
- Replaced the one-line template body with the checks from test_swap_sliding_01_attn_norm.py (BLOCK_TYPE = global, layer 5). The gated metric stays `pcc_swap_out`
  >= 0.98. The test also asserts attn_norm's own output vs golden (PCC >= 0.99, rel L2 <= 0.03) and block out rel L2 <= 0.01 (informational metrics
  `rel_l2_swap_out`, `rel_l2_swap_attn_norm_out`).
- Why: measured on the CPU (layer 5, s4096 chunk 1, block out PCC / rel | step PCC / rel): reference 0.999996 / 0.0029 | 1.0 / 0.0024; `1 + w` 0.9878 / 0.156
  (passes 0.98) | 0.818 / 1.69; no weight 0.9782 / 0.208; wrong weight (post_attention) 0.9766 / 0.215; 5% noise 0.99996 / 0.0094 (under 0.01) | 0.9987 / 0.050; x2
  same as the reference at block out | rel 1.0 at the step; zero 0.962 / 0.274. The step check catches noise and scale errors, and block-out rel L2 catches `1 + w`.
  The measurement script was /tmp/g5_variants.py (CPU only, not kept).
- Verified: BRINGUP_IMPL=reference PASS (0.999996, rel 0.0029, step rel 0.0024); stub FAIL (block PCC 0.962, rel 0.274, step rel 1.0). The device gate already PASSES:
  block PCC 0.999996, rel 0.0028, step PCC 0.999996, rel 0.0030. The `FAIL ... pcc=0.000000` / rel 0.3139 lines come from the precompile pass (comp_pcc stub).
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_swap_global_01_attn_norm.py`

## C.global.attention test (run1, attempt 1)
- Replaced the one-line template body with the sliding attention test's structure (STEP = attention, LAYER = 5). The gated metric stays `pcc_attention_L05` >= 0.99.
  The test also asserts rel L2 <= 0.03 on the whole chunk and on the first 128 rows, a per-token norm ratio in [0.95, 1.05], and a finite output (informational metrics
  `rel_l2_*`, `rel_l2_head_rows_*`, `row_norm_ratio_{min,max}_*`).
- Why: measured on the CPU (layer 5, s4096 chunk 1; PCC / rel whole / rel first 128 rows). RoPE positions counted from 0 pass PCC (0.9991 / 0.042 / 0.062, ratio [0.85, 1.22]).
  PCC already fails the others: no prefix 0.849, V = roped K 0.922, V with k_norm weight 0.929, window 1024 0.877, scale 1/sqrt(512) 0.759, RoPE on all dims 0.929,
  interleaved RoPE 0.972, no q norm 0.973, non-causal 0.943. Accuracy budget: reference 0.0028, bf16 0.0030, bf16 act + emulated bfp8 q/k/o weights 0.0076 (ratio
  [0.996, 1.005]). I used the first 128 rows instead of the sliding test's first `sliding_window` rows because a global layer has no window, and the RoPE error is largest on the
  earliest rows (32: 0.10, 128: 0.062, 1024: 0.044). The measurement script was /tmp/g5att/variants.py (CPU only, not kept).
- Verified: BRINGUP_IMPL=reference PASS (pcc 0.999997, rel 0.0028, first 128 rows 0.0027, ratio [0.9992, 1.0038]); stub FAIL (PCC 0.0). The device gate fails with
  `NotImplementedError: implement step: no device attention for global layer 5 yet` (hooks.py:207). That is expected, because there is no device module yet.
- Note for implement: global = 16 q heads x 512, 2 KV heads x 512 (fewer KV heads than the 4 chips), no v_proj. V = rms_norm(k_proj(x)) without a weight, and K = RoPE(k_norm(k_proj(x))).
  RoPE is proportional, theta 1e6, rotating only dims [0:64] + [256:320] (the other inv_freq entries are 0). Scale is 1.0, and there is no window.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_c_global_attention.py`

## C.global.attention implement (run1, attempt 1)
- Added `tt/attention.py:TtGlobalAttention` (subclasses TtSlidingAttention to reuse `_shard`, `_replicate`, `_head_norm`, `_rope_tables`) and `TtKVCacheGlobal`.
  Chip c holds Q heads 4c..4c+3 and KV head c // 2. The fused per-chip weight is `[Wq_c | Wk_h | Wk_h]` ([H, 3072]), so `nlp_create_qkv_heads(num_heads=4, num_kv_heads=1)`
  returns raw K twice: q = rope(q_norm(q)), k = rope(k_norm(k_raw)), v = rms(k_raw) without a weight. RoPE uses the rotate-half `rotary_embedding` with the proportional
  inv_freq (zeros past 64, so cos = 1 and sin = 0 there, which matches the reference exactly). Scale 1.0, no window. o_proj is row-parallel [2048, H] per chip, then `all_reduce(cluster_axis=1)`.
- Cache: per chip a paged-shaped [max_seq/64, 1, 64, 512] bf16 tensor. With one head per chip this is bit-identical to a contiguous cache, and it uses an identity page table (ERNIE pattern).
  The host [2, S, D] layout is repeat_interleaved to 4 chips on load. `to_torch` reads chips 0 and 2. Writes go through `paged_fill_cache` with a per-chunk page table.
  Chunk 0 runs `scaled_dot_product_attention(is_causal)`, and later chunks run `chunked_scaled_dot_product_attention` over the whole cache, all keyword arguments, `scale=1.0`.
- SDPA q/k chunk 128 (head dim 512; I did not try larger), HiFi4 + fp32 acc, exact exp. Projections HiFi2 + fp32 acc. HiFi4 projections were measured and were *worse*
  (rel 0.0092, ratio max 1.0145 vs 0.0079 / 1.0111), so I kept HiFi2. The error seems to come from bf16 q/k/probabilities, not from matmul fidelity.
- hooks.py: `_attention_module` builds either class (global: no v_proj, asserts `attention_k_eq_v`). The new `_new_kv_cache(mesh, cfg, layer, max_seq)` picks the cache class by layer type
  (used by `device_component` and `_HybridState`). `DEVICE_STEPS["global"] = {attn_norm, attention}`. attn_norm was already device-verified for global (C.global.attn_norm, S.global.01)
  but had not been added yet.
- Gate PASS: pcc_attention_L05 0.999975, rel L2 0.00795, first 128 rows 0.00696, row-norm ratio [0.9952, 1.0111]. The `FAIL ... pcc=0.000000` line is the precompile stub.
- Not covered by this gate: the chunk-0 path (plain SDPA) and the K/V read-back (`to_torch`). The ladder's state metrics exercise both.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_c_global_attention.py`

## S.global.02 test (run1, attempt 1)
- Replaced the one-line template body with the checks from test_swap_sliding_02_attention.py (BLOCK_TYPE = global, layer 5). The global layer has no window,
  so the "prefix rows" are the first 128 rows (HEAD_ROWS, the same as test_c_global_attention.py). The gated metric stays `pcc_swap_out` >= 0.98. The test also asserts:
  each swapped step's own output vs golden (PCC >= 0.99, rel L2 <= 0.03), attn_out rel L2 on the first 128 rows <= 0.03, attn_out per-token norm ratio in [0.95, 1.05],
  block out rel L2 <= 0.02 over the whole chunk and over the first 128 rows, and finite outputs. The limits come from the component test and sliding swap 2. I took no new CPU measurements.
  The RoPE-from-0 bug is caught at the step (attn_out first 128 rows rel 0.062, ratio [0.85, 1.22], from C.global.attention.test).
- Verified: BRINGUP_IMPL=reference PASS (block 0.999996, rel 0.0029 / first 128 rows 0.0026; attn rel 0.0028 / 0.0027, ratio [0.9989, 1.0034]). Stub FAIL (block PCC fails,
  rel 0.314, attn rel 1.0). The device gate PASSES: pcc_swap_out 0.999990, block rel 0.0045 / 0.0033, attn_norm rel 0.0030, attn_out PCC 0.999975, rel 0.0079 / 0.0070,
  ratio [0.9964, 1.0117]. The FAIL / pcc=0.000000 lines come from the precompile pass (comp_pcc stub).
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_swap_global_02_attention.py`
  (BRINGUP_IMPL=reference / stub for the freeze checks).

## C.global.post_attn_norm test (run1, attempt 1)
- Replaced the one-line template body with test_c_sliding_post_attn_norm.py (LAYER = 5). The gated metric stays `pcc_post_attn_norm_L05` >= 0.99. The test also asserts
  rel L2 <= 0.03, a per-token norm ratio in [0.97, 1.03], and a finite output (informational metrics `rel_l2_*`, `row_norm_ratio_{min,max}_*`).
- Measured on the CPU (layer 5 golden, attn_out -> attn_post_norm; the weight was recovered from the golden and lies in [0.0009, 1.10], much smaller than layer 0's):
  reference rel 0.0023, ratio [0.998, 1.002]; bf16 0.0028; 1% noise 0.010; `1 + w` PCC 0.753 (at layer 0 it passed PCC); no weight PCC 0.493;
  sum instead of mean PCC ~1.0 but rel 0.98, ratio 0.019. The script was /tmp/g5pan/v.py (CPU only, not kept).
- Verified: BRINGUP_IMPL=reference PASS (pcc 0.999997, rel 0.0023). Stub FAIL (PCC 0.0). The device gate already PASSES (pcc 0.999996, rel 0.0030, ratio [0.9963, 1.0013]),
  because the device norm is the same module the sliding layers use. The `FAIL ... pcc=0.000000` line comes from the precompile pass (comp_pcc stub).
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_c_global_post_attn_norm.py`

## S.global.03 test (run1, attempt 1)
- Replaced the one-line template body with test_swap_global_02_attention.py (HEAD_ROWS = 128 checks on attn_out, block out rel L2 <= 0.02) plus the norm isolation check
  from test_swap_sliding_03_post_attn_norm.py (each swapped norm vs the CPU norm on the exact device input: rel L2 <= 0.03, per-token ratio in [0.97, 1.03]).
  SWAPPED = attn_norm, attention, post_attn_norm. The gated metric stays `pcc_swap_out` >= 0.98. No new CPU bug measurements; the post_attn_norm bug numbers are from test_c_global_post_attn_norm.py.
- Verified: BRINGUP_IMPL=reference PASS (block 0.999996, rel 0.0029 / 0.0026; post_attn_norm rel 0.0028). Stub FAIL (every check). The device gate PASSES:
  pcc_swap_out 0.999991, block rel 0.0042 / 0.0033, attn_out rel 0.0079 / 0.0070, post_attn_norm rel 0.0084 vs golden, iso 0.0019, ratio [0.9974, 1.0010].
  The device post_attn_norm is the same module as the sliding one, already registered. The pcc=0.000000 lines come from the precompile pass (comp_pcc stub).
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_swap_global_03_post_attn_norm.py`
  (BRINGUP_IMPL=reference / stub for the freeze checks).

## C.global.attn_residual test (run1, attempt 1)
- Replaced the one-line template body with test_c_sliding_attn_residual.py (LAYER = 5). The gated metric stays `pcc_attn_residual_L05` >= 0.99. The test also asserts
  rel L2 <= 0.03, a per-token norm ratio in [0.97, 1.03], and a finite output (informational metrics `rel_l2_*`, `row_norm_ratio_{min,max}_*`).
- Measured on the CPU (layer 5 golden, s4096 chunk 1; ||in|| 4.1e3, ||attn_post_norm|| 1.2e3, so the residual dominates here, unlike layer 0). PCC / rel / ratio:
  reference 0.999997 / 0.0022 / [0.998, 1.002]; bf16 0.0027; 2x passes PCC but rel 1.0; row 0 zeroed 0.99974 / 0.023; last row zeroed 0.99990 / 0.014;
  last 32 rows zeroed 0.993 / 0.117; attention dropped 0.966; residual dropped 0.477. Zeroed rows fail through the ratio check (min 0). Script /tmp/g5res/v.py (CPU only, not kept).
- Verified: BRINGUP_IMPL=reference PASS (pcc 0.999997, rel 0.0022). Stub FAIL (PCC 0.0). The device gate already PASSES (pcc 0.999996, rel 0.0028, ratio [0.9970, 1.0032]).
  The `FAIL ... pcc=0.000000` line comes from the precompile pass (comp_pcc stub).
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_c_global_attn_residual.py`

## S.global.04.test.1 (swap test review, global attn_residual)
- Replaced the rendered one-liner with global swap 3's checks (s4096 chunk 1, HEAD_ROWS = 128) and extended the
  CPU-on-same-inputs isolation check from norms to residuals (`ISO_STEP_KINDS = ("norm", "residual")`), as sliding swap 4.
  Why: PCC misses `2 * (a + b)` and zeroed rows in a residual.
- BRINGUP_IMPL=reference: pass (pcc_swap_out 0.999996, block out rel 0.0029). BRINGUP_IMPL=stub: fails every check.
- Device gate: pass, pcc_swap_out 0.999989, block out rel 0.0046 / 0.0035; attn_residual rel 0.0034 vs golden,
  iso 0.0017, ratio [0.9990, 1.0019].
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_swap_global_04_attn_residual.py`

## C.global.ffn_norm test (run1, attempt 1)
- Replaced the one-line template body with test_c_sliding_ffn_norm.py (LAYER = 5). The gated metric stays `pcc_ffn_norm_L05` >= 0.99. The test also asserts
  rel L2 <= 0.03, a per-token norm ratio in [0.97, 1.03], and a finite output (informational metrics `rel_l2_*`, `row_norm_ratio_{min,max}_*`).
- Measured on the CPU (layer 5 golden, h_mid -> ffn_norm; weight recovered from the golden, in [-0.0002, 93]): reference PCC 0.999997 / rel 0.0023 / ratio [0.9968, 1.0030];
  bf16 0.0028; 1% noise 0.010; `1 + w` PCC 0.768; no weight 0.596; sum instead of mean PCC ~1.0 but rel 0.98; 2x rel 1.0; last row zeroed PCC 0.9997 (ratio min 0);
  last 32 rows zeroed PCC 0.992 / rel 0.125. Script /tmp/g5ffn/v.py (CPU only, not kept).
- Verified: BRINGUP_IMPL=reference PASS (pcc 0.999997, rel 0.0023). Stub FAIL (PCC 0.0). The device gate already PASSES (pcc 0.999996, rel 0.0029, ratio [0.9954, 1.0022]),
  because the device norm is the same module the sliding layers use. The `FAIL ... pcc=0.000000` line comes from the precompile pass (comp_pcc stub).
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_c_global_ffn_norm.py`

## S.global.05.test.1 (swap test review, global ffn_norm)
- Replaced the rendered one-liner with global swap 4's checks (s4096 chunk 1, HEAD_ROWS = 128, block out rel L2 <= 0.02,
  norm/residual isolation vs the CPU step on the same device inputs) and added ffn_norm to SWAPPED, as sliding swap 5 does.
  The isolation check covers ffn_norm, so sum-instead-of-mean and 2x (PCC ~1.0) fail at the step. The router/MoE downstream would blur them.
- BRINGUP_IMPL=reference: pass (pcc_swap_out 0.999996, block out rel 0.0029, ffn_norm rel 0.0026). BRINGUP_IMPL=stub: fails every check.
- Device gate: pass, pcc_swap_out 0.999989, block out rel 0.0047 / 0.0035; ffn_norm rel 0.0067 vs golden, iso 0.0019,
  ratio [0.9970, 1.0010]. The pcc=0.000000 lines come from the precompile pass (comp_pcc stub).
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_swap_global_05_ffn_norm.py`

## C.global.mlp test (run1, attempt 1)
- Replaced the one-line template body with test_c_sliding_mlp.py (LAYER = 5). The gated metric stays `pcc_mlp_L05` >= 0.99. The test also asserts
  rel L2 <= 0.03, a per-token norm ratio in [0.97, 1.03], and a finite output (informational metrics `rel_l2_*`, `row_norm_ratio_{min,max}_*`).
- Measured on the CPU (layer 5 golden, ffn_norm -> mlp_out, row norms 24-375, about 10x smaller than layer 0). PCC / rel / ratio: reference 1.000000 / 0.0018 / [0.998, 1.002];
  bf16 0.0014; emulated bfp8 weights + activations 0.999963 / 0.0093 / [0.991, 1.006] (inside the limits); silu 0.968 / 0.88 (fails PCC here, unlike layer 0);
  2x passes PCC, rel 1.0; row 0 / last row zeroed pass PCC, ratio min 0; last 32 rows zeroed 0.9934 / 0.115; quarter of I missing 0.959. Script /tmp/g5mlp/v.py (CPU only, not kept).
- Verified: BRINGUP_IMPL=reference PASS (pcc 0.999998, rel 0.0018). Stub FAIL (PCC 0.0). The device gate already PASSES (pcc 0.999995, rel 0.0037, ratio [0.9982, 1.0045]),
  because the device MLP is the same module the sliding layers use. The `FAIL ... pcc=0.000000` line comes from the precompile pass (comp_pcc stub).
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_c_global_mlp.py`

## S.global.06.test.1 (swap test review, global mlp)
- Replaced the rendered one-liner with global swap 5's checks (s4096 chunk 1, HEAD_ROWS = 128, block out rel L2 <= 0.02,
  isolation vs the CPU step on the same device inputs). Added mlp to SWAPPED and "mlp" to ISO_STEP_KINDS, as sliding swap 6 does.
  Why: the CPU post_mlp_norm right after mlp undoes any per-row scale, so block out cannot see 2x or zeroed rows in mlp_out. PCC passes both (test_c_global_mlp.py).
- BRINGUP_IMPL=reference: pass (pcc_swap_out 0.999996, block out rel 0.0029, mlp rel 0.0019). BRINGUP_IMPL=stub: fails.
- Device gate: pass, pcc_swap_out 0.999989, block out rel 0.0048 / 0.0037; mlp rel 0.0045 vs golden, iso 0.0032,
  ratio [0.9988, 1.0044]. The pcc=0.000000 lines come from the precompile pass (comp_pcc stub).
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_swap_global_06_mlp.py`

## C.global.post_mlp_norm test (run1, attempt 1)
- Replaced the one-line template body with test_c_sliding_post_mlp_norm.py (LAYER = 5). The gated metric stays `pcc_post_mlp_norm_L05` >= 0.99. The test also asserts
  rel L2 <= 0.03, a per-token norm ratio in [0.97, 1.03], and a finite output (informational metrics `rel_l2_*`, `row_norm_ratio_{min,max}_*`).
- Measured on the CPU (layer 5 golden, mlp_out -> mlp_post_norm; weight recovered from the golden, in [-0.21, 163]). PCC / rel / ratio: recovered-weight norm 0.999997 / 0.0024 / [0.9961, 1.0043];
  bf16 0.0029; 1% noise 0.010; `1 + w` 0.9985 (passes PCC) / 0.058; no weight PCC 0.24; sum instead of mean ~1.0 / 0.98; 2x ~1.0 / 1.0; row 0 / last row zeroed pass PCC, ratio min 0;
  last 32 rows zeroed 0.988 / 0.157. Script /tmp/g5pmn/v.py (CPU only, not kept).
- Verified: BRINGUP_IMPL=reference PASS (pcc 0.999997, rel 0.0024). Stub FAIL (PCC 0.0). The device gate already PASSES (pcc 0.999996, rel 0.0031, ratio [0.9944, 1.0051]),
  because the device norm is the same module the sliding layers use. The `FAIL ... pcc=0.000000` line comes from the precompile pass (comp_pcc stub).
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_c_global_post_mlp_norm.py`

## S.global.07.test.1 (swap test review, global post_mlp_norm)
- Replaced the rendered one-liner with global swap 6's checks (s4096 chunk 1, HEAD_ROWS = 128, block out rel L2 <= 0.02,
  isolation vs the CPU step on the same device inputs) and added post_mlp_norm to SWAPPED, as sliding swap 7 does. post_mlp_norm is a norm step,
  so the existing isolation check covers it. Why: PCC misses `1 + w` (0.9985) and zeroed rows (test_c_global_post_mlp_norm.py).
- BRINGUP_IMPL=reference: pass (pcc_swap_out 0.999996, block out rel 0.0029, post_mlp_norm rel 0.0019). BRINGUP_IMPL=stub: fails every check.
- Device gate: pass, pcc_swap_out 0.999988, block out rel 0.0048 / 0.0038; post_mlp_norm rel 0.0048 vs golden, iso 0.0019,
  ratio [0.9963, 1.0021]. The pcc=0.000000 lines come from the precompile pass (comp_pcc stub).
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_swap_global_07_post_mlp_norm.py`

## C.global.router test (run1, attempt 1)
- Replaced the one-line template body with test_c_sliding_router.py (LAYER = 5). The gated metric stays `pcc_router_L05` >= 0.99. The test also asserts: exactly 8 nonzeros per row,
  non-negative weights, top-8 selection overlap >= 0.995, matched-row weight rel L2 <= 0.005, and a per-row sum ratio in [0.99, 1.01]. Informational metrics are
  `selection_overlap_router_L05`, `matched_rel_l2_router_L05` and `row_sum_ratio_{min,max}_router_L05`. The thresholds are unchanged from sliding.
- Scored mutations on the CPU (layer 5 golden; per_expert_scale in [0.980, 1.023], same as layer 0). The numbers are in the test docstring. Every mutation that passes PCC
  (no/by-rank per_expert_scale, top-7/9, zeroed rows, 1-3% noise, 2x) fails an extra check. Unlike layer 0, renorm-after-per_expert_scale also fails matched rel L2 here (0.0066),
  as well as the row-sum check. Script /tmp/g5rt/v.py (CPU only, not kept).
- Verified: BRINGUP_IMPL=reference PASS (pcc 0.999957, overlap 0.99957, mrel 0.0018). Stub FAIL (PCC 0.0). The device gate already PASSES (pcc 0.999879, overlap 0.99854,
  matched 2024/2048, mrel 0.00283, ratio [0.9961, 1.0034]), because the device router is the same TtRouter the sliding layers use. The `FAIL ... pcc=0.000000` line comes from the precompile pass (comp_pcc stub).
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_c_global_router.py`

## S.global.08.test.1 (swap test review, global router)
- Replaced the rendered one-liner with global swap 7's checks (s4096 chunk 1, HEAD_ROWS = 128, attention head-row rel L2 and norm ratio, block out rel L2 <= 0.02,
  norm/residual/mlp isolation vs the CPU step on the same device inputs), added router to SWAPPED, and copied sliding swap 8's router branch unchanged
  (exactly 8 nnz/row, no negative weights; vs golden / vs CPU router on the same device h_mid: overlap >= 0.99 / 0.995, matched-row rel L2 <= 0.01 / 0.005,
  row-sum ratio in [0.99, 1.01]; no whole-matrix rel L2 for the router). Why: PCC misses per_expert_scale, top-k and zeroed-row bugs (test_c_global_router.py).
- BRINGUP_IMPL=reference: pass (pcc_swap_out 0.999996, router overlap 0.99933 / 1.0, matched rel 0.0019). BRINGUP_IMPL=stub: fails every check.
- Device gate: pass, pcc_swap_out 0.999989, block out rel 0.0047 / 0.0038; router pcc 0.99983, whole-matrix rel 0.018, overlap 0.99805 / 0.99890,
  matched rel 0.00337 / 0.00232, row-sum ratio [0.9958, 1.0033] / [0.9973, 1.0025]. The pcc=0.000000 lines come from the precompile pass (comp_pcc stub).
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_swap_global_08_router.py`

## C.global.moe_norm.test.1 (test review)
- Rewrote the rendered `tests/bringup/test_c_global_moe_norm.py` to the same shape as `test_c_global_ffn_norm.py` /
  `test_c_sliding_moe_norm.py`: gated PCC >= 0.99, plus asserted rel L2 <= 0.03 and per-token norm ratio in [0.97, 1.03]
  (recorded as informational metrics). Reason: PCC is scale-blind (known issue "PCC alone does not gate a norm").
- Layer 5 pre_feedforward_layernorm_2 w in [-0.31, 175]. CPU-only measurements on the golden ([2048, 2816]): reference
  PCC 0.999997 / rel 0.0024 / ratio [0.9983, 1.0014]; bf16 rel 0.0029; `1 + w` PCC 0.535; no weight 0.33; wrong weight
  0.53; sum-instead-of-mean and 2x pass PCC but rel ~1.0; last row zeroed caught by ratio; last 32 rows rel 0.123.
- BRINGUP_IMPL=reference passes; BRINGUP_IMPL=stub fails (PCC 0). Default gate: PCC 0.999996, rel 0.0030,
  ratio [0.9975, 1.0010].
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_c_global_moe_norm.py`

## S.global.09.test.1 (swap test review, global moe_norm)
- Replaced the rendered one-liner with global swap 8's checks (s4096 chunk 1, HEAD_ROWS = 128, attention head-row checks, block out rel L2 <= 0.02,
  norm/residual/mlp isolation vs the CPU step on the same device inputs, router selection/weight checks) and added moe_norm to SWAPPED, as sliding swap 9 does.
  moe_norm is a `norm` step, so the existing checks cover it: vs golden (PCC >= 0.99, rel L2 <= 0.03) and iso (rel L2 <= 0.03, row-norm ratio [0.97, 1.03]).
  Why: PCC misses sum-instead-of-mean, 2x and zeroed rows (test_c_global_moe_norm.py), and post_moe_norm hides per-row scale errors from block out.
- BRINGUP_IMPL=reference: pass (pcc_swap_out 0.999996, block out rel 0.0029, moe_norm rel 0.0026). BRINGUP_IMPL=stub: fails every check.
- Device gate: pass, pcc_swap_out 0.999989, block out rel 0.0048 / 0.0038; moe_norm pcc 0.99998, rel 0.0061 vs golden, iso 0.0019,
  ratio [0.9984, 1.0006]. The pcc=0.000000 lines come from the precompile pass (comp_pcc stub).
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_swap_global_09_moe_norm.py`

## C.global.experts.test.1 (test review)
- Replaced the rendered one-liner with test_c_sliding_experts.py (LAYER = 5). The gated metric stays `pcc_experts_L05` >= 0.99. The test also asserts a finite output,
  rel L2 <= 0.03, a per-token norm ratio in [0.97, 1.03], and the worst per-token rel L2 <= 0.1. Informational metrics: `rel_l2_experts_L05`,
  `row_norm_ratio_{min,max}_experts_L05`, `max_row_rel_l2_experts_L05`. Thresholds are the same as for sliding.
- Layer 5 golden routing is more skewed than layer 0: tokens per expert run from 0 to 1915, expert 93 takes 1915 of 2048 tokens, and expert 127 gets none, so a
  "drop expert 127" mutation is a no-op here. Any per-expert capacity cap in dispatch fails badly (capacity 512: PCC 0.74).
- Scored mutations on the CPU (script /tmp/g5ex/v.py, not kept); the numbers are in the test docstring. These pass PCC but fail an extra check: drop expert 0, drop the smallest pair per token,
  drop token 0's top-1, a zeroed last row or last 32 rows, and 2x. Emulated bfp8 activations and weights pass (rel 0.0153, worst row 0.020). Known gap as at layer 0:
  routing renormalized to sum 1 passes (the router test catches it).
- Verified: BRINGUP_IMPL=reference PASS (pcc 0.999996, rel 0.0027, ratio [0.9963, 1.0032], worst 0.0044). Stub FAIL (PCC 0). Default device gate already PASSES
  (pcc 0.999837, rel 0.0226, ratio [0.9971, 1.0225], worst row 0.030), because the global layer uses the same device experts module as sliding. rel L2 has only about 25% headroom to 0.03.
  The `FAIL ... pcc=0.000000` line comes from the precompile pass (comp_pcc stub).
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_c_global_experts.py`

## S.global.10.test.1 (swap test review, global experts)
- Replaced the rendered one-liner with global swap 9's checks (s4096 chunk 1, HEAD_ROWS = 128, attention head-row checks, block out rel L2 <= 0.02,
  norm/residual/mlp isolation, router selection/weight checks), added experts to SWAPPED, and copied sliding swap 10's `moe` branch and constants unchanged:
  experts vs golden PCC >= 0.99 and rel L2 <= 0.05; vs the CPU experts on the same device moe_norm and router rel L2 <= 0.03, per-token norm ratio in
  [0.97, 1.03], worst row <= 0.1. Why: post_moe_norm hides per-row scale errors from block out, and upstream router flips move whole experts rows vs golden.
- BRINGUP_IMPL=reference: pass (block out rel 0.0029, experts rel 0.0073 vs golden, iso 0.0). BRINGUP_IMPL=stub: fails every check.
- Device gate: pass, pcc_swap_out 0.999973, block out rel 0.0074 / 0.0070; experts pcc 0.99976, rel 0.0246 vs golden, iso rel 0.0224,
  ratio [1.0007, 1.0214], worst row 0.0292. The ratio is above 1 on every row: this is the fused kernel's known overshoot (HiFi2 + fp32 dest, about 1.01-1.02).
  Headroom is tight (iso rel 0.0224 vs 0.03, ratio max 1.021 vs 1.03). The pcc=0.000000 lines come from the precompile pass (comp_pcc stub).
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/gemma4_a4b_d_p/tests/bringup/test_swap_global_10_experts.py`
