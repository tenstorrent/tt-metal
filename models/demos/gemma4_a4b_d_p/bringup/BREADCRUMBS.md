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
