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
