# XiaomiMiMo/MiMo-V2.6-Flash-RL bring-up: breadcrumbs

Append-only log, one section per task attempt: what was done, decisions and why, gotchas, the re-run command, the verdict.

## R.2 reference (attempt 1 -> 2), 2026-09-26

What was done
- `reference/weights.py`: WeightLoader, fp8 128x128 block dequant, per-TP-rank fused qkv dequant + reorder to [q; k; v],
  mxfp4 dequant (low nibble = even column, e8m0 2^(s-127)), `PackedExpert` (mxfp4 kept, dequantized on use).
- `reference/mimo_ref.py`: `MiMoReference` (interface.py). Graphs: dense (layer 0) attn_norm, attention, attn_residual,
  ffn_norm, mlp, mlp_residual; MoE: ... ffn_norm, router ([S,256] fp32 dense routing), experts, ffn_residual.
  State key [Hkv, max_seq, 192] post-RoPE, value [Hkv, max_seq, 128] x 0.707. Sliding attention uses the window-bounded
  key range and a manual sink softmax; full attention uses torch SDPA per 512-row query block.
- `reference/hf_oracle.py` + `hooks.hf_model`: the checkpoint's own MiMoV2ForCausalLM, text only (vision/audio configs
  emptied), fp32, eager, built on meta; routed experts replaced by `PackedExpertMLP` (same math as MiMoV2MLP, mxfp4
  dequantized in forward); other weights dequantized and assigned. Used for both the sanity (full 48 layers) and parity.
- hooks: `reference`, `hf_model`; both cap torch threads at physical cores (16).

Decisions and why
- qkv per-rank layout: scale rows 108 = 4 x 27 for layer 0 (ceil(3392/128) per rank), and dequantized row norms repeat
  with the per-rank period (k rows ~5.5, v rows ~0.45 at the end of each 3392-row slab; layer 1 same with 3712).
- Nibble order: low first; the per-input-column |W| profile of the experts correlates 0.61 with the layer's
  post_attention_layernorm weight (0.34 with the router), high first 0.03 / -0.005. Confirmed by the sanity gate.
- Experts are dequantized at load when <= 8 MoE layers are requested (layers 0-5: ~129 GB fp32), else kept packed
  (48-layer parity run). Both paths give identical weights. Expert GEMM groups padded to 32 rows (known issue).
- HF oracle runs fp32 in both gates (the hook has no dtype argument); the sanity doc says bf16 but fp32 is only more exact.
- HF and the reference share weights.py (the checkpoint's storage definition); parity checks model math, the sanity gate
  (smoke + 0.955 accuracy) checks the dequantization.

Gotchas
- Stock `from_pretrained` fails: index lists `model_mtp.safetensors` (not downloaded, out of scope) and the fp8 path
  does not know mxfp4 or the per-rank qkv.
- modeling_mimo_v2.py (transformers 5.3) passes `input_embeds=` / `cache_position=` to the mask builders; 5.12 rejects
  them. hf_oracle wraps the two names in the modeling module (rename / drop), semantics unchanged.
- The index metadata says `tp_size: 4`; `WeightLoader.tp_size` reads it.

Results (hand run of the gate)
- sanity: revision ok, smoke 'Paris<|im_end|>' ok, text_top1_acc 0.955.
- check_hf --seq 2048 (48 layers, fp32): every pcc_hidden_L* 1.0000000 (maxabs <= 5.4e-3 at L47), logits pcc 1.0000000,
  top1_match 1.0000, text next-token acc 0.955. Both commands 21 min total.
- Extra (R.3 preview): check_reference --seq 4096 --chunk 2048: chunked vs one-shot hidden/state pcc 1.000000000, graph
  replay max abs 0 for layers 0, 1, 5; 2.5 min.

Re-run
    PYTHONPATH=$PWD python -m models.demos.common.bringup.intake.check_hf_sanity && \
    PYTHONPATH=$PWD python -m models.demos.common.bringup.reference.check_hf --seq 2048

## PL.1 plan (attempt 1), 2026-09-26

What was done
- `plan.yaml`: placements for all 73033 checkpoint tensors (0 unplaced). Vision, audio, speech embeddings and layers 6-47
  skipped (text only, spec layer subset); fp8 `weight_scale_inv` and mxfp4 `weight_scale` skipped (folded into the device
  weight at load). Per-chip total 14.65 GiB of 27.2 GiB.
- `plan.md`: per-block-type tables (full_dense, sliding_moe, full_moe, model level), gate totals, departures.
- `components.yaml`: all 20 block steps + embed / final_norm / lm_head mapped (generated with a script, plain YAML, no anchors).
- tasks.yaml unchanged (the PL.0 ledger already has one C/S task per step).

Decisions and why
- TP=4 by head for attention: the checkpoint's fused qkv is stored per TP rank (tp_size 4), so chip r takes rank r's
  slab as stored (3392 rows full, 3712 sliding). 1 KV head per chip (full), 2 (sliding); no duplication.
- V padded 128 -> 192 on device (zero rows in qkv, zero columns in o_proj): plain/chunked SDPA need V dim == QK dim.
  State V counted at 192 in plan.yaml. attention_value_scale 0.707 folded into the V rows.
- Sliding attention: gemma4_a4b_d_p TtSlidingAttention pattern (previous 128 cached rows + chunk, q_pad) with
  `scaled_dot_product_attention(sliding_window_size=128, attention_sink=sink/scale)` (gpt_oss_d_p convention).
- Full attention: gemma4_a4b_d_p TtGlobalAttention pattern (paged-shaped cache, identity page table, chunked SDPA).
- Partial RoPE (64 of 192 dims, rotate-half): slice -> rotary_embedding -> concat planned; a row permutation of q/k weights
  would avoid the slice but changes the cached K order vs the golden.
- EP=4 experts (64 per chip), bfp8 (owner rule), ERNIE moe_unified pipeline with Silu. Pass weights_dtype=bfloat8_b
  explicitly (tt_routed_expert default is bfloat4_b).
- Router: DeepSeek moe_grouped_topk (sigmoid, bias, 1 group), fp32 weights, route_scale 1.0 (config routed_scaling_factor null).
- qkv and dense MLP bf16 (small: 0.35 GiB/chip), accuracy first.

Gotchas for the implement steps
- check_plan counts the mxfp4-packed expert shape (half the values); the other half is an explicit extra entry.
- Silu variant of unified_routed_expert_moe runs LoFi with bf16 dest (known issue); extend GeluTanh's fidelity handling
  if the experts component fails rel-L2 / norm ratio.
- moe_grouped_topk CBs scale with experts/32; DeepSeek runs <= 4096 rows per chip. The s16384 rung has 8192-row chunks.
- Vocab 152576 is not a power of two: the engine pad-id mask (bitwise_and) from gemma4 does not apply; clamp instead.
- check_plan loads the reference for layers 0, 1, 5 in fp32 (~51 GB RAM for the two MoE layers' experts).

Result (hand run): plan_fits 1, unplaced 0, plan_errors 0, component_errors 0, ledger_errors 0, plan_approved 0
(awaiting approval).

Re-run
    PYTHONPATH=$PWD python -m models.demos.common.bringup.plan.check_plan

## C.full_dense.attn_norm test (attempt 1), 2026-09-26

What was done
- Replaced the rendered `run_component_test` call in `tests/bringup/test_c_full_dense_attn_norm.py` with the
  gemma4_a4b_d_p attn_norm template: gated PCC (spec threshold 0.99) plus finite output, rel L2 <= 0.03 and per-token
  norm ratio in [0.97, 1.03]. Records `rel_l2_attn_norm_L00`, `row_norm_ratio_{min,max}_attn_norm_L00` (informational).

Why
- PCC is scale-invariant: on this golden a sum-instead-of-mean RMS scores PCC 0.999994 (passes) but rel 0.98.
  Measured (CPU, golden [2048, 4096] bf16): reference fp32 PCC 0.999999 / rel 0.0016 / ratio [0.9980, 1.0023];
  bf16 math is bit-equal to the golden; eps 1e-2 rel 0.89; `1 + w` or dropped w rel 54 (w mean 0.0056, so `1 + w` is
  a gross error here, unlike Gemma).

Results
- BRINGUP_IMPL=reference: PASS (pcc 0.999999, rel 0.001597). BRINGUP_IMPL=stub: FAIL (pcc 0.0).
- Default mode: FAIL with NotImplementedError (no device module yet; implement step).

Re-run
    PYTHONPATH=$PWD BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p/tests/bringup/test_c_full_dense_attn_norm.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p/tests/bringup/test_c_full_dense_attn_norm.py

## C.full_dense.attn_norm implement (attempt 1), 2026-09-26

What was done
- `tt/rms_norm.py`: TtRMSNorm (copy of gemma4_a4b_d_p/tt/rms_norm.py): replicated `ttnn.rms_norm`, TILE [1,1,1,H]
  bf16 gamma, HiFi4 + fp32 dest acc, eps 1e-6, plain checkpoint `w`. Also the harness-boundary helpers
  `to_device_replicated` / `replicated_to_host` (not used inside any forward).
- `bringup/hooks.py`: `DEVICE_STEPS = {"full_dense": {"attn_norm"}}`, `_NORM_WEIGHTS` (attn_norm ->
  input_layernorm, ffn_norm -> post_attention_layernorm; ffn_norm is mapped but not yet in DEVICE_STEPS),
  `_norm_module` (loads one weight via reference/weights.WeightLoader), `_host_fn`, `device_component` for norm
  steps, and `HybridDeviceModel` / `_HybridState` as `device_model` (CPU reference with DEVICE_STEPS overridden, plain
  embedding with no scale, final norm on CPU). Adding a later norm step = add it to DEVICE_STEPS.

Result
- Gate: pcc_attn_norm_L00 0.999998, rel_l2 0.002118, row_norm_ratio [0.9956, 1.0012]. PASS.

Re-run
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p/tests/bringup/test_c_full_dense_attn_norm.py

## S.full_dense.01 test (attempt 1), 2026-09-26

What was done
- Replaced the rendered `run_swap_test` call in `tests/bringup/test_swap_full_dense_01_attn_norm.py` with the
  gemma4_a4b_d_p swap_sliding_01 pattern: gated pcc_swap_out (spec block 0.98) plus asserted extras: the swapped
  step's own output (finite, PCC >= 0.99, rel L2 <= 0.03, per-token norm ratio in [0.97, 1.03]) and block out
  (finite, rel L2 <= 0.01). Records `rel_l2_swap_out`, `rel_l2_swap_attn_norm`, `row_norm_ratio_{min,max}_swap_attn_norm`.

Why
- The gated metric alone passes the zero stub: pcc_swap_out 0.9884 (rel 0.155). Also passing the gate: sum instead of
  mean 0.9890, eps 1e-2 0.9916. Mutations measured on the CPU (script in /tmp, not kept) are in the test docstring;
  a zeroed last row is caught only by the norm ratio.

Results
- BRINGUP_IMPL=reference: PASS (out 0.999999 / rel 0.0017). BRINGUP_IMPL=stub: FAIL (extra checks; gated PCC 0.988).
- Gate (device TtRMSNorm): PASS, pcc_swap_out 0.999999, rel_l2_swap_out 0.001693, step rel 0.002118, ratio
  [0.9956, 1.0012].

Re-run
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p/tests/bringup/test_swap_full_dense_01_attn_norm.py

## C.full_dense.attention test (attempt 1), 2026-09-26

What was done
- Replaced the rendered `run_component_test` call in `tests/bringup/test_c_full_dense_attention.py` with the
  gemma4_a4b_d_p `test_c_global_attention.py` pattern: gated PCC (spec 0.99) plus asserted finite output, rel L2 <= 0.015
  over the whole chunk and over the first 128 rows, per-token norm ratio in [0.97, 1.03]. Records
  `rel_l2_attention_L00`, `rel_l2_head_rows_attention_L00`, `row_norm_ratio_{min,max}_attention_L00` (informational).

Why
- Golden s4096 chunk 1 (start 2048). On it PCC passes nearly every bug: RoPE from 0 (0.99968, rel 0.025), non-causal
  (0.99960, rel 0.0285, first rows 0.047), scale 128**-0.5 (0.99931, rel 0.040), no value scale (0.9986), no KV prefix
  (0.9932). Gemma's rel 0.03 would pass the first two, hence 0.015.
- Device noise estimate (CPU sim): reference rel 0.0017; bf16 rounding of act/weights/q/k/v/P 0.0017; bfp8 qkv/o
  weights 0.0021; + bfp8 KV 0.0022. Gemma-4 device attention was rel 0.005-0.008. Full table in the test docstring;
  mutation scripts were in /tmp (not kept).

Results
- BRINGUP_IMPL=reference: PASS (pcc 0.999999, rel 0.001697, first-128 0.001716, ratio [0.9992, 1.0008]).
- BRINGUP_IMPL=stub: FAIL (PCC).
- Default mode: FAIL, NotImplementedError (no device attention yet; implement step).

Notes for implement
- The KV prefix arrives in `dctx.extra["state_prefix"]` (bf16 key [4, 4096, 192], value [4, 4096, 128], the full rung
  state: use only [:, :prefix_len]; value already x 0.707);
  RoPE cos/sin must be at positions start..start+S (a from-0 table fails the rel checks).

Re-run
    PYTHONPATH=$PWD BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p/tests/bringup/test_c_full_dense_attention.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p/tests/bringup/test_c_full_dense_attention.py

## C.full_dense.attention implement (attempt 1), 2026-09-26

What was done
- `tt/attention.py`: `TtFullAttention` + `TtKVCacheFull` (from gemma4_a4b_d_p TtGlobalAttention / TtKVCacheGlobal).
  Chip r: Q heads 16r..16r+15, KV head r. Fused per-chip qkv [4096, 16*192 + 192 + 192] bf16 built from
  `reference/weights.qkv_weight` (dequantized per TP rank); V rows x 0.707 and zero-padded 128 -> 192. o_proj
  row-parallel [16*192, 4096] per chip with zero rows for the V pad dims (no slice after SDPA). Partial RoPE: slice
  dims [0:64] -> `rotary_embedding` -> concat with [64:192]; cos/sin [1, 1, max_seq, 64] built once at load for the
  longest rung/target seq (56320), sliced on device per chunk. Cache paged-shaped [nb, 1, 64, 192] per chip, identity
  page table resident, per-chunk page-table slice cut on device; K and V both 192 wide, `to_torch` returns V[..., :128].
  SDPA causal (chunk 0) / chunked (later), scale 192^-0.5, preset A (HiFi2, fp32 acc off, approx exp, q256/k256; env
  `MIMO_SDPA_CFG=base` for HiFi4 + fp32, q128/k128). qkv and o_proj matmuls HiFi4 + fp32 acc. `ttnn.all_reduce(cluster_axis=1)`.
- `bringup/hooks.py`: `_attention_module`, `_new_kv_cache`, `_attention_host_fn`; `device_component("attention")`
  builds a fresh device cache holding the golden prefix per call; `DEVICE_STEPS["full_dense"]` now includes
  `attention`; `_HybridState` keeps device KV caches for device-attention layers, `HybridDeviceModel.layer` passes
  it as `ctx.extra["dev_cache"]`. Sliding layers raise NotImplementedError in `_attention_module`.

Gotchas
- The chunked SDPA binding (noconvert `scale`) rejects 192^-0.5 as a Python double: "incompatible function arguments"
  with matching types. Scale is rounded to fp32 in the module (known_issues Proposed).
- Forward has no host transfer; the harness wrapper (`_attention_host_fn`) does the host in/out.
- The start == 0 path (plain causal SDPA) is not exercised by this gate (golden chunk 1); the ladder exercises it.

Result
- Gate: pcc_attention_L00 0.999987, rel_l2 0.005136, first-128 rel 0.005128, row_norm_ratio [0.9938, 1.0073]. PASS.

Re-run
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p/tests/bringup/test_c_full_dense_attention.py

## S.full_dense.02 test (attempt 1), 2026-09-26

What was done
- Replaced the rendered `run_swap_test` call in `tests/bringup/test_swap_full_dense_02_attention.py` with the swap 01
  pattern plus the gemma4_a4b_d_p swap_sliding_02 head-row check: gated pcc_swap_out (0.98) plus asserted extras:
  block out finite, rel L2 <= 0.01 whole and first 128 rows; each swapped step PCC >= 0.99, per-token norm ratio in
  [0.97, 1.03], rel L2 <= 0.03 (attn_norm) / <= 0.015 (attention, whole and first 128 rows).

Why
- Attention limits copied from the C.full_dense.attention test (RoPE-from-0, non-causal and scale bugs pass PCC and
  rel 0.03 on this golden). Block-out limit 0.01 kept (dense MLP, no MoE amplification: device 0.0027).

Results
- BRINGUP_IMPL=reference: PASS (out rel 0.0017). BRINGUP_IMPL=stub: FAIL (all checks).
- Gate (device): PASS, pcc_swap_out 0.999996, out rel 0.00274, attention rel 0.00519 / first-128 0.00517,
  ratio [0.9938, 1.0061].

Re-run
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p/tests/bringup/test_swap_full_dense_02_attention.py

## C.full_dense.attn_residual test (attempt 1), 2026-09-26

What was done
- Replaced the rendered `run_component_test` call in `tests/bringup/test_c_full_dense_attn_residual.py` with the
  explicit pattern (as test_c_full_dense_attn_norm.py / gemma4 ffn_residual): gated pcc_attn_residual_L00 (0.99) plus
  asserted extras: output size, finite, rel L2 <= 0.01, per-token norm ratio in [0.99, 1.01].

Why
- Step is `h_mid = in + attn_out` (mimo_ref.py). Mutations measured on the golden (PCC / rel): fp32 ref 0.999998 /
  0.0021, bf16 add 0.999997 / 0.0023; 2x 0.999998 / 1.0; single zeroed row 0.9998 / 0.019; last 32 rows zeroed 0.992 /
  0.124; last 32 columns zeroed 0.991 / 0.133 (ratio min 0.987, passes the Gemma [0.97, 1.03]). All pass PCC 0.99, so
  the rel L2 and ratio limits were tightened from the Gemma template's 0.03 / [0.97, 1.03] to 0.01 / [0.99, 1.01].
- Implication for implement: keep the add in bf16 (or fp32); bfp8 on the residual stream is not budgeted.

Results
- BRINGUP_IMPL=reference: PASS (rel 0.00207, ratio [0.9991, 1.0007]). BRINGUP_IMPL=stub: FAIL (PCC).
- Default (device) gate: FAIL with NotImplementedError (no device module for attn_residual yet; implement step).

Re-run
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p/tests/bringup/test_c_full_dense_attn_residual.py

## C.full_dense.attn_residual implement (attempt 1), 2026-09-26

What was done
- `tt/residual.py`: `TtResidualAdd` (from gemma4_a4b_d_p/tt/residual.py, no scale): `ttnn.add(a, b)` on replicated
  [1, 1, S, 4096] bf16 TILE, bf16 output in DRAM, no CCL.
- `bringup/hooks.py`: `_RESIDUAL_STEPS = {"attn_residual"}`, `_residual_host_fn` (two host inputs -> device ->
  add -> chip 0 copy to host); `device_component` returns it; `DEVICE_STEPS["full_dense"]` now includes
  `attn_residual`; `HybridDeviceModel` adds it to the per-layer overrides.

Gotchas
- The log shows `FAIL pcc_attn_residual_L00: pcc=0.000000` first: that is the precompile collect pass (ops are not
  run); the real pass follows with the true value.
- `mlp_residual` / `ffn_residual` (same a + b) are not swapped yet; add them to `_RESIDUAL_STEPS` when their tasks come.

Result
- Gate: pcc_attn_residual_L00 0.999997, rel_l2 0.002393, row_norm_ratio [0.9996, 1.0012]. PASS.

Re-run
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p/tests/bringup/test_c_full_dense_attn_residual.py

## S.full_dense.03 test (attempt 1), 2026-09-26

What was done
- Replaced the rendered `run_swap_test` call in `tests/bringup/test_swap_full_dense_03_attn_residual.py` with the
  swap 02 pattern, SWAPPED = attn_norm, attention, attn_residual. Gated pcc_swap_out (0.98) plus asserted extras:
  block out finite, rel L2 <= 0.01 whole and first 128 rows; each swapped step PCC >= 0.99 and own rel L2 <= 0.03 /
  0.015 / 0.01, per-token norm ratio [0.97, 1.03] (attn_norm, attention) or [0.99, 1.01] (attn_residual);
  attention and attn_residual also checked on the first 128 rows.

Why
- attn_residual limits (rel 0.01, ratio [0.99, 1.01]) copied from test_c_full_dense_attn_residual.py; h_mid here
  also carries the device attn_norm + attention error, measured 0.0042 (2.4x headroom).

Results
- BRINGUP_IMPL=reference: PASS (out rel 0.0017, h_mid rel 0.0017). BRINGUP_IMPL=stub: FAIL (every check).
- Gate (device): PASS, pcc_swap_out 0.999996, out rel 0.00294 / first-128 0.00296, attention rel 0.00519,
  attn_residual rel 0.00420 / first-128 0.00424, ratio [0.9966, 1.0050].

Re-run
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p/tests/bringup/test_swap_full_dense_03_attn_residual.py

## C.full_dense.ffn_norm test (attempt 1), 2026-09-26

What was done
- Replaced the rendered `run_component_test` call in `tests/bringup/test_c_full_dense_ffn_norm.py` with the same body
  as `test_c_full_dense_attn_norm.py` (STEP = ffn_norm): gated PCC (spec 0.99) plus finite output, rel L2 <= 0.03 and
  per-token norm ratio in [0.97, 1.03]. Records `rel_l2_ffn_norm_L00`, `row_norm_ratio_{min,max}_ffn_norm_L00`.

Why
- Measured on the golden (CPU, h_mid [2048, 4096] -> ffn_norm bf16, post_attention_layernorm w mean 0.020):
  reference fp32 PCC 0.999997 / rel 0.0023 / ratio [0.9953, 1.0045]; bf16 math rel 0.0033 / ratio [0.9919, 1.0054]
  (not bit-equal to the golden here, unlike attn_norm); sum-instead-of-mean PCC 0.999997 but rel 0.98; eps 1e-2 PCC
  0.9925 but rel 0.84; `1 + w` rel 7.4; the attn_norm weight PCC 0.17. So a PCC-only gate would let sum and eps bugs through.

Results
- BRINGUP_IMPL=reference: pass (rel 0.0023). BRINGUP_IMPL=stub: fail (PCC 0). Default (device): pass, PCC 0.999996,
  rel 0.0029, ratio [0.9933, 1.0053].

Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p/tests/bringup/test_c_full_dense_ffn_norm.py`

## S.full_dense.04 test (attempt 1), 2026-09-26

What was done
- Replaced the rendered `run_swap_test` call in `tests/bringup/test_swap_full_dense_04_ffn_norm.py` with the body of
  `test_swap_full_dense_03_attn_residual.py`, adding ffn_norm: step rel L2 <= 0.03 and per-token norm ratio in
  [0.97, 1.03] (the C.full_dense.ffn_norm limits). Block out stays at rel L2 <= 0.01, whole chunk and first 128 rows.

Why
- pcc_swap_out alone is weak: a zero ffn_norm gives mlp(0) = 0, so out = h_mid, and the residual dominates out.

Measured
- Device (gate): pcc_swap_out 0.999996, block out rel 0.0030 (first 128 rows 0.0030); ffn_norm pcc 0.999991 / rel
  0.0043 / ratio [0.9893, 1.0071]; attention rel 0.0052, attn_residual rel 0.0042, attn_norm rel 0.0021.
- BRINGUP_IMPL=reference: passes (all rel ~0.0016). BRINGUP_IMPL=stub: fails (pcc 0, rel 1.0 on every check).
- The first metric block in the log with pcc=0.000000 comes from the precompile collect pass (comp_pcc stubbed), not the real run.

Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p/tests/bringup/test_swap_full_dense_04_ffn_norm.py`

## C.full_dense.mlp test (attempt 1), 2026-09-26

What was done
- Replaced the rendered `run_component_test` call in `tests/bringup/test_c_full_dense_mlp.py` with the explicit body
  (as `test_c_full_dense_ffn_norm.py`): gated PCC (spec 0.99) plus finite output, rel L2 <= 0.015, per-token norm ratio
  in [0.98, 1.02], worst per-token rel L2 <= 0.05. Records `rel_l2_mlp_L00`, `row_norm_ratio_{min,max}_mlp_L00`,
  `worst_row_rel_l2_mlp_L00`.

Why (CPU mutations on the golden, ffn_norm [2048, 4096] -> mlp_out, row norms 0.59-1.31)
- Device-like noise: fp32 ref rel 0.0017; bf16 0.0019; bfp8 weights + activations 0.0064 / ratio [0.9905, 1.0082] /
  worst row 0.0126; HiFi2-like 5-bit mantissa weights 0.0141 / [0.981, 1.004].
- Bugs that pass PCC 0.99: 1.02x output (rel 0.020), 2x, all_reduce x4, row 0 or last row zeroed (rel 0.021 / 0.019:
  the Gemma limit of 0.03 would pass them, the norm ratio catches them), one row x1.1 (only worst-row and ratio catch
  it), one TP shard missing / doubled (rel 0.26), last 32 rows zeroed (0.125).
- Tighter than the Gemma template (0.03, [0.97, 1.03]) because this MLP's noise floor is lower; a HiFi2-style build
  sits near the rel limit (known issue: use HiFi4 + fp32 acc, as components.yaml says).

Results
- BRINGUP_IMPL=reference: pass (rel 0.0017, ratio [0.9986, 1.0015], worst row 0.0024). BRINGUP_IMPL=stub: fail (PCC).
- Default (device): fails with NotImplementedError "no device module for mlp yet" (implement step next).

Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p/tests/bringup/test_c_full_dense_mlp.py`

## C.full_dense.mlp implement (attempt 1), 2026-09-26

What was done
- `tt/mlp.py:TtDenseMLP` (from gemma4_a4b_d_p TtDenseMLP): gate/up column-parallel (ShardTensorToMesh dim -1 of
  [H, 16384], 4096 per chip, no pad), `ttnn.linear(..., activation="silu")` for gate, `ttnn.mul`, down row-parallel
  (dim -2 of [16384, H]), one `ttnn.all_reduce(cluster_axis=1)`. bf16 weights (plan.yaml), HiFi4 + fp32 acc.
- hooks.py: `_mlp_module` (fp8 + block scale dequantized via `reference.weights.fp8_weight`), `device_component`
  step "mlp", and "mlp" added to `DEVICE_STEPS["full_dense"]` so the hybrid `device_model` runs it too.

Results (gate): pcc_mlp_L00 0.999995, rel L2 0.0038, row norm ratio [1.0001, 1.0046], worst row rel 0.0055.
The pcc=0.000000 line in the log is the precompile collect pass (comp_pcc stub), not the real run.

Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p/tests/bringup/test_c_full_dense_mlp.py`

## S.full_dense.05 test (attempt 1), 2026-09-26

What was done
- Replaced the rendered `run_swap_test` call in `tests/bringup/test_swap_full_dense_05_mlp.py` with the swap-04 body
  plus mlp: mlp vs golden rel L2 <= 0.015 and per-token norm ratio in [0.98, 1.02] (the mlp component test's limits).
- New check: the swapped mlp against the CPU mlp run on the same device ffn_norm output (rel L2 <= 0.015, ratio
  [0.98, 1.02], worst row rel <= 0.05), recorded as `*_vs_cpu_swap_mlp_out`. Isolates the device mlp from upstream error,
  so mlp scale / TP-shard / row bugs (which PCC and block out miss) fail here. Generic via `CPU_SAME_INPUT` (stateless
  steps only).
- Block out rel L2 <= 0.01 (whole and first 128 rows) kept.

Results
- Device (default impl): pcc_swap_out 0.999995, out rel 0.0034 / first rows 0.0035; mlp vs golden rel 0.0046 ratio
  [0.9981, 1.0081]; mlp vs CPU same inputs rel 0.0033 ratio [1.0004, 1.0044] worst row 0.0048. PASS.
- BRINGUP_IMPL=reference: out 0.999999 / rel 0.0017, mlp rel 0.0017, vs CPU 0. PASS.
- BRINGUP_IMPL=stub: fails every check (out PCC 0, rel 1.0).

Next
- `mlp_residual` is still CPU; it is not in the swap list yet.

Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p/tests/bringup/test_swap_full_dense_05_mlp.py`

## C.full_dense.mlp_residual test (attempt 1), 2026-09-26

What was done
- Replaced the rendered `run_component_test` call in `tests/bringup/test_c_full_dense_mlp_residual.py` with the
  attn_residual test body (STEP = mlp_residual): gated pcc_mlp_residual_L00 (0.99) plus asserted extras: output size,
  finite, rel L2 <= 0.01, per-token norm ratio in [0.99, 1.01].

Why
- Step is `out = h_mid + mlp_out` (mimo_ref.py). Mutations on the golden (PCC / rel): fp32 ref 0.999998 / 0.0021,
  bf16 add 0.999996 / 0.0028; 2x 0.999998 / 1.0; single zeroed row 0.9998 / 0.020-0.022; last 32 rows zeroed 0.992 /
  0.125; all pass PCC 0.99, caught by rel L2 / ratio. Same limits as attn_residual (the stream profile matches).
- Implement: `tt/residual.py:TtResidualAdd` (already used for attn_residual) fits; keep the add bf16 or better.

Results
- BRINGUP_IMPL=reference: PASS (rel 0.00214, ratio [0.9991, 1.0009]). BRINGUP_IMPL=stub: FAIL (PCC).
- Default (device) gate: FAIL with NotImplementedError (no device module for mlp_residual yet; implement step).

Re-run
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p/tests/bringup/test_c_full_dense_mlp_residual.py

## C.full_dense.mlp_residual implement (attempt 1)
- Reused `tt/residual.py:TtResidualAdd` (replicated bf16 `ttnn.add`, DRAM, no CCL). Only change is in `bringup/hooks.py`: added `mlp_residual` to
  `_RESIDUAL_STEPS` (so `device_component` routes it to `_residual_host_fn`) and to `DEVICE_STEPS["full_dense"]` (hybrid `device_model`).
- Gate: pcc_mlp_residual_L00 = 0.999997, rel_l2 0.00277, row norm ratio [1.0003, 1.0023]. A `FAIL ... pcc=0.000000` line from the
  precompile collect pass comes before the real pass and can be ignored.
- `ffn_norm` is handled by `device_component` but is not in `DEVICE_STEPS` yet (not this step's job).
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p/tests/bringup/test_c_full_dense_mlp_residual.py`

## S.full_dense.06 test (attempt 1), 2026-09-26

What was done
- Replaced the rendered `run_swap_test` call in `tests/bringup/test_swap_full_dense_06_mlp_residual.py` with the
  swap-05 body plus mlp_residual (whole dense block on device). mlp_residual's output is block `out`.
- mlp_residual vs golden: rel L2 <= 0.01, per-token norm ratio [0.99, 1.01], also on the first 128 rows (the
  mlp_residual component test's limits).
- mlp_residual vs the CPU add on the same device h_mid / mlp_out (`CPU_SAME_INPUT`): rel L2 <= 0.01, ratio
  [0.99, 1.01], worst per-token rel L2 <= 0.02 (catches a single zeroed row, which gives only 0.02 whole-chunk rel).
- mlp check vs CPU and block-out rel L2 <= 0.01 kept from swap 05.

Results
- Device (default impl): pcc_swap_out 0.999993, out rel 0.0042 / first rows 0.0044, ratio [0.9988, 1.0064];
  mlp_residual vs CPU rel 0.0019, ratio [1.0004, 1.0019], worst row 0.0024. PASS.
- BRINGUP_IMPL=reference: PASS (out rel 0.0017, vs CPU 0). BRINGUP_IMPL=stub: FAIL (out PCC 0, rel 1.0).
- The first FAIL / pcc=0 block in each log is the precompile collect pass, not the real run.

Next
- Every full_dense step is now on the device in the swap harness; the device module for mlp_residual already existed.

Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p/tests/bringup/test_swap_full_dense_06_mlp_residual.py`

## C.sliding_moe.attn_norm test (attempt 1), 2026-09-26

What was done
- Replaced the rendered `run_component_test` body of `tests/bringup/test_c_sliding_moe_attn_norm.py` with the
  full_dense attn_norm test body at LAYER = 1 (same RMSNorm: plain `w`, eps 1e-6). Checks: PCC >= 0.99 (gated),
  output finite, rel L2 <= 0.03, per-token norm ratio in [0.97, 1.03].
- Mutations measured on the layer-1 golden on the CPU (layer-1 input_layernorm w in [-0.13, 2.47], a wider range than layer 0):
  fp32 reference rel 0.0024 / ratio [0.9954, 1.0043]; sum-not-mean PCC ~1.0 but rel 0.98; eps 1e-3 PCC 0.998, rel 0.36;
  eps 1e-2 PCC 0.993, rel 0.74; 1 + w PCC 0.78; one zeroed row PCC ~1.0, ratio min 0; last 32 rows zeroed PCC 0.993,
  rel 0.12. Every one fails at least one assert, so the limits stay as they were.

Results
- BRINGUP_IMPL=reference: PASS (PCC 0.999997, rel 0.00236). BRINGUP_IMPL=stub: FAIL (PCC).
- Default (device) run already PASSES: PCC 0.999996, rel 0.00295, ratio [0.9934, 1.0058]. The norm module from
  full_dense serves this step already. The first `FAIL ... pcc=0.000000` line comes from the precompile pass.

Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p/tests/bringup/test_c_sliding_moe_attn_norm.py`

## S.sliding_moe.01 test (attempt 1), 2026-09-26

What was done
- Replaced the rendered `run_swap_test` call in `tests/bringup/test_swap_sliding_moe_01_attn_norm.py` with the
  body of `test_swap_full_dense_01_attn_norm.py` (BLOCK_TYPE = sliding_moe, layer 1). Gated pcc_swap_out (0.98). Asserted
  extras: attn_norm vs golden PCC >= 0.99, rel L2 <= 0.03, per-token norm ratio in [0.97, 1.03]; block out finite with
  rel L2 <= 0.01.

Why
- CPU mutations measured with a temporary env hook, since removed: x1.05, eps 1e-2 and 5% noise all pass the 0.98 gate.
  x1.05 gives out rel 0.0100, step rel 0.050 and ratio 1.05; eps gives step rel 2.9; noise gives step rel 0.050. Only the
  extras catch these three. Sum-instead-of-mean, a missing weight and a zeroed last row fail the gate as well.
- Unlike layer 0, the zero stub fails the gate here (out PCC 0, rel 16.9).

Results
- BRINGUP_IMPL=reference: PASS (out 0.999996 / rel 0.0030, step rel 0.0024). BRINGUP_IMPL=stub: FAIL.
- Device (default): PASS, pcc_swap_out 0.999995, out rel 0.0032, attn_norm rel 0.0030, ratio [0.9934, 1.0058]; router
  trail PCC 0.9991, experts_out 0.9999. The first FAIL/pcc=0 block in each log is the precompile pass.

Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p/tests/bringup/test_swap_sliding_moe_01_attn_norm.py`

## C.sliding_moe.attention test (attempt 1), 2026-09-26

What was done
- Replaced the rendered `run_component_test` call in `tests/bringup/test_c_sliding_moe_attention.py` with the
  full_dense attention test body at LAYER = 1. Gated PCC (spec 0.99). Asserted extras: finite output; rel L2 <= 0.02 over
  the whole chunk and over the first 128 rows; per-token norm ratio in [0.95, 1.05]; worst per-token rel L2 <= 0.08;
  and a window discriminator: rel L2 to the CPU reference at window 128 must be below rel L2 to the same reference
  at windows 127 and 129 (the test sets `ref.cfg.sliding_window` and restores it after). Records `rel_l2_*`,
  `row_norm_ratio_*`, `worst_row_rel_l2_*` and `rel_l2_vs_cpu_window_attention_L01` (informational).

Why (CPU measurements on the golden; scripts were in /tmp and are not kept)
- On layer 1 the sink takes almost all of the softmax mass (key scores median -26 while the sink is 0.56-1.14).
  attn_out row norms are 0.007-0.057, so most bugs are loud: RoPE from 0, theta 1e7, no window, non-causal, no sink and
  a wrong GQA map all fail PCC.
- Bugs that pass PCC but fail the size checks: sink zero (rel 1.08), sink negated (3.3), no value scale (0.41),
  window 64 (0.25), no KV prefix (0.085 whole, 0.40 first rows), scale 128^-0.5 (0.90), x1.02 (0.020), zeroed rows
  (caught by ratio and worst row).
- The fp32 reference vs golden is already rel 0.0047 with ratio [0.983, 1.015], and the device-noise estimate is rel
  0.0072, ratio [0.975, 1.022], worst row 0.025. So the Gemma/full_dense ratio [0.97, 1.03] and rel 0.015 were
  loosened to [0.95, 1.05] and 0.02. The smallest structural bug the size checks catch is 0.085 (x1.02 sits at 0.020).
- A window off by one (127/129) scores rel 0.010, inside the noise, so it gets the discriminator check. Simulated
  device output at 127/128/129 (with or without 1% extra noise) is always closest to its own window.

Results
- BRINGUP_IMPL=reference: PASS (pcc 0.999989, rel 0.0047, first 128 rows 0.0042, ratio [0.9825, 1.0147], worst row
  0.0176; vs CPU w128 0, w127 0.0085, w129 0.0093). BRINGUP_IMPL=stub: FAIL (PCC).
- Default (device): FAIL, NotImplementedError "no device sliding attention yet" (implement step next).
- The first `FAIL ... pcc=0.000000` line in each log comes from the precompile pass.

Notes for implement
- Mask: key j visible to query i iff i - 128 < j <= i. Only the last 127 rows of the KV prefix are needed
  (`dctx.extra["state_prefix"]`, key [8, 4096, 192], value [8, 4096, 128], already x 0.707; use [:, :prefix_len]).
- Sink: an extra softmax column per q head with logit `attention_sink_bias[h]`, probability dropped. Output scales
  like exp(-sink), so sink precision maps 1:1 onto relative output error.
- RoPE theta 1e4 on dims [0:64], positions start..start+S.

Re-run
    PYTHONPATH=$PWD BRINGUP_IMPL=reference scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p/tests/bringup/test_c_sliding_moe_attention.py
    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p/tests/bringup/test_c_sliding_moe_attention.py

## C.sliding_moe.attention implement (attempt 1), 2026-09-26

What was done
- `tt/attention.py`: new `TtKVCacheSliding` (contiguous [1, 8, max_seq, 192] sharded by head: KV heads 2r, 2r+1 on
  chip r, V zero-padded 128 -> 192) and `TtSlidingAttention` (per chip: fused qkv [4096, 16*192 + 2*192 + 2*192]
  HiFi4 -> nlp_create_qkv_heads -> partial RoPE dims 0-63, theta 1e4, tables built once for max_seq and sliced per chunk
  -> fill_cache at `start` -> the previous 128 K/V rows sliced back from the cache and concatenated, Q front-padded by 128
  -> SDPA(is_causal, sliding_window_size=128, attention_sink [1, 16, 1, 1]) -> drop the pad rows -> concat_heads ->
  row-parallel o_proj -> all_reduce(cluster_axis=1)). Weight building is now shared (`_tp_qkv_o`, general nkv per chip);
  `TtFullAttention` output is unchanged (full_dense attention test still PCC 0.999987, rel 0.0051).
- `hooks.py`: `_attention_module` builds the sliding module for sliding layers (swa theta, sink bias);
  `_new_kv_cache` picks `TtKVCacheSliding`; `DEVICE_STEPS["sliding_moe"] = {attn_norm, attention}` for the hybrid.

Decisions and gotchas
- The SDPA kernel folds the sink with a truncated-bf16 scale (192^-0.5 -> 0.07178). The row max sits ~28 logits below
  the sink, so the output was 4-14% too large. Fix: SDPA scale 2^-4 (exact), the factor 1.1547 folded into the Q rows of
  wqkv, sink / 2^-4 (exact in bf16). Known issue proposed.
- The sliding SDPA uses its own preset, `MIMO_SLIDING_SDPA_CFG` (default `base`: HiFi4, fp32 dest acc, exact exp,
  q128/k128). SDPA-only probe (vs CPU chunk_attention, same inputs): A rel 0.017 mean ratio 1.037; HiFi4 without fp32
  acc 0.016 / 1.032; HiFi2 with fp32 0.014 / 1.056; base 0.005 / 1.004. The sink magnifies QK score errors. It is
  non-streaming, but the window is only 128 keys. Full layers keep `MIMO_SDPA_CFG` (default A).
- Gate result: PCC 0.999972, rel 0.0086, first 128 rows 0.0082, norm ratio [0.9747, 1.0209], worst row 0.0255;
  vs CPU window 128: 0.00708, 127: 0.00802, 129: 0.01419. The margin to window 127 is small (0.0009). Do not add noise
  to this path (bf8 KV, HiFi2) without re-checking it.
- Probes ran from a temporary test file in `tt/` (deleted).

Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p/tests/bringup/test_c_sliding_moe_attention.py`

## S.sliding_moe.02 test (attempt 1), 2026-09-26

What was done
- Replaced the rendered `run_swap_test` call in `tests/bringup/test_swap_sliding_moe_02_attention.py` with the body of
  `test_swap_full_dense_02_attention.py` (BLOCK_TYPE sliding_moe, layer 1), using the C.sliding_moe.* limits. Gated
  pcc_swap_out (0.98). Asserted extras: attn_norm rel <= 0.03 and ratio [0.97, 1.03]; attention rel <= 0.02 (whole chunk
  and first 128 rows), ratio [0.95, 1.05], worst row <= 0.08; attention vs the CPU attention on the same device
  attn_norm input rel <= 0.015; a window discriminator (closer to CPU w128 than to w127 and w129); block out finite with
  rel L2 <= 0.01 (whole chunk and first 128 rows).

Why
- In the precompile pass the attention output is zero, and block out still scores rel 0.0142. The sink keeps attn_out
  small next to the residual, so Gemma's 0.02 block-out limit is too loose. The device scores 0.0031, so 0.01 keeps 3x headroom.
- Attention vs the golden is 0.0145 in the swap (component test 0.0086): the device attn_norm error (0.003) is
  amplified through the sink. Against the CPU attention on the same input it is 0.0070, so that check gets the
  tighter 0.015 limit.

Results
- BRINGUP_IMPL=reference: PASS (out 0.999996 / rel 0.0030; attention rel 0.0038). BRINGUP_IMPL=stub: FAIL (all checks).
- Device (default): PASS. pcc_swap_out 0.999995, out rel 0.0031 / first rows 0.0029, attention rel 0.0145 / 0.0128,
  ratio [0.9882, 1.0420], worst row 0.042. vs CPU w128 0.0070, w127 0.0081, w129 0.0141. The margin to w127 is only 0.0011.
- The first pcc=0 block in each log is the precompile pass.

Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p/tests/bringup/test_swap_sliding_moe_02_attention.py`

## C.sliding_moe.attn_residual test (attempt 1), 2026-09-26

What was done
- Replaced the rendered `run_component_test` call in `tests/bringup/test_c_sliding_moe_attn_residual.py` with the body
  of `test_c_full_dense_attn_residual.py` (layer 1): gated pcc_attn_residual_L01 (0.99), asserted output size, finite,
  rel L2 <= 0.01, per-token norm ratio [0.99, 1.01]. Added two checks on the attention term, delta = out - in:
  coefficient <delta, attn_out> / ||attn_out||^2 in [0.95, 1.05], and ||delta - attn_out|| / ||attn_out|| <= 0.3.

Why
- At layer 1 ||in|| 79.05, ||attn_out|| 0.88 (sink-dominated). Measured on the golden (PCC / rel): attn_out dropped
  0.99993 / 0.0113 (it only fails rel by 0.0013); in + 0.5 attn_out 0.99996 / 0.0060 and ratio [0.989, 0.999]: passes every
  full_dense check; attn_out shifted one row 0.99996 / 0.0059: passes. The delta checks catch all three (coef 0 / 0.5 / 0.88,
  rel 1.0 / 0.5 / 0.49). The bf16 add gives coef 0.9986 and rel 0.15, from output rounding only.

Results
- BRINGUP_IMPL=reference: PASS (rel 0.00236, coef 1.0000, attn rel 0.0000). BRINGUP_IMPL=stub: FAIL (PCC).
- Default (device): PASS already, because `TtResidualAdd` is generic: pcc 0.999996, rel 0.00291, ratio [0.9990, 1.0015],
  coef 1.0200, attn rel 0.1554. The coef margin is 0.03. Keep the add bf16 or fp32 (fp8-like rounding gives rel 0.032).

Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p/tests/bringup/test_c_sliding_moe_attn_residual.py`

## S.sliding_moe.03 test (attempt 1), 2026-09-26

What was done
- Replaced the rendered `run_swap_test` call in `tests/bringup/test_swap_sliding_moe_03_attn_residual.py` with the
  body of `test_swap_sliding_moe_02_attention.py` (every swap-02 check kept: attn_norm, attention, window
  discriminator, block out rel <= 0.01). Added attn_residual (h_mid) checks: rel L2 <= 0.01 (whole chunk and first 128 rows),
  ratio [0.99, 1.01], plus the component test's attention-term checks on delta = h_mid - in against the attn_out the
  swap actually fed in: coef in [0.95, 1.05], ||delta - attn_out|| / ||attn_out|| <= 0.3.

Why
- The sink keeps attn_out ~1% of ||in||, so a dropped, halved or shifted addend passes the whole-output checks
  (C.sliding_moe.attn_residual). The delta check uses the device attn_out, so the attention error does not affect it.

Results
- BRINGUP_IMPL=reference: PASS (h_mid rel 0.0024, coef 1.0000). BRINGUP_IMPL=stub: FAIL (every check).
- Device (gate): PASS. pcc_swap_out 0.999993, out rel 0.0037; h_mid rel 0.0029, ratio [0.9991, 1.0016], coef 1.0198,
  attn rel 0.154; the attention numbers match swap 02 (the w127 margin is still 0.0011).
- When attn_out is all zero, the attn-rel metric divides by a 1e-30 clamp and prints a huge number. That is expected: coef 0 fails as well.

Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p/tests/bringup/test_swap_sliding_moe_03_attn_residual.py`

## C.sliding_moe.ffn_norm test (attempt 1), 2026-09-26

What was done
- Replaced the rendered `run_component_test` call in `tests/bringup/test_c_sliding_moe_ffn_norm.py` with the body of
  `test_c_full_dense_ffn_norm.py` (LAYER = 1): gated PCC (spec 0.99), plus finite output, rel L2 <= 0.03 and per-token
  norm ratio in [0.97, 1.03]. Records `rel_l2_ffn_norm_L01` and `row_norm_ratio_{min,max}_ffn_norm_L01`.

Why
- Measured on the layer-1 golden (CPU; post_attention_layernorm w in [-0.012, 2.33], mean 0.176): fp32 reference rel
  0.0024 / ratio [0.9970, 1.0031]; sum instead of mean passes PCC (0.999997) but rel 0.98; eps 1e-3 and 1e-2 pass PCC
  (0.997, 0.993) but rel 0.35 / 0.74; one zeroed row PCC 0.9998 but ratio min 0; last 32 rows zeroed PCC 0.992 but
  rel 0.13. With eps 1e-5 the rel L2 is 0.0068, a harmless difference that no check catches.

Results
- BRINGUP_IMPL=reference: PASS (rel 0.0024). BRINGUP_IMPL=stub: FAIL (PCC 0).
- Default (device): PASS already, because the generic norm module in hooks handles ffn_norm at layer 1. pcc_ffn_norm_L01
  0.999996, rel 0.0030, ratio [0.9954, 1.0029]. The first pcc=0 line in each log is the precompile pass.

Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p/tests/bringup/test_c_sliding_moe_ffn_norm.py`

## S.sliding_moe.04 test (attempt 1), 2026-09-26

What was done
- Replaced the rendered `run_swap_test` call in `tests/bringup/test_swap_sliding_moe_04_ffn_norm.py` with the body of
  `test_swap_sliding_moe_03_attn_residual.py`. All swap-03 checks are kept. Added ffn_norm to SWAPPED with the
  component-test limits: rel L2 <= 0.03 and per-token norm ratio [0.97, 1.03]. Block out rel L2 stays <= 0.01.

Why
- The ffn_norm component test measured these limits: sum instead of mean, eps 1e-3 / 1e-2, 1 + w, and zeroed rows all
  fail them. The residual dominates block out, so pcc_swap_out on its own would not catch a bad ffn_norm.

Results
- BRINGUP_IMPL=reference: PASS (out rel 0.0030, ffn_norm rel 0.0024). BRINGUP_IMPL=stub: FAIL (every check).
- Device (gate): PASS. pcc_swap_out 0.999993, out rel 0.0037 / first 128 rows 0.0029; ffn_norm rel 0.0035, ratio
  [0.9941, 1.0038]. The CPU MoE on the device ffn_norm adds no visible block-out error (same 0.0037 as swap 03).
  The attention numbers match swap 03; the w127 margin is still 0.0011.

Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p/tests/bringup/test_swap_sliding_moe_04_ffn_norm.py`

## C.sliding_moe.router test (attempt 1), 2026-09-26

What was done
- Replaced the rendered `run_component_test` call in `tests/bringup/test_c_sliding_moe_router.py` with the Gemma
  router-test body (gemma4_a4b_d_p `test_c_sliding_router.py`), adapted to MiMo's router: sigmoid, noaux_tc,
  n_group 1, weights renormalized to sum 1, no per_expert_scale. The gated metric is still PCC >= 0.99. The test also
  asserts exactly 8 nonzeros per row, non-negative weights, selection overlap >= 0.985, matched-row weight rel L2
  <= 0.005, and every row sum within 0.01 of 1.
- The mutation measurements are in the test docstring. They came from a CPU-only tt-probe: golden ffn_norm, layer-1 gate weight and bias, route() with mutations. The probe file was deleted because it was outside this step's paths.

Why
- The PCC gate misses these bugs: weights x1.02 (0.9996), logits x1.1 (0.9965), the last row or the last 32 rows
  zeroed (0.9991 / 0.9909). Other checks catch them: the matched-row rel L2, the row-sum check or the nnz check.
- Overlap limit 0.985: the CPU reference scores 0.99878, logits rounded to bf16 score 0.990, and half the bias scores
  0.950. The PCC gate by itself already implies about 0.98.

Gotchas for implement
- The choice score (sigmoid + bias, around 2.0) must be fp32 or bias-recentred. Rounding the bias to bf16 fails the PCC
  gate (0.970). The logits must be fp32 too: bf16 logits score 0.9943, which leaves almost no margin. Take the weights from the
  unbiased sigmoid and renormalize them; routed_scaling_factor is null, which means 1.0.
- The CPU reference itself scores only PCC 0.99933 on the bf16 golden input: 20 rows flip on near ties.
- The `FAIL pcc_router_L01: pcc=0.000000` line during collection comes from the precompile plugin stub. It is not a test result.

Results
- BRINGUP_IMPL=reference: PASS (PCC 0.999328, overlap 0.99878, matched rel 0.00159, row sums 1.0).
- BRINGUP_IMPL=stub: FAIL (PCC 0).
- Device (gate): FAIL with NotImplementedError. No device router module exists yet; the implement step writes it.

Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p/tests/bringup/test_c_sliding_moe_router.py`

## C.sliding_moe.router implement (attempt 1), 2026-09-26

What was done
- Added `tt/router.py:TtRouter`. It is replicated with no CCL. It does an fp32 `ttnn.linear` (weight [4096, 256] fp32, HiFi4 +
  fp32 acc, fp32 out), then `ttnn.sigmoid`, then `ttnn.add` of the fp32 bias [1, 1, 1, 256] (broadcast), then `ttnn.topk`
  (k=8). The weights are `ttnn.gather` of the unbiased sigmoid, divided by their sum; route_scale is 1.0. The dense
  [S, 256] output comes from a bf16 `ttnn.scatter` into a zero tensor. That tensor is built once at load for max chunk 8192
  and sliced on the device per chunk. Returns (dense, idx, weights).
- hooks.py: `_router_module` and `_router_host_fn`; `device_component` handles "router"; "router" is added to
  `DEVICE_STEPS["sliding_moe"]`; the hybrid `device_model` gets a router override. `_max_chunk(spec)` is the largest
  ladder/target chunk.

Decisions and why
- Departed from the components entry (moe_grouped_topk). On this golden, with exact fp32 logits fed in, the fused op
  scores PCC 0.9855 and overlap 0.975, which fails the gate. The cause is its FPU bias add and TF32 sort keys (step 0.002
  near 2.3; the 8th/9th gap median is 0.0017). With the bias recentred by its mean it scores 0.9975. The fp32 SFPU path
  scores 0.99933, the same as the CPU reference. The fused op stays selectable: `MIMO_ROUTER_MODE=fused` (bias
  recentred, full-shape bias built at load).
- Device logits are accurate: rel err 9.7e-5 against the fp32 CPU logits. Precision is lost in selection only.
- No row slicing needed: both modes ran 8192 rows (s16384 chunk) fine. Warm: fp32 2.24 ms, fused 2.01 ms at 8192
  rows; 1.51 / 1.37 ms at 5120.

Gotchas
- `ttnn.topk` returns uint32 indices here (moe_grouped_topk returns uint16). `ttnn.scatter` accepts either.
- tt-probe dir `tests/ttnn/unit_tests/operations/mimo_router/` was deleted after probing.

Results
- Gate PASS: pcc_router_L01 0.999129; nnz 8 per row; selection_overlap 0.99841; matched rows 2022/2048, matched_rel_l2
  0.00102; row sums [0.9976, 1.0020]. The pcc=0 line is the precompile stub.

Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p/tests/bringup/test_c_sliding_moe_router.py`

## S.sliding_moe.05 test (attempt 1), 2026-09-26

What was done
- Replaced the rendered `run_swap_test` call in `tests/bringup/test_swap_sliding_moe_05_router.py` with swap 04's body
  (the golden, the KV prefix, and every check on attn_norm, attention, attn_residual, ffn_norm and block out). Added a router
  branch `_check_router` modelled on gemma4_a4b_d_p `test_swap_sliding_08_router.py` and adapted to MiMo: rows sum to 1
  (absolute, not a ratio), and the component test's limits apply.
- Router checks: PCC >= 0.99, exactly 8 nonzeros per row, no negative weights, row sums 1 +- 0.01. Against the golden:
  overlap >= 0.985 and matched rel <= 0.01. Against the CPU router on the same device ffn_norm (iso): overlap >= 0.99
  and matched rel <= 0.005. Whole-matrix router rel L2 is recorded but not gated (device 0.050 from near-tie flips).

Results
- BRINGUP_IMPL=reference PASS: out 0.999997; router overlap 0.99866 against the golden and 1.0 iso.
- BRINGUP_IMPL=stub FAIL on every check.
- Device gate PASS: pcc_swap_out 0.999993, out rel 0.0037; router pcc 0.99870, overlap 0.99762 against the golden and
  0.99902 iso, matched rel 0.00147 / 0.00157, row sums [0.9980, 1.0020].
- The precompile pass (zero attention) gives router overlap 0.976 against the golden, which the 0.985 limit catches.

Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p/tests/bringup/test_swap_sliding_moe_05_router.py`

## C.sliding_moe.experts test (attempt 1), 2026-09-26

What was done
- Replaced the rendered `run_component_test` call in `tests/bringup/test_c_sliding_moe_experts.py` with the Gemma experts-test
  body (gemma4_a4b_d_p `test_c_sliding_experts.py`), with LAYER set to 1. The gated metric is PCC >= 0.99. The test also asserts a finite output,
  rel L2 <= 0.03, a per-token norm ratio within [0.97, 1.03], and a worst per-token rel L2 <= 0.1.
- Measured the mutations on the golden with a CPU-only tt-probe (golden ffn_norm and router, the reference's layer-1 expert weights).
  The numbers are in the test docstring. The probe dir was deleted.

Why
- PCC misses: drop expert 0 (0.9983), drop the hottest expert 64 (0.9960), drop each token's smallest pair (0.9956),
  drop token 0's top-1 pair (0.99997), last row zeroed (0.99977), 2x output (0.999997), capacity 512/256 (0.998/0.9935). The extra
  checks catch all of these.
- Device-noise estimate: bfp8 weights blocked along the output dim plus bfp8 activations give rel 0.0091, ratio
  [0.986, 1.014] and worst row 0.028. That is below a third of each limit.
- Known gaps (these pass): drop expert 255 (3 tokens), drop one token's smallest pair, a uniform x1.02 scale.

Gotchas for implement
- Routing on this golden: tokens per expert 0..804 (expert 64 = 804). Pairs per chip (64 experts each):
  3381 / 5292 / 3782 / 3929. Size dispatch capacity for the worst case, not the mean.
- A crude LoFi model gives rel 0.048, which would fail. unified_routed_expert_moe hard-codes LoFi for Silu (known issues).
  Measure early. Gemma needed a variant that honours math_fidelity (HiFi2 + fp32 dest).
- The router golden is bf16, rows sum to 1, 8 nonzeros per row. No shared expert.
- The `FAIL pcc_experts_L01: pcc=0.000000` line during collection is the precompile stub.

Results
- BRINGUP_IMPL=reference: PASS (PCC 0.999997, rel 0.0023, ratio [0.9954, 1.0037], worst row 0.0050).
- BRINGUP_IMPL=stub: FAIL (PCC 0).
- Device gate: FAIL with NotImplementedError ("no device module for experts yet"); the implement step writes it.

Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p/tests/bringup/test_c_sliding_moe_experts.py`

## C.sliding_moe.experts implement (attempt 1), 2026-09-26

What was done
- Added `tt/experts.py:TtExperts`, adapted from gemma4_a4b_d_p `tt/experts.py`. The pipeline is: topk(8) of the dense routing -> masked_bincount + offset_cumsum
  -> TtDispatchModule (1-chip dispatch group, capacity 8 x S rows) -> experts -> TtCombineModule (init_zeros) ->
  TtReduceModule -> `ttnn.all_reduce(cluster_axis=1)`. EP=4 puts 64 experts per chip, with bfp8 weights (a bfp4 request is asserted
  against). Counts and offsets are [1, 256] global. `LazyExpertWeights` dequantizes mxfp4 one expert at a time.
  TtRoutedExpert caches bfp8 tensorbins under `generated/mimo_v2_6_d_p/tt_cache/experts`
  (`layer_{i}.experts.BFLOAT8_B.*`). A complete cache skips the dequant (checked with `check_cache_complete` + `fast_cache_checker`).
- hooks.py: `_experts_module` and `_experts_host_fn`; `device_component` handles "experts"; "experts" is added to
  `DEVICE_STEPS["sliding_moe"]`; the hybrid `device_model` gets an experts override.

Decisions and why
- The default is `MIMO_EXPERTS_MODE=loop`. For each local expert it runs `deepseek_prefill.extract` (cap = chunk length S), then `ttnn.linear` gate and up
  (HiFi2, fp32 dest, auto program config), then `ttnn.multiply` with a SILU input activation, then `ttnn.linear` down, then `insert` into a
  bf16 TILE slab. This departs from the components entry (unified_routed_expert_moe) because the fused Silu path fails the frozen
  test's norm-ratio check: `unified` scores PCC 0.99982, rel 0.019, ratio [0.963, 1.054]; `fused` (moe_fused_swiglu) scores 0.99960 / 0.030 / [0.961, 1.055].
  The brief's fix (make the factory honour fidelity for Silu) is a `ttnn/cpp` change, outside this step's
  allowed paths. Both of those modes stay selectable.
- Cost: warm, S=5120, random uniform routing: loop 148 ms per layer, unified 14 ms. Loop runs 64 experts x 5 ops per chip and
  computes every expert over S rows (extract's static cap). About 0.6 s per chunk over the 4 sliding_moe layers of the subset.
  The perf step should extend the factory's GeluTanh fidelity / fp32-dest handling to Silu
  (`unified_routed_expert_ffn_program_factory.cpp:254` and `:1084`, then `./build_metal.sh`) and switch back to `unified`.

Gotchas
- `routed_expert_ffn` (BH) fixes subblock w 6 and rejects fp32 dest, hence the plain `ttnn.linear` calls.
- `insert` needs its buffer and slab to share a dtype. The output slab is a bf16 TILE copy of the dispatch buffer; extract reads a bfp8 copy.
- Loop mode compiles about 146 programs (extract/insert are keyed on local_expert_id). The first call is slow (55 s cold, 0.4 s JIT-warm).
- The `FAIL pcc_experts_L01: pcc=0.000000` line during collection is the precompile stub.
- The timing probe dir `tests/ttnn/unit_tests/operations/mimo_experts_timing/` was deleted.

Results
- Gate PASS: pcc_experts_L01 0.99995, rel_l2 0.0100, row norm ratio [0.9800, 1.0110], worst row rel L2 0.029 (device-noise
  estimate in the test: 0.0091 / [0.986, 1.014] / 0.028).

Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p/tests/bringup/test_c_sliding_moe_experts.py`

## S.sliding_moe.06.test.1 (swap 06, experts), 2026-09-26

What was done
- Replaced the rendered one-liner in `tests/bringup/test_swap_sliding_moe_06_experts.py` with swap 05's body (all of its
  per-step checks for attn_norm, attention, attn_residual, ffn_norm and router, plus the block-out rel L2 <= 0.01).
  Added `_check_experts`.
- Experts checks: vs golden, PCC >= 0.99 and rel L2 <= 0.03. Vs the CPU experts on the same device ffn_norm and router
  ("iso"): rel L2 <= 0.03, per-token norm ratio in [0.97, 1.03], worst row rel L2 <= 0.1 (the component test limits).

Decisions and why
- The per-row checks run against iso, not the golden. The device router's near-tie flips (up to 39 of 2048 rows)
  change which experts a row uses, so per-row errors against the golden would measure the router, not the experts.

Results
- BRINGUP_IMPL=reference: PASS. Experts golden rel 0.0126, iso 0. Out pcc 0.999997.
- BRINGUP_IMPL=stub: FAIL on every check.
- Device gate: PASS. pcc_swap_out 0.999992. Experts pcc 0.99983, golden rel 0.0185, iso rel 0.0098, iso ratio
  [0.9793, 1.0114], iso worst row 0.029. Out rel 0.0040.
- Gotcha: the iso norm-ratio minimum of 0.979 is close to the 0.97 limit (the same as in the component test). A
  lower-fidelity experts perf change (unified / LoFi) will likely fail it.

Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p/tests/bringup/test_swap_sliding_moe_06_experts.py`

## C.sliding_moe.ffn_residual.test.1, 2026-09-26

What was done
- Replaced the rendered one-liner in `tests/bringup/test_c_sliding_moe_ffn_residual.py` with the body of
  `test_c_sliding_moe_attn_residual.py`, pointed at `ffn_residual`: `out = h_mid + experts_out` (reference
  `mimo_ref.py:421`, plain add).
- Asserted checks: PCC >= 0.99 (gated), output size, finite, rel L2 <= 0.01, per-token norm ratio in [0.99, 1.01]. On
  delta = out - h_mid: experts coefficient in [0.97, 1.03] and experts-term rel L2 <= 0.1.

Decisions and why
- The golden has ||h_mid|| 79.58, ||experts_out|| 12.44 and ||out|| 82.24. The experts term is about 15% of the output
  norm, not 1% as in attn_residual, so rel L2 alone catches a dropped term (0.151), a half term (0.076), a one-row
  shift (0.201) and a 2x error (1.0). All of these except the shift pass PCC 0.99.
- The addend limits are tighter than in attn_residual (0.3 -> 0.1) because bf16 output rounding costs only 0.011 here.
- Host measurements (PCC / rel / ratio): fp32 add 0.999997 / 0.0024 / [0.9991, 1.0010]; bf16 add 0.999996 / 0.0029 /
  [0.9989, 1.0012]; last row zeroed 0.99981 / 0.0196 / min 0.

Results
- BRINGUP_IMPL=reference: PASS (rel 0.0024, ratio [0.9991, 1.0010], coef 1.0, experts rel 0).
- BRINGUP_IMPL=stub: FAIL (PCC 0).
- Gate (device): FAIL with `NotImplementedError: no device module for ffn_residual yet`, which is expected before the
  implement step.
- The `FAIL pcc_ffn_residual_L01: pcc=0.000000` line printed during collection comes from the precompile stub.

Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p/tests/bringup/test_c_sliding_moe_ffn_residual.py`

## C.sliding_moe.ffn_residual implement (attempt 1)
- Reused the existing `tt/residual.py` `TtResidualAdd` (replicated bf16 `ttnn.add`, no CCL). The only change is in `bringup/hooks.py`:
  added `ffn_residual` to `_RESIDUAL_STEPS` (so `device_component` returns `_residual_host_fn`) and to `DEVICE_STEPS["sliding_moe"]` (hybrid `device_model`).
- The inputs are (h_mid, experts_out), in the block graph's order; the add is commutative, so no reordering was needed.
- Gate: pcc_ffn_residual_L01 = 0.999996, rel_l2 0.00289, row norm ratio [0.9993, 1.0016], experts coef 1.0014, rel 0.0114. PASS.
- The log's first `FAIL pcc ... 0.000000` line comes from the precompile collect pass (stubbed outputs). Ignore it; the real pass is the second line.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p/tests/bringup/test_c_sliding_moe_ffn_residual.py`

## S.sliding_moe.07.test.1 (swap 07, ffn_residual), 2026-09-26

What was done
- Replaced the rendered one-liner in `tests/bringup/test_swap_sliding_moe_07_ffn_residual.py` with swap 06's body
  (all per-step checks for attn_norm, attention, attn_residual, ffn_norm, router and experts, block out rel L2 <= 0.01)
  and added `_check_ffn_residual`.
- ffn_residual's output is the block `out`. Checks vs golden: PCC >= 0.99, rel L2 <= 0.01 on the whole chunk and the
  first 128 rows, per-token ratio in [0.98, 1.02]. Vs the CPU add on the device h_mid / experts_out ("iso"): rel L2
  <= 0.01, ratio [0.99, 1.01], worst row <= 0.02. On delta = out - h_mid: experts coef in [0.97, 1.03] and experts-term
  rel L2 <= 0.1 (the component test limits).

Decisions and why
- The golden per-token ratio is looser (0.98..1.02) than the iso one: router near-tie flips change a whole expert in
  some rows, and the experts term is about 15% of ||out||. The iso checks catch add bugs (dropped or scaled operand,
  zeroed row).

Results
- BRINGUP_IMPL=reference: PASS. Out rel 0.0030, iso 0, coef 1.0.
- BRINGUP_IMPL=stub: FAIL on every check.
- Device gate: PASS. pcc_swap_out 0.999991, out rel 0.0044 / first rows 0.0036, golden ratio [0.9958, 1.0042]; iso rel
  0.0017, ratio [0.9997, 1.0011], worst row 0.0023; experts coef 1.0014, rel 0.0114. Upstream steps are unchanged
  from swap 06 (experts iso ratio min 0.9793 is still the tightest margin).
- The first `FAIL pcc_swap_out: pcc=0.000000` block in the log comes from the precompile collect pass.

Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p/tests/bringup/test_swap_sliding_moe_07_ffn_residual.py`

## C.full_moe.attn_norm test (attempt 1)
- Reviewed the rendered test for attn_norm, layer 5, full_moe. Replaced its body with the same checks as test_c_sliding_moe_attn_norm.py, LAYER = 5: PCC gate, output finite, rel L2 <= 0.03, per-token norm ratio in [0.97, 1.03]. The step is the same RMSNorm: plain w, eps 1e-6.
- Measured on the layer-5 golden with a host-only script: fp32 reference rel 0.0024, ratio [0.9972, 1.0018]; bf16 math rel 0.0033. The mutations and what catches them are in the test docstring. Sum instead of mean and one zeroed row pass PCC; rel L2 or the norm ratio catches them. Layer 1's weight used instead of layer 5's fails PCC (0.33).
- BRINGUP_IMPL=reference: pass (PCC 0.999997, rel 0.0024). BRINGUP_IMPL=stub: fails PCC. Default (device) gate run: pass (PCC 0.999996, rel 0.0030, ratio [0.9962, 1.0022]). The existing device norm module already covers layer 5.
- Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p/tests/bringup/test_c_full_moe_attn_norm.py`

## S.full_moe.01 test (attempt 1), 2026-09-26

What was done
- Replaced the rendered `run_swap_test` call in `tests/bringup/test_swap_full_moe_01_attn_norm.py` with the body of
  `test_swap_sliding_moe_01_attn_norm.py`, BLOCK_TYPE = full_moe (layer 5). pcc_swap_out is gated at 0.98. Asserted
  extras: attn_norm vs golden PCC >= 0.99, rel L2 <= 0.03 and per-token norm ratio in [0.97, 1.03]; block out finite
  with rel L2 <= 0.01.

Why
- It is the same RMSNorm step, and the extras catch the same mutations that pass the gate on layers 0 and 1 (x1.05,
  eps, noise; see those tests' docstrings). The mutations were not measured again at layer 5.

Results
- BRINGUP_IMPL=reference: PASS (out 0.999997 / rel 0.0025, step rel 0.0024, ratio [0.9972, 1.0018]).
- BRINGUP_IMPL=stub: FAIL (out PCC 0 / rel 0.345; step rel 1.0).
- Device (default): PASS. pcc_swap_out 0.999994, out rel 0.0034, attn_norm rel 0.0030, ratio [0.9962, 1.0022]; router
  trail PCC 0.9983, experts_out 0.9993. The first FAIL/pcc=0 block in each log comes from the precompile pass.

Re-run: `PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all models/demos/mimo_v2_6_d_p/tests/bringup/test_swap_full_moe_01_attn_norm.py`
