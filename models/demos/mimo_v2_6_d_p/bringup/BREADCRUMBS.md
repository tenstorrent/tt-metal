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
