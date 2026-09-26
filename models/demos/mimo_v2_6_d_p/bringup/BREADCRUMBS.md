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
