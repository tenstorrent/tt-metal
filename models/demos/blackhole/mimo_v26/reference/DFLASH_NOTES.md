# DFlash drafter for MiMo-V2.6-Flash: what the device integration needs

Legend: [Q] quoted/read from code or config, [M] measured by us, [I] inferred (not in any source we have).
Sources: `/mnt/tt-data/MiMo-V2.6-Flash-RL/dflash/{config.json,dflash.py,dflash_draft_model.safetensors,mask_embedding.pt}`, technical report sec 2.4 (PDF p.5-6, Table 1) and p.32-33.
Oracle: `reference/dflash_ref.py`; teacher-forced measurement: `reference/probes/dflash_teacher_forced.py` (LOG track dflash).

## Architecture [Q config / safetensors headers]
5 layers, all sliding-window (window 1024), hidden 4096, 64 q heads / 8 kv heads, head_dim 128, MLP SwiGLU 16384, RMSNorm eps 1e-6 (weight*x_norm), `is_causal: false`,
block_size 8, `target_layer_ids [0,11,23,35,47]`, `mask_token_id 151675`, `attention_value_scale 0.612`, `attention_sink_bias true`, `partial_rotary_factor 0.5`, rope_theta 1e4.
Tensors (bf16, 2.94 GB): `fc.weight [4096,20480]`, `hidden_norm.weight`, `norm.weight`, per layer q/k/v/o proj (q 8192x4096, k,v 1024x4096, o 4096x8192), `q_norm.weight [128]`, `k_norm.weight [128]`,
`attention_sink_bias [64]`, input/post-attention layernorms, gate/up/down. No embedding or lm_head: both come from the TARGET model [Q dflash.py: `target.model.embed_tokens`, `target.lm_head`].

## One round (spec_generate) [Q dflash.py], anchor = position `start`
1. Block tokens = [x_start (the verified token the target just sampled), mask x7] -> `noise_embedding = target embed_tokens(block)`.
   [M] DEVICE-INTEGRATION TRAP: the target's embedding row 151675 is ~zero (norm 2e-5, cos 0.0007 with mask_embedding.pt). The drafter was trained with the vector in `mask_embedding.pt`
   (`{'mask_token_id':151675,'embedding':[4096]}`, norm 0.76); using the table row would be wrong. (dflash.py itself calls the table; the shipped file says otherwise.)
2. Context: `target_hidden = hidden_norm(fc(concat of target hidden states of layers [0,11,23,35,47]))` for the newly committed positions only [Q: `extract_context_feature(...)[:, :acceptance_length+1]`].
   The result is shared by all 5 draft layers; each layer projects it with its own k_proj/v_proj. => per round the target must hand over, for each of the `acceptance+1` accepted rows of the verify pass
   (rows = anchor + accepted drafts), 5 x 4096 bf16 = 40 KB per position (20480 values). The target verify pass has 8 rows; at most 8 positions per round are consumed. At prefill the drafter needs the same 5 hidden states for
   all prompt positions (first round passes `ctx_len = prompt length`); only the last 1024 positions can ever be attended [I: window].
3. Drafter KV [Q: `past_key_values_draft`, `crop(start)`]: per layer per context position K (k_proj -> per-head k_norm -> rope at the absolute position) and V (v_proj, x0.612 [I where applied]).
   Per position per layer 8 kv heads x 128 x (K,V) = 2048 values -> 5 layers = 10240 values = 20 KB bf16 per position; window 1024 => ring of 1024 positions per layer (20 MB/user total). The 8 block
   (noise) rows are computed each round, attended, then DISCARDED (crop to `start`): they never enter the cache. K/V of the accepted positions are appended next round from step 2.
4. Attention for the 8 block rows (queries at positions start..start+7, rope positions absolute): keys = cached ctx positions in [start-1024, start-1] plus the 8 block keys,
   fully bidirectional inside the block (no causal mask). [I: window applies to ctx keys only, per report p.5: "at most 1,024 backbone context positions preceding the anchor".]
5. Output = final `norm`, then TARGET lm_head on the last 7 rows only (rows 1..7) [Q: `[:, -block_size+1:, :]`], greedy argmax -> draft tokens for start+1..start+7. Row 0 (anchor) output is unused.
6. Verify: target runs the 8-token block [x_start, d1..d7] with KV; `posterior = argmax(logits)`; acceptance = leading run where d_i == posterior_{i-1}; commit accepted + the target's next token; next anchor = that token.
   (Tokens accepted per round = acceptance_length + 1.)

## Differences between the shipped dflash.py and config/checkpoint (so the code is NOT a complete reference)  [Q/M]
* dflash.py attention (Qwen3 `eager_attention_forward`) has NO attention sink and NO V scale although config and checkpoint have `attention_sink_bias [64]` per layer and `attention_value_scale 0.612`.
* dflash.py `apply_rotary_pos_emb` rotates the full head dim; config says partial_rotary_factor 0.5. [M] full rotary gives accuracy collapse (top-1 pos1 0.39, accepted length 1.45 vs 2.8): partial (first 64 of 128, split-half) is right.
* `output_hidden_states=True` is not supported by the MiMo HF modeling file (`MiMoV2Model.forward` returns no hidden_states), so spec_generate cannot run against it as shipped.
* `eager_attention_forward` ignores `sliding_window` and attention_mask=None is passed: the 1024 window is only implied by config/report.
* Hidden-state index convention: `extract_context_feature` uses `hidden_states[layer_id + 1]` = output of layer `layer_id` in the HF convention, but for layer 47 HF's last entry is the post-final-norm state [I: HF behaviour].
  [M] On device captures the three conventions (output of layers [0,11,23,35,47] with L47 pre-norm / L47 post-norm / inputs of those layers) give accepted length 2.78 / 2.79 / 2.76 and are not distinguishable at n=1680
  anchors (differences < 0.03). Sink on/off and V scale 0.612 vs 1.0 are likewise not distinguishable (2.78-2.79 vs 2.75). Use: layer outputs, L47 pre-norm [I], sink on, V x0.612 until a larger test says otherwise.

## Draft quality, teacher-forced on device captures [M]  (LOG track dflash)
16 sequences x 128 tokens (GSM8K chat prompt + human gold solution, no target-model run), anchors p=16..120 (1680 anchors), device hidden states of the 5 layers at positions < p, gold token at p,
compare the 7 draft tokens with gold p+1..p+7. Per-position top-1: 0.596 0.460 0.374 0.297 0.248 0.214 0.178; expected accepted length per round (leading matches + 1) = 2.81.
Answer-region anchors only (818, anchor inside the human solution): 0.671 0.555 0.479 0.388 0.331 0.291 0.238, accepted length 3.43; question-text anchors (862): 0.526 ... 0.121, accepted 2.22.
ASSUMPTION / caveat: the target's greedy token is taken to equal the gold text (upper-bound proxy); real acceptance is against the target's own greedy stream, on its own generations, which is
likely different (higher on the model's own chain-of-thought style, lower where the gold text differs). Short context only (<= 128 positions).
