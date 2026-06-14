# BRINGUP LOG: MiniMaxAI/MiniMax-M3

**Model:** `MiniMaxAI/MiniMax-M3`
**Slug:** `minimaxai_minimax_m3`
**Target Device:** bh_galaxy (blackhole)
**Started:** 2026-06-14T11:31:00Z
**Updated:** 2026-06-14T12:08:01Z

## Block Status

| Block | Phase | Status | PCC | Attempts | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| rms_norm | reference | done | 1.000009 | 0 | gemma +1, fp32; key language_model.model.layers.0.input_layernorm.weight |
| rms_norm | ttnn | pending | — | 0 |  |
| rms_norm | debug | n/a | — | 0 |  |
| rms_norm | optimization | pending | — | 0 |  |
| rms_norm | real_weights | pending | — | 0 |  |
| vision_layernorm | reference | done | 1.000000 | 0 | true LayerNorm + bias eps 1e-5; vision_tower.* keys |
| vision_layernorm | ttnn | pending | — | 0 |  |
| vision_layernorm | debug | n/a | — | 0 |  |
| vision_layernorm | optimization | pending | — | 0 |  |
| vision_layernorm | real_weights | pending | — | 0 |  |
| embedding | reference | done | 1.000000 | 0 | plain lookup no scaling; tie_word_embeddings=False |
| embedding | ttnn | pending | — | 0 |  |
| embedding | debug | n/a | — | 0 |  |
| embedding | optimization | pending | — | 0 |  |
| embedding | real_weights | pending | — | 0 |  |
| rope | reference | done | 1.000000 | 0 | partial rope 64/64 half-split rotate_half; theta 5e6; bitwise exact |
| rope | ttnn | pending | — | 0 |  |
| rope | debug | n/a | — | 0 |  |
| rope | optimization | pending | — | 0 |  |
| rope | real_weights | pending | — | 0 |  |
| vision_rope_3d | reference | done | 1.000000 | 0 | 3D rope head_dim80, 78 rotated(3x26)+2 passthrough, band [T\|H\|W\|T\|H\|W], smerge2 coords, NeoX half-split -- TT-LANG SPEC |
| vision_rope_3d | ttnn | pending | — | 0 |  |
| vision_rope_3d | debug | n/a | — | 0 |  |
| vision_rope_3d | optimization | pending | — | 0 |  |
| vision_rope_3d | real_weights | pending | — | 0 |  |
| patch_embedding | reference | done | 1.000000 | 0 | Conv3d [1280,3,2,14,14] kernel=stride no bias; ==linear on flat patch |
| patch_embedding | ttnn | pending | — | 0 |  |
| patch_embedding | debug | n/a | — | 0 |  |
| patch_embedding | optimization | pending | — | 0 |  |
| patch_embedding | real_weights | pending | — | 0 |  |
| qk_norm | reference | done | 1.000000 | 0 | per-head Gemma RMS weight[128], pre-rope, reuses rms_norm_forward |
| qk_norm | ttnn | pending | — | 0 |  |
| qk_norm | debug | n/a | — | 0 |  |
| qk_norm | optimization | pending | — | 0 |  |
| qk_norm | real_weights | pending | — | 0 |  |
| gqa_attention | reference | done | 1.000011 | 0 | qkv->view->qknorm->rope(partial64)->repeat_kv4->64->SDPA causal 1/sqrt128->o_proj; q[8192,6144] kv[512,6144] o[6144,8192] |
| gqa_attention | ttnn | pending | — | 0 |  |
| gqa_attention | debug | n/a | — | 0 |  |
| gqa_attention | optimization | pending | — | 0 |  |
| gqa_attention | real_weights | pending | — | 0 |  |
| sparse_lightning_attention | reference | done | 1.000000 | 0 | lightning indexer block-sparse; FULL tt-lang spec in git notes; block-granular selection + token-granular causality; idxq/k proj+norm, partial rope clamp, max block-score, local+topk select, S=2560 sparsity exercised; PCC 1.0 bit-exact |
| sparse_lightning_attention | ttnn | pending | — | 0 |  |
| sparse_lightning_attention | debug | n/a | — | 0 |  |
| sparse_lightning_attention | optimization | pending | — | 0 |  |
| sparse_lightning_attention | real_weights | pending | — | 0 |  |
| vision_attention | reference | done | 1.000004 | 0 | MHA no GQA, no qk-norm, non-causal, biases, out_proj, reuses vision_rope_3d_forward |
| vision_attention | ttnn | pending | — | 0 |  |
| vision_attention | debug | n/a | — | 0 |  |
| vision_attention | optimization | pending | — | 0 |  |
| vision_attention | real_weights | pending | — | 0 |  |
| swigluoai_mlp | reference | done | 1.000001 | 0 | out=down((up+1)*gate*sigmoid(gate*1.702)); gate.clamp(max=7) up.clamp(+-7); fused gate_up in module |
| swigluoai_mlp | ttnn | pending | — | 0 |  |
| swigluoai_mlp | debug | n/a | — | 0 |  |
| swigluoai_mlp | optimization | pending | — | 0 |  |
| swigluoai_mlp | real_weights | pending | — | 0 |  |
| moe_gate | reference | done | 1.000000 | 0 | sigmoid+bias-for-selection-only, normalize top4 to sum1 then x2.0; gate under block_sparse_moe.gate; no grouped routing; exact index match |
| moe_gate | ttnn | pending | — | 0 |  |
| moe_gate | debug | n/a | — | 0 |  |
| moe_gate | optimization | pending | — | 0 |  |
| moe_gate | real_weights | pending | — | 0 |  |
| moe_experts | reference | done | 1.000000 | 0 | ckpt experts.{i}.{w1=gate,w3=up,w2=down}; HF stacks gate_up_proj[128,2*3072,6144] chunk2 + down_proj[128,6144,3072]; routed-only; x2.0 folded in gate weights |
| moe_experts | ttnn | pending | — | 0 |  |
| moe_experts | debug | n/a | — | 0 |  |
| moe_experts | optimization | pending | — | 0 |  |
| moe_experts | real_weights | pending | — | 0 |  |
| shared_expert | reference | done | 1.000000 | 0 | bare SwiGLU-OAI inter3072; NO gate/scale on shared output; keys block_sparse_moe.shared_experts.{gate,up,down}_proj |
| shared_expert | ttnn | pending | — | 0 |  |
| shared_expert | debug | n/a | — | 0 |  |
| shared_expert | optimization | pending | — | 0 |  |
| shared_expert | real_weights | pending | — | 0 |  |
| vision_mlp | reference | done | 1.000000 | 0 | fc1(1280->5120)->erf GELU->fc2; both bias; ACT2FN[gelu]=exact erf |
| vision_mlp | ttnn | pending | — | 0 |  |
| vision_mlp | debug | n/a | — | 0 |  |
| vision_mlp | optimization | pending | — | 0 |  |
| vision_mlp | real_weights | pending | — | 0 |  |
| dense_decoder_layer | reference | done | 1.000000 | 0 | pre-norm no residual scaling; both norms gemma RMS; dense inter 12288; HF eager needs explicit causal mask (None=bidirectional) |
| dense_decoder_layer | ttnn | pending | — | 0 |  |
| dense_decoder_layer | debug | n/a | — | 0 |  |
| dense_decoder_layer | optimization | pending | — | 0 |  |
| dense_decoder_layer | real_weights | pending | — | 0 |  |
| moe_decoder_layer | reference | pending | — | 0 |  |
| moe_decoder_layer | ttnn | pending | — | 0 |  |
| moe_decoder_layer | debug | n/a | — | 0 |  |
| moe_decoder_layer | optimization | pending | — | 0 |  |
| moe_decoder_layer | real_weights | pending | — | 0 |  |
| vision_encoder_layer | reference | pending | — | 0 |  |
| vision_encoder_layer | ttnn | pending | — | 0 |  |
| vision_encoder_layer | debug | n/a | — | 0 |  |
| vision_encoder_layer | optimization | pending | — | 0 |  |
| vision_encoder_layer | real_weights | pending | — | 0 |  |
| patch_merge_mlp | reference | pending | — | 0 |  |
| patch_merge_mlp | ttnn | pending | — | 0 |  |
| patch_merge_mlp | debug | n/a | — | 0 |  |
| patch_merge_mlp | optimization | pending | — | 0 |  |
| patch_merge_mlp | real_weights | pending | — | 0 |  |
| multimodal_projector | reference | pending | — | 0 |  |
| multimodal_projector | ttnn | pending | — | 0 |  |
| multimodal_projector | debug | n/a | — | 0 |  |
| multimodal_projector | optimization | pending | — | 0 |  |
| multimodal_projector | real_weights | pending | — | 0 |  |
| vision_encoder | reference | pending | — | 0 |  |
| vision_encoder | ttnn | pending | — | 0 |  |
| vision_encoder | debug | n/a | — | 0 |  |
| vision_encoder | optimization | pending | — | 0 |  |
| vision_encoder | real_weights | pending | — | 0 |  |
| final_norm | reference | pending | — | 0 |  |
| final_norm | ttnn | pending | — | 0 |  |
| final_norm | debug | n/a | — | 0 |  |
| final_norm | optimization | pending | — | 0 |  |
| final_norm | real_weights | pending | — | 0 |  |
| lm_head | reference | pending | — | 0 |  |
| lm_head | ttnn | pending | — | 0 |  |
| lm_head | debug | n/a | — | 0 |  |
| lm_head | optimization | pending | — | 0 |  |
| lm_head | real_weights | pending | — | 0 |  |
| mtp_head | reference | pending | — | 0 |  |
| mtp_head | ttnn | pending | — | 0 |  |
| mtp_head | debug | n/a | — | 0 |  |
| mtp_head | optimization | pending | — | 0 |  |
| mtp_head | real_weights | pending | — | 0 |  |

## Use cases

| Name | Input | Output | needs_ar | Generation | Perf |
| :--- | :--- | :--- | :--- | :--- | :--- |
| text_generation | text | text | yes | pending | pending |
| image_text_to_text | image | text | yes | pending | pending |
| video_text_to_text | video | text | yes | pending | pending |

## Recent Ticks

- tick 1 (2026-06-14T11:38:24Z): architecture[all] — ok
- tick 2 (2026-06-14T11:51:59Z): reference[rms_norm,embedding,rope,vision_layernorm] — ok
- tick 3 (2026-06-14T11:56:39Z): reference[vision_rope_3d,patch_embedding,qk_norm,gqa_attention] — ok
- tick 4 (2026-06-14T12:01:58Z): reference[sparse_lightning_attention,vision_attention,swigluoai_mlp,moe_gate] — ok
- tick 5 (2026-06-14T12:08:01Z): reference[moe_experts,shared_expert,vision_mlp,dense_decoder_layer] — ok

## Host-Resident Exceptions

- **patch_embedding**: Conv2d patch embedding (patch 14, 3 ch -> 1280) on run-once per_input vision path; tiny (vpatch ~0.0015 GB). May run host-side as image preprocessing if conv2d patchify is not yet performant on device, then handed to the device encoder. (ref: )
- **patch_merge_mlp**: patch_merge compression (spatial_merge_size 2, temporal_patch_size 2) followed by a 2-layer MLP (patch_merge_mlp.linear_1/linear_2, gelu). Tiny, run-once; reuse qwen3_vl patch_merger. May stay host-side if reshape/merge is awkward on device. (ref: )
- **multimodal_projector**: multi_modal_projector: linear_1 -> gelu -> linear_2 (projector_hidden_act gelu, projector_hidden_size 6144, projection_dim 6144, with bias) mapping vision features into the 6144 text hidden space. Tiny, run-once per image; host-side acceptable. (ref: )
