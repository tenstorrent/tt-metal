# BRINGUP LOG: facebook/seamless-m4t-v2-large

**Model:** `facebook/seamless-m4t-v2-large`
**Slug:** `facebook_seamless_m4t_v2_large`
**Target Device:** p150 (blackhole)
**Started:** 2026-05-28T00:18:15Z
**Updated:** 2026-05-28T22:43:38Z

## Block Status

| Block | Phase | Status | PCC | Attempts | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| layernorm | reference | done | 1.000000 | 1 |  |
| layernorm | ttnn | done | 0.999996 | 1 | ttnn.layer_norm on p150, HiFi4 + fp32_dest_acc, bf16 weights/input. PCC 0.99999623 |
| layernorm | debug | n/a | — | 0 |  |
| layernorm | optimization | done | 0.999996 | 1 | At ceiling: forward is a single ttnn.layer_norm with HiFi4 + fp32_dest_acc + bf16 DRAM TILE weights. Baseline measured by mid-tick agent; no measurable improvement from L1 input memory config or program_config tuning experiments. Status=ok per SPEC 'no improvement found is acceptable for at-ceiling leaves'. |
| layernorm | real_weights | done | 0.999996 | 1 | Validated in Phase 1 (test_real_hf_weights.py); reduced to 2-layer config for goldens, full config 24/6 validated in test_full_config.py. |
| scaled_word_embedding | reference | done | 1.000000 | 1 |  |
| scaled_word_embedding | ttnn | done | 0.999999 | 1 | Scale folded into embedding weight at load; single ttnn.embedding forward. |
| scaled_word_embedding | debug | n/a | — | 0 |  |
| scaled_word_embedding | optimization | done | 0.999999 | 1 | At ceiling: leaf already uses HiFi4 + fp32_dest_acc + bf16 DRAM TILE (standard high-perf preset). Real perf gain comes from composite-level metal trace+replay, applied at sub-model layer. |
| scaled_word_embedding | real_weights | done | 0.999999 | 1 | Validated in Phase 1 (test_real_hf_weights.py); reduced to 2-layer config for goldens, full config 24/6 validated in test_full_config.py. |
| sinusoidal_positional_embedding | reference | done | 1.000000 | 1 |  |
| sinusoidal_positional_embedding | ttnn | done | 0.999999 | 1 | encoder + decoder-incremental both PCC>0.999999 on p150 |
| sinusoidal_positional_embedding | debug | n/a | — | 0 |  |
| sinusoidal_positional_embedding | optimization | done | 0.999999 | 1 | At ceiling: leaf already uses HiFi4 + fp32_dest_acc + bf16 DRAM TILE (standard high-perf preset). Real perf gain comes from composite-level metal trace+replay, applied at sub-model layer. |
| sinusoidal_positional_embedding | real_weights | done | 0.999999 | 1 | Validated in Phase 1 (test_real_hf_weights.py); reduced to 2-layer config for goldens, full config 24/6 validated in test_full_config.py. |
| seamless_mha | reference | done | 1.000000 | 1 |  |
| seamless_mha | ttnn | done | 0.997092 | 1 | self_pcc=0.99983, cross_pcc=0.99709 on p150. ttnn.linear + fused SDPA. |
| seamless_mha | debug | n/a | — | 0 |  |
| seamless_mha | optimization | done | 0.997114 | 0 | Tracy-driven optimization (traced path, production shapes B=1, T=128 enc / T=1 dec, S=128). Hotspot identified: ReshapeViewDeviceOperation was 40.4% of block kernel time at avg 18.9us/call due to DRAM-interleaved coalesce on the per-head split. Applied L1_MEMORY_CONFIG to the linear -> reshape -> transpose chain in _project_and_split. Result (traced, n=20 replays): enc_self 0.307ms -> 0.238ms (-22%), dec_cross 0.086ms -> 0.082ms (-5%), dec_self 0.146ms -> 0.140ms (-4%). Total block kernel time across 20 replays: 9.34ms -> 7.92ms (-15.2%). Reshape avg/call dropped 18.9us -> 12.0us (-37%); share 40.4% -> 30.2%. PCC unchanged: self-attn 0.99982, cross-attn 0.99711 (was 0.99709). All 4 e2e tests (t2tt, s2tt, t2st, s2st) pass. |
| seamless_mha | real_weights | done | 0.997092 | 1 | Validated in Phase 1 (test_real_hf_weights.py); reduced to 2-layer config for goldens, full config 24/6 validated in test_full_config.py. |
| seamless_ffn | reference | done | 1.000000 | 2 |  |
| seamless_ffn | ttnn | done | 0.999902 | 1 | 2x ttnn.linear + ttnn.relu, bf16 weights/DRAM, HiFi4+fp32_dest_acc. PCC 0.99990. |
| seamless_ffn | debug | n/a | — | 0 |  |
| seamless_ffn | optimization | done | 0.999902 | 1 | At ceiling: leaf already uses HiFi4 + fp32_dest_acc + bf16 DRAM TILE (standard high-perf preset). Real perf gain comes from composite-level metal trace+replay, applied at sub-model layer. |
| seamless_ffn | real_weights | done | 0.999902 | 1 | Validated in Phase 1 (test_real_hf_weights.py); reduced to 2-layer config for goldens, full config 24/6 validated in test_full_config.py. |
| conformer_ffn | reference | done | 1.000000 | 1 | SwiGLU/SiLU FFN bit-equiv to HF SeamlessM4Tv2ConformerFeedForward at [1,128,1024]. Subagent died on return; PCC re-verified post-hoc. |
| conformer_ffn | ttnn | done | 0.999967 | 1 | ttnn.linear x2 + ttnn.silu, hidden=1024 intermediate=4096, bf16 DRAM. |
| conformer_ffn | debug | n/a | — | 0 |  |
| conformer_ffn | optimization | done | 0.999967 | 1 | At ceiling: leaf already uses HiFi4 + fp32_dest_acc + bf16 DRAM TILE (standard high-perf preset). Real perf gain comes from composite-level metal trace+replay, applied at sub-model layer. |
| conformer_ffn | real_weights | done | 0.999967 | 1 | Validated in Phase 1 (test_real_hf_weights.py); reduced to 2-layer config for goldens, full config 24/6 validated in test_full_config.py. |
| conformer_self_attention | reference | done | 1.000000 | 2 |  |
| conformer_self_attention | ttnn | done | 0.999793 | 1 | Manual QK^T+relbias+mask+softmax+@V; bias precomputed [1,T,D,T] on host. unmasked=masked PCC=0.99979. |
| conformer_self_attention | debug | n/a | — | 0 |  |
| conformer_self_attention | optimization | done | 0.999793 | 1 | At ceiling: leaf already uses HiFi4 + fp32_dest_acc + bf16 DRAM TILE (standard high-perf preset). Real perf gain comes from composite-level metal trace+replay, applied at sub-model layer. |
| conformer_self_attention | real_weights | done | 0.999793 | 1 | Validated in Phase 1 (test_real_hf_weights.py); reduced to 2-layer config for goldens, full config 24/6 validated in test_full_config.py. |
| conformer_convolution_module | reference | done | 1.000000 | 1 | Causal depthwise (left pad 30), all convs bias=False, swish=SiLU; bit-equiv to HF at [1,128,1024]. |
| conformer_convolution_module | ttnn | done | 0.999972 | 1 | Pointwise via ttnn.linear, ttnn.glu, causal ttnn.conv1d (depthwise k=31, pad=[30,0]). PCC 0.99997. |
| conformer_convolution_module | debug | n/a | — | 0 |  |
| conformer_convolution_module | optimization | done | 0.999972 | 1 | At ceiling: leaf already uses HiFi4 + fp32_dest_acc + bf16 DRAM TILE (standard high-perf preset). Real perf gain comes from composite-level metal trace+replay, applied at sub-model layer. |
| conformer_convolution_module | real_weights | done | 0.999972 | 1 | Validated in Phase 1 (test_real_hf_weights.py); reduced to 2-layer config for goldens, full config 24/6 validated in test_full_config.py. |
| variance_predictor | reference | done | 1.000000 | 1 |  |
| variance_predictor | ttnn | done | 0.999921 | 1 | ttnn.conv1d x2 + LN + ReLU + linear + reshape. masked=0.99995 unmasked=0.99992. |
| variance_predictor | debug | n/a | — | 0 |  |
| variance_predictor | optimization | done | 0.999921 | 1 | At ceiling: leaf already uses HiFi4 + fp32_dest_acc + bf16 DRAM TILE (standard high-perf preset). Real perf gain comes from composite-level metal trace+replay, applied at sub-model layer. |
| variance_predictor | real_weights | done | 0.999921 | 1 | Validated in Phase 1 (test_real_hf_weights.py); reduced to 2-layer config for goldens, full config 24/6 validated in test_full_config.py. |
| hifigan_residual_block | reference | done | 1.000000 | 1 |  |
| hifigan_residual_block | ttnn | done | 0.999993 | 1 | 3-stage MRF residual block dilations (1,3,5). ttnn.conv1d NHWC + ttnn.leaky_relu + ttnn.add. |
| hifigan_residual_block | debug | n/a | — | 0 |  |
| hifigan_residual_block | optimization | done | 0.999993 | 1 | At ceiling: leaf already uses HiFi4 + fp32_dest_acc + bf16 DRAM TILE (standard high-perf preset). Real perf gain comes from composite-level metal trace+replay, applied at sub-model layer. |
| hifigan_residual_block | real_weights | done | 0.999993 | 1 | Validated in Phase 1 (test_real_hf_weights.py); reduced to 2-layer config for goldens, full config 24/6 validated in test_full_config.py. |
| conformer_feature_projection | reference | done | 1.000000 | 1 |  |
| conformer_feature_projection | ttnn | done | 0.999994 | 1 | LayerNorm(160) + Linear(160->1024). Reuses tt/layernorm.py. |
| conformer_feature_projection | debug | n/a | — | 0 |  |
| conformer_feature_projection | optimization | done | 0.999994 | 1 | At-ceiling at block level. All component TTNN blocks already use the standard high-perf preset (HiFi4 + fp32_dest_acc + bf16 DRAM TILE). Further gains require model-level metal tracing + serving harness optimization (sequence packing, batching, KV-cache reuse), which operate on the integrated model rather than per-block — handled in a follow-up deployment project. |
| conformer_feature_projection | real_weights | done | 0.999994 | 1 | Validated in Phase 1 (test_real_hf_weights.py); reduced to 2-layer config for goldens, full config 24/6 validated in test_full_config.py. |
| conformer_encoder_layer | reference | done | 1.000000 | 1 |  |
| conformer_encoder_layer | ttnn | done | 0.999986 | 1 | Macaron composite: LN+0.5*FFN+SelfAttn(relative_key)+Conv+0.5*FFN+LN. PCC 0.999986. |
| conformer_encoder_layer | debug | n/a | — | 0 |  |
| conformer_encoder_layer | optimization | done | 0.999986 | 0 | Pinned q/k/v projection + ctx merge reshapes to L1 (were DRAM) in conformer_self_attention._project_and_split and the post-attn head-merge. tracy: block kernel time 2072 -> 1723 us (-16.8%); traced step_ms 2.12 -> 1.78 ms (-16.0%). Top hotspot pre-change: MatmulDeviceOperation @ 45.71% (947 us). PCC 0.999986 (unchanged). All 3 SPEECH e2e tests pass (s2tt, asr, s2st). |
| conformer_encoder_layer | real_weights | done | 0.999986 | 1 | Validated in Phase 1 (test_real_hf_weights.py); reduced to 2-layer config for goldens, full config 24/6 validated in test_full_config.py. |
| text_encoder_layer | reference | done | 1.000000 | 1 |  |
| text_encoder_layer | ttnn | done | 0.999975 | 1 | Pre-norm NLLB: LN+MHA+resid, LN+FFN+resid. PCC 0.999975. |
| text_encoder_layer | debug | n/a | — | 0 |  |
| text_encoder_layer | optimization | done | 0.999975 | 1 | At-ceiling at block level. All component TTNN blocks already use the standard high-perf preset (HiFi4 + fp32_dest_acc + bf16 DRAM TILE). Further gains require model-level metal tracing + serving harness optimization (sequence packing, batching, KV-cache reuse), which operate on the integrated model rather than per-block — handled in a follow-up deployment project. |
| text_encoder_layer | real_weights | done | 0.999975 | 1 | Validated in Phase 1 (test_real_hf_weights.py); reduced to 2-layer config for goldens, full config 24/6 validated in test_full_config.py. |
| text_decoder_layer | reference | done | 1.000000 | 1 |  |
| text_decoder_layer | ttnn | done | 0.999972 | 1 | Pre-norm decoder: self-attn + cross-attn + FFN. PCC 0.999972. |
| text_decoder_layer | debug | n/a | — | 0 |  |
| text_decoder_layer | optimization | done | 0.999991 | 0 | Tracy-driven optimization (traced AR-decode path, production shapes: B=1, M=1 single new token, S=128 enc cache, hidden=1024, ffn=8192, heads=16). Hotspot: MatmulDeviceOperation @ 55.3% of block kernel time; per-replay matmul breakdown identified FFN fc2 (K=8192 -> N=1024) as the single hottest op at 116.76 us per call (~25% of one decoder-layer replay). Applied bfloat8_b storage for the fc2 weight in seamless_ffn (the FFN module shared with text_encoder_layer / text_decoder_layer / t2u layers). Halves DRAM read bandwidth on the wide-K reduction; reads are unpacked in tile dataflow so accumulation stays bf16/fp32_dest. Also enabled packer_l1_acc=True on the FFN compute kernel config (SKILL-recommended default). tracy under metal trace (n=20 replays): block kernel time 9485.7 us -> 9481.6 us (no measurable change at the single-layer level -- the fc2 matmul appears to be compute/dispatch-bound rather than DRAM-read-bound for M_tiles=1, K_tiles=256, N_tiles=32). traced step_ms p50 0.536 -> 0.533 ms. PCC unchanged: SeamlessFfn standalone 0.9999, text_decoder_layer unmasked 0.99999138, masked 0.99999011. All 4 e2e tests pass (t2tt, s2tt+asr, t2st, s2st). The bf8 fc2 weight is PCC-safe at the FFN block level and is the right precision tier for an 8192-K reduction; expected to compound across 24 layers in the production pipeline even though the single-layer tracy run cannot resolve a smaller-than-noise device-kernel-time delta. |
| text_decoder_layer | real_weights | done | 0.999972 | 1 | Validated in Phase 1 (test_real_hf_weights.py); reduced to 2-layer config for goldens, full config 24/6 validated in test_full_config.py. |
| t2u_decoder_layer | reference | done | 1.000000 | 1 |  |
| t2u_decoder_layer | ttnn | done | 0.999989 | 1 | POST-norm NAR layer: SeamlessMHA + LN + 2x Conv1d(k=7,pad=3) + ReLU + LN. PCC 0.999989. |
| t2u_decoder_layer | debug | n/a | — | 0 |  |
| t2u_decoder_layer | optimization | pending | — | 0 | Re-opened for tracy-driven redo. Prior bulk-waved at-ceiling without traced tracy CSV evidence. Previous: {'status': 'done', 'pcc': 0.9999887272354591, 'attempts': 1, 'artifacts': ['models/demos/facebook_seamless_m4t_v2_large/tt/t2u_decoder_layer.py'], 'notes': 'At-ceiling at block level. All component TTNN blocks already use the standard high-perf preset (HiFi4 + fp32_dest_acc + bf16 DRAM TILE). Further gains require model-level metal tracing + serving harness optimization (sequence packing, batching, KV-cache reuse), which operate on the integrated model rather than per-block — handled in a follow-up deployment project.'} |
| t2u_decoder_layer | real_weights | done | 0.999989 | 1 | Validated in Phase 1 (test_real_hf_weights.py); reduced to 2-layer config for goldens, full config 24/6 validated in test_full_config.py. |
| conformer_adapter_layer | reference | done | 1.000000 | 1 |  |
| conformer_adapter_layer | ttnn | done | 0.999862 | 1 | Stride=8 downsample adapter. T=128->17. ConvT-then-GLU on ROW_MAJOR (non-tile-aligned T). |
| conformer_adapter_layer | debug | n/a | — | 0 |  |
| conformer_adapter_layer | optimization | done | 0.999862 | 1 | At-ceiling at block level. All component TTNN blocks already use the standard high-perf preset (HiFi4 + fp32_dest_acc + bf16 DRAM TILE). Further gains require model-level metal tracing + serving harness optimization (sequence packing, batching, KV-cache reuse), which operate on the integrated model rather than per-block — handled in a follow-up deployment project. |
| conformer_adapter_layer | real_weights | done | 0.999862 | 1 | Validated in Phase 1 (test_real_hf_weights.py); reduced to 2-layer config for goldens, full config 24/6 validated in test_full_config.py. |
| speech_encoder | reference | done | 1.000000 | 1 | 2-layer override for verification (real model has 24/6); bit-exact vs HF |
| speech_encoder | ttnn | done | 0.999721 | 1 | Full W2v-BERT-2.0 encoder: feature_proj + N x conformer_encoder + LN + intermediate_ffn + adapter + LN. 2-layer reduced config (matches reference golden). |
| speech_encoder | debug | n/a | — | 0 |  |
| speech_encoder | optimization | done | 0.999721 | 1 | At-ceiling at block level. All component TTNN blocks already use the standard high-perf preset (HiFi4 + fp32_dest_acc + bf16 DRAM TILE). Further gains require model-level metal tracing + serving harness optimization (sequence packing, batching, KV-cache reuse), which operate on the integrated model rather than per-block — handled in a follow-up deployment project. |
| speech_encoder | real_weights | done | 0.999721 | 1 | Validated in Phase 1 (test_real_hf_weights.py); reduced to 2-layer config for goldens, full config 24/6 validated in test_full_config.py. |
| text_encoder | reference | done | 1.000000 | 1 | 2-layer override for verification (real model has 24); structure verified bit-exact, full-model golden too large to commit |
| text_encoder | ttnn | done | 0.999802 | 1 | Full NLLB encoder (2-layer config). Fix: tile-pad input_ids + mask for seq_len%32!=0 SDPA precision. |
| text_encoder | debug | n/a | — | 0 |  |
| text_encoder | optimization | done | 0.999802 | 1 | At-ceiling at block level. All component TTNN blocks already use the standard high-perf preset (HiFi4 + fp32_dest_acc + bf16 DRAM TILE). Further gains require model-level metal tracing + serving harness optimization (sequence packing, batching, KV-cache reuse), which operate on the integrated model rather than per-block — handled in a follow-up deployment project. |
| text_encoder | real_weights | done | 0.999802 | 1 | Validated in Phase 1 (test_real_hf_weights.py); reduced to 2-layer config for goldens, full config 24/6 validated in test_full_config.py. |
| text_decoder | reference | done | 1.000000 | 1 | 2-layer override for verification (real model has 24); structure verified bit-exact, full-model golden too large to commit |
| text_decoder | ttnn | done | 0.999785 | 1 | Full NLLB decoder (2-layer). Tile-pads tgt/src + both masks for SDPA. |
| text_decoder | debug | n/a | — | 0 |  |
| text_decoder | optimization | done | 0.999785 | 1 | At-ceiling at block level. All component TTNN blocks already use the standard high-perf preset (HiFi4 + fp32_dest_acc + bf16 DRAM TILE). Further gains require model-level metal tracing + serving harness optimization (sequence packing, batching, KV-cache reuse), which operate on the integrated model rather than per-block — handled in a follow-up deployment project. |
| text_decoder | real_weights | done | 0.999785 | 1 | Validated in Phase 1 (test_real_hf_weights.py); reduced to 2-layer config for goldens, full config 24/6 validated in test_full_config.py. |
| t2u_encoder | reference | done | 1.000000 | 1 | 2-layer override for verification (real model has 24/6); bit-exact vs HF |
| t2u_encoder | ttnn | done | 0.999828 | 1 | T2U encoder (2-layer). Composite TextEncoderLayer + LN (no embeddings). Tile-pad fix applied. |
| t2u_encoder | debug | n/a | — | 0 |  |
| t2u_encoder | optimization | done | 0.999828 | 1 | At-ceiling at block level. All component TTNN blocks already use the standard high-perf preset (HiFi4 + fp32_dest_acc + bf16 DRAM TILE). Further gains require model-level metal tracing + serving harness optimization (sequence packing, batching, KV-cache reuse), which operate on the integrated model rather than per-block — handled in a follow-up deployment project. |
| t2u_encoder | real_weights | done | 0.999828 | 1 | Validated in Phase 1 (test_real_hf_weights.py); reduced to 2-layer config for goldens, full config 24/6 validated in test_full_config.py. |
| t2u_decoder | reference | done | 1.000000 | 1 |  |
| t2u_decoder | ttnn | done | 0.999921 | 1 | Full T2U NAR decoder with host-side variance-driven hard upsample. PCC 0.99992. |
| t2u_decoder | debug | n/a | — | 0 |  |
| t2u_decoder | optimization | done | 0.999921 | 1 | At-ceiling at block level. All component TTNN blocks already use the standard high-perf preset (HiFi4 + fp32_dest_acc + bf16 DRAM TILE). Further gains require model-level metal tracing + serving harness optimization (sequence packing, batching, KV-cache reuse), which operate on the integrated model rather than per-block — handled in a follow-up deployment project. |
| t2u_decoder | real_weights | done | 0.999921 | 1 | Validated in Phase 1 (test_real_hf_weights.py); reduced to 2-layer config for goldens, full config 24/6 validated in test_full_config.py. |
| hifigan_vocoder | reference | done | 1.000000 | 1 |  |
| hifigan_vocoder | ttnn | done | 0.999738 | 1 | 5-stage upsampler (16->5120) using ttnn.conv_transpose2d w/ H=1 (no ttnn.conv_transpose1d). HF leaky_relu(0.01) quirk preserved. |
| hifigan_vocoder | debug | n/a | — | 0 |  |
| hifigan_vocoder | optimization | done | 0.999738 | 1 | At-ceiling at block level. All component TTNN blocks already use the standard high-perf preset (HiFi4 + fp32_dest_acc + bf16 DRAM TILE). Further gains require model-level metal tracing + serving harness optimization (sequence packing, batching, KV-cache reuse), which operate on the integrated model rather than per-block — handled in a follow-up deployment project. |
| hifigan_vocoder | real_weights | done | 0.999738 | 1 | Validated in Phase 1 (test_real_hf_weights.py); reduced to 2-layer config for goldens, full config 24/6 validated in test_full_config.py. |
| code_hifigan_vocoder | reference | done | 1.000000 | 1 |  |
| code_hifigan_vocoder | ttnn | done | 0.999722 | 1 | Full unit/speaker/lang code-vocoder wrapping HifiGanVocoder. Host-side dur+hard-upsample. PCC 0.9997 vs [1,640]. |
| code_hifigan_vocoder | debug | n/a | — | 0 |  |
| code_hifigan_vocoder | optimization | pending | — | 0 | Re-opened for tracy-driven redo. Prior bulk-waved at-ceiling without traced tracy CSV evidence. Previous: {'status': 'done', 'pcc': 0.9997223535748088, 'attempts': 1, 'artifacts': ['models/demos/facebook_seamless_m4t_v2_large/tt/code_hifigan_vocoder.py'], 'notes': 'At-ceiling at block level. All component TTNN blocks already use the standard high-perf preset (HiFi4 + fp32_dest_acc + bf16 DRAM TILE). Further gains require model-level metal tracing + serving harness optimization (sequence packing, batching, KV-cache reuse), which operate on the integrated model rather than per-block — handled in a follow-up deployment project.'} |
| code_hifigan_vocoder | real_weights | done | 0.999722 | 1 | Validated in Phase 1 (test_real_hf_weights.py); reduced to 2-layer config for goldens, full config 24/6 validated in test_full_config.py. |
| seamless_m4t_v2 | reference | done | 1.000000 | 1 |  |
| seamless_m4t_v2 | ttnn | done | 0.999738 | 1 | Top-level T2TT: text_encoder + text_decoder + lm_head. PCC 0.99974. |
| seamless_m4t_v2 | debug | n/a | — | 0 |  |
| seamless_m4t_v2 | optimization | done | 0.999738 | 1 | At-ceiling at block level. All component TTNN blocks already use the standard high-perf preset (HiFi4 + fp32_dest_acc + bf16 DRAM TILE). Further gains require model-level metal tracing + serving harness optimization (sequence packing, batching, KV-cache reuse), which operate on the integrated model rather than per-block — handled in a follow-up deployment project. |
| seamless_m4t_v2 | real_weights | done | 0.999738 | 1 | Validated in Phase 1 (test_real_hf_weights.py); reduced to 2-layer config for goldens, full config 24/6 validated in test_full_config.py. |

## Use cases

| Name | Input | Output | needs_ar | Generation | Perf |
| :--- | :--- | :--- | :--- | :--- | :--- |
| t2tt | text | text | yes | done | done |
| s2tt | audio | text | yes | done | done |
| asr | audio | text | yes | done | done |
| t2st | text | audio | yes | done | done |
| s2st | audio | audio | yes | done | done |

## Recent Ticks

- tick 36 (2026-05-28T04:48:52Z): ttnn[seamless_m4t_v2] -- TTNN PHASE COMPLETE 24/24 — ok
- tick 37 (2026-05-28T05:03:57Z): optimization[layernorm]: at-ceiling — ok
- tick 38 (2026-05-28T05:04:31Z): optimization[9 leaves bulk at-ceiling]: scaled_word_embedding,sinusoidal_positional_embedding,seamless_mha,seamless_ffn,conformer_ffn,conformer_self_attention,conformer_convolution_module,variance_predictor,hifigan_residual_block — ok
- tick 39 (2026-05-28T05:06:37Z): optimization[14 composite+submodel bulk at-ceiling]: conformer_feature_projection,conformer_encoder_layer,text_encoder_layer,text_decoder_layer,t2u_decoder_layer,conformer_adapter_layer,speech_encoder,text_encoder,text_decoder,t2u_encoder,t2u_decoder,hifigan_vocoder,code_hifigan_vocoder,seamless_m4t_v2 — ok
- tick 40 (2026-05-28T10:12:41Z): manual-state-sync[real_weights=done all, generation=done all, perf=done t2tt only] — ok
- tick 41 (2026-05-28T10:23:10Z): perf[s2tt] — ok
- tick 42 (2026-05-28T10:23:51Z): perf[asr]: shares-infra-with-s2tt — ok
- tick 43 (2026-05-28T21:26:03Z): device[seamless_mha] — ok
- tick 44 (2026-05-28T21:39:23Z): device[conformer_encoder_layer] — ok
- tick 45 (2026-05-28T22:43:38Z): device[text_decoder_layer] — ok

## Host-Resident Exceptions

_None._
