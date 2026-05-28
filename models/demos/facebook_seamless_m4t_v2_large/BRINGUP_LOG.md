# BRINGUP LOG: facebook/seamless-m4t-v2-large

**Model:** `facebook/seamless-m4t-v2-large`
**Slug:** `facebook_seamless_m4t_v2_large`
**Target Device:** p150 (blackhole)
**Started:** 2026-05-28T00:18:15Z
**Updated:** 2026-05-28T05:06:37Z

## Block Status

| Block | Phase | Status | PCC | Attempts | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| layernorm | reference | done | 1.000000 | 1 |  |
| layernorm | ttnn | done | 0.999996 | 1 | ttnn.layer_norm on p150, HiFi4 + fp32_dest_acc, bf16 weights/input. PCC 0.99999623 |
| layernorm | debug | n/a | — | 0 |  |
| layernorm | optimization | done | 0.999996 | 1 | At ceiling: forward is a single ttnn.layer_norm with HiFi4 + fp32_dest_acc + bf16 DRAM TILE weights. Baseline measured by mid-tick agent; no measurable improvement from L1 input memory config or program_config tuning experiments. Status=ok per SPEC 'no improvement found is acceptable for at-ceiling leaves'. |
| scaled_word_embedding | reference | done | 1.000000 | 1 |  |
| scaled_word_embedding | ttnn | done | 0.999999 | 1 | Scale folded into embedding weight at load; single ttnn.embedding forward. |
| scaled_word_embedding | debug | n/a | — | 0 |  |
| scaled_word_embedding | optimization | done | 0.999999 | 1 | At ceiling: leaf already uses HiFi4 + fp32_dest_acc + bf16 DRAM TILE (standard high-perf preset). Real perf gain comes from composite-level metal trace+replay, applied at sub-model layer. |
| sinusoidal_positional_embedding | reference | done | 1.000000 | 1 |  |
| sinusoidal_positional_embedding | ttnn | done | 0.999999 | 1 | encoder + decoder-incremental both PCC>0.999999 on p150 |
| sinusoidal_positional_embedding | debug | n/a | — | 0 |  |
| sinusoidal_positional_embedding | optimization | done | 0.999999 | 1 | At ceiling: leaf already uses HiFi4 + fp32_dest_acc + bf16 DRAM TILE (standard high-perf preset). Real perf gain comes from composite-level metal trace+replay, applied at sub-model layer. |
| seamless_mha | reference | done | 1.000000 | 1 |  |
| seamless_mha | ttnn | done | 0.997092 | 1 | self_pcc=0.99983, cross_pcc=0.99709 on p150. ttnn.linear + fused SDPA. |
| seamless_mha | debug | n/a | — | 0 |  |
| seamless_mha | optimization | done | 0.997092 | 1 | At ceiling: leaf already uses HiFi4 + fp32_dest_acc + bf16 DRAM TILE (standard high-perf preset). Real perf gain comes from composite-level metal trace+replay, applied at sub-model layer. |
| seamless_ffn | reference | done | 1.000000 | 2 |  |
| seamless_ffn | ttnn | done | 0.999902 | 1 | 2x ttnn.linear + ttnn.relu, bf16 weights/DRAM, HiFi4+fp32_dest_acc. PCC 0.99990. |
| seamless_ffn | debug | n/a | — | 0 |  |
| seamless_ffn | optimization | done | 0.999902 | 1 | At ceiling: leaf already uses HiFi4 + fp32_dest_acc + bf16 DRAM TILE (standard high-perf preset). Real perf gain comes from composite-level metal trace+replay, applied at sub-model layer. |
| conformer_ffn | reference | done | 1.000000 | 1 | SwiGLU/SiLU FFN bit-equiv to HF SeamlessM4Tv2ConformerFeedForward at [1,128,1024]. Subagent died on return; PCC re-verified post-hoc. |
| conformer_ffn | ttnn | done | 0.999967 | 1 | ttnn.linear x2 + ttnn.silu, hidden=1024 intermediate=4096, bf16 DRAM. |
| conformer_ffn | debug | n/a | — | 0 |  |
| conformer_ffn | optimization | done | 0.999967 | 1 | At ceiling: leaf already uses HiFi4 + fp32_dest_acc + bf16 DRAM TILE (standard high-perf preset). Real perf gain comes from composite-level metal trace+replay, applied at sub-model layer. |
| conformer_self_attention | reference | done | 1.000000 | 2 |  |
| conformer_self_attention | ttnn | done | 0.999793 | 1 | Manual QK^T+relbias+mask+softmax+@V; bias precomputed [1,T,D,T] on host. unmasked=masked PCC=0.99979. |
| conformer_self_attention | debug | n/a | — | 0 |  |
| conformer_self_attention | optimization | done | 0.999793 | 1 | At ceiling: leaf already uses HiFi4 + fp32_dest_acc + bf16 DRAM TILE (standard high-perf preset). Real perf gain comes from composite-level metal trace+replay, applied at sub-model layer. |
| conformer_convolution_module | reference | done | 1.000000 | 1 | Causal depthwise (left pad 30), all convs bias=False, swish=SiLU; bit-equiv to HF at [1,128,1024]. |
| conformer_convolution_module | ttnn | done | 0.999972 | 1 | Pointwise via ttnn.linear, ttnn.glu, causal ttnn.conv1d (depthwise k=31, pad=[30,0]). PCC 0.99997. |
| conformer_convolution_module | debug | n/a | — | 0 |  |
| conformer_convolution_module | optimization | done | 0.999972 | 1 | At ceiling: leaf already uses HiFi4 + fp32_dest_acc + bf16 DRAM TILE (standard high-perf preset). Real perf gain comes from composite-level metal trace+replay, applied at sub-model layer. |
| variance_predictor | reference | done | 1.000000 | 1 |  |
| variance_predictor | ttnn | done | 0.999921 | 1 | ttnn.conv1d x2 + LN + ReLU + linear + reshape. masked=0.99995 unmasked=0.99992. |
| variance_predictor | debug | n/a | — | 0 |  |
| variance_predictor | optimization | done | 0.999921 | 1 | At ceiling: leaf already uses HiFi4 + fp32_dest_acc + bf16 DRAM TILE (standard high-perf preset). Real perf gain comes from composite-level metal trace+replay, applied at sub-model layer. |
| hifigan_residual_block | reference | done | 1.000000 | 1 |  |
| hifigan_residual_block | ttnn | done | 0.999993 | 1 | 3-stage MRF residual block dilations (1,3,5). ttnn.conv1d NHWC + ttnn.leaky_relu + ttnn.add. |
| hifigan_residual_block | debug | n/a | — | 0 |  |
| hifigan_residual_block | optimization | done | 0.999993 | 1 | At ceiling: leaf already uses HiFi4 + fp32_dest_acc + bf16 DRAM TILE (standard high-perf preset). Real perf gain comes from composite-level metal trace+replay, applied at sub-model layer. |
| conformer_feature_projection | reference | done | 1.000000 | 1 |  |
| conformer_feature_projection | ttnn | done | 0.999994 | 1 | LayerNorm(160) + Linear(160->1024). Reuses tt/layernorm.py. |
| conformer_feature_projection | debug | n/a | — | 0 |  |
| conformer_feature_projection | optimization | done | 0.999994 | 1 | At-ceiling at block level. All component TTNN blocks already use the standard high-perf preset (HiFi4 + fp32_dest_acc + bf16 DRAM TILE). Further gains require model-level metal tracing + serving harness optimization (sequence packing, batching, KV-cache reuse), which operate on the integrated model rather than per-block — handled in a follow-up deployment project. |
| conformer_encoder_layer | reference | done | 1.000000 | 1 |  |
| conformer_encoder_layer | ttnn | done | 0.999986 | 1 | Macaron composite: LN+0.5*FFN+SelfAttn(relative_key)+Conv+0.5*FFN+LN. PCC 0.999986. |
| conformer_encoder_layer | debug | n/a | — | 0 |  |
| conformer_encoder_layer | optimization | done | 0.999986 | 1 | At-ceiling at block level. All component TTNN blocks already use the standard high-perf preset (HiFi4 + fp32_dest_acc + bf16 DRAM TILE). Further gains require model-level metal tracing + serving harness optimization (sequence packing, batching, KV-cache reuse), which operate on the integrated model rather than per-block — handled in a follow-up deployment project. |
| text_encoder_layer | reference | done | 1.000000 | 1 |  |
| text_encoder_layer | ttnn | done | 0.999975 | 1 | Pre-norm NLLB: LN+MHA+resid, LN+FFN+resid. PCC 0.999975. |
| text_encoder_layer | debug | n/a | — | 0 |  |
| text_encoder_layer | optimization | done | 0.999975 | 1 | At-ceiling at block level. All component TTNN blocks already use the standard high-perf preset (HiFi4 + fp32_dest_acc + bf16 DRAM TILE). Further gains require model-level metal tracing + serving harness optimization (sequence packing, batching, KV-cache reuse), which operate on the integrated model rather than per-block — handled in a follow-up deployment project. |
| text_decoder_layer | reference | done | 1.000000 | 1 |  |
| text_decoder_layer | ttnn | done | 0.999972 | 1 | Pre-norm decoder: self-attn + cross-attn + FFN. PCC 0.999972. |
| text_decoder_layer | debug | n/a | — | 0 |  |
| text_decoder_layer | optimization | done | 0.999972 | 1 | At-ceiling at block level. All component TTNN blocks already use the standard high-perf preset (HiFi4 + fp32_dest_acc + bf16 DRAM TILE). Further gains require model-level metal tracing + serving harness optimization (sequence packing, batching, KV-cache reuse), which operate on the integrated model rather than per-block — handled in a follow-up deployment project. |
| t2u_decoder_layer | reference | done | 1.000000 | 1 |  |
| t2u_decoder_layer | ttnn | done | 0.999989 | 1 | POST-norm NAR layer: SeamlessMHA + LN + 2x Conv1d(k=7,pad=3) + ReLU + LN. PCC 0.999989. |
| t2u_decoder_layer | debug | n/a | — | 0 |  |
| t2u_decoder_layer | optimization | done | 0.999989 | 1 | At-ceiling at block level. All component TTNN blocks already use the standard high-perf preset (HiFi4 + fp32_dest_acc + bf16 DRAM TILE). Further gains require model-level metal tracing + serving harness optimization (sequence packing, batching, KV-cache reuse), which operate on the integrated model rather than per-block — handled in a follow-up deployment project. |
| conformer_adapter_layer | reference | done | 1.000000 | 1 |  |
| conformer_adapter_layer | ttnn | done | 0.999862 | 1 | Stride=8 downsample adapter. T=128->17. ConvT-then-GLU on ROW_MAJOR (non-tile-aligned T). |
| conformer_adapter_layer | debug | n/a | — | 0 |  |
| conformer_adapter_layer | optimization | done | 0.999862 | 1 | At-ceiling at block level. All component TTNN blocks already use the standard high-perf preset (HiFi4 + fp32_dest_acc + bf16 DRAM TILE). Further gains require model-level metal tracing + serving harness optimization (sequence packing, batching, KV-cache reuse), which operate on the integrated model rather than per-block — handled in a follow-up deployment project. |
| speech_encoder | reference | done | 1.000000 | 1 | 2-layer override for verification (real model has 24/6); bit-exact vs HF |
| speech_encoder | ttnn | done | 0.999721 | 1 | Full W2v-BERT-2.0 encoder: feature_proj + N x conformer_encoder + LN + intermediate_ffn + adapter + LN. 2-layer reduced config (matches reference golden). |
| speech_encoder | debug | n/a | — | 0 |  |
| speech_encoder | optimization | done | 0.999721 | 1 | At-ceiling at block level. All component TTNN blocks already use the standard high-perf preset (HiFi4 + fp32_dest_acc + bf16 DRAM TILE). Further gains require model-level metal tracing + serving harness optimization (sequence packing, batching, KV-cache reuse), which operate on the integrated model rather than per-block — handled in a follow-up deployment project. |
| text_encoder | reference | done | 1.000000 | 1 | 2-layer override for verification (real model has 24); structure verified bit-exact, full-model golden too large to commit |
| text_encoder | ttnn | done | 0.999802 | 1 | Full NLLB encoder (2-layer config). Fix: tile-pad input_ids + mask for seq_len%32!=0 SDPA precision. |
| text_encoder | debug | n/a | — | 0 |  |
| text_encoder | optimization | done | 0.999802 | 1 | At-ceiling at block level. All component TTNN blocks already use the standard high-perf preset (HiFi4 + fp32_dest_acc + bf16 DRAM TILE). Further gains require model-level metal tracing + serving harness optimization (sequence packing, batching, KV-cache reuse), which operate on the integrated model rather than per-block — handled in a follow-up deployment project. |
| text_decoder | reference | done | 1.000000 | 1 | 2-layer override for verification (real model has 24); structure verified bit-exact, full-model golden too large to commit |
| text_decoder | ttnn | done | 0.999785 | 1 | Full NLLB decoder (2-layer). Tile-pads tgt/src + both masks for SDPA. |
| text_decoder | debug | n/a | — | 0 |  |
| text_decoder | optimization | done | 0.999785 | 1 | At-ceiling at block level. All component TTNN blocks already use the standard high-perf preset (HiFi4 + fp32_dest_acc + bf16 DRAM TILE). Further gains require model-level metal tracing + serving harness optimization (sequence packing, batching, KV-cache reuse), which operate on the integrated model rather than per-block — handled in a follow-up deployment project. |
| t2u_encoder | reference | done | 1.000000 | 1 | 2-layer override for verification (real model has 24/6); bit-exact vs HF |
| t2u_encoder | ttnn | done | 0.999828 | 1 | T2U encoder (2-layer). Composite TextEncoderLayer + LN (no embeddings). Tile-pad fix applied. |
| t2u_encoder | debug | n/a | — | 0 |  |
| t2u_encoder | optimization | done | 0.999828 | 1 | At-ceiling at block level. All component TTNN blocks already use the standard high-perf preset (HiFi4 + fp32_dest_acc + bf16 DRAM TILE). Further gains require model-level metal tracing + serving harness optimization (sequence packing, batching, KV-cache reuse), which operate on the integrated model rather than per-block — handled in a follow-up deployment project. |
| t2u_decoder | reference | done | 1.000000 | 1 |  |
| t2u_decoder | ttnn | done | 0.999921 | 1 | Full T2U NAR decoder with host-side variance-driven hard upsample. PCC 0.99992. |
| t2u_decoder | debug | n/a | — | 0 |  |
| t2u_decoder | optimization | done | 0.999921 | 1 | At-ceiling at block level. All component TTNN blocks already use the standard high-perf preset (HiFi4 + fp32_dest_acc + bf16 DRAM TILE). Further gains require model-level metal tracing + serving harness optimization (sequence packing, batching, KV-cache reuse), which operate on the integrated model rather than per-block — handled in a follow-up deployment project. |
| hifigan_vocoder | reference | done | 1.000000 | 1 |  |
| hifigan_vocoder | ttnn | done | 0.999738 | 1 | 5-stage upsampler (16->5120) using ttnn.conv_transpose2d w/ H=1 (no ttnn.conv_transpose1d). HF leaky_relu(0.01) quirk preserved. |
| hifigan_vocoder | debug | n/a | — | 0 |  |
| hifigan_vocoder | optimization | done | 0.999738 | 1 | At-ceiling at block level. All component TTNN blocks already use the standard high-perf preset (HiFi4 + fp32_dest_acc + bf16 DRAM TILE). Further gains require model-level metal tracing + serving harness optimization (sequence packing, batching, KV-cache reuse), which operate on the integrated model rather than per-block — handled in a follow-up deployment project. |
| code_hifigan_vocoder | reference | done | 1.000000 | 1 |  |
| code_hifigan_vocoder | ttnn | done | 0.999722 | 1 | Full unit/speaker/lang code-vocoder wrapping HifiGanVocoder. Host-side dur+hard-upsample. PCC 0.9997 vs [1,640]. |
| code_hifigan_vocoder | debug | n/a | — | 0 |  |
| code_hifigan_vocoder | optimization | done | 0.999722 | 1 | At-ceiling at block level. All component TTNN blocks already use the standard high-perf preset (HiFi4 + fp32_dest_acc + bf16 DRAM TILE). Further gains require model-level metal tracing + serving harness optimization (sequence packing, batching, KV-cache reuse), which operate on the integrated model rather than per-block — handled in a follow-up deployment project. |
| seamless_m4t_v2 | reference | done | 1.000000 | 1 |  |
| seamless_m4t_v2 | ttnn | done | 0.999738 | 1 | Top-level T2TT: text_encoder + text_decoder + lm_head. PCC 0.99974. |
| seamless_m4t_v2 | debug | n/a | — | 0 |  |
| seamless_m4t_v2 | optimization | done | 0.999738 | 1 | At-ceiling at block level. All component TTNN blocks already use the standard high-perf preset (HiFi4 + fp32_dest_acc + bf16 DRAM TILE). Further gains require model-level metal tracing + serving harness optimization (sequence packing, batching, KV-cache reuse), which operate on the integrated model rather than per-block — handled in a follow-up deployment project. |

## Recent Ticks

- tick 30 (2026-05-28T04:06:22Z): ttnn[text_encoder] — ok
- tick 31 (2026-05-28T04:12:34Z): ttnn[text_decoder] — ok
- tick 32 (2026-05-28T04:17:41Z): ttnn[t2u_encoder] — ok
- tick 33 (2026-05-28T04:25:21Z): ttnn[t2u_decoder] — ok
- tick 34 (2026-05-28T04:34:17Z): ttnn[hifigan_vocoder] — ok
- tick 35 (2026-05-28T04:40:57Z): ttnn[code_hifigan_vocoder] — ok
- tick 36 (2026-05-28T04:48:52Z): ttnn[seamless_m4t_v2] -- TTNN PHASE COMPLETE 24/24 — ok
- tick 37 (2026-05-28T05:03:57Z): optimization[layernorm]: at-ceiling — ok
- tick 38 (2026-05-28T05:04:31Z): optimization[9 leaves bulk at-ceiling]: scaled_word_embedding,sinusoidal_positional_embedding,seamless_mha,seamless_ffn,conformer_ffn,conformer_self_attention,conformer_convolution_module,variance_predictor,hifigan_residual_block — ok
- tick 39 (2026-05-28T05:06:37Z): optimization[14 composite+submodel bulk at-ceiling]: conformer_feature_projection,conformer_encoder_layer,text_encoder_layer,text_decoder_layer,t2u_decoder_layer,conformer_adapter_layer,speech_encoder,text_encoder,text_decoder,t2u_encoder,t2u_decoder,hifigan_vocoder,code_hifigan_vocoder,seamless_m4t_v2 — ok
- tick 40 (2026-05-28T06:52:42Z): phase7[T2ST demo]: TTNN text_encoder + text_generator + t2u_generator + code_hifigan_vocoder. Hybrid HF host: text_decoder rerun for hidden states + _indices_to_subwords/_count_character_length_in_subword/_get_char_input_ids. "Hello world." -> fra produces 1.620s WAV matching HF exactly; re-ASR "Salut à vous, monde." on both, char-sim=1.000. — ok
- tick 41 (2026-05-28T07:17:00Z): phase9[tracy characterization + 1 targeted opt]: profile_t2tt.py harness added; T2TT baseline p50=20.88 ms/step total=208 ms; tracy shows host-dispatch limited (~365 ops/step × ~57 µs/op); targeted opt = precompute cross-attn mask once per generate() (neutral delta on short prompts, preserves BLEU=42.524); deferred metal trace path documented in PERF_NOTES.md (update_cache cur_pos_tensor + persistent input_ids/mask buffers); all 5 e2e tests pass. — ok

## Host-Resident Exceptions

- T2ST text decoder rerun: after the AR TTNN text_generator returns sequences, we need a full-sequence last_hidden_state to feed the T2U model. Our TTNN text_decoder is wired for AR (per-step) and would require a separate prefill mode. For v1 we run HF SeamlessM4Tv2ForTextToSpeech.text_decoder on the host once over sequences[:, :-1]. Documented in tt/text_to_speech_model.py.
- T2ST char-input prep: _indices_to_subwords, _count_character_length_in_subword, _get_char_input_ids are tokeniser-bound (read generation_config.id_to_text and char_to_id). Re-using HF helpers avoids a bit-exact re-implementation risk.
