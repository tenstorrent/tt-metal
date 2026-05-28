# BRINGUP LOG: facebook/seamless-m4t-v2-large

**Model:** `facebook/seamless-m4t-v2-large`
**Slug:** `facebook_seamless_m4t_v2_large`
**Target Device:** p150 (blackhole)
**Started:** 2026-05-28T00:18:15Z
**Updated:** 2026-05-28T02:09:29Z

## Block Status

| Block | Phase | Status | PCC | Attempts | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| layernorm | reference | done | 1.000000 | 1 |  |
| layernorm | ttnn | done | 0.999996 | 1 | ttnn.layer_norm on p150, HiFi4 + fp32_dest_acc, bf16 weights/input. PCC 0.99999623 |
| layernorm | debug | n/a | — | 0 |  |
| layernorm | optimization | pending | — | 0 |  |
| scaled_word_embedding | reference | done | 1.000000 | 1 |  |
| scaled_word_embedding | ttnn | done | 0.999999 | 1 | Scale folded into embedding weight at load; single ttnn.embedding forward. |
| scaled_word_embedding | debug | n/a | — | 0 |  |
| scaled_word_embedding | optimization | pending | — | 0 |  |
| sinusoidal_positional_embedding | reference | done | 1.000000 | 1 |  |
| sinusoidal_positional_embedding | ttnn | pending | — | 0 |  |
| sinusoidal_positional_embedding | debug | n/a | — | 0 |  |
| sinusoidal_positional_embedding | optimization | pending | — | 0 |  |
| seamless_mha | reference | done | 1.000000 | 1 |  |
| seamless_mha | ttnn | pending | — | 0 |  |
| seamless_mha | debug | n/a | — | 0 |  |
| seamless_mha | optimization | pending | — | 0 |  |
| seamless_ffn | reference | done | 1.000000 | 2 |  |
| seamless_ffn | ttnn | pending | — | 0 |  |
| seamless_ffn | debug | n/a | — | 0 |  |
| seamless_ffn | optimization | pending | — | 0 |  |
| conformer_ffn | reference | done | 1.000000 | 1 | SwiGLU/SiLU FFN bit-equiv to HF SeamlessM4Tv2ConformerFeedForward at [1,128,1024]. Subagent died on return; PCC re-verified post-hoc. |
| conformer_ffn | ttnn | pending | — | 0 |  |
| conformer_ffn | debug | n/a | — | 0 |  |
| conformer_ffn | optimization | pending | — | 0 |  |
| conformer_self_attention | reference | done | 1.000000 | 2 |  |
| conformer_self_attention | ttnn | pending | — | 0 |  |
| conformer_self_attention | debug | n/a | — | 0 |  |
| conformer_self_attention | optimization | pending | — | 0 |  |
| conformer_convolution_module | reference | done | 1.000000 | 1 | Causal depthwise (left pad 30), all convs bias=False, swish=SiLU; bit-equiv to HF at [1,128,1024]. |
| conformer_convolution_module | ttnn | pending | — | 0 |  |
| conformer_convolution_module | debug | n/a | — | 0 |  |
| conformer_convolution_module | optimization | pending | — | 0 |  |
| variance_predictor | reference | done | 1.000000 | 1 |  |
| variance_predictor | ttnn | pending | — | 0 |  |
| variance_predictor | debug | n/a | — | 0 |  |
| variance_predictor | optimization | pending | — | 0 |  |
| hifigan_residual_block | reference | done | 1.000000 | 1 |  |
| hifigan_residual_block | ttnn | pending | — | 0 |  |
| hifigan_residual_block | debug | n/a | — | 0 |  |
| hifigan_residual_block | optimization | pending | — | 0 |  |
| conformer_feature_projection | reference | done | 1.000000 | 1 |  |
| conformer_feature_projection | ttnn | pending | — | 0 |  |
| conformer_feature_projection | debug | n/a | — | 0 |  |
| conformer_feature_projection | optimization | pending | — | 0 |  |
| conformer_encoder_layer | reference | done | 1.000000 | 1 |  |
| conformer_encoder_layer | ttnn | pending | — | 0 |  |
| conformer_encoder_layer | debug | n/a | — | 0 |  |
| conformer_encoder_layer | optimization | pending | — | 0 |  |
| text_encoder_layer | reference | done | 1.000000 | 1 |  |
| text_encoder_layer | ttnn | pending | — | 0 |  |
| text_encoder_layer | debug | n/a | — | 0 |  |
| text_encoder_layer | optimization | pending | — | 0 |  |
| text_decoder_layer | reference | done | 1.000000 | 1 |  |
| text_decoder_layer | ttnn | pending | — | 0 |  |
| text_decoder_layer | debug | n/a | — | 0 |  |
| text_decoder_layer | optimization | pending | — | 0 |  |
| t2u_decoder_layer | reference | done | 1.000000 | 1 |  |
| t2u_decoder_layer | ttnn | pending | — | 0 |  |
| t2u_decoder_layer | debug | n/a | — | 0 |  |
| t2u_decoder_layer | optimization | pending | — | 0 |  |
| conformer_adapter_layer | reference | done | 1.000000 | 1 |  |
| conformer_adapter_layer | ttnn | pending | — | 0 |  |
| conformer_adapter_layer | debug | n/a | — | 0 |  |
| conformer_adapter_layer | optimization | pending | — | 0 |  |
| speech_encoder | reference | done | 1.000000 | 1 | 2-layer override for verification (real model has 24/6); bit-exact vs HF |
| speech_encoder | ttnn | pending | — | 0 |  |
| speech_encoder | debug | n/a | — | 0 |  |
| speech_encoder | optimization | pending | — | 0 |  |
| text_encoder | reference | done | 1.000000 | 1 | 2-layer override for verification (real model has 24); structure verified bit-exact, full-model golden too large to commit |
| text_encoder | ttnn | pending | — | 0 |  |
| text_encoder | debug | n/a | — | 0 |  |
| text_encoder | optimization | pending | — | 0 |  |
| text_decoder | reference | done | 1.000000 | 1 | 2-layer override for verification (real model has 24); structure verified bit-exact, full-model golden too large to commit |
| text_decoder | ttnn | pending | — | 0 |  |
| text_decoder | debug | n/a | — | 0 |  |
| text_decoder | optimization | pending | — | 0 |  |
| t2u_encoder | reference | done | 1.000000 | 1 | 2-layer override for verification (real model has 24/6); bit-exact vs HF |
| t2u_encoder | ttnn | pending | — | 0 |  |
| t2u_encoder | debug | n/a | — | 0 |  |
| t2u_encoder | optimization | pending | — | 0 |  |
| t2u_decoder | reference | done | 1.000000 | 1 |  |
| t2u_decoder | ttnn | pending | — | 0 |  |
| t2u_decoder | debug | n/a | — | 0 |  |
| t2u_decoder | optimization | pending | — | 0 |  |
| hifigan_vocoder | reference | done | 1.000000 | 1 |  |
| hifigan_vocoder | ttnn | pending | — | 0 |  |
| hifigan_vocoder | debug | n/a | — | 0 |  |
| hifigan_vocoder | optimization | pending | — | 0 |  |
| code_hifigan_vocoder | reference | done | 1.000000 | 1 |  |
| code_hifigan_vocoder | ttnn | pending | — | 0 |  |
| code_hifigan_vocoder | debug | n/a | — | 0 |  |
| code_hifigan_vocoder | optimization | pending | — | 0 |  |
| seamless_m4t_v2 | reference | done | 1.000000 | 1 |  |
| seamless_m4t_v2 | ttnn | pending | — | 0 |  |
| seamless_m4t_v2 | debug | n/a | — | 0 |  |
| seamless_m4t_v2 | optimization | pending | — | 0 |  |

## Recent Ticks

- tick 5 (2026-05-28T00:46:55Z): reference[variance_predictor,hifigan_residual_block] — ok
- tick 6 (2026-05-28T00:52:10Z): reference[conformer_feature_projection,conformer_encoder_layer] — ok
- tick 7 (2026-05-28T00:56:00Z): reference[text_encoder_layer,text_decoder_layer] — ok
- tick 8 (2026-05-28T01:02:59Z): reference[t2u_decoder_layer,conformer_adapter_layer] — ok
- tick 9 (2026-05-28T01:28:00Z): reference[text_encoder,text_decoder] — ok
- tick 10 (2026-05-28T01:35:44Z): reference[speech_encoder,t2u_encoder] — ok
- tick 11 (2026-05-28T01:44:33Z): reference[t2u_decoder,hifigan_vocoder] — ok
- tick 12 (2026-05-28T01:51:42Z): reference[code_hifigan_vocoder,seamless_m4t_v2] -- REFERENCE PHASE COMPLETE — ok
- tick 13 (2026-05-28T02:05:47Z): ttnn[layernorm] — ok
- tick 14 (2026-05-28T02:09:29Z): ttnn[scaled_word_embedding] — ok

## Host-Resident Exceptions

_None._
