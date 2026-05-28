# BRINGUP LOG: facebook/seamless-m4t-v2-large

**Model:** `facebook/seamless-m4t-v2-large`
**Slug:** `facebook_seamless_m4t_v2_large`
**Target Device:** p150 (blackhole)
**Started:** 2026-05-28T00:18:15Z
**Updated:** 2026-05-28T00:35:24Z

## Block Status

| Block | Phase | Status | PCC | Attempts | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| layernorm | reference | done | 1.000000 | 1 |  |
| layernorm | ttnn | pending | — | 0 |  |
| layernorm | debug | n/a | — | 0 |  |
| layernorm | optimization | pending | — | 0 |  |
| scaled_word_embedding | reference | done | 1.000000 | 1 |  |
| scaled_word_embedding | ttnn | pending | — | 0 |  |
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
| seamless_ffn | reference | pending | — | 0 |  |
| seamless_ffn | ttnn | pending | — | 0 |  |
| seamless_ffn | debug | n/a | — | 0 |  |
| seamless_ffn | optimization | pending | — | 0 |  |
| conformer_ffn | reference | pending | — | 0 |  |
| conformer_ffn | ttnn | pending | — | 0 |  |
| conformer_ffn | debug | n/a | — | 0 |  |
| conformer_ffn | optimization | pending | — | 0 |  |
| conformer_self_attention | reference | pending | — | 0 |  |
| conformer_self_attention | ttnn | pending | — | 0 |  |
| conformer_self_attention | debug | n/a | — | 0 |  |
| conformer_self_attention | optimization | pending | — | 0 |  |
| conformer_convolution_module | reference | pending | — | 0 |  |
| conformer_convolution_module | ttnn | pending | — | 0 |  |
| conformer_convolution_module | debug | n/a | — | 0 |  |
| conformer_convolution_module | optimization | pending | — | 0 |  |
| variance_predictor | reference | pending | — | 0 |  |
| variance_predictor | ttnn | pending | — | 0 |  |
| variance_predictor | debug | n/a | — | 0 |  |
| variance_predictor | optimization | pending | — | 0 |  |
| hifigan_residual_block | reference | pending | — | 0 |  |
| hifigan_residual_block | ttnn | pending | — | 0 |  |
| hifigan_residual_block | debug | n/a | — | 0 |  |
| hifigan_residual_block | optimization | pending | — | 0 |  |
| conformer_feature_projection | reference | pending | — | 0 |  |
| conformer_feature_projection | ttnn | pending | — | 0 |  |
| conformer_feature_projection | debug | n/a | — | 0 |  |
| conformer_feature_projection | optimization | pending | — | 0 |  |
| conformer_encoder_layer | reference | pending | — | 0 |  |
| conformer_encoder_layer | ttnn | pending | — | 0 |  |
| conformer_encoder_layer | debug | n/a | — | 0 |  |
| conformer_encoder_layer | optimization | pending | — | 0 |  |
| text_encoder_layer | reference | pending | — | 0 |  |
| text_encoder_layer | ttnn | pending | — | 0 |  |
| text_encoder_layer | debug | n/a | — | 0 |  |
| text_encoder_layer | optimization | pending | — | 0 |  |
| text_decoder_layer | reference | pending | — | 0 |  |
| text_decoder_layer | ttnn | pending | — | 0 |  |
| text_decoder_layer | debug | n/a | — | 0 |  |
| text_decoder_layer | optimization | pending | — | 0 |  |
| t2u_decoder_layer | reference | pending | — | 0 |  |
| t2u_decoder_layer | ttnn | pending | — | 0 |  |
| t2u_decoder_layer | debug | n/a | — | 0 |  |
| t2u_decoder_layer | optimization | pending | — | 0 |  |
| conformer_adapter_layer | reference | pending | — | 0 |  |
| conformer_adapter_layer | ttnn | pending | — | 0 |  |
| conformer_adapter_layer | debug | n/a | — | 0 |  |
| conformer_adapter_layer | optimization | pending | — | 0 |  |
| speech_encoder | reference | pending | — | 0 |  |
| speech_encoder | ttnn | pending | — | 0 |  |
| speech_encoder | debug | n/a | — | 0 |  |
| speech_encoder | optimization | pending | — | 0 |  |
| text_encoder | reference | pending | — | 0 |  |
| text_encoder | ttnn | pending | — | 0 |  |
| text_encoder | debug | n/a | — | 0 |  |
| text_encoder | optimization | pending | — | 0 |  |
| text_decoder | reference | pending | — | 0 |  |
| text_decoder | ttnn | pending | — | 0 |  |
| text_decoder | debug | n/a | — | 0 |  |
| text_decoder | optimization | pending | — | 0 |  |
| t2u_encoder | reference | pending | — | 0 |  |
| t2u_encoder | ttnn | pending | — | 0 |  |
| t2u_encoder | debug | n/a | — | 0 |  |
| t2u_encoder | optimization | pending | — | 0 |  |
| t2u_decoder | reference | pending | — | 0 |  |
| t2u_decoder | ttnn | pending | — | 0 |  |
| t2u_decoder | debug | n/a | — | 0 |  |
| t2u_decoder | optimization | pending | — | 0 |  |
| hifigan_vocoder | reference | pending | — | 0 |  |
| hifigan_vocoder | ttnn | pending | — | 0 |  |
| hifigan_vocoder | debug | n/a | — | 0 |  |
| hifigan_vocoder | optimization | pending | — | 0 |  |
| code_hifigan_vocoder | reference | pending | — | 0 |  |
| code_hifigan_vocoder | ttnn | pending | — | 0 |  |
| code_hifigan_vocoder | debug | n/a | — | 0 |  |
| code_hifigan_vocoder | optimization | pending | — | 0 |  |
| seamless_m4t_v2 | reference | pending | — | 0 |  |
| seamless_m4t_v2 | ttnn | pending | — | 0 |  |
| seamless_m4t_v2 | debug | n/a | — | 0 |  |
| seamless_m4t_v2 | optimization | pending | — | 0 |  |

## Recent Ticks

- tick 1 (2026-05-28T00:25:44Z): architecture[all] — ok
- tick 2 (2026-05-28T00:33:43Z): reference[layernorm,scaled_word_embedding,sinusoidal_positional_embedding,seamless_mha] — ok
- tick 2 (2026-05-28T00:35:24Z): reference[layernorm,scaled_word_embedding,sinusoidal_positional_embedding,seamless_mha] — ok

## Host-Resident Exceptions

_None._
