# Qwen-Image-2.1 module PCC report — 2026-09-30

Model revision: `790c92633540aa0cb11d9abf19eb46d861714758`. Raw prompt: “the quick brown fox jumps over the lazy dog”. Prompt expansion is off.

The current CUDA and TT reference runs are complete native pipelines at 384 × 256, 20 steps, seed 42. Their initial random samples differ. The deterministic prompt encoder can be compared directly; denoiser and final-image PCC between those independent samples would not measure correctness. A subsequent CUDA VAE oracle decodes the final native TT latent to provide a valid current-run matched-input decoder comparison. Historical matched-input hardware diagnostics are reported separately below.

PCC is Pearson correlation of flattened finite tensor elements. Recomputed metrics use float64 centering and reductions; bitwise equal tensors receive PCC 1. Relative RMS is RMS(TT − CUDA) / RMS(CUDA). The absolute 0.01 column is an element fraction with rtol 0, not a pass assertion.

## Module summary

| Module / validation scope | Output PCC | Relative RMS | Maximum absolute error |
| --- | ---: | ---: | ---: |
| prompt_encoder_current_native_runs · prompt_embeds | 0.9992962836 | 3.8446% | 32 |
| vae_current_native_tt_final_latent · output | 0.9999514923 | 0.7839% | 0.22070312 |
| prompt_encoder_all_layers · prompt_embeds | 0.9992962836 | 3.8446% | 32 |
| initial_noise_transform · Box-Muller Gaussian | 0.9999999956 | 0.0093% | 0.00390625 |
| denoiser_isolated_components · attn.to_q | 0.9999938892 | 1.2354% | 3 |
| denoiser_complete_matched_input_step · velocity_output | 0.9994167537 | 3.4333% | 0.2109375 |
| vae_complete_decoder · output | 0.9999657032 | 0.7911% | 0.073974609 |
| denoiser_prior_matched_initial_20_step_trajectory · updated_latents_step_019 | 0.9915023425 | 13.1377% | 1.9086914 |
| denoiser_current_fork_matched_input_first_step · velocity_output | 0.9996117208 | 2.8464% | 0.171875 |

## prompt_encoder_current_native_runs

current independent integrated runs, same raw prompt and tokenization; valid deterministic encoder comparison despite independent subsequent RNG.

Workload: `{"height": 256, "input_tokens": 31, "prompt_tokens": 17, "seed": 42, "steps": 20, "width": 384}`.

| Stage | PCC | Relative RMS | Max absolute error | Fraction within abs. 0.01 | Evidence |
| --- | ---: | ---: | ---: | ---: | --- |
| `prompt_embeds` | 0.9992962836 | 3.8446% | 32 | 5.74% | recomputed paired tensors |

## vae_current_native_tt_final_latent

current native TT final latent decoded independently by CUDA VAE oracle; TT native decoder ran without activation injection, and this subsequent CUDA-only oracle isolates final decoder error.

Workload: `{"batch_size": 1, "frames": 1, "height": 256, "latent_channels": 64, "output_channels": 4, "width": 384}`.

| Stage | PCC | Relative RMS | Max absolute error | Fraction within abs. 0.01 | Evidence |
| --- | ---: | ---: | ---: | ---: | --- |
| `output` | 0.9999514923 | 0.7839% | 0.22070312 | 93.31% | recomputed paired tensors |

## prompt_encoder_all_layers

historical complete encoder chain from same raw token IDs, no layer activation injection; intermediate metrics include upstream encoder errors.

Workload: `{"encoder_revision": "790c92633540aa0cb11d9abf19eb46d861714758", "hidden_size": 4096, "input_tokens": 31, "layers": 36, "prompt_tokens": 17}`.

| Stage | PCC | Relative RMS | Max absolute error | Fraction within abs. 0.01 | Evidence |
| --- | ---: | ---: | ---: | ---: | --- |
| `embedding` | 1.0000000000 | 0.0000% | 0 | 100.00% | recomputed paired tensors |
| `rotary_cos` | 1.0000000000 | 0.0000% | 0 | 100.00% | recomputed paired tensors |
| `rotary_sin` | 1.0000000000 | 0.0000% | 0 | 100.00% | recomputed paired tensors |
| `layer_000/input_layernorm` | 0.9999996260 | 0.0875% | 0.0009765625 | 100.00% | recomputed paired tensors |
| `layer_000/self_attn.q_proj` | 0.9999989804 | 0.1432% | 0.00390625 | 100.00% | recomputed paired tensors |
| `layer_000/self_attn.q_norm` | 0.9999977457 | 0.2123% | 0.125 | 98.11% | recomputed paired tensors |
| `layer_000/rotated_q` | 0.9999945235 | 0.3458% | 0.125 | 94.87% | recomputed paired tensors |
| `layer_000/self_attn.k_proj` | 0.9999990958 | 0.1347% | 0.00390625 | 100.00% | recomputed paired tensors |
| `layer_000/self_attn.k_norm` | 0.9999984461 | 0.1764% | 1 | 98.07% | recomputed paired tensors |
| `layer_000/rotated_k` | 0.9999984084 | 0.1786% | 1 | 94.96% | recomputed paired tensors |
| `layer_000/self_attn.v_proj` | 0.9999988041 | 0.1557% | 0.0009765625 | 100.00% | recomputed paired tensors |
| `layer_000/attention_heads_merged` | 0.9999535345 | 1.0097% | 0.0047607422 | 100.00% | recomputed paired tensors |
| `layer_000/self_attn.o_proj` | 0.9999846724 | 0.5934% | 0.015625 | 100.00% | recomputed paired tensors |
| `layer_000/post_attention_layernorm` | 0.9999614330 | 0.8783% | 0.0078125 | 100.00% | recomputed paired tensors |
| `layer_000/mlp.gate_proj` | 0.9999568538 | 0.9138% | 0.057617188 | 99.73% | recomputed paired tensors |
| `layer_000/mlp.up_proj` | 0.9999656475 | 0.8301% | 0.0390625 | 99.94% | recomputed paired tensors |
| `layer_000/mlp_product` | 0.9999540301 | 0.9665% | 0.0625 | 99.99% | recomputed paired tensors |
| `layer_000/mlp.down_proj` | 0.9999730731 | 0.7400% | 0.0625 | 99.94% | recomputed paired tensors |
| `layer_000` | 0.9999777064 | 0.6679% | 0.0625 | 99.92% | recomputed paired tensors |
| `layer_001` | 0.9999805258 | 0.6537% | 0.25 | 99.47% | recomputed paired tensors |
| `layer_002` | 0.9999775016 | 0.7208% | 0.375 | 99.03% | recomputed paired tensors |
| `layer_003` | 0.9999703280 | 0.8152% | 0.5 | 98.38% | recomputed paired tensors |
| `layer_004` | 0.9999639271 | 0.8819% | 0.75 | 96.57% | recomputed paired tensors |
| `layer_005` | 0.9999538989 | 1.0021% | 0.75 | 92.84% | recomputed paired tensors |
| `layer_006` | 0.9999998553 | 1.4344% | 192 | 83.82% | recomputed paired tensors |
| `layer_007` | 0.9999998305 | 1.4347% | 192 | 74.61% | recomputed paired tensors |
| `layer_008` | 0.9999998014 | 1.4350% | 192 | 66.59% | recomputed paired tensors |
| `layer_009` | 0.9999997781 | 1.4353% | 192 | 64.77% | recomputed paired tensors |
| `layer_010` | 0.9999997357 | 1.4356% | 192 | 61.95% | recomputed paired tensors |
| `layer_011` | 0.9999997225 | 1.4357% | 192 | 58.12% | recomputed paired tensors |
| `layer_012` | 0.9999996791 | 1.4359% | 192 | 54.56% | recomputed paired tensors |
| `layer_013` | 0.9999996704 | 1.4361% | 192 | 51.46% | recomputed paired tensors |
| `layer_014` | 0.9999995272 | 1.4368% | 192 | 47.07% | recomputed paired tensors |
| `layer_015` | 0.9999995546 | 1.4368% | 192 | 42.59% | recomputed paired tensors |
| `layer_016` | 0.9999789421 | 1.1615% | 192 | 39.65% | recomputed paired tensors |
| `layer_017` | 0.9999787735 | 1.1595% | 192 | 36.54% | recomputed paired tensors |
| `layer_018` | 0.9999788207 | 1.1600% | 192 | 33.79% | recomputed paired tensors |
| `layer_019` | 0.9999787919 | 1.1603% | 192 | 30.55% | recomputed paired tensors |
| `layer_020` | 0.9999787764 | 1.1606% | 192 | 28.15% | recomputed paired tensors |
| `layer_021` | 0.9999787297 | 1.1610% | 192 | 25.12% | recomputed paired tensors |
| `layer_022` | 0.9999786771 | 1.1613% | 192 | 22.75% | recomputed paired tensors |
| `layer_023` | 0.9999785751 | 1.1620% | 192 | 19.78% | recomputed paired tensors |
| `layer_024` | 0.9999784112 | 1.1629% | 192 | 16.74% | recomputed paired tensors |
| `layer_025` | 0.9999781894 | 1.1644% | 192 | 14.39% | recomputed paired tensors |
| `layer_026` | 0.9999779039 | 1.1663% | 192 | 12.33% | recomputed paired tensors |
| `layer_027` | 0.9999774734 | 1.1693% | 192 | 10.90% | recomputed paired tensors |
| `layer_028` | 0.9999768651 | 1.1733% | 192 | 9.92% | recomputed paired tensors |
| `layer_029` | 0.9999760760 | 1.1787% | 192 | 9.12% | recomputed paired tensors |
| `layer_030` | 0.9999748692 | 1.1864% | 192 | 8.45% | recomputed paired tensors |
| `layer_031` | 0.9999730553 | 1.1996% | 192 | 7.80% | recomputed paired tensors |
| `layer_032` | 0.9999704177 | 1.2192% | 192 | 7.26% | recomputed paired tensors |
| `layer_033` | 0.9999662169 | 1.2540% | 192 | 6.75% | recomputed paired tensors |
| `layer_034` | 0.9999202483 | 2.1016% | 192 | 5.97% | recomputed paired tensors |
| `layer_035` | 0.9995318472 | 3.8359% | 160 | 5.32% | recomputed paired tensors |
| `prompt_embeds` | 0.9992962836 | 3.8446% | 32 | 5.74% | recomputed paired tensors |

## initial_noise_transform

historical isolated Box-Muller transform: CUDA consumes same radial and angular uniforms produced by TT; independent CUDA/TT RNGs are not compared.

Workload: `{"height": 256, "seed": 42, "shape": [1, 384, 64], "width": 384}`.

| Stage | PCC | Relative RMS | Max absolute error | Fraction within abs. 0.01 | Evidence |
| --- | ---: | ---: | ---: | ---: | --- |
| `Box-Muller Gaussian` | 0.9999999956 | 0.0093% | 0.00390625 | 100.00% | recomputed paired tensors |

## denoiser_isolated_components

historical isolated module tests with captured CUDA inputs at attention/MLP boundaries; these validate operators separately and are not the native integrated-run result.

Workload: `{"height": 256, "latent_tokens": 256, "reference_steps": 2, "seed": 42, "sequence_tokens": 273, "tested_step": 0, "width": 256}`.

| Stage | PCC | Relative RMS | Max absolute error | Fraction within abs. 0.01 | Evidence |
| --- | ---: | ---: | ---: | ---: | --- |
| `img_norm1` | 0.9999988738 | 0.1517% | 0.125 | 99.64% | recomputed paired tensors |
| `attn.to_q` | 0.9999938892 | 1.2354% | 3 | 20.54% | recomputed paired tensors |
| `attn.to_k` | 0.9999930939 | 1.2060% | 2 | 17.15% | recomputed paired tensors |
| `attn.to_v` | 0.9999894485 | 1.2188% | 0.75 | 26.55% | recomputed paired tensors |
| `img_mlp.gate_layer` | 0.9999901386 | 1.2027% | 0.125 | 99.02% | recomputed paired tensors |
| `img_mlp.proj` | 0.9999886444 | 1.2258% | 0.09375 | 99.65% | recomputed paired tensors |
| `img_mlp.out` | 0.9999896534 | 6.1934% | 112 | 71.09% | recomputed paired tensors |

## denoiser_complete_matched_input_step

historical full denoiser step 0 from same initial CUDA latent and prompt context, followed by TT scheduler update; all 32 blocks are chained, with no intermediate block injection.

Workload: `{"height": 256, "latent_tokens": 256, "reference_steps": 2, "seed": 42, "sequence_tokens": 273, "tested_step": 0, "width": 256}`.

| Stage | PCC | Relative RMS | Max absolute error | Fraction within abs. 0.01 | Evidence |
| --- | ---: | ---: | ---: | ---: | --- |
| `img_in` | 0.9999863590 | 0.5358% | 0.25 | 97.63% | recomputed paired tensors |
| `modulation` | 0.9999643502 | 1.1230% | 0.109375 | 78.42% | recomputed paired tensors |
| `norm_out` | 0.9981998594 | 6.2000% | 0.859375 | 79.60% | recomputed paired tensors |
| `proj_out` | 0.9993017678 | 3.7526% | 0.69921875 | 19.20% | recomputed paired tensors |
| `time_text_embed` | 0.9999542698 | 1.0440% | 0.015625 | 99.87% | recomputed paired tensors |
| `transformer_blocks.0` | 0.9998400358 | 1.8224% | 120 | 37.75% | recomputed paired tensors |
| `transformer_blocks.1` | 0.9998515934 | 1.8258% | 112 | 21.37% | recomputed paired tensors |
| `transformer_blocks.10` | 0.9994614733 | 7.3054% | 512 | 7.88% | recomputed paired tensors |
| `transformer_blocks.11` | 0.9994800762 | 7.2598% | 576 | 7.49% | recomputed paired tensors |
| `transformer_blocks.12` | 0.9994739606 | 7.3052% | 576 | 7.27% | recomputed paired tensors |
| `transformer_blocks.13` | 0.9994519017 | 7.4108% | 640 | 7.13% | recomputed paired tensors |
| `transformer_blocks.14` | 0.9994546870 | 7.4562% | 640 | 7.07% | recomputed paired tensors |
| `transformer_blocks.15` | 0.9994621403 | 7.5004% | 640 | 7.26% | recomputed paired tensors |
| `transformer_blocks.16` | 0.9994591902 | 7.6126% | 640 | 7.48% | recomputed paired tensors |
| `transformer_blocks.17` | 0.9994647843 | 7.6879% | 640 | 7.52% | recomputed paired tensors |
| `transformer_blocks.18` | 0.9994547919 | 7.8184% | 704 | 7.49% | recomputed paired tensors |
| `transformer_blocks.19` | 0.9994331038 | 7.9171% | 704 | 7.78% | recomputed paired tensors |
| `transformer_blocks.2` | 0.9998250803 | 2.0640% | 128 | 16.10% | recomputed paired tensors |
| `transformer_blocks.20` | 0.9994207692 | 7.9957% | 704 | 8.02% | recomputed paired tensors |
| `transformer_blocks.21` | 0.9994055066 | 8.1073% | 704 | 8.03% | recomputed paired tensors |
| `transformer_blocks.22` | 0.9993735921 | 8.1766% | 704 | 8.07% | recomputed paired tensors |
| `transformer_blocks.23` | 0.9993514906 | 8.2372% | 704 | 7.89% | recomputed paired tensors |
| `transformer_blocks.24` | 0.9993287142 | 8.2768% | 704 | 7.69% | recomputed paired tensors |
| `transformer_blocks.25` | 0.9993323535 | 8.3462% | 704 | 7.55% | recomputed paired tensors |
| `transformer_blocks.26` | 0.9993084738 | 8.4265% | 704 | 7.24% | recomputed paired tensors |
| `transformer_blocks.27` | 0.9993036064 | 8.4880% | 704 | 6.81% | recomputed paired tensors |
| `transformer_blocks.28` | 0.9992768232 | 8.5693% | 704 | 6.16% | recomputed paired tensors |
| `transformer_blocks.29` | 0.9992517837 | 8.6398% | 704 | 5.51% | recomputed paired tensors |
| `transformer_blocks.3` | 0.9998022276 | 2.2765% | 144 | 13.11% | recomputed paired tensors |
| `transformer_blocks.30` | 0.9991888773 | 8.7295% | 704 | 4.98% | recomputed paired tensors |
| `transformer_blocks.31` | 0.9987098463 | 7.8776% | 704 | 4.75% | recomputed paired tensors |
| `transformer_blocks.4` | 0.9996730332 | 3.0040% | 160 | 11.85% | recomputed paired tensors |
| `transformer_blocks.5` | 0.9982622142 | 8.2595% | 352 | 10.86% | recomputed paired tensors |
| `transformer_blocks.6` | 0.9993633570 | 7.2355% | 480 | 10.24% | recomputed paired tensors |
| `transformer_blocks.7` | 0.9994308076 | 7.0956% | 480 | 9.63% | recomputed paired tensors |
| `transformer_blocks.8` | 0.9994615968 | 7.1560% | 480 | 9.20% | recomputed paired tensors |
| `transformer_blocks.9` | 0.9994625377 | 7.2382% | 480 | 8.52% | recomputed paired tensors |
| `txt_in` | 0.9999432655 | 1.1165% | 64 | 50.55% | recomputed paired tensors |
| `velocity_output` | 0.9994167537 | 3.4333% | 0.2109375 | 18.99% | recomputed paired tensors |
| `scheduler.updated_latents` | 0.9987575223 | 5.3942% | 0.203125 | 19.60% | recomputed paired tensors |

## vae_complete_decoder

historical complete decoder chain; CUDA and TT both start from identical final TT latent from earlier 384x256 20-step denoiser run; intermediate stage metrics accumulate decoder error.

Workload: `{"batch_size": 1, "frames": 1, "height": 256, "latent_channels": 64, "output_channels": 4, "width": 384}`.

| Stage | PCC | Relative RMS | Max absolute error | Fraction within abs. 0.01 | Evidence |
| --- | ---: | ---: | ---: | ---: | --- |
| `vae_input` | 0.9999978619 | 0.2267% | 0.0625 | not retained | saved hardware stage report |
| `post_quant_conv` | 0.9999940450 | 0.3718% | 0.03125 | not retained | saved hardware stage report |
| `decoder.conv_in` | 0.9999821318 | 0.6157% | 0.09375 | not retained | saved hardware stage report |
| `decoder.mid_block.resnets.0.norm1` | 0.9999738808 | 0.7267% | 0.0625 | not retained | saved hardware stage report |
| `decoder.mid_block.resnets.0.conv1` | 0.9999765708 | 0.7096% | 0.625 | not retained | saved hardware stage report |
| `decoder.mid_block.resnets.0.norm2` | 0.9999708691 | 0.7650% | 0.15625 | not retained | saved hardware stage report |
| `decoder.mid_block.resnets.0.conv2` | 0.9999729535 | 0.7538% | 0.1875 | not retained | saved hardware stage report |
| `decoder.mid_block.resnets.0` | 0.9999609801 | 0.8921% | 0.21875 | not retained | saved hardware stage report |
| `decoder.mid_block.attentions.0.norm` | 0.9999066409 | 1.4110% | 0.02734375 | not retained | saved hardware stage report |
| `decoder.mid_block.attentions.0.to_qkv` | 0.9999685749 | 0.7995% | 0.026367188 | not retained | saved hardware stage report |
| `decoder.mid_block.attentions.0.proj` | 0.9999232478 | 1.9440% | 0.078125 | not retained | saved hardware stage report |
| `decoder.mid_block.attentions.0` | 0.9999309668 | 1.1805% | 0.25 | not retained | saved hardware stage report |
| `decoder.mid_block` | 0.9999068992 | 1.3503% | 0.25 | not retained | saved hardware stage report |
| `decoder.up_blocks.0.upsampler.resample.0` | 0.9997059983 | 2.4542% | 0.28125 | not retained | saved hardware stage report |
| `decoder.up_blocks.0.avg_shortcut` | 0.9999068992 | 1.3503% | 0.25 | not retained | saved hardware stage report |
| `decoder.up_blocks.0` | 0.9998643634 | 1.6479% | 0.828125 | not retained | saved hardware stage report |
| `decoder.up_blocks.1.upsampler.resample.0` | 0.9993351763 | 3.7129% | 1.0625 | not retained | saved hardware stage report |
| `decoder.up_blocks.1.avg_shortcut` | 0.9998643634 | 1.6479% | 0.828125 | not retained | saved hardware stage report |
| `decoder.up_blocks.1` | 0.9996005090 | 2.8428% | 3.9199219 | not retained | saved hardware stage report |
| `decoder.up_blocks.2.upsampler.resample.0` | 0.9980558452 | 6.2691% | 3.765625 | not retained | saved hardware stage report |
| `decoder.up_blocks.2.avg_shortcut` | 0.9998416077 | 1.8636% | 2.5 | not retained | saved hardware stage report |
| `decoder.up_blocks.2` | 0.9994948000 | 3.1816% | 16.75 | not retained | saved hardware stage report |
| `decoder.up_blocks.3.upsampler.resample.0` | 0.9970081147 | 7.7601% | 4.65625 | not retained | saved hardware stage report |
| `decoder.up_blocks.3.avg_shortcut` | 0.9994948000 | 3.1816% | 16.75 | not retained | saved hardware stage report |
| `decoder.up_blocks.3` | 0.9985480447 | 5.3847% | 50.375 | not retained | saved hardware stage report |
| `decoder.up_blocks.4` | 0.9998417867 | 1.7782% | 15.609375 | not retained | saved hardware stage report |
| `decoder.norm_out` | 0.9999153319 | 1.2263% | 0.17578125 | not retained | saved hardware stage report |
| `decoder.conv_out` | 0.9999651251 | 0.8005% | 0.073974609 | not retained | saved hardware stage report |
| `output` | 0.9999657032 | 0.7911% | 0.073974609 | 96.18% | recomputed paired tensors |

## denoiser_prior_matched_initial_20_step_trajectory

historical 20-step trajectory with same initial CUDA latent and prompt context; TT latents feed subsequent TT steps, so this measures accumulated denoising error, not isolated step error or current independent native outputs.

Workload: `{"height": 256, "seed": 42, "steps": 20, "width": 384}`.

| Stage | PCC | Relative RMS | Max absolute error | Fraction within abs. 0.01 | Evidence |
| --- | ---: | ---: | ---: | ---: | --- |
| `updated_latents_step_000` | 0.9999976131 | 0.2243% | 0.015625 | 99.61% | recomputed paired tensors |
| `updated_latents_step_002` | 0.9999881705 | 0.5154% | 0.0390625 | 96.63% | recomputed paired tensors |
| `updated_latents_step_004` | 0.9999675042 | 0.8582% | 0.046875 | 87.28% | recomputed paired tensors |
| `updated_latents_step_009` | 0.9994273685 | 3.4340% | 0.29296875 | 45.97% | recomputed paired tensors |
| `updated_latents_step_014` | 0.9951620283 | 9.8351% | 0.98681641 | 24.91% | recomputed paired tensors |
| `updated_latents_step_019` | 0.9915023425 | 13.1377% | 1.9086914 | 15.79% | recomputed paired tensors |

## denoiser_current_fork_matched_input_first_step

fresh hardware validation from the migrated fork import path at the current native workload geometry; captured CUDA initial latent and prompt embeddings deliberately supply identical inputs for this separate accuracy test. Complete first denoising step is chained through all 32 TT transformer blocks, followed by TT scheduler update. This diagnostic is separate from the activation-injection-free native integrated generation.

Workload: `{"height": 256, "latent_tokens": 384, "reference_steps": 20, "seed": 42, "sequence_tokens": 401, "tested_step": 0, "width": 384}`.

The fork-path hardware suite passed 12 cases and skipped one in 175.34 seconds. This includes native one-step decode, paired first-step validation, and metadata checks; suite time is not inference time. No current per-block output tensors were retained for individual block PCC.

| Stage | PCC | Relative RMS | Max absolute error | Fraction within abs. 0.01 | Evidence |
| --- | ---: | ---: | ---: | ---: | --- |
| `velocity_output` | 0.9996117208 | 2.8464% | 0.171875 | 24.32% | recomputed paired tensors |
| `scheduler.updated_latents` | 0.9999976131 | 0.2243% | 0.015625 | 99.61% | recomputed paired tensors |

## Limits and provenance

- Fresh fork-path first-step validation at 384 × 256 uses identical captured CUDA inputs in a separate diagnostic. The native integrated runs retain their activation-injection-free provenance.
- Current native TT and CUDA denoiser inputs differ because backend-native RNGs differ. No correctness PCC is reported between their denoiser trajectories or final images.
- Only the deterministic prompt encoder has a direct correctness PCC between the two current independent integrated runs. A separate subsequent CUDA VAE oracle consumes the final native TT latent to isolate current decoder error; this does not inject activations into the native TT run.
- Historical module comparisons cover 256x256 operator/full-step diagnostics and 384x256 VAE/full-trajectory diagnostics. They cannot substitute for matched-input per-module validation of the current native run at every boundary.
- Most VAE intermediate TT tensors are not retained locally; their original valid finite PCC metrics are preserved with explicit source labels.
- The encoder historical report contained rotary PCC values slightly above 1 from numerical roundoff. Paired rotary tensors were rechecked for exact equality and corrected to 1; raw values remain in historical_metrics.
- High PCC does not imply all-element atol=0.01. Large encoder outliers can dominate correlation, so relative RMS and maximum absolute error are also reported.
- No inference workload was changed or re-run while producing this report. CUDA is used only as a previously measured reference, and report calculations run on the CPU.

The companion JSON identifies every paired tensor and source report relative to the artifact root `~/data/qwen-image-2.1`, preserves historical raw metrics, and includes TT RNG repeatability and distribution checks. Machine-specific filesystem links are omitted so the report remains readable in the fork.
