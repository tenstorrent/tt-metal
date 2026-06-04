<!-- Generated from perf/Testing_ACE.xlsx by export_testing_ace_md.py -->

# ACE-Step benchmark results (BH_QB)

Source spreadsheet: [`Testing_ACE.xlsx`](Testing_ACE.xlsx)

Regenerate after updating the Excel log:

```bash
python models/experimental/ace_step_v1_5/perf/export_testing_ace_md.py
```

## Configuration

**Mention the demo command:**

```bash
python models/experimental/ace_step_v1_5/run_prompt_to_wav.py   --variant acestep-v15-base   --lm_variant acestep-5Hz-lm-4B   --mesh-device BH_QB --prompt "Guitar "   --infer_steps 50 --guidance_scale 7 --duration_sec 60      --out /tmp/base_60_4.wav
```

| Varient | base | turbo | stf |
| --- | --- | --- | --- |
| Prompt | Guitar | Guitar | Guitar |
| Duration | 15s/30s/60s | 15s/30s/60s | 15s/30s/60s |
| Mesh Device | BH QB |  |  |
| Steps | 50 | 8 | 50 |
| Branch: | ign/ACE_demo_modified |  |  |

## Data for 15s .wav generation

Varient : acestep-v15-turbo | lm-varient : acestep-5Hz-lm-0.6B

| Metric | Value | Description | OUTPUT AUDIO GOOD/BAD |
| --- | --- | --- | --- |
| Wall Time | 83.39 s | End-to-end time from start to finish | GOOD |
| LM Total Time | 8.13 s | LLM planning (generation + parsing) |  |
| DiT Total Time | 1.03 s | Diffusion (all steps combined) |  |
| VAE Decode Time | 10.10 s | Decode latents to audio waveform |  |
| Tokens/sec | 40.9 tok/s | LLM token generation throughput (238 new tokens) |  |

Varient : acestep-v15-turbo | lm-varient : acestep-5Hz-lm-1.75B

| Metric | Value | Description | OUTPUT AUDIO GOOD/BAD |
| --- | --- | --- | --- |
| Wall Time | 112.21 s | End-to-end time from start to finish | GOOD |
| LM Total Time | 8.45 s | LLM planning (generation + parsing) |  |
| DiT Total Time | 1.04 s | Diffusion (all steps combined) |  |
| VAE Decode Time | 2.08 s | Decode latents to audio waveform |  |
| Tokens/sec | 29.6 tok/s | LLM token generation throughput (250 new tokens) |  |

Varient : acestep-v15-turbo | lm-varient : acestep-5Hz-lm-4B

| Metric | Value | Description | OUTPUT AUDIO GOOD/BAD |
| --- | --- | --- | --- |
| Wall Time | 118.77 s | End-to-end time from start to finish | GOOD |
| LM Total Time | 23.80 s | LLM planning (generation + parsing) |  |
| DiT Total Time | 1.48 s | Diffusion (all steps combined) |  |
| VAE Decode Time | 2.13 s | Decode latents to audio waveform |  |
| Tokens/sec | 10.6 tok/s | LLM token generation throughput (253 new tokens) |  |

Varient : acestep-v15-base | lm-varient : acestep-5Hz-lm-0.6B

| Metric | Value | Description | OUTPUT AUDIO GOOD/BAD |
| --- | --- | --- | --- |
| Wall Time | 76.60 s | End-to-end time from start to finish | GOOD |
| LM Total Time | 6.19 s | LLM planning (generation + parsing) |  |
| DiT Total Time | 5.71 s | Diffusion (all steps combined) |  |
| VAE Decode Time | 2.15 s | Decode latents to audio waveform |  |
| Tokens/sec | 41.2 tok/s | LLM token generation throughput (255 new tokens) |  |

Varient : acestep-v15-base | lm-varient : acestep-5Hz-lm-1.75B

| Metric | Value | Description | OUTPUT AUDIO GOOD/BAD |
| --- | --- | --- | --- |
| Wall Time | 116.99 s | End-to-end time from start to finish | GOOD |
| LM Total Time | 8.24 s | LLM planning (generation + parsing) |  |
| DiT Total Time | 5.62 s | Diffusion (all steps combined) |  |
| VAE Decode Time | 2.16 s | Decode latents to audio waveform |  |
| Tokens/sec | 30.4 tok/s | LLM token generation throughput (250 new tokens) |  |

Varient : acestep-v15-base | lm-varient : acestep-5Hz-lm-4B

| Metric | Value | Description | OUTPUT AUDIO GOOD/BAD |
| --- | --- | --- | --- |
| Wall Time | 104.91 s | End-to-end time from start to finish | GOOD |
| LM Total Time | 11.94 s | LLM planning (generation + parsing) |  |
| DiT Total Time | 1.73 s | Diffusion (all steps combined) |  |
| VAE Decode Time | 2.09 s | Decode latents to audio waveform |  |
| Tokens/sec | 21.2 tok/s | LLM token generation throughput (253 new tokens) |  |

Varient : acestep-v15-sft | lm-varient : acestep-5Hz-lm-0.6B

| Metric | Value | Description | OUTPUT AUDIO GOOD/BAD |
| --- | --- | --- | --- |
| Wall Time | 76.83 s | End-to-end time from start to finish | GOOD |
| LM Total Time | 6.11 s | LLM planning (generation + parsing) |  |
| DiT Total Time | 5.67 s | Diffusion (all steps combined) |  |
| VAE Decode Time | 2.13 s | Decode latents to audio waveform |  |
| Tokens/sec | 41.7 tok/s | LLM token generation throughput (255 new tokens) |  |

Varient : acestep-v15-sft | lm-varient : acestep-5Hz-lm-1.75B

| Metric | Value | Description | OUTPUT AUDIO GOOD/BAD |
| --- | --- | --- | --- |
| Wall Time | 116.88 s | End-to-end time from start to finish | GOOD |
| LM Total Time | 8.23 s | LLM planning (generation + parsing) |  |
| DiT Total Time | 5.57 s | Diffusion (all steps combined) |  |
| VAE Decode Time | 2.14 s | Decode latents to audio waveform |  |
| Tokens/sec | 30.4 tok/s | LLM token generation throughput (250 new tokens) |  |

Varient : acestep-v15-sft | lm-varient : acestep-5Hz-lm-4B

| Metric | Value | Description | OUTPUT AUDIO GOOD/BAD |
| --- | --- | --- | --- |
| Wall Time | 199.93 s | End-to-end time from start to finish | GOOD |
| LM Total Time | 24.34 s | LLM planning (generation + parsing) |  |
| DiT Total Time | 6.07 s | Diffusion (all steps combined) |  |
| VAE Decode Time | 10.29 s | Decode latents to audio waveform |  |
| Tokens/sec | 10.6 tok/s | LLM token generation throughput (257 new tokens) |  |

## Data for 30s .wav generation

Varient : acestep-v15-turbo | lm-varient : acestep-5Hz-lm-0.6B

| Metric | Value | Description | OUTPUT AUDIO GOOD/BAD |
| --- | --- | --- | --- |
| Wall Time | 89.50 s | End-to-end time from start to finish | GOOD |
| LM Total Time | 20.25 s | LLM planning (generation + parsing) |  |
| DiT Total Time | 1.70 s | Diffusion (all steps combined) |  |
| VAE Decode Time | 22.66 s | Decode latents to audio waveform |  |
| Tokens/sec | 24.6 tok/s | LLM token generation throughput (313 new tokens) |  |

Varient : acestep-v15-turbo | lm-varient : acestep-5Hz-lm-1.75B

| Metric | Value | Description | OUTPUT AUDIO GOOD/BAD |
| --- | --- | --- | --- |
| Wall Time | 112.11 s | End-to-end time from start to finish | GOOD |
| LM Total Time | 14.39 s | LLM planning (generation + parsing) |  |
| DiT Total Time | 1.16 s | Diffusion (all steps combined) |  |
| VAE Decode Time | 11.27 s | Decode latents to audio waveform |  |
| Tokens/sec | 22.6 tok/s | LLM token generation throughput (325 new tokens) |  |

Varient : acestep-v15-turbo | lm-varient : acestep-5Hz-lm-4B

| Metric | Value | Description | OUTPUT AUDIO GOOD/BAD |
| --- | --- | --- | --- |
| Wall Time | 103.33 s | End-to-end time from start to finish | GOOD |
| LM Total Time | 15.96 s | LLM planning (generation + parsing) |  |
| DiT Total Time | 1.19 s | Diffusion (all steps combined) |  |
| VAE Decode Time | 11.43 s | Decode latents to audio waveform |  |
| Tokens/sec | 20.5 tok/s | LLM token generation throughput (328 new tokens) |  |

Varient : acestep-v15-base | lm-varient : acestep-5Hz-lm-0.6B

| Metric | Value | Description | OUTPUT AUDIO GOOD/BAD |
| --- | --- | --- | --- |
| Wall Time | 86.65 s | End-to-end time from start to finish | Partially good |
| LM Total Time | 13.38 s | LLM planning (generation + parsing) |  |
| DiT Total Time | 7.05 s | Diffusion (all steps combined) |  |
| VAE Decode Time | 19.11 s | Decode latents to audio waveform |  |
| Tokens/sec | 24.7 tok/s | LLM token generation throughput (330 new tokens) |  |

Varient : acestep-v15-base | lm-varient : acestep-5Hz-lm-1.75B

| Metric | Value | Description | OUTPUT AUDIO GOOD/BAD |
| --- | --- | --- | --- |
| Wall Time | 119.37 s | End-to-end time from start to finish | Partially good |
| LM Total Time | 14.45 s | LLM planning (generation + parsing) |  |
| DiT Total Time | 6.97 s | Diffusion (all steps combined) |  |
| VAE Decode Time | 11.36 s | Decode latents to audio waveform |  |
| Tokens/sec | 22.5 tok/s | LLM token generation throughput (325 new tokens) |  |

Varient : acestep-v15-base | lm-varient : acestep-5Hz-lm-4B

| Metric | Value | Description | OUTPUT AUDIO GOOD/BAD |
| --- | --- | --- | --- |
| Wall Time | 108.96 s | End-to-end time from start to finish | Partially good |
| LM Total Time | 15.85 s | LLM planning (generation + parsing) |  |
| DiT Total Time | 7.13 s | Diffusion (all steps combined) |  |
| VAE Decode Time | 11.40 s | Decode latents to audio waveform |  |
| Tokens/sec | 20.7 tok/s | LLM token generation throughput (328 new tokens) |  |

Varient : acestep-v15-sft | lm-varient : acestep-5Hz-lm-0.6B

| Metric | Value | Description | OUTPUT AUDIO GOOD/BAD |
| --- | --- | --- | --- |
| Wall Time | 78.88 s | End-to-end time from start to finish | GOOD |
| LM Total Time | 13.36 s | LLM planning (generation + parsing) |  |
| DiT Total Time | 7.10 s | Diffusion (all steps combined) |  |
| VAE Decode Time | 11.52 s | Decode latents to audio waveform |  |
| Tokens/sec | 24.7 tok/s | LLM token generation throughput (330 new tokens) |  |

Varient : acestep-v15-sft | lm-varient : acestep-5Hz-lm-1.75B

| Metric | Value | Description | OUTPUT AUDIO GOOD/BAD |
| --- | --- | --- | --- |
| Wall Time | 118.87 s | End-to-end time from start to finish | Partially good |
| LM Total Time | 14.43 s | LLM planning (generation + parsing) |  |
| DiT Total Time | 6.94 s | Diffusion (all steps combined) |  |
| VAE Decode Time | 11.31 s | Decode latents to audio waveform |  |
| Tokens/sec | 22.5 tok/s | LLM token generation throughput (325 new tokens) |  |

Varient : acestep-v15-sft | lm-varient : acestep-5Hz-lm-4B

| Metric | Value | Description | OUTPUT AUDIO GOOD/BAD |
| --- | --- | --- | --- |
| Wall Time | 171.40 s | End-to-end time from start to finish | Partially good |
| LM Total Time | 29.50 s | LLM planning (generation + parsing) |  |
| DiT Total Time | 7.62 s | Diffusion (all steps combined) |  |
| VAE Decode Time | 19.06 s | Decode latents to audio waveform |  |
| Tokens/sec | 11.3 tok/sec | LLM token generation throughput (332 new tokens) |  |

## Data for 60s .wav generation

Varient : acestep-v15-turbo | lm-varient : acestep-5Hz-lm-0.6B

| Metric | Value | Description | OUTPUT AUDIO GOOD/BAD |
| --- | --- | --- | --- |
| Wall Time | 94.45 s | End-to-end time from start to finish | GOOD |
| LM Total Time | 18.66 s | LLM planning (generation + parsing) |  |
| DiT Total Time | 2.09 s | Diffusion (all steps combined) |  |
| VAE Decode Time | 26.37 s | Decode latents to audio waveform |  |
| Tokens/sec | 24.8 tok/s | LLM token generation throughput (462 new tokens) |  |

Varient : acestep-v15-turbo | lm-varient : acestep-5Hz-lm-1.75B

| Metric | Value | Description | OUTPUT AUDIO GOOD/BAD |
| --- | --- | --- | --- |
| Wall Time | 132.28 s | End-to-end time from start to finish | GOOD |
| LM Total Time | 21.06 s | LLM planning (generation + parsing) |  |
| DiT Total Time | 1.88 s | Diffusion (all steps combined) |  |
| VAE Decode Time | 22.81 s | Decode latents to audio waveform |  |
| Tokens/sec | 22.6  tok/s | LLM token generation throughput (476 new tokens) |  |

Varient : acestep-v15-turbo | lm-varient : acestep-5Hz-lm-4B

| Metric | Value | Description | OUTPUT AUDIO GOOD/BAD |
| --- | --- | --- | --- |
| Wall Time | 122.06s | End-to-end time from start to finish | GOOD |
| LM Total Time | 23.39s | LLM planning (generation + parsing) |  |
| DiT Total Time | 1.08s | Diffusion (all steps combined) |  |
| VAE Decode Time | 23.41s | Decode latents to audio waveform |  |
| Tokens/sec | 20.4 tok/s | LLM token generation throughput (328 new tokens) |  |

Varient : acestep-v15-base | lm-varient : acestep-5Hz-lm-0.6B

| Metric | Value | Description | OUTPUT AUDIO GOOD/BAD |
| --- | --- | --- | --- |
| Wall Time | 101.40 s | End-to-end time from start to finish | Partially good |
| LM Total Time | 19.44 s | LLM planning (generation + parsing) |  |
| DiT Total Time | 11.26 s | Diffusion (all steps combined) |  |
| VAE Decode Time | 22.72 s | Decode latents to audio waveform |  |
| Tokens/sec | 24.7 tok/s | LLM token generation throughput (480 new tokens) |  |

Varient : acestep-v15-base | lm-varient : acestep-5Hz-lm-1.75B

| Metric | Value | Description | OUTPUT AUDIO GOOD/BAD |
| --- | --- | --- | --- |
| Wall Time | 140.23 s | End-to-end time from start to finish | Partially good |
| LM Total Time | 20.98 s | LLM planning (generation + parsing) |  |
| DiT Total Time | 10.64 s | Diffusion (all steps combined) |  |
| VAE Decode Time | 22.71 s | Decode latents to audio waveform |  |
| Tokens/sec | 22.7 tok/s | LLM token generation throughput (476 new tokens) |  |

Varient : acestep-v15-base | lm-varient : acestep-5Hz-lm-4B

| Metric | Value | Description | OUTPUT AUDIO GOOD/BAD |
| --- | --- | --- | --- |
| Wall Time | 132.92 s | End-to-end time from start to finish | Partially good |
| LM Total Time | 23.60 s | LLM planning (generation + parsing) |  |
| DiT Total Time | 10.66 s | Diffusion (all steps combined) |  |
| VAE Decode Time | 23.18 s | Decode latents to audio waveform |  |
| Tokens/sec | 20.2 tok/s | LLM token generation throughput (477 new tokens) |  |

Varient : acestep-v15-sft | lm-varient : acestep-5Hz-lm-0.6B

| Metric | Value | Description | OUTPUT AUDIO GOOD/BAD |
| --- | --- | --- | --- |
| Wall Time | 100.52 s | End-to-end time from start to finish | Partially good |
| LM Total Time | 19.48 s | LLM planning (generation + parsing) |  |
| DiT Total Time | 10.63 s | Diffusion (all steps combined) |  |
| VAE Decode Time | 22.90 s | Decode latents to audio waveform |  |
| Tokens/sec | 24.6 tok/s | LLM token generation throughput (480 new tokens) |  |

Varient : acestep-v15-sft | lm-varient : acestep-5Hz-lm-1.75B

| Metric | Value | Description | OUTPUT AUDIO GOOD/BAD |
| --- | --- | --- | --- |
| Wall Time | 141.09 s | End-to-end time from start to finish | Noise |
| LM Total Time | 20.99 s | LLM planning (generation + parsing) |  |
| DiT Total Time | 10.68 s | Diffusion (all steps combined) |  |
| VAE Decode Time | 22.58 s | Decode latents to audio waveform |  |
| Tokens/sec | 22.7 tok/s | LLM token generation throughput (476 new tokens) |  |

Varient : acestep-v15-sft | lm-varient : acestep-5Hz-lm-4B

| Metric | Value | Description | OUTPUT AUDIO GOOD/BAD |
| --- | --- | --- | --- |
| Wall Time | 171.15 s | End-to-end time from start to finish | Noise |
| LM Total Time | 23.69 s | LLM planning (generation + parsing) |  |
| DiT Total Time | 10.70 s | Diffusion (all steps combined) |  |
| VAE Decode Time | 22.65 s | Decode latents to audio waveform |  |
| Tokens/sec | 20.3 tok/s | LLM token generation throughput (482 new tokens) |  |
