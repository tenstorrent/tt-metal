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
python models/experimental/ace_step_v1_5/demo/run_prompt_to_wav.py   --variant acestep-v15-base   --lm_variant acestep-5Hz-lm-4B   --mesh-device BH_QB --prompt "Guitar "   --infer_steps 50 --guidance_scale 7 --duration_sec 60      --out /tmp/base_60_4.wav
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
| Wall Time | 111.68 s | End-to-end time from start to finish | GOOD |
| LM Total Time | 5.71 s | LLM planning (generation + parsing) |  |
| DiT Total Time | 0.70 s | Diffusion (all steps combined) |  |
| VAE Decode Time | 2.109 s | Decode latents to audio waveform |  |
| Tokens/sec | 41.7 tok/s | LLM token generation throughput (238 new tokens) |  |

Varient : acestep-v15-turbo | lm-varient : acestep-5Hz-lm-1.75B

| Metric | Value | Description | OUTPUT AUDIO GOOD/BAD |
| --- | --- | --- | --- |
| Wall Time | 116.04 s | End-to-end time from start to finish | GOOD |
| LM Total Time | 8.37 s | LLM planning (generation + parsing) |  |
| DiT Total Time | 0.70 s | Diffusion (all steps combined) |  |
| VAE Decode Time | 2.09 s | Decode latents to audio waveform |  |
| Tokens/sec | 29.9 tok/s | LLM token generation throughput (250 new tokens) |  |

Varient : acestep-v15-turbo | lm-varient : acestep-5Hz-lm-4B

| Metric | Value | Description | OUTPUT AUDIO GOOD/BAD |
| --- | --- | --- | --- |
| Wall Time | 146.37 s | End-to-end time from start to finish | GOOD |
| LM Total Time | 11.99 s | LLM planning (generation + parsing) |  |
| DiT Total Time | 0.70 s | Diffusion (all steps combined) |  |
| VAE Decode Time | 2.11 s | Decode latents to audio waveform |  |
| Tokens/sec | 21.46 tok/s | LLM token generation throughput (253 new tokens) |  |

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

## Data for 60s .wav generation prompt "guitar, saxophone and prominent drums with clear kick and snare"

Varient : acestep-v15-turbo | lm-varient : acestep-5Hz-lm-0.6B

| Metric | Value | Description | OUTPUT AUDIO GOOD/BAD |
| --- | --- | --- | --- |
| Wall Time | 162.61 s | End-to-end time from start to finish | GOOD |
| LM Total Time | 21.21 s | LLM planning (generation + parsing) |  |
| DiT Total Time | 1.16 s | Diffusion (all steps combined) |  |
| VAE Decode Time | 50.73 s | Decode latents to audio waveform |  |
| Tokens/sec | 15.8 tok/s | LLM token generation throughput (335 new tokens) |  |

Varient : acestep-v15-turbo | lm-varient : acestep-5Hz-lm-1.75B

| Metric | Value | Description | OUTPUT AUDIO GOOD/BAD |
| --- | --- | --- | --- |
| Wall Time | 156.96 s | End-to-end time from start to finish | GOOD |
| LM Total Time | 14.93 s | LLM planning (generation + parsing) |  |
| DiT Total Time | 1.14 s | Diffusion (all steps combined) |  |
| VAE Decode Time | 50.03 s | Decode latents to audio waveform |  |
| Tokens/sec | 22.4  tok/s | LLM token generation throughput (335 new tokens) |  |

Varient : acestep-v15-turbo | lm-varient : acestep-5Hz-lm-4B

| Metric | Value | Description | OUTPUT AUDIO GOOD/BAD |
| --- | --- | --- | --- |
| Wall Time | 197.24 s | End-to-end time from start to finish | GOOD |
| LM Total Time | 26.32 s | LLM planning (generation + parsing) |  |
| DiT Total Time | 1.14 s | Diffusion (all steps combined) |  |
| VAE Decode Time | 50.98 s | Decode latents to audio waveform |  |
| Tokens/sec | 12.7 tok/s | LLM token generation throughput (335 new tokens) |  |

Varient : acestep-v15-base | lm-varient : acestep-5Hz-lm-0.6B

| Metric | Value | Description | OUTPUT AUDIO GOOD/BAD |
| --- | --- | --- | --- |
| Wall Time | 169.06 s | End-to-end time from start to finish | Partially good |
| LM Total Time | 14.03 s | LLM planning (generation + parsing) |  |
| DiT Total Time | 12.31 s | Diffusion (all steps combined) |  |
| VAE Decode Time | 52.73 s | Decode latents to audio waveform |  |
| Tokens/sec | 23.9 tok/s | LLM token generation throughput (335 new tokens) |  |

Varient : acestep-v15-base | lm-varient : acestep-5Hz-lm-1.75B

| Metric | Value | Description | OUTPUT AUDIO GOOD/BAD |
| --- | --- | --- | --- |
| Wall Time | 170.17 s | End-to-end time from start to finish | Partially good |
| LM Total Time | 15.25 s | LLM planning (generation + parsing) |  |
| DiT Total Time | 12.28 s | Diffusion (all steps combined) |  |
| VAE Decode Time | 52.68 s | Decode latents to audio waveform |  |
| Tokens/sec | 22.0 tok/s | LLM token generation throughput (335 new tokens) |  |

Varient : acestep-v15-base | lm-varient : acestep-5Hz-lm-4B

| Metric | Value | Description | OUTPUT AUDIO GOOD/BAD |
| --- | --- | --- | --- |
| Wall Time | 208.22 s s | End-to-end time from start to finish | Bad  |
| LM Total Time | 17.18 s | LLM planning (generation + parsing) |  |
| DiT Total Time | 12.55 s | Diffusion (all steps combined) |  |
| VAE Decode Time | 61.55 s | Decode latents to audio waveform |  |
| Tokens/sec | 19.5 tok/s | LLM token generation throughput (335 new tokens) |  |

Varient : acestep-v15-sft | lm-varient : acestep-5Hz-lm-0.6B

| Metric | Value | Description | OUTPUT AUDIO GOOD/BAD |
| --- | --- | --- | --- |
| Wall Time | 168.67 s | End-to-end time from start to finish | Partially good |
| LM Total Time | 13.91 s | LLM planning (generation + parsing) |  |
| DiT Total Time | 12.33 s | Diffusion (all steps combined) |  |
| VAE Decode Time | 52.67 s | Decode latents to audio waveform |  |
| Tokens/sec | 24.1 tok/s | LLM token generation throughput (335 new tokens) |  |

Varient : acestep-v15-sft | lm-varient : acestep-5Hz-lm-1.75B

| Metric | Value | Description | OUTPUT AUDIO GOOD/BAD |
| --- | --- | --- | --- |
| Wall Time | 170.23 s | End-to-end time from start to finish | Good |
| LM Total Time | 15.37 s | LLM planning (generation + parsing) |  |
| DiT Total Time | 12.31 s   | Diffusion (all steps combined) |  |
| VAE Decode Time | 52.78 s | Decode latents to audio waveform |  |
| Tokens/sec | 21.8 tok/s | LLM token generation throughput (335 new tokens) |  |

Varient : acestep-v15-sft | lm-varient : acestep-5Hz-lm-4B

| Metric | Value | Description | OUTPUT AUDIO GOOD/BAD |
| --- | --- | --- | --- |
| Wall Time | 198.61  s | End-to-end time from start to finish | Noise |
| LM Total Time | 16.98 s | LLM planning (generation + parsing) |  |
| DiT Total Time | 12.29 s | Diffusion (all steps combined) |  |
| VAE Decode Time | 52.87 s | Decode latents to audio waveform |  |
| Tokens/sec | 19.7 tok/s | LLM token generation throughput (335 new tokens) |  |
