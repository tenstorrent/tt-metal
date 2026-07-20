<!-- Generated from perf/Testing_ACE.xlsx by export_testing_ace_md.py -->

# ACE-Step benchmark results (BH_QB)

Source spreadsheet: [`Testing_ACE.xlsx`](Testing_ACE.xlsx)

Regenerate after updating the Excel log:

```bash
python models/experimental/ace_step_v1_5/perf/export_testing_ace_md.py
```

## Configuration

> **Note:** The tables below are a historical Guitar / 15–60 s matrix (turbo 8 / base-sft 50).
> For **client RTF comparison vs A100 / RTX 3090**, re-run with upstream-style inputs:
>
> ```bash
> python models/experimental/ace_step_v1_5/demo/run_prompt_to_wav.py \
>   --mesh-device BH_QB --upstream-benchmark \
>   --lm_variant acestep-5Hz-lm-1.7B \
>   --out /tmp/upstream_rtf_compare.wav
> ```
>
> That forces **170.64 s**, **60 steps**, **guidance 15**, **Euler**, **`acestep-v15-base`**.

**Historical demo command (Guitar matrix):**

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
| Wall Time | 111.75 s | End-to-end time from start to finish | GOOD |
| LM Total Time | 5.87 s | LLM planning (generation + parsing) |  |
| DiT Total Time | 1.26 s | Diffusion (all steps combined) |  |
| VAE Decode Time | 2.13 s | Decode latents to audio waveform |  |
| Tokens/sec | 40.5 tok/s | LLM token generation throughput (255 new tokens) |  |

Varient : acestep-v15-base | lm-varient : acestep-5Hz-lm-1.75B

| Metric | Value | Description | OUTPUT AUDIO GOOD/BAD |
| --- | --- | --- | --- |
| Wall Time | 116.51 s | End-to-end time from start to finish | GOOD |
| LM Total Time | 8.45 s | LLM planning (generation + parsing) |  |
| DiT Total Time | 1.26 s | Diffusion (all steps combined) |  |
| VAE Decode Time | 2.11 s | Decode latents to audio waveform |  |
| Tokens/sec | 29.8 tok/s | LLM token generation throughput (250 new tokens) |  |

Varient : acestep-v15-base | lm-varient : acestep-5Hz-lm-4B

| Metric | Value | Description | OUTPUT AUDIO GOOD/BAD |
| --- | --- | --- | --- |
| Wall Time | 146.81 s | End-to-end time from start to finish | GOOD |
| LM Total Time | 11.79 s | LLM planning (generation + parsing) |  |
| DiT Total Time | 1.25 s | Diffusion (all steps combined) |  |
| VAE Decode Time | 2.14 s | Decode latents to audio waveform |  |
| Tokens/sec | 21.8 tok/s | LLM token generation throughput (253 new tokens) |  |

Varient : acestep-v15-sft | lm-varient : acestep-5Hz-lm-0.6B

| Metric | Value | Description | OUTPUT AUDIO GOOD/BAD |
| --- | --- | --- | --- |
| Wall Time | 113.50 s | End-to-end time from start to finish | GOOD |
| LM Total Time | 5.81 s | LLM planning (generation + parsing) |  |
| DiT Total Time | 1.25 s | Diffusion (all steps combined) |  |
| VAE Decode Time | 2.13 s | Decode latents to audio waveform |  |
| Tokens/sec | 41.0 tok/s | LLM token generation throughput (255 new tokens) |  |

Varient : acestep-v15-sft | lm-varient : acestep-5Hz-lm-1.75B

| Metric | Value | Description | OUTPUT AUDIO GOOD/BAD |
| --- | --- | --- | --- |
| Wall Time | 115.91 s | End-to-end time from start to finish | GOOD |
| LM Total Time | 8.39 s | LLM planning (generation + parsing) |  |
| DiT Total Time | 1.24 s | Diffusion (all steps combined) |  |
| VAE Decode Time | 2.13 s | Decode latents to audio waveform |  |
| Tokens/sec | 29.8 tok/s | LLM token generation throughput (250 new tokens) |  |

Varient : acestep-v15-sft | lm-varient : acestep-5Hz-lm-4B

| Metric | Value | Description | OUTPUT AUDIO GOOD/BAD |
| --- | --- | --- | --- |
| Wall Time | 147.35 s | End-to-end time from start to finish | GOOD |
| LM Total Time | 12.03 s | LLM planning (generation + parsing) |  |
| DiT Total Time | 1.31 s | Diffusion (all steps combined) |  |
| VAE Decode Time | 2.15 s | Decode latents to audio waveform |  |
| Tokens/sec | 21.4 tok/s | LLM token generation throughput (257 new tokens) |  |

## Data for 30s .wav generation

Varient : acestep-v15-turbo | lm-varient : acestep-5Hz-lm-0.6B

| Metric | Value | Description | OUTPUT AUDIO GOOD/BAD |
| --- | --- | --- | --- |
| Wall Time | 141.80 s | End-to-end time from start to finish | GOOD |
| LM Total Time | 22.73 s | LLM planning (generation + parsing) |  |
| DiT Total Time | 1.17 s | Diffusion (all steps combined) |  |
| VAE Decode Time | 22.89 s | Decode latents to audio waveform |  |
| Tokens/sec | 13.2 tok/s | LLM token generation throughput (313 new tokens) |  |

Varient : acestep-v15-turbo | lm-varient : acestep-5Hz-lm-1.75B

| Metric | Value | Description | OUTPUT AUDIO GOOD/BAD |
| --- | --- | --- | --- |
| Wall Time | 125.24 s | End-to-end time from start to finish | GOOD |
| LM Total Time | 15.86 s | LLM planning (generation + parsing) |  |
| DiT Total Time | 0.77 s | Diffusion (all steps combined) |  |
| VAE Decode Time | 20.42 s | Decode latents to audio waveform |  |
| Tokens/sec | 20.5 tok/s | LLM token generation throughput (325 new tokens) |  |

Varient : acestep-v15-turbo | lm-varient : acestep-5Hz-lm-4B

| Metric | Value | Description | OUTPUT AUDIO GOOD/BAD |
| --- | --- | --- | --- |
| Wall Time | 168.83 s | End-to-end time from start to finish | GOOD |
| LM Total Time | 30.29 s | LLM planning (generation + parsing) |  |
| DiT Total Time | 1.17 s | Diffusion (all steps combined) |  |
| VAE Decode Time | 30.23 s | Decode latents to audio waveform |  |
| Tokens/sec | 11.0 tok/s | LLM token generation throughput (328 new tokens) |  |

Varient : acestep-v15-base | lm-varient : acestep-5Hz-lm-0.6B

| Metric | Value | Description | OUTPUT AUDIO GOOD/BAD |
| --- | --- | --- | --- |
| Wall Time | 133.60 s | End-to-end time from start to finish | Partially good |
| LM Total Time | 13.64 s | LLM planning (generation + parsing) |  |
| DiT Total Time | 9.26 s | Diffusion (all steps combined) |  |
| VAE Decode Time | 21.15 s | Decode latents to audio waveform |  |
| Tokens/sec | 22.8 tok/s | LLM token generation throughput (330 new tokens) |  |

Varient : acestep-v15-base | lm-varient : acestep-5Hz-lm-1.75B

| Metric | Value | Description | OUTPUT AUDIO GOOD/BAD |
| --- | --- | --- | --- |
| Wall Time | 151.58 s | End-to-end time from start to finish | Partially good |
| LM Total Time | 26.11 s | LLM planning (generation + parsing) |  |
| DiT Total Time | 11.06 s | Diffusion (all steps combined) |  |
| VAE Decode Time | 21.20 s | Decode latents to audio waveform |  |
| Tokens/sec | 12.4 tok/s | LLM token generation throughput (325 new tokens) |  |

Varient : acestep-v15-base | lm-varient : acestep-5Hz-lm-4B

| Metric | Value | Description | OUTPUT AUDIO GOOD/BAD |
| --- | --- | --- | --- |
| Wall Time | 179.63 s | End-to-end time from start to finish | Partially good |
| LM Total Time | 29.97 s | LLM planning (generation + parsing) |  |
| DiT Total Time | 9.72 s | Diffusion (all steps combined) |  |
| VAE Decode Time | 21.01 s | Decode latents to audio waveform |  |
| Tokens/sec | 11.1 tok/s | LLM token generation throughput (328 new tokens) |  |

Varient : acestep-v15-sft | lm-varient : acestep-5Hz-lm-0.6B

| Metric | Value | Description | OUTPUT AUDIO GOOD/BAD |
| --- | --- | --- | --- |
| Wall Time | 122.81 s | End-to-end time from start to finish | Partially good |
| LM Total Time | 13.25 s | LLM planning (generation + parsing) |  |
| DiT Total Time | 8.98 s | Diffusion (all steps combined) |  |
| VAE Decode Time | 12.59 s | Decode latents to audio waveform |  |
| Tokens/sec | 23.6 tok/s | LLM token generation throughput (330 new tokens) |  |

Varient : acestep-v15-sft | lm-varient : acestep-5Hz-lm-1.75B

| Metric | Value | Description | OUTPUT AUDIO GOOD/BAD |
| --- | --- | --- | --- |
| Wall Time | 127.25 s | End-to-end time from start to finish | GOOD |
| LM Total Time | 15.18 s | LLM planning (generation + parsing) |  |
| DiT Total Time | 8.89 s | Diffusion (all steps combined) |  |
| VAE Decode Time | 12.51 s | Decode latents to audio waveform |  |
| Tokens/sec | 21.4 tok/s | LLM token generation throughput (325 new tokens) |  |

Varient : acestep-v15-sft | lm-varient : acestep-5Hz-lm-4B

| Metric | Value | Description | OUTPUT AUDIO GOOD/BAD |
| --- | --- | --- | --- |
| Wall Time | 155.56 s | End-to-end time from start to finish | Partially good |
| LM Total Time | 17.67 s | LLM planning (generation + parsing) |  |
| DiT Total Time | 9.24 s | Diffusion (all steps combined) |  |
| VAE Decode Time | 12.76 s | Decode latents to audio waveform |  |
| Tokens/sec | 18.8 tok/sec | LLM token generation throughput (332 new tokens) |  |

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
