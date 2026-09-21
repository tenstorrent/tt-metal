# Wan2.2 480p attention comparison

14/14 videos: two prompts, seed 42, 832x480, 81 frames, 40 steps, CFG 4/3. Eight Blackhole chips, SP4/TP2. All choices reuse the same converted weights. No attention fallback.

Stock uses ring attention. Frontier choices use prepared-format KV all-gather then local-Q attention, with native masking of the eight padded tokens. Cross-attention, projections, FFNs, text encoder, scheduler and VAE are unchanged.

## Generation time and CLIP

Generation time is warmed untraced pipeline wall time, including expert reloads but excluding initial setup, pilot diagnostics and video encoding. CLIP is raw OpenAI ViT-B/32 cosine averaged across eight uncompressed sampled frames; it is not a temporal-quality or reference-fidelity metric. Two videos per choice are exploratory.

| Choice | Butterfly seconds | Human seconds | Butterfly CLIP | Human CLIP |
|---|---:|---:|---:|---:|
| stock | 169.23 | 169.68 | 0.30055 | 0.31681 |
| D | 262.65 | 266.20 | 0.29653 | 0.30673 |
| C | 236.99 | 237.14 | 0.29745 | 0.30797 |
| B | 188.48 | 188.55 | 0.29963 | 0.30364 |
| E | 162.87 | 162.99 | 0.29946 | 0.30774 |
| F | 175.38 | 175.74 | 0.29762 | 0.30699 |
| G | 159.38 | 159.56 | 0.30321 | 0.30843 |

Total measured video generation: 45.2 minutes, excluding setup, pilots, scoring and qualification.

## Full-block timings

Blocking mesh trace replays, five warmups and 15 samples per block. Includes attention preprocessing and communication. Inputs come from each variant's own two-step pilot, not a shared captured input; short warmup and dynamic clocks limit small-difference interpretation. These are full-block times, not SDPA FLOP utilization.

| Choice | High-noise block 0 ms | High-noise block 20 ms | Low-noise block 0 ms | Low-noise block 20 ms | Replay exact |
|---|---:|---:|---:|---:|---|
| stock | 37.489 | 47.757 | 37.357 | 47.701 | True |
| D | 70.627 | 71.516 | 70.767 | 71.789 | True |
| C | 63.754 | 70.961 | 63.768 | 70.179 | True |
| B | 48.404 | 55.414 | 48.387 | 55.442 | True |
| E | 42.726 | 48.133 | 42.696 | 48.065 | True |
| F | 46.705 | 52.452 | 46.698 | 52.077 | True |
| G | 42.552 | 47.398 | 42.549 | 47.252 | True |

## Identical real-QKV accuracy

Four D-pilot captures, four selected global heads and 256 selected query rows, all 32,760 valid keys. Reference is FP64 attention on the original BF16 Q/K/V. These are operator errors, not video errors. This bounded sample does not qualify all denoising timesteps or the human prompt.

| Choice | Min–max L2 | Min PCC |
|---|---:|---:|
| D | 0.167%–0.199% | 0.999998 |
| C | 0.193%–0.493% | 0.999988 |
| B | 1.367%–5.022% | 0.998769 |
| E | 1.347%–6.083% | 0.998152 |
| F | 1.017%–4.704% | 0.998902 |
| G | 6.093%–23.692% | 0.971679 |

## Paired outputs

- [Prompt 0 sampled frames](prompt0-comparison.png)
  - [stock video](stock/prompt0-seed42.mp4)
  - [D video](D/prompt0-seed42.mp4)
  - [C video](C/prompt0-seed42.mp4)
  - [B video](B/prompt0-seed42.mp4)
  - [E video](E/prompt0-seed42.mp4)
  - [F video](F/prompt0-seed42.mp4)
  - [G video](G/prompt0-seed42.mp4)
- [Prompt 1 sampled frames](prompt1-comparison.png)
  - [stock video](stock/prompt1-seed42.mp4)
  - [D video](D/prompt1-seed42.mp4)
  - [C video](C/prompt1-seed42.mp4)
  - [B video](B/prompt1-seed42.mp4)
  - [E video](E/prompt1-seed42.mp4)
  - [F video](F/prompt1-seed42.mp4)
  - [G video](G/prompt1-seed42.mp4)

## Integration qualification

All six recipes passed the padded SP4/TP2 poison-tail test and exact trace replay. F initially failed because its BFP8 K unpack format was incorrectly retained when reading the BF16 mask palette. An explicit mask-format reconfiguration fixed this; the native exp formula is unchanged. The fix is gated to the padded adapter, preserving the prior unpadded FLUX path.

Each non-stock manifest records Q/K/V formats for all 80 self-attention blocks. E/F transport BFP8_B KV, G transports BFP4_B KV, D/C/B transport BF16 KV; Q remains BF16. F's FP32 destination is independent of its input storage formats.

Converted-weight misses are forbidden by the run harness. Source provenance is in run-metadata.json; individual manifests retain frame/video hashes, timings, captures and cache-load records.
