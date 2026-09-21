# FLUX.2 attention frontier evaluation

48 generated images out of 48 planned: seven frozen attention recipes plus the separate stock ring control. Three prompts × seeds 0/42; 50 steps, 1024×1024, guidance 4, no prompt upsampling. Checkpoint, prompt embeddings, noise seeds, scheduler, encoder and VAE settings match.

**Exploratory evaluation:** exact replay is measured, but is not an admission requirement for these images. Finite-value, shape and no-fallback checks remain mandatory. Historical stock runs were non-repeatable; the fresh same-input stock control below determines whether that issue persists in this cohort. Cross-variant latent L2 measures end-to-end change relative to D, not per-call SDPA error or error against a ground-truth reference model.

**Hardware cohort:** bh-lb-08, IRD 221619; eight Blackhole devices, firmware 19.13.1.0; TT-KMD 2.9.0. All image variants in this report use this allocation. Earlier bh-51 results, reset attempts and thermal/PCIe failures are retained separately in STATUS.md. Converted-weight caching is enabled with the pinned-host-memory-cache workaround; this setting is common to all choices and does not change attention arithmetic.

Eight Blackhole chips, 2×4 mesh, SP2/TP4, 48 global heads, head width 128. 4096 image + 512 text tokens; frontier Q256/K512. The missing residual and per-head normalization are repaired identically using the main-style fused implementation. Stock conditioning is held fixed; the additional guidance/phase correction is not enabled. Thus this is not a full-reference FLUX.2 numerical qualification; known conditioning differences from Diffusers are deliberately common to all choices.

## Weight caching

All 24/24 initial component loads hit the converted-weight cache. Pipeline setup was 9.03–9.22 s across the eight choices (median 9.07 s). Unexpected conversion fallback was forbidden.

See [the separate cold/warm qualification](../WEIGHT_CACHE.md) for startup timings, exact-weight checks and the pinned-host-memory-cache workaround. Caching changes startup, not the measured attention arithmetic.

## Integration qualification

Failed choices are not substituted or represented by another choice's images. 'Initial gates passed' records only the original qualification calls, not proof of general deterministic execution. See the repeatability investigation for the limits of interpreting paired image differences.

| Choice | Result | Images |
|---|---|---:|
| stock | Exploratory: model replay exact; 0/6 blocks differ | 6 |
| D | Exploratory: model replay exact; 0/6 blocks differ | 6 |
| C | Exploratory: model replay exact; 0/6 blocks differ | 6 |
| B | Exploratory: model replay exact; 0/6 blocks differ | 6 |
| A | Exploratory: model replay exact; 0/6 blocks differ | 6 |
| E | Exploratory: model replay exact; 0/6 blocks differ | 6 |
| F | Exploratory: model replay exact; 0/6 blocks differ | 6 |
| G | Exploratory: model replay exact; 0/6 blocks differ | 6 |

## Stock repeatability control

One process, identical cached prompt embeddings and noise seed, stock attention, 50 denoising steps, three untraced calls followed by three steady traced calls. The first trace-capture call is intentionally discarded. See the measured equality results below; this control uses stock attention, not a frontier replacement.

| Call | Mode | Final-latent L2 vs first | Exact vs first |
|---|---|---:|---|
| 0 | untraced | 0.000% | True |
| 1 | untraced | 0.000% | True |
| 2 | untraced | 0.000% | True |
| 4 | traced | 0.000% | True |
| 5 | traced | 0.000% | True |
| 6 | traced | 0.000% | True |

All measured stock calls were bitwise identical. This finite control does not prove general determinism or identify which changed hardware/runtime condition explains the historical failures.


## CLIP and output differences

Raw OpenAI CLIP ViT-B/32 normalized text/image cosine (not ×100). Six paired images per choice are exploratory, not a statistically powered model evaluation. CLIP measures text alignment, not fidelity to a reference image. D is a comparator, not ground truth.

| Choice | Mean CLIP | Min–max CLIP | Mean paired Δ vs D | Mean final-latent L2 vs D |
|---|---:|---:|---:|---:|
| stock | 0.34869 | 0.32247–0.36181 | +0.00106 | 13.48% |
| D | 0.34763 | 0.32025–0.36369 | +0.00000 | 0.00% |
| C | 0.34792 | 0.32223–0.36300 | +0.00029 | 4.16% |
| B | 0.34746 | 0.32034–0.36048 | -0.00017 | 10.98% |
| A | 0.34725 | 0.31978–0.36147 | -0.00038 | 13.88% |
| E | 0.34814 | 0.31856–0.36397 | +0.00051 | 11.88% |
| F | 0.34782 | 0.31903–0.36154 | +0.00019 | 13.38% |
| G | 0.34737 | 0.32443–0.36379 | -0.00026 | 26.50% |

## Performance

Image-run block timings below are warmed but still subject to clock, temperature and run-order variation. These are measured full-model block costs, not SDPA-only FLOP utilization.

Block times are isolated **full-block blocking mesh trace replays**, including preprocessing and communication, after 250 warmups, with 50 timed samples. They exclude capture, compilation and weight conversion. These are host-observed accelerator trace latencies, not hardware-counter FPU utilization. Denoising-step times are separate traced-pipeline host measurements.

Stock uses joint ring SDPA; all seven frontier variants use local Q plus prepared-format KV all-gather. Stock-versus-frontier speed differences therefore include the communication schedule, not just numerical arithmetic. The frontier adapter is not overlapped ring attention.

| Choice | Median step, ms | Dual 0 | Dual 3 | Dual 7 | Single 0 | Single 23 | Single 47 |
|---|---:|---:|---:|---:|---:|---:|---:|
| stock | 399.69 | 8.648 | 8.579 | 8.449 | 6.554 | 6.715 | 6.590 |
| D | 428.23 | 9.204 | 9.086 | 9.141 | 7.060 | 7.202 | 7.066 |
| C | 417.13 | 9.008 | 8.904 | 8.957 | 6.835 | 6.967 | 6.883 |
| B | 398.40 | 8.757 | 8.567 | 8.699 | 6.605 | 6.766 | 6.556 |
| A | 396.96 | 8.632 | 8.564 | 8.660 | 6.600 | 6.616 | 6.526 |
| E | 389.34 | 8.506 | 8.456 | 8.511 | 6.402 | 6.512 | 6.409 |
| F | 393.98 | 8.611 | 8.523 | 8.522 | 6.478 | 6.611 | 6.565 |
| G | 381.99 | 8.245 | 8.237 | 8.242 | 6.136 | 6.259 | 6.197 |

All block columns are milliseconds. Raw replay samples, min/max and exact-replay checks are retained in each manifest. Values from this 4608-token model run do not characterize the very-long-context precision loss observed in the standalone SDPA experiments.

**Telemetry snapshot:** the recorded sample showed 66.7–71.0°C and 1150–1262 MHz. These are observed timings on this allocation, not qualified peak performance. Thermal/clock variation can bias both run-order comparisons and speedups. This observation does not establish the cause of numerical nondeterminism.

## Sustained-workload block check

To reduce warmup and run-order bias, this follow-up uses the same D-captured inputs for every choice, 5000 initial dual-block replays, then 750 warmups and 50 timed samples per block. The intended schedule is one forward and one reverse pass. Each cell pools the available completed rounds (50 samples per round). The rounds column makes incomplete coverage explicit. Replay differences are recorded; a timing measurement is not a correctness pass. Prefer these warmed block numbers over the initial image-run block measurements.

| Choice | Dual 0 | Dual 3 | Dual 7 | Single 0 | Single 23 | Single 47 | Rounds | Max round-median difference |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| stock | 8.643 | 8.630 | 8.708 | 6.792 | 6.822 | 6.727 | 2 | 2.08% |
| D | 9.168 | 9.212 | 9.254 | 7.226 | 7.276 | 7.122 | 2 | 3.56% |
| C | 9.063 | 9.065 | 9.028 | 7.066 | 7.101 | 6.978 | 2 | 2.84% |
| B | 8.709 | 8.740 | 8.791 | 6.765 | 6.792 | 6.607 | 2 | 0.95% |
| A | 8.705 | 8.756 | 8.803 | 6.758 | 6.798 | 6.645 | 2 | 1.55% |
| E | 8.596 | 8.612 | 8.637 | 6.651 | 6.698 | 6.540 | 2 | 0.46% |
| F | 8.700 | 8.709 | 8.726 | 6.763 | 6.789 | 6.662 | 2 | 0.58% |
| G | 8.546 | 8.571 | 8.578 | 6.594 | 6.637 | 6.520 | 2 | 0.92% |

All block times are milliseconds; raw samples are in `block-sweep.json`.

Replay mismatches in the saved diagnostic sweep: 0/96 block measurements.


## Paired images

- [Prompt 0, seed 0](comparison/prompt0-seed0.png)
- [Prompt 0, seed 42](comparison/prompt0-seed42.png)
- [Prompt 1, seed 0](comparison/prompt1-seed0.png)
- [Prompt 1, seed 42](comparison/prompt1-seed42.png)
- [Prompt 2, seed 0](comparison/prompt2-seed0.png)
- [Prompt 2, seed 42](comparison/prompt2-seed42.png)

Full-resolution originals and final latents are in each variant directory; all six samples per variant are included, without selecting favorable seeds.

## Identical real-QKV operator checks

All seven recipes evaluated on the same D captures from four blocks, four selected global heads, 512 selected query rows and all recorded KV tokens. Reference: original BF16 Q/K/V evaluated in FP64. This is an operator check, not final image error.

| Choice | Min–max L2 across captured blocks | Min PCC |
|---|---:|---:|
| D | 0.166%–0.172% | 0.999999 |
| C | 0.170%–0.217% | 0.999998 |
| B | 1.004%–2.020% | 0.999815 |
| A | 1.055%–2.028% | 0.999810 |
| E | 0.986%–2.504% | 0.999690 |
| F | 0.600%–1.962% | 0.999808 |
| G | 4.122%–11.830% | 0.993023 |

## Reproducibility

Each manifest records the pinned checkpoint, source hashes, numeric defines, observed KV transport types, image/latent hashes, software versions and trace verification. Unsupported inputs fail; no attention-mode fallback is allowed. Exploratory mode records replay mismatches instead of asserting exact equality; it does not change numerical recipes. Earlier noise-producing bring-up runs are excluded from this suite.
