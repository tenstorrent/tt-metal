# FLUX.2 attention frontier evaluation

24 generated images out of 48 planned: seven frozen attention recipes plus the separate stock ring control. Three prompts × seeds 0/42; 50 steps, 1024×1024, guidance 4, no prompt upsampling. Checkpoint, prompt embeddings, noise seeds, scheduler, encoder and VAE settings match.

**Exploratory results, not a clean attention-quality ranking:** the stock control itself fails repeated-run equality (37.7–43.1% final-latent L2 between untraced 50-step runs). The shared repeatability issue remains unresolved. Initial failures are preserved; the optional exploratory mode records equality failures without aborting image generation. Finite-value, shape and no-fallback checks remain mandatory.

**Hardware blocker:** the sustained sweep was interrupted by a PCIe 0xffffffff device-read failure after elevated-temperature telemetry. Active and queued device tests were stopped; the missing A/B/E/G image reruns have not executed. No device reset was performed. See [repeatability and hardware evidence](../REPEATABILITY.md).

Eight Blackhole chips, 2×4 mesh, SP2/TP4, 48 global heads, head width 128. 4096 image + 512 text tokens; frontier Q256/K512. The missing residual and per-head normalization are repaired identically using the main-style fused implementation. Stock conditioning is held fixed; the additional guidance/phase correction is not enabled. Thus this is not a full-reference FLUX.2 numerical qualification; known conditioning differences from Diffusers are deliberately common to all choices.

## Integration qualification

Failed choices are not substituted or represented by another choice's images. 'Initial gates passed' records only the original qualification calls, not proof of general deterministic execution. See the repeatability investigation for the limits of interpreting paired image differences.

| Choice | Result | Images |
|---|---|---:|
| stock | Initial gates passed | 6 |
| D | Initial gates passed | 6 |
| C | Initial gates passed | 6 |
| B | AssertionError: Steady model trace differs from untraced two-step latents | 0 |
| A | AssertionError: Block replay differs for single.0 | 0 |
| E | AssertionError: Block replay differs for dual.0 | 0 |
| F | Initial gates passed | 6 |
| G | AssertionError: Block replay differs for single.47 | 0 |

## Shared repeatability limitation

One process, identical cached prompt embeddings and noise seed, stock attention, 50 denoising steps, three untraced calls followed by three steady traced calls. The first trace-capture call is intentionally discarded. Equality also fails without tracing, so this is not an attention-variant-specific or trace-only diagnosis.

| Call | Mode | Final-latent L2 vs first | Exact vs first |
|---|---|---:|---|
| 0 | untraced | 0.000% | True |
| 1 | untraced | 43.097% | False |
| 2 | untraced | 37.673% | False |
| 4 | traced | 47.213% | False |
| 5 | traced | 47.213% | False |
| 6 | traced | 57.536% | False |

This is a repeatability defect/limitation in the current full-model stack; its root cause is not established. Standalone attention probes passed exact replay on all eight chips, including relocated/alternating inputs and preceding FP32 matmul. Retaining intermediate block outputs changes the full-model behavior, suggesting timing/lifetime sensitivity, but that is not proof of a particular race. Do not interpret cross-variant final-latent L2 as isolated attention error. See the stock repeatability images and raw diagnostic reports.

[View the six stock-repeat images](../pipeline-replay-stock50-01/decoded-cpu/comparison.png).
After the hardware failure, the saved latents were decoded with the same **CPU FP32
reference VAE** for every repeat; these are not additional TT-decoded frontier samples.
All six images are coherent, with visible changes to pose, fur and window details.
CLIP spans **0.35702–0.35979** on this identical prompt/seed. Calls 4 and 5 have
identical latents, PNG hashes and CLIP scores. This is a within-stock variability
control, not a reference full-model image or a statistically powered estimate of
cross-variant quality differences.


## CLIP and output differences

Raw OpenAI CLIP ViT-B/32 normalized text/image cosine (not ×100). Six paired images per choice are exploratory, not a statistically powered model evaluation. CLIP measures text alignment, not fidelity to a reference image. D is a comparator, not ground truth.

| Choice | Mean CLIP | Min–max CLIP | Mean paired Δ vs D | Mean final-latent L2 vs D |
|---|---:|---:|---:|---:|
| stock | 0.34683 | 0.31894–0.36592 | -0.00080 | 39.23% |
| D | 0.34763 | 0.32025–0.36369 | +0.00000 | 0.00% |
| C | 0.34792 | 0.32223–0.36300 | +0.00029 | 4.16% |
| B | — | — | — | — |
| A | — | — | — | — |
| E | — | — | — | — |
| F | 0.34782 | 0.31903–0.36154 | +0.00019 | 13.38% |
| G | — | — | — | — |

## Performance

The first table preserves the initial image-run observations, **not a steady-state block ranking**.
D and F's early dual-block measurements drifted substantially. Use the later
sustained-workload table for block comparisons, subject to its thermal caveat.

Block times are isolated **full-block blocking mesh trace replays**, including preprocessing and communication, after 250 warmups, with 50 timed samples. They exclude capture, compilation and weight conversion. These are host-observed accelerator trace latencies, not hardware-counter FPU utilization. Denoising-step times are separate traced-pipeline host measurements.

Stock uses joint ring SDPA; all seven frontier variants use local Q plus prepared-format KV all-gather. Stock-versus-frontier speed differences therefore include the communication schedule, not just numerical arithmetic. The frontier adapter is not overlapped ring attention.

| Choice | Median step, ms | Dual 0 | Dual 3 | Dual 7 | Single 0 | Single 23 | Single 47 |
|---|---:|---:|---:|---:|---:|---:|---:|
| stock | 869.49 | 17.600 | 17.598 | 17.602 | 14.434 | 14.428 | 14.404 |
| D | 910.12 | 10.145 | 11.129 | 12.008 | 15.076 | 15.082 | 15.054 |
| C | 896.16 | 18.208 | 18.203 | 18.170 | 14.836 | 14.841 | 14.816 |
| B | — | — | — | — | — | — | — |
| A | — | — | — | — | — | — | — |
| E | — | — | — | — | — | — | — |
| F | 868.70 | 9.267 | 9.802 | 10.625 | 14.341 | 14.343 | 14.321 |
| G | — | — | — | — | — | — | — |

All block columns are milliseconds. Raw replay samples, min/max and exact-replay checks are retained in each manifest. Values from this 4608-token model run do not characterize the very-long-context precision loss observed in the standalone SDPA experiments.

**Thermal qualification caveat:** telemetry during the stock repeatability run showed 87.6–95.4°C and 800–1350 MHz, with a reported 90°C throttle threshold and 1350 MHz maximum. These are observed timings on this allocation, not qualified peak performance. Thermal/clock variation can bias both run-order comparisons and speedups. This observation does not establish the cause of numerical nondeterminism.

## Sustained-workload block check

The first D dual-block measurements above drifted during warmup. This follow-up uses the same D-captured inputs for every choice, 5000 initial dual-block replays, then 750 warmups and 50 timed samples per block. The intended schedule is one forward and one reverse pass. Each cell pools the available completed rounds (50 samples per round). The rounds column makes incomplete coverage explicit. Replay differences are recorded; a timing measurement is not a correctness pass. Prefer these warmed block numbers over the initial image-run block measurements.

| Choice | Dual 0 | Dual 3 | Dual 7 | Single 0 | Single 23 | Single 47 | Rounds | Max round-median difference |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| stock | 18.465 | 18.465 | 18.461 | 15.065 | 15.064 | 14.977 | 1 | incomplete |
| D | 19.323 | 19.330 | 19.331 | 15.738 | 15.753 | 15.662 | 1 | incomplete |
| C | 19.086 | 19.086 | 19.088 | 15.481 | 15.488 | 15.407 | 1 | incomplete |
| B | 18.758 | 18.763 | 18.732 | 15.147 | 15.149 | 15.067 | 1 | incomplete |
| A | 18.718 | 18.717 | 18.719 | 15.099 | 15.095 | 15.014 | 1 | incomplete |
| E | 18.595 | 18.595 | 18.595 | 15.055 | 15.031 | 14.963 | 1 | incomplete |
| F | 18.722 | 18.728 | 18.731 | 15.182 | 15.188 | 15.074 | 2 | 0.54% |
| G | 18.552 | 18.554 | 18.553 | 15.021 | 15.022 | 14.935 | 2 | 0.03% |

All block times are milliseconds; raw samples are in `block-sweep.json`.

**Hardware-interrupted:** 60/96 block measurements were durably recorded in the JSON report. The first pass covers all eight choices. During the reverse pass, device discovery failed with a PCIe 0xffffffff read. Active and queued tests were stopped; no reset was performed. The raw log contains additional completed blocks from the interrupted variant, which are not included in the table because its full six-block round was not saved.

Replay mismatches in the saved diagnostic sweep: 0/60 block measurements.

The isolated block equality checks inspect the first device's outputs; they
do not constitute an all-device full-block accuracy comparison. The separate
attention-only replay probes inspect all eight devices, and pipeline latent
checks reassemble the sequence shards.


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
