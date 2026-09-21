# Full-model repeatability limitation

## Current cohort: passing, 2026-09-16 15:38 UTC

On bh-lb-08, reservation 221619, with fresh converted-weight caching and
`TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0`, the repeated 50-step stock control
**passes**: all three untraced calls and all three steady traced calls produce
bitwise-identical final latents. The initial capture call is excluded as before.
Evidence: `replacement-suite-01/stock-repeatability/report.json` and its log.

All eight image variants also completed all six images, with exact equality
for all 48 recorded first-device block checks and all 16 steady two-step model replay checks.
D/C/F reproduce all 18 earlier complete image and latent-file hashes exactly;
the six stock outputs differ from the historical cohort.

This establishes repeatability for these measured calls/configuration, not
general determinism. The earlier failure's root cause is **not established**:
the machine, firmware/kernel, converted-load path and host-pinning setting
changed together. No controlled full-model pinning-only A/B was performed.
Separately, a synthetic cached-load stall was reproduced on the healthy machine
and bypassed specifically by disabling pinned-memory caching; see WEIGHT_CACHE.md.
Do not conflate that isolated loader result with proof of the repeatability cause.

The sections below preserve the earlier bh-51 failures and diagnosis limits.

This is separate from the missing residual/per-head normalization defects
already isolated in [DIAGNOSIS.md](DIAGNOSIS.md). Those common model repairs
are enabled throughout this investigation. Numerical attention recipes have
not been changed.

## Observations

- Initial strict image-suite checks passed for D, C, F and stock. B failed
  full-model replay equality; A, E and G failed an isolated block's replay
  equality. All original failed manifests are preserved in `suite-02`.
- B repeats without isolated block benchmarking still vary: untraced two-step
  final latents differ by 2.94–3.48% L2 from the first run. Traced calls differ
  by 4.19–4.47%. This is not solely a trace-capture problem.
- Retaining cloned first-step block boundaries changes that behavior: all
  three untraced runs and the first two steady traces become identical, but
  the third steady trace differs by 2.84%. This changes both allocation and
  timing, so it does not identify a specific lifetime defect or race.
- Crucially, stock attention also varies in a single-process 50-step control.
  Repeated untraced calls differ by **43.10% and 37.67%** final-latent L2 from
  the first; steady traced calls differ by **47.21%, 47.21%, and 57.54%**.
  The first two steady traces match each other; the third does not.
- The prompt embedding is loaded from the same cache and the pipeline resets
  Torch's noise seed to zero on every generation. Checkpoint, guidance,
  conditioning and model corrections are unchanged.
- The first trace-capture generation is discarded: the current pipeline
  tracer performs an extra preparation step on in-place scheduler state.
  It is not counted among the repeated generations above.

Evidence: `pipeline-replay-B-01/report.json`,
`pipeline-replay-B-02/report.json`, and
`pipeline-replay-stock50-01/report.json`, with adjacent execution logs.
Diagnostic reports use `status=completed` to mean all observations were
collected; the tests deliberately **fail** their final equality assertion.

## What the attention-only checks establish

Four progressively stronger standalone attention probes passed 112 exact
comparisons each. They cover all seven recipes and real inputs from two
blocks at the actual per-device H12/Q2304/K4608/D128 shape. Later probes
compare all eight chips, relocate input buffers, alternate input signs, and
precede attention with a full FP32-accumulating HiFi4 matrix multiplication
and unrelated SFPU work. See `replay-probe-01/02/03/05.json`.

Probe 04 stopped at an invalid harness API name before measuring anything;
probe 05 corrects that name. Probe 04 is not an operator failure or a pass.

These results support local SDPA repeatability under those conditions. They
do not prove that every kernel transition in the full transformer is safe,
or rule out communication, buffer reuse, or reduction-order effects.

## Interpretation and evaluation policy

The exact source of the full-model variability is **not established**.
Bitwise inequality alone does not establish incorrect mathematical output;
however, this variability is large enough to confound a paired image-fidelity
comparison. It cannot fairly be charged only to A/B/E/G when stock also fails.

The original strict records remain unchanged. Separate exploratory runs use
`FLUX2_EXPLORATORY=1` to record equality and L2 differences without aborting
image generation. Finite-value, shape, selected-recipe and no-fallback checks
remain mandatory. This flag changes test acceptance behavior, not the
attention algorithm, preprocessing, data format, or communication schedule.
Exploratory completion is **not** a repeatability qualification pass.

The six saved stock-repeat latents were subsequently decoded using a common
CPU FP32 reference VAE, without touching the failed accelerator. See
[the stock-repeat sheet](pipeline-replay-stock50-01/decoded-cpu/comparison.png).
All six are coherent cat images, with visible pose/detail changes; raw CLIP
cosine ranges from 0.35702 to 0.35979. Calls 4 and 5, whose latents match,
also produce identical PNG hashes. This CPU-only control is separate from
the 24 TT-decoded frontier images and does not replace any missing variant.

Publish all images and CLIP results, but do not interpret cross-variant
final-latent L2 as isolated SDPA error, or tiny CLIP differences as a ranking.
The identical-QKV FP64-reference checks remain a separate, direct operator
accuracy measurement.

## Performance is also hardware-limited

The stock-run snapshot reports 87.6–95.4°C, a 90°C throttle threshold, and
800–1350 MHz clocks. During the sustained block sweep, one chip reached
98.6°C at 800 MHz. Raw snapshots are retained. Warmed block measurements
therefore describe this allocation; they do not qualify peak performance.
No fan, clock, power, firmware or thermal-protection settings were changed.
Thermal observations do not establish the cause of numerical variability.

At approximately 05:16 UTC on September 16, the sustained block sweep stopped
making progress. A subsequent `tt-smi` query failed with
`Read 0xffffffff over PCIe ID 1: the board should be reset.` The process log's
last modification is 05:16:40 UTC; the failed device-discovery output is saved
as `telemetry-block-sweep-02.json` (despite its suffix, it contains an error,
**not valid JSON telemetry**). Sixty of 96 planned block measurements were
durably saved, including the complete forward pass over all eight choices.
Four additional E reverse-pass block results are present only in the raw log.

The active benchmark and queued image/decode launcher were terminated before
the queued jobs could start. No device reset has been performed. The original
24 D/C/F/stock images and all 28 real-QKV checks are preserved. Exploratory
A/B/E/G image runs remain unexecuted, pending hardware recovery. Thermal
stress preceded the PCIe failure, but causation is not established.

Before a PR-quality causal model comparison, reproduce stock repeatability
on a thermally healthy allocation and current main, then isolate the first
divergent full-block component on fixed inputs. Reference conditioning also
needs separate qualification; this suite deliberately holds the stock
conditioning behavior common to every attention choice.
