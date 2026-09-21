# Bring-up status, 2026-09-15

## Complete handoff, 2026-09-16 15:54 UTC

See [replacement-suite-01/REPORT.md](replacement-suite-01/REPORT.md) and
[WEIGHT_CACHE.md](WEIGHT_CACHE.md). The full eight-choice cohort has 48 images,
48 raw CLIP scores, six visually checked paired PNG sheets, and 28 real-QKV
operator measurements. All 96 shared-input forward/reverse timing measurements
completed, with zero first-device replay mismatches. Max round-median drift is
3.56%; small timing differences should not be overinterpreted. The frozen
recipe-definition check passed (1 test). Python compilation and diff checks passed.

The six measured 50-step stock calls are bitwise identical (15/15 pairwise
comparisons); this does not identify the cause of historical bh-51 failures.
All 24 initial model-component loads hit the converted cache, and image-run
pipeline setup was 9.03–9.22 s. Separate fresh-process qualification measured
248.773 s cold (including cache writes) versus 9.068 s warm, with exact checked
weights and identical two-step PNG/latent hashes.

All model/test processes have completed. Reservation 221619 on bh-lb-08 remains
allocated for continued work (last IRD check: about two hours remaining).
Use container `bh-lb-08-special-cglagovich-for-reservation-221619` and the same
remote checkout path `/localdev/cglagovich/flux2-frontier-20260915/tt-metal`.
The old reservations 126327 and 221618 are released. Numeric kernels and frozen
attention recipes were not changed during recovery/cache work. Generated results
and raw logs are retained locally and on the replacement host.

## Images and repeatability complete, 2026-09-16 15:38 UTC

All 48 images completed in `replacement-suite-01`, with 24/24 initial component
cache hits, zero conversion fallbacks, all 48 first-device block replay checks exact, and
all 16 steady model replay checks exact. Pipeline setup ranges 9.03–9.22 s.
CLIP scoring and all six paired PNG sheets are complete and visually checked.
Mean CLIP ranges 0.34725–0.34869 across choices; this small sample does not
establish a quality ranking. D/C/F reproduce all 18 earlier PNG/latent hashes.

The independent stock 50-step repeatability diagnostic passed: three untraced
and three steady traced calls are all bitwise identical. The historical failure's
cause is still not isolated because hardware/runtime conditions changed together.
All 28 real-QKV measurements on fresh D captures completed with finite outputs. The shared-input,
forward/reverse full-block timing sweep is now active; this is the only GPU job.

## Cache qualified; image suite active, 2026-09-16 15:11 UTC

Full-model cache qualification passed in fresh cold/warm processes. All three
warm components hit cache; component loading fell from 244.805 s (including
initial writes) to 5.215 s. Total pipeline setup fell from 248.773 s to 9.068 s
(27.4×). The original stall-site weight matched on all eight shards, and both
two-step smoke tests passed. See `cache-model-qualification-01/report.json`.

`replacement-suite-01` is running serially in order G/stock/D/C/B/A/E/F:
three prompts × seeds 0/42 × 50 steps × 1024², with six block benchmarks per
choice. Frozen numerical recipes and model-repair/conditioning settings are
unchanged. `FLUX2_REQUIRE_WEIGHT_CACHE=1` rejects unexpected cache misses.
Exploratory replay recording remains explicit; fresh stock repeatability and
CLIP/paired-image analysis follow. No accelerator workload overlaps the suite.

## Full-model cache qualification, 2026-09-16 15:07 UTC

The checkpoint copy completed successfully. bh-51 had no remaining task/test
processes and reservation 126327 was released at 15:06 UTC. The working system
is now exclusively bh-lb-08, reservation 221619. Its latest idle telemetry is
33.4–36.8°C. All 28 repeated real-QKV checks match the old results to reporting
precision (maximum L2 difference 3.6e-15 percentage points).

Three fresh-process synthetic warm-cache loads passed with pinned-memory caching
disabled, in 15.9–16.2 ms each; the final run independently verified every shard
against the original Torch source as well. Full-model cold/warm qualification
is active in `cache-model-qualification-01`. It forbids weight-conversion fallback
on the warm pass and records per-component load times plus total pipeline setup.
See [WEIGHT_CACHE.md](WEIGHT_CACHE.md).

## Migration and cache recovery, 2026-09-16 14:44 UTC

The post-reset bh-51 run was stopped after temperatures rose to 98°C again
(800 MHz on the hottest card). Its partial G images are preserved but are not
counted as a completed variant. No fan or power controls were changed.

The active replacement is bh-lb-08, reservation 221619, with all eight devices
exposed, idle temperatures 33.6–36.6°C, and fans already at 100%. The identical
checkout/prebuilt library was copied and its eight-device matmul smoke passed
(1 passed in 8.71s). Firmware is 19.13.1.0, versus 19.8.1.0 on bh-51; the new
results will form a separate hardware cohort. The briefly allocated bh-lb-01
reservation 221618 exposed no TT devices and was released. The 106 GiB raw
checkpoint is still transferring from bh-51; no accelerator workload is active
on that source machine.

A fresh synthetic converted-weight cache reproduced the cached-load stall on
bh-lb-08. Cold conversion/save succeeded; ordinary warm loading stalled and
was terminated. Repeating the warm load with
`TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0` passed, requiring no conversion and
matching SHA256 hashes of all eight device shards of replicated, row-sharded,
and column-sharded BF16 weights. This is an experimentally isolated workaround
for the pinned transfer path, not a claim that the underlying runtime bug has
been repaired. Full FLUX.2 cold/warm cache validation is pending the checkpoint
transfer. The fresh converted-cache root is `dit-cache-v2`; the old cache is
preserved and was not migrated.

## Recovery and continuation, 2026-09-16 13:34 UTC

The user authorized resetting the reserved system or moving to another one.
All eight bh-51 PCI BDFs were explicitly reset with `tt-smi`; the reset exited
successfully. Discovery again sees eight devices at 47.6–58.2°C. The replicated
matmul smoke passed on all eight devices (1 passed in 4.03s). Evidence:
`reset-20260916-01.log`, `telemetry-reset-20260916-01.json`, and
`reset-environment-smoke-01.log`.

Reservation 126327 was extended for four hours. Missing image runs are active
serially in `exploratory-reset-01`, in order G/A/B/E. They keep the frozen
recipes, checkpoint, copied prompt embeddings, prompts, seeds and generation
settings. `FLUX2_EXPLORATORY=1` records replay discrepancies without treating
them as qualification passes; finite-value and no-fallback checks remain.
The original failed/partial results are preserved. A fresh stock repeatability
control is planned after those images. No other accelerator workload overlaps.

## Earlier: partial results saved; hardware stopped, 2026-09-16 05:30 UTC

See [suite-02/REPORT.md](suite-02/REPORT.md) and
[REPEATABILITY.md](REPEATABILITY.md). Completed: 24 images (D/C/F/stock,
six each), CLIP, six paired sheets, 28 identical real-QKV accuracy checks,
and one warmed timing pass over all eight choices. Sixty of 96 planned
forward/reverse block measurements are saved; every saved replay check passed.

A/B/E/G initially failed strict replay checks before generating images.
Subsequent controls show stock also varies across identical prompt/seed
50-step runs: 37.7–43.1% final-latent L2 between untraced calls. The cause is
unresolved, so paired image differences are not isolated attention error.
Separate exploratory image reruns were prepared but did not start.

The timing sweep stopped progressing at 05:16:40 UTC. Device discovery then
reported `Read 0xffffffff over PCIe ID 1: the board should be reset.` Earlier
telemetry reached 98.6°C, with clocks down to 800 MHz. The active hardware
test and queued follow-ups were terminated. No reset or fan/clock/power
change was made. Hardware recovery/cooling direction was requested from the user.

CPU decoding/scoring of all six saved repeatability latents also completed.
The common CPU FP32 VAE produces coherent but visibly varying cat images;
CLIP ranges from 0.35702 to 0.35979. Calls 4/5 have identical PNG hashes.
See `pipeline-replay-stock50-01/decoded-cpu/comparison.png`.
No test or evaluation process remains active. Reservation 126327 is still allocated;
`ird list` reported about nine hours remaining at 05:19 UTC. Its control-host
clock is four hours behind the container clock; use TIME LEFT, not guessed expiry.

The following sections retain the chronological investigation history.

## Larger evaluation, 2026-09-16 UTC

Suite 02 is running on the corrected common model baseline:
`FLUX2_MODEL_REPAIR=main_fused`, `FLUX2_CONDITIONING=stock`. D, stock, C, B,
A, E, F, G each receive three prompts × seeds 0/42, 50 steps at 1024² and
guidance 4. All six representative blocks are benchmarked, and exact trace
replay is mandatory. The reservation had five hours remaining at launch.

The D integration passed the existing two-block reference thresholds after
adapting prompt placement to the pipeline's SP-sharded layout: PCC 0.999973,
normalized RMSE 3.4%, 72.90 seconds. Evidence: `frontier-block-D-01.log`.
The global Torch reference and original thresholds were not changed.
Progress: D, stock, C and F completed all six images and exact model-trace checks.
Their mean CLIP cosines are 0.34763, 0.34683, 0.34792 and 0.34782, respectively.
B stopped before image generation: its first steady full-model trace replay
differed from the untraced two-step reference. All six isolated block traces
passed exact replay. The independent full-shape attention probe also passed
112 exact comparisons across seven recipes and two real-input captures, with
unrelated SFPU operations interleaved. This is an unresolved integration/trace
qualification issue, not a valid B quality result. A subsequently failed isolated
single.0 block replay; E failed dual.0 block replay. G is running separately.

Three attention-only probes passed 112 comparisons each, including full per-device
shape, all eight devices, relocated buffers and alternating positive/negative
inputs. A fourth probe stopped on a harness API-name error before measurements;
that error is corrected for the next run. No numerical kernel has changed.

`pipeline-replay-B-01` reproduces variation without block benchmarking: repeated
untraced two-step runs differ by 2.94–3.48% latent L2; traced results differ by
4.19–4.47% from the first untraced run. With cloned block boundaries retained,
`pipeline-replay-B-02` gives three identical untraced runs and two matching trace
replays, then a 2.84% divergent third replay. This suggests a timing/lifetime
dependency but does not identify its cause. The earlier untraced 50-step stock
control and suite-02's traced stock result also differ by 28.12% latent L2 despite
matching embedding hashes and generation settings. A within-process, 50-step
stock repetition diagnostic is planned before causal quality claims.

The same IRD reservation 126327 was extended from eight to ten hours after an
extended connectivity interruption; no additional machine was reserved.

Initial D dual-block timings drifted during warmup. A separate shared-input,
order-reversed timing sweep is prepared; do not interpret those initial D
dual-block values as steady-state speedups.

## Latest diagnosis

The primary correctness failure is identified: on the pinned source's Linear
topology, the attention output `ColParallelLinear` ignores the supplied gated
residual and returns only the matrix product. Restoring that math alone changes
the existing two-block test from PCC 0.003201 / L2 100.99% to PCC 0.999892 /
L2 3.41% (pass). Missing per-head Q/K normalization is a secondary issue.
Both fixes are already in main commit `2a0f4c55e67` (#55225), after our fork.
The fused main-style repair also passes (PCC 0.999971 / L2 3.38%).

See [DIAGNOSIS.md](DIAGNOSIS.md) for all ablations. Experiment-only fixes are
isolated in `model_fixes.py`; no production fix or branch rebase is applied.
The 50-step `stock-main-fixes-01` image check **passed in 428.61 seconds** with
stock attention and stock conditioning. Visual inspection confirms a coherent
cat on a windowsill at sunset matching the prompt, replacing the noise output.
Diagnosis is complete; the broader frontier image/CLIP/performance suite still
needs to be rerun with the corrected common model baseline.
The older sections below retain the investigation history, not current status.

## Hardware and environment

- IRD reservation 126327, `bh-51`, requested for eight hours.
- Container: `bh-51-special-cglagovich-for-reservation-126327`.
- Eight Blackhole PCIe devices visible through `tt-smi -ls`.
- Host: 503 GiB RAM, approximately 443 GiB available at inspection; 1.3 TiB
  free on `/localdev`.
- Prebuilt `/opt/venv`: torch 2.11.0+cpu, transformers 5.12.1, diffusers 0.38.0.
  Importing `Flux2Transformer2DModel` succeeds.
- Existing `/localdev/cglagovich/tt-metal` has unrelated edits and an older
  build (March 2026); it is untouched.
- Saved research source/build from our previous `yyzo-bh-08` allocation was
  copied into a separate path. This did not reserve or run silicon on that
  old host. No files were deleted.
- A new remote worktree at `/localdev/cglagovich/flux2-frontier-20260915/tt-metal`
  is pinned to `2daa9201e244957b95774d2a0bbe8d1db7c659e4`, with the new smoke
  harness added separately. It reuses copied build/runtime artifacts through
  symlinks; this is NOT a fresh rebuild. `_ttnn.so` SHA256:
  `2c426dd163b6952d220d3e9632147a3974ac9b4252a48362fc3aeecf04cab8ac`.
- The new container lacked `graphviz`. The existing graphviz 0.21 package from
  the saved research environment is exposed via PYTHONPATH; no packages were
  installed. TTNN and the TT FLUX.2 pipeline then imported successfully.
- Weight-free replicated BF16 matmul passed on all eight devices, using the
  actual 2x4 mesh, FABRIC_1D and 64 KiB L1_SMALL: **1 passed in 15.04s**.
  Firmware 19.8.1; KMD 2.9.0. UMD emitted harvesting-count and motherboard
  identification warnings; these did not prevent topology discovery or this
  smoke test. This test does not qualify model execution or attention.

## Checkpoint access resolved

The user's interactive shell sets `HF_HOME=/proj_sw/user_dev/cglagovich/.cache/`.
Noninteractive docker commands did not inherit that setting and initially
reported no credential. With that explicit setting, `HfApi().whoami()` succeeds
and the FLUX.2-dev transformer config downloads successfully. No credential
contents are recorded. Checkpoint revision is pinned to
`26afe3a78bb242c0a8bb181dcc8937bb16e5c66c` for all comparisons.

## Evaluation progress

### Current blocker: model output is noise, not a valid quality comparison

Update: `D-conditioned-01` completed a fresh untraced 50-step run in 422.81s,
but visual inspection still shows noise. The conditioning repair is not
sufficient to fix generation. The isolated repair-wrapper test passed and
reduced conditioning L2 from 6.5–12.7% to 1.0–1.2%; these are component results,
not evidence of a working model. A matching stock-attention control
(`stock-conditioned-01`) is running.

The existing reference transformer PCC test had not yet been run. Its
`single_blocks` case retains one dual-stream and one single-stream block,
including input/output projections and conditioning. The unchanged 2x4 case
collects successfully on this Blackhole allocation; its acceptance thresholds
are PCC >= 0.996 and RMSE/reference-standard-deviation <= 0.09. It is the next
hardware test after the stock control, before any further image sweep.

The stock-attention 50-step control also completed (424.05s), but visual
inspection shows noise. This failure is therefore not unique to the custom
attention adapter. The original block test invocation stopped before inference
because Diffusers requested HF metadata in offline mode. Retry 02 invokes the
same existing test body through `test_existing_blocks.py`, resolving the pinned
checkpoint directly and limiting host PyTorch threads to 16. No numerical
settings or thresholds are changed; this retry has loaded the checkpoint and
is executing weight conversion.

Retry 02 completed with a **numerical failure** in 116.49s: PCC = 0.3201%
(coefficient 0.003201), CCC = 0.0905%, RMSE/reference-standard-deviation = 101.0%.
The expected PCC is >= 99.6% and normalized RMSE <= 9%. This uses stock
attention and original conditioning, not any frontier adapter. The failure is
already present in the two-block reference comparison; next isolate the first
divergent intermediate (input projections, modulation, blocks, output head).

Suite 01 D completed all six 50-step images and passed two exact steady-state
traced-vs-untraced two-step comparisons. However, the inspected 50-step image
is noise. The reference Diffusers FP32 CPU VAE decoding the same final latents
also produces the same noise (`reference-decode-01/`), pointing upstream of the
decoder. Do not use suite 01 images as valid model-quality frontier results.

All seven recipes were also evaluated on identical real D captures at four
blocks (28 measurements, `suite-01/real-attention.json` remotely; test passed
in 18.84 seconds). D remains at 0.1647–0.1690% L2 against FP64. This supports
the local SDPA arithmetic, not whole-model correctness.

Found a concrete conditioning mismatch: installed Diffusers FLUX.2 multiplies
guidance by 1000 before `time_guidance_embed`, while the TT pipeline passes
4 and the TT embedding does not rescale it. The TT sinusoidal frequency factors
are also BF16. `test_conditioning.py` is now isolating original guidance,
guidance×1000, and guidance×1000 with FP32 phases against the checkpoint's
reference embedding, before another full-model run. No conditioning fix has
yet been applied to production model files. No full-quality CLIP scores exist.

### Earlier bring-up evidence

- Isolated stock smoke harness created with fixed prompts/seeds, separate fresh
  output directories, manifest recording and explicit rejection of unimplemented
  variants.
- Local Python syntax checks pass. Stock pipeline pytest collection succeeds.
- Downloaded the pinned Diffusers checkpoint layout (33 files, approximately
  113 GB). CLIP ViT-B/32 OpenAI weights are also available locally.
- The first stock smoke reached VAE decoding during constructor warmup and
  failed: D512 SDPA Q128/K128 circular buffers required 1,688,576 bytes, above
  Blackhole's 1,572,864-byte L1 capacity. The harness now fixes VAE Q64/K64
  identically for every variant; denoiser frontier chunk sizes are unchanged.
- Device adapters reuse the frozen research C++ kernels unchanged, with
  independent rectangular Q/KV lengths. Fifteen tests passed in 53.32 seconds:
  exact recipe audit plus seven rectangular and seven SP2/TP4 joint-attention
  tests, including original-BF16 references, preprocessing oracles, all-device
  replicated output agreement, compressed KV CCL and exact trace replay.
  Evidence: `device-qualification-02.log`.
- Joint mesh L2 percentages: A 2.4866, B 2.4757, C 0.3781, D 0.1784,
  E 3.0269, F 2.8666, G 16.6457. These are synthetic adapter checks, not
  model quality results or broad numerical acceptance claims.
- Stock two-step smoke 03 passed in 455.77 seconds (446.30 seconds of pipeline
  construction/conversion/warmup). It generated one 1024x1024 PNG and finite
  final latents. The image remains noisy at two steps; this is an execution
  smoke, **not** a model-quality pass. Evidence: `stock-smoke-03/manifest.json`
  and `stock-smoke-03.log`.
- D pilot 01 executed real FLUX.2 successfully and captured Q/K/V at dual.0,
  dual.7, single.23, single.47. Six isolated block traces matched their untraced
  outputs bitwise. The model trace check failed on its first capture call:
  the existing `_step` has `prep_run=True`, `clone_prep_inputs=False` and an
  in-place latent update. That call applies an extra denoising update and must
  be discarded, as it is in the standard pipeline trace warmup. The harness now
  discards capture and requires two fresh two-step replays to match untraced
  latents exactly. Suite 01 D is running this gate before its six 50-step images.
- Preliminary block timings drifted during short warmup. The full suite now
  uses 250 warmup replays and 50 timed samples per block rather than 5 and 20.
  No preliminary block speedup claim is made.
- CPU inspection of the four selected heads in pilot 01 found substantial real
  common modes. Define common-mode energy fraction as
  `mean(mean(x, tokens)^2) / mean(x^2)`. For V it was 0.309, 0.823, 0.703,
  0.718 at dual.0, dual.7, single.23, single.47, respectively. Early V RMS was
  5.05 with max absolute 45. These are selected first-step captures, not a
  whole-model distribution survey. No centering or new conditioning is applied.
- Full 50-step images, CLIP scores and transformer block timings are pending.
  Retry 02 stalled for several minutes in `ttnn.load_tensor` on
  `time_guidance_embed.timestep_embedder.linear_2.weight.tensorbin`. SIGTERM did
  not end it promptly; the task-owned process was terminated with SIGKILL.
  An attempted debugger attach did not produce a stack; its task-owned process
  was also terminated. No cache files were deleted. Retry 03 bypasses converted
  caches (`env -u TT_DIT_CACHE_DIR`) and is progressing through the same original
  checkpoint-to-device conversion that worked in the first attempt. The cache
  stall's root cause is not established.

## Remote invocation

SSH with agent forwarding to `yyz-ird`, then to host `bh-51`; execute in
`bh-51-special-cglagovich-for-reservation-126327` as `cglagovich`. The new
container's SSH port is 42837. Existing known-host entries for older containers
may have a different key: do not disable host-key verification indiscriminately.

In the container, from `/localdev/cglagovich/flux2-frontier-20260915/tt-metal`:

```sh
export TT_METAL_HOME=/localdev/cglagovich/flux2-frontier-20260915/tt-metal
export PYTHONPATH=$TT_METAL_HOME:$TT_METAL_HOME/ttnn:$TT_METAL_HOME/tools:/localdev/cglagovich/tt-metal-blackhole-20260908/python_env/lib/python3.10/site-packages
export LD_LIBRARY_PATH=/localdev/cglagovich/tt-metal-blackhole-20260908/build/lib
export TT_METAL_CACHE=/localdev/cglagovich/flux2-frontier-20260915/jit-cache
export TT_DIT_CACHE_DIR=/localdev/cglagovich/flux2-frontier-20260915/dit-cache
export HF_HOME=/proj_sw/user_dev/cglagovich/.cache/
/opt/venv/bin/python -m pytest -s -q experiments/sdpa-l2/flux2-frontier-v1/test_environment.py
```

Run `download_checkpoint.py` to fetch the pinned Diffusers layout before the
README's stock invocation. Original-layout duplicate weights are excluded.
