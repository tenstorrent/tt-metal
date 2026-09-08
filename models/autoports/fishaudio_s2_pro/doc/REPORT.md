# Fish Audio S2 Pro on Blackhole — bring-up report

generated 2026-09-08T23:49:14Z

## Verified

- profile `p150` (mesh 1x1, device P150): tt-model serve/prove/stop passed
- profile `p300` (mesh 1x2, device P300): tt-model serve/prove/stop passed
- profile `p300x2` (mesh 1x4, device P150x4): tt-model serve/prove/stop passed
- slow tower slow_1x1: short/no_ref: top1 0.838 logitsPCC 0.9987, short/ref: top1 0.897 logitsPCC 0.9985, medium/no_ref: top1 0.838 logitsPCC 0.9985
- slow tower slow_1x2: short/no_ref: top1 0.824 logitsPCC 0.9993, short/ref: top1 0.926 logitsPCC 0.9988, medium/no_ref: top1 0.886 logitsPCC 0.9992
- slow tower slow_1x4: short/no_ref: top1 0.838 logitsPCC 0.9994, short/ref: top1 0.926 logitsPCC 0.9988, medium/no_ref: top1 0.895 logitsPCC 0.9993

## Not verified

- profile `p300x4` (8 chips): declared in the package, not runnable on this 4-chip box
- stage 07-codec-tt: not passed (see STATUS.md)

## Stage table

| stage | body | gate | started | ended |
|---|---|---|---|---|
| 00e-metal-build | ok | 0 | 2026-09-08T19:34:24Z | 2026-09-08T19:47:11Z |
| 00-host-prep | ok | 0 | 2026-09-08T19:30:46Z | 2026-09-08T19:32:16Z |
| 01-cpu-baseline | ok | 1 | 2026-09-08T20:23:21Z | 2026-09-08T23:23:59Z |
| 02-weights-config | ok | 0 | 2026-09-08T23:24:21Z | 2026-09-08T23:28:57Z |
| 03-slow-pcc-1chip | ok | 1 | 2026-09-08T20:45:11Z | 2026-09-08T20:48:31Z |
| 04-slow-multichip | ok | 0 | 2026-09-08T20:55:49Z | 2026-09-08T21:00:46Z |
| 05-phase-a-e2e | ok | 1 | 2026-09-08T21:21:16Z | 2026-09-08T21:21:51Z |
| 06-fast-decoder-tt | ok | 0 | 2026-09-08T21:23:21Z | 2026-09-08T21:24:52Z |
| 08-perf-sweep | ok | 1 | 2026-09-08T23:29:30Z | 2026-09-08T23:29:30Z |
| 09-server-smoke | ok | 0 | 2026-09-08T22:28:49Z | 2026-09-08T22:31:48Z |
| 10-package | ok | 0 | 2026-09-08T23:26:47Z | 2026-09-08T23:30:15Z |
| 11-serve-prove | ok | 1 | 2026-09-08T23:39:29Z | 2026-09-08T23:47:15Z |
| 12-report-push | running | - | 2026-09-08T23:49:14Z |  |

## Accuracy thresholds (from bf16-vs-fp32 CPU floors)

```
{
  "cb0_top1": 0.9,
  "cb0_top5": 0.98,
  "cbn_top1": 0.85,
  "cbn_top5": 0.97
}
```

## Advisory findings and known limitations

- `p150`: the container's first boot failed with `Timed out while waiting for active ethernet core`; one bounded `tt-smi -r` + retry booted it. Both cases followed the close of a 1x2 (P300) mesh: closing a 2-chip mesh on this QB2 leaves an ERISC dirty on device 0, so a board reset is needed before the next multi-chip open.
- `p300x2`: the container's first boot failed with `Timed out while waiting for active ethernet core`; one bounded `tt-smi -r` + retry booted it. Both cases followed the close of a 1x2 (P300) mesh: closing a 2-chip mesh on this QB2 leaves an ERISC dirty on device 0, so a board reset is needed before the next multi-chip open.
- codebook-0 top-1 agreement with the fp32 CPU reference is 0.82-0.93 on TT (bfloat8_b weights) vs a CPU-bf16 floor of 0.948; logits PCC >= 0.998 and top-5 = 1.0, ASR WER 0.0 on every clip. Top-1 is advisory; a Phase D dtype sweep (bf16 attention/first+last layers) is the planned fix.
- not real-time yet: RTF 3.39 on one chip (slow 51 ms + fast 49 ms + CPU codec 27 ms per 46 ms frame; prefill 1.8 s untraced). Roofline estimate for slow+fast is ~17 ms/frame; the gap is host sampling/readbacks, untraced prefill and the CPU codec (Phase C/D items).
- tensor-parallel (p300/p300x2) is functionally verified but not faster than one chip for this batch-1 workload: the fast decoder's all-gathers dominate its 10 tiny steps. TP is the shipped default per the plan; DP/replicated-fast-decoder is a follow-up.
- codec decoder runs on the CPU in every profile (`FISH_S2_CODEC_DEVICE=cpu`); the TT codec (stage 07) was not attempted.
- `p300x4` (8 chips, mesh (1, 8), FABRIC_1D_RING) is declared in the package and tagged, but never executed.

## Container proof per profile (`tt-model serve` -> prove_tts.py -> `tt-model stop`)

| profile | mesh | device_name | impl (slow/fast/codec) | critical checks | advisory checks | ASR WER | TTFA (stream) | shutdown |
|---|---|---|---|---|---|---|---|---|
| p150 | 1x1 | P150 | ttnn/tt_transformers/ttnn/torch-cpu | 27/27 | 6/6 | 0.0 | 3.44 s | clean |
| p300 | 1x2 | P300 | ttnn/tt_transformers/ttnn/torch-cpu | 27/27 | 6/6 | 0.0 | 2.76 s | clean |
| p300x2 | 1x4 | P150x4 | ttnn/tt_transformers/ttnn/torch-cpu | 27/27 | 6/6 | 0.0 | 2.58 s | clean |
| p300x4 | (1, 8) | - | - | not run: 4 chips on this box | | | | |

## Performance (ms per 46.4 ms audio frame; RTF = generation s / audio s)

| run | fast impl | codec impl | slow | fast (10 steps) | codec | prefill s | RTF | frames/s |
|---|---|---|---|---|---|---|---|---|
| phaseA/1x1 | torch-cpu | torch-cpu | 36.5 | 304.1 | 28.2 | 1.7 | 8.3 | 2.9 |
| phaseA/1x4 | torch-cpu | torch-cpu | 33.5 | 309.9 | 29.5 | 1.6 | 8.4 | 2.9 |
| phaseB/1x1 | ttnn | torch-cpu | 51.3 | 49.3 | 26.6 | 1.8 | 3.4 | 9.7 |

## Package

- manifest: `~/tt-model-builds/fish-s2-pro/tt_kernel_manifest.json` (schema 5.1, kind tt-dit-server, tt-metal 0.65.2.dev9783+gde57b4fcbd6, image `tt-model/fish-s2-pro:c0f793844a7a` sha256:c0f793844a7a)
- profiles: p150 (p150, mesh P150), p300 (p300, mesh P300), p300x2 (p300x2, mesh P300x2), p300x4 (p300x4, mesh (1, 8)); default `p150`; weights `fishaudio/s2-pro` @ `1de9996b6be3`
- HF push: push skipped: no token
- code: tt-metal branch `jashan/fish-s2-pro`, `models/autoports/fishaudio_s2_pro` (local commits, not pushed)

## Artifacts

- goldens: artifacts/golden/ · gates: artifacts/gates.json (+ gates_calibration.md) · PCC: artifacts/pcc/ · e2e audio: artifacts/e2e/ · proofs: artifacts/prove/ · package: ~/tt-model-builds/fish-s2-pro
- hardware events: logs/hw-events.log
