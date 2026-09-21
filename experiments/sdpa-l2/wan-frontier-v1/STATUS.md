# Wan frontier run status

Exact-shape SDPA tuning is now measured, including independent grid/chunk/
reader controls and uninstrumented sustained replays. See
[SDPA tuning results](SDPA_TUNING.md). The default model dispatch is unchanged.

The full comparison completed successfully on 2026-09-16. All 14 videos,
112 PNGs, CLIP scores, 28 block replay checks and 24 real-QKV comparisons
are verified. See [results](suite-01/REPORT.md) and [findings](FINDINGS.md).
The suite ran on bh-lb-08, IRD 221619, in
`/localdev/cglagovich/flux2-frontier-20260915/wan-frontier-01/suite-01`.
Log: sibling `suite-01.log`. Order: D/C/B/E/F/G/stock. Two prompts per choice,
butterfly and human café scene, seed 42, 40 steps, 81 frames, 480p.

## Preflight

- Native partial-tile mask adapter for SP4/TP2, with compressed KV all-gather.
- Poison-tail qualification: six passed, including exact trace replay.
- Qualification L2/PCC: D 0.178%/0.999998; C 0.375%/0.999993;
  B 2.449%/0.999701; E 3.001%/0.999552; F 2.851%/0.999599;
  G 16.653%/0.986241. These are the artificial poison-tail inputs, not Wan
  model accuracy bands. Their purpose is detecting an ineffective mask.
- F initially failed with L2 1058.6%. The cause was retaining BFP8 K unpack
  format when reading the BF16 mask palette. Explicit SrcA reconfiguration
  fixed it. The initial hypothesis about exp(-inf) was not the cause; no exp
  approximation or clamping change remains. The unpadded FLUX path is unchanged.
- Device kernels were JIT-compiled and executed during qualification.
- Python syntax checks and Black formatting passed for the initial harness.

## Execution results

Suite started around 17:25 UTC on 2026-09-16. Initial model loads all hit
converted caches. The harness forbids any conversion fallback.
Pytest passed in 2907.73 seconds (48m27s), finishing at 18:13 UTC. Sum of
timed video calls was 45.2 minutes; the remainder includes pilots, initial
cached setup, encoding and teardown. Scoring and captured-input tests then
completed successfully; the latter passed in 8.48 seconds. No reconversion
occurred. All 28 sampled full-block replays were exact.

The user was notified that the 40-minute stock-equivalent baseline was
optimistic: D takes about 6.4 seconds/step vs stock's 4.0. Current estimate is
50–60 minutes from video-suite launch for generation/evaluation, additional
to the adapter preflight already completed. Actual generation/evaluation and
artifact review fit that revised estimate.

## Completed evaluation and reproduction

The completed captured-input check ran `test_captured.py` with
`WAN_CAPTURE_MANIFEST=<suite>/D/manifest.json` and
`WAN_CAPTURE_REPORT=<suite>/real-qkv.json`. This evaluates all six recipes on
identical captured D inputs (four blocks, four heads, 256 query rows, full KV)
against FP64 attention.

Scoring used `score_videos.py <suite> --cache-dir
/localdev/cglagovich/flux2-frontier-20260915/clip-cache` and
`summarize.py <suite>`. Videos, sampled PNGs, reports and logs are copied
locally, and both comparison sheets have been visually inspected. Raw QKV
capture tensors remain on the allocated machine; their hashes and measured
results are local. `verify_artifacts.py` checked every video/PNG hash, PNG
size, cache-load record, exact block replay and all 80 transport formats per
experimental choice. Its report is `suite-01/artifact-validation.json`.

`run-metadata.json` records source hashes. Per-variant manifests record
actual Q/K/V formats for all 80 blocks, weight-cache loads, four full-block
trace measurements, video/frame hashes and generation timings. Block inputs
are each variant's own pilot inputs, not an identical-input cross-variant sweep;
short block warmup/dynamic clocks limit fine-grained performance comparisons.

Runtime environment is the same validated FLUX/Wan environment, with the
absolute `wan-frontier-v1` directory prepended to PYTHONPATH (pytest uses
importlib mode). `WAN_CHECKPOINT` is the pinned local checkpoint under
`wan-frontier-01/checkpoint/Wan2.2-T2V-A14B-Diffusers`.
