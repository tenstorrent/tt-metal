# DiffusionGemma live traced denoise-step speed sweep — QB2

> **HISTORICAL PERFORMANCE PROVENANCE (2026-07-10).** This is an explicitly configured,
> prompt-only-prefix, capture-once argmax trace sweep. It is not the plain vLLM default, not
> first-request TTFT, and not current growing-prefix multi-block throughput. Use `plan.md` Part 0
> for the current launch and metric contract.

Date: 2026-07-10
Compact evidence: `live_denoise_step_sweep_results_20260710.json`
Validation record: `live_denoise_step_sweep_validation_20260710.md`

## Result

This is the real patched `tenstorrent/vllm` OpenAI `/v1/completions` path on
bhqb QB2. Each denoise budget used a fresh server lifecycle with an exact
256-token logical prompt, `max_model_len=4096`, four 256-token output blocks,
one same-shape warmup request, and one compile-marker-free timed request.

| max steps K | steady block mean (s) | output tok/s | speedup vs K=48 |
|---:|---:|---:|---:|
| 1 | 1.534773 | **166.800** | **9.127x** |
| 4 | 2.364211 | **108.281** | **5.925x** |
| 8 | 3.509947 | **72.936** | **3.991x** |
| 12 | 4.664967 | **54.877** | **3.003x** |
| 16 | 5.758205 | **44.458** | **2.433x** |
| 20 | 6.907149 | **37.063** | **2.028x** |
| 24 | 8.000544 | **31.998** | **1.751x** |
| 32 | 10.024370 | **25.538** | **1.397x** |
| 40 | 11.997746 | **21.337** | **1.167x** |
| 48 | 14.007196 | **18.276** | 1.000x |

This is a performance-only step-cap experiment. The current model-faithful
setting is 48 steps under issue #48291. Smaller caps can change diffusion
decisions and output quality; the speed rows are not quality results.

## Detailed timing

Block 0 includes Metal trace capture. Blocks 1–3 are the three steady replay
samples used for the mean, median, population standard deviation, and output
rate (`256 / steady_mean`).

| K | model build (s) | prefill (s) | block-0 TTFT (s) | block latencies b0/b1/b2/b3 (s) | steady median (s) | steady σ (s) | denoise ms/step | request wall (s) |
|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 1 | 15.820 | 2.961 | 9.217 | 4.374 / 1.529 / 1.543 / 1.533 | 1.533 | 0.006 | 253.588 | 13.834 |
| 4 | 14.873 | 2.920 | 16.233 | 11.451 / 2.368 / 2.352 / 2.372 | 2.368 | 0.009 | 251.766 | 23.339 |
| 8 | 15.076 | 2.973 | 26.093 | 21.312 / 3.527 / 3.494 / 3.509 | 3.509 | 0.013 | 251.597 | 36.640 |
| 12 | 15.299 | 2.914 | 36.427 | 31.683 / 4.658 / 4.699 / 4.638 | 4.658 | 0.026 | 251.441 | 50.438 |
| 16 | 15.294 | 2.936 | 46.554 | 41.815 / 5.740 / 5.748 / 5.787 | 5.748 | 0.021 | 251.395 | 63.848 |
| 20 | 15.277 | 2.989 | 58.134 | 53.269 / 6.896 / 6.934 / 6.892 | 6.896 | 0.019 | 251.312 | 78.871 |
| 24 | 15.308 | 2.975 | 68.309 | 63.472 / 7.954 / 8.022 / 8.026 | 8.022 | 0.033 | 251.327 | 92.326 |
| 32 | 15.271 | 2.923 | 89.817 | 85.050 / 10.103 / 9.965 / 10.006 | 10.006 | 0.058 | 251.337 | 119.908 |
| 40 | 15.389 | 2.922 | 111.424 | 106.717 / 11.973 / 12.029 / 11.991 | 11.991 | 0.023 | 251.359 | 147.432 |
| 48 | 15.212 | 2.950 | 134.949 | 130.128 / 13.998 / 13.976 / 14.047 | 13.998 | 0.030 | 251.307 | 176.985 |

The warmed denoise cost is effectively flat at about 251.3–251.8 ms per step
for K=4–48. The K=1 estimate is 253.6 ms and is more sensitive to fixed
synchronization overhead. Block latency therefore scales predictably with K,
while commit and other fixed block work limit the end-to-end speedup below the
ideal 48/K ratio.

## Trace proof

Every reported row passed all of these assertions:

- block 0 emitted one capture event with exactly K distinct single-step Metal
  trace IDs;
- all four blocks executed the same trace-ID set, with three steady replays;
- the total execute count was exactly `4*K` and the steady execute count was
  exactly `3*K`;
- there was no eager fallback and no recapture after block 0;
- request completion emitted both trace release and model request-release
  markers;
- the timed request contained zero `Building trisc` compile markers.

| K | traces captured once | block replays | steady replays | total execute calls | trace contract |
|---:|---:|---:|---:|---:|---|
| 1 | 1 | 4 | 3 | 4 | pass |
| 4 | 4 | 4 | 3 | 16 | pass |
| 8 | 8 | 4 | 3 | 32 | pass |
| 12 | 12 | 4 | 3 | 48 | pass |
| 16 | 16 | 4 | 3 | 64 | pass |
| 20 | 20 | 4 | 3 | 80 | pass |
| 24 | 24 | 4 | 3 | 96 | pass |
| 32 | 32 | 4 | 3 | 128 | pass |
| 40 | 40 | 4 | 3 | 160 | pass |
| 48 | 48 | 4 | 3 | 192 | pass |

The configured 10 GiB trace region is a capacity limit. Per-chip used DRAM was
13.461654 GiB after model build, 14.891826–14.893260 GiB after block-0 capture,
and 13.481212 GiB after request release. The trace-resident increase was about
1.430–1.432 GiB/chip across the sweep.

## Full-depth baseline comparison

The inherited 256-context result was 18.246764 output tok/s
(`14.029885 s/block`, displayed as 18.247 tok/s). This isolated K=48 run
measured 18.276320 output tok/s (`14.007196 s/block`), a +0.029556 tok/s or
+0.162% difference. That is immaterial run-to-run variation. The new row used
an isolated server, an explicit same-shape warmup, and a timed-request guard
that rejected first-use compile markers.

## Excluded run

The first K=4 server exited during mesh startup before model build or any
request because device 0's active Ethernet core 29-25 did not resume. Its exact
log and failed JSON are preserved and classified as
`excluded_hardware_fault`. A bounded list/reset/list followed by a passing 1x4
mesh smoke preceded the single retry; the retry passed and is the only K=4 row
in the table. No dense-control run was attempted or included.

The preserved recovery record is:

| action | exact bounded command / result | exit |
|---|---|---:|
| no-process check | `ps ... \| rg '(vllm.entrypoints\|api_server\|EngineCore\|live_context_sweep.py\|...)'`; no matches | 0 |
| list before | `timeout 60 tt-smi -ls --local`; UMD chips 0/1/2/3, Blackhole p300c | 0 |
| reset | `timeout 180 tt-smi -r`; `Resetting all PCI devices: [0, 1, 2, 3]`, reinitialized | 0 |
| list after | `timeout 60 tt-smi -ls --local`; UMD chips 0/1/2/3 present | 0 |
| mesh smoke | `timeout 120` open/close `MeshShape(1, 4)`, trace region 0; `MESH_SMOKE_OK`, firmware 19.8.0 | 0 |
| single retry | `live_denoise_step_k04_20260710_retry.json`; status `passed` | 0 |

The compact JSON points to every exact server log and records its SHA-256.
