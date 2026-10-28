# EmbeddingGemma 2: final benchmark (E8), one Blackhole P150a

This report covers traced latency and throughput for every tower and for the full multimodal model. All runs use the selected precision config `doc/datatype_sweep/selected_precision_config.json` (sha256 `704e3f15...2b44`): text `p3_qkv_gu`, vision `v_exp_bfp8w4`, audio `a_h2b8_qkvbf16_shift2_sharded`.

- **Full sweep:** run `20261028-063930`, `pytest tests/perf/perf.py -s` on code commit `1bcb04d2f0`. Result: 46 passed, 19 skipped (16 x 2cq not supported, 2 profiler-only Tracy cases, 1 batch-probe OOM), exit 0, wall time 318 s.
- **Final code commit `4c2e82e044`:** perf.py now makes every extra opt-in. With `EMBEDDING_GEMMA2_PERF_ALL=1` it collects exactly the 65 test ids that the full sweep executed.
- **Labels:** every number below is **[M]** from `run/perf.jsonl` unless labelled otherwise.
- **No accuracy benchmark:** there is no MTEB or lm-eval run, by decision of the person running the bring-up. The HF checks below only guard the timed path.

## Addendum 2026-10-28 (E9.1-r): text default changed to `p2_sdpa_fast`

The text precision default in `doc/datatype_sweep/selected_precision_config.json` changed from
`p3_qkv_gu` to `p2_sdpa_fast` (re-selected on the 12 x 10 Galaxy grid because `p3_qkv_gu` fails
the E9 b1_s256 contract input; `doc/datatype_sweep/README.md`, "E9.1-r re-selection"). Vision and
audio sections are unchanged. Every text and full-model number below was measured with
`p3_qkv_gu` and is superseded for the current default: the text tower table, the text rows of
the headline, the full-model rows (their text stack runs the same policy), and the text rows of
"Burst vs sustained" and "Comparison with earlier stages". They stay as the record of E8.

Re-measured headline, 13 x 10 P150 path, `pytest tests/perf/perf.py -s` (quick default), commit
`bc9cb3106c` + the new config, order new / control / new **[M]**
(`tt-ops/runs/embeddinggemma-2/e9/01r-geometry-precision/run_c5_perf_13x10_*`):

| shape, mode | `p2_sdpa_fast` (new default) mean ms | `p3_qkv_gu` control mean ms | E8 report |
|---|---|---|---|
| text 1 x 512, forward | 7.099 / 7.098 (p50 7.099 / 7.097) | 6.857 (p50 6.854) | 6.869 (p50) |
| full model text 1 x 512, forward | 7.092 / 7.072 | 6.865 | 6.869 (p50) |

The new default costs +3.5 % at text 1 x 512 (141 vs 146 emb/s). Other text shapes were not
re-measured here (the 12 x 10 re-baseline is the next E9 unit); at 12 x 10 the selection
aggregate (1x512 + 8x512 burst) is 41.84 ms vs 38.23 ms for `p3_qkv_gu`.

Served embeddings change with the precision: the tt-inference-server wrapper
(`demo/generator_vllm.py`) builds `TextEncoder.from_pretrained`, which loads the same default
file, so it serves `p2_sdpa_fast` with no code change. `tests/pcc/test_generator_vllm.py` passes
with the new default at 13 x 10 and 12 x 10 (6 passed each; min serving embedding cosine vs HF fp32
0.99433 over 18 rows, bar 0.99).

## Headline (batch 1)

| workload | shape | forward p50 ms | request (h2d_d2h) p50 ms, excl. host prep | request + host prep, mean ms | public API p50 ms | throughput (forward) | serial throughput (request + host prep) |
|---|---|---|---|---|---|---|---|
| text query / short doc | 1 x 128 tokens | 4.852 | 4.930 | 4.992 | | 206 emb/s, 26.4k tok/s | 200 emb/s |
| text document | 1 x 512 tokens | 6.869 | 6.951 | 7.041 | | 146 emb/s, 74.5k tok/s | 142 emb/s |
| text long document | 1 x 8192 tokens | 93.041 | 93.178 | 93.656 | | 10.7 emb/s, 88.0k tok/s | 10.7 emb/s |
| image (soft tokens) | 1 x 2560 patches | 17.395 | 18.151 | 21.937 | | 57.5 images/s | 45.6 images/s |
| audio 30 s (soft tokens) | 1 x 768 tokens | 11.961 | 12.775 | 19.435 | | 83.6 clips/s | 51.5 clips/s |
| full model, text only | 472 tokens (bucket 512) | 6.869 | 6.951 | 7.062 | 7.992 | 145.6 emb/s | 141.6 emb/s |
| full model, image + text | 267 tokens + 1 image | 24.229 | 24.698 | 28.877 | 33.730 | 41.3 emb/s | 34.6 emb/s |
| full model, audio 10 s + text | 260 tokens + 1 clip | 14.170 | 14.353 | 16.815 | 46.994 | 70.6 emb/s | 59.5 emb/s |

The request (h2d_d2h) column excludes host input preparation (`host_prep`, timed separately; tables
below). "Request + host prep" is the `h2d_d2h` mean plus the `host_prep` mean of the same record;
serial throughput is batch / that sum, for one command queue and no overlap between requests.

## Text tower

The text tower runs through `TextEncoder`, with one trace per shape. Every physical token is a real token: natural text from `tests/common.make_batch`. Each case has 3 warm-up replays, then N = 20 timed iterations. Times are ms per forward.

| batch x bucket | mode | mean | p50 | p90 | p99 | min | tokens/s | emb/s | pipelined mean |
|---|---|---|---|---|---|---|---|---|---|
| 1 x 128 | forward | 4.851 | 4.852 | 4.857 | 4.859 | 4.844 | 26,386 | 206.1 | 4.812 |
| 1 x 128 | h2d_d2h | 4.932 | 4.930 | 4.937 | 4.957 | 4.924 | 25,955 | 202.8 | |
| 1 x 512 | forward | 6.869 | 6.869 | 6.875 | 6.877 | 6.859 | 74,533 | 145.6 | 6.830 |
| 1 x 512 | h2d_d2h | 6.950 | 6.951 | 6.955 | 6.968 | 6.935 | 73,667 | 143.9 | |
| 1 x 2048 | forward | 20.090 | 20.092 | 20.101 | 20.106 | 20.073 | 101,942 | 49.8 | 20.058 |
| 1 x 2048 | h2d_d2h | 20.173 | 20.172 | 20.186 | 20.198 | 20.154 | 101,523 | 49.6 | |
| 1 x 8192 | forward | 93.077 | 93.041 | 93.267 | 93.356 | 92.933 | 88,013 | 10.7 | 93.023 |
| 1 x 8192 | h2d_d2h | 93.174 | 93.178 | 93.249 | 93.362 | 93.003 | 87,922 | 10.7 | |
| 1 x 32768 (own session) | forward | 653.903 | 655.744 | 657.088 | 657.359 | 647.308 | 50,111 | 1.53 | 659.355 |
| 1 x 32768 (own session) | h2d_d2h | 663.253 | 663.022 | 665.284 | 665.643 | 660.650 | 49,405 | 1.51 | |
| 8 x 512 | forward | 24.271 | 24.273 | 24.294 | 24.304 | 24.227 | 168,761 | 329.6 | 24.255 |
| 8 x 512 | h2d_d2h | 24.444 | 24.444 | 24.478 | 24.490 | 24.369 | 167,569 | 327.3 | |
| 32 x 128 | forward | 22.228 | 22.227 | 22.258 | 22.276 | 22.181 | 184,272 | 1,439.6 | 22.203 |
| 32 x 128 | h2d_d2h | 22.759 | 22.756 | 22.794 | 22.801 | 22.723 | 179,974 | 1,406.0 | |

Host input preparation covers padding and building the host ttnn tensors. It is timed outside h2d_d2h. Mean ms:

| shape | host prep mean ms |
|---|---|
| 1 x 128 | 0.060 |
| 1 x 512 | 0.091 |
| 1 x 2048 | 0.165 |
| 1 x 8192 | 0.482 |
| 1 x 32768 | 1.716 |
| 8 x 512 | 0.268 |
| 32 x 128 | 0.266 |

Tokenization is not timed for the towers. The full-model `api` rows include it.

**Batch capacity at 128 tokens.** The probe runs one eager forward per batch, ascending, in one device session with no traces live.
- Batches 32, 64, 128, 256, 512, 1024, 2048 and 4096 fit.
- **Batch 8192 fails with a DRAM OOM:** "Not enough space to allocate 570425344 B DRAM buffer across 8 banks ... free: 57827968 B".
- OOM boundary: the largest tested fit is 4096 x 128 (524,288 tokens) on a power-of-two ladder; the first failure is 8192 x 128, so the boundary is in [4096, 8192).
- Only 32 x 128 was timed.

## Vision tower

The vision tower runs through `VisionEncoder`, using processor outputs of the in-tree test images and video (the `bench_vision.py` inputs). The output is the soft tokens from `embed_vision`, per image or frame. Items are images or video frames.

| batch x bucket | inputs | mode | mean | p50 | p90 | p99 | min | items/s | pipelined mean |
|---|---|---|---|---|---|---|---|---|---|
| 1 x 2560 | landscape image | forward | 17.394 | 17.395 | 17.408 | 17.409 | 17.371 | 57.5 | 17.370 |
| 1 x 2560 | | h2d_d2h | 18.184 | 18.151 | 18.239 | 18.473 | 18.097 | 55.0 | |
| 4 x 2560 | 4 images | forward | 88.418 | 88.443 | 88.567 | 88.610 | 88.178 | 45.2 | 88.398 |
| 4 x 2560 | | h2d_d2h | 90.987 | 90.991 | 91.303 | 91.753 | 90.580 | 44.0 | |
| 8 x 1280 | 8 video frames | forward | 69.202 | 69.211 | 69.260 | 69.315 | 69.071 | 115.6 | 69.157 |
| 8 x 1280 | | h2d_d2h | 71.494 | 71.536 | 71.681 | 71.754 | 71.185 | 111.9 | |

Host input preparation covers pixel padding, positions, RoPE tables, masks and the pooling matrix. Mean ms: 3.753 (1 x 2560), 15.043 (4 x 2560), 11.605 (8 x 1280). At 1 x 2560 it is 17 % of the serial total (3.753 of 3.753 + 18.184 = 21.937 ms), and it is not overlapped with device work.

## Audio tower

The audio tower runs through `AudioEncoder`, using log-mel features of the in-tree speech clips (the `bench_audio.py` inputs). The output is the soft tokens from `embed_audio`, per clip. Items are clips.

| batch x bucket | inputs | mode | mean | p50 | p90 | p99 | min | items/s | pipelined mean |
|---|---|---|---|---|---|---|---|---|---|
| 1 x 256 | 10 s speech | forward | 7.323 | 7.323 | 7.332 | 7.333 | 7.311 | 136.6 | 7.312 |
| 1 x 256 | | h2d_d2h | 7.760 | 7.764 | 7.805 | 7.889 | 7.574 | 128.9 | |
| 1 x 768 | 30 s speech | forward | 11.963 | 11.961 | 11.972 | 11.980 | 11.953 | 83.6 | 11.950 |
| 1 x 768 | | h2d_d2h | 12.757 | 12.775 | 12.899 | 12.981 | 12.475 | 78.4 | |
| 4 x 256 | 4 x 10 s speech | forward | 15.388 | 15.399 | 15.413 | 15.426 | 15.333 | 259.9 | 15.376 |
| 4 x 256 | | h2d_d2h | 16.238 | 16.212 | 16.350 | 16.433 | 16.152 | 246.3 | |

Host input preparation covers mel padding, frame masks and the attention mask. Mean ms: 1.946 (1 x 256), 6.678 (1 x 768), 8.314 (4 x 256). At 1 x 768 the device processes 30 s of speech in 11.96 ms, about 2,500x real time.

## Full model (public API)

The full model runs through `EmbeddingGemma2`, one item per forward, with inputs from `reference/full_model_inputs.py`. tokens/s counts logical tokens. ShapeKeys:
- `b1_s512`: text only, 472 tokens.
- `b1_s512_image1x2560`: `image_dog_caption`, 267 tokens.
- `b1_s512_audio1x256`: `audio_speech10s_text`, 260 tokens.

| case | mode | mean | p50 | p90 | p99 | min | emb/s | tokens/s | pipelined mean |
|---|---|---|---|---|---|---|---|---|---|
| text 472 tok | forward | 6.868 | 6.869 | 6.872 | 6.875 | 6.859 | 145.6 | 68,728 | 6.832 |
| text 472 tok | h2d_d2h | 6.950 | 6.951 | 6.956 | 6.979 | 6.923 | 143.9 | 67,912 | |
| text 472 tok | api | 8.020 | 7.992 | 8.183 | 8.580 | 7.789 | 124.7 | 58,851 | |
| image + text | forward | 24.235 | 24.229 | 24.255 | 24.274 | 24.204 | 41.3 | 11,017 | 24.221 |
| image + text | h2d_d2h | 24.699 | 24.698 | 24.733 | 24.834 | 24.642 | 40.5 | 10,810 | |
| image + text | api | 33.713 | 33.730 | 33.902 | 34.049 | 33.363 | 29.7 | 7,920 | |
| audio 10 s + text | forward | 14.171 | 14.170 | 14.176 | 14.187 | 14.162 | 70.6 | 18,348 | 14.131 |
| audio 10 s + text | h2d_d2h | 14.353 | 14.353 | 14.366 | 14.406 | 14.321 | 69.7 | 18,115 | |
| audio 10 s + text | api | 47.213 | 46.994 | 49.848 | 54.637 | 44.055 | 21.2 | 5,507 | |

Host prep covers the batch tensors and the merge index, timed outside h2d_d2h. Mean ms: 0.112 (text), 4.178 (image), 2.462 (audio).

The `api` rows add the HF processor **[I: by difference of the measured means, api - h2d_d2h - host_prep]**:
- Text: about 1.0 ms (8.020 - 6.950 - 0.112 = 0.96).
- Image: about 4.8 ms (33.713 - 24.699 - 4.178 = 4.84).
- 10 s clip: about 30.4 ms (47.213 - 14.353 - 2.462 = 30.40), from log-mel extraction on the host.

The audio item therefore spends 70 % of its public-API latency off the device (47.213 - 14.171 forward = 33.0 ms of 47.2 ms; the processor alone is 64 %).

The full-model device time equals the sum of its towers plus the text pass **[I]**:
- Image + text: 24.23 ms = vision 1 x 2560 (17.39) + text at 512 (6.87); the sum is 24.26.
- Audio + text: 14.17 ms = audio 1 x 256 (7.32) + text at 512 (6.87); the sum is 14.19.

## Burst vs sustained (b1 headline shape per tower)

- **Burst:** the median of 5 windows of 20 back-to-back replays, each window after 3 s idle.
- **Sustained:** 30 s of back-to-back replays in windows of about 0.5 s, with one sync per window. The reported value is the median of the windows in the last 10 s.

| tower | shape | burst ms | sustained ms (last 10 s) | first window ms | mean over 30 s | sustained / burst |
|---|---|---|---|---|---|---|
| text | 1 x 512 | 6.834 | 6.829 | 6.831 | 6.829 | 0.999 |
| vision | 1 x 2560 | 17.354 | **18.283** | 17.358 | 18.110 | **1.054** |
| audio | 1 x 768 | 11.952 | 11.951 | 11.951 | 11.951 | 1.000 |

Vision slows within 2 s of back-to-back replays: 17.36 ms becomes 17.96 ms at 2.0 s. It keeps drifting to 18.28 ms at 25-30 s; the series is in `run/perf.jsonl`. Text and audio show no drift.

What is measured: the drift is specific to the selected vision policy at its larger single-image shapes. In the E6b post-selection bench (same card, same session for both policies; `doc/datatype_sweep/README.md`, `e6b/post_bench_g1.jsonl`) **[M]**:
- the E5 baseline vision policy is flat at 1 x 2560 (burst 26.115 ms, sustained 26.161 ms);
- the selected policy drifts at 1 x 2560 (17.355 -> 18.242 ms) and 4 x 2560 (88.565 -> 89.620 ms);
- the selected policy is flat at 8 x 1280 (69.204 / 69.218 ms).

The cause is **[I]** the P150a power and current management lowering AICLK under the selected vision load; it is not confirmed. AICLK and power were not read during the run. Whether tt-smi can read telemetry while the benchmark holds the card is untested **[I]**.

The 1 x 32768 text case drifts too **[I: same pattern, not confirmed by telemetry]**: the `forward` samples rise +1.5 % (647.7 -> 657.4 ms) over the first ~10 iterations and then stay flat (mean of the last 10: 656.7 ms). The `h2d_d2h` mode ran after `forward`, so its samples (mean 663.3 ms) all sit on the drifted plateau. The +9.3 ms gap between the `h2d_d2h` and `forward` means is therefore likely that drift (mode order), not upload **[I]**: the ramp inside the forward mean accounts for 2.8 ms of it; the other 6.6 ms (against the forward plateau) is either further drift or transfer, and this run does not separate the two. At 1 x 32768 the uploads are 128 KB of uint32 ids and a 2 MB keep mask (bf16, tile layout pads it to 32 rows); the readback is one 768-wide embedding.

## Protocol

### Selection (final perf.py, `4c2e82e044`)

A plain `pytest tests/perf/perf.py` is quick: headline shapes in `forward` mode only. Every extra is an opt-in env switch, where `"1"` means on.
- Switches combine, and `-k` picks single cases.
- Selection happens at collection time, so `--collect-only` shows exactly what will run.
- Costs assume a warm weight and kernel cache. The plain run and the all-on run were timed as whole runs. The per-option deltas are **[I]**, estimated from the per-class log timestamps of the full-sweep run and rounded to 0.1 min; they sum to the measured 318 s (5.3 min). The sustained delta is 3 towers x (30 s sustained + 5 burst windows after 3 s idle each), about 2.3 min.

| option | adds | about |
|---|---|---|
| (none) | headline shapes, `forward` mode: text 1 x 512, vision 1 x 2560, audio 1 x 768, full model text 472 tokens; finite-output asserts | 0.3 min (measured 15-16 s) |
| `EMBEDDING_GEMMA2_PERF_SHAPES=1` | every shape: text 1 x 128 / 512 / 2048 / 8192, 8 x 512, 32 x 128; vision 1 x 2560, 4 x 2560, 8 x 1280; audio 1 x 256, 1 x 768, 4 x 256; full text, image + text, audio + text | +0.5 min |
| `EMBEDDING_GEMMA2_PERF_MODES=1` | `h2d_d2h`, `api` (full model) and `2cq` (records "not supported") besides `forward` | +0.3 min |
| `EMBEDDING_GEMMA2_PERF_LONG=1` | text 1 x 32768 in its own device session | +0.8 min |
| `EMBEDDING_GEMMA2_PERF_PROBE=1` | text batch-capacity probe at 128 tokens (32 ... 8192, stops at the first OOM) | +0.8 min |
| `EMBEDDING_GEMMA2_PERF_SUSTAINED=1` | 30 s sustained vs burst per tower at its headline shape | +2.3 min |
| `EMBEDDING_GEMMA2_PERF_HF_CHECK=1` | HF fp32 comparison of the timed output (one shape per tower, every selected full case) | +0.3 min |
| `EMBEDDING_GEMMA2_PERF_ALL=1` | all of the above: the full sweep of this report | 5.3 min (measured 318 s) |

The selection logic was checked without rerunning the full sweep (`tt-ops/runs/embeddinggemma-2/e8/logs/collect_check.log`):
- `EMBEDDING_GEMMA2_PERF_ALL=1 --collect-only` collects 65 ids, identical to the ids the full-sweep run executed (empty diff).
- The default collects the 4 headline ids.
- Per-option collection counts: SHAPES 15, MODES 13, LONG 5, PROBE 13, SUSTAINED 7, HF_CHECK 4.
- Two opt-in runs on the card passed on `4c2e82e044`: HF_CHECK on vision, and SHAPES + MODES on audio 4 x 256.

### Timing and checks

- **Device:** one P150a is opened per group: text, text 1 x 32768, batch probe, vision, audio, full. Each opens with `trace_region_size` 128 MiB and one command queue. Each group:
  1. loads its model through the normal constructor (`from_pretrained`, no precision override);
  2. captures every shape of the group with one `capture_all` (compile and allocate first, then capture);
  3. times the cases, releases the traces and closes the device.
- **Inputs:** real data, prepared on the host once per case. The timed loops replay those exact inputs; `request` first loads them into the persistent trace inputs.
- **`forward`:** `execute_trace(blocking=False)` plus `synchronize_device` per iteration.
  - Includes the device time of the whole traced forward (towers, merge, text encoder, pooling, normalize) and one host sync. The sync costs tens of microseconds; the pipelined mean is 0.01-0.04 ms lower at the short shapes.
  - Excludes all host input work and readback.
- **`forward_pipelined`:** N back-to-back replays and one sync, mean only. This is the protocol of the E5 and E6 benches, reported for comparison.
- **`h2d_d2h`:** `copy_host_to_device_tensor` of every prepared host input tensor, then the replay, then the output read to torch. For text and full, the output is the [B, 768] embedding; for vision and audio, the soft tokens with padding rows stripped. It excludes host input preparation, which is timed separately as `host_prep`.
- **`api`** (full model only): `EmbeddingGemma2.encode([item], traced=True)`, covering the HF processor, batch prep, upload, replay and readback.
- **Statistics:** 3 warm-up replays before every timed loop, then N = 20 timed iterations. Reported: mean, p50, p90, p99, min and max, with numpy linear interpolation. With 20 samples, p99 is interpolated between the two largest samples, so treat it as "near max".
- **Throughput:**
  - tokens/s = batch x bucket / mean. For text, every position is a real token; for the full model, logical tokens are counted.
  - embeddings/s and items/s = batch / mean.
  - Requests do not overlap (one command queue). Serial request throughput is batch / (h2d_d2h mean + host_prep mean), because host prep is timed outside `h2d_d2h` and also runs serially. From `run/perf.jsonl`: vision 1 x 2560 45.6 items/s, audio 1 x 768 51.5 clips/s, audio 4 x 256 163 clips/s, text 1 x 512 142 emb/s (headline table, last column).
- **Checks inside the timed path:**
  - Every case asserts a finite output; full-model cases also assert unit norm.
  - With HF_CHECK (on in the full sweep), the output of the timed replay is compared against HF fp32. All bars are 0.99.

    | case | metric | value |
    |---|---|---|
    | text 1 x 512 | embedding cosine | 0.999004 |
    | vision 1 x 2560 | soft-token PCC (flattened cosine) | 0.996577 (0.996576) |
    | audio 1 x 768 | soft-token PCC (flattened cosine) | 0.997668 (0.997656) |
    | full text | embedding cosine | 0.999515 |
    | full image + text | embedding cosine | 0.999532 |
    | full audio + text | embedding cosine | 0.999631 |

  - The HF reference runs on the host before the device opens. Vision, audio and full use the same loaders as the tower tests; text uses `run_reference`.
- **Trace == eager:** `tests/pcc/test_full_model_trace.py` was rerun on the final code commit `4c2e82e044`: 11 passed. The other gates also passed on that commit: `test_full_model.py` 30 passed, `demo/demo.py` 2 passed, `test_precision_config.py` 7 passed. Details are in RUN_NOTES.md.

## Comparison with earlier stages

All runs use the same card and the same inputs. The earlier numbers are burst medians (pipelined protocol), so compare them with the `pipelined mean` column. Where a row has two earlier numbers, the change column lists them in the same order.

| shape | E8 pipelined ms | earlier ms | source | change |
|---|---|---|---|---|
| text 1 x 128 | 4.812 | 4.813 (E6a selected, burst) | doc/datatype_sweep/README.md, post-selection table | 0.0 % |
| text 1 x 512 | 6.830 | 6.833 | same | 0.0 % |
| text 1 x 2048 | 20.058 | 20.054 | same | 0.0 % |
| text 8 x 512 | 24.255 | 24.243 | same | +0.05 % |
| text 1 x 8192 | 93.023 | 93.054 | same | 0.0 % |
| text 1 x 512 | 6.830 | 7.315 (E5a), 18.314 (E4) | doc/optimized_text/README.md | -6.6 % / -62.7 % |
| vision 1 x 2560 | 17.370 | 17.35 (E6b selected); 26.271 (E5b) | doc/datatype_sweep/README.md (Selected tower configs); doc/optimized_vision/README.md | +0.1 % / -33.9 % |
| vision 4 x 2560 | 88.398 | 88.51; 116.020 | same | -0.1 % / -23.8 % |
| vision 8 x 1280 | 69.157 | 69.20; 88.142 | same | -0.1 % / -21.5 % |
| audio 1 x 256 | 7.312 | 7.31 (E6b selected); 8.739 (E5c) | doc/datatype_sweep/README.md; doc/optimized_audio/README.md | 0.0 % / -16.3 % |
| audio 1 x 768 | 11.950 | 11.95; 15.021 | same | 0.0 % / -20.4 % |
| audio 4 x 256 | 15.376 | 15.39; 19.154 | same | -0.1 % / -19.7 % |
| full text 1 x 512 (device) | 6.832 | 18.30 (E4) | doc/full_model/README.md §8 | -62.7 % |
| full image + caption (device / e2e) | 24.221 / 24.699 | 73.90 / 74.37 (E4) | same | -67.2 % / -66.8 % |
| full audio 10 s + text (device / e2e) | 14.131 / 14.353 | 41.99 / 45.20 (E4) | same | -66.3 % / -68.2 % |
| text 1 x 32768, 32 x 128 | 659.355, 22.203 | not measured before (65536 eager 2.23 s, E5a) | doc/context_contract.json | new |

The E8 numbers reproduce the E6 selected-config numbers within 0.1 % **[M]**. The full-model rows are the first measurement of the full model on the optimized, precision-swept code.

## Limitations

- **2cq: not supported.** Every encoder (`TextEncoder`, `VisionEncoder`, `AudioEncoder`, `EmbeddingGemma2`) copies inputs and replays on command queue 0 (`cq_id=0` in `capture_all` and `replay`). The device also opens with `num_command_queues=1`. Input upload therefore cannot overlap the previous replay. A second queue was not built in this stage, which is benchmark only. Each 2cq case records the reason and skips.
- **Batch boundary (text, seq 128):** 4096 fits; 8192 is a DRAM OOM. This was measured eagerly, with no traces live, in its own session. With traces of other shapes resident, the boundary can be lower **[I]**. Only 32 x 128 was timed; batches 64 to 4096 were not.
- **Throttling:** vision sustains 5.4 % slower than burst after 30 s. Text 1 x 512 and audio 1 x 768 showed no drift in this run. Text does throttle at long buckets: E9 telemetry (`doc/prefill/01-baseline/README.md`) measured median AICLK 1312 MHz at 1 x 32768 and 1306 MHz at 1 x 65536 (minimum 1293 / 1268 MHz), and sustained 0.6-1.9 % slower than burst from 1 x 4096 up; text up to 1 x 2048 held 1350 MHz. Burst numbers are the "short request" figure. For a saturated server, use the sustained vision number.
- **p99:** with N = 20 it is near the maximum sample, not a tail estimate.
- **Host work is serial:** vision host prep (3.8-15 ms) and audio feature extraction (about 30.4 ms for 10 s, from the `api` row minus h2d_d2h and host prep) run on the host before upload and do not overlap with device work.
- **Text 1 x 32768:** this bucket has an HF check from E5a (`doc/context_contract.json`). This run only checks that its output is finite. 65536 was not benchmarked; it has no HF reference (see the contract notes).
- **Video:** not part of the full-model sweep. The 8 x 1280 vision trace is the video tower number.
- **Accuracy:** no MTEB or lm-eval, by decision. The HF checks above are cosine and PCC guards only.
- **Option costs:** the per-option costs in the selection table are rounded estimates from log timestamps that sum to the measured 318 s. Only the plain run and the all-on run were timed as whole runs.

## Next (not done here: this stage measures, it does not tune)

- Add a second command queue for input upload (the `2cq` mode). It would hide the 0.1-2.6 ms h2d of the towers.
- Move vision host prep (RoPE tables, pooling matrix) to cached tables per grid shape. Move the audio log-mel to the device or a worker thread. These dominate the image and audio public-API latency.
- Time the text batch sweep beyond 32 at 128 tokens to find the throughput knee; capacity reaches 4096.
- Read AICLK and power during the vision sustained run to confirm the throttle mechanism, using telemetry from a second process if the driver allows it.

## Reproduce

From the tt-metal root, with a built checkout:
1. Activate the tt-metal Python environment (`source python_env/bin/activate`), with `TT_METAL_HOME` and `PYTHONPATH` at the checkout root.
2. Select the card: `export TT_VISIBLE_DEVICES=<card index>`.
3. Run offline against a local snapshot of `google/embeddinggemma-2` at revision `914f7f89`: `export HF_HUB_OFFLINE=1 HF_MODEL=<path to that snapshot>`.

```bash
pytest models/demos/embedding_gemma2/tests/perf/perf.py -s                              # quick (headline, forward)
EMBEDDING_GEMMA2_PERF_ALL=1 pytest models/demos/embedding_gemma2/tests/perf/perf.py -s  # full sweep (this report)
```

Raw data:
- `run/perf.jsonl`: every record, with per-iteration samples and the sustained series.
- `run/perf.csv`: summary.
- `run/run_info.json`: commit, precision hash, versions.
- `run/device_info.json`.

Identity of the run: `identity.json`. Commands, timestamps, reruns and the host paths used for this run: `RUN_NOTES.md`.
