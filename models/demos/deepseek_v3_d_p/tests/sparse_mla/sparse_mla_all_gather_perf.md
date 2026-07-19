# Sparse MLA all-gather performance

Measured on the LoudBox SP=2 × TP=4 proxy with the realtime profiler. The current
implementation includes native `ttnn.all_gather` in MLA and the indexer, plus
row-major top-k-index gathering. It additionally gathers sparse-MLA KVPE directly
from the ND-sharded cache, selecting the active cache slot and transferring only
its populated prefix.

Negative latency deltas mean the current implementation is faster.

## Full code-state E2E: historical main BF16 to target scaled FP8

Fresh measurements on 2026-07-19 compare the complete historical baseline at
`f1f4ff75579` (`origin/main`, before the first all-gather commit) with the current
scaled-FP8 candidate. This includes every model, cache-format, sparse-SDPA,
indexer, and collective change—not only all-gather. Both sides use the same
LoudBox SP=2 × TP=4 workload and the same in-process realtime-profiler rule:
sum device programs after taking the maximum duration across chips for each
program. The baseline model/kernel code is unchanged; only its perf harness was
adapted from legacy multiprocess Tracy to the same realtime timing method.

| Model | Case | Historical main BF16 (ms) | Target scaled FP8 (ms) | Latency reduction | Speedup |
|---|---|---:|---:|---:|---:|
| DeepSeek v3.2 | warm | 9.384 | 7.460 | 20.50% | 1.258× |
| DeepSeek v3.2 | cold | 88.144 | 70.198 | 20.36% | 1.256× |
| DeepSeek v3.2 | long | 19.276 | 13.229 | 31.37% | 1.457× |
| GLM 5.1 | warm | 8.105 | 5.913 | 27.05% | 1.371× |
| GLM 5.1 | cold | 83.116 | 59.400 | 28.53% | 1.399× |
| GLM 5.1 | long | 17.504 | 11.135 | 36.38% | 1.572× |

All twelve measured test executions passed (six baseline and six target). The
target run asserted the receiver-L1 path: one receiver program in warm/long and
eleven over the eleven-forward cold case. The old composite implementation also
cannot execute the forced SP=4 × TP=2 model path on this revision, so the complete
historical A/B uses main's supported SP=2 × TP=4 topology on both sides.

## Latest isolated 8×1 ring bandwidth

Seven-sample steady-state measurements use persistent DRAM output, L1-small
semaphores, and the dtype-specific automatic path. The 64,640-row case is the
canonical long sparse-MLA geometry: `(512,000 cached + 5,120 live) / 8 SP`.
The older 32,768-row case is retained only as a half-size tuning point.
`effective_receive_bw` is useful bytes received by one rank from the other seven
ranks divided by device-program time; it is not raw wire traffic.

The historical baseline is the unchanged old composite all-gather at
`origin/main` (`f1f4ff75579`), with only a harness switch from line to ring. It
cannot accept a persistent output and dispatches two device programs: broadcast
then concat. Its seven samples were 28.755, 28.753, 28.744, 28.737, 28.769,
28.731, and 28.745 ms.

| Rows/rank | Global rows | Implementation | Format | Path | Page size | Median (ms) | Min / p90 (ms) | Effective receive BW |
|---:|---:|---|---|---|---:|---:|---:|---:|
| 64,640 | 517,120 | old composite | BF16 | broadcast + concat | 1152 B | 28.745 | 28.731 / 28.769 | 18.134 GB/s |
| 64,640 | 517,120 | new native | BF16 | direct | 1152 B | 8.600 | 8.589 / 8.625 | 60.612 GB/s |
| 64,640 | 517,120 | new native | scaled FP8 | receiver-L1 | 704 B | 6.319 | 6.305 / 6.338 | 50.408 GB/s |
| 32,768 | 262,144 | new native | BF16 | direct | 1152 B | 4.359 | 4.349 / 4.371 | 60.626 GB/s |
| 32,768 | 262,144 | new native | scaled FP8 | receiver-L1 | 704 B | 3.246 | 3.222 / 3.281 | 49.746 GB/s |

At the production long shape, new-native BF16 is 3.342x faster than the old
BF16 composite (70.1% lower latency). The old median splits into 26.086 ms of
broadcast and 2.658 ms of concat. The scaled-FP8 target is 4.549x faster than
the original BF16 composite (78.0% lower latency), though that comparison also
includes the cache-format reduction.

The Blackhole ring exposes two routing planes in each of its two directions,
giving four aggregate ingress links per rank. The physical link capability is
400 Gb/s (50 GB/s) per link, but the current CCL roofline explicitly models the
devices at half bandwidth, 25 GB/s per link. Therefore:

- current enabled/modelled ceiling: `4 × 25 = 100 GB/s`;
- physical line-rate ceiling: `4 × 50 = 200 GB/s`;
- BF16 reaches 60.6% of the current ceiling (30.3% of physical line rate);
- scaled FP8 reaches 49.7% of the current ceiling (24.9% of physical line rate).

At the production long shape, scaled FP8 finishes 26.5% faster even though its
reported GB/s is lower: it moves 61.1% as many bytes as BF16 but takes 73.5% as
long. Fixed per-page,
fabric, receiver-drain, and DRAM-write costs do not shrink in proportion to the
payload.

| Model | Case | Legacy total (ms) | Current total (ms) | Latency delta | Legacy collective (ms) | Current collective (ms) | Collective delta |
|---|---|---:|---:|---:|---:|---:|---:|
| DeepSeek v3.2 | warm | 9.414 | 8.964 | -4.8% | 1.227 | 0.980 | -20.1% |
| DeepSeek v3.2 | cold | 88.466 | 80.760 | -8.7% | 13.066 | 7.308 | -44.1% |
| DeepSeek v3.2 | long | 19.415 | 17.140 | -11.7% | 7.843 | 6.765 | -13.7% |
| GLM 5.1 | warm | 7.950 | 6.344 | -20.2% | 3.859 | 2.447 | -36.6% |
| GLM 5.1 | cold | 83.342 | 61.892 | -25.7% | 42.721 | 23.630 | -44.7% |
| GLM 5.1 | long | 17.442 | 13.998 | -19.7% | 10.507 | 8.211 | -21.9% |

Collective time is accounted as follows:

- Legacy: `AllGatherAsync + AllBroadcast`.
- Current: native all-gather kernels reported as `ccl`, plus `AllBroadcast`.

The cold proxy is the meaningful prefix-gather comparison: it fills an already
allocated cache over eleven iterations. Fusing the index-cache slot selection and
valid-prefix limit into its gather reduces current total latency by a further
0.081 ms (DeepSeek) and 0.259 ms (GLM) versus the prior native result. Warm and
long allocate their caches to exactly the measured context, so the populated prefix
is the full slab and they are expected to be within run-to-run noise of the prior
result.

Warm/long collective program counts:

| Model | Legacy | Current |
|---|---:|---:|
| DeepSeek v3.2 | 5 `AllGatherAsync` + 2 `AllBroadcast` | 6 `ccl` + 1 `AllBroadcast` |
| GLM 5.1 | 7 `AllGatherAsync` + 2 `AllBroadcast` | 8 `ccl` + 1 `AllBroadcast` |

The remaining current `AllBroadcast` is the broadcast phase of one composite
all-gather fallback. Its tiled input has padding on the gather dimension, whereas
the native path requires that logical and padded extents match. It is unrelated to
the direct KVPE-prefix gather above.

## Scaled-FP8 main KV-cache A/B

On the same LoudBox SP=2 × TP=4 realtime-profiler setup, the main sparse-MLA KV
cache was changed from row-major BF16 (1152 logical bytes/token) to its packed
scaled-FP8 representation (656 logical bytes/token). The direct native cache
gather remains in use for both formats, including its user/layer selection and
valid-prefix transfer. Negative deltas mean scaled FP8 is faster.

| Model | Case | BF16 total (ms) | Scaled-FP8 total (ms) | Latency delta | BF16 collective (ms) | Scaled-FP8 collective (ms) |
|---|---|---:|---:|---:|---:|---:|
| DeepSeek v3.2 | warm | 8.951 | 7.740 | -13.5% | 1.044 | 1.036 |
| DeepSeek v3.2 | cold | 80.935 | 71.762 | -11.3% | 7.843 | 7.755 |
| DeepSeek v3.2 | long | 17.256 | 15.827 | -8.3% | 6.841 | 6.751 |
| GLM 5.1 | warm | 6.425 | 6.232 | -3.0% | 2.510 | 2.482 |
| GLM 5.1 | cold | 62.565 | 60.979 | -2.5% | 24.361 | 24.262 |
| GLM 5.1 | long | 14.110 | 13.890 | -1.6% | 8.325 | 8.262 |

The packed row is 656 B, which occupies a 704-B aligned DRAM page. Native
all-gather now permits this exact height-gather case when output is interleaved:
the physical row stride is preserved, so transferring the aligned page is safe.

## Receiver-confirmed Tensix A/B/A (SP=4 x TP=2)

This is a same-build A/B/A comparison on the available eight-chip P150 system.
`DS_PERF_MESH_SHAPE=4x2` makes the model proxy use the target four-device
sequence axis.  The B leg forces the established direct path; the two A legs use
automatic selection.  Both automatic legs asserted that exactly the intended
main-KV gather contained `multicast_receiver_writer.cpp`, while the direct leg
asserted that no receiver program was present.  Positive speedups mean automatic
receiver selection is faster.  Automatic values are the mean of the two A legs.

| Model | Case | KV format | Direct total (ms) | Auto total (ms) | Total speedup | Direct CCL (ms) | Auto CCL (ms) | CCL speedup |
|---|---|---|---:|---:|---:|---:|---:|---:|
| DeepSeek v3.2 | warm | BF16 | 11.297 | 11.244 | 0.47% | 1.004 | 0.900 | 10.34% |
| DeepSeek v3.2 | warm | scaled FP8 | 10.556 | 10.092 | 4.40% | 0.985 | 0.625 | 36.55% |
| GLM 5.1 | warm | BF16 | 9.351 | 9.318 | 0.35% | 0.997 | 0.888 | 10.95% |
| GLM 5.1 | warm | scaled FP8 | 8.181 | 7.724 | 5.58% | 0.972 | 0.617 | 36.51% |
| DeepSeek v3.2 | long | BF16 | 27.738 | 26.630 | 3.99% | 8.163 | 7.047 | 13.67% |
| DeepSeek v3.2 | long | scaled FP8 | 26.383 | 23.182 | 12.13% | 7.699 | 4.433 | 42.42% |
| GLM 5.1 | long | BF16 | 22.702 | 21.541 | 5.11% | 8.142 | 7.022 | 13.76% |
| GLM 5.1 | long | scaled FP8 | 20.928 | 17.645 | 15.69% | 7.711 | 4.424 | 42.63% |

Device-program accounting is unchanged between direct and automatic execution:
58/60 programs for DeepSeek BF16/FP8 and 71/73 for GLM BF16/FP8.  DeepSeek has
five CCL programs and GLM has four in both modes.  Automatic execution changes
the kernel implementation inside one existing CCL program; it does not dispatch
another user-visible collective.  Each warm/long automatic run observed one
receiver program.  The cold BF16 and FP8 reuse controls observed eleven receiver
programs over eleven forwards.

The older SP=2 table was not a valid receiver A/B: the model did not provide a
persistent output, so profiler inspection showed zero receiver programs in both
columns.  Model integration now owns one persistent interleaved DRAM output for
the main KV-prefix gather and reuses it across forwards.  The ND-sharded DRAM KV
cache is accepted only when its page geometry exactly matches that output.  The
binding returns a temporary tensor wrapper for a supplied output tensor, so the
model deliberately returns its cache-owned wrapper; otherwise downstream cleanup
can deallocate the persistent allocation after the first forward.

Validation after that lifetime correction:

- 11/11 cold forwards pass for both BF16 and scaled FP8 with receiver dispatch;
- full `test_sparse_mla.py`: 39/39 pass across BF16/scaled-FP8 and 2x4/4x2
  in 950.50 s on the final release-built topology-policy candidate;
- the release build passes with `./build_metal.sh --release`; every pytest
  invocation uses `scripts/run_safe_pytest.sh`, and no `--dev` build was used.

Commands used for the B and A legs (replace `force_direct`/`0` with `auto`/`1`):

```bash
DS_PERF_MESH_SHAPE=4x2 \
TTNN_ALL_GATHER_RECEIVER_L1_MODE=force_direct \
TTNN_EXPECT_RECEIVER_L1=0 \
DS_PERF_CSV=sp4_matrix_direct.csv \
scripts/run_safe_pytest.sh \
  models/demos/deepseek_v3_d_p/tests/sparse_mla/test_sparse_mla_perf.py \
  -m perf -k '(warm or long) and not dense' -q -s
```
