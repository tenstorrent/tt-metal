# Optimized profiler provenance

Captured 2026-07-31 UTC at checkout `e5623e55e6997208dd64347103c52747b6e51df8`
on Blackhole P300 device 3.

Capture source SHA256:

- `optimized_decoder.py`: `3589c270b4c8f8b57348479eaf62479ace88702d0775e21d61ff2052133886b3`
- `test_optimized_decoder.py`: `3e09e6704bc5977c1235baea61ad4275e4a72c15c28cd6d003509ccfcf001a27`

The valid traced replay capture was:

```bash
TT_METAL_DEVICE_PROFILER=1 GEMMA4_RANGE_DOWNLOAD=1 \
GEMMA4_OPTIMIZED_DECODER_PERF=1 \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
python -m tracy -v -p --device-trace-profiler \
  -o models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/tracy/optimized_trace_bfp8_lofi_final_raw \
  -m pytest \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py::test_optimized_decoder_perf_profile \
  -q
```

All four optimized cases passed. The compact
`device_trace_latency.csv` transcribes `cpp_device_perf_report.csv`; each
headline device time is the second replay. Architecture metadata is
`blackhole`, all durations are positive, and device/host times agree within
0.36%, so this capture is valid.

An earlier four-case modern per-operation capture overflowed its marker
buffer. That result is retained as historical failure evidence, but it was
superseded by isolated one-case captures with `--op-support-count=10000`:

```bash
GEMMA4_RANGE_DOWNLOAD=1 GEMMA4_OPTIMIZED_DECODER_PERF=1 \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
python -m tracy -r -p --check-exit-code \
  -o models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/tracy/optimized_ops_raw \
  -m pytest \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py::test_optimized_decoder_perf_profile \
  -q
```

```bash
GEMMA4_RANGE_DOWNLOAD=1 GEMMA4_OPTIMIZED_DECODER_PERF=1 \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
python -m tracy -r -p --op-support-count=10000 --check-exit-code \
  -o doc/optimized_decoder/tracy/optimized_ops_exact_<layer> \
  -m pytest '...::test_optimized_decoder_perf_profile[<exact-node-id>]' -q
```

Both isolated captures joined successfully and produced current Blackhole
v2.1 operation reports. The failed combined capture had reported:

```text
AssertionError: Device data missing: Op 1136643 not present in
cpp_device_perf_report.csv for device 3 (trace_id=None)
```

The legacy parser's negative durations and Wormhole metadata remain invalid
and are not combined with current results. Authoritative per-op artifacts are
`sliding_b1_{decode,prefill}_perf_report.csv`,
`full_b1_{decode,prefill}_perf_report.csv`, and their summary CSVs.

## Current per-operation conclusions

At batch-1 decode, sparse expert matmuls are the dominant class: 41.48% of
sliding device time and 37.44% of full-attention device time. The selected
`in0_block_w=11` was attacked against widths 2, 22, and 44. Width 2 was
slower; width 22 was slightly slower at batch 1 and exceeded L1 by 1,280 B at
batch 32; width 44 required 2,319,616 B of circular buffers versus
1,572,864 B physical L1. The selected width 11 is therefore the best correct
geometry at both required batches. The exact extended sweep is in
`../candidates/whole_layer/sparse_extended_geometry.json`.

Dense matmuls account for 19.86% sliding and 24.10% full decode. A coherent
DRAM-sharded BFP8 candidate with wider 8-core/6-core geometry passed
real-weight PCC but lost whole-layer latency at both batches, so the selected
packed interleaved BF16 dense path remains. Decode layer norms account for
16.71% sliding and 15.10% full; the coherent sharded-boundary candidate was
measured as part of the losing DRAM-sharded dense family. SDPA is
2.48%/2.52%, and cache updates are below 1%. Four
`UntilizeWithUnpadding` rows plus one `TilizeWithValPadding` row are about
1.1% combined. They are internal logical-extent/padded-tile boundaries of
the selected matmul composites; the Python forward contains no explicit
tilize/untilize or reshard operation.

Prefill is dominated by the canonical sparse expert class (96.90% sliding,
96.74% full). The explicit large packed dense config is correspondingly
non-material and measured neutral/slower. Sparse prefill uses a distinct
active-128 contract; decode width 11 is not applied to prefill, preserving
the canonical path and non-aligned public sequence contract.

## Same-run performance accounting

The roofline uses the `tt-perf-report` Blackhole specification of 512 GB/s
and BF16 bytes actually consumed by the batch-1 layer: packed attention
projections, packed dense gate/up and down, router, eight routed experts,
and the KV read window. It excludes inactive expert weights.

| Layer | Estimated bytes/token | Roofline ms | Device measured ms | Host measured ms | Host-device gap ms |
|---|---:|---:|---:|---:|---:|
| sliding | 209 MB | 0.408 | 1.815 | 1.886 | 0.071 |
| full | 234 MB | 0.457 | 2.003 | 2.084 | 0.081 |

The 4.4x device/roofline gap is consistent with a heterogeneous layer
of small norms, routing, sparse expert launches, cache operations, and SDPA,
not one ideal streaming matmul. The profiler-instrumented host/device gap is
about 3%; trace replay
has no material host orchestration bottleneck. Final warmed prefill medians
are 669.805 ms sliding and 671.090 ms full. Current per-op prefill reports
show the canonical sparse expert class accounts for 96.90% and 96.74%,
respectively.
