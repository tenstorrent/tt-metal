# Gemma 4 26B-A4B functional decoder

This is the correctness-first, single-device TTNN functional decoder for
`google/gemma-4-26B-A4B-it`. Scope is limited to this decoder stage.

The implementation covers both meaningful layer kinds:

- layer 0 `sliding_attention`: 16 Q heads, 8 KV heads, head dimension 256,
  window 1024, page size 64;
- layer 5 `full_attention`: 16 Q heads, 2 KV heads, head dimension 512,
  page size 128, K reused as V, natural and shared physical cache views.

## Runtime contract

`tt/functional_decoder.py` documents the call contract. Prefill accepts a
positive logical sequence, pads internally to tiles, follows the supplied page
table, fills the paged cache, and returns the logical length. Batch 1 and 2 are
covered. Long prefill dispatches to bounded chunked attention above the
non-chunked 32768-token limit. A non-aligned bounded-modulo tail is written with
exact per-position paged updates so padding cannot wrap over live cache data.

Decode accepts `[1, 1, batch, 2816]` with device-resident current positions,
page tables, and KV caches. It is trace-capture safe at contract batches 1 and
32. The mutable replay test overwrites stable hidden, rotary, position, page
table, and KV-cache buffers in A/B/A order and observes distinct B output plus
deterministic repeated A output.

## Correctness

The acceptance threshold is PCC >= 0.995 against Hugging Face with real
checkpoint weights.

| Layer/cache case | Prefill PCC | Decode PCC |
| --- | ---: | ---: |
| sliding, shared physical cache | 0.998617 | 0.999655 |
| full, natural physical cache | 0.997773 | 0.999861 |
| full, shared physical cache view | 0.997773 | 0.999861 |

Traced decode HF PCC is 0.999418/0.999283 for sliding batch 1/32 and
0.999861/0.999861 for full batch 1/32. Eager-to-replay and repeated-replay PCC
are 1.0. Real-shape batch-2 prefill PCC is 0.997502 sliding and 0.998994 full.
The normal no-download suite also runs deterministic full-target-shape
synthetic HF-vs-TTNN prefill/decode: sliding 0.999496/0.999203 and full
0.998047/0.999635. `synthetic_pcc_*.json` binds those results to the exact
source/test/binary hashes.

Boundary coverage is:

- sliding: `1,31,32,33,63,64,65,1023,1024,1025`, minimum PCC 0.996049;
- full: `1,31,32,33,127,128,129,1023,1024,1025`, minimum PCC 0.998098.

The selected-row long-attention test covers positions 32767, 32768, and 32799.
Minimum PCC is 0.996791 sliding and 0.998736 full. The exact artifacts are
`long_prefill_attention_{sliding,full}_attention.json`.

## Context

There is no capability reduction. Real-weight whole-layer prefill passes for
both layer kinds at aligned 262144 and non-aligned 262143 tokens:

| Layer | 262144 host pass | 262143 host pass |
| --- | ---: | ---: |
| sliding | 175.017 s | 175.017 s |
| full | 262.357 s | 262.378 s |

Traced batch-1 decode passes at current position 262143 with repeat PCC 1.0.
The rolled page tables address nonzero cache baselines and three distinct
history sentinels at logical positions 0, 131071, and 262142. Small device
readbacks prove each sentinel reached its intended physical page and survived
trace replay. The physical K+V cache allocations are 2 GiB sliding and 1 GiB
full. All context artifacts carry decoder/test/binary hashes, checkout SHA,
hardware, and exact commands. See `../context_contract.json`,
`prefill_capacity_*_26214{3,4}.json`, and
`advertised_context_decode_*.json`.

## Precision and layout policy

Runtime matmuls, router, non-chunked/sliding SDPA, layouts, and compute kernels
use framework defaults except for recorded correctness requirements:

- HiFi4, exact math, FP32 destination accumulation for RMS norms and sparse
  expert gate projection;
- the same compute config for full chunked prefill SDPA, whose default PCC was
  0.975739 at position 32768;
- decode SDPA Q32/K64 with a workload-derived grid capped at 8x4. The default
  batch-32 PCC was 0.630007 sliding and 0.958089 full; a full-grid sliding
  control remained 0.629103.

The A/B ledger is in `precision_exception_ab.json` and
`sdpa_program_config_ab.json`. Dense matmuls have no program configs or
hand-tuned shard specs. `ttnn.sparse_matmul` makes its
`MatmulMultiCoreReuseMultiCast1DProgramConfig` a mandatory API argument; the
decoder uses Gemma-4's canonical shape-derived builder for only those three
sparse expert projections. Minimal workload-derived L1 layouts exist only
where paged cache update, decode SDPA, or decode head concat requires sharded
input.

The QKV DRAM-to-L1 promotion was removed after exact-shape op-level DRAM/L1
A/B tests produced PCC 1.0. Post-removal real-weight and traced batch-32
regressions pass with zero promotion hits. No frozen pre-removal
whole-decoder contrast is retained; the decoder JSONs are post-removal
regressions, not an A/B.

## Performance

Sequence/current position is 1024 on one Blackhole P300. Each result has a
warmup. Prefill is a signpost-filtered device-op report; decode is the second
device trace replay.

| Layer | Mode | Batch | Device time | Synchronized host |
| --- | --- | ---: | ---: | ---: |
| sliding | prefill | 1 | incomplete* | 681.667 ms |
| full | prefill | 1 | incomplete* | 682.521 ms |
| sliding | traced decode | 1 | 2.991 ms | 3.038 ms |
| sliding | traced decode | 32 | 68.860 ms | 68.969 ms |
| full | traced decode | 1 | 3.177 ms | 3.204 ms |
| full | traced decode | 32 | 68.628 ms | 68.723 ms |

*The synchronized warmed prefill host window is authoritative. Device marker
buffers filled during the large prefill capture, so the human-readable
`tt-perf-report` tables and filtered CSVs are retained as diagnostic per-op
data, not mislabeled as a complete device window. Sparse host-only calls also
appear as zero-duration rows and missing architecture metadata makes the
report tool print a Wormhole fallback. Dedicated trace profiling did not drop
markers and provides both warmup and measured decode device rows.

The tables, filtered CSVs, compact raw trace CSV,
`decode_trace_latency.{txt,csv}`, and a hash/command manifest are under
`tracy/`.

Provenance SHA-256:

- layer 0 filtered prefill CSV:
  `3d36c1a3baeeed6fd7049a065a1e6b92bb593319493cc0a980c31040f32c91a2`;
- layer 5 filtered prefill CSV:
  `1d18dbf349a699babab6a928dac622aac69c2a299d46396ed499da7ee108037d`;
- decode trace summary CSV:
  `c622e9d9367a4e422db25d5d926bfebd4bde5931f515f835c3dca42de17f303f`.

`tracy/provenance.json` binds every retained CSV/TXT to decoder/test/binary
hashes, checkout SHA, hardware, and exact commands. The modern Tracy
postprocessor failed after dropped markers made the host/device join
incomplete; the supported legacy device-log parser generated the diagnostic
prefill reports.
The supported legacy parser generated prefill reports. Decode therefore retains
the profiler's aggregate trace CSV directly; no per-op replay durations are
fabricated.

## Runtime and device gates

- Static hot-path audit finds no `torch`, `ttnn.from_torch`, `ttnn.to_torch`,
  or host fallback inside a measured pass.
- All correctness and performance runs set
  `TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}'`.
- The final `TT_METAL_WATCHER=10` run passed 9/9 real-weight, traced, and
  mutable-buffer cases in 49.701 s. The 2173-line log has no
  error/fatal/assert/hang match. Its SHA-256 is
  `8bc050722516565cc85d9cf51eab46ed1ca3843b13ea2db224081c2fce27f5b0`.
- Determinism is recorded in all four batch trace artifacts and both mutable
  A/B/A artifacts.

## Reproduction commands

```bash
GEMMA4_RANGE_DOWNLOAD=1 \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
pytest models/autoports/google_gemma_4_26b_a4b_it/tests -q

GEMMA4_RANGE_DOWNLOAD=1 GEMMA4_FUNCTIONAL_DECODER_CONTEXT=1 \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
pytest models/autoports/google_gemma_4_26b_a4b_it/tests/test_functional_decoder.py \
  -k advertised_context_traced_decode -q

GEMMA4_RANGE_DOWNLOAD=1 GEMMA4_PREFILL_CAPACITY_LENGTH=262144 \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
pytest models/autoports/google_gemma_4_26b_a4b_it/tests/test_functional_decoder.py \
  -k prefill_capacity_probe -q

GEMMA4_RANGE_DOWNLOAD=1 GEMMA4_PREFILL_CAPACITY_LENGTH=262143 \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
pytest models/autoports/google_gemma_4_26b_a4b_it/tests/test_functional_decoder.py \
  -k prefill_capacity_probe -q

GEMMA4_RANGE_DOWNLOAD=1 TT_METAL_WATCHER=10 \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
pytest models/autoports/google_gemma_4_26b_a4b_it/tests/test_functional_decoder.py \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_trace_mutable_buffers.py \
  -k 'real_weights_prefill_decode or traced_decode_batch_contract or trace_mutable' -q

GEMMA4_RANGE_DOWNLOAD=1 GEMMA4_FUNCTIONAL_DECODER_PERF=1 \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
python -m tracy -r -p --check-exit-code -o <raw-dir> -m pytest \
  <single-profiler-node-id> -q

python -m tracy.process_ops_logs -o <raw-dir> --force-legacy-device-logs
```
