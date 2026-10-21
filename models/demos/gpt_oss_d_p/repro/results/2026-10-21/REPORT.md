# GPT-OSS-120B: Mateusz guide reproduction and performance sweep

**Measurements complete: 29 prefill-only and 29 P/D cases.** Verified snapshot: 2026-10-21T05:17:28.134445+00:00.
This report uses measured performance and complete migration/stream evidence. It makes no numerical-accuracy claim: no golden generation, golden use or PCC checks were performed, as requested.

## Scope and configuration

- Two Blackhole Galaxies: `bh-glx-120-c02u02` prefill and `bh-glx-120-c06u14` decode; Slurm 140369.
- Full 36-layer GPT-OSS-120B. Prefill SP=4, TP=8; decode four 4×2 reload stages on one Galaxy. Original quantized checkpoint, BF4 prefill expert weights and BF8 KV.
- P/D and sweep: 131,072-token KV capacity, two allocated slots, one active request, native RDMA/DMK migration, 16 MiB staging slabs ×4, decode lockstep, chunk size 8 on decode, no prefill trace.
- One warmup and three measured requests per case. Output cap 32; EOS remains enabled. Actual input/output counts and finish reasons are saved per run. K=1024.
- Fresh nonce prompts, repeated `hello` body, then a short question. P/D may reuse 64 chat-template prefix tokens; that reuse is measured. This is synthetic performance traffic, not an answer-quality evaluation.
- Original standalone reproduction uses one slot / 65,536 capacity to match that guide section. The subsequent standalone sweep uses two slots / 131,072 capacity. These distinct configurations are labelled.

## Source revisions

| Repository | Base commit | Separate working branch |
| --- | --- | --- |
| tt-metal | fbbda775e3dd1a4dcf41ca2d3f9e649ff7be6436 | divanovic/mg-gptoss-pd-repro-20261021-metal |
| tt-blaze | d6d0b09f22f60a161d66fe4db90b9abed81ef82d | Unchanged; use the commit pin |
| tt-d-gen (folder tt-gen) | f5a46649a2d1e9a4ccae8b1e2504c63925b16473 | divanovic/mg-gptoss-pd-repro-20261021-dgen |
| Blaze nested Metal | 44d66500520fda9f2c7060c0f6b41ec48f7ab37e | Pinned dependency |

| Repository | Final commit | Publication |
| --- | --- | --- |
| tt-metal | 3441c663200ca87f7deafe7c48d39ba61c40d5c0 | Measurement harness and method guide, pushed |
| tt-d-gen | 5ca79e3076f193c092962d34a3b69731e39aff91 | Three isolated repairs, pushed |
| tt-blaze | d6d0b09f22f60a161d66fe4db90b9abed81ef82d | Unchanged source pin |

Metal measurement code 3441c663200ca87f7deafe7c48d39ba61c40d5c0 and D-gen repair 5ca79e3076f193c092962d34a3b69731e39aff91 are pushed and remote-verified. Blaze uses its unchanged pin. Raw evidence and measured sources are archived with SHA-256; the final results snapshot is recorded separately from tested code commits.

The minimal decode-lockstep prerequisite comes from upstream `f57f0ccb`; the unmodified guide pin lacks its required configuration field. Metal additions provide explicit no-golden performance harnesses and host timestamp/analysis tools. Model math and precision were not changed.

Measurement provenance: original guide P/D comparisons and sweep cases `64k-c64k` / `32k-c32k` used D-gen `7f6caf7d`; the next 26 P/D sweep cases used `4411a95b`; the final `64k-c1k` retry used admission fix `5ca79e3076f193c092962d34a3b69731e39aff91`. Standalone prefill does not use D-gen. The diagnostic change does not alter data movement; the final repair changes admission scheduling. These are separate source cohorts, not a uniform-binary stability claim.

## Reference versus observed

| Metric | Guide | Measured | Assessment |
| --- | --- | --- | --- |
| Standalone prefill | 25.1K tok/s | 23.625K tok/s (median of 3) | Lower than guide by about 5.9% |
| P/D prefill | 21.8K tok/s | 21961.2 tok/s (mean of 3) | Close: about 0.7% higher |
| TTFT | 3.14s | 3.145s (mean of 3) | Close: about 0.2% higher |
| Decode | ~330 t/s/user | 257.5 t/s/user (mean of 3) | Not reproduced: about 22% lower |
| Successful transfers | 72 / zero failures | [72, 72, 72]; 0 failures in audited windows | Reproduced count; every expected range audited |
| Transfer sum | 1.8–1.9s | 1.819s (mean of 3) | Within stated range |
| Migration span | 2.96s | 2.968s (mean of 3) | Close: about 0.3% higher |
| Exposed final migration tail | 5–9ms | 0.024s (mean of 3) | Not reproduced: about 24ms, outside 5–9ms range |
| P/D vs standalone rate reduction | ~13% | 7.04% (different harness/capacity; see report) | 7.04% observed; not an isolated migration-overhead comparison |

TTFT, P/D prefill throughput, transfer count, transfer sum and migration span are close to the guide at 64K input / 32K chunk. The measured standalone throughput, decode throughput and 32K-chunk exposed tail differ. The guide does not publish a tolerance; these are observations, not a statistical equivalence test. The supplied `async_bench.py` does not compute decode TSU, so the method behind the guide’s ~330 is not established by that client.

The original standalone and P/D runs use different slot/capacity settings and separate harnesses. Their throughput difference must not be attributed entirely to migration contention. The sweep narrows that configuration gap; it still compares synchronized standalone calls with the serving path and measured prefix reuse.

### 64K input: chunk comparison

| Chunk | Guide prefill tok/s | Measured prefill tok/s | TTFT mean ± sample SD (s) | Final KV tail (ms) | Decode t/s/user |
| --- | --- | --- | --- | --- | --- |
| 8K | 20300 | 20423.2 | 3.352 ± 0.008 | 6.06 | 257.4 |
| 16K | 21600 | 21659.1 | 3.173 ± 0.010 | 12.71 | 257.6 |
| 32K | 21800 | 21961.2 | 3.145 ± 0.004 | 23.97 | 257.5 |
| 64K | 20000 | 21123.4 | 3.289 ± 0.005 | 49.05 | 257.5 |

All means use only the three post-warmup runs. Sample variance (N−1), individual requests, prefix reuse, processed counts and migration ranges are available in the live per-run evidence. Three samples describe this experiment; they do not establish long-term variance.

Download [every completed sweep request](SWEEP_RUNS.csv) or [case means and sample variances](SWEEP_STATISTICS.csv). These exports use the same snapshot as this report; unfinished cases are omitted. Blank cells mean N/A.

## Build, loading and server setup

| Observed milestone interval | Elapsed | Scope |
| --- | --- | --- |
| Initial build start → first complete P/D stream | 64.59 min | Observed elapsed experiment time, including build, cold cache population, parallel model startup, orchestration and setup fixes; not an isolated server benchmark. |
| Decoder launch → all 41 images / owners ready | 44.56 min | Same decoder host launch/readiness timestamps; cold start. Prefill preparation overlapped it. |
| Decoder ready → first complete P/D stream | 3.72 min | Approximate cross-host UTC elapsed time: manager/worker attachment, metrics-port correction and first warmup. Not pure service initialization cost. |
| First complete stream → accepted four-run baseline | 3.83 min | Analysis-parser diagnosis and fresh warmup + three measured requests. Model owners remained loaded; the first failed analysis is preserved. |

Global timeline uses recorded UTC timestamps. Request durations use monotonic clocks; concurrent phases are not added.

| Component / phase | Duration | Scope |
| --- | --- | --- |
| prefill Metal native compilation | 332.7s | Wall-clock duration; other components may run concurrently. |
| prefill environment / import verification after compilation | 260.4s | Wall-clock duration; other components may run concurrently. |
| decode Metal native compilation | 334.9s | Wall-clock duration; other components may run concurrently. |
| decode environment / import verification after compilation | 285.9s | Wall-clock duration; other components may run concurrently. |
| Native engine / lockstep build and host checks | 52.5s | Wall-clock duration; other components may run concurrently. |
| Native transport build | 102.2s | Wall-clock duration; other components may run concurrently. |
| Full36 decoder: launch to all owners ready | 2673.9s | One cold launch, 41 reload images across four concurrent stages; not the sum of all image times. |
| 32K guide prefill launch to ready | 194.2s | Populated weight cache; model construction, compile/warmup and KV table export included. |
| repro-c16384 launch to ready | 194.2s | Populated weight cache; model construction, compile/warmup and KV table export included. |
| repro-c65536 launch to ready | 257.9s | Populated weight cache; model construction, compile/warmup and KV table export included. |
| repro-c8192 launch to ready | 163.5s | Populated weight cache; model construction, compile/warmup and KV table export included. |
| sweep-pd-c1024 launch to ready | 128.6s | Populated weight cache; model construction, compile/warmup and KV table export included. |
| sweep-pd-c16384 launch to ready | 133.8s | Populated weight cache; model construction, compile/warmup and KV table export included. |
| sweep-pd-c32768 launch to ready | 115.9s | Populated weight cache; model construction, compile/warmup and KV table export included. |
| sweep-pd-c4096 launch to ready | 130.9s | Populated weight cache; model construction, compile/warmup and KV table export included. |
| sweep-pd-c8192 launch to ready | 129.1s | Populated weight cache; model construction, compile/warmup and KV table export included. |
| cold standalone: imports_finished | 35.1s | Interval from process_started; separate from per-case request warmup. |
| cold standalone: mesh_opened | 21.6s | Interval from imports_finished; separate from per-case request warmup. |
| cold standalone: weights_started | 1.4s | Interval from mesh_opened; separate from per-case request warmup. |
| cold standalone: weights_loaded | 268.9s | Interval from weights_started; separate from per-case request warmup. |
| cold standalone: model_built | 978.6s | Interval from weights_loaded; separate from per-case request warmup. |
| cold standalone: compile_finished | 135.6s | Interval from model_built; separate from per-case request warmup. |
| sweep-prefill-c1024: imports_finished | 16.3s | Interval from process_started; separate from per-case request warmup. |
| sweep-prefill-c1024: mesh_opened | 4.4s | Interval from imports_finished; separate from per-case request warmup. |
| sweep-prefill-c1024: model_build_started | 1.4s | Interval from mesh_opened; separate from per-case request warmup. |
| sweep-prefill-c1024: model_built | 64.0s | Interval from model_build_started; separate from per-case request warmup. |
| sweep-prefill-c1024: compile_finished | 81.0s | Interval from model_built; separate from per-case request warmup. |
| sweep-prefill-c16384: imports_finished | 23.4s | Interval from process_started; separate from per-case request warmup. |
| sweep-prefill-c16384: mesh_opened | 4.6s | Interval from imports_finished; separate from per-case request warmup. |
| sweep-prefill-c16384: model_build_started | 1.5s | Interval from mesh_opened; separate from per-case request warmup. |
| sweep-prefill-c16384: model_built | 58.6s | Interval from model_build_started; separate from per-case request warmup. |
| sweep-prefill-c16384: compile_finished | 37.9s | Interval from model_built; separate from per-case request warmup. |
| sweep-prefill-c32768: imports_finished | 20.7s | Interval from process_started; separate from per-case request warmup. |
| sweep-prefill-c32768: mesh_opened | 6.2s | Interval from imports_finished; separate from per-case request warmup. |
| sweep-prefill-c32768: model_build_started | 1.4s | Interval from mesh_opened; separate from per-case request warmup. |
| sweep-prefill-c32768: model_built | 58.9s | Interval from model_build_started; separate from per-case request warmup. |
| sweep-prefill-c32768: compile_finished | 42.7s | Interval from model_built; separate from per-case request warmup. |
| sweep-prefill-c4096: imports_finished | 17.2s | Interval from process_started; separate from per-case request warmup. |
| sweep-prefill-c4096: mesh_opened | 4.5s | Interval from imports_finished; separate from per-case request warmup. |
| sweep-prefill-c4096: model_build_started | 1.4s | Interval from mesh_opened; separate from per-case request warmup. |
| sweep-prefill-c4096: model_built | 59.0s | Interval from model_build_started; separate from per-case request warmup. |
| sweep-prefill-c4096: compile_finished | 78.1s | Interval from model_built; separate from per-case request warmup. |
| sweep-prefill-c65536: imports_finished | 19.4s | Interval from process_started; separate from per-case request warmup. |
| sweep-prefill-c65536: mesh_opened | 4.4s | Interval from imports_finished; separate from per-case request warmup. |
| sweep-prefill-c65536: model_build_started | 1.4s | Interval from mesh_opened; separate from per-case request warmup. |
| sweep-prefill-c65536: model_built | 59.7s | Interval from model_build_started; separate from per-case request warmup. |
| sweep-prefill-c65536: compile_finished | 39.9s | Interval from model_built; separate from per-case request warmup. |
| sweep-prefill-c8192: imports_finished | 18.4s | Interval from process_started; separate from per-case request warmup. |
| sweep-prefill-c8192: mesh_opened | 4.5s | Interval from imports_finished; separate from per-case request warmup. |
| sweep-prefill-c8192: model_build_started | 1.4s | Interval from mesh_opened; separate from per-case request warmup. |
| sweep-prefill-c8192: model_built | 59.2s | Interval from model_build_started; separate from per-case request warmup. |
| sweep-prefill-c8192: compile_finished | 45.3s | Interval from model_built; separate from per-case request warmup. |

### Concurrent decoder stages

| Stage | Images | Load + graph build | Weight staging | Configure | Readback | Phase sum |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 11 | 1946.9s | 365.5s | 210.7s | 23.5s | 2546.6s |
| 1 | 10 | 1791.7s | 405.8s | 224.4s | 22.9s | 2444.8s |
| 2 | 10 | 1795.9s | 340.9s | 282.4s | 25.5s | 2444.7s |
| 3 | 10 | 1771.4s | 325.7s | 275.8s | 27.5s | 2400.4s |

Do not sum the four stages: they execute concurrently. Reported phase values are rounded by the reload logger. Launcher, mesh, table installation and barrier time are not all represented by image-phase sums. Retaining the decoder avoids repeating its cold setup between chunk groups.

## Sweep: prefill only

29 valid input/chunk combinations. No migration, TTFT or decode exists in this mode; those fields are N/A. Larger chunks than input are excluded.

| ISL | Chunk | Status | Warmup s | Prefill s | Prefill tok/s | TTFT ms | KV MiB | Transfer sum s | Transfer active s | Transfer span s | Overlap s | Non-overlap prefill s | Non-overlap transfer s | Initial prefill s | Final KV tail ms | Decode s | Decode t/s/user | Total s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1K | 1K | DONE | 0.308 | 0.307 | 3335.326 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.307 |
| 2K | 1K | DONE | 0.614 | 0.610 | 3354.968 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.610 |
| 4K | 1K | DONE | 1.217 | 1.221 | 3354.956 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 1.221 |
| 4K | 4K | DONE | 0.336 | 0.312 | 13117.251 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.312 |
| 5K | 1K | DONE | 1.538 | 1.541 | 3322.679 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 1.541 |
| 5K | 4K | DONE | 0.619 | 0.617 | 8302.962 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.617 |
| 8K | 1K | DONE | 2.474 | 2.457 | 3333.813 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 2.457 |
| 8K | 4K | DONE | 0.618 | 0.623 | 13160.919 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.623 |
| 8K | 8K | DONE | 0.325 | 0.325 | 25233.176 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.325 |
| 16K | 1K | DONE | 4.974 | 4.899 | 3344.671 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 4.899 |
| 16K | 4K | DONE | 1.223 | 1.232 | 13293.651 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 1.232 |
| 16K | 8K | DONE | 0.655 | 0.655 | 25025.984 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.655 |
| 16K | 16K | DONE | 0.645 | 0.638 | 25699.715 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.638 |
| 32K | 1K | DONE | 9.802 | 9.793 | 3345.966 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 9.793 |
| 32K | 4K | DONE | 2.468 | 2.459 | 13323.997 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 2.459 |
| 32K | 8K | DONE | 1.352 | 1.365 | 23999.227 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 1.365 |
| 32K | 16K | DONE | 1.320 | 1.321 | 24814.509 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 1.321 |
| 32K | 32K | DONE | 1.306 | 1.309 | 25031.751 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 1.309 |
| 55K | 1K | DONE | 16.779 | 16.743 | 3363.830 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 16.743 |
| 55K | 4K | DONE | 4.287 | 4.297 | 13108.174 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 4.297 |
| 55K | 8K | DONE | 2.495 | 2.478 | 22726.645 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 2.478 |
| 55K | 16K | DONE | 2.744 | 2.744 | 20522.228 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 2.744 |
| 55K | 32K | DONE | 2.707 | 2.705 | 20820.142 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 2.705 |
| 64K | 1K | DONE | 19.602 | 19.549 | 3352.474 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 19.549 |
| 64K | 4K | DONE | 4.931 | 4.961 | 13211.648 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 4.961 |
| 64K | 8K | DONE | 2.871 | 2.869 | 22844.800 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 2.869 |
| 64K | 16K | DONE | 2.811 | 2.814 | 23292.529 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 2.814 |
| 64K | 32K | DONE | 2.774 | 2.774 | 23621.656 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 2.774 |
| 64K | 64K | DONE | 2.899 | 2.894 | 22643.988 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 2.894 |

## Sweep: prefill + migration + decode

| ISL | Chunk | Status | Warmup s | Prefill s | Prefill tok/s | TTFT ms | KV MiB | Transfer sum s | Transfer active s | Transfer span s | Overlap s | Non-overlap prefill s | Non-overlap transfer s | Initial prefill s | Final KV tail ms | Decode s | Decode t/s/user | Total s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1K | 1K | DONE | 0.609 | 0.379 | 2536.841 | 449.158 | 34.664 | 0.071 | 0.071 | 0.365 | 0.070 | 0.309 | 0.002 | 0.015 | 1.774 | 0.117 | 264.668 | 0.556 |
| 2K | 1K | DONE | 0.939 | 0.749 | 2650.389 | 822.139 | 72.914 | 0.152 | 0.152 | 0.738 | 0.148 | 0.600 | 0.004 | 0.014 | 2.771 | 0.117 | 264.545 | 0.929 |
| 4K | 1K | DONE | 1.685 | 1.484 | 2717.011 | 1557.690 | 149.414 | 0.263 | 0.263 | 1.471 | 0.258 | 1.222 | 0.005 | 0.014 | 1.330 | 0.117 | 264.585 | 1.664 |
| 4K | 4K | DONE | 0.635 | 0.391 | 10308.119 | 467.295 | 149.414 | 0.130 | 0.130 | 0.381 | 0.126 | 0.265 | 0.003 | 0.014 | 3.345 | 0.117 | 264.563 | 0.574 |
| 5K | 1K | DONE | 2.036 | 1.871 | 2702.325 | 1947.607 | 187.664 | 0.362 | 0.362 | 1.858 | 0.356 | 1.508 | 0.006 | 0.014 | 1.422 | 0.117 | 264.566 | 2.054 |
| 5K | 4K | DONE | 1.040 | 0.763 | 6630.985 | 839.116 | 187.664 | 0.203 | 0.203 | 0.749 | 0.201 | 0.561 | 0.002 | 0.014 | 1.091 | 0.117 | 264.452 | 0.946 |
| 8K | 1K | DONE | 3.233 | 3.024 | 2688.255 | 3103.362 | 302.414 | 0.566 | 0.566 | 3.010 | 0.555 | 2.458 | 0.011 | 0.015 | 1.498 | 0.117 | 264.503 | 3.210 |
| 8K | 4K | DONE | 0.944 | 0.771 | 10550.881 | 853.004 | 302.414 | 0.278 | 0.278 | 0.760 | 0.273 | 0.496 | 0.005 | 0.014 | 3.470 | 0.117 | 264.565 | 0.960 |
| 8K | 8K | DONE | 0.599 | 0.418 | 19631.088 | 505.499 | 302.414 | 0.236 | 0.236 | 0.411 | 0.229 | 0.189 | 0.007 | 0.015 | 7.024 | 0.117 | 264.538 | 0.612 |
| 16K | 1K | DONE | 6.235 | 5.995 | 2722.176 | 6084.808 | 608.414 | 1.119 | 1.119 | 5.983 | 1.098 | 4.873 | 0.021 | 0.014 | 1.833 | 0.118 | 263.780 | 6.192 |
| 16K | 4K | DONE | 1.746 | 1.500 | 10880.262 | 1587.349 | 608.414 | 0.543 | 0.543 | 1.490 | 0.536 | 0.959 | 0.008 | 0.013 | 2.968 | 0.118 | 263.724 | 1.694 |
| 16K | 8K | DONE | 0.985 | 0.789 | 20696.632 | 883.855 | 608.414 | 0.467 | 0.467 | 0.780 | 0.459 | 0.328 | 0.008 | 0.015 | 5.691 | 0.117 | 263.852 | 0.991 |
| 16K | 16K | DONE | 0.896 | 0.686 | 23779.956 | 786.786 | 608.414 | 0.464 | 0.464 | 0.676 | 0.452 | 0.235 | 0.013 | 0.023 | 12.716 | 0.118 | 263.748 | 0.894 |
| 32K | 1K | DONE | 12.196 | 12.050 | 2713.948 | 12155.742 | 1220.414 | 2.278 | 2.278 | 12.037 | 2.233 | 9.761 | 0.045 | 0.014 | 1.344 | 0.118 | 261.915 | 12.264 |
| 32K | 4K | DONE | 3.296 | 2.978 | 10983.663 | 3086.022 | 1220.414 | 1.042 | 1.042 | 2.966 | 1.029 | 1.939 | 0.013 | 0.015 | 3.153 | 0.118 | 261.927 | 3.194 |
| 32K | 8K | DONE | 1.819 | 1.551 | 21080.671 | 1659.977 | 1220.414 | 0.936 | 0.935 | 1.543 | 0.923 | 0.623 | 0.012 | 0.014 | 6.420 | 0.118 | 261.916 | 1.768 |
| 32K | 16K | DONE | 1.653 | 1.422 | 22994.087 | 1536.416 | 1220.414 | 0.893 | 0.893 | 1.412 | 0.879 | 0.541 | 0.014 | 0.022 | 11.942 | 0.118 | 261.935 | 1.644 |
| 32K | 32K | DONE | 1.659 | 1.410 | 23201.548 | 1538.470 | 1220.414 | 0.909 | 0.909 | 1.394 | 0.884 | 0.526 | 0.025 | 0.040 | 24.898 | 0.118 | 261.835 | 1.646 |
| 55K | 1K | DONE | 20.905 | 20.465 | 2749.107 | 20595.555 | 2100.164 | 3.815 | 3.815 | 20.451 | 3.742 | 16.624 | 0.073 | 0.015 | 1.403 | 0.119 | 261.587 | 20.704 |
| 55K | 4K | DONE | 5.622 | 5.351 | 10512.891 | 5483.451 | 2100.164 | 1.794 | 1.794 | 5.341 | 1.769 | 3.559 | 0.025 | 0.014 | 2.867 | 0.119 | 261.382 | 5.592 |
| 55K | 8K | DONE | 3.017 | 2.829 | 19884.682 | 2961.806 | 2100.164 | 1.730 | 1.728 | 2.820 | 1.714 | 1.107 | 0.015 | 0.015 | 5.688 | 0.119 | 261.492 | 3.070 |
| 55K | 16K | DONE | 3.204 | 2.964 | 18977.170 | 3099.315 | 2100.164 | 1.547 | 1.547 | 2.946 | 1.532 | 1.422 | 0.015 | 0.023 | 5.385 | 0.119 | 261.465 | 3.208 |
| 55K | 32K | DONE | 3.177 | 2.907 | 19350.855 | 3052.146 | 2100.164 | 1.533 | 1.533 | 2.888 | 1.513 | 1.392 | 0.020 | 0.036 | 17.035 | 0.119 | 261.530 | 3.160 |
| 64K | 1K | DONE | 23.994 | 24.113 | 2715.171 | 24256.486 | 2444.414 | 4.641 | 4.641 | 24.101 | 4.555 | 19.454 | 0.087 | 0.014 | 1.717 | 0.120 | 257.569 | 24.366 |
| 64K | 4K | DONE | 6.443 | 6.076 | 10776.030 | 6217.264 | 2444.414 | 2.125 | 2.125 | 6.065 | 2.098 | 3.954 | 0.027 | 0.014 | 3.135 | 0.120 | 257.402 | 6.328 |
| 64K | 8K | DONE | 3.611 | 3.265 | 20058.034 | 3410.337 | 2444.414 | 1.918 | 1.918 | 3.255 | 1.900 | 1.352 | 0.019 | 0.016 | 5.959 | 0.120 | 257.466 | 3.520 |
| 64K | 16K | DONE | 3.280 | 3.025 | 21641.712 | 3177.330 | 2444.414 | 1.889 | 1.889 | 3.014 | 1.868 | 1.149 | 0.021 | 0.024 | 12.795 | 0.120 | 257.438 | 3.288 |
| 64K | 32K | DONE | 3.256 | 2.983 | 21945.424 | 3149.391 | 2444.414 | 1.893 | 1.893 | 2.972 | 1.863 | 1.117 | 0.030 | 0.038 | 26.503 | 0.120 | 257.298 | 3.260 |
| 64K | 64K | DONE | 3.389 | 3.101 | 21113.960 | 3286.165 | 2444.414 | 1.773 | 1.773 | 3.084 | 1.723 | 1.377 | 0.049 | 0.066 | 49.233 | 0.120 | 257.527 | 3.396 |

## Lowest observed mean latency by input length

Each selection is shown only after all valid chunk sizes for that input and mode have finished. These are the lowest means in this three-measurement experiment, not a statistical guarantee or a general tuning recommendation. Prefill-only selects by prefill time; P/D selects by TTFT.

| ISL | Prefill chunk | Prefill mean (s) | Sample SD (s) | P/D chunk | TTFT mean (ms) | Sample SD (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| 1K | 1K | 0.307 | 0.001947 | 1K | 449.158 | 7.717 |
| 2K | 1K | 0.610 | 0.001709 | 1K | 822.139 | 6.333 |
| 4K | 4K | 0.312 | 0.002843 | 4K | 467.295 | 12.209 |
| 5K | 4K | 0.617 | 0.000340 | 4K | 839.116 | 10.734 |
| 8K | 8K | 0.325 | 0.000655 | 8K | 505.499 | 55.342 |
| 16K | 16K | 0.638 | 0.001492 | 16K | 786.786 | 11.627 |
| 32K | 32K | 1.309 | 0.003495 | 16K | 1536.416 | 2.871 |
| 55K | 8K | 2.478 | 0.015559 | 8K | 2961.806 | 29.764 |
| 64K | 32K | 2.774 | 0.000219 | 32K | 3149.391 | 5.747 |

## Measurement definitions and limitations

- **Warmup:** one request after model setup and internal compilation. The warmup column is full request time (prefill-only wall time in that table; end-to-end request time in P/D). Cold model setup appears separately above.
- **Prefill:** standalone measures input conversion, chunk calls and final device synchronization. P/D measures host-observed compute-call intervals through existing asynchronous completion handling; no extra device synchronization was introduced. These are serving/wall durations, not isolated kernel cycle counts.
- **Rates:** standalone uses all real input tokens; padded/processed counts are saved separately. P/D uses actual prefilled tokens, excluding measured prefix reuse. P/D computed tokens including padding are derived from observed chunk calls × configured fixed width; the raw `prefill_tokens_processed` field is the engine’s logical prompt counter and does not include that tail padding. The guide’s input-token/service-time convention can be reconstructed from saved `input_tokens` and `prefill_engine_busy_s`.
- **Migration:** source-manager command to successful completion, including control-plane scheduling/retries. Payload is the sender’s wire-byte counter, not receiver drain bytes or link headers. Same-ID admission/local-child log repeats are recorded, not counted as additional transfers. Every expected layer/range must complete once, and fault lines invalidate the run.
- **Overlap:** intersection of measured prefill-active and transfer-active interval unions. Non-overlap is the remaining union duration on each side. The first prefill-only segment and final transfer tail are additional boundary measurements, not disjoint extra costs to add again.
- **TTFT:** local HTTP POST start to first nonempty content or reasoning delta, using immediate stream chunks. SSE event count is not token count.
- **Decode:** native inter-token interval count / corresponding interval seconds. With 32 outputs, this is normally 31 intervals. These are host steady-clock timestamps immediately after the decode committer publishes (`engine/src/control/telemetry_source.cpp:44`), not device-only cycle timing or terminal print speed. The final prompt block is computed on decode; migration ends at the reusable block boundary.
- **Time alignment:** client, prefill instrumentation and source-manager timestamps are collected on the same prefill host. Decoder throughput uses native duration counters, not cross-host timestamp subtraction.
- **Scope:** one active request and short output cap. No claim about long generation, multi-user saturation, answer correctness, PCC or all future runs. Minimal timestamp file appends between prefill chunks remain part of the instrumented serving experiment.

## Obstacles and fixes

- Published d-gen revision lacks decode_lockstep support required by its supplied worker config.
- Recorded standalone runs used a repeated shorter prompt and explicitly skipped PCC. New measurements will use prompt tokens without any golden data.
- 55K/32K first attempt stopped on a misleading empty-SP-queue warning. All 1,012,032 requested table chunks and 2,202,181,632 wire bytes audited; a diagnostic-only manager correction is being retried. Original failure preserved under evidence/sweep/failed/.
- Resolved cache-path mismatch: standalone and common runner use different directory layouts; runner now aliases the same populated cache. Failed attempt preserved.
- Remote login-to-compute SSH authentication failed. Attachments now use the authenticated laptop ProxyJump route; running model owners remain detached on the Galaxies.
- Resolved decoder metrics port conflict with model-rank telemetry; worker metrics moved to18089 without model restart.
- Final 64K-input/1K-chunk attempt timed out in migration admission after its warmup and first two measured requests passed. Original evidence preserved. The admission fix passed 80 host tests and a fresh warmup plus three measured requests; all 58 accepted cases passed the saved-evidence audit. See FINAL-CASE-INVESTIGATION.md.
- Measurement parser initially rejected native admission/local-child duplicate command lines; saved logs and source established their meaning. The first failed analysis is preserved, and an entirely fresh warmup + three measured requests produced the accepted 32K baseline. Ten device-free parser tests now pass.
- The supplied guide assigns `TT_GEN_NATIVE_BUILD` to the executable and later appends `/kv_manager`; the launched command uses the actual executable once. Built Python bindings and dependency-library locations also differ from the guide examples.
- Setup repairs and harness additions are isolated on the Metal and D-gen task branches; Blaze remains unchanged; original branches and failed attempts are preserved. No hardware reset has been needed.

- At 55K input / 32K chunk, an expected empty intersection with the final SP shard emitted 576 misleading stale-receiver warnings. Saved-table audit covered all 1,012,032 requested chunks and exactly matched the 2,202,181,632 wire payload bytes. The manager now logs that per-device condition at DEBUG; the aggregate empty-transfer error remains. The initial failed case is preserved, and a fresh warmup + three measured requests passed after the diagnostic-only rebuild. Both model owners survived that repair.

The final 64K/1K attempt exposed a separate staging-buffer admission deadlock on its third measured request. A deterministic host regression reproduced it; a narrow exception for already-admitted local children passed all 80 host control-plane tests. The failed attempt and its first three successful requests remain preserved. See [the investigation](FINAL-CASE-INVESTIGATION.md). The final case passed a fresh warmup plus three measurements at 05:04:41 UTC (TTFT mean 24.256486s), followed by a complete 58-case audit with zero errors; earlier completed cases were not replaced.

## Exact execution and evidence

Shared root: `/data/divanovic/gptoss-mg-repro-20261021`. Access uses `ssh -J slurm-login-01.exabox.tenstorrent.com divanovic@HOST`. Active services and device jobs must not be duplicated.
```text
Reproduction owner launch: runtime/start-decode-owner.sh
Standalone baseline: runtime/start-standalone-baseline.sh
Common prefill: runtime/start-prefill-owner-v2.sh
Fresh chunk-group configs: runtime/prepare-chunk-group.py
Services: runtime/start-service-v3.py --run GROUP SERVICE
Measurement: tt-metal/models/demos/gpt_oss_d_p/repro/run_pd_case.py
Standalone sweep: tt-metal/models/demos/gpt_oss_d_p/repro/prefill-sweep-group.py
Original results: evidence/reproduction/pd-*/case.json
Sweep results: evidence/sweep/{prefill,pd}/*/case.json
Every P/D invocation: command-N.json; every request: prompt, SSE events, before/after counters and log windows
```
The runtime files contain the exact environment, addresses, cache paths, deadlines and commands actually executed.
The serial sweep coordinator has exited. It preserved completed cases and retained the loaded decoder between chunk groups.
Source and command archive: `/data/divanovic/gptoss-mg-repro-20261021/archives/gptoss-mg-repro-20261021-source-evidence.tar.gz`.
Archive SHA-256: `edeff1737e8f3201e43b53a05f23823ad84b5e1c42aadd4bd988d8275a24e552`. The archive preserves the measured-source and raw-evidence snapshot, including failures; its interim workflow-status documents are historical. This final report and the live completion status are published separately.
Cleanup verified: Both task hosts released: Slurm 140369 absent from active queue after scancel; task processes and device holders verified empty before release. Slurm accounting is disabled.

Additional historical records in Mateusz’s private workspace were blocked by automatic approval review as outside the explicitly shared inputs. This report relies on the supplied guide, checked-in source and our own measurements; that restriction does not block this sweep.
