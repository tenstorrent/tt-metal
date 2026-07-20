# DiffusionGemma live traced speed by context — QB2

> **HISTORICAL PERFORMANCE PROVENANCE (2026-07-10).** These rows used explicit sparse/tuned
> argmax trace settings and replayed a prompt-only frozen-prefix shape across output blocks.
> They are not current default-server TTFT, not growing-prefix multi-block throughput, and not
> evidence for an implicit vLLM launch. Use `plan.md` Part 0 and `doc/vllm_integration/README.md`
> for the current execution contract.

Date: 2026-07-10
Evidence: `live_context_sweep_results_20260710.json`

## Result

This is the real `tenstorrent/vllm` OpenAI `/v1/completions` path on bhqb QB2, not the vLLM-free serving-session benchmark. The optimized run explicitly set:

```text
DG_VLLM_TRACE=1
DG_VLLM_GUMBEL_MODE=argmax
DG_SPARSE_MOE=1
DG_DEDUP_ARGMAX=1
DG_SPARSE_MOE_TUNED=1
DG_TRACE_REGION_SIZE=10737418240
```

All requests used `--generation-config vllm`, `--max-num-seqs 1`, on-device sampling, temperature 0, ignored EOS, four real 256-token blocks, and the model-faithful 48 denoise steps per block.

### Primary warmed actual-prompt scaling at `max_model_len=4096`

Each target first ran one untimed same-shape warmup, followed by three timed
requests. Every timed request had zero first-use kernel compile markers. The
statistics below combine nine steady blocks per prompt target.

| logical/cache tokens | timed requests | steady samples | median prefill (s) | median block-0 TTFT (s) | steady mean ± σ (s) | p99 (s) | output tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 3 | 9 | 0.203 | 130.840 | 13.842 ± 0.023 | 13.875 | **18.495** |
| 256 | 3 | 9 | 2.969 | 135.619 | 14.012 ± 0.035 | 14.074 | **18.270** |
| 1024 | 3 | 9 | 15.070 | 148.964 | 14.569 ± 0.047 | 14.677 | **17.571** |
| 2048 | 3 | 9 | 31.686 | 166.759 | 15.309 ± 0.023 | 15.358 | **16.722** |

The priority handoff occurred after all four targets above completed. The
lower-priority 3072 warmed rerun was intentionally omitted; no larger context
work was resumed. The raw warmed file is labeled `interrupted` rather than
`running` and records the completed and omitted targets.

Every primary row committed 1024 tokens per request. Position progressions
were `32 → 288 → 544 → 800 → 1056`, `256 → 512 → 768 → 1024 → 1280`,
`1024 → 1280 → 1536 → 1792 → 2048`, and
`2048 → 2304 → 2560 → 2816 → 3072`.

### Earlier single-request context rows (provenance, non-primary)

These rows motivated the warmed rerun. They remain for provenance and for the
requested historical 18.247 tok/s comparison, but the primary table above
supersedes them where a warmed target exists. The 3072 row was not repeated
under the compile-marker guard.

| logical/cache tokens | prefill (s) | block-0 TTFT (s) | block latencies b0/b1/b2/b3 (s) | steady mean ± σ (s) | output tok/s |
|---:|---:|---:|---|---:|---:|
| 32 | 0.616 | 127.514 | 126.132 / 13.653 / 13.600 / 13.542 | 13.598 ± 0.045 | **18.826** |
| 256 | 2.994 | 134.789 | 129.895 / 14.040 / 13.997 / 14.053 | 14.030 ± 0.024 | **18.247** |
| 1024 | 15.203 | 148.899 | 131.403 / 14.629 / 14.631 / 16.172 | 15.144 ± 0.727 | **16.904** |
| 2048 | 38.825 | 178.405 | 137.222 / 16.905 / 16.631 / 16.871 | 16.802 ± 0.122 | **15.236** |
| 3072 | 57.576 | 187.835 | 129.484 / 17.277 / 17.251 / 17.231 | 17.253 ± 0.019 | **14.838** |

The 3072-token prefill required `--max-num-batched-tokens 4096`: the TT
scheduler disables chunked prefill, so its default 2048-token scheduling
budget left the request waiting without entering the model.

### DRAM at `max_model_len=4096`

Values are per chip, used/free GiB.

| prompt | after model build | trace resident after block 0 | after request release |
|---:|---:|---:|---:|
| 32 | 13.462 / 8.406 | 14.892 / 6.975 | 13.482 / 8.385 |
| 256 | 13.462 / 8.406 | 14.895 / 6.972 | 13.484 / 8.383 |
| 1024 | 13.462 / 8.406 | 14.899 / 6.968 | 13.488 / 8.379 |
| 2048 | 13.462 / 8.406 | 14.904 / 6.963 | 13.494 / 8.373 |
| 3072 | 13.462 / 8.406 | 14.897 / 6.970 | 13.252 / 8.615 |

## Allocation scaling versus actual prompt scaling

A fixed 32-token prompt remains flat as the allocated model context grows. The slowdown comes from the actual prefix read, not merely from `max_model_len`.

| max_model_len | model-build DRAM used/free (GiB) | 32-token steady block (s) | 32-token output tok/s |
|---:|---:|---:|---:|
| 4096 | 13.462 / 8.406 | 13.598 | 18.826 |
| 8192 | 13.719 / 8.148 | 13.579 | 18.852 |
| 16384 | 14.235 / 7.632 | 13.551 | 18.892 |
| 32768 | 15.266 / 6.601 | 13.581 | 18.850 |

Real longer prompts show the expected prefix-dependent cost:

| max_model_len | logical/cache tokens | prefill (s) | block-0 TTFT (s) | block latencies b0/b1/b2/b3 (s) | steady mean ± σ (s) | output tok/s | trace-resident DRAM used/free (GiB) |
|---:|---:|---:|---:|---|---:|---:|---:|
| 8192 | 6144 | 106.640 | 245.804 | 136.559 / 20.170 / 19.948 / 20.445 | 20.188 ± 0.203 | **12.681** | 15.158 / 6.709 |
| 16384 | 8192 | 139.407 | 280.148 | 138.169 / 21.401 / 21.660 / 21.562 | 21.541 ± 0.107 | **11.884** | 15.675 / 6.192 |
| 32768 | 16384 | 270.260 | 414.364 | 141.475 / 25.849 / 27.419 / 27.665 | 26.978 ± 0.804 | **9.489** | 16.710 / 5.157 |

The long-prompt position progressions were `6144 → 7168`, `8192 → 9216`, and `16384 → 17408`, in 256-token increments. Each request returned all 1024 committed tokens and released its traces.

## Trace proof

For every reported request:

- block 0 emitted one capture event containing exactly 48 distinct Metal trace IDs;
- block 0 executed those 48 IDs once;
- blocks 1–3 replayed the same IDs, for 144 steady execute calls and 192 total execute calls;
- there was no eager fallback and no recapture after block 0;
- prefill and commit remained outside the denoise trace;
- request completion emitted the trace-release marker before vLLM removed the row.

The configured 10 GiB trace region is a capacity limit, not the physically resident trace footprint. The observed trace-resident increase was about 1.41–1.44 GiB/chip, so the full 48-step path passed at `max_model_len=32768` with 5.157 GiB/chip still free. This live result supersedes the earlier estimate that compared free DRAM directly with the configured region size.

## Fit verdict

The bounded 8192, 16384, and 32768 allocated-context probes all passed. The largest allocated context tested was 32768 and the largest real prompt measured was 16384 tokens. This does not establish the absolute ceiling; no larger allocation or 256K run was forced.

## Rejected dense control

Before the explicit performance stack was added, the traced path measured 1.225 tok/s at 32 prompt tokens and 1.218 tok/s at 256. Those runs omitted `DG_SPARSE_MOE`, `DG_DEDUP_ARGMAX`, and `DG_SPARSE_MOE_TUNED`; they are retained only as `rejected_dense_control` and are not optimized results.
