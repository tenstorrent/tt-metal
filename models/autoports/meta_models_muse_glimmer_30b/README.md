# Muse-Glimmer-30B on Blackhole

This port serves `meta-models/Muse-Glimmer-30B` as an OpenAI-compatible vLLM
endpoint for agentic coding. One `tt-model` container image supports tensor
parallel widths 1, 2, and 4 through named profiles.

## Release contract

| item | value |
|---|---|
| weights | `meta-models/Muse-Glimmer-30B` |
| immutable weight revision | `f84ecc3a0ea984a4c04542a84269e3d065350a6e` |
| package | `tt-hous/muse-glimmer-30b` |
| source branch | `hous/muse-glimmer-30b-p150-p150x2` |
| architecture | Blackhole |
| API | OpenAI Chat Completions on the port selected by `tt-model serve` |
| context limit | 131,072 tokens |
| tool parser | `muse_glimmer` (ATEM) |
| reasoning parser | `muse_glimmer` (channel-aware) |

## Hardware profiles

| profile | devices | tensor parallelism | max sequences | qualification |
|---|---:|---:|---:|---|
| `p150` | 1 | 1 | 1 | hardware-qualified: 24/24 tool-call samples and full-context OSL-512 control pass |
| `p150x2` | 2 | 2 | 1 | hardware-qualified: 24/24 tool-call samples and full-context OSL-512 control pass |
| `p150x4` | 4 | 4 | 32 | previously validated on the four-chip QB2/P300x2 configuration |

The lower-device evidence is committed as the
[`p150` tool-call sweep](doc/serving_perf/benchmarks/p150_tool_call_latency.json),
[`p150x2` tool-call sweep](doc/serving_perf/benchmarks/p150x2_tool_call_latency.json),
[`p150` fixed-OSL control](doc/serving_perf/benchmarks/p150_fixed_osl_latency.json),
and [`p150x2` fixed-OSL control](doc/serving_perf/benchmarks/p150x2_fixed_osl_latency.json).

P150 uses the optimized single-chip decoder and shares one full-context RoPE
cache across all layers. P150x2 uses the multichip decoder with a topology-
specific, exactly divisible MLP grid. P150x4 remains the default profile and
preserves the original release behavior.

## Serve

```bash
tt-model pull tt-hous/muse-glimmer-30b
tt-model serve tt-hous/muse-glimmer-30b --profile p150
```

For two chips, change the final profile to `p150x2`. The package enables tool
choice and reasoning parsing automatically. Do not set `VLLM_PLUGINS`; vLLM
treats it as an allowlist and can silently suppress the Tenstorrent platform
plugin.

Once the server reports ready, verify the API and run the bounded coding task:

```bash
curl -fsS http://127.0.0.1:20000/health
python models/autoports/meta_models_muse_glimmer_30b/tests/tool_calling_harness.py \
  --base-url http://127.0.0.1:20000 \
  --model meta-models/Muse-Glimmer-30B
```

`tt-model serve` chooses port 20000 and walks upward if it is occupied; use the
port printed by the command.

## Performance evidence

The live OpenAI API sweep uses a system-plus-user agentic-coding prompt with
real source context and requires the exact structured `record_latency_probe`
call on every response. Each table cell is the median of three measured calls
after a per-shape warmup; OSL is the model's actual completion length. Every
latency/performance table uses the standard column order: ISL, OSL,
concurrency, decode tok/s/u, TTFT, E2EL.

### P150 tool-calling API

| ISL | OSL | concurrency | decode tok/s/u | TTFT | E2EL |
|---:|---:|---:|---:|---:|---:|
| 512 | 135 | 1 | 10.50 | 613.7 ms | 13.37 s |
| 1,024 | 189 | 1 | 10.35 | 731.9 ms | 18.89 s |
| 4,096 | 171 | 1 | 10.12 | 1.58 s | 18.38 s |
| 8,192 | 168 | 1 | 9.97 | 2.84 s | 19.59 s |
| 16,384 | 117 | 1 | 9.74 | 5.77 s | 17.67 s |
| 32,768 | 167 | 1 | 9.11 | 12.21 s | 30.43 s |
| 65,536 | 168 | 1 | 8.18 | 27.75 s | 48.17 s |
| 130,560 | 162 | 1 | 6.80 | 69.59 s | 93.26 s |

### P150x2 tool-calling API

| ISL | OSL | concurrency | decode tok/s/u | TTFT | E2EL |
|---:|---:|---:|---:|---:|---:|
| 512 | 135 | 1 | 28.19 | 283.8 ms | 5.04 s |
| 1,024 | 180 | 1 | 27.68 | 364.2 ms | 6.83 s |
| 4,096 | 278 | 1 | 83.26 | 7.80 s | 11.13 s |
| 8,192 | 147 | 1 | 26.98 | 1.70 s | 7.11 s |
| 16,384 | 128 | 1 | 26.37 | 3.54 s | 8.36 s |
| 32,768 | 168 | 1 | 25.00 | 7.49 s | 14.17 s |
| 65,536 | 110 | 1 | 23.04 | 16.99 s | 21.72 s |
| 130,560 | 135 | 1 | 19.54 | 42.03 s | 48.89 s |

At P150x2 ISL 4096, the reasoning/tool parser buffers early output until late
in the completion. Client-visible TTFT therefore absorbs decode work and the
API-derived decode rate is correspondingly inflated; the fixed-OSL control
below is the authority for raw device-path decode throughput. All 48 measured
tool-calling samples across P150 and P150x2 passed the structured-call gate.

### Fixed-OSL device control

Every row below generated exactly 512 tokens at concurrency 1. These runs
bypass the API parser and provide comparable underlying device-path latency.

#### P150 fixed-OSL control

| ISL | OSL | concurrency | decode tok/s/u | TTFT | E2EL |
|---:|---:|---:|---:|---:|---:|
| 128 | 512 | 1 | 10.23 | 177.3 ms | 50.12 s |
| 1,024 | 512 | 1 | 10.00 | 318.8 ms | 51.42 s |
| 4,096 | 512 | 1 | 9.66 | 1.13 s | 54.03 s |
| 8,192 | 512 | 1 | 9.42 | 2.29 s | 56.54 s |
| 16,384 | 512 | 1 | 8.97 | 5.01 s | 61.96 s |
| 32,768 | 512 | 1 | 8.23 | 10.99 s | 73.10 s |
| 65,536 | 512 | 1 | 7.07 | 25.33 s | 97.62 s |
| 130,560 | 512 | 1 | 5.55 | 63.56 s | 155.62 s |

#### P150x2 fixed-OSL control

| ISL | OSL | concurrency | decode tok/s/u | TTFT | E2EL |
|---:|---:|---:|---:|---:|---:|
| 128 | 512 | 1 | 27.36 | 81.0 ms | 18.76 s |
| 1,024 | 512 | 1 | 26.35 | 214.2 ms | 19.61 s |
| 4,096 | 512 | 1 | 25.25 | 729.0 ms | 20.97 s |
| 8,192 | 512 | 1 | 24.48 | 1.49 s | 22.36 s |
| 16,384 | 512 | 1 | 23.10 | 3.31 s | 25.43 s |
| 32,768 | 512 | 1 | 20.73 | 7.21 s | 31.85 s |
| 65,536 | 512 | 1 | 17.22 | 16.45 s | 46.12 s |
| 130,560 | 512 | 1 | 12.86 | 39.33 s | 79.06 s |

A minor, measured performance degradation is acceptable: a same-topology
median regression of at most 5% across three stable samples may ship when it is
documented. Cross-topology deltas are informational rather than regressions.
Correctness, valid structured tool calls, stable full-context serving, memory
headroom, and clean shutdown are hard gates; an unexplained or larger
same-topology regression blocks publication.

### Existing P150x4 baseline

The existing four-chip batch-1 baseline uses 512 output tokens:

| ISL | OSL | concurrency | decode tok/s/u | TTFT | E2EL |
|---:|---:|---:|---:|---:|---:|
| 128 | 512 | 1 | 42.38 | 69.5 ms | 12.1 s |
| 1,024 | 512 | 1 | 40.02 | 144.6 ms | 12.9 s |
| 4,096 | 512 | 1 | 37.54 | 454.5 ms | 14.1 s |
| 8,192 | 512 | 1 | 35.90 | 912.8 ms | 15.1 s |
| 16,384 | 512 | 1 | 33.05 | 2.08 s | 17.5 s |
| 32,768 | 512 | 1 | 28.34 | 4.48 s | 22.5 s |
| 65,536 | 512 | 1 | 22.17 | 10.17 s | 33.2 s |
| 130,560 | 512 | 1 | 15.44 | 25.12 s | 58.2 s |

The last row saturates the 131,072-token context with 512 generated tokens.
See [tool calling](doc/tool_calling/README.md) and the
[latency release gate](doc/serving_perf/LATENCY_RELEASE_GATE.md) for the exact
acceptance flows.
