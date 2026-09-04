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
after a per-shape warmup; `output` is the model's actual completion length.

### P150 tool-calling API

| input | output | TTFT | derived TPOT | end-to-end | tokens/s/user | result |
|---:|---:|---:|---:|---:|---:|:---:|
| 512 | 135 | 613.7 ms | 95.20 ms | 13.37 s | 10.50 | pass |
| 1,024 | 189 | 731.9 ms | 96.61 ms | 18.89 s | 10.35 | pass |
| 4,096 | 171 | 1.58 s | 98.80 ms | 18.38 s | 10.12 | pass |
| 8,192 | 168 | 2.84 s | 100.34 ms | 19.59 s | 9.97 | pass |
| 16,384 | 117 | 5.77 s | 102.65 ms | 17.67 s | 9.74 | pass |
| 32,768 | 167 | 12.21 s | 109.75 ms | 30.43 s | 9.11 | pass |
| 65,536 | 168 | 27.75 s | 122.28 ms | 48.17 s | 8.18 | pass |
| 130,560 | 162 | 69.59 s | 147.03 ms | 93.26 s | 6.80 | pass |

### P150x2 tool-calling API

| input | output | TTFT | derived TPOT | end-to-end | tokens/s/user | result |
|---:|---:|---:|---:|---:|---:|:---:|
| 512 | 135 | 283.8 ms | 35.48 ms | 5.04 s | 28.19 | pass |
| 1,024 | 180 | 364.2 ms | 36.13 ms | 6.83 s | 27.68 | pass |
| 4,096 | 278 | 7.80 s | 12.01 ms | 11.13 s | 83.26 | pass |
| 8,192 | 147 | 1.70 s | 37.06 ms | 7.11 s | 26.98 | pass |
| 16,384 | 128 | 3.54 s | 37.91 ms | 8.36 s | 26.37 | pass |
| 32,768 | 168 | 7.49 s | 40.00 ms | 14.17 s | 25.00 | pass |
| 65,536 | 110 | 16.99 s | 43.41 ms | 21.72 s | 23.04 | pass |
| 130,560 | 135 | 42.03 s | 51.18 ms | 48.89 s | 19.54 | pass |

At P150x2 ISL 4096, the reasoning/tool parser buffers early output until late
in the completion. Client-visible TTFT therefore absorbs decode work and the
derived TPOT is correspondingly low; the fixed-OSL control below is the
authority for raw device-path TPOT.

### Fixed-OSL device control

Every row below generated exactly 512 tokens. These runs bypass the API parser
and provide the comparable underlying device-path latency.

| input | P150 TTFT | P150 TPOT | P150 E2E | P150x2 TTFT | P150x2 TPOT | P150x2 E2E |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 177.3 ms | 97.74 ms | 50.12 s | 81.0 ms | 36.55 ms | 18.76 s |
| 1,024 | 318.8 ms | 100.00 ms | 51.42 s | 214.2 ms | 37.95 ms | 19.61 s |
| 4,096 | 1.13 s | 103.53 ms | 54.03 s | 729.0 ms | 39.61 ms | 20.97 s |
| 8,192 | 2.29 s | 106.17 ms | 56.54 s | 1.49 s | 40.84 ms | 22.36 s |
| 16,384 | 5.01 s | 111.44 ms | 61.96 s | 3.31 s | 43.29 ms | 25.43 s |
| 32,768 | 10.99 s | 121.55 ms | 73.10 s | 7.21 s | 48.24 ms | 31.85 s |
| 65,536 | 25.33 s | 141.48 ms | 97.62 s | 16.45 s | 58.06 ms | 46.12 s |
| 130,560 | 63.56 s | 180.15 ms | 155.62 s | 39.33 s | 77.74 ms | 79.06 s |

A minor, measured performance degradation is acceptable: a same-topology
median regression of at most 5% across three stable samples may ship when it is
documented. Cross-topology deltas are informational rather than regressions.
Correctness, valid structured tool calls, stable full-context serving, memory
headroom, and clean shutdown are hard gates; an unexplained or larger
same-topology regression blocks publication.

### Existing P150x4 baseline

The existing four-chip batch-1 baseline uses 512 output tokens:

| input tokens | TTFT | TPOT | end-to-end | tokens/s/user |
|---:|---:|---:|---:|---:|
| 128 | 69.5 ms | 23.60 ms | 12.1 s | 42.38 |
| 1,024 | 144.6 ms | 24.99 ms | 12.9 s | 40.02 |
| 4,096 | 454.5 ms | 26.64 ms | 14.1 s | 37.54 |
| 8,192 | 912.8 ms | 27.86 ms | 15.1 s | 35.90 |
| 16,384 | 2.08 s | 30.26 ms | 17.5 s | 33.05 |
| 32,768 | 4.48 s | 35.29 ms | 22.5 s | 28.34 |
| 65,536 | 10.17 s | 45.10 ms | 33.2 s | 22.17 |
| 130,560 | 25.12 s | 64.76 ms | 58.2 s | 15.44 |

The last row saturates the 131,072-token context with 512 generated tokens.
See [tool calling](doc/tool_calling/README.md) and the
[latency release gate](doc/serving_perf/LATENCY_RELEASE_GATE.md) for the exact
acceptance flows.
