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
| `p150` | 1 | 1 | 1 | implementation and offline topology tests pass; hardware sweep pending |
| `p150x2` | 2 | 2 | 1 | implementation and offline topology tests pass; hardware sweep pending |
| `p150x4` | 4 | 4 | 32 | previously validated on the four-chip QB2/P300x2 configuration |

The table records current evidence, not intended status. Replace the two
pending cells with links to the measured artifacts before publishing the
updated package.

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

The P150 and P150x2 release tables will be generated from the tool-calling API
sweep and committed here before publication. A minor, measured performance
degradation is acceptable: a same-topology median regression of at most 5%
across three stable samples may ship when it is documented. Cross-topology
deltas are informational rather than regressions. Correctness, valid structured
tool calls, stable full-context serving, memory headroom, and clean shutdown are
hard gates; an unexplained or larger same-topology regression blocks
publication.

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
