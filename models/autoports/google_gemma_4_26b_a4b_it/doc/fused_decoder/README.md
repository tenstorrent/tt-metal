# Gemma4 fused decoder

Stage 02 for `google/gemma-4-26B-A4B-it`, revision
`4d7ae4984b7db7de8f8457170b3f1a419ee76d52`. Stage review is **clean-pass**; runtime gates, controls and evidence are complete. No later pipeline stage is part of this change.

## Runtime and contract

`tt/fused_decoder.py:FusedDecoder` supplies the fused computation and inherits
the functional decoder's public chunking, request-slot and cache orchestration.
The real-weight regression disables `FunctionalDecoder._forward` and exercises
both layer kinds through the fused path. There is no functional fallback.

The context contract remains 262144 tokens with arbitrary valid logical lengths,
a default physical chunk of 1024 (configurable under the inherited contract),
BF16 paged K/V and 32-token pages. Sliding attention (layer 0, Q16/KV8/D256)
and full attention (layer 5, Q16/KV2/D512, tied K/V) both include a shared MLP
and top 8 of 128 routed experts. Acceptance remains PCC >= .995.

Selected changes:

- Broadcast FP32 QKV/router products without repeated activations, coalesce QKV
  groups to 16384 rows, and project full-attention tied K/V once. QKV output
  goes directly to the head split's L1 input.
- Pack routed gate/up weights, remove identity movement, merge prefill tiles
  into 64-token expert calls, and concatenate once. Merge accurate GELU into
  shared multiplication and expert prefill; separate expert decode GELU is faster.
- Batch independent decode KV-head attention. Merge exact subtract/exp while
  retaining the accurate FP32 softmax reduction. Concatenated heads remain
  sharded through the output projection.
- Reuse common residual normalization. Use native unweighted RMSNorm plus
  external FP32 gamma at validated boundaries; sliding head norms retain
  precise SFPU arithmetic with merged epsilon/rsqrt.
- Write shared-down and expert-mixing matmul outputs directly into their norms'
  width shards. Shared-down decode uses the program's inferred LoFi/BF16 policy;
  expert mixing uses HiFi4 with FP32 accumulation for sliding and BF16 for full.
- Preserve output norm sharding, merge addition into the combined norm's residual
  argument, then consume its shard directly in the final add/scale/BF16 output.
- Sliding uses native HF rotary in decode, fused K/V updates and unary-chain
  cast-to-shard. Full retains precise rotary, separate cache writes and its
  faster explicit cast/shard; tied K/V normalization is shared.

`FusedDecoder.DEFAULT_FUSIONS` records exact policies. Select the delivered default
with `--decoder fused`; `--fusion` and `--group-size` retain experiment controls.
[PATTERNS.md](PATTERNS.md) covers every skill pattern, adaptations, measured
rejections and the five additional boundaries found by independent review.

## Measured headline

Exactly 4096 input tokens, 128 subsequent decode steps, batch 1, one request,
on one Blackhole ASIC. Device times include every layer operation and internal gap.

| Layer kind | Prefill device ms, before → after | Traced decode device ms, before → after |
| --- | ---: | ---: |
| sliding_attention | 4998.537 → 2812.752 | 9.811 → 5.147 |
| full_attention | 5009.443 → 2825.397 | 10.894 → 5.585 |

Prefill improves 1.777x/1.773x and decode 1.906x/1.951x, respectively.
Final headline PCC is .998553/.998985 (prefill/minimum decode) for sliding and
.999419/.999736 for full. Direct TTNN equivalence also passes.
[PERFORMANCE.md](PERFORMANCE.md) gives roofline assumptions and report paths;
[ACCURACY.md](ACCURACY.md) gives before/after PCC and semantic coverage.

## Reproduce and inspect

From the repository root with the configured environment:

```bash
python_env/bin/python -m models.autoports.google_gemma_4_26b_a4b_it.tests.run_decoder \
  --decoder fused --layer 0 --length 4096 --real --decode --steps 128 \
  --verify-program-cache --output /tmp/fused_sliding.json
python_env/bin/python -m pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/test_fused_decoder.py
```

Use layer 5 for full attention. `verified_validation_commands.json` records
exact commands, exit statuses and frozen source hashes for final equivalence,
request reuse, batch 32, continuation, pytest, watcher, Tracy and context checks.
Watchers and profilers run separately. `tracy/{sliding,full}_verified/` contains
whole-layer summaries, per-replay CSV, tt-perf-report tables/CSV and provenance.
Large raw captures and saved comparison tensors remain local.

`candidate_summary.json` contains 66 rederived headline timing rows. Its `passed`
field means headline acceptance only; broader failing controls invalidate the
faster native-normalization candidates. Final unprofiled defaults reproduce the
chosen combined candidates within .03%. Host replay timings are not device fields.

## Limits and stage evidence

This is single-layer evidence, not autoregressive full-model text generation.
Full-context TT prefill computes every query/cache row; the HF oracle compares
291 sampled rows. End-of-context decode changes cache state between positions,
so repeated fixed-state headline/batch replay establishes determinism.

A controlled S65 sliding decode position swaps the eighth expert under BF16 KV
rounding in both functional and fused paths; CPU HF controls reproduce it.
The full8-position direct-equivalence regression passes the unchanged0.995 bar;
see `ACCURACY.md`. Headline FP32-HF gates remain unchanged.

The remaining embedding-internal cache untilization is retained after slower
exact tiled-gather controls. Setup-only weight conversion is outside measured
forward paths; runtime audits reject torch/from_torch/to_torch fallback.
`AUTOFIX.md`, `AUTOFIX_profiler.md` and `work_log.md` classify numerical,
profiler and test-harness anomalies. Invalid/interrupted captures supply no result.

Only Python, tests and docs changed; no C++/CMake build is required.
Final acceptance is recorded in `STAGE_REVIEW.md`; local checkpoint SHAs are
recorded in `work_log.md`. The resumed attempt's compact packet is
`bringup/artifacts/multigoal-runs/20260925T171711Z/telemetry/packets/e73d7165-fc65-44f4-828b-1f979f0dcf75.json`.
