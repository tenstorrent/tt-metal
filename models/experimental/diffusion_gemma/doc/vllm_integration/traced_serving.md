# Traced serving decode, context/DRAM scaling, and production-Gumbel tracing

Status: current for the eager-vs-traced result and the DRAM table; the per-request /
growing-prefix capture lifecycle it describes is superseded by the model-lifetime up-front capture
(see [serving README](README.md)). Over the 100-line cap because two open contradictions, four
refutations and the traced-Gumbel design traps are never cut for length.
Owns: eager-vs-traced serving decode, the per-chip DRAM-vs-`max_model_len` table and the KiB/tok
disagreement derived from it, the trace-region capacity result, and the traced-Gumbel constraints.
See also: [refuted list](../REFUTED.md) · [serving README](README.md) ·
[trace hazards](../optimize_perf/README.md)

## The trace win

Full-depth 30L, K=48, msl=4096, on the serving session the vLLM adapter delegates to:

| config | output t/s | steady block (s) | TTFT (s) | steps | commit sha |
|---|---:|---:|---:|---|---|
| eager | **6.86** | 37.32 | 40.0 | [48,48] | `8f015a49e4e31a63` |
| **traced** | **17.93** | 14.28 | 123.9 | [48,48] | `8f015a49e4e31a63` |
| early_halt | 17.85 | 14.34 | 123.9 | [48,48] | `8f015a49e4e31a63` |

**2.61x, byte-identical commit.** That same sha also equals dg-08's 30L no-halt fixed-48 traced
commit, so traced serving reproduces the generator's committed argmax exactly and the eager path is
unchanged. The traced TTFT includes the **one-time** 48-single-step-trace capture on block 0
(~110 s); the served rate is the steady per-block latency, so capture amortizes over the request's
blocks rather than being paid per block. Decode is context-flat: eager 6.86 t/s at msl 4096 vs
6.51 t/s at msl 32768.

The deleted `vllm_speed_by_context.md` stated this same eager-vs-traced ratio as **2.72x** against
the measured **2.61x** here; both readings are recorded, and its eager control rows now live in
[serving README](README.md#live-serving-evidence).

> **OPEN CONTRADICTION (unexplained):** early halt. This file measured a **uniform no-op** —
> `traced_early_halt_block` (scheme A, threshold 0.005), 5 chat-templated prompts at seed 0, every
> block ran the full 48 steps, **0/5 halted**, because the entropy gate never clears 0.005 (mean
> entropy floors ~0.14–0.51 nats per dg-08's `probe_halt_gap`), costing ~0.5% for nothing (17.93
> traced vs 17.85 early_halt, within block-timing noise; dg-08's per-step-sync upper bound was ~2%).
> [early halt](../optimize_perf/early_halt.md) records measured halts at [9,17,2]/48 and K=10–43
> under the concat MoE. Not explained.

**MEASUREMENT TRAP:** the "35-step halt" quoted from live serving came from a raw,
non-chat-templated 6-token "Hello" prompt on a different seed/run and does not reproduce under a
controlled chat-templated measurement.

## Context and DRAM

Post-build per-chip DRAM vs `max_model_len` on QB2 (21.87 GiB/chip):

| max_model_len | used (GiB) | free (GiB) |
|---:|---:|---:|
| 1024 | 13.27 | 8.60 |
| 4096 | 13.46 | 8.41 |
| 8192 | 13.72 | 8.15 |
| 16384 | 14.24 | 7.63 |
| 32768 | 15.27 | 6.60 |

> **OPEN CONTRADICTION (unexplained):** the contiguous KV cost per token derived from *this one
> table*. This file derives **~66 KiB/tok** — `(15.27-13.46) GiB / (32768-4096) tok` = 66.2 and
> `(13.46-13.27)/(4096-1024)` = 64.8. The deleted `doc/vllm_integration/work_log.md` states
> **~63 KiB/tok** from the same rows. Not explained. `doc/optimize_perf/bisect_trace_region.sh:9`
> cites this table by line number as `traced_serving.md:86-89`, so renumbering it breaks that
> reference.

**REFUTED:** the earlier estimate that treated `DG_TRACE_REGION_SIZE` as immediately-resident DRAM.
The live server proves it is a **capacity** limit — 48 resident traces increased used DRAM by only
~1.41–1.44 GiB/chip (~1.430–1.432 across the K sweep) against a configured 10 GiB region. The
consequence, that "at msl=32768 traced serving is trace-region-gated before the eager KV ceiling",
is refuted with it. At msl=32768 both a 32-token prompt and a real 16384-token prompt captured and
replayed all 48 traces while leaving 5.17 / 5.16 GiB free. The bounded probes establish 32768 as
passing; they do **not** establish the absolute ceiling, and 256K was never forced — see the served
ceiling contradiction in [serving README](README.md#served-context--an-open-contradiction).

Allocation scaling and actual-prompt scaling are **distinct**: a fixed 32-token prompt stayed flat
at 18.83–18.89 t/s for allocated limits 4096→32768, while real 6144 / 8192 / 16384-token prompts
measured 12.68 / 11.88 / 9.49 t/s. The frozen-prefix *read* is material at real long context even
though allocation alone is not. Warmed denoise cost stayed ~251.3–251.8 ms/step for K=4–48 (K=1's
253.6 ms estimate is more sensitive to fixed synchronization overhead), so block latency scales
predictably with K. Full sweep rows: [serving README](README.md#live-serving-evidence).

## Production Gumbel under trace (2026-07-13)

- **DESIGN CONSTRAINT:** materialized Gumbel is forced to **one-step** trace windows — sharing one
  full-vocab tensor across multiple steps inside a grouped trace would silently reuse one draw.
- **DESIGN CONSTRAINT:** grouped trace windows cannot refresh a per-step noise/seed input inside the
  window without changing the captured graph, which is why dynamic Gumbel modes use one-step traces.
- **TRAP:** ordinary `ttnn.rand(seed=<Python int>)` would **bake block 0's seed into replay**. The
  traced path instead uses a DG-local `ttnn.generic_op` uniform kernel that reads the seed from one
  persistent device tile, refreshed before each single-step replay, with all vocab chunks reusing
  one persistent uniform buffer. Chunk-selection constants are allocated during warmup so capture
  contains no host writes. Kernels live under `models/experimental/diffusion_gemma/tt/kernels/`; no
  shared Gemma4 or TTNN source is modified.

QB2 gates — `test_trace_seeded_uniform_refresh_matches_rand`,
`test_traced_materialized_gumbel_refresh_matches_fixed_loop_across_blocks`,
`test_traced_chunked_gumbel_dynamic_seed_matches_fixed_loop_across_blocks`,
`test_trace_capture_guard_recovers_after_injected_failure` → **4 passed in 1.20 s**. The
dynamic-seed uniform output is bit-identical to `ttnn.rand` for both the capture seed and a
different replay seed. The injected capture-failure gate ends and releases the aborted trace, then
successfully captures and replays a second trace on the same device. The reduced growing-prefix run
matches eager end-to-end at `committed_sha256=7b7d…fbba`, while a frozen-prefix A/B has the **same**
block-0 hash and a **different** block-1 hash — proving committed KV affects later blocks'
decisions.

2026-07-13 growing-prefix throughput: reduced 1-layer K=2 two blocks **37.19** output tok/s (2
captures, 4 traces, 4 executes); full 30-layer K=2 two blocks **25.10** t/s at 32→288→544; released
full 30-layer K=48 two blocks block-0 TTFT 179.36 s and block-1 including recapture 180.82 s for
**1.42** output tok/s, 48 traces per prefix shape, 96 captured / 96 executed, clean release. Because
the contiguous prefix tensor shape changes by 256 each block, the controller releases and recaptures
at the new shape; paged or fixed-shape prefix inputs are required to recover capture-once replay.

**MEASUREMENT TRAP:** the July-09/10 same-ID cross-block rows held the denoise prefix at the initial
*prompt* length, so they are same-shape performance provenance and **not** correct
block-autoregressive growing-prefix evidence.

Artifacts: `vllmtraced_msl4096.json`, `vllmtraced_msl32768.json`,
`traced_chunked_gumbel_20260713.json`. This is a perf feature, not Tracy/device profiling: no
profiler was run against a live server.
