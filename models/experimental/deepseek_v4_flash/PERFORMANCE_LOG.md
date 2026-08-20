# DeepSeek V4 Flash — Decode Performance Log

This log tracks the decode throughput (tokens/s, generally per-user unless
noted) of the `models/experimental/deepseek_v4_flash` implementation over
time, reconstructed from the git history. Each entry cites the commit that
reported the new number and a short description of the optimization that
produced the gain.

All numbers are per-user (tok/s/u) tokens/second at batch=1 unless a batch
size is explicitly called out.

## Summary chart

| # | Date (UTC) | Commit | Perf | Δ vs prev | What changed |
|---|------------|--------|------|-----------|--------------|
| 1 | 2026-06-29 | [`789fc6f`](../../../) | **5.99 tok/s** | baseline | First working full decode pipeline gets a real measurement. |
| 2 | 2026-07-02 | `71d1bed` | **7.36 tok/s** | +23% | Optimized the mixed **HyperConnection (mHC)** op — added a fused C++ device op path in `common.py`/`fused_hyperconnection.cpp` instead of composing it from several ttnn ops. |
| 3 | 2026-07-07 | `578ffe8` | **8.1 tok/s** | +10% | **Batched matmul in attention** — replaced per-head/per-tensor matmuls with a single batched matmul call in `attention.py`/`layers.py`. |
| 4 | 2026-07-25 | `7c367fa` | **10.5 tok/s** | +30% | **HCA/CSA rework** — restructured hyperconnection/cross-stream-attention handling and added vLLM generator plumbing (`generator_vllm.py`, `model.py` rewrite). |
| 5 | 2026-07-26 | `5faf162` | **10.7 tok/s** | +2% | **Slice-write instead of concat** for KV cache updates in attention — avoids the cost of a full concat op on every decode step. |
| 6 | 2026-07-31 | `d2d5aa3` | **11.6 tok/s** | +8% | **Removed redundant RoPE computation** — RoPE tables were being recomputed per layer; hoisted/cached at the model level. |
| 7 | 2026-07-31 | `0701a26` | **11.8 tok/s** | +2% | **Fused "mix streams" op** — new custom device op (`mix_streams.cpp`, kernels) to fuse the MoE gate/up stream mixing instead of separate ttnn ops. |
| 8 | 2026-08-01 | `8999002` | **11.95 tok/s** | +1% | **Fused pre/post + split in HyperConnection** — merged the pre-process/post-process/split steps of the hyperconnection op into one kernel (`fused_pre_post_*`). |
| 9 | 2026-08-02 | `ca92831` | **12 tok/s** | +0.4% | **Removed unnecessary memory_config on sharded tensors** — cut redundant reshard/copy overhead in `layers.py`/`model.py`. |
| 10 | 2026-08-03 | `42dc762` | **12.3 tok/s/u** | flat | **Asserted sharded-only RMSNorm** — forced RMSNorm inputs to always be sharded, removing a conditional interleaved fallback path. |
| 11 | 2026-08-03 | `e73ea42` | **12.3 tok/s/u** (PGS=1) | flat | **Fused `mix_streams` into a real custom op** (device op + program factory + kernels) rather than the composed version from step 7, enabling `PIPELINE_GROUP_SIZE=1`. |
| 12 | 2026-08-10/11 | `dcf6bca` | **14 tok/s** | +14% | **WIP: added batch (B>1) support** — extended attention/hyperconnection/sinkhorn ops to handle multiple users per call. |
| 13 | 2026-08-10 | `ba43b8a` | — | — | Got a working decoder layer at batch **B=8**. |
| 14 | 2026-08-10 | `ceebba9` | **4.3 tok/s/u** at **B=64** | -69%/user, but ~275 tok/s aggregate | Scaled up to **batch=64**; per-user throughput drops (as expected under heavy batching) while aggregate cluster throughput rises substantially. |
| 15 | 2026-08-11 | `4de494d` | **14.2 tok/s/u** | — | **Reusable GCB (Global Circular Buffer)** for `matmul_decode` — the prefetcher's GCB is now allocated once and reused across calls instead of being rebuilt each time. |
| 16 | 2026-08-11 | `63f55dd` | **15.4 tok/s** | +8% | **Prefetcher used in MoE** — added `decode_prefetch.py` and wired the weight-prefetcher into the MoE matmuls, removing weight-fetch stalls. |
| 17 | 2026-08-11 | `6150d00` | **15.7 tok/s/u** | +2% | **Optimized MoE router** — reworked `moe.py`'s hash router and `fused_experts` device op/kernels (expert-id computation) to cut router overhead. |
| 18 | 2026-08-12 | `a4b9672` | **16.2 tok/s/u** | +3% | **Batched `MatmulDecode` combined with the Prefetcher** — extended the prefetcher-backed matmul decode op to operate on batched inputs, fusing two previous optimizations. |

## Overall trajectory

- **5.99 → 16.2 tok/s/u** at batch=1: a **~2.7x** improvement over ~6 weeks
  (2026-06-29 → 2026-08-12).
- Early gains (steps 1-6, 5.99 → 11.6 tok/s, ~+94%) came from **algorithmic/op
  fusion work**: fusing HyperConnection, batching attention matmuls,
  restructuring HCA/CSA, switching KV-cache updates from concat to slice-write,
  and removing redundant RoPE recomputation.
- Middle gains (steps 7-11, 11.6 → 12.3 tok/s, ~+6%) were **op-fusion and
  correctness tightening**: fusing MoE stream mixing into a dedicated custom
  device op (twice — first a composed version, then a full custom op with
  kernels), fusing pre/post/split steps of HyperConnection, dropping redundant
  memory-config/resharding calls, and forcing sharded-only RMSNorm.
- The **batching push** (steps 12-14) is a separate axis: rather than only
  optimizing single-user latency, the model was extended to support B>1 and
  then B=64 users concurrently. Per-user tok/s dips under heavy batching
  (4.3 tok/s/u @ B=64) because compute/bandwidth is shared, but total
  cluster throughput (aggregate tok/s across all users) is far higher.
- The **final push** (steps 15-18, 14.2 → 16.2 tok/s/u, ~+14%) focused on the
  **weight-prefetcher infrastructure**: making the prefetcher's global
  circular buffer (GCB) reusable, extending prefetching to MoE matmuls,
  optimizing the MoE router, and finally combining batched `MatmulDecode`
  with the prefetcher so both optimizations compound.

## Commit reference

```
789fc6fc489  DeepSeek 5.99 tps
71d1bede1a6  Opt mHC. 7.36 toks/s
578ffe8d18a  Batched Matmul in Attn. 8.1 toks/s
7c367fa5f9b  Update HCA/CSA. 10.5 tok/s
5faf1624806  Using slice write instead of concat. 10.7 toks/s
d2d5aa32553  Remove redundant RoPE Calculation. 11.6 toks/s/s
0701a26ca5a  Fused Mix Streams. 11.8 toks/s
89990027627  Fused Pre Post with Split in Hyperconnection. 11.95 tok/s/u
ca92831363b  Removed memory config of sharded. 12 toks/s
42dc762c32a  Assert only sharded rms_norm. 12.3 toks/s/u
e73ea425d18  Fused mix_streams op. 12.3 toks/s/u with PGS=1
dcf6bca2d2f  WIP. Adding support for B > 1. 14 toks/s
ba43b8a664f  Working decoder layer with B=8
ceebba9f3ed  Batch=64. 4.3 tok/s/u
4de494dec1d  Reusable GCB. 14.2 tok/s/u
63f55dd160c  Prefetcher in MoE 15.4 toks/s
6150d00e3ce  Optimized Router. 15.7 tok/s/u
a4b967209214 Batched MatmulDecode with Prefetcher. 16.2 tok/s/u
```

*(Generated from `git log --oneline -- models/experimental/deepseek_v4_flash`
and per-commit diffstats; see individual commits for full diffs.)*
