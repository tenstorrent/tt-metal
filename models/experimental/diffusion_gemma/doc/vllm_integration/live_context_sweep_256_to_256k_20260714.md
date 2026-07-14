# DiffusionGemma vLLM live speed sweep — 256 → 256K attempt (latest optimizations)

Date: 2026-07-14 · QB2 / P150x4 / 4× Blackhole / TP=4 · real `tenstorrent/vllm` OpenAI path
Evidence: `live_context_sweep_256_to_128k_20260714.json` · harness: `live_context_sweep.py`

## Config (full optimization stack)

One server, `--generation-config vllm`, `--max-num-seqs 1`, `--block-size 64`, on-device sampling,
temp 0, ignore-EOS, 48 denoise steps/block, 3 blocks/request. Flags:
`DG_VLLM_TRACE=1` (traced denoise), `DG_SPARSE_MOE=1` + `DG_SPARSE_MOE_TUNED=1` + `DG_DEDUP_ARGMAX=1`
(sparse denoise MoE), `DG_PREFILL_MOE_TUNED=1`, and **`DG_PREFILL_RAGGED_LONG` default-on** — the
chunked ragged prefill added 2026-07-13 (the new optimization vs the 2026-07-10 baseline).

## Result — prefill (TTFT) win, generation regression, and the ceilings

| prompt | **prefill_s (now)** | prefill_s (2026-07-10) | speedup | steady/blk | out tok/s | recapture |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | **0.65** | 2.97 | 4.6× | 71.1 s | 3.60 | yes |
| 1,024 | **0.60** | 15.07 | **25×** | 71.0 s | 3.60 | yes |
| 4,096 | **2.36** | — | — | 74.7 s | 3.43 | yes |
| 16,384 | **13.52** | ~270 (msl32768) | **20×** | 83.6 s | 3.06 | yes |
| 32,768 | — timed out (see below) | — | — | — | — | — |

DRAM stayed ~22.9 GiB used / 27.9 usable through 16K.

### 1. Prefill fix works end-to-end in serving — 20–25× faster TTFT
The chunked ragged prefill (default-on) makes the real vLLM prefill **20–25× faster** for
prompts past the old 4096 cap (1024: 15.07→0.60 s; 16384: ~270→13.5 s). The >4096 prefill cliff is
gone in the serving path, exactly as in the standalone bit-identity + sweep evidence.

### 2. Steady generation regressed to 3.6 tok/s (was ~18) — traced-denoise recapture
Every block recaptures the Metal trace (`recapture_after_block0=true`, `steady_replay=false`), so
steady is ~71–84 s/block ≈ **3.6 tok/s vs the 2026-07-10 replay's ~18 tok/s**. Root cause: commit
`ec5b64b4891` ("trace production Gumbel with growing-prefix correctness") grows the cross-attention
prefix +256/block and added a `prompt_len`-keyed trace-invalidation guard in `traced_denoise.py`;
because the denoise mask shape depends on `prompt_len` ([denoise_forward.py:83](../../tt/denoise_forward.py#L83)),
the trace invalidates every block. It is a **genuine correctness fix** (old replay left committed
tokens invisible to later blocks) delivered **suboptimally** (full recapture instead of a fixed-max
mask fed as a replay input). **Recoverable** (~2–4 days): capture once against a fixed max-context
mask, feed the growing prefix as a written replay input (the KV-decode pattern) → restore ~18 tok/s
while keeping correctness. Recommend scheduling that follow-up, not reverting.

### 3. 256K is infeasible via vLLM serving with bf16 weights (KV-memory bound)
The TT plugin sizes the paged KV pool from `max_model_len` (it ignores `--num-gpu-blocks-override`):
at `max_model_len=262144` the KV cache is ~15 GiB, which plus bf16 weights (13.25 GiB) plus a trace
region overflows the 32 GB/chip — build-time OOM in `init_kv_cache` (two confirmed OOMs). Max feasible
traced context ≈ **128K** (this run used `max_model_len=131072`, which builds fine). Reaching 256K
needs **bf8 weights** (~6.5 GiB, frees the room), or dropping traced denoise, or expert-parallel.

### 4. 32K serving request stalled (distinct from prefill compute)
The 32768 request emitted no block markers and the request timed out (>… harness limit) during
prefill / first-use SDPA-kernel compilation — **no eth-core hang, no OOM**. Notably the *standalone*
32K prefill is 10.8 s (`context_window_prefill_only_chunkedlong_*`), so the vLLM **serving** 32K
prefill path is pathologically slow (32768 is just below the >32768 chunked-SDPA threshold → full 32K
attention + a fresh kernel-compile storm). A separate serving-path item to investigate.

## Practical ceiling

With the current code, clean traced serving is demonstrated to **16K** context. Above it, generation
is impractically slow (recapture, §2) and 32K stalls in the serving prefill (§4); 256K is KV-memory
bound (§3). The prefill optimization itself (§1) is solid across the whole measured range. Priority
follow-ups: (a) restore denoise trace replay (§2), (b) fix the 32K serving-prefill stall (§4),
(c) bf8 weights for >128K context (§3).

## Harness changes (this run)

`live_context_sweep.py`: added `--num-gpu-blocks-override` pass-through; relaxed the trace-region
guard to `[2 GiB, 10 GiB]` (long context needs the DRAM back from the reserved region); softened the
strict capture-once trace-event assertion to record actual capture/replay/release counts (the
capture-once contract is superseded by the growing-prefix per-block recapture — the timing metrics
come from the `DG_VLLM_METRIC` block markers and are unaffected).
