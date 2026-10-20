# Gemma4 chunked prefill: traced vs. untraced (#60913)

## Verdict

Disabling trace is a large regression at small chunk sizes and a moderate one at 8k:

| Chunk | Chunks | Traced total | Untraced total | Δ | Δ % |
|---|---|---|---|---|---|
| 2048 | 128 | 10.31 s | 30.21 s | +19.90 s | **+193%** |
| 4096 | 64 | 7.19 s | 13.55 s | +6.36 s | **+88%** |
| 8192 | 32 | 6.40 s | 7.64 s | +1.24 s | **+19%** |

Totals are summed per-chunk latency (dispatch/replay + synchronize) over a 256k-token prefill. Wall time, which also
includes host staging, shows the same deltas (+185%, +85%, +19%).

Untraced execution is host-bound. Python and TTNN take about **205–235 ms of host time to dispatch one chunk**, almost
independent of chunk size, and the device then needs only another 1–8 ms after the host returns. An untraced chunk
therefore costs roughly `max(~220 ms, traced device time)`:

- 2048 and 4096: every traced chunk is shorter than the dispatch floor, so every untraced chunk sits on it.
- 8192: early chunks are host-bound (114 ms traced → 222 ms untraced); as ring history grows the device time
  overtakes the floor and the last chunk is within 3% (291 → 300 ms).

Kimi and GLM did not regress because their fastest chunks (~400 ms) already exceed a dispatch floor of this size.
Gemma4's 2k and 4k chunks take 54–151 ms traced, so they cannot hide it.

## Per-chunk latency (ms)

| Chunk | Metric | Traced | Untraced | Δ | Δ % |
|---|---|---|---|---|---|
| 2048 | first | 53.6 | 236.0 | +182.4 | +340% |
| 2048 | median | 80.3 | 234.1 | +153.9 | +192% |
| 2048 | last | 107.6 | 234.8 | +127.2 | +118% |
| 4096 | first | 75.5 | 218.6 | +143.1 | +190% |
| 4096 | median | 111.8 | 210.4 | +98.6 | +88% |
| 4096 | last | 151.0 | 211.5 | +60.5 | +40% |
| 8192 | first | 113.8 | 221.7 | +107.8 | +95% |
| 8192 | median | 198.4 | 223.6 | +25.2 | +13% |
| 8192 | last | 291.0 | 300.4 | +9.3 | +3% |

Untraced host dispatch per chunk (time for the model call to return, before synchronize), median [min, max]:
2048: 232.9 [230.0, 252.9] ms; 4096: 207.3 [204.9, 225.2] ms; 8192: 212.5 [210.6, 232.1] ms.

Repeatability: a second 2048 pair gave 10.32 s traced (+0.05%) and 30.10 s untraced (−0.4%).

## Workload and configuration

- Gemma4-31B-it, `models/demos/gemma4_d_p`, Blackhole Galaxy 8×4 (CP8/TP4), bf16, `FABRIC_1D_RING`.
- 256k-token context (262144), text token source (prefix of the cached Gutenberg corpus), one user.
- `main` at `e5cf5c83c2f` plus the `test_prefill_long_context_untraced` change described below. Release build,
  Tracy enabled.
- Environment: `TT_METAL_SHM_TRACKING_DISABLED=1`, `LOGURU_LEVEL=ERROR`, `HF_HUB_OFFLINE=1`.

Both modes use one code path, `_measure_traced` in `demo/text_demo_prefill.py`. Each runs one warm-up chunk to fill
the program cache and stages tokens, ring metadata and RoPE positions into the same persistent device tensors before
each chunk. The only difference is that the traced mode captures the forward once and replays it with
`execute_trace`, while the untraced mode calls the model for every chunk. Chunk timing starts after staging and ends
after `synchronize_device`.

## Commands

```bash
export HF_MODEL=google/gemma-4-31B-it HF_HOME=/mnt/models/huggingface \
       TT_CACHE_PATH=/mnt/models/huggingface/tt_cache/gemma4_d_p/google--gemma-4-31B-it \
       HF_HUB_OFFLINE=1 TT_METAL_SHM_TRACKING_DISABLED=1 LOGURU_LEVEL=ERROR
for C in 2048 4096 8192; do
  pytest "models/demos/gemma4_d_p/demo/text_demo_prefill.py::test_prefill_long_context_traced[blackhole-ctx_256k-chunk${C}-text-8x4]" -s --timeout 3600
  pytest "models/demos/gemma4_d_p/demo/text_demo_prefill.py::test_prefill_long_context_untraced[blackhole-ctx_256k-chunk${C}-text-8x4]" -s --timeout 3600
done
```

Each run writes its per-chunk timings to `$PREFILL_SUMMARIES/prefill_trace/*.json` because `LOGURU_LEVEL=ERROR`
hides the `[traced_perf]` / `[eager_perf]` log lines.
