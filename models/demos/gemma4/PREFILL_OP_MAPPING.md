# Prefill op mapping — Gemma-4 31B (single sliding layer @ ISL≈128)

## How this was produced

```bash
# Capture (already on disk)
python -m tracy -p -r -v -m pytest \
  models/demos/gemma4/tests/unit/test_prefill_single_layer_tracy.py \
  -k "sliding-prefill_128-1x8" -v -s --timeout=1800

# Report — MUST use stop (test emits start/stop, not end)
CSV=generated/profiler/reports/2026_08_12_09_06_18/ops_perf_results_2026_08_12_09_06_18.csv
tt-perf-report "$CSV" --start-signpost start --end-signpost stop > prefill/exp1.txt
```

- Report dir: `generated/profiler/reports/2026_08_12_09_06_18`
- Signposts: CSV has `start` @ row 1664, `stop` @ 2081 (no `end`)
- Mesh: Wormhole 1×8 (T3K), TP=8
- Stack: `num_layers=1` (sliding layer 0 only) + embed + final norm + lm_head
- Prompt pad: logical seq **96** tiles (report shapes `…x96x…`), not 128 — Tracy kernel pad / valid length

## Run configuration

| Item | Value | Evidence |
|------|--------|----------|
| Model | google/gemma-4-31B-it | HF_MODEL |
| Hidden / TP shard | 5376 / 672 | Embed out `96×672` → AG → `96×5376` |
| Vocab / TP shard | 262144 / 32768 | lm_head `32×32768` → AG → `32×262144` |
| Heads (sliding) | Q=4 local (32/8), KV=2 local (16/8), hd=256 | NlpCreateHeads `4×96×256`, KV `2×…` |
| Weights | BFLOAT8_B DRAM-IL | all matmul `INPUT_1` |
| Act | BF16 | |
| Prefill matmul MF | HiFi2 | QKV/O/MLP |
| lm_head MF | HiFi4 | |
| Norm / SDPA / RoPE MF | HiFi4 | |
| Chunk policy | 2048 on T3K (unused @ seq≤2048) | generator_trace policy |

## Where the time goes

From `prefill/exp1.txt` stacked report (device-time, **excluding** ~49.7 ms host gap before Embeddings):

| Bucket | ~% device | Device μs (merged) | Notes |
|--------|-----------|--------------------|-------|
| AllGather | ~42% | ~1650 | Vocab AG alone ~1330–1629 μs (1 core!) |
| Matmul | ~34% | ~1340 | lm_head 894 μs DRAM 68%; rest 1-layer |
| Untilize | ~7% | 266 | Full-vocab TILE→ROW_MAJOR for host |
| ReduceScatter | ~5% | ~180 | Attn O + MLP down (split all_reduce) |
| LayerNorm / Binary / DM | ~8% | | Width-sharded norms + residuals |
| SDPA + RoPE + Fill | ~2% | ~80 | Cheap at ISL~96 |

**Host gap:** Embeddings shows 49,740 μs op-to-op gap **inside** start→stop. Persists with correct `stop` signpost → real host work at start of measured replay (not a filter bug). Treat as Phase 5 / demo-optimization; do not count as device FLOPs.

**Full-model extrapolation:** lm_head+vocab AG+Untilize are **once**; layer matmul+CCL+SDPA **×60**. At ISL 128, re-rank after a 6-layer or full-model Tracy.

## Per-block tables

Memory: `DRAM-IL` / `L1-IL` / `L1-WS`. Representative Perf IDs from exp1 merged report.

### Prologue — embed + RoPE cache slice

| Perf ID | Perf OP | ttnn / block | In loc | In shape | Out loc | Out shape | Weight | Act / MF |
|---------|---------|--------------|--------|----------|---------|-----------|--------|----------|
| 1755 | Embeddings | `embed_tokens` — TP-sharded vocab table | DRAM-IL UINT32 | `1×96` ids | L1-IL TILE | `96×672` | DRAM-IL BF16 `V×672` | HiFi4 |
| (dev0#1) | AllGather | gather hidden across TP | L1-IL | `96×672` | L1-IL | `96×5376` | — | — |
| 1669+ | Slice | RoPE cos/sin slice from max cache | DRAM-IL | `4096×256` | DRAM-IL | `96×256` | — | — |
| 1741 | Copy | L1→DRAM before width-shard | L1-IL | `96×5376` | DRAM-IL | same | — | — |

### Layer 0 (sliding) — attention

| Perf ID | Perf OP | ttnn / block | In | Out | Weight | MF |
|---------|---------|--------------|----|-----|--------|-----|
| 1749 | LayerNorm (WS) | `input_layernorm` | L1-WS `96×5376` | L1-WS | scale DRAM-IL RM | HiFi4 |
| 1895 | Matmul | fused QKV `5376→2048` (local) | L1-IL | DRAM-IL `96×2048` | BFP8 `5376×2048` | HiFi2 |
| 1940 | NlpCreateHeads | split Q/K/V | DRAM `96×2048` | Q `4×96×256`, K/V `2×…` | — | — |
| 1898–99 | LayerNorm | Q/K (/V) head norms | DRAM | DRAM | scale | HiFi4 |
| 1900–01 | RotaryEmbedding | HF-style RoPE Q,K | DRAM | DRAM | cos/sin | HiFi4 |
| 1902 / 1947 | PagedFillCache | K,V into paged cache | DRAM | DRAM | — | — |
| 1904 | SDPA | causal+sliding W=1024 | DRAM Q/K/V | DRAM `4×96×256` | — | HiFi4 |
| 1819 | NLPConcatHeads | attn out → `96×1024` | DRAM | L1-IL | — | — |
| 1908 | Matmul | O-proj `1024→5376` | L1-IL | DRAM | BFP8 | HiFi2 |
| 1953 | ReduceScatter | row-parallel O | DRAM | DRAM `96×672` shard | — | HiFi4 |
| 1954 | AllGather | complete all_reduce | DRAM `96×672` | L1-WS `96×5376` | — | — |
| 1937 | LayerNorm | `post_attention_layernorm` | L1-WS | L1-WS | scale | HiFi4 |
| 1927 | BinaryNg | residual add | DRAM | DRAM | — | HiFi4 |

Source: `tt/attention/prefill.py`, `tt/attention/operations.py`, `tt/layer.py`, `tt/ccl.py` (RS+AG = split all_reduce).

### Layer 0 — SharedMLP (GeGLU)

| Perf ID | Perf OP | ttnn / block | In | Out | Weight | MF |
|---------|---------|--------------|----|-----|--------|-----|
| 2017 | LayerNorm | `pre_feedforward_layernorm` | L1-WS | L1-WS | scale | HiFi4 |
| 1742 | Matmul | fused gate+up `5376→5376` local (`2×2688`) | L1-IL | L1-IL | BFP8 `5376×5376` | HiFi2 |
| 1787–88 | Slice | split gate / up | L1-IL | `96×2688` | — | — |
| 1833 | BinaryNg | GeGLU (silu(gate)*up) | L1-IL | L1-IL | — | HiFi4 |
| 1834 | Matmul | down `2688→5376` | L1-IL | L1-IL | BFP8 | HiFi2 |
| 1967 | ReduceScatter | row-parallel down | L1-IL | DRAM shard | — | HiFi4 |
| 1968 | AllGather | complete all_reduce | DRAM | L1-WS | — | — |
| 1999 | LayerNorm | `post_feedforward_layernorm` | L1-WS | L1-WS | scale | HiFi4 |
| 1825 | BinaryNg | residual + `layer_scalar` path | DRAM | DRAM | — | HiFi4 |

Source: `tt/shared_mlp.py` (WH: tuned 1D / interleaved, not DramShardedLinear).

### Epilogue — final norm + lm_head + softcap + host readback

| Perf ID | Perf OP | ttnn / block | In | Out | Weight | MF |
|---------|---------|--------------|----|-----|--------|-----|
| (in stream) | LayerNorm | final RMSNorm | L1-WS | … | scale | HiFi4 |
| 2022 | Slice | last-token tile `get_last_token` → 32 rows | DRAM `96×5376` | `32×5376` | — | — |
| 2038 | Matmul **DRAM** | lm_head `5376→32768` | L1-IL | L1-IL | BFP8 `5376×32768` | HiFi4 |
| 2046–61 | Binary / Unary / Binary | softcap `tanh(x/30)*30` | L1-IL | L1-IL | — | HiFi4 |
| 2067 | AllGather | vocab gather TP×8 | L1-IL `32×32768` | DRAM `32×262144` | — | — **1 core** |
| 2079 | Untilize | TILE→ROW_MAJOR for host logits | DRAM | DRAM RM | — | HiFi4 |

Source: `tt/model.py` `_apply_lm_head`, `process_output_prefill` / untilize note ~1750.

## Optimization notes (Phase 1+)

### Phase 1 status (lm_head + vocab AG + Untilize)

| Lever | Result |
|-------|--------|
| lm_head progcfg | Already sweep winner `1d_c64_bw1` + L1 out + HiFi4 + bfp8 (`dram_sharded.lm_head_decode_config`). Near DRAM ceiling (~68%) — no further config headroom. |
| Untilize | Kept on host full-vocab path (cheaper than TILE height-slice + host read). Real skip = on-device sampling. |
| Vocab AllGather | **Skip when on-device sampling consumes TP-sharded logits** — same contract as decode. Wired via `allow_sharded_prefill_logits` / `process_logits_after_prefill_trace(..., allow_sharded=True)`. |

Expected device saving on the device-sampling path: **~1.6–1.9 ms** (AG ~1.3–1.6 ms + Untilize ~0.27 ms) per prefill first-token. Host-path Tracy (`GEMMA4_HOST_SAMPLE=1` default) still pays AG+Untilize by design.

### Phase 1 measured (exp2, 2026-08-12)

```bash
GEMMA4_TRACY_DEVICE_SAMPLE=1 python -m tracy -p -r -v -m pytest \
  models/demos/gemma4/tests/unit/test_prefill_single_layer_tracy.py \
  -k "sliding-prefill_128-1x8" -v -s --timeout=1800
tt-perf-report … --start-signpost start --end-signpost stop > prefill/exp2_devicesample.txt
```

| Metric | exp1 (host logits) | exp2 (device sample) | Δ |
|--------|--------------------:|---------------------:|--:|
| Device sum | 3,937 μs | 2,744 μs | **−1,193 μs (−30%)** |
| Vocab AllGather | 1,330 μs | **0** | −1,330 |
| Untilize (BF16 vocab) | 266 μs | **0** | −266 |
| TopK + Sampling (+tiny TM) | — | ~350 μs | +350 |
| lm_head `32×5376×32768` | 894 μs | 891 μs | ~flat |
| Host gap @ Embeddings | 49,740 μs | 8,133 μs | (path-dependent; Phase 5) |

Net epilogue: removed AG+Untilize (~1.6 ms), paid sampling (~0.35 ms) → **~1.2 ms device win** on this 1-layer window. Full-model TTFT still dominated by ×60 layers + host gap.

Demo: set `GEMMA4_HOST_SAMPLE=0` so Generator passes `sampling_params` and takes the skip-AG path.

### Later phases

1. **Per-layer RS+AG (~200 μs/layer)** — ×60 on full model; tune split-AR worker knobs; keep Linear on T3K.
2. **Prefill matmuls under-cored (24–42)** — Phase 3 at ISL≥2k with 2D-mcast.
3. **SDPA** — ignore until ISL 2048 capture.
4. **Host gap ~50 ms** — investigate Python before first device op inside signposted replay; not a matmul config issue.

## Next measurement

```bash
# Optional Phase 0 matrix (same recipe, correct stop)
-k "sliding-prefill_2048-1x8 or full-prefill_128-1x8"
tt-perf-report … --start-signpost start --end-signpost stop
```
