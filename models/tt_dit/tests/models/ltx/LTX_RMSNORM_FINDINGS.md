# LTX-2.3 AV DistributedRMSNorm fusion: coverage & findings

Benchmark suite: `test_ltx_distributed_rmsnorm_bench_fused.py`. Covers every distinct
RMSNorm shape×fusion-pattern from the LTX-2.3 AV transformer block (see
`distributed_rmsnorm_av.md`), at TP=2 (SP=4) and TP=4 (SP=8), both video stages —
14 distinct configs/TP, 3 fusion patterns. Baseline = composite RMSNorm +
the *unfused* trailing op LTX uses today (`ttnn.addcmul` for adaLN, standalone
`ttnn.experimental.rotary_embedding_llama` for RoPE); fused = the single device op.

## Feature support: nothing missing at the API level
Every LTX fusion opportunity maps onto the existing op — no new kernel capability needed:
- **adaLN addcmul** `normed·(1+scale)+shift` → `weight=(1+scale)` + `bias=shift` (both broadcast). `has_bias` path works.
- **static QK affine** → `weight`.
- **create_heads** → `num_heads_per_device` (emits BHNE).
- **per-head RoPE** `(1,H,N,head_dim)` → auto-detected (`rope_cos.shape[1]==num_heads_per_device`).
- whole-row norm + per-head rope → `per_head_norm=False` + per-head rope (independent flags).

## What fuses today (measured, TP=4 galaxy LINE, 4 links)
| config | pattern | feat | hd | rows | speedup |
|---|---|---:|---:|---:|---:|
| a_selfattn_qk | qk + per-head rope | 512 | 64 | 32 | **1.72×** |
| a2v_audioK | qk + per-head rope | 512 | 64 | 256 | **1.43×** |
| v_textcross_q | qk (no rope) | 1024 | 128 | 4864 | **1.48×** |
| v_block / a_block | block + adaLN (weight+bias) | 1024/512 | — | — | runs ✓ |

i.e. block norms, audio QK (self + cross), A→V audio-K, and no-rope video QK all fuse.

## Two host-side L1/chunk-sizing bugs — now FIXED

These were tuned for Wan's 40-tile-col / broadcast-RoPE shapes; LTX's wider features
(64 cols) and per-head RoPE exposed them. Fixed in `compute_sizing` (single source of
truth for chunk/window/page sizing, so the program and the caller-allocated stats buffer
agree); `create_stats_buffer` now forwards weight/RoPE + uses `fp32_dest_acc=true`.

**Bug A — per-head RoPE overflowed L1 at feat ≥ 1024.** The per-head cos/sin CBs are
`chunk × num_tile_cols` fp32 tiles, uncounted by the L1 estimate; `chunk_h_cap=4` made the
resident path ~1.86 MB. **Fix:** per-head RoPE now uses `chunk_size_rows=1` resident — one
row of cos/sin fits feat ≤ 1024, and chunk-1 also dodges a per-head-RoPE deadlock that
appears at chunk ≥ 2 with many rows. ⟹ video self-attn QK (feat 1024) and A↔V video-Q/K
now fuse.

**Bug B — streaming path asserted `chunk_size_rows==1` for wide features** (feat 2048 →
`chunk_h_cap=2`). **Fix:** `compute_sizing` clamps `chunk_size_rows=1` whenever streaming
is selected. ⟹ TP=2 wide-feature video block / text-cross now fuse.

### Measured after the fix (fused µs, WH galaxy LINE, TP=4 — all 14 configs fuse)
| config | pattern | feat | rows | fused µs |
|---|---|---:|---:|---:|
| v_block_s1 / s2 | block+adaLN | 1024 | 1216/4864 | 129 / 190 |
| a_block | block+adaLN | 512 | 32 | 25 |
| v_selfattn_qk_s1 / s2 | qk + per-head rope | 1024 | 1216/4864 | 177 / 538 |
| a_selfattn_qk | qk + per-head rope | 512 | 32 | 31 |
| a2v_videoQ_s1 / s2 | qk + per-head rope | 512 | 1216/4864 | 110 / 291 |
| a2v_audioK | qk + per-head rope | 512 | 256 | 55 |
| v_textcross_q_s1/s2/k | qk (no rope) | 1024 | 1216/4864/1024 | 113 / 183 / 78 |
| a_textcross_q / k | qk (no rope) | 512 | 32/1024 | 23 / 67 |

(Speedup vs baseline not shown — the unfused baseline `rotary_embedding_llama` at large
rows is slow/flaky on this galaxy; these are fused timings.) Wan is unaffected
(correctness PCC 0.99998).

## Remaining limitation
**feat-2048 per-head RoPE** (TP=2 video self-attn + A↔V video-Q/K) still exceeds L1 even
at chunk 1 — the 64-wide fp32 cos/sin (512 KB) + intermediate/rotated don't fit. It now
fails as a **clean compile-time OOM** (not a hang). Fitting it needs **cos/sin streaming**
(read per block rather than holding the whole row) — a larger kernel change, tracked as a
follow-up. All other LTX configs fuse.

## Note on the benchmark
The *baseline* (composite norm + standalone `rotary_embedding_llama`) at large row counts
can be slow / flaky on this galaxy; the bench catches per-config failures and keeps
sweeping. Run fused-only via `LTX_BENCH_METHODS=fused`. The fused op's correctness (incl.
per-head rope) is covered by
`test_wan_fused_distributed_rmsnorm_device_op.py::test_wan_fused_distributed_rmsnorm_tp1_rope`.
