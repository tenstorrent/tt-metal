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

## Blocked by two host-side L1/chunk-sizing bugs (NOT kernel-capability gaps)

**Bug A — per-head RoPE overflows L1 at feat ≥ 1024.** The per-head cos/sin CBs are
`chunk × num_tile_cols`-sized (fp32), but `decide_streaming_low_l1()` estimates resident
L1 with a flat "~192 KB of small CBs" and never counts them. With `chunk_h_cap =
128/num_tile_cols` (=4 at feat 1024) the resident path allocates ~1.86 MB > 1.5 MB.
Blocks: **video self-attn QK** (feat 1024, head_dim 128) at both stages, and the
**A↔V video-Q/K at TP=2** (feat 1024). Fix: include per-head cos/sin (chunk-scaled) in
the L1 estimate, and/or cap chunk for per-head rope.

**Bug B — streaming path asserts `chunk_size_rows==1` for wide features.** At feat 2048
(TP=2 video, 64 tile-cols), `chunk_h_cap = 128/64 = 2`, but the streaming-low-L1 path
requires `chunk_size_rows==1` → `TT_FATAL`. Blocks **all TP=2 video configs** (block /
self-attn / text-cross, regardless of RoPE). Fix: clamp `chunk_size_rows=1` when
`streaming_low_l1` is selected.

Both stem from the op being tuned for Wan's 40-tile-col / broadcast-RoPE shapes; LTX's
wider features (64 cols) and per-head RoPE expose the gaps.

## Note on the benchmark
The *baseline* (composite norm + standalone `rotary_embedding_llama`) at large row counts
(video stage-2, rows 4864) can be slow / flaky on this galaxy; the bench catches per-config
failures and keeps sweeping. The fused op's correctness (incl. per-head rope) is covered by
`test_wan_fused_distributed_rmsnorm_device_op.py::test_wan_fused_distributed_rmsnorm_tp1_rope`.
