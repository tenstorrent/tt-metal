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

## Two host-side L1/chunk-sizing bugs — FIXED (chunk clamp)

Both were tuned for Wan's 40-tile-col / broadcast-RoPE shapes; LTX's wider features
(64 cols) and per-head RoPE exposed them. Fixed by clamping `chunk_size_rows=1` for
per-head RoPE (resident — NOT streaming) and for the streaming-low-L1 fallback, applied
identically in `compute_sizing` (buffer) and the program factory; `create_stats_buffer`
forwards weight/RoPE + uses `fp32_dest_acc=true` so the buffer's window/pages match.
Crucially the clamp is applied ONLY to those two cases — every other shape keeps the
exact original chunk, so **Wan is byte-identically unaffected** (PCC 0.999991, perf
unchanged). All 14 TP=4 LTX configs fuse (see table above; block 106/191/25, self-attn
176/541/31, A↔V 110/290/56, text-cross 94/183/79/23/67 µs).

- **Bug A** (per-head RoPE OOM at feat ≥ 1024): chunk-1 resident keeps cos/sin to one row
  → feat 1024 fits, and chunk-1 also dodges the per-head-RoPE chunk≥2 deadlock.
- **Bug B** (streaming `chunk_size_rows==1` FATAL at feat 2048): clamp to 1 when streaming.

## Known software bugs still open (NOT machine flakiness — hangs during kernel exec)
1. **Per-head RoPE deadlocks at chunk ≥ 2 + many rows** (worked around by forcing chunk-1;
   the underlying compute deadlock remains).
2. **`tp2_a_selfattn_qk` hangs even at chunk-1** — feat 1024 per-head RoPE on the TP=2
   (ring_size=2), 2-tile-row, single-worker legacy-writer path. A distinct per-head-RoPE
   execution hang specific to that path; needs `run_safe_pytest.sh --dev` + tt-triage.
3. **feat-2048 per-head RoPE** (TP=2 video self-attn / A↔V video) exceeds L1 even at
   chunk-1 → clean compile-time OOM; needs cos/sin streaming (a kernel change).
4. The **composite baseline** (`use_device_op=False` norm + standalone
   `rotary_embedding_llama`) hangs at large rows (video stage-2) — a bug in that path,
   which is why the LTX unfused-baseline column isn't cleanly measurable yet.

The fused op's correctness (incl. per-head rope) is covered by
`test_wan_fused_distributed_rmsnorm_device_op.py::test_wan_fused_distributed_rmsnorm_tp1_rope`.
