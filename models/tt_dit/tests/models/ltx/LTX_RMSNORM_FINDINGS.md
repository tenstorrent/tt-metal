# DistributedRMSNorm fusion (LTX-2.3 AV + Wan2.2): coverage & findings

Benchmark + correctness suite: `models/tt_dit/tests/test_distributed_rmsnorm_fused.py`
(one parametrized file covering **both** LTX-2.3 AV and Wan2.2 — they drive the same
fused device op). Covers every distinct RMSNorm shape×fusion-pattern from the LTX-2.3 AV
transformer block (see `distributed_rmsnorm_av.md`; 14 configs/TP, 3 fusion patterns) and
the 7 Wan2.2 720p attention call sites. Baseline = composite RMSNorm + the *unfused*
trailing op LTX uses today (`ttnn.addcmul` for adaLN, standalone
`ttnn.experimental.rotary_embedding_llama` for RoPE); for Wan the composite C++ op fuses
weight+RoPE in-op. Fused = the single device op.

## Feature support: nothing missing at the API level
Every LTX fusion opportunity maps onto the existing op — no new kernel capability needed:
- **adaLN addcmul** `normed·(1+scale)+shift` → `weight=(1+scale)` + `bias=shift` (both broadcast). `has_bias` path works.
- **static QK affine** → `weight`.
- **create_heads** → `num_heads_per_device` (emits BHNE).
- **per-head RoPE** `(1,H,N,head_dim)` → auto-detected (`rope_cos.shape[1]==num_heads_per_device`).
- whole-row norm + per-head rope → `per_head_norm=False` + per-head rope (independent flags).

## TP=4 RING — the production galaxy config (full 4×8 mesh, 4 links)
This is the canonical BH-Galaxy config (`distributed_rmsnorm_av.md` §0): open the full
**4×8** mesh, ride TP on the **4-wide axis 0** (a *closed* ring on the galaxy torus — the
only axis with a real wrap link; a 1×4 sub-row of the 8-wide axis is open, hence the
"TP=4 LINE only" constraint that bit us before), and **replicate** activations / weights /
RoPE across the 8-wide axis 1. Run with `test_corr_det[...ring]` / `test_bench[...ring]`.

**Correctness + determinism (Ring):** every config of both models passes — Wan 7/7 and
LTX 14/14 `det=OK` (0/9 over 10 fresh-pob runs, bit-exact), `pcc(fused:torch)` 99.99–100%,
`pcc(fused:composite)` ≈100%. Identical to LINE — topology changes routing, not the math.

**Wan2.2 — TP=4 RING** (`fused-split` = `WAN_RMSNORM_RING_SPLIT=1`, see split-sender section below):

| config | pattern | feat | rows | baseline µs | fused µs | fused-split µs | split vs fused |
|---|---|---:|---:|---:|---:|---:|---:|
| self_sp4_N18944 | qk+rope | 1280 | 18944 | 1154.84 | 897.93 | 897.81 | ~0% |
| self_sp8_N9472 | qk+rope | 1280 | 9472 | 572.98 | 505.71 | 505.32 | ~0% |
| self_sp32_N2368 | qk+rope | 1280 | 2368 | 187.49 | 191.49 | 192.03 | ~0% |
| cross_q_sp4_N18944 | qk | 1280 | 18944 | 944.48 | 604.21 | 585.43 | **−3.1%** |
| cross_q_sp8_N9472 | qk | 1280 | 9472 | 472.84 | 335.90 | 336.14 | ~0% |
| cross_q_sp32_N2368 | qk | 1280 | 2368 | 141.48 | 141.57 | 141.54 | ~0% |
| cross_k_prompt_L512 | qk | 1280 | 512 | 73.69 | 67.20 | 67.17 | ~0% |

(`fused` µs above are the dual-direction default. The earlier cross_q numbers were re-measured here for a same-session comparison.) LTX spot-checks of the two largest AG-exposed configs: `v_block_s2` 207.02→207.03 µs, `v_textcross_q_s2` 186.52→187.25 µs — both ~0%.

**LTX-2.3 — TP=4 RING:**

| config | pattern | feat | hd | rows | baseline µs | fused µs | speedup |
|---|---|---:|---:|---:|---:|---:|---:|
| v_block_s1 | block+adaLN | 1024 | — | 1216 | 139.86 | 110.35 | **1.27×** |
| v_block_s2 | block+adaLN | 1024 | — | 4864 | 432.84 | 207.02 | **2.09×** |
| a_block | block+adaLN | 512 | — | 32 | 33.70 | 24.93 | **1.35×** |
| v_selfattn_qk_s1 | qk+per-head rope | 1024 | 128 | 1216 | 145.28 | 128.45 | **1.13×** |
| v_selfattn_qk_s2 | qk+per-head rope | 1024 | 128 | 4864 | 453.22 | 258.78 | **1.75×** |
| a_selfattn_qk | qk+per-head rope | 512 | 64 | 32 | 51.86 | 29.58 | **1.75×** |
| a2v_videoQ_s1 | qk+per-head rope | 512 | 64 | 1216 | 107.02 | 104.69 | 1.02× |
| a2v_videoQ_s2 | qk+per-head rope | 512 | 64 | 4864 | 289.68 | 183.40 | **1.58×** |
| a2v_audioK | qk+per-head rope | 512 | 64 | 256 | 80.11 | 57.68 | **1.39×** |
| v_textcross_q_s1 | qk (no rope) | 1024 | 128 | 1216 | 89.05 | 101.90 | 0.87× |
| v_textcross_q_s2 | qk (no rope) | 1024 | 128 | 4864 | 247.11 | 186.52 | **1.32×** |
| v_textcross_k | qk (no rope) | 1024 | 128 | 1024 | 82.70 | 84.74 | 0.98× |
| a_textcross_q | qk (no rope) | 512 | 64 | 32 | 32.93 | 22.85 | **1.44×** |
| a_textcross_k | qk (no rope) | 512 | 64 | 1024 | 72.19 | 74.48 | 0.97× |

Takeaways (both topologies): the fused op wins biggest on large-token configs (LTX
`v_block_s2` 2.1×, `v_selfattn_qk_s2` 1.75×; Wan `cross_q_sp4` 1.6×). Small/dispatch-bound
configs (≤2368 rows) hover near 1.0×, and a couple of short no-trailing-op QK configs are
a slight wash (0.87–0.98×) — there the fused op has marginally more setup than a bare
composite norm with nothing to fold in. Per-head-rope no longer regresses (the old chunk≥2
deadlock is dodged by forcing chunk-1; self-attn now wins 1.1–1.75×).

> **NB (full-mesh memory):** the Ring config runs on all 32 devices (4 TP × 8 replicas),
> ~8× the buffer/trace pressure of a 4-device LINE submesh. On the (flaky) galaxy the LTX
> 14-config ring sweep can trip a `system_memory_manager` throw mid-run; the numbers above
> were gathered in small `RMS_BENCH_ONLY` batches with a `tt-smi -glx_reset` between each.

### Split-sender ring AG (`WAN_RMSNORM_RING_SPLIT`) — marginal, kept off by default
Default ring AG: every MUX worker mcasts BOTH directions (fwd arc + bwd arc, each
~⌈(ring−1)/2⌉ hops). Split mode instead partitions a link's workers into fwd-only and
bwd-only halves; each does ONE full-wrap mcast (ring−1 hops, reaching all peers one way
around) and each MUX serves half the channels. Hypothesis: fewer fabric injections per
worker + a shorter MUX channel loop. Confined to the kernel layout (`create_at`); the
writer is unchanged (both `num_targets`=ring−1, per-worker `connection_valid` picks the
direction). **Correctness verified:** Wan TP=4 ring 7/7 `det=OK`, PCC 99.99–100%, identical
to non-split.

**Result: essentially neutral.** Only `cross_q_sp4` (the largest no-RoPE shape, 18944 rows,
where the AG is most exposed) improves, ~3.1% (604→585 µs). Everywhere else it's within
noise: RoPE configs are compute-bound (AG isn't the bottleneck), small configs are
dispatch-bound, and the LTX AG-exposed configs (block / no-rope QK) show 0%. The two wins
are real but the all-gather isn't on the critical path for any shape except the single
biggest. It's also **more fragile**: the longer one-directional full-wrap traffic makes the
traced back-to-back sweep trip the cumulative `system_memory_manager` hang sooner (the split
sweep needed one config per process to complete; the dual-direction sweep tolerated 7).
So split mode stays **off by default**; the flag is retained for future ring-size sweeps
(larger rings = more hops saved by halving the arc, where it may pay off more).

## Baseline vs fused — TP=4 galaxy LINE, 4 links (1×4 submesh)
Same op, LINE topology on a 1×4 submesh (4 devices). Within noise of the RING numbers
above (topology only changes fabric routing). All 14 TP=4 configs fuse.

| config | pattern | feat | hd | rows | baseline µs | fused µs | speedup |
|---|---|---:|---:|---:|---:|---:|---:|
| v_block_s1 | block+adaLN | 1024 | — | 1216 | 143.63 | 108.50 | **1.32×** |
| v_block_s2 | block+adaLN | 1024 | — | 4864 | 458.41 | 210.41 | **2.18×** |
| a_block | block+adaLN | 512 | — | 32 | 34.95 | 24.80 | **1.41×** |
| v_selfattn_qk_s1 | qk+per-head rope | 1024 | 128 | 1216 | 149.60 | 125.53 | **1.19×** |
| v_selfattn_qk_s2 | qk+per-head rope | 1024 | 128 | 4864 | 476.94 | 258.56 | **1.84×** |
| a_selfattn_qk | qk+per-head rope | 512 | 64 | 32 | 52.95 | 29.55 | **1.79×** |
| a2v_videoQ_s1 | qk+per-head rope | 512 | 64 | 1216 | 111.17 | 100.82 | **1.10×** |
| a2v_videoQ_s2 | qk+per-head rope | 512 | 64 | 4864 | 323.70 | 186.19 | **1.74×** |
| a2v_audioK | qk+per-head rope | 512 | 64 | 256 | 80.01 | 54.69 | **1.46×** |
| v_textcross_q_s1 | qk (no rope) | 1024 | 128 | 1216 | 93.44 | 100.41 | 0.93× |
| v_textcross_q_s2 | qk (no rope) | 1024 | 128 | 4864 | 274.62 | 190.76 | **1.44×** |
| v_textcross_k | qk (no rope) | 1024 | 128 | 1024 | 85.71 | 81.31 | **1.05×** |
| a_textcross_q | qk (no rope) | 512 | 64 | 32 | 33.89 | 22.77 | **1.49×** |
| a_textcross_k | qk (no rope) | 512 | 64 | 1024 | 75.45 | 70.24 | **1.07×** |

(TP=2 not tabulated: its per-head-rope configs OOM (feat 2048) or hit the TP=2 hang, bug #2.)

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
