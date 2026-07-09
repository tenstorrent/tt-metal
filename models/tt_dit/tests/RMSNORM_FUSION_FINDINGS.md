# Distributed RMSNorm fusion (Wan2.2 + LTX-2.3 AV + FLUX): findings & speedups

Single source of truth for the fused distributed-RMSNorm op
(`ttnn.experimental.wan_fused_distributed_rmsnorm`) — one device program that does
pre-allgather (sum-of-squares) → TP all-gather of partial stats → post-allgather
(norm + weight/bias + per-head/broadcast RoPE), folding the trailing affine / adaLN
`addcmul` / RoPE into the post step. Both production models drive the same op.

**Test + bench:** `models/tt_dit/tests/test_distributed_rmsnorm_fused.py` (one parametrized
file, both models). `test_bench` = traced baseline-vs-fused timing (+CSV); `test_corr_det`
= fused vs fp32-PyTorch ref AND vs on-device composite, plus 10× bit-exact determinism.
**Spec / module mapping:** `distributed_rmsnorm_av.md` (which LTX modules fuse, shapes).

**Adding a config:** one row in `_WAN_RAW` (Wan: `cid, seq_len, use_rope`), one `mk(...)`
line (LTX), or one entry in the `_make_cfgs` FLUX branch → flows into perf + PCC +
determinism automatically. A new mesh/topology = one row in `_BENCH_PARAMS`/`_CORR_PARAMS`
`(mesh, device_params, model, tp, topology, links, tp_axis, full_mesh)` — `full_mesh=True`
keeps the whole 2D mesh with TP on axis 1 (the 8-wide closed ring), used by FLUX TP=8 RING.
Isolate with `RMS_BENCH_ONLY=`/`CORR_ONLY=`; pick methods with `RMS_BENCH_METHODS=`.

Baseline = composite RMSNorm (`use_device_op=False`) + the *unfused* trailing op LTX uses
today (`ttnn.addcmul` for adaLN, standalone `rotary_embedding_llama` for RoPE); for Wan the
composite C++ op fuses weight+RoPE in-op. Fused = the single device op.

---

## Blackhole port (TARGET) — Wormhole was the proxy

The shipping target is **Blackhole galaxy**; all numbers below were gathered on **Wormhole
galaxy** as a stand-in. Hardware deltas to account for on BH:

| | Wormhole galaxy (proxy) | Blackhole galaxy (target) |
|---|---|---|
| fabric links | 4 | **2** |
| compute grid | 8×9 (72 cores) | **12×10 (120 cores)** |
| topology | mesh/ring | **4×8 torus** |
| mem-BW / FLOP | — | different (changes the compute-bound balance) |

**Carries over unchanged:** the fabric-forwarder AG, the grid-derived row-aligned worker cap
(auto-adapts: `floor((grid−links)/grid.x)·grid.x`), the unconditional-fp32-internals invariant,
and the whole op/test API.

**BH porting checklist:**
1. **Links:** run with `WAN_GALAXY_LINKS=2` (forwarder count = `min(links, workers)` = 2).
2. **Worker cap:** the grid-derived cap yields ~108 on 12×10 — but the WH "64 = 8×8 rows"
   optimum was *geometric* and won't transfer. **Verify** the row-alignment orientation
   (`grid.x` = the row-major fill dim) and **re-sweep** the optimum via `WAN_RMSNORM_WORKER_CAP`
   / `WAN_RMSNORM_FORCE_WORKERS` — don't assume 108.
3. **Topology:** the AG uses `Ring` (fwd+bwd mcast); a torus's TP axis is a closed ring, so it
   should map — verify the fabric `num_targets` routing on the wraparound.
4. **Re-sweep / re-ablate:** `chunk=1` and the "compute-bound" finding are WH-only — re-run the
   chunk sweep (`WAN_RMSNORM_FORCE_CHUNK`) and the ablation (`WAN_ABLATION`) on BH; the I/O vs
   fabric vs compute balance will shift with BH's BW/FLOP ratio.

**Headline speedup vs composite baseline (fill BH as you go):**

| representative shape | WH ↑ (proxy) | BH ↑ (target) |
|---|---:|:--:|
| flux_tp8_N16384 (TP=8 large) | 2.34× | _TBD_ |
| self_sp4_N18944 (WAN large) | 1.82× | _TBD_ |
| LTX v_block_s2 | 2.71× | _TBD_ |
| flux_tp4_N8192 | 1.63× | _TBD_ |
| (mid/small) | 1.2–1.6× | _TBD_ |

Detailed per-shape WH numbers: `REBENCH_baseline_vs_fused.md` (perf) and `ABLATION_LEVERS.md`
(component breakdown) — both have a Wormhole section and a Blackhole stub to fill.

---

## Speedups — TP=4 on WH Galaxy (4×8), 4 links

LINE = 1×4 submesh; RING = full 4×8 mesh, TP on the closed 4-axis, replicate the 8-axis
(the production galaxy config, `distributed_rmsnorm_av.md` §0). Times are µs/iter (traced);
↑ = baseline/fused speedup. LINE and RING are within run-to-run noise of each other —
topology only changes fabric routing, not the math.

**Correctness + determinism (both topologies):** all configs pass — Wan 7/7 and LTX 14/14
`det=OK` (0/9 over 10 fresh-pob runs, bit-exact), `pcc(fused:torch)` 99.99–100%,
`pcc(fused:composite)` ≈100%.

### Wan2.2 14B (feat/dev 1280, head_dim 128, broadcast RoPE)

| config | rows | pattern | LINE base | LINE fused | LINE ↑ | RING base | RING fused | RING ↑ |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| self_sp4_N18944 | 18944 | qk+rope | 1267.55 | 934.35 | **1.36×** | 1154.84 | 897.93 | **1.29×** |
| self_sp8_N9472 | 9472 | qk+rope | 615.30 | 518.75 | 1.19× | 572.98 | 505.71 | 1.13× |
| self_sp32_N2368 | 2368 | qk+rope | 196.78 | 190.77 | 1.03× | 187.49 | 191.49 | 0.98× |
| cross_q_sp4_N18944 | 18944 | qk | 1060.46 | 604.21 | **1.76×** | 944.48 | 584.56 | **1.62×** |
| cross_q_sp8_N9472 | 9472 | qk | 527.53 | 345.64 | **1.53×** | 472.84 | 335.90 | **1.41×** |
| cross_q_sp32_N2368 | 2368 | qk | 151.05 | 140.72 | 1.07× | 141.48 | 141.57 | 1.00× |
| cross_k_prompt_L512 | 512 | qk | 75.53 | 64.09 | 1.18× | 73.69 | 67.20 | 1.10× |

### LTX-2.3 AV (video feat/dev 1024 hd 128, audio 512 hd 64, per-head RoPE)

| config | rows | pattern | LINE base | LINE fused | LINE ↑ | RING base | RING fused | RING ↑ |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| v_block_s1 | 1216 | block+adaLN | 143.63 | 108.50 | **1.32×** | 139.86 | 110.35 | **1.27×** |
| v_block_s2 | 4864 | block+adaLN | 458.41 | 210.41 | **2.18×** | 432.84 | 207.02 | **2.09×** |
| a_block | 32 | block+adaLN | 34.95 | 24.80 | **1.41×** | 33.70 | 24.93 | **1.35×** |
| v_selfattn_qk_s1 | 1216 | qk+rope | 149.60 | 125.53 | **1.19×** | 145.28 | 128.45 | **1.13×** |
| v_selfattn_qk_s2 | 4864 | qk+rope | 476.94 | 258.56 | **1.84×** | 453.22 | 258.78 | **1.75×** |
| a_selfattn_qk | 32 | qk+rope | 52.95 | 29.55 | **1.79×** | 51.86 | 29.58 | **1.75×** |
| a2v_videoQ_s1 | 1216 | qk+rope | 111.17 | 100.82 | **1.10×** | 107.02 | 104.69 | 1.02× |
| a2v_videoQ_s2 | 4864 | qk+rope | 323.70 | 186.19 | **1.74×** | 289.68 | 183.40 | **1.58×** |
| a2v_audioK | 256 | qk+rope | 80.01 | 54.69 | **1.46×** | 80.11 | 57.68 | **1.39×** |
| v_textcross_q_s1 | 1216 | qk | 93.44 | 100.41 | 0.93× | 89.05 | 101.90 | 0.87× |
| v_textcross_q_s2 | 4864 | qk | 274.62 | 190.76 | **1.44×** | 247.11 | 186.52 | **1.32×** |
| v_textcross_k | 1024 | qk | 85.71 | 81.31 | **1.05×** | 82.70 | 84.74 | 0.98× |
| a_textcross_q | 32 | qk | 33.89 | 22.77 | **1.49×** | 32.93 | 22.85 | **1.44×** |
| a_textcross_k | 1024 | qk | 75.45 | 70.24 | **1.07×** | 72.19 | 74.48 | 0.97× |

**Takeaways:** fused wins biggest on large-token configs (LTX `v_block_s2` ~2.1×,
`v_selfattn_qk_s2` ~1.8×; Wan `cross_q_sp4` ~1.7×). Small/dispatch-bound configs (≤2368
rows) hover near 1.0×, and a couple of short no-trailing-op QK configs are a slight wash
(0.87–0.98×) — there the fused op has marginally more setup than a bare composite norm with
nothing to fold in.

> **Full-mesh RING memory:** RING runs on all 32 devices (4 TP × 8 replicas), ~8× the
> buffer/trace pressure of the 4-device LINE submesh. On the (flaky) galaxy a long traced
> ring sweep can trip a `system_memory_manager` throw mid-run; gather big sweeps in small
> `RMS_BENCH_ONLY` batches with `tt-smi -glx_reset` between each. Per-config error isolation
> in `test_bench` keeps the table from being lost when one config fails.

---

## FLUX shapes (full dim 6144, 48 heads, head_dim 128, broadcast RoPE)

RING only (no LINE bench). TP=4 RING = full 4×8 mesh, TP on the closed 4-axis (feat/dev
1536, 12 heads/dev), replicate the 8-axis. TP=8 RING = full 4×8 mesh, TP on the closed
8-axis (feat/dev 768, 6 heads/dev), replicate the 4-axis. Baseline = Wan-style composite
(weight+RoPE fused in-op); fused = the single device op. µs/iter (traced, 4 links).

| config | TP | rows | pattern | base | fused | ↑ |
|---|---:|---:|---|---:|---:|---:|
| flux_tp4_N512  | 4 | 512   | qk+rope | 117.99 | 92.07  | **1.28×** |
| flux_tp4_N64   | 4 | 64    | qk+rope | 92.90  | 104.73 | 0.89× |
| flux_tp4_N2048 | 4 | 2048  | qk+rope | 175.17 | 168.15 | 1.04× |
| flux_tp4_N8192 | 4 | 8192  | qk+rope | 577.97 | 483.43 | **1.20×** |
| flux_tp8_N1024 | 8 | 1024  | qk+rope | 123.07 | 107.70 | **1.14×** |
| flux_tp8_N128  | 8 | 128   | qk+rope | 96.55  | 64.33  | **1.50×** |
| flux_tp8_N4096 | 8 | 4096  | qk+rope | 279.75 | 255.43 | 1.10× |
| flux_tp8_N16384| 8 | 16384 | qk+rope | 953.79 | 780.48 | **1.22×** |

**Correctness + determinism:** all 8 `per_head_norm=False` configs pass on both TP=4 and
TP=8 RING — `det=OK` (0/9, bit-exact), `pcc(fused:torch)` 99.985–100.00%, flagged NONE.
Same big-shape-wins / small-shape-wash pattern as Wan/LTX (the N=64 TP=4 config is the one
slight regression at 0.89×, a dispatch-bound shape with little to fold).

> **`per_head_norm=True` (FLUX.2 QK-norm) on `ring_size>1` — FIXED (2026-06-23).**
> All 8 FLUX TP=4/TP=8 ring `per_head_norm=True` configs now run deterministically
> (det=OK, PCC 99.81–100.00% vs fp32 torch), and run by default.
>
> *Real root cause:* NOT the matmul-reduce → pack wedge the "reduce fan-out" theory
> assumed (every LLK-level workaround failed because the LLK was never the problem).
> The program factory passed the compute kernel an `is_tp_1` compile-time arg of
> `(ring_size==1)` only, while the writer got `(ring_size==1) || per_head_norm`. So
> for `per_head_norm && ring_size>1` the compute took the `is_tp_1==0` branch →
> `stats_dest_cb = stats_local_cb`, which for per-head is sized **1 tile** with **no
> consumer** (the writer is drain-only). PRE produces `num_heads` stat tiles/row, so
> it blocked on `cb_reserve_back` at the **2nd head's** reduce — read as a "reduce wedge".
>
> *Fix:* set compute's `is_tp_1` arg to the factory-level `is_tp_1` (incl. `per_head_norm`).
> PRE then pushes the `num_heads` per-row stat tiles straight into `stats_gathered_cb`
> (sized for `num_heads`) and POST consumes them locally — self-contained, no AG, matching
> the drain-only writer. One line in the program factory.

---

## Tuning knobs & heuristic (workers, chunks)

Current heuristic (committed): `num_workers = min(tile_rows, derive_worker_cap(grid, links))`
where the cap = the compute grid minus the forwarder cores, **rounded down to whole grid rows**
(`floor((grid.x·grid.y − links)/grid.x)·grid.x`; = **64** on the WH 8×9 galaxy). Workers tile
complete rows for NoC locality; using the full grid (ragged final row) regresses 3–9%. The cap
is **device-derived (already arch-adaptive)** — no hardcoded constant. `chunk_size_rows = 1`
(`kMaxChunkSizeRows`): a Wormhole sweep (chunk 1–4) found chunk=1 best-or-tied everywhere.
The worker cap was +20–44% over the old hardcoded-32 cap on large all-gather-bound shapes.

> **Blackhole caveat:** the `chunk=1` choice is **Wormhole-only**, and the row-aligned worker
> cap (geometric "64 = 8×8" knee) needs re-validation on BH's 12×10 grid. BH's different
> mem-BW/FLOP ratio can flip the chunk>1 tradeoff — **re-sweep, don't assume.** Sweep with no
> code change via `WAN_RMSNORM_FORCE_WORKERS`, `WAN_RMSNORM_WORKER_CAP`, `WAN_RMSNORM_FORCE_CHUNK`
> (all read through the single sizing path so buffer + kernel stay consistent). The cap formula
> already adapts to the BH grid; only `kMaxChunkSizeRows` would need arch-conditioning if BH
> prefers chunk>1.

A split-sender ring AG (each worker one full-wrap mcast instead of two arc mcasts) was
tried and **reverted** — essentially neutral (only the largest no-RoPE shape gained ~3%)
and more fragile under traced replay. Dual-direction arc AG is the only path.

---

## Feature support — nothing missing at the API level
- **adaLN addcmul** `normed·(1+scale)+shift` → `weight=(1+scale)` + `bias=shift` (broadcast).
- **static QK affine** → `weight`.
- **create_heads** → `num_heads_per_device` (emits BHNE).
- **per-head RoPE** `(1,H,N,head_dim)` → auto-detected (`rope_cos.shape[1]==num_heads_per_device`).
- whole-row norm + per-head rope → `per_head_norm=False` + per-head rope (independent flags).

---

## Resolved gotchas & current limits
- **chunk=1 AG uninitialized-read race (FIXED, `0ef362b10e6`):** the packed-page fused
  write+atomic_inc used `flush=false`, so the receiver's inc could beat the payload commit
  → sem-gated read of uninit DRAM (intermittent non-determinism on large no-RoPE shapes).
  `flush=true` orders write-before-inc.
- **stats-buffer / kernel sizing mismatch (FIXED):** `create_stats_buffer` and the program
  factory must agree on the `num_links` worker rounding and chunk size; the buffer is
  otherwise chip-global (num_workers-independent). Pass weight/RoPE + `num_links` to the buffer.
- **TP=2 multi-chunk hang — two fixes:**
  - *Compute/MUX side (FIXED, `11bc6a0e056`):* the matmul stats-reduce wedged the packer at
    `ring_size>1` × multi-chunk; replaced with an FPU eltwise-add (requires even `ring_size`
    = TP, always 2/4/8).
  - *Legacy single-worker writer side (FIXED, `34c606ef7fe`):* small TP=2 shapes
    (≤2 tile-rows → `num_workers=1` → legacy direct-fabric writer) hit a second hang. Two
    coupled bugs: (a) `stats_gathered_cb` was sized `chunk_size_rows*ring_size` but the legacy
    writer reserves `num_tile_rows*ring_size` → `cb_reserve_back` blocked; (b) it forwarded all
    rows then pushed gathered stats once at the end, deadlocking against the compute's
    per-chunk PRE/POST interleave. Now sized for `num_tile_rows` on the legacy path and the
    writer releases each row's slots as its gather completes (+ `flush=true`).
- **Host L1/chunk-sizing bugs (FIXED):** LTX's wider features + per-head RoPE exposed two
  clamps; `chunk_size_rows=1` is forced for per-head RoPE (keeps cos/sin resident, fits feat
  1024) and the streaming-low-L1 fallback. Applied identically in `compute_sizing` and the
  factory so the buffer matches; **Wan is byte-identically unaffected**.
- **feat-2048 per-head RoPE → L1 OOM (FIXED, `222870958b5` + `df735764a5b`):** the TP=2 video
  self-attn QK norms (`tp2_v_selfattn_qk_s1/s2`; video dim 4096 → feat/dev **2048**, head_dim
  128, 16 heads/dev). The sub-phase-major POST keeps `intermediate_cb` + `rotated_input_cb`
  resident for a whole row, which overflowed L1 (1,593,632 B > 1,499,136 B). Fix: when the
  resident estimate would overflow (per-head only), use a **conditional block-major POST** —
  fuse the matmul-rotate + RoPE finalize per block so `rotated_input_cb` is block-local — and
  **stream cos/sin** (cap the CB at a few `block_size` groups). Both gated on the overflow
  estimate, so the resident fast path (all TP=4 shapes) is untouched. The block-major POST is
  auto-selected on the L1-overflow estimate; `WAN_RMSNORM_ROPE_STREAM_BLOCKS` tunes the cos/sin
  streaming depth.
- **Underlying — per-head RoPE chunk≥2** compute path is avoided by pinning chunk=1; the
  deeper chunk≥2 deadlock isn't separately fixed (no need at chunk=1).

**TP=2 status: now fully correct.** With the fixes above, the entire LTX TP=2 (SP=4) sweep
passes — **14/14 `det=OK`, PCC 99.99–100% vs PyTorch**, incl. the previously-OOM video
self-attn (`v_selfattn_qk`) and the previously-hanging small-audio configs (`a_block`,
`a_selfattn_qk`, `a_textcross_q`). Wan TP=2 is 7/7. Only *correctness/determinism* is
validated at TP=2; perf isn't benchmarked there (TP=4 is the shipping config, so the speedup
tables above stay TP=4-only).

The fused op's per-head-rope correctness also has a standalone regression test:
`test_wan_fused_distributed_rmsnorm_device_op.py::...tp1_rope`.
