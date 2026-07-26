# LTX-2.3 TP×SP on BH Galaxy 4×8 — slide narrative

Deliverables in this dir: `ltx_tpsp_report.html` (interactive, published as an Artifact),
`ltx_tpsp_stage_1.png` / `ltx_tpsp_stage_2.png` (slide drop-ins), model/analysis scripts.

Status: the **(4,8) anchor is measured**; all other splits are the roofline **calibrated through
that anchor** (labeled on the charts). The measured sweep is blocked on a Galaxy `-glx_reset`
(eth core wedged at `open_mesh_device`); harness is ready to fill in the remaining points.

---

## Slide 1 — The decision & the constraint
- 48-layer DiT denoiser; per-layer × 48 ≈ denoiser step. Measure **one layer**.
- CFG=1 ⇒ **TP · SP = 32** fixed. One knob: the split, plus which physical torus axis carries SP.
- **Shipped: TP=4 / SP=8.**

## Slide 2 — Method (answers "how we measured")
- Reuse the block unit test → one video-only `LTXTransformerBlock`, random weights, signposted forward, profiled per-op.
- Bucket per-op device time → **matmul+TP-CCL / ring-attention / overhead**. One trace gives total *and* breakdown (buckets sum to total, overlap already accounted).
- Isolation microbenches (matmul short-sweep, CCL AG/RS by axis, ring-SDPA by SP) explain the curve; they're the cheaper path than 6× full-pipeline runs.

## Slide 3 — The measured anchor (2 facts)
- **Ring attention's share 18% → 31% stage1→stage2.** Attention O(N²), matmuls O(N) ⇒ at HD the SP collective is what to optimize.
- **Fused-op FLOP util splits compute from comm:** AllGather-matmul ~5% util (comm-bound), ReduceScatter-matmul ~53% (compute), RingJointSDPA transport-bound. This is the calibration hook.

## Slide 4 — The tradeoff curve  (`ltx_tpsp_stage_2.png`)
- U-shaped. Floor at **TP=4/SP=8** for stage 2. Stage 1 floor at TP=2/SP=16, shipped point +9%.
- Left of floor: ring attention (orange) explodes. Right of floor: matmul+TP-CCL (blue) explodes.
- **1-D extremes lose on both** — exactly the "thin shard starves matmul, fat ring inflates collectives" point.

## Slide 5 — Why (the physics that makes it robust)
- **Per-device compute is invariant to the split** (matmul `N·W/32`, attention `N²·hd` — TP cancels both). So it's *not* about FLOPs.
- Two opposing collectives: **TP all-gather ∝ N·TP** (↑ with TP), **SP ring K/V-gather ∝ N/TP** (↑ with SP). Sum ⇒ U, floor at `TP*≈√(k_sp/k_tp)`.
- `k_sp/k_tp` bigger at longer N ⇒ optimum TP rises stage1(2)→stage2(4). Shipping the stage-2-optimal split is correct because stage 2 dominates wall-clock.
- `TP=32/SP=1` is **architecturally invalid** — SP=1 can't mask padded keys (needs ring `logical_n`). It can't run, not just slow.

## Slide 6 — Topology (answers "how the 2-D torus affects it")
- BH Galaxy = native **8×4 2-D torus**: wraparound both axes, **4 eth links/pair, 200 Gbps bi-dir each**.
- **Ring axis sustains both directions ≈ 2× linear BW** → LTX uses `FABRIC_1D_RING` + `Topology.Ring`.
- **Shorter ring = fewer hops** (len-4 → 2 hops, len-8 → 4, len-32 → 16).
- Placement rule: heaviest/most-frequent collective on the axis that serves it best. The **K/V ring-gather runs every attention and moves the most bytes → put SP on the long (len-8) axis for bandwidth**; TP's smaller fused all-gather tolerates the short (len-4) axis. → *matmul-best is one orientation, attention-best the other.* Shipped `sp_axis=1(8), tp_axis=0(4)` is the attention-favoring choice.

## Slide 7 — Verdict
- Optimal at stage 2 (dominant), ~9% off at stage 1, one config for both (no re-shard).
- Balanced 2-D shard keeps every op near its compute-bound floor.
- Rides the torus: dominant K/V ring on the long, double-bandwidth axis.

---

## Open items the measured sweep closes (post-reset)
1. Absolute crossover position (model's compute:comm split assumption → confirm k_sp/k_tp).
2. SP-on-len-8 (bandwidth) vs SP-on-len-4 (fewer hops) — the `*_altaxis` harness rows measure this directly.
3. Untuned-chunk penalty magnitude for non-(4,8) — currently these are conservatively "untuned*".
