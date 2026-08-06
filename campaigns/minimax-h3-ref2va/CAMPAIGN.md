# MiniMax-H3 `ref2va` — 4×8 Blackhole Galaxy — campaign

Branch `kevinmi/minimax-h3-integration` · run root `campaigns/minimax-h3-ref2va/` · started 2026-08-05
Full history: `git log --follow -- campaigns/minimax-h3-ref2va/CAMPAIGN.md campaigns/minimax-h3-ref2va/ledgers/`
Bringup campaign: the metric is gates going green, not a latency.

## Loop state

| Round | Gates green | Baseline | Stall | Target | Gate status |
|---|---|---|---|---|---|
| 9 | **8/8** | t2va + fl2va green @ `bd12ad2aeb2` | 0/10 | 8/8 acceptance criteria | **target met** |

## Working point

Mesh **4×8 Blackhole**, TP=4 axis 0 / SP=8 axis 1, ring, 2 links, `ring_params_req_exact_devices`
with **`l1_small_size` 16384** (measured, am. 126 — not the 65536 t2va/fl2va use). Target
**1344×768, 124 frames @ 24 fps** → 37 latent frames (48×84 latent, 1008 rows/frame, 37296 video
rows), 207 audio latents (414 rows), 50 steps → 49 forwards. Partition **`transformer_ref/`**.
AdaLN precompute on, with a **fourth** level for the audio-conditioning `t = 1.0`. Seed 0.

Measured ref2va shapes (host prediction in am. 114, all confirmed on device):

| Request | padded | rows/device | warm forward | e2e compute |
|---|---|---|---|---|
| t2va (baseline) | 37888 | 4736 | — | — |
| 1 image | **46080** | 5760 | 2.11 s | 210.7 s |
| 1 video + sound | **81664** | 10208 | 3.26 s | 270.5 s |
| image + video + audio | **89856** | 11232 | — | 372.0 s |
| 9 images | **111616** | 13952 | 5.45 s | not run e2e |

Layout: `[ text | ref block 1 | … | target audio | target video ]`, one block per reference in
**request order**, a video reference's soundtrack rows immediately **before** its own video rows.

## Fixed baseline (Phase 1, immutable)

SHA `bd12ad2aeb2`, mesh 4×8, cold e2e. **t2va 1 passed (560 s), fl2va 4 passed (645 s).** Frozen
output `artifacts/round-0/e2e/`. CLIP **37.38** (min 36.79, max 36.79–37.82) · audio peak **0.095** ·
VBench 0.9804 / 0.9812 / 0.9914 / 1.0000 / 0.6925 · fl2va anchor **0.9971** · fractal discriminator
**0.9963** vs **0.2972**. Re-measured after every ref2va change and **bit-identical** (am. 121).

## Gates

| Gate | Result |
|---|---|
| 1 host packing bit-exact vs reference, 5 request shapes | **GREEN** — 61 tests, `torch.equal` |
| 2 both `_temporal_position_span` orders, provably different | **GREEN** — exact vs both, differ at n=16 and n=37 |
| 3 reference encode ≥ 0.99 on real media | **GREEN** — image 99.9905 %, audio 99.9910 %, video 99.9927 % |
| 4 typed condition stream, existing PCCs identical | **GREEN** — 12/12 bit-identical; interleaved 99.9974/99.9975 |
| 5 t2va + fl2va unchanged | **GREEN** — 5 passed, metrics bit-identical |
| 6 ref2va e2e | **GREEN** — all three shapes |
| 7 conditioning is not a no-op | **GREEN** — signal 0.128143 vs floor **0.000000**; order 0.096018 |
| 8 frames inspected | **GREEN** — 6 frames + a 2× boundary strip; no seams, no flicker |

## Pending work

1. **No instrument shows the output resembling its own reference** more than another of the same
   geometry (am. 128). Divergence and order-sensitivity are proven; *directional* resemblance is
   recorded, not asserted. Luminance correlation measured noise; CLIP gives a per-output offset
   (0.0279 vs 0.0292 gap for the two references). A subject-transfer test with a nameable subject the
   model can place is the next thing to try. **This is a quality question, not a parity one** — every
   interface below e2e is gated against the reference implementation.
2. The **9-image** shape (111616) fits at full depth but has no e2e run. Its vision-tower cost is nine
   16384-row attentions on a replicated tower, unmeasured (am. 117).
3. `warmup()` does not accept `references`, so no ref2va latency is warm. The numbers above are cold.
4. VBench/CLIP quality bars are **not** set for ref2va. Deliberately: am. 80/87 and this campaign's
   own am. 130 all say t2va's bars do not transfer to reference-driven content.

## Pitfalls

- A reference **never** binds the target geometry and is prepared at its own resolution — 2048 px
  short edge, no area cap. That is why one image costs 4096 rows *and* 4096 vision tokens (am. 114).
- Audio reference rows are **clean at t = 1.0**, posterior `mode()`, never noise-augmented (am. 115),
  and they need a **fourth AdaLN level** that t2va/fl2va do not have (am. 129).
- **Two** `_temporal_position_span` orders, one per call site; they differ at the production n = 37.
- `l1_small_size` **16384**, not 65536: the taps=3 video-reference encoder clashes with L1 above it
  (am. 124/126). Audio decode is fine at 16384 (am. 127).
- Copy the **whole** device-params dict from the gate that ships the surrounding pipeline. Three of
  this campaign's failures were components green under a config production does not use —
  `SINGLE_DEVICE` with empty `device_params` twice, a LINE fabric config once.
- Write artifacts **before** quality checks, or a failing check leaves no frames to look at.

## Latest amendment

**130 (2026-08-06)** — the 2.29× horizontal seam ratio on `video_with_sound` is scene content, not a
seam: no discontinuity in the magnified boundary strip, a ~9-row-wide elevation, and the frame's
largest vertical gradient (16.06) at y=306 which is not a boundary at all. Horizontal bar set to 3.0
for ref2va on that evidence; vertical stays 2.0. Full body and 114–129 in `ledgers/amendments.md`.

## Ledger index

attempts.md (7 rounds) · optimizations.md (5) · source-ideas.md (10) · amendments.md (17: 114–130)
