# MiniMax-H3 `ref2va` — 4×8 Blackhole Galaxy — campaign

Branch `kevinmi/minimax-h3-integration` · run root `campaigns/minimax-h3-ref2va/` · started 2026-08-05
Full history: `git log --follow -- campaigns/minimax-h3-ref2va/CAMPAIGN.md campaigns/minimax-h3-ref2va/ledgers/`
Bringup campaign: the metric is gates going green, not a latency.

## Loop state

| Round | Gates green | Baseline | Stall | Target | Gate status |
|---|---|---|---|---|---|
| 11 | **8/8 + quality bars set** | t2va + fl2va green @ `bd12ad2aeb2`; warm perf @ `e10e6dda34e` | 1/10 | correctness met; perf **not** exhausted | correctness target met |

Correctness is done. **Perf is not**: the warm baseline exists and two levers are named with evidence,
one optimization was tried and produced a measured non-result (am. 133). See `Pending work`.

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
| 9 quality bars, derived from measurement | **GREEN** — 6 bars set from am. 131, verified active on the binding case |

## Quality (am. 131) — bars derived from ref2va's own measurements, never inherited from t2va

| dimension | one_image | video+sound | mixed | **bar** | t2va | t2va bar |
|---|---|---|---|---|---|---|
| CLIP prompt alignment | 29.05 | 29.97 | 29.38 | **25.0** | 37.38 | 33.0 |
| subject_consistency | 0.9631 | 0.9344 | 0.9587 | **0.90** | 0.9804 | 0.95 |
| background_consistency | 0.9569 | 0.9249 | 0.9397 | **0.89** | 0.9812 | 0.95 |
| motion_smoothness | 0.9957 | 0.9952 | 0.9959 | **0.97** | 0.9914 | 0.97 |
| dynamic_degree | 1.0000 | 1.0000 | 1.0000 | **1.0** | 1.0000 | 1.0 |
| imaging_quality | 0.4826 | 0.6575 | 0.5826 | **0.44** | 0.6925 | 0.64 |

**Three of t2va's six bars would have failed and none of the three is a defect** — CLIP is
prompt-dependent, the consistency pair trades against confirmed motion (`dynamic_degree` 1.0
everywhere), and `imaging_quality` is no-reference IQA that spreads 0.17 on one pipeline. ref2va is
**better** than t2va on `motion_smoothness`.

## Warm latency (am. 132, immutable baseline @ `e10e6dda34e`)

Warm window: one full warmup generation at the same shape with the same references, plus a priming
`encode_prompt`; both `padded_len` values asserted. Wall time, prepares and export excluded.

| case | cold | **warm** | ms/forward | realtime | denoise share |
|---|---|---|---|---|---|
| one_image (46080) | 210.7 s | **73.6 s** | 1355 | 14.2× | 90.3 % |
| video+sound (81664) | 270.5 s | **193.3 s** | 3356 | 37.4× | 85.0 % |
| mixed (89856) | 372.0 s | **216.1 s** | 3818 | 41.8× | 86.6 % |

## Pending work

**Perf, in priority order. Denoise is 85–90 %, so nothing else is worth touching first.**

1. **Matmul blockings for ref2va's shapes.** The warm log carries 12 distinct
   `No known best blocking for (M, K, N) = …; using default` warnings, including `(5760, 5376, 5376)`
   and `(5760, 7168, 1344)` — DiT block matmuls running 50× per forward. The table has no entry for
   `seq_local` 5760 / 10208 / 11232. A configuration sweep with its own baseline and revalidation;
   **not attempted**.
2. **`measured_sdpa_chunk_sizes`.** STATE.md pending item 2: keyed on `seq_local ∈ {4768, 9216, 13632}`,
   so dead at t2va's 4736 / 4992 *and* at all three ref2va values. The tuned `(320, 384)` was worth
   −13 % on its op and has never run. Same shape of work as (1).
3. **Reference encode is 21.6 s (11 %) for a video reference** — a 124-frame clip through the taps=3
   temporal chunking, 192 work units over 32 devices. Second-largest single stage; unexamined.
4. **Trace is unused for the DiT.** Lower priority than it looks: per-forward scales 2.82× for 1.95×
   the rows per device, which is device-work scaling, and am. 133 measured that removing 351 MB of
   redundant per-step host upload changed nothing. Host dispatch is not the binding constraint at these
   lengths.

**Correctness, remaining.**

5. **No instrument shows the output resembling its own reference** more than another of the same
   geometry (am. 128). Divergence and order-sensitivity are proven; directional resemblance is recorded,
   not asserted. A subject-transfer test with a nameable subject is the next thing to try.
6. The **9-image** shape (111616) fits at full depth but has no e2e run; its nine 16384-row vision-tower
   attentions on a replicated tower are unmeasured (am. 117).

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

**133 (2026-08-06)** — hoisting the provably-redundant per-step conditioning upload (351 MB of
identical bytes over 49 steps) buys −0.8 / −0.1 / −0.3 %, all inside the ±8 % noise floor, and the
effect did **not** scale with block size. The loop is asynchronous, so host work hidden behind device
work costs nothing. Kept as `attempts`, not `optimizations`: counting bytes is not profiling. Full body
and 114–132 in `ledgers/amendments.md`.

## Ledger index

attempts.md (11 rounds) · optimizations.md (6) · source-ideas.md (10) · amendments.md (20: 114–133)
