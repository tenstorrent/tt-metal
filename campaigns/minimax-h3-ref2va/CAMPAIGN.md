# MiniMax-H3 `ref2va` — 4×8 Blackhole Galaxy — campaign

Branch `kevinmi/minimax-h3-integration` · run root `campaigns/minimax-h3-ref2va/` · started 2026-08-05
Full history: `git log --follow -- campaigns/minimax-h3-ref2va/CAMPAIGN.md campaigns/minimax-h3-ref2va/ledgers/`
Bringup campaign: the metric is gates going green, not a latency.

## Loop state

| Round | Gates green | Baseline | Stall | Target | Gate status |
|---|---|---|---|---|---|
| 0 | 0/8 | t2va + fl2va green @ `bd12ad2aeb2` (running) | 0/10 | 8/8 acceptance criteria | none fired |

## Working point

Mesh **4×8 Blackhole**, TP=4 axis 0 / SP=8 axis 1, ring, 2 links. Target **1344×768, 124 frames @ 24
fps** → 37 latent frames (48×84 latent, 1008 rows/frame, 37296 video rows), 207 audio latents (414
rows), 50 steps → 49 forwards. Partition **`transformer_ref/`** for ref2va (config byte-identical to
`transformer/`). AdaLN precompute on. Seed 0.

ref2va padded lengths, measured host-only (am. 114) — `sp_factor * TILE` = 256:

| Request | seq | padded | `seq_local` |
|---|---|---|---|
| t2va (baseline) | 37710 | 37888 | 4736 |
| 1 image 1:1 | 45910 | 46080 | 5760 |
| 1 video + sound 16:9 | 81488 | 81664 | 10208 |
| image + video + audio | 90102 | 90112 | 11264 |
| 9 images 1:1 | 111446 | 111616 | 13952 |

Layout: `[ text | ref block 1 | ref block 2 | … | target audio | target video ]`, one block per
reference in **request order**, a video reference's soundtrack rows immediately **before** its own
video rows.

## Fixed baseline (Phase 1, immutable)

SHA `bd12ad2aeb2`. Command (verbatim), mesh 4×8, cold e2e (these gates run cold; not comparable to a
warm latency number):

```
MINIMAX_H3_DIFFUSERS_DIR=/data/cglagovich/MiniMax-H3-diffusers \
MINIMAX_H3_MODEL_PATH=/data/cglagovich/MiniMax-H3-diffusers \
TT_DIT_CACHE_DIR=/data/kevinmi/tt_dit_cache \
timeout 9000 scripts/run_safe_pytest.sh --run-all \
  models/tt_dit/tests/models/minimax_h3/test_pipeline_minimax_h3.py \
  models/tt_dit/tests/models/minimax_h3/test_pipeline_fl2va_minimax_h3.py
```

Result: **GREEN — t2va 1 passed (560 s), fl2va 4 passed (645 s), exit 0.** Frozen output in
`artifacts/round-0/e2e/`. Measured on this SHA, not carried over:

| Gate | Measured | Bar |
|---|---|---|
| CLIP prompt alignment | **37.38** (min 36.79, max 37.82) | 33.0 |
| audio | peak **0.095**, rms 0.0154, 2ch 5.175 s @ 32 kHz, 0 % clipped | — |
| VBench subject / background consistency | 0.9804 / 0.9812 | 0.95 |
| VBench motion_smoothness / dynamic_degree | 0.9914 / 1.0000 | 0.97 / 1.0 |
| VBench imaging_quality | 0.6925 | 0.64 |
| fl2va first-keyframe anchor | 0.9971 | 0.95 |
| fl2va fractal discriminator | 0.9963 to the keyframe vs **0.2972** to t2va's own frame 0 | — |

CLIP 37.38 and audio peak 0.095 match STATE.md's fingerprint exactly, and the fractal discriminator
reproduces its 0.9963 / 0.2972. **Only CLIP is a quality signal**; the rest catch an *unexplained*
move. The fl2va half ran on `5c8adce1e85`, which adds new files only and touches no shared path.

Caveat worth keeping: the first attempt had all four fl2va cases **skip silently** because
`MINIMAX_H3_ARTIFACT_DIR` and `MINIMAX_H3_T2VA_ARTIFACT_DIR` pointed at different directories, so the
fl2va gate could not find the calibrated `t2va.mp4` it keys its thresholds to. Point both at one
directory. This is STATE.md's "a test reading its input from the directory it writes skips silently"
pitfall, met from the other side.

## Gates

| Gate | Command | Result |
|---|---|---|
| 1 host packing bit-exact | `pytest .../test_packing_ref2va_minimax_h3.py` | not written |
| 2 both span orders | same, `-k span` | not written |
| 3 reference encode pcc ≥ 0.99, real media | `pytest .../test_references_minimax_h3.py` | not written |
| 4 typed condition stream, existing PCCs identical | `pytest .../test_transformer_minimax_h3.py` | not written |
| 5 t2va + fl2va unchanged | `pytest .../test_pipeline_minimax_h3.py .../test_pipeline_fl2va_minimax_h3.py` | baseline running |
| 6 ref2va e2e | `pytest .../test_pipeline_ref2va_minimax_h3.py` | not written |
| 7 conditioning is not a no-op | same, `-k discriminate` | not written |
| 8 frames inspected | `artifacts/round-N/frames/` | not run |

## Pending work

1. [#2] Phase 1 fixed baseline — **running**; capture the existing transformer PCCs after it.
2. [#3] Port host `packing_ref2va.py` + its bit-exact gate (no device). In flight alongside #2.
3. [#4] Phase 0 shape probe at 46080 / 81664 / 111616 — full depth, real weights, real AdaLN table.
   **Its verdict sets the Phase 6 case list.**
4. [#5] `condition_blocks` typed condition stream; existing PCCs must be identical, not merely ≥ 0.9995.
5. [#6] Reference encode on device (image via `encode_clip`, video via `encode`, audio via the
   existing `MiniMaxH3AudioEncoder`, zero-padded to a multiple of 800).
6. [#7] Pipeline plumbing, `transformer_ref` selection, and the three cache-key fixes of am. 118.
7. [#8] ref2va e2e plus the two-references-identical-geometry discriminator.

## Pitfalls

Campaign-specific only; the general tt_dit ones are in `STATE.md` and `shared/known-issues.md`.

- A reference **never** binds the target geometry, and is prepared at its own resolution. This is what
  makes the packed sequence up to 3× t2va's (am. 114) and is easy to mistake for a bug.
- Audio reference rows are **clean at t = 1.0** and take `posterior.mode()` — not noise-augmented, not
  sampled (am. 115). Nothing would catch getting this wrong.
- **Two** `_temporal_position_span` orders, one per call site; they differ at the production n = 37
  (am. 116).
- The reference audio VAE **right-pads with zeros** to a multiple of 800; our device encoder asserts
  divisibility instead, so the host path must pad.
- `_decode_audio` must drop the leading reference audio rows: `unpack_audio_tokens` reshapes to
  `(2, num_audio_latents, C)` and asserts nothing.
- The vision tower's merged tokens are consumed **in run order** by `_scatter_rows`, so they must be
  assembled in *presentation* order, not image-batch-then-video-batch order.

## Latest amendment

**119 (2026-08-05)** — the ref2va vision-tower coverage was adopted after all, in consolidated form:
`test_qwen3vl_vision_tower.py::test_tower_on_device`'s `GRIDS` covers every ref2va grid including
`max_load` (nine images + three videos, 168192 patches, 18 blocks). The tower half needs no porting
work. Full body and 114–118 in `ledgers/amendments.md`.

## Ledger index

attempts.md (0 rounds) · optimizations.md (0) · source-ideas.md (10) · amendments.md (6: 114–119)
