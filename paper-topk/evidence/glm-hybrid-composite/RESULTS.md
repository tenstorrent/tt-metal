# Multi-rectangle tree factory + hybrid composite (2026-08-17, p150a 130-core)

The plain call — exactly as the GLM/DSA indexer makes it, zero call-site changes —
now routes rows>grid through an op-internal hybrid: row-parallel full waves +
a multi-rectangle remainder wave (one P-core tree per row, concurrent) + concat.

Measured (Tracy device kernel duration, 5-iter medians, per-cell subprocess):

| cell | before | after | parts |
|---|---|---|---|
| 160x65536 k=2048 plain (GLM) | 712.3 us | **467.0 us (1.53x)** | rp 359.4 + rect 99.0 + concat 8.8 |
| 160x65536 k=512 plain (DS-V4) | 559.5 us | **358.3 us (1.56x)** | rp 280.9 + rect 73.9 + concat 3.2 |
| 185x65536 k=2048 plain (margin class) | ~713 us | 553.9 us (1.29x) | rp 359.4 + rect 184.0 + concat 10.5 |
| 30x65536 k=2048 explicit num_slices=4 | 356.5 us (RP) | 98.8 us (3.6x) | 30 concurrent 2x2 trees, 120c |
| 160 rows explicit num_slices=2 (single program) | 712.3 | 542.4 us (1.31x) | 65 concurrent 1x2 trees, 3 row-waves |

Design decisions (why, not just what):
- Device op NEVER auto-selects multi-row rects: the non-stable op breaks bf16
  ties differently across engines, and tile_output/uint16 variants can't run
  rects (ROW_MAJOR-only writers), so auto-routing would break the
  "opt-ins change layout, never results" contract the nightly suite guards
  (its matches_default tests caught exactly this in round 1). Multi-row rects
  run only via explicit num_slices or the wrapper's internal remainder window.
- num_rects = full tiling capacity (empty rects run zero rows): rows stay
  runtime-only, one cached program per rectangle layout — preserves the
  program-cache row-count contract (suite test) and serves 60-vs-30-row calls
  from one program (validated).
- Rectangle fit prefers max tiling capacity among equal-P shapes (1x4 tiles a
  13x10 grid 26x, 2x2 tiles it 30x — the fit flaw cost 191.2 vs 98.8 us in
  round 1, found by measurement, model-confirmed).
- Wrapper margin >= 12.5% modeled win (was 2x): covers concat+dispatch; the
  2x threshold would have ceded the rows 180-195 class (r2=50-65, P=2-only).
- Row window (row_start/row_count) is composite-internal: readers offset into
  global input rows, writers stay output-relative; no input slice/copy.

Gates: correctness battery 28/28 (twice; program-cache-hit reruns, 60->30-row
redistribution, valid_length poisoned-tail, values arm, k=512 P=16, composite
plain/valid_length/values, cache-contract shape); nightly suite 181 passed +
2 known env-gated IOMMU perf pins (stale anyway — re-pin on the IOMMU runner,
now reflecting the composite). Grid-portable: on the production BH Galaxy
12x10 grid the split re-derives as 120 RP rows + 40 rows at P=3 (cap 40 = r2
exactly). The op is single-chip SPMD on the mesh — no cross-chip traffic; the
one mesh-validation want is the composite under mesh dispatch, covered
incidentally by any GLM 8x4 PCC suite run (same run that gates fe1930d50c2).

Next capture (routing layer, NOT the op): the rows=32 sampling/TP=8 scenario
shapes model at 2.8-2.9x via P=3 trees — routed ttnn.topk already changes
engine vs stock (ties I5-audited), so topk.cpp routing can pass num_slices.
