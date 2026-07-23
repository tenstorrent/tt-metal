# fused_rms_minimal non-rectangular-grid stale-L1 demo (#45175 / tt-xla #5738)

`fused_rms_minimal` (rms_allgather) on a **non-rectangular** width-shard grid is correct
or garbage depending only on **what was in L1 before the op ran**. The stats reduction
walks the shard grid's bounding-box rectangle, so the "phantom" cells (in the bounding
box but holding no shard data) are read into E[x^2]: 0 on a fresh device (harmless),
leftover junk on a used device (corrupts). This L1-history dependence is why the bug is
non-deterministic — it hides after a reset and only bites inside a running model.

## Run
Freshly reset the device first, then run on Blackhole (>=11-wide worker grid):
```
tt-smi -r        # phantom cells must start at 0 for the FRESH rows
python rms_allgather_stale_l1_demo.py
```

## Expected output (Blackhole qb2, 2x2 mesh, on main WITHOUT the validator fix)
```
=== fused_rms_minimal: stale-L1 dependence on a non-rectangular grid ===
condition                                    PCC    rel_l2    out_max   ref_max
fresh  | non-rect (11x6, 2 phantom)      0.99996   0.01796      11.25      11.5
fresh  | rect 8x8 (no phantom)           0.99992    0.0234      11.12      11.3
stale  | non-rect (11x6, 2 phantom)      0.70285    0.7125      9.438      11.7  <-- CORRUPT
stale  | rect 8x8 (no phantom)           0.99995   0.02066      9.562      9.62
```

Reading:
- **FRESH L1**: both grids correct (phantom cells are 0).
- **STALE L1** (after priming the bounding-box cells): **only the non-rectangular grid
  breaks** — it reads the primed phantom cells. The rectangular 8x8 grid has no phantom
  cells and stays correct regardless of L1 state.

Notes:
- The synthetic prime here produces a per-row scale corruption (PCC ~0.70). A real model's
  arbitrary leftover garbage pushes it all the way to PCC ~0 / NaN / ~1e36 (the tt-xla
  #5738 symptom).
- Grid geometry: non-rect = `[(0,0)-(10,4)]`(55) + `[(0,5)-(8,5)]`(9) = 64 cores, bbox
  11x6=66 -> 2 phantom cells (9,5),(10,5). This is the llama-3.1-70B TP decode norm layout.
- This demo runs on `main`; with the op validator fix (branch
  `mvasiljevic/5738-rms-allgather-nonrect-correctness`) the non-rect rows instead raise a
  clear `TT_FATAL` up front.
