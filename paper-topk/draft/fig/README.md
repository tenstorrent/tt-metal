# Figures — data sources and plotting rules

Every figure is generated from a committed CSV or an evidence-pack artifact;
no number enters a figure that is not in
`paper-topk/evidence/paper/evidence.md`.

| Figure | Owner brief | Data source |
|---|---|---|
| fig-b1-tensix-mesh (schematic) | 02-background | none (architecture drawing; facts from Hot Chips talk) |
| fig-d1-design-space (map) | 03-design-space | RADIX_BUCKET_GPU.md §7.1 (conceptual) |
| fig-d2-engine-shootout (bars) | 03-design-space | cyc/elem numbers per evidence.md Fig-4 row (cgtceq-debug.md, risc_scan_bench commit numbers); regenerate raw CSVs from committed benches if needed |
| fig-s1-operator (schematic) | 04-system | program-factory structure (conceptual) |
| fig-s2-skip-law (curves) | 04-system | `paper-topk/evidence/tileskip/forecast.md` §2 + `sim_tier1_skip.py` |
| fig-e1-pscaling (lines + model overlay) | 06-evaluation | `tests/ttnn/unit_tests/operations/reduction/baselines/comp3/psweep4_full.csv`; unit constants forecast.md §3 |
| fig-e2-skip-ab (bars) | 06-evaluation | `paper-topk/evidence/tileskip/{baseline,skipon,skipgated}.csv` |

## Plotting rules (binding — style guide "Tufte-Style Rules of Thumb")

- `ax.grid(False)`; top/right spines off; left/bottom spines 0.5 pt.
- Legends: `loc='lower center', bbox_to_anchor=(0.5, 1.0), frameon=False`
  above the plot — or better, direct line-termination labels.
- One color per measured arm, held constant across ALL figures.
- Flat 2D only; semi-transparent Rectangle patches for highlight regions
  (underperformance `#FFE5E5` α=0.5; aggregate `#FFD700` α=0.2).
- Captions ≤8 words, declarative noun phrase, no "This figure shows".
- Any axis or annotation in µs derived from cycle measurements carries the
  1.35 GHz busy-clock caveat in its caption (Tracy-measured µs are direct
  and exempt).
- `×` never `x`; `≈` never `~` in annotations.
