# Reference tables

Reduced results of the SDPA and TopK measurement campaign on one Blackhole p100a (11x10 grid = 110
worker cores, 1350 MHz, firmware bundle 19.9.0). They are here so that a rerun can be diffed against
the numbers the analysis was built on, instead of only against a prose summary. Every file carries
its own `# PROVENANCE:` header naming the date, the card, the kernel revision and the reducer that
produced it.

| File | Rows | What it holds |
|---|---|---|
| `targets_table.csv` | 142 | Every device wall of the campaign: configuration, measured cycles and microseconds, core count, the run tag it came from. The widest table, and the one to diff a rerun against. |
| `refit_r2_walls.csv` | 110 | The same walls priced by the model, with signed error per wall and the fit / prediction / hold-out set each belongs to. Includes the 2026-09-16 MLA decode slice and cores-per-group sweeps. |
| `floor_verification_configs.csv` | 20 | Perf counter readings (FPU, SFPU, MATH) per configuration on the unmodified kernel, per-core mean and wall-setting core. |
| `model_components_grid1.csv` | 14 | Measured wall against the model's named wall terms, per configuration. |
| `floor_verification_fits.json` | - | The fitted floor constants and the overlap fits behind them. |

## How to use them

Reproduce a row with the sweep it names (`analysis/RUNBOOK.md` sections 3 to 5 for the sweeps and the
counter capture, section 6 for the reducers), then compare. Two things to keep in mind:

- The wall is the device wall: max end minus min start over all cores, which is what the tracy ops
  report's `DEVICE KERNEL DURATION` gives. A per-core median understates it (section 11.3).
- These are one card's numbers. Run-to-run repeats on the same card landed within 0.15 percent, but
  another p100a or another firmware bundle can shift the DRAM-bound rows; the per-row configuration
  is in the table so a mismatch can be attributed rather than guessed at.

The analysis write-ups that interpret these tables are internal. The findings that came out of them
are public on the tt-metal tracker, issue #55767 and its children.

The repository ignores `*.csv` (see `.gitignore`), so these four were added with `git add -f` on this
branch. That is deliberate for a handoff branch and is not meant for upstream: a rerun writes its own
CSVs, which stay ignored as usual.
