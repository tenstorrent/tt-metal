---
name: profiler-viz
description: Turn a tt-metal device-profiler dump (profile_log_device.csv) from a blaze fused decoder layer into a per-op HTML visualization — a relative-cost bar chart plus a per-op table (time_us, % wall, busy cores, chips). Two-step pipeline (gen_csv.py → gen_html.py). Use when you have a profile_log_device.csv (or a profiler output dir) and want a readable per-op report/chart, or need the intermediate ops.csv/coregrid.csv tables.
---

# Profiler CSV → per-op HTML viz

Converts a raw device-profiler dump into a readable per-op report. The raw
`profile_log_device.csv` is huge (~1.3M rows for a full-per-op fused layer, one
row per DeviceZoneScopedN marker per core per RISC per iteration). This skill
reduces it to a small `ops.csv` and renders an HTML chart + table.

Bundled scripts live next to this file in `scripts/`:
- `scripts/gen_csv.py` — reduce `profile_log_device.csv` → `ops.csv` + `coregrid.csv`
- `scripts/gen_html.py` — render `ops.csv` → self-contained `perf.html`

## Inputs

A device-profiler dump. Either:
- a `profile_log_device.csv` directly, or
- a profiler output dir — the CSV is at `<dir>/.logs/profile_log_device.csv`
  (`find <dir> -name profile_log_device.csv`).

Produced by a profiling run — see the `gptoss-blaze-profiling` skill for how to
generate one (CI uses `TT_METAL_DEVICE_PROFILER=1` + plain pytest; local uses
`python -m tracy -r -p -o <out> -m pytest <nodeid>`).

## Two-step pipeline

```bash
S=~/.claude/skills/profiler-viz/scripts

# 1) reduce the big CSV to per-op tables (run this ON the box — the CSV is huge)
python "$S/gen_csv.py" <profile_log_device.csv> --out <outdir>
#   -> <outdir>/ops.csv, <outdir>/coregrid.csv
#   prints: PFX=... reduce-root=PCIe N | ops=K | WALL=... | checkpoint_samples=M

# 2) render the HTML (ops.csv is small — do this anywhere, e.g. locally)
python "$S/gen_html.py" <outdir>/ops.csv --out <outdir>/perf.html \
    --title "GPT-OSS-120B global @128" --h1 "Global decoder layer @128" \
    --subtitle "TP=8, slow dispatch." --wall 0
```

Then view: open `perf.html` in a browser, or publish it with the **Artifact**
tool to get a shareable link. The HTML is self-contained (inlined CSS/JS),
theme-aware, and needs no server.

## Output columns (`ops.csv`)

`op, phase, time_us, pct_wall, busy_cores, n_chips, prev, next`
- `phase` ∈ {attn, moe, comm, infra} (colors the chart).
- `time_us` — per-op time = **max over reduce-root-chip busy cores of the
  median (across RISCs/iterations) zone duration**. This is the op's critical
  core; peer chips/cores wait.
- `pct_wall` — `time_us / wall`. Shares **overlap** (ops run in parallel), so
  they do NOT sum to 100 — the Σ row is for *attribution* (which op is biggest),
  not an additive latency budget.
- `busy_cores` — cores with median > 0.3 µs; `n_chips` — chips the op ran on.

`coregrid.csv` (`op,chip,core_x,core_y,dur_us`) gives the per-core breakdown for
heat-map / spatial analysis.

## Method (how per-op time is produced)

1. Each row is a zone marker: `PCIe_slot, core_x, core_y, RISC, timer_id,
   time[cycles], …, zone_name, ZONE_START|END`.
2. Per `(chip,core,RISC,zone)`: pair each START with the next END →
   `duration = (end−start)/1350` µs (Blackhole @ 1350 MHz). Drop > clamp (mis-pairs).
3. Per `(chip,core,zone)`: pool all RISC instances, take the **median** (robust
   to DM/other RISCs that idle inside the zone).
4. Busy core = median > 0.3 µs.
5. Reduce-root chip = chip with the largest `MOE__REDUCE_TO_ONE` core.
6. Per-op time = **max over reduce-root busy cores**.
7. Layer prefix (`GPTOSS_GLOBAL_LAYER__` / `GPTOSS_WINDOWED_LAYER__`) is
   auto-detected and stripped from op names.

## Wall time (optional — often not needed)

`gen_csv.py` derives WALL from `STAGE_CHECKPOINT` zones (a per-iteration wall
marker) when present. **If the kernels don't emit `STAGE_CHECKPOINT`** (it was
removed from `kernel_codegen.py`), there are no checkpoint samples and WALL
falls back to a nominal **67.5 µs** (or `--wall <n>`). In that case `pct_wall`
is attribution-only — treat per-op `time_us` as the source of truth and ignore
the wall row. Pass `--wall 0` to `gen_html.py`/`gen_csv.py` to make the nominal
wall explicit, or a real number if you measured latency separately.

## Practical notes

- **Run `gen_csv.py` on the box**, then scp the small `ops.csv`/`coregrid.csv`
  back. Never scp the 1.3M-line raw CSV.
- **One profiler dir per config.** Don't mix zone sets from different configs
  (e.g. global vs windowed) in one dump — the layer prefix auto-detect and the
  reduce-root pick assume a single layer type.
- **Dropped-zone caveat.** A full fused layer emits ~327 distinct zones; the
  16-bit `timer_id` hash space collides (~1 pair), so one zone label is dropped
  by the profiler. Impact is negligible (the affected op usually has other zone
  instances), but if an expected op is missing from `ops.csv`, that's why — see
  the `gptoss-blaze-profiling` skill's collision notes.
- **Tuning:** `--clamp` (per-op mis-pair cap, default 40 µs) and `--cpcap`
  (checkpoint cap, default 500 µs) reject impossible durations from unpaired
  START/END across buffer wraps.
