# Embedded Activation Results Layout

This directory uses the canonical embedded result layout. Prefer these stable
paths over older timestamped run directories when linking, reviewing, or
archiving results.

## Canonical Directories

```text
results/
|-- frontier/
|   |-- bf16/
|   |   |-- data/csv/
|   |   |-- logs/
|   |   |-- plots/
|   |   `-- summary.txt
|   `-- fp32/
|       |-- data/csv/
|       |-- logs/
|       |-- plots/
|       `-- summary.txt
`-- native_vs_embedded/
    `-- bf16/
        |-- data/csv/
        `-- logs/
```

`frontier/<dtype>/data/csv/` contains the shard CSVs from the frontier sweep and
any TTNN reference CSVs available for that dtype. `logs/` contains worker,
plotting, TTNN reference, and failure logs where those artifacts exist. `plots/`
contains generated PNGs. `summary.txt` is the text summary emitted by the
frontier plotting pass.

`native_vs_embedded/bf16/data/csv/` contains the native-vs-embedded comparison
summary CSVs that feed tier comparison plots. This comparison currently has a
canonical BF16 result set only.

Timestamped run directories such as `frontier_4chip_*`,
`frontier_fp32_4chip_*`, and `native_vs_embedded_4chip_*` are retired. Do not
add new checked-in plots or CSVs under those names.

## Plot Families Present Now

The canonical tree currently carries these generated plot families:

- `frontier/<dtype>/plots/scatter/*.png`: per-activation Tracy runtime
  versus maximum ULP scatter plots for the frontier sweep. BF16 and FP32
  canonical runs both have this family.
- `frontier/bf16/plots/tiers/*.png`: per-activation comparison of
  selected native-vs-embedded BF16 tiers against the TTNN reference. This family
  depends on BF16 native-vs-embedded summary CSVs.

The FP32 frontier run does not currently have `tiers` plots because
there is no canonical `native_vs_embedded/fp32` summary set.

## Plot Families Requiring Additional Artifacts

Some older plot families mentioned elsewhere in this example predate this
embedded result layout. They require raw or broader data artifacts that are not
part of the canonical result set above:

- ULP-by-input plots require explicit raw hardware dump CSVs with `input,output`
  columns. These are consumed by `tools/plotting/ulp_by_input.py`; summary
  frontier CSVs are not sufficient.
- Pareto-winner IO plots use `frontier/<dtype>/data/csv/pareto_winners.csv` as
  a manifest, raw dumps under `frontier/<dtype>/data/dumps`, and generated PNGs
  under `frontier/<dtype>/plots/ulp_by_input`.
- Hardware error analysis and ULP distribution plots require raw hardware output
  dumps from the retired `*_hardware_outputs` or `data/hardware_outputs` style
  layouts, or equivalent raw dumps migrated into a documented location.
- Legacy architecture-wide Pareto, depth-degree heatmap, runtime comparison,
  and activation comparison plots require the older sweep/result datasets those
  scripts were written for. They are not regenerated from only
  `frontier/<dtype>/data/csv` and `native_vs_embedded/<dtype>/data/csv`.

When adding a new plot family, document the exact required inputs here before
publishing generated PNGs.
