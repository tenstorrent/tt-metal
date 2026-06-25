# Embedded Plotting Tools

These tools operate on explicit result paths. They should write generated
artifacts under the canonical result layout documented in
`../../results/README.md`.

## Supported Plot Families

Future sweeps should write into stable run roots with the shell tools'
`--run-dir` option, for example:

```bash
tt_metal/programming_examples/generic_lut_activation_embedded/tools/frontier_sweep.sh \
  --precision bf16 \
  --shard 0 \
  --num-shards 4 \
  --run-dir tt_metal/programming_examples/generic_lut_activation_embedded/results/frontier/bf16
```

`frontier_scatter.py` reads frontier shard CSVs and writes per-activation plots:

```bash
python3 tt_metal/programming_examples/generic_lut_activation_embedded/tools/plotting/frontier_scatter.py \
  tt_metal/programming_examples/generic_lut_activation_embedded/results/frontier/bf16/data/csv/frontier_chip*.csv \
  --ttnn tt_metal/programming_examples/generic_lut_activation_embedded/results/native_vs_embedded/bf16/data/csv/ttnn_ref.csv \
  --tier best=tt_metal/programming_examples/generic_lut_activation_embedded/results/native_vs_embedded/bf16/data/csv/best_native_vs_embedded.csv \
  --tier best99=tt_metal/programming_examples/generic_lut_activation_embedded/results/native_vs_embedded/bf16/data/csv/best99_native_vs_embedded.csv \
  --tier best95=tt_metal/programming_examples/generic_lut_activation_embedded/results/native_vs_embedded/bf16/data/csv/best95_native_vs_embedded.csv \
  --outdir tt_metal/programming_examples/generic_lut_activation_embedded/results/frontier/bf16/plots
```

The same tool writes:

- `scatter/*.png` from frontier shard CSVs.
- `tiers/*.png` when `--tier` CSVs are supplied.

`plot_experiment.py` is the canonical wrapper. By default it reads BF16
frontier CSVs from `results/frontier/bf16/data/csv`, writes plots under
`results/frontier/bf16/plots`, and discovers TTNN/tier CSVs from the canonical
`data/csv` directories when supplied:

```bash
python3 tt_metal/programming_examples/generic_lut_activation_embedded/tools/plotting/plot_experiment.py \
  --frontier tt_metal/programming_examples/generic_lut_activation_embedded/results/frontier/bf16 \
  --native-vs-embedded tt_metal/programming_examples/generic_lut_activation_embedded/results/native_vs_embedded/bf16 \
  --all
```

`ulp_by_input.py` reads explicit raw dump CSVs with `input,output` columns and
writes one requested PNG. It is intentionally separate from frontier summaries:

```bash
python3 tt_metal/programming_examples/generic_lut_activation_embedded/tools/plotting/ulp_by_input.py \
  --activation gelu \
  --precision bf16 \
  --series embedded=/path/to/raw_embedded_dump.csv \
  --out /path/to/ulp_by_input/gelu.png
```

## Artifact Requirements

The canonical result tree has enough data for the current frontier scatter
family, and enough BF16 native-vs-embedded summary data for BF16 tier comparison
plots.

The older plot families that analyze pointwise error, ULP distributions, or
architecture-wide sweeps need additional raw/data artifacts. Do not imply that
those plots can be regenerated from the canonical summary CSVs alone; they need
raw hardware output dumps or the legacy sweep datasets they were designed to
consume.
