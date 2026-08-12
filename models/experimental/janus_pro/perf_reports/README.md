# Per-stage profiler reports

Alongside these, [`DEAD_ENDS.md`](DEAD_ENDS.md) collects the negative results: levers that were
measured and did not pay, plus one that did pay and is deliberately not in the tree because it
breaks an accuracy gate.

[`PROFILER_NOTES.md`](PROFILER_NOTES.md) covers how to read tt-perf-report and the per-RISC
counters on this tower, including three ways their output misleads.

One file per row of [PERF.md](../PERF.md)'s change log, named `NN-slug.md` where `NN` is the row
number. Each holds the per-op-code and per-matmul-shape breakdown of a single trace replay plus
the kernel total, which is what the change log's figures are matched against.

The raw `ops_perf_results_*.csv` a tracy run produces is ~12 MB and 107 columns. It cannot be
checked in — the repo's pre-commit hook rejects anything over 500 KB — so these condensed
reports are the durable artifact. Each records the commit it measured in its header.

## Regenerating one

```bash
# 1. put the arithmetic at that stage, keeping the trace harness current
tools/stage_checkout.sh <sha>

# 2. run the perf command from PERF.md's "How to reproduce"

# 3. condense the run into a report
python -m models.experimental.janus_pro.tools.perf_stage_report \
  generated/profiler/reports/<stamp>/ops_perf_results_<stamp>.csv \
  --stage NN-slug --sha <sha> --note "<one line>"

# 4. back to current
tools/stage_checkout.sh --restore
```

Roughly two device-minutes per stage.

**Why the checkout is split.** `forward_device`, `prepare_patches` and the perf test itself only
appear part-way through the sequence, so early stages have no traced entry point of their own.
`stage_checkout.sh` therefore reverts only the files that carry the arithmetic — attention, the
transformer block, the MLP, the aligner, the layer norm and `model_config.py` — and holds the
device-entry plumbing at HEAD. Measuring every stage through one harness is the only way the
figures compare; `model_config.py` is byte-identical across the range, so nothing leaks.

## Stage to commit

| # | commit | # | commit |
|--:|---|--:|---|
| 0 | `f7f7d7cd87f` | 13 | `d9b9a3ba0d2` |
| 1 | `16f0bf214c9` | 14 | `9fcbc1edfbf` |
| 2 | `0ef4d8efc93` | 15 | `33aebf75626` |
| 3 | `8dfc3cf9198` | 16 | `45b59edfb69` |
| 4 | `56e4f367883` | 17 | `5a47c877785` |
| 5 | `76e533f24e9` | 18 | `4b6b23530ba` |
| 6 | `8becfcd193e` | 19 | `da1f869da23` |
| 7 | `176d4978e6a` | 20 | `43c86bd2fa3` |
| 8 | `0c3409ab2d6` | 21 | `a709b232923` |
| 9 | `9a559f35bed` | 22 | `744b677f553` |
| 10 | `2c43754521f` | 23 | `c2548366aa5` |
| 11 | `839efe8eb06` | 24 | `09c20b5fde7` |
| 12 | `0b4a907677e` | | |

Stages 15 and 16 each landed over several commits — the layer-norm class in `ab17d22a140`,
`395885e429e`, `33aebf75626`; `in0_block_w` in `8e58a8f7151` then `45b59edfb69`. The commit listed
is the last of each group, i.e. the state the row describes.

## Known gaps

A stage with no file here has not been re-measured under the current harness. Its change-log row
carries the figure recorded at the time, in whatever metric was in use then, and says so.
