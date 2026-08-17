# Tracy performance artifacts

One directory per measured case (`<layer kind>_<mode>`), produced by
`../../../tests/run_perf.sh`. Contents:

| file | committed | notes |
|---|---|---|
| `<mode>_perf_report.txt` | yes | the human-readable `tt-perf-report` table for the signposted window |
| `<mode>_perf_report.csv` | yes | the same rows as CSV (`tt-perf-report --csv`) |
| `<mode>_perf_report.console.log` | yes | stdout of the `--csv` run (provenance: signposts used, warnings, roofline) |
| `<mode>_perf_report_stacked.csv` / `.png` | yes | `tt-perf-report`'s stacked breakdown |
| `tracy_run.log.gz` | yes | full Tracy + pytest transcript |
| `<mode>_ops.csv.gz` | **only when under 500 KB** | the raw post-processed Tracy ops CSV the report was built from |

This repository's `check-large-files` pre-commit hook rejects committed files over 500 KB, and
the two decode ops CSVs exceed it even gzipped (`linear_decode` 1.6 MB, `full_decode` 707 KB —
one traced decode of this layer records ~900 ops per iteration). They are therefore not in git.
Everything derived from them *is*: the filtered per-op CSV and the human-readable table above
are exactly the signpost-filtered view of those rows.

Regenerate any of them with one command, e.g.:

```bash
./models/autoports/qwen_qwen3_6_35b_a3b/tests/run_perf.sh decode linear
```

To re-run `tt-perf-report` against a committed ops CSV, `gunzip -k` it first.
