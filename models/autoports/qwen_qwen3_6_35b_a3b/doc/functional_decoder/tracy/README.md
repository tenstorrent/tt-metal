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
| `<mode>_ops.csv.gz` | **only when under 500 KB** | the raw post-processed Tracy ops CSV the report was built from. `run_perf.sh` gzips it after both `tt-perf-report` runs, so `gunzip -k` it before re-running the tool |

This repository's `check-large-files` pre-commit hook rejects committed files over 500 KB, and
the two decode ops CSVs exceed it even gzipped (`linear_decode` 1.72 MB,
`full_decode` 731 KB — one traced decode of this layer records 109 (full) /
121 (linear) ops per iteration and the committed window holds 8 iterations). They are therefore
excluded by `.gitignore` in this directory; the two prefill ops CSVs
(`full_prefill` 151 KB, `linear_prefill` 432 KB)
are under the limit and committed.
Everything derived from them *is*: the filtered per-op CSV and the human-readable table above
are exactly the signpost-filtered view of those rows.

Regenerate **all four** cases (no arguments) before trusting the derived numbers:

```bash
./models/autoports/qwen_qwen3_6_35b_a3b/tests/run_perf.sh
```

A single case is available for debugging one artifact — `run_perf.sh decode linear` — but it is not
how the committed set should be produced: `perf_summary.json` and README section 5 aggregate all four,
and the script only resets `perf_host_summary.jsonl` on a full run (it warns on a partial one). The
work log's section 6 profiler incident is also why the committed set comes from one contiguous run
rather than being spliced case by case.

To re-run `tt-perf-report` against a committed ops CSV, `gunzip -k` it first.
