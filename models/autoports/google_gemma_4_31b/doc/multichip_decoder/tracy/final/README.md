# Final multichip profiler provenance

These are four independent Tracy captures on the fixed 1x4 Blackhole P150b
mesh. The test fixture opens the mesh directly; it does not construct the
single-chip latency baseline inside a profiler process.

## Capture command

```bash
GEMMA4_MULTICHIP_PROFILE=1 MPLCONFIGDIR=/tmp/mpl \
LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH \
python -m tracy -r -p -v -o <folder> -m pytest \
'models/autoports/google_gemma_4_31b/tests/test_multichip_decoder.py::test_multichip_profile[<mode>-<layer>]'
```

Folders and source op CSVs:

- `sliding_decode/reports/2026_07_14_07_28_49/ops_perf_results_2026_07_14_07_28_49.csv`
- `full_decode/reports/2026_07_14_07_29_50/ops_perf_results_2026_07_14_07_29_50.csv`
- `sliding_prefill/reports/2026_07_14_07_30_47/ops_perf_results_2026_07_14_07_30_47.csv`
- `full_prefill/reports/2026_07_14_07_31_51/ops_perf_results_2026_07_14_07_31_51.csv`

## Analysis command

```bash
MPLCONFIGDIR=/tmp/mpl PYTHONPATH=/tmp/tt-perf-report-env \
/usr/bin/python3 $HOME/.local/bin/tt-perf-report <ops-csv> \
  --start-signpost MC_<layer>_<MODE> \
  --end-signpost MC_<layer>_<MODE>_END \
  --no-color --csv <folder>/filtered.csv \
  --summary-file <folder>/summary.csv

MPLCONFIGDIR=/tmp/mpl PYTHONPATH=/tmp/tt-perf-report-env \
/usr/bin/python3 $HOME/.local/bin/tt-perf-report <ops-csv> \
  --start-signpost MC_<layer>_<MODE> \
  --end-signpost MC_<layer>_<MODE>_END --no-color --no-summary
```

The second command's human-readable table and advice are appended to
`report.txt`. Raw device logs and `.tracy` captures were removed after the
auditable source op CSV, filtered CSV, summary CSV/PNG, and report were
generated.

## SHA-256

| Mode | source op CSV | filtered CSV | report |
|---|---|---|---|
| sliding decode | `f1897ba9617ca526773e3b8cce356cad95d2d4be3b3e8df9b80577b61c60a988` | `c8888fd4bb25f248ae9d23282c30e8156bff54532455d15a1799e9f640ec7fd4` | `eddb9c54affea6a7e9b5556704e45dc9c9206b7d63139c50391f6b575aee2432` |
| full decode | `0ea7c6c7bf18a155f7fedcc28d6de4fd31752a0831ec4012a7cf03eb41c51569` | `ece48cb236864e7b1ba333a338664f7c09b254f7b29562aad41f6f754bcba193` | `1be627b5824500e68b61d409caa96bfe2d8921aff69fe35a669eae96e83097b8` |
| sliding prefill | `a2144144b452a90640ded6351df29569bb90dde9fc2605c52b4076ea1fcfa8ba` | `cb1aed6f3ce3b6a8434b74b33803e401d763bc6b7081a5c487b2f15b22a0f370` | `f13bd1aebc004abebe60b2ff382bc0b31c4e2d294eda344741b20011ecb7694b` |
| full prefill | `6debbed4506bb0fb34f79a6503ae04c25fa9390c5e8c31973ab557f033fec8b3` | `72cf1f39fdd1b6c483aa1dde8bf26172dae2ead80e5a2668c20ff4fb94225aa1` | `ece4aa3488c76d4a7e63408a46dd0903fd70e9cda48286f90b6c04b95c5265c2` |
