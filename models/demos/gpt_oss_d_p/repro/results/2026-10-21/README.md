# October 21, 2026 performance reproduction

- [Full reference-versus-observed report and all 58 cases](REPORT.md)
- [Per-request measurements: 58 warmups and 174 measured runs](SWEEP_RUNS.csv)
- [Means and sample variances](SWEEP_STATISTICS.csv)
- [Commands, prerequisites and measurement definitions](../../PERFORMANCE_MEASUREMENTS.md)
- [Final migration-stall diagnosis, regression and device retry](FINAL-CASE-INVESTIGATION.md)
- [Saved-evidence audit](sweep-audit.json)
- [Raw source/evidence archive location and SHA-256](source-evidence-archive.json)
- [Allocation release evidence](allocation-release.json)

No golden data or PCC checks were used. All cases validate execution and performance accounting, not numerical model accuracy. The original and repaired native-source cohorts are identified in the report and per-run CSV. Failed attempts remain in the raw evidence archive. Metal code commit `3441c663200ca87f7deafe7c48d39ba61c40d5c0` contains the harness; this results commit does not change measured code.
