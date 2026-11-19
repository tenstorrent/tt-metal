# Offline evidence and streaming review

This is a source/CPU review, not stage acceptance or a hardware result.
No runtime files were changed and no device was opened.

The streaming join now rejects fallback IDs regardless of host CSV operation
type, and rejects newly enqueued device operations inside the decode replay
window. Its report adapter also binds the supplied profile JSON to the exact
SHA256 saved by the merge. Negative tests cover all three cases. Existing
checks retain exact dispatcher-ID ranges, original markers, complete execution
ordinals, increasing replay cycle intervals, final-replay selection, loss and
flush diagnostics, and the exact selected operation count after tt-perf-report.

The functional-decoder skill explicitly permits reporting the filtered
`Device Time` kernel-time column with its units. The proposed output therefore
provides warmed on-device performance evidence and human-readable tt-perf-report
tables, subject to successful hardware collection. It reports per-operation
durations and their sum; it does not measure whole-pass device elapsed latency.
All gap fields remain blank/null. Host replay timing is diagnostic only.
Streaming instrumentation and producer stalls can perturb execution; source
shows stall counters and warnings, which the merge retains in provenance.
Even zero reported stalls do not prove zero instrumentation overhead. There is
no measured uninstrumented-overhead estimate and no such claim.

`tests/summarize_functional_evidence.py` adds a CPU-only inventory. Run:

```bash
PYTHONPATH=. python_env/bin/python \
  -m models.autoports.qwen_qwen3_8_flash_next.tests.summarize_functional_evidence \
  --output /home/ttuser/qwen38-flash/functional_evidence_inventory.json
```

The output records the context contract verbatim, process status/exit code,
command arguments, changed runtime sources, changed recorded test sources,
and log hashes. It verifies concatenated archive parts, compressed identities,
decompressed identities/sizes, and separately checks any remaining original.
An archive matching its historical source does not validate a changed original.
The helper never rewrites artifacts, archive manifests, status, or context claims.

The initial inventory found 43 valid archives, all with matching local
originals, and 70 process provenance records. Of these, 64 have runtime source
differences and six match the recorded runtime; matching source alone does not
imply a passed test (some matching records are failed or still running).
These counts are a point-in-time inventory while the parent is still working,
not final stage evidence. Whole-runtime comparison is conservative: an
unrelated module change still appears explicitly for dependency review.

By default all 20 required gate categories are unbound. Optional
`--gate-manifest` accepts explicit artifact/provenance paths, scope, required
source identities and JSON-pointer checks; its schema is in the helper's
docstring. Successful checks are labeled `declared_checks_satisfied_review_required`,
never `clean-pass`. Source staleness, unsuccessful processes, changed command
logs, unbound output paths and failed predicates remain unmet. Current artifact
hashes are recorded, but historical `run_recorded` provenance does not include
result-file hashes: matching commands/logs cannot prove an output was not
subsequently rewritten. This limitation stays explicit. Stage review must
assess complete scope, physical capacity, every layer kind, and final commits.

CPU tests:

```bash
PYTHONPATH=. python_env/bin/python \
  -m models.autoports.qwen_qwen3_8_flash_next.tests.test_streaming_profile_merge_cpu
PYTHONPATH=. python_env/bin/python \
  -m models.autoports.qwen_qwen3_8_flash_next.tests.test_functional_evidence_cpu
```

These exercise the actual host CSV parser and tt-perf-report, plus split
archive verification, stale originals, changed runtime sources, failed PCC
predicates and deliberately unbound gates. TTNN is not imported.
