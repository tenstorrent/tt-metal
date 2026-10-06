# V4 validation evidence generator

Run from the checkout root; this command only reads evidence and writes the
two summary JSONs. It does not import TTNN or access hardware.

```bash
python models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/summarize_validated_v4.py
```

Outputs are `validated_v4_validation_summary.json` and
`validated_v4_watcher_summary.json`. Exit status is 0 for a complete pass,
1 for failed or contradictory evidence, and 2 for incomplete evidence. There
is no polling. `--output-dir /tmp/gemma_v4_summary_check` writes a separate
snapshot without replacing the canonical summaries.

The generator pins runtime
`3d51014f98128dfb21bb484fcece50993524b967754f68f6dfe825ae7f472ba9` and checks:

- All 18 completed command records and the 16 named public-contract reports,
  with matching runtime, fixtures, default policies, and process results.
- B32 heterogeneous positions, prefix/cache preservation, request reuse,
  BF16-cache override coverage, and traced headline checks.
- Both attention kinds at 262144 and 262143, including the exact 291-query
  sample set, each row's numeric .995 threshold, explicit enforced row-gate
  flag, aggregate/end-query/decode checks, and saved oracle hashes.
- Exactly four passing pytest cases, with each saved optimized case report
  checked against the frozen runtime and its recorded-input metadata.
- Both 1025/512 stress streams using the four raw HF reports, nested command
  journal, raw direct-comparison rows, report hashes, and saved tensor hashes.
- Watcher initialization, no disabled features, clean device/console logs,
  completion, and profiler separation from the captured validation-process
  environment and saved driver source.

`validated_v4_environment_snapshot.json` contains only whitelisted TT debug
and profiler variables read from the live orchestrator. The exact driver is
retained as `validated_v4_driver_snapshot.txt`; its hash binds the documented
watcher interval and inherited environment. This is evidence about that
validation process, not the later summary process's environment.

The final execution found 18 commands, no pending evidence, and no errors.
Each of the four long-context runs passed all 291 rows. Minimum long-row PCC
was .995099172920 for sliding and .996014477122 for full attention, at both
lengths. The paired stress minimum optimized HF decode PCCs were
.995525862233 and .996240662329; direct fused-comparison minima were
.995146483599 and .996320298618.

`validated_v4_summary_generator_checks.json` records 11 successful negative
checks: stale runtime, missing strict gate, below-threshold row with a passing
flag, missing query row, unexpected override, profiler-enabled watcher,
failed stress subprocess, failed direct comparison, failed public process,
missing final command, and a watcher error line. Python compilation and Black
with Python 3.10/120-column settings also passed.

The generator excludes historical `verified_*` artifacts from current
completion: their e810 runtime used aggregate and endpoint gates and had
diagnostic interior sliding misses. It does not reinterpret those old results
as strict row-gate passes. Sampled layer checks are not all-token or full-model
accuracy claims. Multi-GB input fixtures are matched by recorded manifest/run
hashes rather than rehashed; compact saved oracles and stress tensors are
rehashed. No runtime or test-acceptance code was edited by this packaging task.
