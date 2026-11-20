# Marking failures non-retryable (`ci-no-retry`)

Nightly auto-retry (`.github/workflows/_auto-retry-nightly-workflows.yaml`) re-runs the failed jobs of
the workflows it watches, up to 3 attempts. Some failures repeat on every re-run of the same commit,
for example a perf value outside its baseline band. Re-running them wastes runners and delays the red
result. A job can mark such a failure, and then:

- if every unsuccessful job of a run is marked, auto-retry does not re-run the run;
- otherwise the run is re-run as before, and each marked job fails within seconds at its first step
  (`skip-non-retryable`) instead of running again. This applies only to re-runs started by auto-retry: a
  re-run started by hand (UI, `gh run rerun`) runs every job again, so a perf miss can be measured again.

Everything is opt-in. A job that doesn't opt in, or whose records are missing, is retried exactly as before.
Any API error also leads to a normal re-run.

## What counts as non-retryable

A job is non-retryable if **at least one** of its failures is a perf verdict from a run that completed.
That verdict repeats on a re-run (by policy), so the job cannot turn green whatever else failed with it.

| Marked | Not marked (still retried) |
|---|---|
| A measured perf value outside its band, slower or faster | No measurement: no perf JSON, profiler inactive, nothing timed |
| A baseline that no longer matches the test's shape (e.g. deeper context) | A perf verdict after a crashed producer or runner |
| | Hangs, crashes, timeouts, setup or runner failures |
| | Accuracy / PCC failures, pytest usage errors |

## Opting in

### A workflow job

```yaml
permissions:
  actions: read # skip-non-retryable reads the previous attempt's jobs
  checks: read # ... and their annotations
jobs:
  test:
    env:
      # Unique per run, attempt and job. Put it on shared storage if tests write from other hosts.
      CI_FAILURE_RECORDS: /ci/ci_failure_records/${{ github.run_id }}-${{ github.run_attempt }}/<unique job key>
    steps:
      - uses: actions/checkout@...
      - name: Skip if the previous attempt failed deterministically
        id: skip-non-retryable
        uses: ./.github/actions/skip-non-retryable # first, before any expensive setup
        with:
          job-id: ${{ job.check_run_id }}
      # ... setup ...
      - name: Run tests
        id: run-test
        run: ...
      - name: Classify failure for auto-retry
        if: failure() && steps.run-test.outcome == 'failure'
        uses: ./.github/actions/classify-failure # before any step that re-echoes the test log
```

The workflow must be listed in the auto-retry workflow's `workflow_run.workflows`. A reusable workflow's
callers must also grant `actions: read` and `checks: read`. If tests write records on other hosts (e.g.
under `mpirun`), make sure the variable reaches them; see the per-rank check in
`blaze-models-prefill-tests-impl.yaml`.

### A pytest test

Raise `PerfGateFailure` for the perf verdict; the root `conftest.py` records it. It subclasses
`AssertionError`, so pytest output, `-x` and `pytest.raises(AssertionError)` are unchanged.

```python
from models.common.ci_failure import assert_perf_gate

assert_perf_gate(lower <= measured <= upper, f"{measured:.3f} ms outside [{lower:.3f}, {upper:.3f}] ms")
```

### A script

From Python outside pytest, call `models.common.ci_failure.record_deterministic_failure(reason)`. Do this
before any `except` that would swallow the failure. From a shell script, run
`python3 -m models.common.ci_failure "<reason>"`, or write a one-line `.txt` file into
`$CI_FAILURE_RECORDS` yourself (write it under another name, then rename it into place).

Only record a verdict from a run that completed. If the run crashed first, leave `CI_FAILURE_RECORDS`
unset for the gate; see `run_multirank_pcc.sh`.

### Tests of gates

A test that triggers a gate failure on purpose, inside a job that opted in, must not record it. Under
pytest, a failure that the test catches is never recorded. A direct `record_deterministic_failure` call
records immediately, so blank `CI_FAILURE_RECORDS` around it, as `test_run_multirank_pcc.py` does.

## Pieces

| File | Role |
|---|---|
| `models/common/ci_failure.py` | `PerfGateFailure`, `assert_perf_gate`, `record_deterministic_failure`, CLI. Stdlib only. |
| `conftest.py` (`pytest_runtest_makereport`) | Records any `DeterministicFailure` a test raises |
| `.github/actions/classify-failure` | Records → one `ci-no-retry` annotation, in its own step (GitHub caps annotations per step) |
| `.github/actions/skip-non-retryable` | On a re-run, fails a job whose previous attempt was marked |
| `.github/actions/skip-non-retryable/ci_no_retry.py` | The API logic for both. Runnable locally: `GH_TOKEN=$(gh auth token) python3 ci_no_retry.py run-decision --run-id <id> --attempt <n>` |
