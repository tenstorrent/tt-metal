# AutoFix Report: nanobind shutdown diagnostics

## Starting Evidence

- `../logs/full_suite.log` and `../watcher/pytest.log` pass their tests and close
  the device cleanly, then report two leaked nanobind instances
  (`MemoryConfig`, `CoreRangeSet`), 20 types, and 250 functions.
- Hypothesis: the new functional decoder retains TTNN binding objects and owns
  the shutdown diagnostic.
- Prediction: a stage-specific no-device import/control should reproduce the
  warning while an unrelated repository pytest should not.

## Hypothesis Experiments

- Hypothesis: base TTNN import universally emits the diagnostic.
  Experiment: `python autofix/nanobind_import_control.py`.
  Result: exit 0; no nanobind warning.
  Verdict: refuted.
  Evidence: `nanobind_import_control.log`.

- Hypothesis: importing the stage implementation or its existing shared Qwen
  dependencies is sufficient to emit the diagnostic.
  Experiments:
  `python autofix/nanobind_shared_qwen_control.py`,
  `python autofix/nanobind_shared_gated_control.py`,
  `python autofix/nanobind_shared_test_utility_control.py`, and
  `python -c 'import models.autoports.qwen_qwen3_8_27b.tests.test_functional_decoder'`.
  Result: every command exits 0 without the warning.
  Verdict: refuted.
  Evidence: corresponding `nanobind_*control.log` files.

- Hypothesis: the warning is stage-owned rather than shared pytest/binding
  teardown behavior.
  Experiment A:
  `pytest -q models/autoports/qwen_qwen3_8_27b/tests/test_functional_decoder.py::test_target_config_contract -s`.
  Experiment B (unrelated pre-existing, CPU-only control):
  `pytest -q models/experimental/gated_attention_gated_deltanet/tests/test_gated_attention.py::test_gated_attention_output_shape -s`.
  Result: both tests pass and exit 0 without opening TT hardware; both emit the
  same two instance classes, 20 leaked types, and 250 leaked functions.
  Verdict: stage ownership refuted; verified shared pytest/nanobind teardown
  behavior in the existing gated-attention test environment.
  Evidence: `nanobind_decoder_import_control.log` and
  `nanobind_unrelated_gated_pytest_control.log`.

- Control: an unrelated pure-Python repository pytest
  (`tests/sweep_framework/test_homogenize_master_json.py::test_merges_distinct_configs_under_same_operation`)
  passes without the warning, confirming it is tied to the TT/gated binding
  test environment rather than pytest alone. Evidence:
  `nanobind_unrelated_pytest_control.log`.

## Final Status

- Limitation outside stage ownership; no stage code fix is justified.
- The warning occurs after successful test completion, including in an unrelated
  CPU-only existing test, and is not evidence of a device hang, leaked stage
  cache, watcher failure, or improper device close.
- All controls were intentionally no-device, so they did not conflict with
  concurrent hardware work.
