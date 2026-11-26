# KDA single-op tests

One standard way to test a `ttnn.experimental.kda` op. Every op gets the same file layout, the same checks under
the same names, and the same way to select tests.

Status: `test_summarize_chunk_recurrence.py` follows the standard; the other op modules do not yet.

## Files

| File | Holds |
| --- | --- |
| `test_<binding>.py` | One per public binding: the op's cases, tolerances, harness and tests. |
| `kda_op_checklist.py` | The checklist: each item's name pattern, selection group, whether it is enforced, and why it exists. The single source for names. |
| `kda_op_checks.py` | The shared checks a test calls (output contract, determinism, cache rebinding, default config, program counts, device-time band). Op tests import measurement and profiler helpers only through it. |
| `recurrent_chunk_scan_test_utils.py` | The recurrence protocol and its golden oracles. |
| `conftest.py` | Fixtures, and the hook that marks tests for selection. |
| `test_kda_op_standard_conformance.py` | Checks every module under the standard against this README. No device needed. |

## Adding an op

1. Create `test_<binding>.py` with these sections, in this order:
   - a docstring: the op's contract in one or two sentences, and which models reach it;
   - `pytestmark` with the module device and its settings;
   - `NOT_APPLICABLE`: each enforced checklist item the op has no test for, with the reason. A module that declares
     it is under the standard: it gets selection marks and conformance checks;
   - cases: a frozen `Case` dataclass. Every case has an `id` and a `production` flag; production cases carry the
     model's geometry, and performance cases their measured device-time reference with where and when it was measured;
   - tolerances: `checks.Tolerance` per case or output, from the clean value measured for it plus headroom;
   - the harness: `host_inputs`, `reference`, `device_inputs`, `run` (the only place the binding is called),
     and `work` when the op has a performance model;
   - the tests, grouped by checklist group in checklist order.
2. Name each test `test_<binding>_<suffix>` with a suffix from `kda_op_checklist.py`. Parametrize device tests over
   cases with `checks.case_params(...)`, which sets the ids and marks production cases.
3. Call one shared check per standard test. Write op-specific assertions next to it in the test body.
4. Run `pytest test_kda_op_standard_conformance.py`. It names any enforced item that is missing or undeclared.

`test_summarize_chunk_recurrence.py` is the reference module.

## Checklist

`kda_op_checklist.py` lists every item with its reason. The enforced items:

| Leaf | Test suffix | Group | Shared check |
| --- | --- | --- | --- |
| 1a | `accuracy` | accuracy | `assert_close` |
| 3 | `is_deterministic` | determinism | `assert_deterministic`: 10 eager runs, 10 trace replays, replay vs eager |
| 4a | `rejects_<what>` | contract | one table per test: id → (mutation, message) |
| 4b | `default_compute_config_matches_explicit` | contract | `assert_default_config_matches` |
| 4f | `output_contract` | contract | `assert_output_contract` |
| 5a | `cache_hit_rebinds_fresh_tensors` | caching | `assert_cache_hit_rebinds` |
| 7a | `performance` | perf | `@checks.performance_test` and `assert_device_time` (two-sided band) |
| 7c | `work_golden` | perf | — |

The other items are candidates. Their names are reserved, so tests written for them now already conform.

## Selecting tests

The conftest marks every test of a module under the standard twice: with its checklist group, taken from its
name, and with `host` (no device), `unit` (a device test without a production case) or `production`. Combine
them with `-m`:

| To run | Command |
| --- | --- |
| Everything for one op | `pytest test_<binding>.py` |
| All but performance | `-m "not perf"` |
| Fast development loop | `-m "unit and not perf"` |
| Unit performance | `-m "perf and unit"` |
| Accuracy and determinism | `-m "accuracy or determinism"` |
| Production geometry only | `-m production` |
| No device | `-m host` |

Groups: `accuracy`, `stability`, `determinism`, `contract`, `caching`, `safety`, `perf`.

Use `-m`, not `-k`, to select groups or shapes. `-k` matches substrings of the test's path too: `-k unit` selects
everything under `unit_tests/`, and `-k "not performance"` deselects all of `test_kda_performance_model.py`.
Performance tests still end in `_performance`, so CI's existing `-k "not performance"` filter keeps them out of
the functional jobs. `@checks.performance_test` adds `requires_host_iommu`, which sends them to the profiler job.
