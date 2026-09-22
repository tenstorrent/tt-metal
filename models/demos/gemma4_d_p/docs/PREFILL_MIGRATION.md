# Gemma4 prefill migration tests

Both tests run on TT hardware using input tokens and reference KV from a GPU capture.

| Test | Purpose |
| --- | --- |
| Mock | Compare prefill KV against the GPU reference and check sampled migration-table addresses. No migration endpoint is needed. |
| Loopback | Perform the same checks and copy KV between local cache slots through a migration endpoint, verifying that destination bytes match the source. |

## Run

From the repository root, apply the [environment setup](PREFILL_SERVICE.md). Set `PREFILL_TRACE_DIR` to select a GPU capture, or use the adapter's default.

Run the mock test:

```bash
pytest models/demos/gemma4_d_p/tests/test_prefill_migration.py::test_prefill_migration[mock-256k] -sv
```

For loopback, first [build and start the migration endpoint](PREFILL_TEST_FLOWS.md#loopback-16k), then run:

```bash
GEMMA4_TEST_LOOPBACK=1 pytest models/demos/gemma4_d_p/tests/test_prefill_migration.py \
    -k loopback -sv --basetemp=/tmp/gemma4-migration-loopback
```

The loopback command runs all loopback cases. List available cases with:

```bash
pytest models/demos/gemma4_d_p/tests/test_prefill_migration.py --collect-only -q
```

Use a quoted node ID from this list to select another case.

## Reported metrics

The test prints a table with one row per layer and an overall row, pooling all compared heads:

| Metric | Meaning |
| --- | --- |
| PCC | Pearson correlation between TT and GPU values; higher is better. |
| Relative RMSE | RMS error divided by the GPU reference's RMS magnitude; lower is better. |
| RMSE | Root mean squared error in the tensor's units; lower is better. |

The JSON report also includes per-head PCC, minimum PCC by cache type, and validation timings. Pass/fail criteria are defined in the [test](../tests/test_prefill_migration.py).

Each case saves `runner.log`, `producer.log`, and `gemma4_slot*.json` in its pytest temporary directory. Follow `runner.log` for live progress; the metrics table appears in the pytest console after validation.
