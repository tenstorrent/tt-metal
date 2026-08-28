# Kimi Delta Attention tests

Run every device test through `scripts/run_safe_pytest.sh`. A passing hardware run must end with `SAFE_PYTEST_RESULT: PASS`; a skip is not a pass.

## Policy

- Hermetic tests use deterministic synthetic weights and never depend on local checkpoint discovery.
- Real-weight tests require an explicit `KIMI_K3_CKPT` path, provided by `conftest.py`.
- Numerical result assertions use only the three contracts in `utils.py`; do not add local wrappers
  or direct Torch tensor comparisons.
- `assert_accurate` compares an oracle with an implementation: both tensors must be finite and their
  PCC must meet the test's threshold.
- `assert_equal` checks finite oracle/implementation tensors for identical shape, dtype, and values.
- `assert_bit_identical` compares implementation repetitions without computing a CPU oracle.
- Dedicated determinism tests run the implementation three times from identical inputs and state.
  They do not compute a CPU reference; every output and final-state tensor must be bit-identical.
- CPU-reference determinism uses T=32 and compares two repetitions with the initial result.
- Required performance acceptance is real Kimi-K3, B=1, T=5120 on SP1xTP8, SP2xTP4, and SP4xTP2.
- LoudBox references, five-session dispersion, and regression limits live in `perf/perf_targets/bh_loudbox.json`.
- Rebaseline only when the workload, hardware/runtime contract, or accepted baseline changes.
- `model/test_real_weights.py` checks output and both states against the independent Torch reference
  and requires usable realtime program records.
- `perf/test_layer_perf.py` checks those endpoints on a synchronized eager forward, then gates the
  median of five warm trace-replay samples. Its 900-second item timeout covers a cold CPU-oracle cache.
  Timing repetitions are not accuracy or determinism samples.
- Use synchronized trace wall time for routine latency and Tracy only for targeted attribution.

## Catalogue

```text
tests/
├── conftest.py                         — Pinned checkpoint fixture plus perf marker registration.
├── checkpoint_utils.py                 — Indexed Kimi-K3 layer loading for tests.
├── test_cache_fingerprints.py           — Persistent checkpoint-content, config, and placement identities.
├── test_numeric_validation.py          — Accuracy, equality, bit-identity, and finiteness contracts.
├── utils.py                            — Three numeric contracts; case builders,
│                                         reconstruction, and profiling support.
├── checkpoint/
│   └── test_checkpoint.py              — Indexed-shard loading, failures, and weight
│                                         validation, and padded K3 A_log normalization.
├── operations/
│   ├── test_chunk.py                   — chunk recurrence accuracy, grouped invariance, and
│   │                                     bit-identical implementation determinism.
│   ├── test_distributed_affine.py      — Prefix accuracy, cache/trace replay, and
│   │                                     bit-identical implementation determinism.
│   └── test_halo.py                    — Halo accuracy on both TP axes and
│                                         implementation determinism.
├── model/
│   ├── test_distributed_layer.py       — SP accuracy, segmented continuity, and
│   │                                     bit-identical implementation determinism.
│   ├── test_layer.py                   — Composed accuracy, state/cache contracts, immutable
│   │                                     trace replay, and bit-identical determinism.
│   ├── test_real_weights.py            — Kimi-K3 layer-1 accuracy on all layouts
│   │                                     with required realtime program records.
│   └── test_weights.py                 — TP placement and output-projection accuracy plus
│                                         bit-identical projection determinism.
└── perf/
    ├── perf_targets/
    │   ├── bh_loudbox.json             — T=5120 median trace-wall references, session dispersion,
    │   │                                 provenance, and one-sided regression limits.
    └── test_layer_perf.py              — T=5120 accuracy and trace-wall acceptance on
                                          SP1xTP8, SP2xTP4, and SP4xTP2.
```

Direct tests for the six split device operations live under
`tests/ttnn/nightly/unit_tests/operations/experimental/kda/`; do not duplicate
them in the model test tree.

CPU-reference tests live beside the implementation:

```text
models/demos/deepseek_v3_d_p/reference/kda/tests/
├── test_config.py                       — Config mapping and validation contracts.
├── test_layer.py                        — Transition accuracy and bit-identical determinism.
└── test_ops.py                          — Torch operation identities and accuracy checks.
```

## Commands

Hermetic and real-weight correctness, excluding perf:

```bash
KIMI_K3_CKPT=/path/to/pinned/kimi-k3 \
scripts/run_safe_pytest.sh --run-all \
  models/demos/deepseek_v3_d_p/tests/kda \
  models/demos/deepseek_v3_d_p/reference/kda/tests \
  --ignore=models/demos/deepseek_v3_d_p/tests/kda/perf -q -s
```

Independent real-weight PCC:

```bash
KIMI_K3_CKPT=/path/to/pinned/kimi-k3 \
scripts/run_safe_pytest.sh \
  models/demos/deepseek_v3_d_p/tests/kda/model/test_real_weights.py -q -s
```

Required performance matrix:

```bash
KIMI_K3_CKPT=/path/to/pinned/kimi-k3 PERF_REPS=10 \
scripts/run_safe_pytest.sh --run-all \
  models/demos/deepseek_v3_d_p/tests/kda/perf/test_layer_perf.py -q -s
```

Add `--profile` only for a specific Tracy investigation and use an exact node ID; `run_safe_pytest.sh --profile` does not preserve a spaced `-k` expression as one argument.
