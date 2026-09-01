# Kimi Delta Attention tests

Run every device test through `scripts/run_safe_pytest.sh`. A passing hardware run must end with `SAFE_PYTEST_RESULT: PASS`; a skip is not a pass.

## Policy

- Hermetic tests use deterministic synthetic weights and never depend on local checkpoint discovery.
- Real-weight tests require an explicit `KIMI_K3_CKPT` path, provided by `conftest.py`.
- Numerical result assertions import the three shared contracts from `tests/ttnn/unit_tests/operations/experimental/kda/kda_test_utils.py`; do not add local wrappers
  or direct Torch tensor comparisons.
- `assert_accurate` compares an oracle with an implementation: shape and dtype must match, both tensors
  must be finite, and PCC must meet the test's threshold.
- `assert_equal` checks finite oracle/implementation tensors for identical shape, dtype, and values.
- `assert_bit_identical` requires matching shape and dtype, finite values, and identical bit patterns across
  implementation repetitions without computing a CPU oracle.
- Matching accuracy and determinism cases share one three-run workload. They retain the first result
  for the CPU oracle and reduce repeat mismatches on device before transferring scalar evidence.
- CPU-reference determinism uses T=32 and compares two repetitions with the initial result.
- Required performance acceptance is real Kimi-K3, B=1, T=5120 on SP1xTP8, SP2xTP4, and SP4xTP2.
- LoudBox references and their symmetric 3% regression limits live beside the performance test.
- Rebaseline only when the workload, hardware/runtime contract, or accepted baseline changes.
- `model/test_real_weights.py` checks output and both states against the independent Torch reference.
- `model/test_synthetic_kimi_k3.py` provides checkpoint-free production-dimension CI accuracy and
  device-side determinism, with SP2xTP4 local validation and SP8xTP4 Blaze selection.
- `perf/test_layer_perf.py` checks those endpoints on synchronized eager and trace-replay forwards,
  then gates the median of five warm trace-replay samples. Its 900-second item timeout covers a cold CPU-oracle cache.
  Timing repetitions are not accuracy or determinism samples. The SP2xTP4 case also logs a non-additive,
  overlap-aware per-device-program breakdown from one separate warm eager forward once after the five gated samples.
  Its checkpoint-free SP8xTP4 case is initially calibration-only until a hosted high-power Galaxy baseline is recorded.
- Use synchronized trace wall time for routine latency and Tracy only for targeted attribution.

## Catalogue

```text
tests/
├── conftest.py                         — Pinned checkpoint fixture plus perf marker registration.
├── checkpoint_utils.py                 — Indexed Kimi-K3 layer loading for tests.
├── test_cache_fingerprints.py           — Persistent checkpoint-content, config, and placement identities.
├── test_host_config.py                 — Host-only recurrence and program-config validation.
├── test_weight_schema.py               — Host-only TT weight canonicalization and validation.
├── utils.py                            — Case builders, reconstruction, and device-case support.
├── checkpoint/
│   └── test_layer_checkpoint.py        — Indexed loading and failure contracts for one KDA layer.
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
│   │                                     against the independent Torch reference.
│   ├── test_synthetic_kimi_k3.py       — Checkpoint-free production K3 accuracy and determinism.
│   └── test_weights.py                 — TP placement and output-projection accuracy plus
│                                         bit-identical projection determinism.
└── perf/
    └── test_layer_perf.py              — T=5120 local real-weight acceptance plus checkpoint-free
                                          SP8xTP4 Galaxy performance calibration/acceptance.
```

Shared numeric assertion contract tests live at
`tests/ttnn/unit_tests/operations/experimental/kda/test_kda_test_utils.py`.

Direct tests for the six split device operations live under
`tests/ttnn/nightly/unit_tests/operations/experimental/kda/`; do not duplicate
them in the model test tree.

CPU-reference tests live beside the implementation:

```text
models/demos/deepseek_v3_d_p/reference/kda/tests/
├── test_config.py                       — Config mapping and validation contracts.
├── test_layer.py                        — Transition accuracy and bit-identical determinism.
├── test_ops.py                          — Torch operation identities and accuracy checks.
└── test_weights.py                      — Reference weight-name and shape validation.
```

## Commands

Hermetic and real-weight correctness, excluding perf:

```bash
KIMI_K3_CKPT=/path/to/pinned/kimi-k3 \
scripts/run_safe_pytest.sh --run-all \
  models/demos/deepseek_v3_d_p/tests/kda \
  models/demos/deepseek_v3_d_p/reference/kda/tests \
  tests/ttnn/unit_tests/operations/experimental/kda/test_kda_test_utils.py \
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
KIMI_K3_CKPT=/path/to/pinned/kimi-k3 KDA_PERF_SKU=bh_loudbox \
scripts/run_safe_pytest.sh --run-all \
  models/demos/deepseek_v3_d_p/tests/kda/perf/test_layer_perf.py -q -s
```

Add `--profile` only for a specific Tracy investigation and use an exact node ID; `run_safe_pytest.sh --profile` does not preserve a spaced `-k` expression as one argument.
