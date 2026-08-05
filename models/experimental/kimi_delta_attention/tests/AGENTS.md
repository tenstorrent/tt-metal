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
- CPU-reference determinism uses T=128 and three repetitions. It is marked `long_running` and skips
  unless `KDA_RUN_LONG_TESTS=1`.
- Required performance acceptance is real Kimi-K3, B=1, T=5120 on SP1xTP8, SP2xTP4, and SP4xTP2.
- LoudBox references and regression limits live in `perf/perf_targets/bh_loudbox.json`.
- Fusion decisions and scoped historical evidence live in `perf/perf_targets/bh_loudbox_fusion_ab.json`.
- Rebaseline only when the workload, hardware/runtime contract, or accepted baseline changes.
- `model/test_real_weights.py` checks output and both states against the independent Torch reference
  and requires usable realtime program records.
- `perf/test_layer_perf.py` checks those endpoints on a synchronized eager forward, then measures
  warm trace-replay wall time. Timing repetitions are not accuracy or determinism samples.
- Use synchronized trace wall time for routine latency and Tracy only for targeted attribution.

## Catalogue

```text
tests/
├── conftest.py                         — Pinned checkpoint fixture plus perf and optional
│                                         long-running marker registration.
├── test_cache_fingerprints.py           — Persistent tensor and CPU-oracle cache identities.
├── utils.py                            — Three numeric contracts; case builders,
│                                         reconstruction, and profiling support.
├── checkpoint/
│   └── test_checkpoint.py              — Indexed-shard loading, failures, and weight
│                                         validation, and padded K3 A_log normalization.
├── reference/
│   ├── test_config.py                  — Config mapping and validation contracts.
│   ├── test_layer.py                   — Transition accuracy and optional bounded
│   │                                     bit-identical CPU-reference determinism.
│   ├── test_numeric_validation.py      — Accuracy, equality, bit-identity, and finiteness
│   │                                     contract coverage.
│   └── test_ops.py                     — Torch operation identities with complete
│                                         accuracy checks.
├── operations/
│   ├── test_chunk.py                   — chunk_kda accuracy, grouped invariance, and
│   │                                     bit-identical implementation determinism.
│   ├── test_convolution.py             — Fused four-tap Q/K/V convolution accuracy and
│   │                                     bit-identical implementation determinism.
│   ├── test_distributed_affine.py      — Prefix accuracy, cache/trace replay, and
│   │                                     bit-identical implementation determinism.
│   ├── test_gated_rms_norm.py          — Gated-RMS accuracy, cache/trace replay, and
│   │                                     bit-identical implementation determinism.
│   └── test_halo.py                    — Halo accuracy on both TP axes and
│                                         implementation determinism.
├── model/
│   ├── test_distributed_layer.py       — SP accuracy, segmented continuity, and
│   │                                     bit-identical implementation determinism.
│   ├── test_layer.py                   — Composed accuracy, state/cache contracts, and
│   │                                     bit-identical implementation determinism.
│   ├── test_real_weights.py            — Kimi-K3 layer-1 accuracy on all layouts
│   │                                     with required realtime program records.
│   └── test_weights.py                 — TP placement and output-projection accuracy plus
│                                         bit-identical projection determinism.
└── perf/
    ├── perf_targets/
    │   ├── bh_loudbox.json             — T=5120 trace-wall references and provenance;
    │   │                                 and one-sided regression limits.
    │   └── bh_loudbox_fusion_ab.json   — Historical MMRS, convolution, and gated-RMS A/B
    │                                     decisions with scope and provenance.
    └── test_layer_perf.py              — T=5120 accuracy and trace-wall acceptance on
                                          SP1xTP8, SP2xTP4, and SP4xTP2.
```

## Commands

Hermetic and real-weight correctness, excluding perf:

```bash
KIMI_K3_CKPT=/path/to/pinned/kimi-k3 \
scripts/run_safe_pytest.sh --run-all \
  models/experimental/kimi_delta_attention/tests \
  --ignore=models/experimental/kimi_delta_attention/tests/perf -q -s
```

Independent real-weight PCC:

```bash
KIMI_K3_CKPT=/path/to/pinned/kimi-k3 \
scripts/run_safe_pytest.sh \
  models/experimental/kimi_delta_attention/tests/model/test_real_weights.py -q -s
```

Required performance matrix:

```bash
KIMI_K3_CKPT=/path/to/pinned/kimi-k3 PERF_REPS=10 \
scripts/run_safe_pytest.sh --run-all \
  models/experimental/kimi_delta_attention/tests/perf/test_layer_perf.py -q -s
```

Optional bounded CPU-reference determinism:

```bash
KDA_RUN_LONG_TESTS=1 pytest -q -s \
  models/experimental/kimi_delta_attention/tests/reference/test_layer.py \
  -k reference_layer_determinism
```

`perf/perf_targets/bh_loudbox_fusion_ab.json` is historical decision evidence. Its controls and
raw Tracy captures are not part of the repository, so the historical A/B is not reproducible.

Add `--profile` only for a specific Tracy investigation and use an exact node ID; `run_safe_pytest.sh --profile` does not preserve a spaced `-k` expression as one argument.
