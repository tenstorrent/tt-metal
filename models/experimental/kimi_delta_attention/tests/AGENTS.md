# Kimi Delta Attention tests

Run every device test through `scripts/run_safe_pytest.sh`. A passing hardware run must end with `SAFE_PYTEST_RESULT: PASS`; a skip is not a pass.

## Policy

- Hermetic tests use deterministic synthetic weights. Tests never switch behavior based on local checkpoint availability.
- Real-weight tests require an explicit `KIMI_K3_CKPT` path, provided by `conftest.py`.
- Required performance acceptance is real Kimi-K3, B=1, T=5120 on SP1xTP8, SP2xTP4, and SP4xTP2.
- LoudBox reference values and one-sided regression limits live in `perf/perf_targets/bh_loudbox.json`. Tests compare current measurements with those checked-in values; do not rerun the baseline arm for each change.
- Fusion decisions and their scoped real-K3 A/B evidence live in `perf/perf_targets/bh_loudbox_fusion_ab.json`. Full-layer and isolated-component numbers are labeled explicitly; do not compare them as the same metric.
- Rebaseline only when the workload, hardware/runtime contract, or accepted implementation baseline changes. Update provenance and targets in a dedicated reviewable commit; keep full logs and Tracy artifacts outside Git.
- A perf case checks output, recurrent-state, and convolution-state PCC on its realtime-profiled forward before trace timing. Its device-repeat PCC is reproducibility evidence; `model/test_real_weights.py` supplies the independent Torch-reference PCC gate.
- Realtime capture is used once for correctness and program records, not inside timing loops. On SP1xTP8 it increased one warm forward from 10.803 ms to 112.108 ms because record collection is host-heavy.
- Use synchronized trace wall time for routine latency. Use Tracy only for overlap, unexplained regressions, or per-core/kernel attribution. The MMRS decision required Tracy because standalone program sums do not expose its critical-path overlap.

## Catalogue

| Test | Intent | Required? |
|---|---|---|
| `reference/test_config.py` | Model-config mapping and validation contracts | Yes |
| `reference/test_layer.py` | Stateless full-layer transition, segmented equivalence, and caller-owned state immutability | Yes |
| `reference/test_ops.py` | Independent Torch operation identities | Yes |
| `checkpoint/test_checkpoint.py` | Indexed-shard loading, failure contracts, and padded K3 normalization | Yes; evolve with checkpoint API |
| `operations/test_chunk.py` | Direct `chunk_kda` PCC across layouts, shapes, fidelity, and grouped summaries | Yes |
| `operations/test_convolution.py` | Direct fused four-tap Q/K/V convolution PCC | Yes while fused convolution remains |
| `operations/test_halo.py` | 2D-mesh convolution-halo correctness on both TP axes | Yes while SP convolution remains |
| `operations/test_distributed_affine.py` | Distributed-prefix equivalence, cache reuse, and trace replay | Yes while SP affine prefix remains |
| `operations/test_gated_rms_norm.py` | Direct K3-geometry gated-RMS PCC, cache reuse, and trace replay | Yes while KDA uses the fused gated-RMS operation |
| `model/test_layer.py` | Small synthetic composed-layer PCC, offline cache/cache-only construction, validation, segmented state, and external state | Yes |
| `model/test_distributed_layer.py` | Synthetic SP layer PCC and segmented-prefill state continuity | Yes |
| `model/test_weights.py` | TP placement plus TP/2D composed-layer correctness | Yes; direct placement and integration contracts |
| `model/test_real_weights.py` | Offline cache build and cache-only independent Torch-reference PCC with pinned Kimi-K3 layer-1 weights; correctness forward emits realtime records | Yes; primary real-weight accuracy gate |
| `perf/test_layer_perf.py` | Real-K3 T=5120 profiled-result PCC, profiler overhead, records, and trace latency on the three target layouts | Yes; primary perf acceptance |
| `perf/perf_targets/bh_loudbox.json` | Versioned LoudBox trace-wall references, workload provenance, and regression limits | Yes; executable perf source of truth |
| `perf/perf_targets/bh_loudbox_fusion_ab.json` | Versioned MMRS, convolution, and gated-RMS A/B evidence, scope, provenance, and reproduction | Yes; source of truth for fusion decisions |
| `perf/test_fusion_ab.py` | Real-K3 fused/unfused convolution and gated-RMS PCC/perf experiments | Development evidence for retained fusions |
| `perf/test_operation_perf.py` | Exact-shape `chunk_kda` microprofile | Development probe; keep while public op is tuned |
| `perf/test_distributed_operation_perf.py` | Exact K3 SP4xTP2 distributed-prefix microprofile | Development probe; keep while SP prefix is tuned |
| `utils.py` | Deterministic synthetic config/weight builders | Support module, not a test |
| `conftest.py` | Explicit pinned Kimi-K3 checkpoint fixture | Support module, not a test |

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

Retained-fusion A/B matrix:

```bash
KIMI_K3_CKPT=/path/to/pinned/kimi-k3 KDA_FUSION_AB_REPS=10 \
scripts/run_safe_pytest.sh --run-all \
  models/experimental/kimi_delta_attention/tests/perf/test_fusion_ab.py -q -s
```

The MMRS result in `perf/perf_targets/bh_loudbox_fusion_ab.json` is retained decision evidence. Its removed control
implementation and raw Tracy capture are not part of this PR, so that historical A/B is not directly reproducible.

Add `--profile` only for a specific Tracy investigation and use an exact node ID; `run_safe_pytest.sh --profile` does not preserve a spaced `-k` expression as one argument.
