# V6 full chunk64 output/state with corrected FMA

The parent-run length128 component now reproduces **every FP32 output bit, every BF16 output bit, and every FP32 final-state bit** of the canonical HF chunk64 reference. This is the real propagated graph, with no cache transplant or oracle runtime inputs beyond the ordinary setup-only same-operand component fixture. It is not yet full decoder or latency acceptance.

The [wrapper](../../../tests/v6_chunk64_corrected_fma_output_probe.py) composes the existing scaled-MKL exponential, double-single cumulative log decay, canonical triangular sum, and full chunk output/state equations. It replaces only setup and `mm` dispatch with `CorrectedFma`/`CorrectedMatmul`. The CPU reference, output/state equations, logical-length padding/cropping, and quality gates are unchanged. The source is frozen at SHA `8568bee017b255a738b1034bb97e1fa4ff975be734307c4a1e6a93c7a594906c`.

## Measured comparison

Both rows use actual layer34/seed154 length128 operands, zero initial state, the same scaled-MKL exp/cumsum/triangular-sum helpers, and the unchanged HF oracle.

| GEMM implementation | FP32 output code differences | BF16 output code differences | FP32 final-state code differences | BF16 final-state code differences |
| --- | ---: | ---: | ---: | ---: |
| Native sequential addcmul (`v6_chunk64_mkl_exp_scaled_output_device.json`) | 140,102 | 12 | 98,856 | 16 |
| Corrected split-product/expansion FMA (`v6_chunk64_corrected_fma_output_device.json`) | **0** | **0** | **0** | **0** |

The corrected output and state PCC are exactly 1.0 in this component. The parent recorded clean device closure. Detailed intermediate comparisons, tensors, dependency snapshots, hashes, and command provenance are adjacent to the device JSON. The wrapper source was not edited while the parent ran hardware.

## CPU contract and source checks

CPU reconstruction passes against unchanged HF outputs and state for lengths **1, 65, 127, 128**, with both **zero** and **nonzero prefix64** initial state. These are separate CPU validations; hardware evidence for the extra lengths/states must come from the parent's runs. Artifacts:

- `v6_chunk64_corrected_fma_output_cpu_zero.json`, `.log`, `.cpu_matrices.pt`, `.provenance.json`, and dependency snapshots.
- `v6_chunk64_corrected_fma_output_cpu_prefix64.json`, `.log`, `.cpu_matrices.pt`, `.provenance.json`, and dependency snapshots.
- `v6_chunk64_corrected_fma_output_audit.json` is the initial source audit before a metadata-only correction; its own dependency snapshot preserves that earlier wrapper. Use the CPU setup/device artifacts above for the frozen current source.

Fourteen source dependencies are hashed and snapshotted. The metadata asserts unchanged output-call identity and attached state-call identity, and explicitly verifies that `mm` dispatch is installed on the corrected callable. The runtime AST audit finds no Torch or host conversions in the adapter or matrix wrapper. The original diagnostic observer still copies intermediates for comparison; omitting the observer leaves the mathematical runtime on device. Generic exponential tables are setup-owned mathematical constants, not oracle caches. Compatibility CLI labels do not enable a native-addcmul fallback.

Both nonaligned sequence padding and nonzero initial-state semantics are inherited unchanged. Internal M/N dimensions presented to the corrected GEMM are tile aligned by the existing chunk64 implementation. No public sequence alignment restriction or reduced context contract is introduced.

## Commands and remaining limits

CPU-only commands used:

```bash
HF_HOME=/home/ttuser/qwen36-35b-a3b/hf python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_chunk64_corrected_fma_output_probe --oracle-only --lengths 1 65 127 128 --initial-state zero --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_chunk64_corrected_fma_output_cpu_zero.json
HF_HOME=/home/ttuser/qwen36-35b-a3b/hf python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_chunk64_corrected_fma_output_probe --oracle-only --lengths 1 65 127 128 --initial-state prefix64 --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_chunk64_corrected_fma_output_cpu_prefix64.json
```

Parent-controlled hardware command:

```bash
python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_chunk64_corrected_fma_output_probe --lengths 128 --initial-state zero --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_chunk64_corrected_fma_output_device.json
```

The corrected primitive's proof is bounded; it is not an exact general FP32 FMA. Independently evaluated actual transformed-K and state-contribution matrices still show tiny differences, up to roughly `3.85e-34`, while their final state-update products are bit exact. The fully propagated length128 output/state result above is direct evidence and does not infer correctness from those isolated products. Broader nonaligned/nonzero hardware runs, actual decoder integration, long stress, watcher, and final performance remain necessary. Per-K launches/broadcasting/sharding are diagnostic and carry no warmed-latency claim. No production or previously measured source file was edited in this preparation.
