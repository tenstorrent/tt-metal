# Full-output scaled MKL exp wrapper preparation

`tests/v6_chunk64_mkl_exp_scaled_output_probe.py` is frozen at SHA256 `5b75128be0dda666823c012ad28c6aa5e8853929111e57628a1f89f452576477`. It reuses the full-output graph unchanged and replaces only graph initialization and exp dispatch in process. The replacement helper is frozen at `c55f1249fae3671bdf98496d582346a947f5d27163ac2bc8a43b224346738748`. Both state and output terms use the same scaled helper.

The AST/setup-only command passed without importing TTNN or reconstructing the model:

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_chunk64_mkl_exp_scaled_output_probe --audit-only --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_chunk64_mkl_exp_scaled_output_setup.json
```

The JSON records output/state method identity checks, the unchanged state-method AST proof, the scaled exp AST proof, and ten source dependency hashes with adjacent exact snapshots. Python compilation and Black passed. Existing source files were not edited. Prior CPU full-output/state evidence remains `v6_chunk64_output_cpu_zero.json` and `v6_chunk64_output_cpu_nonzero.json`; this setup did not repeat it or claim scaled device agreement.

Parent-owned device command:

```bash
HF_HOME=/home/ttuser/qwen36-35b-a3b/hf python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_chunk64_mkl_exp_scaled_output_probe --lengths 128 --variant sfpu_fma --sum-order canonical --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_chunk64_mkl_exp_scaled_output_device.json
```

The helper deliberately flushes sufficiently negative inputs outside its proven normal range; the remaining SFPU addcmul rounding limitation also remains. Component results must decide whether these approximations satisfy the integrated accuracy gates.

## Standalone snapshot builder strategy — integration held pending results

Use a separate builder under tests, with ordinary Python AST manipulation performed only by the builder. Pin every input SHA before reading. Emit ordinary static Python definitions to a new endurance snapshot; emit no `exec`, `compile`, AST imports, or diagnostic-test imports in the decoder snapshot. Preserve `dcaa36081fbce0169fb625a7ff2438643b0180a6a34379d96694330b6cdb668c` unchanged as the source baseline.

1. Retain the baseline's already-materialized `Chunk64StateGraph`, canonical triangular sum, and double-single helper. Verify its state-call AST equals the proven combined state method before attaching output behavior.
2. Extract the original MKL helper's mathematical table generator and class, then apply the two **builder-time** AST substitutions proven by the scaled helper: add `16 << 23` to exponent_bits and evaluate `mul(mul(reduced, 2**-16), exponent)`. Add the exponent-bias tensor in setup. Resolve SAFE_ABS to its frozen scalar. Put Decimal/struct/Torch imports in the setup-only mathematical-table generator or constructor. The emitted runtime `__call__` must contain only device operations and metadata checks.
3. Add this static MKL helper to the state graph's constructor and replace its exp method with the helper call. Extract `OutputChunk64Graph.__call__` unchanged from the frozen output source. Emit a static subclass of the embedded state graph with `state_call = Chunk64StateGraph.__call__`; this binding needs no runtime dynamic compilation. Its output arithmetic must retain the source AST hash, modulo the explicitly recorded class-name binding.
4. Change decoder setup to instantiate that full-output graph. Replace only the linear prefill native chunk32 plus custom-state block with one expanded-q/k full graph call. Keep the existing recurrent-state copy destination stable. The graph result supplies both BF16 output and FP32 new state; no oracle tensors or diagnostic observer are passed.
5. Apply the public-relative chunk64 plan, explicit prefill-tail routing, projection-policy propagation, and output-norm choice documented in `v6_chunk64_integration_proposal.md`. Keep output norm as an independently named candidate change so a state/output helper result is not confused with norm evidence. Preserve the full-attention planner and true decode path.
6. Emit a provenance JSON listing input hashes, exact changed definitions, unchanged-definition AST hashes, embedded helper AST hashes, and runtime forbidden-call audit. Compile the emitted source without importing it. Reconstructing/importing an integrated decoder and all device execution remain separate parent-owned validation steps.

At initial wrapper preparation, integration was held pending component results. The subsequent parent-authorized standalone build is recorded in `v6_chunk64_integrated_snapshot_build.md`; it preserves baseline public one-row recurrence at all starts and emits separate native-padded and explicit-norm candidates.
