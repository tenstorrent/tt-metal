# Corrected-FMA standalone integration

Two standalone diagnostic snapshots embed the measured corrected-FMA implementation without changing the frozen structural candidates:

| Norm variant | Snapshot | SHA256 |
|---|---|---|
| Native padded | `v6_inline_router_silu_chunk64_corrected_fma_native_norm.py.snapshot` | `8b173253e4b6017042335c189723368f4e536fb96ba42bae9aa2558e3783fc76` |
| Explicit RMS + SiLU | `v6_inline_router_silu_chunk64_corrected_fma_explicit_norm.py.snapshot` | `8391717476ae224ee9b82c203d7dec21ceab69632c5edd0abd89bf8cc4daeb03` |

`tests/v6_build_corrected_fma_snapshot.py` proves that only `Chunk64StateGraph.__init__` and `Chunk64StateGraph.mm` change relative to the corresponding frozen v2 snapshot. It embeds `CorrectedFma`, `CorrectedMatmul`, and their fully generated compute-source string. Adapter runtime AST is unchanged except that setup reads the literal string instead of invoking the test source generator. Matrix-wrapper AST is unchanged. There are no test imports, runtime AST/exec/compile/eval, or Torch/host-conversion calls in the corrected adapter. Every chunk64 matmul dispatches to increasing-K corrected FMA; there is no native addcmul GEMM fallback. Existing norm choices, public S=1 policy, and decode code remain unchanged.

The compute string SHA256 is `be3b5c5c44aae30dc927e2977949508675638276bbd8e1c88a65f19f4675a7cd`, matching the measured component. Each snapshot has adjacent `.compute.cpp.snapshot`, `.builder.py.snapshot`, and provenance JSON recording the base, helper, component, generated-source, and AST identities. Source compilation and builder formatting passed; no Torch or TTNN was imported by the builder. This agent ran no hardware.

Parent-owned component evidence:

- `v6_chunk64_corrected_fma_output_device.json`: length128, zero initial state; complete FP32/BF16 output and final FP32 state codes match HF exactly; device closed cleanly. This result gates both generated snapshots.
- `v6_chunk64_corrected_fma_output_nonzero_device.json`, SHA256 `aae7704314a74968ec5007fa617de24de9f82af23538bb27d207f477b640604f`: lengths127/65/1 with prefix64 nonzero initial state; complete FP32/BF16 output and final FP32 state codes match HF exactly; device closed cleanly. This additional result was inspected after snapshot emission, so it is recorded here without rewriting frozen provenance.

Reproduce with fresh output/report paths:

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_build_corrected_fma_snapshot --norm native --output NEW_NATIVE_SNAPSHOT --report NEW_NATIVE_PROVENANCE
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_build_corrected_fma_snapshot --norm explicit --output NEW_EXPLICIT_SNAPSHOT --report NEW_EXPLICIT_PROVENANCE
```

These are integrated candidates, not accepted decoders. Full decoder stress, continuation, determinism, watcher, profiling, and speed gates remain parent-owned work. The corrected arithmetic has a bounded measured domain, not universal subnormal correctness. Its per-K broadcast, placement, and kernel dispatch are intentionally still diagnostic and carry no performance claim.
