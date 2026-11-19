# Standalone chunk64 output/state integration candidates

After the parent-owned full-output scaled-exp component completed, `tests/v6_build_chunk64_output_snapshot.py` emitted two independent standalone diagnostic decoder snapshots. No production decoder or preexisting helper source changed, and this agent ran no hardware.

| Candidate | Snapshot | SHA256 |
|---|---|---|
| Structural integration; existing norm with internal padding | `v6_inline_router_silu_chunk64_output_mkl_native_norm_v2.py.snapshot` | `1446f3bbb5d00aa69976be0021b534d895b48fc0824d5120d2db67738ec7f2fa` |
| Same integration; explicit RMS boundaries and universal SiLU lookup | `v6_inline_router_silu_chunk64_output_mkl_explicit_norm_v2.py.snapshot` | `d57f87095b9c3d8705386b1c7c2f54d9541f3389bb33d6bf9d1821e7f6fd507f` |

Use the v2 snapshots. The first emitted versions remain preserved; v2 changes their planner docstring to describe the new linear planner accurately. Every snapshot has adjacent provenance with exact source hashes and modified-method AST hashes. The builder's default cached-singleton policy is `recurrent`: public S=1 prefill preserves the baseline recurrence at **all** starts, while internal singleton tails of a larger public call use chunk64. This was the parent's explicit isolation choice. Fresh HF first-call S=1 versus recurrent behavior remains a separate validation case, not silently redefined by the candidate.

Both candidates statically embed the full output graph, canonical triangular sum, double-single cumsum, scaled MKL exp class, and mathematical table generator. The scaled exp runtime AST hash matches the proven helper after resolving its private constant name; the state-call module AST and canonical/cumsum helper ASTs match the frozen component sources. `ROW_SNAPSHOTS=(1,2,31,63)` is explicitly embedded because the output graph uses an internal observer, activating a reference that the earlier state-only integrated runtime did not evaluate. No diagnostic-test imports, runtime exec/compile/eval, oracle data, or host conversions were added to graph runtime.

Only these existing method definitions change: decoder setup, linear attention, prefill planning, prefill dispatch, chunk graph setup, and chunk exp dispatch. Decode_forward, full attention, MoE, general linear projections, and decoder input/post-attention normalization retain their baseline ASTs. Both projection packs receive the same public projection_k, preventing an internal singleton's output projection from silently reverting to K512. Singleton MoE/expert shape branches remain baseline for this experiment; any remaining mismatch there requires separate evidence.

Native norm pads only core and packed gate to a tile-aligned logical length, then crops before multiplying logical Z. Explicit norm flattens token/head rows, preserves decode's established BF16 boundaries, gathers FP32 SiLU for all Z codes, and produces logical `[1,S,4096]` without decode concat-head sharding. Their remaining state arithmetic is identical, including the known SFPU FMA limitation.

The builder compiles emitted source without importing it. `v6_chunk64_integrated_planner_cpu.json` executes the actual emitted planner definitions in isolation: each passes all 36,864 linear length/start cases and 110 full-attention plan comparisons against the original snapshot. No Torch/TTNN import or model reconstruction is involved. These are source/planner checks, not integrated accuracy or performance results.

Reproduction (choose the corresponding norm and fresh output paths):

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_build_chunk64_output_snapshot --norm native_padded --cached-singleton recurrent --component-result models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_chunk64_mkl_exp_scaled_output_device.json --emit NEW_NATIVE_SNAPSHOT --report NEW_NATIVE_PROVENANCE
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_build_chunk64_output_snapshot --norm explicit_silu --cached-singleton recurrent --component-result models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_chunk64_mkl_exp_scaled_output_device.json --emit NEW_EXPLICIT_SNAPSHOT --report NEW_EXPLICIT_PROVENANCE
```

The recorded component result is cleanly closed but still has 12 BF16 output-code differences and 16 BF16 cache-code differences at length128. Thus neither standalone candidate is acceptance eligible without full decoder testing, including the long failing stream, continuation cases, unaligned tails, and the unchanged stage gates.
