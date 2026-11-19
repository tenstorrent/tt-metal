# Current-source native decode-output controls

These are diagnostic snapshots only. No production source or test files were changed, and this investigation did not import TTNN or execute hardware. The root can run them through `run_native_output_snapshot.py`, which reuses the real `run_decoder` benchmark and its runtime fallback audit, sets the candidate class explicitly, and writes snapshot hashes into both output and provenance. It refuses to overwrite output. The standard runner owns chip 0 and closes it in finally. Candidate results are labeled ineligible for final acceptance.

Two native APIs are distinct and are tested separately:

- `chunk_gdn_output_only.py.snapshot` uses `ttnn.transformer.chunk_gated_delta_rule` only for the one-token output core. Rank-4 normalized Q/K and rank-4 V permit T=1; the public native wrapper pads time to chunk size 32. It supplies all four existing setup constants, so the native wrapper does not rebuild/upload constants. Because this API needs log-space g, decode performs a full gate-table lookup in addition to the existing compact decay lookup. All adapter work is inside measured decode. Native final state is not requested.
- `kda_output_only.py.snapshot` revisits the actual historical `ttnn.experimental.kda.recurrent_chunk_scan` adapter. It places one real row before 31 zero rows, with an identity inverse and setup-owned start/end scalars. It preserves the current exact decay lookup for the real row; `final_decay = decay - 1` describes the same one-token transition, and native final state is discarded. This is the direct current-source discriminator for the previously unresolved historical KDA pattern.

Both snapshots retain the current exact ordered FP32 state update. `_recurrent_step(compute_output=False)` skips Q preparation and the exact output contraction; it does not compute an unused exact core before invoking the native alternative. Public state is round/widened once into a shared FP32 tensor for the native core and exact update. This materialized boundary adds cost compared with current fused state-input activation, and that necessary adaptation is included in timing. Internal non-public boundaries retain their existing policy. The native output passes through the unchanged BF16 core/unweighted-norm/gamma boundaries and FP32 SiLU gate. No HF state/core substitution is used.

## Source contract

`chunk_gated_delta_rule.cpp` derives `pad=(C-T%C)%C` for rank-4 input, expands grouped Q/K heads, applies query scale and pads Q/K/V; token-major gates retain T=1 and native readers zero later rows. The rank-3 flat path requires aligned T, which is why this diagnostic deliberately uses the existing rank-4 decode Q/K/V. The operation header documents all-constants-or-none behavior; the snapshot supplies `self.constants` unchanged. Dtypes, dimensions K=V=128 and head counts H=16/HV=32 match the current prefill native contract.

`experimental/kda/recurrent_chunk_scan/device/recurrent_chunk_scan_device_operation.cpp` accepts interleaved TILE FP32 protocol tensors with shapes `[BH,NC,32,V]`, `[BH,NC,32,K]`, `[BH,NC,32,32]`, `[BH,NC,K,32]`, `[BH,NC,K,1]`, and entry state `[BH,K,V]`; the snapshot uses BH=32, NC=1, K=V=128. The original adapter used precisely these shapes. The first snapshot incorrectly used setup-owned ROW_MAJOR INT32 start/end tensors. The installed native validation requires UINT32; its failure is preserved, and the corrected `kda_output_only_uint32.py.snapshot` uses UINT32. This was an adapter setup bug, not a model precision failure.

The KDA factory still uses FPU `UnpackToSrc` for many protocol/output terms while seeds/carry use full FP32 unpack. This is a reason to test output accuracy, not sufficient proof that its output PCC must fail. Retaining the exact model state removes recurrent accumulation of native-state error, and the completed short checks establish the output PCC and speed below; no long-stream output claim is made. Do not claim algebraic equivalence implies bitwise native output equivalence.

## Root commands

Run serially through the normal bounded wait wrapper, with watcher/profiler and allocation tracking disabled for timing:

```sh
python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/run_native_output_snapshot.py --snapshot models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/kda_output_only.py.snapshot --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/kda_output_only_benchmark.json
python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/run_native_output_snapshot.py --snapshot models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/chunk_gdn_output_only.py.snapshot --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/chunk_gdn_output_only_benchmark.json
```

Defaults match the selected S256/context512 control with 10 warmed prefills and 60 traced decodes. A correctly measured slower full decode can reject a candidate without an 8192-token qualification; a faster candidate still needs unchanged stateful, PCC, determinism and context gates. A failed binding or shape control must be investigated against the installed native fingerprint rather than attributed to model precision. These snapshots must never replace final source provenance or delay freezing the selected production path.

## Completed device controls: both rejected on speed

Root ran both complete adapters serially at S256/context512 with 10 warmed prefills and 60 traced warmed decodes. The frozen comparison is `final_0.json`, 154.096084 ms prefill and 1.455763 ms decode.

| Adapter | Prefill ms | Decode ms | Decode slowdown | Decode PCC | Artifact |
|---|---:|---:|---:|---:|---|
| KDA, corrected UINT32 scalars | 154.572919 | 1.584314 | 8.83% | 0.999979155 | [kda_output_only_uint32_benchmark.json](kda_output_only_uint32_benchmark.json) |
| Chunk GDN | 154.625607 | 1.642141 | 12.80% | 0.999980941 | [chunk_gdn_output_only_benchmark.json](chunk_gdn_output_only_benchmark.json) |

Both pass short PCC, determinism and runtime-fallback checks, but are rejected because their complete traced decode is slower. Prefill is unchanged structurally; its small measured differences are not optimization wins. These diagnostic controls do not establish 8192-token eligibility and are not final acceptance artifacts. The initial INT32 KDA validation failure remains in `kda_output_only_benchmark.json.snapshot.json` with its provenance; the corrected control uses a distinct snapshot/output. The original command above deliberately reproduces that preserved failure; use the `_uint32` snapshot and a fresh output path to repeat the successful KDA measurement.
