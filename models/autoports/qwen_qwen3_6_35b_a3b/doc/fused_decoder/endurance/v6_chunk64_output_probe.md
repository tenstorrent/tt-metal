# Full chunk64 output and state diagnostic

`tests/v6_chunk64_output_probe.py` adds the output boundary to the unchanged canonical triangular-sum and double-single-cumsum state diagnostic. Its SHA256 is `4c17aca9c01f3e19ed60b99a5f4880bf7a83053713778316c19cba781737b5d6`. This is an AutoFix component experiment, not decoder acceptance or performance evidence. Preparation used CPU only; the parent owns device execution.

The installed `transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py:244–322` is the reference. After normalization and FP32 conversion, query is scaled by the same Python expression `1 / (Dk ** 0.5)`. For each chunk the output is `(q * exp(cumulative_g)) @ previous_state + ((q @ k.T) * decay_mask) @ v_new`. The lower-triangular decay mask includes the diagonal. Output is cropped to the original sequence length, transposed to `[B,S,H,Dv]`, then cast to BF16.

The wrapper retains immutable device handles for the previous state and the state graph's cumulative g, decay mask, and v_new. It computes output after the new state has been allocated, using the previous state. It changes no state arithmetic. The inherited combined wrapper verifies restoration to the ordered state AST; its only replacement remains the cumulative-g assignment. Dependencies are snapshotted and hashed with each result, and are checked again after device execution.

CPU evidence:

| Initial state | Logical lengths | FP32 output differences | BF16 output differences | FP32 state differences |
|---|---|---|---|---|
| Zero | 128, 127, 65, 1 | 0 at every length | 0 at every length | 0 at every length |
| HF prefix64 state | 128, 127, 65, 1 | 0 at every length | 0 at every length | 0 at every length |

Commands, from the repository root:

```bash
HF_HOME=/home/ttuser/qwen36-35b-a3b/hf python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_chunk64_output_probe --oracle-only --lengths 128 127 65 1 --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_chunk64_output_cpu_zero.json
HF_HOME=/home/ttuser/qwen36-35b-a3b/hf python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_chunk64_output_probe --oracle-only --initial-state prefix64 --lengths 128 127 65 1 --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_chunk64_output_cpu_nonzero.json
```

Each result has an adjacent `.cpu_matrices.pt`, `.provenance.json`, and five `.dependencyN.<name>.snapshot` files. These retain actual normalized inputs, reference outputs and cache, every selected intermediate, and all local source dependencies. The result also records the installed HF source hash. Python compilation passed, and both CPU runs asserted TTNN was not imported.

Device API/layout review: runtime uses FP32 TILE tensors in interleaved DRAM, inherited helpers, explicit zero-padding to 64, and device-only output concat/slice/permute/typecast. All `Shape` slices convert to `tuple` first. ADD/SUB leave the unsupported FP32 accurate-rounding flag unset; inherited MUL retains its supported accurate flag. The new output graph introduces no new arithmetic flags. `slice/slice.cpp:387–388` rounds physical tile ends while preserving logical shape, permitting the unaligned final crop. `permute/device/permute_device_operation.cpp:37–45` selects the tile-row-invariant path because width remains the final axis; its validation checks rank. These source checks do not substitute for device execution. The graph supports every positive logical S, while the actual-prefix fixture is limited to its available 128 rows.

Suggested parent-owned device command:

```bash
HF_HOME=/home/ttuser/qwen36-35b-a3b/hf python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_chunk64_output_probe --lengths 128 --variant sfpu_fma --sum-order canonical --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_chunk64_output_sfpu_fma_device.json
```

The optional observer copies intermediates to CPU for diagnostics; omit it for device-only graph execution. This probe is not timed. Existing SFPU addcmul and exp accuracy limitations still apply; CPU transcription agreement does not establish device equality. No production decoder or existing helper source changed in this preparation.
