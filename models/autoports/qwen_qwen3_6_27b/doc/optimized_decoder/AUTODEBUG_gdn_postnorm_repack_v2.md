# Postnorm repack probe: immutable API repair v2

The first `affine124_probe` stopped at the Python guard before launching any kernel: TTNN `Shape` supports integer indexing/iteration but not Python slice indexing. The original helper used `tuple(packed.shape[:2])`. This is not evidence against the material layout/affine/gate fusion.

`candidates/gdn_affine_gate_flat_v2.py` changes only that expression to `tuple(packed.shape)[:2]`. `candidates/gdn_affine_gate_decode_v2.py` embeds the repaired helper over the same delivery118 source; the policy and numerical/kernel code are unchanged. The original candidate and failure evidence remain intact. The new generator is `candidates/make_gdn_affine_gate_decode_v2.py`; its generation JSON records hashes.

The helper's remaining shape accesses are scalar indices, length checks, or full tuple conversions. The isolated runner contains no native Shape slicing. Descriptor construction, RuntimeArgs, TensorAccessorArgs and generic_op follow the established stage-local generic-op helpers. Kernel compilation and execution remain unverified by this source-only repair.

`python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/test_gdn_affine_gate_mapping_v2.py` passes all 64 mapping cases, five worker partitions, unchanged prefill/default AST, deterministic source reproduction/hashes, and a new AST guard rejecting direct native Shape slices. `py_compile` passes for both v2 helper and integrated candidate. Results: `gdn_affine_gate_v2_cpu_checks.json`.

Rerun the existing `tests/run_gdn_affine_gate_probe.py` with `--candidate models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/gdn_affine_gate_flat_v2.py`; the integrated candidate is selected with `gdn_fused_affine_gate=True`. The isolated gate remains exact BF16 equality, with repeated-output and stable-address checks. No device was opened by this investigation.
