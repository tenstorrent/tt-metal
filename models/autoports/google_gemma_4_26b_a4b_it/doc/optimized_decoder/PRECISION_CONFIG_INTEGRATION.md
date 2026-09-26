# Optional precision/configuration integration

`precision_config_integration.patch` changes only the owned
`tt/optimized_decoder.py`. It was applied after parent authorization. Existing constructor defaults
and default execution paths are retained; the precision-policy report now lists
QKV and output weights separately. No C++, imported demo, or test code is changed.

Base runtime SHA256:
`c96cf65fff7ec69e8193a0df502610bce39f8bd5ea2ab2a0bec3dfbeeda4090a`.
Proposed runtime SHA256:
`e0691dc70a93b1e2d76d3028f022980f137cec4b0ac2ea944a25da425f19ea22`.
The companion JSON records hashes and validation.

| Constructor option | Default | Effect when selected |
| --- | --- | --- |
| `qkv_weight_dtype` | `None` | Cast each already-built lane weight independently, then cast the source prefill weight and rebuild its FP32 rows, exactly at the attention probe's boundary. Also handles the direct and compensated projection sources. |
| `output_weight_dtype` | `None` | Cast the shared attention output-projection weight, as in the attention probe. |
| `expert_activation_dtype` | `None` | Cast only decode gate/up inputs; prefill, intermediate/down activations, routing and outputs retain their policy. |
| `native_sdpa_fidelity` | `HiFi4` | Native decode fidelity with FP32 destination/full synchronization enabled and approximation/packer accumulation disabled. |
| `native_sdpa_grid` | `(8, 8)` | Native decode grid; existing chunk sizes, exp mode and core limit remain unchanged. |
| `prefill_attention_fidelity` | `None` | Prebuild a compute config for direct and paged/chunked prefill attention only. |
| `generalized_router` | `False` | Decode top-8/softmax composite with the tested BF16 logits, fixed padding, face packing and per-expert scales; prefill delegates to the original router. Mutually exclusive with lane-router projection. |

All are available through the existing `run_optimized_decoder --defaults
--default-overrides` JSON path. The harness converts `*_dtype` names and
`*_fidelity` names to TTNN enum values. For example, an optional policy can contain
`{"qkv_weight_dtype":"bfloat4_b","output_weight_dtype":"bfloat8_b",
"expert_activation_dtype":"bfloat8_b","native_sdpa":true,
"native_sdpa_fidelity":"HiFi2","native_sdpa_grid":[8,4],
"prefill_attention_fidelity":"HiFi2","generalized_router":true}`.
This illustrates parameter encoding; select values from the parent-controlled
experiments before measuring the integrated combination.

The owned `ConfiguredChunkedPrefillAttention` preserves the imported helper's
scalar-offset call sequence: the same environment-selected chunk geometry,
request page-table slice, Q slicing/padding, cache geometry override, native call,
unpadding and deallocation. Full-attention optimized prefill supplies scalar
offsets. Prefix continuation already uses `decode_forward`. The original helper
is still used when `prefill_attention_fidelity=None`. Config construction occurs
during setup, and no runtime monkeypatch is needed.

CPU/source checks passed: existing constructor defaults unchanged; eight exact
mocked chunk-helper call-sequence comparisons including padded tails, multiple
chunks, request slots, geometry overrides and environment overrides; isolated
native/prefill config checks; promoted generalized-router equivalence to the
tested probe, including prefill delegation; Black (Python 3.10, 120 columns),
`py_compile`, and `git apply --check`. These checks do not establish integrated
device parity or latency. The CPU scripts and proposed source are retained under
`/tmp/gemma4-owned-runtime-integration/` for this session.

After all processes measuring or hashing the current runtime have exited, run
from the repository root:

```bash
git apply --check models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/precision_config_integration.patch
git apply models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/precision_config_integration.patch
python_env/bin/python -m py_compile models/autoports/google_gemma_4_26b_a4b_it/tt/optimized_decoder.py
```

Apply only after the check succeeds and the base hash matches. `git apply`
updates this one file as a unit. Verify the resulting hash against the proposed
hash, then run the existing actual-input audited headline/stress commands with
explicit overrides. Rebase the proposal if intervening edits changed the base.
