# AutoFix: vLLM prefill return contract

## Starting evidence

`AUTODEBUG_prefill_contract.md` traces the actual runner and adapter boundary.
The original reduced S31/G3 server request failed unpacking the output at
`TTModelRunner.submit_prefill:2687` (`reduced_rope_contract_failure.log`).
After the parent added the tuple, the retry failed at line 2690 because a
Python integer has no `.item()` (`reduced_rope_type_failure.log`).

Reduced server command from the parent work log:

```bash
QWEN_VLLM_TEST_LAYERS=0,3 bash models/autoports/qwen_qwen3_8_27b/tests/run_vllm_stage.sh --stages serve --sampling-profile smoke
```

## Hypothesis and experiment

**Hypothesis:** The outer tuple is correct, but its RoPE delta elements must be
scalar tensor values accepted by the actual caller's `.item()` expression.

Added `tests/test_vllm_prefill_host.py`, which invokes the real adapter and real
`TTModelRunner.submit_prefill`, `_forward_with_model_input`, and
`_get_output_tokens`. A fake generator records the adapter's calls and returns
host tensors instead of executing model/device effects; readback and TTNN host
conversion are patched. No mesh, model, weight, or device tensor is constructed.

Before the repair, the five-test run exited 1 with five errors (including both
one- and two-prompt subtests). Four errors reproduce the exact runner
`AttributeError: 'int' object has no attribute 'item'`; the direct output test
also rejects the list as lacking tensor shape. Cache rejection already passed.
Evidence: `prefill_host_before.log`. **Hypothesis verified.**

## Smallest repair and verification

Changed only `Qwen38ForCausalLM.prefill_forward`'s final return expression:

```python
return result, torch.zeros(len(ends), dtype=torch.int64)
```

The same suite then passed all five tests in 0.004 seconds
(`prefill_host_after.log`). It verifies:

- Sampled tokens are CPU int64 `[N, 1]`, packed in input order, with no padded
  token rows leaking through. The real runner accepts N=1 and N=2, extracts
  device tokens, and updates only the participating requests' RoPE deltas.
- Explicit host compatibility returns floating logits `[N, 1, vocab]` and CPU
  int64 zero RoPE deltas `[N]`, and the runner consumes that tuple successfully.
- Absolute chunk ends become the correct token slices and relative lengths.
  Packed rows map to noncontiguous slots `[3, 1]`; page rows outside those slots
  remain intact. Only the fresh request resets its recurrent slot. Sampling
  parameters retain packed row order, and prefill invalidates the decode binding.
- Both adapter/cache and generator/cache identity mismatches reject before
  reset, sampling, or prefill effects.

Exact host verification command, run before and after the repair:

```bash
source ../run-env.sh
PYTHONPATH="$VLLM_ROOT:$PYTHONPATH" python_env/bin/python -m models.autoports.qwen_qwen3_8_27b.tests.test_vllm_prefill_host
python_env/bin/python -m black --check models/autoports/qwen_qwen3_8_27b/tests/test_vllm_prefill_host.py
```

Black exited 0 with no changes; it warned about its inferred Python 3.14 target
under Python 3.12. No C++ or CMake changed, so no build is required.

## Final status

The observed Python contract defect is repaired and verified at the actual
host caller boundary. **Original server S31/G3 runtime verification is pending**
with the parent after device recovery. These tests do not verify numerical
quality, device sampling correctness, device queue ordering, or real KV writes.
