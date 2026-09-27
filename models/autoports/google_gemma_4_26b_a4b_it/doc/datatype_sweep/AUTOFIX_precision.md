# AutoFix: precision construction

## Verified hypothesis

The original generator/model API had no precision argument, and model construction hardcoded the decoder, cache and terminal policy. `AUTODEBUG_precision.md` identifies those source boundaries. A selected JSON could not affect the measured runtime.

## Change

`tt/precision_policy.py` provides complete per-kind defaults, strict JSON/dictionary loading, layer exceptions and fixed-assumption validation. `precision_config` reaches the model through the generator factory's existing keyword forwarding. Model construction consumes head/cache settings and passes each resolved layer policy into the decoder. Decode attention, expert and shared weight/fidelity groups, activation casts and CCL payloads are configurable; prefill and sensitive norm/router assumptions remain explicit and fail closed. The selected artifact is loaded by default when present; `{}` explicitly requests baseline defaults.

The runtime summary reads weights and compute configs and compares them with the resolved request. Cache allocation records actual K/V dtype, and external caches must agree. Embedding, decoder output and logits dtypes are checked during execution. Candidate recovery constructs from retained original BF16 checkpoint uploads or raw checkpoint tensors; it never widens an already selected low-precision tensor to claim recovered information.

## Focused experiments

- Pure-host policy checks passed: complete baseline round-trip; partial policy merge and layer precedence; unsupported dtype/fidelity, fixed assumptions and unknown-key rejection; path/default artifact loading; explicit baseline override; deliberate actual/requested mismatch rejection.
- `python_env/bin/python -m pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/test_precision_policy.py`: **8 passed**. This test file is owned by the main agent.
- `python3 -m py_compile models/autoports/google_gemma_4_26b_a4b_it/tt/{precision_policy,multichip_decoder,model,generator}.py`: passed (run with individually expanded paths).
- `pre-commit run --files models/autoports/google_gemma_4_26b_a4b_it/tt/precision_policy.py models/autoports/google_gemma_4_26b_a4b_it/tt/multichip_decoder.py models/autoports/google_gemma_4_26b_a4b_it/tt/model.py models/autoports/google_gemma_4_26b_a4b_it/tt/generator.py`: passed after formatting.
- `git diff --check`: passed.
- Main-agent hardware smoke on layers 0/5 and 33-token input passed construction verification and execution; see `smoke_baseline.json`. Full-stack candidate validation is owned by the main agent and remains separate from these host checks.

## Allocation correction found during source audit

The initial plumbing redundantly called `ttnn.typecast` when QKV/output already had the requested BFP8 dtype. The current C++ typecast path allocates an output even for equal dtypes (`ttnn/cpp/ttnn/operations/copy/typecast/device/typecast_device_op.cpp`), so that introduced separate prefill/decode weight copies. Explicit dtype-equality guards now preserve the original aliases. The main agent was notified to preserve the first full baseline run as diagnostic and repeat it under the final source; numerical settings are unchanged, but allocation-sensitive timing must use the final version.

## Status

Construction plumbing and host verification complete. No device was opened by this subagent. Full-model accuracy, warmed timing, final selected-policy review and any maximum-context changes remain the main stage's responsibility. Python-only changes require no C++ build.
