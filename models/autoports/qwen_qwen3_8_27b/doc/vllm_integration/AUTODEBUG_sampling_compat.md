# AutoDebug: explicit sampling compatibility

## Starting evidence

Source-only investigation, 2026-09-14. No device or server execution.

- `vllm/plugins/vllm-tt-plugin/tests/tt/test_request_isolation.py` requests
  stochastic `top_k=100` and non-neutral presence, frequency, and repetition
  penalties. The selected Qwen adapter uses canonical `TTSampling` with runtime
  `k<=32` and does not implement penalty transforms.
- `TTModelRunner.check_perform_device_sampling` checks host logits processors,
  structured outputs, and logprobs, but never checks model-specific top-k or
  penalty support. It therefore selects device sampling for these requests.
- `InputBatch.add_request` normalizes unrestricted top-k to vocabulary size.
  Greedy requests must ignore that normalized top-k; positive-temperature
  requests with unrestricted top-k cannot be silently reduced to 32.
- Existing host sampling already handles full logits and these features.
  `min_p` and TP4 logprobs already select that route. The adapter requires
  `QWEN_VLLM_HOST_COMPATIBILITY=1` before returning host logits.
- Only gathered-DP rank zero loads a model instance. An instance-only hook
  would miss incompatible requests on other ranks.

## Hypothesis experiment

Hypothesis: an optional model class capability callback can preserve existing
device routing for supported requests and select the existing host route for
unsupported requests when the model explicitly permits compatibility.

Prediction: a fake class callback rejecting a live `top_k=100` request changes
the runner decision from device to host. Inactive rows must not affect the
decision, including gaps in lane batches. Existing host-only decisions and
models without a callback must preserve their behavior. A callback error must
propagate rather than enabling an implicit fallback.

Implemented contract:

```python
@classmethod
def supports_device_sampling(cls, params, *, is_decode: bool) -> bool:
    ...
```

`params` is a `TTSamplingParams` containing CPU tensors for live request rows,
without persistent-buffer padding. The runner caches the class callback at
initialization on every DP rank. It calls the hook only after existing shared
host-only gates have accepted device sampling. False selects existing host
sampling; True preserves device sampling; exceptions propagate.

The Qwen adapter must return True for its supported subset, return False for
unsupported requests only when its explicit compatibility environment is set,
and otherwise raise a descriptive error. Existing host-only gates are enforced
by the adapter's host-logits guard. Canonical-only batches retain device sampling
even with compatibility enabled. The runner has one mode per execution batch:
an unsupported request also sends co-batched requests through host sampling.

## Results and retained fix

Verdict: **verified and fixed at the routing boundary**. The production change
is 16 added lines in
`vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py`: cache the
optional class callback and consult it after existing shared device gates.
No sampling algorithm or existing parameter value is changed.

Commands below ran from `/home/mvasiljevic/qwen38-full-rerun/vllm`, using the
existing `tt-metal/python_env/bin/python`; imports load TTNN, but these tests
do not open a device or run model computation.

```bash
python -m pytest plugins/vllm-tt-plugin/tests/test_sampling_capability.py -q --tb=short
```

- Before the production edit: **16 failed, 12 passed**. The failures show
  unsupported requests still returning True, swallowed callback rejection,
  and absent class-hook initialization.
- After the production edit: **28 passed**. This covers both prefill and
  decode, unbounded and top-100 requests, each penalty, mixed batches, ordinary
  stale tails, sparse lane rows, explicit errors, existing shared host-only
  gates, legacy models, and class initialization on both gathered-DP ranks.

```bash
python -m pytest plugins/vllm-tt-plugin/tests/test_sampling_capability.py plugins/vllm-tt-plugin/tests/test_lane_input_batch.py plugins/vllm-tt-plugin/tests/test_lane_model_runner.py plugins/vllm-tt-plugin/tests/test_state_slots.py -q --tb=short
```

**69 passed.** Only existing import/deprecation warnings were emitted.

A further manual host probe imported the parent's actual
`Qwen38ForCausalLM.supports_device_sampling` into this test's real `InputBatch`
and runner helper. **28 combinations passed:** compatibility 0/1, prefill/decode,
and seven request policies (greedy unrestricted, stochastic k=32, stochastic
k=100, stochastic unrestricted, and each of the three non-neutral penalties).
Supported cases returned device=True in both modes; unsupported cases raised
with compatibility disabled and returned device=False with compatibility
enabled. This verifies the cross-repository callback contract without model
initialization.

```bash
python -m black --target-version py312 --check plugins/vllm-tt-plugin/tests/test_sampling_capability.py
git diff --check
```

Both passed. Ruff is not installed in the existing environment or cached
pre-commit environment; no dependency installation was attempted. Python-only
changes need no C++ build.

## Remaining limits

- The original server sampling tests and real host-logits extraction were not
  run by this investigation. They remain the coordinating hardware owner's
  validation responsibility; host routing tests cannot establish device
  correctness, numerical output quality, or performance.
- Compatibility is selected per execution batch, including gathered DP, not
  independently per co-batched request. Switching host/device modes can change
  RNG streams; the runner already documents that behavior. These changes add
  no stronger cross-mode determinism claim.
- The callback operates after generic host-only gates. The adapter must retain
  its explicit host-logits guard for features such as min_p and TP4 logprobs,
  which bypass the new callback through those existing gates.
- The class hook must be pure host policy and safe before loading model
  weights or opening hardware. It must use identical policy on all DP ranks.
- A server imported before this change must restart to use the new hook.
