# Traced Executor `can_trace` Plan

## Purpose

This plan focuses on one issue: `can_trace` policy in `TracedLLMExecutor`.

`TracedLLMExecutor` should continue to own the shared code that LLMs have in common:

- prefill batching over users;
- padded prefill length selection;
- page table selection for traced shapes;
- trace capture/replay lifecycle;
- decode trace capture/replay lifecycle;
- trace cache ownership;
- cleanup;
- warmup and benchmark integration.

The goal is not to turn `TracedLLMExecutor` into a generic trace runner. The goal is to make trace eligibility strict and centralized inside `TracedLLMExecutor`.

The existing model post-processing hook should remain:

```python
logits = self.model.post_process_prefill_output(logits, last_token_idx)
```

That pattern is acceptable: the shared LLM executor owns common flow, while the model owns model-specific output conversion.

## Current Problem

The traced prefill loop currently computes `can_trace`, then silently falls back to eager when tracing is unsupported:

```python
can_trace = self.model_args and self.model_args.can_enable_trace(prefill_seq_len, num_cached_tokens)

if can_trace:
    page_table_user = _get_prefill_trace_user_page_table(...)
else:
    page_table_user = _get_prefill_user_page_table(...)

if can_trace:
    logits = self._easy_trace_prefill(...)
    logits = self.model.post_process_prefill_output(logits, last_token_idx)
else:
    logits = self._eager._prefill_single_user(...)
```

That is the architectural issue.

`TracedLLMExecutor` is a traced executor. If it cannot trace a request, it should reject the request with a clear reason. It should not silently run eager code.

Silent eager fallback causes several problems:

- performance benchmarks can report mixed traced/eager behavior;
- trace regressions can be hidden;
- code paths become harder to reason about;
- validation and trace capture scope become tangled;
- `Traced*` APIs no longer mean “traced only.”

## Review Panel

The useful review group for this narrower question is:

- `TracedLLMExecutor owner`: owns shared LLM trace policy and lifecycle.
- `ModelConfig owner`: owns static trace capability fields such as supported sequence lengths.
- `Llama model owner`: validates that shared policy still supports Llama-specific constraints.
- `TTNN trace/runtime expert`: validates that trace-unsafe cases are rejected before capture.
- `Perf/demo owner`: ensures unsupported trace paths fail before timing and do not fall back to eager.
- `Serving/vLLM integration owner`: confirms prefix caching and chunked prefill behavior is explicit.

## Expected Panel Feedback

### TracedLLMExecutor Owner

`TracedLLMExecutor` should own the decision “can this executor trace this request?” because that is part of the traced executor contract.

The decision can consult `model_args`, but callers should not call `model_args.can_enable_trace(...)` directly inside the prefill loop. The traced executor can expose one internal assertion helper:

```python
self._assert_prefill_trace_supported(prefill_seq_len, num_cached_tokens)
```

The error message should include enough detail to debug the unsupported case.

The traced executor should not contain eager fallback branches in traced prefill. If trace is unsupported, raise.

### ModelConfig Owner

`model_args.can_enable_trace(...)` can remain as a static capability query. It should answer whether the model configuration supports a trace shape.

The executor policy should combine:

- model config capability;
- requested padded prefill length;
- number of cached tokens;
- max prefill chunk size;
- max sequence length;
- currently unsupported executor features such as prefix caching or chunked prefill.

If the model config lacks required trace capability information, constructing `TracedLLMExecutor` should fail early.

### Llama Model Owner

Keep `post_process_prefill_output` as a model hook. The current pattern is still a good shared-executor design:

- traced prefill returns hidden states;
- the model hook converts hidden states to logits for the selected token block.

That is a small model-specific hook, not an argument for moving the whole trace flow into `TracedLlamaExecutor`.

### TTNN Trace Runtime Expert

Trace support should be checked before any trace capture starts. Unsupported cases should never reach `begin_trace_capture`.

The trace support result should include a reason because trace failures are expensive to debug. Example reasons:

- no `model_args`;
- no trace-supported sequence lengths;
- padded sequence length unsupported;
- prefix caching unsupported;
- prefill length exceeds max chunk size;
- prefill length exceeds max sequence length.

### Perf/Demo Owner

Performance tests should assert that they use a traced path. If the traced executor cannot trace, fail before timing starts.

Do not allow traced perf tests to silently fall back to eager. That makes TTFT/decode measurements ambiguous.

### Serving/vLLM Integration Owner

Prefix caching and chunked prefill should be explicit trace policy decisions. Today, prefix caching disables trace through `num_cached_tokens == 0`. That should remain visible in the traced executor’s rejection reason.

Future support for prefix-cached tracing can be added by changing the traced executor policy in one place.

## Proposed Design

### 1. Centralize Prefill Trace Policy in `TracedLLMExecutor`

Add a small helper on `TracedLLMExecutor`, or keep the check inline if that reads better after the patch:

```python
def _assert_prefill_trace_supported(self, prefill_seq_len: int, num_cached_tokens: int) -> None:
    if not self.model_args:
        raise RuntimeError("Traced prefill requires model_args")
    if not self.model_args.can_enable_trace(prefill_seq_len, num_cached_tokens):
        raise RuntimeError(
            "Traced prefill is unsupported for "
            f"prefill_seq_len={prefill_seq_len}, num_cached_tokens={num_cached_tokens}"
        )
```

Do not add a result object or custom exception unless a caller actually needs to catch unsupported trace separately. A clear `RuntimeError` is enough for the first patch.

`model_args.can_enable_trace(...)` can keep the existing detailed policy. The architectural boundary is that `TracedLLMExecutor` owns when that check is applied and whether fallback is allowed.

### 2. Replace Eager Fallback With Explicit Rejection

In traced prefill, replace this:

```python
if can_trace:
    ...
else:
    logits = self._eager._prefill_single_user(...)
```

with this:

```python
self._assert_prefill_trace_supported(prefill_seq_len, num_cached_tokens)
```

Then the remaining code can assume trace is supported:

```python
page_table_user = _get_prefill_trace_user_page_table(...)
logits = self._easy_trace_prefill(...)
logits = self.model.post_process_prefill_output(logits, last_token_idx)
```

This keeps `post_process_prefill_output` exactly where it is.

### 3. Keep Eager Behavior Out of `TracedLLMExecutor`

If a caller wants eager behavior, it should instantiate/use `EagerLLMExecutor` or a model-specific eager wrapper.

Do not add flags to traced executors that change them into eager executors.

This aligns with removing `enable_trace` from `Traced*` APIs.

## Validation Scope

This plan does not primarily address module validation, but it interacts with the validation problem.

Once traced prefill no longer has eager fallback inside the traced path, validation can be scoped more clearly:

- validate eager warmup directly;
- avoid wrapping broad compile calls that capture traces;
- keep validation wrappers out of `begin_trace_capture` / `end_trace_capture`;
- eventually remove `suspend_module_input_validation()` from trace capture once validation scope is narrowed.

That is a follow-up. The first change should focus on `can_trace`.

## Migration Plan

### Phase 1: Add the Trace Support Assertion

- Add `_assert_prefill_trace_supported(...)` to `TracedLLMExecutor`, or keep an equivalent inline assertion if that is clearer.
- Use `RuntimeError` with a message that includes `prefill_seq_len` and `num_cached_tokens`.
- Do not add a dataclass, custom exception, or construction-time validation in the first patch.

Success criteria:

- `model_args.can_enable_trace(...)` is called only from the traced prefill support check.
- Unsupported trace cases fail with a clear message.

### Phase 2: Remove Eager Fallback From Traced Prefill

- Replace local `can_trace` branches with a required support check.
- Always use `_get_prefill_trace_user_page_table(...)` in traced prefill.
- Always run `_easy_trace_prefill(...)` for traced prefill.
- Keep `self.model.post_process_prefill_output(...)` after traced prefill.

Success criteria:

- `TracedLLMExecutor.prefill_forward(...)` never calls `_eager._prefill_single_user(...)` as fallback.
- Unsupported requests raise before trace capture.

### Phase 3: Update Tests

Add unit tests for:

- supported prefill trace request passes;
- unsupported sequence length raises;
- prefix caching raises;
- traced prefill no longer calls eager fallback.

Keep existing tests for:

- trace replay copies only mutable inputs;
- validation suspension behavior until validation scope is narrowed.

### Phase 4: Update Call Sites

- Ensure perf/demo paths instantiate traced executors only when they expect traced execution.
- If a path needs eager fallback, route it to `EagerLLMExecutor` before calling forward.
- Remove any remaining caller assumptions that `Traced*` can silently run eager.

## Non-Goals

This plan intentionally does not:

- introduce a full trace-plan abstraction;
- move `post_process_prefill_output` out of the model hook;
- make `TracedLLMExecutor` a generic trace runner detached from LLM semantics;
- redesign page table helpers;
- solve prefix-cached tracing;
- solve module validation scoping in the same patch.

## Recommended First PR

The first PR should be narrow:

- add `_assert_prefill_trace_supported(...)`, or an equivalent inline assertion;
- reject unsupported trace requests instead of falling back to eager;
- keep `post_process_prefill_output` unchanged;
- add focused tests around the new `can_trace` behavior.

This addresses the pointed architectural issue without over-abstracting the shared LLM executor.
