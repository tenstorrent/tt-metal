# AutoDebug: vLLM prefill return contract

Source inspection on 2026-09-14, before this investigator changed implementation
or ran the reproducer. No model, device, or server execution was performed.

## Evidence and diagnosis

The reduced S31/G3 request first failed in `TTModelRunner.submit_prefill` at
`model_runner.py:2687`: `ValueError: not enough values to unpack (expected 2,
got 1)`. See `reduced_rope_contract_failure.log:155`. The parent then changed the
adapter to return `(result, [0] * N)`. The retry failed at line 2690 with
`AttributeError: 'int' object has no attribute 'item'`; see the preserved
`reduced_rope_type_failure.log:155` and `readiness_vllm/server.log:155`.

The actual caller lives in
`/home/mvasiljevic/qwen38-full-rerun/vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py`.
It enables `request_specific_rope` from `model_config.uses_mrope` (line 143),
unpacks `(output, rope_deltas)`, and stores `rope_deltas[i].item()` in request
order (2686–2691). A list of Python integers meets tuple arity but fails the
element protocol. Text-only zero offsets are appropriate; return a CPU int64
tensor with one zero per packed prompt row.

## Complete relevant boundary

- The runner sends full-prefix token rows. `prompt_lens` holds absolute chunk
  ends; `start_pos` holds previously computed token counts (1100–1123).
  The adapter must slice `[start:end]` and pass chunk length `end-start` plus the
  absolute start to the canonical generator. Its current implementation does so.
- `submit_prefill` passes the allocated cache unchanged and forwards
  `prefill_empty_slots` even with DP=1 (2640–2684). Slots map packed input rows
  to fixed recurrent/cache rows; output rows remain packed in input order.
  The adapter preserves cache identity, merges page rows into the named slots,
  and resets recurrent state only for starts equal to zero.
- Device sampling parameters arrive as tensors, are converted by the runner to
  `TTSamplingParams` lists, and are supplied only when device sampling is active
  (2656–2671). The adapter returns integer tokens `[N, 1]`; explicit host
  compatibility returns floating logits `[N, 1, vocab]`.
- `_forward_with_model_input` consumes the outer RoPE tuple through
  `submit_prefill` before handling any optional sampling/logprob tuple
  (2717–2730). `_get_output_tokens` gathers packed prompt rows and reshapes
  device tokens to `[N, 1]`, or indexes host logits at `[rows, -1, :]`
  (2957–2976, 3094–3136). TP4 logprobs select host sampling (2613–2624);
  no extra logprob tuple is required for the current device sampling path.

## Focused verification plan

Invoke actual `TTModelRunner.submit_prefill`, `_forward_with_model_input`, and
`_get_output_tokens` against the actual adapter with a fake generator replacing
device effects. Require the existing integer-list return to reproduce the
reported AttributeError. Then change only the return expression and require
one- and two-request output shapes/dtypes, tensor RoPE deltas, request ordering,
noncontiguous slot/page mapping, chunk slicing, fresh-only resets, and exact
cache identity. Verify host-logit shape separately with explicit compatibility.

The original hardware/server S31/G3 request must be rerun by the parent after
device recovery. Host contract checks do not prove device sampling, numerical
correctness, or serving completion.
