# Stage 09 sampling compatibility contract

Date: 2026-09-27. Status: **source diagnosis and host contract tests verified; optional compatibility implementation and device validation remain outstanding**.

Follow-through: the optional implementation and expanded 38-test host verification are recorded in [AUTOFIX_sampling_contract.md](AUTOFIX_sampling_contract.md). This document preserves the original pre-implementation diagnosis and source locations.

Starting evidence: finding 5 in [AUTODEBUG_contracts.md](AUTODEBUG_contracts.md), and the remaining sampling limits in [AUTOFIX_plugin_contracts.md](AUTOFIX_plugin_contracts.md). This is the focused AutoFix verification of that existing diagnosis. No implementation file was changed, accelerator opened, server launched, or performance measured. The working runtime is installed vLLM **0.26.0** at `/home/container_app_user/tt-metal/python_env/lib/python3.10/site-packages/vllm`, with the source plugin at `/workspace/tt-metal/vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin`.

## Verified diagnosis

1. **The plugin already routes every TP4 logprob request to host sampling.** `model_runner.py:2610-2656` accepts device logprobs only with 8 or 32 devices per DP rank. `logprobs=0` counts as a request: only `None` means disabled. Therefore TP4 requests for 0, 1, 3, 5, or 10 logprobs all set `perform_device_sampling=False`, regardless of `sample_on_device_mode="all"`. Top-N support is additionally detected only for model type `gpt_oss` at `model_runner.py:161`. The first serving failure is not the common sampler's TP4 logprob guard; these requests never enter its configuration path.
2. **Host mode is signaled by omission of `sampling_params`.** `TTModelRunner.submit_prefill` (`model_runner.py:2658-2720`) and `TTAsyncDecodeController.submit_decode` (`async_decode.py:552-659`) add that keyword only for device sampling. The adapter defaults it to `None` and immediately raises `ValueError("Host sampling is not enabled for the autoport serving adapter")` in both phases (`generator_vllm.py:124-176`). Actual plugin submission into an adapter constructed with a fake generator reproduces both failures without touching TTNN hardware.
3. **The same boundary blocks required host-only cases.** `check_perform_device_sampling` rejects batches with `allowed_token_ids`, bad words, active logits processors (`min_p`, `logit_bias`, `min_tokens`), custom logits processors, or structured outputs. The shared `tests/tt/test_host_only_params.py` covers the first five; `tests/tt/test_logprobs.py` covers logprobs 0/1/3/5/10 at several batch fractions and an all-vocabulary chat case. Its all-vocabulary test conditionally skips a server-declared max-logprobs limitation. Ordinary tests use temperature 1.0 by default (`tests/tt/utils.py:RequestConfig`), so a greedy-only fallback cannot satisfy their contract.
4. **The existing generator `host_sampling=True` flag is a different API.** `generator.py:_replay` reads full logits, runs host `argmax`, and uploads the chosen IDs. `_validate_sampling` still rejects logprobs and permits only greedy, unpenalized compatibility requests. `_capture` still precompiles and captures the device sampler. Enabling this flag does not expose logits to vLLM's host sampler, and bypassing the logprob guard cannot produce the requested metadata.

## Exact producer/consumer contracts

| Boundary | Required value and ownership |
| --- | --- |
| Host prefill | Adapter returns a materialized CPU `torch.Tensor` shaped `[number_of_prefills, 1, full_vocab]`. Rows follow the packed incoming prompts, even if `empty_slots=[2,0]` maps cache writes into different slots. The final token is selected using each logical `prompt_len`, not rectangular padding. |
| Host decode | Adapter returns materialized CPU logits `[padded_decode_slots, 1, full_vocab]`, preserving slot order. The plugin selects active rows after the return. It must not receive sampled IDs or only one TP vocabulary shard. |
| Synchronous decode finalization | `finalize_decode` accepts a Torch tensor directly. For a raw TT result it instead calls `process_decode_output_host(..., is_tokens=False)`. The current adapter rejects that branch. It does not first call `read_decode_output` in the synchronous path. |
| Host sampling | `_get_output_tokens` indexes `tt_out[rows,-1,:]`, applies grammar constraints, assembles `SamplingMetadata`, and calls the installed vLLM `Sampler`. That sampler owns temperature/top-k/top-p, penalties, allowed tokens, bad words, logits processors, seeds, and returned logprob tensors. The adapter must not duplicate this logic. |
| Lane host extraction | `input_batch.py:_host_logits` scatters packed prefill logits into the supplied scheduled slots, and retains all decode slots. Returning only active compacted decode rows would change request ownership. |
| Device token path | With non-`None` `sampling_params`, return token IDs through the existing split model/sampler traces. Default performance execution must retain zero full-logit readbacks and zero host argmax. |
| Optional asynchronous read | `submit_decode(async_read=True)` calls an existing `read_decode_output` method even when `decode_forward` returned a Torch tensor. If host compatibility can reach this path, implement Torch passthrough returning `(output, [])`. Do not feed Torch logits into `ttnn.get_device_tensors`. Host fallback does not meet steady device-feedback eligibility (`async_decode.py:318-344`). |

`Gemma4Model.logits` already includes terminal normalization, the selected LM head, and final logit softcapping (`model.py:197-215`). Host compatibility should transfer these exact outputs; it should not reimplement the tail on CPU.

## Smallest explicit optional extension

The following is a proposed intervention, **not an implemented or validated device fix**:

1. Add an adapter option such as `allow_host_sampling=False`, set only by an explicit test-server opt-in such as `GEMMA4_AUTOPORT_ALLOW_HOST_SAMPLING=1`. Keep the existing refusal when `sampling_params is None` and this option is false. Keep the normal platform device-sampling setting `"all"`; the option permits the plugin's existing per-batch fallback instead of moving supported requests off device. Do not set `generator.host_sampling=True`.
2. For opted-in host prefill, reuse `gen.prefill_forward(..., return_all_logits=False, slots=empty_slots)` and the same per-layer page tables/cache. This already produces `[N,1,V/4]` sharded logits. Read them through the canonical generator's logits conversion and preserve `[N,1,V]`. Do not request all prompt positions merely to extract the last one; `return_all_logits=True` pads shorter prompts, so its final rectangular row is not necessarily the last real token.
3. Expose a pure low-level generator decode output mode, for example `decode_forward(..., return_logits=False)`, with the default unchanged. Factor the existing replay boundary so this mode executes the same model trace and returns its logits **before** sampler execution; it must not invoke either device sampling or the legacy host argmax branch. Reuse `_bind`, table refresh, token/position refresh, `_forward`, and trace state restoration. This needs no duplicate model implementation.
4. Make trace capture aware of the output mode: host-logit capture skips `sampler.precompile` and `sampler.capture_trace`; device capture retains them. Store the mode with the trace binding and recapture when it changes. Otherwise a trace created in host mode has no valid device sampling capture on return to device mode. Also avoid advancing device sampler seeds while generating host logits. Preserve the original token/position/cache-position state around warmup and capture in both modes.
5. For every host decode, force `device_feedback=False` and upload scheduler-owned tokens and positions. The plugin omits `reset_batch` in host mode; it cannot be used as evidence that device-sampled tokens are available. The new low-level mode can reject `device_feedback=True` explicitly. Read `[1,1,32,V/4]` model trace logits with the existing TP concatenation, slice the logical bound slot count, and return `[slots,1,V]`. A small public `read_logits` wrapper around the existing `_read_logits` can keep readback counters and mesh composition in the generator.
6. Clear the adapter `_sampling_signature` whenever host compatibility runs, and require sampler reconfiguration plus token/position reset on the next device-sampled call. Restore prompt/output penalty history through the plugin's existing fields. The signature must not suppress configuration merely because the device parameters match those used before a host-only step. Mode changes are synchronization boundaries; do not advertise host fallback as overlapped device feedback.
7. Retain separate runtime counters and artifacts: compatibility requests intentionally increment `full_logits_readbacks`; performance artifacts must demonstrate zero such reads and continued `sampling_replays`. A passing host compatibility test is not proof of on-device TP4 logprob support or performance.

## Focused verified host checks

Added [test_vllm_sampling_contract.py](../../tests/test_vllm_sampling_contract.py), **26 passed, 16 warnings in 15.45 seconds**:

- Actual plugin selection for no-logprob versus logprobs 0/1/3/5/10 in both phases, and the host-only selector branches.
- Actual prefill/decode submission into `AutoportGemma4ForCausalLM.__new__` with a fake generator. Both omit `sampling_params`, preserve per-layer tables, and hit the current explicit refusal before any generator operation. Prefill retains noncontiguous cache slots; host decode omits `reset_batch`.
- Actual `_get_output_tokens` plus the installed vLLM `Sampler` consumes `[B,1,V]`, excludes the padded decode row, and returns correct sampled-token log probabilities and top-3 shapes. Torch compilation is disabled only around this CPU numerical check, using `torch.compiler.set_stance("force_eager")`; no model, device, or sampler implementation is substituted.
- Actual decode finalization passes materialized host logits through without invoking device processing.
- Actual lane `_host_logits` preserves decode rows and scatters prefill rows `[2,0]` correctly.

Exact command, from `/workspace/tt-metal`:

```bash
python -m pytest models/autoports/google_gemma_4_26b_a4b_it/tests/test_vllm_sampling_contract.py -q --disable-warnings --tb=short
```

Formatting passed with `python -m black --check --target-version py310 models/autoports/google_gemma_4_26b_a4b_it/tests/test_vllm_sampling_contract.py`; `git diff --check` also passed. This Python/tests/docs change does not require a C++ build.

These passing tests verify the refusal and surrounding host interface; they do not claim that compatibility execution has been implemented. When the option is added, retain the default-refusal checks and add fake-generator opt-in acceptance checks for both phases, expected logits shape/mesh conversion, mandatory scheduler feedback, device-to-host-to-device switching, and Torch readback passthrough. Use deliberately different per-slot token IDs and logits to expose accidental flattening or compaction.

After implementation, device tests still need a reduced real-layer prefill/decode comparison using the same terminal tail, distinct hybrid tables, scheduler page growth, and changed active slots. Exercise host parameter enforcement and per-token logprob numerics before the full shared server suite. Then separately rerun the unchanged split-sampling performance path and confirm its counters. The full test suite, default refusal, reduced hardware checks, and serving performance have distinct claims and must be reported separately.
