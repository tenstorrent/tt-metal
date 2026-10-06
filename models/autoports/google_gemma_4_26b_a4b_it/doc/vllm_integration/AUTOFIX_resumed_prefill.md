# AutoFix: retained output state during resumed prefill

## Verified contract gap

Supported preemption frees KV and replays the full retained prefix. `TTScheduler._preempt_request` (`vllm_tt_plugin/scheduler.py:233`) delegates KV release to the base scheduler and discards outstanding async placeholders. `apply_cached_req_state_update` (`input_batch.py:86`) replaces resumed block IDs and computed-token count without deleting the request's retained output IDs. `_prepare_model_inputs` classifies resumed IDs as prefill and passes the logical total prefix length.

Before this repair, the adapter treated every token in that prefix as an original prompt token, reset output penalty history to zero, and configured the base seed. With `k` retained outputs, resumed prefill must instead:

- Keep the whole prefix as model input to rebuild KV.
- Mark only the original prompt, whose length is `logical_prefix_length - k`, as prompt history.
- Restore the generated suffix as output history, including repeated-token counts.
- Sample the next token with `base + k`. Prefill does not execute the decode trace's seed increment, so its offset is `k`, whereas decode restoration uses `k - 1` before replay.

The earlier decode lifecycle correction did not cover this supported prefill path.

## Negative controls

Before implementation, actual plugin producer/consumer and adapter/canonical-sampler CPU tests produced **4 failures, 21 passes, 52 deselected**:

```sh
python -m pytest models/autoports/google_gemma_4_26b_a4b_it/tests/test_vllm_sampling_contract.py -q -k 'resumed_prefill or plugin_produces_padded_counts or plugin_forwards_counts' --disable-warnings --tb=short
```

Two producer cases lacked retained-output counts, the prefill consumer omitted the metadata, and the sampler-state test observed the base seed rather than `base + 2`. The last test uses actual `Gemma4Generator.configure_sampling`, `sample_prefill`, and `TTPenalties.reset_prompt_tokens/reset_output_tokens` with only TT primitives/model compute replaced by CPU stand-ins. Its mixed batch contains a resumed row with two copies of output token 8, a fresh row, and stale padding token 19. It checks exact prompt masks, output masks, output counts, seed position, and unchanged model tokens/logical prefix lengths.

Evidence: `readiness_vllm/resumed_prefill_cpu_before.log`.

## Minimal repair

The existing optional `output_token_counts` field now covers both prefill and decode for opted-in DP1 models. `submit_prefill` forwards it only for opted-in device sampling. Existing models receive no new keyword; explicit host sampling still owns its sampling state in vLLM.

The adapter validates count vector shape/type/nonnegativity and rejects counts exceeding the logical prefix. For retained outputs, it masks original prompt history and generated suffix separately, passes seed offsets `k` into the canonical generator, and restores output history before sampling. The full unmodified prefix and total logical lengths continue into model prefill. Ordinary zero-count prefill uses its existing masking/configuration path without allocating an additional output-history tensor.

The decode count validator is shared with prefill. Decode seed offset/state restoration and its steady device feedback remain unchanged. Preemption, context limits, scheduler capacity, and precision policy remain supported as before.

## Host validation

The entire sampling contract file passed **77 tests in 33.40 seconds**, including the original negative controls and all earlier optional-host, decode RNG, penalty, and trace-retention tests:

```sh
python -m pytest models/autoports/google_gemma_4_26b_a4b_it/tests/test_vllm_sampling_contract.py -q --disable-warnings --tb=short
```

Evidence: `readiness_vllm/resumed_prefill_cpu_after.log`. A separately added prefix-bound validation test is tracked in `resumed_prefill_count_validation.log`.

## Reduced device oracle

The parent agent owns hardware and runs:

```sh
python -m models.autoports.google_gemma_4_26b_a4b_it.tests.check_vllm_resumed_prefill --output models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/resumed_prefill_reduced.json
```

The helper uses reduced layers 0 and 5. It generates a real three-token retained suffix, invokes adapter resumed prefill, snapshots actual device seed and penalty masks/counts before sampling, and clones those unpenalized device logits. It then configures the canonical sampler explicitly with the original prompt, retained outputs, and seed offset `k`, and samples the cloned logits. Exact state and next-token agreement are required.

The uninterrupted fourth token is recorded for information only. Prefill and decode use different selected precision policies, so exact equality across those distinct model computations is not assumed. The same-logits oracle isolates the sampling-state contract without hiding a numerical-model difference. The helper performs device execution only when the parent invokes it; its `--help` path was checked locally without opening a device.
