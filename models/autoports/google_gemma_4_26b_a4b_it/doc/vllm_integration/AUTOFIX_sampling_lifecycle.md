# AutoFix: request sampling state across decode batch resets

## Verified defects

A surviving seeded request restarted its device seed sequence when a shorter co-batched request finished and changed the sampling-parameter signature. Separately, a slot reorder with identical sampling parameters failed to restore the requests' prompt/output penalty histories. These defects are independent of the previously corrected prefill prompt-padding mask.

The earlier shared `mixed_params` failure involved a first-token discrepancy; its targeted rerun passed after the prompt-mask correction. That result does not establish the cause of these later-token lifecycle failures. The lifecycle findings below have separate source, CPU, and HTTP evidence.

## Source contract

- `tt/generator.py:114` configures the canonical sampler. Its original implementation always wrote `hash(request_seed, 0)` into device seed state. The model trace advances the seed once immediately before each decode sample (`generator.py:296`). Prefill consumes the base seed.
- If `k` output tokens already exist, the current decode input is the last of those tokens. Restoring `base + k - 1` before replay yields `base + k` for the next sample. The current canonical sequence is incremental, not `hash(seed, k)`; changing to the latter would change established standalone output.
- `InputBatch.make_output_token_ids_tensor` in `vllm_tt_plugin/input_batch.py:624` computes output length as `num_tokens - num_prompt_tokens`, includes the latest applied output, and pads token histories with `-1`.
- `TTModelRunner.build_model_input` drains pending decode results after discovering a layout change and before constructing inputs (`model_runner.py:1415`). Counts constructed on that reset are therefore current. Stable overlapped steps can have stale host data and must retain device-owned sampling state.
- Previously, `_prepare_model_inputs` copied prompt/output matrices only when penalties were active. Seeded requests without penalties had no output progress metadata. The added scalar-count transport avoids requesting full prompt histories for that case.
- Before this fix, adapter decode reconfigured only when `repr(sampling_params)` changed. Thus changed parameters reset surviving seeds, while same-parameter row remaps preserved the wrong per-row histories.

## Experiments before the repair

Two durable host tests executed actual adapter methods, canonical `configure_sampling`, and canonical `_forward`, with CPU stand-ins only for TT device primitives and model computation. The seed test observed `[777315, 777316, 777315]` for seed 6, while the correct progression was `[777315, 777316, 777317]`. The row-remap test retained the old prompt/output history after `reset_batch=True` with unchanged parameters. Both failed against the previous implementation:

```sh
python -m pytest models/autoports/google_gemma_4_26b_a4b_it/tests/test_vllm_sampling_contract.py -q -k 'sampling_reconfiguration_preserves or identical_sampling_parameters_refresh' --disable-warnings --tb=short
```

Result: **2 failed, 42 deselected**, preserved in `readiness_vllm/seed_lifecycle_cpu_before.log`. The independent probe artifact is `readiness_vllm/mixed_sampling_state_probe.json`.

The parent agent ran the HTTP probe against the previous loaded full-model server, after the prompt-mask correction:

```sh
python -m models.autoports.google_gemma_4_26b_a4b_it.tests.check_vllm_seed_lifecycle --url http://localhost:8000 --tokens 24 --peer-tokens 3 7 --output models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/seed_lifecycle_before.json
```

The isolated before/after controls matched exactly. All four co-request comparisons failed: first mismatches at output index 3 with the three-token peer, and index 9 with the seven-token peer, in both submission orders. `seed_lifecycle_before.json` records exact token IDs and request parameters. HTTP concurrency alone is not proof of the exact scheduler batch; the observed boundary-dependent divergence corroborates the isolated CPU state proof.

## Minimal repair

1. `TTModelInput.output_token_counts` is optional and defaults to `None`. The normal DP1 builder populates a CPU `int32[B]` vector only for a model advertising `needs_sampling_output_counts`. Active rows contain `num_tokens - num_prompt_tokens`; wire-padding rows contain zero. Both reset and normal builds may carry the vector so a host-to-device sampling transition has current progress even without a layout reset. Existing prompt/output history selection is unchanged.
2. `TTAsyncDecodeController.submit_decode` forwards the vector only to an opted-in model when device sampling is selected and counts are present. Other adapters receive no new keyword. The Gemma4 adapter advertises the capability; it already requires DP1.
3. On a sampling-signature change or trusted batch reset, the adapter validates one nonnegative integer per wire row and derives `max(k - 1, 0)`. Existing direct callers with output histories may derive counts from their nonnegative entries. Stable decode ignores host counts and keeps device feedback.
4. Canonical `configure_sampling` accepts optional `seed_offsets`; its default remains zero. Changed signatures configure the existing sampler and restore current output penalty history.
5. Unchanged signatures use canonical `restore_sampling_state`, which updates only seeds/history and preserves parameter and trace bindings. Greedy batches do not upload seed state in this helper. Prompt/output reset methods remain no-ops when penalties are inactive. This preserves the existing greedy page-growth path without unnecessary trace capture or parameter uploads.

No host argmax, full-logits readback, duplicated sampling implementation, or per-token sampling upload was introduced. Explicit optional host compatibility remains controlled by `GEMMA4_AUTOPORT_ALLOW_HOST_SAMPLING=1`.

## Verification after the repair

```sh
python -m pytest models/autoports/google_gemma_4_26b_a4b_it/tests/test_vllm_sampling_contract.py -q --disable-warnings --tb=short
# From vllm/plugins/vllm-tt-plugin:
python -m pytest tests/test_hybrid_cache_contracts.py -q --disable-warnings --tb=short
```

Results: **64 passed in 29.13 s**, and **25 passed in 5.86 s**. Logs are `readiness_vllm/seed_lifecycle_cpu_after.log` and `seed_lifecycle_plugin_regressions.log`.

Coverage includes both original failing controls, actual producer/consumer opt-in behavior, seeded requests without penalty histories, padded counts, malformed row shape/integrality/negative counts, unchanged device feedback with stale host counts, and actual adapter-to-generator page growth retaining trace ID 17. That page-growth test also verifies zero parameter resets and zero seed uploads for the greedy case.

Model Python files pass Black with `--target-version py310`. The plugin Black diff includes pre-existing unrelated assert-format differences; changed lines were formatted without rewriting those unrelated sections. This is a Python-only change and requires no C++ build.

## Live verification

The parent agent reran both HTTP probes on the loaded decode fix. The default combined-penalty probe and the `--no-penalties` probe both passed: isolated controls and all four shorter-peer comparisons matched exact token IDs. Artifacts are `readiness_vllm/seed_lifecycle_after.json` and the separately recorded seed-only after artifact. The latter exercises scalar-count transport without output-history fallback. Full shared-suite validation must still use final loaded source. A subsequent review identified the distinct resumed-prefill boundary; its extension and evidence are documented in `AUTOFIX_resumed_prefill.md`. No serving performance improvement is claimed from these correctness checks.
