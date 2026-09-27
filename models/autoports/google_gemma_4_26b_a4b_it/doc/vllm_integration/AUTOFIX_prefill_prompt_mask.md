# Prefill repetition-history padding repair

Date: 2026-09-27. Status: **host-verified; serving process must restart and rerun the final path**.

## Verified cause

The adapter passed rectangular prefill `tokens` directly as `prompt_tokens` to `gen.configure_sampling`. The plugin prepares those tokens from `input_batch.token_ids_cpu_tensor[req_indices, :max_prefill_tokens]` (`model_runner.py:1148-1152`), which includes padding or stale buffer contents after shorter prompts. `InputBatch.add_request` writes only the new live prefix (`input_batch.py:326-328`).

`SamplingGenerator.reset_prompt_tokens` forwards active penalty history to `TTPenalties.reset_prompt_tokens`. The latter counts every ID satisfying `0 <= id < vocab_size` and builds a presence mask (`tt_penalties.py:218-223`, `258-290`). It does not know logical prompt lengths. The repetition penalty combines this prompt mask with generated-token history, so a padded zero or stale positive ID can change the first sampled token. Presence/frequency penalties use output history; this particular prompt-mask defect affects repetition.

## Smallest retained fix

In `tt/generator_vllm.py:prefill_forward`, build a sampler-only copy with positions at or beyond each `prompt_len` replaced by `-1`, then pass it to `configure_sampling`. The original `tokens` object remains the model input. The canonical generator, device sampler, precision policy, and cache routing are unchanged.

Decode already satisfies the contract. The plugin calls `InputBatch.make_prompt_token_ids_tensor(req_indices)` when decode penalties are active. That helper masks each selected row beyond its original `num_prompt_tokens` to `-1`, in requested row order (`input_batch.py:610-622`). Padded batch rows are also `-1` (`model_runner.py:1274-1282`). No decode implementation change was required.

## Discriminating checks

Added two parametrized prefill cases and one decode-history case to `tests/test_vllm_sampling_contract.py`:

- Prefill rows have lengths 3 and 5. The short row's two trailing entries are either zero or stale ID 19; the long row contains a legitimate token zero. The test calls the actual common `SamplingGenerator.reset_prompt_tokens`, actual `TTPenalties.reset_prompt_tokens`, and actual `_token_counts_host`, replacing only the final device upload with a capture. It checks the complete 32-row mask, preserves duplicates correctly as mask membership, and verifies the original model tensor is untouched.
- The actual plugin decode prompt helper is called in reordered request order `[1,0]` against stale input storage. It returns the expected `-1` suffix without changing the backing token buffer.

Before repair, the focused checks produced **2 failed, 1 passed, 39 deselected**: both prefill padding cases failed and the existing decode helper passed. After repair, the full adapter CPU suite produced **42 passed, 16 warnings in 21.61 seconds**.

Exact commands from `/workspace/tt-metal`:

```bash
python -m pytest models/autoports/google_gemma_4_26b_a4b_it/tests/test_vllm_sampling_contract.py -k 'prefill_penalty_mask or plugin_decode_prompt_history' -q --disable-warnings --tb=short
python -m pytest models/autoports/google_gemma_4_26b_a4b_it/tests/test_vllm_sampling_contract.py -q --disable-warnings --tb=short
python -m black --check --target-version py310 models/autoports/google_gemma_4_26b_a4b_it/tt/generator_vllm.py models/autoports/google_gemma_4_26b_a4b_it/tests/test_vllm_sampling_contract.py
git diff --check
```

The focused command records the pre-repair result; the full suite and formatting/whitespace checks passed afterward. Transcripts are `readiness_vllm/prefill_prompt_mask_baseline.log` and `readiness_vllm/prefill_prompt_mask_host_tests.log`. No server request, hardware execution, restart, or active-suite-log overwrite was performed by this repair task. No C++ build is needed.

The currently running server may still contain the old adapter class. The main stage must restart it after its active suite finishes and rerun the final serving path, including heterogeneous prompt lengths with repetition penalties. These CPU results prove history construction and mask contents, not accelerator output behavior.
