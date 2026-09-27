# Prefill state slots versus page-table rows

2026-09-27. Source and host investigation only. No accelerator or server access.

## Diagnosis before host reproduction

The adapter at diagnosis time conflated two independent indices. Plugin `TTModelRunner._prepare_model_inputs` builds page tables in compact input-row order using `InputBatch.block_tables_for_rows(range(num_reqs), width)`. Its `_block_tables_per_layer` pads trailing rows with zero; it does not scatter requests to persistent device-state slots. Separately, `_alloc_prefill_state_slots` avoids overwriting the recurrent state of live off-batch requests. For example, when A holds state slot0, a new one-row prefill for B has table row0=B but `prefill_empty_slots=[1]`.

`TTModelRunner.submit_prefill` forwards those state slots as `empty_slots`. `AutoportGemma4ForCausalLM.prefill_forward` forwards them to `Gemma4Generator.prefill_forward(slots=empty_slots)`. The generator passes each slot as `Gemma4Model.prefill_forward(user_id=slot)`, which propagates the value to the selected decoder. `OptimizedAttention.prefill` passes it as `paged_fill_cache(batch_idx=user_id)`. The native operation documents that `batch_idx` indexes the page-table row, not a physical-cache or persistent-state slot (`paged_fill_cache_device_operation.cpp:59`). Thus B selects padded row1 (physical page0) instead of its own row0. With two newcomers B/C and A already live, slots `[1,2]` make B write C's row and C write the padded row.

This model's prefill does not retain separate slot-indexed recurrent state. The minimal adapter repair is to use compact row indices for prefill cache filling: omit `slots=empty_slots`, letting the generator choose `range(len(prompt_lens))`, or pass that explicit range. Keep the plugin state-slot mechanism intact for models that require it. Preserve the generic generator's existing `slots` API for standalone callers whose tables are deliberately indexed by slots.

## Relation to the live failure

The parent observed correct prefill-generated tokens followed by divergent first decode tokens for later requests. Single-chunk prefill attention reads its fresh Q/K/V tensors directly, while decode reads the paged cache, so a wrong cache destination can preserve token0 and corrupt token1. The source mechanism therefore matches the observed boundary.

The additional prompt-list order reversal (first request correct, second wrong) does not establish that both prefills ran in one engine step. Requests in one HTTP list can still enter scheduling separately. If actual `empty_slots` is `[0,1]` for an actual two-row prefill, this specific mapping bug is not exercised. Record prefill token shape, slots, and first physical page IDs to distinguish that case; the parent's independent direct B2 probe with identity slots is also discriminating.

## Other mechanisms inspected

The model and `OptimizedDecoder.decode_forward` consistently slice hidden row, `current_pos[:,slot:slot+1]`, `cache_pos[slot:slot+1]`, and `page_table[slot:slot+1,:]`. No Python index mismatch was found. A host forwarding test can verify those arguments, but cannot validate TTNN's physical row slicing.

`MultichipDecoder.allreduce` reuses persistent scratch by role and shape. Its `_forward` consumes attention/routed/shared results before returning a fresh residual result; the slot loop retains those final outputs. There is no demonstrated direct returned-tensor alias from source. A device-level CCL synchronization defect remains possible, but is lower confidence than the concrete prefill contract violation. No speculative collective fix is recommended without a failing isolated device probe.

## Host reproduction

Added `tests/test_vllm_prefill_slots.py`. Its five cases call real `InputBatch`, `TTModelRunner._prepare_model_inputs`, `submit_prefill`, adapter prefill, and `Gemma4Generator.prefill_forward`. Only TTNN operations and model arithmetic are replaced with host tensors. The model stub records the exact `user_id` and physical page IDs forwarded to each layer.

The main agent independently removed `slots=empty_slots` while this reproduction was prepared. The repaired implementation passed **5/5** cases: fresh one/two requests, one existing holder with one/two newcomers, and three existing holders with one newcomer. The one-holder/two-newcomer case explicitly verifies `empty_slots=[1,2]` while the selected KV rows remain `[0,1]`.

An in-memory negative control restored only the prior adapter keyword `slots=empty_slots`, without changing any source file. It produced **3 failures and 2 passes**. Both fresh identity cases passed. With one holder/one newcomer, the selected physical pages were `[0,0]` instead of `[10,100]`; with two newcomers, the first selected the second request's `[18,108]`, and the second selected `[0,0]`; three holders again selected padded zero pages. This isolates the prefill index contract violation independently of model arithmetic and device behavior.

Exact repaired command, from the repository root:

```bash
python -m pytest models/autoports/google_gemma_4_26b_a4b_it/tests/test_vllm_prefill_slots.py \
  -q --disable-warnings --tb=short
```

The parent also reports a passing direct accelerator B2 probe with compact tables and identity slots, consistent with the host identity controls. Accelerator runs and the subsequent serving retry remain owned by the parent. This subagent made no implementation changes or device calls. The host results establish argument/cache-address semantics, not numerical accelerator correctness.
