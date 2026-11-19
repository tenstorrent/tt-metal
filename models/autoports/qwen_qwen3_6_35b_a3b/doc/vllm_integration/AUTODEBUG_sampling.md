# AutoDebug: Stage 9 sampling sentinel mismatch

## Finding

**The adapter forwards the pinned plugin's `num_logprobs=-2` (no logprobs) unchanged into a model validator that only accepts `None` or `0` as neutral. This directly explains the first request's `Device num_logprobs is not supported` exception.** `enable_log_probs=False` does not prevent the exception: the validator checks the two fields independently.

The smallest repair belongs in the model-owned `tt/generator_vllm.py` translation boundary. Normalize the plugin's disabled sentinel for both prefill and decode, retain all real feature requests, and leave shared plugin code and prepared device sampling unchanged. Fixing only prefill leaves the same failure on the first device-sampled decode parameter reload.

Confidence is high for this exception's cause. Successful serving after the repair remains unverified; this investigation ran no hardware, server, model execution, or tests.

## Scope and revision consistency

- Model root, abbreviated `M` below: `models/autoports/qwen_qwen3_6_35b_a3b`.
- Plugin root, abbreviated `P` below: `/home/ttuser/qwen36-35b-a3b/vllm-tt-plugin`.
- Inspected tt-metal HEAD: `47c0bbd606a691e6111b46c5ee1eb9e01fd423d8`.
- Inspected plugin HEAD: `042a94863ad27d88207bcf71abdf67122c6f4c27`, with a clean plugin worktree. This matches `M/doc/vllm_integration/manifest.json:5-8`.
- Model adapter and Stage 9 evidence are untracked local work; `tt/generator.py` already has local changes. Findings concern the inspected worktree, not an assertion that these files are in tt-metal HEAD. Relevant logged call sites match the inspected source.
- Primary evidence: `M/readiness_vllm/reduced_v2/server.log`, `lifecycle.json`, and `vllm_run_summary.json`.
- The existing root `AUTODEBUG.md` is deliberately preserved. This task-specific report is the requested output override.

## Direct observations

1. `lifecycle.json:3-18` records one request: length 129, variant 0, `sampled=false`, HTTP 500 after approximately 0.019 seconds. Later requests in the script were not reached: `M/tests/run_vllm_lifecycle.py:51-54` stops after the initial failure.
2. The actual scheduler dump at `server.log:621` contains `temperature=0.0`, `top_k=0`, `top_p=1.0`, `seed=42`, neutral penalties, and `logprobs=None`. Although the client script supplies `top_k=1`, the dump gives the effective upstream request values; these must not be conflated.
3. The server is configured for device sampling in prefill and decode, decode-only tracing, capacity 3, and a 256-token prefill budget (`vllm_run_summary.json:7-29`). The log identifies the actual mesh as 1x4, reduced layers `[0,3]`, and capacity 3 (`server.log:68,620`); the summary's `mesh_device` label alone is not the hardware geometry.
4. Warmup completes and captures `model_fixed` and `sampling` (`server.log:541-548`). Health returns HTTP 200 (`:618`). The sole request reaches prefill with starts `[0]`, ends `[129]`, slot `[0]`, and sampling `device` (`:619`).
5. The traceback is `submit_prefill` -> adapter `prefill_forward` -> generator `prefill_forward` -> `prefill_with_sampling` -> `_neutral_device_features` (`server.log:659-678`). It names precisely the `num_logprobs` check.
6. Replay and readback counters remain zero at failure and release; both captured traces are subsequently released (`server.log:679-680`). This does not prove general device health or queued-work correctness, but is consistent with the synchronous validator failing before this request submits its prefill computation.

The log does not dump the resulting `TTSamplingParams` object. Its exact `[-2]` / `[False]` values are derived below from the pinned production path and the observed request, rather than claimed as a separately logged sample.

## Causal chain and concrete values

| Boundary | Code evidence | Effective value for this request |
|---|---|---|
| No logprobs requested | Scheduler dump `server.log:621` | `logprobs=None` |
| Store per-request sampling fields | `P/src/vllm_tt_plugin/input_batch.py:59,524-532` | `num_logprobs=-2` (`LOGPROBS_NONE_SENTINEL`) |
| Select submitted rows | `P/src/vllm_tt_plugin/model_input.py:86-105`; runner `model_runner.py:2583` | `num_logprobs=tensor([-2])`, `enable_log_probs=tensor([False])` |
| Choose device route | `model_runner.py:3998-4082`; actual event `server.log:619` | No penalties, host-only processors, grammar, or logprob request requires host routing; temperature zero avoids stochastic top-k bounds |
| Build device prefill parameters | `model_runner.py:4112-4126` | Tensors become lists: `num_logprobs=[-2]`, `enable_log_probs=[False]`; only seed sentinel is translated |
| Model adapter | `M/tt/generator_vllm.py:119-145` | Forwards sampling parameters unchanged |
| Serving dispatch | `M/tt/generator.py:340-352` | Non-`None` sampling parameters select `prefill_with_sampling` |
| Neutral feature validation | `M/tt/serving.py:68-78,486-494` | `False` passes its own check; `-2 not in (None,0)` raises the observed `ValueError` |

The explicit pinned contract is documented in executable source at `P/src/vllm_tt_plugin/model_input.py:91-104`: `-2` means no logprobs, while `0` means the sampled token's own logprob and sets `enable_log_probs=True`. The model's neutral convention instead matches `models/common/sampling/sampling_params.py:23-24`, whose defaults are `False` and `0`. These are two representations of the same disabled state, with no conversion between them.

This also explains why preparation passes: adapter warmup calls `generator.prepare()` (`generator_vllm.py:103-117`), whose prefill warmup calls use `device_output=True` without plugin sampling parameters (`generator.py:493-496`). They do not exercise the failing translation and validator combination. Health checking likewise does not exercise a model request.

### Completion boundary

The reported failure is a synchronous Python rejection. `prefill_with_sampling` checks these fields at `serving.py:489`, before its low-level model call at `:523` and before sampling trace enqueue at `:549`. Decode performs the same validation at `:305-306`, before its reload executor and nonblocking trace enqueues at `:379-385`. There is no need to infer a kernel, synchronization, allocator, or teardown failure to explain this exception. Warmup's device work and capture are separate earlier execution and are not end-to-end serving acceptance.

## Minimal model-owned intervention

Add one small adapter helper and use it at both forwarding sites, `generator_vllm.py:144` and `:157`, when `sampling_params` is present:

1. Copy the incoming sampling object without mutating it. The plugin dataclass is frozen (`P/src/vllm_tt_plugin/model_input.py:68`). `dataclasses.replace` is suitable for the pinned list-valued device contract.
2. Map each known `num_logprobs == -2` sentinel to the model's neutral `0` (or `None`). Preserve row count and order, including inactive decode padding rows. Retain the supplied `enable_log_probs` field.
3. Preserve `0`, positive counts, and other unexpected values as supplied. In particular, do not interpret `num_logprobs=0` as proof that logprobs are disabled: the paired `enable_log_probs=True` requests a real feature and must remain visible to validation.
4. Preserve temperature, top-k, top-p, seed, penalties, and every decode command. Keep `sampling_params=None` as `None`, since that selects host logits. Do not derive parameter lengths from stale decode tokens or positions.

For the ordinary pinned payload, the only semantic change is:

```text
plugin: num_logprobs=[-2, -2, ...], enable_log_probs=[False, False, ...]
model:  num_logprobs=[ 0,  0, ...], enable_log_probs=[False, False, ...]
```

The helper should handle absent/`None` `num_logprobs` without manufacturing a feature request; the observed plugin device path supplies lists. An inconsistent `(-2, True)` payload still fails because the true enable flag is retained. Avoid replacing the entire parameter object with just temperature/top-k/top-p/seed: that would discard nonneutral penalties or logprob requests and could silently change semantics.

An alternative localized change is teaching `_neutral_device_features` that `-2` is also neutral. That would cover both generator entry points, but expands the plugin-neutral serving layer's accepted representation. The adapter is the narrower owner of plugin-to-model translation, and already owns mRoPE and multimodal compatibility conversions. No plugin change, new capability, environment switch, automatic sampling fallback, or change to captured sampler buffers is needed for this mismatch.

### Decode and inactive rows are part of the same repair

`P/src/vllm_tt_plugin/async_decode.py:1373-1390` repeats the prefill conversion: every sampling tensor becomes a list, the no-seed sentinel becomes `None`, and `num_logprobs=-2` remains unchanged. The model adapter forwards that object at `generator_vllm.py:157`.

At the first device decode after prefill, the plugin starts a new decode chain and reloads inputs, sampling parameters, and sampling state (`async_decode.py:409-411,457-483`). This reaches `serving.py:305-306`. Merely fixing prefill therefore exposes the same error during decode. Later resident steps without a sampling reload do not call this validator; their lack of an exception would not demonstrate that the translation is correct.

For capacity 3 and one request, `_sampling_params_for_padded_decode` pads sampling fields to the physical decode row count and derives the enable flags (`P/src/vllm_tt_plugin/model_runner.py:1230-1261,2583-2589`). Padding defaults themselves use `num_logprobs=-2` (`input_batch.py:206-217`). Validation checks every submitted row, including inactive rows, before parameter updates filter active rows (`serving.py:305-310,362-366`). Normalize all parameter rows, not just the live request's row.

## Nearby parameter audit

The following closely related paths were inspected rather than assumed to fail with the sentinel:

| Candidate | Inspection result |
|---|---|
| `enable_log_probs=False` as a list | Already accepted: `_parameter_rows` accepts lists/tensors and `_neutral_device_features` accepts `False`. The count sentinel alone is the mismatch. |
| Real `logprobs=0` requests | Correctly distinct from no logprobs. On this four-device mesh the plugin routes any requested logprobs to host sampling (`model_runner.py:4070-4080`). Retaining the true enable flag also protects the direct device entry point. |
| Greedy top-k after upstream normalization | Observed effective `top_k=0` becomes the real vocabulary size in `input_batch.py:468-471`. Both the plugin bound check (`model_runner.py:4034-4039`) and model bound check (`serving.py:64`) apply only to nonzero temperature. `generator.py:233-239` maps greedy parameters to the prepared sampler's `k=1,p=0` convention. No second greedy top-k blocker is established. |
| No-seed sentinel | Both plugin submission paths explicitly convert it to `None`; model sampling accepts `None` (`serving.py:62-63`, `generator.py:228-231`). The failing request uses explicit seed 42. |
| Neutral penalty fields | Plugin defaults `0,0,1` match model validation. Model capabilities route nonneutral penalties to host (`serving_capabilities.py:36`; `model_runner.py:4041-4044`). Do not strip these fields as part of the repair. |
| Decode row padding and explicit commands | Sampling rows are padded with the input rows. The first transition requests authoritative reload and state reset; ordinary resident commands retain host-stale token/position behavior. Sentinel normalization needs no change to those commands. |
| Ordinary mRoPE kwargs | Prefill returns zero deltas for text; decode removes and validates the plugin's `rope_deltas_all_users` before forwarding (`generator_vllm.py:145,148-150`). No unknown ordinary decode kwarg is established here. |

## Other Potential Issues: continuation can lose its state-slot owner

**Separate static finding, not the cause of the recorded exception:** the front-packed runner can assign an unfinished chunked-prefill request a different physical slot without moving its retained state. This requires multiple requests and a row-layout change between prompt chunks. The failing artifact has only one request at start position zero and does not demonstrate this scenario.

`P/src/vllm_tt_plugin/model_runner.py:1295-1336` allocates state slots for every prefill, including continuations (`:2422-2433,2768-2774`). It protects only *off-batch* owners in `held` (`:1311-1316`). For a request being continued in this batch, it ignores that request's previous owner and prefers the new host row (`:1318-1334`). The prefill path sends the chosen slots but no state permutation; `_decode_state_slot_remap` only runs in the decode branch (`:2765-2774`).

Hand-evaluated concrete sequence, using capacity 3, budget 256 and the logged 128-token partial-prefill alignment:

| Step | Host/ownership state | Consumed model state |
|---|---|---|
| Initial prefill | A has a 128-token prompt and `max_tokens=1`; B has a 512-token prompt. Schedule A's complete 128 tokens and B's first 128. Allocation records `{A:0, B:1}`. | A's state is written at slot 0; B's prefix state at slot 1. Both intervals satisfy the scheduler's alignment rule (`scheduler.py:819-856`). |
| Finish A | The mixed final/intermediate prefill uses host logits (`model_runner.py:2665-2669`); A's one output token finishes it while B remains mid-prompt. On a subsequent step A is released and its claim dropped (`:988-1003,1024,1290-1291`). Host condensation moves B to row 0 (`:1089-1092`). | Model `cache_contract.py:315-318` clears slot 0; B's prefix state still lives at slot 1. |
| Continue B | `_alloc_prefill_state_slots([B])`: B is excluded from `held`, so `held={}`; row 0 is available; B's owner is overwritten from 1 to 0. Start position is 128. | `empty_slots=[0]` is forwarded without a remap. `generator.py:405-415,430-434` consumes the suffix in slot 0 and only resets on `start==0`. `prefill.py:210-214` selects `states[slot]` for each layer. The expected prefix recurrence at slot 1 is not used. |

The model's prefill validator checks dimensions, tokens, extents and page ownership, not request identity or prefix-state correspondence (`M/tt/prefill_validation.py:87-108,133-156`). Rebinding B's correct page table restores full-attention KV addressing, but cannot restore its per-slot recurrent/convolution state. This makes the slot mismatch meaningful for this hybrid model, rather than a harmless host-row reorder.

This is a production-code ownership contradiction for the stated sequence; the sequence and resulting numerical corruption were not executed. Its smallest intervention is preserving and reserving existing continuation owners in the plugin's state-slot allocator before allocating new requests. The adapter receives no request IDs with which to repair that map directly. Keep it separate from the model-owned sentinel repair; if the parent remains scoped to model-only changes, carry this finding as a follow-up rather than silently adding identity reconstruction or claiming the sentinel fix establishes mixed-request correctness.

## Test gap and follow-up validation for the parent

Existing model-owned CPU serving probes use dictionaries containing only temperature/top-k/top-p/seed (`M/tests/run_serving_commands_cpu.py:134,233`; device probe `run_serving_commands_probe.py:163`). They omit the fields that trigger this bug. Existing adapter host tests mock the inner generator (`M/tests/test_vllm_adapter_host.py:46-78`); they check forwarding and mRoPE, without passing the pinned disabled-logprob representation through the real neutral-feature validator. The plugin's own reload fixtures use the actual `-2` / `False` representation (`P/tests/test_decode_reload_contract.py:68-79`), so this is a model/plugin boundary coverage gap.

Suggested checks after implementation (not run by this investigator):

1. Host-only adapter coverage using actual pinned `TTSamplingParams` device payloads for both prefill and decode: all-disabled rows, mixed greedy/stochastic rows, and capacity-3 rows with inactive padding. Exercise the real neutral-feature validator, not only a mock that accepts every argument.
2. Verify no mutation of the incoming frozen dataclass or its lists, unchanged seeds and row order, and preservation of `sampling_params=None` host mode. Ensure nonneutral penalties, `enable_log_probs=True` with count `0`, and positive counts are still rejected on direct device entry.
3. Check a resident decode with `reload_inputs=False` and `reload_sampling_params=False` still does not inspect stale tokens/positions or reset sampling state because of the helper.
4. Repeat the existing reduced lifecycle: initial 129-token greedy request must reach decode; then the script's concurrent 130/517/131-token prompts, reordered requests, bounded stochastic request, and return to greedy. Existing evidence stops before all of these; none is presently a passing contrast. Record prefill/decode commands and trace replay/readback/release evidence.
5. Separately exercise the 128-token A / 512-token B sequence above with A limited to one output token. Record B's physical state owner across prompt chunks and compare against a stable-slot continuation. This tests the secondary ownership concern; the existing lifecycle does not contain that exact controlled sequence.

## Claim review and remaining limits

The headline was challenged against the exact recorded request, the pinned request storage/slicing/conversion, both model entry points, the validator ordering, and the warmup bypass. It explains successful preparation, first-request timing, the exact failing field, and why no request replay occurred. It does not depend on a malformed request, nonneutral penalty, stochastic sampler behavior, or a device failure.

No additional nearby parameter mismatch is necessary to explain this request. The separate continuation-ownership finding requires a different, explicitly stated schedule and remains unobserved in this log. Reduced layers `[0,3]` exercise integration mechanics only; this report provides no accuracy or full-model performance claim. Later serving behavior, mixed-request state ownership, output completion, and long-context execution require parent validation after the synchronous blocker is repaired.
