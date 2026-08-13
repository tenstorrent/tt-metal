# AutoDebug: Mistral vLLM long-prompt corruption after the first token

## 2026-08-13 final resolution

Later raw-ID and model-versus-sampler diagnostics superseded the hypotheses
below. The traced model logits were already corrupt while positions advanced,
and sampled IDs matched model argmax. Startup warmup had executed a decode at
position 0 into vLLM's real BFP8 serving cache; the first real prefill then
rewrote already-quantized cache tiles. Compile warmup now uses a disposable
exact-geometry scratch cache, capture records without replay, serving state is
restored, and startup resets the vLLM cache before the first request. Final
production raw IDs and all qualitative outputs are coherent. The older sections
remain as investigation history and should not be read as the final diagnosis.

## 2026-08-13 follow-up: fast-tokenizer / async-scheduling correlation

### Headline conclusion

The new evidence **rules out `FastIncrementalDetokenizer` as a direct source of
the corrupt sampled IDs**.  It may change frontend/output-processing timing and
therefore expose a latent TT async state bug, but it has no path back into the
scheduler, model runner, KV cache, trace token buffer, or sampled-token
selection.

This distinction is source-backed:

- `vllm/vllm/v1/engine/detokenizer.py:55-65` selects the fast implementation
  only from the tokenizer object's runtime type.  Both implementations are
  request-local output processors.
- `BaseIncrementalDetokenizer.update()` at lines 102-140 consumes already
  sampled `new_token_ids` and appends them to its private `token_ids` and
  `output_text`; `FastIncrementalDetokenizer.decode_next()` at lines 205-242
  only advances `tokenizers.DecodeStream`.
- `RequestState._new_completion_output()` in
  `vllm/vllm/v1/engine/output_processor.py:377-408` obtains the response token
  IDs from that output state after engine-core output processing.  The OpenAI
  response then forwards `output.token_ids` unchanged at
  `vllm/vllm/entrypoints/openai/chat_completion/serving.py:1515-1518`.
- The observed `return_token_ids` are already corrupt after the correct prefill
  token.  Therefore bad text decoding cannot explain the observation.

Exact slow/fast full-chat `input_ids` equality also excludes the renderer,
system prompt, regex correction, prompt length, and prompt token contents as
the direct delta.  A fresh-server comparison is still required before treating
the tokenizer class as causal: tokenizer type is otherwise disjoint from the
TT model-input and sampling path.

### Smallest discriminating experiment

Run the same exact token-ID request in fresh, identically reset server
processes, record raw returned IDs, and vary only one control at a time:

1. fast tokenizer, `async_scheduling=True` (reported failure);
2. fast tokenizer, `async_scheduling=False`;
3. slow tokenizer with a ByteLevel-correct `convert_tokens_to_string`,
   `async_scheduling=True`.

Interpretation:

- (1) fails and (2) passes: investigate the TT async-ahead feedback lifecycle;
- (1) fails and (3) passes with byte-identical prompt IDs: tokenizer selection
  is only a timing/lifecycle trigger, and the slow wrapper is a safe narrow
  workaround while the race is isolated;
- (1) and (2) both fail: the regression is server/device/trace state rather
  than vLLM async scheduling;
- all fresh runs pass: the earlier result came from uncontrolled process,
  trace, cache, or device state.

The proposed slow-tokenizer workaround should preserve the existing slow
scheduler/output path and override only `convert_tokens_to_string` with the
checkpoint tokenizer's ByteLevel decoder.  It is the smallest practical change
because it fixes literal `Ġ`/`Ċ` output without changing prompt IDs or
selecting `FastIncrementalDetokenizer`.  It must be described as a workaround /
discriminator, not a proven fix for corrupt sampled IDs.

### Ranked remaining hypotheses

1. **Latent TT async-ahead feedback/state race exposed by a timing change
   (plausible, not yet proven).** `TTAsyncDecodeController` deliberately permits
   one step of host-state lag.  `MistralSmall24BGenerator.decode_forward()` at
   `tt/generator.py:918-930` then ignores scheduler token/position input when
   `reset_batch is False`, trusting the device-resident trace token and current
   position.  A missed drain, stale completion, or wrong reset classification
   would predict exactly a correct synchronous prefill sample followed by bad
   decode IDs.  Mocked stale-token tests establish intended control flow but do
   not reproduce timing, device writes, or trace replay ordering.

2. **Uncontrolled server/device/trace-state difference between tokenizer runs
   (equally important until A/B is clean).** No source path makes tokenizer
   type an input to TT sampling.  Separate live launches can differ in retained
   processes, trace allocation, cache contents, or device health.  The fresh
   three-way A/B above is required to distinguish this from hypothesis 1.

3. **Fast incremental detokenization itself (refuted as the cause of sampled
   ID corruption).** It can explain text differences only.  It cannot alter
   the raw IDs emitted by EngineCore/model runner.

### Most useful instrumentation if async-only failure is confirmed

For the first prefill token and next two decode submissions, log a compact
per-request ledger at the boundaries of
`TTAsyncDecodeController.capture_submitted_step_context()`,
`complete_decode_step()`, and `apply_completed_decode_step()`:
request ID, live row, captured request-state identity, scheduler input token and
position, trace-resident token and position before replay, sampled token read
back, `reset_batch`, `slot_remap`, and pending/completed queue depth.  The first
divergence determines whether corruption enters before trace replay, in device
sampling/readback, or only during deferred host-state application.  This is
stronger than another mocked test because the existing tests cannot observe
device-resident feedback or real completion timing.

This follow-up supersedes any interpretation elsewhere in this report that
attributes the new fast-tokenizer-correlated raw-ID failure directly to text
detokenization.  The earlier non-aligned trace-warmup/cache finding is a
separate, already-tested failure mode and does not establish the cause of this
new comparison.

## Scope and current evidence

This is a fresh, inspection-only diagnosis of the serving-quality regression
for `mistralai/Mistral-Small-24B-Instruct-2501`. No implementation or hardware
state was changed.

Observed facts:

- The corrected Mistral tokenizer and served chat rendering are token-exact
  (`readiness_vllm/vllm_chat_template_exact_match.json`).
- The six served prompts are about 185--200 tokens. Their first greedy token is
  plausible (`Data`, `Sure`, `Once`, `The`, `Certain`), but subsequent device-
  sampled decode collapses into repeated subwords such as `outh`, `WS`, and
  `mp` (`vllm_qualitative_outputs.json`).
- The same six prompt templates are coherent through the standalone full-model
  generator, including traced split sampling (`doc/full_model/qualitative_suite`).
- The final 73-test sampling profile passes. That suite proves sampling API
  semantics and reproducibility, but not natural-language quality; it does not
  contradict a deterministic decode-attention corruption.
- Prefix caching and vLLM chunked prefill are disabled in `server.log`.

## Headline finding

The strongest source-level explanation is an **invalid block-table tail read by
the paged SDPA decode kernel's rounded 128-token window**.

The recent request-isolation fix in
`vllm_tt_plugin/input_batch.py::block_tables_for_rows()` correctly stops stale
physical block IDs from a previous request from escaping row reuse, but replaces
every entry after `num_blocks_per_row` with `-1`. The Mistral decoder uses
`k_chunk_size=128`. TT paged SDPA computes
`valid_seq_len = nearest_n(cur_pos + 1, k_chunk_size)` and physically reads all
K/V tiles in that rounded window before masking positions after `cur_pos`.
Consequently, a first decode at position 185 rounds to 256 tokens: eight 32-token
pages are physically read even though only six pages contain the sequence. The
two tail lookups see `-1`, which is consumed as an unsigned physical-page ID by
the dataflow kernel. At position 192, the scheduler can own seven pages while
the 256-token read still requires eight table entries.

This explains the otherwise unusual boundary cleanly:

1. Dense prefill produces the plausible first token and fills only scheduler-
   owned pages.
2. The first decode switches to `paged_scaled_dot_product_attention_decode`.
3. Its 128-token-rounded physical read dereferences the newly introduced `-1`
   tail; causal masking is too late to make an invalid physical read safe.
4. Standalone generation is clean because its default page table contains valid
   physical pages across the full fixed-width row rather than `-1` sentinels.

The timing also fits: replacing stale tail IDs with `-1` fixed the ordered-suite
cross-request contamination, after which the full sampling suite passed, but
the long qualitative check exposed deterministic decode gibberish. The two
symptoms are different consequences of the same fixed-width block-table tail:
stale foreign pages leak request state; invalid `-1` pages make rounded reads
undefined.

## Important correction: this is not suffix-prefill decode

The production generator default is `prefill_chunk_size=576`, and the server
does not override it. Prompts of 185--200 tokens therefore remain entirely in
`_run_initial_prefill`; `_run_suffix_decode_device_final()` is not entered.
The relevant transition is **dense padded prefill -> paged traced decode**, not
an internal 128-token prefill suffix. The number 128 comes from the decoder's
paged-SDPA `k_chunk_size`, not the prefill chunk size.

## Smallest verify/refute experiments, in order

### 1. Safe-tail A/B on one exact prompt

Keep the scheduler's valid prefix unchanged. For each active row only, replace
the emitted tail after `num_blocks_per_row` with that row's first owned physical
page instead of `-1`. This is a read-safe alias for causally masked padding; it
does not expose another request's page and does not allocate or claim hidden KV
ownership. Leave zero-block inactive rows unchanged because position `-1`
returns before paged lookup.

Run one exact 185-token chat prompt for 16 greedy tokens before and after. The
hypothesis predicts:

- first token unchanged;
- subsequent output becomes coherent in the safe-tail run;
- no `outh`/`WS`/`mp` collapse.

This is the highest-value experiment because it changes only the suspected
invalid addresses while preserving model, sampler, trace, scheduler, page
ownership, and workload shape.

### 2. Rounded-coverage logging/assertion

At first decode, log per active request:

- request ID, row, `cur_pos`, and valid block count;
- `rounded_tokens = ceil((cur_pos + 1) / 128) * 128`;
- `rounded_blocks = rounded_tokens / 32`;
- the table prefix through `rounded_blocks`.

For the failing prompt, the current code should show six valid blocks followed
by two `-1` entries at position 185. This establishes the exact first invalid
lookup without reading full KV tensors.

### 3. `k_chunk_size=32` discriminator

As a diagnostic only, use paged decode with `k_chunk_size=32` for the same
prompt. The rounded window then stays within the current logical page. Coherent
generation would independently confirm rounded tail coverage. Revert this
performance-changing control after the experiment; it is not the preferred
fix.

### 4. Eager-versus-traced decode on the same pooled cache

With the original `-1` tail, compare host/eager decode and traced device decode
after an identical pooled-cache prefill. Both call the same paged SDPA and
should corrupt if the tail is causal. If eager is clean and trace corrupts,
redirect investigation to trace-bound page-table/current-position state.

### 5. Boundary sweep

After the focused A/B, sweep prompt lengths 127, 128, 129, 191, 192, and 193
with a fixed continuation. Record first divergence and the corresponding
`cur_pos`, valid pages, and rounded pages. A cliff aligned with 128-token SDPA
windows is confirming evidence; a cliff aligned only with trace capture or row
movement would refute the current ranking.

## Smallest fix hypothesis

Change the TT-emitted constant-width table, not vLLM's scheduler ownership:

- preserve the first `num_blocks_per_row` scheduler-owned IDs exactly;
- for an active row with at least one valid block, fill the causally masked tail
  with one block already owned by that same row (the first block is sufficient);
- retain an assertion that every logical page through `cur_pos` is genuinely
  scheduler-owned; the alias is permitted only beyond that semantic coverage;
- keep inactive rows at a harmless default because their `cur_pos=-1` skips the
  paged reader.

Add a unit regression that reuses a row `[11, 12, 13] -> [21]` and expects an
emitted width-four table of `[21, 21, 21, 21]`, not stale IDs and not `-1`.
Then rerun the exact qualitative request, the full 73-test profile, stale-token/
position/page-table adapter tests, and the six-prompt qualitative suite.

If same-row tail aliasing is not acceptable as the durable interface contract,
the deeper alternative is a TT paged-SDPA kernel change that never dereferences
pages wholly beyond `cur_pos` (or explicitly supports a sentinel by zero-filling
those reads). That is a substantially larger intervention and should be pursued
only if experiment 1 proves the mechanism but the plugin workaround is rejected.

## Ranked alternatives

1. **Rounded paged-SDPA read through the `-1` table tail -- very high.** It
   explains the exact prefill/decode boundary, standalone/vLLM difference,
   recent tail-mask chronology, and deterministic garbage.
2. **Trace page-table/current-position lifecycle -- medium-low.** Still testable
   with experiment 4, but short trace tests and standalone traced generation
   pass, and the first decode shares paged SDPA with eager mode.
3. **Pooled page ownership or prefill fill error in the valid prefix -- low.**
   The first prefill token is plausible and the earlier active-row/stale-tail
   fixes cover the known ownership hazards.
4. **Sampling/token feedback/CCL error -- low.** Both greedy and sampled output
   degrade after the boundary, while the canonical split sampler and trace are
   clean standalone and the full sampling profile passes.

## Required next evidence

Run experiment 1 before any broad trace, sampler, cache-reset, or collective
change. A clean safe-tail result verifies the smallest causal mechanism and
identifies a fix that preserves vLLM cache ownership and the production traced
sampling path.

## Follow-up: max-num-seqs 1 versus 32 after device reset

New evidence materially changes the ranking above:

- after a board reset, an exact 192-token request is coherent through both
  eager host decode and traced device decode when the server/model is built
  with `max_num_seqs=1`;
- the production `max_num_seqs=32` server remains first-token-correct and then
  corrupt;
- a direct pooled-cache generator built with `max_batch_size=32` produces a
  coherent continuation through eager host decode for the same exact prompt
  and page layouts (`readiness_vllm/paged_cache_ab.json`);
- same-row safe-tail aliasing and `k_chunk_size=32` do not restore coherent
  production output.

Therefore the rounded-tail hypothesis is refuted as the sufficient cause, and
physical batch-32 model math is not intrinsically broken. The unresolved delta
is the **vLLM one-live-request -> 32-row decode padding and traced sampling
contract**.

### Precise source mismatch

`TTModelRunner.build_model_input()` pads a one-request decode to
`tt_per_lane_max_num_seqs` (32):

- token rows 1--31 become token 0;
- position rows 1--31 become `-1`;
- block-table rows 1--31 become all zero.

The adapter passes that physical tensor unchanged. In
`MistralSmall24BGenerator.decode_forward()`, `active_batch` is computed as
`tokens.reshape(-1).numel()`, so the generator treats the padded request as 32
active sampler/trace rows. It captures a batch-32 model and sampling trace and
`set_sampling_params(..., active_batch=32)` includes every padded row. Cache
update and paged SDPA skip `position=-1`, but the rest of the decoder still
computes embeddings, residual/MLP, final norm, LM-head logits, and Sampling1D
for those 31 rows. Unlike the clean batch-1 server, the sampled-token feedback
buffer is consequently rewritten from a 32-row logits tensor containing 31
non-semantic rows.

This is a contract bug even before the exact corrupting op is localized: wire
capacity is being used as semantic active batch. The generator has no explicit
active-row count or mask at decode, despite the runner retaining the true count
in `model_input.unpadded_batch_size`.

### Smallest decisive test

Use one full-model load and the exact 192-token prompt with
`max_batch_size=32`, one active request, and identical pooled cache/page table:

1. Eager-host control using the 32-row padded token/position inputs; save the
   active-row logits/top-8 token IDs for the first three decode steps.
2. Traced device path with the same inputs; save sampled active-row token IDs.
3. Repeat traced device sampling after replacing logits rows 1--31 immediately
   before `sample_split` with a trace-stable zero/neutral logits tensor, without
   changing active row 0, KV, positions, page table, or model trace.

Predictions:

- if eager row-0 logits are coherent and device output corrupts, and neutral
  inactive logits fix it, the bug is Sampling1D/feedback contamination from
  padded rows;
- if eager row-0 logits are already corrupt only when inactive block-table rows
  are zeros, repeat with rows 1--31 set to `-1`; this isolates runner table
  padding despite the SDPA `position=-1` early return;
- if eager is coherent and neutral inactive logits do not fix traced output,
  compare eager versus traced row-0 logits before sampling to isolate model
  trace state rather than the sampler.

A lighter first probe can reuse the direct generator A/B and vary only inactive
block-table rows (`0` versus `-1`) in eager mode. This costs no trace capture and
separates the only page-table difference between the current direct control and
the vLLM wire input.

### Smallest fix hypothesis

Thread the true active decode row count from
`TTModelInput.unpadded_batch_size` through `async_decode.submit_decode()` and
`generator_vllm.decode_forward()` into `MistralSmall24BGenerator.decode_forward`.
Keep the physical model/trace shape 32, but:

- distinguish `wire_batch=32` from `active_batch=1`;
- build sampling parameters with only the true active count;
- ensure inactive LM-head rows are neutral before Sampling1D, or extend the
  canonical split sampler to accept an active-row count/mask;
- format/read only active sampled tokens;
- retain `position=-1` for inactive cache/SDPA rows.

Do not shrink the model to batch 1 or disable trace: serving must retain
concurrency 32 and the production split-sampling path. A regression should run
the same exact prompt at configured batch 1 and configured batch 32 with one
live request and require identical greedy token IDs for at least 16 tokens,
then repeat with multiple active rows.

### Revised ranking

1. **Wire batch incorrectly treated as semantic active sampler batch -- high.**
   It is a directly visible contract mismatch and exactly distinguishes the
   clean max-num-seqs=1 server from the corrupt max-num-seqs=32 server.
2. **Inactive zero block-table rows perturb batch-32 paged decode -- medium.**
   Source says `position=-1` should skip, but it is the remaining input delta
   from the clean direct batch-32 control and is cheap to test.
3. **Batch-32 trace-only model state issue -- medium-low.** Distinguish by
   comparing row-0 logits before sampling; eager batch-32 is already clean.
4. **Rounded page-tail lookup -- refuted as sufficient.** Safe tail and
   `k_chunk_size=32` failed, while direct logical `-1` tables are coherent.

## Follow-up: first-decode BFP8 cache tile is updated twice

The 185-versus-192 alignment evidence identifies a stronger mechanism than the
wire-batch hypothesis above:

- a production traced request at prompt length 185 corrupts after the first
  plausible token;
- the same path at exactly 192 tokens is coherent;
- a direct host/eager decode that updates the first decode position once is
  coherent;
- the KV cache uses a BFP8 dtype and the paged update kernel rewrites a complete
  32-row cache tile for a one-row update.

At position 185, rows 160--184 of the same cache tile already contain prompt KV.
Re-reading and re-packing that tile changes those existing BFP8 values. At
position 192, the update starts a new 32-row tile, so no preceding prompt rows
share the rewritten tile. This exactly predicts the observed alignment cliff.

### Trace-capture semantics: one correction, same root cause

TT mesh trace capture is record-only on the fast-dispatch path. `record_begin()`
enables system-memory bypass mode, and `enqueue_mesh_workload()` records a trace
node and returns instead of dispatching the workload
(`tt_metal/distributed/fd_mesh_command_queue.cpp:342-355`, `:1155-1172`). Thus
the model call between `begin_trace_capture()` and `end_trace_capture()` does
**not** execute a second cache update.

Before the inactive-position change, however, `_capture_decode_traces()` still
did this:

1. copy the real first-decode token and position into trace state;
2. execute an eager model warmup against the caller-owned live KV cache;
3. record the model and sampler traces (no device execution here);
4. restore the same real first-decode state;
5. let the caller execute the first model-trace replay.

The live first-decode position was therefore updated twice, not three times:
once by warmup and once by first replay. The paged-update reader loads all `Wt`
tiles for the containing 32-row block, compute untilizes/re-tilizes the whole
block, and the writer writes all `Wt` tiles back. Only one row is replaced in
L1 (`cache_tile_offset_B = update_idx % 32 * Wbytes`), so every repeated update
re-packs the other 31 rows too
(`reader_update_cache_interleaved_start_id.cpp:132-146`,
`update_cache.cpp`, `writer_update_cache_interleaved_start_id.cpp:116-149`).

This mechanism also explains why direct eager decode is clean: it performs the
semantic update once. It supersedes the earlier wire-batch/sampler hypothesis
as the leading explanation, although the wire/active-batch mismatch remains a
separate contract issue worth retaining as a regression target.

### Safety of inactive warmup and capture

Using fixed-shape tokens with every current position set to `-1`, then copying
the real token/current-position/rotary-position state immediately before the
first replay, is the smallest source-supported fix:

- paged-update reader and writer explicitly interpret `uint32(-1)` as
  `skip_update`, do not read the page table/cache tile, and suppress the DRAM
  write (`reader_update_cache_interleaved_start_id.cpp:85-122,132-146` and
  `writer_update_cache_interleaved_start_id.cpp:76-106,132-149`);
- paged SDPA reader, compute, and writer each explicitly return for
  `cur_pos == UINT32_MAX`, documented there as the skipped-user sentinel
  (`reader_decode_all.cpp:150-155`, `sdpa_flash_decode.cpp:149-155`, and
  `writer_decode_all.cpp:129-136`);
- `plus_one(current_pos, skip_negative_entries=True)` preserves `-1`;
- `_copy_trace_state()` rebuilds rotary positions from the supplied current
  positions and overwrites token/current/rotary buffers with real state after
  capture, before replay;
- trace program identity depends on tensor/program shape and addresses, not the
  runtime values in the position tensor, so the recorded graph remains the
  production graph. The real page table can remain installed throughout.

The dummy warmup can produce semantically meaningless logits because every
attention row is skipped, and dummy sampler output may overwrite the trace
token. That output is harmless because the final real-state copy overwrites the
token and positions, and first model replay rewrites the captured logits buffer
before sampling replay consumes it. No standalone or scratch KV allocation is
needed, preserving vLLM cache ownership.

The only residual validation risk is whether an all-inactive SDPA invocation is
runtime-clean on this exact model configuration despite each kernel's explicit
skip branch. That is a smoke-test concern, not a source-level cache-safety
concern.

### Focused verification and durable regression

Run the exact 185-token production request after a fresh device reset with the
inactive warmup/capture state. Require a coherent greedy continuation and keep
192 tokens as the aligned control. To verify the mechanism directly, snapshot
one BFP8 KV tile containing positions 160--191 before capture setup and after
capture setup but before first replay: rows 160--184 must remain unchanged.
After first replay, only the semantic effect of writing row 185 should appear;
there must be no second setup-time re-pack.

Add a targeted traced-generator regression with caller-owned paged BFP8 KV,
physical batch 32, one active request, prompt length 185, and greedy split
sampling. Compare at least 16 tokens with the clean eager control. Also assert
that capture setup leaves the already-prefilled prefix tile unchanged, then run
the 192-token control. This catches the actual failure more reliably than a
trace-stat assertion alone.

### Revised ranking

1. **Live BFP8 cache tile updated by eager warmup and first replay -- very
   high.** It uniquely explains non-aligned failure, aligned-192 success, and
   clean single-update eager decode from the exact source sequence.
2. **Wire batch treated as semantic active sampler batch -- medium-low.** It is
   a real interface mismatch but no longer explains the 185/192 cliff as
   directly.
3. **Inactive block-table padding or rounded tail reads -- low/refuted as
   sufficient.** The focused safe-tail and chunk-size experiments did not fix
   production output.
