# Serving plugin change records

Pinned base: `tenstorrent/vllm-tt-plugin` commit
`042a94863ad27d88207bcf71abdf67122c6f4c27`; upstream vLLM `0.31.0`.
PC-01/02/03 are implemented and locally committed with the evidence below.
PC-04 was explicitly approved by the owner on 2026-11-16 and locally committed as
`cf2ce6d499040f9bce0c0a0e263329da7771f1e7`. PC-05 was explicitly approved on2026-11-16 and implemented in local commit `da1587235b84b13f2826cfd7e8b448b340322a54`.
Upstream vLLM source remains unchanged. Stage acceptance remains incomplete.

## PC-01: final-row sampling in chunked prefill

**Status: implemented; local plugin commit `1e35b653ae57f28914e60b3759ad07bc38f32a21`.**

1. **Requirement and failure.** The stage requires canonical on-device split sampling
   in `sample_on_device_mode=all`, without full-logits readback, including scheduler
   chunking with concurrent decode. The pinned plugin forces the entire prefill
   batch to host sampling whenever any row is an intermediate chunk. The exact
   current-source CPU reproducer is `probe_plugin_contract.py`; its JSON shows
   an otherwise device-eligible `[intermediate, final]` batch becomes host sampled
   and receives neither sampling parameters nor row-finality metadata.
   Run: `source doc/vllm_integration/env.sh` from this model directory, then
   `python doc/vllm_integration/probe_plugin_contract.py`.
2. **Owner.** Missing shared behavior. `docs/MODEL_EXECUTION.md` explicitly
   documents host sampling for intermediate chunks; this is not a violation of
   that current contract. `src/vllm_tt_plugin/model_runner.py:2665` forces the
   route, and `submit_prefill` omits the finality mask. The plugin alone knows
   which scheduled rows finish their logical prompts.
3. **Model-side alternative.** The adapter cannot derive finality from `tokens`,
   `start_pos`, or `prompt_lens`: these contain only scheduled history and chunk
   endpoints. The same arguments can represent a complete prompt or an
   intermediate chunk of a longer prompt. Sampling every row advances RNG and
   token feedback for intermediate rows; returning fake logits corrupts final
   rows. Reconstructing scheduler state from private plugin globals is an
   undocumented interface. Disabling chunking removes required coverage.
4. **Existing mechanisms checked.** `supports_chunked_prefill` permits chunking,
   but carries no per-row sampling mask. `supports_sample_on_device`,
   `max_device_top_k`, `sample_on_device_mode=all`, and seeded sampling `auto`
   do not override the intermediate-chunk route. `empty_slots` describes state
   placement, not completion. `prompt_lens` is the exclusive chunk end.
   Runtime class registration only selects the adapter. Decode reload commands,
   deferred grammar sampling, and host compatibility do not convey prefill
   finality. No existing documented option expresses this behavior.
5. **Exact proposed interface and effect on other models.** Add documented
   Boolean capability `supports_prefill_sampling_mask` (default `False`).
   Opted-in adapters receive keyword `sampling_mask: list[bool]` on device
   prefill calls, in submitted row order. `True` means this row reaches its
   logical input-history end (including generated history on resumed replay)
   and may sample; `False` means execute prefill but preserve
   sampler/RNG/token-feedback state and produce a suppressed placeholder ID.
   All-intermediate batches perform no sampling or logits transfer. Mixed
   batches sample only final rows with the generator's existing sampler.
   Other device sampling eligibility checks remain in force. Models without
   the capability retain the pinned behavior and call signature. This is
   usable by every chunk-capable adapter; no model/architecture branch.
6. **Authorization.** Explicit owner message delivered by the multigoal runner on
   2026-11-16: “PC-01 is approved exactly as proposed in plugin_changes.md: the new
   documented capability supports_prefill_sampling_mask (default False) and the
   prefill_forward(sampling_mask=...) keyword with the semantics described there.”
   The approval covers exactly item 5 above.
7. **Planned validation.** Plugin regression tests exercising `_prepare_model_inputs`
   and `submit_prefill`: all-intermediate, mixed final/intermediate, final-only,
   and non-opted-in legacy batches, in both front-packed and lane paths;
   preserve unrelated host-feature routing.
   Generator tests prove masked rows retain RNG/feedback, final rows match
   final-only seeded controls, and no logits D2H occurs in eligible chunks.
   Run the plugin host suite, then reduced-layer chunk/async/readback tests,
   allocation-tracked mixed-shape serving, and all final serving gates.
8. **Implementation candidate.** Capability validation, both routing paths,
   submission mask, lane extraction and documentation updated after approval.
   `pc01_before.log`: 12 expected failures / 103 passing controls on pinned code.
   Expanded 223-case matrix covers all-intermediate, mixed, final-only, resumed
   history, both layouts, legacy adapters, and unrelated host-feature routing.
   `pc01_host_suite.log`: 2530 passed, 16 skipped. `pc01_hooks_final.log`: hooks pass.
   `adapter_prefix_final.json` passes the representative-device mask/RNG/token
   feedback checks with allocation tracking and zero full-logits reads.
   `readiness_vllm/full_sync_serial_retry/lifecycle.json` passes all-layer
   concurrent chunk continuations and final tails: 713 device forwards, zero
   host forwards, four stable trace signatures, no retirements. This is affected
   synchronous serving validation, not completion of the final async/full suite.
   Explicit pre-commit: `pc01_pc02_commit_hooks.log`, pass. Local SHA above.

## PC-02: preserve continuing prefill state slots

**Status: implemented; local plugin commit `1e35b653ae57f28914e60b3759ad07bc38f32a21`. No new shared interface.**

1. **Requirement and failure.** A continuing GDN request must retain its recurrent
   state across chunks. `probe_plugin_contract.json` reproduces a live decoder
   in slot 0 and continuations in slots 2/1 being silently reassigned to 1/2.
2. **Owner.** Plugin defect. `docs/MODEL_LIFECYCLE.md` requires state ownership
   preservation and documents state moves via decode `slot_remap`; prefill
   receives no such remap. `_alloc_prefill_state_slots` discards a continuing
   request's existing slot before assigning by row.
3. **Model-side alternative.** Guessing request identity from page addresses is
   incorrect: pages are scheduler-owned and can be reclaimed/reallocated.
   The adapter is given neither request IDs nor old-to-new prefill mapping.
   Replaying the entire prefix is possible but duplicates work and conceals
   the runner's false ownership record; preserving claims repairs the actual
   contract for all incremental stateful models.
4. **Existing mechanisms.** `empty_slots` specifies only destination;
   `slot_remap` is decode-only; `note_state_slots_moved` reports committed
   decode moves. None transfers continuing prefill state.
5. **Effect on other models.** Preserve existing claimed slots for continuing
   requests and reserve them before assigning new prefills. New requests still
   prefer their row when free. Stateless models are unchanged semantically;
   all recurrent adapters benefit. No default capability or new API.
6. **Authorization.** Adds no shared interface; covered as a documented plugin
   defect repair, subject to regression and host-suite validation.
7. **Planned validation.** Extend `tests/test_block_model_runner.py` with reordered
   continuations while another request decodes, plus mixed new/continuing rows.
   Confirm failure on the pinned source, then pass on the repair; run full host
   suite and served mixed-shape correctness after PC-01 is resolved.
8. **Implementation candidate.** `model_runner.py` preserves existing slots;
   `MODEL_LIFECYCLE.md` documents continuation ownership. Three regressions fail
   before (`slot_regression_before.log`) and pass in the full host suite after
   (`plugin_host_fixed.log`:2306 passed,16 skipped). Formatting is recorded in
   `plugin_hooks_final.log`. Actual affected serving checks and local commit
   SHA were pending at the initial checkpoint. Affected serving now passes in
   `readiness_vllm/full_sync_serial_retry/lifecycle.json`: all nine greedy
   repetition/reordering/isolation comparisons exact with concurrent chunked
   prefill and allocator growth. The final combined host suite has 2530 passes,
   16 skips (`pc01_host_suite.log`); explicit pre-commit passes in
   `pc01_pc02_commit_hooks.log`. Local SHA above.

## PC-03: initialize vLLM before the tile-size collection import

**Status: implemented; local plugin commit `ea251ee12377d247fd4b80af67c2cf7b8ac4d7a1`. No shared interface.**

1. **Requirement/failure.** The documented `python -m pytest tests/tt` entrypoint
   must collect. `sampling_collection_final.log` shows collection fails because
   `test_tile_size.py` imports the plugin platform before vLLM initialization.
   Reproducer: `PYTHONPATH=tests/host-stubs python -m pytest tests/tt --collect-only -q --tt-server-url http://localhost:8000 --tt-model-name Qwen/Qwen3.6-27B --tt-max-num-seqs 32`.
2. **Owner.** Plugin test initialization defect; AGENTS.md's Testing notes
   explicitly warn that importing the platform while `vllm.platforms` is
   partially initialized raises `cannot import name current_platform`.
3. **Model alternative.** None: collection fails before model import/HTTP.
4. **Existing mechanisms.** The serving fixture intentionally neutralizes the
   host fixture to avoid exactly this cycle. Runtime registration, TT config
   and adapter declarations cannot fix Python test import ordering.
5. **Effect.** Explicit vLLM initialization before the test imports the plugin
   platform. All tile-size assertions remain unchanged; no skipped tests.
6. **Authorization.** No shared interface or semantics added.
7. **Validation plan.** Repeat the exact collection command, then plugin host
   suite. Live sampling remains a required later gate.
8. **Implementation candidate.** Added explicit vLLM initialization before
   plugin import. `sampling_collection_fixed.log` now collects142 tests with
   exit0 (previously one collection error). Full host suite passes as above;
   no tile-size assertion changed. Live smoke collected and ran: 2 passed,
   1 failed (independent sampling isolation defect), 1 skipped (server all-vocab
   logprob cap). That failure is not a collection fix acceptance or a waived
   sampling gate. Explicit pre-commit: `pc03_commit_hooks.log`, pass. Local SHA above.

Post-AutoDebug focused control: `slot_regression_final.log` passes all four
continuation cases, including accumulated history after a companion finishes.
The subsequent full-model lifecycle and local commits are recorded above.

## PC-04: original prompt boundary for device prefill sampling history

**Status: implemented; owner approved on 2026-11-16; local plugin commit
`cf2ce6d499040f9bce0c0a0e263329da7771f1e7`.**

1. **Requirement and failure.** Device presence/frequency penalties must count
   generated tokens, including the first token sampled after preemption replay;
   repetition uses prompt-or-output membership. Qwen currently lacks device
   penalties and must implement them in its canonical generator sampler.
   The present prefill call loses the original prompt/output boundary.
   `probe_prefill_history.py` executes the current plugin input builder and
   `submit_prefill` with a recording CPU model. A replay with prompt `[0]` and
   outputs `[1,2]` and a fresh prompt `[0,1,2]` produce identical adapter
   arguments, but correct generated-token counts differ. Result:
   `probe_prefill_history.json`, `missing_history_boundary_confirmed`.
   Reproduce from the tt-metal root:
   ```bash
   source models/autoports/qwen_qwen3_6_27b/doc/vllm_integration/env.sh
   PYTHONPATH="$VLLM_TT_PLUGIN_ROOT/tests/host-stubs:$VLLM_TT_PLUGIN_ROOT:$PYTHONPATH" \
     timeout 120 python models/autoports/qwen_qwen3_6_27b/doc/vllm_integration/probe_prefill_history.py
   ```
   The live penalty-companion probe separately demonstrates device→host→device
   routing and sampled divergence at token index2. That does not fully attribute
   the original shuffled smoke mismatch; `sampling_AUTODEBUG.md` preserves the
   distinction. This proposal supplies history metadata, not a sampler-routing fix.
2. **Owner.** Missing shared behavior: the plugin owns original request history
   and preemption, but its prefill interface cannot express this distinction.
   `model_runner.py::_prepare_model_inputs` builds combined history for replay;
   `submit_prefill` sends only chunk start/end, combined tokens, state slots and
   sampling parameters/mask. `MODEL_EXECUTION.md` documents chunk endpoints and
   the separate decode histories. This is not a claim that pinned `auto` routing
   violates its documented batch-wide semantics. Implementing penalties remains
   model-owned; transmitting the otherwise unknowable boundary is plugin-owned.
3. **Model-side alternative.** Fix canonical penalties, output-count masks,
   slot lifecycle/remap, reset histories and warmup in the generator. This is
   required and is not replaced by PC-04. That implementation alone cannot
   distinguish the two reproduced prefill calls. Retaining a slot's prior
   counts is wrong on release/reuse; release carries no request identity or
   finish/preemption reason. Guessing from physical KV pages or token content
   cannot recover identity. Waiting for decode histories misses the first
   resumed sample. Sampling on host or skipping penalty tests would hide the gap.
4. **Existing mechanisms checked.** `supports_device_penalties` describes
   existing sampler support; it does not authorize extra prefill keywords and
   legacy implementations do not all accept them. `prompt_lens` means scheduled
   chunk end, not original prompt size. `start_pos`, `empty_slots`, decode-only
   `slot_remap`, and PC-01 `sampling_mask` do not identify generated history.
   Decode `prompt_tokens`/`output_tokens` arrive too late for resumed prefill.
   Registration and TT sampling/seed policies do not supply history. Reusing
   the approved PC-01 key with new meaning would exceed its exact authorization.
5. **Exact proposed interface and effect on other models.** Add documented
   Boolean capability `supports_prefill_sampling_history`, default `False`.
   Present values must be Boolean; `True` requires `supports_sample_on_device=True`.
   On every device-sampled prefill, including neutral penalties and
   all-intermediate chunks, opted-in adapters receive
   `original_prompt_lens: list[int]`, one original request prompt token count
   per submitted row, in the same order as `tokens` and `sampling_mask`.
   This is the full original prompt count, unchanged across chunks/replay,
   and may exceed the current chunk end or tensor width for an intermediate
   chunk. For row `i`, visible prompt history is
   `tokens[i, :min(original_prompt_lens[i], prompt_lens[i])]`; visible committed
   output history is `tokens[i, original_prompt_lens[i]:prompt_lens[i]]` when
   the chunk end exceeds that boundary, otherwise empty. At a sampleable final
   row the complete committed history is visible. Intermediate chunks must not
   count an un-emitted token; PC-01 finality semantics remain unchanged.
   The plugin must derive counts from original request metadata and preserve
   row ordering in front-packed, lane-DP and per-rank standard-DP execution.
   Gather `num_prompt_tokens` using front-packed `req_indices` or lane
   `plan.input_rows`; do not pad to state-slot capacity or index the lengths
   using `empty_slots`. Host prefill and decode receive no new keyword.
   No new routing rule, env var, TT config or model-name branch is introduced.
   Legacy adapters receive no keyword and retain their existing behavior.
   This metadata can serve any incremental sampler requiring separate prompt
   and generated histories; Qwen is the first planned consumer.
6. **Authorization.** The owner explicitly approved exactly the default-False
   capability and `prefill_forward(original_prompt_lens=...)` keyword/semantics
   in item5 on 2026-11-16. This is a separate approval from PC-01.
   No new routing policy is part of this approval.
7. **Planned validation.** Before/after plugin regressions must exercise the
   indistinguishable replay/fresh pair through the actual builders/submission,
   repeated outputs, prompt chunks ending before/at/after the boundary,
   all-intermediate/mixed/final masks, reordered noncontiguous state slots,
   release/reuse, front-packed and lane-DP paths, standard-DP rank-local rows,
   capability absent/False, explicit host sampling, and invalid capability types.
   Legacy calls must have identical keywords. Run the full plugin host suite
   and explicit pre-commit. Then implement and validate canonical device
   penalties against fixed logits/count references (including signs,
   multiplicity, slot remap, masked count updates, first-prefill→decode reset,
   and replay versus fresh distinction), prepare counted sampling before traces,
   run allocation-tracked reduced tests, and rerun the targeted live companion
   and unchanged mixed-params test before smoke/full sampling and remaining
   full-model gates. Do not enable `supports_device_penalties` until that
   complete model contract passes. This metadata alone is not gate completion.
8. **Status.** `implemented`, local plugin commit
   `cf2ce6d499040f9bce0c0a0e263329da7771f1e7`. The183focused
   before/after regressions and2713pass/16skip full host suite are recorded in
   `pc04_validation.log`, including explicit pre-commit. Canonical tracked
   arithmetic/history/remap/mask proof is `penalties_probe_v4.json`; neutral
   stale-token/current-position/page-only/deferred-read regression is
   `adapter_penalties_neutral_retry.json`. Reduced async32-slot live artifacts
   in `readiness_vllm/pc04_reduced_server/` include exact penalty-companion
   agreement(258device/0host), mixed_params_batch pass(21device/0host), and
   lifecycle pass(712device/0host,5stable signatures,0retirements). All-layer
   full64/B32 synchronous affected checks now pass in
   `readiness_vllm/pc04_full_sync/`: penalty companion exact(258device/0host)
   and8penalty/isolation tests passed(905device/0host). Explicit final hooks:
   `pc04_hooks_before_commit.log`. No push. The stage remains incomplete.

## PC-05: opt-in request-stable sampler ownership

**Status: implemented; local plugin commit `da1587235b84b13f2826cfd7e8b448b340322a54`. Final acceptance is complete; P2 remains critical and unwaived. Independent stage review returns `more-work-needed`.**

1. **Requirement and reproduced failure.** The vLLM-integration skill requires
   a seeded request to retain its sampler throughout its lifetime. After PC-04
   and canonical device penalties, a late penalty companion is fully device
   sampled and reproduces. A genuinely host-routed companion still changes the
   active request's sampler. On the same reduced representative async server,
   `readiness_vllm/pc04_reduced_server/topk_route.json` records a seeded top-k20
   request, followed after its first token by a short top-k64 companion. It
   changes device→host→device and differs from its solo output at token index4;
   greedy control remains identical. The audit records242device/16host forwards.
   The preceding penalty control has258device/0host forwards and exact seeded
   and greedy agreement; unchanged mixed_params_batch passes. Reproducer:
   ```bash
   source models/autoports/qwen_qwen3_6_27b/doc/vllm_integration/env.sh
   timeout 600 python models/autoports/qwen_qwen3_6_27b/tests/probe_vllm_sampling_route.py \
     --companion-feature top_k --server-log <running-server.log> \
     --output <readiness_vllm/output.json>
   ```
   The all-layer64/B32 synchronous confirmation is now
   `readiness_vllm/pc04_full_sync/topk_route.json`: seeded first difference at
   token index2, greedy exact,242device/16host forwards. It proves this is also
   present on the complete model, not only the reduced async diagnostic.
   This is request admission/routing evidence, not a reduced-model quality
   result. `topk_route_audit.json` saying passed means the probe completed without
   server errors; it does not mean sampler ownership passed.
2. **Owner.** Missing shared behavior, not a violation of the pinned `auto`
   policy: README.md documents its batch-wide, per-forward route decision.
   `model_runner.py::check_perform_device_sampling` makes that decision and
   includes batch-global checks; the model receives a single sampling mode for
   a whole submission. The plugin owns request identity, admission, replay and
   scheduler/async lifetime. Those are necessary to keep one request's route
   from changing when another request arrives. Host-only processors and other
   documented host compatibility features remain supported.
3. **Model alternative.** Canonical penalties and original prompt history have
   now been fixed in the generator; they no longer trigger this failure. A
   wider device top-k would only remove that one trigger, not provide lifetime
   routing for the plugin's host-only controls or arbitrary Python logits
   processors. The model cannot infer request identities from padded rows,
   recover intrinsic controls omitted by a batch-wide host decision, or split
   scheduler ownership from a single token/logit output contract. Reimplementing
   the plugin's host sampling inside the adapter would violate canonical device
   sampling and create a separate sampler. Rejecting valid host-only requests,
   reducing max-num-seqs to1, or moving all seeded requests to host does not meet
   the serving contract. This proposal preserves real host feature boundaries;
   it is not an exemption for missing device penalties.
4. **Existing mechanisms checked.** `seeded_sampling_policy=auto` has the
   reproduced batch-wide behavior. `host` permanently routes seeded requests
   to host and loses the required performance path. `sample_on_device_mode`
   controls phases, not lifetime ownership. `max_device_top_k`,
   `supports_device_penalties`, grammar and logprob capabilities report actual
   feature support; they do not isolate otherwise supported device requests.
   `sampling_rows` alone cannot remove batch-global penalties/processors/logprob
   predicates. Decode reload flags and async draining preserve state but do not
   choose independent samplers. Registration, PC-01 and PC-04 convey no routing
   policy. Lane/rank partitioning is not an alternative hardware configuration
   for this one TP4 replica.
5. **Exact proposed interface and other-model effect.** Add one opt-in value
   `"request"` to the existing `tt.seeded_sampling_policy` configuration key.
   Keep the default `"auto"` and existing `"host"` behavior unchanged. Under
   `"request"`, classify each request's intrinsic sampler eligibility before
   its first emitted token, accounting for every phase and declared model
   capability. Preserve that host/device ownership across prefill chunks,
   admissions/departures, slot moves, preemption/replay and async completion;
   discard ownership only on finish/abort. Phase-dependent controls such as
   min_tokens are classified conservatively for the whole lifetime. Unsupported
   request features retain the host sampler. Scheduled model submissions must
   contain only requests with the same sampler owner; use fair homogeneous
   cohorts through the existing forward/output and reload contracts. Retain
   the current resident/async single-forward fast path for homogeneous device
   traffic. No new model capability, environment variable, output union,
   model-name branch, or upstream-vLLM patch is proposed. All models that opt
   into this value may use the shared policy; non-opted-in models are unchanged.
   The exact internal cohort implementation must meet fairness and state gates;
   changing other default policies remains outside this authorization request.
6. **Authorization.** The owner explicitly approved PC-05 in the stage9 runner
   message on 2026-11-16: “PC-05 is approved exactly as proposed in
   plugin_changes.md (opt-in request-stable sampler ownership, default off,
   other models unaffected).” This authorizes
   `tt.seeded_sampling_policy="request"` and item5's lifetime ownership and
   homogeneous-cohort semantics. Defaults remain unchanged.
   The proposed final Qwen server would explicitly use this value instead of
   `auto`; it must retain device-sampled headline performance.
7. **Planned validation.** Add plugin regressions that fail under the current
   batch route and pass under the new opt-in policy: initial and late admission
   of top-k64, min_p, logprob and filter companions; both arrival orders;
   departure/rearrival, abort/reuse, reordering and slot compaction, chunked
   prefill and replay; front-packed, lane and standard-DP rank-local execution;
   synchronous and async pending-result transitions. Check persistent owner,
   exact history/RNG attribution, fair bounded progress, and legacy auto/host
   behavior. Run full host suite and explicit pre-commit. Repeat representative
   allocation-tracked serving, fixed-logit/greedy and seeded isolation controls,
   same-server mixed-shape trace reuse and automatic page growth. Then rerun
   all-layer sampling/qualitative/P2/P3/P5 and primary/CI/prefill benchmarks;
   compare homogeneous device performance to the unchanged-policy baseline and
   report mixed-cohort scheduling cost separately. No profiler in serving.
8. **Status.** `implemented`, local plugin commit `da1587235b84b13f2826cfd7e8b448b340322a54`.
   Host suite2784pass/16skip (`pc05_host_suite_final.log`);71new focused
   regressions pass (`pc05_new_tests.log`); wrapper-bypass witness16expected
   failures (`pc05_cohorts_before.log`). Explicit pre-commit passes
   (`pc05_precommit.log`). Representative real layers0/3, P300x2/B32,
   fullcontext262144/cache8224, allocation tracking enabled: all15 host-feature
   admission cases pass exact active/companion seeded and greedy equality;
   17-request mixed-shape lifecycle passes with five stable trace signatures,
   zero retirements,29consumed new pages,777device/0host steps. Artifacts:
   `readiness_vllm/pc05_reduced_async/`. No device process remains after its
   drained shutdown; no reset used. Final full-model acceptance completed. The round1 P2 top-k40 blocker was subsequently fixed in the canonical tt-metal sampler, without another plugin change: current P2 passes all15 checks and the32-user reasoning ratio is0.95027. Current full sampling67pass75skip, lifecycle, fixed logits and qualitative checks pass in `readiness_vllm/p2_final_async/`. Original P3/P5 and once-only context evidence remain valid; see the stage README for exact scopes. The current unsupported top-k companion uses128 because64 is now supported. Historical top-k64 negative/positive controls above describe the pre-widening sampler.
