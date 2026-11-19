# AutoDebug: continuing prefill loses state ownership; chunked device sampling is blocked by the existing host-output contract

Inspection date: 2026-11-16. **Inspection only:** no model execution, device access, test execution, builds, source edits, or AutoDebug relaunch. This investigation writes only this report. Three independent xhigh source investigations checked state ownership, sampling flow, and interface alternatives; important claims were rechecked against source.

**Findings:**

1. **Confirmed defect in the pinned commit:** `_alloc_prefill_state_slots` can change a continuing request's recorded slot without transferring recurrent/conv state. A later decode derives its remap from the already incorrect ownership map. Preserve existing live claims before allocating new prefill slots; this needs no new interface. An external working-tree patch implementing this preservation appeared during post-draft review; this investigation did not author or execute it.
2. **Confirmed integration blocker, with ownership distinguished:** any intermediate prefill row forces the entire prefill submission onto the host-output route, and the adapter receives neither sampling parameters nor row finality. This matches the pinned plugin documentation. The current Qwen adapter rejects that call by default; its explicit compatibility mode transfers vocabulary logits. The strict device-sampling goal therefore cannot pass a prefill submission containing an intermediate row; merely enabling chunking does not establish that such a submission occurred.
3. **A new capability is not proved uniquely necessary merely to device-sample final rows.** A generic runner could partition final and intermediate prefills using existing calls. Eliminating intermediate logits transfers as well exposes a missing documented execution/return semantic. The proposed opt-in `supports_prefill_sampling_mask` plus `prefill_forward(sampling_mask=...)` is a justified, small shared extension under that strict requirement and requires approval of those exact items. This investigation is not such approval.

There is no supplied hardware failure trace proving observed text corruption. The state failure below is a source-derived reachable sequence; adapter rejection is a direct control-flow consequence. In the current default Qwen path, sampling rejection can stop execution before latent continuation-state corruption is reached.

## Provenance and notation

| Name | Inspected location/revision |
|---|---|
| `P` | `/home/ttuser/qwen36-27b/vllm-tt-plugin`, HEAD **`042a94863ad27d88207bcf71abdf67122c6f4c27`** |
| `Q` | `/home/ttuser/qwen36-27b/tt-metal/models/autoports/qwen_qwen3_6_27b`, tt-metal HEAD **`818f5e81574b25c1a0784051172e7f91dc96ae52`** |
| Runner | [model_runner.py](/home/ttuser/qwen36-27b/vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py) |
| Generator | [generator.py](/home/ttuser/qwen36-27b/tt-metal/models/autoports/qwen_qwen3_6_27b/tt/generator.py) |
| Draft adapter | [generator_vllm.py](/home/ttuser/qwen36-27b/tt-metal/models/autoports/qwen_qwen3_6_27b/tt/generator_vllm.py), untracked at inspection |

`P/...:line` and `Q/...:line` below refer to these roots. **All plugin source/document line references describe the pinned commit**, checked with `git show` when the working tree subsequently changed. All three requested documents were read: `P/docs/MODEL_EXECUTION.md`, `MODEL_LIFECYCLE.md`, and `DECODE_RELOAD_CONTRACT.md`. Also checked `MODEL_INTERFACE.md`, relevant capability/scheduling documentation, plugin `AGENTS.md`, and the installed bringup workflow's plugin-change rules.

Inspected SHA256 values:

```text
Q/tt/generator.py
43c6d811d3f5d1008a3cd9d6ca92552f130188bed17d1c395bdf7c0a5a505566
Q/tt/generator_vllm.py
42feece228400208a7d112076b42e57fed1ac367bf42c351ca9fa85ca78184c6
P/src/vllm_tt_plugin/model_runner.py
8ba39fb80b37ce88a7ca8546c80d5cf963f56d06fb5010c3fa21a211e2bb305a
P/src/vllm_tt_plugin/input_batch.py
2f10eb9efbed4838edbe872cf1cfeca81b62d7f1e2a702292f4bb2ceeb2d368f
```

The plugin was initially clean. During inspection, external changes appeared in its allocator, lifecycle documentation, and tests; this investigation did not make or execute them. The allocator diff now reserves all live claims and reuses an existing request's slot, matching the core repair below. The added tests cover reordered and mixed new/continuing allocation. This is a source comparison, not a claim that the local patch passed tests or serving. The runner hash above is the pinned/original source; the later working-tree runner hash was `0b643b63e30da3035c9aa1363d61e3aab4090cfe9a42bd1137709699c7e4a56f`. The unchanged `input_batch.py` still matched the pinned commit at that check. Findings and baseline-test gaps refer to the pinned version, not the evolving repair.

The untracked `Q/doc/vllm_integration/probe_plugin_contract.{py,json}` was read as pre-existing evidence, not executed. Its local reassignment fixture agrees with source but is not a hardware or end-to-end scheduler reproduction. New working notes/proposals appearing there are not treated as human authorization.

## 1. Continuing prefill can silently lose its recurrent history

### Direct source facts

- `_req_state_slot` records where request-owned device state resides (`P/src/vllm_tt_plugin/model_runner.py:451`). The allocator excludes **every submitted prefilling request** from `held`, even one with an existing live claim (`:1311`). It then prefers the current tensor row or first free slot and overwrites `_req_state_slot[req_id]` (`:1318`, `:1334`). There is no old-slot check or physical transfer.
- Cached chunk continuations explicitly take the prefill branch (`:2417`). At `:2768`, prefill gets only `prefill_empty_slots`; the decode branch alone constructs `slot_remap`.
- `submit_prefill` supplies `empty_slots`, absolute chunk starts/ends, and pages (`:4094`, `:4127`). It supplies no old slot or prefill remap. The draft adapter forwards these arguments (`Q/tt/generator_vllm.py:146`).
- Qwen maps page-table rows into destination slots (`Q/tt/generator.py:643`), clears state only for `start==0` (`:651`), and runs only `tokens[start:end]` there (`:666`). Page-table updates do not move recurrent history.
- `BoundedPrefill._import_state` copies conv/recurrent state from the supplied physical slot (`Q/tt/bounded_prefill.py:129`, called at `:402`); `_publish_state` writes back there (`:138`, `:439`). GDN consumes the imported state, including `initial_state=state.recurrent` (`Q/tt/multichip_decoder.py:1333`). No incorrect TTNN copy/gather kernel is needed: the caller selects the wrong history slot before these operations run.

### Concrete failing and passing state ledger

Assume Qwen host compatibility is enabled, or another conforming stateful adapter executes these calls; default Qwen stops at the first intermediate call. Use configured capacity 2, an existing decoder A in slot0, and long prompt B. During B's prefill, A is temporarily unscheduled but remains live.

| Event | Host claims after event | Physical recurrent/conv state |
|---|---|---|
| A decodes | A→0 | slot0=A, slot1=unused |
| B first chunk `[0:512]`, only prefill row0 | A→0, B→1 | A holds0, so B-prefix is written into1 |
| A is cancelled before B's next chunk | B→1 | release clears0; B-prefix remains in1 |
| B continuation `[512:1024]`, again row0 | allocator changes B→0 | **No gather:** slot0 remains cleared, slot1 remains B-prefix |
| Qwen executes B's continuation | B→0 | continuation imports zero history from0 and publishes its resulting state into0 |
| B later decodes in row0 | B→0, identity remap | old B-prefix in1 is not recovered |

Cancellation is sufficient; A need not make decode progress during B's prefill. `_update_states` releases finished requests through their current slots before removing slot claims (`P/.../model_runner.py:988`, `:999`, `:1022`); Qwen release zeroes conv/recurrent state (`Q/tt/generator.py:422`). Temporarily unscheduled requests retain their claims (`P/.../model_runner.py:1265`). B remains a prompt continuation because its computed count is below its prompt length (`:2396`).

**Passing contrast:** keep A alive. `held={0}` keeps B in1 on every chunk. When B later decodes alone in row0, the decode permutation is `[1,0]`; Qwen gathers B to0 and A to1. This explains why fresh arrivals and stable companion populations can work while completion/cancellation or row reorder exposes corruption.

The same defect occurs with reordered continuations: the pre-existing probe starts with `{decode:0, continue_a:2, continue_b:1}` and assigns submitted `[continue_a,continue_b]` to `[1,2]`. That fixture demonstrates the local calculation. The cancellation ledger establishes a scheduler-compatible sequence without assuming an arbitrary prompt-row permutation.

### Why downstream lifecycle handling cannot repair it

`_decode_state_slot_remap` reads the overwritten map (`P/.../model_runner.py:1351`), so it no longer knows the prior slot. Its whole-capacity permutation and accepted-submission commit otherwise provide the appropriate mechanism (`:1373`, `:1396`; `P/.../async_decode.py:1479`). Qwen applies physical remaps in decode (`Q/tt/generator.py:813`, `:227`). `note_state_slots_moved` is notification after a gather, not a prefill transfer hook (`P/docs/MODEL_LIFECYCLE.md:52`). Reloading tokens/positions/pages also does not remap recurrent state (`P/docs/DECODE_RELOAD_CONTRACT.md:46`).

Do not overstate the documentation: `MODEL_LIFECYCLE.md:90` explicitly warns that row and slot can change between chunks for stage-input-derived work. There is no universal promise that slots never change. The defect is the missing continuity mechanism: the API requests computation of `tokens[start:end]` (`MODEL_EXECUTION.md:51`), identifies prefill destinations and live-state preservation (`MODEL_LIFECYCLE.md:50`), and represents continuing ownership transfers through decode `slot_remap` (`DECODE_RELOAD_CONTRACT.md:79`, `:90`). A destination-only prefill cannot incrementally consume state the runner silently left elsewhere.

### Minimum plugin repair

Keep the change generic and local to `_alloc_prefill_state_slots`:

1. Reserve **all existing live slot claims before visiting submitted rows**, including current continuations.
2. Reuse a submitted request's current valid claim when it has one.
3. Allocate row-preferred/free slots only for requests without a claim.
4. Retain release/preemption semantics: explicit preemption removes the claim, so resumed rebuild can use a fresh slot. Preserve off-batch holders and the accepted-decode permutation protocol.

Update allocator comments and lifecycle documentation. Planning/validating all assignments before committing would be additional hardening for invalid ownership/capacity, not necessary to repair the demonstrated normal-state loss.

Adding only an in-loop reuse check is insufficient: if B owns0 and rows are `[NEW,B]`, NEW must not steal0 before B is visited. No capability, environment read, configuration key, new argument, or move hook is necessary. This repairs existing ownership implementation rather than introducing a new execution mode.

### A model-only state workaround exists

The runner supplies token history from0 through the maximum scheduled end (`P/.../model_runner.py:2471`). Qwen could rebuild the prefix in each assigned destination before processing the requested interval, or replay `[0:end]` and return the boundary output. This is source-feasible; claiming that *no* model-only state workaround exists would be false.

It duplicates prefix computation, can approach quadratic work across scheduler chunks, and needs validation of accepted recurrent precision/publication boundaries. Rebuilding only GDN state is nontrivial because deeper layers' prefix activations depend on preceding layers. Guessing identity from KV page addresses or token-prefix equality is weaker: neither is a request identity. Model replay would compensate for lost ownership information for that implementation; retaining existing claims directly repairs the runner's false state-location record for every incremental stateful adapter.

## 2. Intermediate chunks select host output; Qwen raises or transfers logits

### Actual control/data chain

1. `_prepare_model_inputs` calculates `end = start + scheduled_count`, then `intermediate_prefill_mask = end < input_batch.num_tokens` (`P/.../model_runner.py:2454`). The comparison includes generated history on resumed replay; it is not just original prompt length.
2. Normal device eligibility is decided first. Any intermediate entry then unconditionally sets `perform_device_sampling=False` (`:2665`), even in `sample_on_device_mode="all"` and even with otherwise eligible greedy rows. Lane-DP duplicates this (`P/.../input_batch.py:1591`).
3. The mask remains internal (`P/.../model_input.py:202`; runner `:2806`). `submit_prefill` does not forward it and attaches `sampling_params` only when device sampling was selected (`runner:4112`).
4. Qwen sees `sampling_params=None` and raises `ValueError("Host sampling requires explicit QWEN_VLLM_HOST_COMPATIBILITY=1")` before generator execution (`Q/tt/generator_vllm.py:124`). This applies to an all-intermediate B1 first chunk as well as mixed intermediate/final rows.
5. With that **model-owned compatibility mode explicitly enabled**, Qwen executes the host-output branch. It calls `_read_logits` (`Q/tt/generator.py:679`), which increments `full_logits_readbacks` and composes vocabulary shards onto the host (`:278`). It does not use the token-only path.
6. For a mixed intermediate/final submission, the runner samples host logits. Seeded intermediate rows get cloned host generators (`P/.../model_runner.py:1416`) and their outputs are suppressed. `_build_chunked_prefill_output` emits `[]` for intermediate rows and applies/enqueues only finals (`:3899`). Suppression does not undo earlier logits transfers.

**Important correction:** for an **all-intermediate** submission, `_get_output_tokens` returns zero placeholders without calling the host sampler (`runner:4311`); lane extraction does likewise (`input_batch.py:1683`). The adapter has already returned, so current compatibility Qwen has already transferred logits. “Every intermediate chunk advances host RNG” is false; “every such call takes the host-output route, and current compatibility Qwen reads logits” is supported.

This is deliberate pinned-plugin behavior: `MODEL_EXECUTION.md:57` says prefill returns host logits or sampled IDs, supplies no separate prefill readback hook, and selects host sampling for intermediate chunks to avoid device-sampler advancement. `:215` requires ordinary adapters to return logits when host sampling is selected. `MODEL_CAPABILITIES.md:84` says device support permits, rather than forces, the route. Missing prefill device finality is a **shared feature gap relative to this goal**, not proof that pinned host fallback violates its own contract. Qwen's default refusal of a legitimate host-output call is a separate adapter conformance limitation.

### Concurrent requests do not imply a mixed-phase submission

`TTScheduler` permits all-prefill or all-decode batches (`P/.../scheduler.py:329`). `_schedule_prefill_only` hides decoders (`:957`); `_schedule_decode_only` hides partial prefills and waiting queues (`:983`). Default scheduling interleaves them (`:893`). A workload can have active decoders while new prompts/chunks progress; simultaneous prefill+decode in one forward is not supported.

For example, D can own slot0 while one prefill step submits A `[512:1024]` of 1536 tokens into1 and final F `[0:64]` into2. Here `intermediate_prefill_mask=[True,False]`; the proposed `sampling_mask` would be its inverse, `[False,True]`. Default Qwen rejects; compatibility Qwen reads both rows' logits, samples on host, emits A:`[]` and F:`[token]`. An interleaved decode-only step can advance D/F while A stays off-batch. A later final chunk can be device-sampled again. Prefill fallback does **not** automatically force adjacent decodes to host.

## 3. Existing mechanisms, alternatives, and shared-interface boundary

### Existing mechanisms audited

| Mechanism | Why it does not supply the missing prefill behavior |
|---|---|
| `supports_chunked_prefill` | Admits resumable scheduler chunking; no row finality and no override of the host route. |
| `supports_sample_on_device`, `sample_on_device_mode="all"`, `max_device_top_k`, seeded policy `auto` | Permit eligible device sampling; intermediate override runs afterward. |
| `prompt_lens`, `start_pos`, token width | Scheduled intervals/history through largest scheduled end, not each request's complete logical history length. |
| `empty_slots`, decode `slot_remap`, movement notification | State location/movement, not permission to sample a prompt row. |
| `TTSamplingParams` | Existing fields are sampling controls, without a mask (`P/.../model_input.py:68`); inventing a sentinel changes semantics. |
| `sample_rows`, compact/selective host readback | Current lane-DP **decode host-output** facilities; still return vocabulary logits and do not provide prefill finality (`DECODE_RELOAD_CONTRACT.md:191`, `:208`). Qwen declares neither capability. |
| Deferred device grammar sampling | Decode-specific retained-logits hook; structured prefill remains host-sampled (`MODEL_EXECUTION.md:223`). No ordinary prefill equivalent. |
| Stage inputs or runtime registration | Stage inputs are request payloads, not scheduler completion metadata; registration selects a class, not an execution protocol. |
| Model-owned host-compatibility switch | Expresses the existing host route, failing the strict device/no-logits requirement. No plugin environment read was added. |

Finality cannot be inferred reliably: the same submitted token prefixes, starts, scheduled ends, destination slots and page mapping can correspond to different uncomputed tails. In a two-row host-routed batch, hidden totals can produce `[intermediate,final]` or `[final,intermediate]` with identical submitted intervals and prefix contents. Token values and padding are not general completion delimiters. Supplying true totals or a mask conveys new information.

### Alternatives adjudicated

| Alternative | What it can meet | Limitation |
|---|---|---|
| Existing host compatibility | Existing-contract functional host fallback | Reads full logits; mixed finals use host sampling. Label compatibility evidence separately. |
| Runner partitions final/intermediate rows into separate existing prefill calls | **Final rows** can remain on canonical device sampling without a new model argument | Intermediate calls still require host logits under the documented contract; requires careful metadata slicing/scattering, stable slots and batching review. |
| Partition and omit intermediate output/readback | Runtime-feasible in the current all-intermediate extraction branch | `tt_out` is ignored there, but no existing argument tells Qwen a no-params call is guaranteed unused. Treating `None`/placeholders as supported requires resolving/documenting a new or broadened return semantic. Permissive code alone is not a documented interface. |
| Sample every chunk and discard its token | Might appear to work for special greedy/reseeded models | Violates generic no-intermediate-sampler-advance behavior, changes mutable RNG/feedback/penalty state, and mismatches host-output interpretation unless runner routing changes. |
| Fabricate logits encoding a device-selected token | No valid general solution | Not actual logits for host probabilities, logprobs, constraints or stochastic sampling; introduces a different strategy. |
| Disable chunking, force B1 or serialize all admission | Avoids selected triggers | Removes required concurrent/chunked coverage. |

Thus **“a new Boolean capability is logically unavoidable” is too strong**. Partitioning is a real alternative for the weaker final-row-only requirement. For the full requirement—eligible chunked calls, no full-vocabulary readback including intermediates, documented adapter contracts—some supported finality/state-only execution or return semantic is missing. A single opt-in mask is a smaller proposal than a deferred-prefill hook, output type, sampler or scheduling mode.

### Concrete minimal proposal, not an implementation

The local PC-01 proposal names:

- Boolean `model_capabilities["supports_prefill_sampling_mask"]`, default `False`.
- Optional `prefill_forward(sampling_mask=...)`, a Boolean list in **submitted request-row order**, only on opted-in device-prefill calls. `True` may sample; `False` computes context while preserving request sampler/RNG/token-feedback state and returns a suppressed placeholder in the existing `[B,1]` token result.
- For **otherwise device-eligible opted-in calls**, all-False masks fully execute/publish prefill state, perform no sampling or logits/token readback, and return host placeholders. Missing mask retains existing all-row sampled-prefill semantics. Non-opted-in adapters retain their signature and host fallback. Other eligibility checks remain in force; this proposal does not supply a separate state-only route for calls selecting host output because of unsupported controls.

Implementation scope, if that exact proposal is authorized:

1. Condition both intermediate-host overrides on opt-in (`runner:2665`, `input_batch.py:1591`), and forward the inverse internal mask from `submit_prefill`. No Qwen-name branch or silent new meaning for `supports_chunked_prefill`.
2. Account for lane extraction's assertion rejecting device-sampled intermediate rows (`input_batch.py:1687`), while retaining suppression and deferred application. A front-packed-only patch must not claim complete shared support.
3. Validate/document capability and keyword, retaining absent-key compatibility and fallback for unsupported controls. This design needs no new environment variable, JSON option, output type, readback hook or sampler.
4. In Qwen, map row booleans through `empty_slots`. Reuse persistent-logit staging, canonical sample trace and token-only readback (`generator.py:405`, `:674`, `:683`). Prefill currently marks **all submitted slots** sample-active (`:690`); it must select only finals. Configure/reset mutable sampler state only as the mask semantics permit.
5. Existing `_sample_graph` restores inactive tokens and seed tensors for B>1 (`:312`), and `_reset_rng(..., preserve_inactive=True)` exists (`:382`). These are useful building blocks, not proof of the proposed path. All-false must bypass sampling, particularly because B1 does not restore inactive state. Common sampling reseeds immediately before drawing (`models/common/sampling/tt_sampling.py:1460`); actual masked RNG/feedback preservation still needs hardware validation.
6. Fully consume bounded-prefill iterators even when suppressing logits. `_publish_state` is **after the final yield** (`bounded_prefill.py:436`); breaking at a discarded yield leaves state unpublished. Preserve prefill→decode invalidation (`P/.../async_decode.py:409`).

The slot repair is independent and needs no such capability. Removing the host override alone is insufficient: current Qwen would sample every submitted slot and other adapters could advance intermediate RNG state.

### Exact authorization requirement

Plugin [AGENTS.md:332](/home/ttuser/qwen36-27b/vllm-tt-plugin/AGENTS.md:332) says: **“A human must explicitly request or approve the specific model capability”** before implementation. It also requires checking existing mechanisms and forbids disguising a bug as a support boundary (`:340`, `:348`).

The installed workflow's [plugin-changes.md:16](/home/ttuser/qwen36-27b/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.24/skills/vllm-integration/references/plugin-changes.md:16) includes new hooks/marker methods, changed meanings, and default changes in shared additions; `:26` says: **“Make such a change only after a human authorizes that exact item.”** A general bringup/fix request is insufficient. This workflow restriction is broader than capability/env/config wording in plugin `AGENTS.md`.

A reviewable approval item is:

> Approve adding `supports_prefill_sampling_mask` (Boolean, default False) and `prefill_forward(sampling_mask=...)` for otherwise device-eligible opted-in prefill calls. The mask follows submitted-row order: False entries execute context without advancing request sampling/RNG/token feedback; all-False masks transfer no logits. Non-opted-in adapters retain existing behavior.

No approval of these items is present in this investigation request. `Q/doc/vllm_integration/plugin_changes.md` records PC-01 as proposed/pending; a proposal is not authorization. If another design changes unopted defaults or permits omitted intermediate returns, that exact alternative also needs scope review under the workflow. This report does not ask for approval or block completion of the requested inspection.

## 4. Completion boundaries and diagnosis scope

The ownership mismatch is not delayed device visibility: it selects the wrong physical history before prefill operations are enqueued. Decode correctly commits proposed ownership after accepted `decode_forward` (`P/.../async_decode.py:1479`); finalization waits for read events before host conversion (`:1653`). Qwen queues its async read before an output buffer can be overwritten (`Q/tt/generator_vllm.py:172`). No evidence establishes a read-event or TTNN kernel defect.

Prefill invalidates resident decode chaining (`P/.../async_decode.py:409`). Transition drains/reloads are covered by `DECODE_RELOAD_CONTRACT.md:106` and `:141`. They do not recreate missing recurrent history or add finality. Under async scheduling, final-prefill tokens are deferred for state application (`runner:3953`), so a future fix must preserve forced-reset/finished-request handling.

Current Qwen declares `supports_async_decode=False` (`generator_vllm.py:37`); requested async scheduling is disabled without it (`P/.../platform.py:2440`). Concurrent-request coverage must not be reported as proof of async overlap or steady resident-input optimization. Those remain separate adapter/evidence work.

| Case | Source-supported outcome at inspected revisions |
|---|---|
| Otherwise eligible, unchunked/final-only prefill | Canonical device sampler reachable; no runtime pass claimed. |
| B1 intermediate chunk | Host-output route; default adapter raises. Compatibility transfers logits, then runner skips sampling. |
| Mixed intermediate/final prefill | Host-output route; default adapter raises. Compatibility transfers logits, host-samples and emits only finals. |
| Continuing prefill retains same destination | Does not itself trigger lost state. |
| Continuing prefill changes destination without rebuild | Imports wrong history; latent behind default sampling rejection until that path executes. |
| Temporarily omitted live decoder | Must retain state; held-slot logic handles fresh arrivals but not the continuation's own old claim. |
| Explicitly preempted request resumes via rebuilt history | Old state intentionally released; a fresh destination is valid. Distinct from nonzero-start continuing chunks. |
| Lane-DP | Stable slots avoid this front-packed allocator defect; independent intermediate-host override remains. Not evidence that Qwen supports lane-DP. |

## 5. Validation needed outside this inspection

`P/tests/test_state_slots.py:117` covers fresh arrivals and decode moves. Its re-prefill check at `:147` already has row0/slot0 and cannot catch changed destinations. `_prefill` (`:69`) models bookkeeping only; physical-history ledger coverage (`:228`) covers decode gathers, not incremental prefill reads. Chunk tests check classification (`tests/test_model_runner.py:200`) and host-RNG cloning/all-intermediate sampler suppression (`tests/test_lane_model_runner.py:299`, `:323`, `:343`), not masked device prefill or avoided logits transfers.

Required regressions and controls:

1. **Slot repair, host-only:** capacity-two cancellation ledger; reordered continuations; `[NEW,CONTINUING]` where the continuation owns NEW's preferred row; continuation after a decode gather; unchanged fresh/resumed allocation; off-batch preservation. Track accumulated history markers, not only labels/indices. Atomic failure for invalid ownership/capacity is additional hardening coverage if that behavior is changed.
2. **Any sampling extension, host-only:** all-intermediate, mixed, final-only, non-opted-in legacy batches; row mask/slot correspondence; front-packed and lane paths; unchanged routing for penalties, grammar, top-k, logprobs and host-only controls; suppression/deferred application after cancellation/reset.
3. **Model/device follow-up:** reduced representative layers first. Compare resumed recurrent state/output against stable-slot controls, including dropped-history negatives. Count actual logits transfers and sampler executions; prove only finals advance request seed/feedback, all-intermediate reads nothing, nonzero destination slots map correctly, and finals agree with final-only canonical controls. Verify no recapture and preserve iterator publication. Then exercise scheduler-budget-crossing prompts with concurrent decoders and required full-model gates.

Exact text divergence, production frequency, performance cost and masked sampling hardware behavior are unmeasured. No test-pass, serving-success, trace-correctness or performance claim is inferred from this inspection or pre-existing probe output.

## Post-draft adversarial review

Rechecked allocator, prefill builder/submission, state import/publication, sampler extraction, lifecycle docs and alternatives. A standard-library-only static check parsed source ASTs and compared SHA256/revision contents without importing or executing target modules. It detected the external allocator edit; the original allocator and affected references were then checked against `git show 042a948...`. Attractive claims were rejected or narrowed:

- **Rejected:** mask omission violates pinned sampling contract. That contract explicitly selects host output for intermediates.
- **Rejected:** every intermediate chunk invokes the host sampler. All-intermediate batches skip it, after current Qwen readback.
- **Rejected:** plugin submits prefill/decode in one batch. It interleaves separate phases while preserving off-batch state.
- **Rejected:** every continuation loses state, or decode remapping is absent. Destination change is necessary; decode cannot recover overwritten provenance.
- **Rejected:** docs universally guarantee fixed prefill slots. The demonstrated defect is missing recurrent continuity when destination changes.
- **Rejected:** no model-only state workaround exists. Full-prefix replay is possible, with implementation/precision/performance consequences.
- **Rejected:** a new Boolean is the only way to device-sample finals. Partitioning is possible; strict zero-transfer and documented-return requirements distinguish the proposed extension.
- **Not established:** a kernel defect, async completion race, measured hardware corruption or PC-01 authorization. None is needed for the source-level findings.

Minimum intervention: repair existing slot ownership without adding interface; then resolve strict chunked sampling through a reviewed design, obtaining exact human approval before implementing the proposed shared interface.
