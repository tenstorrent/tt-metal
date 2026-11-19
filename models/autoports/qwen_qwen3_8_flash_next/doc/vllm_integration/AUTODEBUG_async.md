# AutoDebug: reduced mixed-serving greedy differences

## Findings

**The six string differences are real, but this comparison does not isolate async correctness. Every wave changes its physical decode-width history.** The async run performs one additional singleton decode before admitting the second request, then switches the surviving request back to singleton one position earlier. These changes reach an explicit singleton FP32 versus multi-row BF16 reduction boundary. This is the leading source-supported explanation to test; the artifacts do **not** prove that numerical sensitivity caused these particular six outputs.

**No concrete stale/pending-token, page-growth, position, or snapshot-lifetime defect was established.** The inspected producer/consumer and transition ordering is coherent. The passing adapter probe supports individual mechanisms, but misses their full serving combination. It cannot clear server correctness, and the existing failed equality result must remain failed.

**The exact first generated-token divergence is unavailable in these artifacts.** They retain text and sample counts, not emitted token IDs or per-submission request identities. Exact first character divergences and the first differing execution geometries are derived below. Inferring an original token index by re-tokenizing these strings would be incorrect.

One independent documentation discrepancy was verified: the seven intermediate-prefill calls do force host-logits transfer, but all are singleton intermediate calls, for which the pinned plugin **skips the host sampler**. The broader prefill-interface limitation remains. No interface change was implemented.

## Scope and revision boundary

Inspection date: 2026-11-16. No model, server, target test, or hardware reproduction was executed. Work consisted of source inspection, Git reads, JSON analysis, hashing, and a local tokenizer-only check. This investigation writes only this report and did not relaunch AutoDebug.

Paths below use:

- `M/`: `models/autoports/qwen_qwen3_8_flash_next/` in this tt-metal checkout.
- `P/`: `/home/ttuser/qwen38-flash/vllm-tt-plugin/`.
- `V/`: `/home/ttuser/qwen38-flash/vllm-env/lib/python3.12/site-packages/vllm/`.

tt-metal HEAD is `b6c08997041167c909b9db11dcc3366cb47643bd`; the plugin is clean at `042a94863ad27d88207bcf71abdf67122c6f4c27`. Both saved server commands use vLLM 0.31.0, checkpoint `812d96e10a0dd82b69d749e68693698cf6c36daa`, capacity two, context 1024, page32, token budget 256, no prefix caching, and `sample_on_device_mode=all`. Their recorded command difference is `--no-async-scheduling` versus `--async-scheduling`. Construction records select real layers `(0,1,3)`, TP4/EP4, and `combined_shared4_mixed`; this is reduced debugging, not full-model acceptance.

Current source must not be mistaken for historical instrumentation:

- The checked-in adapter currently advertises `supports_async_decode=False`. `M/doc/vllm_integration/work_log.md:64,70` records enabling it for the experiment and restoring false afterwards. Independently, the saved runs contain 465/482 and 470/487 decode submissions with `reload_inputs=false`; `P/src/vllm_tt_plugin/async_decode.py:430-474` forces reload when the capability is false. Thus current false does not show that historical resident decode was disabled.
- During inspection, working-tree diagnostic additions were present in `generator_vllm.py` and `check_vllm_mixed.py`. Added `active_slots`, `submitted_positions`, and token-ID fields are absent from v3/v4. This report does not attribute those edits to the investigation or retrofit their information into old evidence. Harness line numbers below refer to `git show HEAD:M/tests/check_vllm_mixed.py` where specified.
- The saved runs have no complete source hash manifest. Historical numeric fixtures support the arithmetic mechanism, not a claim that every source byte and runtime state matches this serving experiment.

Evidence SHA-256:

| Artifact under `M/` | SHA-256 |
|---|---|
| `readiness_vllm/reduced_control_v3/mixed.json` | `ff2edf842f55b2f1e72e0e6e3543500112f3753902872e2a4c3b813e5483c628` |
| `readiness_vllm/reduced_async_v4/mixed.json` | `d3cac22e5dc1ad216787646b2e896f3c887afda31c242924a242ae8fa3b046cb` |
| `doc/vllm_integration/adapter_async_v1.json` | `c2044d8e3363fabd0d5f4e1758be20ff9f0784d1cc45a5d97a12b82be3ff3d58` |

## Direct observations: what first differs

Both reports complete ten requests, each with 96 output tokens according to API usage. Greedy rows are `0,1,2,3,4,5,8,9`; rows 6/7 are sampled and excluded from equality. Both reports retain seven host-prefill submissions, 17 total prefill submissions, 92 internal prefill chunks, identical scheduler start sets `{0,256,512}`, unique captures, no workload retirement, and no observed out-of-vocabulary sample. Decode submissions differ: **482 control versus 487 async**.

The following offsets are zero-based Python string character offsets, not bytes or token indices. Suffixes start at the first unequal character.

| Request index | Prompt length / variant | First unequal character | Control suffix | Async suffix |
|---|---|---:|---|---|
| 0 | 129 / 0 | 21 | `Advertisementsymar…` | ` BáLEEP-Core…` |
| 1 | 513 / 1 | identical, 388 characters | — | — |
| 2 | 130 / 0 | 23 | `rgasOUStry…` | `ckt Główirth…` |
| 3 | 515 / 1 | 420 | ` someloch` | `nyeoload` |
| 4 | 131 / 0 | 141 | `Prostitmier…` | `unanattia…` |
| 5 | 257 / 1 | identical, 398 characters | — | — |
| 8 | 513 / 0 | 13 | `践ejo_RG…` | ` shyrach…` |
| 9 | 129 / 1 | 412 | `勇往直前` | ` inher` |

The first response in workload order already differs: row 0 agrees through `spreDTVÓSurts shyrach`. Rows 3 and 9 differ only in their final short suffixes. This early-left/late-right pattern is consistent with admission and retirement width changes; it is not a token-by-token causal proof.

The saved harness accumulates text and a timestamp for each nonempty streaming chunk; it does not preserve each chunk's text or token IDs (HEAD harness:50,70-76). The saved adapter's `sampled_tokens` event contains only `count` and OOV count. A tokenizer-only check using the local checkpoint's `tokenizer.json` re-encoded greedy strings into **101-112 tokens**, despite API usage of 96; even text round trips do not recover original token segmentation. Therefore no original token ID/index, exact first wrong logit, or exact token-to-event correspondence can be asserted from these reports.

## Execution-history reconstruction

### Why identical HTTP payloads do not imply identical execution

The harness waits for the first **nonempty streamed text**, then submits the other request using a second client thread (HEAD harness:70-76,89-101). It does not wait at a fixed model step or enforce scheduler admission/retirement order. The control also overlaps requests (`client_overlap=true`); the name `nonoverlapped_greedy_control` means async scheduling is disabled, not that requests execute alone. Equality compares only corresponding final strings (HEAD harness:124-129).

vLLM's queued engine submits additional work before resolving the oldest future (`V/v1/engine/core.py:697-745`); the async scheduler tracks output placeholders (`V/v1/core/sched/async_scheduler.py:25-50`). That can change the number of singleton steps completed before HTTP-triggered admission. It is not itself evidence of stale or duplicated output.

### Observable first scheduling divergence

All event indices below are zero-based within `workload_events`.

In wave 0, both runs prefill row 0 to 129 and decode from `[129,0]` at event 10. At the next event boundary:

- **Control:** event 12 begins row 1 prefill `[0,256)`; event 21 continues `[256,512)`; event 30 decodes the left request at 130; event 32 finishes row 1 `[512,513)`; event 35 is the first two-request decode, with pre-call cache positions `[131,513]`.
- **Async:** event 12 is another singleton decode at `[130,0]`; prefill starts at event 14 and continues at 23; event 32 decodes the left request at 131; event 34 finishes row 1; event 37 is the first two-request decode, with pre-call cache positions `[132,513]`.

See [control event 10 onward](/home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/reduced_control_v3/mixed.json:2796) and [async event 10 onward](/home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/reduced_async_v4/mixed.json:2835). This is an observed execution-history difference before the first joint decode, independent of any explanation of the output text.

### Full wave matrix

`1×a → 2×b → 1×c` describes decode invocations with one active request, then two, then the surviving right request. Prefill calls interleave with the initial singleton segment and are excluded from these counts.

| Wave / request rows | Prompt lengths | Control decode-width history | Async decode-width history | First joint decode: left position control → async | First right-only position control → async |
|---|---|---|---|---|---|
| 0 / 0,1 | 129 / 513 | `1×2 → 2×93 → 1×2` | `1×3 → 2×92 → 1×3` | 131 → 132 | 606 → 605 |
| 1 / 2,3 | 130 / 515 | `1×2 → 2×93 → 1×2` | `1×3 → 2×92 → 1×3` | 132 → 133 | 608 → 607 |
| 2 / 4,5 | 131 / 257 | `1×1 → 2×94 → 1×1` | `1×2 → 2×93 → 1×2` | 132 → 133 | 351 → 350 |
| 3 / 6,7, sampled | 130 / 129 | `1×1 → 2×94 → 1×1` | `1×2 → 2×93 → 1×2` | 131 → 132 | 223 → 222 |
| 4 / 8,9 | 513 / 129 | `1×1 → 2×94 → 1×1` | `1×2 → 2×93 → 1×2` | 514 → 515 | 223 → 222 |

Method: enqueue decode submissions and device-prefill submissions, associate formatted sample events in submission order, and use their valid sample count to derive active count. Device-prefill samples are removed from the decode table. This handles v4's two outstanding decode submissions rather than incorrectly attaching each sample to the adjacent event. In these plain UniProc runs, worker calls execute inline and output futures finalize at `.result()` (`V/v1/executor/uniproc_executor.py:38-48,102-117`); the engine resolves oldest futures (`core.py:725,740-745`). Transition drains visit pending submissions in append order (`P/src/vllm_tt_plugin/async_decode.py:735,935-945`). Prefill follows that drain and samples synchronously. Each wave's ledger empties and sums to 192 formatted samples including its two final-prefill samples, matching usage. Thus five extra **invocations** here accompany one fewer two-row invocation per wave, without an observed increase in emitted tokens. Sample counts establish execution width; they do not independently identify request IDs or rule out later output suppression.

The logged `positions` field is the complete internal cache-position array **before** the call, not submitted `start_pos` or active membership. Two positive positions do not imply two active requests. For example, v4 event 221 logs `[605,224]` but produces one active sample; the following cache array becomes `[606,0]`. The table uses sample counts plus the source path, not array length. Request reordering also explains position-array swaps such as `[131,513]` becoming `[514,132]`; missing historical slot-remap metadata prevents treating those arrays alone as proof of request ownership correctness.

The earliest decode at which physical width differs for each greedy request is consequently:

| Request | Input position | Control width → async width | Output ordinal computed from position* |
|---|---:|---|---:|
| 0 | 131 | 2 → 1 | 3 |
| 1 | 605 | 2 → 1 | 93 |
| 2 | 132 | 2 → 1 | 3 |
| 3 | 607 | 2 → 1 | 93 |
| 4 | 132 | 2 → 1 | 2 |
| 5 | 350 | 2 → 1 | 94 |
| 8 | 514 | 2 → 1 | 2 |
| 9 | 222 | 2 → 1 | 94 |

\* Prefill emits output ordinal 0; decode at input position `p` emits ordinal `p - prompt_length + 1`. These are **geometry-change ordinals**, not measured first unequal token indices. Rows 1 and 5 still match despite their late geometry difference: width-dependent rounding need not change argmax. Changed prefill interleaving and peer histories can also precede a request's own width transition, so this table does not claim complete execution equivalence before that position.

## Source mechanism: capacity two still selects two numeric paths

The actual reduced serving route is `QwenGenerator.decode_forward → FactoredDecode.replay → layer.mlp`. It does not use `model.decode_batch` as its replay implementation.

1. `M/tt/generator_vllm.py:35-37` advertises buckets `(1,2)`. The plugin chooses a supported wire-row bucket at `P/src/vllm_tt_plugin/model_runner.py:2346-2370`; padded positions are `-1` at 2525 onward.
2. `M/tt/generator.py:903-905` selects active rows using submitted positions `>=0`. `_replay_decode` delegates to factored replay at 521-528.
3. `M/tt/factored_decode.py:157-164` chooses `bucket_for_count(len(slots))`. Its MoE step at 95-99 packs the rows and calls the real layer MLP.
4. `M/tt/multichip_decoder.py:819` derives logical `n` from the packed input. `_reduce` at 893-932 dispatches `n==1` to `_reduce_single`; `n==2` reaches native `post_combine_reduce` after padding only the reduction to 32 rows.

| Item | Singleton | Two active rows |
|---|---|---|
| MoE input | `[1,1,1,2560]` | `[1,1,2,2560]` |
| Expert `input_m_tiles` | 1 | 1 |
| Combine input extent | 320 rows | 640 rows |
| Reduction implementation | HiFi4 matmul, FP32 DEST, BF16 output | Native HiFi4, BF16 DEST/scratch accumulation, BF16 output |
| Native reduction padding | not used | combined `[1,1,32,10,2560]`; BF16 weights and UINT16 IDs `[1,1,32,10]` |

The singleton config comes from `M/tt/decoder_components.py:33-41`, consumed by `_reduce_single` at `multichip_decoder.py:934-946`. Native lowering in `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/post_combine_reduce/device/post_combine_reduce_program_factory.cpp:45-92,327-330` yields 32 token rows, ten selected experts, hidden width 2560, three scratch tiles per feature row, 80 output tiles, and one participating core; FP32 DEST is disabled. `device/kernels/deepseek_moe_post_combine_reduce_compute.cpp:122-139` overwrites scratch with the first owned expert product and accumulates later products through the BF16 packer path. Equal tiled height at an earlier expert matmul does not make these reductions numerically equivalent.

This is a reachable, concrete arithmetic difference, not a newly established kernel defect. Historical primary evidence in `M/doc/full_model/moe_decode_capture_v1.json:254-450` isolates the first nonexact stage to reduction despite matching input, route IDs/weights, owned combined outputs, and shared branch: 8,901/20,480 differing elements, max absolute difference `0.000244140625`. Synthetic controls include one-owned-expert discrepancies, so “only a different sum association” is too narrow. The four native C++ hashes recorded in `M/doc/full_model/moe_reduction_rows_v2.json:1589` still match current source; the historical decoder hash differs. No such intermediate tensors were captured for v3/v4.

`M/doc/full_model/batch_geometry_disposition.md` explicitly preserves the failed cross-width comparison and scopes exact reproducibility to matched histories. `batch32_hf_factored_v2.json` records exact same-history logits/tokens with passing independent HF checks. Those prior controls motivate this diagnosis; they do not waive the new serving failure or establish its cause.

## Pending-token and completion-boundary audit

| Boundary | Inspected behavior | Implication / limit |
|---|---|---|
| Resident decode | `generator.py:917-935,992-1008` uses resident positions and device tokens when `reload_inputs=false`; host positions still supply active/inactive sentinels | Positive host position/token staleness is intentional, not itself a defect |
| Page-only growth | `_bind(..., positions=None)` at 992-994; table shadows cloned at 700 | Updating a block map does not overwrite resident token/position |
| Token producer/consumer | model replay then sample; sample saves/restores inactive lanes (`generator.py:559-580`) | Next model consumes the preceding sample under queue order |
| Host snapshot | immutable submission metadata; each `.cpu(blocking=False)` creates/enqueues its own host read; event recorded afterwards (`generator.py:1017-1025,1169-1197`) | Shared device token address is not shared host snapshot storage |
| Wait before formatting | `P/src/vllm_tt_plugin/async_decode.py:1653-1677` waits for events; finalization guarded at 228 onward | Formatting does not merely trust successful enqueue |
| Submission ownership | residency committed at accepted submission; page tables cloned (`async_decode.py:379-407,486-509`) | Pending host completion does not make the device chain uncommitted |
| Prefill/layout transition | steady path rejected; pending finalized before build (`async_decode.py:580-607,947-966`; `model_runner.py:3208-3221`) | No obvious reload-before-drain hole |
| Apply after lifecycle | scheduler state update, then completed tokens by request identity/current row, then input preparation (`model_runner.py:2881-2894,4617` onward) | Reordering is accounted for rather than indexing old host rows blindly |

The queue ledger is `model₁ → sample₁ → host-copy₁ → event₁ → model₂ → sample₂ → host-copy₂ → event₂`. A later sample can overwrite persistent device tokens safely after the earlier host copy is queued. The lower `.cpu` implementation enqueues a read (`ttnn/core/tensor/tensor_ops.cpp:233-255`); model traces and events use CQ0. The default queue path is `tt_metal/distributed/mesh_device.cpp:978-989` and `tt_metal/impl/host_api/tt_metal.cpp:1095-1105`; no plugin queue override was found. Hardware visibility is not newly proven by source inspection.

The local token-feedback helper also has coherent bounds: `M/tt/token_feedback.py:25-41` reads three 128-byte buffers, waits, selects current/saved values using membership, writes 128 bytes, and waits for that write. No concrete count, coordinate, or lifetime contradiction was identified there.

### What the passing direct probe actually covers

`M/tests/probe_vllm_async.py` and `adapter_async_v1.json` establish:

- Stale host token/position: synchronous singleton slot 1, exact tokens and request-state match against authoritative reload; actual device positions checked (169-200).
- Page growth: synchronous singleton crossing position 32; physical page 13 written, old page 10 untouched; exact continuation match (203-247).
- Inactive preservation: inactive slot state remains exact (250-258).
- Two queued snapshots: both copies queued before waiting, active width 2 then 1, immutable metadata, distinct host handles despite one device address, reverse formatting, exact final state (260-318).

**Coverage gap:** the queued section's two calls both default to `reload_inputs=True` through its helper at 158-167. It does not combine a resident no-reload chain with pending snapshots and page growth. Submission order `(1,0)` tests row ordering, not actual `slot_remap`; no interleaved prefill or runner/scheduler input construction is exercised. Therefore this pass and the serving string failure are compatible under either a geometry explanation or an as-yet-unidentified integration defect.

## Candidate adjudication

| Candidate | Direct contradiction / contrast | Explains observations | Missing evidence | Smallest discriminator |
|---|---|---|---|---|
| Unmatched width/timing histories | Proven differing histories; proven reachable FP32/BF16 arithmetic branch | Predicts early left sensitivity and late survivor sensitivity; allows rows 1/5 to keep argmax; agrees with matched-history adapter pass | Actual first differing logits/tokens and state of these requests | Same timeline with sync/async reads, plus both timelines replayed synchronously |
| Stale/pending token across steady decode or transition | No inspected contract violation; existing drains/snapshots support intended behavior | Could still cause text differences and evade the limited adapter probe | A wrong consumed token/position, identity mapping, or ordering violation | Exact matched-history resident/queued replay including transition and page growth |
| Forced intermediate host route | Seven calls in both runs, independently real | Explains `device_sampling_only=false`; can affect timing | No evidence it directly changes the six greedy choices; wave 4 right request is an ordinary final prefill | Keep it identical in controls; investigate separately under the existing interface proposal |

The leading conclusion is **an uncontrolled numerical/execution-history comparison**, not “async is correct” or “the differences are harmless.” The same burden of proof applies to the geometry and stale-token hypotheses. No source change, FP32 widening, singleton fallback, precision-policy change, or relaxed oracle is justified by this investigation.

## Minimal discriminating follow-up; retain the exact oracle

Use the existing reduced model, selected precision, page32 pool, and exact prompts. Start with just the failing final greedy pair **513/129** (rows 8/9), whose first geometry difference is especially small: decode the left request at **position 514** with width two in the control history versus width one in the async history. This is not a claim that its first text mismatch has already been mapped to that token.

Run a deterministic adapter/runner replay whose admission, prefill insertion, retirement, slots/remaps, page tables and active-width sequence are scripted, not triggered by HTTP stream timing. The old artifacts establish width histories and cache-position transitions, **not** a complete historical input/slot/page-table schedule. An initial fixture therefore reconstructs transition metadata and tests the mechanism. To attribute the original serving failure, first obtain a new instrumented serving pair with that missing metadata and confirm its correspondence to the saved text/history differences; do not invent historical mappings. Compare this small matrix:

| Fixed execution history | Immediate completion/readback finalization | Deferred finalization with two outstanding resident snapshots |
|---|---|---|
| Control history | baseline C-sync | C-async must match C-sync exactly |
| Async history | baseline A-sync | A-async must match A-sync exactly |

Within each history require **exact per-step generated IDs, logits where captured, and final request state**, with identical precision and inputs. Keep `reload_inputs`, `reload_page_table`, sampling-reset flags and remaps identical in the two columns: immediate finalization must not silently switch to authoritative host reload while the deferred column uses resident feedback. Keep device feedback enabled in both columns, and deliberately stale positive host tokens/positions on the second resident submission. Include one page-only growth at the same logical position in both controls. At admission/retirement reproduce the normal drain and authoritative reload/slot-remap, rather than bypassing those transitions. If resident timing controls agree, an additional same-history authoritative-reload control can detect a resident-input defect shared by both columns. An adapter-level pass is only the first discriminator; a scripted runner replay must include the plugin boundary before clearing serving correctness.

Capture immutable submission sequence numbers and request IDs, request-to-slot maps, active masks/buckets, remaps, submitted versus resident positions, authoritative-reload token IDs, page-table contents/checksums, reload flags, prefill spans, and output token IDs tied to their submission. Queue debug snapshots before buffer reuse; do not attach a late device read to an earlier submission. At the first divergence, force the same known prefix in an additional diagnostic to separate differing inputs from differing arithmetic; compare logits/top-two margin and state before and after the first width-sensitive operation. Forced-prefix diagnostics supplement, not replace, autoregressive equality.

Interpretation:

1. **C-sync ≠ C-async or A-sync ≠ A-async:** width history cannot explain the whole failure. Locate the first wrong input, state, readback, or logit within that identical history; follow the pending/transition boundary.
2. **Both matched-history comparisons exact, but C-sync ≠ A-sync:** supports width/history causation only when the replay reproduces the serving divergence. A newly instrumented, history-matched serving pair must supply the first differing token for that correspondence; v3/v4 alone supply only the saved text divergence. Matching inputs/router/owned expert outputs followed by differing reduction localizes the numeric boundary. Merely obtaining some cross-width difference is insufficient.
3. **Both sync histories identical while served outputs differ:** the simple width explanation is incomplete; inspect the missing runner/scheduler or nondeterministic state dimension.
4. **A difference occurs before the first differing execution operation:** falsifies attribution to that later width switch. Logits may first differ before text does; compare both.

Do not replace equality with usage-only, OOV-only, PCC, top-k membership, or a full-layer run. Retain v3/v4 as failed evidence. Do not relabel their equality check passed because earlier full-model work accepted a scoped cross-width limitation. Any eventual change in acceptance classification requires the new controlled evidence.

## Other potential issues and documentation corrections

**Intermediate host path is real; host sampling is overstated for this workload.** `P/src/vllm_tt_plugin/model_runner.py:2665-2669` forces host routing for any intermediate prefill. `M/tt/generator.py:837-845` reads the host logits. But all seven recorded intermediate calls have one row and do not finish that request's prompt. At `model_runner.py:4311-4323`, an all-intermediate batch bypasses the host sampler, creates placeholders, and later suppresses them (`3856-3867`, `_build_chunked_prefill_output`). Mixed final/intermediate batches take the host-sampling branch and protect intermediate RNG using clones. Thus `M/doc/vllm_integration/plugin_changes.md` is correct about forced transfer/interface limitations, but its assertion of host sampling on these seven calls is inaccurate. This is not evidence that host RNG caused the greedy differences. The proposed prefill-mask capability remains pending; nothing was implemented.

**Control validation is incomplete.** The checked-in harness zips requests and compares text without validating prompt/config/execution-history identity. `--reduced` labels output rather than enforcing reduced construction. In these saved runs the command/construction records and request lengths supply the missing static context, but the harness should not independently be treated as a controlled async oracle. This is a test attribution gap, not a reason to weaken its result.

**No broader acceptance claim.** Counts, OOV checks, stable traces, and clean completion do not prove greedy correctness, full-model accuracy, or every queued-state invariant. No hardware follow-up was run here.

## Post-draft claim review

The report was rechecked against the saved events and executable branches. In particular: width was derived from formatted active counts and replay selection, not the always-length-two cache array; current false async capability and newly added diagnostic fields were not projected backward; original token indices were not fabricated from text; the actual factored replay path was followed into native reduction; the direct probe's reload defaults were checked; all-intermediate host sampling was demoted/corrected after inspecting the sampler bypass. The remaining numerical cause and complete serving-state correctness are explicitly unverified.
