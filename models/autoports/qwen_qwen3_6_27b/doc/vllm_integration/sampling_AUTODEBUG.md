# AutoDebug: Qwen3.6-27B mixed-request sampling failure

## Findings

**Confirmed: penalty co-admission can change an already-running seeded request from device sampling to host sampling and back.** The saved controlled probe shows its first sampled-token difference exactly at the first host-sampled token; the corresponding greedy control stays identical. Qwen's missing device penalties trigger this behavior. It is a serving-routing defect under the stage's no-mid-request-switch requirement, but **the pinned plugin explicitly documents batch-dependent routing under `seeded_sampling_policy=auto`**. Do not misreport it as a violation of a plugin promise of sampler affinity.

**The exact smoke failure is not fully attributed.** In `test_mixed_params_batch`, all decode forwards in both runs are host-sampled. Only one first-run prefill is device-sampled, and its request identity is missing from the log. It belongs to either Count or Random. Consequently Random may have changed device→host in run 1, or may have stayed on host in both runs. The later controlled probe does not supply the missing original mapping. There is no established stale-token, KV, recurrent-state, or host-RNG ownership bug explaining a same-route Random mismatch.

**For the confirmed penalty-triggered route switch, the smallest relevant implementation boundary is the model's canonical sampler.** Device penalties are explicitly disabled/rejected, despite the existing common sampler providing penalty primitives. Implementing them requires correct count ownership, masking, reload, and warmup. A capability flip alone is invalid. This is not yet a proven repair for the original Random mismatch. Complete resume support also exposes a genuine missing prefill history distinction; extending that shared model interface needs approval beyond PC01.

## Scope and provenance

Inspection date: 2026-11-16. No model/serving execution, hardware operations, HTTP, resets, server/process control, source edits, or AutoDebug relaunch were performed. Only this report was written. Verification used source reads, AST parsing, standard-library JSON/byte-offset analysis, offline tokenizer encoding/minimum-piece vocabulary analysis, and static artifact inspection, with three xhigh subinvestigations and a post-draft claim audit.

Paths below use:

- `Q/` = `models/autoports/qwen_qwen3_6_27b/` in this tt-metal checkout.
- `R/` = `Q/readiness_vllm/full_sync_serial_retry/`.
- `P/` = `/home/ttuser/qwen36-27b/vllm-tt-plugin/`.

The requested `models/autoports/qwen_qwen3_27b/readiness_vllm` does not exist here; the named failing artifacts are under `Q/`. tt-metal HEAD is `818f5e81574b25c1a0784051172e7f91dc96ae52`; plugin HEAD/pinned base is `042a94863ad27d88207bcf71abdf67122c6f4c27`. Both working trees contain pre-existing changes, including the adapter/generator and plugin PC01–03 candidates. This analysis describes the **prepared working tree**, not pristine plugin HEAD. Existing plugin-change records still say some validation/commits are pending; this investigation does not complete those records.

`R/runner.log` records full-model serving, capacity 32, context 262144, block size 32, chunk budget 512, device sampling `all`, trace mode `all`, and `--no-async-scheduling`. The stage work log records `QWEN_VLLM_HOST_COMPATIBILITY=1`. The launch TT config contains no seeded policy override; `P/src/vllm_tt_plugin/config.py:41` defaults it to `auto`. Server startup records default top-k 20. Decode tensor buckets are 1/32; the mixed test submits ten logical requests, not 32 simultaneously active requests. Full64 refers to model layers, not this test's request count.

Contracts read: plugin `AGENTS.md`, `docs/MODEL_INTERFACE.md`, linked setup/execution/lifecycle/capability documents, `DECODE_RELOAD_CONTRACT.md`, README sampling policy, and the installed vLLM integration skill plus `references/plugin-changes.md`. PC01 authorizes only `supports_prefill_sampling_mask` and prefill `sampling_mask` with the recorded semantics. It does not authorize new sampler ownership policy, history keywords, or routing capabilities.

## Direct observations

### Exact smoke window: do not combine different tests' routes

`R/sampling_smoke_audit.json` bounds the smoke at bytes **466427–501259**. Byte 466008 is the previous successful logprob-audit endpoint. Offsets here are zero-based; server line numbers are one-based.

| Workload | Recorded event order | Evidence in `R/server.log` |
|---|---|---|
| Isolation run 1 | Device prefill of slot 0, length 3; host prefill of slots 1–9; nine host decode forwards | L1803 @466485; L1806 @467273; L1819–1854 |
| Isolation run 2 | Host prefill of seven rows, then three rows; nine host decode forwards | L1856 @475378; L1865 @477150; L1871–1906 |
| Isolation end | All second-run responses complete before the next workload | L1907 @484101; next prefill L1908 @484168 |
| Following top-1/greedy test | Three device prefills and eighteen device decode forwards | L1908–1968 |
| Following min-p test | Two host prefills and nine host decode forwards | L1970 onward |

Thus isolation contributes **1 device prefill + 3 host prefills + 18 host decode forwards**. The audit's remaining device activity belongs to the subsequent top-1 test. Its totals reconcile exactly: device prefill 4, device decode 18, host prefill 5, host decode 27; **22 device / 32 host forwards**. These are forwards, not per-request token counts. An aggregate mixture is not proof that Random switched.

The first run's initial device slot 0 and host-prefilled slot 2 both have prompt length 3 and five generated tokens; they are Count and Random in unknown order. The unique longer prompts and 3/5/8/10-token departures, followed through `slot_remap`, constrain the other identities but cannot distinguish this pair. The failing strings identify Random's five-token request; the failure's shuffled client index "Request 2" is **not** a server slot. Both strings require at least five pieces in the cached tokenizer byte vocabulary, excluding the other temp=2/seed=42 request Number, whose maximum is three tokens.

Tokenizer provenance: `/home/ttuser/qwen36-27b/hf/hub/models--Qwen--Qwen3.6-27B/snapshots/8f33b98937c6031bb49e713dee3e8e185aafd26f/tokenizer.json`, SHA256 `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`. Minimum-piece byte-vocabulary analysis gives five for each string; this is offline evidence, not captured output-token IDs.

`P/tests/tt/utils.py:136–188` uses concurrent coroutines and `asyncio.gather`. It preserves returned result order but does not guarantee server admission order. Neither socket-port order nor client-list order supplies the missing mapping. The shuffle permutation and request IDs are not recorded alongside prefill slots.

The two remaining possibilities are:

| Run-1 singleton | Random routes, run 1 | Random routes, run 2 | Consequence |
|---|---|---|---|
| Random | D,H,H,H,H | H,H,H,H,H | Routing can explain the mismatch; first host draw starts from a CPU stream not advanced by the device draw. |
| Count | H,H,H,H,H | H,H,H,H,H | A Random route-switch explanation is false for this execution; inspect same-prefix logits, parameters, and host RNG. |

The smoke reports one failure, two passes, one skip. `test_chat_logprobs_all_vocab` was skipped, so it is not all-vocabulary logprob evidence. The test stops at its first failed assertion; it does not prove every other configuration reproduced.

### Independent controlled probe establishes a real mid-request switch

`R/sampling_route.json`, its corresponding log, and `Q/tests/probe_vllm_sampling_route.py` provide stronger identity information than the smoke: one active request has 129 prompt tokens and requests 64 output tokens; its only companion has 137 prompt tokens, requests eight outputs, and sets presence penalty 1. The companion is submitted only after the active request's first streamed token. The sampled active bodies match exactly: temperature .7, k20, p.9, seed42, ignore EOS.

| Saved case | Server byte interval | Active request's token route, zero-based indices | Result |
|---|---|---|---|
| Sampled solo | 501588–517745 | D: 0–63 | Prefix `[264,11,198,...]` |
| Sampled + penalty companion | 517745–536861 | D: 0–1; H: 2–8; D: 9–63 | Prefix `[264,11,197,...]`; first difference **2** |
| Greedy solo | 536861–552747 | D: 0–63 | 64 outputs |
| Greedy + same companion | 552747–571955 | D: 0–1; H: 2–8; D: 9–63 | All 64 outputs equal solo |

The active request is identifiable throughout from its unique length, output count, positions, and remaps. In the sampled mixed case, active device prefill is L2162 @517812, its first device decode is L2165 @518609, and companion host prefill is L2169 @519121. The first shared host decode is **L2172 @519943**, with positions `[137,130]` and remap `[1,0,...]`. After seven host decodes, the eight-token companion completes; **L2193 @523646** returns the active request to device sampling.

This directly proves route instability and sequence divergence coinciding with its first host token. It does not isolate RNG from all simultaneous shape/remap/numerical effects, and it does not retroactively identify the smoke singleton. A late nonpenalized companion would be the useful missing control.

### Passing controls are substantial but bounded

`R/lifecycle.json` and its audit pass on the same full-model synchronous server: 713 device forwards / zero host forwards; greedy repeat/reorder comparisons equal; 27 allocator-grown page columns consumed; 23 page-only updates; 663 unchanged-table steps; four retained trace signatures; zero retirements; program-cache entries constant at 24853. This makes a broad stale-input or trace-recapture account less economical than the observed routing boundary. It does not exclude a sampled-only or particular-request state bug.

`R/logits.json` records twelve concurrent-vs-solo top-20 comparisons with maximum error 0 and six standalone comparisons with all 20 IDs shared and error 0. These are **first generated-token** checks for six other, prompt-formatted controls, using explicit host logprobs. They do not establish identical full-vocabulary logits for Random's later sampled prefixes. The route probe uses raw-token continuation stress, not a qualitative model-quality evaluation.

## Source causal chain and ownership

1. **Model support is deliberately missing.** `Q/tt/generator_vllm.py:31–40` declares device sampling, k≤32, masked prefill, and `supports_device_penalties=False`. `_sampling_mode` at :149–154 maps absent sampling parameters to explicit host compatibility. It does not force eligible seeded requests to host.
2. **The plugin chooses one sampler for each submitted forward.** `P/src/vllm_tt_plugin/model_runner.py:3989–4088` checks eligibility. At :4047–4050 active penalties plus the false capability force host for the whole batch. All four penalty-bearing configurations in the isolation test remain potential triggers even though explicit top-k values fit 32. `sampling_mask` only removes the intermediate-chunk fallback for opted-in adapters; it does not override penalties. The smoke prompts are 3–7 tokens, all final, so PC01 finality is not the active blocker.
3. **There is no route affinity under auto.** Prefill decides at :2653–2675; decode decisions are recomputed as scheduled membership changes. README :290–304 and `docs/MODEL_EXECUTION.md:218–232` document one batch-wide route. The comment at runner :2761–2765 explicitly acknowledges companion-dependent determinism. The stage skill separately says a seeded request must not change samplers mid-request. That stronger acceptance condition is not a promise already implemented by the pinned plugin.
4. **The streams differ.** Request CPU generators are created in `P/.../input_batch.py:151–155`, attached/moved with requests at :515–517 and :661–664, and rekeyed by runner `_build_host_generators` at :1418–1445. Device sampling does not consume that CPU generator. Qwen `_reset_rng` at `generator.py:415–436` derives device seeds from request seed plus absolute prediction-input position; `_sample_graph` at :345–355 increments once and restores inactive token/seed rows. A correct reload cannot make this algorithm produce the same draws as vLLM's host exponential-noise sampler. Artificially advancing the CPU generator on device steps is not an equivalence fix.
5. **Mode switches already command input reloads.** `P/.../async_decode.py:426–483` detects sampling-mode changes, invalidates the resident chain after prefill, reloads authoritative inputs on host steps, and resets sampler state on device transitions. Qwen `generator.py:938–988` remaps before reload/reset, binds pages, uploads authoritative token/position controls, and selects decode/sample traces. The recorded host steps have `reload_inputs=true`. This is positive evidence against a simplistic "host steps reuse stale device feedback" account.

The ordinary host RNG paths were also checked: `P/.../host_sampler.py:39–48` prefetches with private clones; :67–99 keys cached noise by generator identity and checks full state; :117–119 commits one advance. Mixed seeded/greedy batches can use upstream `random_sample`, which overwrites seeded rows with their own generator's noise (`python_env/lib/python3.12/site-packages/vllm/v1/sample/ops/topk_topp_sampler.py:554–565`). No source contradiction proving a same-route RNG leak was found. This is not a runtime proof that every numerical/filter path is batch-invariant.

### Completion and buffer boundaries

Trace replay is nonblocking, so a Python forward return alone is not evidence of completion. Device prefill `_read_tokens` materializes output. Decode adapter `read_decode_output` at :270–275 queues a CPU transfer before output-buffer reuse; `process_decode_output_host` at :277–292 performs required readback/conversion. The plugin finalizes reads before host sampling; overlapped readbacks additionally wait on recorded events. This server explicitly disables async scheduling. The relevant evidence is completed HTTP responses/token counts and saved results, not just trace enqueue messages. No uncompleted producer/consumer mismatch was found at this boundary, and no hardware result is inferred from AST checks.

## Smallest valid fixes and their limits

### A. Implement the missing canonical device penalties; do not flip the flag first

This is the correct owner for the penalty trigger. Current `Q/tt/generator.py` rejects nondefault penalties at :367–376, hardcodes neutral values at :395–403, and calls `_run_sampling(..., penalties_on=False)` at :351. `models/common/sampling/generator.py` already owns `TTPenalties`, prompt/output reset helpers, penalty application, and counted sampling (:203–211, :415–444).

The necessary implementation work is:

- Carry all three penalty parameters through canonical generator configuration, prefill slot placement, decode parameter merging, and the sampling signature. Apply repetition to prompt-or-output membership, and presence/frequency to generated outputs only, before temperature/top-k/top-p.
- Populate prompt masks and committed output counts for the correct request slots; remap or rebuild them before sampling, including remaps accepted during host compatibility. Release/reuse must clear only the owning slot. Consume the existing authoritative decode `prompt_tokens`/`output_tokens` histories on commanded resets; `output_tokens` is presently accepted but unused by this generator.
- Count exactly one emitted token per selected row. Common `_run_sampling` updates counts before Qwen restores inactive token/seed rows. `TTPenalties.update_output_tokens` at :366–413 has no active-row mask and scatters a source of ones. Simply enabling penalties would count inactive and masked intermediate rows. Mask count updates or restore the complete affected count state; preserving token/seed vectors alone is insufficient.
- Preserve PC01 semantics: intermediate rows execute/publish model state without synthetic sampling or count advancement. New-owner slot initialization may clear the previous owner's counts; that is distinct from counting an un-emitted token. Preserve continuing rows' sampler parameters, RNG and token feedback, and build the final prompt mask from complete history when the row becomes sampleable.
- Warm the exact penalties/count-update path before any retained captures. The current common `precompile` has an existing `compile_token_update=True` option (:491–513); default precompile skips that update. Configure a genuinely penalties-active warmup before invoking it, then clear synthetic counts. Common reset helpers no-op while penalties are inactive (:203–211, :447–458), so slot/global lifecycle cleanup must remain correct regardless of the current parameter toggle. Prepare every reachable sampling trace variant and stable buffers; avoid runtime compilation or changing a Python Boolean as if it could alter an already-captured trace.
- Preserve raw host-compatibility logits and avoid applying penalties twice. `TTPenalties.apply` normally rewrites its input in place (`tt_penalties.py:32–74`); warmup/capture needs fresh or scratch logits, and repeated sampling cannot reuse an already-penalized value as raw input.

These are model/common-sampler implementation changes, not grounds for a new plugin routing exemption. Keep `supports_device_penalties=False` until its full contract is implemented and validated. Turning it true after implementation uses an existing documented capability, not a new interface, but needs evidence for all admitted paths. The fresh-prompt core can be implemented and reviewed independently; that alone does not close the serving gate because the false capability still routes companions to host.

### B. Complete device penalty resume semantics needs a real history distinction

The existing prefill interface cannot always tell original prompt tokens from replayed generated tokens. Runner :2443–2475 explicitly includes generated history on resume. `submit_prefill` at :4090–4154 sends combined `tokens`, absolute chunk starts/ends, slots and, for eligible sampling, parameters/mask. It sends no original prompt boundary. Both normal finish and preemption invoke `release_request(slot)` (:998–1020), with no request identity or reason exposed.

A concrete indistinguishability example:

- World A: original prompt `[10]`, committed outputs `[20,20]`, preempt/release, then replay `[10,20,20]` in a reused slot.
- World B: the same prior request has the identical model-visible history, then finishes/releases; a fresh request's prompt is `[10,20,20]` in that same slot. Only the undisclosed reason for release differs before the identical next prefill.

The adapter can receive the same tokens, starts `[0]`, ends `[3]`, slot, page mapping, mask `[True]`, seed and parameters in both worlds. Correct output count for token 20 is two in A and zero in B. Presence/frequency penalties differ. Retaining token content or slot history, guessing from page addresses, and waiting for the next decode history cannot fix the already-emitted first resumed token. Repetition's union happens to match, which does not rescue the other penalties.

**Minimum information needed is the original prompt boundary.** One original-prompt length per submitted prefill row, interpreted against existing full-history `tokens`, would suffice; alternatively extend the existing decode `prompt_tokens`/`output_tokens` history keywords to prefill. The latter reuses names but still extends the shared call contract. Neither is authorized by PC01. A future implementer must choose one precise interface, document fresh/chunked/resumed semantics, defaults and legacy-model behavior, write the complete proposed plugin-change record, and obtain exact owner approval before implementation. A new sampling capability/environment/config knob is not necessary to represent this history fact. Compatible delivery must still be demonstrated: existing adapters do not automatically accept a new prefill keyword because they support penalties. If delivery requires an opt-in marker, include that exact additional interface in the proposal and authorization.

### C. Do not disguise generic auto routing as a small model fix

Implementing device penalties removes this test's feature-triggered fallback once all penalty paths are supported. It does not make actual host-only features device-compatible, and it does not by itself establish lifelong sampler affinity.

Generic policy changes such as splitting incompatible cohorts into separate forwards or retaining sampler ownership require a concrete scheduling/contract design. One batch-wide sampler cannot simultaneously preserve an existing request's device ownership and sample a new host-only companion in the same forward. Making `auto` enforce a new affinity rule changes documented default behavior and falls under the skill's owner-authorization rule. Per-row hybrid output/sampling would be a still larger model-facing contract. Do not introduce either as an unrecorded "bug fix".

The existing `seeded_sampling_policy=host` may be useful as a separately labeled diagnostic, but **is not the proposed production repair** and does not establish correct device serving. Do not route every seeded request to host for reproducibility, remove the penalty cases, weaken the test assertion, change seeds/top-k recipes, or relabel the failed gate as passed.

## Discriminating checks for follow-up work

These checks were not executed in this inspection. Start with CPU metadata/fixed-logit checks; hardware checks belong to the implementing session.

1. **Resolve the exact smoke mapping.** Record request ID or stable prompt/config hash, submitted rows, owner slots, route and reason, effective parameters, and emitted-token index for each forward. Retain the shuffled client permutation and token IDs. If Random was run-1 slot 2, reject its route-switch explanation and investigate the all-host path. Existing logs cannot recover this missing identity.
2. **Test host ownership using fixed logits.** Exercise add/remove/re-admit/condense and different admission groupings with the same seeded request; include mixed unseeded greedy rows, all-seeded rows, and transitions into the optimized host-sampler path. Compare committed generator state and outputs after each draw. A cached noise entry may be discarded without advancing a live stream. This can falsify the static host-ownership assessment without hardware.
3. **Add the missing routing control.** Repeat the saved late-companion pattern with penalties neutralized while keeping both prompts, admission timing and output lengths. Verify active D throughout, and compare the first two outputs and the next common-prefix logits. Use enough distinct greedy fixtures as well as the existing raw-token stress; a repetitive greedy continuation has limited sensitivity.
4. **At the first divergent common prefix, compare route, parameters, full candidate/logit data, and RNG state before drawing.** Same route/prefix/parameters/RNG but different logits points to model or filtering numerics/state; identical inputs and RNG but different samples yields a small sampler reproducer. Different routes already invalidate a same-stream reproducibility claim. Do not compare logits after the prefixes diverge. Adding a public `logprobs` request on this four-chip model itself forces host sampling, so it is not a control for the original device route; use separate diagnostic observation/teacher forcing and label any logits readback outside measured serving.
5. **Validate the actual penalty mechanism.** Fixed-logit references should check positive/negative repetition scaling, prompt-only versus generated occurrences, repeated-output multiplicity, combined penalties, nonidentity remaps, released/reused slots, and masked/all-intermediate prefill. Check first-prefill→decode reset explicitly: the first emitted token is already counted and appears in authoritative output history, so reset must replace history without losing or double-counting it. Include A/B from the history example; they must produce different presence/frequency behavior from identical concatenated histories. Keep real count updates inside the captured sampling graph, then verify traced versus reference behavior with no new compile/capture after warmup and no inactive-row count changes.
6. **Run targeted serving before the full profile.** Keep `auto`, device `all`, k≤32 and the unchanged mixed-params test. Verify zero host forwards for this now-device-supported penalty mix and stable request routes; rerun the saved companion probe, greedy lifecycle and logprob controls. Only then run smoke/full sampling and the remaining required qualitative/probe gates. The skill permits separate classification only after correctness, logprobs, crash-free serving, and qualitative output all pass and the remaining failures are demonstrated reproducibility-only. Same-route execution plus logits/state checks alone is insufficient; do not assume that classification now.

## Other potential issues, not explanations of this smoke failure

| Candidate | Source evidence and smallest intervention | Why secondary |
|---|---|---|
| Front-packed **host resumed-prefill** loses output history | Runner :2691–2700 builds penalty/bad-word histories only on decode; extraction :4384–4414 gives prefill empty output history and treats combined replay as prompt. Presence/frequency and multi-token bad-word matching therefore miss generated history on the resumed first sample. Build/use original histories through existing internal `TTModelInput` fields for host prefill. Lane host metadata already does this at `input_batch.py:1211–1223`. A fixed-logit resume-vs-decode regression needs no new adapter API. | All smoke starts are zero with 3–7-token fresh prompts and no recorded resume. This real contract gap does not establish the observed Random mismatch. Distinguish this internal repair from the new **device-prefill** metadata contract above. |
| Selected-row eligibility is inconsistent for lane-DP | `check_perform_device_sampling` scopes seed/top-k to `sampling_rows` (:4023–4046) but penalties use global `InputBatch.no_penalties` (:4047). Lane prefill supplies a selection (`input_batch.py:1586–1590`), so an unsubmitted resident penalty row can influence route. A selected-row predicate plus matching contract/tests is a narrow candidate. | Current capability docs explicitly describe the global predicate; do not call it an unequivocal documented-contract violation. Front-packed DP1 removes unscheduled rows before build (:1026–1042); this does not fix the observed live DP1 companion transition. |
| Same-route numeric or sampling reproducibility | Different batch shapes, filtering paths, ties and CPU optimized/native paths are not ruled out for Random by the other prompts' top-20 checks. | No captured common-prefix logits/RNG for the exact failure. No evidence justifies headlining stale state, low precision, or an allowed benign difference as the cause. |

## Evidence pins and post-draft review

Current-source SHA256 values (AST parsing passed for all five):

| File | SHA256 |
|---|---|
| `Q/tt/generator_vllm.py` | `00bd28cbe48f1efe713260d3c79cc7a95217508ce9b901211521d8096b1cb770` |
| `Q/tt/generator.py` | `c9aaaf7c263f42afb6f3cf9e5e5e1c4020bb9985aa419b6646033426860d10ca` |
| `P/src/vllm_tt_plugin/model_runner.py` | `401436bbf7871cb8d3d566666b9621f41513ffa95b52a499ebc2daa8ec4b0cf8` |
| `P/src/vllm_tt_plugin/input_batch.py` | `4e0f079231b8e1f9dd785dac119842fe11263bff25c90afebd3f4f1cbd021305` |
| `P/src/vllm_tt_plugin/async_decode.py` | `083f87d99525e45295e05ebb252f6bdb9085185a9ee1004831e307b316eab4f4` |

Post-draft adjudication specifically challenges the favored routing story: the isolation device-decode count belongs to another test; singleton identity remains ambiguous; documented auto semantics are not a sampler-affinity guarantee; the controlled probe establishes its own identity/boundary but not equal logits; and secondary resume/lane bugs lack the smoke's preconditions. Independent reviews confirmed the event/count mappings and history-information argument. Corrections clarified that masked rows must avoid synthetic count advancement while new-owner initialization clears old counts, penalty warmup/reset helpers depend on activation, in-place penalty application needs scratch/raw-logit discipline, and reproducibility classification needs all correctness/quality predicates. The standard-library audit passed count reconciliation, saved route/log correspondence, token boundaries, lifecycle counts, and all five source pins/ASTs; `git diff --check -- AUTODEBUG.md` passed. No runtime root-cause proof or completed serving gate is claimed. The concrete established blocker is missing device penalties plus demonstrated route instability; attribution of the original Random mismatch remains conditional on the missing mapping.
