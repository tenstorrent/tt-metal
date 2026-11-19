# Forced-preemption parity investigation

## Findings

**Plugin revision `215effd` has a concrete same-step resume bug in its front-packed persistent batch: replacement KV block IDs are appended to the previous row instead of replacing it.** This is independent of the recently repaired original-prompt and penalty-history contract. A host-only execution of the production update methods demonstrates the inconsistent state. Fix or falsify this bug before attributing the complete parity failure to BF8 numerics.

**The first resumed model scores also differ, before penalties or sampling.** The saved host logprobs establish this at output index 22, while greedy token IDs remain equal through index 61. Prefill and decode execute different rounded graphs, but inspection cannot establish that the difference is acceptable, or that it is the only remaining cause. In particular, the page-table bug does not by itself explain the first resumed score difference under the five-page reuse example below.

All output indices in this report are **zero-based**. No device, server, reset, process-control, or network calls were made. No source was edited; this report is the only requested artifact. The live reduced server was left untouched.

## Scope and revision ledger

Source abbreviations used below:

- `M`: `models/autoports/ornith_ai_ornith_1_5_9b` in this tt-metal checkout.
- `P`: `/home/ttuser/ornith-9b/dependencies/vllm-tt-plugin/src/vllm_tt_plugin`.
- `V`: `/home/ttuser/ornith-9b/tt-metal/python_env/lib/python3.12/site-packages/vllm`.
- `E`: this `M/doc/vllm_integration/diagnostic8_history` directory.

Prepared tt-metal HEAD is `d1c644f35289ee46bfa4a23598aa607d5ea96522`, with modified `M/tt/generator.py` and untracked serving/test artifacts including `generator_vllm.py`. The inspected generator SHA256 is `95052193c2099a3cddf7fbd78101ce872d3d932e32e4bb40cfa2a7acd2549d61`. These working-tree changes, rather than HEAD alone, are the model under investigation.

At initial inspection the plugin was clean at `215effdee4e473f8c50af608d9812fafc56fd48b` (`Restore resumed-prefill penalty history`). Its audited `model_runner.py` SHA256 was `9619c1c6535d80cdfba2631df145a981f208b614b83e17891d1e75d979f58fad`; `P` line references below refer to that revision. Installed vLLM reports 0.31.0 in the saved log, whose responses carry fingerprint `vllm-0.31.0-32c593ab`. The lifecycle analysis below uses the **installed current upstream source**; there is no complete saved upstream source snapshot proving byte-for-byte equality to the historical process. Installed `V/v1/core/sched/scheduler.py` SHA256 is `f820bee4f42b878b262a30f3728a97d7413477b2a277748e20f2528ac2ec2f2a`.

During final report review, a concurrent working-tree edit appeared in `P/model_runner.py`, adding an existing-row resumed branch that calls `block_table.add_row` and marks the layout changed; `tests/test_resumed_persistent_blocks.py` also appeared in the plugin repository. This investigation did not author or modify those files. That edit changes the diagnosed branch and has not been executed or validated here; the bug finding and AST result are pinned to the audited `215effd` source.

`E/server.log:71-72` records policy `bfp8_all_lofi` and registration of `OrnithServingDiagnostic`; `M/tt/generator_vllm.py:261-264` selects layers `(0,3)`. The saved failing run and direct adapter regression therefore use the same reduced layer kinds. Neither is full-model correctness evidence.

## Direct observations

`M/tests/vllm_preemption_probe.py:18-56` submits the three modes below with prompt `[9707 + i % 13 for i in range(129)]`, 100 outputs, presence 0.75, frequency 0.5, repetition 1.1, and `ignore_eos=True`. At 20 client-observed tokens it awaits `/pause?mode=keep&clear_cache=false`, then `/reset_prefix_cache?reset_running_requests=true`, then `/resume`.

Parsing `E/preemption.json` gives:

| Mode | Outputs baseline / forced | First unequal token | Reset response |
| --- | --- | --- | --- |
| Device greedy | 100 / 100 | 62: baseline 37, forced 387 | success |
| Device temperature .8, top-k 32, top-p .9, seed 42 | 100 / 100 | 22: baseline 264, forced 51 | success |
| Host greedy with logprobs | 100 / 100 | 62: baseline 37, forced 387 | success |

Device and host greedy sequences are exactly identical to each other within each baseline/forced branch. Every request ends with `finish_reason="length"`. Client count 20 is not the model replay boundary: all three logged resumed prefills have `ends=[151]`, `original_prompt_lens=[129]`, `starts=[0]`, slot 0 (`E/server.log:349-358,560-569,771-780`). Thus 22 already generated outputs are replayed, and the resumed prefill produces output index 22.

For host greedy, raw selected-token logprobs are exactly equal at indices 0-21 and differ at **all 40 indices 22-61**, despite identical emitted tokens throughout that interval. At index 22, both branches choose token 0 (`!`):

| Raw logprob | Baseline | Forced |
| --- | --- | --- |
| Selected `!` | -6.95357608795166 | -7.005373001098633 |
| Reported top token, space | -5.26607608795166 | -5.302248001098633 |
| Space minus `!` | 1.6875 | 1.703125 |

The changing gap rules out a common log-normalization shift alone. These are **raw prepenalty** scores: `P/host_sampler.py:187-191` constructs the default upstream sampler, `V/v1/sample/sampler.py:61-69` defaults to `raw_logprobs`, and `:79-102` computes them before logits processors. `P/model_runner.py:4450` calls the host sampler without a mode override. A penalty-only or seed-only explanation cannot account for this observation, although either kind of bug could coexist with model-state differences.

## Confirmed source bug: a resumed persistent row retains the old block table

### Reachability through the reported control sequence

1. In the installed upstream implementation, keep-mode pause enters `PAUSED_ALL`; the scheduler reports zero unfinished requests in that state (`V/v1/core/sched/scheduler.py:2738-2740`). `EngineCore.step_with_batch_queue` drains existing futures but does not submit an empty worker build when `scheduler.has_requests()` is false (`V/v1/engine/core.py:696-757`). Pause completion waits for `has_work()` to become false, including an empty batch queue (`:1457-1463,2028-2033`). Device synchronization at pause completion does not remove runner batch rows (`:875-879`).
2. The explicit reset delegates to the scheduler (`V/v1/engine/core.py:831-836`). It preempts running requests and clears `prev_step_scheduled_req_ids`, explicitly documenting **same-step preemption plus resumption** (`V/v1/core/sched/scheduler.py:2794-2810`). `_preempt_request` frees blocks, resets computed tokens, and preserves the request for replay (`:1556-1581`). Resume changes pause state (`V/v1/engine/core.py:911-913`).
3. A sole resumed request goes through the plugin's prefill-only scheduling path, which delegates to upstream scheduling (`P/scheduler.py:896-931,957-981`). Upstream marks the cached request as resumed and supplies its replacement blocks (`V/v1/core/sched/scheduler.py:1695-1715`). There need not be an intervening worker step in which the request is absent.
4. The front-packed runner releases model state for the preemption (`P/model_runner.py:1011-1026`), but `_release_model_request` only invokes the model hook (`:962-979`) and `_release_dead_state_slots` only clears a separate ownership map (`:1292-1295`). Neither removes the persistent input row.
5. Removal at `P/model_runner.py:1033-1044` covers **unscheduled** requests. The immediately resumed request is scheduled, so its existing row survives.

### The contradictory updates

`P/input_batch.py:190-196` correctly **replaces** `req_state.block_ids` when `resumed_from_preemption` is true. However, `P/model_runner.py:1068-1079` finds the existing persistent row and unconditionally calls `block_table.append_row(new_block_ids, req_index)`. It does not branch on the resumed flag there.

This is an actual append: `V/v1/worker/block_table.py:157-177` starts writing at `num_blocks_per_row[row]` and increases that count. `add_row`, in contrast, resets the count to zero. The wire page table is then read from this persistent table, not rebuilt from the corrected cached request (`P/input_batch.py:778-808`; `P/model_runner.py:2383-2394`). Model input preparation therefore propagates the discrepancy.

There is unusually direct sibling-contract evidence: `V/v1/worker/gpu_model_runner.py:1240-1247` explicitly identifies forced `reset_prefix_cache` as the case in which resumed IDs may still be cached, and uses `cached_req_ids - (scheduled_req_ids - resumed_req_ids)` to remove those rows before re-admission. The plugin's corresponding calculation omits the resumed subtraction. TT pins the V1 runner contract (`P/platform.py:982-1011`), so this is not explained by V2 folding resumed requests into new requests. Between `_update_states` and input preparation, the plugin only applies token completions; no block-table rebuild intervenes (`P/model_runner.py:2891-2904`).

A cheap local AST harness executed the actual `_update_states`, `_release_model_request`, `_release_dead_state_slots`, `apply_cached_req_state_update`, and installed `BlockTable.append_row`, with NumPy host storage and minimal model/batch recorders. It imported no model/runtime and performed no device operations. Its same-row fixture produced:

```text
old persistent pages:       [11,12,13,14,15]
replacement scheduler pages:[15,14,13,12,11]
cached req_state.block_ids: [[15,14,13,12,11]]
persistent/wire pages:      [11,12,13,14,15,15,14,13,12,11]
persistent live-page count: 10   (expected 5)
remove/add/condense calls:  none
model release(slot=0):      called; ownership map cleared
original prompt/output:    129 / 22, retained
decode-layout change flag: false
```

The fixture page IDs are illustrative, not values recovered from the log. A five-page reversal is source-realistic with prefix caching disabled: freeing reverses allocation order (`V/v1/core/single_type_kv_cache_manager.py:581-589`); uncached free blocks are prepended without reversing that list (`V/v1/core/block_pool.py:794-804`; `V/v1/core/kv_cache_utils.py:416-430`); allocation pops from the front (`V/v1/core/block_pool.py:684`). Prefix-cache reset clears hashes rather than rebuilding this queue (`:849-869`). Actual old/replacement page IDs and live counts were not logged, so this does not claim a recovered physical mapping for the saved process.

### How the bad row can cause delayed corruption

Instantiate the five-page fixture at the reported boundary:

| Phase | Mapping consumed | Consequence |
| --- | --- | --- |
| Replay positions 0-150 | Old first five pages | All five are still allocated in the reversal example. Prefill rewrites them coherently in wire order. |
| Padded prefill rows beyond 151 | Fill entries are -1 | They do not overwrite the live prefix: `M/tt/prefill_ops.py:218-224`. |
| Decode positions 151-159 | Still logical page 4 | The stale prefix can remain internally coherent. |
| Decode position 160 | Logical page 5 maps to page 15, already used by logical page 4 | The new token overwrites storage for a different causal position; subsequent attention has aliased pages. |
| Scheduler allocates the real sixth page | Runner appends it after the erroneous ten-page live prefix | Normal incremental append does not repair the causal columns. |

At position 160, the output being generated has index `160 - 129 + 1 = 32`. **This predicts the earliest aliasing boundary in this fixture, not the first unequal greedy token.** Matching argmax tokens through index 61 does not prove that causal cache entries remained correct. Conversely, the observed index 62 alone does not prove this aliasing occurred.

The chain reaches the executable coordinate contract: `M/tt/multichip_decoder.py:1334` passes the wire table and absolute positions into `paged_fused_update_cache`. Tiled K/V select the tiled factory (`ttnn/cpp/ttnn/operations/experimental/paged_cache/device/fused_update_cache/paged_fused_update_cache_device_operation.cpp:16-34`); it takes block size from cache dimension 2 and selects the tiled fused writer (`paged_tiled_fused_update_cache_program_factory.cpp:179-197,466`). That writer computes `virtual_block_id = update_idx / block_size`, selects `page_table[virtual_block_id]`, and writes the corresponding physical page plus within-page row (`../kernels/dataflow/writer_paged_fused_update_cache_interleaved_start_id.cpp:108-127`). With block size 32, positions 128 and 160 therefore target the same physical page/row in this fixture. The caller does not reject duplicate causal page IDs (`M/tt/generator.py:895-920`). This is an address/ownership error, not a numerical tolerance issue.

**Smallest intervention:** replace or remove/re-add an already resident row when a cached request is marked resumed, so the persistent table is rebuilt from replacement `req_state.block_ids`, and report the resulting layout transition consistently. Preserve accepted token history and request RNG. The lane implementation already explicitly frees/re-adds resumed rows (`P/input_batch.py:1035-1042`). No new capability or public setting is needed.

**Causal limit:** this bug is proven in the inspected production methods and matches the immediate-resume sequence. The saved artifacts omit wire page IDs; its actual contribution to the saved wrong tokens is unmeasured. In the five-page reuse example it does not explain the first resumed score change at index 22, before the aliasing boundary. It must not be presented as the sole explanation of all three cases.

## Original boundaries, sampling reset, and history: no remaining mismatch demonstrated

`M/tt/generator_vllm.py:182-218` requires the authoritative original prompt lengths and forwards them unchanged. `M/tt/generator.py:547-551` resets from `tokens[:129]` as prompt and `tokens[129:151]` as 22 generated outputs. `M/tt/sampling_state.py:353-394` reconstructs `prompt_mask`, `output_mask`, `output_counts`, and `output_counts_gathered` from exact integer history. The helper's transient all-prompt reset (`prefill_contract.py:388-392`) is overwritten before any resumed sampling; it is not an extant double-count finding.

The current seed ledger is consistent:

| Boundary | Prior output count | Seed42 device stream |
| --- | --- | --- |
| Fresh prefill, before output 0 draw | 0 | base = 275415 |
| Resumed prefill end151/original129, before output 22 draw | 22 | 275437 |
| Immediately after that draw | 23 | 275438 |
| First resumed decode, input position151, before output 23 draw | 23 | 275438 |
| Immediately after first resumed decode | 24 | 275439 |

`generator.py:293-327,337-368` derives the logical offset from explicit histories or immutable prompt origin plus authoritative position. `_sample_logits` advances the seed once (`:675-687`); common sampling applies previous penalties before counting the new token (`models/common/sampling/generator.py:422-444`). First decode does not restart the already advanced prefill seed. The decode-only planner also cannot cause a token sampled earlier by resumed prefill.

For host prefill, the plugin now calls `prefill_penalty_history` with the delivered exclusive end and original boundary (`P/model_runner.py:4395-4405`). This repairs the older host-prefill empty-output-history bug; old investigation notes describing that bug do not describe `215effd`.

The existing host-only command was run:

```text
python_env/bin/python -m unittest models.autoports.ornith_ai_ornith_1_5_9b.tests.vllm_seed_continuity_test -v
Ran 15 tests ... OK
```

These AST/fake-operation tests exercise production control methods, not silicon sampling behavior. They support the seed/history arithmetic and do not validate KV ownership.

## Replay numerics: real graph differences, unresolved acceptability

The source gives concrete reasons for score differences when 22 single-token updates are replaced by a prefill replay. The selected policy uses BF8 weights, CCL payloads and KV storage, BF16 residual/logit storage, and FP32 recurrent state. Nominal Torch BF16 cache specs are translated to selected BF8 storage by the adapter (`M/tt/generator_vllm.py:95-105`); nominal scheduler dtype is not evidence of a wrong cache type.

| Component | Prefill replay | Decode | Source |
| --- | --- | --- | --- |
| CCL reduction | At logical heights511/480, BF16 inputs remain BF16 | Explicit cast to BF8 payload, BF16 result | `M/tt/multichip_decoder.py:473-517` |
| MLP | >128-row packed projection with fused SwiGLU | Separate gate/up projections and multiply | `:785-806,846-852` |
| Linear recurrence | Masked 32-token chunk GDN | Single-token pooled conv/gates/ordered update | `M/tt/prefill_ops.py:92-153`; `multichip_decoder.py:1137-1151` |
| Full attention | Chunked prefill SDPA | Paged decode SDPA | `prefill_ops.py:325-365`; `multichip_decoder.py:1357-1372` |
| KV write | Cast then paged fill | BF16 input to fused update of BF8 cache | `prefill_ops.py:348-350`; `multichip_decoder.py:1334` |
| Terminal head | Ordinary linear | Normally DRAM-sharded LMHead1D | `M/tt/model.py:109-150,218-235` |

The KV write interfaces permit these different input types (`ttnn/cpp/ttnn/operations/experimental/paged_cache/paged_cache_nanobind.cpp:38-43,91-94`); their difference alone is not a caller-contract violation.

A seed selects a reproducible draw from a distribution; it does not make changed distributions identical. Thus changed logits can explain why sampled output diverges immediately while greedy remains stable for many steps. **This is a plausible explanation of timing, not proof that these particular deltas are expected BF8 error.** The first differing operation, cache-state differences, penalized margins, top-k/top-p support, and actual first resumed device seed were not captured. No target-specific numeric-kernel defect has been established or ruled out by this inspection.

### Why the direct adapter pass does not settle this case

`original_boundary_device.json` and `M/tests/vllm_original_boundary_probe.py:97-155` provide meaningful exact checks of four penalty tables and per-slot seeds; the test also compares resumed third/fourth sampled IDs (`:174-203`). However:

- Prompts33/35/37 with two replayed outputs select linear/full profiles64. Prompt129 and replay151 both select linear511/full480 (`M/tt/prefill_plan.py:29-51`). The live pair does **not** cross a bucket boundary, but it differs from the passing probe's graph and >128-row MLP branch.
- Context256 selects SDPA decode key chunk64; context262144 at batch3 selects key chunk128. Selection uses cache capacity rather than current position (`multichip_decoder.py:71-83,1270-1272`).
- The test never asserts raw logit, live KV, conv, or recurrent-state equality. Its host-final case checks geometry and intentionally injects the uninterrupted token (`vllm_original_boundary_probe.py:223-239`).
- Direct calls supply coherent page tables; they bypass the plugin persistent-row replacement bug.

Consequently this pass validates a short adapter state transition, not forced-reset wire ownership or long replay numerical parity.

## Async completion and pending-output drops

The current controller distinguishes published output from applied runner history. It finalizes pending steps before a nonsteady/prefill build or forced-reset discard processing (`P/async_decode.py:935-966`), then counts completed frames oldest-to-newest and excludes only the newest counted stale suffix from state application (`:857-911`). Frames remain published so upstream can consume its stale placeholders; completed accepted frames are not blanket-dropped (`:537-547,837-850`). The scheduler carries frame counts on the output (`P/scheduler.py:1042-1067,949-955`).

The model itself queues model and sampling traces on CQ0, then queues owning host transfers and an event before a later trace can overwrite the source (`M/tt/generator.py:1079-1094,1140-1163`). Host processing synchronizes that event before reading/retiring the submission (`:1165-1198`). A host enqueue returning is not completion.

For this specific awaited keep-pause, upstream drains its queue and synchronizes the device before reset. Client20/replay22 is consistent with two outputs already accepted ahead of streaming consumption; it is not evidence of a two-token loss. The logs do not contain per-request submission/accepted/discard ledgers sufficient to prove all pending-history correspondence. No concrete newest-suffix or duplicated-token bug was found in the inspected path. Do not undo the existing suffix-discard logic on the basis of client count alone.

An accepted completed result can still await runner-state application after scheduler delivery. The next build applies those completions after updating scheduler-owned state and before preparing replay inputs (`P/model_runner.py:2891-2904`). Although the bad existing-row fixture leaves the sticky layout flag false, prefill separately invalidates the decode chain (`P/async_decode.py:409-411`), so the next device decode still reloads inputs and resets sampling (`:457-483`). The false sticky flag is not evidence of a skipped first-decode reset here.

## Other potential issues and rejected headline claims

- **Stale recurrent or convolution state on full replay:** start0 admission resets the selected model rows (`M/tt/prefill_contract.py:139-147,371-385`); runtime reset invalidates old boundary metadata (`prefill_runtime.py:202-209,462-463`). Start0 does not restore an old unaligned tail. This does not explain the observed sequence from source alone.
- **Inactive snapshot overwrites active slot0:** snapshots are refreshed immediately before each masked layer and restored immediately afterward (`M/tt/model.py:261-271`; `batch_state.py:122-151`). The copy kernel skips active rows and uses read/write barriers (`kernels/inactive_state_copy/copy.cpp:17-31`). No supporting source contradiction.
- **Padded prefill writes clobber causal pages:** fill pages beyond the valid interval are -1 (`prefill_ops.py:218-224`), even if the persistent table incorrectly calls more columns live. This is distinct from the confirmed later decode aliasing bug.
- **Decode reset seed off by one / all replayed outputs treated as prompt:** contradicted by the current explicit split and seed ledger above. These were relevant historical bugs, not established current causes.
- **BF8 quantization is harmless because greedy matches:** contradicted as a proof by differing raw scores from index22 and the concrete ownership bug. Matching tokens and aggregate quality cannot replace the missing controlled comparison.

## Candidate adjudication ledger

| Candidate | Code/evidence status | What it accounts for | Unresolved or contrary dimension | Smallest falsification |
| --- | --- | --- | --- | --- |
| Same-step persistent-page append in215effd | Direct contract contradiction, source-reachable on awaited pause/reset/resume; host AST demonstration | A real ownership error with delayed page aliasing; absent from direct adapter test | Exact saved wire IDs absent; cannot alone explain index22 scores in the five-page example | Compare scheduler and wire tables on resume and next page allocation |
| Different rounded replay/decode graphs | Concrete source differences plus immediate raw-score differences | Early sampled divergence with later greedy divergence is possible | Does not prove benign error; cannot excuse page aliases or exclude a numeric kernel defect | Identical-token replay, raw logits/live state, then same-input per-component contrasts |
| Original boundary/history/seed reset | Correct current split and arithmetic; short exact-state evidence and15 host tests | Explains why older missing-history diagnoses no longer apply | Actual failing run's tables/seeds not dumped | Capture all four tables and seed before output22 |
| Pending-output suffix misapplication | Current queue/discard sequence is coherent; awaited pause drains scheduler frames | Client20 versus replay22 does not require loss | Per-frame historical correspondence not logged | Request-keyed accepted/completed/discard ledger |

## Minimal discriminating checks, in priority order

1. **Host-only persistent-row regression.** Use a real front-packed input batch, keep a request resident, and pass one `SchedulerOutput` in which it is both preempted and resumed with replacement pages. Assert wire table == cached replacement table, live count5, then add a sixth page and assert it occupies logical column5. Compare with ordinary preemption that has an intervening unscheduled build. This directly detects the confirmed defect without TT hardware.
2. **Page-ownership ledger on a future parent-owned run.** Without changing the parity criterion, capture scheduler replacement blocks, persistent live counts/table prefix, wire table, request identity, and absolute position at replay and at159/160 and191/192. The causal prefix must match scheduler ownership and contain distinct real pages. This falsifies or confirms the suspected contribution to the saved delayed divergence. Fixing the row lifecycle is the smallest controlled intervention; rerun the same request afterward.
3. **Exact129/+22 direct replay with authoritative shared tokens.** At output22, compare seed275437 immediately **before** sampling, all four history tables, raw logits, conv/recurrent state, and live KV against uninterrupted decode. Keep the same diagnostic layers, policy, batch capacity, and context as the failing serving path. Feed identical subsequent tokens into both branches so sample feedback cannot obscure the first model difference. Inspect only valid KV positions, not arbitrary padding.
4. **Separate model from sampler.** If seed/tables match but samples differ, run the same saved logits and state through eager prefill sampling and captured decode sampling. Compare penalized logits, candidate IDs, top-k/top-p support, and seed advancement. If the raw logits already differ, isolate terminal heads on the same hidden tensor, then prefill/decode reductions on the same input, then layer-level live cache/hidden outputs. A BF16 control can localize sensitivity but is not permission to relax parity.
5. **Only if pending history remains suspect:** record accepted output count, completed-frame ordering, stale/discard counts and submitted positions under one request ID. A no-in-flight direct replay is the small control; an unpaused forced reset is a separate lifecycle test, not a substitute for the saved awaited-pause sequence.

The first draft was rechecked against call sites, persistent-table consumers, the lower KV-address calculation, explicit sampler ordering, and both saved contrasts. The page-table finding remains a concrete source bug; its exact hardware manifestation and the classification of the first resumed score difference remain unverified. No test loosening is justified by the available evidence.
