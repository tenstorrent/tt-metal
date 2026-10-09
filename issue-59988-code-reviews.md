# Issue #59988 — implementation code reviews

Reviewed on 2026-10-09 against the uncommitted implementation based on `48468ff598adb857c29af74f8848580aa001ab6e`. These are code reviews, separate from the earlier [plan reviews](issue-59988-reviews.md). The requested acceptance scope is KDA-layer behavior on the local eight-device Blackhole LoudBox, plus the synthetic transformer bonus.

## Review provenance

- **Codex Astra:** fresh `gpt-6-astra` instance, `/root/astra_code_review`, started without inherited conversation. Initial independent source review followed by a focused MPI teardown check prompted by additional source evidence from the primary agent.
- **Claude Code:** installed CLI 2.1.295, explicitly requested `claude-opus-5-5`; the session initialization confirmed that exact model. A fresh noninteractive session used only Read, Glob and Grep, with no fallback model or delegation.
- Both reviewers received the same implementation scope and were instructed not to consult each other's findings or the earlier plan reviews. They inspected source and the validation record; they did not rerun device tests or builds.
- Snapshot: 43 modified tracked files and seven new files (four test files plus the three existing issue documents). The tracked diff SHA-256 is `811f25edf1c931f1f9b2598a5bb7eee1b47f22662b88f447ec1b98cc192c359e`. File hashes and raw review artifacts are retained locally under `/tmp/issue-59988-code-review-f7m5od7x/`.

## Consolidated verdict

Both reviewers found no confirmed KDA computational defect in the native readers or Python routing. **Address the migration failure path and performance-test scope before merging.** The LB measurements remain valid evidence for the scoped kernel/layer behavior; they do not establish shared-driver or Galaxy CI readiness.

| Priority | Finding | Evidence and disposition |
| --- | --- | --- |
| P2, confirmed source defect | Multi-rank completion rejection can hang MPI shutdown. | Astra follow-up and Claude F3; fix coordinated failure propagation. Details below. |
| P2, confirmed scope expansion; runtime failure unproven | New policy/request-head assertions run on Galaxy before the existing unsupported-topology skip. | Claude F2; verified `test_layer_perf.py:467–510`, with the skip at 511. Existing Galaxy CI selects this node at `tests/pipeline_reorg/blaze_models_prefill_tests.yaml:435`. Limit new gates to an explicitly LB-only node/block and preserve the earlier Galaxy behavior. |
| P2, CI compatibility risk; CI regression unproven | A local 8.17 ms measurement replaces the shared 8.758 ms synthetic LB reference. | Claude F1; verified `test_layer_perf.py:63,94–98` and `tests/pipeline_reorg/blackhole_e2e_tests.yaml:218`. A machine still meeting the previous reference would exceed the new upper bound. Keep same-run policy comparisons separate and rebaseline shared CI only with CI data. |

The original review findings are preserved below. **The three P2 findings were addressed during draft-PR preparation:**

- Rank zero now publishes a completion status to validators on both success and failure before any resident-state broadcast. Prefix/channel eligibility is checked before scheduling work. Host tests exercise the real driver entry point for prefix rejection, absent channels, timeouts and excess acknowledgements; they also cover non-K3 acknowledgement counting with MTP.
- Policy/request-head comparisons now live in `test_request_initialization_perf_loudbox`, parametrized only for LB SP2xTP4. The existing Galaxy node is unchanged.
- The original shared 8.758 ms synthetic LB reference is restored. The new LB comparison uses same-run ratios, without an absolute rebaseline.

The cleanup was checked locally. The original independent reviews predate these fixes; a focused Codex Astra follow-up found no new correctness issues and confirmed the identified completion-failure paths are resolved. See [the validation record](issue-59988-validation.md) for rerun results. Cross-host execution remains deferred.

## Confirmed finding

**P2 — multi-rank migration completion failures can hang.** At [migration_driver.py](models/demos/common/prefill/runners/migration_driver.py), line 890, rank zero can raise before participating in the resident-state collective. Other ranks already wait in that collective at line 716. The MPI context registers `MPI_Finalize` with `atexit`, so ordinary exception-driven exit can itself block while peers remain in the unmatched `MPI_Allgather`.

Triggers include a K3 prefix ending in KDA, a missing completion channel, or incomplete/timed-out/unexpected acknowledgements. This is a source-confirmed control-flow defect, not a hardware-reproduced hang. Other uncoordinated exception paths predate the patch, but the patch introduces these expected completion-failure paths. It does not invalidate successful LB KDA-layer measurements.

Required correction: coordinate failure with all ranks using a matched collective error protocol or an explicit distributed abort. Check deterministic prefix/channel eligibility before submitting prefill work; that alone does not fix asynchronous timeout failures. Add failure-propagation tests. This was open at review time and is addressed by the cleanup described above; no real multi-host hang/recovery run is claimed.

## Material tradeoffs and validation limits

- **Broader migration contract:** the shared driver now requires exact model-aware completion for all adapters, even with verification disabled. K3 migration prefixes ending in KDA are rejected. Test non-K3 callers and move eligibility checks earlier.
- **Stale state and API compatibility:** reused carries/slabs contain the previous request until a completed commit/export. External readers must respect that lifecycle. In-tree reset callers were updated; out-of-tree callers of the removed `reset()` or `reset_streams()` APIs need migration.
- **Potential continuation cost:** the enabled chain reader waits for the start scalar before issuing the external seed read. The measured LB comparison passed its 3% gate, but tiny differences are within timing uncertainty. The 1.69% request-head improvement is a local measurement, not guaranteed production throughput.
- **Accuracy coverage:** production-state fixtures retain CPU PCC/RMSE and bit-exact explicit-zero control checks, but omit the recurrent peak-error bound. Most adapter cases grade transitions from the device's incoming carry; the bonus adds a short independent CPU trajectory. Longer independent trajectories and calibrated production peak bounds would strengthen qualification.
- **Explicitly deferred:** Galaxy/SC4, full-checkpoint and AttnRes correctness, cross-host completion ordering, and production throughput. Policy-enabled SP1 direct execution remains unsupported and rejects at construction; generic default-off direct execution is preserved. Traced multi-slot operation remains restricted by #59977.

## Reconciliation of lower-priority findings

- Claude F4 is a useful constructor-contract hardening suggestion. The current production factory sets the flag correctly, so no active missing-policy call site was identified. Prefer a public policy property/validation over coupling the adapter to a private field.
- Claude F5 identifies coverage limits, but it does not establish a missing LB acceptance case: enabled distributed behavior is tested through the real adapter on FABRIC_1D; the FABRIC_2D component suite checks generic default-off behavior and was changed to avoid required LB cases being skipped. Policy-enabled SP1 direct execution is intentionally rejected, with generic direct regressions retained.
- The production-width peak gate was omitted for newly added 128-wide cases; the existing small-fixture threshold was not weakened. That still leaves a production peak-error qualification limit, as both reviews note.
- Do not simply accept excess acknowledgements as completion: without request/generation provenance, extra queued acknowledgements can also hide missing current work. Claude's suggestion to treat extras diagnostically needs a concrete validity argument before relaxing the gate.
- Claude's native-affine guard reference `affine_exclusive_scan_device_operation.cpp:293–296` refers to the wrong line numbers; the verified source is lines 38–42. Its migration-contract reference `KDA_STATE_MIGRATION.md:836–843` is actually lines 74–81. Original report wording is preserved below. Its claim that SP1 affine “should not be slower” is an inference, not a measured guarantee. The recorded +0.016% is start-zero policy overhead; the measured positive-start ratio was +0.001%, both below what these samples can establish as statistically significant.

Claude completed successfully with the exact requested model, 68 turns, and no permission denials. The original reviewer reports follow, including Astra's correction after the MPI follow-up.


## Codex Astra — initial independent report

Independent read-only review: **no confirmed correctness bug found in the implementation within the agreed LB KDA-layer scope.** I did not run builds/device tests, edit files, or inspect the other review.

The native changes look sound: absolute `actual_start` is read before overwriting chronology storage; local zero fills complete before publication or output writes; positive starts retain real seed reads; convolution preserves predecessor history; distributed downstream scans keep the policy disabled. The new attribute is threaded through operation parameters and compile-time arguments, while public defaults remain off. SP1 direct rejection matches actual dispatch, and appended configuration fields preserve positional compatibility.

Material tradeoffs and remaining concerns, prioritized:

1. **Migration compatibility/failure handling — medium, intentional behavior change with an untested failure-path concern.**
   `models/demos/common/prefill/runners/migration_driver.py:75–81` now makes a completion channel and exact acknowledgements mandatory for **all adapters**, including when verification is disabled. K3 prefixes ending in KDA deliberately fail (`.../tt/runners/adapters/kimi_k3.py:115`). This is defensible fail-closed behavior, but materially broader than a KDA memory optimization. Static rejection currently happens only after the entire prefill (`migration_driver.py:884–890`); move model/channel eligibility checks ahead of submitting work. Also, failure occurs before the resident broadcast at line 893 while validator ranks wait at line 716. This creates a potential distributed shutdown/hang issue unless the launcher reliably aborts peers; I have **not confirmed a hang**. Add a collective error/abort path and host tests covering non-K3 adapters and distributed failure propagation.

2. **Slab lifecycle/API break — medium, intentional.**
   `.../tt/kimi_k3/kda_state.py:44–45,110–121`: a reused slot retains the previous request’s bytes until a completed commit/export. Removing `reset()` and `reset_streams()` also breaks out-of-tree callers. In-tree call sites appear migrated, and the documented new contract is clear. Completion gating does not introduce generation/validity metadata into the slab itself; external readers must obey the contract. This is an integration requirement, not a confirmed current bug. Keep remote consumer qualification explicitly deferred.

3. **Numerical coverage is narrower than an end-to-end production accuracy claim — low, evidence limitation.**
   `.../tests/kimi_k3/test_kda_padding.py:213–230` reconstructs the CPU reference seed from the device’s incoming carry for each transition; this was preexisting and tests transition correctness rather than accumulated long-stream drift. New production-width cases omit the recurrent peak-error bound at lines 279–281. They retain CPU PCC/RMSE and exact explicit-zero-control comparisons, which are useful for this initialization change, but shared arithmetic bugs or localized peak errors can pass. Existing small-fixture thresholds were not weakened. The bonus transformer provides an independent two-chunk CPU trajectory, but only for its explicitly selected plain residual fixture. A longer independent-reference trajectory and calibrated production peak bound would strengthen follow-up qualification.

4. **Performance tradeoff/evidence — low, no invalid comparison found.**
   `.../chain_affine_transforms/.../reader_writer_chain_affine_transforms.cpp:73–89` serializes positive-start seed reads behind the metadata barrier; there is a real potential continuation cost. The bracketed same-position measurements in `.../tests/kda/perf/test_layer_perf.py:211–224` are a reasonable control for gradual drift. The request-head comparison also includes the historical reset and ordinary commit/export. However, the continuation microbenchmark repeatedly reads one fixed nonzero carry (`:190–198`), rather than committing an evolving stream, and five bracketed groups under substantial clock drift do not establish statistical significance for tiny ratios. Treat the reported result as “no >3% regression in this local fixture,” and the 1.69% request-head improvement as a local measurement, not guaranteed production speedup.

**Verdict:** no blocking finding from source inspection; reasonable to accept as **KDA-layer implementation validated on LB**, subject to independently checking the recorded test artifacts. Galaxy/SC4, cross-host completion ordering, full checkpoint/AttnRes correctness, and production throughput remain unqualified as documented.


## Codex Astra — MPI follow-up and corrected verdict

**Correction: promote this to a confirmed P2 distributed failure-handling defect.** Source establishes a newly introduced unmatched-collective path that can hang; execution is not needed to establish the control-flow error. I have not reproduced its wall-clock behavior.

**Finding: `[P2] Coordinate completion failures before leaving validator ranks blocked`**
Location: `models/demos/common/prefill/runners/migration_driver.py:890`.

Trigger: a multi-rank invocation with any new completion rejection: K3 trailing-KDA prefix, missing channel, or partial/timed-out/unexpected acknowledgement count.

Source chain:

- `prefill_producer.py:1481–1488` initializes the distributed context.
- Python `ttnn.init_distributed_context()` calls `DistributedContext::create` (`ttnn/core/distributed/distributed_nanobind.cpp:1119–1125`), which selects `MPIContext` in an `OPEN_MPI` build (`tt_metal/distributed/multihost/distributed_context.cpp:18–26`).
- Nonzero ranks enter `_run_validator` and block in `_mr_bcast_resident` (`migration_driver.py:716`). Its first operation is `distributed_context_allgather_int` (`prefill_producer.py:1493–1494`), ultimately a blocking `MPI_Allgather` (`mpi_distributed_context.cpp:515`).
- Rank zero raises at the new `_require_prefill_completion` call, before joining that collective. `main()` has no handler coordinating this failure.
- Ordinary Python exit runs the registered `MPI_Finalize` callback (`mpi_distributed_context.cpp:236`). Validators cannot reach finalization because their allgather lacks rank zero. Explicit `MPI_Abort` exists separately (`:649`) and is not invoked here.

Thus “the launcher will notice rank zero exit and kill peers” is insufficient: rank zero can itself remain inside collective finalization before exiting. External timeout/forced termination can break the hang, but that does not correct the program.

**Scope:** the underlying driver already has other uncoordinated exception paths; this change newly exposes that defect for expected completion-validation failures. It does not affect successful LB layer execution or single-process migration. Remote numerical qualification being deferred does not remove this shared-driver regression.

**Fix:** catch completion failure and either explicitly abort the initialized distributed context or deliver failure through a matched collective protocol so every validator exits coherently. Move deterministic prefix/channel eligibility checks before submitting prefill work, but early validation alone is insufficient for acknowledgement timeouts; those also need coordinated failure handling.

Updated verdict: **no confirmed KDA computational bug; one confirmed P2 migration-driver failure-path bug.**


## Claude Code / Opus 5.5 — independent report

# Code review: #59988 Kimi-K3 device-side request initialization (uncommitted diff on `48468ff598a`)

## Overall verdict

**Approve the core change, but fix the test/CI scope first.** I found no confirmed correctness bug in the native readers or the Python routing. The confirmed problems are in the surrounding surface: the perf test, CI expectations and the shared migration driver. Two of them (F1, F2) touch CI jobs this PR doesn't own and shouldn't ship as-is. This review was read-only; I didn't build or run anything.

**What I verified in source, not taken from the plan or validation docs:**

- **Raw start, not modulo position.** All three readers set `fresh_request` from the raw `actual_start` word before it gets overwritten:
  - chain at `reader_writer_chain_affine_transforms.cpp:77`, before `store()` at line 79;
  - affine at `reader_writer_affine_exclusive_scan.cpp:190`, before the `actual_end` read reuses `words[0]` at line 193;
  - convolution at `reader_qkv_causal_conv1d_silu.cpp:74`.

  A start of one full SP cycle (rank 0 is first again) therefore continues the request rather than restarting it.
- **NoC ordering.**
  - In the chain reader with the policy on, the seed DFB is filled (zeros plus `write_zeros_l1_barrier`, or a read plus `async_read_barrier`) before `write_state(initial, …)` at :98 and before `push_back` at :100.
  - The order in which the chronology and seed are pushed to compute is unchanged.
  - With the policy off, the old overlapped read is preserved (:65-67).
  - In the affine reader, workers outside the active groups still exit at :203-205, before any seed handling. Every active worker zero-fills its own seed and waits on the barrier before `push_back`.
  - In the convolution reader, the zero fill of the window happens after the previous work item's tap reads have been waited on (:157), so the window isn't still in use.
- **Convolution scope.** Only the request head's external history is zeroed: `!initial_from_predecessor && mt == 0` (:114-120). The predecessor read at :134-135 still takes priority, so later SP ranks keep this chunk's predecessor tokens.
- **SP1 grouped path.** The aliased `tail_entry_states=initial_state` (`recurrence.py:309`, and `_scan_chunks` at :317) is never read on SP1:
  - `derive()` never sets `local_split` when there is one partition, so `reset_group` and `reset_chunk` resolve to "no tail";
  - the recurrent scan reads the tail only when `reset_chunk != 0` (`reader_recurrent_chunk_scan.cpp:188`).
- **SP>1 routing.** The policy reaches only `chain_affine_transforms`. The downstream `affine_exclusive_scan` (`recurrence.py:445-459`) stays default-off, and a native check rejects enabling it on SP>1 (`affine_exclusive_scan_device_operation.cpp:293-296`).
- **Program cache.** The flag is both a field in each params struct (hashed by reflection; there are no custom hashes under `kda/`) and a kernel compile-time argument. The `actual_start` value stays runtime data.
- **Construction and removed APIs.**
  - `build_attention` (`attention.py:348-353`) is the only place that constructs a K3 `ttKDA`.
  - SP1 direct execution with the policy on is rejected before weights load (`kda.py:124-125`).
  - No in-repo references to `reset_streams`, `kda_states.reset` or `_reset_kda_carries` remain.
  - `allocate_state` uses `ttnn.zeros` (`kda.py:212-225`), so the harness's synthetic-prefix restore really is a zero seed.

## Findings

### F1 — Medium (confirmed change; whether it breaks CI is uncertain): the LoudBox perf gate in CI was re-baselined from one dev machine

- **Evidence:**
  - `test_layer_perf.py:61-63` adds `_LOUDBOX_SYNTHETIC_REFERENCE_MS = 8.17`.
  - `:94-98` substitutes it for SP2xTP4.
  - That test runs in CI at `tests/pipeline_reorg/blackhole_e2e_tests.yaml:218` (`-k "SP2xTP4"`) with a two-sided ±3% band.
- **Scenario:** a CI LoudBox that ran at the old 8.758 ms reference was inside the old band. Against the new one ([7.925, 8.415]) it now fails the lower-is-fine/upper-is-fail check.
- **Impact:** possible CI break from a change unrelated to the feature. The validation record itself shows the old gate already failed on this machine before any code changed, which points at hardware or setup variance, not this change.
- **Fix:**
  - Revert the re-baseline in this PR.
  - Keep the new policy gates, which are ratios against a control run in the same session and don't depend on the machine.
  - Re-baseline separately using CI data.

### F2 — Medium (confirmed): the Galaxy perf job now runs unvalidated policy and request-head gates

- **Evidence:**
  - `test_layer_perf.py:467-510` runs `_paired_policy_samples_ms` and `_request_wall_samples_ms` for every layout. The latter allocates `KdaStates` and captures two more traces.
  - It asserts on both, and this runs *before* the existing "TP axis not wrapped → skip" check at :511-515.
  - The SP8xTP4 case of this test runs in CI at `blaze_models_prefill_tests.yaml:435`.
- **Scenario:**
  - On Galaxy, any failure in 8x4 slab geometry or trace-region capacity, or a noisy ratio sample, fails a job the plan says is deferred.
  - On a Galaxy where TP isn't wrapped (Linear), the test used to skip; now it can fail before reaching the skip.
- **Impact:** CI risk outside the approved LoudBox scope, and extra runtime on Galaxy.
- **Fix:** guard the new block with `if layout != "SP8xTP4":`, or move it into its own LoudBox-only test node. At minimum, put it after the skip.

### F3 — Medium-Low (confirmed behaviour change; downstream effect uncertain): the stricter migration completion check changes the shared driver for every model and skips the normal failure path

- **Evidence:**
  - `migration_driver.py:64-82` now raises when the ack channel is missing, the count is zero, a drain is partial or times out, or the count exceeds the expected value.
  - The call at :890 sits before `_mr_bcast_resident` (:892-893), before the verdict collection, and before the `PREFILL_SEND_SHUTDOWN` sentinel (:964-974).
- **Scenarios:**
  - **Multiple ranks:** rank 0 raises while the validator ranks are blocked in `distributed_context_allgather_int` (`prefill_producer.py:1493`). The job ends by MPI abort or hangs, not through `_mr_allgather_verdict`.
  - **Other models (DeepSeek MTP, DFlash, MiniMax, Gemma):** these previously drained acks on a best-effort basis and ignored the count. They now need an exact `_ack_layers_per_chunk(kv_table) * pushes`. `try_consume_all` takes every queued ack, so leftover acks from an earlier or timed-out run push the total over and fail an otherwise correct run.
  - I could not confirm whether MTP acks fire on every chunk, including partial final chunks where `mtp_provided_levels` returns 0.
  - Only the K3 path was tested, and only against host test doubles.
- **Fix:**
  - Catch the failure and feed it into the existing `migrate_ok=False` / verdict / shutdown path instead of letting the exception escape from `main()`.
  - Either make the exact-count requirement opt-in per adapter (K3 already has `validate_migration_completion`), or validate it on one dense model.
  - Consider treating "more acks than expected" as a separate diagnostic rather than the same hard error.

### F4 — Low (hardening; not a live bug): the K3 adapter doesn't enforce the policy it now depends on

- **Evidence:** `TtK3KdaAttention.__init__` (`attention.py:202-208`) accepts any `ttKDA`. With `reset()` and `reset_streams()` gone, nothing else clears carries.
- **Scenario:** a future call site, or a test, wraps a default-off `ttKDA` (for example one built with the plain `kimi_k3_program_config()`) in the K3 adapter. Each new request then silently continues from the previous request's carry.
- **Fix:** add `assert kda._zero_initial_state_on_start` (or expose a public property) in `TtK3KdaAttention.__init__` or in `KdaStateCache`.

### F5 — Low (test coverage is narrower than the claims suggest)

- **Native test is single-device only.** `test_request_start_policy.py:44-162` exercises the chain op with one SP rank, so with the policy on, multi-rank chain rotation is only covered indirectly through the adapter tests.
- **Component chain suite.**
  - The suite (`components/test_chain_affine_transforms.py:31-36`) moved LoudBox from FABRIC_1D to FABRIC_2D, while production LoudBox runs FABRIC_1D.
  - It doesn't exercise `zero_initial_state_on_start=True` at all.
  - The fabric change is reasonable for getting the cases to run, but it's out of scope.
- **Production-width peak-error check removed.**
  - For production width (96 heads, K=V=128), `test_kda_padding.py:279` drops the 0.6 L∞ check against the CPU oracle.
  - The bit-exact comparison that replaces it is against a generic grouped control that shares all the arithmetic except seed selection.
  - So it proves the seed choice is equivalent, not that the result is accurate. PCC and RMSE still apply.
- **SP1 cases now use grouped scan.** Under `build_layer(..., zero_initial_state_on_start=True)` (`utils.py:479`), the K3 padding test's `(1,8)` cases switched from direct to grouped. That matches production, but the K3 adapter no longer has any direct-path coverage.

### F6 — Low (documented tradeoff): reused slots expose the previous request's state

- **Evidence:** `KDA_STATE_MIGRATION.md:836-843`.
- **Behaviour:** until the new request's first export, a reused slot's slab holds the previous request's carry. The old `reset()` exported zeros instead.
- **Impact:**
  - The in-repo driver is gated (F3).
  - Any consumer outside the repo, such as the dgen worker or other per-layer-ack consumers, could read another request's state.
  - This isn't verified and is deferred.
- **Fix:** keep the documentation. Track the external-consumer audit as a blocker for the production qualification work.

## Pre-existing (not introduced here)

- No device-side check that `actual_start`/`actual_end` are aligned or non-empty when only device scalars are supplied (`chronology.hpp:87-91`).
- When `end == start`, the affine scan's workers outside the active groups leave their outputs unwritten.
- The ack count proves only "this many acks arrived", not ordering across hosts.

## Tradeoffs and limitations

- **Static opt-in flag.** It keeps generic callers compatible, but it means two compiled program variants per op. It also adds a C++/nanobind parameter that must stay consistent with the K3 construction site. Direct-strategy K3 configs are now rejected.
- **Chain reader serialization.** With the policy on, the seed read waits behind the `actual_start` read. This was measured as negligible only at SP2xTP4 (+0.016% at start 0); SP4 and SP8 weren't measured. The SP1 affine path swaps a DRAM read for a local zero fill, so it should not be slower.
- **Perf evidence is reasonable but narrow.**
  - The bracketed ratios against a control run alongside each sample are sound.
  - The "1.69% faster at request head" control includes eager host dispatch of the reconstructed seven-program reset. That's fair to the legacy behaviour, but it's a host-plus-device comparison, not device-only.
  - The profile counts device programs (49 with and without the policy; the 7 reset programs are gone). It doesn't show DRAM transactions, and the validation record says so.
- **Scope and maintenance.** The perf re-baseline, the fabric switch in the component test, and the shared-driver strictness are each independent changes. Splitting them out would shrink the review surface and the CI blast radius.

## Would more validation change the verdict?

- **On the core feature: no.** The reader logic and routing check out on inspection, and the dirty-seed/NaN and continuation tests target the right failure modes.
- **On F1-F3: yes.**
  - A CI LoudBox run of `test_synthetic_kimi_k3_perf[…SP2xTP4…]` would settle F1.
  - One run of the Galaxy SP8xTP4 job (or the guard) would settle F2.
  - A multi-rank failure-path test of the migration driver, plus one non-K3 adapter (MTP/DFlash) with the strict count, would settle F3.


## Codex Astra — draft-PR cleanup follow-up

A read-only follow-up of the final cleanup found no new correctness findings. Rank zero now announces failure through a matched collective before validators enter the resident broadcast. Setup, scheduling and acknowledgement failures are covered. Prefix/channel checks run before scheduling or attaching migration. The original performance baseline and Galaxy behavior are restored; policy benchmarks are isolated to the explicit LB test.

Other distributed exceptions outside the coordinated region remain preexisting limitations. Mocked collective tests establish control flow and do not replace a real multi-rank smoke test. The reviewer ran `git diff --check`, with no edits or device tests.


## Codex Astra — integration with unaligned prompt ends

During draft preparation, main gained #59473 (non-32-aligned KDA prompt ends). A plain merge would have retained old convolution-history rows for fresh one/two-token requests. The final integration adds a native `select_request_history` reader to the existing chronological-selection implementation. It directly gathers the selected rows and synthesizes zeros only for missing external-history rows at absolute start zero. Generic explicit-state callers retain their existing selection path.

Astra reviewed the new kernel, validation/factory, cache bindings and both Python routes, finding no confirmed correctness bug. The review checked index mapping, predecessor preservation, scratch separation/barriers, trace-bound metadata, allocation ownership and inactive-rank placeholders. Follow-ups were addressed with direct input-rebinding/invalid-input tests and a documented selection-record provenance precondition. Post-merge execution and profile evidence are recorded in the validation document.


## Claude Code / Opus 5.5 — integration follow-up

A fresh read-only invocation confirmed model `claude-opus-5-5`, completed successfully in 30 turns, and found no confirmed defects in the selector, Python routing, or integration with unaligned prompt ends. The reviewer inspected source independently and did not execute tests. Raw stream and report are local artifacts under `/tmp/kda-59988/claude-merge-review*`.

The suggested width hardening is included: positive channel widths must be multiples of 32, keeping each BF16 row aligned to 64 bytes, with a rejection test for width 16. Consolidating identical per-coordinate programs, using NoC zero-fill for the rare missing-prefix rows, and additional targeted multi-device microtests remain optional follow-ups. The adapter matrix already covers predecessor-based short tails against exact generic controls, and the NaN short-request matrix covers fresh one/two-token requests on all three LB mesh layouts.

A device run during development caught an unaligned scalar read in the initial selector. The final kernel reads the scalar at the aligned scratch base, saves the predicate after a barrier, then reads selection metadata at the same base. The corrected direct tests passed; the earlier failed diagnostic remains in the local logs.
