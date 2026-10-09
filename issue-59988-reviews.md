**Independent reviews of the #59988 implementation plan**

Reviewed on 2026-10-09 against repository commit `48468ff598a` and the initial [implementation plan](issue-59988-plan.md). Both reviewers inspected repository source independently, with no implementation edits or device tests.

- **Claude Code:** installed CLI 2.1.295, reported model `claude-opus-5-5`; independent noninteractive session with only Read, Glob, Grep, and WebFetch tools. Completed successfully with no permission denials.
- **Codex Astra:** separate `gpt-6-astra` agent, `/root/astra_plan_review`, started without the parent conversation. A subsequent focused exchange reconciled the API/scope differences between the reviews.

Both support the device-side zero-seed design and preserving generic explicit-state semantics. Astra found a concrete migration-completion prerequisite. Claude identified useful scope reductions and more precise test/harness changes. The plan has been amended; these are design reviews, not evidence that an implementation passes device validation.

**Subsequent user-approved scope change:** validation is now limited to KDA-layer behavior on the local eight-device Blackhole LoudBox, using synthetic fixtures. Real KDA/attention-adapter trace replay, local slab round trips, and host completion-contract tests remain required. Full serving-runtime, Galaxy/SC4, checkpoint/golden, and remote migration qualification are deferred. A synthetic KDA-focused transformer LB case is a bonus. This scope decision was made after the reviews below; their original reports and dispositions are preserved as review history.

**Disposition of findings**

| Finding | Decision incorporated into the plan |
| --- | --- |
| Padding reference must reset logically at start zero (Claude 1; Astra 2) | Accepted. Use a zero reference for new requests, retain dirty device carries, and enable the policy in the native comparison and direct-construction K3 fixtures. |
| Positive-offset benchmark reseeding (Claude 2) | Accepted. Keep a harness-local temporary zero-state allocation and in-place commit outside measurement, preserving the existing synthetic benchmark semantics without retained zero storage or a production reset API. True prefix correctness requires matching prefix state. |
| Affine misuse guard and worker barriers (Claude 3) | Accepted. Require SP1 for policy-enabled native affine scan, zero every active worker's seed, and complete every worker's zero barrier. Preserve inactive-worker and unused-tail invariants. |
| Direct recurrence scope (Claude 4) | Accepted after Astra cross-check. Production K3 is grouped. Defer direct-reader support and reject only policy-enabled SP1 direct construction before expensive setup; SP>1 remains supported regardless of the local strategy label. |
| Migration scope (Claude 5; Astra 1) | Partially accepted. Avoid a new general validity framework, but retain a concrete completion-fence prerequisite and failure-path tests. Claude's positive-length resident filter does not establish completed compute: Astra verified incorrect ack counts, ignored incomplete drains, and an input-only H2D barrier. |
| Constructor-owned policy and runtime assertions (Claude 6) | Accepted after Astra cross-check. Add a default-off frozen config field, enable it only at K3 construction, keep generic helper defaults and forward signatures unchanged, and update construction/validation tests. |
| Coalesced convolution fill and NoC ordering (Claude 7) | Accepted. Zero the first three history rows as one aligned range at a fresh request boundary and complete the zero barrier before any write/publication. |
| Native test inventory, runtime device-only metadata, omitted host bounds (Claude 8; Astra 3) | Accepted. Add chain/padding coverage and exercise runtime replay with packed metadata and absent host bounds. Host validation remains distinct from the device-value caller contract. |
| Enabled-policy benchmarks (Astra 4) | Accepted. Construct enabled-policy benchmark layers, compare zero and positive offsets, and retain default-off controls. |

**Scope reconciliation**

Astra checked Claude's constructor-fixed and grouped-only suggestions after its initial independent review. It agreed with both, while emphasizing that synthetic K3 fixtures bypass `build_attention` and must opt in explicitly. The direct-path rejection must depend on the actual SP1 execution path, not only the `local_scan_strategy` string. Astra also confirmed that the native affine guard cannot identify every semantically derived SP1 input; correct Python routing remains part of the contract.

Astra recommended retaining the migration prerequisite after reviewing Claude's documentation-only alternative. This is an existing consumer weakness, not a defect introduced by zero-seed kernels. The plan now calls for a narrow fix or explicit prerequisite that proves completion of the relevant exports, including trailing KDA layers or KDA-only slices when supported. It does not add multi-slot trace work from #59977.

Astra then read the amended plan and confirmed that all its findings were addressed, with no material contradiction remaining. Its final wording suggestion—make explicit that the downstream recurrent scan API stays unchanged—was incorporated. Claude reviewed the original plan; it did not perform a second review of the amended document.

The original reviewer reports follow. Their plan line references refer to the version before these amendments; repository source references remain at the reviewed commit. Abbreviated `tt/` and model `tests/` paths refer to `models/demos/deepseek_v3_d_p/` unless a full root-relative path is given.

**Claude Code — original independent review**

# Review: Implementation plan for #59988 (device-side KDA zero-state initialization)

## Verdict

**The design is sound. I found no confirmed correctness blockers.** The plan's main claims hold against the source:

- **Two paths, not one.** The issue names only `chain_affine_transforms` and the convolution history read, but the SP=1 grouped path reads the external seed through `affine_exclusive_scan`. The plan is right to cover it.
- **Convolution predecessor history.** Only the request-history branch should be zeroed. Zeroing every rank would erase valid predecessor history.
- **Default must stay off.** Generic callers depend on the current behaviour, so an opt-in policy that defaults to off is required.

The plan does need amendments before implementation. One required test (`test_kda_padding.py`) changes meaning, not just its reset call. A benchmark harness loses its only way to restore state. The policy has no guard against being enabled where it would silently give wrong results. And two sections (the direct path and migration) are larger than this issue needs.

---

## Findings (highest priority first)

### 1. HIGH: `test_kda_padding.py` changes meaning; it isn't just a reset caller

**Evidence:**
- `models/demos/deepseek_v3_d_p/tests/kimi_k3/test_kda_padding.py:61-64`: the cases list starts with `(0, capacity)` and appends a second `(0, capacity)` after several nonzero-start cases. The comment at `:59` says it deliberately builds up "nonzero carried state".
- `:95-109`: each case's reference is built from the actual incoming carry (`incoming = cache.read(1)`).
- `:82-83`: the traced variant runs two warm forwards before capture, then calls `cache.reset()` at `:87`.

**Why it matters:** With the K3 policy on, the final `(0, capacity)` case ignores the incoming carry. The reference at `:106-109` will still use the dirty carry, so the test fails even though the code is correct. Section 5 of the plan only lists this file as "reset-dependent".

**Amendment:** In section 5/6, say that this test's reference must use `KDAReferenceState` zeros whenever `start == 0`. Delete `cache.reset()` at `:87` so the warm forwards leave dirty state. That turns this test into the dirty-state and warmup/capture coverage from the section 6 table, so no new test is needed for that row. Also assert that the stored addresses (`:58`) are unchanged afterwards.

### 2. HIGH: The `preload_isl > 0` harness needs a concrete way to restore state

**Evidence:**
- `models/demos/deepseek_v3_d_p/tests/test_prefill_transformer_chunked.py:2140-2147`: `_reset_kda_carries`.
- `:2117-2122`, `:2176-2181`: warmup and capture run at `actual_start=preload_isl`.
- `:2228`, `:2241`: resets after capture and before every iteration.
- `:2230-2233`: `determinism_check` compares iterations.

**Why it matters:** When `preload_isl > 0`, nothing zeroes the carries anymore. Each iteration then starts from the previous one's state, which reproduces the determinism failure that the comment at `:2237-2240` warns about. The plan says to "restore the intended prefix state explicitly … or reject". But section 5 also deletes `reset()` and forbids keeping a reset method, so it leaves no mechanism. Also, the old behaviour was never "the intended prefix state"; it was just deterministic zeros.

**Amendment:** Pick one of these and write it down:
- **(a) Recommended:** a harness-local helper that allocates a temporary `layer.allocate_state()`, copies it into each carry with `ttnn.copy`, then frees it. This costs no persistent memory and keeps addresses stable, so it isn't the misleading reset section 5 warns about.
- **(b)** Make `determinism_check`/`check_pcc` skip or fail when `preload_isl > 0` and KDA layers are present.

When `preload_isl == 0`, delete both reset calls. Device-side zeroing already covers the warm and capture passes there.

### 3. MEDIUM: Enabling the policy on a non-seed input fails silently; add host-side guards

**Evidence:**
- `ttnn/.../affine_exclusive_scan/device/kernels/dataflow/reader_writer_affine_exclusive_scan.cpp:233-234`: every group worker reads `initial_state`, not just group 0.
- `:228-230`: `tail_entry_states` is read only by the reset worker.
- `chronology.hpp:94`, `:39`, `:51`: with `partitions == 1`, `local_split` is false, so `reset_group == groups`. The tail is never read on SP=1.
- `recurrence.py:435-449`: on SP>1 the same op takes the computed `local_entry_state`, which is nonzero for ranks after the first even when `actual_start == 0`.

**Why it matters:**
- The SP=1 grouped claim holds, but only because of this chronology fact. The plan should state it, because `_ordinary_group_scan` passes the external carry as `tail_entry_states` too (`recurrence.py:306`, `:313`).
- If the flag were set on the SP>1 call, ranks after the first would get a zeroed entry state when `actual_start == 0`. Nothing would error.
- The zero fill has to happen on every worker of a head, not only group 0. Non-reset workers also need their own `write_zeros_l1_barrier()`; today the barrier at `:231` only runs inside `if (reset_worker)`.

**Amendment:**
- Add `TT_FATAL` in device-op validation:
  - `affine_exclusive_scan`: allow the policy only when the SP axis size is 1.
  - `recurrent_chunk_scan`: allow it only for `!summary && groups_per_head == 1 && sp_size == 1`, if that path is kept at all (see finding 4).
- Note the `tail_entry_states` argument in a comment so it isn't silently reintroduced.
- State explicitly: "zero-fill on all G workers; each needs its own zero barrier before `push_back`".

### 4. MEDIUM (scope): K3 never runs the direct path

**Evidence:**
- `models/demos/deepseek_v3_d_p/tt/kda/config.py:123-125`: `kimi_k3_program_config` always sets `local_scan_strategy="grouped"`.
- `tt/kimi_k3/attention.py:348`: K3 always uses it.
- `recurrence.py:527-541`: direct is only chosen when SP=1 and the strategy is not grouped.

**Why it matters:** Section 2's direct-path row, the reader change, the factory/binding plumbing and the "SP1 direct" test row add a third native op change that K3 never runs.

**Amendment:** Make `KDARecurrence` raise `ValueError` when the policy is on with the direct strategy, and drop the direct changes to `recurrent_chunk_scan`. That leaves two native ops plus the convolution. If direct support is wanted for generality, mark it as optional and separate, not required for completion.

### 5. MEDIUM (scope): The migration section is broader than the risk

**Evidence:**
- `models/demos/common/prefill/runners/migration_driver.py:143-148`, `:151-156`: only resident slots with `real_len > 0` are migrated.
- `KDA_STATE_MIGRATION.md:71`: the only stated promise is that reset zeroes the slab.

**Why it matters:**
- After `reset()`, a zeroed slab is the correct state only for an empty prefix. Any consumer reading a slot with `real_len > 0` before its first commit already gets wrong data today, so removing the zeroing opens no new window for the in-repo consumer.
- The plan's "validity tracking must use the request lifecycle … distinguish successive requests reusing the same slot" plus a "reject/defer a read" test reads like a new mechanism.

**Amendment:**
- Change section 4 to: (1) update `KDA_STATE_MIGRATION.md:71`; (2) confirm in writing that `migration_driver` reads only after the producer has finished prefill; (3) ask the issue reviewers whether any external or decode consumer relies on "zero slab = empty prefix".
- Build validity tracking only if (3) finds such a consumer.
- Keep the export/import comparison tests. Drop the "reject/defer" test unless a consumer that can read early exists.

### 6. MEDIUM: Where the static flag lives, and existing exact-kwargs assertions

**Evidence:**
- `tests/kimi_k3/test_runtime_contract.py:192-198` and `:250` check the exact `kda.forward` keyword arguments.
- `:268-279` is named `…before_reset_or_replay` and asserts `kda_states.reset.assert_not_called()`. On a `Mock` this keeps passing after `reset` is deleted, so it would prove nothing.
- `ttKDA` describes itself as constructor-fixed (`kda.py:78-85`, `recurrence.py:477-478`).

**Why it matters:** A per-`forward` keyword breaks those assertions. It also leaves room for a K3 call path that forgets to pass it, for example a future second K3 attention adapter or a test that calls `ttKDA` directly.

**Amendment:**
- **Preferred:** add a `KDAProgramConfig` field that defaults to `False` and is set only where `build_attention` builds the K3 layer (`attention.py:348`, e.g. `replace(..., request_start_zero_state=True)`). Don't change `kimi_k3_program_config()`'s default: generic tests get it through `tests/kda/utils.py:340`, and `test_stateful.py:31,95` relies on explicit-state semantics at offset zero.
- **Either way:** list the `test_runtime_contract.py` edits. Rename the `:268` test and assert that `prefill_chunk` delegates without touching KDA state, instead of asserting on a mock attribute that no longer exists.

### 7. LOW-MEDIUM: Simplify the convolution change and pin the zero/write ordering

**Evidence:**
- `reader_qkv_causal_conv1d_silu.cpp:61-69`: at `actual_start == 0`, `split` is false (`chronology.hpp:94`), so `row_floor` is always 0.
- `:119-123`: the external `history` is read only when `rank == first_rank == 0`, for `mt == 0` and window rows 0–2.
- `noc.h:744-757`: the L1 zero must finish with its barrier before any other NoC write. On Quasar it borrows command buffer 0.

**Why it matters:** "Fill exactly the requested BF16 row range" invites substituting zeros row by row inside the read loop, with zero calls interleaved among `async_read`s. Only one shape actually happens.

**Amendment:**
- Specify: `if (fresh && !initial_from_predecessor && mt == 0)`, issue one `noc.async_write_zeros(window, history_rows * block_row_bytes)` at offset 0, then `write_zeros_l1_barrier()`, before the per-row loop, and skip the `history` read for rows 0–2. This is aligned by the static_assert at `:78`.
- For `chain_affine_transforms`, require `write_zeros_l1_barrier()` before the `write_state(initial, …)` at `:77`.
- Also note that the `actual_start` word lives in `window` (`:64`) and is overwritten by the zero. That's fine, but document it.

### 8. LOW: Validation list gaps

- **Missing native test file:** add `tests/ttnn/nightly/unit_tests/operations/experimental/kda/test_chain_affine_transforms.py`; it exists and the plan's native list omits it. `test_padding_prefix.py` and `test_padding_recurrence.py` should also run both policy values.
- **Traced runtime test:** add one runtime-level traced test through `TtKimiK3Runtime.prefill_chunk` with two requests on slot 0. That is the real serving path: `tt_prefill_runtime.py:995-1060` with host bounds from `prefill_runner.py:432-438`.
- **Remove the `None`-bounds dependency:** today the reset branch only runs when the host passes `actual_start` (`runtime.py:292`). Any traced caller passing `None` skips it silently, and device-side zeroing removes that dependency. Say so in the completion criteria.
- **`reset_streams` callers:** `test_chunked_prefill.py:229` and `test_prefill_perf.py:152,190` always start at 0, so plain deletion is correct for them. Note that the single-chunk perf case passes `actual_start=None` (`:154`), which becomes device 0 through `K3KdaChunk` (`attention.py:67`). Keep that covered.

---

## Confirmed correct (no change needed)

- **SP>1 seed:** `_scan_sp_grouped_chunks` reads the external recurrent state only through `chain_affine_transforms` (`recurrence.py:424-433`). `prefix_final_state` is derived from it, so the unsplit final-state selection (`:471`) is correct with a zero seed.
- **Padding:** `valid_chunks == 0` early exits (`reader_recurrent_chunk_scan.cpp:157-159`, writer `:121-123`) and `active_groups` returns (`affine_exclusive_scan:200-202`) don't depend on the seed. Padding behaviour is unchanged.
- **SP=1 convolution:** `predecessor = incoming_layer_carry` (`kda.py:281`) aliases the external history but is never read, because `initial_from_predecessor` is false and `row_floor == 0`. Zeroing the `history` branch is enough.
- **Program cache:** a new field in `*Params` structs (e.g. `chain_affine_transforms_device_operation_types.hpp:13-22`) gets hashed with the other attributes. There's no custom `compute_program_hash` under `kda/`. The scalar stays runtime data.

## Residual validation (after the amendments)

Run on real Blackhole meshes; skipped parametrizations don't count:

- SP>1 in both axis orientations, plus an SP=1 grouped geometry.
- The rewritten `test_kda_padding` (dirty state, warmup/capture, start 0 → positive → 0 → positive in one trace).
- The `test_chunked_prefill` slab-equals-carry check.
- Chain-op latency with the serialized `actual_start` read (baseline ≈78 µs per the factory comment at `chain_affine_transforms_program_factory.cpp:40`).
- Measured DRAM before and after `_zeros` removal on the SC4 stage geometry.

**Codex Astra — original independent review**

Verdict: **approve the kernel approach, but amend migration completion gating before implementation removes reset.** No confirmed blocker in the proposed zero-seed routing itself.

1. **P1 — The plan overstates existing migration completion protection.**
   `issue-59988-plan.md:72` treats the producer barrier/ack drain as established gating. In fact:

   - `models/demos/common/prefill/runners/migration_driver.py:869` waits for `NUM_LAYERS * pushes`, although K3 only emits MLA acknowledgements (`tt/kimi_k3/block.py:209`; `tt/kimi_k3/runtime.py:257`).
   - `prefill_producer.py:278` returns `None` when the ack connection fails; `_drain_layer_acks` returns zero for that case and merely warns on timeout (`:308–326`).
   - The driver ignores the drained count and proceeds to migration (`migration_driver.py:900`).
   - `H2DStreamService::barrier()` only establishes input transfer completion, not model execution or slab export completion (`ttnn/core/services/h2d_socket_service.cpp:1067`).

   **Amendment:** explicitly require a fail-closed, request-completion fence before migration; fix the expected acknowledgement count where MLA acknowledgements are sufficient. Also account for trailing KDA layers or KDA-only slices: an earlier MLA acknowledgement cannot fence later KDA exports. Test missing acknowledgements, incomplete/time-out drains, and reused-slot migration before completion. Keep this scoped to proving the existing consumer’s completion contract; a new general slot-validity framework is unnecessary.

   This is an existing consumer weakness exposed by the plan’s removal prerequisite, not a newly introduced kernel defect. It directly matters to the issue author’s requirement that migration wait for the new request’s first commit. [Issue #59988](https://github.com/tenstorrent/tt-metal/issues/59988)

2. **P2 — Make the padding-test semantic changes explicit.**
   Merely removing `cache.reset()` will not correctly update `test_kda_padding.py`. Its cases revisit zero at lines 61–64; its reference always incorporates incoming dirty state at lines 97–114; its native comparison calls the default generic `layer.forward` at line 119.

   **Amendment:** use an explicit-zero reference state whenever `start == 0`, retain reconstructed incoming state for continuation, and enable the new policy in the native comparison. Keep the persistent device carry dirty at request starts. Otherwise a correct implementation fails its existing oracle, or a weakened test accidentally preserves the old behavior.

3. **P2 — Add one runtime-level replay test with host bounds absent.**
   The shared serving path intentionally accepts `actual_start=None`, `actual_end=None`, with chronology supplied exclusively by packed `metadata_msg` (`tt/tt_prefill_runtime.py:977`, `:1027`, `:1059`). The existing host reset condition never handles that request boundary.

   **Amendment:** drive `TtKimiK3Runtime.prefill_chunk` through packed device metadata for `0 → positive → 0 → positive`, including dirty warmup/capture state. The attention-level test proposed at plan line 101 is valuable but does not exercise metadata staging and runtime dispatch. Also clarify that preserving host validation does **not** validate device-only scalar contents (`tt/kimi_k3/attention.py:35`); those remain a caller contract.

4. **P3 — Ensure the proposed performance benchmark actually opts in.**
   `tests/kda/perf/test_layer_perf.py:132` and `:140` call generic `layer.forward` with its default behavior. Running it unchanged measures the compatibility path.

   **Amendment:** explicitly benchmark enabled-policy first chunks and positive-offset continuation, retaining the default-policy result as a control. This is a validation improvement, not a correctness blocker.

The main design choices are supported by the code:

- **Static opt-in is justified.** Generic tests deliberately seed nonzero state at start zero (`tests/kda/components/test_chain_affine_transforms.py:52`) and continue streams with zero chronology (`tests/kda/layer/test_stateful.py:31`). Changing the universal default would alter an established explicit-state API.
- **Recurrence routing is correct.** SP1 direct consumes the external seed at `_run_direct`; SP1 grouped consumes it in `_ordinary_group_scan`; distributed SP consumes it in `_distributed_prefix`. Downstream scans must preserve computed group/rank entry states (`tt/kda/recurrence.py:281`, `:435`, `:621`).
- **Convolution handling is correctly selective.** The predecessor branch at `reader_qkv_causal_conv1d_silu.cpp:119` must survive, including later segments of a first request chunk.
- **Padding requires no new arithmetic design.** Existing recurrence writers already publish the last valid group’s state into the final physical carry slot (`writer_recurrent_chunk_scan.cpp:120`). Preserve that behavior.
- **Keeping stable buffers and in-trace commit/export is essential.** The plan correctly separates this from device-side slot selection and retains the #59977 guard. [Issue #59977](https://github.com/tenstorrent/tt-metal/issues/59977)

Review was read-only; no builds or device tests were run.
