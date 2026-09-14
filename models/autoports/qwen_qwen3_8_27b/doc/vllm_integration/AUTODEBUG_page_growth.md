# AutoDebug: async decode page growth

## Scope and verdict

Source-only inspection on 2026-09-14. The inspected vLLM checkout is
`/home/mvasiljevic/qwen38-full-rerun/vllm` at
`03fa3af2e15b5f8dc07cbaa67d92f979aa00be11`; tt-metal is at
`3f48cac393ac8609517b9e8dee76828a87821e5d`. Both working trees were clean when
inspection began. No target execution, hardware access, server operation, or
implementation edit was performed. This document is the only authored artifact.

**The missing layout invalidation on block append is real, but does not by itself
prove a stale-page bug.** The pinned runner rebuilds and forwards current CPU
page tables on every decode, including steady decode. The current shared
tt-metal generator explicitly refreshes only the page-table trace input when its
contents change. The autoport's existing `QwenGenerator` already provides the
same separation. An adapter that preserves this separation does not need a runner
change for ordinary page growth.

The smallest supported intervention is therefore in the new adapter: always
forward the current, normalized CPU page table to `QwenGenerator.decode_forward`,
independently of `reset_batch`; suppress host token/position inputs while the
device owns steady decode state. Actual host-input resets must occur only after
the runner finalizes and applies pending decode outputs. Do not turn every block
append, or every non-`None` allocation payload, into a full input reset.

This verifies the source contract and identifies a concrete failure mechanism if
the adapter violates it. It is not evidence that an existing autoport serving run
has failed, or that a repair has passed on hardware: no `generator_vllm.py` existed
in this autoport at the start of the investigation, and no failing runtime log was
provided to this investigator.

## Direct observations

Paths below prefixed `plugin/` mean
`vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/`; `autoport/` means
`tt-metal/models/autoports/qwen_qwen3_8_27b/`.

1. `plugin/model_runner.py:660–684` updates cached block IDs and appends new IDs
   to the persistent batch's block table. This does not mark
   `_decode_layout_changed_since_last_decode`. Only removal, addition, and
   condensation set the local layout flag, consumed at lines 700–703.
2. `plugin/async_decode.py:249–291` does not reject a steady-decode candidate
   merely because a cached request received new blocks. It does reject actual
   layout changes, prompts/resumes, and unsupported sampling constraints.
3. `plugin/model_runner.py:1027–1030` calls `block_tables_for_rows` on every input
   preparation. That helper, `plugin/input_batch.py:648–667`, clones current CPU
   rows and normalizes their width. `plugin/async_decode.py:571–583` then passes
   these tables to the model on every submission. No steady-decode branch
   bypasses this path or substitutes a cached `TTModelInput`.
4. `plugin/model_runner.py:1126–1135` reads host tokens and positions and puts the
   layout flag in `reset_batch`. These host inputs can lag device feedback while
   an async output is pending. Their presence in the arguments does not establish
   their authority to overwrite device state.
5. `autoport/tt/generator.py:462–492` refreshes the page table unconditionally,
   but writes tokens only when `tokens is not None`, and positions/RoPE only when
   `start_pos is not None`. `_refresh_table`, lines 219–233, compares values,
   skips unchanged tables, writes into the existing bound device tensor, and
   clones the new host snapshot. Passing an unchanged CPU table each step incurs
   host comparison, not a device table write. A device tensor argument instead
   asserts that the caller already maintains the bound tensor in place; passing
   it while neglecting the current CPU table would lose allocator updates.
6. The sibling shared generator already follows this contract:
   `tt-metal/models/tt_transformers/tt/generator.py:2210–2243` compares page-table
   values and performs a page-only copy when `reload_inputs` is false. Its comment
   explicitly identifies async host token/position lag. This is a direct
   counterexample to the claim that the pinned runner's absent layout flag
   necessarily produces stale device tables.
7. `plugin/model_runner.py:2348–2358` applies ready outputs and drains pending
   outputs before a predicted non-steady step. `build_model_input`, lines
   1360–1378, additionally drains after `_update_states` discovers an unexpected
   layout change and **before** preparing host inputs. Thus removals and
   condensation discovered after the initial candidate decision are covered.
8. `plugin/async_decode.py:377–387` drives `ensure_finalized()` on each pending
   output, then applies completed output to runner state. An event-only wait is
   insufficient: deferred output may not yet have been requested by the engine.
   `DeferredDecodeOutput.ensure_finalized`, lines 102–110, protects readback with
   a lock and caches the result; subsequent engine consumption must not read the
   same submission twice. State application at
   `plugin/model_runner.py:3246–3269` resolves the current request row and checks
   request identity before appending the sampled token.
9. `vllm/v1/core/sched/scheduler.py:429–436` obtains allocations from
   `KVCacheManager.allocate_slots`; lines 1041–1043 serialize the result through
   `get_block_ids(allow_none=True)`. The latter returns `None` when all groups are
   empty (`vllm/v1/core/kv_cache_manager.py:78–80`). In mixed group payloads,
   some groups may still be empty. `BlockTable.append_row` itself ignores an
   empty list (`vllm/v1/worker/block_table.py:100–116`). Truthiness of an outer
   tuple such as `([],)` is not a valid test for actual allocation growth.

## Causal model and repair hypothesis

Consider a continuing request whose device state is `(next_token, position=p)`
and whose host state still reflects the previous output. The scheduler allocates
the physical block for logical page `floor(p / P)` and the runner appends its ID.
The runner correctly supplies the new table while leaving `reset_batch=False`.

- If the adapter ignores tables while `reset_batch=False`, the trace reads its
  previous device table and can address an old or padding block. The first
  semantic failure is expected near the first newly used logical page, although
  identical padding/physical IDs or insensitive output tokens can mask it.
- If the adapter refreshes all host inputs whenever the table changes, it can
  overwrite `next_token` and `p` with lagging host values. The apparent page fix
  then introduces duplicate, skipped, or position-shifted decode work.
- If it forwards the new table but passes `tokens=None, start_pos=None` for a
  continuing steady step, the existing generator refreshes the mapping while
  preserving the device token/position chain. Unchanged table values do not
  trigger a device write.

The adapter should keep fixed page-table geometry and device tensor identity
across allocations. Normalize scheduler group IDs into the autoport's physical
page IDs before comparing or copying, including every full-attention group used
by the model. Keep a cloned host snapshot if the adapter adds its own change
detection. Do not compare Python object identity: the runner intentionally
produces a new tensor object each step.

For first decode, post-prefill transition, actual admission/removal/remap, or an
explicit supported reset, host inputs may be required. The adapter must not
invent a late reset based only on page growth after the runner's drain decision.
If any adapter-owned condition requires host authority beyond the existing
runner flags, expose that condition before input preparation or explicitly
finalize and apply pending outputs before preparing those inputs. Simply waiting
for a device event after stale host tensors have already been constructed does
not repair them.

No runner patch is justified by the source evidence for this page-growth case.
If an alternate implementation nevertheless uses the layout flag to force a
full reset on growth, it must drain before preparation and detect actual IDs
with a group-aware predicate such as
`new_block_ids is not None and any(new_block_ids)`. That is a broader workaround
with extra synchronization, not the preferred page-only adapter design.

## Focused host verification to implement

Use the actual runner/controller methods with a fake input batch and a spy
autoport generator; replace device copies/replays with recorded effects. Do not
duplicate the algorithm in an independent toy model and call that integration
coverage. The existing plugin `tests/test_state_slots.py` demonstrates invoking
runner methods against `SimpleNamespace` state without device execution.

| Test | Setup and required assertions |
| --- | --- |
| Real page growth during steady decode | Seed a cached request with one block, a false layout flag, and an unresolved output. Feed a `ScheduledCachedRequestData` allocation containing a distinct new block. Invoke actual state update/input preparation/submission and the adapter. Assert the model receives the new normalized table; exactly the table device buffer changes; token, position, and RoPE writes remain zero; trace identity remains stable. Use deliberately different stale host tokens/positions so an accidental reload is observable. |
| Empty and unchanged allocations | Cover `None`, `([], )`, multiple empty groups, and a newly cloned but value-identical CPU table. No table write, token/position write, or reset may be introduced. A payload with one nonempty group and other empty groups must refresh only the relevant mapping. |
| Snapshot ownership | After one refresh, mutate the caller's original CPU tensor in place. The next call must detect changed values; previously saved state must not alias the caller. Also verify that an unrelated new tensor with equal values is a no-op. |
| Late real layout change | Have the initial steady candidate return true, then make actual `_update_states` discover removal/condensation. Leave a pending output whose finalization appends a distinct token. Assert event order `finalize -> apply token -> prepare host inputs -> reset decode`; the reset must consume the newly appended token and its matching position. |
| Deferred finalization exactly once | Let both engine consumption and reset draining resolve the same wrapper. Assert one readback and one state application, with the second consumer receiving cached output. This protects the reset path from duplicated tokens. |
| Submission output lifetime | Submit two steady decodes without host finalization between them using distinguishable generated tokens. Each deferred read must refer to its own submitted output, not the latest contents of a shared feedback buffer. Assert output ordering and values after both resolve. |

The page-growth tests should fail for two temporary test-local mutations:
dropping table forwarding when `reset_batch=False`, and forwarding host
tokens/positions on every page change. This proves the tests distinguish the two
failure mechanisms instead of only observing a copy counter.

## Required allocator-driven runtime control

Host tests establish forwarding and ordering, not device correctness. Before this
serving contract passes, run a small fixed-seed greedy request through the real
pinned vLLM scheduler/allocator and the actual autoport adapter, with async
scheduling and device sampling enabled. Keep cache dtype, compute fidelity,
attention policy, and page geometry identical between controls.

1. Derive scheduler allocation block size, native page size, KV group mapping,
   and page-table width from the actual integration. The current autoport native
   page size is 32 (`tt/model.py:143–148`); do not assume vLLM's group block size
   equals it after hybrid-cache unification. Use an exact token-count prompt
   near a block boundary and enough output to cause at least two subsequent
   **real allocator extensions** for the same continuing request. For an actual
   32-token allocation block, a 31-token prompt and at least 68 generated tokens
   is a compact example. Confirm allocations in the trace rather than inferring
   them from requested lengths.
2. Record scheduler step, request ID, allocated IDs by group, logical/native
   page index, authoritative absolute decode position, normalized table values
   before/after refresh, and copy/reset/drain counters. Require at least one
   allocation event with unchanged request layout and `reset_batch=False`.
   Table copies must follow value changes, not number of generated tokens.
3. Compare exact generated token IDs against a control that finalizes pending
   output before each next step but uses the same dynamic allocator, adapter,
   device sampling, and precision. At the same time compare with the validated
   full-model/reference correctness contract; agreement between two equally
   stale adapters alone is insufficient. Check the entire output stream, length,
   and first divergence index, not just successful HTTP completion or fluent text.
4. Add a second request that remains active across the first request's block
   growth; then finish or admit a request to exercise a genuine reset with
   pending output. Verify unaffected request state, no duplicate/omitted tokens,
   and reset finalization order. Finalize all outputs before declaring success.
5. Keep a preallocated-capacity or allocator-over-allocation control as a
   localization aid, not the page-growth acceptance run. Existing standalone
   `QwenGenerator.generate` allocates the complete request capacity up front
   (`tt/generator.py:611`), so its zero-table-write steady-state tests cannot
   establish dynamic allocator behavior.

An additional coverage issue must remain separate from the page-refresh claim.
TP4 policy currently sets `sdpa_k=128`
(`tt/multichip_decoder.py:61`), while native pages hold 32 tokens. The effective
decode chunk can change with table geometry
(`tt/optimized_decoder.py:747–770`). Log the lowered chunk size `K` and compare
allocated native pages with the attention read-window bound
`ceil((p + 1) / K) * K / P`, as well as the minimum write page `floor(p / P)`.
If failure persists at a page/chunk boundary after verifying the actual uploaded
mapping, use a same-dtype allocator-over-allocation control before attributing it
to numerical precision. Whether this kernel actually consumes unmapped future
page entries in the selected configuration requires lower-level/runtime evidence;
this report does not assert that an allocation-coverage bug has been proved.

## Inspection checks performed and remaining uncertainty

Read the relevant source paths with `rg` and numbered `sed` views; confirmed both
commit IDs and initial clean status with `git rev-parse HEAD` / `git status --short`.
A `python` AST-only check (no target imports or execution) confirmed:

```text
PASS: block append 684 does not set layout flag; flag assignment only 703.
PASS: model input table rebuilt 1028 and passed to decode independently of reset_batch.
PASS: QwenGenerator refresh_table 462 runs before token-only conditional 463.
PASS: discovered layout reset drains 1375 before host input preparation 1378.
```

Re-reviewed the headline against the shared generator's page-only branch. This
refuted the broad runner-only diagnosis and narrowed the necessary intervention
to preserving the adapter's page-refresh/token-state separation. The host tests
and runtime controls above are proposed follow-up work, not executed results.
Device queue ordering, per-submission token readback ownership, normalized hybrid
page geometry, and allocator-driven output correctness remain unverified here.
