# Independent P2 table-snapshot review — 2026-11-16

Scope: source-only review under AutoFix, independent of the implementation
author. Read `AUTODEBUG_P2.md`, the pending `tt/generator.py` diff, cache pool,
slot migration, prefill checkpoint, and focused CPU regression sources. No
runtime imports, tests, device operations or HTTP requests were performed by
this reviewer. This is not a stage-review verdict or a P2 performance pass.

## Finding and resolution

The initial shortcut accepted `remap is None`, which also admits calls from
`prefill_forward`. A reused slot can receive start-zero prefill with an unchanged
full page table and an end in the same page extent as its old cache position.
The shortcut then returns `None` before prefill calls `_reset_cache`, which
deletes `self.page_tables[slot]`. `_bind(None, ...)` does not restore the host
ownership snapshot. Shared KV reset leaves the device table allocated, so this
can escape the immediate forward and fail subsequent retained-slot validation.

Reported to the implementation author and root before approval. The revised
source adds private `_table_plan(..., reuse_decode_snapshot=False)` and enables
it only from `decode_forward`. This is stronger than inferring caller identity
from remap presence: prefill retains original normalization and snapshot binding
even with equal metadata. Reviewed the added same-table, same-page start-zero
reset/rebind regression and the AST assertion that only decode opts in.
The identified source defect is resolved in the reviewed revision.

Implementation-owner evidence `table_plan_snapshot_cpu_v4.json` records 14
passing focused CPU tests and no TTNN import. This reviewer inspected the saved
result and regression source but did not execute them. The final source uses
the explicit private flag described above; test-only formatting followed that
artifact. Hardware and P2 gate results are separate evidence.

## Other reviewed invariants

- `_bind` retains cloned, normalized per-slot tables. Equality compares complete
  submitted rows, not tensor identity or only the active prefix. A caller's
  mutable/strided table view therefore cannot silently change live IDs or future
  reservations. Changed invalid future IDs reach original bounds validation;
  changed valid reservations are retained through normal binding.
- Host tensor, integer dtype, dimensionality and submitted-row coverage checks
  precede the shortcut. Shape inequality, short tables, negative suffix padding,
  per-layer count mismatch and nonidentity remaps retain full validation. Per-
  layer tables remain authoritative over the common table.
- The guard compares each next live-page count with its current slot page count.
  Growing across a page boundary revalidates the complete retained-slot view,
  including inactive live requests. A previously legal future reservation that
  overlaps another live slot cannot become live through the shortcut.
- All fixed slots must still reference the generator's pool, and each nonempty
  slot must retain a mapping. Pool replacement is rejected after allocation;
  allocation clears warmup snapshots, and `prepare` freezes binding. Cache
  references keep the pool alive. This optimization creates no new ownership
  or allocation lifetime.
- Nonidentity remaps use the existing complete snapshot validation before state
  migration. Identity still passes `remap_slot_mapping` validation rather than
  relying only on equality (equal-valued float indices must be rejected).
- Authoritative decode repositioning within a page still calls `_bind` with
  positions and reloads host-selected tokens. Only unchanged table preparation
  is omitted. Position writes, recurrent state, sampling, device-token feedback,
  trace capture and readback behavior are unchanged.
- With prefill excluded, odd-tail checkpoint restore and start-zero resets keep
  their original paths. Checkpoints deliberately exclude page tables and KV;
  the current scheduler submission remains their ownership authority.

The unchanged-snapshot premise relies on generator-owned snapshots and fixed
pool specs remaining private implementation state; callers may mutate their
submitted tensors, but must not mutate generator internals. This is the existing
ownership contract, not a new restriction introduced by the optimization.

## Source references

All model paths are relative to `models/autoports/qwen_qwen3_8_flash_next/`:

- `tt/generator.py`: `_resolve_caches`, `allocate_kv_cache_per_layer`,
  `_reset_cache`, `_table_plan`, `_bind`, `prefill_forward`, `decode_forward`.
- `tt/cache_pool.py`: `SharedKVPool.bind`, `freeze`, `normalize_tables`,
  `validate_active_pages`.
- `tt/slot_state.py`: `validate_remap`, `remap_slot_mapping`, `SlotStateMover`.
- `tt/prefill_checkpoint.py`: `compact_state`, `can_restore`, `restore`.
- `tests/test_table_plan_snapshot_cpu.py`: real-method AST extraction and
  mutation, retained-overlap, ownership, dtype, row routing and remap controls.

No additional required source correction was identified beyond the prefill
reset/rebind finding. CPU benchmark savings alone cannot establish the P2
end-to-end threshold, hardware state equivalence or full48 serving correctness.
Those remain the root's serialized validation work.
