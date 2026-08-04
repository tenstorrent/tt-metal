# `mcast_pipe` effort status

**Start here in a new session.** This page states the current rollout state,
the active review queue, and where to look next. Do not read the historical
design and migration artifacts unless a task below links to them.

- Last reviewed: 2026-08-04
- Branch: `sjovic/mcast-migration`
- Baseline: `origin/llk_helper_library` at `4a1d6a97ca9`
- Current helper API: v9

## Where the effort is now

The v9 rollout is reconciled through the width-sharded Conv migration at the
current branch HEAD. The machine-readable ledger records:

- 13 migrated kernel rows;
- 12 migrated host bindings;
- 78 deferred kernel rows;
- no pending or `needs_recheck` rows.

The completed migrations have correctness and JIT-path evidence, but the API
and migration review is not finished. Newly recorded feedback is intentionally
outside the old rollout report and is the active work queue.

## Next important work

1. Resolve the open API feedback before beginning another broad migration pass.
   API-001 and API-002 may change the CT/RT wire; API-003 changes signal-only
   handshake semantics. Settling those first avoids immediately revisiting new
   ports.
2. Implement API-004's offset-grid `Mcast1D`. Use matmul-2D in1 as the first
   production migration test and apply MIG-003's semaphore ownership and opaque
   CT/RT block insertion in the same atomic change.
3. Address the remaining migration-specific cleanup: MIG-001's Conv CT/RT
   offset chaining and MIG-002's sort row-start handshake.
4. Fix PERF-002 using its measured root cause: the per-send completion fence is
   the dominant SDXL VAE cost, followed by the send hot path not remaining fully
   inline. PERF-003's SegFormer width-sharded regression still needs equivalent
   ablation. PERF-001 found no regression for the rectangular SDXL GroupNorm
   model shape, but still needs deliberate wrapped-group coverage.

No priority below this list should be inferred from the order of historical
documents or ledger rows.

## Active review queue

- [`api_feedback.md`](api_feedback.md) — open helper-contract decisions
  (API-001 through API-004).
- [`migration_feedback.md`](migration_feedback.md) — concrete robustness issues
  in existing ports (MIG-001 through MIG-003).
- [`perf_feedback.md`](perf_feedback.md) — measured performance comparisons and
  open follow-ups (PERF-001 through PERF-003).

These three files are intake logs. When an item is resolved, update its status
there and record an implemented API change in `changelog.md`.

## Authoritative workflow state

- [`migration/ledger.json`](migration/ledger.json) — machine source of truth for
  per-kernel and host-binding migration status.
- [`census.txt`](census.txt) — production multicast/handshake inventory used by
  reconcile.
- [`primitive_contracts.md`](primitive_contracts.md) — authoritative primitive
  recognition family used by recall sweeps.
- [`migration/test_map.json`](migration/test_map.json) — durable test inventory
  and dispatch mapping.
- [`migration/ledger.md`](migration/ledger.md) — human-readable ledger view.
- [`migration/report.md`](migration/report.md) — latest completed rollout run
  summary. It does not supersede the active feedback queue above.

## Consult only when needed

These are retained because the helper workflows consume them or because they
preserve evidence behind existing decisions. They are not session entry points.

- `intent.md` — original scope; still used to classify recall-sweep exclusions.
- `hazards_catalog.md` — synchronization hazards consumed by migration review.
- `api_feasibility.md` — accumulated feasibility decisions and census-backed
  API analysis.
- `style_bakeoff.md` — historical on-device measurements and correctness
  decisions used as the migration baseline.
- `proposed_helpers.md` — superseded as an API description, but still the
  persisted Step-F artifact expected by the tune/apply workflow.
- `migration_audit/` and `kernel_annotations/` — classification evidence used by
  reconcile; not current-status prose.
- `migration/log/` — per-unit validation and JIT evidence.
- `migration/reconcile_*.md` — dated audit trail; the ledger points to relevant
  reports.
- `migration/tiers.md` — the latest generated rollout worklist and historical
  tier outcomes.
- `changelog.md` — API evolution history; use the implementation and API version
  as the current contract.
