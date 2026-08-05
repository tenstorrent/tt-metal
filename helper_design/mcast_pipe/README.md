# `mcast_pipe` effort status

**Start here in a new session.** This page states the current rollout state,
the active review queue, and where to look next. Do not read the historical
design and migration artifacts unless a task below links to them.

- Last reviewed: 2026-08-05
- Branch: `sjovic/mcast-migration`
- Baseline: `origin/llk_helper_library` at `4a1d6a97ca9`
- Current helper API: v10

## Where the effort is now

The v10 rollout is reconciled through the final release gate at the current
branch HEAD. The machine-readable ledger records:

- 13 migrated kernel rows;
- 12 migrated host bindings;
- 78 deferred kernel rows;
- no pending or `needs_recheck` rows.

The completed migrations have correctness, fresh-JIT, and matched performance
evidence. All seven gates in `mcast_feedback_plan_2026-08-04.md` are green.

## Remaining review work

API-002 face metadata and RT compaction remain intentionally open and were
explicitly deferred from this rollout. API-001, API-003 through API-006, and
MIG-001 through MIG-004 are implemented. Any future API-002 work should begin
as a separate planned wire revision rather than extending this completed gate
sequence.

## Active review queue

- [`api_feedback.md`](api_feedback.md) — open helper-contract decisions
  plus implemented contracts that future migrations must preserve.
- [`migration_feedback.md`](migration_feedback.md) — concrete robustness issues
  and migration-specific validation gaps.

These two files are the active review queue. When an item is resolved, update
its status there and record an implemented API change in `changelog.md`.

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
