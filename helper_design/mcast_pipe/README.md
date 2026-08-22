# `mcast_pipe` helper rollout

Start here. This directory records the design and repository-wide rollout of the
kernel `mcast_pipe` helper and its paired host helper. The rollout was explicitly
resumed for the 2026-08-22 migration-feedback pass.

- Reviewed: 2026-08-22
- Branch/head at feedback intake: `sjovic/mcast-migration` / `cea14afbea9`
- Recorded branch baseline: `llk_helper_library` at `dc9282be7d5`
- Materialized helper API: v13
- Ledger write-back API: v13

## Four files that matter first

Read these in order:

1. **This README** — current state and the next logical action.
2. **[`migration/ledger.json`](migration/ledger.json)** — machine source of truth
   for every kernel and required host binding.
3. **[`migration/test_map.json`](migration/test_map.json)** — dispatch conditions
   and the tests that prove each route.
4. **[`changelog.md`](changelog.md)** — why the API and production integrations
   changed. Current code and `MCAST_PIPE_API_VERSION` win over old prose.

## Current state

The 2026-08-14 reconciliation found 91 existing call-site paths. The former
text inventory matched the ledger exactly and has now been folded into it:

| State | Kernels | Host bindings |
|---|---:|---:|
| migrated, verified at v13 | 31 | 27 |
| pending | 2 | 5 |
| deferred | 71 | 0 |
| quarantined | 0 | 0 |

The migrated fleet was updated to the v13 tagged, operation-first ABI and
append-style host bindings during the 2026-08-22 feedback pass. The host build,
36 helper host tests, all 80 helper device/wire tests under `--dev`, all 26
source audits, and focused sequential device gates passed. The v13-specific
gates cover present, absent, and chained helper blocks.
Exact evidence is recorded in `migration_feedback_tracker.md`.

Two pending kernels and five pending host bindings retain their existing status;
the feedback pass did not broaden the approved migration inventory.

## Current handoff

All items in `migration_feedback.md` are resolved and tracked. The block-sharded
Conv activation reader and all other deferred/pending units retain their prior
dispositions; resolving review feedback did not authorize migrating those units.

Current human views:

- [`migration/ledger.md`](migration/ledger.md) — concise ledger explanation.

Generated tier and rollout reports remain intentionally absent. The active
machine state is the v13 ledger plus test map, while the feedback tracker is the
execution record for this pass.

## Supporting evidence

- [`api_feedback.md`](api_feedback.md) — helper-contract review queue.
- [`migration_guardrails.md`](migration_guardrails.md) — durable rules distilled
  from completed migration feedback.
- [`migration/ledger.json`](migration/ledger.json) — both the durable call-site
  inventory and mutable rollout state. `design/primitive_contracts.md` supplies
  the recognition family used by reconciliation.
- `migration_audit/`, `kernel_annotations/`, and `migration/log/` — detailed
  classification, implementation, validation, and JIT evidence.
- [`design/`](design/) — still-valid contracts, hazards, feasibility analysis,
  and bake-off evidence; consult when changing the API or investigating a gap.
- [`proposed_helpers.md`](proposed_helpers.md) — the active helper proposal and
  migration classifications.
- [`archive/`](archive/) — completed plans and superseded reports retained for
  provenance, not as instructions for the next agent.

Generated dashboards are intentionally not kept. They duplicated a partial,
stale view of the ledger; derive status directly from `ledger.json` instead.
