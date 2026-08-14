# `mcast_pipe` helper rollout

Start here. This directory records the design and repository-wide rollout of the
kernel `mcast_pipe` helper and its paired host helper. The rollout is paused; do
not start migration or change ledger status unless the user explicitly resumes it.

- Reviewed: 2026-08-14
- Branch/head at reconciliation: `sjovic/mcast-migration` / `9686814ea22`
- Baseline: `origin/llk_helper_library` at `4a1d6a97ca9`
- Materialized helper API: v11
- Ledger write-back API: v10

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

The 2026-08-14 reconciliation found 91 existing census paths with an exact
census/ledger match:

| State | Kernels | Host bindings |
|---|---:|---:|
| migrated, recorded at v10 | 17 | 14 |
| pending | 4 | 10 |
| deferred | 70 | 0 |
| quarantined | 0 | 0 |

The migrated fleet is paper-stale because the helper is v11; this does not mean
the current source is known broken. The current host build, 32 host-helper tests,
and all 80 helper device/wire tests passed on 2026-08-14. Those intake checks do
not replace the mapped per-operation validation required for ledger write-back.

Four pending kernels are already integrated in source: Matmul in0 sender,
receiver, and block-sharded hybrid, plus block-sharded Conv2D activation. Their
ten required factory bindings are now represented explicitly in the ledger and
test map. Three migrated kernels also carry `needs_recheck`; see
[`migration/ledger.md`](migration/ledger.md).

## Next logical action — only when migration resumes

Re-enter the apply workflow from the reconciled v11 state. First verify/stamp the
v10 fleet and clear the three `needs_recheck` flags, then validate the pending
Matmul and Conv units under their mapped inventories. No apply run is currently
approved and no run mode has been selected.

Current human views:

- [`migration/ledger.md`](migration/ledger.md) — concise ledger explanation.
- [`migration/tiers.md`](migration/tiers.md) — prepared future work order.
- [`migration/report.md`](migration/report.md) — latest reconciliation and intake
  result, not a completed v11 rollout report.
- [`migration/reconcile_2026-08-14.md`](migration/reconcile_2026-08-14.md) — exact
  reconciliation evidence.

## Supporting evidence

- [`api_feedback.md`](api_feedback.md) — helper-contract review queue.
- [`migration_guardrails.md`](migration_guardrails.md) — durable rules distilled
  from completed migration feedback.
- `census.txt` and `primitive_contracts.md` — recognition inventory used by
  reconciliation.
- `migration_audit/`, `kernel_annotations/`, and `migration/log/` — detailed
  classification, implementation, validation, and JIT evidence.
- `intent.md`, `hazards_catalog.md`, `api_feasibility.md`, `style_bakeoff.md`, and
  `proposed_helpers.md` — design inputs and historical rationale; consult when a
  current task points to them.
- [`archive/`](archive/) — completed plans and superseded reports retained for
  provenance, not as instructions for the next agent.

Generated dashboards are intentionally not kept. They duplicated a partial,
stale view of the ledger; derive status directly from `ledger.json` instead.
