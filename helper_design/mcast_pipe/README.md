# `mcast_pipe` helper rollout

Start here. This directory records the design and repository-wide rollout of the
kernel `mcast_pipe` helper and its paired host helper. The rollout is paused; do
not start migration or change ledger status unless the user explicitly resumes it.

- Reviewed: 2026-08-14
- Branch/head after experimental rollback: `sjovic/mcast-migration` / `9d870bf2da9`
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

The 2026-08-14 reconciliation found 91 existing call-site paths. The former
text inventory matched the ledger exactly and has now been folded into it:

| State | Kernels | Host bindings |
|---|---:|---:|
| migrated, recorded at v10 | 17 | 14 |
| pending | 3 | 9 |
| deferred | 71 | 0 |
| quarantined | 0 | 0 |

The migrated fleet is paper-stale because the helper is v11; this does not mean
the current source is known broken. The 2026-08-14 intake host build, 32
host-helper tests, and 80 helper device/wire tests passed before the experimental
Conv rollback. Current rollback validation is recorded in the changelog. These
checks do not replace mapped per-operation validation required for write-back.

Three pending kernels are already integrated in source: Matmul in0 sender,
receiver, and block-sharded hybrid. Their nine required factory bindings are
represented explicitly in the ledger and test map. Two migrated kernels also
carry `needs_recheck`; see
[`migration/ledger.md`](migration/ledger.md).

## Next logical action — only when migration resumes

Re-enter the apply workflow from the reconciled v11 state. First verify/stamp the
v10 fleet and clear the two `needs_recheck` flags, then validate the pending
Matmul units under their mapped inventories. The block-sharded Conv activation
reader remains deferred on the R4 streaming design gap. No apply run is currently
approved and no run mode has been selected.

Current human views:

- [`migration/ledger.md`](migration/ledger.md) — concise ledger explanation.

Generated tier and rollout reports are intentionally absent while migration is
paused. `apply-dm-helper` must regenerate them from the current ledger and test
map after its intake and planning gates.

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
