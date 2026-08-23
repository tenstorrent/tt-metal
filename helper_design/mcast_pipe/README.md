# `mcast_pipe` helper rollout

Start here. This directory records the design and repository-wide rollout of the
kernel `mcast_pipe` helper and its paired host helper. The rollout was explicitly
resumed for the 2026-08-22 migration-feedback pass.

- Reviewed: 2026-08-23
- Branch/head at feedback intake: `sjovic/mcast-migration` / `cea14afbea9`
- Recorded branch baseline: `llk_helper_library` at `dc9282be7d5`
- Materialized helper API: v14
- Ledger write-back API: v14

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
| migrated, verified at v14 | 31 | 27 |
| pending | 2 | 5 |
| deferred | 71 | 0 |
| quarantined | 0 | 0 |

The migrated fleet was updated to the v14 template-owned runtime-base ABI during
the 2026-08-23 feedback pass. Runtime-sized and genuinely optional compile-time
operation tails now follow the opaque helper block and derive their start from
it; fixed-width layouts retain ordinary helper tails. The final Conv review
also restored operation terminal drains, removed migration-only source-lifetime
synchronization, clarified independent input ownership, and verified dense
versus divergent ACK-count policy. The host build, 36 helper host tests, all 80
helper device/wire tests under `--dev`, all 32 source audits, and focused
sequential Matmul, Conv2D, and Conv3D device gates passed. The inherited v13
gates still cover present, absent, and chained helper blocks.
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
machine state is the v14 ledger plus test map, while the feedback tracker is the
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
