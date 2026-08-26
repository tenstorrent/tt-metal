# `mcast_pipe` helper rollout

Start here. This directory records the design and repository-wide rollout of the
kernel `mcast_pipe` helper and its paired host helper. The rollout was explicitly
resumed for the 2026-08-22 migration-feedback pass.

- Reviewed: 2026-08-23
- Branch/head at feedback intake: `sjovic/mcast-migration` / `cea14afbea9`
- Recorded branch baseline: `llk_helper_library` at `e6d0562cfaa` (rebased 2026-08-26; was `dc9282be7d5`)
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

The 2026-08-23 repository-wide recall reconciled 108 call-site paths. It added
four previously omitted TT-Train kernel faces without changing any existing
rollout disposition:

| State | Kernels | Host bindings |
|---|---:|---:|
| migrated, verified at v14 | 31 | 27 |
| pending | 2 | 5 |
| deferred | 75 | 0 |
| quarantined | 0 | 0 |

The migrated fleet was updated to the v14 template-owned runtime-base ABI during
the 2026-08-23 feedback pass. Runtime-sized and genuinely optional compile-time
operation tails now follow the opaque helper block and derive their start from
it; fixed-width layouts retain ordinary helper tails. The final Conv review
also restored operation terminal drains, removed migration-only source-lifetime
synchronization, clarified independent input ownership, and verified dense
versus divergent ACK-count policy. Kernel offset chaining now uses an existing
named constexpr helper object wherever one is available, without changing the
static API or wire. The host build, 36 helper host tests, all 80 helper
device/wire tests under `--dev`, all 33 source audits, and focused sequential
Matmul, Conv2D, Conv3D, Move, GroupNorm, and LayerNorm device gates passed. The inherited v13
gates still cover present, absent, and chained helper blocks.
Exact evidence is recorded in `migration_feedback_tracker.md`.

Two pending kernels and five pending host bindings retain their existing status;
the feedback pass did not broaden the approved migration inventory.

## Current handoff

The feedback tracker is complete through MCAST-007. Two later review items remain
open in `migration_feedback.md`: GROUP-ATTN-MATMUL-001 and SDPA-DECODE-001.
The block-sharded Conv activation reader, newly inventoried TT-Train families,
and all other deferred/pending units retain their prior dispositions; this
documentation reconciliation did not authorize migrating them.

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
- `migration_audit/`, `kernel_annotations/`, and `migration/log/` — historical
  classification, implementation, validation, and JIT evidence. Their local
  READMEs define how to interpret dated status claims.
- [`design/`](design/) — still-valid contracts, hazards, feasibility analysis,
  and bake-off evidence; consult when changing the API or investigating a gap.
- `ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp` and
  `ttnn/cpp/ttnn/kernel_lib/host/mcast_host.hpp` — the materialized API v14
  contract; superseded intent and proposal documents are archived.
- [`archive/`](archive/) — completed plans and superseded reports retained for
  provenance, not as instructions for the next agent.

Generated dashboards are intentionally not kept. They duplicated a partial,
stale view of the ledger; derive status directly from `ledger.json` instead.
