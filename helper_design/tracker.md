# `mcast_pipe` unattended rollout tracker

Date: 2026-08-16
Branch start: `sjovic/mcast-migration` at `db246d49b89978d436d944d57f0ba326ef698416`
Plan: `helper_design/mcast_pipe/plan.md`

This is the resumption-oriented progress log requested by the user. The machine sources of truth remain
`helper_design/mcast_pipe/migration/ledger.json`, `test_map.json`, and the per-unit logs.

## Authorized scope

- Tier 0 units 1-2.
- Tier 1 unit 6.
- Tier 2 units 7-9 only.
- API v11 as-is. Any API extension remains separately approval-gated by the plan.
- No rebase, push, reset, worktree, or historical checkout.

## Initial repository state

- The user named `helper_design/plan.md`; the only applicable plan is
  `helper_design/mcast_pipe/plan.md`, which names this exact branch/HEAD and the requested tiers.
- Existing user change preserved: `tt_metal/third_party/tt_ops_code_gen` is dirty at submodule commit
  `7e974cd3ecd46e8d06f831f27686f78e4c056a71` instead of recorded `4860704b721a2938e3dcc2dc00634f73dcd9513d`.
- Ledger: API v10; helper source: API v11; 91 kernel entries, 23 host bindings, 12 migrated kernels
  flagged `needs_recheck`.
- Plan reconciliation additions: 13 kernel/receiver inventory paths plus two shared Conv support headers
  that remain atomic-scope dependencies rather than ledger rows.

## Claude consultations

### C1 — plan identity and unattended approval gate

Claude independently read the plan, ledger, and reconciliation skill. Recommendation:

- Treat `helper_design/mcast_pipe/plan.md` as the intended plan and create this tracker at the literal
  requested root path.
- The current instruction satisfies the plan implementation gate.
- Apply reconciliation automatically only when the audit exactly matches or is a strict subset of
  Appendices A/B. Do not apply any unexpected removed/clobbered row or new disposition unattended.
- Tier 0 unit 1 may be verified, but its historical performance checkout and final write-back remain
  blocked because the plan explicitly requires separate authorization for that checkout.
- Preserve the dirty submodule; use no rebase/push/reset/worktree; stop a unit on API, LOC,
  correctness, coverage, or performance gate failure and leave no half-migrated production edits.

## Progress

| Unit | State | Current finding / next gate |
|---|---|---|
| Reconciliation | in progress | Auditing current tree against plan Appendices A/B before pre-approved write-back |
| Tier 0.1 Matmul in0 interleaved | pending | Historical matched performance checkout is not authorized; verification can proceed |
| Tier 0.2 Matmul in0 block-sharded | pending | Matched performance evidence exists in archived 2026-08-07 plan |
| Tier 1.6 DeepSeek sampling | pending | Required behavior and test collection not started |
| Tier 2.7 DRAM-sharded Matmul | pending | Required behavior and test collection not started |
| Tier 2.8 group-attention Matmul | pending | Required behavior and post-flag-barrier proof not started |
| Tier 2.9 Conv3D weight sharing | pending | Required behavior and test collection not started |

## Chronological findings

1. Loaded the mandatory helper-specific context before task actions.
2. Confirmed the working branch baseline is `origin/llk_helper_library`, not `main`; no baseline rewrite
   is authorized or planned.
3. Confirmed `helper_design/mcast_pipe/plan.md` already records user approval dated 2026-08-16,
   including D4/D5 deferrals.
