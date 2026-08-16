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

### C2 — Appendix-B reconciliation schema and classifications

The first broad Claude invocation was terminated after ten minutes without output. A bounded retry with
the same required command returned `REVISE`; all corrections were incorporated:

- Use exact controlled-vocabulary family names.
- Classify the programming Matmul receiver as `matmul / receiver / clean`, not an example `ref`.
- Classify the two Matmul Decode readers as `refactor-high` pending their prototype-first Tier-3 gate,
  rather than prejudging them `defer`.
- Classify Quasar in0 receiver twins `refactor-high` to match current production truth; keep Quasar in1
  and Conv weight receivers `clean`.
- Reuse/extend existing family annotations; create only the genuinely new Matmul Decode protocol document.
- Keep deferred factories out of `host_bindings` and document their complete companion map in the report.

Claude confirmed the 13 paths exactly match Appendix B, both support headers are correctly excluded from
ledger rows, no 14th candidate exists, and there is no integrity blocker.

### C3 — Tier 0 API-v11 and write-back gate

The first bounded Tier 0 review was terminated after six minutes without output. A narrower retry with
the same required model returned PASS:

- clear all 12 `needs_recheck` flags and stamp the existing 17-kernel / 14-binding fleet at API v11;
- migrate Tier 0.2's hybrid kernel and four bindings at API v11 using the plan-authorized inherited
  matched medians (+0.643% and -0.045%);
- keep Tier 0.1's two kernels and five bindings pending because its route-specific historical baseline
  requires the separately authorized checkout that was not performed;
- no correctness, API, LOC, or integrity issue blocks those exact ledger mutations.

Claude requested that the 302/188 Matmul result explicitly state that the mapped
`MM-IN0-INTERLEAVED` and `MM-BLOCK-SHARDED-HYBRID` nodes were included; the tracker and ledger now do so.

### C4 — Tier 1.6 DeepSeek sampling barrier

Claude returned PASS/KEEP with no API expansion for each material decision:

- a fixed-sender, fully-inside `Mcast2D` with `handshake=false` preserves the legacy signal-only
  `EXCLUDE_SOURCE` fan-out;
- using the dense 11x10 bounding rectangle preserves the raw 109-destination multicast while
  initializing the helper semaphore on all landed cells; kernels and per-core runtime arguments remain
  restricted to the sparse 101 active cores;
- common and per-core NCRISC runtime argument spaces are separate, so `McastArgs<0, 0>` is correct;
- leaving the helper-owned sender-local Flag at VALID is safe dead state because the fixed sender never
  receives or reads it, and every signal rewrites it;
- an unskipped 100-iteration top-k run can clear the barrier gate only if its selection failure exactly
  matches the raw baseline. That condition was met byte-for-byte for the decoded `p_indices` and
  `p_scores` vectors and assertion signature.

Two broader final write-back prompts and two compact retries were terminated after returning no output.
No approval was inferred from those timeouts; write-back follows the earlier explicit KEEP verdict after
its raw-baseline condition was satisfied. API expansion: NO.

## Progress

| Unit | State | Current finding / next gate |
|---|---|---|
| Reconciliation | complete | 104 unique entries preserved; current rollout state is 19 migrated, 2 pending, 83 deferred |
| Tier 0.1 Matmul in0 interleaved | blocked-writeback | Correctness/JIT verification passed; historical matched performance checkout is not authorized |
| Tier 0.2 Matmul in0 block-sharded | complete | Migrated API v11 after correctness, fresh-JIT, inherited performance, and Claude gates passed |
| Tier 1.6 DeepSeek sampling | complete | API v11 at `2840fc28361`; 101-core correctness/cache, raw-signature top-k barrier proof, and matched performance passed |
| Tier 2.7 DRAM-sharded Matmul | pending | Required behavior and test collection not started |
| Tier 2.8 group-attention Matmul | pending | Required behavior and post-flag-barrier proof not started |
| Tier 2.9 Conv3D weight sharing | pending | Required behavior and test collection not started |

## Chronological findings

1. Loaded the mandatory helper-specific context before task actions.
2. Confirmed the working branch baseline is `origin/llk_helper_library`, not `main`; no baseline rewrite
   is authorized or planned.
3. Confirmed `helper_design/mcast_pipe/plan.md` already records user approval dated 2026-08-16,
   including D4/D5 deferrals.
4. Reconciliation matched the approved boundary exactly. Added 13 kernel/receiver rows, kept two
   `conv_reader_common.hpp` headers support-only, regenerated the deferred factory map in
   `archive/reconciliation/reconcile_2026-08-16-plan-inventory.md`, and preserved all existing rollout state.
5. Rebuilt the current checkout with `./build_metal.sh`; the Release build passed. No production source
   was changed during Tier 0 verification.
6. Ran one exact `--dev --no-precompile` operation node for each Tier 0 route. Matmul 2D in0 multicast,
   GroupNorm legacy, LayerNorm pre-allgather, TopK, Sort, Conv height-sharded, Conv block-sharded, and
   Conv width-sharded all passed. The isolated caches contained the expected sender/receiver artifacts
   for Matmul, GroupNorm, LayerNorm, TopK, Sort, and the Conv 1D/2D weight routes. The width and
   block-sharded probes were repeated with the runtime's canonical `TT_METAL_CACHE` variable; both
   reported zero cache hits and contained their exact width-activation and hybrid Matmul artifacts.
7. Revalidated the helper fleet: `test_mcast_pipe.py` 80 passed, source audit 17 passed, and
   `McastHostFixture.*` 32 passed.
8. Ran the complete mapped Tier 0 operation inventories sequentially through `run_safe_pytest.sh`:
   Matmul 302 passed / 188 expected skips (including `MM-IN0-INTERLEAVED` and
   `MM-BLOCK-SHARDED-HYBRID`); sparse Matmul 11 passed; GroupNorm 235 passed / 10
   device-shape skips; distributed LayerNorm 190 passed / 10 pre-existing non-running xfails; sharded
   LayerNorm 280 passed; TopK 25 passed / 12 expected BFLOAT8_B pad xfails; Sort 9 passed; Conv DRAM
   routes 17 passed. The three exact Conv feature nodes also passed.
9. Tier 0.1 cannot satisfy its performance gate without the plan's separately authorized reversible
   historical checkout at `45033178088b`; no checkout was attempted. Tier 0.2 has the plan-approved
   inherited matched medians (+0.643% and -0.045%).
10. After Claude C3, advanced `ledger.json.current_api_version` from 10 to 11, stamped all 17 existing
    migrated kernel rows and 14 existing migrated host bindings at v11, cleared the 12 `needs_recheck`
    flags, and migrated Tier 0.2's one kernel plus four bindings. Tier 0.1 remains unchanged and pending.
11. Migrated Tier 1.6 in `2840fc28361`: replaced the raw single-device loop barrier with a helper-owned
    fixed-sender Flag pipe; removed manual physical-coordinate conversion and five named compile-time
    arguments; kept the mesh path and the operation-owned global semaphore protocol unchanged. Production
    files shrank independently (kernel 36 deletions / 9 additions; host 22 deletions / 10 additions).
12. Tier 1.6 validation: Release build passed; cold `--dev` argmax passed with 0/523 JIT hits; the mapped
    normal suite reported 4 argmax passed and the repository's 3 existing Blackhole top-k skips, with
    533/533 warm JIT hits. The temporarily unskipped 100-iteration top-k node completed without a hang,
    selected the expected index 85, and then reproduced the raw baseline's exact pre-existing `p_scores`
    mismatch. Final source audit: 17 passed.
13. Tier 1.6 matched Tracy device-kernel durations: argmax raw 18,789 ns vs migrated 18,836 ns (+0.25%);
    top-k raw 1,558,235 ns vs migrated 1,557,464 ns (-0.05%). The original test skip was restored and
    the test file is unchanged.
