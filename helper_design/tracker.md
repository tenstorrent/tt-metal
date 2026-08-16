# `mcast_pipe` unattended rollout tracker

Date: 2026-08-16
Branch start: `sjovic/mcast-migration` at `db246d49b89978d436d944d57f0ba326ef698416`
Plan: `helper_design/mcast_pipe/plan.md`

This is the resumption-oriented progress log requested by the user. The machine sources of truth remain
`helper_design/mcast_pipe/migration/ledger.json`, `test_map.json`, and the per-unit logs.

## Authorized scope

- Tier 0 units 1-2.
- Tier 1 unit 6.
- Tier 2 units 7-9 in the initial authorization.
- Tier 2 units 10-16, including independent units 16a-16c, authorized by the user's 2026-08-16
  continuation instruction after units 7-9 completed.
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

### C5 — Tier 2.7 DRAM-sharded Matmul

The first architecture prompt and a bounded retry returned no output. A second, fact-complete bounded
retry returned DEFER for API v11 as-is:

- the runtime ACK override resolves the historical destination-count/ACK-count split;
- normal type-2 sender+compute and `SKIP_MCAST` type-1 sender-only behavior are expressible;
- normal type-1 sender-only behavior is not equivalent under `send()`: because the sender is inside the
  rectangle and the sharded source differs from the compute destination, loopback inference adds an
  unowned local data write and leaves the sender's helper Flag VALID;
- `SKIP_MCAST` type 2 is not expressible: `send()` couples data and signaling, while `send_signal()` is
  EXCLUDE-source and cannot satisfy the sender's local readiness dependency;
- the smallest proposed API extension is an explicit default-preserving self-mode (`AUTO`, `INCLUDE`,
  `EXCLUDE`) on data and signal sends. That would be an API-v12 design decision and is not authorized.

No production source was edited and no correctness/performance tests were credited for this blocked unit.

### C6 — Tier 2.8 group-attention Matmul

Three Claude consultations were attempted with the required Opus command: a broad architectural review,
a compact decision request, and a post-diff/evidence review. They timed out after 180, 120, and 180
seconds respectively without producing a verdict. A fourth final review returned REVISE for evidence
completeness only: cite per-round coordinate indexing and cross-round ordering, attribute the large
speedup, and classify the expected skips. The migration log now resolves all four items from source and
profile evidence; no code or API change was requested. No approval was inferred from earlier silence,
and no API expansion was made. A bounded re-review returned PASS for ledger write-back after the
evidence revisions; it requested that the cross-round proof also live in the kernel annotation and the
skip categories in `test_map.json`, which are now recorded.

The independently verified API-v11 decision was KEEP:

- model the fixed dense receiver rectangle as one `Mcast2D`, with the first 32 logical cores rotating
  through the sender role;
- move the helper semaphore IDs into compile-time arguments, retain only the per-core divergent ACK
  count as operation-owned runtime state, and append `McastArgs` without disturbing existing ABI slots;
- use direct sender construction for the rotating face and `McastArgs::receiver()` for the receiving
  face; helper loopback inference exactly covers the `CB2 -> CB1`, `CB1 -> CB1`, inside, and outside
  cases;
- remove the historical per-round post-flag barrier. API v11 flushes the remote Flag before state reset
  and barriers local loopback; one final write barrier preserves last-send completion at kernel exit.

Correctness, fresh-JIT, LOC, build, and two matched 800 MHz performance gates all passed. API expansion:
NO.

### C7 — Tier 2.9 Conv3D weight sharing

The initial architectural consultation returned REVISE. It required the migration to use default
`SourceL1Guard`, prove that the weight CB exists on every rectangle core, keep an unconditional
four-word helper runtime ABI and resume through `next_runtime_args_offset()`, assert that every group
strip shares the representative compile-time helper configuration, and adopt the existing semaphore IDs
without changing descriptor allocation order. All five conditions were implemented before validation.

Two broad post-validation reviews timed out after three minutes and 150 seconds without verdict; neither
was treated as approval. A final bounded, fact-complete review returned PASS, API EXPANSION NO, LEDGER
YES. Claude confirmed that active Mcast roles use existing API-v11 `Mcast2D`, the Chain/Disabled paths
remain untouched, and the evidence justifies write-back.

### C8 — Tier 2.10 post-allgather LayerNorm

The architecture consultation returned API EXPANSION NO: API v11 can express the dense `mcast_1d`
channel directly, and the plan's operation-owned local copy plus helper remote pipe preserves the
outside-sender non-1D protocol without a loopback knob. It also confirmed that dense landed cells own
the required CB/semaphore state on ragged logical grids.

Two final post-validation review attempts used the required Opus command and timed out after 240 and
150 seconds without output. Neither silence was treated as a new verdict. Ledger write-back follows the
earlier explicit API-v11 architecture decision and the completed mandatory build, correctness, fresh-JIT,
LOC, host/helper, and matched-performance gates; no helper API expansion was made.

## Progress

| Unit | State | Current finding / next gate |
|---|---|---|
| Reconciliation | complete | 104 unique entries preserved; current rollout state is 21 migrated, 2 pending, 81 deferred |
| Tier 0.1 Matmul in0 interleaved | blocked-writeback | Correctness/JIT verification passed; historical matched performance checkout is not authorized |
| Tier 0.2 Matmul in0 block-sharded | complete | Migrated API v11 after correctness, fresh-JIT, inherited performance, and Claude gates passed |
| Tier 1.6 DeepSeek sampling | complete | API v11 at `2840fc28361`; 101-core correctness/cache, raw-signature top-k barrier proof, and matched performance passed |
| Tier 2.7 DRAM-sharded Matmul | deferred-design-gap | API v11 cannot preserve sender-only EXCLUDE plus type-2 signal-only INCLUDE behavior; API expansion is not authorized |
| Tier 2.8 group-attention Matmul | complete | API v11 at `6e8eb763885`; exact/full correctness, fresh JIT, barrier proof, and both matched performance gates passed |
| Tier 2.9 Conv3D weight sharing | complete | API v11 at `a290ce20281`; exact/full correctness, fresh JIT, helper guards, and both matched performance gates passed |
| Tier 2.10 Sharded LayerNorm post-allgather | complete | API v11 at `6cc49825476`; exact/full correctness, ABI guards, fresh JIT, host topology, and both matched performance gates passed |

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
14. Tier 2.7 architecture audit confirmed that API v11's runtime ACK override closes the historical
    count split but not the full protocol. Sender-only normal sends require forced EXCLUDE despite a
    non-aliasing in-rectangle source, while sender+compute `SKIP_MCAST` requires an INCLUDE-source
    readiness signal without data. The public helper exposes neither operation. Claude returned DEFER;
    the API-extension gate remains closed, so the kernel/factory were left untouched.
15. Migrated Tier 2.8 in `6e8eb763885`: replaced the raw rotating multicast and manual physical
    coordinates with independent `Mcast2D` sender/receiver faces over the fixed dense rectangle. Helper
    compile-time and runtime arguments were appended after the existing TensorAccessor and operation
    ABI. The host retains only runtime slot 20 for each core's divergent ACK count. Production files
    shrank independently (factory 55 deletions / 20 additions; kernel 132 deletions / 24 additions).
16. Proved the raw per-round post-flag full barrier redundant: helper `send()` flushes the remote Flag
    before state reset and completes local loopback writes; a single final write barrier preserves the
    last remote send at kernel exit. Release build passed, the exact q16 fully-sharded ROW_MAJOR `--dev`
    probe passed with fresh artifacts, and the complete `group_attn_matmul` inventory reported 322 passed,
    132 expected skips, 299 deselected, with 351/351 warm JIT hits.
17. Tier 2.8 matched 800 MHz Tracy used three independent 25-iteration sessions per source state and
    shape, discarding the first five operation samples in every session. Median-of-run-medians improved
    from 57,846.5 ns to 39,219.5 ns (-32.20%) for q16 and from 405,652 ns to 294,888 ns (-27.31%) for
    q48. Raw source was restored reversibly from the parent commit, verified byte-identical, then the
    migrated commit was restored exactly and rebuilt. Empty profiler attempts that selected zero tests
    were not credited.
18. Final Tier 2.8 guards passed on the restored committed source: exact normal q16 passed,
    `test_mcast_pipe.py` passed 80/80, the source audit passed 17/17, and `McastHostFixture.*` passed
    32/32.
19. Claude's final Tier 2.8 review requested documentation-only revisions. Source citations now show
    that sender selection and receiver ACK targets index `tile_row_id` every round, and that every
    next-round ACK occurs only after the prior Flag was observed/cleared while the sender fence flushes
    the linked data+Flag chain before local reset. The 32-round removal of the raw intermediate flush
    plus post-Flag full barrier explains the measured speedup; both profiles retained 25 operation rows
    and 110 cores. The 132 skips are 96 optional-preallocated sharded-output exclusions plus 36 remaining
    duplicate COL_MAJOR interleaved-input cases.
20. Migrated Tier 2.9 in `a290ce20281`: each Conv3D weight-sharing group strip is now an independent
    fixed-sender `Mcast2D`. The factory adopts the existing semaphore IDs, appends helper CT arguments
    after the three TensorAccessor blocks, and supplies one fixed four-word helper RT block at slot 19
    for every mode. The kernel resumes through the helper's named next offset. Chain unicast forwarding
    and Disabled/local loading remain unchanged. Production files shrank independently (factory 49
    deletions / 44 additions; kernel 58 deletions / 15 additions).
21. The sender uses default `SourceL1Guard` because its one-block CB can be refilled before the next
    iteration's ACK wait. The host now asserts that all rectangle cores own the weight CB and that every
    exact group helper is active and compile-time-identical to the representative. Passive receivers use
    semaphore-only `receive()` iterations and retain the final atomic barrier.
22. Tier 2.9 validation passed: Release build; a zero-hit exact Mcast/NOC1 writer compile with PCC
    `0.9999914190473849`; 12/12 focused shapes; unit 27 passed / 1 pre-existing skip; nightly 1606 passed
    / 5 expected skips / 2 pre-existing width-sharded page-alignment xfails; helper host 32/32, helper
    device 80/80 under Watcher, and source audit 18/18 after adding the Conv3D fixed-ABI guard.
23. Matched 800 MHz Tracy used three independent 25-iteration sessions per source state and shape,
    discarding the first five Conv3d samples per session. The non-grouped median improved from 14,977 ns
    to 14,855 ns (-0.815%); the grouped median improved from 70,343 ns to 70,133.5 ns (-0.298%). Claude's
    final decision was PASS with no API expansion and approval for ledger write-back.
24. The user authorized the remainder of Tier 2 in the same unattended, Claude-reviewed fashion. API
    v11 remains the implementation ceiling: any unit that cannot satisfy its observable protocol with
    the existing API stops before production edits and is recorded as deferred; no API expansion is
    inferred from this continuation.
25. Migrated Tier 2.10 in `6cc49825476`: generalized the shared host descriptor across distributed
    LayerNorm variants. Dense post-allgather uses helper loopback. Each non-1D row/column uses a helper
    remote rectangle with the sender outside it, while the kernel retains an operation-owned local
    CB21-to-CB15 copy and completes it before publishing the CB. Pre-allgather behavior is preserved.
26. The production LOC gate passed independently: receiver 10/12, sender 26/45,
    `layernorm_op_multi_core_sharded.cpp` 8/13, shared helper cpp 52/53, and shared helper hpp 12/20
    additions/deletions. No helper source or API changed.
27. Tier 2.10 validation passed: Release build; a fresh-cache exact `--dev --no-precompile` node with
    0/47 JIT hits and both artifacts; post 136/136; pre 126/126; plain sharded 208/208; host 34/34;
    helper 80/80; and source audit both before and after write-back 18/18.
28. Matched 800 MHz Tracy used three independent 25-operation sessions per source state and norm type,
    discarding the first five samples. LayerNorm improved from 4009 to 3880 ns (-3.2%); RMSNorm improved
    from 4020 to 3617.5 ns (-10.0%).
29. The mapped post inventory is `mcast_1d` only. A non-1D probe exposes the existing operation contract:
    stats must be sharded on one core, but multiple sender lines each require stats, so only the first
    line has valid input. This is recorded as an operation coverage limit, not a helper API gap; the
    outside-sender wire is covered by the host fixture.
