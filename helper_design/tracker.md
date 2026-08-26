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

### C9 — Tier 2.11 plain sharded LayerNorm

The initial architecture consultation and the focused non-rectangular-grid follow-up both used the
required Opus command and timed out after 180 seconds without a verdict. Neither silence was treated as
approval. The plan-prescribed API-v11 composition was implemented experimentally: one handshaked Flag
pipe carried the readiness phase and one non-handshaked Counter pipe carried repeated final-stat blocks.
No helper API expansion was made.

The first full-suite attempt exposed a migration bug on the `1_line_plus_1` ragged grid: the helper's
dense `Mcast2D` default waited for ACKs from bounding-box holes whose `IDLE_CORE` readers return without
acking. The corrected host composition passes the legacy operation wait count explicitly
(`grid.num_blocks - 1`) while retaining the dense multicast rectangle. Release build, all 36 host
multicast tests, the exact formerly hanging `--dev` parametrization, 280 sharded LayerNorm cases, 190
pre/post-allgather guards, and 80 helper-device cases passed.

The matched-performance gate nevertheless failed for the single-stage case: migrated median 7,904 ns
versus raw 7,742.5 ns (+2.086%, above the +2.0% ceiling). The two-stage case passed at 44,687.5 ns versus
44,636.5 ns (+0.114%). The narrow single-stage delta is consistent with the stronger completion required
by API-v11's monotone Counter (`inc_multicast` plus atomic-ack drain) compared with the legacy value flag.
A final focused defer-versus-change consultation returned no output. Per the approved plan, the unit was
deferred and all Tier 2.11 production/test changes were reverted; no gate was waived and no API change
was attempted.

### C10 — Tier 2.12 interleaved GroupNorm

The architecture consultation used the required Claude command but returned no verdict. No approval was
inferred from silence. The implementation follows the plan's already-migrated sharded-v2 precedent:
three API-v11 `Mcast2D` wires partition the mid, first-edge, and last-edge rectangles; the legacy path
uses separate `send_signal`/`receive_signal` and `send`/`receive` phases, while Welford uses the ordinary
data exchange. The shared aggregate receiver-ready counter and gather tails remain operation-owned.
The no-mcast factory emits the same opaque ABI with inactive singleton wires. No helper source or API
was changed.

Raw matched real-time baselines at 800 MHz used three independent 20-operation sessions after three
warmups. Median-of-run-medians is 74,173.140 ns for legacy and 103,521.069 ns for Welford. The new
benchmarks assert that both non-v2 sender/receiver source pairs are JIT-compiled; the existing SDXL
benchmarks remain sharded-v2 guards.

Post-migration median-of-run-medians is 75,115.544 ns for legacy (+1.271%) and 103,191.783 ns for
Welford (-0.318%). Both pass the mandatory +2% ceiling, and legacy remains below the +1.5% five-run
extension threshold. Complete operation, helper-host, helper-device, source-audit, build, and LOC gates
also pass.

The final fact-complete Claude review used the required command and timed out after 240 seconds without
output. Silence was not treated as approval. Ledger write-back follows the plan-authorized API-v11
design and the independently completed mandatory gates; no API expansion was made.

### C11 — Tier 2.13 SDPA-decode `read_k` star

The fact-complete architecture consultation used the required Claude command and timed out after 240
seconds without a verdict. No approval was inferred. The plan-authorized API-v11 design uses one
no-handshake fixed `Mcast2D` per vertical replicated-Q sharing group, adopts the existing K semaphore,
and retains the runtime sender-role scalar because the fixed helper wire intentionally has no role tag.
The sender uses `CallerManaged` followed by the existing full write barrier, preserving source lifetime
and the proven Blackhole post-signal completion point; the receiver conservatively retains its atomic
barrier. No helper API change is proposed.

A second fact-complete consultation after build, full correctness, LOC, and performance evidence used
the required command and timed out after five minutes without a verdict. Silence was not treated as
approval. The plan-authorized gates independently pass, and no API expansion was made.

### C12 — Tier 2.14 Argmax multicore control

The architecture consultation used the required Claude command and timed out after five minutes
without a verdict. No approval was inferred. The plan-authorized API-v11 design composes two fixed
no-handshake Counter control wires over the two disjoint worker rectangles and adopts the existing
start semaphore. Rectangle 0 contains and excludes the reducer; rectangle 1 uses it as a separate
sender. Counter readiness preserves the reset-free `wait_min` behavior required by free-running
reduce-all, while result unicasts and the operation-owned done counter remain unchanged. No helper API
change is proposed.

The final fact-complete consultation after build, full correctness, LOC, and matched performance used
the required command and timed out after five minutes without a verdict. Silence was not treated as
approval. The plan-authorized gates independently pass, and no API expansion was made.

### C13 — Tier 2.15 Move overlap control

The architecture consultation used the required Claude command and timed out after five minutes
without a verdict. No approval was inferred. The implemented API-v11 decomposition keeps the existing
worker return counter operation-owned and gives each of the two or three disjoint worker rectangles its
own fixed no-handshake Flag wire. The host always emits three opaque helper blocks; an absent third
rectangle is an inactive controller singleton. A runtime region selector preserves the shared
controller/worker binary, and source/destination addresses remain slots 0/1 for cache override. No
helper API change is proposed.

The final fact-complete review after Release builds, complete correctness, fresh JIT, LOC, source audit,
helper guards, and matched performance also timed out after five minutes without a verdict. Silence was
not treated as approval. The mandatory plan gates independently pass in both tiled and row-major paths,
and no API expansion was made.

### C14 — Tier 2.16a routed-expert FFN

The architecture audit found that the in0, in1-down, and activated-data channels can each be
described by API-v11 pipes, but phase 1's active in1 protocol cannot. It sends two discontiguous L1
payloads (`gate`, then `up`) under one receiver-ready ACK and publishes one valid Flag after both
linked writes. `send()` couples one payload to one signal; two handshaked calls would require two ACKs
and publish two signals, while a no-handshake second pipe would not protect repeated destination
lifetime. No public data-only stage exists.

The required extension invariant was not found in a second unrelated production family. The H2D/D2H
units need arbitrary GlobalSemaphore targeting instead, so they do not make the routed-FFN extension
general. The required Claude consultation timed out after five minutes without a verdict; silence was
not treated as approval. The kernel and factory remain raw, and no helper API change is proposed.

### C15 — Tier 2.16b/c persistent host-I/O services

Both persistent service kernels signal worker-grid `GlobalSemaphore` L1 addresses allocated outside
their single-core service programs. API-v11 `Semaphore<>` accepts only a program semaphore ID and
`SenderPipe` always signals the corresponding owned local address. Neither emitter creates a worker
program semaphore, and substituting one would change the cross-program address and lifetime contract.

H2D has a second independent gap: metadata and the worker-ready event are separated by a DRAM barrier
and PCIe completion publication, whereas `send()` fuses its one payload directly to its signal. D2H
otherwise exactly matches a no-handshake Counter `send_signal()` including its atomic barrier; its sole
helper gap is external-address binding. Claude independently returned DEFER — DESIGN-GAP +
COVERAGE-GAP for both and agreed that two directions of one socket subsystem fail the unrelated-family
generality gate.

The current Python and C++ worker-sync tests require Blackhole Galaxy/UBB and skip on this single-chip
P100a. The plan's stale single-device coverage statement was corrected. No production code or helper
API was changed.

## Progress

| Unit | State | Current finding / next gate |
|---|---|---|
| Reconciliation | complete | 104 unique entries preserved; current rollout state is 31 migrated, 2 pending, 71 deferred |
| Tier 0.1 Matmul in0 interleaved | blocked-writeback | Correctness/JIT verification passed; historical matched performance checkout is not authorized |
| Tier 0.2 Matmul in0 block-sharded | complete | Migrated API v11 after correctness, fresh-JIT, inherited performance, and Claude gates passed |
| Tier 1.6 DeepSeek sampling | complete | API v11 at `2840fc28361`; 101-core correctness/cache, raw-signature top-k barrier proof, and matched performance passed |
| Tier 2.7 DRAM-sharded Matmul | deferred-design-gap | API v11 cannot preserve sender-only EXCLUDE plus type-2 signal-only INCLUDE behavior; API expansion is not authorized |
| Tier 2.8 group-attention Matmul | complete | API v11 at `6e8eb763885`; exact/full correctness, fresh JIT, barrier proof, and both matched performance gates passed |
| Tier 2.9 Conv3D weight sharing | complete | API v11 at `a290ce20281`; exact/full correctness, fresh JIT, helper guards, and both matched performance gates passed |
| Tier 2.10 Sharded LayerNorm post-allgather | complete | API v11 at `6cc49825476`; exact/full correctness, ABI guards, fresh JIT, host topology, and both matched performance gates passed |
| Tier 2.11 Plain sharded LayerNorm | deferred-performance | Correctness passed, but single-stage matched performance regressed +2.086%; experimental changes reverted |
| Tier 2.12 Interleaved GroupNorm | complete | API v11 at `40e209daad9`; all build, LOC, correctness, helper, source-audit, and matched-performance gates passed |
| Tier 2.13 SDPA-decode `read_k` star | complete | API v11 at `f760425fe06`; full mapped correctness, fresh-JIT, shared-consumer audit, and q-factor 2/4 matched performance passed |
| Tier 2.14 Argmax multicore control | complete | API v11 at `5aaaf5b5aa5`; exact two-rectangle, reduce-all, cache, full unit, fresh-JIT, source-audit, and both matched-performance gates passed |
| Tier 2.15 Move overlap control | complete | API v11 at `a25603ae2c0`; TILE/ROW_MAJOR fresh JIT, complete Move/cache inventory, helper/source guards, and both matched-performance gates passed |
| Tier 2.16a Routed-expert FFN | deferred-design-gap | API v11 cannot stage gate+up under one ACK and one final linked signal; no unrelated family earns an extension |
| Tier 2.16b H2D host-I/O service | deferred-design-and-coverage-gap | GlobalSemaphore target is unbindable; metadata and ready signal also straddle completion publication; Galaxy/UBB route unavailable |
| Tier 2.16c D2H host-I/O service | deferred-design-and-coverage-gap | Counter control fits API v11 except for its GlobalSemaphore target; Galaxy/UBB route unavailable |

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
30. Tier 2.11 raw matched-performance baselines at 800 MHz used three independent 25-operation sessions,
    discarding the first five samples. The single-stage legacy median-of-run-medians is 7,742.5 ns; the
    two-stage Welford median is 44,636.5 ns.
31. The experimental plain sharded LayerNorm migration composed two existing API-v11 channels: a
    handshaked Flag readiness pipe over the existing sender/receiver semaphores and a reset-free,
    non-handshaked Counter data pipe for each final-stat block. Remote reads and the second-stage
    semaphore remained operation-owned. The host build passed, and a fresh-cache exact legacy node
    passed with 0/29 JIT hits.
32. The first complete sharded-LayerNorm run found a deterministic hang on the ragged
    `1_line_plus_1` grid. The dense multicast rectangle contains inactive landed cores whose reader
    kernels use `IDLE_CORE` and do not ACK, whereas the legacy sender waited only `grid.num_blocks - 1`.
    Passing that existing operation count to `Mcast2D` fixed the mismatch without changing the helper
    API or multicast footprint. The exact node passed under Watcher after a device reset, all 36
    multicast host tests passed, and every touched production file satisfied additions < deletions.
33. Tier 2.11 broad validation passed before the performance decision: 280/280 complete sharded
    LayerNorm, 190 passed plus 10 documented xfails in the distributed pre/post guard inventory, and
    80/80 helper-device cases. Three migrated 25-operation Tracy sessions per shape discarded the first
    five samples. The two-stage median-of-run-medians was 44,687.5 ns versus 44,636.5 ns raw (+0.114%),
    but single-stage was 7,904 ns versus 7,742.5 ns raw (+2.086%), outside the mandatory +2.0% gate.
    API-v11's Counter pipe cannot drop its multicast-atomic acknowledgement drain, and repeated Flag
    signaling would weaken the monotone protocol. A focused Claude consultation returned no verdict;
    the approved plan independently requires deferral on any failed gate. The entire Tier 2.11 source
    experiment and temporary profiler shim were reverted, leaving the ledger unchanged at API v11.
34. Tier 2.12 raw baselines added two route-asserting interleaved GroupNorm cases. Three independent
    20-operation sessions produced median-of-run-medians of 74,173.140 ns (legacy) and 103,521.069 ns
    (Welford). Each record names the intended non-v2 sender and receiver sources.
35. The Tier 2.12 implementation composes mid/first/last API-v11 wires in both mcast and no-mcast
    factories. Legacy preserves its aggregate ACK plus early-go and later-data phases; Welford keeps
    its aggregate ACK plus single data phase. The four kernels resume through named opaque CT/RT
    boundaries, all six touched production files independently satisfy additions < deletions, and the
    Release host build passed. No helper API changed.
36. Exact legacy and Welford interleaved nodes passed after fresh compilation. Watcher mode cannot
    currently compile the generated C++17 vararg accessor because its lightweight ASSERT expands to
    `asm` in a constexpr function; the same exact nodes pass without `--dev`. The complete DRAM suite
    passed all 181 valid cases with five architecture skips. Its sole strict XPASS is the pre-existing
    negative garbage-padding probe, whose contract expects intentionally invalid padding to corrupt the
    legacy analytic correction. The general GroupNorm inventory passed 345 with ten expected skips.
37. Tier 2.12 nightly passed 203 with six expected skips; `McastHostFixture.*` passed 34/34; helper
    device tests passed 80/80 under Watcher; and the expanded source audit passed 19/19. Three migrated
    performance sessions per algorithm produced 75,115.544 ns legacy (+1.271%) and 103,191.783 ns
    Welford (-0.318%) median-of-run-medians. Both mandatory +2% gates pass, and no five-run extension
    is required.
38. The final Tier 2.12 Claude review timed out after 240 seconds without a verdict; no approval was
    inferred. The source migration was committed at `40e209daad9`, then the four kernel rows and three
    host bindings were written back at API v11 from the independently satisfied gates. Rollout state is
    now 27 migrated, 2 pending, and 75 deferred kernels. API expansion: NO.
39. Tier 2.13 replaces only the `read_k` vertical star. Host code emits an inactive singleton wire for
    ordinary SDPA-decode and exact per-group wires for replicated-Q MLA; kernel code resumes both CT
    and RT through `McastArgs`. Every touched production file has additions below deletions, the Release
    build passed, and a fresh-cache factor-4 replicated-Q route passed with 0/16 JIT hits and PCCs above
    0.9998. A first compile attempt exposed only that the shared header is also parsed by the writer;
    moving the helper include into that header fixed all consumers without API expansion.
40. Tier 2.13 mapped validation passed: unit SDPA 11 passed / 1 grid skip; flexible geometry 10/10;
    unit MLA 2/2; replicated-Q q-factor 4 and q-factor 2 stress plus ordinary guards 4/4; nightly SDPA
    75 passed / 13 declared skips; cache 10/10; nightly MLA 40/40. The sink file retained its 18 existing
    Blackhole issue skips. The host helper suite passed 34/34, the device helper suite passed 80/80
    under Watcher, and the expanded source audit passed 20/20. Release builds passed before and after
    the matched source-state profiling cycle.
41. Matched 800 MHz Tracy used three independent 25-operation sessions per source state and sharded
    replicated-Q geometry, discarding the first five SDPA operation samples per session. The q-factor-4
    median-of-run-medians improved from 52,478.5 ns to 51,850 ns (-1.20%); q-factor 2 improved from
    51,833 ns to 51,473 ns (-0.69%). Raw production files were restored reversibly from the source
    commit parent, rebuilt, then the migrated files were restored byte-identically and rebuilt.
42. The second Tier 2.13 Claude review timed out after five minutes without a verdict; no approval was
    inferred. Source was committed at `f760425fe06`, then the kernel and host binding were written back
    at API v11 from the independently satisfied mandatory gates. Rollout state is now 28 migrated,
    2 pending, and 74 deferred kernels. API expansion: NO.
43. Tier 2.14 replaces the reducer's two raw start-semaphore multicasts with two fixed no-handshake
    Counter wires. Both adopt the existing start semaphore; their receiver sets are disjoint, and the
    reducer is inside rectangle 0 but separate from rectangle 1. Signaling starts at k=1, so helper
    Counter round 1 exactly replaces legacy value 2 as the first gate. Result unicasts, their barriers,
    and the independent done counter are unchanged. The architecture Claude consultation timed out
    after five minutes without a verdict; no approval was inferred. API expansion: NO.
44. Release build passed after two mechanical host integration corrections. Production LOC passed:
    factory 49 additions / 61 deletions and kernel 25 additions / 40 deletions. A fresh-cache exact
    two-rectangle reduce-all route passed under Watcher with 0/19 JIT hits. The explicit two-rectangle
    last-dimension and reduce-all cache routes passed, the complete unit file passed 69/69, nightly
    Argmax passed 16 with 8 declared skips, host helpers passed 34/34, device helpers passed 80/80
    under Watcher, and the expanded source audit passed 21/21.
45. Matched 800 MHz Tracy used three independent 25-operation sessions per source state and case,
    discarding the first five operation samples per session. `[64,128]`, dim=-1 improved from 68,504.5
    ns to 68,310 ns (-0.28%); the multi-round two-rectangle case moved from 693,623 ns to 695,722 ns
    (+0.30%). Raw production files were restored from the source checkpoint parent, rebuilt, then the
    migrated files were restored byte-identically and rebuilt.
46. The final Tier 2.14 Claude review timed out after five minutes without a verdict; no approval was
    inferred. Source was committed at `5aaaf5b5aa5`, then the kernel and host binding were written back
    at API v11 from the independently satisfied mandatory gates. Rollout state is now 29 migrated,
    2 pending, and 73 deferred kernels. API expansion: NO.
47. Tier 2.15 decomposes the legacy dual-use control word into the unchanged operation-owned worker
    return counter and three helper-owned fixed release flags. Two or three disjoint worker rectangles
    are represented by three opaque no-handshake Flag wires, with an inactive controller singleton for
    the absent third rectangle. Controller/worker role selection remains runtime under one kernel ABI;
    src/dst cache slots remain 0/1. Both architecture and final Claude consultations timed out after
    five minutes without a verdict; no approval was inferred. API expansion: NO.
48. Release build and production LOC gates passed: tiled kernel 30 additions / 40 deletions, row-major
    kernel 31/41, factory 46/47. Separate fresh-cache TILE and ROW_MAJOR overlap nodes passed under
    Watcher with 0/19 hits each. The complete Move inventory passed 136 valid cases with 128 intentional
    non-L1 skips, including all eight cache cases. Host helpers passed 34/34, device helpers 80/80 under
    Watcher, and source audit 22/22 before and after ledger promotion.
49. Matched 800 MHz Tracy used three independent 25-operation sessions per source state and layout,
    discarding the first five Move samples per session. TILE improved from 4,144.5 ns to 4,125 ns
    (-0.47%); ROW_MAJOR improved from 6,190 ns to 6,189 ns (-0.02%). Raw production files were restored
    from the source checkpoint parent, rebuilt, then the migrated files were restored byte-identically
    and rebuilt. Source was committed at `a25603ae2c0`; the two kernel rows and host binding were then
    written back at API v11. Rollout state is 31 migrated, 2 pending, and 71 deferred kernels.
50. Tier 2.16a stopped before production edits at the API-v11 design gate. The phase-1 in1 path requires
    gate and up to retain one linked path under one receiver-ready handshake and one final valid Flag;
    the public helper has no data-only stage and two `send()` calls change the protocol. The prior
    multi-device-only ledger coverage claim was corrected to the routed-expert test inventory. Claude's
    five-minute architecture consultation timed out without a verdict. API expansion: NO.
51. Tier 2.16b/c source audit proved that both worker events target cross-program GlobalSemaphore L1
    addresses, while API-v11 binds only program semaphore IDs. Claude independently returned DEFER for
    both, confirmed that D2H is otherwise an exact no-handshake Counter control pipe, and identified
    H2D's separate metadata/barrier/completion/signal ordering gap. The twins do not satisfy the
    unrelated-family extension gate. API expansion: NO.
52. The H2D/D2H Python service modules and C++ worker-sync tests require Blackhole Galaxy/UBB and skip
    on the current single-chip P100a, so no skipped route or performance result was credited. The stale
    Tier 2.16 plan wording was corrected and both rows record design-gap plus coverage-gap separately.
53. Every remaining Tier 2 unit now has a terminal disposition. Tier 2.16 made no production changes,
    so rollout counts remain 31 migrated, 2 pending, and 71 deferred kernels with 27 migrated and 5
    pending host bindings. The final static source audit passed 22/22. Helper API remains v11.
