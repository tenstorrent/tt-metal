# Archived: mcast_pipe rollout report through 2026-08-06

## Tier 3 sharded LayerNorm pre-allgather migration — PASS

`layernorm-sharded-pre-allgather` is fully end-to-end migrated at v10. The
shared compile-time reader builder was first split into variant-specific
builders and landed independently as `4ef7e9a57a6`; its rebuild and all three
guard inventories passed before the production wire changed.

The host now describes the reduce-ready channel with a handshaked Flag
`Mcast2D` for whole-grid reduction or fixed-sender `Mcast1D` lines for
two-stage reduction. Both kernels decode a complete opaque CT/RT block. The
global sender and additional two-stage line coordinators call `send_signal()`;
the other participants call `receive_signal()`. Gather reads, CB ownership,
the second-stage semaphore, and final write/atomic barriers remain
operation-owned.

- `./build_metal.sh`: passed.
- Exact 8x4 BFLOAT8_B RMSNorm case under `--dev` from a fresh isolated cache:
  passed with sender and both receiver-variant JIT artifacts confirmed.
- `LN-PRE-ALLGATHER`: 126 passed; `LN-POST-ALLGATHER`: 136 passed;
  `LN-SHARDED`: 208 passed.
- `McastHostFixture.*`: 28/28, including three new offset and two-stage
  geometry cases; `test_mcast_pipe.py`: 77/77.

The profiled exact node contains four pre-allgather calls with device-kernel
durations 2,583, 2,564, 2,563, and 2,656 ns (median 2,563.5 ns). A per-kernel
delta is **N/A**: there is no operation-matched pre-migration LayerNorm
bakeoff, and the reported data-movement envelopes include other kernels in the
operation.

The current ledger state is 17 migrated kernels and 14 current host bindings
across eight atomic units. Two kernel rows and five bindings remain pending in
the Matmul in0 unit. The production migration is `4acd98259b6`, after the
separate builder-split prerequisite `4ef7e9a57a6`.

## Tier 2 TopK final-readiness migration — PASS

`topk-multicore-final-readiness` is fully end-to-end migrated at v10: one host
binding and both kernel faces. The factory now owns one sender-separate
no-handshake Counter `Mcast2D`, adopts readiness descriptor 1 with explicit
`INVALID` (`0`) host initialization, and prepends the complete opaque helper
CT/RT blocks. `reader_final_topk` uses `send_signal()` and
`writer_local_topk` uses `receive_signal()`; value/index unicast, the arrival
counter, CB ownership, and data/atomic barriers remain operation-owned.

- `./build_metal.sh`: passed.
- Exact W=8192, k=50, BFLOAT16_B case under `--dev` from a fresh isolated cache:
  passed with both migrated JIT artifacts confirmed.
- `TOPK-MULTICORE`: 14 passed, 12 expected BFLOAT8_B pad xfails, 26 selected.
- `McastHostFixture.*`: 25/25; `test_mcast_pipe.py`: 77/77.
- Production diff: +77 / -75 (reader -23 lines, writer -19 lines); net +2.

The profiled exact node reports a 238,281 ns device-kernel duration, with a
32,003 ns reader/NCRISC envelope and a 238,280 ns writer/BRISC envelope. A
per-kernel delta is **N/A**: no operation-matched pre-migration TopK bakeoff
exists, and each processor envelope includes another TopK data-movement kernel.
The raw F2 helper bakeoff has different work and geometry and is not presented
as a comparable baseline.

At completion of this tier, the ledger held 15 migrated kernels and 13 current
host bindings across seven atomic units. The LayerNorm tier above supersedes
those interim totals.

## Final release gate — 2026-08-05

All mapped correctness inventories, the host build, helper host/device suites,
and the 10-test opaque-boundary audit are green. Fresh artifacts cover all 13
migrated kernels, all 12 required host bindings are build-covered, and no
migrated row carries `needs_recheck`.

All ten performance cases pass the 1.5% gate. The only apparent failure,
legacy GroupNorm, was resolved with a controlled worktree comparison using
independent builds and Python environments for the actual pre-migration
baseline `4a1d6a97ca9`, previously passing migrated snapshot `28356d43846`,
and current `2699996541a`. Their medians were `49,694.26516945126`,
`49,850.05759004791`, and `49,836.38882787317 ns`, respectively. Current is
+0.285996% versus the freshly reproduced baseline and therefore passes. The
older `48,593.7037037037 ns` artifact did not reproduce on either historical
snapshot under the current firmware/profiler environment, so no production
code change was warranted.

## API v10 update — 2026-08-05

API-001 is implemented. `MCAST_PIPE_API_VERSION=10` adds `rotating_span` as the sixth uniform CT
word and removes the third template argument from `McastArgs`. All 13 migrated kernels and 12 host
bindings remain current; API-002 compile-time sender/receiver-face enforcement remains deferred.

- `./build_metal.sh`: passed.
- `McastHostFixture.*`: 25/25; complete helper device suite: 73/73.
- Fresh-JIT focused cases passed for Matmul, Conv height/block/width, GroupNorm legacy/Welford, and
  Sort. Width-sharded Conv passed at PCC `0.9999992597711427` with 0/26 JIT hits.
- Full mapped inventories: Matmul 302 passed / 188 expected skips; each Conv route 48/16 plus its
  DRAM-config case and shared DRAM 14/14; GroupNorm legacy 108/2, Welford 108/2, fixed/default 19/6;
  Sort long 7/7 and deadlock 2/2.
- The opaque-boundary audit remains green and now also rejects a third `McastArgs` template argument.

The remainder of this report retains the prior v9 rollout history and migration anchors.

### Signal-only handshake and Sort follow-up

API-003 and MIG-002 are implemented under v10. Signal-only methods now honor the existing handshake
policy. Sort uses separate handshaked row-start and no-handshake sub-stage Counter channels; the raw
reader-ready semaphore is removed and writer completion remains operation-owned.

The build, cold-cache helper suite (77/77), exact cold-JIT long Sort case, both deadlock regressions
(2/2), and all long cases (7/7) passed. The median of three performance-run medians was
`145,201,100.41355687 ns`, +1.195124% versus baseline and within the 1.5% gate.

### GroupNorm three-rectangle performance follow-up

MIG-004 is closed. The actual sharded-v2 host constraints and generated group partitions classify
every mapped production configuration as zero-edge. The one- and two-edge wrapped partitions remain
defensive splitter behavior and now have direct synthetic host coverage alongside zero-edge geometry;
`GroupNormMcastGeometry` passed 3/3 and `McastHostFixture` passed 25/25 after a successful rebuild.

The supported zero-edge class reuses the matched Blackhole p100a measurements already recorded for
the SDXL `(1, 1920, 32, 32)` production shape: legacy +0.248% and Welford -0.485% versus baseline.
Both are within the 1.5% gate, so no hot-path change was required.

## Prior v9 run header

- Helper: `mcast_pipe`, `MCAST_PIPE_API_VERSION=9` (unchanged)
- Entry mode: **re-entry**
- Invocation mode: **`run-all`**
- Baseline: `origin/llk_helper_library` at `4a1d6a97ca9`
- Branch: `sjovic/mcast-migration`; pre-rebase branch preserved at
  `backup/mcast-migration-prerebase-20260803`
- Re-entry worklist: one Tier-6 unit, `conv2d-activation-width-sharded`, containing one hybrid
  rotating sender/receiver kernel and one host binding.
- Device: single-chip Blackhole p100a (`bh-41-special-sjovic-for-reservation-53855`) — matches the
  `test_map.json` baseline machine
- Test runner: `scripts/run_safe_pytest.sh`

Production code commit `fe866a1d0c4c32b78aae8a76e875c0da109f51c8` replaces the width-sharded
Conv2D activation kernel's hand-packed rotating multicast wire with the existing API-v9 host
`Mcast2D` plus kernel `McastArgs` sender/receiver faces. No helper change or API bump was required.

## Rollout state at v9

| State | Count |
|---|---:|
| kernel-current | 13 |
| host-binding-current | 12 |
| fully end-to-end current | 12 bindings / 13 kernels |
| host-pending | 0 |
| kernel-pending | 0 |
| quarantined | 0 |
| **open `needs_recheck`** | **0** (was 6 at the start of this run) |
| deferred | 78 kernels / 0 host bindings |

The fleet is **fully current at v9**: nothing stale, nothing pending, no advisory flags open.
The helper-neutral sort writer remains one of the deferred rows because it has no Pipe face.

Deferred first dropped 82 → 81 because `reconcile_2026-08-03.md` removed the deepseek_prefill
`reader_dispatch.cpp` row after the kernel was deleted upstream by `af00262e51d` (#48694). Its
F2-counter coverage is retained by `reader_combine.cpp` and 5 other census entries.
It then dropped 81 → 79 when the sort coordinator and reader migrated, and 79 → 78 when the
width-sharded Conv2D activation kernel migrated.

"Fully end to end" remains channel-specific: e.g. height-sharded Conv2D has a weights-multicast
binding but reads activations locally. Block-sharded Conv2D activation multicast stays in the
deferred kernel census.

## This run — width-sharded Conv2D activation result

| Unit | Rows | Kind | Result |
|---|---:|---|---|
| `conv2d-activation-width-sharded` | 1 hybrid kernel + 1 host binding | Tier-6 migration | **PASS** |

- `./build_metal.sh`: passed.
- Exact BF16/BF16 filter-3 TILE-output node under `--dev` from a fresh isolated cache: passed at
  PCC `0.999956503`; the activation-reader JIT artifact was confirmed.
- Complete width-sharded feature inventory: 48 passed, 16 legitimate row-major+bfloat8 skips.
- Width-sharded DRAM-config route: 1 passed at PCC `0.998234911`; current activation-reader JIT path
  confirmed.
- Post-integration helper suite: 72/72 passed.
- The current ACK-fenced real-loopback completion behavior resolves the earlier v9 port's 25
  numerical regressions; no partial migration or quarantine remains.

## Prior same-day sort migration results

| Unit | Rows | Kind | Result |
|---|---:|---|---|
| `sort-single-row-control` | 2 migrated kernels + 1 helper-neutral companion + 1 host binding | Tier-5 migration | **PASS** |

- Step G added four control-only Counter cases; complete helper suite passed 72/72 before and after
  production integration.
- `./build_metal.sh`: passed.
- Exact `[1,524288]` long-tensor node under `--dev` from a fresh isolated cache: passed; all three
  sort JIT artifacts confirmed.
- Ht=2 deadlock regression: 2/2 passed. Full long-tensor inventory: 7/7 passed.
- Reconcile found all 91 ledger kernel paths present and no new raw multicast primitive callsites.

## Prior same-day verify-only results

| Unit | Rows | Kind | Result |
|---|---:|---|---|
| `matmul-in1-mcast-padding-host` | 2 kernels + 4 host bindings | verify-only (`needs_recheck`) | **PASS — flag cleared on all 6** |

Why the rows were flagged: the matmul mcast 1d/2d factories were churned upstream
(`54d8dfb7bef`→`4a1d6a97ca9`, +203/−13 and +93/−4, touching `mm_in1_sender_writer_args`), then
reworked again by `c946da17d29` + `eb05b3929a3`, both of which postdate the ledger's last update
`62f82dd4a64`. The recorded `last_verified: 2026-07-30` therefore predated the current tree.

Static pre-check carried in from the reconcile (not redone here): both kernel files byte-identical to
the pre-rebase verified state, and the `McastArgs` wire confirmed intact on both factories — sender CT
block idx 10–14 (next = 15 = `KtNt`), sender RT idx 2–5, receiver CT idx 4–8, receiver RT idx 0–3,
`MCAST_ARGS` set at `2d:618` / `1d:1512`, `SKIP_MCAST` coexistence coherent.

### Validation

- `./build_metal.sh`: passed — **already current**. `_ttnn.so` (13:57) postdated both churned
  factories (13:47, 13:48) and its mtime did not change, so nothing needed recompiling.
- Exact compile-focused 2D node under `run_safe_pytest.sh --dev`: **PASSED**, no watcher or assert
  trips. Device-verified that **both** kernels ran — JIT-built at 14:36:22 under the **new** cache
  root `tt-metal-cache12312614508320308860`: sender `6509650342639884602`, receiver
  `4791675444625965894` + `5078604005037224472`.
  - The 2026-07-30 hashes (`4616781822959825899` / `4167676435791909128`) live under the old root
    `tt-metal-cache15548382223525479139`. Hashes are **not comparable across the rebase** (both the
    cache root and the CT args moved), so hash equality is not the check — a green run of both
    kernels at the current state is.
- `MM-IN1-ALL`, re-run in 4 chunks (`-x` per chunk, `--precompile` for the cold cache):

  | chunk | selection | result |
  |---|---|---|
  | A | `test_matmul_2d_multiple_output_blocks_per_core` (128) | 56 passed, 72 skipped |
  | B | `test_matmul_2d_tiny_tile` (96) | 46 passed, 50 skipped |
  | C | `test_matmul_1d_tiny_tile` (96) | 46 passed, 50 skipped |
  | D | remaining 16 test functions (170) | 154 passed, 16 skipped |
  | **total** | **490 selected** | **302 passed, 188 expected skips** |

  Chunked because the changed cache root meant everything compiled cold; the reconstructed `-k`
  selection was confirmed against collection to select exactly 490. The result is an **exact match**
  to the recorded baseline (302 / 188 / 490).
- `McastHostFixture.*`: 19 passed.
- `test_mcast_pipe.py`: 68 passed.

Ledger write-back: `needs_recheck` cleared on all 6 rows, `last_verified` = 2026-08-03,
`verified_at_commit` = `eb05b3929a3`. Each row's `commit` deliberately still points at its migration
commit (`aeeb28ff007`) — its documented role is the revert/bisect anchor, not "last verified at".
`test_map.json` baseline refreshed `54d8dfb7bef` → `4a1d6a97ca9`.

## Cumulative rollout

| Tier / atomic unit | Bindings | Failed | Quarantined | Production diff |
|---|---:|---:|---:|---:|
| 1 — `conv2d-weights-single-sender-rect` | 1 | 0 | 0 | +48 / −51 |
| 2 — `conv2d-weights-fixed-line` | 1 | 0 | 0 | +40 / −96 |
| 3 — `matmul-in1-mcast-padding-host` | 4 | 0 | 0 | +196 / −70 |
| 4 — `groupnorm-sharded-v2-mcast-host` | 4 | 0 | 0 | +168 / −376 |
| 5 — `sort-single-row-control` | 1 | 0 | 0 | +57 / −79 |
| 6 — `conv2d-activation-width-sharded` | 1 | 0 | 0 | +48 / −150 |
| re-entry Tier 2 — `topk-multicore-final-readiness` | 1 | 0 | 0 | +77 / −75 |
| Total | 13 | 0 | 0 | +634 / −897 |

Net reduction of 263 production lines. The TopK run measured current
performance but cannot claim a comparable delta because no operation-matched
pre-migration baseline exists; prior units retain their recorded results.

| Kernel | Status | Validation | File deletions |
|---|---|---|---:|
| Conv2D 1D weights sender | migrated, fully end-to-end | exact JIT; height 49/16 skips; DRAM 14/14 | 19 |
| Conv2D 1D weights receiver | migrated, fully end-to-end | exact JIT; height 49/16 skips; DRAM 14/14 | 11 |
| Conv2D fixed-line weights sender | migrated, fully end-to-end | exact PerRow/PerColumn JIT; block 49/16 skips; DRAM 14/14 | 15 |
| Conv2D fixed-line weights receiver | migrated, fully end-to-end | exact PerRow/PerColumn JIT; block 49/16 skips; DRAM 14/14 | 12 |
| Matmul in1 padding sender | migrated, fully end-to-end | **re-verified 2026-08-03**: exact `--dev` 2D node; 302/188 skips | 40 |
| Matmul in1 padding receiver | migrated, fully end-to-end | **re-verified 2026-08-03**: exact `--dev` 2D node; 302/188 skips | 25 |
| GroupNorm v2 legacy sender | migrated, fully end-to-end | exact JIT; parameterized 108/2 skips | 111 |
| GroupNorm v2 legacy receiver | migrated, fully end-to-end | exact JIT; parameterized 108/2 skips | 17 |
| GroupNorm v2 Welford sender | migrated, fully end-to-end | exact JIT; parameterized 108/2 skips | 113 |
| GroupNorm v2 Welford receiver | migrated, fully end-to-end | exact JIT; parameterized 108/2 skips | 20 |
| Sort single-row coordinator | migrated, fully end-to-end | exact fresh-cache JIT; long 7/7; Ht=2 2/2 | 43 |
| Sort single-row reader | migrated, fully end-to-end | exact fresh-cache JIT; long 7/7; Ht=2 2/2 | 21 |
| Conv2D width-sharded activation | migrated, fully end-to-end | exact fresh-cache JIT; features 48/16 skips; DRAM 1/1 | 150 |
| TopK final readiness sender | migrated, fully end-to-end | exact fresh-cache JIT; multicore 14/12 expected xfails | 23 |
| TopK local readiness receiver | migrated, fully end-to-end | exact fresh-cache JIT; multicore 14/12 expected xfails | 19 |

Conv2D and GroupNorm rows were **not** in this run's scope — their factories are byte-identical to the
pre-rebase verified state, so they needed no recheck. Their evidence dates from 2026-07-30.

## Coverage gaps — UNCHANGED by this run

No migrated kernel lacks device runtime coverage. Four host-binding variants have narrower coverage
and stay flagged in the ledger. These are properties of *who calls the constructors*, not of the
rebase, so the verify-only pass neither widened nor narrowed them:

- **Matmul legacy 1D and 2D constructors** compile in the host build, but their callers are fused CCL
  factories; the mapped single-chip inventory runtime-exercises the **descriptor** constructors only.
  (`legacy-runtime-coverage-gap`, `coverage_confidence: medium` on those two rows; the two descriptor
  rows are `high`.)
- **GroupNorm legacy and Welford `use_mcast=false` bindings** have host-oracle and degenerate
  device-wire coverage, but no mapped operation test reaches the v2 sender-only route. The same sender
  kernels are JIT-verified through the multicast route.

## Deferred and pending kernel work

The 72 deferred and 4 pending kernel rows remain outside the completed units. Principal production
blockers and prerequisites:

- matmul in0 control channels needing typed/custom control values or independent data/signal loopback;
- block-sharded Conv2D activation multicast, whose producer-overlapped chunked send the current helper
  cannot express;
- LayerNorm channels needing acknowledged signal-only, mixed-mode streaming, or explicit
  include-source loopback;

The earlier reconcile listed ten deferred rows whose upstream churn touched protocol lines. The
three sort rows were re-audited here: coordinator and reader migrated, while writer was explicitly
classified helper-neutral. The remaining churned rows stay deferred for later focused audits.

No partial migration for a deferred or quarantined unit remains in the worktree.

## Commits

Code and doc commit hashes below are the **post-rebase** ones (the pre-rebase hashes previously
recorded here resolve only on `backup/mcast-migration-prerebase-20260803`; remapped 1:1 by subject,
confirmed by patch-id).

| Unit | Code commit | Ledger/log commit | pre-rebase code hash |
|---|---|---|---|
| Conv2D height weights | `991b5b6b6386a90726d15007002fe1f5a77d8487` | `baa86dc7116` | `75b977e1a04` |
| Conv2D fixed-line weights | `51dfb1f1ed61045ed10dc679269960b6d2ccac9e` | `5320c2d69bd` | `261e322ed22` |
| Matmul in1 | `aeeb28ff007807c71b1f60842cca85e5c41efa7f` | `53724c12419` | `2d0280d3dac` |
| GroupNorm v2 | `bc24a55bf80a8ab2a4d702be2a91b827c1dcbeb0` | `49e559dcb55` | `0a796a025c9` |
| Sort single-row control | `7337302b5649b7cd169764cd95c0b0343e88950d` | `8479210e61e` | n/a |
| Conv2D width-sharded activation | `fe866a1d0c4c32b78aae8a76e875c0da109f51c8` | `30927931918` | historical v8 only |
| TopK final readiness | `b5c99d43fd5` | same commit | historical v8 only |

Two on-branch commit *messages* still cite pre-rebase hashes (`baa86dc7116` "…for 75b977e1a04",
`5320c2d69bd` "…for 261e322ed22") — history is immutable; this table is the key.

The authoritative machine state is `migration/ledger.json`; this report is its regenerated run view.
