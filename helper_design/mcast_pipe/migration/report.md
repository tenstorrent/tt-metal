# mcast_pipe rollout report — API v9, verify-only pass 2026-08-03

## Run header

- Helper: `mcast_pipe`, `MCAST_PIPE_API_VERSION=9` (unchanged — this skill only reads it)
- Entry mode: **re-entry**
- Invocation mode: **`halt`**
- Baseline: `origin/llk_helper_library` at `4a1d6a97ca9` (HEAD 17 commits ahead, at `eb05b3929a3`)
- Branch: `sjovic/mcast-migration`; pre-rebase branch preserved at
  `backup/mcast-migration-prerebase-20260803`
- Re-entry worklist: **0 stale kernels, 0 stale/pending host bindings, 0 pending kernels, 0 net-new
  units — the entire worklist was the 6 `needs_recheck` rows** raised by
  `reconcile_2026-08-03.md`, handled as Phase 2's priority-1 **verify-only** pass
- Device: single-chip Blackhole p100a (`bh-41-special-sjovic-for-reservation-53855`) — matches the
  `test_map.json` baseline machine
- Test runner: `scripts/run_safe_pytest.sh`; gtest under `flock /tmp/tt-device.lock`

**No production code was modified in this run.** A verify-only pass re-runs a row's `validation_set`
and clears the flag; it never rewrites a call site. There is therefore no diff, no commit of
production code, and nothing to quarantine or revert. Per-tier subagents were not used (standing
instruction not to spawn agents unprompted); the single verify-only unit ran inline.

## Rollout state at v9

| State | Count |
|---|---:|
| kernel-current | 10 |
| host-binding-current | 10 |
| fully end-to-end current | 10 bindings / 10 kernels |
| host-pending | 0 |
| kernel-pending | 0 |
| quarantined | 0 |
| **open `needs_recheck`** | **0** (was 6 at the start of this run) |
| deferred | 81 kernels / 0 host bindings |

The fleet is **fully current at v9**: nothing stale, nothing pending, no advisory flags open.

Deferred dropped 82 → 81 because `reconcile_2026-08-03.md` removed the deepseek_prefill
`reader_dispatch.cpp` row after the kernel was deleted upstream by `af00262e51d` (#48694). Its
F2-counter coverage is retained by `reader_combine.cpp` and 5 other census entries.

"Fully end to end" remains channel-specific: e.g. height-sharded Conv2D has a weights-multicast
binding but reads activations locally. Block- and width-sharded Conv2D activation multicast stay in
the deferred kernel census.

## This run — verify-only results

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

## Cumulative rollout (from the original migration, unchanged by this run)

| Tier / atomic unit | Bindings | Failed | Quarantined | Production diff |
|---|---:|---:|---:|---:|
| 1 — `conv2d-weights-single-sender-rect` | 1 | 0 | 0 | +48 / −51 |
| 2 — `conv2d-weights-fixed-line` | 1 | 0 | 0 | +40 / −96 |
| 3 — `matmul-in1-mcast-padding-host` | 4 | 0 | 0 | +196 / −70 |
| 4 — `groupnorm-sharded-v2-mcast-host` | 4 | 0 | 0 | +168 / −376 |
| Total | 10 | 0 | 0 | +452 / −593 |

Net reduction of 141 production lines. No in-context performance run was requested, so no performance
delta is claimed.

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

## Deferred kernel work

The 81 deferred kernel rows remain outside this rollout. Principal production blockers:

- matmul in0 control channels needing typed/custom control values or independent data/signal loopback;
- width-sharded Conv2D activation multicast (prior port failed 25 numerical cases; baseline restored);
- block-sharded Conv2D activation multicast, whose producer-overlapped chunked send the current helper
  cannot express;
- LayerNorm channels needing acknowledged signal-only, mixed-mode streaming, or explicit
  include-source loopback;
- TopK no-handshake receiver initialization hazards.

`reconcile_2026-08-03.md` additionally lists **10 deferred rows whose upstream churn touched protocol
lines**, so their tags/notes may have drifted (loudest: the three `sort/` kernels, whose
`cores_to_coordinator_semaphore` was split into `ready` + `done`, and
`unified_routed_expert_ffn_reader.cpp`, which lost a `noc.async_write_multicast`). Nothing was
re-tagged — a re-audit is an open follow-up, not part of this pass.

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

Two on-branch commit *messages* still cite pre-rebase hashes (`baa86dc7116` "…for 75b977e1a04",
`5320c2d69bd` "…for 261e322ed22") — history is immutable; this table is the key.

The authoritative machine state is `migration/ledger.json`; this report is its regenerated run view.
