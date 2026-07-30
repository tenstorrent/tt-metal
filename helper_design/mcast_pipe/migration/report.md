# mcast_pipe rollout report — API v9, completed 2026-07-30

## Run header

- Helper: `mcast_pipe`, `MCAST_PIPE_API_VERSION=9`
- Entry mode: re-entry
- Invocation mode: `run-all`
- Baseline: `origin/llk_helper_library` at `54d8dfb7bef`
- Branch: `sjovic/mcast-migration`
- Re-entry worklist: 0 stale kernels, 0 stale host bindings, 10 pending
  host integrations across 4 atomic units, and 0 net-new kernel units
- Device: single-chip Blackhole p100a
- Test runner: repository environment plus `scripts/run_safe_pytest.sh`

The user first approved the easiest height-sharded Conv2D unit, then expanded
the run to every remaining host-helper unit. Device tests were serialized. In
`run-all` mode, a failed unit would have been restored and quarantined before
continuing; no unit ultimately failed or required quarantine.

## Rollout state at v9

| State | Count |
|---|---:|
| kernel-current | 10 |
| host-binding-current | 10 |
| fully end-to-end current | 10 bindings / 10 kernels |
| host-pending | 0 |
| kernel-pending | 0 |
| quarantined | 0 |
| deferred | 82 kernels / 0 host bindings |

All required bindings in the host census are current at v9. “Fully end to end”
is channel-specific: for example, height-sharded Conv2D has a weights
multicast binding but reads activations locally. Block- and width-sharded
Conv2D activation multicast kernels remain in the deferred kernel census and
were not eligible for this host-only worklist.

## Summary by tier

| Tier / atomic unit | Bindings migrated | Failed | Quarantined | Production diff |
|---|---:|---:|---:|---:|
| 1 — `conv2d-weights-single-sender-rect` | 1 | 0 | 0 | +48 / -51 |
| 2 — `conv2d-weights-fixed-line` | 1 | 0 | 0 | +40 / -96 |
| 3 — `matmul-in1-mcast-padding-host` | 4 | 0 | 0 | +196 / -70 |
| 4 — `groupnorm-sharded-v2-mcast-host` | 4 | 0 | 0 | +168 / -376 |
| Total | 10 | 0 | 0 | +452 / -593 |

The production rollout removed 593 lines and added 452, a net reduction of
141 lines. No in-context performance run was requested, so no performance
delta is claimed.

## Per-kernel result

| Kernel | Status | Validation | File deletions |
|---|---|---|---:|
| Conv2D 1D weights sender | migrated, fully end-to-end | exact JIT; height 49/16 skips; DRAM 14/14 | 19 |
| Conv2D 1D weights receiver | migrated, fully end-to-end | exact JIT; height 49/16 skips; DRAM 14/14 | 11 |
| Conv2D fixed-line weights sender | migrated, fully end-to-end | exact PerRow/PerColumn JIT; block 49/16 skips; DRAM 14/14 | 15 |
| Conv2D fixed-line weights receiver | migrated, fully end-to-end | exact PerRow/PerColumn JIT; block 49/16 skips; DRAM 14/14 | 12 |
| Matmul in1 padding sender | migrated, fully end-to-end | exact offset 1D and both 2D orientations; full 302/188 skips | 40 |
| Matmul in1 padding receiver | migrated, fully end-to-end | exact offset 1D and both 2D orientations; full 302/188 skips | 25 |
| GroupNorm v2 legacy sender | migrated, fully end-to-end | exact JIT; parameterized 108/2 skips | 111 |
| GroupNorm v2 legacy receiver | migrated, fully end-to-end | exact JIT; parameterized 108/2 skips | 17 |
| GroupNorm v2 Welford sender | migrated, fully end-to-end | exact JIT; parameterized 108/2 skips | 113 |
| GroupNorm v2 Welford receiver | migrated, fully end-to-end | exact JIT; parameterized 108/2 skips | 20 |

Factory deletions account for the remaining 210 removed lines. Detailed
runtime layouts, JIT hashes, refactor notes, and exact commands are in the four
per-unit logs under `migration/log/`.

## Validation

- Host rebuild passed after every atomic host-code unit.
- Conv2D height-sharded: 49 passed, 16 expected skips; shared DRAM 14/14.
- Conv2D block-sharded: 49 passed, 16 expected skips; shared DRAM 14/14.
- Matmul in1 mapped inventory: 302 passed, 188 expected skips.
- GroupNorm legacy: 108 passed, 2 expected skips.
- GroupNorm Welford: 108 passed, 2 expected skips.
- GroupNorm fixed/default routing: 19 passed, 6 expected skips.
- `McastHostFixture.*`: 19/19 after each unit.
- `test_mcast_pipe.py`: 68/68 after each unit.

The first Conv2D fixed-line PerColumn smoke hung because the migrated sender
read its trailing booleans before advancing over the helper runtime block.
Changing the cursor to `McastArgs::next_runtime_args_offset()` fixed the issue;
the exact retry, opposite orientation, full block inventory, and every shared
regression then passed.

## Coverage gaps

No migrated kernel lacks device runtime coverage. Four host-binding variants
have narrower coverage and remain flagged in the ledger:

- Matmul legacy 1D and 2D constructors compile in the host build, but their
  callers are fused CCL factories; mapped single-chip tests runtime-exercise
  the descriptor constructors only.
- GroupNorm legacy and Welford `use_mcast=false` bindings have direct host
  oracle and device-wire degenerate coverage, but no mapped operation test
  reaches the v2 sender-only route. The same sender kernels are JIT-verified
  through the multicast route.

## Deferred kernel work

The 82 deferred kernel rows remain outside this host rollout. The principal
production blockers are:

- matmul in0 control channels that require typed/custom control values or
  independent data/signal loopback behavior;
- width-sharded Conv2D activation multicast, whose prior port failed 25
  numerical cases and was restored;
- block-sharded Conv2D activation multicast, whose producer-overlapped chunked
  send is not expressible by the current helper;
- LayerNorm channels requiring acknowledged signal-only, mixed-mode streaming,
  or explicit include-source loopback semantics;
- TopK no-handshake receiver initialization hazards.

No partial migration for a deferred or quarantined unit remains in the
worktree.

## Commits

| Unit | Code commit | Ledger/log commit |
|---|---|---|
| Conv2D height weights | `75b977e1a04ee7a14df5d8039393c7844f33fdae` | `de4122b2ca4894aafeebaab751306e5b204a8bde` |
| Conv2D fixed-line weights | `261e322ed2284175e3b4b7b80f98e947b569fe10` | `200e157a8278a25386c8fa440da3a25ca89d1961` |
| Matmul in1 | `2d0280d3dacf8a2ba24882b35816c6a1fbffb7dd` | `080237302623cbdb49e37854cf38a35d43715bee` |
| GroupNorm v2 | `0a796a025c9dc678387e2a7fa52518c898737dc9` | `3cac5e6d0a9d716d640167750852d92913953d07` |

The authoritative machine state is `migration/ledger.json`; this report is its
regenerated run view.
