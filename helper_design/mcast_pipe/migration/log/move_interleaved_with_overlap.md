# Move overlap control — tiled path migrated at API v11

Date: 2026-08-16

## Verdict

Migrated in `a25603ae2c0`. The old API-v7 design-gap finding is obsolete. API v11 allows
host-selected semaphore IDs and opaque per-core runtime roles, while the legacy dual-use semaphore is
cleanly decomposed into the operation-owned worker return counter and helper-owned release flags.

## Protocol mapping

- Semaphore 0 remains the operation-owned return counter over all work cores. Every worker atomically
  increments it at the controller coordinates, and the controller waits for `num_cores - 1` arrivals.
- Three fixed, no-handshake `Mcast2D` Flag wires cover the disjoint worker rectangles. The host always
  emits three opaque CT/RT blocks; when only two rectangles exist, the third is an inactive controller
  singleton. Helper semaphores 1-3 cannot alias the return counter.
- Runtime `release_region` selects the appropriate receiver face under the shared kernel binary. The
  controller sentinel selects the three sender faces. No helper API role extension is needed.
- The controller sends one signal on each active rectangle only after the operation return counter is
  complete. Workers wait on only their selected release wire before writing the staged CB to output.
- Source/destination buffer addresses remain runtime slots 0/1, so the existing program-cache override
  continues to patch exactly the mutable fields. All helper-generated arguments are appended as opaque
  ranges after the fixed operation arguments.

## Validation

- Production LOC: tiled kernel 30 additions / 40 deletions; factory 46 / 47.
- Release builds passed after migration, in the exact raw source state, and after byte-identical restore.
- Fresh-cache TILE `[1,3,320,384]` BFLOAT16 L1-to-L1 overlap passed under Watcher with 0/19 JIT hits.
- Complete `test_move.py`: 136 passed / 128 intentional non-L1 skips, including all cache cases.
- Shared guards: `McastHostFixture.*` 34/34, `test_mcast_pipe.py` 80/80 under Watcher, source audit 22/22.
- Matched 800 MHz Tracy, three 25-operation sessions per state with the first five samples discarded:
  raw 4,144.5 ns, migrated 4,125 ns, -0.47%.

## Claude consultation

The architecture consultation used the user-required Opus command and timed out after five minutes
without a verdict. No approval was inferred. A final fact-complete post-validation review was also
requested; its result is recorded in the rollout tracker. No helper API expansion was made.
