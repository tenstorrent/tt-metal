# Move overlap control — row-major path migrated at API v11

Date: 2026-08-16

## Verdict

Migrated with the tiled twin in `a25603ae2c0`. It uses the same three helper release wires and separate
operation-owned worker return counter. The stick/page data path and aligned-page stride are unchanged.

## Protocol and ABI

- Semaphore 0 remains the return counter; helper Flag semaphores 1-3 release the disjoint rectangles.
- Three opaque helper CT/RT blocks are emitted unconditionally, with an inactive controller singleton
  when the third rectangle is absent.
- Runtime `release_region` selects controller or receiver behavior in the shared kernel binary.
- Fixed runtime slots 0/1 remain source/destination addresses for cache override; row-major slot 5
  remains `aligned_page_size`, and the first helper wire starts after it.
- No raw multicast primitive or parallel legacy ABI remains, and API v11 required no expansion.

## Validation

- Production LOC: row-major kernel 31 additions / 41 deletions; shared factory 46 / 47.
- Fresh-cache ROW_MAJOR `[1,3,320,384]` BFLOAT16 L1-to-L1 overlap passed under Watcher with 0/19 hits.
- Complete operation, cache, host-helper, device-helper, source-audit, and Release-build evidence is
  shared with the tiled atomic unit and passed in full.
- Matched 800 MHz Tracy, three 25-operation sessions per state with the first five samples discarded:
  raw 6,190 ns, migrated 6,189 ns, -0.02%.

## Claude consultation

The shared architecture consultation timed out after five minutes without a verdict; no approval was
inferred. The final fact-complete review result is recorded in the rollout tracker. No API expansion
was made.
