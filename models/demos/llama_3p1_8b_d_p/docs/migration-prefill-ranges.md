# Native prefill ranges, continuation and slot reuse

**Result: PASSED on 18 September 2026.** The test ran seven real 32-layer Llama-3.1-8B-Instruct prefill calls. The native tt-d-gen manager moved six selected ranges into passive destination caches. Every selected packed page matched its saved source. Both managers and both device owners shut down cleanly.

## Configuration and result

| Item | Value |
|---|---|
| Source | One Galaxy, SP=4 / TP=8, 32 layers |
| Destination | One Galaxy with passive buffers; no decoder |
| Cache | BFP8_B, two slots, 2,048-token capacity |
| Compute chunk | 1,024 tokens |
| Transfer page | 32 tokens, 4,352 packed bytes |
| Configurations | 8 K heads and 8 V heads |
| Full-model calls | 7 |
| Post-write layer acknowledgements | 224 |
| Native layer commands | 224 |
| Exact destination checks | 6 generations |
| Selected page visits | 24,576 |
| Selected packed bytes | 106,954,752 |
| Controller, owners, bridges and managers | Successful completion and clean shutdown |

The source bridge code is published at [tt-d-gen 6eeb1a9](https://github.com/tenstorrent/tt-d-gen/commit/6eeb1a9c4a86e4b4e1795665bd7a504b216e504f). The [JSON report](migration-prefill-ranges.json) records the checks and evidence hashes.

## What each transfer tests

Intervals use an exclusive end: [32, 65) includes tokens 32 through 64.

| Generation | Source → destination slot | Prompt length | Selected interval | Purpose |
|---|---|---:|---|---|
| A100 | 0 → 0 | 1,033 | [0, 33) | A short selected range inside a longer computed prompt. |
| A101 | 0 → 0 | 1,033 | [32, 257) | A nonzero start and an overlapping boundary page. |
| A102 | 0 → 0 | 1,033 | [256, 1,033) | A range that crosses the 1,024-token compute boundary and reaches a partial tail. |
| B103 | 1 → 1 | 257 | [0, 257) | A different prompt in the other slot. |
| C104 | 0 → 1 | 33 | [0, 33) | Reuse a drained source slot for a distinct prompt, with crossed destination mapping. |
| C105 | 0 → 1 | 65 | [32, 65) | Continue C from the rounded-down tile boundary and preserve its earlier prefix. |

A100 computes [0, 1024) and [1024, 1033). A101 and A102 replay the same final compute tail to exercise new transfer requests over resident data. B103, C104 and C105 each make one compute call. The test does not treat those A-tail replays as new prompt progress.

## How the comparison works

1. Initialize both caches and save their packed bytes.
2. Register the source request and let the passive receiver arm its selected range.
3. Send real tokens through the persistent H2D input service. Complete each model write and synchronize it. Save source bytes before publishing layer readiness.
4. Transfer the ready selected range through the native manager.
5. Compare every selected destination page with its saved source page, including BFP8 exponent bytes. Compare every unselected destination page with its previous snapshot.
6. Retire the transfer before advancing to the next generation. Reclaim A before reusing its source slot for C.
7. Drain work and verify both exact manager exits before either cache owner releases its buffers. Then verify both device closes.

All decoded source values must be finite. Every valid row must contain a nonzero value and must not remain entirely equal to its initial pattern. Eligible new writes must also change at least one valid value in each configuration/layer group. C105 must change newly valid rows, not just its replayed overlap. The exact A-tail replays have a narrow exemption from this progress check.

Transfer boundaries select **whole 32-token pages**. For A100, the page containing token 32 also contains valid A tokens 33 through 63. They are copied as part of that page. They are not padding and are not required to be zero. Padding after the actual valid source tail is checked separately.

## What this result establishes

The test covers selected-range addressing, all 16 K/V configurations, all 32 layers, ordinary and crossed slot mappings, continuation from resident prefill KV, slot isolation, safe slot reuse and cache lifetime through native shutdown.

This is a byte-preservation and runtime-contract test. Use the accepted 2K numerical report for model accuracy. This report adds no new PCC threshold, timing claim, larger-capacity result, cancellation result, decode run or decode-to-prefill session import.

The recorded device run used the frozen deployment fixture. The portable publication of that fixture has separate host validation. Relocating the fixture is not a new device run.
