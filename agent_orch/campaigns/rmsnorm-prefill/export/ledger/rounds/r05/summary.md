# Round r05 summary (policy v4, W=4, R=3, root r04-b04-a02 = 1.3997, all HiFi4)

8 attempts in 3 steps. v4 closed b02 and b04 after step 1. All committed, verified (incl. the new forbidden_patterns check) and valid.
An earlier round-5 start rooted at the HiFi2 node r04-b01-a03 was aborted before any commit (ledger: aborted/r05-hifi2-root).

## Best
r05-b01-a01, score 1.5696: 11.19 / 12.08 / 14.59 / 15.84 µs (1.518 / 1.512 / 1.610 / 1.643x). PCC 0.9999985, all HiFi4.

## Grid (score)
| branch | a01 | a02 | a03 |
|---|---|---|---|
| b01 | 1.570 | 1.432 | 1.466 |
| b02 | 1.408 (closed) | | |
| b03 | 1.517 | 1.422 | 1.488 |
| b04 | 1.194 (closed) | | |

## What worked
- k=2 column split run as two 20-core AG waves (b01 with a forked two-wave forwarder, b03 with the stock forwarder): wave B's input read overlaps wave A's AG, and A's drain overlaps B's AG. +12% over the root. It combines the r01 column split and the r03 two-wave idea, which both failed on their own.
- Un-gating each wave's stick push from the streamed-gamma barrier (b01-a03, b03-a03): it repaired the 4-wave regressions, and both reflections say it should also help the 2-wave best.

## What failed and why
- 4 waves on 80 quarter-row workers (b01-a02, b03-a02, found independently in parallel): the stick push was gated on gamma landing after the whole input stream. Repaired by a03, but still below 2 waves.
- Half-row two-round pipeline on 20 cores with the stock forwarder (b04): -15%.
- Row stat via matmul diagonal at HiFi4 (b02): within noise of the root.

## Human decision this round
- No reduced math fidelity (HiFi2/HiFi3/LoFi) and no new approximate modes. r04-b01-a02, r04-b01-a03 and r04-b02-a03 are overridden as forbidden_edit. campaign.yaml rules + forbidden_patterns are enforced by verify_node and told to every worker.

## Policy observations (input for dreaming)
- The r04 replay changes: its best is now 1.3997 (the HiFi2 nodes are invalid).
- The biggest jump of the campaign came from a structural recombination of two earlier failed ideas. Width plus history made that possible.
- Parallel same-step duplication again: both open branches tried 4 waves in step 2 with the same bug.
- Step 3 repairs recovered but did not beat the leader. The obvious next step (port the un-gate fix to the 2-wave best) needs a new round rooted at r05-b01-a01.

## Ops
- Root re-measure: no drift. v4 replay incl. r05: mean V 1.3444.
