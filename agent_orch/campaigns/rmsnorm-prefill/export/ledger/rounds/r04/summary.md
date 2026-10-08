# Round r04 summary (policy v3, W=4, R=3, root r03-b02-a02 = 1.3333)

12 attempts in 3 steps. No branch closed (the gap and stall rules never fired). All committed, verified and valid.

## Best
r04-b01-a03, score 1.4475: 11.67 / 13.05 / 16.35 / 17.35 µs (1.456 / 1.399 / 1.436 / 1.500x). PCC 0.9999985, max_abs 0.022-0.024.
NOTE: this node runs PRE (x*x and the row-sum matmul) at MathFidelity::HiFi2 (from r04-b01-a02). Whether lower
fidelity is a legal optimization is still an open question for the human. Best all-HiFi4 node: r04-b04-a02, 1.3997.

## Grid (score)
| branch | a01 | a02 | a03 |
|---|---|---|---|
| b01 | 1.344 | 1.391 (HiFi2) | 1.448 (HiFi2) |
| b02 | 1.299 | 1.318 | 1.422 (HiFi2) |
| b03 | 1.354 | 1.391 | 1.389 |
| b04 | 1.367 | 1.400 | 1.395 |

## What worked
- Three independent writer-side wins in step 1: posted (no-ack) output drain (b04), ack-free stick push (b03), streamed gamma ported onto the best node (b01).
- Stacking them across branches in steps 2-3 (b03-a02, b04-a02, b01-a03, b02-a03): each stack beat its parts.
- PRE at HiFi2 (b01-a02): about +3.5% with unchanged PCC.

## What failed and why
- Forwarder-push / streamed-multicast AG release (b02): below the stack on its own; only competitive after porting the stack.
- Per-worker DRAM bank de-phasing on the 20-core layout (b03-a03): neutral, reverted.
- Position-aware input-read depth (b04-a03): neutral.

## Policy observations (input for dreaming)
- Width paid off again: the three step-1 winners were different mechanisms, and the best nodes are cross-branch stacks.
- Workers port ideas across branches inside a round (they read sibling nodes via git), so cross-branch merging happens without merge commits.
- The v3 stop rule never fired, because every branch was still improving. In replay, v3 on r04 gets 1.4275 (best 1.448, all 12 attempts).
- Remaining bottleneck named by several reflections: cross-device launch skew (dev0 vs dev3 differ by ~2 µs at h7168).

## Ops
- Root re-measure: no drift. One transient ssh disconnect during verify; retried fine.
