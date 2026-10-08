# r04-b03-a02: stack r04-b04-a01's posted (no-ack) output-drain writes onto this node's ack-free stick-push handshake

## Motivation
Two round-4 wins sit on sibling branches of the same root (r03-b02-a02) and touch disjoint phases of the same writer kernel:
- r04-b03-a01 (parent, 1.3543): the AG-start stick push drops the write-ack + atomic-ack round trips
  (W_PUSH 0.64 -> 0.25 µs). AG-wait end moved 0.28-0.39 µs earlier on every shape. h6144 did not move because its
  post-AG tail is the per-core-bound output drain (drain end lags pack POST end by ~2 µs).
- r04-b04-a01 (1.3666): drain writes are NocOptions::POSTED. Per-tile drain rate is ~7% faster and the drain tail after
  pack POST end shrinks 0.18-0.46 µs (h6144: 1.98 -> 1.55 µs).

They act on different parts of the critical path: before the AG and after the AG. The parent's own reflection says h6144
only moves if the drain gets faster. That is exactly what the posted drain does.

## Mechanism
Writer kernel only (`dit_rmsnorm_fused_worker_writer.cpp`):
- Output tiles go out with `async_write<NocOptions::POSTED>` on both `noc` and `noc_alt`.
- Each block flushes with `async_writes_flushed<NocOptions::POSTED>()` before `pop_front`.
- At kernel end: write barrier, posted flush, and the parent's deferred atomic barrier for the arrival incs.

The code is r04-b04-a01's diff applied on top of the parent. The only conflict was the kernel-end barrier block,
which now keeps both lines.

## Why this is not a repeat
This is a combination of two validated, orthogonal pieces that no node has tried together. r04-b04-a01 lacks the push
handshake change and r04-b03-a01 lacks the posted drain. No new mechanism, by design: the parent's #1 suggestion (gamma
streaming port) is the obvious choice for sibling branches b01/b04, so I take the other pairing.

## Expected effect and risk
The gains should be roughly additive: parent µs minus r04-b04-a01's per-shape drain gain.
- h3584 ~12.1, h4096 ~13.4, h6144 ~17.2-17.3, h7168 ~18.3. Score ~1.38.
- h6144 may gain more than either alone, because its earlier AG end is no longer absorbed by the drain.

Risk is low. Both pieces were valid on HW with identical PCC/max_abs. The push uses non-posted stick writes plus a
same-VC inc, and runs before any posted drain traffic, so they don't interact. Same production caveat as r04-b04-a01:
posted writes may still be in flight at kernel end.
