# r04-b04-a02: stack round 4's two other writer-only wins on the posted drain: ack-free stick push (r04-b03-a01) + streamed gamma chunks (r04-b01-a01)

## Motivation
Round 4 produced three wins, all off r03-b02-a02, all confined to `dit_rmsnorm_fused_worker_writer.cpp`, each
attacking a different cost:
- r04-b04-a01 (parent, 1.3666): posted output-drain writes, ~7% faster drain rate, -0.2..-0.67 µs (post-AG tail).
- r04-b03-a01 (1.3543): stick push flush + inc on the same VC, no ack waits: W_PUSH 0.63 -> 0.25 µs, AG ends
  0.28-0.39 µs earlier on every shape (AG-start critical path).
- r04-b01-a01 (1.3436): gamma streamed to compute in 8-page sticky-trid chunks: removes the dev-0 h7168 cross-call
  straggler (9/10 -> 0/10 calls), h7168 -0.64 µs.
The parent's reflection #1 and r04-b03-a01's #1 both name these ports as the cheapest next best. No node has all three.

## Mechanism
Writer only. Apply r04-b01-a01's W_GAMMA diff and r04-b03-a01's push_stick diff on top of the parent:
- `push_stick`: `async_writes_flushed()` instead of `async_write_barrier()` before `fwd_arrival_sem.up`, drop the
  per-push `async_atomic_barrier()`; one atomic barrier at kernel end.
- W_GAMMA: 8-page chunks with sticky read trids 1..4, chunk barrier + `push_back` 2 chunks behind the issue front,
  bank rotation within a chunk, stick poll in the loop; default trid restored after.
- Drain keeps the parent's posted writes. The only merge point is the kernel-end fences: write barrier, posted
  flush, atomic barrier.

## Why this is not a repeat
A combination of three validated, disjoint pieces (drain / push / gamma code blocks). Each was measured only alone on
r03-b02-a02. The push fix moves the AG earlier; the posted drain shortens the tail after it; on h6144 the push fix was
neutral because that shape is drain-bound, which the posted drain addresses, so they may compound there.

## Expected effect and risk
Roughly additive: h3584 ~12.0, h4096 ~13.3, h6144 ~17.2, h7168 ~18.0-18.3 µs; score ~1.39-1.40.
Risk is low (each piece is HW-valid). Interactions to watch: the push now runs inside the gamma trid loop with a
non-posted flush (writes, unaffected by the sticky read trid); posted and non-posted counters are separate.
A regression on one shape vs the parent would point to an interaction (check W_PUSH and W_GAMMA zones).
