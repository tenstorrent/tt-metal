# r04-b03-a01 result: 1.3543 (ok)

## What happened vs expected
Valid on every shape. PCC 0.9999985 and max_abs 0.0204-0.0240 are the parent's values, so the forwarder never read a
stale stick. Parent r03-b02-a02 -> this node (µs, chip mean):

| shape | parent | this | change |
|---|---|---|---|
| h3584 | 12.59 | 12.33 | -2.1% |
| h4096 | 14.05 | 13.76 | -2.1% |
| h6144 | 17.55 | 17.54 | -0.1% (noise) |
| h7168 | 19.32 | 18.94 | -2.0% |

Score 1.3543 vs 1.3333 (+1.6%, outside the ±1% noise band). That makes it the new campaign best. I predicted
-0.2..-0.3 µs per shape and +1.5-2%. Three shapes landed there; h6144 did not move on the chip mean.

## Why (profiler evidence)
`push.py` (in this dir; `python3 push.py $DREAM_HOME/rmsnorm-prefill/reports/<node>`; outputs `push_out.txt` and
`push_parent_out.txt`). Medians over measured calls x 4 chips, µs from the call's first worker BRISC start:

| shape | W_PUSH dur med | slowest push end | F_COLLECT end (last fwd) | AG-wait end max | drain end max |
|---|---|---|---|---|---|
| h3584 | 0.64 -> **0.24** | 4.96 -> 4.51 | 4.84 -> 4.62 | 7.73 -> 7.45 | 11.95 -> 11.71 |
| h4096 | 0.63 -> **0.25** | 5.62 -> 4.98 | 5.47 -> 5.07 | 8.53 -> 8.23 | 13.71 -> 13.31 |
| h6144 | 0.63 -> **0.25** | 7.39 -> 7.27 | 7.25 -> 7.36 | 10.40 -> 10.12 | 17.16 -> 17.10 |
| h7168 | 0.64 -> **0.25** | 8.19 -> 7.79 | 8.08 -> 7.88 | 11.53 -> 11.14 | 18.68 -> 18.23 |

- **The push handshake lost ~0.39 µs on every core and every shape.** The parent's min push was 0.48 µs and its
  median 0.64. Now both are 0.24-0.25. Two round trips (write ack, atomic ack) used to sit inside the push; now the
  push is just issue + flush + inc issue.
- **The forwarder now sees the inc ~0.10 µs after the slowest worker's push returns.** In the parent it saw it
  0.12-0.14 µs *before* the push returned, because the worker was still waiting for the atomic ack. So the inc reaches
  the forwarder ~0.2-0.4 µs earlier. AG-wait end max moved 0.28-0.39 µs earlier on every shape, h6144 included.
- **h6144:** the AG ended 0.28 µs earlier, yet the drain end max moved only 0.06 µs and the chip mean was flat. That
  shape's post-AG tail is the per-core-bound drain (r03-b02-a03, r03-b04-a02: drain end lags TRISC end by up to
  ~2 µs), so starting it earlier shifted less into the kernel end. The other shapes carried ~0.25-0.4 µs through.
- Ordering held: same NoC + same VC + flushed before the inc means the inc is delivered after the payload (the fabric
  EDM relies on the same rule). No accuracy change in 13 calls x 4 shapes x 4 chips. As the proposal noted, the test
  can't catch a stale-slot race after the first call, because every call writes identical sticks. The safety
  argument is architectural.

## Classification
win (+1.6% geomean, three shapes -2.0..-2.1%, h6144 neutral). It removes a fixed ~0.39 µs from the AG-start critical
path on every core.

## What a child of this node should try next
1. **Port r03-b04-a03's streamed gamma (8-page chunks, sticky trid) onto this node.** It is writer-only and orthogonal.
   It removed a dev-0 straggler on h7168 (-0.8 µs there) that this lineage still has. Keep this node's push
   (`push_stick` is also called from inside the gamma loop, so merge the two carefully). Expected score ~1.37-1.38.
2. **Same "flush, then inc on the same VC" rule on the forwarder side** (fork `dit_fused_norm_forwarder.cpp` into
   the op dir, as r02-b02-a04 did). In F_FABRIC, the forwarder runs `async_write_barrier` + `async_atomic_barrier`
   before it releases go, then an atomic barrier after the 10 go incs per round. The release incs don't need acks
   before the kernel ends. Smaller expected gain (the barriers there are mostly hidden behind the out_ready wait), so
   measure F_FABRIC end -> W_AGWAIT end first.
3. **The remaining 0.24 µs push** is mostly `async_writes_flushed` plus zone overhead. One 128 B write instead of two
   64 B writes would need compute to pack the stat row contiguously (face_00 row 0 + face_01 row 0 are 1 KB apart).
   Probably <0.05 µs. Low priority.
4. **h6144 is drain-bound after the AG.** Any further AG-path cut will show up on h3584/h4096/h7168 but not h6144. To
   move h6144, the drain itself has to get faster (per-core NoC0 share, r02-b02-a01 #1).
