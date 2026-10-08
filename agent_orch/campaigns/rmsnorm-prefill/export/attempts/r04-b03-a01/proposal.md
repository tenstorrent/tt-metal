# r04-b03-a01: stick-push handshake without the write-ack round trip: flush the two 64 B stick writes, then fire the arrival inc on the same NoC/VC (in-order delivery), atomic barrier deferred to kernel end

## Motivation
The all-gather can't start until the slowest worker's stick has arrived at its forwarder (F_COLLECT waits on
fwd_arrival_sem). The slowest pusher's W_PUSH end gates F_COLLECT on every shape (r03-b01-a03 `pre.py`: W_PUSH end
max = 4.8 / 5.6 / 7.6 / 8.1 µs, then AG). W_PUSH lasts **0.63 µs on every core even when the stat is already
there** (r03-b01-a03: pushes fired from inside the gamma poll, where the stat was checked ready beforehand, still take
0.63 µs). That is ~850 cycles for two 64 B writes and one atomic. The push does today:

1. 2 x 64 B NoC writes to the forwarder's packet buffer
2. `async_write_barrier()`: waits for the write **acks** (a full round trip to the forwarder core)
3. `fwd_arrival_sem.up(noc, fwd)`: atomic inc
4. `async_atomic_barrier()`: waits for the atomic **ack** (a second round trip)
5. pop the stat CB

The forwarder only needs the inc to arrive after the payload. It doesn't need the worker to see the write acks first.

## Mechanism
In `dit_rmsnorm_fused_worker_writer.cpp` `push_stick`:
- replace `noc.async_write_barrier()` with `noc.async_writes_flushed()`. That waits only until the write requests
  have left this core's NIU, which is also the condition for reusing the source CB slot.
- issue the inc on the same `noc` object with the default `NOC_UNICAST_WRITE_VC`. The stick writes use that VC too.
  Same source, same destination, same VC, same dimension-ordered route, so the NoC delivers the inc after the
  payload. This is the ordering the fabric EDM relies on for its flush=true fused write+inc
  (`fabric_edm_packet_transmission.hpp` NOC_FUSED_UNICAST_ATOMIC_INC: `flush_write_to_noc_pipeline` then
  `noc_semaphore_inc` on the same noc/vc).
- drop the per-push `async_atomic_barrier()`. Nothing in the kernel depends on the inc's ack. Add one
  `noc.async_atomic_barrier()` at kernel end, so no atomic is outstanding when the kernel exits.

Only that file changes. The forwarder, compute, reader and host are untouched.

## Why this is not a repeat
- r03-b01-a03 moved *when* the push runs (from inside the gamma read poll). This node shortens the push itself.
  r03-b01-a03's reflection #1 proposed exactly this and nobody has tried it.
- r02-b02-a04 changed the go *release* (forwarder multicast) and address pre-staging: other end of the AG, neutral.
- r03-b02-a03 / r03-b03-a03 tried more cmd bufs and VCs on the *drain*. Here the write and the inc deliberately share
  one VC so they stay ordered.

## Expected effect and risk
- W_PUSH duration ~0.63 -> ~0.2-0.35 µs. The forwarder sees the slowest worker's arrival ~0.2-0.3 µs earlier, so
  the AG starts and ends earlier and the whole post-AG tail shifts left: ~-0.2..-0.3 µs per shape, ~+1.5-2% score.
  Check: W_PUSH dur and W_PUSH end max vs r03-b02-a02 in the profiler log; F_COLLECT end earlier.
- Accuracy: if the ordering assumption were wrong, the forwarder would occasionally ship a stale stick. That shows as
  a PCC/max_abs failure. Caveat: every call feeds the same input, so a stale slot (packet_buf[r%2]) would hold the
  previous call's identical stick, and the test could miss the race in all calls but the first warmup. The guarantee
  is architectural, not measured: same-source same-VC in-order delivery is what fabric's fused write+inc depends on
  in production.
- No hang risk: the counts are unchanged, only the ack waits move.
