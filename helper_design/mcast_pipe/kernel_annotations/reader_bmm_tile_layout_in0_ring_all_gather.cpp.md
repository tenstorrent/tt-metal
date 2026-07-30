# reader_bmm_tile_layout_in0_ring_all_gather.cpp — annotation

Role: **Ring / neighbor unicast forward**, NOT a multicast Pipe. Object API. Included for census completeness (grep family would NOT flag it — no multicast at all).

## What it does
Each worker forwards its shard to the NEXT core in a ring and signals it:
- L62-63: `local_shard_read_addr`, `l1_write_addr_in0` setup.
- L76: **wait** `signal_sem.wait_min(shard_cnt)` — COUNTER-style (monotone, no reset). Wait until predecessor has delivered.
- L81-88: **data UNICAST** `noc_obj.async_write(CoreLocalMem(read), UnicastEndpoint, shard_size_bytes, ..., {.noc_x=next_core, .noc_y=next_core, .addr})` — point-to-point to the single next core. NOT a multicast.
- L91: **signal-forward** `signal_sem.up(noc_obj, next_core_noc_x,y, 1)` — increment the next core's counter.
- L96: `cb_in2.push_back`.
- L106: final `async_atomic_barrier()`.

## Fork signature
- **F1**: barrier (atomic barrier at end only).
- **F2**: pure COUNTER (`wait_min` + `up`, monotone, never reset).
- **F3**: N/A (unicast, no rectangle, no src semantics).
- **KNOB**: N/A.

## Verdict
**defer/raw — outlier.** No multicast, no rectangle, no two-sided sender/receiver-rect handshake. It is a ring all-gather using unicast + a monotone semaphore. It should NOT migrate to a mcast Pipe; it would belong to a different "ring forward" helper if anything. The only shared DNA is `Semaphore::up`/`wait_min` and `CoreLocalMem`/`UnicastEndpoint`.
