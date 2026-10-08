# r03-b01-a03: stick push no longer waits behind the gamma read barrier: BRISC polls for the row-0 stat while its broadcast-gamma reads land, instead of blocking in async_read_barrier

## Motivation
The all-gather starts when the forwarder has all 20 sticks (F_COLLECT end = the slowest worker's W_PUSH end). On
this lineage BRISC issues 2 x num_tile_cols broadcast-gamma face-row reads at kernel start and polls for compute's
row-0 stat between read issues, so it can push the stick early (r01-b04-a04). But after the issue loop it calls a
**blocking** `noc.async_read_barrier()` for the gamma reads. A stat that becomes ready while BRISC sits in that barrier
is not pushed until the last gamma read lands.

Evidence from the parent's profile (`reports/r03-b01-a02`, all 4 chips, measured calls; script `gam.py` in this dir):

| shape | W_PUSH inside the gamma issue loop | W_PUSH within 0.15 us after W_GAMMA end | W_PUSH later than that | slowest pusher: push start - W_GAMMA end |
|---|---|---|---|---|
| h3584 | 740 / 800 | 60 | **0** | -1.00 us (in loop) |
| h4096 | 678 / 800 | 122 | **0** | -1.22 us (in loop) |
| h6144 | 566 / 800 | 234 | **0** | **+0.05 us** |
| h7168 | 344 / 800 | 456 | **0** | **+0.06 us** |

No push ever starts more than 0.15 us after W_GAMMA ends, and on h6144/h7168 the push of the slowest pusher (the one
that gates F_COLLECT) starts right after the gamma barrier. If the stat were independent of gamma, some pushes would
land later. So the stat is ready during the barrier and BRISC is holding it. For the slowest pusher at h7168 the push
starts 1.16 us after its input read ends, while the input-landed -> stat-ready tail at the fastest cores is ~0.64 us
(`pre.py`: push_start - R_INPUT end, min/median/max = 0.64 / 0.83 / 1.18 at h7168, 0.64 / 0.84 / 1.30 at h6144).

## Mechanism
`kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp` only. After the gamma issue loop, replace the blocking
`noc.async_read_barrier()` with a poll loop: while the gamma reads have not all landed (the same NIU read-response
counter check that `noc_async_read_barrier` spins on, `ncrisc_dynamic_noc_reads_flushed` in DM_DYNAMIC_NOC), keep
checking `cb_stats_local.pages_available_at_front(num_stats)` and push the stick as soon as it is there. Then the
(now instant) barrier and `cb_weight.push_back` as before. Compute needs gamma only after it has produced the stat
(x*gamma runs after PRE), so the weight push is not delayed in the common case. A nested `W_GBAR` zone around the poll
makes the gamma-landing wait visible.

## Why this is not a repeat
- r01-b04-a04 added the in-loop poll (stick preempts the gamma issue loop) and saw "the poll never fired" because PRE
  was slower then. Now it fires on 43-93% of core-calls, but the window after the issue loop (the barrier) was never
  covered. This node closes that hole.
- r03-b03-a02 / r02-b02-a0x tried to make the stat ready earlier in compute; this is the BRISC side: the stat is
  already ready and waits for BRISC.
- No change to compute, reader, forwarder, NoC choice, or the push handshake itself.

## Expected effect and risk
- h6144 / h7168: F_COLLECT and everything after it earlier by ~0.3-0.5 us (slowest pusher drops from ~1.2 us after
  its read end to ~0.7-0.9). Kernel end moves the same, since the post-AG chain is a fixed shift. ~-2% on those shapes.
- h3584 / h4096: slowest pusher is already in-loop; ~0 to -0.1 us.
- Geomean expected +1 to +1.5%, so near the noise band. The per-core table from `gam.py` (no push right after
  W_GAMMA end on the slowest pusher; W_PUSH end max earlier) is the real check.
- Risk: the push writes + atomic go out on BRISC's NoC while gamma reads are outstanding. That already happens with
  the in-loop preempt, so it's not new. Accuracy is unchanged (same data). No CB / L1 change. If the counter check
  were wrong it would hang or push weight early; I use the same check the barrier uses, followed by the real barrier.
