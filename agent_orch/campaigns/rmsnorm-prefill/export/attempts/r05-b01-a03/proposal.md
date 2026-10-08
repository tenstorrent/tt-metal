# r05-b01-a03: un-gate the stick push from the streamed-gamma trid barrier: the W_GAMMA loop spins on `is_read_trid_flushed` and keeps polling for the row-0 stat instead of blocking in `async_read_barrier<TXN_ID>`, so each of the 4 waves pushes as its own PRE finishes

## Motivation
The parent r05-b01-a02 (4 waves of 20 quarter-row workers, 1.4320) lost to its 2-wave parent r05-b01-a01 (1.5696).
Its reflection found why, and r05-b03-a02 (the same 4-wave design on another branch, 1.4219) found the same thing
independently:
- The reads pipelined as designed. Wave read ends at h7168: 1.97 / 3.36 / 4.85 / 6.18 µs.
- The stick pushes didn't. The push ends were 3.64 / 6.68 / 6.92 / 7.21 µs at h7168 and 4.40 / 4.36 / 4.44 / 4.81 µs
  at h3584. Every gated push ends at W_GAMMA end.
- The cause is in the writer. `poll_stick()` runs only between read issues and between chunks. The chunk barrier
  `noc.async_read_barrier<TXN_ID>` blocks without polling.
- The gamma face-row reads queue in DRAM behind all 80 cores' input reads, so they land only when the whole input
  stream has drained, at about the last wave's read end.
- At ≤ 8 gamma pages per core (h3584/h4096) there is one chunk, so every wave waits on it. At 12/14 pages there are
  2 chunks: wave 0 escapes and waves 1-3 wait.
- Result: the forwarder sends bunch up (F_SEND 0.25-0.3 µs apart), the gos come ~0.55 µs apart, and the drains
  overlap and contend. The 4 waves give no AG overlap, only extra serial releases.

## Mechanism
One change in `dit_rmsnorm_fused_worker_writer.cpp`, in the W_GAMMA loop. Replace the blocking
`noc.async_read_barrier<NocOptions::TXN_ID>({.trid = t})` with:
```
while (!noc.is_read_trid_flushed(t)) { poll_stick(); }
noc.async_read_barrier<NocOptions::TXN_ID>({.trid = t});  // returns at once; keeps the L1 invalidate
```
`poll_stick()` pushes the first row's stick as soon as compute's row-0 stat is in `stats_transposed_local_cb`. Compute
pushes that stat before its x*gamma pre-pass waits on `weight_cb`, so there is no deadlock: the stat never depends on
gamma. Everything else is the parent's code: S=4, the 4-wave forwarder, chained start sems, and the 1280 B group read.

## Why this is not a repeat
This is the repair both 4-wave nodes named as their #1 next step: r05-b01-a02 #1 and r05-b03-a02 #1 (the
`is_read_trid_flushed` variant). Neither node ran it. Earlier push-path changes did different things:
- r03-b01-a03 polled during the old single gamma barrier.
- r04-b03-a01 made the push handshake ack-free.

Neither applies to the streamed-gamma chunk barrier, which r04-b01-a01 introduced.

## Expected effect and risk
Wave pushes should track each wave's read end + ~0.5-0.9 µs. Targets: h7168 ≈ 3.0 / 4.4 / 6.0 / 7.2 and h3584 ≈ 2.2 /
3.2 / 4.0 / 4.8. The forwarder sends and gos should step up per wave by roughly the read stagger. The model
`end ≈ T_read + chain + T_drain/4` gives ~13 µs at h7168 if the chain holds. A realistic outcome is somewhere between
the parent (17.0) and that.

Risks:
- The narrow shapes may still lose to 2 waves. The 4-link start chain costs ~0.5 µs, and these shapes are bound by
  host launch skew.
- The per-wave AG (~3 µs send -> go, partly cross-chip skew) may still stack up at the forwarder's in-order release.
- Accuracy can't change: same data, same math.
- Hang risk is low: a non-blocking spin over the same condition the barrier waits on.

Judge with the parent's `analysis/wavesN.py` (push end and go per wave).
