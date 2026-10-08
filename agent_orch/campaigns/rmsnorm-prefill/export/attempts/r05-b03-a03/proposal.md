# r05-b03-a03: un-gate the stick push from the streamed-gamma barrier on the 4-wave row pipeline (spin on the gamma chunk's read trid while polling for the stat, instead of a blocking trid barrier)

## Motivation
The parent r05-b03-a02 (4 AG waves of 20 quarter-row workers, score 1.4219) lost to its 2-wave parent (1.5174) because
the waves' all-gathers bunched up. Its reflection (and r05-b01-a02's, which hit the same thing) pinned it down:
- the per-wave input reads DID pipeline (R_INPUT end max per wave at h7168: 2.08 / 3.51 / 5.19 / 6.38 µs);
- but the W_PUSH end per wave tracked W_GAMMA end, not R_INPUT end (h7168: 2.96 / 6.55 / 6.82 / 7.41; h3584: 4.50 /
  4.46 / 4.56 / 5.08, while the reads end at 1.35..4.29);
- the cause is in the writer's W_GAMMA loop: `poll_stick()` only runs between read issues, and the chunk barrier
  `noc.async_read_barrier<TXN_ID>` blocks without polling. The gamma face-row reads sit in the DRAM queues behind all
  80 cores' input reads, so they land only at ~the last wave's read end. With ≤ 2 gamma chunks per core, every wave
  (except wave 0 on the 2-chunk shapes) pushes its stick only then.
So the 4-wave premise (wave k's AG overlaps wave k+1's read) was never tested.

## Mechanism
Writer only (`dit_rmsnorm_fused_worker_writer.cpp`, W_GAMMA loop): before each gamma chunk's trid barrier, spin on
`noc.is_read_trid_flushed(trid)` and call `poll_stick()` inside the spin, so the stick goes out as soon as compute has
the stat even while gamma is still in flight. The blocking barrier is kept after the spin (it returns at once and does
the L1 cache invalidate). Gamma chunk pushes to compute are unchanged. No deadlock risk: compute produces the stat
before it needs any gamma page (the x*gamma pre-pass runs after PRE), and if the stat is not ready the spin just ends
when gamma lands, as before.

## Why this is not a repeat
It is the repair both r05-b03-a02 #1 and r05-b01-a02 #1 prescribed. No committed node has implemented it. The 4-wave
plumbing (sizing, start-sem chain, 8-bit wave fields, 1280 B row read, 16-partial combine) is unchanged from the parent.

## Expected effect and risk
Expected wave pushes ≈ read end + 0.6-0.9 µs (h7168 ≈ 3.0 / 4.4 / 6.1 / 7.3, h3584 ≈ 2.2 / 3.3 / 4.4 / 5.2), so the
forwarder sends and gos step up per wave. If the AG per wave stays ~3 µs and the gos spread out ~R/4, the drains
de-overlap and the tail shrinks: hope for ~1-2 µs on the wide shapes and enough on the narrow ones to beat 1.5174.
Risk: 4 waves may still lose on narrow shapes (start-sem chain cost, 4 serial go fan-outs, 16-partial combine); then
the result says 4 waves are worse than 2 even with staggered pushes. Accuracy cannot change (no data path change).
Judge with `analysis/push.py` (W_PUSH end must track R_INPUT end per wave) and `wavesN.py` (F_GO spacing).
