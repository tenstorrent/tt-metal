# r05-b01-a02 result: 1.4320 (ok)

## What happened vs expected
Valid on every shape on the first build and the first device run: no hang, no JIT error. PCC is 0.9999985 and max_abs
0.0204-0.0237, the parent's values. So these all work on HW:
- the S=4 decomposition (80 quarter-row workers, 7/8/12/14 tiles each);
- the 4-wave forwarder (8-bit wave fields in arrival and out_ready);
- the chained start sems (middle waves wait and signal);
- the 1280 B group read;
- the 16-partial combine through 64 B-page tile views.

But it is slower than the parent r05-b01-a01 on every shape (µs, chip mean):

| shape | parent (2 waves) | this (4 waves) | change |
|---|---|---|---|
| h3584 | 11.19 | 13.02 | +16% |
| h4096 | 12.08 | 13.28 | +10% |
| h6144 | 14.59 | 15.30 | +5% |
| h7168 | 15.84 | 17.04 | +8% |

Score 1.4320 vs 1.5696. I expected ~13.0 µs at h7168 and ~9 at h3584. The pipeline didn't form: the four waves'
all-gathers start almost together, not staggered.

## Why (profiler evidence)
Scripts are in `analysis/`. Run `python3 wavesN.py <report> S` for the per-wave timeline (wave = row-major worker
index % S; it reproduces the parent's waves.py exactly with S=2). Run `python3 percore.py <report> <shape> <dev>
<call>` for one call's per-core zones. Outputs: `wavesN_out.txt`, `wavesN_parent_out.txt`, `fwd_out.txt`,
`percore_h3584_dev0.txt` and `percore_h7168_dev0.txt`.

Per-wave medians of the max over the wave, µs, waves 0/1/2/3:

| | h3584 | h7168 |
|---|---|---|
| read end | 1.41 / 2.39 / 3.20 / 3.95 | 1.97 / 3.36 / 4.85 / 6.18 |
| push end | 4.40 / 4.36 / 4.44 / 4.81 | 3.64 / 6.68 / 6.92 / 7.21 |
| forwarder send start (`fwd.py`) | 4.65 / 4.94 / 5.17 / 5.41 | 4.51 / 6.77 / 7.10 / 7.48 |
| go | 8.49 / 9.39 / 9.97 / 10.52 | 9.21 / 10.58 / 11.16 / 11.71 |
| drain end | 10.99 / 11.85 / 12.48 / 12.92 | 12.48 / 16.57 / 16.85 / 16.87 |

1. **The reads pipelined as designed.** The last wave's read ends at 3.95 µs (h3584) and 6.18 µs (h7168), vs the
   parent's 3.43 and 6.29. So 20 cores per wave are still aggregate-bound. The 4-link start-sem chain costs ~0.5 µs
   on the narrow shapes and nothing on h7168.
2. **The stick push is gated by the gamma read, not by PRE. This is the bug that collapsed the pipeline.**
   - The writer pushes the stick from `poll_stick()` inside the streamed-gamma loop. That loop blocks in
     `async_read_barrier<TXN_ID>` for each gamma chunk, and it can't poll the stat while it is blocked.
   - The gamma face-row reads (issued at ~1.3 µs, after the ~1.2 µs scalar setup) sit in the DRAM queues behind
     the bulk input reads, so they land only when the input traffic drains. W_GAMMA ends at ~3.4-4.1 µs at h3584
     and ~6.2-6.6 µs at h7168 on every wave.
   - Per-core medians (`percore_*`), push start minus own read end:
     - h3584: 1.93 / 1.01 / 0.72 / 0.42 µs for waves 0-3. Waves 0-1 push 0.3 µs before W_GAMMA ends, i.e. at the
       barrier release.
     - h7168: 0.85 / 2.12 / 1.63 / 0.62 µs. Wave 0 pushes mid-issue at 2.65 µs and isn't gated. Waves 1 and 2 are.
   - So waves 0-2 (h3584) and 1-3 (h7168) push within ~0.5 µs of each other, at about the last wave's read end.
     Their AGs then go out back to back (F_SEND 0.25-0.3 µs apart).
3. **With the sends bunched, the forwarder serializes the releases.** Go #k is spaced ~0.55 µs (F_GO = write barrier
   + 20 serial go incs). The waves' drains then overlap and contend: h7168 waves 1-3 each take 3.3-3.6 µs per core
   vs 2.4 for the lone wave 0. The kernel ends at the bunched drains' end.
4. **Each wave's AG is still ~3-4 µs from send to go** (wave 0: 4.51 -> 8.61 µs median at h7168, partly cross-chip
   skew). The fixed chain didn't shrink, so the only way 4 waves pay is if the sends are staggered.
5. The parent has the same gate, but milder. Its wave A push ended 1.0 µs (h3584) and 0.9 µs (h7168) after A's read,
   while W_GAMMA ran to 3.6-4.0. With 28 tiles (4 chunks) its loop polls between more barriers. So the parent's
   wave A chain is probably ~0.3-0.5 µs later than it needs to be too.
6. Wave-3 (and h6144/h7168 wave-2/3) BRISC kernels start up to 1.3 µs late ("worker BRISC start max" 1.34-1.37 on
   h6144/h7168), the usual carry-over from finishing last in the previous call. Those waves are gated by the start sem
   anyway, so it is hidden.

## Classification
repairable failure (bug: the stick push is blocked behind the streamed-gamma trid barrier). Gamma lands only when the
input DRAM traffic drains, so every wave's AG start was pushed to about the end of the whole read, which removed the
stagger the waves exist for. The S=4 / 4-wave plumbing (sizing, decomposition, chained start sems, 8-bit wave
fields, 1280 B group read, 16-partial combine) is correct on HW and reusable. The idea itself isn't tested yet.

## What a child of this node should try next
1. **Un-gate the push (writer-only, ~5 lines), then re-measure the 4 waves.** In the W_GAMMA loop of
   `dit_rmsnorm_fused_worker_writer.cpp`, before the first blocking `async_read_barrier<TXN_ID>` (k == kGammaLookahead),
   push the stick if it isn't pushed yet:
   `if (!first_stick_pushed) { push_stick(0); first_stick_pushed = true; }`.
   - This blocks on compute's stat, but compute produces the stat before it needs any gamma page (the x*gamma
     pre-pass runs after the push), so it can't deadlock.
   - All S=4 rows have ≤ 2 gamma chunks (≤ 14 tiles), so every gamma read is already issued at that point.
   - Alternative: replace the blocking barrier with a spin on the trid's completion that calls `poll_stick()`.
   - Expected: push ≈ read end + 0.5-0.9 µs for every wave, sends staggered by about one wave read (~0.8-1.5 µs).
     That is when the model (h7168 ~13 µs) can be tested.
   - Judge with `wavesN.py`: push end and go must step up per wave by about the read stagger.
2. **Apply the same fix to the parent's 2-wave node r05-b01-a01** (and any S=2 lineage). Its wave A push also waits
   on the gamma barrier (A push end 4.51 vs read end 3.59 at h7168), so it is a cheap independent gain there.
3. **Forwarder release cost**: F_GO = async_write_barrier + 20 serial incs ≈ 0.55 µs per wave, ×4 waves. With
   staggered sends this overlaps the next wave's AG and matters less. If go #k still bunches, release the go incs
   flush-then-inc style (no write barrier before the incs, r04-b03-a01 #2).
4. If S=4 still loses after (1), compare with S=2 + fix (2) before abandoning more waves. The narrow shapes pay
   ~0.5 µs for the 4-link start chain (h3584 last read end 3.95 vs 3.43) and are host-launch-skew bound
   (r05-b01-a01 #3). A per-shape S choice (S=2 for ≤ 32 tile-cols, S=4 for 48/56) is a valid follow-up.
