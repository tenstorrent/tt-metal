# -mk-tail results (2026-11-10, d27u08 dev0, MK1 skeleton: 4 GDN-type layers, stalls 67 + 18 us, MIXN 1)

All arms from `run_si_mktail.sh <tag> 4 67 18 1 "<env>"` with the device profiler on. CSVs are kept on the laptop
(.claude/tmp/r16/mktail/csv). Tail = C-DN end (TRISC_0 median) minus C-GU end (TRISC_0 max), i.e. last gate|up core → down
done everywhere.

| arm | defines / env | kernel us (runs) | tail L1 / L2 (us) | ROOT-MC med (us) | last-layer PCC |
|---|---|---|---|---|---|
| inst (baseline + zones) | — | 581.2, 585.3, 575.8 | 6.9-7.3 / 6.9-7.2 | 4.5-5.7 | 0.99998 |
| pipe | MK_DNPIPE=1 | 578.2 | 6.6 / 6.8 | 5.9 | 0.99998 |
| flush | MK_FLAG_FLUSH=1 | 587.3 | 6.8 / 6.8 | 4.3 | 0.99998 |
| both | MK_DNPIPE + MK_FLAG_FLUSH | 576.2, 572.1 | 6.3 / 6.3-6.7 | 4.6-4.7 | 0.99998 |
| **bothh8** | both + MK_H_BFP8=1 | **569.6, 568.6** | **5.3-5.5 / 5.3-5.7** | **2.8-2.9** | 0.99993 |
| cold | bothh8 + MK_COLD_MIXFIRST=1 | 572.8, 575.9 | 6.6-10.2 / 4.4-11.6 | 3.3-4.2 | 0.99993 |

- **Down tail:** about 7 → 5.3-5.7 us per layer (the sim said ~4). Kernel time drops from a mean of 580.8 to 569.1 us
  over 4 layers, about −2.9 us per layer, i.e. about −0.19 ms per 64-layer pass for the MLP root path. Root
  multicast is the largest piece: bf16 h 34 KB per root takes 4.5-5.7 us while the weight stream is active on the same
  NoCs. BFP8 h halves it to 2.9 us. DNPIPE (down accumulates root chunk by root chunk) and FLAG_FLUSH (flag after
  writes are flushed, not acked) are numerically identical (same PCC and max err). BFP8 h changes numerics slightly
  (PCC 0.99998 → 0.99993).
- **Cold start (layer 0):** the mixer slice is gated on the whole layer's 19 MB plus layer 1's issue (median 69 / max
  103 us vs ~22 us steady). MK_COLD_MIXFIRST (separate mixer transaction ids + the reader held until its mixer
  landed) cuts the median to 23-25 us, but the max stays at 75-78 us: cores whose mixer landed early release their
  19 MB streams while other cores' mixer reads are still queued. With a per-core GO, layer 0's chain is set by the
  max, so the kernel time is not better (572.8 / 575.9). It is a once-per-pass cost (~50 us of ~12 ms). Not worth a
  global cold-start barrier now. The option uses program semaphore 12 (reserved for barriers in DESIGN §6), so leave
  it off or re-assign the id.
- **Remaining straggler, not a tail-path issue:** in layers 1-2, W-MIXSTALL ends up to ~20 us late on a few cores
  (the same cores each run), which delays their GO → gate|up → h for everyone. This is the spin_cycles stand-in
  (BRISC busy-wait on the wall clock while the NoCs stream into that core). The real GO (all-reduce receive) replaces
  it, so check it again at MK4 rather than tuning the stand-in.
