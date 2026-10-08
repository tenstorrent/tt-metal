# r02-b02-a01 result: 1.2385 (ok)

## What happened vs expected
Valid on every shape. Accuracy is bit-identical to the root (PCC 0.9999985, max_abs 0.0217-0.0243): same bytes, same
addresses, only the NoC carrying them changed. Each shape against the round root r01-b04-a04 (1.2132):

| shape | root µs | this µs | change |
|---|---|---|---|
| h3584 | 14.17 | 13.87 | -2.1% |
| h4096 | 15.43 | 15.20 | -1.5% |
| h6144 | 19.08 | 18.73 | -1.9% |
| h7168 | 20.98 | 20.42 | -2.7% |

Geomean is +2.1% over the root. Every shape is outside the ±1% noise band. I expected ~1.25-1.30 and got the low end:
the drain tail shrank by about half of what the link model promised.

## Why (profiler evidence)
I ran `analysis/tails.py` on `reports/r02-b02-a01`. Values are medians of measured calls, devices 0-1, in µs relative
to the last W_AGWAIT end, root -> this node:

| shape | last drain end - AG | median drain end - AG | POST (AG -> TRISC end) |
|---|---|---|---|
| h3584 | 5.75-5.95 -> 5.36-5.40 | 5.10 -> 4.95 | 3.98 -> 4.09 |
| h4096 | 6.44-6.49 -> 6.01-6.12 | 5.50 -> 5.40 | 4.23 -> 4.31 |
| h6144 | 8.47-8.49 -> 7.79-7.81 | 7.25 -> 7.10 | 5.30 -> 5.38 |
| h7168 | 9.36-9.47 -> 7.92 (dev1) | 8.1 -> 7.5 | 5.81 -> 5.91 |

- **The drain tail past compute is roughly halved at h7168:** 3.7 -> 2.0 µs on dev 1.
- **The worst cores improved most.** The median drain end moved less than the max, which fits unloading the shared
  hot links.
  - h7168 call 58, per-core drain end (`analysis/startspread.py`): the spread went from 18.4-21.1 µs (2.7 µs,
    x=5..11 late) to 19.7-20.6 µs (0.9 µs).
  - The x-gradient is gone. A small row effect remains: y=3 is ~0.3 µs later than y=2.
- **The next call's start skew shrank too.** At h7168 the worker kernel starts now spread 0.87-1.50 µs (0.63 µs), down
  from 0.01-1.41 µs (1.4 µs). `analysis/gap.py` shows why: all workers now end within ~0.65 µs of each other, so they
  all restart together. This confirms the cross-call coupling in the proposal: late-draining cores cost twice.
- **Dynamic-NoC mode is close to free.** R_INPUT max at h7168 is 6.03-6.24 vs 5.94-6.06 µs (+0.1-0.2 µs, near noise).
  POST is up ~0.1 µs on 3 of 4 shapes. The trid-pipelined reader works unchanged in DM_DYNAMIC_NOC.
- **What remains at h7168:** after the AG, POST takes ~5.9 µs (~105 ns/tile, HiFi4 fp32 x bcast), and the drain still
  ends ~2 µs after compute.
  - The model predicted the hottest write link would drop from 4.25 B to 3.0 B (-29%). The measured drain moved about
    as far as a 15-20% cut would explain.
  - So either part of the limit is not those links (DRAM-side, or parking-lot arbitration on the remaining NoC1
    rows), or the 50%-of-short-destinations share is not optimal.
- **NoC labels for future readers:** on BH the reader (NCRISC) is on NoC0 and the writer (BRISC) on NoC1
  (kernel_types.hpp preferred_noc_for_dram_read/write). The r01 reflections had them swapped.
- `analysis/nocmodel.py` models physical coords from the profiler and the soc-descriptor DRAM endpoints. It predicts the
  read rate (~90 GB/s per hot link) and r01-b02-a04's NoC0 disaster on the left cores, which are the long eastward
  wraps.

## Classification
win (+2.1% geomean over the round root, every shape outside noise; best committed node when written). It is a repair of the dual-NoC
drain idea (r01-b02-a04, r01-b03-a04). The bug in those attempts was a position-blind split: half of all tiles went on
NoC0, including the long eastward wraps. Choosing the NoC per destination fixes that, and issuing both NoCs from BRISC
in DM_DYNAMIC_NOC removes the cross-RISC handshake.

## What a child of this node should try next
1. **Tune the NoC0 share.** This node sends 50% of the short-eastward destination visits on NoC0 (`(page/8)&1==0`).
   - `analysis/opt.py` finds per-(core,bank) assignments with a max link load of 147/238 (-38%) vs the rule's -29%.
   - It also says row-2 and row-3 cores want different shares. The rule's shares are row-blind, and y=3 now finishes
     ~0.3 µs later.
   - Cheap test: share 2/3 or 3/4 for y=3 cores only, or a per-core 8-bit mask passed as an RT arg (host computes
     from the model). Watch W_DRAIN per core with `startspread.py`: the target is a flat table.
2. **Stack onto the column split** (r01-b03-a02/a03, k=4, 80 workers). The drain is the column split's critical path
   (~200 GB/s), and its cores span rows 2-9. Re-run `nocmodel.py` with that worker set first: the best per-core rule
   will differ, and the short-eastward rule generalizes.
3. **POST is now ~70% of the post-AG time at h7168** (5.9 of 7.9 µs). Options:
   - Lower fidelity for the x*gamma*(1/rms) multiply.
   - A bf16 (not fp32) x*gamma intermediate, which halves unpack bytes. Check max_abs against the 0.05 gate.
4. Don't split the input read the same way. The reads already run at ~420 GB/s (near the DRAM peak), and the model
   only gives -18% on the hot link.
