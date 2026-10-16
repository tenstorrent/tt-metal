# r04-b03-a01 result: 1.5945 (ok). Measurement node; -1.0% vs root (1.6103), at the edge of the ±1% noise band, part of it the probe's start-up cost. Outputs bit-identical. It answers the campaign's open question: the ~31 cycles per tile-mult is NOT the matmul; the compute is waiting for the ROW streams (x in gate/up, activated in down)

## What happened vs expected
- All 9 shapes valid; PCC / max_abs / mean_abs equal root and baseline on every digit. No hang, no JIT error, no
  kernel-config overflow (+~1.9 KB of TRISC code in profiler builds; non-profiler binaries are unchanged in size).
- Every RE-HW probe record (indices 16-19) and every RE-UW record arrived, all 117 launches.
- Chip µs vs root: dsv3 514.0 -> 519.7, k2.7 729.5 -> 732.6, k3 1365.8 -> 1359.7, glm 451.2 -> 455.6, mistral 185.9 -> 194.9,
  dsv4-flash 316.9 -> 316.9, dsv4-pro 1070.9 -> 1069.7, mm3 365.5 -> 370.8, gpt-oss 261.8 -> 263.9.
  - The probe is not fully hidden: first RE-GU starts 1-2 µs later (`/tmp/r04b03a01/ends.py`: dsv3 6.7 -> 8.4, glm
    6.7 -> 8.5, mistral 3.9 -> 5.9, dsv4-flash 9.1 -> 10.9). That is ~0.3-1% on the short shapes. mistral's +9 µs is
    mostly its usual lottery (mean core end 165.2 -> 167.7, +2.5 µs).

## The measurement (`/tmp/r04b03a01/probe.py` on reports/r04-b03-a01; means over the 10 measured calls)
1. In-situ probe, cycles per tile-mult of the real gate/up matmul with NO data wait possible (same init, 1 x W
   matmul_block over in0_block_w K-steps, gate -> up CB switch, one dst acquire each), vs the RE-GU window:

| shape | probe cyc/mult | probe FPU / UNP0 per mult | RE-GU window cyc/mult | RE-DN window cyc/mult |
|---|---|---|---|---|
| dsv3, glm, k2.7, mistral, dsv4-flash (W = 6) | **21.0** | 16.0 / 17.2 | 32.9 / 32.9 / 45.5 / 31.2 / 39.5 | 30.5 / 31.2 / 44.5 / 29.0 / 47.1 |
| k3, dsv4-pro, mm3, gpt-oss (N = 9 shapes) | **25.1-25.4** | 16.0 / 17.5 | 43.3 / 47.6 / 30.5 / 41.1 | 61.0 / 48.5 / 29.2 / 29.3 |

   - The matmul pipeline itself runs a tile-mult in 21 cycles (1x6 subblocks) and ~25 (the N = 9 shapes' narrower
     subblocks). Unpack and FPU DO overlap: 16 + 17.2 would be 33. r04-b01-a01's "SrcA behaves single-buffered" is
     refuted. The gap to the LLK harness's 17.2 (1x8x32) is the per-subblock overhead of 16-K-step / 6-wide blocks
     (dst hand-off, gate/up CB switch, per-call unpack protocol).
2. UNPACK wall-clock waits, % of the UNPACK-side window (x / gate / up are checked in that order each block, so a
   later wait only counts what is left after the earlier one returned):

| shape | GU: x | gate | up | owner tilize | DN: activated | down weights | per-core GU wait min/med/max |
|---|---|---|---|---|---|---|---|
| dsv3 | **30.3%** | 9.8 | 0.1 | 2.1 | **31.9%** | 6.3 | 20/40/53% |
| kimi-k2.7 | **41.9** | 14.7 | 0.1 | 1.6 | **46.8** | 11.3 | 38/58/65 |
| kimi-k3 | **28.2** | 8.8 | 0.1 | 1.5 | **57.5** | 9.3 | 24/36/59 |
| glm | **29.1** | 9.9 | 0.1 | 2.3 | **33.3** | 6.4 | 17/39/53 |
| mistral | **25.9** | 5.6 | 0.1 | 3.1 | **34.3** | 3.6 | 3/30/57 |
| dsv4-flash | **33.0** | 11.6 | 0.1 | 2.4 | **49.8** | 9.0 | 29/45/53 |
| dsv4-pro | **41.5** | 15.3 | 0.1 | 1.1 | **49.7** | 11.3 | 45/56/72 |
| minimax-m3 | **24.8** | 6.0 | 0.1 | 2.1 | **30.4** | 4.6 | 8/30/60 |
| gpt-oss | **39.6** | 2.8 | 0.1 | 3.4 | **39.0** | 2.9 | 17/42/99 |

   - Consistency check: window cyc/mult x (1 - wait share) ≈ probe (dsv3 GU 32.9 x 0.60 = 19.7 vs 21.0; mistral DN
     29.0 x 0.62 = 18 vs 21). The windows are the 21-25-cycle matmul plus waiting.
   - The compute waits mostly on the ROW stream: x in gate/up (25-42% of the window), activated in down (30-58%).
     The column weight stream adds only 3-15% after x has landed. The owner's tilize is 1-3%.
   - Caveat: x is waited first, so a block whose weights arrive even later than x shows up partly as x. But weights
     arriving later than x would show as gate wait, and that is small; and on the receivers the next weight ack is
     itself gated by x's arrival (lock-step reader loop), so x is upstream of the weights either way.

## Why
- The pcM-scaling of cycles/mult (31 at pcM 3, 45 at pcM 2) is a per-block delivery chain rate, not a compute or a
  pure DRAM rate. Per K-block every row core must receive x(kb) from the row's rotating owner, whose multicast waits for
  ALL 11 row cores' acks, and each core acks kb+1 only after it received kb (x and weights) and its compute freed kb-1.
  In down, activated(kb) comes from row core gx = kb, which needs its own compute's cb_activated plus all 10 row acks.
  The row coupling (11 cores, every block, rotating sender) is what the compute waits on.
- kimi-k3 / dsv4-flash / dsv4-pro DN (activated 50-58%) are the worst: many small down blocks, each a full row
  barrier.

## Classification
neutral (measurement node; -1.0% is mostly noise plus ~1-2 µs of probe start-up). It settles (A) vs (B): data
starvation by the row streams, with a 21-25 cycle matmul underneath.

## What a child of this node should try next
1. **Decouple the row streams; this is where the time is.** Expected value: up to the wait shares above (25-58% of
   the windows) on every shape, i.e. far more than any LLK work.
   - Down / activated (biggest: 30-58%): make act_valid a sequence number and let receivers ack activated kb+1 as soon
     as their cb_in0_down_full slot is free, not after receiving kb (r04-b02-a01 next #2 proposes exactly this; with 3
     ring slots the act sender for kb+1 could then multicast while kb is being consumed). Also start the act multicast
     as soon as cb_activated holds the block instead of after the in1_down section.
   - Gate/up / x: r04-b04-a01 (in flight) decouples the x and weight acks; the same sequence-number valid lets receivers
     ack x(kb+1) when the x slot frees. An x owner further ahead (tilize 2 blocks ahead when the x ring has 3 slots)
     shortens the chain too.
   - Judge any of these with this node's RE-UW tallies: the x / activated shares must fall.
2. **Don't spend attempts on the LLK matmul.** The pipeline already overlaps (21 cycles vs 16 FPU / 17.2 unpack). Even
   reaching the harness's 17.2 is at most -18% of the non-waiting part, and only where the windows are not waiting.
   (The N = 9 shapes' 25 cycles is subblock width: 1x3 / ragged calls; a 1x9 or 1x5+1x4 split was r01-b02-a04, small.)
3. Weights are not the limiter in-window (gate 3-15%), so read-ahead / VC / ring-depth work on the weight stream is done.
4. Tooling: keep `re_uw` (RE-UW, TRISC0) and `/tmp/r04b03a01/probe.py`; they cost nothing outside the profiler.
   Drop or gate the probe (`re_probe_gu` call in kernel_main) in score-bearing children: it delays the first GU by
   1-2 µs. `/tmp/r04b03a01/ends.py` gives first-GU start, mean / max core end per shape. `/tmp/r04b03a01/sz/kc.sh` does
   a local 3-TRISC compile + size check against root.
