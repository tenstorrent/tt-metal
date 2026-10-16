# r03-b04-a04 result: 1.6078 (ok). Neutral by the metric: +0.4% over the parent (1.6010), inside the ±1% noise band. Mechanistically real on the many-expert shapes

## What happened vs expected
- All 9 shapes are valid.
  - PCC, max_abs and mean_abs are bit-identical to the parent and the baseline.
  - So the look-ahead slot math is right: the next ring slot, the wrap at fifo_limit, and the writer's
    `(down_blk + kb) % IN1_D_SLOTS` slot staying in step with the early down_go.
  - No hang and no JIT error on the first run.
  - The eval rebuilt the shared checkout (20 s; another node had changed host files). This node is kernel-only.
- I expected the gain on the few-expert shapes (mistral, minimax-m3, gpt-oss, dsv4-flash). It landed on the
  many-expert shapes instead.

Chip-mean µs vs the parent, and MATH zones from `/tmp/r03b04a04/z.py`. Zones are the mean over busy cores of the
per-call zone sums, mean of the 10 measured calls.

| shape | chip µs | mean core end | RE-DN (mean core) | RE-GU |
|---|---|---|---|---|
| dsv3 | 516.4 -> 509.5 (-1.3%) | 497.4 -> 483.6 | **145.6 -> 134.2 (-8%)** | 335 -> 333 |
| kimi-k2.7 | 735.8 -> 727.1 (-1.2%) | 727.9 -> 714.6 | **223.0 -> 213.4 (-4%)** | 484 -> 481 |
| kimi-k3 | 1358.7 -> 1342.1 (-1.2%) | 1272.5 -> 1254.9 | **503.9 -> 473.1 (-6%)** | 742 -> 756 |
| glm | 450.7 -> 456.0 (+1.2%) | 433.6 -> 427.0 | **127.3 -> 118.5 (-7%)** | 290 -> 292 |
| dsv4-pro | 1075.2 -> 1064.3 (-1.0%) | 1058.1 -> 1049.4 | 342.6 -> 337.4 (-2%) | 702 -> 698 |
| dsv4-flash | 315.9 -> 323.0 (+2.2%) | 295.7 -> 294.9 | 98.5 -> 96.6 (-2%) | 187 -> 188 |
| mistral | 189.8 -> 187.9 (-1.0%) | 169.2 -> 168.2 | 49.0 -> 48.3 (-1%) | 112 -> 112 |
| minimax-m3 | 372.1 -> 373.8 (+0.5%) | 352.0 -> 350.5 | 112.7 -> 110.3 (-2%) | 235 -> 236 |
| gpt-oss | 263.3 -> 258.4 (-1.9%) | 230.3 -> 232.6 | 70.1 -> 71.7 (+2%) | 155 -> 156 |

## Why (profiler evidence)
1. **Where it works: the 8-28-expert shapes, whose down phase is a chain of per-block handshakes.**
   - The down phase shrank by 9-31 µs per core: dsv3 -11, glm -9, kimi-k2.7 -10, kimi-k3 -31, dsv4-pro -5.
   - RE-GU did not move (kimi-k3's +13 is partly DN time it gives back, as in r03-b04-a03).
   - The mean core end fell 7-18 µs on all five shapes.
   - So on these shapes the DN stream was not purely aggregate-DRAM-bound. The reader-half read, issued only
     after the receivers' acks, was exposed once per down block, and ~11 blocks × experts adds up.
   - This contradicts the "mean core end = weight bytes / DRAM BW" reading (r02-b02-a03) for the down phase
     at least.
   - The chip time (max core) only partly followed the mean:
     - dsv3, kimi-k2.7, kimi-k3 and dsv4-pro moved -1.0..-1.3%;
     - glm's max core went +5 µs (the group-tail lottery) although its mean core went -7 µs.
2. **Where it does nothing: the 4-expert shapes (mistral, minimax-m3, gpt-oss) and dsv4-flash.** RE-DN moved
   by -2..+2%, which is noise. Two possible reasons:
   - The early read did not happen. The sender's own compute was still on block kb-1 when it finished
     multicasting kb (`pages_reservable_at_back(2 blocks)` false), so the read fell back to the top of kb+1, as
     before but without the ack gate.
   - Or the chain there is set by something else: the activated multicast (one row core per block sends its
     cb_activated to the row and needs all 10 row acks), or the group's DRAM share.

   I added no counter for whether the look-ahead fired, so this run can't separate the two. mistral's DN is
   ~4.4 µs per block against ~2.6 µs of compute, so the per-block gap is still there on these shapes.
3. The chip-time moves on gpt-oss (-1.9%) and dsv4-flash (+2.2%) are max-core lottery. Their mean core end
   moved +2.3 and -0.8 µs.

## Classification
Neutral (within noise) by score, +0.4%. The mechanism is a real, measured DN saving on the many-expert shapes.
- It cuts 9-31 µs of down phase per core and 7-18 µs of mean core end.
- It is bit-identical and cheap: one reader-kernel change, ring and semaphores unchanged.
- It is masked in the chip time by the max-core / group-tail spread.

Keep it. It is orthogonal to the compute-side work on this lineage, and it lowers the floor the tail sits on.

## What a child of this node should try next
1. **Find out why the 4-expert shapes didn't move. One profiler-only counter first.**
   - On the down sender (NCRISC), count per call how often the look-ahead fired versus fell back.
   - Put a TS_DATA or zone around the act sender's `cb_activated` wait + `act_ready` wait, and around the
     receivers' `act_valid` / `in1_valid` waits.
   - If the look-ahead rarely fires, the sender's compute is the slow member: start the read when the slot
     frees instead of polling once. For example, re-check `pages_reservable_at_back` while spinning on
     act_valid in step 5.
   - If the act multicast is the gate, the activated stream is the next chain to decouple. Its sender for
     block kb is row core gx=kb, which must wait for all 10 row acks for kb. Those acks come only after each
     core received kb-1. The fix is the same pattern: ack earlier, or let receivers ack kb+1 as soon as their
     in0_down_full slot is free, before they wait for kb's in1_valid.
2. **Apply the same "issue before the ack wait" idea to the row-shared x multicast in gate/up.**
   - r03-b03-a02 measured receivers waiting on x 61-202 µs per call, more than on weights.
   - The x owner's multicast for block kb needs in0_ready from all 10 row cores, sent only at their top of kb.
   - With a 2-3 slot cb_in0_x ring, receivers could ack block kb+1 as soon as that slot is free, decoupled
     from finishing kb's reads. That changes the ack protocol (one ack per slot-free, counted per block), so
     check sem ordering carefully.
3. The tail is still the biggest waste: max - mean is 25-87 µs on most shapes. This node lowered the mean on
   the many-expert shapes, so the split-a-last-wave-expert-by-N idea (r02-b02-a03 next #2) is now worth
   relatively more there.
4. Don't retry streaming x in the gate/up matmul for speed without first checking r03-b04-a01's per_core_M fit.
   It puts the marginal cost per tile-row at the LoFi ideal, so the matmul rate is not the limit.
