# r01-b03-a03 result: 1.4574 (ok)

## What happened vs expected
All 9 shapes valid and bit-identical to baseline (PCC / max_abs / mean_abs equal to every printed
digit), no hang. So the cross-unit block-0 feed is correct, and the deadlock argument held under the
real 2-row and 4-row group schedules. The score is 1.4574 against r01-b01-a02's 1.4716, whose code this node
carries over verbatim apart from the writer change. That is -1%, at the edge of the ±1% noise band. Per shape vs
b01-a02: dsv3 563.1 -> 559.9 (+0.6%), kimi-k2.7 771.5 -> 767.5 (+0.5%), kimi-k3 1478.8 -> 1489.1
(-0.7%), glm 486.9 -> 488.9 (-0.4%), mistral 185.9 -> 192.4 (-3.5%), dsv4-flash 340.7 -> 343.1 (-0.7%),
dsv4-pro 1148.9 -> 1151.5 (-0.2%), minimax 378.2 -> 384.2 (-1.6%), gpt-oss 364.5 -> 374.7 (-2.8%).
Against the parent r01-b03-a02 (0.9626) this is +51%, all of it from adopting b01-a02's row-group code.

Expected: +1.5-2.5% on kimi-k3 and ~1% on the 2-3-expert-per-group shapes. Got nothing measurable.

## Why (best explanation, with profiler evidence)
Per-group math-end times (TRISC_1 kernel end per 2-row group, averaged over the 10 measured calls,
profile_log_device.csv of this node vs reports/r01-b01-a02):
- Mean group finish (the work actually done, without the tail lottery): kimi-k3 1433 -> 1442,
  dsv3 529 -> 531, kimi-k2.7 740 -> 738, dsv4-pro 1120 -> 1109, glm 460 -> 457. On the shapes where
  the early feed triggers (several units per group), the mean group time didn't move. So the expert
  boundary bubble I targeted (one up read plus one gate/up multicast per unit) is not a
  measurable cost. The likely reason: at the boundary, the compute's own work is the down tail,
  the output copy pass and the x(0) tilize+multicast, and the weight fill already overlaps those. My
  estimate of ~16 us/expert of fixed cost on kimi-k3 is real but isn't this.
- The biggest regressions are on shapes where the new code path does nothing. mistral and
  minimax run one expert per group, so there is never a next unit. gpt-oss's reader blocks on the
  single-buffered bias CB, so up_go is never granted during the drain. For senders with
  no next unit the drain loop is the same wait as before. These shapes' run-to-run max/mean
  group-end ratio is 1.06-1.09 (b01-a02 analysis in context.md), and their per-call max ranged
  179-222 us here (177-191 before). I read the -2.8/-3.5% as tail-lottery noise on those shapes, not
  as an effect of the change. Treat the whole node as neutral.
- Side finding, useful for the next node: with weight bytes plus x reads plus output writes, the
  large shapes already move 410-434 GB/s (dsv3 234 MB, kimi-k2.7 335 MB, dsv4-pro 474 MB). A p150b
  in this repo measured a 367 GB/s matmul DRAM ceiling (models/experimental/voxtral_tts). So dsv3,
  kimi-k2.7, glm and dsv4-pro are within ~5-10% of the DRAM floor, and DM-ordering tweaks can't buy much there.
  The remaining non-DRAM-bound shape is gpt-oss: 68 MB in ~365 us, groups balanced to <1%, about
  32-39 cycles per tile-mult with 1x3 subblocks, bias and SwiGLU-OAI. That makes it compute- or
  latency-bound.

## Classification
neutral (within noise): flawed idea, because the expert-boundary weight bubble is not on the
critical path. The implementation was correct (bit-identical, no hang).

## What a child of this node should try next
- Don't spend more attempts on unit-boundary or in-loop weight-read timing. With this node there
  have been four (b03-a01, b04-a01, b04-a02, this one), all neutral or negative.
- The largest measurable inefficiency left is the random group tail. Groups doing equal work finish
  3-9% apart (max/mean: mistral 1.09, dsv4-flash 1.064, dsv3 1.061, mm3 1.058, glm 1.056). The slow
  group changes from run to run, which looks like DRAM/NoC share unfairness, not schedule imbalance.
  Fixing it needs sub-expert work sharing or fairer DRAM issue (e.g. per-row NoC/VC choice for the
  weight reads). Expert-granular work stealing won't help, because the spread is smaller than one expert's duration.
- gpt-oss (compute-bound) is the one shape far from its DRAM floor. Levers: ragged 1x5+1x4 gate/up and
  down subblocks instead of 1x3 (fewer matmul_block calls and dst rounds; the down path already
  supports a ragged tail). Also skip the fully phantom 11th down K-block (K 90-98 on gpt-oss), which
  costs a full handshake round per expert, and double-buffer the bias CBs so expert boundaries don't serialise on
  them.
- Do not try taller subblocks (out_subblock_h > 1) to cut unpacks. The BH matmul_block LLK
  unpacks the streamed operand once per tile-mult for every (rt, ct) shape (unpacks = rt*ct + min(rt,ct)
  per call), so taller subblocks never reduce unpack work below what 1 x ct already does.
- The writer refactor here (feed_up_block lambda, next-unit lookahead via a GroupScheduler copy) is
  harmless and can stay, or be reverted with no effect.
