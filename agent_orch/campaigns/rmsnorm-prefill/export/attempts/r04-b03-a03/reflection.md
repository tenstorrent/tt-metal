# r04-b03-a03 result: 1.3894 (ok)

## What happened vs expected
Valid on every shape. PCC 0.9999985 and max_abs 0.0204-0.0240 are the same as the parent, so the slot -> column map
is consistent across the reader, the gamma slots and the drain. I expected ~1.40-1.42. The score is 1.3894 against
the parent's 1.3906, i.e. **neutral**. Parent r04-b03-a02 -> this node (µs, chip mean):

| shape | parent | this | change |
|---|---|---|---|
| h3584 | 12.25 | 11.99 | -2.1% (cross-chip skew, see below) |
| h4096 | 13.42 | 13.31 | -0.9% (noise) |
| h6144 | 16.98 | 17.05 | +0.4% (noise) |
| h7168 | 18.15 | 18.70 | +3.0% |

Per-device means (from eval/ops.csv) show what drives the h3584 and h7168 moves:
- **h3584:** the parent spans 14.20 / 12.88 / 11.58 / 10.34 µs on dev0..3, and this node spans 13.13 / 12.20 /
  11.41 / 11.22. The gain is a smaller launch skew between chips. That is call-to-call variance, not the rotation.
- **h7168:** all four chips are +0.5 µs (18.64-18.86 vs 18.06-18.33). That is a uniform slowdown, not a straggler.

## Why (profiler evidence)
`rot.py` is in this dir: `python3 rot.py $DREAM_HOME/rmsnorm-prefill/reports/<node>`. Outputs: `rot_out.txt` for
this node and `rot_parent_out.txt` for r04-b03-a02. Values are medians over measured calls x 4 chips, in µs from
the first worker BRISC start:

| shape | R_INPUT end med (max) | drain dur med (max) | AG-wait end max | AG-wait end max -> drain end max |
|---|---|---|---|---|
| h3584 | 3.15 (3.35) -> 3.17 (3.40) | 3.31 (3.61) -> 3.35 (3.62) | 7.76 -> 7.46 | 4.14 -> 4.13 |
| h4096 | 3.64 (4.00) -> 3.63 (4.00) | 3.84 (4.52) -> 3.78 (4.30) | 8.32 -> 8.24 | 4.93 -> 4.62 |
| h6144 | 5.18 (5.83) -> 5.24 (5.75) | 5.32 (6.06) -> 5.27 (5.84) | 10.32 -> 10.30 | 6.47 -> 6.19 |
| h7168 | 6.11 (6.51) -> 6.01 (6.48) | 5.65 (6.09) -> 5.65 (6.34) | 11.01 -> 11.34 | 6.60 -> 6.81 |

- **The input read is not bank-phase bound at 20 cores.** R_INPUT end did not move on any shape (±0.1 µs). The
  parent already gave the hint: per-chip read throughput is ~360-380 GB/s on every shape, including h3584, whose
  rows already alternate two bank phases. With a 4-block (16-tile) trid lookahead, each core's outstanding reads
  already cover all 8 banks twice, so lockstep never builds a single-bank queue. That explains why r01-b03-a03's
  big h4096 read win (80 cores, per-block barriers, 80 cores per bank) does not carry over.
- **The drain is not bank/column-phase bound either.** The median per-core drain duration is unchanged on every
  shape. The worst core improved by 0.2 µs on h4096 and h6144 (tail after the AG -0.28..-0.31 µs), but got worse by
  0.25 µs on h7168. The kernel end did not follow on h4096 or h6144 (chip means flat).
- h7168 +0.5 µs: half of it is a later AG end (+0.33 µs AG-wait end max, push end max +0.11). That sits upstream of
  the drain and is on every chip, so it could be fabric/run variance. The other half is a worse worst-case drain.
  One run can't separate them. Either way there is no gain to bank.

## Classification
neutral (within noise as a geomean; flawed idea for this lineage). Per-core DRAM bank de-phasing does not help the
20-core trid-pipelined read or the posted drain. With 16 reads in flight per core the bank load is already
balanced, and the per-core drain cap is not a bank or column hot spot. Don't port the rotation further, and don't
retry "rotate the first output bank" (r04-b02-a03 #1) as a drain-staggering fix: per-core drain durations didn't
change.

## What a child of this node should try next
1. **Revert this change.** Continue from the parent r04-b03-a02 or, better, from the stacked best (r04-b02-a03,
   1.4216, or whatever "pull + 3 writer wins + HiFi2" node a sibling produces).
2. **The drain cap is still unexplained** (~95-118 ns per 2 KB tile per core, posted). Ruled out so far:
   issue/cmd-buf (r03-b02-a03), VC (r03-b03-a03), ack (r04-b04-a01), aggregate bandwidth (r03-b04-a02 waves), and now
   bank/column phase. What's left to measure:
   - Per-tile timestamps inside W_DRAIN, i.e. whether it waits on `cb_output.wait_front` (compute) or on the posted
     flush. If it waits on compute, POST (unpack-bound ~65 ns/tile, but HiFi4 fp32 x*gamma*1/rms) is the real tail.
     Then lower POST fidelity or bf16 intermediate would matter.
   - Drop the per-block posted flush. output_cb holds 2 rows, so flush/pop once per row.
3. **Fixed AG-path costs** (each ~0.1-0.3 µs on every shape): forwarder-side flush-then-inc release
   (r04-b03-a01 #2), and the ~0.45 µs PRE tail at HiFi2 (r04-b01-a02 #2).
