# r01-b01-a01 result: 1.1089 (ok)

## What happened vs expected
Valid on all 4 shapes, PCC 0.9999985 (unchanged), max_abs 0.022-0.024 (baseline ~0.022). Speedups: h3584 1.123,
h4096 1.132, h6144 1.105, h7168 1.077 (geomean 1.109, well outside the ±1% noise). Expected ~1.12-1.2; the small
shapes landed there, the wide shapes gained less.

## Why (best explanation, with profiler evidence)
Zones (`reports/r01-b01-a01`, chip-relative ns, min/max over the 20 workers), baseline -> this node:
- h3584: TRISC end 14.1-14.8 -> 11.9-12.5 us, W_DRAIN end 15.1-16.9 -> 12.7-14.4 us. x*gamma (28 tiles) fits inside
  the ~3 us AG wait as planned; post-AG is now one tile-op per column.
- h7168: TRISC end 21.5-22.4 -> 19.2-20.7 us, W_DRAIN end 23.0-26.1 -> 20.4-24.0 us. Two things eat the gain:
  1. Gamma is still late. Even as one deep batch, the 2 x 56 x 64 B face-row reads land ~3 us after the input row
     (NCRISC end 10.7-12.0 us vs PRE/W_PUSH end ~8.6-9.9 us). The x*gamma pass waits on cb_weight, so it starts ~11 us
     instead of ~9.5 us. It is issued after the input pass barrier, behind the per-block-barriered input read.
  2. The drain tail is now the critical path. Compute ends ~20 us, but W_DRAIN ends 20.4-24.0 us: the 20 cores x
     112 KB output writes contend for DRAM, so slow cores finish 1-3.5 us after compute.
- The R_INPUT read is unchanged: ~7.5 us for 56 tiles. read_input_pass barriers every block_size(=4) tiles, even
  though its comment says "deep read". It is latency bound at ~0.5 us per 8 KB block, not DRAM-bandwidth bound.

## Classification
win (+10.9% geomean; per-shape gains all > noise)

## What a child of this node should try next
- Issue the broadcast gamma reads BEFORE (or interleaved with) the first input block: it is only 7 KB per core,
  and then it is resident when PRE ends. That recovers ~1.5 us on h6144/h7168. Cheap, reader-only.
- Make the input read actually deep: keep >1 block in flight (e.g. issue block k+1 before barriering block k, or use
  BH read transaction IDs per block) so R_INPUT approaches DRAM bandwidth. PRE, the stick push and the AG all start
  earlier. This is likely the biggest remaining lever, since everything downstream is serialized on it.
- Spread the output drain: the tail after compute is DRAM write contention (W_DRAIN spread 3.5 us). Try write
  ordering or bank-interleaved starting offsets per worker, or more workers via a column split (needs on-chip
  partial-stat combine: a 1-forwarder packet holds only 34 sticks, and the forwarder kernel is outside allowed_paths).
- Bigger structural idea: the grid has ~110 cores and only 21 are used. A column split (2-3 cores per tile-row) would
  halve read, PRE, POST and drain per core. It needs the partner cores' partial sum-of-squares tiles added on chip
  before the stick push, plus a go-relay to the partners.
