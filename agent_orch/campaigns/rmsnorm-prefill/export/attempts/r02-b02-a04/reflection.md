# r02-b02-a04 result: 1.2346 (ok)

## What happened vs expected
The run is valid on all shapes: PCC 0.9999985, max_abs 0.0204-0.0240, the same as the parent. Per shape, this node vs
the parent r02-b02-a03 (µs, chip mean): h3584 14.01 / 13.95, h4096 15.12 / 15.12, h6144 18.87 / 18.67,
h7168 20.41 / 20.38. Geomean is 1.2346 vs 1.2396 (-0.4%), inside the ±1% noise band. I expected +1-2%. The
multicast release did engage (no hang, correct data), but it didn't make the go arrive earlier.

## Why (profiler evidence)
`go.py` (in this dir) gives per-core medians over the measured calls on all 4 chips, in µs after the forwarder's
F_FABRIC end. `comb.py` gives the new compute zones. Run both as `python3 X.py $DREAM_HOME/rmsnorm-prefill/reports/<node>`.

| go arrival after F_FABRIC end | first worker | last worker | spread |
|---|---|---|---|
| parent (20 serial unicast incs) | 0.17 | 0.52 | 0.35 |
| this node (2 set_multicast) | 0.28 | 0.49 | 0.21 |

1. **The multicast go is slower to arrive, not faster.** The earliest worker now gets go 0.11 µs *later*, and the
   last one only 0.03 µs earlier.
   - Inside each row, arrival falls with x (x=1 0.36 -> x=13 0.28). The NoC1 multicast starts at the east end.
   - The second row-segment (y=3) lands ~0.1 µs after the first, which is the cost of the second multicast plus
     multicast path reservation.
   - So a NoC1 set_multicast to 11 cores costs ~0.1 µs more than a unicast atomic, and two of them in sequence give
     back most of the serial-inc saving. The median go time is unchanged. y=3 still drains last (~0.25 µs after y=2).
2. **Pre-staging the stick reads saved nothing, and the slowest core got worse.** W_DRAIN start - go is 0.67 µs at
   min (same as before) and 0.80 µs at max (was 0.71). The address math was not the cost. The 8 x 64 B DRAM read round
   trip is. Releasing all 20 workers at nearly the same time also puts their 160 reads on the 4 scratch pages at once,
   which plausibly explains the worse max.
3. **New, and the most useful finding: the post-AG fixed cost is the compute stat combine.** These are TRISC_0
   (unpack) zones, the only thread whose zone start waits on the CB:
   - C_COMB starts 0.02 µs after the sticks are pushed. Its unpack part takes only 0.13 µs.
   - Then the unpacker idles **1.27-1.28 µs** before C_POST starts, on every shape. That is the math + pack half of
     the combine: 3 ELWADDs, transpose_dest<fp32>, mul_unary, add_unary, rsqrt_tile on a full 32x32 fp32 tile, pack
     to reduce_result_cb, and the CB handoff back to unpack.
   - It is a fixed ~1.3 µs on the critical path between "all stats in L1" and the first POST tile. That is ~6-9% of
     the kernel.
   - POST itself runs at 63-64 ns/tile on unpack (h3584 1.77 µs / 28 tiles, h7168 3.58 µs / 56 tiles). The pack
     thread's POST lasts 4.8-6.3 µs because output_cb backs up behind the drain. The drain still ends 1.0-1.9 µs
     after pack's POST end.
   - The math/pack C_COMB zones start early (those threads don't wait on the CB), so their durations include idle
     time. Ignore them.

## Classification
neutral (within noise). A go multicast on NoC1 has higher latency than a unicast inc, so it can't beat a 20-way
serial fan-out by much. Address pre-staging targets a non-cost. This is a flawed idea at this group size, not a bug.
The forwarder fork (`dit_rmsnorm_forwarder.cpp`, inside allowed_paths) is reusable plumbing: future nodes can now
change the AG protocol.

## What a child of this node should try next
1. **Cut the ~1.28 µs combine chain in compute (largest fixed post-AG cost).** Options, cheapest first:
   - Run mul/add/rsqrt only on the faces that hold the stat. After transpose_dest the stat is in col 0 (faces 0 and 2),
     so use VectorMode::C via the LLK `_llk_math_eltwise_unary_sfpu_params_` path or `*_tile(idst, VectorMode::C)`
     where available. Better, fold `*1/H` into the gathered sum: fold it into PRE's ones-matrix / scalar so PRE emits
     s/H. Then the chain is add eps + rsqrt only.
   - Drop transpose_dest: let the writer drop the gathered sticks into COLUMN 0 instead of row 0. The 32 floats per
     device go to strided words, too slow on BRISC; see r02-b02-a02. Or produce col 0 by accumulating 4 matmuls
     I * G_d^T into DST (one op type, no transpose_dest). Measure with C_COMB/C_POST on TRISC_0 (keep these zones).
   - Overlap the rsqrt with POST's unpack setup: issue mul_bcast_cols_init / reconfig on unpack before waiting on
     reduce_result.
2. Revert the multicast go to the unicast inc loop (or keep it only with NoC0 / a single rectangle). The unicast order
   at least gets slot 0 out 0.11 µs sooner. Don't spend more attempts on the release fan-out: its whole spread is
   ~0.35 µs and the median doesn't move.
3. The stick read after go (0.67 µs DRAM round trip) only goes away if the gathered data lands in L1. That needs a
   persistent, mesh-coherent L1 buffer, not a CB, because the op deliberately uses caller-owned persistent scratch +
   ping-ponged semaphores for cross-chip safety. It is possible from the factory (a MeshBuffer held in shared
   variables), but it is a larger change.
4. Don't try spreading the drain over a DRAM bank's other NoC ports. On BH the non-preferred DRAM NIUs are in stream
   mode, so writes there land in DRISC L1, not GDDR (tt_metal/hw/inc/experimental/drisc_mode.h).
