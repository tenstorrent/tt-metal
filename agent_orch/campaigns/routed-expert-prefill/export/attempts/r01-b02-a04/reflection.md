# r01-b02-a04 result: 1.4705 (ok), neutral (parent r01-b02-a03 1.4665, best r01-b01-a02 1.4716)

## What happened vs expected
All 9 shapes are valid and bit-identical to the parent and baseline (PCC, max_abs and mean_abs equal to
every printed digit). So the ragged gate/up grid (5 + 4), the chunked SiLU copy and multiply loops, the
row-granule ring checks and gpt-oss's ragged 5 + 4 down grid are all correct, and one matmul_block_init at
ct = 5 does serve the ct = 4 tail calls. No hang.

Speed did not move. Per shape vs the parent (noise ±1%): the four TARGET shapes are kimi-k3 1480.1 ->
1477.6 (-0.2%), dsv4-pro 1147.6 -> 1143.8 (-0.3%), minimax-m3 376.5 -> 380.0 (+0.9% slower) and gpt-oss
366.4 -> 365.2 (-0.3%). The untouched shapes are dsv3 551.8 -> 559.9, kimi-k2.7 784.2 -> 778.1, glm
492.7 -> 485.5, mistral 187.4 -> 186.9 and dsv4-flash 345.8 -> 340.7. Those untouched shapes run identical
code, so their ±1.5% is run-to-run noise. Expected +3-10% on the target shapes; got nothing.

## Why (best explanation, with profiler evidence)
Mean per-core TRISC1 (math) end, which is the work rate without the tail lottery (tail.py on
profile_log_device.csv), parent -> this node: kimi-k3 1433 -> 1434, dsv4-pro 1115 -> 1104 (-1%),
minimax-m3 351 -> 352, gpt-oss 359 -> 357. gpt-oss is the most compute-bound shape (186 GB/s, groups
balanced to <1%), and both of its matmul phases went from 1x3 to 5 + 4, yet it gained only 0.6%. So:
- **The per-matmul_block-call / per-row LLK unpack-context cost is NOT a meaningful part of the
  per-tile-mult time.** One third fewer context rounds and srcB unpacks buys < 1%. The cost per tile-mult
  (~30-40 cycles against the LoFi ideal of 16) is in the per-TILE work: the srcA (bf4 weight) unpack per
  tile-mult, which every subblock shape pays once per mult (it carries its own CFG STALLWAIT in the MOP), plus
  the math itself. Together with r01-b03-a03 / r01-b04-a03 (taller subblocks don't cut unpacks), subblock
  geometry is now exhausted as a lever.
- My premise compared kimi-k2.7 (36 cycles per tile-mult, 1x6) with dsv4-pro (41.5) and kimi-k3 (45), which
  have the same DRAM bytes per tile-mult, and blamed the 1x3 subblocks. That was wrong. The other
  difference between those shapes is the ACTIVATION: kimi-k2.7 uses SiLU (silu_tile + an FPU multiply),
  kimi-k3 SiTU-GLU (two tanh polynomials + exp + Newton recip per element), dsv4-pro clamped SiLU-GLU
  (exp + recip + clamps), gpt-oss SwiGLU-OAI. kimi-k3 also has 7 experts per group, so 7 serial activation
  phases per core. I estimate ~2k SFPU cycles per tile, which on kimi-k3 is ~25-35 us per expert per core:
  up to 10-15% of its time, fully serial between gate/up and down (down block kb needs column kb's
  activation). This is unmeasured.

## Classification
neutral (within noise): flawed idea. The per-call LLK overhead is not on the critical path. The
implementation is correct and bit-identical, and harmless to keep: it removes the "largest divisor" cliff
for any future per_core_N with no good divisor, and the SiLU path is generalized to ragged grids.

## What a child of this node should try next
1. **Measure the activation phase before changing anything else.** Put one DeviceZoneScopedN around
   binary_activation_phase / multiply_phase on TRISC1 (MATH), and one around the down phase, and read
   them from profile_log_device.csv for kimi-k3 / dsv4-pro / gpt-oss vs kimi-k2.7. If activation is
   >= 5% per expert, the lever is to take it off the serial gate/up -> activation -> down path. For example,
   start the down K-loop on blocks whose activated columns are ready (column kb finishes its activation
   first if it activates its own N-slice in kb order), or activate per N-subblock as soon as the last
   gate/up K-block has packed it, interleaved with the remaining subblocks' last-block matmuls. Do not touch
   the approximation mode (campaign rule).
2. Don't spend more attempts on matmul subblock shape (h or w) or per-call overhead: this node, r01-b03-a03
   and r01-b04-a03 cover it.
3. The other big inefficiency is still the random group tail (per call, groups doing identical work end
   up to 30% apart on mistral: 147-197 us; max/mean 1.03-1.10). An untried cheap probe is NoC
   virtual-channel separation for the weight DRAM read REQUESTS. Today every reader uses the default
   request VC, and tech_reports/Saturating_DRAM_bandwidth notes that requests within one VC are served
   FCFS, so one reader can starve another. The dram-sharded matmul assigns vc = bank_id & 3. Pass
   read_req_vc = (my_mt / group_rows) & 3 (one VC per row group) to the weight reads in weight_runs.hpp
   (noc.async_read with a custom VC). It is numerically free, so judge it by the per-group spread
   (grp.py in /tmp/r01b02a04).
