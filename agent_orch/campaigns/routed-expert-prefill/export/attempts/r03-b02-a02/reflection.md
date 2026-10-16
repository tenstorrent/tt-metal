# r03-b02-a02 result: 1.5952 (ok). Neutral: +0.4% over the parent r03-b02-a01 (1.5887), inside the ±1% noise band

## What happened vs expected
**First run: runtime_error on the three h7168 shapes. Repaired before any timing.**
- dsv3, kimi-k2.7 and dsv4-pro failed with `Program size (71056 / 72608 / 71328) too large for kernel config buffer (70656)`, the same failure as r03-b01-a01. The parent sat just under the limit.
- Measured with a local LTO compile of the parent vs this node (/tmp/r03b02a02/sz, the JIT's generated chlkc files plus representative CT args):
  - The owner test, inlined at two sites of the gate/up phase, cost ~+0.8-0.9 KB of compute code.
  - The reader's growth was not measured.
- Repair, inert for numerics and the plan:
  - Ownership is now computed once per expert (owner-slot count + own slot). The gate/up phase tests it with a wrapping slot counter, and the reader calls an out-of-line `x_owner_col`.
  - On TRISC builds, the launch planner region of adaptive_chunk.hpp is compiled at -O2 (`#pragma GCC optimize`), not -O3. TRISC0 shrinks by 4.4-4.9 KB, so compute is 3.8-5.2 KB under the parent overall.
- The second run is the only measurement of the mechanism.

**Second run:**
- All 9 shapes are valid. PCC, max_abs and mean_abs are bit-identical to the parent and the baseline, so the new owner assignment is consistent between reader and compute, and there was no hang.
- Score 1.5952 vs 1.5887. Per shape vs the parent:

| shape | parent µs | this µs | Δ |
|---|---|---|---|
| dsv3 | 517.3 | 519.7 | +0.5% |
| kimi-k2.7 | 735.9 | 733.0 | -0.4% |
| kimi-k3 | 1374.7 | 1371.5 | -0.2% |
| glm | 462.3 | 457.5 | -1.0% |
| mistral | 192.4 | 191.5 | -0.5% |
| dsv4-flash | 322.4 | 318.5 | -1.2% |
| dsv4-pro | 1075.9 | 1069.2 | -0.6% |
| minimax-m3 | 372.6 | 366.1 | -1.7% (noisy shape, ±5%) |
| gpt-oss | 262.0 | 266.1 | +1.6% (noisy shape) |

I expected -3..-10% on the few-expert shapes. Nothing moved outside noise.

## Why (profiler evidence)
MATH zones, mean core per call (/tmp/r02b01a02/zones.py, parent -> this):

| shape | GU | DN | mean-core end | first-GU-start |
|---|---|---|---|---|
| mistral | 109.1 -> 106.0 | 49.5 -> 51.2 | 168 -> 167 | 3.4 -> 3.7 |
| minimax-m3 | 228.3 -> 227.5 | 111.8 -> 111.9 | 344 -> 344 | 3.5 -> 3.6 |
| gpt-oss | 152.9 -> 157.7 | 76.9 -> 70.7 | 234 -> 233 | 3.9 -> 4.3 |
| dsv4-flash | 184.4 -> 181.1 | 105.0 -> 105.2 | 298 -> 296 | 7.4 -> 8.7 |
| dsv3 | 329.4 -> 325.5 | 144.5 -> 147.1 | 492 -> 492 | 5.6 -> 6.4 |
| kimi-k2.7 | 486.6 -> 477.5 | 219.7 -> 225.3 | 730 -> 727 | 6.7 -> 7.8 |
| kimi-k3 | 730.5 -> 730.2 | 519.8 -> 519.8 | 1271 -> 1276 | 17.2 -> 21.9 |

- **The coupling I targeted is not on the critical path.** Before this node, in every row group on every gate/up K-block, one row's x owner was that column's weight sender. Its NCRISC's gate read barrier also covered ~96 x-stick reads, and its next gate read waited behind the x multicast. Removing all of that changed GU per core by -3..+3% and the mean-core end by ≤ 1%.
  - The 2-slot x / weight rings absorb the extra latency on that one column per block.
  - It also confirms r01-b02-a03's empirical reading ("the collision does not show up as a loss"), even though that node underestimated how often the collision happens.
- So the ~2x-ideal GU time per K-block on mistral / mm3 / gpt-oss (13.7 µs per block vs ~6.8 µs of LoFi matmul on mistral; gpt-oss ~26 vs ~10-13) is **not** sender-side x work. Ruled out so far: x/weight sender coupling (this node), read-ahead / ring depth / VCs / down look-ahead (earlier nodes), and per-block tilize + reconfig (r01-b03-a02).
- **Side effect: the TRISC planner at -O2 is slightly slower.** First-GU-start went up ~0.3-1.3 µs, and 4.7 µs on kimi-k3. So UNPACK's own planning is partly on the compute critical path after all, in parallel with the reader's planning, not hidden behind it. kimi-k3's mean-core end +5 µs is consistent with that. It is within noise everywhere.

## Classification
neutral (within noise). Flawed premise: the x-owner / weight-sender collision is real and happens on every block, but the ring slack hides it. The implementation is correct and bit-identical, and harmless to keep.

The size repair is worth keeping on its own: TRISC0 gains ~4.5 KB of kernel-config-buffer headroom (h7168 programs were within ~0-2 KB of the 70656 B limit). The cost is a few µs of first-GU-start on kimi-k3.

## What a child of this node should try next
1. **Use the headroom (this lineage was at the size limit):** port r02-b04-a04's SiLU body (`re_act::silu_act`) to PACK for dsv3 / glm / kimi-k2.7 / mistral. RE-ACT is still ~5 µs per expert there (the FPU multiply pass plus the last subblock's SiLU).
   - If size is tight again, r03-b01-a02's plan-once-per-core (reader plans, L1 CB shares the plan) frees ~12 KB more and removes UNPACK planning.
   - With plan-once, the -O2 pragma here becomes moot and its kimi-k3 first-GU cost disappears.
2. **Find what really costs ~2x per gate/up K-block on the few-expert shapes before changing more DM.** Put DeviceZoneScopedN zones on NCRISC (recorded reliably; TRISC_0 zones are not) on one shape (mistral):
   - the sender's gate read+barrier,
   - the sender's wait for mcast_done / receivers' in1_ready,
   - the receiver's in1_valid / in0_valid waits.

   Compare per-block wait time against 13.7 µs/block. Candidates left: the writer's serial up-read -> multicast chain per block (UP_SPLIT: the up read of kb+1 starts only after the multicast of kb), and the unpack rate of bf4 weights at 1xN subblocks.
3. If you keep this node's owner rotation, every row still pays exactly one owner per K-block. The non-sender owners now do ~2x the tilize work of before (1 in 5-6 blocks on 2-row groups). That showed no cost either.
4. Don't retry: x-owner placement (this node), request VCs, down look-ahead / deeper rings, read-ahead, writer-owned down stream.
