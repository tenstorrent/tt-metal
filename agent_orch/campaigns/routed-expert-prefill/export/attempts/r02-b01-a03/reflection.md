# r02-b01-a03 result: 1.5275 (ok). +2.6% vs parent r02-b01-a02 (1.4889); 0.3% under the best r02-b04-a01 (1.5314), which is inside noise

## What happened vs expected
All 9 shapes are valid. PCC, max_abs and mean_abs are bit-identical to the parent and the baseline on every
shape. So the following all reproduce today's numerics exactly:
- the pack-thread SFPU (SwiGLU-OAI, SiTU-GLU, clamped SiLU-GLU, SiLU)
- the per-subblock partials hand-off
- the LW-wide last-block subblocks
- the fused gate+up acquire
- the bias add moved into the pipeline (gpt-oss)

The first device run had no JIT error and no hang.

Per shape vs the parent (µs; noise ±1%, ±5% on the one-expert-per-group shapes mistral and minimax-m3):

| shape | parent | this | Δ |
|---|---|---|---|
| minimax-m3 (SwiGLU-OAI) | 392.4 | 372.0 | **-5.2%** |
| gpt-oss (SwiGLU-OAI + bias) | 341.7 | 330.8 | **-3.2%** |
| glm (SiLU) | 486.5 | 475.4 | **-2.3%** |
| kimi-k2.7 (SiLU) | 767.1 | 755.8 | -1.5% |
| kimi-k3 (SiTU-GLU) | 1366.5 | 1353.5 | -1.0% |
| dsv4-pro (clamped SiLU-GLU) | 1128.1 | 1118.4 | -0.9% |
| dsv4-flash (clamped SiLU-GLU) | 326.0 | 326.3 | 0 |
| dsv3 (SiLU) | 555.0 | 556.7 | +0.3% |
| mistral (SiLU) | 202.2 | 184.5 | -8.7% (mostly lottery: the parent's 202 was a high draw; typical 186-200) |

Expected about +2% geomean. Got +2.6%, of which roughly 0.5% is mistral's noise. The shapes that gained are
the ones not pinned to the DRAM ceiling, as predicted.

## Why (profiler evidence)
Zones on TRISC_1, mean core per call, 10 measured calls (/tmp/r02b01a02/zones.py on both nodes'
profile_log_device.csv). RE-GU now contains the pipelined activation. RE-ACT is now only the SiLU
multiply_phase plus the SFPU tail. RE-DN starts as soon as MATH has issued its last copy.

| shape | GU+ACT parent → this | DN parent → this | mean-core end | critical-core end |
|---|---|---|---|---|
| kimi-k3 | 805 → 722 (-83) | 437 → 524 (+88) | 1252 → 1257 | 1365 → 1352 |
| gpt-oss | 244 → 207 (-37) | 82 → 108 (+26) | 330 → 319 | 340 → 329 |
| minimax-m3 | 249 → 235 (-14) | 105 → 108 (+2) | 357 → 346 | 391 → 371 |
| dsv4-flash | 177 → 165 (-12) | 79 → 91 (+12) | 260 → 260 | 325 → 325 |
| dsv4-pro | 645 → 621 (-24) | 309 → 325 (+15) | 964 → 956 | 1127 → 1117 |
| glm | 260 → 251 (-10) | 110 → 115 (+6) | 375 → 371 | 486 → 474 |
| dsv3 | 299 → 297 (-2) | 126 → 127 | 430 → 429 | 554 → 555 |

- **The activation is hidden on the binary shapes.** kimi-k3 saved ~15 µs per expert of its 39.5 µs
  ACT. gpt-oss saved ~18 µs per expert, more than the 16 µs last-block FPU estimate, because the
  bias adds also run off the critical SFPU path now.
- **But on the DRAM-bound and weight-stream-bound shapes, DN grew by about what GU+ACT saved.**
  There are two reasons:
  - (a) DN now begins while PACK still runs the SFPU of the LAST subblock. That subblock's
    activation can't overlap anything, so ~3 tiles of SFPU (~4-6 µs) moved from ACT into DN's
    wait for the first activated multicast.
  - (b) More importantly, the parent's ACT phase gave the reader 20-40 µs per expert to prefetch
    the first down-weight blocks into the 2-slot down ring, for free. Now DN starts right after
    the gate/up stream ends and waits for its first down blocks to arrive.
  - kimi-k3 (~406 GB/s), dsv4-flash and dsv4-pro are bandwidth-bound, so removing compute just
    moves the wait.
  - gpt-oss and minimax-m3 are not bandwidth-bound, so they keep most of the gain (DN +13 µs per
    expert on gpt-oss is about two down blocks of delivery at ~6.5 µs each).
- **SiLU shapes gain less than binary ones.** silu is cheaper per tile than the binary ops. With LW=3
  the per-subblock switch cost (unpack/math re-init for copy↔matmul, packer format + L1_ACC flips,
  roughly 0.5 µs) eats most of what SiLU overlap can save. glm and kimi-k2.7 still gained 1.5-2.3%.
  dsv3 is DRAM-bound and flat.

## Classification
win (+2.6% over the parent, bit-identical; measured gains on the 4 shapes that are not
bandwidth-bound). The mechanism is orthogonal to the scheduling line (mixed-height layouts r02-b04-a01,
two-phase r02-b03-a02), which does not touch the compute kernel's phase structure.

## What a child of this node should try next
1. **Combine with r02-b04-a01's mixed-height layouts (best 1.5314).** That is the obvious +2-3%.
   - The two touch different code: adaptive_chunk.hpp + host topology sets vs fused_swiglu.cpp's
     GU phase.
   - gpt-oss there runs one expert per 3/2-row group at pcM 3-4, so the pipelined activation
     hides even more (more last-block subblocks, more FPU per subblock).
   - Expect ~1.55-1.56.
2. **Give DN its down-weight head start back.** The time saved now shows up as DN waiting for down
   blocks 0-1. Options:
   - Deepen the down ring (cb_in1_down + cb_in0_down_full) to 3-4 slots where L1 allows.
     r01-b04-a03 found 3-slot rings neutral, but at that time ACT was providing the prefetch
     window. Re-test on this lineage, down ring only.
   - Or let the reader issue down block 0/1 reads during the last gate/up block, after its last
     gate multicast.
   - Judge it by the RE-DN zone on kimi-k3, gpt-oss and dsv4-flash: DN should fall back toward
     the parent's 78 / 41 / 49 µs per expert.
3. **The last subblock's SFPU is still exposed** (~3 tiles). One fix is to order the last block so
   the final subblock is the narrowest: for example, a 1-tile LW for the last subblock only. Another
   is to start the down phase's first block on MATH while PACK finishes it. That second option
   needs care: the down matmul's first block needs column 0's activation, which every column
   finishes at the same time.
4. Don't revisit the SiLU switch cost unless it is measured. If you do, keep the silu chunk
   coarser (one switch per tile-row instead of per LW subblock), since silu is cheap per tile.
5. Code notes:
   - The pipeline lives at the K-loop of `matmul_phase_fused_gu`. The SFPU init is `PACK_ACT_INIT`
     at kernel start, and MATH runs no SFPU op anymore. `binary_activation_phase` is deleted.
   - The partials CBs are now pushed per LW-subblock in the last block, so `pack_tile<true>`
     indices there are relative to the pushed pointer.
   - Any change to the K-loop must keep `pushed + pad == EFF_OUT_MAX` and the cb_activated /
     cb_gate_intermed order flat (m, n).
