# r03-b04-a03: hide the gate/up activation behind two K-blocks of matmul: on r03-b02-a02's code, the last two gate/up K-blocks are walked subblock by subblock, and each subblock's PACK-thread SFPU activation is split over both blocks' matmul acquires

## Motivation
- The PACK-thread activation pipeline (r02-b01-a03, kept by r03-b02-a01/a02, best node 1.5952) runs subblock s's SFPU
  activation while MATH computes subblock s+1's last-K-block matmul. But BH's MATH commit waits for the SFPU to go idle
  (`dest_section_flip`: `STALLWAIT(STALL_CFG, MATH | SFPU1)`, tt_llk_blackhole cmath_common.h). So each activation
  chunk can hide behind exactly ONE dst acquire of matmul.
- That acquire is small next to the SFPU work. With LW = 3 (pcN 6 or 9), one acquire is gate + up over 3 tiles x
  16 K-tiles = 96 LoFi tile-mults, about 1.5k cycles. The activation of the same 3 tiles is 3 x 32 SFPU iterations of a
  25-40-instruction body, about 3-4.5k cycles. So in the last block MATH stalls at every commit, and the
  activation is still mostly serial. Evidence:
  - r03-b02-a02's RE-ACT zone is still ~10 µs per expert on the SiLU shapes (dsv3, glm, kimi-k2.7, mistral) after the
    pipeline. That is mostly MATH's multiply pass waiting for PACK's silu backlog.
  - On the binary shapes RE-ACT is ~0, but the same stall sits inside RE-GU. kimi-k3 gained only ~3 µs per expert from
    the pipeline (r03-b02-a01 vs r02-b02-a04, after removing the prologue share), against ~25 µs of SFPU per expert.
- The PACK SFPU sits idle during every earlier K-block. More FPU work per activation chunk is the only way to hide it
  further without touching the math.

## Mechanism
Compute kernel only (`kernels/compute/fused_swiglu.cpp`, `matmul_phase_fused_gu`), on top of r03-b02-a02's kernels.
This node first brings branch b04 to r03-b02-a02's op code: all four kernel files are taken from that node, and the
b04 lineage's planner trim and x-trid reader are dropped. Both were neutral.
- When num_blocks >= 3, the K-loop stops before the LAST TWO K-blocks instead of the last one. Both blocks' x / gate /
  up are waited on together. The rings are 2-3 slots deep, and the reader already pushes block N-1 once block N-3
  was popped.
- The walk visits every LW-subblock. For each one it runs the block N-2 matmul acquire (gate + up, packed L1_ACC
  onto the subblock's partials slots), then the block N-1 acquire (packed onto the same slots after a packer drain,
  the same barrier as at every K-block boundary), then hands the slots to the activation.
- The previous subblock's activation is split into one part per matmul acquire (2 parts when gate + up fuse, 4 when
  they need a dst half each). Each part runs on PACK's SFPU while MATH computes one acquire. So each subblock's
  SFPU work now overlaps two acquires of FPU work instead of one.
- Block N-1's tiles may sit at the ring start, behind the read pointer. UNPACK computes each walked block's tile
  offset from the ring's read pointer and limit. When it wraps, the offset is negative mod 2^32, which the
  unpacker's `base + index * page_size` address arithmetic resolves exactly. Pops stay one block at a time.
- num_blocks <= 2 keeps today's one-block walk (`kPipeBlocks = 1` gives today's code path).

Numerics: every output tile accumulates the same K-blocks in the same order through the same packer L1_ACC adds,
with the same formats. Only the order of subblocks inside the last two blocks changes, so the result is
bit-identical. No host change, so no build.

## Why this is not a repeat
- r02-b01-a03 / r02-b03-a04 / r03-b02-a01 built the one-block pipeline. This node deepens it to two blocks, which
  is the only way past the dest_section_flip limit of one acquire per SFPU chunk. Nobody has walked more than the
  last block.
- It is not an SFPU instruction cut (r02-b03-a03, r02-b04-a04). The bodies are untouched.
- It is not a DM change. Every DM tweak (read-ahead, rings, VCs, x owner, trids) was neutral. Profiles of
  r03-b02-a02 / r03-b04-a02 show mistral's gate/up stream already near the DRAM peak (~450 GB/s across 4 groups at
  the 10.5 µs median block period). The compute-side activation stall is a different resource.

## Expected effect and risk
- About one LW-subblock matmul (~1.5k cycles) is saved per subblock per expert wherever the activation is
  SFPU-bound, i.e. ~pcM x LN x 1.1 µs per expert. Estimates: gpt-oss -4..-6% (15 subblocks on the pcM-5 group, one
  expert), mistral / minimax-m3 -2..-3%, kimi-k3 -2..-3% (6 experts x 6 subblocks), dsv3 / glm / dsv4-flash -1..-2%,
  kimi-k2.7 / dsv4-pro ~-1%. On the DRAM-bound shapes less shows. Expect +1-2% geomean over r03-b02-a02.
- Risks:
  - kernel config buffer overflow on the h7168 shapes. A local compile shows +1.0-1.5 KB of compute code; r03-b02-a02
    left ~4 KB+ of margin. An overflow would show as runtime_error on dsv3 / kimi-k2.7 / dsv4-pro.
  - a ring-offset mistake would read the wrong x / weight block: accuracy_fail or a PCC change.
  - a mis-ordered hand-off would hang.
  The eval separates these. A bit-identical PCC / mean_abs on all shapes confirms the order is right.
