# r02-b01-a03: pipeline the gate/up activation into the last gate/up K-block, with the SFPU on the PACK thread

## Motivation
r02-b01-a02 measured the per-expert MATH phases for the first time (RE-GU / RE-ACT / RE-DN zones).
The binary activations are pure serial SFPU work after the gate/up K-loop:
- kimi-k3 39.5 µs per expert (18% of the expert)
- gpt-oss 25.9 µs (16%)
- minimax-m3 38.7 µs (11%)
- dsv4-flash 18.3 µs (11%)
- dsv4-pro 22.9 µs (6%)

The SiLU shapes (dsv3, kimi-k2.7, glm, mistral) do the same kind of work inside RE-GU: a copy pass with
`silu_tile` on the MATH thread after the K-loop. At ~1-1.5 µs per tile that is roughly 10-25 µs per
expert. During this time the FPU is idle. r02-b04-a02 points at the same lever.

Today all of it runs after the whole K-loop, on the MATH thread, so neither the SFPU nor the FPU
overlaps with the other. The only FPU work that can legally overlap with an expert's activation
without reordering the weight streams is the expert's own LAST gate/up K-block. Subblock s of the
output is final as soon as the last K-block's matmul for s has been packed, so its activation can
run while the FPU computes the last-block matmuls of subblock s+1.

The last K-block's FPU time is about 1/num_blocks of RE-GU:
- gpt-oss: 6 blocks of w=15, ~16 µs
- minimax-m3: 12 blocks, ~17 µs
- dsv4-pro: 14 blocks, ~16 µs
- dsv4-flash: 8 blocks, ~8-10 µs
- kimi-k3: 14 blocks, ~7 µs
- SiLU shapes: 8-14 blocks, ~8-15 µs

## Mechanism
This is a compute-kernel-only change (`device/kernels/compute/fused_swiglu.cpp`). There is no
host or dataflow change.
1. The activation's SFPU moves to the PACK thread. This is the `apply_activation_from_pack` /
   moe_gpt pattern: SEMWAIT on the math commit, set the packer dest offset, run the SFPU tile op,
   STALLWAIT for the SFPU, then pack. The SFPU init runs on PACK at kernel start. Binary ops use
   the same `calculate_*` sfpi functions and templates as today; SiLU uses `silu_tile_pack`, which
   is the same `calculate_silu<fp32,8>` as `silu_tile`. The numerics should stay bit-identical.
2. The last gate/up K-block of `matmul_phase_fused_gu` gets its own loop:
   - The last block runs in subblocks of width LW: SB if SB <= 4, else SB/2, so that pcN 6 runs
     as 3+3.
   - For LW <= 4, the gate and up matmuls share one dst acquire (gate in dst[0,LW), up in
     dst[LW,2LW)). Same MACs, same packs.
   - After subblock s is packed (L1_ACC, same as before), PACK `push_back`s its LW slots on both
     partials. The pushes are incremental, and the pack index is relative to the pushed pointer, so
     the slots are unchanged.
   - Before the matmul of subblock s+1, UNPACK/MATH switch to the activation's partials->dst pass
     for subblock s: copy_tile, or the bias bcast-add on gpt-oss, the same ops as today. PACK runs
     the SFPU on that dst half. MATH then switches back to the matmul and computes s+1 on the
     other dst half while the SFPU runs.
   - The last subblock is activated after the loop, and the CB pads (pointer-only) keep every
     CB's block size constant.
   - Binary activations write cb_activated directly, and `binary_activation_phase` is no longer
     called.
   - SiLU writes cb_gate_intermed as today, and `multiply_phase` is unchanged.

## Why this is not a repeat
- r02-b04-a02 decoupled the down-weight stream (DM) and did not touch the activation.
- r02-b01-a02 and r02-b04-a02 only measured the activation.
- No node has moved SFPU work off MATH or overlapped it with FPU work. The "ACT ∥ next expert's
  GU" idea (cross-expert pipelining) would need a DM reorder. This one stays inside one expert and
  one K-block, so reader/writer stay byte-for-byte unchanged.
- The only per-K-block change is in the last block. r01-b02-a04 showed that subblock shape and
  per-call overhead are neutral, so running LW-wide subblocks in the last block should cost
  nothing.

## Expected effect and risk
- Per expert I expect to save up to ~(S-1)/S of the last block's FPU time, where S is the number
  of last-block subblocks (6-9 on pcN 9, 2*pcM on pcN 6), minus a few hundred cycles of
  reconfig per subblock.
- Estimates:
  - gpt-oss -5..-8%
  - minimax-m3 -3..-4%
  - dsv4-flash -3..-5%
  - SiLU shapes -2..-4% where compute-bound (mistral, glm)
  - kimi-k3 / dsv4-pro / kimi-k2.7 / dsv3: less, because they sit near the DRAM ceiling
- Geomean about +2%.
- Risks:
  - SFPU/FPU concurrency hazards: the MATH commit's dest_section_flip waits for SFPU idle, which
    limits the overlap to one acquire per SFPU chunk. That is why gate and up share an acquire.
  - A wrong pack index or pad count would show as an accuracy failure or a hang (CB imbalance).
  - Format reconfig mistakes would show as accuracy failures (PCC collapse).
- Bit-identical PCC/max_abs/mean_abs vs the parent is the correctness check. The RE-GU / RE-ACT
  zones show where the time went.
