# r01-b01-a04: cross-unit x look-ahead: the row-shared x pipeline (RM read 2 blocks ahead, owner tilize 1 block ahead) runs on across chunk/expert boundaries, so a unit's first x blocks are read and tilized during the previous unit

## Motivation
The parent r01-b01-a03 (pacing, 1.0363) is a dead end; its own reflection says to build on
r01-b01-a02 (1.4716). This node reverts a03's code (pacing board, CB_PACE) back to a02 and adds
one mechanism on top.

In a02's row-shared tilize, the x look-ahead stops at every gate/up unit (chunk) boundary:
- reader: at unit start the owners of blocks 0 and 1 issue their row-major x reads and BARRIER
  on them synchronously ("Blocks 0 and 1 of the chunk are read up front"). The owner of block 1 does
  this before its own weight-sender work for block 0.
- compute: the owner of block 0 tilizes it only after it has finished the previous unit's down
  matmul and output copy pass. Then its reader multicasts the bf8 tiles to the row, and only then
  can any core in the row start the unit's first matmul.
So every unit start puts (tilize of x block 0 + its multicast handshake) on the whole row's critical
path, plus the up-front read latency if the reader has not run far enough ahead. r01-b03-a03
moved the weight side of this boundary earlier and measured nothing. Its reflection lists what
remains on the boundary's compute critical path: "the down tail, the output copy pass and the x(0)
tilize+multicast". This node removes the x(0) part. Units per group: kimi-k3 7, kimi-k2.7 and
dsv4-pro 3, dsv3, glm, dsv4-flash and gpt-oss 2 (the first unit's bubble at kernel start stays).
The x(0) tilize is 16-48 tiles plus a tilize init/uninit and reconfig, about 2-5 us, plus the
multicast handshake, about 1-2 us.

## Mechanism
Row-major x path only, and only when a unit has >= 2 gate/up K-blocks (compile-time). Otherwise,
and on the TILE path, the code is unchanged.
- Gate/up blocks are already numbered globally per row (x_blk). The look-ahead now uses the
  global block index across units: at the global block g the owner of g+2 issues its RM read, and
  before its matmul of g the owner of g+1 tilizes it, even when g+1 / g+2 fall in the NEXT unit. The
  next unit is the next chunk of this expert, or chunk 0 of the next expert this row group owns.
- reader: a small XUnit descriptor (count, tile count, per_core_M, first row, x start stick)
  parametrises issue_x_rm_read. The next expert this group owns is found with a copy of the group
  scheduler (same deterministic schedule). The up-front reads of blocks 0/1 run only for the
  row's first unit.
- compute: UNPACK scans one expert ahead (one mailbox value per expert, as now, but sent one expert
  earlier), so MATH/PACK know the next unit's per_core_M. matmul_phase_fused_gu takes the next
  unit's per_core_M (0 = none) and whether block 0 was already tilized. The tilize lambda takes
  the strip count as a parameter.
- Ownership, CB block sizes, push/pop sequences, the multicast protocol and the math are all
  unchanged. Only WHEN the owner reads/tilizes changes, so numerics stay bit-identical.

Files: `unified_routed_expert_ffn/device/kernels/dataflow/unified_routed_expert_ffn_reader.cpp`,
`.../kernels/compute/fused_swiglu.cpp` (both kernels JIT, no host rebuild apart from the revert of
a03's factory change).

## Why this is not a repeat
- r01-b03-a03 moved the next unit's first WEIGHT block (up read + gate/up multicast) ahead of the
  drain. Weights were already prefetched by the 2-slot rings, and that node was neutral. The x(0)
  path is different. It needs the owner's COMPUTE (tilize) and could not be prefetched at all,
  because the tilize ran after the down phase and the RM read after the down DM.
- r01-b03-a02 batched x reads (burstier x, failed). This node does not change x granularity.
- Not pacing (a03) and not ring depth (b04-a03).

## Expected effect and risk
- Expected: about 3-6 us per unit boundary. Best on kimi-k3 (6 boundaries per group, about 1-2%),
  then kimi-k2.7 / dsv4-pro (2), dsv3 / glm / dsv4-flash / gpt-oss (1). mistral and minimax-m3
  (one unit per group) should be unchanged. Probably +0.5-1% overall, which is near the ±1% noise.
- Cost moved elsewhere: the owner of the next unit's block 0 now tilizes it before its last gate/up
  matmul of the current unit, which delays that one core's activation by a few us. Only the
  down block that multicasts its column waits on it, and that block is usually not the first.
- Risks: hang if reader and compute disagree on "has next unit" / "block 0 pre-tilized" (CB
  wait on cb_x_rm / cb_x_tl never satisfied), and wrong x rows (PCC collapse) if the next unit's
  rows/start are wrong. Any accuracy change at all means a bug.
