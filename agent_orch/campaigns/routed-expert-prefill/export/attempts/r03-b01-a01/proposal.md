# r03-b01-a01: combine r02-b02-a04's mixed-height phase-B planner with r02-b03-a04's PACK-thread activation pipelined into the last gate/up K-block

## Motivation
The two best round-2 lines both sit on r02-b03-a03's code and touch disjoint files:
- r02-b02-a04 (1.5697, best): planner only (`kernels/adaptive_chunk.hpp`). The two-phase phase B may use a
  mixed-height layout (4+3+3), so dsv3 / glm lose their group tail (-6.4% / -7.9%).
- r02-b03-a04 (1.5630) and r02-b01-a04 (1.5657): compute only. The op-local short SFPU bodies run on the PACK
  thread, pipelined into the last gate/up K-block, so RE-ACT is ~0 on every binary shape. Median core moved down
  on 8 of 9 shapes: gpt-oss -4.5%, mm3 -2.9%, dsv3 -2.5%, mistral -4%, dsv4-pro -1.3%.
r02-b02-a04 has the old serial activation, and the PACK-pipeline nodes have the old phase-B planner. r02-b02-a04's
reflection lists this combination as next #2. Nobody has run it.

## Mechanism
Apply r02-b03-a04's diff vs r02-b03-a03 (`compute/fused_swiglu.cpp`, `compute/binary_act_sfpu.h`) onto
r02-b02-a04. r02-b02-a04 differs from r02-b03-a03 only in `adaptive_chunk.hpp`, so the patch applies cleanly onto
the identical base files. I chose b03-a04 over b01-a04 because it is compute-only and kernel-only (no host rebuild,
no reader/writer churn from b01-a02's `start` folding and zones). Its score is within noise of b01-a04's.

## Why this is not a repeat
- r02-b01-a04 / r02-b03-a04 had the pipeline but used the uniform-only phase B (dsv3 549-560, glm 492-497).
- r02-b02-a04 had the mixed phase B but the serial MATH activation (gpt-oss 267, mm3 382, dsv4-pro 1081).
No node has both.

## Expected effect and risk
- dsv3 / glm: keep r02-b02-a04's ~529 / ~453 (DRAM-bound now, so at most -1..-2% from the pipeline).
- gpt-oss / mm3 / mistral / dsv4-pro / kimi-k3: about -1.5..-2% each, as in r02-b03-a04.
- Expected geomean ~1.58-1.59.
- Risk: the pipelined K-loop has not run under a mixed 4-row phase-B group (pcM 2 with phantom tile rows). The
  CB push/pad accounting (`pushed + pad == EFF_OUT_MAX`) is per-chunk and layout-agnostic, so I expect no issue.
  A hang or PCC fail on dsv3 / glm only would point there.
