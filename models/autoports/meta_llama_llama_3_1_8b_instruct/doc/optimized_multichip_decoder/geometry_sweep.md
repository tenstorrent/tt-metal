# Final-topology projection geometry sweep

All rows use the selected TP4 1x4 ring, BF16 residual and minimal-all-reduce
payload, BFP4 weights, LoFi math, batch 1, logical sequence 18, and traced
warmed decode.  Each row passed PCC against the invariant optimized
single-chip output.  The initial role isolation used the then-current 24-core
QKV candidate.  After QKV block32 won, packing and every near-winning role
candidate were rerun cumulatively under QKV32 with three independent
100-prefill/1000-replay processes.  Gate and up have identical shapes and
deliberately share one role config.

| Role/candidate | active cores / grid | per-core N / subblock | K block | matching output memory | decode ms | Decision |
| --- | --- | --- | --- | --- | ---: | --- |
| old O | 64 / 11x6 | 2 / 1x2 | 8 | 64-core width-sharded | 0.246752 QKV32/O8 median | replaced |
| O core32 | 32 / 8x4 | 4 / 1x4 | 16 | 32-core width-sharded | 0.248511 | reject |
| O subblock1 | 64 / 11x6 | 2 / 1x1 | 8 | 64-core width-sharded | 0.247637 | reject |
| O max K | 64 / 11x6 | 2 / 1x2 | 32 | 64-core width-sharded | 0.247458 under QKV16 | advance |
| retained gate and up | 56 / 11x6 | 2 / 1x2 | 8 | 56-core width-sharded | 0.246686 final | keep |
| gate/up core28 | 28 / 7x4 | 4 / 1x4 | 16 | 28-core width-sharded | 0.255470 | reject |
| gate/up non-power N | 16 / 8x2 | 7 / 1x1 | 16 | 16-core width-sharded | 0.265605 | reject |
| gate/up subblock1 | 56 / 11x6 | 2 / 1x1 | 8 | 56-core width-sharded | 0.248815 | reject |
| gate/up K32 / K64 | 56 / 11x6 | 2 / 1x2 | 32 / 64 | 56-core width-sharded | 0.257140 / 0.257329 | reject |
| retained down | 64 / 11x6 | 2 / 1x2 | 8 | 64-core width-sharded | 0.246686 final | keep |
| down non-power K14 | 32 / 8x4 | 4 / 1x4 | 14 | 32-core width-sharded | 0.248917 | reject |
| down non-power K28 | 16 / 8x2 | 8 / 1x4 | 28 | 16-core width-sharded | 0.252272 | reject |
| down subblock1 | 64 / 11x6 | 2 / 1x1 | 8 | 64-core width-sharded | 0.248032 | reject |
| down non-power K56 | 64 / 11x6 | 2 / 1x2 | 56 | 64-core width-sharded | 0.253300 | reject |
| QKV block16 | 24 / 8x3 | 2 / 1x2 | 16 | 24-core width-sharded | 0.247549 old median | replaced |
| QKV block32 | 24 / 8x3 | 2 / 1x2 | 32 | 24-core width-sharded | 0.246752 with old O8 | advance |
| QKV block64 | 24 / 8x3 | 2 / 1x2 | 64 | 24-core width-sharded | 0.247075 | reject |

The required cumulative QKV32 family comparison is:

| Coherent QKV32 family | decode samples, ms | median | PCC | Decision |
| --- | --- | ---: | ---: | --- |
| separate gate/up, O block32 | 0.246689 / 0.246691 / 0.246665 | **0.246689** | 0.9999998070869076 | promote |
| separate gate/up, O subblock1 | 0.246817 / 0.246810 / 0.246818 | 0.246817 | 0.9999998068156027 | reject |
| separate gate/up, down subblock1 | 0.247242 / 0.247234 / 0.247238 | 0.247238 | 0.9999998068156027 | reject |
| packed gate/up through SwiGLU/down | 0.248682 / 0.248581 / 0.248601 | 0.248601 | 0.9999998068156027 | reject |

Six subsequent executions through the promoted `default` path reproduce a
0.246686-ms median at PCC 0.9999998070869076.  Thus the final family is QKV
block32 plus O block32; gate/up and down retain block8.

The one-core-per-output-tile candidates are physical blockers rather than
untried knobs: O/down need 128 active workers and gate/up need 112, while this
Blackhole exposes 110.  Other arithmetically legal per-core N values are not
physical blockers: the bounded family search measured the retained advisor
core count, material 2x/4x core-count reductions, role-specific subblocks and
matching output layouts, then cumulatively reran every near-winner under
QKV32.  The down K dimension has 112 tiles, so 14, 28, and 56 explicitly cover
its material larger non-power-of-two K-block divisors.  O has 32 K tiles and
was swept through its maximum K block.  Gate/up and QKV have 128 K tiles and
their K blocks were swept beyond the retained value through 64.

Packed gate/up was also adapted under the exact final QKV32 family: one packed
56-core projection (`per_core_N=4`), L1-interleaved split, SiLU-multiply, and
the retained down projection.  It passes PCC `0.9999998068156027` but measures
0.248601 ms median versus 0.246689 ms for separate gate/up, so the two-row
group is retained.
