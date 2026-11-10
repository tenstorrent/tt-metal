# Fused MLP decode block (prototype, r16)

One `ttnn.generic_op` for a Qwen3.8 TP8 MLP decode block: y = (silu(x @ Wg) * (x @ Wu)) @ Wd, M = 32 rows
(8 users x 4 verify rows), BFP4 weights, on a 10x8 core rectangle (68 gate|up cores, 80 down cores).

- Weights are re-laid out so every core's slice (one gate|up pair, two down columns) is contiguous in the
  nearest DRAM bank (interleaved pages with p = 8 * slot + bank).
- NCRISC (NOC1) streams the even gate|up block groups; BRISC (NOC0) issues the odd groups and the down slice
  at t = 0, gathers x / h to the root core (core 0) by unicast, and the root multicasts them once per rectangle.
- Correct vs a torch golden (PCC 0.99998) in ttsim and on silicon (d30u20).

Silicon (P150, device 0, trace replay x200): fused 77.9 us/call (65 us kernel) vs DSMC gate_up + down 57.9 us.
Weight drain with all reads issued at t = 0: 18.8 MB in ~48 us (~390 GB/s) vs DSMC 325 GB/s. The remaining cost is
x / h distribution over the NOC (~17 us for x, ~10-17 us for h), which a persistent multi-layer kernel overlaps
with the next layer's weight stream. See bench_fused_mlp.py (FM_GRP, FM_DRAIN, FM_REPS).
