# routed-expert-prefill: discovery history

Generated 2026-10-08T23:50 from git tags `dream/routed-expert-prefill/n/*` and the ledger.

**Baseline** (µs, per-chip mean device kernel time; noise ±1.0%): deepseek-v3-h7168-i2048 801.27, kimi-k2-7-h7168-i2048 1161.99, kimi-k3-latent-h3584-i3072 2570.42, glm-5-3-h6144-i2048 704.06, mistral-small-4-h4096-i2048 261.79, deepseek-v4-flash-h4096-i2048 525.15, deepseek-v4-pro-h7168-i3072 1604.12, minimax-m3-h6144-i3072 477.77, gpt-oss-120b-h2880-i2880 572.05
**Best valid:** `r02-b02-a04` score 1.5697 (on r02-b03-a03's code, the two-phase schedule's phase B may use a mixed-height layout (4+3+3 etc.), one leftover expert per group largest-first, so dsv3/glm's 3 leftover experts run at per_core_M 2 on the rows that idle today)
**Attempts:** 32 committed (31 valid) over 2 round(s); 0 lost.

## Leaderboard (top 10 valid)

| node | score | mechanism | deepseek-v3-h7168-i2048 µs | kimi-k2-7-h7168-i2048 µs | kimi-k3-latent-h3584-i3072 µs | glm-5-3-h6144-i2048 µs | mistral-small-4-h4096-i2048 µs | deepseek-v4-flash-h4096-i2048 µs | deepseek-v4-pro-h7168-i3072 µs | minimax-m3-h6144-i3072 µs | gpt-oss-120b-h2880-i2880 µs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `r02-b02-a04` | 1.5697 | on r02-b03-a03's code, the two-phase schedule's phase B may use a mixed-height layout (4+3+3 etc.), one leftover expert per group largest-first, so dsv3/glm's 3 leftover experts run at per_core_M 2 on the rows that idle today | 529.07 | 736.47 | 1405.94 | 453.49 | 195.63 | 327.93 | 1080.73 | 381.63 | 267.00 |
| `r02-b01-a04` | 1.5657 | combine r02-b03-a03's lineage (mixed-height layouts, single-round two-phase plan, op-local short binary SFPU bodies) with this branch's PACK-thread activation pipelined into the last gate/up K-block, running the short SFPU bodies on PACK | 560.29 | 739.33 | 1386.02 | 492.21 | 189.10 | 325.66 | 1068.83 | 368.25 | 262.25 |
| `r02-b03-a04` | 1.5630 | pipeline the op-local fast SFPU activations into the last gate/up K-block on the PACK thread (port of r02-b01-a03's pack-thread pipeline onto r02-b03-a03's bodies and the two-phase/mixed-layout schedule) | 549.23 | 740.94 | 1399.52 | 497.22 | 189.47 | 326.10 | 1077.54 | 369.40 | 261.90 |
| `r02-b03-a03` | 1.5435 | same-math binary SFPU activations in fewer instructions: op-local SwiGLU-OAI / SiTU-GLU / clamped SiLU-GLU with constants hoisted into LRegs / spare CRegs, SFPSWAP clamps, and replayed unrolled bodies | 565.01 | 734.20 | 1405.32 | 492.21 | 197.93 | 326.16 | 1093.15 | 376.87 | 267.59 |
| `r02-b04-a04` | 1.5427 | same-math SFPU activations in fewer instructions for every activation: port r02-b03-a03's op-local SwiGLU-OAI / SiTU-GLU / clamped SiLU-GLU bodies and give the plain-SiLU gate pass the same treatment (constants in LRegs / Prgm1-2 CRegs, unrolled for sfpi replay) | 558.45 | 771.42 | 1361.04 | 480.50 | 193.98 | 325.63 | 1128.53 | 382.41 | 267.19 |
| `r02-b04-a01` | 1.5314 | mixed-height row-group layouts (3+3+2+2 etc.) with a height-aware LPT expert assignment, taken only when the cost model says they strictly beat the uniform 10/5/2 layouts | 555.66 | 767.50 | 1396.97 | 486.27 | 192.56 | 322.37 | 1131.80 | 388.78 | 277.13 |
| `r02-b02-a01` | 1.5279 | mixed-height expert row groups: runtime layouts 3+3+2+2 and 3+3+3 (beside 10 / 5+5 / 2x5) with a height-aware largest-first expert deal | 564.96 | 765.90 | 1391.68 | 492.92 | 190.65 | 332.77 | 1125.55 | 392.74 | 268.96 |
| `r02-b03-a02` | 1.5278 | combine r02-b04-a01's mixed-height row-group layouts with the two-phase schedule, taking two-phase only when the leftover experts fit in one phase-B round | 555.93 | 736.01 | 1404.53 | 500.32 | 196.67 | 330.49 | 1083.53 | 393.39 | 281.10 |
| `r02-b01-a03` | 1.5275 | pipeline the gate/up activation into the last gate/up K-block: per-subblock hand-off of the final partials, SFPU (binary op or SiLU) on the PACK thread overlapping the next subblock's last-block matmul on MATH | 556.67 | 755.77 | 1353.55 | 475.43 | 184.54 | 326.32 | 1118.38 | 372.05 | 330.77 |
| `r02-b04-a03` | 1.5234 | single-row expert groups: a 10 x 1-row layout with no weight multicast (row-major x only), so kimi-k3's 28 small experts run 2-3 per row at their own tile count instead of 5-6 per 2-row group at padded per_core_M 2; reverts the parent's writer-owned down stream | 564.58 | 773.08 | 1364.98 | 491.63 | 194.26 | 327.20 | 1125.31 | 383.23 | 286.22 |

## Round r01

Policy `v0` (beta 0.6), W=4, R=4, root `dream/routed-expert-prefill/root`. Plan: v0: fixed campaign defaults

### Branch b01

| node | parent | mechanism | tags | score | Δ parent | fail_class | next |
|---|---|---|---|---|---|---|---|
| `r01-b01-a01` | root | runtime expert-parallel row groups: split the 8 M-rows into 1/2/4 groups that run different experts concurrently | work-partition, expert-parallel, multicast-topology, runtime-schedule | 1.4376 | +0.4376 | ok | **4-expert shapes / gpt-oss:** the remaining gap is latency within one expert's pipeline. Two |
| `r01-b01-a02` | r01-b01-a01 | row-shared x tilize: each gate/up K-block of row-major x is tilized once per M-row by a rotating owner core and multicast as bf8 tiles | compute, tilize, x-multicast, dedup, row-major-path | 1.4716 | +0.0340 | ok | 1. **Group imbalance / tail (largest remaining prize, ~5-10% on few-expert shapes).** Groups doing |
| `r01-b01-a03` | r01-b01-a02 | expert-group pacing: a row group more than ~3% ahead (fraction of its weight K-blocks) holds its weight reads until the slowest group catches up, via a grid-multicast progress board | scheduling, load-balance, expert-parallel, dram-fairness, data-movement | 1.0363 | -0.4353 | ok | Don't build on this node. Children should start from r01-b01-a02 (1.4716). |
| `r01-b01-a04` | r01-b01-a03 | cross-unit x look-ahead: on top of r01-b01-a02 (reverting a03's pacing), the row-shared x pipeline (RM read 2 blocks ahead, owner tilize 1 block ahead) runs on across chunk/expert boundaries, so each unit's first x blocks are read and tilized during the previous unit | x-path, tilize, expert-boundary, pipelining, revert-parent, row-major-path | 1.4576 | +0.4213 | ok | Start from r01-b01-a02 (or keep this node; same performance). Don't spend more attempts on |

### Branch b02

| node | parent | mechanism | tags | score | Δ parent | fail_class | next |
|---|---|---|---|---|---|---|---|
| `r01-b02-a01` | root | row-group expert parallelism: runtime G-row groups run disjoint experts concurrently | scheduling, expert-parallel, mcast, dram-issue | 1.4017 | +0.4017 | ok | 1. Balance the expert -> group deal: when experts % groups != 0 the last wave runs on part of the |
| `r01-b02-a02` | r01-b02-a01 | collision-free DM reader placement for row groups: fixed staggered x sender, groups >= 2 rows, no bias reads for other groups' experts | data-movement, sender-placement, expert-parallel, repair | 1.4460 | +0.0443 | ok | 1. The 4-expert shapes (mistral, minimax-m3, gpt-oss) have no cross-expert overlap inside a group |
| `r01-b02-a03` | r01-b02-a02 | port r01-b01-a02 row-shared x tilize (one owner core per M-row tilizes each RM x K-block and multicasts bf8 tiles) onto the collision-free row-group schedule | compute, tilize, x-multicast, dedup, row-major-path, combination | 1.4665 | +0.0205 | ok | 1. Stop porting between the b01 and b02 branches: the schedule details are now worth < 1%. |
| `r01-b02-a04` | r01-b02-a03 | balanced ragged matmul subblocks: per_core_N 9 runs as 1x5 + 1x4 instead of 3 x 1x3 (gate/up on the hidden-3072/2880 shapes, down on gpt-oss), cutting per-row LLK unpack-context rounds by a third | compute, matmul-subblock, llk-overhead, ragged-subblock | 1.4705 | +0.0040 | ok | 1. **Measure the activation phase before changing anything else.** Put one DeviceZoneScopedN around |

### Branch b03

| node | parent | mechanism | tags | score | Δ parent | fail_class | next |
|---|---|---|---|---|---|---|---|
| `r01-b03-a01` | root | one-block DRAM read-ahead of weight blocks on the sender cores (writer up in gate/up, both halves in down) when the next ring slot is already free | dataflow, weight-delivery, prefetch, pipelining | 0.9416 | -0.0584 | ok | Don't retry read-ahead in this form. If you revisit it, issue the next block's reads AFTER the |
| `r01-b03-a02` | r01-b03-a01 | batch x over up to 4 gate/up K-blocks per x CB block at small per_core_M (one stick read, mcast, tilize init and matmul re-init per batch; RM path), reverting the parent read-ahead | compute, x-path, tilize, handshake-amortization, revert-parent | 0.9626 | +0.0210 | ok | Don't spend more attempts on per-K-block overhead or weight/x read timing in the root schedule. |
| `r01-b03-a03` | r01-b03-a02 | on r01-b01-a02's row-group code: weight-sender writers feed the next unit's first gate/up block (up read + gate/up mcast) while waiting to drain this unit's output, instead of after the drain | data-movement, expert-boundary, prefetch, rebase-onto-best, writer | 1.4574 | +0.4948 | ok | Don't spend more attempts on unit-boundary or in-loop weight-read timing. With this node there |
| `r01-b03-a04` | r01-b03-a03 | span the device's 2 spare core rows: add 10-row expert-group layouts (5 groups of 2 rows, 2 groups of 5 rows) that the runtime picker takes only when they strictly beat the 8-row layouts | work-partition, expert-parallel, grid-size, row-groups, runtime-schedule | 1.4856 | +0.0282 | ok | 1. **Finer than whole-expert balancing on kimi-k3 / kimi-k2.7.** With 5 groups the critical groups |

### Branch b04

| node | parent | mechanism | tags | score | Δ parent | fail_class | next |
|---|---|---|---|---|---|---|---|
| `r01-b04-a01` | root | software-pipeline weight streams: issue next K-block DRAM read before multicasting the current one (writer up, reader down) | data-movement, pipelining, weight-mcast | 0.9330 | -0.0670 | ok | Start from root, not from this node. Directions this result points at: |
| `r01-b04-a02` | r01-b04-a01 | halve gate/up K-block count: in0_block_w_gu 16->32 paid by per_core_M_max 4->2, drop M-doubling step (reverts parent read-ahead) | k-block-geometry, l1-budget, host-config, revert-parent | 0.0000 | - | accuracy_fail | Don't build on this node's config. Restore in0_block_w_gu <= the baseline width (16, or 8 for |
| `r01-b04-a03` | r01-b04-a02 | port r01-b01-a02 (row groups + row-shared x tilize) and deepen the gate/up (x, gate, up) and down (down weights, activated) input rings to 3 slots when L1 has room | cb-depth, pipelining, l1-budget, port-best, row-groups | 1.4622 | - | ok | Treat this node as r01-b01-a02 for practical purposes. It ports that code into branch b04, so |
| `r01-b04-a04` | r01-b04-a03 | use all 10 Blackhole worker rows: 11x10 grid with expert row groups of 10/5/2 rows (was 11x8 with 8/4/2) | work-partition, grid-size, expert-parallel, row-groups, host-config | 1.5011 | +0.0389 | ok | 1. **Group-tail imbalance is now the largest, measured waste.** With 5 groups the expert counts |

## Round r02

Policy `v0` (beta 0.6), W=4, R=4, root `dream/routed-expert-prefill/n/r01-b04-a04`. Plan: v0: fixed campaign defaults

### Branch b01

| node | parent | mechanism | tags | score | Δ parent | fail_class | next |
|---|---|---|---|---|---|---|---|
| `r02-b01-a01` | root | DRAM-aware group-size picker: model groups as sharing DRAM (fluid phases) so few-tile expert sets run 2 groups of 5 rows at per_core_M 1 instead of 5 groups of 2 rows with a compute-bound tail | runtime-schedule, cost-model, dram-bandwidth, row-groups, work-partition | 1.4854 | -0.0157 | ok | Start from r01-b04-a04, not this node (or revert this header change). |
| `r02-b01-a02` | r02-b01-a01 | fold the per-core `start` DRAM page read into the one-core counts multicast (reader+writer copies), on r01-b04-a04's picker (reverts parent), plus per-expert MATH phase zones RE-GU/RE-ACT/RE-DN | prologue, dram-hotspot, multicast, measurement, revert-parent | 1.4889 | +0.0035 | ok | 1. **Hide or shrink the binary activation phase.** It costs kimi-k3 ~18%, gpt-oss ~16%, dsv4-flash and mm3 |
| `r02-b01-a03` | r02-b01-a02 | pipeline the gate/up activation into the last gate/up K-block: per-subblock hand-off of the final partials, SFPU (binary op or SiLU) on the PACK thread overlapping the next subblock's last-block matmul on MATH | compute, sfpu-fpu-overlap, activation, pack-thread-sfpu, kernel-only | 1.5275 | +0.0386 | ok | 1. **Combine with r02-b04-a01's mixed-height layouts (best 1.5314).** That is the obvious +2-3%. |
| `r02-b01-a04` | r02-b01-a03 | combine r02-b03-a03's lineage (mixed-height layouts, single-round two-phase plan, op-local short binary SFPU bodies) with this branch's PACK-thread activation pipelined into the last gate/up K-block, running the short SFPU bodies on PACK | combination, sfpu-fpu-overlap, pack-thread-sfpu, runtime-schedule, row-groups, sfpu | 1.5657 | +0.0382 | ok | 1. **Cut the planner prologue (measured 5-29 µs per call, ~+0.7-1% geomean, kernel-only).** |

### Branch b02

| node | parent | mechanism | tags | score | Δ parent | fail_class | next |
|---|---|---|---|---|---|---|---|
| `r02-b02-a01` | root | mixed-height expert row groups: runtime layouts 3+3+2+2 and 3+3+3 (beside 10 / 5+5 / 2x5) with a height-aware largest-first expert deal | work-partition, expert-parallel, row-groups, runtime-schedule, load-balance | 1.5279 | +0.0268 | ok | 1. **The per-expert fixed cost (~1.3 per_core_M units on gpt-oss) is now measured and is the biggest |
| `r02-b02-a02` | r02-b02-a01 | per-sender NoC request VCs for the weight DRAM reads ((column + row group) & 3) so concurrent column/group weight streams are arbitrated round-robin instead of FCFS on one VC | data-movement, noc-vc, dram-fairness, group-tail, kernel-only | 1.5191 | -0.0088 | ok | 1. Start from r02-b02-a01's code (revert this node's VC argument), or keep it with `weight_read_vc` returning |
| `r02-b02-a03` | r02-b02-a02 | down-weight look-ahead across the activation gap: the reader streams down K-blocks Dw-2 ahead of the activated multicast from a deeper, down-weight-only cb_in1_down ring sized from leftover L1 (reverts parent request VCs) | data-movement, down-phase, prefetch, activation-gap, cb-depth, l1-budget, revert-parent | 1.5059 | -0.0132 | ok | 1. **Start from r02-b02-a01's code** (or r02-b04-a01, the best node). Revert this node's reader |
| `r02-b02-a04` | r02-b02-a03 | on r02-b03-a03's code, the two-phase schedule's phase B may use a mixed-height layout (4+3+3 etc.), one leftover expert per group largest-first, so dsv3/glm's 3 leftover experts run at per_core_M 2 on the rows that idle today | runtime-schedule, two-phase, row-groups, load-balance, mixed-height, planner, rebase-onto-best, kernel-only | 1.5697 | +0.0638 | ok | 1. **dsv3/glm are now DRAM-bound (~440 GB/s), so don't spend more schedule attempts there.** The only |

### Branch b03

| node | parent | mechanism | tags | score | Δ parent | fail_class | next |
|---|---|---|---|---|---|---|---|
| `r02-b03-a01` | root | two-phase expert schedule: full waves on the small row groups, then the leftover experts on larger (5/10-row) groups so the tail runs at lower per_core_M instead of leaving groups idle | scheduling, expert-parallel, row-groups, load-balance, runtime-schedule | 1.4973 | -0.0038 | ok | 1. **Cheapest fix, likely a net win (~+0.7% score):** only take a two-phase plan when the leftover |
| `r02-b03-a02` | r02-b03-a01 | combine r02-b04-a01's mixed-height row-group layouts with the two-phase schedule, taking two-phase only when the leftover experts fit in one phase-B round | runtime-schedule, row-groups, combination, load-balance, repair | 1.5278 | +0.0305 | ok | 1. **Cut the planner prologue (cheap, kernel-only, likely about +0.5-1%).** In `make_plan`, reject right after |
| `r02-b03-a03` | r02-b03-a02 | same-math binary SFPU activations in fewer instructions: op-local SwiGLU-OAI / SiTU-GLU / clamped SiLU-GLU with constants hoisted into LRegs / spare CRegs, SFPSWAP clamps, and replayed unrolled bodies | compute, sfpu, activation, instruction-count, kernel-only | 1.5435 | +0.0157 | ok | 1. **The SiLU shapes' activation is untouched.** silu_tile runs in the gate copy pass of |
| `r02-b03-a04` | r02-b03-a03 | pipeline the op-local fast SFPU activations into the last gate/up K-block on the PACK thread (port of r02-b01-a03's pack-thread pipeline onto r02-b03-a03's bodies and the two-phase/mixed-layout schedule) | compute, sfpu-fpu-overlap, activation, pack-thread-sfpu, combination, kernel-only | 1.5630 | +0.0195 | ok | 1. **SiLU body treatment (still untried).** silu_tile_pack runs the stock calculate_silu (exp + recip |

### Branch b04

| node | parent | mechanism | tags | score | Δ parent | fail_class | next |
|---|---|---|---|---|---|---|---|
| `r02-b04-a01` | root | mixed-height row-group layouts (3+3+2+2 etc.) with a height-aware LPT expert assignment, taken only when the cost model says they strictly beat the uniform 10/5/2 layouts | work-partition, row-groups, runtime-schedule, expert-parallel, non-uniform-groups | 1.5314 | +0.0303 | ok | 1. **Fix the cost model's fixed term.** Measured per-expert fixed cost F is ~2 units (dsv4-pro) to ~5 |
| `r02-b04-a02` | r02-b04-a01 | writer-owned down-weight stream: the column sender reads+multicasts down K-blocks on NoC 1 with its own ready/valid sems, so down weights run ahead of the activated multicast (reader keeps only the activated stream); plus MATH phase zones | data-movement, down-phase, decoupling, pipelining, writer, measurement | 1.3612 | -0.1702 | ok | **Start from r02-b04-a01 (the parent), not this node.** If you want the phase zones, copy only |
| `r02-b04-a03` | r02-b04-a02 | single-row expert groups: a 10 x 1-row layout with no weight multicast (row-major x only), so kimi-k3's 28 small experts run 2-3 per row at their own tile count instead of 5-6 per 2-row group at padded per_core_M 2; reverts the parent's writer-owned down stream | work-partition, row-groups, runtime-schedule, single-row-groups, revert-parent | 1.5234 | +0.1622 | ok | 1. **Keep or drop freely.** It is worth ~0.25% geomean through kimi-k3. On this lineage it is now |
| `r02-b04-a04` | r02-b04-a03 | same-math SFPU activations in fewer instructions for every activation: port r02-b03-a03's op-local SwiGLU-OAI / SiTU-GLU / clamped SiLU-GLU bodies and give the plain-SiLU gate pass the same treatment (constants in LRegs / Prgm1-2 CRegs, unrolled for sfpi replay) | compute, sfpu, activation, instruction-count, kernel-only, port | 1.5427 | +0.0193 | ok | 1. **Combine with two-phase (r02-b03-a02's `plan_launch` / phase loop).** kimi-k2.7 (771 here vs 734 there) |

## Reading a node in full

```bash
git show dream/routed-expert-prefill/n/<id>:agent_orch/campaigns/routed-expert-prefill/attempts/<id>/proposal.md
git show dream/routed-expert-prefill/n/<id>:agent_orch/campaigns/routed-expert-prefill/attempts/<id>/reflection.md
git show dream/routed-expert-prefill/n/<id>:agent_orch/campaigns/routed-expert-prefill/attempts/<id>/eval/summary.md
git diff dream/routed-expert-prefill/n/<id>~1 dream/routed-expert-prefill/n/<id> -- . ':!agent_orch'
```
