# routed-expert-prefill: discovery history

Generated 2026-10-08T22:22 from git tags `dream/routed-expert-prefill/n/*` and the ledger.

**Baseline** (µs, per-chip mean device kernel time; noise ±1.0%): deepseek-v3-h7168-i2048 801.27, kimi-k2-7-h7168-i2048 1161.99, kimi-k3-latent-h3584-i3072 2570.42, glm-5-3-h6144-i2048 704.06, mistral-small-4-h4096-i2048 261.79, deepseek-v4-flash-h4096-i2048 525.15, deepseek-v4-pro-h7168-i3072 1604.12, minimax-m3-h6144-i3072 477.77, gpt-oss-120b-h2880-i2880 572.05
**Best valid:** `r01-b04-a04` score 1.5011 (use all 10 Blackhole worker rows: 11x10 grid with expert row groups of 10/5/2 rows (was 11x8 with 8/4/2))
**Attempts:** 16 committed (15 valid) over 1 round(s); 0 lost.

## Leaderboard (top 10 valid)

| node | score | mechanism | deepseek-v3-h7168-i2048 µs | kimi-k2-7-h7168-i2048 µs | kimi-k3-latent-h3584-i3072 µs | glm-5-3-h6144-i2048 µs | mistral-small-4-h4096-i2048 µs | deepseek-v4-flash-h4096-i2048 µs | deepseek-v4-pro-h7168-i3072 µs | minimax-m3-h6144-i3072 µs | gpt-oss-120b-h2880-i2880 µs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `r01-b04-a04` | 1.5011 | use all 10 Blackhole worker rows: 11x10 grid with expert row groups of 10/5/2 rows (was 11x8 with 8/4/2) | 547.46 | 764.28 | 1378.84 | 487.63 | 193.00 | 331.22 | 1128.00 | 381.96 | 338.76 |
| `r01-b03-a04` | 1.4856 | span the device's 2 spare core rows: add 10-row expert-group layouts (5 groups of 2 rows, 2 groups of 5 rows) that the runtime picker takes only when they strictly beat the 8-row layouts | 559.74 | 762.23 | 1373.10 | 491.21 | 195.51 | 344.67 | 1149.55 | 383.02 | 337.44 |
| `r01-b01-a02` | 1.4716 | row-shared x tilize: each gate/up K-block of row-major x is tilized once per M-row by a rotating owner core and multicast as bf8 tiles | 563.09 | 771.50 | 1478.76 | 486.89 | 185.94 | 340.66 | 1148.93 | 378.16 | 364.53 |
| `r01-b02-a04` | 1.4705 | balanced ragged matmul subblocks: per_core_N 9 runs as 1x5 + 1x4 instead of 3 x 1x3 (gate/up on the hidden-3072/2880 shapes, down on gpt-oss), cutting per-row LLK unpack-context rounds by a third | 559.87 | 778.10 | 1477.62 | 485.47 | 186.88 | 340.71 | 1143.76 | 380.05 | 365.18 |
| `r01-b02-a03` | 1.4665 | port r01-b01-a02 row-shared x tilize (one owner core per M-row tilizes each RM x K-block and multicasts bf8 tiles) onto the collision-free row-group schedule | 551.76 | 784.21 | 1480.08 | 492.72 | 187.36 | 345.77 | 1147.58 | 376.51 | 366.39 |
| `r01-b04-a03` | 1.4622 | port r01-b01-a02 (row groups + row-shared x tilize) and deepen the gate/up (x, gate, up) and down (down weights, activated) input rings to 3 slots when L1 has room | 561.88 | 767.65 | 1477.70 | 488.19 | 193.20 | 337.56 | 1153.21 | 385.60 | 368.45 |
| `r01-b01-a04` | 1.4576 | cross-unit x look-ahead: on top of r01-b01-a02 (reverting a03's pacing), the row-shared x pipeline (RM read 2 blocks ahead, owner tilize 1 block ahead) runs on across chunk/expert boundaries, so each unit's first x blocks are read and tilized during the previous unit | 556.25 | 773.42 | 1487.19 | 493.72 | 195.09 | 343.70 | 1152.81 | 382.10 | 366.62 |
| `r01-b03-a03` | 1.4574 | on r01-b01-a02's row-group code: weight-sender writers feed the next unit's first gate/up block (up read + gate/up mcast) while waiting to drain this unit's output, instead of after the drain | 559.90 | 767.55 | 1489.08 | 488.94 | 192.42 | 343.11 | 1151.53 | 384.24 | 374.68 |
| `r01-b02-a02` | 1.4460 | collision-free DM reader placement for row groups: fixed staggered x sender, groups >= 2 rows, no bias reads for other groups' experts | 566.53 | 776.37 | 1472.38 | 504.38 | 195.76 | 343.01 | 1148.13 | 381.12 | 382.93 |
| `r01-b01-a01` | 1.4376 | runtime expert-parallel row groups: split the 8 M-rows into 1/2/4 groups that run different experts concurrently | 568.04 | 777.91 | 1484.68 | 497.46 | 200.07 | 352.70 | 1142.75 | 384.82 | 382.28 |

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

## Reading a node in full

```bash
git show dream/routed-expert-prefill/n/<id>:agent_orch/campaigns/routed-expert-prefill/attempts/<id>/proposal.md
git show dream/routed-expert-prefill/n/<id>:agent_orch/campaigns/routed-expert-prefill/attempts/<id>/reflection.md
git show dream/routed-expert-prefill/n/<id>:agent_orch/campaigns/routed-expert-prefill/attempts/<id>/eval/summary.md
git diff dream/routed-expert-prefill/n/<id>~1 dream/routed-expert-prefill/n/<id> -- . ':!agent_orch'
```
