# routed-expert-prefill: discovery history

Generated 2026-10-09T04:11 from git tags `dream/routed-expert-prefill/n/*` and the ledger.

**Baseline** (µs, per-chip mean device kernel time; noise ±1.0%): deepseek-v3-h7168-i2048 801.27, kimi-k2-7-h7168-i2048 1161.99, kimi-k3-latent-h3584-i3072 2570.42, glm-5-3-h6144-i2048 704.06, mistral-small-4-h4096-i2048 261.79, deepseek-v4-flash-h4096-i2048 525.15, deepseek-v4-pro-h7168-i3072 1604.12, minimax-m3-h6144-i3072 477.77, gpt-oss-120b-h2880-i2880 572.05
**Best valid:** `r05-b02-a01` score 1.6667 (rebalance the weight DRAM reads ~60/40 toward NoC 0: the reader also reads the first ~0.2w up K-rows of each gate/up block and the leading ~60% tiles of every down K-row (tile split instead of row split), so the hot NoC-1 response links in the DRAM columns carry ~40% instead of 50%)
**Attempts:** 61 committed (57 valid) over 5 round(s); 0 lost.

## Leaderboard (top 10 valid)

| node | score | mechanism | deepseek-v3-h7168-i2048 µs | kimi-k2-7-h7168-i2048 µs | kimi-k3-latent-h3584-i3072 µs | glm-5-3-h6144-i2048 µs | mistral-small-4-h4096-i2048 µs | deepseek-v4-flash-h4096-i2048 µs | deepseek-v4-pro-h7168-i3072 µs | minimax-m3-h6144-i3072 µs | gpt-oss-120b-h2880-i2880 µs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `r05-b02-a01` | 1.6667 | rebalance the weight DRAM reads ~60/40 toward NoC 0: the reader also reads the first ~0.2w up K-rows of each gate/up block and the leading ~60% tiles of every down K-row (tile split instead of row split), so the hot NoC-1 response links in the DRAM columns carry ~40% instead of 50% | 504.86 | 701.38 | 1243.64 | 436.68 | 190.11 | 298.41 | 993.18 | 361.90 | 260.97 |
| `r05-b01-a01` | 1.6457 | plan once per core: the reader publishes the launch plan in a one-page L1 CB for the writer and compute (two planner copies removed, start-page read overlapped with planning), and the freed kernel-config space carries r04-b03-a03's idle-row tie pass so minimax-m3 / mistral run 3+3+2+2 rows instead of 2x5 with two rows idle | 510.25 | 729.24 | 1327.57 | 443.91 | 176.18 | 313.12 | 1063.55 | 350.99 | 254.00 |
| `r04-b02-a01` | 1.6275 | weight senders start block kb+1's DRAM reads as soon as their own ring slot is free in both phases: gate/up (reader gate read + writer up_go) no longer waits for the sender row's x block kb, and r03-b04-a04's down look-ahead is ported with a re-check while the sender spins on act_valid | 512.82 | 717.28 | 1342.79 | 449.24 | 183.76 | 308.69 | 1056.41 | 372.86 | 255.75 |
| `r05-b03-a01` | 1.6177 | stagger the row-shared x owner rotation by M-row (group index + 2 x row-in-group), so a gate/up K-block's x stick reads, tilize and row multicast are spread over ~10 grid columns instead of landing on the same 1-2 columns in every row | 507.74 | 719.17 | 1331.28 | 451.38 | 182.28 | 319.17 | 1061.48 | 379.70 | 260.17 |
| `r04-b03-a02` | 1.6119 | decouple the down phase's row (activated) and column (down-weight) streams in the reader: one non-blocking polling loop where each stream acks its next block as soon as its own previous block landed and its slot is free, the act sender multicasts as soon as its row acks are in, and the in1 sender's DRAM read is polled instead of barriered; parent's probe compiled out | 519.96 | 744.80 | 1314.56 | 452.13 | 183.02 | 309.34 | 1087.80 | 368.83 | 264.44 |
| `r03-b02-a03` | 1.6103 | measurement node: profiler-only Tensix hardware perf counters (FPU/SFPU busy, unpacker busy, SrcA/SrcB write and valid-wait, per-thread stalls) and FW matmul-throttle samples on the MATH thread, accumulated per gate/up and down window, to find what sets the ~30-40 cycles per tile-mult on the compute-bound shapes | 513.98 | 729.53 | 1365.81 | 451.23 | 185.90 | 316.95 | 1070.94 | 365.46 | 261.82 |
| `r04-b02-a02` | 1.6093 | the phase-4 activated row stream runs on its own: a non-blocking pump (spun in every down-loop wait) lets receivers ack activated block a+1 as soon as a is in and a slot is free and senders multicast as soon as acked, with a running-count act_valid, instead of one block per slowest column's down-weight arrival; profiler-only MATH counters made opt-in for kernel-config headroom | 520.08 | 740.38 | 1329.12 | 451.70 | 178.67 | 314.67 | 1084.18 | 371.92 | 267.69 |
| `r03-b04-a04` | 1.6078 | down-weight sender issues block kb+1's DRAM read (both DOWN_SPLIT halves) right after multicasting block kb, into its own already-free ring slot, instead of after all column receivers ack kb+1 | 509.47 | 727.12 | 1342.05 | 456.05 | 187.90 | 322.99 | 1064.33 | 373.82 | 258.40 |
| `r04-b03-a03` | 1.6066 | put the idle rows to work on a tie: when the best uniform row-group layout leaves groups idle, the planner takes a mixed-height layout with one expert per group, the same modelled makespan and a lower summed expert cost, so mistral / minimax-m3 run 3+3+2+2 rows (6-tile experts at per_core_M 2) instead of 2x5 with rows y10-11 idle | 519.48 | 755.28 | 1320.69 | 463.25 | 183.13 | 320.58 | 1080.59 | 351.98 | 265.67 |
| `r04-b04-a02` | 1.6037 | activated-stream run-ahead in the down phase: row cores grant an activated ring slot (ack its sender) as soon as it is free and senders multicast as soon as all grants are in, with per-slot valid flags and an activated ring of up to 4 slots from leftover L1 (on r04-b02-a01's reader) | 513.43 | 742.65 | 1327.13 | 457.84 | 189.71 | 314.93 | 1063.68 | 379.94 | 258.72 |

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

## Round r03

Policy `v0` (beta 0.6), W=4, R=4, root `dream/routed-expert-prefill/n/r02-b02-a04`. Plan: v0: fixed campaign defaults

### Branch b01

| node | parent | mechanism | tags | score | Δ parent | fail_class | next |
|---|---|---|---|---|---|---|---|
| `r03-b01-a01` | root | combine r02-b02-a04's mixed-height phase-B planner with r02-b03-a04's PACK-thread activation pipelined into the last gate/up K-block (short op-local SFPU bodies on PACK) | combination, sfpu-fpu-overlap, pack-thread-sfpu, two-phase, mixed-height, kernel-only | 0.0000 | - | runtime_error | Keep this code and cut at least ~2.1 KB (aim for 4 KB+ of margin) from the per-core binary set, then re-eval. |
| `r03-b01-a02` | r03-b01-a01 | repair r03-b01-a01's kernel-config-buffer overflow by planning once per core: the reader runs plan_launch and publishes the plan in a new L1 scratch CB that the writer and compute (UNPACK) read instead of planning themselves | repair, binary-size, planner, plan-sharing, combination, pack-thread-sfpu, host-config | 1.5882 | - | ok | 1. **Stack a planner trim on top (expect +0.5-1% via kimi-k3 / dsv4-flash / dsv3).** r03-b04-a01's bit-identical |
| `r03-b01-a03` | r03-b01-a02 | make the reader's single launch planner fast: port r03-b04-a01's bit-identical planner trim onto the plan-once-per-core lineage, overlap the reader/writer start reads with planning, and build the planner region at -O3 on the dataflow RISCs | planner, prologue, kernel-only, bit-identical, opt-level, combination | 1.5917 | +0.0035 | ok | 1. **Stop working on the planner.** Remaining RE-PLAN is 2.4-5.7 µs on 8 of 9 shapes and 14 µs on kimi-k3. |
| `r03-b01-a04` | r03-b01-a03 | stream x through SrcB instead of the bf4 weights through SrcA: at per_core_M >= 3 the non-last gate/up K-blocks and the down K-loop call matmul_block with rt = per_core_M, ct = 1 (reuse the weight tile, stream x), bit-identical | compute, matmul-reuse, unpack-bound, srcb-streaming, kernel-only, bit-identical | 1.5952 | +0.0035 | ok | 1. **Don't retry operand swaps, subblock shapes or reuse modes** (r01-b02-a04 width, this node's rt/ct swap). With |

### Branch b02

| node | parent | mechanism | tags | score | Δ parent | fail_class | next |
|---|---|---|---|---|---|---|---|
| `r03-b02-a01` | root | combine r02-b02-a04's planner (mixed-height phase B) with r02-b01-a04's PACK-thread pipelined activation, and trim the planner prologue (counts cached once per RISC, uniform layouts scheduled once, divide-free cost model) | combination, planner-prologue, pack-thread-sfpu, sfpu-fpu-overlap, runtime-schedule, kernel-only | 1.5887 | +0.0190 | ok | 1. **Finish the prologue trim on kimi-k3 (17 µs -> ~6 µs, ~+0.3% geomean via one shape; kernel-only).** |
| `r03-b02-a02` | r03-b02-a01 | keep the row-shared x owner off weight-sender cores: gate/up block j of a row is read, tilized and multicast by the row's non-weight-sender columns in turn, so no weight-sender NCRISC carries x stick reads (under its gate read barrier), the tilize wait or the x multicast | data-movement, x-path, sender-placement, row-shared-tilize, critical-path, kernel-only, kernel-size, planner-opt-level | 1.5952 | +0.0065 | ok | 1. **Use the headroom (this lineage was at the size limit):** port r02-b04-a04's SiLU body (`re_act::silu_act`) to PACK for dsv3 / glm / kimi-k2.7 / mistral. RE-ACT is still ~5 µs per expert there (the FPU multiply pass plus the last subblock's SiLU). |
| `r03-b02-a03` | r03-b02-a02 | measurement node: profiler-only Tensix hardware perf counters (FPU/SFPU busy, unpacker busy, SrcA/SrcB write and valid-wait, per-thread stalls) and FW matmul-throttle samples on the MATH thread, accumulated per gate/up and down window, to find what sets the ~30-40 cycles per tile-mult on the compute-bound shapes | measurement, perf-counters, fw-throttle, compute, profiler-only, kernel-only | 1.6103 | +0.0151 | ok | 1. **Stream x instead of the weights in the non-last gate/up K-blocks.** This is the direct lever on the measured bottleneck: up to ~40% off compute-bound GU (mistral, minimax-m3, gpt-oss, dsv3/glm phase A). |
| `r03-b02-a04` | r03-b02-a03 | column-streaming matmul groups: at per_core_M >= 3 the non-last gate/up K-blocks and the down K-loop hold each bf4 weight tile in SrcA and stream the per_core_M x/activated tiles through SrcB (LLK ct < rt mode) instead of unpacking a weight tile per tile-mult; all matmul calls go through out-of-line helpers so the binaries shrink | compute, matmul-operand-streaming, unpack-bound, llk-reuse-mode, kernel-size, kernel-only, bit-identical | 1.5717 | -0.0386 | ok | 1. **Start from the parent's fused_swiglu.cpp** (r03-b02-a03), or set `kCsMinRows` > 8 here to disable CS. |

### Branch b03

| node | parent | mechanism | tags | score | Δ parent | fail_class | next |
|---|---|---|---|---|---|---|---|
| `r03-b03-a01` | root | cheap launch planner: each RISC caches the experts' tile counts once and plans on that array with division-free costs and one uniform pass (identical plan), so the first DRAM read / matmul starts earlier | prologue, planner, runtime-schedule, kernel-only, measurement | 1.5751 | +0.0054 | ok | 1. **Combine with the PACK-thread activation pipeline** (r03-b01-a01 / r03-b02-a01 are porting |
| `r03-b03-a02` | r03-b03-a01 | unblock the weight-sender row's gate/up DRAM stream: the row-major x owner multicasts while its weight reads are in flight, and the reader reads gate+up so the writer only multicasts (block k+1's reads overlap block k's multicast); plus profiler-only DM wait counters | data-movement, pipelining, x-path, weight-delivery, sender-row, measurement, host+kernel | 1.4983 | -0.0768 | ok | 1. **Start from the parent's code (r03-b03-a01), or revert this node's factory `up_mode` change.** That |
| `r03-b03-a03` | r03-b03-a02 | plain SiLU runs as a fused binary SiLU-GLU SFPU op on PACK (silu(gate)*up in one pass, pipelined into the last gate/up K-block), dropping the separate FPU multiply pass; code base = best node r03-b02-a02 (reverts parent's all-NoC-0 reads) | compute, sfpu, activation, pack-thread-sfpu, silu, rebase-onto-best, kernel-only | 0.0000 | - | accuracy_fail | 1. **Don't spend more on the SiLU multiply pass for speed alone.** Even made accuracy-neutral, it is |
| `r03-b03-a04` | r03-b03-a03 | reuse-B matmul at per_core_M >= 3: hold the bf4 weight tile in SrcA and stream x / activated rows through SrcB (rt = per_core_M, ct = 2) in the non-last gate/up K-blocks and the down K-blocks; all matmuls via one out-of-line helper (code base = best node r03-b02-a03) | compute, matmul-reuse, unpack-bound, subblock-shape, kernel-only, rebase-onto-best, kernel-size | 1.5554 | - | ok | 1. **Do not retry operand-reuse / subblock-shape changes** (this node, r01-b02-a04). Unpack volume |

### Branch b04

| node | parent | mechanism | tags | score | Δ parent | fail_class | next |
|---|---|---|---|---|---|---|---|
| `r03-b04-a01` | root | bit-identical launch-planner prologue trim: plan from a once-read tile-count byte cache and a largest-first order (no per-call L1 re-reads, no divides for single-chunk costs, uniform layouts scheduled once, one-pass phase-A model, infeasible mixed layouts pruned), and overlap the start-page DRAM read with planning | prologue, planner, runtime-schedule, kernel-only, bit-identical, measurement | 1.5813 | +0.0116 | ok | 1. **Stack it with the round-3 PACK-thread activation pipeline combination** (r03-b01-a01 / r03-b02-a01 were porting r02-b03-a04's fused_swiglu.cpp onto this root). The two touch different things: this node changes `adaptive_chunk.hpp`, two lines each in reader/writer, and a zone in fused_swiglu's UNPACK block. Both gains should add: about +0.7% from this node on top of theirs. |
| `r03-b04-a02` | r03-b04-a01 | take the row-shared x look-ahead DRAM read off the gate/up weight path: x reads get their own NoC transaction id so the gate sender's weight barrier and each reader's per-block weight push no longer wait for them, and the x block is pushed as soon as it lands | data-movement, x-path, noc-trid, barrier-scope, row-major-path, kernel-only, measurement | 1.5761 | -0.0052 | ok | 1. **Start from r03-b02-a01 (best, 1.5887), not from here.** This node's reader change is harmless but buys nothing. |
| `r03-b04-a03` | r03-b04-a02 | hide the gate/up activation behind two K-blocks of matmul: on r03-b02-a02's code, the last two gate/up K-blocks are walked subblock by subblock and each subblock's PACK-thread SFPU activation is split over both blocks' matmul acquires | compute, sfpu-fpu-overlap, pack-thread-sfpu, activation, pipelining, kernel-only, bit-identical, rebase-onto-best | 1.6010 | +0.0249 | ok | 1. **Stop spending attempts on compute-side activation hiding** (this node, r02-b01-a03 and its ports). On 8 of 9 |
| `r03-b04-a04` | r03-b04-a03 | down-weight sender issues block kb+1's DRAM read (both DOWN_SPLIT halves) right after multicasting block kb, into its own already-free ring slot, instead of after all column receivers ack kb+1 | data-movement, down-phase, pipelining, ack-decoupling, kernel-only, bit-identical | 1.6078 | +0.0068 | ok | 1. **Find out why the 4-expert shapes didn't move. One profiler-only counter first.** |

## Round r04

Policy `v1` (beta 0.6), W=4, R=4, root `dream/routed-expert-prefill/n/r03-b02-a03`. Plan: campaign defaults: every recorded round ran 4 x 4 and the next round re-roots on the best node

### Branch b01 (closed: head 1.5915 is more than 1.1x noise below the round's best 1.6275)

| node | parent | mechanism | tags | score | Δ parent | fail_class | next |
|---|---|---|---|---|---|---|---|
| `r04-b01-a01` | root | op-local matmul unpack MOP that advances the streamed SrcA tile address with one CFGSHIFTMASK per tile instead of the stock RDCFG / ADDDMAREG / STALLWAIT / WRCFG chain | compute, llk-unpack-mop, cfgshiftmask, unpack-bound, op-local-llk, kernel-only, bit-identical | 1.6011 | -0.0092 | ok | 1. **Measure the bank hand-off before touching the LLK again.** |
| `r04-b01-a02` | r04-b01-a01 | activated stream one block ahead in the down phase: receivers ack activated block kb+1 as soon as their own cb_in0_down_full slot for it is free and its sender multicasts it while the row still waits for block kb (two parity valid flags), on r04-b02-a01's down weight look-ahead | data-movement, down-phase, activated-stream, handshake, lookahead, port, kernel-size, kernel-only, bit-identical | 1.5915 | -0.0096 | ok | 1. **Don't build on this node. Start from r04-b02-a01** (best, 1.6275). If a child wants this node's size savings |

### Branch b02 (closed: head 1.6093 is more than 1.1x noise below the round's best 1.6275)

| node | parent | mechanism | tags | score | Δ parent | fail_class | next |
|---|---|---|---|---|---|---|---|
| `r04-b02-a01` | root | weight senders start block kb+1's DRAM reads as soon as their own ring slot is free in both phases: gate/up (reader gate read + writer up_go) no longer waits for the sender row's x block kb, and r03-b04-a04's down look-ahead is ported with a re-check while the sender spins on act_valid | data-movement, weight-delivery, lookahead, decoupling, lock-step, port, kernel-only, bit-identical | 1.6275 | +0.0172 | ok | 1. **Measure whether the gate/up look-ahead fires** before keeping or extending it. Add a profiler-only counter on |
| `r04-b02-a02` | r04-b02-a01 | the phase-4 activated row stream runs on its own: a non-blocking pump (spun in every down-loop wait) lets receivers ack activated block a+1 as soon as a is in and a slot is free and senders multicast as soon as acked, with a running-count act_valid, instead of one block per slowest column's down-weight arrival; profiler-only MATH counters made opt-in for kernel-config headroom | data-movement, down-phase, activated-stream, decoupling, handshake, kernel-only, bit-identical, kernel-size | 1.6093 | -0.0182 | ok | 1. **Make the act send non-blocking and keep it out of the weight-critical window** (same file, small change). |

### Branch b03 (closed: head 1.6066 is more than 1.1x noise below the round's best 1.6275)

| node | parent | mechanism | tags | score | Δ parent | fail_class | next |
|---|---|---|---|---|---|---|---|
| `r04-b03-a01` | root | measurement node: an in-situ, data-independent gate/up matmul probe at kernel start (perf-counter window on MATH) plus UNPACK-thread wall-clock tallies of every input-CB wait (x, gate, up, activated, down) and the owner's x tilize, to decide whether the ~31 cycles per tile-mult is the matmul pipeline or the compute waiting for data | measurement, profiler-only, compute, in-situ-probe, cb-wait, kernel-only | 1.5945 | -0.0158 | ok | 1. **Decouple the row streams; this is where the time is.** Expected value: up to the wait shares above (25-58% of |
| `r04-b03-a02` | r04-b03-a01 | decouple the down phase's row (activated) and column (down-weight) streams in the reader: one non-blocking polling loop where each stream acks its next block as soon as its own previous block landed and its slot is free, the act sender multicasts as soon as its row acks are in, and the in1 sender's DRAM read is polled instead of barriered; parent's probe compiled out | data-movement, down-phase, handshake, stream-decoupling, polling-loop, kernel-only, bit-identical | 1.6119 | +0.0174 | ok | 1. **Decouple only the activated stream.** Keep the polling loop's act half, which is where the k3 / dsv4-flash / mistral win comes from. Put the down-weight receivers' ack back behind the activated arrival: ack w(kb+1) only after act(kb) is pushed, as the lock-step loop did. That stops the down reads from running ahead into the gate/up groups' DRAM share. This is a one-line condition in the weight branch (`w_ack == w_rx && act_rx > w_rx`). Expect it to keep most of the k3 / dsv4-flash gain and to remove the k2.7 / dsv4-pro loss. |
| `r04-b03-a03` | r04-b03-a02 | put the idle rows to work on a tie: when the best uniform row-group layout leaves groups idle, the planner takes a mixed-height layout with one expert per group, the same modelled makespan and a lower summed expert cost, so mistral / minimax-m3 run 3+3+2+2 rows (6-tile experts at per_core_M 2) instead of 2x5 with rows y10-11 idle | planner, runtime-schedule, row-groups, idle-rows, tail, mixed-height, kernel-only, bit-identical | 1.6066 | -0.0053 | ok | 1. **Keep the tie pass, but find the size somewhere other than the UNPACK planner.** |

### Branch b04 (closed: head 1.6037 is more than 1.1x noise below the round's best 1.6275)

| node | parent | mechanism | tags | score | Δ parent | fail_class | next |
|---|---|---|---|---|---|---|---|
| `r04-b04-a01` | root | decouple the gate/up x and weight handshakes on the receivers: each stream's next-block ack goes out as soon as that stream's own block lands (and its next slot is free), not after both streams' blocks land | data-movement, handshake, stream-decoupling, x-path, weight-delivery, kernel-only, bit-identical | 1.6002 | -0.0101 | ok | 1. **Start from the root's reader (r03-b02-a03) unless you need nothing else in the reader.** This node |
| `r04-b04-a02` | r04-b04-a01 | activated-stream run-ahead in the down phase: row cores grant an activated ring slot (ack its sender) as soon as it is free and senders multicast as soon as all grants are in, with per-slot valid flags and an activated ring of up to 4 slots from leftover L1 (on r04-b02-a01's reader) | data-movement, down-phase, activated-stream, run-ahead, handshake, cb-depth, host+kernel, bit-identical | 1.6037 | +0.0035 | ok | 1. **Don't spend more attempts on DM handshakes or ack timing in either phase.** |

## Round r05

Policy `v2` (beta 0.6), W=4, R=1, root `dream/routed-expert-prefill/n/r04-b02-a01`. Plan: 4 wide, 1 step(s) deep: in r02-r04 step 1 set the round's largest gain and steps 2-4 never paid for their attempts; the next round re-roots on the best node and runs W wide from it

### Branch b01

| node | parent | mechanism | tags | score | Δ parent | fail_class | next |
|---|---|---|---|---|---|---|---|
| `r05-b01-a01` | root | plan once per core: the reader publishes the launch plan in a one-page L1 CB for the writer and compute (two planner copies removed, start-page read overlapped with planning), and the freed kernel-config space carries r04-b03-a03's idle-row tie pass so minimax-m3 / mistral run 3+3+2+2 rows instead of 2x5 with two rows idle | planner, plan-sharing, kernel-size, row-groups, idle-rows, tie-pass, host+kernel, bit-identical, port, combination | 1.6457 | +0.0182 | ok | 1. **Spend the headroom.** Nodes that overflowed or had to drop pieces on this lineage can now fit. Candidates: |

### Branch b02

| node | parent | mechanism | tags | score | Δ parent | fail_class | next |
|---|---|---|---|---|---|---|---|
| `r05-b02-a01` | root | rebalance the weight DRAM reads ~60/40 toward NoC 0: the reader also reads the first ~0.2w up K-rows of each gate/up block and the leading ~60% tiles of every down K-row (tile split instead of row split), so the hot NoC-1 response links in the DRAM columns carry ~40% instead of 50% | data-movement, noc-balance, dram-bandwidth, weight-delivery, noc-congestion, host+kernel, bit-identical | 1.6667 | +0.0392 | ok | 1. **Rebalance the plan for the new row speeds (largest, cheapest next gain).** The row groups no longer run at equal |

### Branch b03

| node | parent | mechanism | tags | score | Δ parent | fail_class | next |
|---|---|---|---|---|---|---|---|
| `r05-b03-a01` | root | stagger the row-shared x owner rotation by M-row (group index + 2 x row-in-group), so a gate/up K-block's x stick reads, tilize and row multicast are spread over ~10 grid columns instead of landing on the same 1-2 columns in every row | data-movement, x-path, noc-hotspot, owner-placement, row-shared-tilize, kernel-only, bit-identical, measurement | 1.6177 | -0.0098 | ok | 1. **Pin down the row gradient of the first block's reads. It is the biggest structured waste left on the |

### Branch b04

| node | parent | mechanism | tags | score | Δ parent | fail_class | next |
|---|---|---|---|---|---|---|---|
| `r05-b04-a01` | root | torus-aware weight-sender rows: every group column's gate/up sender is the group's bottom row and its down sender the top row, so the writer's NoC-1 (-y) and the reader's NoC-0 (+y) column multicasts start at their rectangle corner instead of wrapping around the whole column of the torus | data-movement, noc-topology, multicast-geometry, sender-placement, torus-wrap, host+kernel, bit-identical | 0.0000 | - | hang | 1. **Split the change to find the hang.** Do it in this order, and treat each step as the repair candidate. |

## Reading a node in full

```bash
git show dream/routed-expert-prefill/n/<id>:agent_orch/campaigns/routed-expert-prefill/attempts/<id>/proposal.md
git show dream/routed-expert-prefill/n/<id>:agent_orch/campaigns/routed-expert-prefill/attempts/<id>/reflection.md
git show dream/routed-expert-prefill/n/<id>:agent_orch/campaigns/routed-expert-prefill/attempts/<id>/eval/summary.md
git diff dream/routed-expert-prefill/n/<id>~1 dream/routed-expert-prefill/n/<id> -- . ':!agent_orch'
```
