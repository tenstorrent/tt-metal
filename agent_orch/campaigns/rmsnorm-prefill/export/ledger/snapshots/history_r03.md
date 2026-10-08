# rmsnorm-prefill: discovery history

Generated 2026-10-08T15:47 from git tags `dream/rmsnorm-prefill/n/*` and the ledger.

**Baseline** (µs, per-chip mean device kernel time; noise ±1.0%): kimi-k3-latent-moe-h3584 16.99, deepseek-v4-flash-h4096 18.26, glm-5-3-h6144 23.48, kimi-k2-7-h7168 26.03
**Best valid:** `r03-b02-a02` score 1.3333 (land the all-gather stats scratch in L1 (mesh-coherent L1-interleaved persistent buffer) instead of DRAM: gathered-stick read after go 0.69 -> 0.60 us, fabric landing unchanged)
**Attempts:** 35 committed (35 valid) over 3 round(s); 0 lost.

## Leaderboard (top 10 valid)

| node | score | mechanism | kimi-k3-latent-moe-h3584 µs | deepseek-v4-flash-h4096 µs | glm-5-3-h6144 µs | kimi-k2-7-h7168 µs |
|---|---|---|---|---|---|---|
| `r03-b02-a02` | 1.3333 | land the all-gather stats scratch in L1 (mesh-coherent L1-interleaved persistent buffer) instead of DRAM: gathered-stick read after go 0.69 -> 0.60 us, fabric landing unchanged | 12.59 | 14.05 | 17.55 | 19.32 |
| `r03-b04-a03` | 1.3318 | stream gamma to compute in 8-page chunks (sticky-trid BRISC reads, bank-rotated within each chunk) so the x*gamma pre-pass starts at PRE end instead of after the last gamma page; parent's two-wave AG reverted to r03-b04-a01 | 12.77 | 14.20 | 17.75 | 18.72 |
| `r03-b01-a03` | 1.3255 | stick push no longer waits behind the gamma read barrier: BRISC polls for the row-0 stat while its broadcast-gamma reads land, instead of blocking in async_read_barrier | 12.54 | 14.17 | 17.77 | 19.45 |
| `r03-b02-a01` | 1.3231 | post-AG stat combine on row 0 only: one fused SFPU add_rsqrt (sum*1/H + eps, rsqrt) over the 4 SFPU iterations holding tile row 0, before transpose_dest, replaces three full-tile SFPU passes; POST unpack reconfig/init hoisted ahead of the 1/rms wait | 12.72 | 14.13 | 17.72 | 19.41 |
| `r03-b01-a02` | 1.3220 | land the all-gathered stat sticks in L1, not DRAM: the persistent stats scratch becomes an L1-interleaved mesh buffer, so the fabric writes and the worker post-go stick read are L1 transactions | 12.66 | 14.09 | 17.92 | 19.41 |
| `r03-b03-a01` | 1.3175 | post-AG stat chain on row 0 only: one fused SFPU add_rsqrt (x*1/H + eps, rsqrt) with VectorMode::R over 2 iterations/face before the fp32 transpose_dest, instead of three full-tile SFPU passes after it | 12.86 | 14.07 | 17.86 | 19.46 |
| `r03-b04-a01` | 1.3131 | post-AG stat finalize on row 0 only: *1/H, +eps and rsqrt as 2-iteration VectorMode::R SFPU passes before transpose_dest instead of full-tile passes after it | 12.89 | 14.20 | 17.83 | 19.53 |
| `r03-b01-a01` | 1.3129 | post-AG stat combine: run *1/H + eps + rsqrt on the SFPU over the stat row only (VectorMode::R, 2 iterations/face) before transpose_dest, instead of over the full 32x32 fp32 tile after it | 12.97 | 14.24 | 17.73 | 19.48 |
| `r03-b03-a02` | 1.3120 | PRE row stat straight out of the accumulating DST: fp32 transpose_dest + SFPU column-sum (sfpu_reduce<SUM,REDUCE_COL>) puts sum(x^2) per row into row 0 before the single pack, replacing the S pack -> L1 -> unpack -> ones*S^T matmul -> pack round trip | 12.90 | 14.26 | 17.83 | 19.51 |
| `r03-b03-a03` | 1.3023 | output drain round-robins its DRAM tile writes over the 4 unicast request VCs (0-3) instead of the single static VC 1, so a core keeps several 2 KB packets injecting at once | 12.70 | 14.57 | 18.25 | 19.51 |

## Round r01

Policy `v0` (beta 0.6), W=4, R=4, root `dream/rmsnorm-prefill/root`. Plan: v0: fixed campaign defaults

### Branch b01

| node | parent | mechanism | tags | score | Δ parent | fail_class | next |
|---|---|---|---|---|---|---|---|
| `r01-b01-a01` | root | apply gamma (x*w) during the all-gather wait so POST is a single x*rsqrt pass | compute, overlap, post-phase, reader-weight-batching | 1.1089 | +0.1089 | ok | Issue the broadcast gamma reads BEFORE (or interleaved with) the first input block: it is only 7 KB per core, |
| `r01-b01-a02` | r01-b01-a01 | pipeline the reader input row with per-block NoC transaction ids and interleave the gamma reads into the gaps | dataflow, reader, read-pipelining, transaction-id | 0.8994 | -0.2095 | ok | 1. Take the gamma read OFF the NCRISC input path entirely. Have the WRITER (BRISC, NoC0) issue the broadcast gamma |
| `r01-b01-a03` | r01-b01-a02 | read gamma on the idle writer RISC (BRISC/NoC0) at kernel start with per-worker rotated start tile, so the NCRISC trid input read carries no gamma traffic | dataflow, writer, gamma-read, dram-bank-spread, repair | 1.1868 | +0.2874 | ok | The remaining critical path, h7168, µs rel. kernel start: input 0->5-6, PRE tail -> stick 7-10 (spread 2.8 us; the AG |
| `r01-b01-a04` | r01-b01-a03 | accumulate sum(x^2) in fp32 DST via the ELWMUL dest-MAC under one acquire per row, so PRE packs once per row instead of an L1-acc fp32 pack per tile | compute, pre-phase, dst-accumulation, pack-reduction | 1.2042 | +0.0174 | ok | 1. **Measure before cutting PRE further.** Add a compute-side zone (DeviceZoneScopedN in the PRE loop, e.g. "C_PRE") |

### Branch b02

| node | parent | mechanism | tags | score | Δ parent | fail_class | next |
|---|---|---|---|---|---|---|---|
| `r01-b02-a01` | root | column-split each tile-row across multiple worker cores (more cores, same single AG round) | parallelism, work-decomposition, col-split | 1.0025 | +0.0025 | ok | Keep this node's plumbing (reader col_offset/row_stride RT args, writer col_offset, factory |
| `r01-b02-a02` | r01-b02-a01 | keep DRAM I/O deep: per-block transaction-ID input read and one flush per row in the output drain | dataflow, noc-depth, reader, writer-drain | 1.0338 | +0.0313 | ok | 1. **Stack with x*gamma (r01-b01-a01's compute change, score 1.109).** It is orthogonal: b01 shortens POST, this node |
| `r01-b02-a03` | r01-b02-a02 | stack x*gamma under the AG wait (r01-b01-a01) on top of the deep trid input read (r01-b02-a02) | compute, overlap, post-phase, combination, reader-weight-batching | 1.1721 | +0.1383 | ok | 1. **Column split on this lineage (port r01-b03-a02's k | num_tile_cols, 80-worker decomposition).** It is the third |
| `r01-b02-a04` | r01-b02-a03 | split the output drain across both NoCs: the idle reader (NCRISC/NoC1) writes the odd output blocks while the writer (BRISC/NoC0) writes the even ones | dataflow, writer-drain, dual-noc, reader | 0.9853 | -0.1868 | ok | 1. If you retry dual-NoC at all, make the share per core and position-aware: cores at physical x >= 10 (the right |

### Branch b03

| node | parent | mechanism | tags | score | Δ parent | fail_class | next |
|---|---|---|---|---|---|---|---|
| `r01-b03-a01` | root | Column-split each tile-row across k=floor(64/rows) worker cores; row leader FPU-sums follower partial sum-of-squares before the forwarder AG | parallelism, core-count, work-split, ag-protocol | 0.9867 | -0.0133 | ok | 1. Keep the column split but make dispatch cheap. Pick col_splits so slices are equal-width, |
| `r01-b03-a02` | r01-b03-a01 | Column split with equal-width slices only (k divides num_tile_cols, k=4 -> 81 cores): one kernel group, no dispatch stall | parallelism, work-split, dispatch, repair | 1.0942 | +0.1075 | ok | 1. **De-phase the DRAM bank access (likely the biggest lever, cheapest to try).** Give each core a start offset |
| `r01-b03-a03` | r01-b03-a02 | De-phase DRAM bank access: each column-split core walks its slice from a greedy per-core rotation so concurrent cores hit different interleaved banks | dataflow, dram-banks, access-order, reader, writer-drain | 1.1313 | +0.0371 | ok | 1. **Split the output drain across both NoCs.** NCRISC (reader, NoC1) is idle after its gamma read (ends well before the |
| `r01-b03-a04` | r01-b03-a03 | split the output drain across both NoCs: BRISC writes even CB positions on NoC0, the idle reader (NCRISC) writes odd positions on NoC1 | dataflow, writer-drain, noc-balance, reader, dual-noc | 1.1653 | +0.0340 | ok | 1. **Per-core NoC split ratio instead of 50/50.** The two gradients are mirror images, so give each core a split |

### Branch b04

| node | parent | mechanism | tags | score | Δ parent | fail_class | next |
|---|---|---|---|---|---|---|---|
| `r01-b04-a01` | root | precompute x*gamma during the all-gather wait so POST is a single x*gamma*(1/rms) pass | compute, overlap, post-phase | 1.0761 | +0.0761 | ok | 1. Make the output drain non-blocking: drop the per-block `async_writes_flushed` in |
| `r01-b04-a02` | r01-b04-a01 | trid-pipelined input read with gamma face-row reads interleaved per block | reader, dram-latency, noc-trid, overlap | 0.8397 | -0.2364 | ok | 1. Keep the trid-pipelined input pass (reader `read_input_pass_pipelined`) but call it with `with_weight=false` |
| `r01-b04-a03` | r01-b04-a02 | read broadcast gamma on the idle writer (BRISC) at kernel start; reader keeps only the trid-pipelined input | dataflow, writer, gamma-read, overlap, repair | 1.2075 | +0.3678 | ok | 1. **Port this lineage onto the column split (r01-b03-a02, 1.094: k=4, 80 workers, one kernel group).** The pieces |
| `r01-b04-a04` | r01-b04-a03 | accumulate sum(x^2) in DST across the row (ELWMUL accumulates) and pack once per row instead of an L1-acc pack per tile; stick push may preempt the BRISC gamma issue loop | compute, pre-phase, dst-accumulation, ag-start, writer | 1.2132 | +0.0057 | ok | 1. **Attack the fixed PRE tail, not per-tile cost.** Add TRISC zones (DeviceZoneScopedN around reduce, transpose, |

## Round r02

Policy `v1` (beta 0.6), W=4, R=4, root `dream/rmsnorm-prefill/n/r01-b04-a04`. Plan: campaign defaults (1 earlier round(s); no wider/deeper plan in support)

### Branch b01 (closed: anchor 1.1058 trails leader 1.2385 by more than 6.8%)

| node | parent | mechanism | tags | score | Δ parent | fail_class | next |
|---|---|---|---|---|---|---|---|
| `r02-b01-a01` | root | route each output tile on the NoC with the short (non-wrapping) x-path to its DRAM column: left-half cores send west-bank tiles on NoC1, right-half cores send east-bank tiles on NoC1, BRISC issues both | dataflow, writer-drain, dual-noc, noc-routing, dram-column | 1.1058 | -0.1074 | ok | 1. **Keep only the right-half rule** (one-line change in `want_noc1`: `!core_left_half && !dst_west`). Left-half |

### Branch b02

| node | parent | mechanism | tags | score | Δ parent | fail_class | next |
|---|---|---|---|---|---|---|---|
| `r02-b02-a01` | root | path-aware dual-NoC output drain: BRISC (DM_DYNAMIC_NOC) sends short-eastward DRAM destinations on NoC0 every other bank visit, the rest on its default NoC1 | dataflow, writer-drain, dual-noc, noc-routing, dynamic-noc | 1.2385 | +0.0253 | ok | 1. **Tune the NoC0 share.** This node sends 50% of the short-eastward destination visits on NoC0 (`(page/8)&1==0`). |
| `r02-b02-a02` | r02-b02-a01 | sum(x^2) per row as the diagonal of a DST-accumulated x*x^T matmul; BRISC gathers the 32 diagonal words into the stick, so the reduce + transpose leave the AG-start path | compute, pre-phase, ag-start, matmul, writer | 1.2292 | -0.0093 | ok | 1. **Get a row-0 stat out of the FPU with ONE op, so no gather is needed.** Keep the parent's DST-accumulated |
| `r02-b02-a03` | r02-b02-a02 | row-0 stat from ONE matmul: C = ones*S^T on the DST-accumulated S = sum x*x replaces reduce<SUM,ROW> + transpose; writer keeps its two 64 B face-row stick writes (no BRISC diagonal gather) | compute, pre-phase, ag-start, matmul, repair | 1.2396 | +0.0104 | ok | 1. **Go back to the parent's diagonal matmul (stat comes straight out of the accumulating DST, no S round trip, |
| `r02-b02-a04` | r02-b02-a03 | AG release by multicast: forwarder (forked into the op dir) sets the go-sem on each worker row-segment with one multicast instead of 20 serial unicast incs; workers pre-stage the gathered-stick CB reserve + NoC addresses before the go wait | ag-release, forwarder, multicast, writer, latency | 1.2346 | -0.0050 | ok | 1. **Cut the ~1.28 µs combine chain in compute (largest fixed post-AG cost).** Options, cheapest first: |

### Branch b03 (closed: anchor 1.0312 trails leader 1.2385 by more than 6.8%)

| node | parent | mechanism | tags | score | Δ parent | fail_class | next |
|---|---|---|---|---|---|---|---|
| `r02-b03-a01` | root | destination-aware dual-NoC output drain: each output tile goes on the NoC with the shorter torus path to its DRAM bank (BRISC/NoC0 + idle NCRISC/NoC1) | dataflow, writer-drain, dual-noc, noc-routing, reader | 1.0312 | -0.1820 | ok | 1. If dual-NoC is retried at all, use the only variant all three nodes support: **right-half cores only** put their x=9 |

### Branch b04 (closed: anchor 0.8470 trails leader 1.2385 by more than 6.8%)

| node | parent | mechanism | tags | score | Δ parent | fail_class | next |
|---|---|---|---|---|---|---|---|
| `r02-b04-a01` | root | spread the 20 AG-path worker cores over all 10 grid rows (2 full columns, column-major) instead of packing them row-major into 2 rows, to cut per-row NoC link congestion on the input read and output drain | placement, noc-congestion, dataflow, host-factory | 0.8470 | -0.3662 | ok | 1. **Don't retry column stacking.** If placement is revisited, use a diagonal or knight's-move layout (logical |

## Round r03

Policy `v2` (beta 0.6), W=4, R=3, root `dream/rmsnorm-prefill/n/r02-b02-a03`. Plan: R=3 (beta 0.6): recorded 4th attempts never lifted a round best by more than noise; the next round re-roots at this round's best, so depth continues there (2 earlier round(s))

### Branch b01

| node | parent | mechanism | tags | score | Δ parent | fail_class | next |
|---|---|---|---|---|---|---|---|
| `r03-b01-a01` | root | post-AG stat combine: run *1/H + eps + rsqrt on the SFPU over the stat row only (VectorMode::R, 2 iterations/face) before transpose_dest, instead of over the full 32x32 fp32 tile after it | compute, post-phase, stat-combine, sfpu, ag-release | 1.3129 | +0.0733 | ok | 1. **Shrink the rest of the combine (~0.4 µs).** Same trick on the remaining full-tile work: |
| `r03-b01-a02` | r03-b01-a01 | land the all-gathered stat sticks in L1, not DRAM: the persistent stats scratch becomes an L1-interleaved mesh buffer, so the fabric writes and the worker post-go stick read are L1 transactions | ag-end, stats-scratch, l1, latency, host-factory | 1.3220 | +0.0091 | ok | 1. **Port r03-b02-a01's combine (fused add_rsqrt on row 0 + POST unpack init hoisted before the 1/rms wait)** onto |
| `r03-b01-a03` | r03-b01-a02 | stick push no longer waits behind the gamma read barrier: BRISC polls for the row-0 stat while its broadcast-gamma reads land, instead of blocking in async_read_barrier | writer, ag-start, scheduling, gamma-read, latency | 1.3255 | +0.0035 | ok | 1. **Shorten the stick-push handshake itself (W_PUSH = 0.63 µs on every core, on the AG-start critical path).** Today: |

### Branch b02

| node | parent | mechanism | tags | score | Δ parent | fail_class | next |
|---|---|---|---|---|---|---|---|
| `r03-b02-a01` | root | post-AG stat combine on row 0 only: one fused SFPU add_rsqrt (sum*1/H + eps, rsqrt) over the 4 SFPU iterations holding tile row 0, before transpose_dest, replaces three full-tile SFPU passes; POST unpack reconfig/init hoisted ahead of the 1/rms wait | compute, post-phase, ag-end, sfpu, combine | 1.3231 | +0.0835 | ok | 1. **The remaining 0.38 µs combine gap.** What's left: 2 full-tile ELWADDs, transpose_dest<fp32> (cfg RMWs + MOVs), |
| `r03-b02-a02` | r03-b02-a01 | land the all-gather stats scratch in L1 (mesh-coherent L1-interleaved persistent buffer) instead of DRAM: gathered-stick read after go 0.69 -> 0.60 us, fabric landing unchanged | ag-landing, l1-scratch, stick-read, host-factory, latency | 1.3333 | +0.0102 | ok | 1. **Remove the worker-side gathered-stick read entirely.** The scratch is now in L1, so the forwarder can push the |
| `r03-b02-a03` | r03-b02-a02 | output drain keeps two writes in flight per NoC: BRISC alternates its write cmd buf 0 and its idle read/atomic cmd buf 1 (source coord set before, return coord restored after), same per-tile NoC routing as the parent | writer-drain, noc-cmd-buf, issue-pipelining, dataflow, per-core-bound | 1.3000 | -0.0333 | ok | 1. **Revert this drain change** (start from the parent r03-b02-a02 or drop the `two_cmd_bufs` path). Don't add more |

### Branch b03

| node | parent | mechanism | tags | score | Δ parent | fail_class | next |
|---|---|---|---|---|---|---|---|
| `r03-b03-a01` | root | post-AG stat chain on row 0 only: one fused SFPU add_rsqrt (x*1/H + eps, rsqrt) with VectorMode::R over 2 iterations/face before the fp32 transpose_dest, instead of three full-tile SFPU passes after it | compute, post-phase, ag-combine, sfpu, latency | 1.3175 | +0.0779 | ok | 1. **Cut the remaining 0.38 µs combine gap.** |
| `r03-b03-a02` | r03-b03-a01 | PRE row stat straight out of the accumulating DST: fp32 transpose_dest + SFPU column-sum (sfpu_reduce<SUM,REDUCE_COL>) puts sum(x^2) per row into row 0 before the single pack, replacing the S pack -> L1 -> unpack -> ones*S^T matmul -> pack round trip | compute, pre-phase, ag-start, sfpu, dst-resident | 1.3120 | -0.0055 | ok | 1. **Measure the PRE tail before cutting it again.** Add a MATH-thread (TRISC_1) zone around the last input block's |
| `r03-b03-a03` | r03-b03-a02 | output drain round-robins its DRAM tile writes over the 4 unicast request VCs (0-3) instead of the single static VC 1, so a core keeps several 2 KB packets injecting at once | dataflow, writer-drain, noc-vc, injection-concurrency | 1.3023 | -0.0097 | ok | 1. **Measure where a drain tile's ~147 cycles go before trying another drain mechanism.** In the writer, accumulate |

### Branch b04

| node | parent | mechanism | tags | score | Δ parent | fail_class | next |
|---|---|---|---|---|---|---|---|
| `r03-b04-a01` | root | post-AG stat finalize on row 0 only: *1/H, +eps and rsqrt as 2-iteration VectorMode::R SFPU passes before transpose_dest instead of full-tile passes after it | compute, post-ag, combine, sfpu, critical-path | 1.3131 | +0.0735 | ok | 1. **Squeeze the remaining 0.43 µs combine gap.** Options, cheapest first: |
| `r03-b04-a02` | r03-b04-a01 | two-wave AG pipeline: even workers read first, odd workers start as their neighbour's read lands; a forked forwarder gathers + releases each wave separately (16-bit per-wave sem fields), so wave A's AG overlaps wave B's read and B's AG overlaps A's drain | ag-pipeline, overlap, forwarder, dataflow, scheduling | 1.2440 | -0.0691 | ok | 1. **Revert to the parent's single wave** (or keep `use_waves=false`). Don't retry waves on 20 cores. |
| `r03-b04-a03` | r03-b04-a02 | stream gamma to compute in 8-page chunks (sticky-trid BRISC reads, bank-rotated within each chunk) so the x*gamma pre-pass starts at PRE end instead of after the last gamma page; parent's two-wave AG reverted to r03-b04-a01 | gamma-read, pipelining, writer, straggler, transaction-id, revert | 1.3318 | +0.0878 | ok | 1. **Port this gamma streaming onto the best node r03-b02-a02.** It is the cheapest likely new best. The change is |

## Reading a node in full

```bash
git show dream/rmsnorm-prefill/n/<id>:agent_orch/campaigns/rmsnorm-prefill/attempts/<id>/proposal.md
git show dream/rmsnorm-prefill/n/<id>:agent_orch/campaigns/rmsnorm-prefill/attempts/<id>/reflection.md
git show dream/rmsnorm-prefill/n/<id>:agent_orch/campaigns/rmsnorm-prefill/attempts/<id>/eval/summary.md
git diff dream/rmsnorm-prefill/n/<id>~1 dream/rmsnorm-prefill/n/<id> -- . ':!agent_orch'
```
