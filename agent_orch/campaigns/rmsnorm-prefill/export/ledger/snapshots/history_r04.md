# rmsnorm-prefill: discovery history

Generated 2026-10-08T16:44 from git tags `dream/rmsnorm-prefill/n/*` and the ledger.

**Baseline** (µs, per-chip mean device kernel time; noise ±1.0%): kimi-k3-latent-moe-h3584 16.99, deepseek-v4-flash-h4096 18.26, glm-5-3-h6144 23.48, kimi-k2-7-h7168 26.03
**Best valid:** `r04-b01-a03` score 1.4475 (stack r04-b04-a02's writer (posted output drain + ack-free stick push) onto the HiFi2 PRE node r04-b01-a02)
**Attempts:** 47 committed (47 valid) over 4 round(s); 0 lost.

## Leaderboard (top 10 valid)

| node | score | mechanism | kimi-k3-latent-moe-h3584 µs | deepseek-v4-flash-h4096 µs | glm-5-3-h6144 µs | kimi-k2-7-h7168 µs |
|---|---|---|---|---|---|---|
| `r04-b01-a03` | 1.4475 | stack r04-b04-a02's writer (posted output drain + ack-free stick push) onto the HiFi2 PRE node r04-b01-a02 | 11.67 | 13.05 | 16.34 | 17.35 |
| `r04-b02-a03` | 1.4216 | port the round-4 stack (posted drain + ack-free stick push + streamed gamma from r04-b04-a02, PRE at HiFi2 from r04-b01-a02) onto the streamed-multicast AG release | 11.74 | 13.35 | 16.78 | 17.65 |
| `r04-b04-a02` | 1.3997 | stack round 4's two other writer-only wins on the posted drain: ack-free stick push (r04-b03-a01) + streamed gamma chunks (r04-b01-a01) | 11.96 | 13.62 | 16.91 | 17.93 |
| `r04-b04-a03` | 1.3949 | position-aware input-read depth: workers left of the x=9 DRAM column keep 6 trid blocks (24 tiles) of input reads in flight instead of 4, so the slow side's read and stick push stop gating F_COLLECT | 12.07 | 13.46 | 17.06 | 18.08 |
| `r04-b01-a02` | 1.3911 | PRE at HiFi2: the per-tile x*x ELWMUL and the ones*S^T row-sum matmul run at MathFidelity::HiFi2 instead of HiFi4; POST and the post-AG combine stay HiFi4 | 12.04 | 13.51 | 17.13 | 18.16 |
| `r04-b03-a02` | 1.3906 | stack r04-b04-a01's posted (no-ack) output-drain writes onto this node's ack-free stick-push handshake: AG-start push cut + drain per-tile ack cut together | 12.25 | 13.42 | 16.98 | 18.15 |
| `r04-b03-a03` | 1.3894 | de-phase DRAM banks per worker on the 20-core pull path: CB slot k <-> column (k+rot) mod W with rot putting each worker's first page in bank tile_row % 8, applied to the input read, the BRISC gamma read and the posted output drain (compute untouched) | 11.99 | 13.31 | 17.05 | 18.70 |
| `r04-b04-a01` | 1.3666 | posted (no-ack) output-drain writes: BRISC issues each 2 KB output tile with NocOptions::POSTED and flushes posted-sent before each pop, instead of non-posted writes | 12.31 | 13.64 | 17.35 | 18.64 |
| `r04-b03-a01` | 1.3543 | stick-push handshake without the write-ack round trip: flush the two 64 B stick writes, then fire the arrival inc on the same NoC/VC (in-order delivery), atomic barrier deferred to kernel end | 12.33 | 13.76 | 17.54 | 18.94 |
| `r04-b01-a01` | 1.3436 | port r03-b04-a03's streamed gamma (8-page sticky-trid chunks pushed 2 chunks behind the issue front) onto the best node r03-b02-a02, removing its dev-0 h7168 cross-call straggler | 12.43 | 14.12 | 17.73 | 18.68 |

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

## Round r04

Policy `v3` (beta 0.6), W=4, R=3, root `dream/rmsnorm-prefill/n/r03-b02-a02`. Plan: R=3 (beta 0.6): recorded 4th attempts never lifted a round best by more than noise; the next round re-roots at this round's best, so depth continues there (3 earlier round(s))

### Branch b01

| node | parent | mechanism | tags | score | Δ parent | fail_class | next |
|---|---|---|---|---|---|---|---|
| `r04-b01-a01` | root | port r03-b04-a03's streamed gamma (8-page sticky-trid chunks pushed 2 chunks behind the issue front) onto the best node r03-b02-a02, removing its dev-0 h7168 cross-call straggler | gamma-read, pipelining, writer, straggler, combination, port | 1.3436 | +0.0103 | ok | With the straggler gone, every core follows the same timeline, so per-core fixed costs now map straight to kernel time: |
| `r04-b01-a02` | r04-b01-a01 | PRE at HiFi2: the per-tile x*x ELWMUL and the ones*S^T row-sum matmul run at MathFidelity::HiFi2 instead of HiFi4; POST and the post-AG combine stay HiFi4 | compute, pre-phase, math-fidelity, ag-start, accuracy-tradeoff | 1.3911 | +0.0475 | ok | 1. **Stack the two writer-only round-4 wins on this node:** r04-b04-a01's posted output-drain writes (-0.2..-0.7 µs, drain |
| `r04-b01-a03` | r04-b01-a02 | stack r04-b04-a02's writer (posted output drain + ack-free stick push) onto the HiFi2 PRE node r04-b01-a02 | combination, port, writer, posted-writes, ag-start, math-fidelity | 1.4475 | +0.0564 | ok | Where h7168's 16.7 µs goes now (max over cores): the input read ends at 6.4, the PRE tail is 0.5, the push 0.25, the |

### Branch b02

| node | parent | mechanism | tags | score | Δ parent | fail_class | next |
|---|---|---|---|---|---|---|---|
| `r04-b02-a01` | root | forwarder-push AG release: stats scratch height-sharded on the forwarder core, forwarder multicasts the gathered pages + go to the worker rectangle and compute unpacks each worker's stick in place (64 B-page gathered CB), replacing 20 serial go incs + 8 post-go remote stick reads | ag-release, forwarder, multicast, stick-read, l1-scratch, compute-indexing | 1.2988 | -0.0345 | ok | 1. **Overlap the multicasts with the cross-chip skew instead of serialising them after the last arrival.** F_FABRIC's |
| `r04-b02-a02` | r04-b02-a01 | streamed forwarder multicast: peers' fused incs carry a per-source-device bit field in out_ready, so the forwarder multicasts each peer's gathered page to the worker rectangle as soon as it lands; only the last page + go remain after the final arrival | ag-release, forwarder, multicast, overlap, cross-chip-skew, semaphore-fields | 1.3175 | +0.0187 | ok | 1. **Port r04-b04-a01's posted drain writes onto this node** (writer W_DRAIN only, orthogonal). With posted writes the |
| `r04-b02-a03` | r04-b02-a02 | port the round-4 stack (posted drain + ack-free stick push + streamed gamma from r04-b04-a02, PRE at HiFi2 from r04-b01-a02) onto the streamed-multicast AG release | combination, port, ag-release, multicast, posted-writes, math-fidelity, gamma-read | 1.4216 | +0.1041 | ok | 1. **Stagger the drain starts deliberately, but keep the early go.** This is the only way this lineage's ~0.3 µs |

### Branch b03 (closed: stalled: last 1 refinement(s) within 1.0% noise of the anchor (-0.09%))

| node | parent | mechanism | tags | score | Δ parent | fail_class | next |
|---|---|---|---|---|---|---|---|
| `r04-b03-a01` | root | stick-push handshake without the write-ack round trip: flush the two 64 B stick writes, then fire the arrival inc on the same NoC/VC (in-order delivery), atomic barrier deferred to kernel end | ag-start, writer, noc-ordering, handshake, latency | 1.3543 | +0.0210 | ok | 1. **Port r03-b04-a03's streamed gamma (8-page chunks, sticky trid) onto this node.** It is writer-only and orthogonal. |
| `r04-b03-a02` | r04-b03-a01 | stack r04-b04-a01's posted (no-ack) output-drain writes onto this node's ack-free stick-push handshake: AG-start push cut + drain per-tile ack cut together | combination, writer-drain, posted-writes, ag-start, writer, port | 1.3906 | +0.0363 | ok | 1. **Port r03-b04-a03 / r04-b01-a01's streamed gamma** (8-page sticky-trid chunks; writer W_GAMMA block only). It is |
| `r04-b03-a03` | r04-b03-a02 | de-phase DRAM banks per worker on the 20-core pull path: CB slot k <-> column (k+rot) mod W with rot putting each worker's first page in bank tile_row % 8, applied to the input read, the BRISC gamma read and the posted output drain (compute untouched) | dataflow, dram-banks, access-order, reader, writer-drain, port | 1.3894 | -0.0012 | ok | 1. **Revert this change.** Continue from the parent r04-b03-a02 or, better, from the stacked best (r04-b02-a03, |

### Branch b04 (closed: stalled: last 1 refinement(s) within 1.0% noise of the anchor (-0.34%))

| node | parent | mechanism | tags | score | Δ parent | fail_class | next |
|---|---|---|---|---|---|---|---|
| `r04-b04-a01` | root | posted (no-ack) output-drain writes: BRISC issues each 2 KB output tile with NocOptions::POSTED and flushes posted-sent before each pop, instead of non-posted writes | writer-drain, posted-writes, noc, per-core-bound, dataflow | 1.3666 | +0.0333 | ok | 1. **Stack with the gamma streaming of r03-b04-a03** (writer W_GAMMA block only, orthogonal to this drain change). It |
| `r04-b04-a02` | r04-b04-a01 | stack round 4's two other writer-only wins on the posted drain: ack-free stick push (r04-b03-a01) + streamed gamma chunks (r04-b01-a01) | combination, port, writer, ag-start, gamma-read, posted-writes | 1.3997 | +0.0331 | ok | 1. **Cross-device launch skew on h7168** (dev0 span 19.15 vs dev3 16.98 µs). The chip mean pays for dev0 waiting at |
| `r04-b04-a03` | r04-b04-a02 | position-aware input-read depth: workers left of the x=9 DRAM column keep 6 trid blocks (24 tiles) of input reads in flight instead of 4, so the slow side's read and stick push stop gating F_COLLECT | reader, read-depth, load-balance, ag-start, noc-position | 1.3949 | -0.0048 | ok | 1. **Break the cross-call launch-skew loop. It is worth ~0.3-0.5 µs on whichever shapes are skewed in a run, and it is a |

## Reading a node in full

```bash
git show dream/rmsnorm-prefill/n/<id>:agent_orch/campaigns/rmsnorm-prefill/attempts/<id>/proposal.md
git show dream/rmsnorm-prefill/n/<id>:agent_orch/campaigns/rmsnorm-prefill/attempts/<id>/reflection.md
git show dream/rmsnorm-prefill/n/<id>:agent_orch/campaigns/rmsnorm-prefill/attempts/<id>/eval/summary.md
git diff dream/rmsnorm-prefill/n/<id>~1 dream/rmsnorm-prefill/n/<id> -- . ':!agent_orch'
```
