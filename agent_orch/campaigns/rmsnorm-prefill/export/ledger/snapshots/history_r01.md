# rmsnorm-prefill: discovery history

Generated 2026-10-08T13:46 from git tags `dream/rmsnorm-prefill/n/*` and the ledger.

**Baseline** (µs, per-chip mean device kernel time; noise ±1.0%): kimi-k3-latent-moe-h3584 16.99, deepseek-v4-flash-h4096 18.26, glm-5-3-h6144 23.48, kimi-k2-7-h7168 26.03
**Best valid:** `r01-b04-a04` score 1.2132 (accumulate sum(x^2) in DST across the row (ELWMUL accumulates) and pack once per row instead of an L1-acc pack per tile; stick push may preempt the BRISC gamma issue loop)
**Attempts:** 16 committed (16 valid) over 1 round(s); 0 lost.

## Leaderboard (top 10 valid)

| node | score | mechanism | kimi-k3-latent-moe-h3584 µs | deepseek-v4-flash-h4096 µs | glm-5-3-h6144 µs | kimi-k2-7-h7168 µs |
|---|---|---|---|---|---|---|
| `r01-b04-a04` | 1.2132 | accumulate sum(x^2) in DST across the row (ELWMUL accumulates) and pack once per row instead of an L1-acc pack per tile; stick push may preempt the BRISC gamma issue loop | 14.17 | 15.43 | 19.08 | 20.98 |
| `r01-b04-a03` | 1.2075 | read broadcast gamma on the idle writer (BRISC) at kernel start; reader keeps only the trid-pipelined input | 14.46 | 15.27 | 19.14 | 21.09 |
| `r01-b01-a04` | 1.2042 | accumulate sum(x^2) in fp32 DST via the ELWMUL dest-MAC under one acquire per row, so PRE packs once per row instead of an L1-acc fp32 pack per tile | 14.15 | 15.35 | 19.32 | 21.49 |
| `r01-b01-a03` | 1.1868 | read gamma on the idle writer RISC (BRISC/NoC0) at kernel start with per-worker rotated start tile, so the NCRISC trid input read carries no gamma traffic | 14.51 | 15.52 | 19.47 | 21.80 |
| `r01-b02-a03` | 1.1721 | stack x*gamma under the AG wait (r01-b01-a01) on top of the deep trid input read (r01-b02-a02) | 14.56 | 15.75 | 19.73 | 22.20 |
| `r01-b03-a04` | 1.1653 | split the output drain across both NoCs: BRISC writes even CB positions on NoC0, the idle reader (NCRISC) writes odd positions on NoC1 | 14.59 | 16.06 | 20.29 | 21.63 |
| `r01-b03-a03` | 1.1313 | De-phase DRAM bank access: each column-split core walks its slice from a greedy per-core rotation so concurrent cores hit different interleaved banks | 15.29 | 16.39 | 20.44 | 22.60 |
| `r01-b01-a01` | 1.1089 | apply gamma (x*w) during the all-gather wait so POST is a single x*rsqrt pass | 15.13 | 16.13 | 21.25 | 24.17 |
| `r01-b03-a02` | 1.0942 | Column split with equal-width slices only (k divides num_tile_cols, k=4 -> 81 cores): one kernel group, no dispatch stall | 15.38 | 18.05 | 20.57 | 23.15 |
| `r01-b04-a01` | 1.0761 | precompute x*gamma during the all-gather wait so POST is a single x*gamma*(1/rms) pass | 15.23 | 16.22 | 22.32 | 25.62 |

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

## Reading a node in full

```bash
git show dream/rmsnorm-prefill/n/<id>:agent_orch/campaigns/rmsnorm-prefill/attempts/<id>/proposal.md
git show dream/rmsnorm-prefill/n/<id>:agent_orch/campaigns/rmsnorm-prefill/attempts/<id>/reflection.md
git show dream/rmsnorm-prefill/n/<id>:agent_orch/campaigns/rmsnorm-prefill/attempts/<id>/eval/summary.md
git diff dream/rmsnorm-prefill/n/<id>~1 dream/rmsnorm-prefill/n/<id> -- . ':!agent_orch'
```
