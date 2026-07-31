# Regime-A fused all-gather matmul: design specification

## Goal and contract

Build a new multi-device TTNN op that consumes `in0[M, K/TP]`, K-sharded over an **even** tensor-parallel
group (`TP=4` or `8` normally), and a device-local `in1[K, N]`, and produces the replicated result
`all_gather(in0, dim=-1) @ in1`. It must preserve the accepted single-chip Regime-A choices—`Ns`, `Pk`,
`Sm`, core placement, local in0 ring, in1 DRAM-sharded readers, reduction, tails, precision, and fusions.
The objective is to hide fabric all-gather behind in1 DRAM reads and matmul, not to perform a complete
all-gather followed by matmul.

The op must support both ring and linear device topologies through the established CCL/fabric APIs and
**mux v2**. Do not add multi-device branches to `regime_a_matmul`; implement a separate op, initially named
`ttnn.experimental.all_gather_regime_a_matmul_async`.

## Device-level schedule

Use a balanced bidirectional all-gather. For the ring, send each transport chunk in both directions. With
even `TP`, a device receives work in waves:

```text
wave 0 (local):       K/TP
waves 1..TP/2-1:    2K/TP  (one full shard from each direction)
final antipode wave: K/TP   (half of the antipode payload on each direction)
```

The linear topology uses the repository's existing balanced bidirectional line schedule, with the same
local-first and progressive-consumption rules. Never process a duplicate shard, and never wait for the
whole gathered tensor. A received block is made compute-visible as soon as its payload is safe; it is then
forwarded if another device still needs it.

Only the **first N traversal** forwards A over fabric. If a worker owns several N sub-blocks, later
traversals reuse resident A and must not resend it. Assign global K blocks to the existing `Pk` groups so
each availability wave gives useful work to as many groups as possible; a purely contiguous assignment can
leave most `Pk` groups idle during fabric startup. The in1 reader must use the identical global-K order.

## On-chip dataflow and ownership

Keep the current local pipeline:

```text
fabric/local A arrival -> fixed L1 slot -> local in0 writer publishes CB0
                       -> eight-core in0 ring -> progressive matmul
in1 DRAM banks         -> bank-adjacent in1 readers -> matmul
matmul partials        -> existing Pk reduction -> fusion once -> output
```

CB0 already has capacity for the worker's complete `K/Pk` slice and remains resident across N sub-blocks.
The in0 writer must remain CB0's sole producer: fabric workers write fixed L1 staging slots and signal
readiness; the writer performs `cb_push_back`. Use per-slot epochs/credits rather than one ambiguous
cumulative counter, because bidirectional arrivals can be out of order and the antipode is split.

An A transport chunk crosses fabric **once per destination device**. `Ns` groups need identical A; they may
share forwarding work, but must not emit duplicate fabric copies. After ingress, distribute the payload to
all local `Ns` consumers over NoC. `Sm` groups own distinct M rows and may receive distinct slices. Do not
put fabric work on the critical in1-reader RISC. Compare one designated injector per direction against
striping chunks across eligible in0-ring cores/`Ns` groups; choose by measurement, not symmetry.

## Two implementation phases

1. **DRAM-staged reference.** Read local A once for both local compute and fabric egress. Remote ingress
   writes transport chunks to a deterministic DRAM staging region, then signals readiness; local readers
   wait per chunk and read it into the normal in0 path. This is the simplest correctness and overlap
   baseline, but its extra fabric-to-DRAM write plus DRAM-to-L1 read may compete with in1 and cap performance.
2. **Direct-L1 streaming.** Fabric ingress writes bounded L1 slots, signals payload readiness, and receives a
   credit only after all local consumers release a slot. Partition/seed the eight local ring stripes without
   rereading A from DRAM. Size the window from authoritative L1 accounting; do not assume the whole gathered
   A fits. Retain DRAM staging as an A/B diagnostic until direct L1 is proven correct and faster.

Transport granularity is independent of compute `kb`: transfer `C*kb` blocks to amortize fabric overhead,
but publish contained `kb` blocks progressively. Optimize for the default **4 KiB fabric packet**. If fabric
is measured critical, repeat the same configuration at **8 KiB** and report the difference. Signal once per
transport chunk/slot, not once per packet; payload must precede readiness, and source reuse must wait for the
appropriate flush/credit. Drain writes and non-posted atomics before kernel exit.

## Correctness, fusion, and performance gates

- Exact tile ownership: every global K tile is consumed once on every device; ring/line, both directions,
  tails, split antipode, cache replay, and fresh semaphore sets must be tested.
- Preserve BF16/compute fidelity and all supported epilogues: bias, activation, addcmul, and output chunking.
  Apply epilogues only after the complete global-K result has reached the existing local reduction endpoint.
- Test `TP=4` and `8`, narrow and wide N, `Pk>1`, `Ns>1`, `Sm>1`, multiple N sub-blocks, and non-divisible
  logical tails. Use watcher and per-RISC profiling; host dispatch time is not evidence of overlap.
- Record four baselines with identical shapes/configs: single-chip full-K matmul, standalone all-gather,
  unfused all-gather+matmul, and fused AGMM. Report all relaunches and
  `overlap_efficiency = max(T_mm, T_ag) / T_fused`; also report fabric and DRAM bytes.
- Galaxy RevB DRAM peak is **448 GB/s**, not the p150 value of 512 GB/s. All bandwidth percentages on this
  machine must use 448 GB/s and identify the board revision. Phase A is successful when correct and visibly
  overlapped; the production target is direct-L1 fused time within 10% of `max(T_mm, T_ag)` on DRAM-bound
  goldens, with no material regression versus unfused execution.

Start by reading the production `regime_a_matmul`,
`experimental/ccl/all_gather_minimal_matmul_async`, and current mux-v2 unit tests. Reuse fabric connection,
packet, teardown, and flow-control primitives; do not invent a private mux protocol.
