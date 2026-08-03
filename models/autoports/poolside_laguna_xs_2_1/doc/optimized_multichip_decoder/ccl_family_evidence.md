# CCL family evidence — optimized multichip decoder

All measured on-device, 1×4 Blackhole p300c, `FABRIC_1D_RING`, `cluster_axis=1`, 2 links, traced +
warmed, payload `[1,1,32,2048]` bf16 (the decode residual all_reduce payload). Harnesses:
`/tmp/ccl_bench.py`, `/tmp/ccl_bench2.py`.

## Isolated collective costs (µs per single collective)

| candidate | µs | note |
|-----------|----|------|
| `ttnn.all_reduce` (composite, current) | **43.74** | lowers to ring reduce-scatter + all-gather |
| `ttnn.experimental.all_reduce_async` (deepseek form, DRAM out) | 43.73 | **no gain vs composite** |
| `ttnn.experimental.all_reduce_async` (L1 out) | 43.61 | no gain |
| `reduce_scatter` + `all_gather` (explicit split) | 43.81 | confirms all_reduce == RS+AG |
| `reduce_scatter` alone | **30.91** | AG ≈ 13 µs; RS ≈ 31 µs |

Conclusion: the residual all_reduce is **latency-bound** on this 4 KB payload (43.7 µs is fabric +
dispatch latency, not bandwidth). Async CCL gives nothing over the composite; both lower to the
same ring RS+AG primitives, and inside a trace the semaphore/setup is captured once regardless.

## Persistent buffers (OPT-009)

`all_reduce_async` with a preallocated persistent output buffer + `create_global_semaphore` was
exercised; inside a warmed trace the buffer/semaphore allocation is already amortized at capture,
so it does not beat the 43.7 µs latency floor. The 2 decode collectives/layer already run every
token inside the single decode trace with no per-call host allocation. No persistent-buffer win.

## Fused matmul-CCL (`llama_rs_matmul`, `all_gather_matmul_async`, `matmul_reduce_scatter_async`) — rejected with reason

These ops exist but (a) are Llama-Galaxy-shape-specialized (require persistent interim buffers,
worker sub-devices, and specific sharded layouts), and more fundamentally (b) **cannot reduce the
collective count for this model**: a fused matmul→reduce-scatter produces an H/4-sharded output,
but the very next consumers on this path — the residual add against the replicated hidden, the
full-hidden RMSNorm, and especially the **replicated MoE router** (full 256-wide sigmoid + top-8
selection needs the whole hidden) — all require the replicated full-H tensor. So an all-gather back
to full H is **architecturally unavoidable regardless of residual layout**, not a lazy "immediate
restore". The RS-alone lower bound (30.9 µs) plus the unavoidable AG (13 µs) = 43.9 µs ≈ the current
all_reduce; fusion can at best overlap the small decode matmul (14–25 µs) with the collective, which
cannot beat the latency floor and would add distributed-norm stat all-gathers.

## Sharded / fractured residual family (OPT-008) — rejected with reason

Carrying an H/4-fractured residual would require, per layer: RS after WO + AG before QKV + RS after
down + AG before router/gate-up = the same **2 RS + 2 AG** as the current 2 all_reduce, PLUS two
tiny distributed-RMSNorm stat all-gathers, for strictly ≥ cost. The replicated-residual + local-
exact-RMSNorm + ring-all_reduce contract is comm-optimal at H=2048. This is the same conclusion the
multichip stage reached, now confirmed with isolated collective measurements rather than reasoning.

## Kept

No CCL change. The inherited `ttnn.all_reduce` (ring RS+AG, 2/layer) is the measured-optimal choice.
Inter-layer residual contract unchanged: replicated BF16, no inter-layer collective.
