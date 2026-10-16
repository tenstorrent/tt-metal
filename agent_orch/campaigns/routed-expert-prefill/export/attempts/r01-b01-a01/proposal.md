# r01-b01-a01: runtime expert-parallel row groups (split the 8 M-rows into 1/2/4 groups that run different experts concurrently)

## Motivation
The op runs every local expert on all 8 M-rows of the 11x8 grid, one expert after another. Each
core's work per expert is `per_core_M x K x per_core_N` and `per_core_M = ceil(count_tiles / 8)`.
In the benchmark every expert is small. Tile-rows per expert, simulated from the test's routing
(seed 0):

| shape | experts/chip | tile-rows per expert |
|---|---|---|
| dsv3 / glm | 8 | 5-6 |
| kimi-k2.7 | 12 | 3-5 |
| kimi-k3 | 28 | 3-4 |
| mistral / minimax-m3 | 4 | 5-6 |
| dsv4-flash | 8 | 4-5 |
| dsv4-pro | 12 | 3 |
| gpt-oss | 4 | 8-9 |

So every core computes 1 tile-row per expert (2 for gpt-oss's 9-tile experts), and 2-5 of the 8
rows compute only padding. The baseline profile has all five RISCs busy for the whole kernel. DS-V3
works out to about 33 cycles per tile-matmul against the ~16 a LoFi matmul needs, and
198 MB of bf4 weights in 801 us is only about 250 GB/s. The time goes to per-expert, per-K-block
latency and handshakes in one serial pipeline, not to raw MACs or DRAM bandwidth.

## Mechanism
Split the 8 M-rows into G = 8/R independent groups of R rows (R in {8, 4, 2}). Each group runs
its own subset of experts, so G experts are in flight at once:
- A group's rows hold all of an expert's tile-rows: `per_core_M = ceil(count_tiles / R)`. Fewer
  rows go to padding. There are G times more independent weight streams (one weight sender per
  group-column) and G pipelines that overlap each other's latency.
- Each kernel picks R at **runtime** from the device-side counts. All five RISCs read the same
  counts and run the same integer cost model, so they agree without host syncs. Cost per expert
  is `units + chunks`, where units = sum of per_core_M over its chunks and chunks counts the
  weight re-reads. Experts go to the least-loaded group by list scheduling in index order.
  R = argmin over makespan, with ties going to the larger R. A single expert, or large token
  counts that would need more chunks, keep R = 8, which is the old behaviour.
- An expert that isn't in my group is skipped exactly like a count-0 expert. That is the existing
  uniform skip path (no CB traffic, no semaphores).
- Weight (in1) multicast topology per R: the sender is row `group_base + gx % R`, the rectangle
  covers the group's rows, and there are R-1 receivers. The host passes all three topologies as
  extra runtime args, and the kernels select one. The x multicast and the activated multicast are
  per-row and unchanged. Every row in a group processes the same experts.
- Compute: UNPACK sends MATH/PACK a header (R), then the count of each expert it works on, then a
  sentinel 0. It still writes one mailbox entry per worked expert, as today. FUSE_BIAS: the reader
  pushes biases only for worked experts, and compute pops them only for those.
- Host: `num_chunks_max` is now the bound for the smallest group (R=2), and the compute kernel
  gets `gy` as a runtime arg.

Files: `unified_routed_expert_ffn/device/kernels/adaptive_chunk.hpp`,
`.../kernels/dataflow/unified_routed_expert_ffn_reader.cpp`,
`.../kernels/dataflow/unified_routed_expert_ffn_writer.cpp`,
`.../kernels/compute/fused_swiglu.cpp`, `.../device/unified_routed_expert_ffn_program_factory.cpp`.

## Why this is not a repeat
This is the first node of the campaign. No prior attempts exist.

## Expected effect and risk
Cost-model picks (units+chunks): dsv3, glm, mistral, mm3, dsv4f, dsv4p, kimi-k2.7 and kimi-k3 get
R=2; gpt-oss gets R=4. If time scales with per-core MAC rows, the MAC-row reduction would be
1.3x (5-6-tile shapes) up to 2x (3-4-tile shapes). On top of that come more DRAM stream
parallelism and overlapped pipelines. The floor is DRAM bandwidth (~450 GB/s, so roughly
440 us for DS-V3). Numerics should be bit-identical: each output tile is computed with the same
K-block order on whichever core owns it.
Risks: hangs from a semaphore or multicast topology mistake (the eval would show `hang`). A
wrong row mapping would show up as a PCC collapse. Group pipelines that are DRAM-bound could make
the gain smaller than the MAC estimate.
