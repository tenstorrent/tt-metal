# Collective/projection candidate audit

CPU source inspection only; no device experiments or measured speed claims. Target: 1x4 Blackhole, mesh axis1, hidden-sharded residual `[1,1,S,512]`; full hidden2048. Source inspected: current `tt/multichip_decoder.py`, the experimental CCL implementations named below, Gemma QB2 `_GatherProjection`, and GPT-OSS batched-decode all-reduce. Candidate selection must follow whole-layer measurements and PCC, not this ranking alone.

## Recommended order

1. Full-attention O projection: fused `matmul_reduce_scatter_prefill`, including decode padded internally to32 rows. Consume its `[1,1,S,512]` result directly in residual add and the following norm. This is the simplest exact topology match and avoids a materialized full partial plus a separate RS launch.
2. Full-attention QKVG: keep distributed norm output sharded, then `all_gather_matmul_async`; retain the returned gathered tensor only when another branch needs it. Measure this together with candidate1 under the same BFP8/HiFi2 policy.
3. Replicated-residual control: replace current RS followed immediately by AG with the *direct*, scratch-buffer overload of `all_reduce_async`. Its Blackhole input must be L1 WIDTH_SHARDED; DRAM is explicitly rejected. Include conversions and the following local norm in timing.
4. Shared-expert gather projection: fuse AG with the packed288-wide projection and reuse the returned full2048 activation for router/experts. Do not add another AG for that branch.

The linear-attention ordered projections use custom SFPU product/scan arithmetic. Native fused GEMM cannot preserve that instruction sequence. They require a separate numerical candidate against the ordered path; a full-attention success does not qualify them. Likewise, the default ordered linear norm gathers *before* its FP32 arithmetic, so it cannot immediately feed an AG-MM with sharded normalized input.

## Boundary contracts and traffic

Let `Mp=32*ceil(S/32)`. Numbers below are nominal per-device collective payloads for four-way bandwidth-optimal exchange, before headers, topology forwarding and chunk alignment. These are comparison estimates, not measured fabric byte counters.

| Family | Producer/consumer contract | BF16 activation payload | Persistent storage / qualification |
|---|---|---:|---|
| Separate MM + RS, sharded residual | local `[Mp,2048]` partial -> RS `[Mp,512]` -> residual add -> distributed norm | RS `3072*Mp` bytes | Current path; partial materialized in DRAM |
| Separate MM + direct AR, replicated residual | local partial -> replicated2048 -> add -> local norm | AR `6144*Mp` bytes | L1 WIDTH_SHARDED input/output; scratch at least4x each output shard |
| Fused MM-RS, sharded residual | O `[Mp,1024] @ [1024,2048]` -> `[Mp,512]` -> add/norm | Same RS payload class; internal accumulation/transport can be FP32 | Optional persistent BF16 output; no immediate AG |
| AG-MM after distributed norm | normalized512 -> fused gather2048 and local QKVG2560 | AG `3072*Mp` bytes | Per-live-trace gathered2048 buffer and semaphores |
| Shared AG-MM with branch reuse | normalized512 -> gathered2048 + packed288; gathered tensor also feeds router/experts | Same AG payload; no duplicate gather | Retain gathered buffer until all branch consumers finish |
| Shared down fused MM-RS alone | local128 -> reduced512; combine with routed residual contribution | Extra RS unless routed/shared reduction is restructured | Not an isolated whole-layer win: current code sums routed+shared locally before one RS |

At S1, BF16 AG/RS nominal payload is96KiB and AR192KiB because of physical32-row tiles. FP32 partial RS doubles its corresponding estimate. Preserve FP32 for the ordered linear output control; changing CCL dtype is a separate numerical candidate.

## Fused O projection: exact supported call

The local full-attention O weight is already `[1024,2048]`; input is `[1,1,S,1024]`. `matmul_reduce_scatter_prefill` supports rank2–4 A with leading dimensions1, rank2 W, default32x32 tiles, DRAM/L1 **interleaved** operands/output, BF16/BFP8 A and BF16/BFP8/BFP4 W. M/K/N must be logically tile-aligned; scattered N must divide `32*group_size=128`. N2048 meets this condition. Supported fabric topologies are Linear and Ring, with1 or2 links. Output is always BF16; FP32 destination/full synchronization are forced by the op.

```python
# Setup: W=[1024,2048], out=[1,1,Mp,512], both DRAM interleaved.
# For S1/tails, pad A's logical M to Mp; crop only the output rows afterward.
y = ttnn.experimental.matmul_reduce_scatter_prefill(
    a_padded, w_o, cluster_axis=1, scatter_dim=-1,
    topology=ttnn.Topology.Linear, num_links=2,
    compute_kernel_config=attention_compute,
    memory_config=ttnn.DRAM_MEMORY_CONFIG, output_tensor=out,
)
# y is already hidden-sharded. Add residual and run next norm directly.
```

Padding here affects a pure projection, so it need not write padded KV/recurrent state. A padded singleton must never be forwarded into attention/cache update as32 tokens. Warm every required padded signature before capture. Output must not alias A or W, including views. Log `_matmul_reduce_scatter_prefill_last_plan()` to identify the selected resident/stream schedule, K block, grid and L1 footprint. The repo's Blackhole tests include padded-decode M32 cases; the API name is not a decode exclusion.

Shared down `[Mp,128] @ [128,2048]` satisfies these shape constraints too. However, fusing it separately adds communication unless the routed branch's reduction is reorganized. First compare the complete routed+shared boundary. Do not apply a replicated shared scalar gate once per TP rank after AR in a way that duplicates its contribution; multiplying the *reduced shared result* once by its common scalar is mathematically valid, with rounding differences to check.

## Fused gather projection: exact supported call

`all_gather_matmul_async` requires rank4 input/weight, first two padded A dimensions1, gather dim3, and an explicit **1D or2D multicast** matmul program. A DRAM-sharded matmul program is not accepted. Input/weight/bias must use distinct backing buffers. If the gathered output is intra-chip sharded, its core-shard count must equal the mesh gather size4; use DRAM interleaved for the first probe.

For QKVG use local input `[1,1,S,512]`, W `[1,1,2048,2560]`, persistent gathered `[1,1,S,2048]`. The persistent buffer dtype must equal the activation dtype. Start with BF16 activations, BFP8 weights, HiFi2 FP32 destination, matching the selected attention policy. Reserve nonoverlapping core ranges for matmul and AG workers.

```python
full_x, qkvg = ttnn.experimental.all_gather_matmul_async(
    norm_local, w_qkvg_4d, persistent_output_buffer=gather_buffer,
    dim=3, multi_device_global_semaphore=ag_semaphores,
    all_gather_core_grid_offset=ag_offset,
    barrier_semaphore=ag_barrier, num_links=1,
    topology=ttnn.Topology.Linear,
    memory_config_ag=ttnn.DRAM_MEMORY_CONFIG,
    memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
    program_config=multicast_program,
    compute_kernel_config=attention_compute, dtype=ttnn.bfloat16,
)
```

This API has no `cluster_axis` keyword; on this1x4 mesh its full storage ring is the intended group. A shape-faithful decode starting geometry is 1D multicast with40 active cores, `per_core_M=1`, `per_core_N=2` (80 output tiles), `out_subblock_h=1`, `out_subblock_w=2`, `mcast_in0=True`; reserve AG cores outside that compute rectangle. Sweep legal local-ready-slice K divisors `in0_block_w=1,2,4,8,16`: each512-wide slice contains16 tiles. Actual grid/worker allocation must fit the queried physical grid. Prefill needs its own 2D multicast geometry.

For packed shared projection, W is `[1,1,2048,288]` (nine N tiles). A1D geometry with nine N workers and `per_core_N=1` avoids silently padding the public output. Packed scalar occupies the final32 columns. Reuse `full_x` for router/expert consumers; otherwise fusion would accidentally add a second gather. Shared BFP8/HiFi2 and router FP32 arithmetic remain independently specified.

An additional source-supported family is `strided_all_gather_minimal_matmul_async` (MinimalMatmul config, explicit `cluster_axis`, persistent gather output). It can fuse SwiGLU, but only with tile-pair-interleaved gate/up weights and N divisible by64. Current packed288 includes a scalar and fails that SwiGLU packing contract. Adapt by splitting the scalar and packing gate/up256 as alternating32-column tiles; reuse the gathered activation for the scalar/router. This is a real candidate, not rejected by the first packing mismatch.

## Direct all-reduce control

Use the overload accepting `buffer_tensor` and one `multi_device_global_semaphore`. The other overloads accepting RS/AG semaphore vectors compose RS+AG and are not the desired direct-kernel comparison.

```python
# partial: L1 WIDTH_SHARDED [1,1,Mp,2048]
# scratch: same dtype/layout, width8192 on the same core grid;
# each scratch shard must hold at least4x the output shard volume.
y = ttnn.experimental.all_reduce_async(
    partial, scratch, cluster_axis=1, mesh_device=mesh,
    multi_device_global_semaphore=ar_sem,
    memory_config=residual_l1_width_sharded,
    dtype=ttnn.bfloat16, topology=ttnn.Topology.Linear,
    num_links=2, fp32_dest_acc=True,
)
```

The direct op rejects Blackhole DRAM input, non-width-sharded input/scratch/output, odd device counts and undersized scratch. Its output is replicated **across chips**, despite being width-sharded **within each chip**. Benchmark persistent scratch and separately owned semaphores for attention/MoE (or prove serial lifetime reuse). For the replicated family, keep add and next norm in the L1 residual layout where supported; charging an avoidable DRAM roundtrip would not evaluate the coherent family. Try BF16 first; BFP8 CCL is a separate real-weight PCC candidate, not an assumed equivalent.

## Other fused MM-RS options and guardrails

- `matmul_reduce_scatter_async` requires dim3 and2D multicast program config, explicit persistent intermediate/output buffers and separate CCL core offset. It returns both MM and RS outputs. It has no explicit mesh-axis keyword. This is a valid alternate family if the planner-selected prefill op loses; it is not compatible with an unchanged DRAM-sharded matmul config.
- `minimal_matmul_strided_reduce_scatter_async` supports explicit axis and a MinimalMatmul config. L1 MM output requires a bounded `mm_window_blocks` plus caller-owned progress **and** credit counters; the returned MM buffer is only a recycled window. DRAM MM output avoids that window contract for a first probe. Do not allocate per-layer counter buffers unnecessarily; CCLManager exposes reusable counter buffers.
- Fused add/residual operations before a TP sum must not add the replicated residual four times. Add the residual after RS unless the op explicitly implements an ownership-aware fused addition.
- Measure `projection -> collective -> add -> next norm/projection` using the same precision, persistent buffers and residual contract. The mandatory sharded candidate must not immediately gather just to restore the former interface. Test stack compatibility and valid tail lengths along with trace replay/lifetimes.
- No candidate above has been measured by this audit. First shape/API failures require compatible repacking/config adjustment or diagnosis before rejection.

## Source pointers

- `ttnn/cpp/ttnn/operations/experimental/ccl/matmul_reduce_scatter_prefill/`: nanobind contract; device `check_support`, `check_operands`, `check_group`, output alias checks.
- `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_matmul_async/`: nanobind signature; device validation; program factory merges disjoint AG/MM core descriptors.
- `ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/`: direct overload and Blackhole/shard/scratch validation.
- `ttnn/cpp/ttnn/operations/experimental/ccl/strided_all_gather_minimal_matmul_async/`: SwiGLU packing, fused ternary restrictions, MinimalMatmul contract.
- `ttnn/cpp/ttnn/operations/experimental/ccl/minimal_matmul_strided_reduce_scatter_async/`: window/counter lifetime requirements.
- `models/demos/gemma4_26b_a4b_qb2/tt/multichip_decoder.py:_GatherProjection`; `models/demos/gpt_oss/tt/batched_decode.py` direct AR.
- `tests/ttnn/unit_tests/operations/ccl/blackhole_CI/box/nightly/test_matmul_reduce_scatter_prefill_bh.py`; `tests/nightly/t3000/ccl/test_minimal_all_gather_matmul_async.py`.
