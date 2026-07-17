# Fractured-residual topology probe plan

This is a read-only design result; no device experiment was run while preparing
it.  The probe below is the smallest shape-faithful steady-state comparison
between the selected replicated-residual/two-all-reduce implementation and the
main rejected-alternative family required by the multichip audit.

## Measured disposition

The probe was implemented and extended through the full boundary needed for a
fair decision. Exact BFP4 QKV weights, batch 32 shapes, persistent gather
storage, PCC checks, and 1,000 trace replays were used.

- A projection-only `RS -> AG -> QKV` boundary was 11.1% faster, so it was not
  accepted as sufficient rejection evidence.
- Adding the required local residual add and distributed RMS statistics made
  the supported fractured boundary 0.099587 ms versus 0.083456 ms for the
  selected all-reduce/replicated-norm boundary: 19.3% slower.
- Rank PCC values were 0.9999763312 and 0.9999763298.
- The fused primitive was retried after adapting weights to rank 4,
  DRAM-interleaved BFP4 placement, and a persistent output with exactly two
  gather shards. It then hit the physical topology gate: this P300 pair has no
  wrap edge, TTNN resolves the usable topology to Linear, and fused AGMM
  explicitly rejects Linear.
- The supported separate-AG candidate is therefore the measured alternative,
  and the replicated residual remains the final implementation.

The original design and repair ladder below are retained as provenance. One
detail learned during execution is that the persistent gather output must have
two `[32,2048]` slices (one per rank); the 32-core `[32,128]` description is
valid for an ordinary compute-sharded full-hidden tensor but not for this
primitive's persistent gather buffer.

## Recommendation

Probe a **hybrid fractured-residual chain**, not a fully column-parallel O/down
rewrite:

1. Keep the decoder boundary fractured across TP ranks: each device owns
   `[1, 1, 32, 2048]` of the global BF16 `[1, 1, 32, 4096]` residual.
2. Compute distributed RMSNorm from the two local 2048-wide pieces.
3. Fuse the hidden all-gather with the rank-local column-parallel QKV matmul.
4. Run local-head attention.  Keep O row-parallel, produce a full-width partial,
   and reduce-scatter it back to a 2048-wide residual shard.
5. Add the local residual, compute the second distributed RMSNorm, and fuse its
   hidden all-gather with the local gate matmul.  Reuse the all-gathered tensor
   returned by the fused op for the up matmul, avoiding a second collective.
6. Keep down row-parallel and reduce-scatter its full-width partial.  Add the
   local residual and return another `[1, 1, 32, 2048]` shard for the next layer.

This is the meaningful steady-state alternative: each all-reduce's all-gather
phase moves to the beginning of the next column-parallel projection, where
`all_gather_matmul_async` can overlap it with compute.  It preserves the current
O/down sharding and local KV-head ownership.  A fully column-parallel O/down
variant forces an all-gather of the 7168-wide local MLP intermediate before down
and has materially more traffic; it is not the first candidate to implement.

## Exact TP2 tensor contract

Target is the fixed P300 `MeshShape(1, 2)`, ring topology, two links, decode
batch 32, BF16 activations/collectives, BFP4 weights, and the existing LoFi
matmul kernels.

| Value | Global shape | Per-device physical shape | Mesh placement |
| --- | ---: | ---: | --- |
| residual / normalized residual | `[1,1,32,4096]` | `[1,1,32,2048]` | shard dim 3 |
| packed QKV weight | `[4096,6144]` | `[4096,3072]` | shard output dim 1; each shard is local Q16/K4/V4 |
| packed QKV output | `[1,1,32,6144]` | `[1,1,32,3072]` | local Q/K/V columns |
| attention context | `[1,1,32,4096]` | `[1,1,32,2048]` | 16 local query heads |
| O weight | `[4096,4096]` | `[2048,4096]` | shard input dim 0 |
| O partial | replicated logical width | `[1,1,32,4096]` | one partial per rank |
| O reduce-scatter result | `[1,1,32,4096]` | `[1,1,32,2048]` | shard dim 3 |
| gate weight | `[4096,14336]` | `[4096,7168]` | shard output dim 1 |
| up weight | `[4096,14336]` | `[4096,7168]` | shard output dim 1 |
| local gate / up | `[1,1,32,14336]` | `[1,1,32,7168]` each | local MLP features |
| down weight | `[14336,4096]` | `[7168,4096]` | shard input dim 0 |
| down partial | replicated logical width | `[1,1,32,4096]` | one partial per rank |
| down reduce-scatter result | `[1,1,32,4096]` | `[1,1,32,2048]` | shard dim 3 |
| RMSNorm gamma | `[1,1,128,32]` | `[1,1,64,32]` | shard tile dim 2, row-major DRAM |
| local RMS statistic | n/a | padded `[1,1,32,32]` | one BF16 tile per rank |
| gathered RMS statistics | n/a | `[1,1,32,64]` on every rank | one-core L1 width shard |

`_pack_tp_qkv` already constructs the required rank-local Q/K/V order before
mesh sharding.  Gate and up can remain separate.  As a secondary experiment,
`_pack_tp_gate_up` can form a per-rank `[4096,14336]` weight and one local
`[1,1,32,14336]` output, but the primary probe should fuse gate only and reuse
the returned gathered hidden tensor for up.  That avoids the packed output's
larger circular buffers and the interleave/slice/reshard sequence.

## In-device memory and program configurations

Use an 8-by-4 rectangular compute grid (32 cores), which fits the P300 11-by-10
worker grid.

- Local fractured residual: L1 width-sharded, shard shape `[32,64]` on 8x4.
- Full hidden all-gather result: L1 width-sharded, shard shape `[32,128]` on 8x4.
- Local QKV output: L1 width-sharded, shard shape `[32,96]` on 8x4.
- Local gate/up output: L1 width-sharded, shard shape `[32,224]` on 8x4.
- O and down input/output matmuls retain the current production DRAM-sharded
  configs.  Convert their full-width sharded partial to L1 interleaved before
  reduce-scatter, matching `Attention1D._all_reduce_output_decode`.
- Fused-AGMM QKV/gate/up weights must be DRAM **interleaved**.  The common fused
  path explicitly uses interleaved weights; the current `_dram_weight_memory_config`
  tensors are not a valid assumption for the fused probe.

The fused QKV program is:

```python
ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
    compute_with_storage_grid_size=(8, 4),
    in0_block_w=4,       # 4096 / 32 / 32 cores
    out_subblock_h=1,
    out_subblock_w=3,    # divides per_core_N=3
    per_core_M=1,        # padded decode rows = 32
    per_core_N=3,        # 3072 / 32 / 32 cores
    fuse_batch=True,
    fused_activation=None,
    mcast_in0=True,
)
```

The fused gate and ordinary up matmuls use the same config except
`per_core_N=7` and `out_subblock_w=1` (`7168 / 32 / 32 = 7`).  If packed
gate/up is tested, use `per_core_N=14`, `out_subblock_w=2`.  All use
`in0_block_w=4`.  These configurations preserve the selected BFP4 weight and
LoFi compute policy; a BF16 weight run is diagnostic only.

Distributed RMSNorm uses the local 2048-wide shard:

```python
norm_pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
    compute_with_storage_grid_size=(8, 4),
    subblock_w=2,
    block_h=1,
    block_w=2,  # 2048 / 32 cores / 32 columns per tile
    inplace=False,
)

stats = ttnn.rms_norm_pre_all_gather(x_local, program_config=norm_pc)
stats = ttnn.experimental.all_gather_async(
    stats,
    persistent_output_buffer=stats_buffer,  # replicated [1,1,32,64]
    dim=3,
    multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(),
    barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
    num_links=2,
    topology=ttnn.Topology.Ring,
    memory_config=stats_one_core_l1,          # shard [32,64]
    chunks_per_sync=10,
    num_workers_per_link=2,
    num_buffers_per_channel=2,
)
x_norm_local = ttnn.rms_norm_post_all_gather(
    x_local,
    stats,
    epsilon=eps,
    weight=local_gamma,
    program_config=norm_pc,
    dtype=ttnn.bfloat16,
    memory_config=fractured_residual_memcfg,
)
```

The exact fused hidden gather plus projection call is:

```python
gathered_hidden, local_projection = ttnn.experimental.all_gather_matmul_async(
    x_norm_local,
    local_column_weight,
    persistent_output_buffer=full_hidden_buffer,  # replicated [1,1,32,4096]
    dim=3,
    multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(),
    all_gather_core_grid_offset=(0, 4),
    barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
    num_links=2,
    topology=ttnn.Topology.Ring,
    memory_config_ag=full_hidden_l1_memcfg,        # 8x4, shard [32,128]
    memory_config_mm=local_projection_l1_memcfg,
    program_config=agmm_program_config,
    compute_kernel_config=existing_lofi_kernel,
    chunks_per_sync=10,
    num_workers_per_link=2,
    num_buffers_per_channel=2,
)
```

For the MLP, call that with gate weight, then call `ttnn.linear` on
`gathered_hidden` with the local up weight and the gate/up program config.  This
is one hidden all-gather, two local column-parallel matmuls.

After the existing O/down row matmul, reduce-scatter the full partial:

```python
partial = ttnn.sharded_to_interleaved(partial, ttnn.L1_MEMORY_CONFIG, ttnn.bfloat16)
local_result = ttnn.experimental.reduce_scatter_minimal_async(
    partial,
    persistent_output_buffers=[rs_intermediate, rs_output],
    dim=3,
    multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(),
    barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
    num_links=2,
    memory_config=fractured_residual_memcfg,
    intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    topology=ttnn.Topology.Ring,
    chunks_per_sync=10,
    num_workers_per_link=2,
    num_buffers_per_channel=2,
)
```

For ring reduce-scatter, allocate the persistent intermediate as replicated
BF16 DRAM `[1,1,32,4096]` and the persistent output as replicated storage with
local result shape `[1,1,32,2048]` in `fractured_residual_memcfg`.  Use distinct
O and down outputs while first proving correctness; alias only after a trace
liveness audit.

## Calculated communication payload

The values below are per device, per layer, for batch 32 BF16.  They count the
logical peer payload of each ring phase, consistently for both candidates.

Selected replicated-residual path:

- O all-reduce: RS half `32*2048*2 = 131,072 B`, then AG half `131,072 B`.
- Down all-reduce: the same `262,144 B`.
- Total: **524,288 B/device/layer**.

Fractured hybrid:

- QKV fused hidden AG: `32*2048*2 = 131,072 B`.
- O RS: `131,072 B`.
- Gate fused hidden AG (gather reused by up): `131,072 B`.
- Down RS: `131,072 B`.
- Two RMS-stat AGs: `2 * (32*32*2) = 4,096 B` because each local statistic is
  tile padded.
- Total: **528,384 B/device/layer**, only **4,096 B / 0.78125%** above the
  selected path.  It may still win because two hidden AGs overlap projection
  compute and residual/norm/add work is half width.

Do not run separate hidden all-gathers for gate and up: that becomes 659,456 B,
25.8% above selected.  A fully column-parallel O/down family requires the QKV,
context, gate, and 7168-wide intermediate gathers plus stats:
`131,072 + 131,072 + 131,072 + 458,752 + 4,096 = 856,064 B`, 63.3% above
selected.  The hybrid is the only fractured candidate worth a full first run.

## Smallest shape-faithful test

Add one environment-gated test beside `test_multichip_decoder.py`; it should use
the exact production dimensions, dtypes, two-chip ring, and batch 32 but may
replace SDPA/cache with a deterministic dependency that preserves the projection
shapes:

1. Generate a global `[1,1,32,4096]` residual and shard dim 3.
2. Run distributed norm and fused QKV AGMM.  Use the first 2048 local Q columns
   as a deterministic local context so QKV cannot be dead work.
3. Run the current row-parallel O, reduce-scatter, and local residual add.
4. Run distributed norm, fused gate AGMM, up from the returned gathered hidden,
   SwiGLU, current row-parallel down, reduce-scatter, and local residual add.
5. Concatenate final rank shards on dim 3 and compare to a torch implementation
   of the same synthetic graph.  Also check distributed-norm output, local QKV,
   each reduce-scatter result, gate/up, and final output.  Use the stage's
   production BFP4 PCC floor rather than an equality check.
6. Build a replicated synthetic chain with the same weights and current generic
   all-reduces.  Compile each once, capture one trace, replay at least 50 warmed
   iterations, verify post-replay PCC/determinism, and report candidate latency,
   selected latency, speedup, and efficiency under separate Tracy signposts.
7. Profile the winning/rejected runs separately.  The TT perf table must show the
   two AGMMs, two RS operations, two stats AGs, DRAM weight reads, and reshards;
   otherwise the timing window is not the intended graph.

The test should seed rank-distinct values and weights.  For the torch QKV
reference, preserve `_pack_tp_qkv`'s rank-local `[Q,K,V]` pack order instead of
assuming ordinary global `[all Q, all K, all V]` order.

## Expected blockers and repair order

1. **P300/TP2 fused AGMM coverage.**  The common `Attention1D` auto-enables its
   fused path only for eight devices, and the available general fused-AGMM tests
   exercise T3000/Galaxy rather than P300 TP2.  The API is topology-general, but
   a P300 compile/validation failure is plausible.
2. **BFP4 fused weight coverage.**  Existing fused-AGMM tests primarily use BF16.
   First reproduce an API/geometry failure at exact shapes with BF16, then retry
   the repaired path with BFP4.  BF16 is not an acceptable performance result.
3. **Weight placement.**  Fused AGMM expects DRAM-interleaved weights in the
   common path.  The current multichip QKV/gate/up are DRAM width-sharded; the
   probe needs separately loaded interleaved tensors and must include their
   memory cost in the comparison.
4. **Worker-grid overlap.**  `(0,4)` is proven by the common ring path but its
   program uses an 8x1 matmul grid.  The proposed 8x4 P300 grid leaves rows 4-9
   for CCL workers; if validation rejects it, try the common 8x1 geometry before
   declaring AGMM unsupported.
5. **Packed gate/up CB pressure.**  The stage already observed CB pressure at
   large gate/up shapes.  Prefer fused gate plus gathered-input reuse for up.
   Packed gate/up is a secondary measured candidate only.
6. **Distributed-norm stats layout.**  Post-all-gather requires the gathered
   stats on one width-sharded core and local gamma sharded by norm tiles.  DRAM
   interleaved stats or replicated 4096-wide gamma can compile yet violate the
   intended local contract.
7. **Async lifetime and trace reuse.**  AG and RS persistent buffers must remain
   live for the trace.  Do not deallocate returned gathered tensors before the
   up matmul or reuse O/down RS outputs until liveness is established.
8. **Standalone versus stack boundary.**  This probe measures steady state.  A
   real implementation must change the stacked-decoder input/output contract to
   dim-3 mesh shards, shard embedding output once, and adapt final norm/head once.
   Per-layer gather-to-replicated output would erase the candidate's benefit.

If fused AGMM fails, the diagnostic ladder is exact BFP4 8x4, exact BF16 8x4,
exact BF16 8x1, then separate `all_gather_async + ttnn.linear` to prove the
fractured/norm/RS math.  A separate AG plus matmul can establish correctness but
does not satisfy the fused candidate's performance claim.  Preserve the exact
compile/runtime error as rejection evidence if the autofix loop exhausts these
variants.
