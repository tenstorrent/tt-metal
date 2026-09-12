# Inter-layer residual contract

This is the selected Stage5 default contract. Default-path evidence is indexed
by the stage README.

The target is a local Blackhole `MeshShape(1,4)`. Set
`TT_MESH_PASS_THROUGH_THREAD_POOL=1` before opening that mesh; the optimized
launchers select it by default. Paired local TP4 controls reduce eager prefill
by 6–9% across both kinds and their stack while traced decode is unchanged.
This changes host task dispatch, not device execution or TP ownership.
Use the same setting when reproducing this stage or assembling its layers. Each device owns a quarter
of attention heads and MLP intermediate channels. Residual hidden values are
replicated across the mesh. This is tensor-parallel execution with a replicated
residual, not four replicated model executions.

| Boundary | Logical tensor | Per-device storage |
| --- | --- | --- |
| Batch1 decode input/output | BF16 TILE `[1,1,5120]` | L1 WIDTH_SHARDED, rectangular10x4 grid, shard `[32,128]` |
| Batch2–32 decode input/output | BF16 TILE `[B,1,5120]` | DRAM INTERLEAVED; public TILE layout pads each user's time axis to32 |
| Packed internal decode residual | BF16 TILE `[1,1,B,5120]` | L1 WIDTH_SHARDED on40 cores, one32-row shard per core for B<=32 |
| Prefill input/output | BF16 TILE `[B,S,5120]` | DRAM INTERLEAVED, logical S unrestricted within context |

Pass one layer's output directly to the next layer. Do not insert a mesh
gather, reduce-scatter, all-reduce, or local reshard at that boundary. Batch1
`decode_forward`'s input memory request already matches the preceding output
and is a no-op. Batch2–32 pack/unpack their local public rows inside each layer;
the public boundary stays in DRAM to avoid retaining B*32 rows in L1.

Each layer internally sums its row-parallel attention-output and MLP-down
projections. Decode uses two native async all-reduces, axis1, Ring, two links,
BF16 payload, L1 width-sharded input/output on the same40 residual cores.
Those reductions compute the layer's mathematical outputs; none is added
between layers. Prefill uses async reduce-scatter and all-gather in DRAM.

Construct every layer in an ordered CQ0 stack with the same `TT_CCL` context.
The direct-all-reduce workspace `[1,1,32,20480]` is pooled once per context
and layout, not once per layer. It occupies1,310,720 bytes per device,
32,768 bytes on each of40 cores (`[32,512]` BF16 shards). Keep the context,
workspace, semaphores, input tensors, states and trace outputs alive until all
replays finish. Do not share that context across concurrently executing stacks.

The caller owns logical positions, replicated page tables and request state.
Each full-attention rank stores one local KV head in BFP8 TILE
`[physical_pages,1,32,256]`. Linear ranks store12 FP32 recurrent heads and
BF16 row-major convolution history. Trace replay must refresh caller-owned
device inputs without reallocating their storage; comparison readbacks are
outside the forward and timed replay.

Sharded hidden1280 residual alternatives were implemented through two unlike
layers, including distributed/fused norms, residual adds, RS, row AGMM, MMRS,
column AGMM, BF16/BF8 payloads and persistent buffers. Their comparison gather
is outside the measured stack. These coherent alternatives lose to the selected
replicated stream; see `topology_family_results.json` and the stage README.

This contract does not begin full-model assembly. It records the decoder
boundary that later assembly must preserve.
