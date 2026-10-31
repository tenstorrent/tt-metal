# AUTOTRIAGE

## Diagnosis
Fused matmul-reduce-scatter stalls in its fabric producer on the linear mesh; root cause is not yet proved. The fused factory unconditionally constructs ring RS kernels whereas the passing unfused linear operation selects line kernels. Test matching ring fabric/topology before rejecting fusion.

## Triage Evidence
triage_fused.txt identifies running op716 MatmulReduceScatterAsyncDeviceOperation, preceding op715 ReshapeView. Local input[1,1,1,2048], BFP4 weight[2048,6144], BF16 intermediate[2,1,32,6144], output[1,1,1,1536]. All4 chips participate. Writer stacks on endpoints0 and3 stop in FabricMuxV2Sender::wait_until_ready_blocking/flush while sending remote chunks (ring_reduce_scatter_minimal_async_writer.cpp:293,783). Binary integrity and core magic pass. NoC counters sampled while live are not proof of board damage.

## Source Evidence
matmul_reduce_scatter_async_program_factory.cpp:74 calls build_ring_reduce_scatter_minimal_async_program_descriptor with topology supplied by caller. Standalone reduce_scatter_minimal_async_op_device_operation selects the line factory for Linear. Current candidate supplies Linear/FABRIC_1D and two links. Intermediate has doubled leading dimension appropriate to the standalone line API; ring candidate requires input-shaped tiled intermediate. Row geometry is padded per output shard to1536, with1344 useful channels,8x1 matmul cores,N24 tiles/core,out block12,subblock1x4. Output normalization consumes1344 channels after internal slicing, preserving the sharded residual contract.

## Downstream Effects
Host output read waits for unfinished fused WO; downstream norms/MLP are queued and do not identify the initiating fault. Sparse broken-core samples from triage are not a board-health verdict.

## Proposed Fix
Probe FABRIC_1D_RING at mesh1x4; if supported, use Ring for the fused op and an input-shaped persistent intermediate. Keep real weights, local input shapes, BFP4/LoFi and delayed-gather residual contract. This is an adaptation within the stage's test scope, not a tt-metal kernel change.

## Uncertainty
The ring helper receives topology and may support some line cases; its name alone does not prove incompatibility. The matched-fabric experiment must establish whether it resolves the observed send stalls. No fused performance or correctness result exists yet.
