## Files read
- agent_orch/WORKER.md, campaign.yaml: rules, allowed paths, metric.
- kernels/compute/fused_swiglu.cpp (all): GU K-loop (rt=1 x ct subblocks, reuse_a, the weight tile streamed in SrcA), two-block walk + PACK activation, down matmul_phase, kernel_main.
- tt-llk blackhole llk_unpack_AB_matmul.h / llk_math_matmul.h: reuse_a = ct >= rt decides which operand streams. The math is symmetric; unpack MOP per streamed tile.
- ttnn matmul_program_config.cpp SUBBLOCK_HW_CHOICES: big GEMMs prefer (4,2), i.e. they stream SrcB.
- kernels/dataflow/unified_routed_expert_ffn_reader.cpp: GU loop (writer-mcast mode decouples gate/up reads from receiver acks) and the phase-4 down loop. There the sender waits for in1_ready from all receivers BEFORE issuing its NoC-0 half of the down read.
- kernels/dataflow/unified_routed_expert_ffn_writer.cpp: DOWN_SPLIT half reads slot (down_blk+kb) % IN1_D_SLOTS on down_go.
- program_factory.cpp: up_mode (UP_SPLIT), kWriterMcastsIn1, DOWN_SPLIT, d_slots (2 or 3).
- tt_metal/hw/inc/api/dataflow/{circular_buffer.h,dataflow_api.h}: pages_reservable_at_back; CB pointers in bytes; wrap rule.
- reports/r03-b04-a03/run.log: per-shape token counts, weight GB/s per chip (gpt-oss ~200, mistral ~275, mm3 ~320, others 335-400).

## Nodes consulted
- history.md: all 44 nodes (index + next lines).
- r03-b04-a03 (parent): the tail is mostly DN; the down read is gated on receivers' acks (its next #2).
- r03-b02-a03: perf counters, FPU at 16 cyc/mult; proposes streaming x. Not taken: r03-b04-a01's per_core_M fit puts the marginal cost at the LoFi ideal, and r03-b03-a02's DM counters show receivers mostly waiting on DM. So the matmul rate is not the limit.
- r03-b04-a01, r03-b04-a02, r03-b03-a02: DM wait breakdowns. Gate/up block cadence 9-16 µs vs ~6.8 µs compute. Weight streams must stay split across both NoCs.
- r02-b02-a03: deeper down ring + look-ahead across the activation gap redistributed DRAM between groups and widened the tail (-1.3%). This node keeps the ring and never reads block 0 early.
- r02-b04-a02: an all-on-one-NoC down stream lost 10-22%. Not repeated: both halves keep their NoCs.

## Docs / external references
- tech_reports/GEMM_FLOPS/GEMM_FLOPS.md: LoFi 16 cycles/tile ideal.
