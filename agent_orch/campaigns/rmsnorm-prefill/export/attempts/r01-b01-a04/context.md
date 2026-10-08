## Files read
- agent_orch/WORKER.md, campaign.yaml: process, allowed paths, gate (pcc 0.99999, max_abs 0.05).
- kernels/compute/dit_rmsnorm_fused_compute.cpp: PRE (resident branch) does a per-block mul_tiles into DST 0..3 plus an L1-acc fp32 pack per tile into pre_intermediate, then reduce<SUM,ROW> + transpose + stick push. The x*gamma pre-pass and single POST pass come from r01-b01-a01.
- device/dit_fused_distributed_rmsnorm_program_factory.cpp (compute config): HiFi4 (default in dit_fused_distributed_rmsnorm.cpp), fp32_dest_acc_en required, block_size = dst_reg_count.
- tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_binary.h: standard ELWMUL MOP always uses dest_accum_en=0, and HiFi phases accumulate in dest.
- tt_metal/tt-llk/tt_llk_blackhole/llk_lib/experimental/llk_math_eltwise_binary_custom.h: documents that ELWMUL with dest_accum_en=0 is a dest MAC, dest cleared by ZEROACC, and uses it for a head reduction with one pack per column. This is the basis of this node.
- tt_metal/hw/inc/api/compute/eltwise_binary.h: binary_tiles_init / mul_tiles API.
## Nodes consulted
- All 12 nodes (r01-b01..b04 a01-a03): proposals, reflections, scores.
- r01-b01-a03 (parent): PRE tail 2-3.7 µs after R_INPUT at h7168, which gates the AG start.
- r01-b02-a02, r01-b02-a03, r01-b04-a03: same PRE-gates-AG finding, ~125 ns/tile, each suggests DST accumulation.
- r01-b03-* : column split (structural lever). I left it to other branches for diversity; this node is orthogonal and stackable.
## Docs / external references
- none beyond the in-tree LLK sources above.
