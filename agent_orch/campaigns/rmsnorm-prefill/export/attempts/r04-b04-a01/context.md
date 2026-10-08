## Files read
- agent_orch/WORKER.md, campaigns/rmsnorm-prefill/campaign.yaml — job + allowed paths
- device/kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp — W_DRAIN loop: non-posted per-tile writes, per-block flush, dual-NoC routing
- tt_metal/hw/inc/api/dataflow/noc.h — NocOptions::POSTED on async_write / async_writes_flushed
- tt_metal/hw/inc/api/dataflow/dataflow_api.h, internal/tt-1xx/blackhole/noc_nonblocking_api.h — posted path under DM_DYNAMIC_NOC (POSTED_WRITES_NUM_ISSUED counter, NIU_MST_POSTED_WR_REQ_SENT), posted drops NOC_CMD_RESP_MARKED
- device/dit_fused_distributed_rmsnorm_program_factory.cpp — dual_noc_drain flag, output_cb sizing
## Nodes consulted
- all 35 nodes' proposal/reflection (history.md + git show), in detail round 3
- r03-b02-a03, r03-b03-a03, r03-b04-a02 — drain per-core cap is not VC, cmd buf, or aggregate; posted writes named as next test
- r03-b02-a02 — parent/best; drain tail 1.1-1.9 µs after POST end
- r03-b04-a03 — gamma streaming (likely ported by a sibling branch; not repeated here)
## Docs / external references
- none
