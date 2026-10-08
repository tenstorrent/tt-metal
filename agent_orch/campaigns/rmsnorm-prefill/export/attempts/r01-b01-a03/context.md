## Files read
- agent_orch/WORKER.md, campaigns/rmsnorm-prefill/campaign.yaml — process, allowed paths, metric
- kernels/dataflow/dit_rmsnorm_fused_reader.cpp — parent's pipelined input read with interleaved gamma (issue_weight_pages, read_input_row_pipelined), deferred broadcast weight read
- kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp — BRISC timeline: scalar setup -> wait stats_local (PRE) -> stick push -> go wait -> gather read -> drain; idle from start until PRE ends
- kernels/compute/dit_rmsnorm_fused_compute.cpp (diff vs root) — pre_ag_weight x*gamma waits cb_weight cumulatively per block
- device/dit_fused_distributed_rmsnorm_program_factory.cpp — reader/writer CT args, common RT args, override_runtime_arguments
## Nodes consulted
- r01-b01-a01 — x*gamma under AG (best, 1.109); gamma late at h7168
- r01-b01-a02 (parent) — early gamma helps post-AG (-2.4 us) but interleaving it on NCRISC delays the input 3-7 us
- r01-b04-a02 — same interleave failure; ~70 ns per gamma read, same-page hot spot across 20 workers
- r01-b02-a02 — trid deep input read alone: +3.4%, R_INPUT -2.3 us at h7168; drain flush change neutral
- r01-b03-a02 — column split k=4 (1.094); bank camping hypothesis (h4096)
- r01-b02-a01, r01-b03-a01, r01-b04-a01 — col split infeasibility / dispatch-skew diagnosis / x*gamma variant
## Docs / external references
- none
