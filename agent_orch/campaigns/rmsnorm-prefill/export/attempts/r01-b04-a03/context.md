## Files read
- agent_orch/WORKER.md, campaigns/rmsnorm-prefill/campaign.yaml — rules, allowed paths, accuracy gate
- kernels/dataflow/dit_rmsnorm_fused_reader.cpp — parent's `read_input_pass_pipelined(…, with_weight)`, the deferred
  broadcast-weight read, and `weight_pushed` gating
- kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp — BRISC timeline: scalar setup, then it blocks on
  stats_local (PRE end). That makes it idle for the whole input read
- kernels/compute/dit_rmsnorm_fused_compute.cpp (lineage diff) — `prescale_weight` x*gamma pass waits on
  cb_weight cumulatively (whole-row push OK); sub-phase 2 also waits cumulatively
- device/dit_fused_distributed_rmsnorm_program_factory.cpp — reader/writer CT-arg layout (accessors last),
  writer common args {output, trans_mat, stats_dram}, override_runtime_arguments, block_major_post/use_mux flags
- tests/.../test_fused_rms_norm_prefill.py — bf16 broadcast [1,1,1,H] TILE weight, no bias/RoPE
## Nodes consulted
- r01-b04-a02 (parent) — gamma interleave on NCRISC serialized the input; suggests BRISC gamma / isolate trid read
- r01-b01-a02 — same failure; confirmed early gamma cuts post-AG by 2.4 µs at h7168
- r01-b01-a01, r01-b04-a01 — x*gamma under AG wins; gamma lands late on wide shapes
- r01-b02-a02 — trid deep input read works (~400 GB/s); PRE then gates the AG; drain-flush change did nothing
- r01-b02-a01, r01-b03-a01/a02 — column split; dispatch stall from 2 kernel groups; bank-phase hypothesis
## Docs / external references
- none
