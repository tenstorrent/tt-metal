## Files read
- agent_orch/WORKER.md, campaigns/rmsnorm-prefill/campaign.yaml — job definition, allowed paths, gate
- $HISTORY and every node's reflection.md (31 nodes; dumped via git show into /tmp) — mechanisms tried and outcomes
- device/kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp — gamma loop (one barrier, one push), drain, stick push
- device/kernels/dataflow/dit_rmsnorm_fused_reader.cpp — trid-pipelined input read pattern (reused idea)
- device/kernels/dataflow/dit_rmsnorm_scalar_setup.hpp, ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.inl — the ~1.2 us before W_GAMMA (scalar tiles; already NoC-zeroed)
- device/kernels/compute/dit_rmsnorm_fused_compute.cpp — x*gamma pre-pass waits cumulatively per block on weight_cb; scalars waited at kernel start
- device/dit_fused_distributed_rmsnorm_program_factory.cpp — output_cb = 2 padded rows; wave plumbing (reverted)
- tt_metal/hw/inc/api/dataflow/noc.h, dataflow_api.h, internal/tt-1xx/blackhole/noc_nonblocking_api.h — trid semantics
  (TXN_ID read path sets the sticky NOC_PACKET_TAG and polls outstanding count per read; plain one-packet reads don't
  touch the tag; NOC_MAX_TRANSACTION_ID_COUNT=255 on BH; dynamic-NoC counters)
## Nodes consulted
- r03-b04-a02 (parent) — waves flawed at 20 cores; reverted
- r03-b04-a01 (grandparent) — base code; its profiler report is the evidence for the straggler loop / gamma timing
- r03-b02-a02 (best) — same dev-0 straggler pattern in its report
- r01-b04-a03, r01-b01-a03 — gamma on BRISC; r01-b01-a02 / r01-b04-a02 — don't put gamma on NCRISC
- r02-b02-a01 — cross-call coupling: late-finishing cores start late next call
## Analysis scripts (in analysis/)
- core.py <csv> <run> <dev> — per-core zone timeline of one call
- late2.py <csv> — per shape x device: last-vs-median drain end, POST-lag straggler count, gamma slack
- start.py, late.py, tl.py — kernel start skew / last core per call
- analysis outputs: late2_out.txt (this node), late2_r03-b04-a01.txt, late2_r03-b02-a02.txt, core_dev0_h7168_call.txt
