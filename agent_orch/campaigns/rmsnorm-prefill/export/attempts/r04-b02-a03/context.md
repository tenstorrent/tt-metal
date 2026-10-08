## Files read
- agent_orch/WORKER.md, campaigns/rmsnorm-prefill/campaign.yaml — job definition, allowed paths, accuracy gate
- device/kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp (this lineage, merged) — checked that the push, gamma
  stream and posted drain diffs sit correctly next to the gather_mcast branches (slot offset, reserve-before-go)
- device/kernels/compute/dit_rmsnorm_fused_compute.cpp — PRE block (HiFi2 port), POST gather indexing untouched
- git diff r03-b02-a02..r04-b04-a02 (writer), r03-b02-a02..r04-b01-a02 (compute) — the code being ported

## Nodes consulted
- history.md index for all 43 nodes. Full proposal/reflection for every round-4 node:
  - r04-b02-a01/a02 (my lineage): multicast release; POST is earlier but synchronized drains absorb it.
  - r04-b04-a01/a02: posted drain plus the stack of all three writer wins (best, 1.3997).
  - r04-b03-a01/a02: ack-free push.
  - r04-b01-a01/a02: streamed gamma, PRE HiFi2.
- Earlier rounds (via history.md "next" columns and the r03 reflections quoted in round 4): the dual-NoC,
  cmd-buf, VC and placement drain variants are ruled out; two-wave AG was flawed.

## Docs / external references
- none beyond the code
