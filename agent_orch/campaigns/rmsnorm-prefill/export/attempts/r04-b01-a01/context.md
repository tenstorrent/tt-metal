## Files read
- agent_orch/WORKER.md, campaigns/rmsnorm-prefill/campaign.yaml — job definition, allowed paths, gate.
- dit_rmsnorm_fused_worker_writer.cpp — W_GAMMA block (whole-row gamma under one barrier on HEAD); verified HEAD's
  writer is identical to r03-b04-a01's writer so r03-b04-a03's diff applies unchanged.
- dit_rmsnorm_fused_compute.cpp — x*gamma pre-pass waits on weight_cb cumulatively per block (line ~385), so chunked
  pushes are consumed incrementally; POST waits on the absolute col index (resident weight).
## Nodes consulted
- All r01-r03 reflections (history.md + git show). Key ones:
- r03-b04-a03 — the gamma streaming mechanism, straggler-loop diagnosis, its #1 recommendation (this port).
- r03-b02-a02 — the base (best, L1 stats scratch), still has the dev-0 h7168 straggler 9/10 calls.
- r03-b02-a01 / r03-b03-a01 / r03-b04-a01 — combine variants; confirms the base's compute is the fastest combine.
- r03-b02-a03, r03-b03-a03, r03-b04-a02 — drain mechanisms that failed (cmd bufs, VCs, waves): not revisited.
## Docs / external references
- none
