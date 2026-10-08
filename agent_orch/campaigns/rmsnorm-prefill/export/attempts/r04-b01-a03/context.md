## Files read
- agent_orch/WORKER.md, campaign.yaml: process, allowed paths, accuracy gate.
- history.md: every node's mechanism/score/next; round 4 is the relevant frontier.
- `git diff n/r04-b01-a02 n/r04-b04-a02 -- ttnn`: only two files differ. The compute file differs by the HiFi2 PRE edit.
  The writer differs by exactly r04-b03-a01's flush-then-inc push and r04-b04-a01's posted drain. Both lineages
  inherit r04-b01-a01's streamed-gamma writer, so the port is a clean file checkout.
- dit_rmsnorm_fused_worker_writer.cpp (r04-b04-a02 version): push_stick lambda, poll_stick in the gamma loop, drain loop,
  kernel-end barriers.
## Nodes consulted
- r04-b01-a02 (parent): HiFi2 PRE win, profile of the PRE tail; #1 suggestion = this stack.
- r04-b04-a02 (best): writer wins stacked, near-additive except h4096.
- r04-b03-a02, r04-b03-a01, r04-b04-a01: the individual writer mechanisms and their measured effects.
- r04-b01-a01: streamed gamma (shared base of both lineages).
- r04-b02-a01/a02: forwarder multicast release; lesson that synchronized drains contend.
- r03-b03-a02: DST-resident PRE stat neutral; PRE tail was the math, which HiFi2 then confirmed.
- Rounds 1-3 via history.md tables (dual-NoC drain variants, column split, placement: all superseded or closed).
## Docs / external references
- none
