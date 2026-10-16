## Files read
- agent_orch/WORKER.md, campaign.yaml — rules, allowed paths, gate.
- history.md — full index of r01-r06.
- reflections of every node (r01-r06, via git show on the tags), proposals of r05/r06 nodes.
- kernels/adaptive_chunk.hpp — planner, layout table, r06-b04-a01's sender_row_in_group / x_owner_key, the
  per-RISC pragma block (writer -Os planner) that conflicts with r05-b01-a01's plan-once.
- program factory (group topology loop ~l.1330-1380) — host places corner senders for EVERY layout in the
  table, so the tie pass's 3+3+2+2 choice is covered without host changes beyond the plan CB.
- reader / writer / fused_swiglu.cpp — plan_launch call sites replaced by store_plan / load_plan.

## Nodes consulted
- r06-b04-a01 (parent code, best 1.6931) — next #2 asks for exactly this port; next #4: only a few hundred
  bytes of h7168 headroom without plan-once.
- r05-b01-a01 (1.6457) — the code ported here (plan-once + tie pass): mm3 -5.9%, mistral -4.1%.
- r04-b03-a03 — tie pass origin, and its -Os UNPACK planner penalty (why plan-once, not -Os).
- r03-b01-a02 — original plan-once (latency-neutral, ~12 KB headroom).
- r05-b02-a01 — the NoC split on this lineage.
- r05-b04-a01 / r06-b04-a01 next #1 — gate/up wrap-free multicasts need reader code room; this node frees it.

## Docs / external references
- none
