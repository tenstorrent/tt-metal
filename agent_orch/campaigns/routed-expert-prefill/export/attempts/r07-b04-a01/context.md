## Files read
- ttnn/.../unified_routed_expert_ffn/device/kernels/adaptive_chunk.hpp — the launch planner. `schedule_in_order`
  (uniform single-phase deal), `GroupScheduler::assign` (model's deal), `schedule_lpt` (mixed), `make_plan`
  (two-phase), and `plan_launch`. Ties go to the lowest group index = top grid rows.
- reader.cpp — `plan_launch` call site; the owner[] array is consumed generically (count-0 skip for other groups).
- program_factory.cpp — the host builds per-layout topologies and doesn't replicate the deal, so no host change.
- hybrid_routed_expert_ffn kernels — own adaptive_chunk.hpp copy, unaffected.
## Nodes consulted
- r06-b04-a01 (parent, best 1.6931): reflection, eval; per-row ends recomputed from its profile.
- r06-b01-a01, r06-b02-a01: the row-gradient measurements and the "use it in the planner" recommendations.
- r06-b03-a01: down-only DRAM endpoint remap idea (not taken here).
- r05-b01-a01 / r04-b03-a03: the idle-row tie pass (layout change for mm3 / mistral), not on this lineage.
- r05-b02-a01: next #1 rebalance proposal (kimi-k3 5/5/6/6/6).
- r05-b03-a01, r05-b04-a01: x(0) arrival gradient by row, wrap diagnosis.
- history.md: the whole index for rounds r01-r06.
## Tools
- /tmp/r06b01a01/ends.py (per-row kernel end from the device profile CSV).
- /tmp/r07b04/pt.cpp (copy of r06-b01-a01's native planner replay on the test's counts), base.txt / new.txt.
