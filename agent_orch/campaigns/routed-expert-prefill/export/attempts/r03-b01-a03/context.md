## Files read
- agent_orch/WORKER.md, campaigns/routed-expert-prefill/campaign.yaml — rules, eval, allowed paths.
- kernels/adaptive_chunk.hpp (parent and r03-b04-a01 versions) — the planner, plan sharing helpers.
- kernels/dataflow/unified_routed_expert_ffn_reader.cpp / _writer.cpp — start read, plan publish/wait.
- tt_metal/impl/program/program.cpp (opt_level defaults: DM O2, compute O3), tt_metal/jit_build/build.cpp.
- tt_metal/api/tt-metalium/kernel_types.hpp, program_descriptors.hpp — opt_level fields.

## Nodes consulted
- All 40 reflections (history.md + git show of every node's reflection.md); round 2-3 in full.
- r03-b01-a02 (parent) — plan once per core; next #1 = stack a planner trim; first-GU-start numbers.
- r03-b04-a01 — bit-identical planner trim, RE-PLAN zone measurements, rvsim + fuzz tooling.
- r03-b02-a01 / r03-b03-a01 — other planner trims, measured gains.
- r03-b02-a02 — -O2 pragma precedent on TRISC (guarded by COMPILE_FOR_TRISC), size limit history.
- r03-b01-a01 — kernel config buffer overflow (70656 B) on h7168 shapes.

## Docs / external references
- /tmp/rep-r03b04a01-rO2J/sim (rvsim.py, prog.cpp, link.ld) and fuzz.cpp, reused in /tmp/r03b01a03/{sim,fz}.
