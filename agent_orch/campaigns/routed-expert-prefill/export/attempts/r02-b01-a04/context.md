## Files read
- agent_orch/WORKER.md, campaigns/routed-expert-prefill/campaign.yaml — rules
- compute/fused_swiglu.cpp (parent) — PACK_ACT_* macros, pipelined last-block activation
- r02-b03-a03 compute/binary_act_sfpu.h and its fused_swiglu.cpp diff — op-local SFPU bodies, MATH-thread macros, plan_launch mailbox
- tt-llk BH llk_pack.h, tilize_helpers.inl, api/compute/tilize.h — pack replay use only in (fast-)tilize inits, re-recorded per init
- tt-llk race-audit-all skill doc / ckernel_structs.h — replay buffer is 32 entries per thread
## Nodes consulted
- all 28 nodes' reflections (history.md + git show)
- r02-b01-a03 (parent) — pipeline, next step #1 = combine with mixed-height layouts
- r02-b03-a02/a03 — schedule combo + short SFPU bodies (best 1.5435)
- r02-b04-a01, r02-b02-a01 — mixed-height layouts
- r02-b02-a03, r02-b04-a02 — what not to do with down stream / DRAM timing
## Docs / external references
- none
