## Files read
- agent_orch/WORKER.md, campaign.yaml — rules, allowed paths.
- kernels/dataflow/unified_routed_expert_ffn_reader.cpp — the phase-4 loop. The activated ack is
  sent at the top of each block, after the previous block's down weight has landed. The look-ahead
  is from r04-b02-a01.
- device/unified_routed_expert_ffn_program_factory.cpp — ring depth (d_slots 2/3), semaphores start
  at 0, act sem ids.
- kernels/compute/re_hwprof.hpp, fused_swiglu.cpp — the profiler-only MATH counters (size cut).
- tt_metal/hw/inc/api/semaphore.h, dataflow/noc.h, circular_buffer.h, dataflow_api.h — non-blocking
  primitives: `value()`, `pages_reservable_at_back`, `pages_available_at_front`,
  `ncrisc_noc_reads_flushed`.
## Nodes consulted
- r04-b03-a01 — UNPACK wait tallies. Activated is 30-58% of the DN window, the biggest wait.
- r04-b02-a01 (parent) — down look-ahead with spin re-check, next #2 = this idea.
- r04-b04-a01 — gate/up ack decoupling was neutral, and why (the x owner was not ready). The kernel
  config buffer limit is ~0.3 KB away on k2.7.
- r03-b04-a04 — down read before the ack; next #1 suggests decoupling the activated stream.
- r02-b04-a02 — the writer-owned down stream regressed because of the NoC move and the single-RISC
  serialization. Don't touch the weight delivery.
- r01-b04-a03 — deeper rings were neutral (explained by the ack protocol).
- r03-b02-a02, r03-b01-a01/a02 — kernel-config-buffer overflows and their repairs.
- r04-b01-a01, r03-b02-a03 — compute side is not the limiter.
- All reflections via /tmp/r04b02/refl.md (concatenation made by the parent's worker) and history.md.
## Docs / external references
- Local compile tooling: /tmp/r04b04/cc/build.sh (reader size), /tmp/r04b01/sz/kc.sh (TRISC size;
  a copy pointing at this worktree is in /tmp/r04b02a02w/kc.sh).
