## Files read
- agent_orch/WORKER.md, campaign.yaml — task, rules, allowed paths.
- $DREAM_HOME/routed-expert-prefill/history.md — index of all 69 nodes and the leaderboard.
- kernels/adaptive_chunk.hpp — layouts, sender_row_in_group, plan_launch / PhaseDealer (phase A/B), num_chunks.
- kernels/dataflow/unified_routed_expert_ffn_reader.cpp — phase loop, gate/up K-loop (top-of-block up_go +
  issue_gate_read, try_gu_lookahead), down K-loop (release_writer_down / issue_down_read,
  try_down_lookahead), linked-multicast windows, final atomic barrier. These are the four weight-read
  issue sites the pacer gates.
- kernels/weight_runs.hpp, kernels/compute/re_hwprof.hpp — weight read helper; profiler-only MATH counters.
- device/unified_routed_expert_ffn_program_factory.cpp — semaphore list (15 of 16 used), reader defines,
  per-core runtime args (M-row NoC table, per-layout topologies), my_mt = gy, my_nt_gu = gx.
- tt_metal/impl/program/dispatch.cpp — semaphores: only the 4-byte initial value is written per launch
  (16-byte slots), so words 1-3 of a slot must be cleared by the kernel.
- tt_metal/hw/inc/api/dataflow/dataflow_api.h, blackhole noc_nonblocking_api.h — noc_semaphore_inc /
  noc_fast_atomic_increment select the 32-bit word in a 16-B slot with NOC_AT_IND_32((addr>>2)&3);
  get_semaphore = sem base + id * L1_ALIGNMENT.
- tests/ttnn/nightly/unit_tests/operations/fused/test_routed_expert_prefill.py — shapes, routing.

## Nodes consulted
- All reflections (and proposals where needed) of rounds 1-7, dumped via git show to /tmp/r08b03/.
- r07-b02-a01 (root) — per-row ends; gate/up multicast wrap fix; linked-multicast hazard (no read /
  unicast / atomic on a NoC while a linked multicast is open on it).
- r07-b04-a01, r06-b02-a01 — the gradient is relative to the active set and first-mover; static fixes fail;
  both ask for a dynamic share cap.
- r06-b01-a01, r07-b03-a01, r05-b02-a01, r06-b03-a01 — NoC byte placement moves bandwidth between rows.
- r01-b01-a03 — the earlier pacing attempt and why it lost (publisher overhead on the critical core).
- r07-b03-a01 — RE_HWPROF opt-in (~-1 KB TRISC1 in profiled builds) and kernel-config size data
  (r06-b04-a01 lineage was within ~50-100 B of the 70656 B limit; r07-b02-a01 freed ~320 B).
- r05-b01-a01, r07-b01-a01 — plan-once frees space (not used here; one mechanism per node).

## Docs / external references
- Profile of the root: reports/r07-b02-a01/.logs/profile_log_device.csv with /tmp/r06b01a01/zones.py
  (per-row GU/ACT/DN start+duration per unit).
- Local size check: /tmp/r07b02/sz/b.sh (reader -O2 profiler compile). Base reader 18928 B text,
  this node 19936 B (+1008 B; group_pacer take 368 B, begin 296 B at -Os).
- Result analysis: /tmp/r06b01a01/zones.py and ends.py on reports/r07-b02-a01 vs reports/r08-b03-a01
  (per-row GU/DN per unit and per-row kernel end, mean of the 10 measured calls).
