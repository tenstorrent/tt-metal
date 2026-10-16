## Files read
- `kernels/dataflow/unified_routed_expert_ffn_reader.cpp` (parent and r04-b02-a01) — the phase-4 activated /
  down-weight handshake: receivers ack act kb+1 only after act(kb) and w(kb) landed; act data is ready at
  phase start; single 0/1 act_valid flag forces one block in flight.
- `kernels/compute/fused_swiglu.cpp` (parent, r03-b02-a02) — RE-HW counters and re_mm_block_init are the only
  diffs vs r03-b02-a02; compute is otherwise unchanged since.
- `device/unified_routed_expert_ffn_program_factory.cpp` — semaphore list (in0/act/in1 pairs), cb_in0_down_full
  ring (d_slots blocks, Bfp8_b), L1 footprint.
- `tt_metal/hw/inc/api/semaphore.h`, `api/dataflow/semaphore_dm_impl.h` — Semaphore<> carries a runtime scope;
  passing it by reference into a noinline helper bloats code, so helpers take ids.
- `tt_metal/hw/inc/api/dataflow/circular_buffer.h` — pages_reservable_at_back / pages_available_at_front.
- `tt_metal/impl/program/program.cpp` — kernel config buffer = RTAs + sems + CBs + all 5 kernel binaries.
- `tests/ttnn/nightly/unit_tests/operations/fused/test_routed_expert_prefill.py` — seed-0 routing; per-chip
  tile counts (/tmp/r04b01a02/counts.py): mistral/mm3 [5,6,6,5], gpt-oss [8,9,9,8], dsv3/glm 5-6, k2.7 3-5,
  k3 3-4, dsv4-pro all 3.

## Nodes consulted
- All 52 nodes' proposal/reflection (dumped to /tmp/rep_r04b01a02/all.md); in detail:
- r04-b03-a01 — UNPACK wait tallies: DN waits on activated 30-58% of its window; matmul probe 21-25 cyc/mult.
- r04-b02-a01 — best node; down look-ahead + spin re-check (DN -4..-30 µs), GU look-ahead neutral; its next #2
  is this node's mechanism. Its reader is ported here.
- r04-b04-a01 — gate/up early acks fired 19-34% of blocks yet changed nothing (producer not ready); kernel
  config overflow of +240 B at its run-1 reader size, used to calibrate the size budget.
- r04-b01-a01 (parent) — CFGSHIFTMASK MOP measured null; reverted for size.
- r03-b02-a03 — RE-HW counters (profiler-only, ~1.46 KB TRISC1); removed for size.
- r03-b01-a01 / r03-b02-a02 — previous kernel-config-buffer overflows and repairs.
- r02-b02-a01, r02-b04-a01 — mixed-height layouts; mistral/mm3 leave 2 rows idle.

## Tools
- /tmp/r04b01a02/build_o2.sh (from /tmp/r04b04/cc/build2.sh, -O2 like the JIT's DM build; RSRCP=<file>
  compiles another reader source): reader sizes base 16352, r04-b02-a01 18324, r04-b04-a01 run-1 19648
  (overflowed by 240 B), this node 20136 (mistral-like CTA, profiler on).
- /tmp/r04b01a02/sz/kc.sh (from /tmp/r04b01/sz) — 3-TRISC compute compile: parent 20176/14680/14992,
  this node 19500/13220/14992.
- /tmp/r04b01a02/rows.py — per-row (y) end / GU / DN per shape from the profile CSV.
- /tmp/r04b04/zones2.py — MATH zones mean / crit core.
