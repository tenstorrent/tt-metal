# r05-b04-a01 context

## Files read
- `agent_orch/WORKER.md`, `campaign.yaml`, `history.md`: rules, metric, the 57 earlier nodes.
- `device/unified_routed_expert_ffn_program_factory.cpp`:
  - L1 fit, ring depths, semaphores, NoC/RISC assignment (IN1_WRITER_MCAST, UP_SPLIT, DOWN_SPLIT);
  - the per-layout group topology args, i.e. the sender row `group_base + gx % rows`;
  - the x-sender collision TT_FATAL.
- `kernels/dataflow/unified_routed_expert_ffn_reader.cpp` (all):
  - gate/up loop: x owner, gate read, look-ahead, acks;
  - phase-4 down loop: in1_down sender, act sender, look-ahead;
  - the pre-down writer-multicast drain.
- `kernels/dataflow/unified_routed_expert_ffn_writer.cpp` (all): UP_SPLIT + NoC-1 gate/up multicast (rectangle corners
  swapped for NoC 1), DOWN_SPLIT half-read, output drain.
- `kernels/adaptive_chunk.hpp` (all): layouts, planner, x-owner rotation helpers.
- `kernels/compute/fused_swiglu.cpp`: gate/up K-loop (owner tilize one block ahead, waits), compute-side x-owner use.
- `kernels/compute/re_hwprof.hpp`: RE-HW record layout (MATH counters per GU/DN window).
- `models/demos/deepseek_v3_d_p/tt/moe/tt_routed_expert.py::routed_expert_weight_memory_config`: ND shard =
  1 tile-row x per_core_N, round-robin over banks.
- The test: shapes, routing, how the log reports weight GB/s.

## Nodes consulted
Every node's reflection (all 57) and most proposals. Most relevant here:
- r04-b03-a01: the RE-UW UNPACK wait tallies; the only data with per-core weight-wait times, which this node's
  parity / wrap analysis uses.
- r04-b04-a01, r04-b04-a02, r04-b02-a02, r04-b03-a02: handshake and ack decoupling don't change the block period.
  This node changes the multicast path, not the timing.
- r03-b02-a02: x owner moved off weight-sender cores, measured neutral. This node gives that back on the bottom row.
- r01-b02-a02: the staggered sender placement (RISC collisions), which this node replaces for the row-group
  topologies.
- r02-b02-a02 (request VCs), r02-b04-a02 / r03-b03-a02 (whole weight stream on one NoC: -8..-20%): this node keeps
  every stream on the NoC and RISC it uses today.
- r04-b02-a01: parent / round root (1.6275), whose profile is used for the tail analysis.

## Analysis (scripts in analysis/, run on reports/<node>/.logs/profile_log_device.csv)
- `rows.py`, `grp2.py`, `strag.py` on r04-b02-a01: per-row and per-call ends.
  - Per-call max - median core is 2-90 µs, very uneven call to call (glm: 5 5 5 86 85 ...).
  - Whole groups have slow phases: glm phase-B GU 142 vs 88 µs; mistral DN 73-110 vs ~45 µs.
- `slowuw.py` on r04-b03-a01: in a slow group the extra time is the ROW stream wait (x in GU, act in DN), on both
  rows of the group.
- `wrap.py` on r04-b03-a01: receivers' weight wait split by whether their column multicast wraps. The sender row
  isn't the rectangle's NoC start corner: bottom for NoC-1 gate/up, top for NoC-0 down.
  - gate/up: 1.4-4.3x longer when wrapping, on all 5 shapes with a known layout.
  - down: 2-11x longer when wrapping.
- `collate.py`: per-column lateness. No single hot column beyond a few µs (x14 slightly late in DN on h7168).

## Local compile checks (not committed)
- r04-b04-a02's NCRISC/BRISC harness (`build.sh`) against this worktree vs HEAD:
  - reader 14296 -> 14212 B, writer 11520 -> 11508 B (-Os, profiler build);
  - the TILE-x, bias and no-writer-mcast variants also compile.
- The 3-TRISC compile (`kc.sh` with r04-b04's JIT chlkc files retargeted to this worktree) compiles.

## Docs / external references
- tt-metal NoC conventions as used in this op's own comments:
  - NoC 0 routes +x then +y;
  - NoC 1 routes -x / -y, so its rectangle corners are given in reverse order;
  - the worker grid is a 2D torus.
