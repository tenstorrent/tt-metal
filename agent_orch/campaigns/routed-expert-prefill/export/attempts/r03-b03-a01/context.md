## Files read
- agent_orch/WORKER.md, campaigns/routed-expert-prefill/campaign.yaml: the worker protocol and the
  campaign rules (precision, all paths, allowed_paths).
- device/kernels/adaptive_chunk.hpp (whole file): the launch planner. `plan_launch` -> `make_plan`
  (`pick_group_rows`, thresh loop, leftover walk, 2 x `model_phase`, phase-B candidates) ->
  `plan_groups` (3 uniform `schedule_in_order` + 3 mixed `schedule_lpt` + owner pass). Every pass
  calls `tiles_of`, and the model passes run `expert_cost` / `num_chunks` (divides).
- device/kernels/dataflow/unified_routed_expert_ffn_reader.cpp (prologue through the phase loop):
  COUNTS_BCAST, the `start` page read, and `tiles_of` (two volatile L1 loads + band + clamp). The plan
  runs before the first weight / x read.
- device/kernels/dataflow/unified_routed_expert_ffn_writer.cpp (plan call): the same plan, before the
  `up` reads.
- device/kernels/compute/fused_swiglu.cpp (expert_count_tiles, UNPACK plan + mailbox loop): UNPACK
  plans and then mailboxes per-expert (count, rows) to MATH/PACK. Its `tiles_of` also re-reads two CB
  interfaces per call.
- tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h: TRISC local memory is 4 KB (stack
  headroom for the per-expert cache).

## Nodes consulted
- Every node of rounds 1 and 2: proposal, score and reflection (dumped to /tmp/r03b03/all.md).
- r02-b01-a04: measured first-GU-start of 29.2 µs on kimi-k3 and 10-11 µs on dsv3/glm/k2.7/dsv4 with the
  b03 planner, vs 6.4 / 3.2-3.9 with the old picker. Its next #1 is "cut the planner prologue".
- r02-b03-a02: median-core shift (+21 µs kimi-k3, +10 µs dsv4-flash) attributed to make_plan +
  plan_groups. Its next #1 is the same trim (reject before modelling, cache tiles, reuse the
  uniform pick).
- r02-b03-a04: its zones give the root-like first-GU-start values (I re-ran /tmp/r02b01a02/zones.py
  on its profile): k3 29.0, dsv3/k2.7/glm/dsv4f/dsv4p 9.8-10.9, mistral/mm3/gpt-oss 4.6-5.4 µs.
- r02-b01-a02: the start fold. Prologue outside the planner is only ~1 µs (RE-PRO).
- r02-b02-a04 (root): the mixed phase-B scoring in make_plan, which this node keeps exactly.
- Round 3 in progress: r03-b01 and r03-b02 are both porting the PACK-thread activation pipeline
  (r02-b03-a04) onto the root, so this node takes an orthogonal mechanism.

## Tools / measurements
- JIT ELF disassembly of the reader (campaign jitcache, riscv-tt-elf-objdump): BH has hardware
  divu/remu. The planner code holds 43 of them.
- /tmp/r03b03a01/{old,new}.cpp + main.cpp: a host fuzz of the root's plan_launch vs this node's.
  0 mismatches on 4 x 300k random cases (10- and 8-row grids, 16- and 32-bit counts). A planted bug
  gives 4224 mismatches out of 50k cases, so the check has teeth.
- /tmp/r03b03a01/gc/est.sh: gcov line / tiles_of / divide counts of the planner on the test's seed-0
  routing (vectors from /tmp/routed_r02b02a04/pt.cpp).
  - k3: 13270 lines, 656 tiles_of, ~820 divides -> 10686 lines, 28 tiles_of, 2 divides.
  - dsv3: 2967 / 104 / 267 -> 2601 / 8 / 2.
  - A cost model (1.75 cycles per line, 20 per tiles_of, 5 per divide), calibrated against the
    measured first-GU-start, predicts planner time k3 30 -> 14 µs, dsv3/glm 6.4 -> 3.5,
    k27/dsv4p ~9 -> ~4.8, dsv4f 9.9 -> 5.0, 4-expert shapes 2.5 -> 1.4.

## Docs / external references
- none beyond the code.
