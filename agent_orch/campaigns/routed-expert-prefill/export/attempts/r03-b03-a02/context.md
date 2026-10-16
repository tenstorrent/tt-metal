## Files read
- agent_orch/WORKER.md, campaign.yaml: the job, the allowed paths, the accuracy gate.
- tests/ttnn/nightly/unit_tests/operations/fused/test_routed_expert_prefill.py: shapes, routing (seed 0).
  - Per-expert tile counts (/tmp/r03b03a02/route.py):
    - mistral / minimax-m3: [5, 6, 6, 5].
    - gpt-oss: [8, 9, 9, 8].
    - dsv4-flash: [5, 4, 5, 5, 4, 4, 4, 4].
    - dsv3 / glm: 8 experts of 5-6 tiles.
    - k2.7: 12 experts of 3-5 tiles.
    - dsv4-pro: 12 experts of 3 tiles.
    - kimi-k3: 28 experts of 3-4 tiles.
- kernels/dataflow/unified_routed_expert_ffn_reader.cpp (whole file).
  - Gate/up K-loop: the row-major x owner multicasts only after its own weight-read barrier and the
    up_done wait.
  - Row cores wait in0_valid before starting their next block.
  - The DN loop is serial per block: read -> mcast -> activated mcast.
- kernels/dataflow/unified_routed_expert_ffn_writer.cpp (whole file).
  - UP_SPLIT + IN1_WRITER_MCAST loop: up(kb) read -> barrier -> mcast_go -> in1_ready -> multicast
    -> next up read. So no up read is in flight during a multicast.
- kernels/weight_runs.hpp: one NoC transaction per K-row slice (ND-shard coalescing).
  - The disassembly of the writer's read loop (jitcache brisc.elf) is ~45 instructions per read, so
    issuing 16 reads costs ~0.7 µs per block.
- kernels/compute/fused_swiglu.cpp, matmul_phase_fused_gu: the per-block wait x/gate/up -> matmul ->
  packer drain -> pop order, and the owner's one-block-ahead tilize.
- unified_routed_expert_ffn_program_factory.cpp:
  - up_mode / UP_SPLIT / IN1_WRITER_MCAST / DOWN_SPLIT selection.
  - L1 fit and ring depth. A replica in /tmp/r03b03a02/fit.py says most shapes get 3-slot gate/up
    rings; mm3 and dsv4-pro get 2.
  - Reader and writer compile-time args.
- tt_metal/tools/profiler/kernel_profiler.hpp: DeviceTimestampedData exists. DO_SUM zones need a
  profiler option the eval does not set, so I used my own wall-clock counters plus one TS record per
  counter pair.
- tt_metal/hw/inc/internal/tt-1xx/blackhole/noc_nonblocking_api.h: a write multicast issue loops over
  16 KB bursts and waits for the command buffer on each.
- tt_metal/jit_build/build.cpp and fake_kernels_target/CMakeLists.txt: the JIT flags, used for an
  offline compile harness (/tmp/r03b03a02/cc/build.sh).
  - Reader and writer compile in profiler and non-profiler builds, on the row-major, TILE, bias and
    DS_NO_WRITER_MCAST paths.
  - Size delta vs parent: profiler build reader +1.5 KB, writer ~0. Non-profiler build reader
    +0.2 KB, writer -0.4 KB.

## Nodes consulted
- All 36 nodes' reflections, from history.md and git show; proposals for the read-ahead nodes.
- r03-b03-a01 (parent): planner cache, same plan. Its RE-PLAN NCRISC zone is kept.
- r03-b02-a01 (best): its MATH zones are the source of the per-block cycle analysis. Run with
  /tmp/r03b03a02/percore.py on reports/r03-b02-a01.
  - mistral GU ~110 µs over 8 blocks, DN ~50.
  - gpt-oss GU 156-190 by group.
- r03-b04-a01 next #3: asks for exactly this DM wait breakdown; F ≈ 100 µs per expert is not matmul.
- r01-b03-a01 / r01-b04-a01: read-ahead issued BEFORE the multicast on the multicasting RISC, root
  schedule; both lost. This node never puts a read in front of a multicast on the same RISC.
- r02-b04-a02: moving down reads to NoC 1 and serialising read + multicast on one RISC lost 10-22%.
  The gate/up reads here stay on NoC 0 (reader).
- r02-b02-a03, r01-b04-a03: deeper rings / down look-ahead, neutral; not a ring-depth change.
- r02-b01-a01, r02-b03-a01: aggregate weight bandwidth vs number of concurrent group streams
  (1 group ~250, 2 ~340-390, 5 ~440 GB/s). The 4-group shapes sit at ~340.
- r02-b04-a03: the "group-tail lottery" is a gate/up delivery effect, sometimes one row of a group.
  That is consistent with the per-row x coupling found here.

## Docs / external references
- None beyond the tt-metal sources above.
