## Files read
- agent_orch/WORKER.md, campaigns/routed-expert-prefill/campaign.yaml: the rules, metric and accuracy gate.
- $DREAM_HOME/routed-expert-prefill/history.md: the index of all 57 nodes. The round-5 root is r04-b02-a01.
- Every node's reflection and proposal (extracted with git show into /tmp/r05b03-a01-nikola/{refl,prop}.md). I read
  rounds 3-4 in full and rounds 1-2 for their conclusions.
- tests/.../test_routed_expert_prefill.py, models/.../tt_routed_expert.py: shapes, LoFi config (bf16 dst, packer
  L1 acc), and the weight ND-shard spec (one tile-row x per_core_N, round-robin over banks).
- kernels/adaptive_chunk.hpp: layouts (10 / 5+5 / 5x2 / 3322 / 433 / 4222).
  - `x_owner_slots / x_owner_rank / x_owner_col`: block j goes to slot j % slots, which is independent of the
    grid row, so every row's owner of block j is in the same 1-2 columns.
- kernels/dataflow/unified_routed_expert_ffn_reader.cpp:
  - start-up: counts multicast, plan, up-front x(0)/x(1) stick reads by the owners;
  - the gate/up K-loop: x owner reads 2 blocks ahead on NoC 0, the owner multicasts after its tilize and all 10
    row acks, weight look-ahead;
  - the receivers' x / weight waits.
- kernels/dataflow/unified_routed_expert_ffn_writer.cpp: UP_SPLIT / DOWN_SPLIT reads and the writer multicast.
  The writer of a non-sender core is idle in gate/up. I considered and rejected a two-RISC x read for this node.
- kernels/compute/fused_swiglu.cpp: matmul_phase_fused_gu (owner tilize one block ahead, x_slot from x_blk_base),
  and the UNPACK -> MATH/PACK per-expert mailbox word (group rows, row in group).
- hybrid_routed_expert_ffn/device/kernels: it has its own adaptive_chunk.hpp / fused_swiglu.cpp copies, so this
  change does not touch the hybrid op.
- tt_metal/soc_descriptors/blackhole_140_arch.yaml: DRAM at physical x = 0 / 9. These are the NoC-0 / NoC-1 worker
  endpoints per channel, used to reason about where NoC-0 read responses travel.
- tt_metal/hw/inc/internal/tt-1xx/blackhole/noc_nonblocking_api.h: read-completion counters are per NoC, not per
  RISC. That is why I did not move x reads to the reader's NoC 1.
- tt_metal/hw/ckernels/.../experimental/llk_math_custom_mm*.h: the decode custom_mm. It supports only 1-8-row in0
  tiles and accumulates in a different order, so it is not usable here.
- tests/scripts/test_moreh_microbenchmark.py: the BH DRAM read benchmark bound is 490-512 GB/s. This op moves
  ~450-466 GB/s total on the big shapes.

## Measurements made while choosing (scripts in /tmp/r05b03-a01-nikola/ana)
- rows.py / zones2.py on reports/r04-b02-a01: per-row GU is systematically longer on the top rows (mistral y2-3 120
  vs y8-9 93 µs, mm3 242 vs 211, dsv4-flash 190 vs 163). DN is flat across rows.
- calls.py: per call, mistral's slowest group is y2-3 in 6 of 10 calls; mm3's is y4-5 in 6 of 10.
- uwrow.py on reports/r04-b03-a01 (UNPACK x-wait per row): the GU-window gap between rows equals the x-wait gap
  (mistral y2 x-wait 39 vs y8 21 µs).
- fill2.py on reports/r03-b04-a02 (RE-GW): the block-1 read starts (≈ x(0) arrival) at y2 ~37 µs vs y9 ~19 µs.
- skew.py: kernel launch skew across cores is < 0.1 µs, so it is not the cause.
- owners.py: rows per owner column for a given block. Base: 5.0 (5x2), 7.1 mean / 9 max (10-row). With
  x_off = group + 2 * row-in-group: 1.0 (5x2), ~2 elsewhere.
- Local compiles (/tmp/r05b03-a01-nikola/build.sh, kc.sh from r04-b04-a02):
  - reader +36 B non-profiler, +292 B with the RE-X0 zone;
  - TRISC0/1/2 +24/+20/+16 B.

## Nodes consulted
- r04-b03-a01 (UNPACK wait tallies: compute waits on the row x stream), r04-b04-a01 / r04-b02-a02 / r04-b04-a02
  (handshake decoupling is null or redistributive), r03-b04-a02 (x read timing not late in steady state; RE-GW
  zones), r03-b04-a03 (first-block fill, row-group stagger y8 first / y2 last), r03-b02-a02 (owner off the weight
  sender: neutral), r03-b03-a02 (both NoCs needed for weight reads).
- r04-b03-a03 (tie pass, -Os UNPACK planner cost) and r03-b01-a02 (plan once per core): the obvious
  combination, left to other branches.
- r04-b04-a02, r04-b01-a02, r04-b04-a01: kernel-config-buffer margins on the h7168 programs (~0.9 KB on kimi-k2.7
  for r04-b02-a01).
