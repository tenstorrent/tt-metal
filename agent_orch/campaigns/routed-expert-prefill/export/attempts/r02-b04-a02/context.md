## Files read
- agent_orch/WORKER.md, campaigns/routed-expert-prefill/campaign.yaml — job definition, rules, allowed paths.
- device/kernels/adaptive_chunk.hpp — row-group layouts + LPT plan (parent r02-b04-a01); replayed the
  test routing with /tmp/r02b04a01/plan.py (copied into /tmp/r02b04a02/sim.py with a fixed-cost knob F):
  raising the cost model's per-chunk constant to 2-3 only moves dsv4-pro back to 2x5 (measured equal),
  so recalibrating F alone buys nothing.
- device/kernels/compute/fused_swiglu.cpp — phase structure per chunk: fused gate/up K-loop
  (packer L1_ACC, per-block drain), activation (SFPU on MATH), down K-loop over num_blocks_d blocks of
  in0_block_w_d = per_core_N_gu K-rows.
- device/kernels/dataflow/unified_routed_expert_ffn_reader.cpp — gate/up loop (x owner tilize +
  multicast, gate read, IN1_WRITER_MCAST hand-off) and phase-4 loop. Key finding: the down-weight
  delivery shared each K-block iteration with the activated multicast, so a receiver acked down block kb
  only after block kb-1's act_valid; during the activation phase only down block 0 could be in flight.
- device/kernels/dataflow/unified_routed_expert_ffn_writer.cpp — UP_SPLIT up reads, IN1_WRITER_MCAST
  gate/up multicast on NoC 1, DOWN_SPLIT half reads, cb_out drain.
- device/unified_routed_expert_ffn_program_factory.cpp — semaphore allocation (13 sems), CB ring depths
  (gu_slots / d_slots 2-3), runtime-arg layout (DOWN_SEM_RT slots 34/35 reader, 6/7 writer), per-layout
  topology sets {is_sender, n_recv, rect, sender xy}.
- tt_metal/hw/inc/internal/tt-1xx/blackhole/noc_nonblocking_api.h — NOC_0_X(noc, size, x) = x on
  Blackhole, so a NoC-1 unicast semaphore inc can use the host's worker coordinates as is.
- tt_metal/hw/inc/hostdev/profiler_common.h — 250 optional markers per RISC per program run, so 3 zones
  per expert on MATH (<= 21 on kimi-k3) are safe.
- hybrid_routed_expert_ffn/device/hybrid_program_factory.cpp — the hybrid op compiles its OWN copies
  of the reader/writer/compute kernels, so this change does not touch it.
- tests/.../test_routed_expert_prefill.py — weights are DRAM ND-sharded (coalesced runs per K-row), bf4.

## Nodes consulted
- All r01 and r02 reflections/proposals (history.md + git show of every tag).
- r02-b04-a01 (parent) — measured per-expert fixed cost F ~150 µs on gpt-oss; F ~2U on dsv4-pro.
- r02-b02-a01 — same fixed cost on gpt-oss (~70 µs per removed expert).
- r02-b01-a01 / r02-b03-a01 — few large groups are slower per expert than many small ones (stream
  rate per group is limited), consistent with weight delivery latency in the group pipeline.
- r01-b03-a01 / r01-b04-a01 — read-ahead before the multicast on the same RISC regressed; this node
  keeps read -> multicast order per block and decouples the stream from the activated chain instead.
- r01-b04-a03 — 3-slot rings neutral; here the down ring could not fill because of the act coupling.
- r01-b03-a03 / r01-b01-a04 — unit-boundary feeds neutral (different mechanism).
- r01-b02-a04 / r01-b01-a04 — requested per-phase DeviceZoneScopedN measurements (added here).

## Derived numbers (my estimates, not measured)
- Per pcM unit LoFi matmul (16 cyc/tile-mult, 1.35 GHz): mistral/dsv4-flash ~27.6 µs, minimax-m3 ~62 µs,
  gpt-oss ~29 µs. F = measured - U*pcM: mistral ~90, dsv4-flash ~95, minimax-m3 ~175, gpt-oss ~150 µs.
