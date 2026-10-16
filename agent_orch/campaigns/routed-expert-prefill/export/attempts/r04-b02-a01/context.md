## Files read
- agent_orch/WORKER.md, campaigns/routed-expert-prefill/campaign.yaml: the rules, metric and accuracy gate.
- $DREAM_HOME/routed-expert-prefill/history.md: the index of all 48 nodes in rounds 1-3.
- kernels/dataflow/unified_routed_expert_ffn_reader.cpp: the whole gate/up K-loop (sender read, IN1_WRITER_MCAST
  hand-off, UP_SPLIT `up_go`/`up_done`, `mcast_done` slot reuse, x owner multicast, receivers' step 3) and the down
  K-loop.
  - The sender's next gate read and `up_go` are issued only at the top of the next iteration, after the sender
    core's own `in0_valid` (x block kb) wait. That couples the column's weight stream to the sender row's x stream.
- kernels/dataflow/unified_routed_expert_ffn_writer.cpp: the UP_SPLIT read + IN1_WRITER_MCAST loop.
  - It is serial per block: `up_go` -> read up -> `up_done` -> `mcast_go` -> `in1_ready` -> multicast ->
    `mcast_done`.
  - The slot index is counter-based (`(gu_blk + kb) % IN1_GU_SLOTS`), so an earlier `up_go` needs no writer change.
- kernels/weight_runs.hpp: the coalesced ND-shard reads, one transaction per K-row.
- program_factory.cpp:
  - ring depths `gu_slots` / `d_slots` (2 or 3 depending on L1);
  - `kWriterMcastsIn1` (on unless an env var is set) and UP_SPLIT (on), so every benchmark shape runs the
    writer-multicast + UP_SPLIT mode;
  - `kEnableSplitDown`.
- tt_metal/hw/inc/api/semaphore.h, semaphore_dm_impl.h: `value()` is a fresh read (fence + load), so it is usable
  in a spin.
- dataflow_api.h: `cb_push_back` wraps `fifo_wr_ptr` at `fifo_limit`, so `get_write_ptr()` after a push is the
  next slot. `cb_pages_reservable_at_back` is non-blocking.
- tt_metal/soc_descriptors/blackhole_140_arch.yaml: DRAM has 3 NoC endpoints per channel, but workers use one per
  NoC, and the noc0 endpoint is pinned to avoid SYS-1419. I rejected endpoint spreading as too risky.
- models/demos/deepseek_v3_d_p/tt/moe/tt_routed_expert.py: the weight ND-shard spec (one tile-row high,
  per_core_N wide, round-robin over banks).

## Nodes consulted
- Every node's reflection in rounds 1-3 (extracted with git show into /tmp/r04b02/refl.md).
- r03-b04-a04: the down look-ahead. I applied its diff verbatim (its parent's dataflow files equal this root's)
  and turned the look-ahead into a lambda that is re-tried while spinning on `act_valid` (its next #1).
- r03-b04-a02, r03-b03-a02: the gate/up block period has 3-8 µs that is neither the sender's read nor matmul; the
  sender waits for x 3-4 µs per block. Also: never put a whole weight stream on one NoC.
- r01-b03-a01, r01-b04-a01: read-ahead regressions came from delaying the critical multicast. Here no read is
  issued before a pending multicast on the same RISC.
- r02-b02-a03: look-ahead across the activation gap widens the group race, so down block 0 keeps its order.
- r03-b02-a03 (root), r03-b01-a04, r03-b02-a04, r03-b03-a04: the compute side is settled (16 FPU + ~15
  non-FPU cycles per mult, not unpack volume), so this node stays on the DM side.

## Docs / external references
- Local compile harness /tmp/rep-routed-r03b03a02/cc/build.sh (from r03-b03-a02; mistral-like CT args). The
  reader compiles in every mode: RM/TILE x, bias, LEGACY up, no writer multicast. Profiler-build reader .text:
  13028 B (root), 13364 B (down port only), 14296 B (this node).
