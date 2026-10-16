## Files read
- agent_orch/WORKER.md, campaign.yaml — rules, allowed paths, eval.
- kernels/dataflow/unified_routed_expert_ffn_reader.cpp — whole file; the phase-4 (down) loop is lock-step:
  ack both senders at the top of kb, then wait act_valid(kb), then in1_valid(kb). Rewritten here.
- kernels/compute/fused_swiglu.cpp — down phase consumption order (cb_in0_down_full then cb_in1_down per block),
  activation pushes cb_activated before RE-DN; probe call site (now behind RE_PROBE).
- kernels/dataflow/unified_routed_expert_ffn_writer.cpp — DOWN_SPLIT half-read protocol (down_go/down_done,
  slot = (down_blk + kb) % slots), which requires the reader to reserve/push cb_in1_down in block order: kept.
- kernels/weight_runs.hpp — reads are issue-only, caller barriers.
- device/unified_routed_expert_ffn_program_factory.cpp — ring depths (gu_slots/d_slots 2 or 3), intermed = Bfp8_b.
- tt_metal/hw/inc/api/semaphore.h, api/dataflow/semaphore_dm_impl.h — Semaphore::value() is an invalidate + volatile
  load on DM, usable for non-blocking polls; wait() is equality.
- tt_metal/hw/inc/api/dataflow/circular_buffer.h — pages_reservable_at_back / pages_available_at_front.
- tt_metal/hw/inc/api/dataflow/dataflow_api.h, internal/tt-1xx/blackhole/noc_nonblocking_api.h —
  noc_async_read_barrier = spin on ncrisc_(dynamic_)noc_reads_flushed; used as a non-blocking check.
## Nodes consulted
- r04-b03-a01 (parent) — UNPACK wait tallies: DN waits on activated 30-58% of the window, down weights 3-11%;
  matmul 21-25 cyc/mult with no wait. Its next #1 (decouple the down row stream) is this node.
- r04-b02-a01 (best) — down read look-ahead + spin re-check: DN -4..-30 µs; next #2 proposes the counting/early act ack.
- r04-b04-a01 — gate/up x/weight ack decoupling was neutral (next block's senders not ready); motivates also
  decoupling the act SEND, not only the acks, and doing it in down where the act data is already resident.
- r04-b01-a01, r03-b02-a03 — counters; LLK work is not the limiter.
- r03-b04-a04, r02-b02-a03, r02-b04-a02 — earlier down-phase weight-stream changes.
- Round 1-3 leaderboard/history (history.md) for the lineage and what not to repeat.
## Docs / external references
- none beyond the tt-metal sources above.
