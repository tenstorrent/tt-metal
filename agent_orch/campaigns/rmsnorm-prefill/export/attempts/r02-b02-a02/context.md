## Files read
- agent_orch/WORKER.md, campaigns/rmsnorm-prefill/campaign.yaml: job, metric, allowed paths.
- device/kernels/compute/dit_rmsnorm_fused_compute.cpp: PRE (DST-accumulated x^2 -> pack -> reduce<SUM,REDUCE_ROW> ->
  transpose -> stats_transposed_local_cb), x*gamma pre-pass, post-AG chain. Changed: resident packed-AG PRE now
  accumulates x·x^T with matmul_tiles(transpose=1) and packs it straight to stats_transposed_local_cb.
- device/kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp: push_stick (two 64 B face-row writes + write barrier +
  arrival sem + atomic barrier), gamma-loop preemption, drain. Changed: diagonal gather + one 128 B write.
- device/dit_fused_distributed_rmsnorm_program_factory.cpp: writer/compute CT arg layout, streaming_low_l1,
  is_layernorm, use_mux (worker writer only exists when use_mux). Changed: stick_from_diag CT arg.
- tt_metal/hw/inc/api/compute/matmul.h: matmul_init(in0, in1, transpose) transposes B (in1); matmul_tiles
  accumulates into DST (DST += A*B). SDPA uses the same flag for Q·K^T.
- tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h / llk_unpack_AB.h: matmul init sets the Haloize
  (within-face transpose) bit; the eltwise AB unpack init (used by mul_bcast_rows_init for x*gamma) resets it.
- tt_metal/hw/inc/internal/tt-1xx/cache.h: invalidate_l1_cache() is a `fence` on BH (used around the BRISC gather).
- tt_metal/hw/inc/api/dataflow/dataflow_api.h: noc_async_writes_flushed semantics (only departure, not ack) - decided
  NOT to replace the stick write barrier with a flush (ordering vs the arrival atomic not documented).

## Nodes consulted
- All 20 nodes (proposals + reflections). Most relevant:
  - r01-b04-a04 / r01-b01-a04: DST accumulation of x^2; showed the PRE tail (~1-1.4 µs after the input) is a fixed
    per-row chain (reduce + transpose + handoffs), not per-tile cost. r01-b04-a04 #1 suggests removing it.
  - r02-b02-a01 (parent, 1.2385): dual-NoC drain; timeline and NoC labels (reader NoC0, writer NoC1 on BH).
  - r02-b01/b03-a01, r01-b02-a04, r01-b03-a04, r02-b04-a01: drain / placement attempts -> the drain has had 7 attempts;
    chose a different part of the critical path for diversity.

## Analysis
- tl.py (in this node dir) on reports/r02-b02-a01 (all chips, measured calls): stat ready (W_PUSH start) trails the last
  input tile by ~0.9-1.0 µs on every shape; the W_PUSH NoC part is ~0.48 µs; F_COLLECT -> F_FABRIC ~2.3-3.0 µs;
  AG end -> first drain ~0.7 µs; drain ends ~1-2 µs after TRISC.

## Docs / external references
- none beyond the in-tree sources above.
