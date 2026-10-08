## Files read
- agent_orch/WORKER.md, campaigns/rmsnorm-prefill/campaign.yaml — job definition, metric, allowed paths, accuracy gate.
- device/kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp — BRISC: gamma issue loop with in-loop stick-push poll,
  then a BLOCKING gamma read barrier (the gap this node closes); push_stick handshake; go wait; stick reads; drain.
- device/kernels/compute/dit_rmsnorm_fused_compute.cpp — PRE (DST-accumulated x*x, ones*S^T matmul -> row-0 stat),
  then x*gamma (waits weight_cb), so gamma is only needed after the stat is produced.
- device/kernels/dataflow/dit_rmsnorm_fused_reader.cpp — trid-pipelined input read (block pushes, lookahead 4).
- device/dit_fused_distributed_rmsnorm_program_factory.cpp — output_cb is 2 rows, block_size = 4 (fp32 dest),
  writer/reader in DM_DYNAMIC_NOC, forwarder kernel lives in dit_fused_norm_common (outside allowed_paths).
- ttnn/.../dit_fused_norm_common/kernels/dataflow/dit_fused_norm_forwarder.cpp — F_COLLECT waits for all 20 sticks.
- tt_metal/hw/inc/api/dataflow/dataflow_api.h (noc_async_read_barrier, cb_pages_available_at_front),
  tt_metal/hw/inc/internal/tt-1xx/blackhole/noc_nonblocking_api.h (ncrisc_dynamic_noc_reads_flushed: NIU
  RD_RESP_RECEIVED vs both RISCs' L1 issue counters) — the poll uses exactly the barrier's check.
- tt_metal/fabric/hw/inc/edm_fabric/fabric_edm_packet_transmission.hpp — the fabric's flush=true fused write+inc uses
  "write flushed, then atomic on same noc/vc" ordering (considered for a push-handshake idea, not used here).
- tt_metal/tt-llk/tests/python_tests/helpers/golden_generators.py (_apply_fidelity_masking) — BH HiFi2 drops bf16
  srcB's last mantissa bit (considered PRE HiFi2; emulated sum bias -0.28%, max_abs +0.007; not used here).

## Nodes consulted
- All 31 nodes' reflections (r01..r03). Key ones:
- r01-b04-a03 / r01-b04-a04 — gamma read moved to BRISC; in-loop stick preempt poll ("never fired" then).
- r02-b02-a01 — drain path-aware dual NoC; drain is now ~DRAM-rate (~380 GB/s aggregate by my estimate).
- r03-b04-a02 — waves fail because per-core phases; its "drain per-core bound" claim is weak (A's tail overlapped B).
- r03-b03-a02 — stat-in-DST did not make the stat earlier; suggested measuring the PRE tail.
- r03-b01-a02 (parent) / r03-b02-a02 — L1 scratch; post-go stick read ~0.6 us is not memory latency.

## Analysis scripts (this dir)
- pre.py — per-shape AG-start path: R_INPUT end, push start/end, slowest pusher, post-AG timings.
- gam.py — classifies each core-call's W_PUSH start: in the gamma issue loop / right after W_GAMMA end / later.
- rin.py — per-core R_INPUT / W_PUSH / W_DRAIN ends by grid position.
- xchip.py — per-call per-chip F_FABRIC duration and kernel end (AG skew).
Run as `python3 X.py $DREAM_HOME/rmsnorm-prefill/reports/<node>`.

## Docs / external references
- tenstorrent/tt-metal#58723 (via Glean): BH ELWMUL math cycles/tile LoFi 16.8, HiFi2 34.6, HiFi3 58.6, HiFi4 82.6.
- tech_reports/matrix_engine/matrix_engine.md — fidelity phase bit split.
