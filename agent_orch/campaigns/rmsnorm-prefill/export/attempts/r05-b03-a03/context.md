## Files read
- agent_orch/WORKER.md, campaign.yaml — rules, allowed paths, gate.
- ttnn/.../kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp — W_GAMMA streamed-gamma loop, poll_stick/push_stick,
  wave-pipeline gather; the blocking chunk barrier is the gate.
- tt_metal/hw/inc/api/dataflow/noc.h:640 — `Noc::is_read_trid_flushed` (non-blocking trid poll).
- tt_metal/hw/inc/api/dataflow/dataflow_api.h:2770 — `noc_async_read_barrier_with_trid` = same spin + invalidate_l1_cache.
- parent's analysis/push.py, wavesN.py — per-wave push/read/go timelines.
## Nodes consulted
- r05-b03-a02 (parent) — diagnosed the gamma-gated push; prescribed this repair.
- r05-b01-a02 — same 4-wave design, same gate, same prescription.
- r05-b03-a01, r05-b01-a01 — the 2-wave wins; wave B's push is also on W_GAMMA end there.
- r05-b04-a01 — fabric AG ~2.3 µs floor per round; release fan-out ~0.67 µs.
- r05-b02-a01 — PRE stat reformulations at HiFi4 are exhausted.
- all earlier reflections (Classification lines) — gamma-on-NCRISC failures (r01-b01-a02, r01-b04-a02) and the BRISC
  gamma read / streamed gamma wins (r01-b01-a03, r03-b04-a03, r04-b01-a01).
## Docs / external references
- none
