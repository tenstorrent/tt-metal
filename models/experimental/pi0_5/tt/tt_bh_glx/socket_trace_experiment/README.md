# socket_trace_experiment (isolated — does not touch the production pipeline)

Tests the hypothesis: **the original per-1×1-mesh socket pipeline can be traced if
we capture a trace on EACH per-chip submesh** (instead of on the parent, which was
the original `capture_trace` bug → empty trace + full-mesh-finish deadlock).

Background / proof of the mechanism (already in `tests/perf/`):
- `_socket_in_trace_repro.py` — a socket in a 1×1-submesh trace replays (2 chips, 3×).
- `_socket_chain_32_trace.py` — a 31-socket relay across 32 1×1 submeshes, captured
  per-submesh, replays 5× with data intact. So sockets-in-trace are NOT the problem.

This experiment goes further: the **real eager `Pi0_5GLXPipeline`** (vision 4 + prefill
18 + denoise 6, per-chip 1×1 submeshes, FABRIC_2D sockets) with **per-submesh trace
capture**. It reuses the production pipeline unchanged and only wraps its
`_sample_actions_device` body with begin/end_trace_capture on every per-chip submesh.

The key unknown it probes: can ~28 submesh traces be **captured concurrently**
(begin-all → run the monolithic device body → end-all) and replayed correctly?

Run (needs a 32-chip BH Galaxy):
```
source _bench_runs/pi05_production.env
export PI05_CHECKPOINT_DIR=/path/to/pi05_libero_upstream
tt-smi -glx_reset
python_env/bin/python models/experimental/pi0_5/tt/tt_bh_glx/socket_trace_experiment/run_socket_traced.py
```

Outcomes:
- captures + replays + actions match eager → the original socket design IS traceable
  per-submesh (just less efficient than the 3-trace per-stage p2p design: ~28 traces).
- begin_trace_capture errors on the 2nd concurrent submesh → concurrent multi-mesh
  capture isn't supported; would need to capture/replay stage-by-stage instead.
