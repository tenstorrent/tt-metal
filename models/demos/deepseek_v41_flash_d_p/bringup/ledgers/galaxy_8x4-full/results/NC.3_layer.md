# NC.3 layer check: the per-layer fabric sends as the native C++ ttnn.layer_send vs the Python prepared route

Written by `models/demos/deepseek_v41_flash_d_p/tests/native/test_native_sends.py` (op level on the held mesh, no model load; fixed-seed bytes in the model's io shapes and the callers' arguments). Model-wide verification is deferred to the end of the port series (owner, NC.8).

| case | program | mismatch | repeat | cache growth (C++) | host us python (before) | host us native (after) | device us python | device us native | calls a chunk | host ms a chunk saved |
|---|---|---|---|---|---|---|---|---|---|---|
| window_fused8192 | WindowSend.fused | 0.0 | 0.0 | 0 | 54.7 | 43.7 | 42.65 | 42.63 | 20 | 0.22 |
| window8192 | WindowSend.__call__ | 0.0 | 0.0 | 0 | 44.0 | 40.6 | 39.1 | 38.98 | 20 | 0.07 |
| window_fused5120 | WindowSend.fused | 0.0 | 0.0 | 0 | 54.0 | 44.7 | 39.3 | 38.82 | 20 | 0.19 |
| snap_dyn_own | SnapshotSend._call_dyn | 0.0 | 0.0 | 0 | 61.1 | 47.4 | 59.33 | 47.12 | 20 | 0.27 |
| snap_static_own | SnapshotSend static | 0.0 | 0.0 | 0 | 70.8 | 55.7 | 59.47 | 55.21 | 20 | 0.30 |
| snap_dyn_mixed | SnapshotSend._call_dyn | 0.0 | 0.0 | 0 | 61.8 | 47.1 | 59.33 | 47.04 | 20 | 0.29 |
| snap_static_mixed | SnapshotSend static | 0.0 | 0.0 | 0 | 72.9 | 56.9 | 60.56 | 57.04 | 20 | 0.32 |
| snap_multi4 | SnapshotSend.multi | 0.0 | 0.0 | 0 | 130.0 | 84.2 | 127.38 | 87.14 | 20 | 0.92 |
| glued4x2k_mixed | GluedSend | 0.0 | 0.0 | 0 | 116.1 | 66.5 | 66.12 | 66.08 | 20 | 0.99 |
| glued4x2k_temps | GluedSend | 0.0 | 0.0 | 0 | 117.7 | 67.6 | 66.1 | 66.1 | 20 | 1.00 |

Host us: a warm call, perf_counter over 40 calls, the median of 5 rounds alternating the routes (python = the class's prepared route: ttnn.bringup.prepare_program / run_prepared; native = ttnn.layer_send). Device us: real-time profiler, slowest chip, 20 calls a section. Calls a chunk: one a layer for layers 0-19 (the brief); a chunk runs one of the window forms and one of the snapshot forms a layer (the glued form on packed chunks), so the forms' savings do not add up.
