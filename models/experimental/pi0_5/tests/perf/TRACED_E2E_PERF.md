# pi0.5 tt_bh_glx — fully-traced e2e perf (BH Galaxy, per-stage layout)

Measured on Blackhole Galaxy, 5 denoise steps, N_CAMS=3 (prefix_len=1024),
production perf flags baked in (`_bench_runs/pi05_production.env`). Correctness:
traced e2e matches torch `Pi0_5Model.sample_actions` at **PCC 0.9988**.

Reproduce:
```
tt-smi -glx_reset
PI05_E2E_PERF=1 PI05_CHECKPOINT_DIR=<ckpt> \
  python_env/bin/python models/experimental/pi0_5/tests/perf/_trace_e2e_full.py
# add PI05_E2E_PCC=1 for the torch PCC check, PI05_PREFILL_PROFILE=1 for the snake breakdown
```
The 3 traced-stage replays are timed over 2 warmup + 20 iters (execute_trace is
pure replay; the cross-stage hand-offs allocate device buffers so they're timed
single-shot inline).

## Per-inference latency (5 denoise steps)

| Component                | Before opt | After opt | Note |
|--------------------------|-----------:|----------:|------|
| vision trace replay      |    8.66 ms |   7.98 ms | SigLIP 27-block core, 3-chip p2p chain |
| vision→prefix hand-off   |   ~4.8 ms  |  ~4.8 ms  | host-bounce (on-device under PI05_VISION_SOCKET=1) |
| **prefill trace replay** |   45.75 ms |  **32.55 ms** | 18 VLM layers, snake p2p — multi-core p2p cut transfer 29→16 ms |
| KV migration             |   ~11 ms warm | ~11 ms warm | on-device p2p-gather + adjacent socket (2 conns) |
| denoise trace replay     |   26.17 ms |  26.08 ms | 5 Euler steps, KV cross-attn (p2p small here) |
| **Traced compute**       |   80.6 ms  | **66.6 ms** | sum of the 3 replays |
| **Total / inference (warm)** | ~96 ms | **~82 ms** | ~15% faster |

Prefill is the dominant stage; after the p2p optimization it is block-compute
bound (`block_fwd` ~30 ms traced), not transfer bound.

## Fabric transfer findings (what drove the optimization)

Adjacent BH-Galaxy chips have **2 forwarding links** (probed via
`_p2p_multicore_probe.py`).

| Mechanism                    | Bandwidth (1024×2048 bf16) | Notes |
|------------------------------|---------------------------:|-------|
| `point_to_point` (orig)      | 2.7 GB/s (1 core, 1 link)  | hardcoded use_cores={1,1}, link_idx=0 |
| `point_to_point` (multi-core)| **5.3 GB/s** (2 cores, 2 links) | this change; trace-safe, used by the snake |
| `send_direct_async` 1 conn   | 8.3 GB/s                   | direct-write socket; faster kernel |
| `send_direct_async` 2 conns  | **15.5 GB/s** (2 cores, 2 links) | used by KV migration + on-device vision→prefix |

- p2p hop *distance* is free (1-hop == 3-hop); only the number of transfers and
  per-transfer bandwidth matter.
- `send_direct_async` cannot replace `point_to_point` inside a trace (sockets
  carry flow-control state that trace replay doesn't reset → deadlock). So the
  in-trace snake uses multi-core `point_to_point`; the eager cross-stage
  hand-offs use multi-connection `send_direct_async`.

## Cross-stage hand-off (KV migration) — host-bounce vs on-device

| | host-bounce | on-device (p2p-gather + socket) |
|---|---:|---:|
| KV migration | ~265 ms | **~11 ms warm** (24×) |

(host-bounce did 36 full-mesh `ConcatMeshToTensor` gathers + 6 reshards.)
