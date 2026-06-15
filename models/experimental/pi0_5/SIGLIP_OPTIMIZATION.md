# SigLIP Vision Stage Optimization (BH Galaxy)

This document describes the latency optimization work on the PI0.5 SigLIP vision
encoder as it runs on the Blackhole Galaxy (GLX) pipeline, the env flags that
gate each optimization, and the scripts to reproduce the perf numbers.

All optimizations are **env-gated and default-off** (except where noted), so the
production behavior is unchanged unless the flags are set.

---

## TL;DR

The SigLIP vision stage runs as a 4-chip slice of the GLX pipeline:

```
chip0 = patch embed   chip1 = layers 0-8   chip2 = layers 9-17   chip3 = layers 18-26 + projector
```

Input: 3 camera images `(3, 3, 224, 224)` → output `(3, 256, 2048)` (bs=3).

| Metric (3 images, bs=3)                       | Baseline | Optimized | Delta   |
|-----------------------------------------------|----------|-----------|---------|
| Per-forward device-kernel time                | 8.06 ms  | 5.47 ms   | −32%    |
| Single-shot wall-clock (trace + sockets)      | —        | ~7.8 ms   |         |
| Continuous throughput (pipelined)             | —        | 2.3 ms/frame (≈439 fps) | |

PCC vs torch reference stays ≥ 0.999 (gate 0.997 for the stage, 0.90 full tower).

**Key principle proven along the way:** L1 *weight residency* does NOT help the
SigLIP (or VLM-prefill) matmuls — they are math/FPU-bound and read each weight
once per forward, so L1 placement gives 1.00× (measured). L1 residency only pays
off for the denoise stage, which reuses weights across N Euler steps. The real
prefill wins are **structural** (block-sharded slice path, fidelity, transport),
not memory placement.

---

## The optimizations

Each is a separate env flag. They stack additively and are PCC-safe.

### 1. Matmul LoFi — `PI0_SIGLIP_MM_HIFI=0` + `PI0_SIGLIP_MM_FP32_DEST=0`
Drops the SigLIP attention/MLP matmuls from HiFi2 to LoFi math fidelity. The
weights are bf8 (7-bit mantissa) so LoFi is sufficient; it roughly halves the
MAC phases. ~−0.8 ms/forward.

Per-region overrides also exist for finer control:
`PI0_SIGLIP_ATTN_HIFI` and `PI0_SIGLIP_MLP_HIFI` (the PCC loss is entirely from
the attention matmuls; MLP-LoFi alone keeps PCC ≈ 0.9996).

### 2. Fold host-prep — `PI0_SIGLIP_USE_FOLD=1` + `PI0_SIGLIP_FOLD_HOST_PREP=1`
Moves the pixel patch-embed pre-processing (permute / untilize / reshape) off
the device and onto the host CPU. On-device embed one-off ops drop to ~0. The
full benefit is realized under trace + 2 command queues (host work overlaps the
device). ~−0.9 ms/forward of device time.

### 3. SDPA query chunk — `PI0_SIGLIP_SDPA_QCHUNK=256`
SigLIP attention is Sq = Skv = 256. The default picker uses q_chunk = 64
(conservative, tuned for S = 512). Setting q_chunk = 256 runs the attention in a
single query chunk (matches the canonical `models/demos/multimodal/siglip`
config). Pure scheduling knob, no precision impact.

### 4. Block-sharded slice path — `PI0_GLX_SIGLIP_BS=1`
The biggest single win. The GLX vision slices originally used the slow
interleaved `forward()` (a "perf comes later" TODO). Each slice owns 9
contiguous on-chip layers, so the residual stream can enter block-sharded once
and stay block-sharded across all 9 layers — eliminating the per-layer
LayerNorm↔matmul reshards. Host bounces only happen between slices.
~−0.9 ms/forward (matmul + SDPA + reshard savings).

### Recommended combined config

```bash
PI0_SIGLIP_MM_HIFI=0 \
PI0_SIGLIP_MM_FP32_DEST=0 \
PI0_SIGLIP_USE_FOLD=1 \
PI0_SIGLIP_FOLD_HOST_PREP=1 \
PI0_SIGLIP_SDPA_QCHUNK=256 \
PI0_GLX_SIGLIP_BS=1
```

---

## Trace + fabric sockets (how the device win becomes wall-clock)

The −32% device-kernel-time only converts to faster wall-clock under **trace
mode** + **fabric-socket transport**. Two facts make this necessary:

1. A single TTNN trace cannot span submeshes or contain host I/O. The vision
   stage spans 4 chips, so the fast path captures **one trace per chip** and
   stitches them with transport that stays *between* the traces.
2. The inter-chip handoff must avoid the host. `SocketTransport`
   (`ttnn.experimental.send_direct_async` / `recv_direct_async`) does a fabric
   direct-write (bit-exact, no `to_torch`/`from_torch`). This works on a single
   4×4 island, which is where the vision stage lives.

Progression (optimized config, bs=3):

| Path                                  | Wall-clock |
|---------------------------------------|------------|
| eager + host transport                | ~39 ms     |
| per-chip traces + host transport      | 16.4 ms    |
| per-chip traces + fabric sockets      | 9.2 ms     |
| + upload hidden (2nd CQ)              | 7.3 ms     |
| streaming throughput (frames pipelined) | 2.3 ms/frame |

The single-shot 5.47 ms device time is a serial *sum* across the 4 chips (each
waits for the previous). In continuous streaming the chips work on different
frames concurrently, so steady-state throughput is the slowest single chip
(~1.75 ms) plus transport ≈ 2.3 ms/frame.

---

## Scripts — perf testing

All scripts live in `models/experimental/pi0_5/tests/perf/`. Source your env
first (activates the venv, sets the checkpoint path, etc.) and run from the
repo root.

### Stage benchmark + PCC (4-chip slice)
```bash
PI0_SIGLIP_MM_HIFI=0 PI0_SIGLIP_MM_FP32_DEST=0 PI0_SIGLIP_USE_FOLD=1 \
PI0_SIGLIP_FOLD_HOST_PREP=1 PI0_SIGLIP_SDPA_QCHUNK=256 PI0_GLX_SIGLIP_BS=1 \
python_env/bin/python models/experimental/pi0_5/tests/perf/bench_siglip_l1.py
```
Times the full 4-chip vision forward and reports PCC vs the torch reference.
`BENCH_SLICES=1` prints a per-slice breakdown. `PI0_NUM_CAMERAS` sets the batch.

### Device-kernel-time profile (tracy)
```bash
PI0_SIGLIP_MM_HIFI=0 PI0_SIGLIP_MM_FP32_DEST=0 PI0_SIGLIP_USE_FOLD=1 \
PI0_SIGLIP_FOLD_HOST_PREP=1 PI0_SIGLIP_SDPA_QCHUNK=256 PI0_GLX_SIGLIP_BS=1 \
PROF_ITERS=1 python_env/bin/python -m tracy -v -r -p \
  -o generated/profiler/siglip models/experimental/pi0_5/tests/perf/prof_siglip.py
```
Produces `generated/profiler/siglip/reports/.../ops_perf_results_*.csv`. Note
`prof_siglip.py` profiles a warmup forward + `PROF_ITERS` measured forwards, so
`PROF_ITERS=1` captures **2** forwards (verify via matmul op count = 36/forward
per layer-chip). Divide the per-op sums accordingly.

#### Pretty-print the CSV with tt-perf-report
```bash
uv pip install --python python_env/bin/python tt-perf-report   # one-time
python_env/bin/tt-perf-report --no-merge-devices --min-percentage 1.0 \
  generated/profiler/siglip/reports/*/ops_perf_results_*.csv
```
Use `--no-merge-devices` — the 4 chips run *different* pipeline slices, so
device-merge is wrong. Read the **Stacked report** at the bottom (per-op sums per
device). Chips 17/18/19 are the representative layer-chips
(Matmul ≈ 58% / SDPA ≈ 13% / LayerNorm ≈ 11.5%); exclude `ConcatDeviceOperation`
(one-time weight construction) for steady-state timing.

### Latency demos (trace + fabric sockets)
```bash
# single-shot latency
... env flags ... python_env/bin/python \
  models/experimental/pi0_5/tests/perf/demo_siglip_single_shot.py

# continuous / streaming throughput (fill latency + steady-state ms/frame)
... env flags ... python_env/bin/python \
  models/experimental/pi0_5/tests/perf/demo_siglip_continuous.py
```

### MLP matmul sweep (minimal_matmul vs linear) — pytest
```bash
PI0_MLP_SWEEP=1 pytest -xvs \
  models/experimental/pi0_5/tests/perf/test_mlp_minimal_vs_linear.py
```
Sweeps `ttnn.experimental.minimal_matmul` vs traditional `ttnn.linear` for the
SigLIP and VLM-prefill MLP shapes, on **single-chip** and **4-chip** meshes,
across a range of block / grid configs. Every config is PCC-gated; a ranked
table + winner is printed per shape. Pin with `PI0_MLP_SWEEP_SHAPE` /
`PI0_MLP_SWEEP_DEVICE`.

Finding: minimal_matmul wins the wide-N matmuls (fc1 1152→4608, gate/up
2048→16384) as a CB-budget escape hatch; traditional linear wins the shapes that
fit a 2D mcast (fc2, down). End-to-end on the BS path, however, the BS linear is
preferred for SigLIP because BS↔interleaved reshards cost more than the kernel
savings.

---

## Supporting microbenchmarks

| Script | Purpose |
|--------|---------|
| `bench_siglip_matmul.py` / `bench_siglip_matmul2.py` | isolated matmul fidelity / grid / K-block sweeps |
| `bench_minimal_mm.py`    | minimal_matmul vs BS linear microbench (SigLIP fc1/fc2) |
| `bench_vlm_mlp_l1.py`    | L1-vs-DRAM weight microbench at VLM MLP shapes (proves math-bound) |
| `bench_siglip_block.py`  | op-level per-block profile |
| `bench_siglip_trace.py`  | per-chip traces stitched by host transport |
| `bench_siglip_trace_socket.py` | per-chip traces + fabric sockets (the fast path) |
| `pcc_siglip_fidelity.py` | full single-chip tower PCC across fidelity configs |

---

## L1 migration helpers (gated, default-off)

`tt/tt_bh_glx/_l1_migration.py` adds `migrate_siglip_weights_to_l1` and
`migrate_prefill_vlm_weights_to_l1`, wired into `pipeline.py` behind
`PI0_GLX_SIGLIP_L1` / `PI0_GLX_PREFILL_VLM_L1`. **These do not improve SigLIP
latency** (math-bound; measured 1.00× and they OOM because the fc1 weight
doesn't fit in L1 next to block-sharded activations). They are kept as gated,
measured negative results — use `minimal_matmul` (see the MLP sweep) if you ever
need to unblock the CB clash, but expect no speedup for SigLIP.

## WIN 5 — multi-connection fabric sockets (`PI05_SOCK_CONN=2`, default ON)

Ported from branch `pi05_openpi_upstream_bh_glx_trace` (commit `7009cd6a47c`).
`SocketTransport._pair` now opens **N `SocketConnection`s** (sender core `(i,0)`
→ receiver core `(i,1)`) instead of one, so `send_direct_async` spreads the
inter-chip transfer across the adjacent chip pair's **2 forwarding fabric
links** (~2× bandwidth, 2.7→5.3 GB/s).

A/B (cumulative-optimized config, this 4×4 island):

| metric | `PI05_SOCK_CONN=1` | `PI05_SOCK_CONN=2` | delta |
|---|---|---|---|
| single-shot | 7.789 ms | **7.525 ms** | −3.4% |
| streaming | 2.255 ms/frame (443 fps) | **2.081 ms/frame (481 fps)** | −7.7% |
| PCC vs torch | 0.9992 | 0.9990 | both PASS |

`PI05_SOCK_CONN=3` hard-fails (`Available links: 2, Requested pairs: 3`) — there
are exactly 2 fabric links between adjacent BH-GLX chips, so **2 is both the
optimum and the hardware ceiling**. Default is 2; set `PI05_SOCK_CONN=1` to A/B
the original single-connection path.
