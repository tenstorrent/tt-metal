# socket_trace_experiment — all-socket per-1×1-mesh traced pi0.5 e2e

Isolated harness (does **not** modify the production pipeline) that runs the real
`Pi0_5GLXPipeline` — vision 4 + prefill 18 + denoise 6, every chip a 1×1 submesh, every
cross-stage hand-off a **fabric socket** (`send_direct_async`/`recv_direct_async`, no
`point_to_point`) — and **traces it per-submesh** (begin/end_trace_capture on each of the
28 submeshes, not the parent). It reuses the pipeline's pure-device body
(`_sample_actions_device`) unchanged.

**Status: RESOLVED — it works.** The full 28-chip all-socket pipeline captures
per-submesh (28 concurrent submesh traces, no deadlock) and replays at PCC 1.000000 vs
eager / 0.998796 vs torch, including the N=5 denoise Euler loop. The original
`capture_trace` hang was capturing on the PARENT while ops ran on 1×1 children (empty
trace + full-mesh-finish deadlock at capture time), not a socket-replay problem.

---

## Quick start

Production perf flags are **auto-applied** from `_bench_runs/pi05_production.env`
(via `setdefault` at script startup) — no manual `source` needed. Only the checkpoint
path is machine-specific:

```bash
export PI05_CHECKPOINT_DIR=/home/tt-admin/sdawle/tt-metal/models/experimental/pi0_5/weights/pi05_libero_upstream
export PYTHONPATH=$PWD TT_METAL_HOME=$PWD
python_env/bin/tt-smi -glx_reset                       # clean device state (do this before every run)
python_env/bin/python models/experimental/pi0_5/tt/tt_bh_glx/socket_trace_experiment/run_socket_traced.py
```

Default run = capture + 3 PCC-vs-eager replays + a `PERF_ITERS`-averaged perf number.

> **Always start from a clean `tt-smi -glx_reset`.** Without the production flags (or on a
> dirty device) the e2e runs ~67 ms with PCC-vs-eager ~0.98; with them it's ~50–54 ms at
> PCC 1.0. The flags being off — not device state — was the cause of an earlier perf "drift".

---

## Run modes (env vars)

| Env | Effect |
| --- | --- |
| `TRACE_SCOPE=full` *(default)* | Whole pipeline: vision → build_prefix → prefill → KV migration → N-step denoise. Also `vp` (vision+prefill only) and `denoise` (denoise-only, KV pre-populated eager) for localization. |
| `FIXED_NOISE=1` *(default)* | Pin one noise tensor across eager+capture+replay. **Required** for a valid PCC comparison — `_refresh_noise_buffer()` otherwise draws fresh `torch.randn` every call, so eager and replay would see different noise and the PCC looks "corrupted" when it isn't. |
| `PI05_E2E_PCC=1` | After replay, compare the all-socket actions against the torch reference (target ≥ 0.95; production ≈ 0.9988). |
| `PI05_SOCK_CONN=2` *(default)* | SocketConnections per hop. 2 spreads `send_direct_async` across the adjacent pair's 2 fabric links → ~7% faster e2e (see Results). `=1` for the single-link baseline. |
| `PERF_ITERS=20` *(default)* | Replays averaged for the `PERF:` latency line. |
| `EAGER=1` | **No trace.** 1 warm-up + 1 profiled iter run eagerly, with tracy SIGNPOSTS around each phase. Use under `python -m tracy` to get true per-op device-kernel durations (no trace-replay socket-WAIT inflation). |
| `TRACY=1` | Capture + exactly 1 warm-up replay + 1 profiled replay, then stop (keeps the device profiler log small). Use under `python -m tracy --device-trace-profiler`. |

---

## PCC + perf (no profiler)

```bash
TRACE_SCOPE=full FIXED_NOISE=1 PI05_E2E_PCC=1 PI05_SOCK_CONN=2 \
  python_env/bin/python models/experimental/pi0_5/tt/tt_bh_glx/socket_trace_experiment/run_socket_traced.py
```
Prints, per replay, `PCC vs eager` (trace fidelity, expect 1.000000), then
`PCC vs torch` (numerics, ≈ 0.998796), then
`PERF: traced all-socket e2e replay = <ms>/inference (avg of <PERF_ITERS>)`.

### Results (validated 2026-06-14, clean reset, production flags on)

| sockets | e2e replay | infer/s | PCC vs eager | PCC vs torch |
| --- | --- | --- | --- | --- |
| `PI05_SOCK_CONN=1` | 53.67 ms | 18.6 | 1.000000 | 0.998796 |
| `PI05_SOCK_CONN=2` *(default)* | **49.94 ms** | **20.0** | 1.000000 | 0.998796 |

2 connections is ~7% faster at identical numerics. The C++ "multiple sender cores on a
single device" line is a warning only — no correctness impact here. The gain is modest
(not 2×) because the pipeline is **serialization/dispatch-bound, not socket-bandwidth-
bound**: the busiest single chip does only ~4 ms of compute; the ~50 ms is the critical
path through the 18-hop prefill snake + the N-step denoise loop + host dispatch.

---

## Tracy profiling

Both modes write a tracy report under `-o <dir>/reports/<name>/<timestamp>/`.

**Non-traced (EAGER)** — true per-op device-kernel durations, phase-signposted:
```bash
EAGER=1 FIXED_NOISE=1 \
  python_env/bin/python -m tracy -p -r -v --op-support-count 100000 \
    -o /path/to/tracy_out -n eager_pf \
    models/experimental/pi0_5/tt/tt_bh_glx/socket_trace_experiment/run_socket_traced.py
```

**Traced (TRACY)** — 1 warm-up + 1 profiled replay:
```bash
TRACE_SCOPE=full TRACY=1 FIXED_NOISE=1 PI05_SOCK_CONN=2 \
  python_env/bin/python -m tracy -p -r -v --op-support-count 100000 --device-trace-profiler \
    -o /path/to/tracy_out -n traced_pf \
    models/experimental/pi0_5/tt/tt_bh_glx/socket_trace_experiment/run_socket_traced.py
```

> Note: the traced run's *per-op* `ops_perf_results` report can fail to assemble
> (`AssertionError: Device data missing ... trace_id=None`) — a tracy limitation mapping
> replayed ops back to device-perf entries. The raw device data still lands in
> `<dir>/.logs/cpp_device_perf_report.csv` (per-device trace-replay kernel durations) and
> `profile_log_device.csv`. The non-traced EAGER report assembles cleanly and is the one
> to annotate.

---

## Annotation

The EAGER tracy CSV (`ops_perf_results_*.csv`) is annotated two ways:

1. **Phase rollup** — `annotate_eager_perf.py <ops_csv> <out_dir>` emits
   `socket_e2e_phase_summary.csv`, `socket_e2e_phase_optype.csv`, and a gzipped per-op
   dump. Per-phase device-kernel ms / socket-wait ms / busiest-chip ms.

2. **Full per-op annotation** — `annotate_full.py <eager_ops_csv> [out.csv]` keeps every
   original column + row and prepends five columns:

   | column | values |
   | --- | --- |
   | `PHASE` | `init_one_time`, `warm-up`, `iter:vision`, `iter:build_prefix`, `iter:prefill`, `iter:kv_migration`, `iter:denoise`, `teardown` (from the tracy signposts) |
   | `STAGE` | `vision` / `prefill` / `denoise` (by DEVICE ID), or `signpost` |
   | `STEP` | `1`–`5` for `iter:denoise` rows (the N Euler steps), blank elsewhere |
   | `LAYER` | 1-based, resets per stage: SigLIP 1–27, VLM prefill 1–18, denoise 1–18 |
   | `SUBSTAGE` | `attn`/`mlp` (per layer) + `head` (one-time denoise-loop setup, step 1) + `tail` (per-step velocity-wrap / Euler update) + `output` (step 5's final wrap) |

   LAYER/SUBSTAGE for vision+prefill come from `_bench_runs/annotate_ops_csv_v4.py`
   (SDPA-boundary segmentation). The denoise loop is re-segmented so each Euler step owns
   its trailing velocity-wrap, giving an identical 602-op body per step.

   `annotate_full.py` verifies the run captured exactly **one warm-up + one profiled iter**
   (warm-up op count == sum of the iter phases) and that there are 135 SDPA/inference.

---

## Per-stage timing (actual profiled iter, EAGER device kernel, production flags)

Device-kernel summed across all 28 chips (work, not wall-clock — chips run in parallel;
busiest single chip ≈ 4 ms). `init_one_time` is the one-time setup and is **excluded**
from the iter total.

| stage | ops | kernel ms |
| --- | ---: | ---: |
| `init_one_time` *(excluded)* | 1084 | 10.85 |
| iter:vision (SigLIP) | 425 | 9.38 |
| iter:build_prefix | 4 | 0.06 |
| iter:prefill (VLM, 18-hop snake) | 395 | 28.67 |
| iter:kv_migration | 72 | 2.66 |
| iter:denoise (5 Euler steps) | 3090 | 22.61 |
| **ITER TOTAL** *(init excluded)* | **3986** | **63.39** |

Denoise per step — body (the 18 transformer layers) is identical every step; the total-op
differences are boundary I/O (step 1 carries the one-time loop head; step 5's wrap is the
final output, lighter than the inter-step wraps):

| step | total ops | kernel ms | body | head | tail/output |
| --- | ---: | ---: | --- | ---: | ---: |
| denoise_step_1 | 629 | 4.59 | 602 op, 4.43 ms | 0.06 | 0.10 |
| denoise_step_2 | 618 | 4.52 | 602 op, 4.40 ms | – | 0.12 |
| denoise_step_3 | 618 | 4.51 | 602 op, 4.41 ms | – | 0.10 |
| denoise_step_4 | 618 | 4.55 | 602 op, 4.44 ms | – | 0.11 |
| denoise_step_5 | 607 | 4.45 | 602 op, 4.41 ms | – | 0.04 (output) |

### Kernel vs socket vs data-movement (profiled iter)

| category | ops | % ops | device-kernel ms | % ms |
| --- | ---: | ---: | ---: | ---: |
| Compute / kernel ops | 2501 | 62.7% | 44.15 | 69.7% |
| Socket ops (Send/Recv) | 174 | 4.4% | 16.34 | 25.8% |
| Data-movement / layout ops | 1311 | 32.9% | 2.90 | 4.6% |

Socket ops are 4.4% of the op count but ~26% of kernel time — that time is fabric **WAIT**,
not work. The direct-write transport adds **no** per-iter support kernels (receiver buffers
are pre-allocated once at init); the only socket-adjacent op is the bf8 KV typecast after
recv (`kv_migration.py`).

---

## Background / mechanism proofs (in `tests/perf/`)

- `_socket_in_trace_repro.py` — a socket in a 1×1-submesh trace replays (2 chips, 3×).
- `_socket_chain_32_trace.py` — a 31-socket relay across 32 1×1 submeshes, captured
  per-submesh, replays 5× with data intact. So sockets-in-trace are not the blocker.

Localization helpers here: `localize_denoise.py`, `socket_*_minitest.py`,
`loop_persist_minitest.py`.
