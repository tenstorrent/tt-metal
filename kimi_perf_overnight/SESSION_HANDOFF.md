# Session handoff — Kimi chunked-prefill H2D-service "dispatch tax" experiment

Self-contained context to resume on another machine. Branch: **`ppopovic/investigation`** (pushed to origin).
Date: 2026-06-16.

## Objective
Measure the **H2D stream-service dispatch tax** on Kimi K2.6 **chunked prefill**: compare per-chunk
prefill time WITHOUT the service (`nosvc`) vs WITH an *unused* H2D service merely spinning in the
background (`h2dsvc`). Vehicle = a **no-PCC** chunked-prefill transformer test, **DEVICE_FP32** gate,
**L61 / 11 chunks / 10 iters**, mesh 8x4 Blackhole.

## USE THIS BRANCH: `ppopovic/investigation`
It is the ONLY branch where the whole experiment runs end-to-end:
- chunked prefill via `transformer.forward_chunk(...)` (the "forward_chunk" lineage)
- Kimi **single-group** device gate works (`moe_grouped_topk` handles `n_groups==1` → DEVICE_FP32 OK)
- the no-PCC + `with_h2d_service` test is committed here (commit `62c9326cd9a`)
- DEVICE_FP32 **fits L1** here (no OOM)

Branches that do NOT work (verified this session):
- `main` / `ppopovic/main_investigation` (= main + 2 mla commits): chunked prefill uses a DIFFERENT API
  `transformer.forward(..., actual_start=...)` (no `forward_chunk`), AND the device gate **L1-OOMs by
  ~3.8 KB** for Kimi chunked L61 (`Statically allocated circular buffers clash with L1 buffers`,
  addresses 1559104/1562896, independent of `dispatch_buffer_capacity_factor`). Kimi is HOST_ALL-only there.
- `ppopovic/chunked_prefill_runner_integration` (06-13 fork): has `forward_chunk` + the runner's
  `_build_h2d_service`, BUT its `moe_grouped_topk_device_operation.cpp` is DeepSeek-only
  (`TT_FATAL summed_experts_per_group == 2`, `n_groups == 8`) → Kimi single-group DEVICE_FP32 FAILS.

## The test (on this branch)
`models/demos/deepseek_v3_d_p/tests/test_prefill_transformer_chunked.py`
- `run_chunked_transformer_no_pcc(...)` + `test_kimi_prefill_transformer_chunked_no_pcc`
- Parametrize axes: `num_layers` {L1,L10,L61}, `n_chunks` {chunks11}, `num_iters` {iters1,iters10,iters20},
  `with_h2d_service` {nosvc, h2dsvc}, variant kimi, mesh-8x4. Gate = `GateComputeMode.DEVICE_FP32`.
- Builds the transformer once (`forward_chunk`, `mla_seq_len=SEQ_CACHE`, `dispatch_buffer_capacity_factor=8`,
  `is_chunked=True`, `slot_num=1`), runs `num_iters` of 11-chunk prefill with
  `return_layer_outputs=False` (no host readback, no PCC).
- `h2dsvc`: `mesh_device.clear_loaded_sub_device_manager()` then
  `build_h2d_service(...)` (from `tt/runners/runner_utils.py`) → the service's persistent receiver kernel
  spins on its worker core for the whole run; it is NEVER fed (no producer/socket). Released after the loop.
- Logs to parse: `iter {it} chunk {c}: X.XXXs` and `iter {it} done (11 chunks) in X.XXXs`. Use WARM iters
  (1..9, skip iter 0 = cold compile); per-chunk ms = iter_time / 11.

## How to run
```bash
# 1. build (this branch's impl differs from others -> rebuild after any switch)
./build_metal.sh --release --enable-ccache

# 2. run both variants (nosvc then h2dsvc) — env paths below are bh-glx specific, adjust per machine
source python_env/bin/activate
env \
  KIMI_K2_6_HF_MODEL=/data/nbabin/Kimi-K2_6-dequantized \
  TT_KIMI_PREFILL_TTNN_CACHE=/mnt/models/Kimi-K2_6-Cache/Kimi-K2_6-Cache-prefill \
  TT_KIMI_PREFILL_HOST_REF_CACHE=/mnt/models/kimi-prefill-cache/golden \
  TT_DS_PREFILL_TTNN_CACHE=/mnt/models/Kimi-K2_6-Cache/Kimi-K2_6-Cache-prefill \
  TT_DS_PREFILL_HOST_REF_CACHE=/mnt/models/kimi-prefill-cache/golden \
  python -m pytest \
    "models/demos/deepseek_v3_d_p/tests/test_prefill_transformer_chunked.py::test_kimi_prefill_transformer_chunked_no_pcc" \
    -k "iters10 and L61 and chunks11 and kimi and mesh-8x4" -s --timeout=0
```
A golden trace is optional (tokens fall back to a synthetic in-vocab pattern if the trace dir is absent;
no PCC is computed either way). Trace default: `DEEPSEEK_PREFILL_TRACE_DIR` /
`/mnt/models/deepseek-prefill-cache/golden/kimi-26/.../longbook_qa_eng_prefill_56320_nopad`.

## Findings already established (do NOT redo)
- **Service-presence tax is real & large.** DEVICE_FP32, L61, warm: `nosvc ≈ 1950 ms/chunk`,
  `h2dsvc ≈ 3450 ms/chunk` (≈ +1.5 s/chunk, ~1.8×). The service merely being resident (unused) ~doubles
  per-chunk prefill.
- **Profiler-independent.** Same nosvc↔h2dsvc delta across Tracy / non-Tracy (`--disable-profiler`) /
  RT-profiler-fully-disabled builds. The realtime profiler (PR #42905) reserves a core + does a D2H clock
  sync but adds ~0 ms to timings (measured). Note: it is NOT gated by `TRACY_ENABLE`; to fully remove it,
  early-`return {.enabled=false}` in `evaluate_realtime_profiler_eligibility`
  (`tt_metal/distributed/realtime_profiler_manager.cpp`). Not necessary for the experiment.
- **Per-iter climb.** `h2dsvc` warm iters climb (~36→40 s/iter at L61) while `nosvc` is flat — real,
  profiler-independent; something accumulates while the service is resident across iterations. OPEN thread.
- **Not MoE/sub-device specific.** MLA-only (MoE path commented out) still shows the tax
  (~760 nosvc → ~1180 h2dsvc ms/chunk). Disabling MoE sub-device overlap
  (`overlap_shared_expert_with_dispatch=False`) only removes ~11% of the tax.
- **Mechanism.** The service's persistent receiver kernel on its worker core imposes a per-op-launch
  dispatch/coordination tax that disrupts dispatcher run-ahead every layer (op2op gaps recur per layer
  WITH service; fill once at layer 0 WITHOUT). On-device kernel/compute time is unchanged; the slowdown
  is entirely op-to-op (dispatch) latency.
- **Fix direction (for later):** isolate the service's dispatch domain — give its resident program its
  own sub-device, or suspend it during `forward_chunk` — an H2DStreamService (C++) change. Core placement
  (PREFILL_H2D_WORKER_COL/ROW) and a 2nd command queue (PREFILL_NUM_CQ=2) did NOT help.

## Operational gotchas (hit this session)
- **Device wedge / eth-core timeout** (`Timed out while waiting for active ethernet core … to become
  active again`): usually leftover from a crashed run. Fix: `tt-smi -glx_reset` (only when no co-users —
  it's a whole-tray Galaxy reset), then retry. `ETH_LIVE_STATUS: 0x0` is benign/idle, NOT a wedge signal.
- **Rebuild after any branch switch** — the lib must match the checked-out impl, else mismatch/hang.
- **Stale-log race:** when polling a fresh run's log, the driver truncates it at start; grep can match the
  previous run's content in the truncation window. Verify by log mtime / fresh markers.
- **Don't `pkill -f` patterns that match your own shell** (self-kill). Kill by PID; wait for full process
  reap before relaunching (a zombie holds the chip lock / sysmem).
- After a kill, `rm -f /dev/shm/tt_prefill_layer_acks_ds_prefill /dev/shm/tt_h2d_stream_service_ds_prefill*`.

## Current git state
- Branch `ppopovic/investigation`, HEAD `62c9326cd9a` (no-PCC + H2D-service test) + this handoff commit.
- Pushed to `origin/ppopovic/investigation`.
- The `tt/runners/runner_utils.py:build_h2d_service` helper and `forward_chunk` impl are already on this
  branch (no extra porting needed). Scratch/results from this session live in `kimi_perf_overnight/`
  (NOT committed — machine-local).
