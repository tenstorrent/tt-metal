# WIP: BH Galaxy Qwen3-32B — Prefetcher + Fused `matmul_reduce_scatter_async`

Status doc for resuming later. Branch: `mgiermakowski/bh-fused-test`
(branched from `mgiermakowski/bh-qwen-prefetcher`, which is the working prefetcher-only baseline).

Last updated: 2026-08-05.

---

## Goal

Run Qwen3-32B decode on Blackhole (BH) Galaxy with **both** the `dram_prefetcher` **and** a
fused matmul + reduce-scatter op for the MLP FF1/FF3, for maximum decode perf.

- Prefetcher-only (no fused op) already works end-to-end: ~44.8 t/s/u decode. That is the shippable
  baseline and lives on `mgiermakowski/bh-qwen-prefetcher`.
- The fused op is an incremental gain layered on top. It is what this branch adds, and it is
  **not yet working end-to-end** (see "Remaining blocker").

## TL;DR of current state

- The fused op is integrated (MLP FF1/FF3), compiles, and is **numerically correct**:
  decode PCC **0.9978 / 0.9979** for tokens 0 and 1, matching the unfused baseline (~0.9975).
- It **deadlocks on the 3rd decode token (token index 2)**, every run, deterministically.
- The deadlock is **not** in the fused op's own kernels — it is in a *downstream* op (a
  `ttnn.slice` reader, or a standard `all_gather`) running on cores the fused ring matmul used.
  The fused op leaves persistent L1 / circular-buffer / semaphore state that corrupts a later op,
  and it accumulates until it deadlocks on iteration 3.

## How to reproduce

From `/home/mgiermak/tt-metal`:

Fused (hangs at token 2):
```
HF_MODEL=Qwen/Qwen3-32B TT_CACHE_PATH=/home/mgiermak/bh_qwen_cache \
  QWEN_BH_PREFETCHER=1 QWEN_BH_UNFUSED_CCL=1 QWEN_BH_FUSED_ASYNC_RS_MATMUL=1 \
  ./python_env/bin/pytest -svq \
  models/demos/llama3_70b_galaxy/tests/unit_tests/test_qwen_decoder.py
```

Baseline that passes all 10 tokens (drop the fused flag):
```
HF_MODEL=Qwen/Qwen3-32B TT_CACHE_PATH=/home/mgiermak/bh_qwen_cache \
  QWEN_BH_PREFETCHER=1 QWEN_BH_UNFUSED_CCL=1 \
  ./python_env/bin/pytest -svq \
  models/demos/llama3_70b_galaxy/tests/unit_tests/test_qwen_decoder.py
```

The decoder test runs `generation_length = 10` iterations and prints
`All 10 Llama decode iterations Passed!` on success. It hangs after token 1's PCC print.

Useful debug env vars:
- `TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1` — enable the Metal watcher (10s poll).
- `TT_METAL_WATCHER_DISABLE_ASSERT=1` — let it run past the soft assert to the real deadlock,
  then read the last complete dump in `generated/watcher/watcher.log`.
- `QWEN_DBG_FUSED_SYNC=1` — (added in `llama_mlp.py`) drains the worker sub-device after the two
  fused ops. Diagnostic only; does NOT fix the hang (see below).

Standalone single-shot op test (passes, PCC ~1.0 — does not exercise the decode loop):
`models/demos/llama3_70b_galaxy/tests/unit_tests/test_matmul_reduce_scatter_async_bh.py`

NOTE: the device on this host is shared and occasionally needs a `tt-smi -glx_reset` between runs
(look for "Timed out while waiting for active ethernet core ... to become active" at
`open_mesh_device` — that is a dirty device, not a code bug).

## Design of the fused op (what was built)

`matmul_reduce_scatter_async` originally supported only a 2D-multicast matmul with no global CB.
It was generalized to fuse the BH prefetcher's **1D gathered ring matmul** (streaming global-CB
weights) with the BH-working `reduce_scatter_minimal_async` backend, in a single program.

Signaling co-design ("Option B"): the prefetcher's 1D gathered matmul natively emits the
`LLAMA_REDUCE_SCATTER` signal, but the BH-working `reduce_scatter_minimal_async` consumes the
`REDUCE_SCATTER` `OpSignaler` protocol. Rather than debug the BH 2D-torus fabric for the llama-RS
writer, the 1D gathered matmul reader was taught to also emit `REDUCE_SCATTER` signals via
`OpSignaler`, and the RS reader consumes them via `ReduceScatterOpReceiver::wait_for_matmul_batch`.

Core placement (single fused program, so matmul and RS must be on disjoint cores):
- col 0: prefetcher senders (resident global CB producers)
- cols 1–3: ring matmul workers
- cols 4–10: reduce-scatter workers (confined via new `reduce_scatter_sub_core_grid` param;
  an additive `core_grid_offset` overflows off-grid, so an explicit sub-core-grid was needed)
- col 11: tensix dispatch

Model wiring: `use_bh_fused_async_rs_matmul` flag (env `QWEN_BH_FUSED_ASYNC_RS_MATMUL=1`).
`TT_CCL.matmul_reduce_scatter_async` wrapper + double-buffered persistent buffers
(`get_decode_fused_rs_matmul_buffers`). `llama_mlp.py` FF1/FF3 branch calls it and skips the
post-branch `line_reduce_scatter`. Deallocation of the persistent output buffers is guarded off
in the fused path (they are reused across layers/tokens).

## Bugs FIXED along the way

1. **PCC gap 0.984 -> 0.9978 (persistent buffer logical width).**
   The fused op's `persistent_intermediate` / `persistent_output` were allocated at the *padded*
   FF-hidden width (3840, and 960/device) instead of the *logical unpadded* width
   (`intermediate_dim_per_tp` = 3200, and 800/device). The reduce-scatter derives its scatter
   partition from the matmul output's logical width, so allocating at the padded width folded the
   640 padding columns into the logical result. Fix: allocate `torch.zeros` at the logical width and
   let the memory-config handle physical padding (`llama_ccl.py::get_decode_fused_rs_matmul_buffers`,
   uses `self.model_args.intermediate_dim_per_tp`).

2. **Persistent output buffer deallocated between layers.**
   `llama_mlp.py` was deallocating `w1/w3_out_reduced`, which in the fused path ARE the TT_CCL
   persistent buffers -> "Input Tensor is not allocated" on the next layer. Fix: guard the
   deallocate off when `use_bh_fused_async_rs_matmul`.

3. **Fused signal-semaphore never reset across cached dispatches (latent leak, real but not the
   deadlock cause).** `ReduceScatterOpReceiver::wait_for_matmul_batch` uses a cumulative
   `wait_min(batch_idx+1)` on a semaphore created once via `CreateSemaphore(..., 0)`. Under program
   caching the L1 value persists and accumulates across decode iterations. Added
   `ReduceScatterOpReceiver::reset()` (zeroes the semaphore) and call it after the RS batch loop in
   both `ring_` and `line_reduce_scatter_minimal_async_reader.cpp`. Correct to keep, but it does
   **not** resolve the token-2 hang.

## Remaining blocker (token-2 deadlock) — investigation

Symptom: tokens 0,1 pass with good PCC; token 2 hangs, deterministically. Baseline (no fused op)
runs all 10 tokens.

Watcher localization (`generated/watcher/watcher.log`), device 0:
- Asserts ON: `ttnn/.../ccl/all_gather/device/kernels/multicast_reader.cpp` trips on core `(1,0)`.
- Asserts OFF (runs to the real hang): deadlocks in
  `ttnn/.../data_movement/slice/device/kernels/dataflow/slice_reader_unary_unpad_dims_rm_sharded.cpp`
  (kernel id 1038), NCRISC stuck at waypoint `R` (NOC read wait), on cols 1,2,3,5,6.
  Prefetcher senders (col 0) also waiting.

Key deductions:
- The stuck kernels are **downstream data-movement / all_gather ops**, NOT the fused matmul or RS
  kernels. So the fused op itself completes and returns correct data (hence good PCC), but leaves
  corrupt persistent state on the cores it used (cols 1–3), which a later op inherits.
- The failure point **moves under the watcher** (token 2 without it; token 0 with it), i.e. it is
  timing-sensitive — consistent with a producer/consumer ring pointer that drifts per iteration.

Hypotheses RULED OUT:
- Fused signal-semaphore leak — reset added, no change.
- Async overlap / drain hazard — a full `ttnn.synchronize_device` on the worker sub-device after
  the fused ops (`QWEN_DBG_FUSED_SYNC=1`) does NOT fix it, so the corruption is persistent L1 state,
  not unfinished work.
- Wrapper semaphore bookkeeping — verified identical to the proven-good standalone
  `reduce_scatter_minimal_async` usage in `llama_ccl.py` (which loops fine in the baseline).
- Global-CB *release* path in the matmul reader (non-streaming `ENABLE_GLOBAL_CB && !STREAMING_IN1`
  branch) is unchanged by the edits; only the signaling branch was added.

Prime remaining suspect:
- The co-designed 1D gathered matmul reader's **remote/global circular-buffer pointer accounting**
  (`experimental::remote_cb_pop_front` / `update_remote_cb_config_in_l1(remote_cb_id)` at the end of
  `reader_bmm_tile_layout_in1_ring_all_gather.cpp`). A small per-iteration drift in the remote-CB
  ring would desync after 2–3 prefetcher refills and strand a downstream op's NOC read — matches the
  token-2, timing-sensitive signature and the fact that the prefetcher senders are also parked.
- Secondary suspect: L1 semaphore-id / CB-config aliasing between the fused program and the
  downstream slice/all_gather programs on the shared cores (cols 1–3).

## Suggested next steps

1. Add device-side `DPRINT` in `reader_bmm_tile_layout_in1_ring_all_gather.cpp` to trace the
   remote-CB read pointer / `remote_cb_pop_front` counts per batch across iterations; run 3 tokens
   and diff iteration 0 vs 2. (DPRINT perturbs timing, so also compare against the watcher dumps.)
2. Narrow the repro: fuse only FF1 (leave FF3 unfused) to confirm whether a single fused call per
   layer still deadlocks — isolates per-call vs per-layer accumulation.
3. If it is remote-CB drift: ensure the fused reader restores the exact remote-CB config the plain
   prefetcher matmul (`ttnn.linear` with `global_cb`) leaves, i.e. compare
   `update_remote_cb_config_in_l1` behavior between the two on the non-streaming path.
4. Otherwise escalate to the CCL/matmul op team with this doc + the watcher dumps; this is
   fused-op remote-CB lifecycle co-design.

## Files changed on this branch (relative to `mgiermakowski/bh-qwen-prefetcher`)

Model (Python):
- `models/demos/llama3_70b_galaxy/tt/llama_mlp.py` — FF1/FF3 fused branch; `QWEN_DBG_FUSED_SYNC`
  diagnostic; guarded deallocate.
- `models/demos/llama3_70b_galaxy/tt/llama_ccl.py` — `matmul_reduce_scatter_async` wrapper;
  `get_decode_fused_rs_matmul_buffers` (logical-width fix); stores `self.model_args`.
- `models/demos/llama3_70b_galaxy/tt/qwen_model_config.py` — `use_bh_fused_async_rs_matmul` flag.
- `models/demos/llama3_70b_galaxy/tests/unit_tests/test_matmul_reduce_scatter_async_bh.py` — new
  standalone op test (single-shot, passes).

ttnn C++ (host): generalize `matmul_reduce_scatter_async` for global_cb + 1D gathered program
config + `reduce_scatter_sub_core_grid`:
- `.../experimental/ccl/matmul_reduce_scatter_async/matmul_reduce_scatter_async{.cpp,.hpp,_nanobind.cpp}`
- `.../experimental/ccl/matmul_reduce_scatter_async/device/matmul_reduce_scatter_async_device_operation{.cpp,.hpp}`
- `.../experimental/ccl/matmul_reduce_scatter_async/device/matmul_reduce_scatter_async_device_operation_types.hpp`
- `.../experimental/ccl/matmul_reduce_scatter_async/device/matmul_reduce_scatter_async_program_factory{.cpp,.hpp}`
- `.../experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_program.cpp`
- `.../experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_{ring,line}_program_factory.hpp`
- `.../ccl/reduce_scatter/device/reduce_scatter_program_factory.cpp`
- `.../matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp`

ttnn C++ (device kernels — JIT, no host rebuild needed):
- `.../ccl/kernel_common/worker_sync_utils.hpp` — `ReduceScatterOpReceiver::reset()`.
- `.../experimental/ccl/reduce_scatter_minimal_async/device/kernels/{ring,line}_reduce_scatter_minimal_async_reader.cpp`
  — call `matmul_receiver.reset()` after the batch loop.
- `.../matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_ring_all_gather.cpp` — REDUCE_SCATTER
  `OpSignaler` path + `cb_sync` handshake for streaming.
- `.../matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp`
  — `RS_STREAMING_SYNC` credit.

NOTE: a full `ttnn` C++ rebuild is required for the host-side changes (the `.so`). The device
kernel `.cpp`/`.hpp` recompile automatically by hash on first run.
