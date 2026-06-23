# Pipelined-Prefill Layer-Completion — Integration Notes (2026-06-23)

Status of wiring the layer-completion routing transport into the N-rank
(pipeline-parallel D2D) prefill runner. Companion to the plan at
`docs/superpowers/plans/2026-06-19-pipelined-prefill-layer-completion-routing.md`.

## Branch topology (why this was split three ways)

The work lived across three branches, which is the source of most of the confusion:

- **`snijjar/prefill-layer-completion-aggregation`** — the C++ `ttnn.layer_completion`
  transport (message + Vyukov MPSC SHM ring + reorder buffer + MPI router) and its
  nanobind binding. Built on a pre-D2D tree, so its runner wiring targets the OLD
  single-host `TtDeepSeekPrefillPipeline` and hard-guards `world_size > 1`.
- **PR #47420 `jjovicic/pipeline-prefill-d2d`** — pipeline-parallel prefill over D2D
  fabric sockets. Rank-sliced model (`first_layer_idx` / `is_first_rank` /
  `is_last_rank`), `TtPrefillRuntime` (renamed from the pipeline class). Per-layer
  LayerAck + migration are explicitly **disabled for `num_ranks > 1`**.
- **`snijjar/pp-d2d-layer-completion-wiring`** (this branch) — built on #47420. Already
  contained the Python reconciliation (commit `42d723d0944`) but was MISSING the C++
  transport it calls into.

## Key correctness fact: completions fire the GLOBAL layer index

On a pipeline-sliced runtime each block is constructed with the global index
`layer_idx = first_layer_idx + local_idx` (`tt_prefill_transformer.py`), which
propagates to `ttMLA(layer_idx=...)`, so MLA fires `on_layer_complete(self.layer_idx)`
with the **global** index. The KV-cache slot, separately, uses the **local**
`cache_layer_idx` (`cache_user_id * num_my_layers + local_idx`).

Consequence for the sink: `seq = request_id * NUM_LAYERS + layer_idx` is dense and
gap-free across ranks **only if the stride is the GLOBAL `NUM_LAYERS`, not the rank's
slice count**. `build_layer_completion_sink` is wired with `num_layers=NUM_LAYERS`.
This is the load-bearing footgun — wiring it with `runtime.config.num_layers` (now the
local slice count) would collapse seq and collide across ranks.

## #1 — reconcile the two mutual guards: DONE (source level)

`42d723d0944` (pre-existing) did the Python side:
- Removed #47420's "disable LayerAck for `num_ranks > 1`" refusal.
- Added `TtPrefillRuntime.set_layer_completion_sink()` (mutually exclusive with
  `set_layer_ack_channel()`); driver picks one by `num_ranks`.
- `runners/layer_completion_sink.py`: `build_layer_completion_sink()` with the GLOBAL
  `NUM_LAYERS` stride; `request_id = runtime.current_chunk_idx` (set per chunk in
  `_compute_and_send`).
- `_serve_request`: single-rank keeps the direct counter-channel inject; `num_ranks > 1`
  stands up `LayerCompletionRouter` (+ host-local ring producer) on every rank and
  registers the sink. `master_rank` via `PREFILL_MASTER_RANK` (default 0).

This commit's blocker was that `ttnn.layer_completion` (the C++ transport) was not in
the branch. Resolved on 2026-06-23 by cherry-picking the 7 contiguous transport /
nanobind / test commits `c3c81d12c3d..bcf0ae0d043` from
`snijjar/prefill-layer-completion-aggregation`, **excluding** the two old-runner
commits (`ff898166ecc`, `9601fc4a988`) that conflict with #47420's reworked runner.
Cherry-pick was clean (additive auto-merges on `ttnn-nanobind/__init__.cpp`,
`ttnn/sources.cmake`, `ttnn/ttnn/__init__.py`); transport + binding + build
registration now all present in this tree.

### Verified
- Sink seq-math (pure Python, no build): both tests in
  `tests/ttnn/unit_tests/base_functionality/test_pp_layer_completion_sink.py` pass —
  completions tile `0..N-1` densely across 2 ranks × 4 layers × 2 chunks, and a full
  ring raises rather than dropping.
- nanobind call signatures match the wiring (`LayerCompletionRouter(rank, world_size,
  master_rank, ring_shm_name, scheduler_channel_shm_name)`, `LayerCompletionQueue.connect`,
  `try_push(seq, source_rank, layer_idx, request_id)`).

### Not yet done for #1
- **C++ build of the combined tree + the single-rank e2e test**
  (`test_sink_through_real_router_single_rank`, currently `importorskip`-gated). The
  cherry-picked sources built on their origin branch, but the combined-tree compile is
  unverified.

## Follow-up topics (explicitly deferred)

- **#2 — per-layer `synchronize_device` defeats pipelining.** The completion callback in
  MLA is gated together with `zero_padded_kv_cache` + an unconditional
  `ttnn.synchronize_device(self.mesh_device)` (mla.py ~607–622) BEFORE
  `on_layer_complete`. Enabling completions in pipeline mode forces a per-layer global
  device sync on every rank, serializing the overlap D2D pipelining exists to create.
  Must decouple the notification from the pad-zero/sync (or make the sync async).
- **#3 — migration KV-chunk-table publish is single-rank only.** Completion *messages*
  carry `source_rank` + global `layer_idx` (enough to locate the KV), but the chunk
  address table the consumer DMAs from is published single-rank. Each rank must publish
  its slice's table; the consumer must union them and map global layer → (rank, local
  slot). Separate work.
- **#4 — MPI thread-safety.** The router does `send`/`irecv` on a background thread while
  the main path runs MPI barriers + D2D fabric setup. Needs `MPI_THREAD_MULTIPLE`;
  confirm the `DistributedContext` requests it and the completion tag (4242) can't
  collide with host-side socket/fabric traffic.
- **#5 — reorder-buffer head-of-line stall.** A single lost / never-emitted completion
  wedges the whole scheduler stream and grows the `std::map` unbounded. No
  watchdog/timeout. More likely under real multi-rank + fabric.

Also open (flagged in `42d723d0944`): per-rank chunk-counter alignment (relies on FIFO
+ the one warm-up barrier) and clean cross-rank router teardown (currently rough
SIGKILL shutdown).
