# Interleaving traced prefill and traced decode on shared weights

Goal: load the model once, then for each question run prefill followed by decode, with
prefill and decode sharing weights and both running from traces captured once.

Status: design notes only. Nothing here has been run on device yet.

## The problem

- Decode keeps weight-prefetch GCBs (about 288 KB per receiver core), their config pages,
  and a few L1-resident tensors alive for its whole lifetime.
- Prefill's static circular buffers use almost all of L1. A clash error from an earlier
  run showed the static circular-buffer region ending at 1,541,120 B of 1,572,864 B, on
  cores 0-0 through 11-7.
- The previous plan was to free decode's L1 before each prefill (release the GCBs), restore
  it afterwards, and recapture the decode traces for every question. That plan has two
  problems:
  - One GCB per device was never freed (see "GCB leak" below).
  - `release_prefetch_buffers` has to stop the prefetcher. Stopping it drops the prefetch
    requests recorded in the decode traces, so the decode traces must be recaptured for
    every question.

## Proposal: let prefill and decode alias the same L1

Capture prefill while decode's L1 is not allocated, then allocate decode's L1 and capture
decode. The two sets of addresses overlap, but they are never in use at the same time.

### Why tt-metal allows it

- The check that static circular buffers don't overlap allocated L1 buffers
  (`ProgramImpl::validate_circular_buffer_region`) only runs when a program is enqueued, in
  `compile_and_allocate`. That covers the compile run and trace capture, not trace replay
  (`tt_metal/impl/program/program.cpp` around lines 2992-3008).
- The check uses the lowest occupied L1 address across all compute cores (lockstep banks),
  not only the program's own cores. So when prefill is captured, no decode L1 allocation
  may sit below prefill's circular-buffer high-water mark on any core.
- Each chip's command queue runs programs in order, so prefill and decode never run at the
  same time on a chip.
- Trace nodes hold `shared_ptr<ProgramImpl>` (`tt_metal/impl/trace/trace_node.hpp`), so
  clearing the program cache while traces are alive does not free their programs.

### Keep the prefetcher running (don't stop it between phases)

- Prefetch requests recorded during trace capture are re-sent on every replay
  (`tt_metal/api/tt-metalium/experimental/tensor_prefetcher.hpp`, the
  `QueueTensorPrefetcherRequest` doc).
- With no requests queued, the DRISC senders park on `socket_wait_for_pages` and write
  nothing.
- Decode only prefetches ahead within a step: `_next_layer_on_submesh` hoists the next
  layer on the same submesh (`tt/model.py` around line 2406). No request crosses a step
  boundary, so every request is consumed when a step's last matmul finishes. That is the
  "prefetcher queue is empty" condition, reached without any extra synchronization.
  - If DSpark / MTP (`hoist_prefetch`) is turned on, check that it doesn't queue
    requests past the end of a step.
- Because `stop_tensor_prefetcher` is never called, the captured decode traces stay valid.
  Decode is captured once, not once per question.
- The prefetcher must be running for decode's compile and warm-up run, because
  `matmul_decode` waits on GCB pages. It can't be started only after prefill.

### What a prefill replay overwrites, and how to handle each item

Anything decode allocates in L1 after prefill capture sits under prefill's circular
buffers or intermediates, so prefill overwrites it.

1. **GCB data region.** Safe. It is empty between steps, and every page is written by the
   sender before it is read.
2. **GCB config pages, on every receiver core.** Must be restored. Each page holds the
   read pointer (`fifo_ptr`) and the `pages_sent` / `pages_acked` credit counters
   (layout in `GlobalCircularBuffer::setup_cb_buffers`, `tt_metal/impl/buffers/global_circular_buffer.cpp`).
   The DRISC sender increments `pages_sent` in this page by NoC atomic, and kernels read the
   page and write it back across programs. If prefill overwrites it, the next decode matmul
   sees bad credits and either hangs or reads the wrong pages.
   - Fix: after decode finishes, read the config buffer (ordered on the command queue); after
     prefill, write it back. The config buffer isn't exposed to Python, so this needs a small
     C++ API, for example `experimental::ReadGlobalCircularBufferConfig(gcb)` and
     `WriteGlobalCircularBufferConfig(gcb, bytes)` built on `EnqueueRead/WriteMeshBuffer` of
     `cb_config_buffer_`, plus nanobind bindings and `./build_metal.sh`.
   - The DRISC-side state (the per-mesh DRISC L1 arena) is in DRAM-core L1, which prefill
     doesn't touch.
3. **Decode's L1-resident tensors** (CSA windows, position bias, and the rest of what
   `release_prefetch_buffers` parks). Must be restored.
   - Tensors that the commit rewrites from prefill state every question are already fine.
   - The rest need a re-upload in place after each prefill with
     `ttnn.copy_host_to_device_tensor(host_copy, device_tensor)`. That keeps their addresses,
     so the decode traces stay valid.
   - Alternative: keep them in DRAM permanently, at some decode cost.
4. **Decode sockets** (D2D between submeshes on cores (0,0) and (0,1), H2D input, D2H
   output, MTP). Protected: they are created in the decode constructor, before prefill
   capture, so the allocator keeps prefill away from them. If they sit inside prefill's
   circular-buffer range, prefill capture fails loudly with the clash error. The fix then
   is to release and recreate them around the capture, which is safe before any decode
   trace exists.
5. **Decode DRAM state.** Protected if it is allocated before prefill capture: weights,
   KV caches, and the static session from `prepare_static_decode`. The decode trace's
   persistent outputs are allocated later, but decode writes them before reading them.
6. **Prefill's own state.** Protected by the allocator: it is alive when decode compiles.
   Any overlap between decode's circular buffers and prefill's L1 buffers fails loudly.
7. **Exported prefill states.** They are allocated eagerly after the decode traces exist,
   so they must be committed and then freed (`free_traced_states`) before the next decode
   or prefill replay.

### Order of operations

1. Build decode (weights, sockets, GCBs, L1 tensors), then run `prepare_static_decode`
   sized for the longest question, so the KV caches live in DRAM before the prefill trace
   exists.
2. Call `release_prefetch_buffers()` once. There are no decode traces yet, so this is
   allowed. It needs the GCB leak fix.
3. Call `decode.build_prefill(rope, weights, ...)` (shared experts, embedding, and
   optionally the LM head), then `prefill.prepare_traced_prefill(max_len, C)`: one trace
   per stage.
4. Call `restore_prefetch_buffers()`, which also starts the prefetcher. Then run decode's
   compile and warm-up and capture the decode traces. Do not stop the prefetcher after
   this point.
5. For each question:
   1. Snapshot the GCB config pages.
   2. Replay prefill.
   3. Write the config pages back, then re-upload the non-state L1 tensors.
   4. Commit the prefill states into decode (with `bias_slots`), then `free_traced_states`.
   5. Replay the tail, then generate with the decode trace.
6. At teardown: release the traces, then stop the prefetcher.

Compared with the per-question release and restore plan, this removes the per-question
GCB rebuild, decode recapture, and program-cache clear.

### Open risks (need device runs)

- Both sets of traces (8 prefill stage traces plus decode's) must fit in `trace_region_size`.
- A decode L1 tensor not found by the `_gcb_holders(self, _is_l1_tensor)` walk that is read
  before it is written would be silently corrupted.
- Whether prefill's circular buffers actually reach the GCB config pages can be read from
  the `DEEPSEEK_V4_DUMP_L1=1` memory dumps. If they don't, the snapshot step can be skipped.

## GCB leak (one GCB per device not freed by `release_prefetch_buffers`)

- Cause: `ttnn/cpp/ttnn-nanobind/global_circular_buffer.cpp` binds `receiver_cores` and
  `sender_cores` with `nb::rv_policy::reference_internal`. Any Python-held `CoreRangeSet`
  returned from them keeps the whole `GlobalCircularBuffer` alive. Its `cb_buffer_` and
  `cb_config_buffer_` are shared `AnyBuffer`s, and copies share ownership.
- Places that keep one:
  - `LinearDecode._init_prefetched_weight` stores `self.receiver_cores`, built from
    `global_cb.receiver_cores()`.
  - `LinearDecode.b_core_grid()` returns `global_cb.receiver_cores()`; its callers are in
    `decode/attention.py` (around lines 41, 947, 1010, 1234, 1410-1413) and
    `decode/moe.py` (around line 260).
  - `_prefetch_output_memory_config` and the `BatchedLinearDecode` equivalents in
    `tt/layers.py` (around lines 1310, 1343, 1394).
- Fix, either:
  - change both bindings to `nb::rv_policy::copy` (one line each; `CoreRangeSet` is cheap
    to copy; needs a rebuild), or
  - copy the grid on the Python side wherever it is stored.
- Ruled out: the program cache. `matmul_decode` attributes hold an
  `optional<GlobalCircularBuffer>`, but ttnn's program cache doesn't keep the operation
  attributes. `CircularBuffer` only keeps a raw `shadow_global_circular_buffer_` pointer.
  An earlier `clear_program_cache()` in `release_prefetch_buffers` didn't free the block.
  - `matmul_decode` folds the GCB's config and data addresses into its program hash, so a
    GCB rebuilt at the same addresses reuses the cached program, and one rebuilt at new
    addresses compiles a new program.

## First experiment: prefetcher off

Before building the snapshot and restore machinery, validate the aliasing flow with decode
built without the DRISC prefetcher. There are then no GCBs, no config pages to protect, and
no prefetch requests in the decode traces. That leaves only the L1-tensor re-upload (item 3)
and the allocation order to get right.

Decode is not fully switchable today:
- `DeepSeekV4Model.__init__` hardcodes `self.use_prefetcher = True` (`tt/model.py` around
  line 387) and passes it down to the layers (around lines 555 and 977).
- `decode/attention_csa.py` (around line 515) builds the indexer's compressor with
  `use_prefetcher=True` regardless of the flag.
- The `use_prefetcher=True` settings in `decode/moe.py` sit inside `if use_prefetcher:`
  branches, so they already follow the flag.
- `prefetcher_session`, `prefetch_weights`, `release_prefetch_buffers`, and
  `restore_prefetch_buffers` need to be no-ops when `_prefetch_buffers_by_device` is empty.
- `tests/decode/test_multi_user_paged_decode_demo.py` already reads a
  `DEEPSEEK_V4_PREFETCHER` env var, which is a precedent for exposing the switch.

Expected cost: decode reads weights from DRAM through `ttnn.linear` / DRAM-sharded
matmuls, which is slower per step. That's acceptable for proving correctness of the
interleave.

## Single-trace traced prefill (done, host-verified only)

- One trace per stage at `C = chunk_size`, reused for any 128-aligned prompt length up to
  `max_len`.
- The last chunk is padded with repeats of its own tokens; causality keeps the real tokens
  exact.
- The head selects the last real token with a one-hot `eq(ramp, last)` followed by a
  matmul.
- The export slices the real tail rows, the overlap row, and the real FIFO entries.
- The first device check is `pytest -s models/experimental/deepseek_v4_flash/tests/prefill/test_prefill_traced.py`
  (defaults: `LEN=1152`, `CHUNK=256`, `DEEPSEEK_V4_TRACED_MAX_LEN=2048`, so the last chunk
  is padded).
