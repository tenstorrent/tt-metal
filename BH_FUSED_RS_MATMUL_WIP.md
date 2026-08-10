# WIP: BH Galaxy Qwen3-32B — Prefetcher + Fused `matmul_reduce_scatter_async`

Status doc for resuming later. Branch: `mgiermakowski/bh-fused-test`
(branched from `mgiermakowski/bh-qwen-prefetcher`, which is the working prefetcher-only baseline).

Last updated: 2026-08-07 (deadlock FIXED — see "FIXED" section; e2e demo validation in progress).

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
  decode PCC **~0.9977** per token, matching the unfused baseline (~0.9975).
- The former token-2 deadlock is **FIXED** (2026-08-07): it was NOT the fused op — `ttnn.pad`'s
  row-major sharded program factories placed CBs on the full device grid, and dispatch multicast
  the CB config onto the prefetcher sender column, corrupting the cached DramPrefetcher kernel
  text. See the "FIXED" section near the bottom for the full story.
- `test_qwen_decoder.py` passes all 10 decode iterations with prefetcher + fused op enabled.

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

Prime remaining suspect (SUPERSEDED — see BREAKTHROUGH section below):
- ~~The co-designed 1D gathered matmul reader's remote/global circular-buffer pointer accounting~~
- ~~L1 semaphore-id / CB-config aliasing between the fused program and downstream programs~~
- Actual mechanism: dispatch-level worker done-signal deficit; see next section.

## BREAKTHROUGH (2026-08-06): live tt-triage post-mortem — it is a DISPATCH-LEVEL go/done deficit

The watcher-based localization above is superseded. The watcher itself is an artifact: the
baseline (prefetcher, no fused op) also hangs at token 0 under `TT_METAL_WATCHER=10`, so all
watcher-derived conclusions (slice reader / all_gather multicast_reader) were observer effects.

New method that works (use this next time):
- Reproduce WITHOUT watcher; while pytest is HUNG (do not kill it — the inspector RPC must stay
  alive), run `python tools/tt-triage.py --llm-output --llm-output-path=...` in parallel.
  Needs `tt-exalens==0.3.27` (upgraded in python_env). On the shared broker, emit heartbeat
  output every ~25 s or the 300 s no-output kill fires before triage runs.

Findings from the frozen mesh (all 32 devices identical):
- Op window: token 1 completed fully (both fused ops included; PCC 0.9979). Token 2's
  `DramPrefetcherOperation` (op id 56) is the only RUNNING op; ops 57+ (first worker op of
  token 2, a Reshard) were **never dispatched**.
- Prefetcher `reader_dram` NCRISC is parked in `cb_pop_front` with a clean callstack — starved,
  not corrupted, waiting for consumers that never launch.
- `cq_dispatch` BRISC is stuck in `process_wait` (wait_stream loop, cq_dispatch.cpp:1059).
  Reading its locals via ttexalens (`last_wait_stream`@0xffb00d14, `last_wait_count`@0xffb00d18,
  BRISC private mem, dispatch core logical (12,1)) and the live stream counters
  (`0xFFB40000 + stream*0x1000 + 297*4`; stream 48 = sub-device 0/prefetcher,
  49 = sub-device 1/worker; base 48 from `dispatch_stream_base_`):
  **wait is on stream 49 (WORKER sub-device), target 10700, counter stuck at 10600.**
- The worker sub-device has exactly 100 cores and every mesh program on a sub-device produces one
  done-increment per sub-device core, so the counts are multiples of 100:
  **exactly ONE worker program slot's entire done cycle (100 signals) is missing.**
- `cq_dispatch_subordinate` (go-signal sender) is likewise parked in `wait_for_workers`.
  (Its `last_wait_count` is in NCRISC private mem, unreadable on BH — "ncrisc does not have
  debug hardware" — so whether the missing slot's GO was ever multicast is still unconfirmed.)
- `check_binary_integrity` reported ~6k `.text` mismatches, but the same per-RISC offsets repeat
  across unrelated kernels and the RUNNING kernels backtrace cleanly — treat as a stale-slot
  checker artifact, not real corruption, unless later evidence says otherwise.

Interpretation: everything previously suspected (OpSignaler semaphores, global-CB pointer
accounting, L1 CB aliasing) is off the table as the primary cause. The fused program breaks the
mesh dispatch go/done accounting such that, ~107 worker-program slots in (early token 2), one
program's 100 done-signals never arrive and the dispatcher deadlocks waiting for them. This also
explains why `QWEN_DBG_FUSED_SYNC=1` (worker-sub-device sync after each fused op) passed at
tokens 0/1: the counter was still consistent at those wait points.

Probe script: `/home/mgiermak/probe_dispatch_wait.py` (run under the broker while hung).
Artifacts: `/home/mgiermak/triage_fused_hang.txt` (4 MB), `/home/mgiermak/qwen_fused_hang3.log`.

## BREAKTHROUGH 2 (2026-08-07): the dispatch deficit is a SYMPTOM — token-1's DramPrefetcher never exits

Fresh repro + three new frozen-mesh probes (`probe_worker_mailboxes.py`, `probe_gcb_credits.py`,
fixed `probe_dispatch_wait.py`) overturned the "one worker program loses its 100 done-signals"
reading:

1. **Worker launch-mailbox dump (all 130 tensix cores, device 0)**: every one of the 100 worker
   sub-device cores is `DONE` at `launch_msg_rd_ptr=2`, uniformly. `host_assigned_id` in the launch
   ring == the ttnn op id (`fetch_and_increment_device_operation_id`), which lets us map slots to
   ops: the decoder test dispatches 53 ops/token, prefetchers are op 3 (token 0), 56 (token 1),
   109 (token 2). Workers consumed ALL 106 worker programs of tokens 0-1 (stream 49 = 10600 = 106
   x 100, exactly). No worker program lost any dones.
2. **Prefetcher (col-0) mailboxes**: slot 0 (op 3, token-0 prefetcher) consumed -> token 0 drained
   cleanly. The cores are in `GO` at rd_ptr=1 — **still running op 56, token-1's
   DramPrefetcherOperation** — with token-2's op 109 launch message pre-written at slot 2, its GO
   forever queued in dispatch_s behind op 56's completion (gos are fully serialized per sub-device:
   each go's `wait_count` = expected-workers-completed *before* that program).
3. **Global-CB credits (scanned config page @0xa9200, counters @0xa9240)**: perfectly balanced —
   every sender (0,0)-(0,7) shows `pages_sent == pages_acked == 216832` for all 3 receivers, and
   216832 = 2 x 108416 = exactly two tokens' worth. **The prefetcher is NOT starved by the fused op
   under-consuming weights**: token-1's streaming completed and was fully acked.

So the failure chain is: token-1 `DramPrefetcherOperation` finishes streaming but **never exits its
kernels** (core stays GO) -> token-2's prefetcher go can't be sent (dispatch_s serialized) ->
dispatcher's config-buffer wait for a token-2 worker program (stream 49 target 10700; some devices
10800) never satisfies -> mesh-wide stall. The worker-side "deficit" was just the not-yet-sent go.

Where can the writer/reader pair be stuck AFTER all pushes are acked? The exit path is:
`remote_cb_sender_barrier` (polls the exact L1 words the probe read as equal -> should pass) ->
`update_remote_cb_config_in_l1` -> `noc_async_atomic_barrier` -> writer pushes `sync_cb` ->
reader's final `cb_wait_front(sync_cb)` (reader_dram.cpp:114) -> exit. The old triage found the
reader parked at that final sync wait, i.e. the WRITER never delivered the exit credit — despite
balanced remote credits. Current suspicion is therefore in the writer's post-stream sequence or
the reader/writer local-CB handshake state across cached re-runs, possibly corrupted by something
the fused-op program does to shared L1 state on the *receiver* columns (the prefetcher's
`update_remote_cb_config_in_l1` persists GCB pointers for the next program; the fused matmul does
the same from the receiver side).

Why the baseline doesn't hang: identical prefetcher, identical worker consumption totals — the
only delta is the fused matmul+RS program structure. Token 0 (program-cache miss) drains; the hang
requires the cached-run token 1. A triage callstack of the sender cores' BRISC (writer_l1) +
NCRISC (reader_dram) on the frozen mesh is queued (run script now includes a parallel tt-triage
pass) to pin the exact parked line.

Probe artifacts: `/home/mgiermak/mailbox_probe4.log` (+5), `/home/mgiermak/gcb_credits_probe5.log`,
`/home/mgiermak/dispatch_probe5.log`, `/home/mgiermak/inspector_fused_hang5/`.

## ROOT CAUSE FOUND (2026-08-07, evening): kernel-TEXT CORRUPTION on the prefetcher sender cores

New probes (`probe_reader_pc.py`, `probe_sync_cb.py`) on the frozen mesh nailed the mechanism:

1. **The writer_l1 BRISC finished cleanly** (parked in firmware `wait_ncrisc_trisc`, PC moving) and
   **delivered the exit credit**: the reader-writer sync CB (stream 3) shows
   `pages_received=1, pages_acked=0` on every sender core. No credit bug.
2. **The reader_dram NCRISC is HARD-FROZEN at PC 0xbad4** (identical, non-moving, on all 8 senders
   of every device).
3. From the launch mailbox, op 56's kernel config base is 0xad80 with NCRISC text offset +0x940 →
   **reader text loads at L1 [0xb6c0, 0xbb78)**. Byte-diffing L1 against the cached
   `reader_dram/.../ncrisc.elf.xip.elf` shows **68 corrupted words at L1 [0xb9c0, 0xbad0)**
   (ELF VMA 0x6450-0x655c) — exactly the code for the tail of the block loop, the final barrier,
   and the `cb_wait_front/cb_pop_front(sync_cb)` exit handshake. The frozen PC (VMA 0x6564) sits
   just past the corrupted region: the reader streamed all 3600 blocks fine (local CB counters
   3600/3600), then walked into clobbered instructions and died before consuming the exit credit.
4. **The corrupting payload is a kernel-config CB TABLE**, not data: 17 four-word records with
   entries {CB0: addr 0xa0680, 8 KB, 32 pages of 256 B}, {CB1: addr 0x1b380 (= L1 unreserved base),
   256 B, 1 page}, {CB16: addr 0xa3fc0, 8 KB, 32 pages of 256 B}, rest zero.
5. **The same 68 words appear at the SAME address 0xb9c0 on worker core (4,0)** (a fabric-mux core
   of the fused program), where they are LEGITIMATE program config. Ring matmul cores (1-3,0) and
   RS writer cores (6,0)/(9,0) have different content there. So a kernel-config multicast for some
   worker program's kernel group is ALSO being sent to logical cores (0,0)-(0,7) — the prefetcher
   senders — where 0xb9c0 lands inside op 56's kernel text.
6. Why only token 2: config-ring layout luck. Op 3 (token 0) text sat at [0xa8c0, 0xadd8) — the
   0xb9c0 write missed it. Op 56 (token 1, cached re-dispatch) text at [0xb6c0, 0xbb78) — hit.

Suspicious detail: (0,0)-(0,7) is EXACTLY what `choose_worker_cores(8 workers, sub-device 0)`
returns — and sub-device 0 is the PREFETCHER sub-device. I.e. some CCL helper inside the fused op
(or an op it pulls in) is choosing cores with a defaulted `sub_device_id` (`get_sub_device_ids().at(0)`)
and creating CBs there. Kernels get sub-device validation; CBs do not, so this slips through and
dispatch happily multicasts the CB config onto foreign cores.

A temporary host-side debug log (`TT_DBG_COL0_CB=1`, in `ProgramImpl::add_circular_buffer_`) is in
place to identify the exact program; map its id via inspector `kernels.yaml`.

## FIXED (2026-08-07, night): the offender was `ttnn.pad` (row-major sharded factories), not the fused op

The `TT_DBG_COL0_CB` hunt (with C++ backtrace) caught it immediately. Two per-token programs
created CBs on the FULL 12x10 grid `[0-0 - 11-9]`, i.e. including prefetcher sender column 0:

```
COL0-CB: program=320 range=[0-0 - 11-9] total_size=8192 globally_allocated=true   <- input shard CB
COL0-CB: program=320 range=[0-0 - 11-9] total_size=8192 globally_allocated=true   <- output shard CB
COL0-CB: program=320 range=[0-0 - 11-9] total_size=256  globally_allocated=false  <- pad-value stick CB
backtrace: ... PadDeviceOperation ... pad_impl ... invoke_rm ... ttnn::to_layout ...
```

That is EXACTLY the corrupting CB table found at 0xb9c0 (CB0 8 KB / CB1 256 B / CB16 8 KB).
The call chain is `ttnn.to_layout` → `ttnn::pad` (row-major sharded path) in the per-token
host input prep (rot-mat / page-table massaging), NOT the fused op. It was always there; the
fused op only changed the config-ring layout so token-1's prefetcher reader text happened to
land under the stray 0xb9c0 write (see "why only token 2" above).

Root bug: `pad_rm_sharded_height_only_program_factory.cpp` and
`pad_rm_sharded_width_only_program_factory.cpp` placed all three CBs on
`CoreRange({0,0}, compute_with_storage_grid_size - 1)` (the whole device grid) while their
kernels only run on the shard grids. Dispatch multicasts CB configs to every CB core — including
column 0, which belongs to the *prefetcher sub-device* whose kernel-config ring uses the same
addresses for kernel TEXT. CBs get no sub-device validation, so this silently corrupted the
cached DramPrefetcher `reader_dram` kernel.

Fix (both factories): place CBs only on `input_shard_grid.merge(output_shard_grid)`.

Validation: `test_qwen_decoder.py` with `QWEN_BH_PREFETCHER=1 QWEN_BH_FUSED_ASYNC_RS_MATMUL=1`
now passes **all 10 decode iterations**, PCC ~0.9977 each token. No more token-2 deadlock.
The temporary `TT_DBG_COL0_CB` instrumentation has been removed.

## E2E results with the fix (2026-08-07, night)

Full demo (`text_qwen_demo.py -k batch-32 --max_generated_tokens 64`, prefetcher + fused):
- **PASSED**, all 64 decode iterations, no hang.
- **47.82 tok/s/user average** (1530 tok/s throughput, 20.91 ms/iter) vs **45.98 t/s/u** for the
  unfused control run on the same build — **+4% decode perf** from the fused op
  (vs the older 44.8 t/s/u baseline number: +6.7%).

Accuracy (`test_qwen_accuracy.py`, 511 teacher-forced tokens):
- fused:   Top-1 **27.2%**, Top-5 34%
- unfused: Top-1 **27.4%**, Top-5 34% (same build, same flags minus fusion)
- => The fused op does NOT regress accuracy — it exactly matches the unfused prefetcher path.

**Open issue (pre-existing, NOT caused by the fused op):** the prefetcher decode path
(`QWEN_BH_PREFETCHER=1 QWEN_BH_UNFUSED_CCL=1`) scores ~27% Top-1 on this accuracy test,
uniformly from the first token window (20-43% per 100-token window; no drift over time), and demo
text is garbled. The 96%-Top-1 runs from July were on the non-prefetcher e2e branch. Wrong tokens
repeat a small set of junk vocab entries (脏, _CAMERA, satu, 漂) which smells like corrupted logits
for specific vocab shards (lm_head gather / fast_reduce_nc path?). Needs its own investigation on
the prefetcher baseline branch, independent of this fused-op work.

## RESOLVED (2026-08-08): prefetcher-path accuracy 27% -> 89% Top-1

Two independent bugs, both now fixed:

1. **lm_head DRAM-sharded weights broken on BH (27% -> ~61%).** The lm_head decode ring matmul's
   `create_dram_sharded_mem_config_lm_head` bank/offset layout is hardcoded for WH's 12 DRAM banks;
   on BH's 8 banks the ring cores read misaligned weight slices, producing near-random logits in
   specific vocab shards (the junk-token clusters). Fix: on BH keep the lm_head decode weight
   DRAM-**interleaved** (`lm_head.py`), and make `LM_HEAD_OUT_RING_MEMCFG` use the input ring
   grid's core ordering (`qwen_model_config.py`).

2. **Force-argmax sampling gather raced the untilize/argmax under trace (61% -> 89%).**
   `tt_sampling.py`'s force-argmax `all_gather_async` did not pass `subdevice_id`, so
   `choose_worker_cores` defaulted to `get_sub_device_ids()[0]` — which, with the prefetcher
   decode sub-device manager loaded, is the **prefetcher/senders sub-device (col 0)**, not the
   worker sub-device where the downstream untilize/argmax run. Dispatch (and trace replay) only
   serializes programs *within* a sub-device stream (`simple_trace_allocator.cpp` skips nodes with
   a different `sub_device_id`), so under trace the untilize launched concurrently with the gather
   and read the gather's output buffer before this step's writes landed -> argmax returned the
   *previous* step's tokens (the +3-chunk "displacement" was stale data, not a permutation).
   Eager runs masked it because per-op host dispatch latency exceeds the gather runtime.
   Repro: `test_qwen_sampling_argmax.py` with `QWEN_SAMPLING_TEST_TRACE=1` (fails 24-32/32 rows
   stale-by-one-iteration before the fix, deterministic with `QWEN_SAMPLING_TEST_SYNC_BEFORE=1`).
   Fix: pass `subdevice_id=tt_ccl.worker_sub_device_id` to the gather (`tt_sampling.py`), matching
   every other CCL call in `llama_ccl.py`.

Post-fix (`test_qwen_accuracy.py`, 511 teacher-forced tokens, prefetcher on): Top-1 **89%**,
Top-5 **~98%** (assert threshold 98% is borderline: observed 97.7-98.1 across runs). With
`QWEN_ACC_HOST_ARGMAX=1` the on-device sampled token matched the host argmax of the logits on
**all 511 steps** — the sampling path is now exact. The residual ~7-point Top-1 gap vs the 96%
non-prefetcher baseline is numerics of the prefetcher compute path (bfp8 ring matmuls / CCL
dataformats), not corruption.

## Suggested next steps (2026-08-06 list — items 1-2 RESOLVED by Breakthrough 2)

1. Identify the missing program slot. Only ~3 worker programs run between token-1's PCC readback
   (which waited successfully) and the stuck op 57 (two Embeddings + possibly a reshard) — but if
   the wait targets launch-ring-slot reuse (wait for slot N-ring_size), 10700 may map to an
   earlier slot, i.e. the fused op itself. Read a worker core's mailbox launch ring / go-message
   state (dev_msgs.h `mailboxes_t`) on the frozen mesh to see whether the missing slot's GO was
   delivered.
2. Audit mesh `EnqueueProgram`/dispatch_s expected-worker accounting for the fused program: it has
   kernel groups on disjoint, non-rectangular core sets (ring matmul cols 1–3 incl. hop cores + RS
   sub-grid cols 4–10) inside a 100-core sub-device. Compare `num_active_cores` /
   go-mcast rectangles vs the plain (non-fused) matmul and RS programs.
3. Narrow the repro: fuse only FF1 (one fused call per layer) — if the hang moves to token 4-ish,
   it is a per-fused-call accounting leak; if still token 2, it is per-program-structure.
4. Escalate to the runtime/dispatch team with this doc: "one worker-sub-device program slot loses
   all 100 done-signals after N dispatches of a multi-kernel-group fused program under sub-devices
   + program cache on BH" is their domain.

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

## Decode perf: 47.6 -> 50.7 t/s/u (2026-08-08)

Traced-decode profiling (qwen_trace_pf_b32_v2 capture, device 16, one replay = one token,
~20.4 ms span):

- Worker stream: 15.6 ms busy + 4.9 ms dispatch gaps across 2673 ops/token (~42 ops/layer on the
  unfused-CCL path vs ~25 on WH's fused-CCL path — that op-count delta is the structural gap).
- Per layer (~330 us): FF1/FF3 ring matmul pair 73 us (the second one is gated by the prefetcher's
  DRAM stream), 2x reduce_scatter_minimal_async 45 us, 3x all_reduce 26 us, 2x distributed-norm
  chains (reshard+LN+AllGather+LN+reshard) ~28 us, QK-norm with 4 reshards ~10 us, 4 rotary
  cos/sin slices ~9 us.
- Sampling trace (separate trace, serialized after decode, every token): gather 206 us +
  untilize 201 us + **argmax 2.1 ms**. Argmax is a scalar-RISC compare loop over the full gathered
  padded vocab (32 x 155648) and was pinned to the 40-core `sub_core_grids`.

### Fix landed: wide force-argmax grid (+3 t/s/u)

`qwen_model_config.py` now sets `force_argmax_sub_core_grids` = the full 100-core worker
sub-device (cols 1-10, rows 0-9) on the BH prefetcher path; `tt_sampling.py` prefers it over
`sub_core_grids` for the force-argmax untilize/argmax. Argmax/untilize CBs are tiny, so unlike
top-k they coexist with the resident global CB on receiver columns 1-3.

Result (batch-32, 64 tokens, prefetcher + fused async RS matmul):
**19.73 ms @ 50.68 tok/s/user (1622 tok/s)**, steady-state iterations ~20.07 ms — up from
20.99 ms @ 47.64 t/s/u. Accuracy re-verified after the change (see accuracy section).

### Tried and parked: rotary slice hoist (QWEN_BH_HOIST_ROT_SLICES=1, default off)

The 4 rotary cos/sin slices + 4 to_layouts per layer are loop-invariant (same rot_mats object,
static head shapes) — hoisting them to once-per-token would save ~0.5 ms. Implemented as a
tt_ccl-cached memo (llama_attention.compute_decode_rot_slices) with a model-level preamble
preslice, but the cached tensors persist through the layer loop and the BH ring FF matmul's
static CB region (ends 444480 on cols 1-6) leaves ~zero persistent L1 headroom on those cores:
trace capture trips "static circular buffers clash with L1 buffers" (lowest buffer 365568-392320
< 444480) even when the tensors allocate at the very top of the decode forward. Env-gated off.
To revive: shrink to a single cos/sin pair (q and k bounds are identical => 2 tensors, 16 KB) or
carve headroom out of the ring matmul CB budget.

### Fix landed: distributed force-argmax (+2.1 t/s/u, 2026-08-08)

`tt_sampling._distributed_force_argmax` (default on via `QWEN_BH_DIST_ARGMAX`, BH prefetcher path
only): each column device argmaxes/maxes its local 32 x 19456 shard, only the per-column
(value, index) candidate tiles are gathered (1-link `all_gather_async`, worker sub-device), the
winner column comes from a small argmax over the gathered values, and the token id is
reconstructed exactly with int32 SFPU arithmetic (hi/lo byte split for the TF32-safe one-hot
select, then `<< 8 | lo` + column offset — see the TF32 note in the code).

Demo-flow engagement gotcha: the galaxy generator passes the decode trace's frozen token-input
buffer (RM uint32 [1,1,1,32]) as `tt_out_tok`, and the first cut of the guard bailed on any
caller-provided buffer — so the first "with dist argmax" perf run (19.48 ms @ 51.33) actually ran
the old path. The write-back is now a no-op `ttnn.bitwise_or(tok, 0, output_tensor=tt_out_tok,
sub_core_grids=worker_grid)`: `ttnn.copy` can't be used because its factory grids from core
(0,0), which under the split senders/worker sub-device manager lands the copy on the senders
sub-device and races the worker-sub-device producers under trace replay (same failure mode as
the unpinned all_gather). binary_ng honors `sub_core_grids` directly and supports RM uint32.
Verified by `test_qwen_sampling_argmax.py` (now passes a demo-style `tt_out_tok`; 8 trace
replays, 0 mismatches).

Result (batch-32, 64 tokens): **18.93 ms @ 52.82 tok/s/user (1690 tok/s)** — up from
19.73 ms @ 50.68.

Also landed: rotary K-share (`compute_decode_rot_slices` reuses the Q cos/sin slices for K when
the bounds differ only in the heads dim) — removes 2 slice + 2 to_layout ops per layer on the
per-layer (non-hoisted) path.

### Rotary hoist retry with K-share: still 19 KB short (2026-08-08)

With the K-share halving the cached slices to one cos/sin pair, `QWEN_BH_HOIST_ROT_SLICES=1`
now fails trace capture with lowest buffer 425088 < 444480 (was 365568-392320 with 4 tensors) —
the deficit is now exactly the persistent footprint (2 x 8 KB shards + alignment). L1 buffer
allocation is lock-step across banks, so any persistent addition lowers the global floor by its
size regardless of allocation order or shard placement; the only remaining lever is carving
>=19392 B out of the ring FF matmul CB budget (region ends 444480 on cols 1-6), which is
perf-critical matmul surgery. Parked again.

### Fresh profile of the 18.93 ms build (qwen_trace_pf_b32_v3, 2026-08-08)

Capture: batch-32 demo, 6 tokens, QWEN_PROFILE_DRAIN=1 under tracy. Device 16, decode trace id 9
(2404 ops/replay = one token) + sampling trace id 10 (19 ops/replay — the distributed argmax).
Analysis from `cpp_device_perf_report.csv` only (the 44 GB `profile_log_device.csv` post-process
step is unnecessary and was skipped; report CSV is written incrementally by the drain).

Union-of-intervals busy per token = 18.87 ms == the measured wall (18.93), i.e. no dispatch
holes are left; the time is in the serialized op chain. Critical-path attribution (increment
each op adds to the busy frontier, one full token):

| op | n/token | critical | note |
|---|---|---|---|
| MatmulReduceScatterAsync (fused FF pair) | 128 | 5.43 ms | FF3 kern ~62 us: gated by prefetcher DRAM stream |
| Matmul (QKV / attn-out / FF2 / lm_head) | 193 | 2.85 ms | QKV kern 41 us, attn-out 38 us |
| AllReduceAsync | 192 | 2.79 ms | 3/layer, 14.5 us critical each (FW window ~30-47 us incl. wait) |
| LayerNorm | 386 | 1.83 ms | 6/layer: 2 distributed-norm pre/post pairs + 2 QK norms |
| AllGatherAsync | 100 | 1.22 ms | |
| Reshard | 671 | 1.02 ms | 10.5/layer, ~1.5 us each |
| SdpaDecode | 64 | 0.68 ms | |
| AllGather (legacy, 2-core norm-stats gather) | 74 | 0.64 ms | 8.6 us critical each — fabric-latency bound |
| BinaryNg | 198 | 0.56 ms | kern 1.3-2.8 us; FW window up to 64 us = waiting for all 100 cores free |
| everything else (incl. sampling ArgMax 0.20) | | ~1.9 ms | sampling total now ~0.4 ms |

### Remaining ideas towards ~55 t/s/u (need ~0.75 ms), effort-ranked

- Distributed-norm chain (reshard + LN-pre + 2-core legacy AllGather + LN-post + reshard, x2 per
  layer): LN 1.83 + legacy AG 0.64 + a chunk of the reshards ~= 2.5-3 ms/token. WH fuses this
  into RMSAllGather (`fused_rms_minimal`), whose 1D-multicast writer no-ops on the BH 2D-torus
  fabric — porting it is the same class of kernel co-design as the fused RS matmul was, and the
  single biggest lever left (~1.5-1.8 ms -> ~57-58 t/s/u by itself). Merely swapping the 2-core
  legacy stats gather for all_gather_async is NOT a win: the async gathers average the same
  ~12 us critical (fabric-latency bound at this size).
- QK-norm reshard sandwich (4 reshards + 2 LNs per layer on 16/8-core grids).
- Rotary hoist: blocked on 19392 B of L1 under the ring-matmul CB ceiling (see above); would
  need CB-budget surgery on the gathered matmul.
- FF3 kernel time (62 us vs FF1's 21 us) is prefetcher-DRAM-stream bound — check
  enable_performance_mode / reader tuning headroom before touching anything else.

### Accuracy with distributed argmax (2026-08-08)

Full accuracy run (512-token teacher-forced, batch 32) with the distributed argmax engaged:
**Top-1 93% | Top-5 99%** — up from 88-89% / 97.7% with the single-op full-vocab argmax. The
distributed path typecasts the logits shard to bf16 before both max and argmax (so reduce and
argmax see identical values) and reconstructs indices with exact int32 arithmetic, which appears
to have removed a residual quantization artifact of the old path as a side effect.

## Fused distributed norm (fused_rms_minimal) PORTED to BH 2D torus (2026-08-08)

The RMSAllGather writer (`rms_allgather/device/kernels/dataflow/rms_writer.cpp`) built raw 1D
`MulticastRoutingCommandHeader`s for its stats all-gather, which no-op on the BH 2D-torus fabric
(PACKET_HEADER_TYPE = HybridMeshPacketHeader needs dst mesh/chip ids + per-direction hop counts).
Port = the same kernel co-design pattern as the fused RS matmul:

- Host (`rms_allgather_program_factory.cpp`): compute neighbor coords
  (`get_physical_neighbor_from_physical_coord`) and emit 6+6 multicast route CT args via
  `get_forward_backward_line_mcast_configuration` (exactly what the BH-proven
  all_gather_async llama-sharded writer does).
- Kernel: `ccl_routing_utils::fabric_set_line_multicast_route(pkt_hdr_fwd/bwd, route_info)`
  replaces the raw 1D headers. The fused write+atomic-inc packet works fine over the 2D
  multicast route (all_reduce_async uses the same combination).
- On 1D fabrics the route helper reproduces the old behavior bit-for-bit -> WH unaffected.

Model wiring (BH prefetcher only, gated `QWEN_BH_FUSED_RMS`, default ON, requires unfused-CCL path):
- `use_bh_fused_rms` in qwen_model_config; DistributedNorm decode routes back through
  `tt_sharded_distributed_rmsnorm` (fused op) instead of the unfused pre/AG/post chain.
- Fused norm runs on the LN grid at col 7 ((7,0)-(8,4)) to stay clear of the prefetcher GCB
  receiver cols; input is resharded there first (`ln_sharded_input_memcfg`).
- LAYERNORM persistent stats buffer moved to (7,0) (must live on the op's sender core).
- Dedicated global-semaphore pool on the full worker grid (cols 1-10) since `sub_device_crs`
  excludes col 7.

Unit test `models/demos/llama3_70b_galaxy/tests/unit_tests/test_bh_fused_rms_allgather.py`
(exact model geometry: 5120 hidden / 4-way row fracture / Ring / 10-core grid at (7,0), stats on
sender core): eager + 3x trace replay all pass PCC 0.999 on all 8 mesh rows.

### E2E result: 18.51 ms @ 54.03 t/s/u (from 18.93 @ 52.82)

Demo (batch-32, 64 tokens): early iterations 18.44-18.51 ms, drifting to ~18.8-19.0 by iter 11+
(same drift pattern existed before; average 18.51). Only ~0.4 ms of the predicted 1.5-1.8 ms
materialized. Follow-ups queued: accuracy run + fresh tracy profile to see where the fused chain
actually lands (suspects: the added reshard-in to col 7, fused-op stats-gather serialization,
or the norm chain simply overlapping less than the old profile suggested).

### Residual grid moved to LN cols 7-8: 18.4 ms @ 54.36 t/s/u (2026-08-08)

Post-fused-norm tracy showed ReshardDeviceOperation at 0.836 ms critical; the biggest slice was
the per-norm reshard of the residual (cols 1-2) onto the fused-norm LN grid (cols 7-8),
129 instances/token. Fix: when `use_bh_fused_rms`, DECODE_RESIDUAL_MEMCFG now lives on
(7,0)-(8,4) directly (qwen_model_config), making the norm's reshard-in a no-op — all other
residual consumers (adds, attn/MLP output all-reduces, embedding) are grid-agnostic.

One trap: `all_reduce_async` requires every *output* core to hold a shard of the persistent
interim buffer (the buffer doubles as per-output-core reduction scratch;
`buffer.grid.contains(output.grid)` is validated). The axis-0 CCL buffer was sharded on
sub_core_grids (cols 1-3, 5-6) only, so the dense-out/FF2 all-reduce writing to the relocated
residual grid failed validation. Fixed in llama_ccl.get_persistent_buffers: merge the residual
grid into the axis-0 buffer CRS (50 -> 60 cores, ~35 KB extra L1 on each of the 10 LN cores).

Accuracy with the relocated residual grid is unchanged: Top-1 93%, Top-5 100% (511 tokens).

Net: 18.4 ms @ 54.36 t/s/u (was 18.51 @ 54.03) — only ~0.1 ms, so the norm-input reshard was
NOT the dominant reshard cost; the remaining Reshard critical time is spread across norm-out
-> ring grid and attention-side reshards. Remaining gap to 58 t/s/u is ~1.35 ms, with the top
critical-path items being MatmulReduceScatterAsync (5.45 ms), AllReduceAsync (3.19 ms) and
Matmul (3.11 ms).

### llama_rs_matmul BH hang ROOT-CAUSED + FIXED (2026-08-09)

Two independent bugs, both now fixed:

1. **RS writer fabric routing (fixed earlier today)**: writer_llama_reduce_scatter used
   hop-count-only unicast routes (fabric_set_unicast_route(num_hops)) which the BH 2D-torus
   HybridMesh routers cannot resolve. Ported to host-computed {mesh_id, chip_id} routes via
   ccl_routing_utils (same mechanism as the fused-RMS/all_gather_async ports). Standalone
   test_bh_llama_reduce_scatter passes (PCC 0.9998, eager+trace).

2. **Packet-worker / matmul-ring core overlap (the actual model hang)**: the fused
   llama_rs_matmul builds ONE program with the ring matmul + RS. The matmul factory subtracts
   the RS cores from its worker set (restricted_cores, matmul_multicore_reuse_mcast_1d
   factory line ~1998) and silently drops any ring core inside the RS grid. On WH the ring
   (PREFETCHER_NOC1_GRID) and PACKET_WORKER_CRS are disjoint by design (the comment in
   qwen_model_config even documents the constraint); on BH the ring is cols 1-3 rows 0-7,
   which fully covers the old PACKET_WORKER_CRS ((1,1)-(3,2),(1,3)-(2,3)). Result: 8 of 25
   ring cores got no matmul kernel and the in0 ring gather deadlocked at
   noc_semaphore_wait_min (confirmed by tt-triage on the hung unit test: only 17 ring cores
   running, no RS kernels, readers parked in signal_sem.wait_min).

   Fix: BH PACKET_WORKER_CRS moved off-ring to (5,0)-(6,2) + (5,4)-(6,4) (8 cores, avoiding
   the RS sender row (5,3),(6,3) and hop core (3,8)). qwen_model_config.py.

Unit test: models/demos/llama3_70b_galaxy/tests/unit_tests/test_bh_llama_rs_matmul.py —
fused ring-matmul + RS + signaler geometry (single-weight rs_tensor variant; the two-weight
variant requires the prefetcher global CB and DRAM-sharded weights, rejected standalone by
matmul validation). PASSES: RS PCC 0.9998, matmul PCC 0.9997, eager + trace.

Demo run with QWEN_BH_FUSED_RS_MATMUL=1 QWEN_BH_FUSED_ASYNC_RS_MATMUL=0 in flight.
Result: 56.38 tok/s/u, accuracy 93% top-1 / 100% top-5.

### llama_rs_create_heads BH hang ROOT-CAUSED + FIXED (2026-08-09)

Despite carrying the same {mesh_id, chip_id} routing port as the (working) plain
llama_reduce_scatter, the create-heads variant still deadlocked: tt-triage on the hung unit
test showed all packet-worker readers parked in noc_semaphore_wait (reader line 158, waiting
for 3 fabric increments) with every sender kernel already finished - packets entered the
fabric and vanished, wedging eth cores (board needed glx_reset afterwards).

Root cause: **fabric injection direction vs. source-computed 2D route mismatch**. On the 2D
torus, fabric_set_unicast_route(HybridMeshPacketHeader, dst_chip, dst_mesh) encodes the FULL
hop-by-hop route into header->route_buffer from the *sender's* routing table
(decode_route_to_buffer), so the packet must be injected into the EDM connection matching the
table's first hop. The create-heads writer load-balances ring 2-hop targets by chip parity
(odd chips inject them into the BACKWARD connection) while the routing table resolves 2-hop
ties FORWARD (the plain-RS writer always picks forward on ties, which is why it worked).
Backward-injected packets carried a forward route program and were never delivered.

Fix (kernel-only, llama_reduce_scatter_create_heads writer): on 2D fabrics choose the
injection direction from shortest-path-forward-on-ties (matching the routing table); 1D
fabrics keep the parity load-balancing so WH 6U is unchanged.

Unit tests (all PASS): test_bh_llama_rs_create_heads (eager+trace, q/k/v PCC 0.99996),
test_bh_all_gather_concat (eager+trace x wh_order/bh_ring_order after the factory fixes:
dynamic concat_arg_cores, sender-worker exclusion by actual cores, runtime-arg NOC coords,
dynamic semaphore mcast ranges/dest counts).

Demo run with all three fused CCL ops (QWEN_BH_FUSED_RS_MATMUL=1 QWEN_BH_FUSED_QKV_RS=1
QWEN_BH_FUSED_AG_CONCAT=1): 56.71 tok/s/u, accuracy 93% top-1 / 99% top-5 (accuracy test
needed its hardcoded _IS_BLACKHOLE rot-table gates switched to tt_model.use_fused_rope,
since the BH fused-QKV path now uses the fused-qk rope tables).

### Traced-decode device profile with all fused ops (2026-08-09)

Tracy capture (qwen_prof_allfused): per decode token on device 0, worker sub-device is ~94%
busy (op sum ~16.5 ms of ~17.6 ms token). Per-layer big rocks (us/call):
Matmul_RS 70.9 | plain Matmuls 3x ~9.2 | w3 ReduceScatterMinimalAsync 20.8 |
RMSAllGather 2x 10.1 | AllReduceAsync 2x 9.9 | SDPA 17.8 | AllGatherConcat 16.7 |
w2-in AllGatherAsync 12.6 | qk-norm chain (2 LayerNorm + 8 reshard + tilize/untilize) ~17.
Sampling tail ~0.5 ms (2x ArgMax 125 us + 3x Reduce 87 us).

Follow-up experiments:
- w3 RS via the ported llama_reduce_scatter instead of reduce_scatter_minimal_async:
  ~1 ms/token SLOWER in the full model (54.2 vs 56.7) despite the standalone op being
  faster - it serializes against the fused Matmul_RS packet workers / interim buffers.
  REVERTED (note left in llama_ccl.line_reduce_scatter).
- Paged SDPA k_chunk 64 -> 128 (QWEN_BH_SDPA_K_CHUNK, default now 128): K/V CBs ~128 KB
  still clear the resident-global-CB L1 clash, halves flash-decode K iterations.
  56.92 tok/s/u (17.57 ms). KEPT.

Remaining gap analysis (why not faster yet): Matmul_RS at 70.9 us is ~9x its pure-compute
cost (~8 us); FF1+FF3 weights are 8.7 MB/device/layer, so the ring matmuls are paced by the
prefetcher stream rate (~134 GB/s effective vs >500 GB/s DRAM). All ring matmuls together
are ~6.3 ms/token of the 17.6 ms budget. The single biggest future lever is the BH
dram_prefetcher streaming rate (readers x receivers geometry, global-CB churn, mcast
cadence), not another CCL fusion; the CCLs are at or near their 2-link fabric floors.
