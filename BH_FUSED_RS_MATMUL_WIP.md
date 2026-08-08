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
