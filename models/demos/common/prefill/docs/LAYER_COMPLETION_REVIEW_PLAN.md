# Layer-completion v2 branch: review, cleanup, correctness and test plan

Branch: `asaigal/layer-completion-v2-ack-ring` (tenstorrent/tt-metal), three commits not on `main`
(merge base `a08819ddbe2`):

| commit | what |
|---|---|
| `d19f8a14c41` | PR #55286: v2 structured layer-completion protocol; unified per-protocol plumbing (23 files, +2644/-280) |
| `4b760872ed2` | LayerAckService nanobind ctor takes `ack_layer_ids` as a Python sequence (works around a list-caster corruption) |
| `b6281b87631` | D2H layer acks: record ring on the service core, credit-based reader (fixes the single-slot overwrite and the `==` wedge) |

Version 2 (2026-10-10). Version 1 was vetted by four independent reviewers reading the code (C++ core, Python
and runtime, tests, comments and naming); their confirmations, disputes and misses are folded in below. Every
finding carries a file reference.

Ground rules for the cleanup: no new knobs; comments terse and in one place; one commit per logical step; nothing is
pushed until asked.

---

## 1. What the branch does

**Producer side (per rank).** A completion event is "layer L of chunk k on slot S covering positions [a,b) is done".
Two transports raise it:

- host callback (`kv_ack.py` → `TtPrefillRuntime._layer_complete_cb` / traced controller) → a
  `LayerCompletionSink` (`layer_completion_sink.py`) → the host-local SHM ring (`LayerCompletionQueueT<Msg>`);
- device (`PREFILL_LAYER_ACK_D2H=1`): the ack op (`outbound_socket_service_sync`) ships the chunk's 12-byte
  metadata record through the D2H service; `LayerAckService::reader_loop` turns each record into a message using a
  per-rank counter (`k / local_layers` = chunk, `k % local_layers` = ack index) plus the record's identity words.

**Router (per host, `LayerCompletionRouter`).** Drains the host-local ring; subordinates MPI-send every message to
the master; the master also receives from every subordinate. v1: reorder by dense `seq` and `inject(n)` into the
scheduler-facing `InterProcessCounterChannel`. v2: forward as arrived into a scheduler-facing
`LayerCompletionQueueV2` ring, with backpressure when that ring is full.

**Consumer side.** tt-d-gen's prefill reader (lcv2 tree) connects to the scheduler-facing segment and, under v2,
retires chunks by `(slot, pos_start, pos_end)` with per-layer coverage. The in-tree Python consumer
(`layer_completion_drainer.py`) does the same with coverage keyed by `request_id`, for the producer harness and
tests.

**Wire contract.** v1 message 24 B, magic `LCQ1`, 32 B cells; v2 message 40 B, magic `LCQ2`, 64 B cells; ring
capacity 1024 (65,664 B segment for v2). `seq` is the reorder key in v1 and diagnostic in v2. End-of-stream sentinel
in `reserved` (v1) / `flags` (v2).

**ACK space.** Hybrid stacks ack only on KV-writing layers, so every counter-derived index runs in "ACK space"
(`num_ack_layers`, `ack_first_idx`, `ack_local_count`, `ack_idx_of_layer`, `my_ack_layer_ids`), derived in
`prefill_runner._serve_request` and handed to both transports.

**What CI exercises today.** Kimi e2e stage: D2H on, eager, protocol 1. Multirank kimi27 / glm53 / kimi_k3: D2H on,
traced, protocol 1. The host-callback transport is not exercised through the runner in CI at all; protocol 2 is not
exercised anywhere in CI.

---

## 2. Findings

Severity: **S1** wrong or unsafe, **S2** fragile or will break under maintenance, **S3** quality.

### 2.1 Correctness, races, logic

**F1 (S1) The shared scheduler shm name corrupts in both directions.** Both protocols publish at
`/tt_prefill_layer_acks_<service_id>` (`prefill_runner.py:703`, `layer_completion_drainer.py:277,293`).
`InterProcessCounterChannel::connect` (`inter_process_counter_channel.cpp:107-146`) validates only the name and
mmaps 128 B of whatever is there. Counter layout: `producer_counter` at 0, `consumer_cursor` at 64,
`prior_clean_shutdown` at 68; v2 ring header: `enqueue_pos` at 0, `dequeue_pos` at 64. A protocol-1 consumer on a
protocol-2 runner reads the low word of `enqueue_pos` as its count, writes its cursor into the low word of
`dequeue_pos` (`:221`) and writes `prior_clean_shutdown` into the high word at attach and at shutdown
(`:157`, `:261`), leaving `dequeue_pos >= 2^32`: every later v2 consumer is wedged. The reverse direction fails in
`NamedShm::open`'s size check, but the in-tree drainer swallows that with a warning and "skips drain"
(`layer_completion_drainer.py:279-281`); only tt-d-gen fails loudly. In practice the mismatch happens because the
runner and the producer read `PREFILL_LAYER_COMPLETION_PROTOCOL` independently and `_apply_manifest_env` has no key
for it (`prefill_producer.py:40-90`). In-tree v1 consumer on the same name: `gemma4_d_p/tests/test_prefill_service.py:112`.
Fix: protocol-specific names; the runner unlinks both stale names; one helper owns the naming (C1).

**F2 (S1) `request_id` means different things per transport under v2.** Host sink: the runner's chunk id
(`prefill_runner.py:401` → sink). D2H: `chunk = k / local_layers` (`layer_ack_service.cpp:151-152`, emitted at
`:193`). They coincide (both 0-based, warm-up records drained before `start()`) until the first desync, after which
the ranks disagree. The drainer keys coverage on it (`RequestCoverage`); the tt-d-gen reader keys on
`(slot, pos_start, pos_end)` and ignores it. Fix: consumers key on identity (C3).

**F3 (S1) Two incompatible full-ring policies; one kills the runner.** `_push_with_spin`
(`layer_completion_sink.py:61,77,88-93`) raises after 30 s of a full ring and warns on every blocked message;
`LayerAckService::push_blocking` (`layer_ack_service.cpp:118-126`) waits forever. The design is backpressure all the
way back (scheduler ring full → master spins → subordinate MPI sends block → per-rank rings fill → producer waits);
a timeout that aborts the runner contradicts it. Fix: one policy (C2).

**F4 (S1, new) Protocol 2 with no consumer kills the runner.** With `verify` off the producer attaches nothing
(`prefill_producer.py:1557-1571`); nothing drains the v2 scheduler ring; the master spins
(`layer_completion_router.cpp:235-247`), the per-rank rings fill (1024 messages, about 16 chunks of 61 layers) and
the sink raises after 30 s. v1 was harmless because a counter accumulates. Resolution (changed after vetting): the producer must not
attach unconditionally, because in a deployment the real consumer owns the ring and a second consumer would
steal messages. With C2 the runner waits on its full ring with a warning every 10 s instead of dying; a
protocol-2 runner without a consumer is a misconfiguration that is now visible and lossless. The producer's
log says so when CHECK_PCC is off.

**F5 (S1, new) Protocol 2 crashes the producer on MTP / DFlash.** Extra acks land at global layers at or past
`NUM_LAYERS` (`tt_prefill_runtime.py:372,1173-1189`, `tt_dflash_drafter.py:572-579`,
`tt_prefill_transformer.py:638`); the drainer's span bound is `PREFILL_NUM_LAYERS` (`layer_completion_drainer.py:124-127,339`)
so `add_span` raises `ValueError`, which `_drain_layer_completion_ring` does not catch (`:341-344`, TimeoutError
only). The e2e `glm53_mtp` scenario (`test_producer_runner_e2e.py:182-183`) therefore fails under protocol 2. Fix:
the drainer's bound is an explicit parameter equal to the widened global layer count (C3).

**F6 (S1, new) The PR deleted functions a live caller still uses.** `migration_driver.py:843,869` call
`producer._connect_layer_ack_channel` and `producer._drain_layer_acks`, removed from `prefill_producer.py` by
`d19f8a14c41`: `AttributeError` on every run of that driver. It also still computes `NUM_LAYERS * total_pushes`,
wrong on hybrid and MTP models. Fix: route it through `connect_layer_completion_channel` /
`drain_layer_completions` with the ack-layer count (C1).

**F7 (S2) Desync handling in `LayerAckService` is warn-and-continue.** After a detected drop (identity changes at
`slice_idx != 0`, `layer_ack_service.cpp:303-323`) the service keeps emitting messages it knows are mislabelled.
With the ring fix, host backpressure can no longer drop records, so a desync now means a bug (remaining sources:
the non-traced path starts the service without draining warm-up records, `prefill_runner.py:822` versus the traced
drain at `:1004-1010`; `runtime.layer_ack_layers` widening against a fixed `local_layers`). Fix: fail loudly, but
keep draining the D2H FIFO so the device never stalls (C4). Decision for the user.

**F8 (S2, new) Teardown can hang in the D2H destructor.** `~D2HStreamService` calls an untimed `barrier()`
(`d2h_socket_service.cpp:657` → `d2h_socket.cpp:736`, spins until host-acked == sent). The runner's `finally`
(`prefill_runner.py:1036-1040`) stops the ack service and then drops the D2H service; any unread record (abort
mid-chunk, or a reader that stopped reading) hangs the process in the destructor. Fix: the reader keeps draining
after a desync (C4) and the destructor's barrier gets a bound (C9).

**F9 (S2, new) No exception boundary on either background thread.** `reader_loop` TT_FATALs (`:254`, the
cross-socket mismatch in `read_metadata_from_sockets:1252`) and listener MPI throws reach `std::terminate` with no
Python-visible error. Fix: catch, store an `exception_ptr`, surface through `stop()` and a `check()` (C4, C5).

**F10 (S2, new) The v2 teardown drop bound starts at the first blocked push, not at `stop()`**
(`layer_completion_router.cpp:234-245`). A scheduler legitimately 30 s behind (the runner's teardown timeout is 30
s, `prefill_runner.py:734`) loses its whole backlog at the first poll after `stop()`, and `scheduler_gone` then
discards the rest. Fix: start the deadline at `stop_` (C5).

**F11 (S2, new) The sink's shutdown abort has no clean exit.** The `RuntimeError` from `_push_with_spin`
propagates out of `controller.replay()` into `run_request_loop`'s `finally` (`:1039-1049`), but `release_trace()`
and `close_mesh_device` run only on normal return (`:648-656`), which `release_trace`'s docstring says segfaults.
Fix: on shutdown the sink drops the message and returns; the request loop exits normally at the next chunk (C2).

**F12 (S2) Environment reads at import time.** `prefill_runner.py:170` evaluates `current_protocol()` at import;
`layer_completion_sink.py:61` reads the timeout env at import. Partial: the runner reads about twenty env vars at
import, so this alone does not make it importable; it still removes the one that raises. Fix in C8.

**F13 (S2) Polymorphism by name only.** `LayerCompletionQueueBase` and `SchedulerEgress` exist so the router can
hold either protocol in one member; every use site then `static_cast`s (`layer_completion_router.cpp:112,205,226,267,270`).
A mismatch is undefined behaviour with no check. `LayerAckService` has two `unique_ptr` producers branched on
`protocol_` (`layer_ack_service.hpp:112-113`). Fix: `std::variant` state chosen once at construction (C5).

**F14 (S2) ACK-space layout is derived twice and inline.** `compute_layer_split` at `prefill_runner.py:605` and
`:755`; inline derivation `:757-796`; no test references any of it. `compute_layer_split` also reads
`PREFILL_PP_LAYER_COUNTS` (`runner_utils.py:214`), so passing `layer_split` in is what removes the hidden env read.
Hybrid plus MTP/DFlash would TT_FATAL in the service ctor (`layer_ack_layers` widens `ack_local_count` but not
`my_ack_layer_ids`, `:773-777,795-796` versus `layer_ack_service.cpp:60-66`); unreachable today only because the
kimi_k3 adapter enables neither. Fix: one helper that owns the widening and is unit-tested (C6).

**F15 (S2, new) Hybrid models cannot complete under v2 with the tt-d-gen reader.** The host sink emits the raw
global span `[l, l+1)` (OPENS.md §2); the lcv2 reader retires a chunk when `[0, layers_per_chunk)` is covered and
fatals on `layer_end > layers_per_chunk`. A 24-layer Kimi-K3 rank with 6 acks never retires. The span policy
(OPENS.md §2 options a/b/c) must be decided before any hybrid model runs under protocol 2. None of the models in
this plan's matrix is hybrid; recorded as a blocker, not fixed here.

**F16 (S3) v1 reorder buffer stalls forever on a permanent gap** (`layer_completion_reorder_buffer.cpp:34-40`).
Out of scope beyond a rate-limited warning while `pending_` is non-empty and `next_expected` has not moved.

**F17 (S3) The nanobind caster corruption is unexplained.** No custom `std::vector` caster exists in ttnn (only
SmallVector/Span/bfloat16), so intra-module ODR is not it. Keep the workaround; the list-ctor path runs in CI via
`run_multirank_pcc.sh kimi_k3`; add a device unit test so a revert to `nb::init` is caught (test T4).

**F18 (S3) D2H record ring (my commit).** Slot arithmetic is right for any `num_workers` because the designated
worker reads the tally before its own increment. `num_workers == 1` is the only configuration; rather than test
multi-worker, assert `num_workers == 1 || ring_slots == 1` in the factory. The ack op's wait has no termination
path (a dead host stalls the forward); say so once, in the kernel header. Counter zeroing on construction is
correct (`d2h_socket_service.cpp:492,497,549`).

**F19 (S3) Small things.** `check_mapping_alignment` guards a condition mmap cannot produce
(`layer_completion_queue.cpp:157-173`); `kLayerCompletionRingMagic` (`ring_layout.hpp:58`) has no references;
`capacity` is bound through the v1 class (`layer_completion.cpp:184`); `_retry_side_queues` sorts every request id
on every step; `connect()` polls only for path existence and validates once, so a connector inside the owner's
create window fails instead of retrying to the deadline (`layer_completion_queue.cpp:221`); a producer or consumer
dying between claiming a Vyukov cell and committing it wedges the ring with no recovery (document in the ring
header; N2 covers it with one SIGKILL); `LayerCompletionRouter.stop` and the dtor join up to 30 s holding the GIL
(`layer_completion.cpp:317`); `start()` throwing after `running_ = true` leaves a stuck state
(`layer_ack_service.cpp:213-223`); the master's teardown clock starts at its own `stop_` so a subordinate exiting
later than the timeout logs warnings on a clean run (`:171,174,332`); stale comment "LayerAckService ignores the
record's contents" (`tt_dflash_drafter.py:403,569-571`).

### 2.2 Abstractions and naming

- ACK-space parameters carry layer-space names: `LayerAckService(num_layers, first_layer_idx, local_layers)` and
  the sinks' `num_layers`; `first_layer_idx` collides with `TtPrefillRuntime.config.first_layer_idx` (layer space).
  Use the runner's existing vocabulary, not a new word: `num_ack_layers`, `ack_first_idx`, `ack_local_count`.
- Sink interface `actual_start/actual_end` versus wire `pos_start/pos_end`: one quantity; the sink takes `pos_*`.
- `CountedLayerCompletionSink` / `kCountOnlyV1` / "count protocol": one spelling, `CountOnlyLayerCompletionSink`.
- `build_layer_completion_sink[_v2]` are pure kwarg forwarders: delete; the runner instantiates the class.
- `expected_layers` and `expected_total_layers` name one quantity: `expected_ack_layers`.
- Unversioned `LayerCompletionQueue` / `LayerCompletionMessage` / `LayerCompletionCell` mean v1 beside a `V2`;
  the C++ aliases become `...V1` (the Python names stay for tt-d-gen).
- "layer acks" (v1 names) versus "layer completions" (v2 names): the C1 helper settles one word per protocol.
- D2H ring geometry is spelled three times (three getters, three `D2HMetadataArgs` fields, three op params, three
  CT args): one `MetadataRingLayout{slots, slot_stride, data_offset}` struct, one getter.
- `run_master` (chooses policy) / `run_master_impl` (fan-in loop): `fan_in<MsgT>`.
- `have_prev_identity_` plus three `prev_*`: `std::optional<std::array<uint32_t, 3>> prev_identity_`.
- `seq` in v2 is computed from ACK space in two places and declared diagnostic: define it once as emission order
  per rank (`record_count_` on D2H, a per-sink counter on the host path); the v2 sink then needs neither
  `ack_idx_of_layer` nor the stride.
- The drainer's `expectation` seam (`on_first_completion`, `RequestCoverage.expectation`) is speculative and unused
  outside one test: delete.
- The D2H service has two sources of truth for a v2 message: counter-derived `(chunk, ack_idx)` and the record's
  identity. After C3 consumers key on identity; the counter supplies only the layer index.
- Python `Completion` relies on the positional order of the nanobind `try_pop` tuple: pin it in one test against
  the real binding (T6).

### 2.3 Comments and docs

Rule: a comment states the non-obvious invariant in one or two lines; rationale lives in the PR, the commit
message or OPENS.md; nothing is said twice. Categories: R restates code, O over-explains, D repeated, F defensive,
H history, S stale.

Homes: the protocol essay (v1 reorder/HoL versus v2 as-arrived) lives in `layer_completion_message.hpp` (about five
lines); it is repeated eight times today (`queue.hpp:19-25`, `router.hpp:10-21`, `sink.py:19-30`,
`drainer.py:13-19`, runner `:162-169`, `layer_completion.cpp:191-194`, `tt_prefill_runtime.py:1514-1521`): one line
plus a pointer each. The D2H ring layout lives in `persistent_d2h_writer.cpp:27-29`; the other three copies
(`outbound ... writer.cpp:46-48`, `d2h_socket_service.cpp:536-537`, `.hpp:94-95`) become one line each. The D2H
record layout lives in `layer_ack_service.cpp:97-98`; delete `prefill_runner.py:805-808` and `.hpp:45-47`.

To delete or cut (file:line → action):
- `layer_completion_router.cpp`: five downcast comments (gone with C5); `:218-225` v2 egress block → two lines;
  `:30-31` "a job is homogeneous" (F, D).
- `layer_completion_router.hpp`: `:105-106`, `:113` (R); `:108-109` duplicates the cpp; `:118-119` downcast;
  `:80-82` "ONE shm name" (S after C1).
- `layer_completion_message.hpp`: `:12` "fails at connect() rather than corrupting" (S, F1); `:77-80` seq comment
  (S after C7 → "Per-rank emission order; diagnostic"); `:91-93` → "Global layer range"; `:132` (R); `:19` second
  clause (D).
- `layer_completion_queue.hpp`: `:107` duplicates `queue.cpp:176`; `:104-105` `// v1` `// v2` (R); forward decls
  `:37,42,44,46` lose "fwd — defined in".
- `layer_completion_queue.cpp`: `:29-32` comment and the function (F19).
- `layer_completion_ring_layout.hpp`: `:18-24` false-sharing paragraph (H) → "V2 cells are one cache line; v1
  cells stay packed 32 B (frozen)"; delete `:46`, `:53`; `:38-39` → "magic + cell alignment per message type";
  `:57` alias delete; `:96-102` banner → one line; `:104-106` (R); `:71-72` → "Shared by both versions; frozen".
- `layer_ack_service.hpp`: `:29-33`, `:36` (S: v2 exists); `:45-58` two paragraphs → the six-line derivation
  table; `:65-66` (H); `:73` contradicts `:51` (fixed by the rename); `:77-84` → one line each; `:126-128` (D of
  cpp, runner, OPENS §4) keep the cpp's, trimmed to "all records of a chunk share an identity, so a change must
  land on slice_idx == 0".
- `layer_ack_service.cpp`: `:54-55` (R of the TT_FATAL text); `:102-103` (gone with C4); `:110-111` (D of ctor);
  `:140-141` → "unaligned buffer"; `:150-152` (D); `:163` parenthetical (H); `:184-190` TODO → one line pointing at
  OPENS §2; `:187` (R).
- `ttnn-nanobind/layer_ack_service.cpp:67-102` docstring: "globally-dense ordering key", "never touches the
  scheduler counter channel" (S); the Args block repeats the header a third time; cut to about twelve lines after
  the rename.
- `ttnn-nanobind/layer_completion.cpp`: `:53-55` (gone with C5); `:62` capacity (unused, delete); `:189` "Appended
  ..." (R); `:191-194` (D).
- `layer_completion_sink.py`: `:22-24` (H); `:32-46` TODO → one line + OPENS §2; `:59-63` (gone with C2);
  `:101-102` → "Fired once per completion event"; `:119-123` → two lines; `:136-138` → "seq in ACK space, layer_idx
  global; KeyError = producer bug"; `:159-160` (D); `:193-195` → "No-op sink for warm passes (see
  TtPrefillRuntime.compile)"; `:204-207` says `main()` but it is `_serve_request` (S).
- `layer_completion_drainer.py`: `:11`, `:353` "(num_layers per chunk)" (S); `:21-42` duplicates the class
  docstring (`:145-159` is home); `:33-38`, `:104-107` expectation seam (delete with the feature); `:64-66`
  "Mirrors prefill_runner's read" (S); `:121-122` (H) → "Raises on overlap or out-of-bounds"; `:208-209` (S after
  C3); `:265-268` banner (D).
- `prefill_runner.py`: `:159-160` (R); `:162-169` → two lines; `:737-754` → one line pointing at the helper;
  `:768-772`, `:778-780`, `:791-794` fold into C6; `:784-790` error text → two clauses; `:825-834` → one line +
  OPENS §3.
- `tt_prefill_runtime.py`: `:981-983` old request_id paragraph left under the new one (S, D); `:856-857` second
  sentence (R); `:1522-1527` "reads _trace_request_id" (S: four fields).
- `tt_dflash_drafter.py:403,569-571` (S).
- D2H ring kernels: `outbound ... writer.cpp:86` "tally" → `data_ready_counter`; `:90-91` → "wait while the slot
  holds an unsent record (host a full ring behind); no termination path: a dead host stalls the forward here";
  `d2h_socket_service.cpp:77` trailing comment (R).
- `LAYER_COMPLETION_OPENS.md`: §5 is stale (device runs and the ring tests exist; it is contradicted by
  `b6281b87631`); rewrite §5, add §6 (segment names, F1) and §7 (hybrid under v2, F15).

### 2.4 Test coverage

| area | today | gap |
|---|---|---|
| ring gtests | fifo, full, wrap, MPSC, v2 fields, cross-version ring connect (both directions) | counter channel attaching to a v2 segment (F1) |
| router gtests | v1 reorder, v2 as-arrived, backpressure, teardown drop, config; all `world_size = 1` | subordinate/sentinel path (needs `mpirun -np 2`); v2 drop bound from `stop()` (F10) |
| `LayerAckService` | one C++ device gtest, v1 only, one chunk, `first_layer_idx = 0`, no `ack_layer_ids` (`tests/ttnn/unit_tests/gtests/tensor/test_d2h_stream_service.cpp:948`, needs `TTNN_BUILD_TESTS`, galaxy only) | v2, multi-chunk identity, non-zero slice, `ack_layer_ids`, desync, the derivation as a unit |
| sinks | fields, span split, spin, timeout, shutdown, ACK remap | policy change (C2), seq definition (C7) |
| drainer | coverage, side queues, errors, dispatch | identity keying with eviction (C3), explicit bound (F5), real-binding tuple order |
| runner routing | none | `ack_space_layout` (C6) |
| D2H ring | queue-while-host-sleeps, wait-when-full | `num_workers` assertion (F18) |
| nanobind | none (the list-ctor path runs only in the kimi_k3 CI stage) | ctor with a non-empty list (T4) |
| build | `distributed_unit_tests` now built here; the binary hangs on `--gtest_list_tests` and on a filtered run (investigate before relying on it: probably MPI world init at static time) | — |

---

## 3. Cleanup plan (ordered; one commit each)

**C1 Protocol-specific scheduler segment names; fix the broken caller.** v1 keeps `/tt_prefill_layer_acks_<svc>`;
v2 publishes `/tt_prefill_layer_completions_<svc>`. One helper `scheduler_shm_name(service_id, protocol)` in
`layer_completion_drainer.py`, used by the runner, the drainer helpers and `migration_driver.py` (F6, which also
takes the ack-layer count instead of `NUM_LAYERS`). The runner unlinks both stale names. `_apply_manifest_env`
learns the protocol key so the producer and runner cannot disagree. Docs: `LayerCompletionRouterConfig`,
`PREFILL_MIGRATION_TESTING.md:475`, the minimax runbooks, `gemma4_d_p/tests/test_prefill_service.py:112`.
tt-d-gen follows in its own tree (`mock_prefill_runner.cpp` `prefill_ack_shm_name()`,
`test_layer_completion_channel.cpp:122-133`, the lcv2 model JSONs' `ack_shm_name`, `prefill_bringup.md`,
`docs/design/layer_completion_v2.md`, the deploy skill's preflight).
Tests: T1 (counter channel on a v2 segment) documents why; a drainer test asserts the two names differ.

**C2 One backpressure policy; no runner death; v2 always has a consumer.** `_push_with_spin` waits, warns once
then every 10 s, and on `is_shutdown()` drops the message and returns (no exception, F11); the env knob and
`LAYER_COMPLETION_PUSH_SPIN_TIMEOUT_S` go. `LayerAckService::push_blocking` gets the same rate-limited warning.
Under protocol 2 `prefill_producer` attaches and drains the completion ring regardless of `verify` (F4).
Tests: `test_sink_timeout_raises` → T8 policy test; shutdown test asserts drop-and-return.

**C3 Consumers key on identity; explicit bound; eviction.** `RequestCoverage` keyed by
`(slot_id, pos_start, pos_end)`, evicted on completion; a message for an evicted key is an error ("completion after
chunk complete"); `request_id` kept for diagnostics only. `drain_layer_completions(channel, expected_ack_layers, *,
max_layer)` where `max_layer` is the widened global layer count (NUM_LAYERS plus MTP levels plus drafter layers,
computed by the producer from its table and manifest, F5); `_drain_layer_completion_ring` no longer catches only
`TimeoutError`. `_retry_side_queues` iterates insertion order. The expectation seam is deleted.
Tests: drainer tests updated; slot-reuse test (same identity twice, sequentially, completes twice); MTP bound test;
T6 real-binding tuple order.

**C4 Desync and thread failures are errors (pending the user's decision on F7).** `LayerAckService`: on a
detected drop stop emitting but keep draining the D2H FIFO (so the device never stalls, F8), set
`std::atomic<bool> failed_` and an `exception_ptr`; the loop body is wrapped so TT_FATALs on the reader thread are
stored, not terminated (F9); `check()` rethrows and is bound to Python; `stop()` rethrows once. The runner calls
`check()` per chunk. `start()` resets `running_` on throw. The per-record derivation becomes a header-only
`AckDerivation{k, prev_identity}.step(record) -> variant<Msg, Desync>` under
`tt_metal/api/internal/disaggregation/` so `distributed_unit_tests` can test it (T3).

**C5 Router and service without downcasts; teardown fixes.** Router state is
`std::variant<V1{queue, counter}, V2{queue, ring}>`, declared before `listener_`; `run_master`, `run_subordinate`,
`stop` use `std::visit`; `CounterChannelEgress`, `RingEgress`, `SchedulerEgress` and `LayerCompletionQueueBase`
go (the nanobind base has no Python users; `shutdown`/`shm_name` are bound per class, `capacity` dropped). The
v2 drop deadline starts at `stop_` (F10). The listener thread stores exceptions (F9). `connect()` retries
validation failures until the deadline (F19). `LayerAckService` holds
`std::variant<std::monostate, unique_ptr<QueueV1>, unique_ptr<QueueV2>>`. Python `stop()` releases the GIL across
the join. Ring header documents the dead-participant limitation.
Tests: existing router gtests unchanged; T2 MPI subordinate test added.

**C6 ACK-space layout helper.** `ack_space_layout(num_layers, layer_split, kv_slot_layer_ids, rank,
extra_ack_layers)` in a ttnn-free module returning `AckSpaceLayout(num_ack_layers, ack_first_idx, ack_local_count,
ack_idx_of_layer, ack_layer_ids_of_rank)`; it owns the MTP/DFlash widening (extending `ack_layer_ids_of_rank` too,
F14) and the zero-ack-rank check; `_serve_request` takes `layer_split` from `main()`. It is the single parameter
object both transports take (the service ctor and the sinks take the layout, not five numbers). Renames per 2.2.
Tests: T7.

**C7 `seq` on v2 is emission order.** Per-sink counter on the host path; `record_count_` on D2H. The v2 sink
drops `ack_idx_of_layer` and the stride. Tests: v2 sink tests assert monotonic seq per sink.

**C8 Env at call time; comment pass; docs.** F12; the full list in 2.3; OPENS.md §5 rewritten, §6 and §7 added;
`kLayerCompletionRingMagic` and `check_mapping_alignment` deleted; `MetadataRingLayout` struct and the
`num_workers` assertion (F18).

**C9 Bounded D2H destructor barrier** (`d2h_socket_service.cpp:657`): the destructor barrier takes the service's
teardown bound and logs what was left unread instead of hanging (F8).

**Not in scope (recorded in OPENS.md):** dense acks (§1); the v2 hybrid span policy (§2 / F15); request id in the
D2H record (§4); D2H on MiniMax/GPT-OSS (§3); v1 permanent-gap recovery (F16).

---

## 4. Test plan

### 4.1 Host-only, tt-metal (gate for every commit)

```
build/test/tt_metal/distributed/distributed_unit_tests --gtest_filter='LayerCompletion*:LayerAck*'   # after the hang is understood
mpirun -np 2 --oversubscribe build/test/tt_metal/distributed/distributed_unit_tests --gtest_filter='LayerCompletionRouterMpi.*'
python -m pytest tests/ttnn/unit_tests/base_functionality/test_layer_completion_sink.py \
                 tests/ttnn/unit_tests/base_functionality/test_layer_completion_drainer.py \
                 tests/ttnn/unit_tests/base_functionality/test_layer_completion_drainer_binding.py \
                 models/demos/deepseek_v3_d_p/tests/test_kv_ack.py \
                 models/demos/common/prefill/tests/test_ack_space_layout.py
pre-commit run --files <changed>
```
Baseline on the branch: 39 Python tests pass. `distributed_unit_tests` is built (`TT_METAL_BUILD_TESTS=ON`) but
hangs even on `--gtest_list_tests`; resolve first (suspect: MPI world construction in a static initializer; try
under `mpirun -np 1`).

New tests:
- **T1** `LayerCompletionQueue.CounterChannelAttachesToV2Segment` (host gtest): a v2 ring at X;
  `InterProcessCounterChannel::connect(X)` succeeds today, which is the F1 hazard; after C1 the Python helper test
  asserts the names differ.
- **T2** `LayerCompletionRouterMpi.SubordinateSentinelEndsMaster` (host, `mpirun -np 2`, `GTEST_SKIP` at world
  size 1): rank 1 pushes v2, rank 0's scheduler ring receives in order with `source_rank = 1`; rank 1 `stop()`
  ends rank 0's `stop()` well under the teardown bound; plus the cancel path.
- **T3** `LayerAckDerivation.*` (host gtest): dense, hybrid, slice > 0 → `(chunk, ack_idx, seq, layer)`; identity
  change at `slice != 0` → `Desync`.
- **T6** `test_layer_completion_drainer_binding.py`: the real `LayerCompletionQueueV2` feeding the drainer; pins the
  tuple order; the two-transport `request_id` case.
- **T7** `test_ack_space_layout.py`: dense 61/4 → `[16,15,15,15]`; Kimi-K3 24L/2 ranks with
  `kv_slot_layer_ids [3,7,11,15,19,23]` → `acks_per_rank [3,3]`, rank 1 `ack_first_idx 3`, ids `[15,19,23]`; MTP
  last-rank widening; `PREFILL_PP_LAYER_COUNTS` override; a rank with no KV-writing layer errors.
- **T8** `test_sink_backpressure_policy`: N refusals → exactly one warning then the 10-s cadence (monkeypatched),
  success returns; shutdown drops and returns.

### 4.2 Device unit, one galaxy (a10u02)

```
pytest tests/ttnn/unit_tests/base_functionality/test_d2h_stream_service.py -k "metadata_ack"
pytest tests/ttnn/unit_tests/base_functionality/test_layer_ack_service.py
```
- **T4** `test_v2_identity_and_layer_per_record`: `D2HStreamService(global_spec=None, fifo 4 pages, one worker
  core, 12 B records)`, `LayerCompletionQueueV2.create` for the ring (created before `start()`), `LayerAckService(
  protocol=2, ack_first_idx=3, ack_local_count=3, ack_layer_ids=[15,19,23])`, four chunks with distinct
  `(slot, pos)` via `outbound_socket_service_sync` → twelve messages with the record's identity, the layer from
  `ack_layer_ids`, monotonic seq; a v1 variant through `LayerCompletionRouter` + `InterProcessCounterChannel`.
  Constructs with a non-empty list (F17).
- **T5** `test_desync_is_reported`: identity flips mid-chunk → today a logged warning (capfd), after C4 `stop()`
  raises and the destructor does not; the D2H FIFO is empty afterwards.
- After C4: assert the FIFO is empty after the warm-up drain in the traced path (through the runner, 4.4).

### 4.3 tt-d-gen, lcv2 tree (`/data/asaigal/tt-d-gen-lcv2`)

```
ctest --test-dir build-dbg --output-on-failure            # CPU, 858 tests
ctest --test-dir build-dbg-blaze --output-on-failure      # blaze-linked, 906 tests
build-dbg-blaze/engine/tests/test_layer_completion_channel
adapters/dynamo/.venv/bin/python -m pytest adapters/dynamo/tests
```
After C1 the lcv2 worker configs, `mock_prefill_runner`, the channel test and the docs take the v2 segment name;
the mismatch check runs both ways (N1).

### 4.4 Integration on hardware (bh-glx-110-a10u02, a10u14; a10u20 once allocated)

Rules: every host checked against `squeue` before any reset, kill or recovery (the allocation guard);
`HOSTS=<pair> recover-hosts --skip-cross-host-port-down` until it passes between multi-galaxy runs; `tt-smi
-glx_reset` for a single galaxy after a failed run; mock KV manager without NFS logging; no requests while a load
runs. No 3-galaxy pipeline descriptor exists, so multi-rank legs run at 2 ranks.

| leg | model / ranks | transport, mode | sides | load | pass criteria |
|---|---|---|---|---|---|
| I1 | DeepSeek-R1, 1 rank | D2H, traced and eager | v2 (lcv2 worker), v1 (main worker) | smoke + 3×50K at 2 slots | all 200; 0 `prefill: completion for slot`/`out of range` fatals; 0 `saw a new chunk`; 0 `[layer-completion] ring full` |
| I2 | MiniMax-M3, 1 rank | host callback | v2, v1 | c=20, c=40 × 120/240 | all ok; v2 within noise of v1 |
| I3 | GLM-5.3, 2 ranks (a10u02+a10u14) | D2H, traced | v2, v1 | c=20, c=40 | all ok; v2 ≥ v1 req/s at c=40, tails not worse; per-rank logs free of `teardown timed out` / `master not receiving` |
| I4 | Kimi K2.7, 2 ranks | D2H, traced | v2, v1 | c=20, c=40 | same (same code path as I3; kept because the ask is every model) |
| I5 | `test_producer_runner_e2e.py` incl. `glm53_mtp` scenarios (after C3) | D2H, eager | protocol 1 and 2 | its scenarios | passes under both protocols |
| N1 | protocol mismatch both ways (1-rank runner) | — | — | worker attach | after C1: attach fails with the connect timeout on the unpublished name within the configured budget; never attaches silently |
| N2 | worker SIGINT + restart mid-load; once with SIGKILL | D2H | v2 | I1 load | runner keeps acking; new worker attaches and serves; stragglers at attach are drained; completions for the dead worker's in-flight chunks are reported as the plan's chosen behaviour (fatal today, `prefill_reader.cpp:145`); the SIGKILL case documents the Vyukov limitation |
| N3 | worker `SIGSTOP` 25 s mid-load, `SIGCONT` | D2H | v2 | I1 load | no desync, no loss, completes; the backpressure warnings are expected here and excluded from the I-leg "no warnings" criterion |

Hybrid models (Kimi-K3) under protocol 2 are not run: blocked by F15.

Estimated hardware time: I1 30 min, I2 45 min, I3 and I4 60 min each, I5 30 min, N1–N3 40 min; about 5 h plus
recoveries.

### 4.5 Exit criteria

- All host-only suites green on tt-metal and tt-d-gen; the gtest hang resolved.
- Device unit tests green, including T4 and T5.
- Every I-leg green with no warning-level lines from the layer-completion path; N1–N3 behave as specified.

---

## 5. Order of execution

1. User decisions (section 6). Resolve the gtest hang.
2. C1, C2, C3 with host-only tests; commit each.
3. C4 and T3; commit. C5 and T2; commit. C6, C7 and T7/T8; commit. C8, C9; commit.
4. 4.2 device unit on a10u02 (T4, T5, ring tests).
5. tt-d-gen: C1 follow-through, rebuild bindings, 4.3 suites.
6. 4.4 integration legs, recovering between runs.
7. Update OPENS.md and tt-d-gen's `docs/design/layer_completion_v2.md` with results.

## 6. Decisions needed from the user

- F1: v2 segment name `/tt_prefill_layer_completions_<service_id>` (proposed); tt-d-gen configs change with it.
- F7: desync and reader-thread failures fatal via `check()`/`stop()` (proposed) versus warn-and-continue.
- F3/F11: remove the 30 s push timeout and its env knob; shutdown drops and returns (proposed).
- F4: under protocol 2 the producer always attaches and drains (proposed).
- N2: completions for a dead worker's in-flight chunks: keep the fatal (proposed) or drop them.
- 2.2: `seq` on v2 as emission order (proposed) versus reserved.

---

## 7. Results (2026-10-10)

### 7.1 Host-only and device unit (after C1–C9, tt-metal lcv2 branch at a8b99586d44)

| suite | where | result |
|---|---|---|
| `distributed_unit_tests --gtest_filter='LayerCompletion*:LayerAck*'` | a10u02 | 26/26 |
| `test_layer_ack_service.py` + `test_d2h_stream_service.py -k 'not sweep'` | a10u02 | 5/5 |
| `test_layer_completion_drainer.py`, `test_layer_completion_sink.py`, `test_ack_space_layout.py` | host | 35/35 |
| pre-commit on every touched file | host | clean |

### 7.2 tt-d-gen lcv2 tree (segment rename follow-through included)

| suite | result |
|---|---|
| `ctest --test-dir build-dbg` (CPU Debug) | 857/858; the 1 failure is the pre-existing CMake-4.4 `ENGINE_FSM_TRACE` artifact (`TraceCheckerFatal.DeathOnIllegalTransition`) |
| `ctest --test-dir build-dbg-blaze` on the launch host | 888/907; same artifact + 18 device fixtures (`PrefillPipelineMeshTest`, `SequenceParallelism/PrefillDeviceFixture`) that need a galaxy — rerun on a10u02, see 7.3 |
| `test_layer_completion_channel` (real shm ring) | 5/5 |
| adapter `pytest tests` | 315 passed; 4 `tests/frontend` SSE keep-alive failures and 2 `tests/integration/test_e2e.py` failures (`Using direct-ZMQ KV event ingress` never logged by this dynamo venv) reproduce identically on untouched main; 7 `FileNotFoundError: build-dynamo/.../mock_kvm_server` cleared by `ENGINE_BUILD_DIR=build-dbg-blaze` (35/37 with the same 2 env failures) |

### 7.3 Hardware (bh-glx-110-a10u02; a10u14 for the 2-rank legs)

Filled in per leg below as they complete.

**I1 (DeepSeek-R1, 1 rank, D2H, protocol 2, eager, lcv2 worker):** PASS. Smoke 200; three concurrent ~50K prompts at 2 slots
all 200 (19.6 s / 35.8 s / 53.8 s); 0 worker fatals, 0 runner desync/fatal, 0 layer-completion warnings (the runner's
6,780 warning lines are the ttnn `all_gather` deprecation notice).

**N2 SIGINT (worker restart mid-load):** PASS. Dynamo drains the in-flight requests before the worker exits (two of the
three returned 200 before the exit, the third 200 at 140.9 s once the new worker attached); the new worker attached with
nothing stranded; short and fresh-50K follow-ups 200.

**N2 SIGKILL, first run:** FOUND A WEDGE. The new worker dropped 425 stranded records at attach, but the runner was still
computing the killed worker's chunk and its remaining records landed after that drain; the reader hit the
`never injected` fatal (`prefill_reader.cpp:145`), the reader thread died, and the worker stayed `ready` (the stall
verdict only looks at slot progress with admitted work, and the control loop had exited so admits were never counted).
Every later request hung (short follow-up 600 s timeout). Fixed in the tt-d-gen lcv2 tree: a completion whose identity
is not an open chunk is dropped and counted (`tt_engine_stray_completions_total`, WARN on the first and every 256th),
and the adapter's stall verdict fails liveness when any engine thread's beat is older than `health_stall_ms`
(`test_prefill_completions` 19/19, adapter health-gauge tests 16/16). Re-run below.

**N2 SIGKILL, re-run with the fix:** the layer-completion side is clean (attach dropped the stranded record; no reader
fatal) but the leg is NOT survivable end to end: the killed worker's last H2D request (slot 1 [10240,15360)) was torn
in the request ring, `H2DSocket::connect` warned "prior connector process exited without running its destructor. State
has been recovered from SHM, but downstream effects (in-flight ...)" and the runner decoded the next worker's first
request as `slot=61 [7562,5224)` and asserted (`tt_prefill_runtime.py:940`). That is the H2D request path, outside
this branch: a hard-killed worker needs a runner restart, and the new liveness verdict now makes the orphaned worker
report `notready` instead of hanging. Production stops workers with SIGINT, which drains (see N2 SIGINT).

**N3 (SIGSTOP 25 s):** the Dynamo etcd lease (default ~10 s, `ETCD_LEASE_TTL`) expires during the pause and the runtime
shuts itself down at SIGCONT, so the leg runs as an 8 s pause on the stock worker and a 25 s pause with
`ETCD_LEASE_TTL=120`; results under the traced runner below.

**N1 (protocol-1 worker against the protocol-2 runner):** PASS. The main-tree worker with a 60 s attach budget logged
the attach timeout on the unpublished v1 name and never came up ready.

**I1 traced (DeepSeek-R1, 1 rank, D2H, protocol 2, `PREFILL_USE_TRACE=1`, after a `tt-smi -glx_reset`):** PASS. Smoke 200
(19.6 s, trace capture), three concurrent ~50K prompts all 200 (19.7 s / 35.9 s / 53.9 s), 0 fatals, 0 desync.

**N3 on the traced runner:** PASS both ways. 8 s pause on the stock worker: three fresh ~50K prompts all 200 (124 s), 0
desync, 0 fatals. 25 s pause with `ETCD_LEASE_TTL=120`: all 200 (135 s), worker still `live=1 ready=1`, no lease expiry,
0 desync, 0 fatals, `stray_completions_total 0`. Neither pause filled the 1024-record ring (DeepSeek at 2 slots produces
~30 records/s), so the ring-full backpressure path stays covered by the device unit test (lost record + FIFO keeps
draining) and the gtests rather than by this leg.

**N1 (protocol-1 worker vs protocol-2 runner), redone with the verdict keyed on the engine's own error:** PASS. With
`ENGINE_PIPELINE_ATTACH_WAIT_MS=60000` the main-tree worker exited after 66 s with
`attach: prefill layer-completion ring "/tt_prefill_layer_acks_deepseek_prefill" never appeared within
ENGINE_PIPELINE_ATTACH_WAIT_MS=60000ms`; with the default budget (0 = wait forever) it logs "not published yet" and never
comes up ready. It never attached to the v2 segment.

**Correction to the N1 entries above:** the DeepSeek run directory's `start_worker.sh` hard-coded the lcv2 tree, so
both "main-tree worker" N1 runs actually ran the protocol-2 worker pointed at the protocol-1 *name* (the error text
`prefill layer-completion ring ... LayerCompletionQueue::connect timed out` is the lcv2 engine's). They still show a
mismatched name never attaching, but the main-tree (protocol-1) worker had not been exercised; it is re-run below
after the launcher fix. The same slip put the protocol-2 worker on the protocol-1 runner's published counter segment
(`I1 ds_p1`), which exposed a real behaviour: `LayerCompletionQueue::connect` retried the 128-byte segment (short file,
zero magic) for the whole `connect_timeout_ms` (1 h in the deployed config) instead of refusing it. Fixed: the
short-file / zero-magic retry is bounded by a 2 s initialisation grace, after which the segment is refused as "not a
valid ring for this protocol version" (gtest `LayerCompletionConnect.RefusesAForeignSegmentInsideTheBudget`); tt-d-gen's
wrapper rewords that as "not a protocol-2 ring; launch the runner with PREFILL_LAYER_COMPLETION_PROTOCOL=2".

**Device gtests of the tt-d-gen blaze build (a10u02, after a `tt-smi -glx_reset`):** `test_prefill_pipeline_device`
22/22, `test_prefill_device` 15/15 — the 18 fixtures that could not run on the launch host.

**Allocation note (04:15 UTC):** the Slurm job holding bh-glx-110-a10u14 (129407) ended during the DeepSeek legs; only
a10u02 (job 128696) remains in my allocation and a10u20 never was. The 2-rank GLM-5.3 and Kimi K2.7 legs (I3, I4) are
gated on the allocation guard and skipped unless a10u14 is re-allocated; the scripts (`gk_run.sh glm53|kimi27 trace`,
16-user 2-rank manifests and 16-slot worker configs) are ready to run.

### 7.4 Final regression on the final code (tt-metal at db87d721778; tt-d-gen lcv2 with the stray-drop and liveness fixes)

| suite | result |
|---|---|
| tt-metal `distributed_unit_tests --gtest_filter='LayerCompletion*:LayerAck*'` (a10u02) | 27/27 (includes the new connect-grace test) |
| tt-metal `test_layer_ack_service.py` + `test_d2h_stream_service.py -k 'not sweep'` (a10u02) | 5/5 |
| tt-d-gen `ctest --test-dir build-dbg` | 857/858 (the CMake-4.4 `ENGINE_FSM_TRACE` artifact) |
| tt-d-gen `ctest --test-dir build-dbg-blaze` minus the device fixtures (launch host) | 874/875 (same artifact) |
| tt-d-gen device fixtures (`test_prefill_pipeline_device`, `test_prefill_device`, a10u02) | 22/22, 15/15 |
| tt-d-gen `test_layer_completion_channel` against the rebuilt tt-metal | 5/5 |
| tt-d-gen `test_prefill_completions` | 19/19 |

### 7.5 Decisions applied as proposed defaults (overrule and I redo the leg)

- F1/C1: per-protocol scheduler segment names (`/tt_prefill_layer_acks_<svc>` v1, `/tt_prefill_layer_completions_<svc>` v2),
  followed through in the tt-d-gen lcv2 tree (mock runner, every `ack_shm_name`, docs, deploy preflight).
- F7: desync and reader failures are fatal through `LayerAckService::check()`/`stop()`.
- F3/F11: the 30 s push timeout and its env knob are gone; a push waits with warnings and drops only on shutdown.
- F4: the producer does not attach unconditionally; a protocol-2 runner without a consumer waits with warnings.
- v2 `seq` is per-rank emission order, diagnostic only.
- Hybrid stacks stay on protocol 1 (F15).
- N2: a completion for a chunk this connector never injected is dropped and counted, not fatal (found on hardware:
  the fatal left the worker ready with a dead reader). A dead engine thread now fails the adapter's liveness verdict.
- Foreign segment at a ring name: refused after a 2 s initialisation grace instead of the connect budget.

### 7.6 Not covered

- I3/I4 (GLM-5.3 and Kimi K2.7, 2 ranks): blocked when the a10u14 allocation ended; scripts and manifests are ready.
- A hard-killed (SIGKILL) worker: the layer-completion side survives, the H2D request ring does not (tt-metal
  `H2DSocket` documents it); a runner restart is required. SIGINT is the supported stop.
- The ring-full backpressure path on hardware: a process pause long enough to fill 1024 records exceeds the Dynamo etcd
  lease on this stack; covered by the device unit test and the gtests instead.

**I2 (MiniMax-M3, 1 rank, host-callback acks, traced, 8 slots, 4,800-token prompts, 4 output tokens, mock decode):** PASS,
v2 within noise of v1. Each side on its own runner bring-up after a `tt-smi -glx_reset`; 0 runner desync/fatal, 0 worker
fatals on both.

| side | c | ok | req/s | tok/s | TTFT p50 / p90 / p99 (s) |
|---|---|---|---|---|---|
| v2 (lcv2 worker, protocol 2) | 20 | 120/120 | 1.540 | 7,190 | 6.8 / 12.0 / 13.3 |
| v2 | 40 | 240/240 | 1.610 | 7,518 | 12.9 / 22.6 / 25.0 |
| v1 (main worker, protocol 1) | 20 | 120/120 | 1.533 | 7,158 | 6.9 / 12.0 / 13.3 |
| v1 | 40 | 240/240 | 1.600 | 7,473 | 13.0 / 22.7 / 25.1 |

(The main-tree worker's bindings had been overwritten by a CPU-only `--dynamo` build on 10-09 22:06 and threw
"PrefillConfig requires a TT_DGEN_ENABLE_BLAZE=ON build" at start; rebuilt from `build-main-blaze` at 04:13 before
this leg. The DeepSeek protocol-1 leg with the main worker is re-run below.)

**I1 protocol 1 (DeepSeek-R1, 1 rank, D2H, main-tree worker with its rebuilt blaze bindings):** PASS. Smoke 200; three
concurrent ~50K prompts all 200 (19.7 s / 35.9 s / 53.9 s, the same as protocol 2); 0 worker fatals, 0 runner desync.

**N1 reverse (protocol-2 worker against the protocol-1 runner), with the real pairing:** PASS. With a 60 s attach budget
the lcv2 worker exited: `attach: prefill layer-completion ring "/tt_prefill_layer_completions_deepseek_prefill" never
appeared within ENGINE_PIPELINE_ATTACH_WAIT_MS=60000ms`. It never touched the v1 segment.

**I5 (`test_producer_runner_e2e.py`, D2H, eager, a10u02) under protocol 2:** 6/6 passed in 54.5 min — Kimi K2.7
single-user full depth (PCC 0.99963), round-robin 4 users (0.99964), random 8 users (0.99965), GLM-5.3 full depth with the
KV table (kvpe 0.8554, index 0.9830), GLM-5.3 MTP-4 and MTP-7 (mtp 0.8221, mtp_index 0.9839). Protocol 1 below.

**I5 under protocol 1:** 6/6 passed in 47.6 min, identical PCC values. Both protocol runs ended with a `tt-smi -glx_reset`.
