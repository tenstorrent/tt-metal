# Streams port into the experiment tree

- `test_infra_6dad.patch` (2026-11-05): the study's test-infra changes on the port (6dad5fec638), uncommitted, kept
  apart from the router series.
  - **Source and application.** It is the study branch's diff of tests/tt_metal/tt_fabric/test_infra since its last
    main merge (6d8a1032acc..e569c5deb20), applied three-way. Apply from the repo root with `git apply`. Checked:
    `git apply --check` is clean on 6dad5fec638, e5b75d1fbcd and c53b5a5cbbd.
  - **Worktree.** /home/aagarwal/tt-metal-port-testinfra (6dad5fec638 plus the patch, uncommitted). Its own host
    build: test_tt_fabric links. Mock vet: `vet_6dad.sh` (results below).
  - **Contents.** 19 changed files and the new `test_yamls/test_fabric_stream_sanity.yaml`:
    - **Streams.** Segments, laps, idle segments, per-segment and per-pattern `flush`. Includes the VC2 replay fix.
    - **Benchmark send loops.**
      - A single-pattern benchmark sender runs the stream loop as a one-segment stream: a local-memory copy of the
        connection, free-slot checks only when the known count runs out, header replay by the credit-only update.
      - Several patterns run the round-robin benchmark loop.
      - Both write stream counters to the result buffer (words 8-13): header copies, slot waits, busy cycles.
        The bandwidth profiler reads and logs them.
    - **Receiver start gate for sync tests (streams_rxready.patch).** Receivers count themselves ready on the sync
      core, and the host releases the sync kernel once every receiver has started.
    - **Bandwidth profiler.**
      - Per-link byte and packet loads, for links that carry flows of different sizes.
      - Node-id (inter-mesh) traffic traced through the routing tables. Main's profiler cannot measure VC1 inter-mesh
        rows in benchmark mode (bad_optional_access on hops).
      - A link may be shared by several sender cores (e.g. VC0 and VC2): it gets the longest sender's cycles, and
        each core is logged.
      - A zero-payload atomic segment's credits are limited by its packet count.
  - **Left out:**
    - **The phase profiler** (tt_fabric_test_phase_profiler.{cpp,hpp}, its CMake source line, TestContext's
      begin/end hooks and their calls in test_tt_fabric.cpp). It reads the study router's phase-profile buffer
      (tt_metal/fabric/hw/inc/edm_fabric/telemetry/fabric_phase_profile_types.hpp), which only the study router has.
    - **test_tt_fabric.cpp's diagnostic launch delay** (TT_FABRIC_TEST_DIAG_LAUNCH_DELAY_MS): diagnostic only,
      never to merge.
    - **TT_FABRIC_NO_BW_TELEMETRY:** the port carries it as its own commit (7d442203c01).
    - **The study's microbenchmarks** (drisc_microbench, erisc_load_ubench, eth_txq_probe, and their CMake targets).
      They are study tooling, not the fabric test, and drisc_microbench needs the study's DRISC offload.
  - **What depends on study-only router behavior:**
    - **`flush` overrides.** The study sends `NocFlushType::FLUSH_RELAXED`, the study router's relaxed flush mode
      (study commit 5ef745cb2a0, checkpoint 2). On the port the header's flush flag is main's bool, so `flush: true`
      gets main's full flush. Rows with a flush override do not compare across the two trees.
    - The patch differs from the study's files in two edits only: `flush_or` returns bool, and the phase-profiler
      hooks are removed from TestContext.
    - Nothing else in the patch reads router internals. Header replay (a credit-only update makes the router resend
      the header already in the slot) is what main's own `send_packets_stateful` does after its warmup lap.
  - **Effect on benchmark numbers** (a09u32 job 798, consolidation/RECONCILE_DEVICE.md §8.1).
    - The single-pattern stream loop alone, on the port, changes 2x4 mesh 1-hop rows:
      - writes at 16 B: -4.2 to -4.5%;
      - writes at 4352 B: -1.0 to -1.1%;
      - atomics: -0.2 to -0.3%.
    - With main's sender those write rows were bound by the worker. Benchmark numbers taken with this patch do not
      compare to main's test infra.
  - **Device history.** The same test files, except that `flush_or` comment, ran on lb2 in the receiver tuning sweep
    on 22cdb1e55c3 (/home/aagarwal/tt-metal-port-tune; consolidation/RECEIVER_TUNING.md).
  - **Mock vet** (`vet_6dad.sh`; CPU only, mock P150 2x4, benchmark off, one test per case).
    - **Arms.**
      - base: 6dad5fec638 as built, from port-gap's binary and the runtime root
        /home/aagarwal/tt-metal-port-gap-roots/asbuilt;
      - patched: this worktree.
    - **Results.**
      - The 6 stream-sanity tests launch on the patched arm.
      - 11 cases from main's yamls launch on both arms: sanity_common (4), at_least_2x2 (2), mesh_sanity,
        vc2_at_least_2x2 (2, VC2 workers), ubench neighbor exchange, and pvs_ns.
      - Their built tests (`--dump-built-tests`) are identical except for the random seeds.
      - Router images are identical by loadable bytes: 152 / 152 ELFs, 104 E0 and 70 E1 distinct.
      - The test kernels compile.
    - **Benchmark-mode senders**, recompiled from the logged commands with BENCHMARK_MODE = 1: the single-pattern
      stream loop, the stream benchmark, and the round-robin loop (2 connections, 7 configs) all compile and link.
    - **A VC2 worker in benchmark mode does not compile**, on main as on this patch:
      - main's `fabric_detail::update_credits_and_slots` takes only VC0 / VC1's adapter;
      - the study templated it on the adapter type in the worker adapter header (tt_metal/, not tests/).
  - **`adapter_vc2_credits_6dad.patch`** (optional, separate; tt_metal/fabric/hw/inc/edm_fabric/
    edm_fabric_worker_adapters.hpp, 3 hunks): the study's change.
    - Applies cleanly to 6dad5fec638.
    - With it, the VC2 benchmark sender compiles.
    - It does not change the router, receiver or non-benchmark sender images (mock caches, loadable bytes), nor the
      VC0 benchmark sender's. Only test kernels call the function.

- `streams_rxready.patch` (2026-10-26, v2; RESULTS §98 end): receiver start gate for sync tests. A receiver that
  starts after its sender has sent more packets than its buffer holds waits forever (the VC1 4 KiB all-to-all hangs);
  the global sync spans one mesh, and nothing ordered receivers before senders across hosts.
  - Each receiver kernel, on entry (after it marks itself started, before it reads any packet), increments a
    receivers-ready count on its chip's sync core (local NoC atomic; compile-time arg SIGNAL_READY, set in sync tests).
  - `TestContext::launch_programs` launches, then reads that one word per chip until it reaches the chip's receiver
    count (30 s timeout, then a warning and release anyway), joins the host barrier, and writes the release word.
  - The sync kernel waits for the release before its first global sync. Both words are in the sync core's global sync
    region (words 1 and 2), cleared before launch; the memory map layout is unchanged.
  - Unchanged: sender kernels (byte-identical); receiver kernels outside sync tests (byte-identical, checked .text
    and .data); tests without `sync`, latency tests, mock clusters (no gate).
  - Files: tests/tt_metal/tt_fabric/test_infra/{kernels/tt_fabric_test_sync.cpp, kernels/tt_fabric_test_receiver.cpp,
    tt_fabric_test_memory_map.hpp, tt_fabric_test_device_setup.cpp, tt_fabric_test_device_setup.hpp,
    tt_fabric_test_context.cpp, tt_fabric_test_context.hpp}. Made in the worktree first. Rebuild test_tt_fabric.
  - On a tree that carries v1 (as the experiment tree does since jobs 334-336): `git apply
    fabric_exp/streams/port/streams_rxready_v2_delta.patch`. Equivalent: `git apply -R
    fabric_exp/streams/port/streams_rxready_v1.patch && git apply fabric_exp/streams/port/streams_rxready.patch`.
  - `streams_rxready_v1.patch`: the first form (the host polled every receiver core's status word), kept for the
    revert.

- `diag_launch_delay.patch` (2026-10-26, v2): DIAGNOSTIC ONLY, experiment tree only (not in the worktree, never to
  merge). `TT_FABRIC_TEST_DIAG_LAUNCH_DELAY_MS=<ms>` delays the launch of every process whose global launch rank is not
  0, so a multi-rank run reproduces the receiver start race on demand. The rank comes from tt-run's `TT_RUN_RANK`,
  else `OMPI_COMM_WORLD_RANK`, else the unsplit MPI world. Every process logs one line per test ("diag: rank R ...
  delaying launch by N ms" or "diag: rank 0 ... launches without delay", with world and current-world rank/size), so
  the log shows whether it fired. v1 (`diag_launch_delay_v1.patch`) read the current world's rank and never fired in
  check A. On a tree with v1: `git apply fabric_exp/streams/port/diag_launch_delay_v2_delta.patch`. File:
  tests/tt_metal/tt_fabric/test_infra/test_tt_fabric.cpp. Independent of the other patches.

- `streams_idle.patch` (2026-10-26): stream idle segments `{idle: <cycles>}` (bursty senders). Host: parse, expand into
  `StreamProgram::idle_after`, validate, dump; device setup passes STREAM_MODE = 2 and the idle cycles; the sender
  kernel waits after a segment's packets. Made in the worktree first (uncommitted there, on top of the flush work).
  Apply from the repo root with `git apply fabric_exp/streams/port/streams_idle.patch`; undo with `git apply -R`.
  Files: tests/tt_metal/tt_fabric/test_infra/{tt_fabric_test_common_types.hpp, tt_fabric_test_config.cpp,
  tt_fabric_test_device_setup.cpp, kernels/tt_fabric_test_sender.cpp}. Rebuild test_tt_fabric after applying.

- `streams_flush.patch` (2026-10-22, §64): per-segment / per-pattern `flush` for atomic_inc and fused_atomic_inc
  (host parse + merge + stream expansion + traffic args; the kernel decodes it from the top two bits of the
  increment word). Made in the worktree first (uncommitted there), applied here with `git apply`. Undo:
  `git apply -R streams_flush.patch`.
- `streams_vc2_replay.patch` (2026-10-29, RESULTS §98 VC2): the benchmark loop's header replay
  (`update_credits_and_slots`, VC0's adapter only) called for every adapter, so any VC2 test's sender kernel failed to
  compile (`test_fabric_vc2_at_least_2x2_mesh.yaml`: "cannot convert WorkerToFabricEdmSenderBase<false, 0, 2>*").
  The loop now replays through `replay_current_slot`, which copies the header for other adapters. VC0 benchmark
  sender binaries byte-identical (3 replayed from job 138's logged commands). Made in the worktree first (uncommitted
  there), the same edit here. Undo: `git apply -R streams_vc2_replay.patch`.

The experiment tree currently carries `streams_p3f.patch` (applied 2026-10-16 after reversing `streams_p3d.patch`; the faster benchmark loops also run the existing single-pattern and round-robin benchmark sends).

- `streams_p2.patch`: test_infra changes from branch `aagarwal/fabric-test-streams` (worktree
  /home/aagarwal/tt-metal-streams) through P2, including the new `test_yamls/test_fabric_stream_sanity.yaml`.
  It leaves out the branch's measurement-only telemetry switch in test_tt_fabric.cpp (this tree already has it).
- `streams_p1.patch`: the earlier P1 port, kept for reference.
- Undo: `git apply -R streams_p3f.patch` from the repo root, if nothing else changed those files since.
  Otherwise restore from the backups: `exp_tree_test_infra_backup.tgz` (files touched by P1) and
  `exp_tree_p2_extra_backup.tgz` (the two more files P2 touches), both taken before any patch, then delete the
  new `test_fabric_stream_sanity.yaml`.
