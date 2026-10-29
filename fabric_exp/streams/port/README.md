# Streams port into the experiment tree

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
