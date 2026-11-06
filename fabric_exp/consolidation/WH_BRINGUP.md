# Wormhole bring-up notes for the new fabric router

> Temporary working notes for the Wormhole bring-up. Drop this file before the branch is merged.

For the agent doing the Wormhole bring-up on a WH machine. Written 2026-11-03 from MAINLINE_PLAN.md (the read-only
port audit), updated 2026-11-04 for the pushed branch. The router code is the Blackhole-validated final build (study
variant `v4txa11`), ported onto origin/main.

Study documents referenced below (MAINLINE_PLAN.md, TEST_PLAN.md, V4_SIZE.md, V4_SINGLE_ERISC.md) are on branch
`origin/aagarwal/fabric-router-study` under `fabric_exp/consolidation/`; the study variant is under
`fabric_exp/variants/v4txa11/` there.

## Starting point

- **Branch:** `aagarwal/fabric-router-mainline`, on main d2bd12f. The port is six commits:
  1. NoC API: command-buffer choice for inline writes, node-addressed stateful writes (`*_to_node_with_state`), and
     `skip_cmdbuf_chk` on the set-trid and inline `*_with_state` forms. Wormhole and Blackhole get the new forms;
     Quasar v2/v3 get only the `skip_cmdbuf_chk` parameter (the `_to_node_` forms are guarded off there).
  2. Fabric library pieces (packed-header load/unpack, counter-credit accumulate/flush, the flush template argument,
     compile-time args).
  3. Host compile-time args for the new router.
  4. `TT_FABRIC_NO_BW_TELEMETRY` in test_tt_fabric (keeps router bandwidth telemetry off).
  5. The router itself: `fabric_erisc_router.cpp` plus `tt_metal/fabric/hw/inc/edm_fabric/router/*.hpp`; the old
     implementation is gone.
  6. The streaming-profiler hook (compiled out unless `PROFILE_STREAMING`).
  Cleanup commits (comments, unused headers) follow on the same branch.
- **The new router replaces main's router under main's own kernel name** (`fabric_erisc_router.cpp`).
- **Blackhole is the only arch validated so far.** P150x8 LoudBox at 400G and BH Galaxy at 200G: 1D, 2D, ring, torus,
  VC0/VC1/VC2 (VC2 is neighbour exchange, as on main), one and two ERISCs, mux, UDM, channel trimming. The pushed
  branch passed the P150 validation set (two ERISCs, one ERISC, inline write + UDM, VC1 / VC1+VC2).
- **The router's supported-configuration checks (router_common.hpp) reject non-Blackhole today.** Lifting them for
  Wormhole is the first step (the "Wormhole commit" section below).
- **Line references** like `rc:294` below are to the study variant's files:
  - rc = router/router_common.hpp, rd = router_downstream.hpp, rs = router_sender.hpp, rr = router_receiver.hpp;
  - ri = router_init.hpp, rld = router_local_delivery.hpp, rrt = router_routing.hpp, k = the kernel file.
  The port shifts lines; search by name.
- **tt-run gotcha (cost one validation run on Blackhole):** tt-run puts `<cwd>/build/lib` on `LD_LIBRARY_PATH`. If that
  directory holds an installed `libtt_metal.so` from an older build (ninja-only rebuilds don't re-install), multi-rank
  runs load the stale library and the router fails to compile on missing named args. Run `cmake --install` after
  rebuilding, or make sure `build/lib` has no stale copy.

## 6. Wormhole and other arches

### 6.1 Compile blockers on Wormhole

| Item | Where | Needed |
| --- | --- | --- |
| `static_assert(is_blackhole_arch)` | `rc:294` | lift for Wormhole bring-up (behind the router selection) |
| `sync_ready` used outside the `#if defined(ARCH_BLACKHOLE)` that declares it | `rd:237-262` | a Wormhole build fails here (a non-dependent name; GCC diagnoses it at definition; no Wormhole build was run). The §4 API form removes the `#if` |
| Host gives stream-register credits on a single TX queue | `fabric_builder_context.cpp` (main: counters only when `multi_txq_enabled`) | v4 `static_assert`s counter credits (`rs:94`, `rr:173`). The credit plan must give counters to v4 on Wormhole (§3.6) |
| Worker credit on `AT` | §3.5 | the host's role-based buffer choice must apply on Wormhole too |

### 6.2 Behavior that differs on Wormhole (check during bring-up)

- **One ERISC per Ethernet core.**
  - `NUM_ACTIVE_ERISCS == 1` gives `single::run` (`k:36-37`).
  - `CommandBuffers::presets` is false (`rc:199-204`): stateless forwards and whole-command local writes.
    `FullPresetForwards` / `PartialPresetForwards` are never instantiated, and the Wormhole arm of the full form is
    dead there.
- **One TX queue** (Wormhole TXQs 0 and 1 exist, but the host sets `receiver_txq_id = 1` only with two-ERISC Blackhole,
  merge-base 270-273).
  - Data and credit frames share the queue.
  - v4 zeroes all four counter blocks itself for one TXQ (`ri:163-170`), because the host clears them only for
    two-TXQ routers.
- **TX-queue busy test.**
  - `read_txq_status` / `txq_busy` (`rc:159-163`) read STATUS bit `ETH_TXQ_STATUS_CMD_ONGOING_BIT`. That is the
    Blackhole (BH-55) form.
  - Wormhole's `eth_txq_is_busy` tests `ETH_TXQ_CMD != 0` (`tunneling.h:73-75`).
  - Used on every send (`rs:545`, `rs:566`) and in the receiver's grouped read under first-level acks (`rr:748`). It
    needs §4's per-arch API.
- **Keepalive.** The host sets it on Blackhole only, so Wormhole gets 0 and `set_keepalive` is a no-op (`ri:114`).
  Accept-ahead is written on Wormhole as main does.
- **Heartbeat** at Wormhole's `0x1F80` (`rc:78`), as main.
- **Arch-aware already:**
  - `update_counter = is_blackhole_arch` on forwards (`rd:163`, `rd:321`);
  - posted connection opens on Wormhole (`rd:529`, `rd:587`);
  - `noc_clear_packet_tags` Blackhole only (`ri:383-385`);
  - `set_l1_data_cache` is a no-op off Blackhole (`risc_common.h:150-151`);
  - `worker_handshake_noc` is `sender_channel_ack_noc_ids[0]` on Wormhole (NoC 0), not `noc_index`.
- **`CommandBuffers` keys buffers by id, not (NoC, id).** On one Wormhole ERISC, worker credits go on NoC 0 and local
  writes on NoC 1. The table treats them as one buffer. That is conservative, and with presets off only the atomics
  rule is checked.
- **NoC equality.** `static_assert(local_chip_noc_equals_downstream_noc)` (`rr:503`). The Wormhole host defaults are
  forwarding NoC 1 and local-write NoC 1 (`erisc_datamover_builder.hpp:144-145`), so it holds.
- **Stream credit reads.** Wormhole's `get_ptr_val` returns the raw register (no mask). v4 masks to 11 bits
  (`rc:70-71`, asserted against `REMOTE_DEST_WORDS_FREE_WIDTH`, defined on both arches).
- **Code space.**
  - v4's largest one-ERISC image is 16,680 B at -O3 (V4_SIZE.md, v4txa11), against main's 12,604 B.
  - Wormhole keeps router `.text` in IRAM (V4_SINGLE_ERISC.md §1.5). Measure offline against Wormhole's limit before
    the first run (limit not checked here).
- **`wait_for_notification`** has no termination check on Wormhole (`tt_fabric_utils.h:78-84`), as main.
- **Not verified on Wormhole:** 2D on Wormhole Galaxy with v4's codec use. The codec is arch-neutral, so this is
  expected to work.

### 6.3 Host for Wormhole

On the pushed branch the host has no router-kind switch any more (one router). Still to check for Wormhole: credit
plan "counters everywhere" (on one TXQ too); worker-credit buffers by role; opt level by code size (the branch builds
-O3, or -Os only when VC1 and VC2 are both on; under review, measure first); slot policy: main's tier table until
measured.

### 6.4 Quasar (API only; the new router does not target it)

Done in commit 1 of the branch: Quasar v2/v3 `noc_fast_write_dw_inline_with_state` take the `skip_cmdbuf_chk`
parameter, so `noc_inline_dw_write_with_state` keeps compiling there; the `_to_node_` forms are not provided on
Quasar. Quasar v1 was already incompatible upstream and is untouched. Not built for Quasar.

---


## The Wormhole commit (MAINLINE_PLAN §7.2, commit 7)

Not on the branch yet; this is the first Wormhole commit to make:
- arch assert relaxed when `ARCH_WORMHOLE`;
- the `sync_ready` fix (still present on the branch: `FullPresetForwards::forward` in router_downstream.hpp declares it
  inside `#if defined(ARCH_BLACKHOLE)` and reads it after the `#endif`);
- TXQ status through a per-arch API (MAINLINE_PLAN §4's Ethernet API commit), or a local per-arch `txq_busy` as a
  stopgap;
- host counter credits on one TXQ.

Expect an IRAM size question first.

## Validation to aim for on Wormhole

The Blackhole validation list, run on WH systems (N150/N300, T3K, WH Galaxy/TG): the fabric mesh / 1D / ring /
torus sanity yamls, the fabric unit-test fixtures (1D, 2D, mux), inline-write and flushed-atomic fixtures, and UDM
where Wormhole supports it. Then the CCL suites listed for T3K / WH Galaxy in TEST_PLAN.md. TEST_PLAN.md marks which
WH tests run only in CI (the exabox Slurm cluster is Blackhole-only).
