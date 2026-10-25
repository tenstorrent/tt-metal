# Router v3a: faithful consolidation of s79mc

v3a is router v2 as built at `TT_FABRIC_EXP_FLAGS=13129088` (= `variants/s79mc`), rewritten as mainline code: no
experiment flags, no DRISC offload, no phase profiler, no layout hints. It keeps v2's loop shape, channel types, fenced
load groups and templating, and takes the same compile-time and run-time args (host builder unchanged).

**Result.** Offline, over all 440 logged configs, v3a compiles to exactly the same instruction stream as s79mc with
its five study-added branch hints deleted, at -O3, -O2 and -Os. Everything else in this rewrite is codegen-neutral.
The only difference from s79mc is therefore the hints (section 6).

## 1. Files (`fabric_exp/variants/v3a/`)

| File | Status | What |
| --- | --- | --- |
| `fabric_erisc_router_v3.cpp` | new | The router. 3573 lines (v2: 3610). Code lines 3057 (v2: 3347); comment lines 280 (v2: 16, see section 5) |
| `tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_v3_speedy_path.hpp` | new | v3's speedy (neighbor-exchange) path: main's `fabric_erisc_router_speedy_path.hpp`, verbatim plus a 2-line header note |
| `fabric_erisc_router_ct_args.hpp` | shared, modified | `rx_batch_completion` policy; `NUM_TRANSACTION_IDS` uses it |
| `fabric_edm_packet_transmission.hpp` | shared, modified | Strict-flush check uses `rx_batch_completion`; v2's deferred-atomic semaphore path also for v3 |
| `fabric_erisc_datamover_channels.hpp` | shared, modified | `// STABILITY_PROBE_STATE` marker; stride-advance comment no longer says "Experiment" |

v1 (`fabric_erisc_router.cpp`) and v2 (`fabric_erisc_router_v2.cpp`) are untouched. With the three modified shared
headers, both compile to the same `.text` as before in all 440 configs (section 8).

## 2. Removed

| What | Where in v2 |
| --- | --- |
| DRISC offload | every `FABRIC_DRISC_OFFLOAD` block: `drisc_offload_*` helpers and static_asserts, the push-to-DRISC receive path, the speedy-mode DRISC receive branch, the DRISC term in the completion-batch decision, the init handshake (routing-table copy, E1-ready, waiting for DRISC credits, NoC state), the DRISC credit stream init, and the teardown DRISC signal and credit drain |
| Phase profiler | the `phase_profiler` global and every `open/close/mark_*/count_*/end_iteration/flush/init` call (all compiled to nothing at mode 0) |
| Microbenchmark | `run_fabric_erisc_microbench` call and the `fabric_erisc_microbench.hpp` include |
| Experiment-only paths | TX/RX delays (`fabric_experiment_spin`); the speedy header's bits 0, 5, 6 and its profiler hooks (v3 includes its own speedy header instead) |
| Dead helpers | `count_serviced` (profiler init only), `exp_local_delivery_is_data_only` (speedy bit 0 only), `exp_stream_unused_in` (DRISC static_asserts only) |
| Branch hints | `[[unlikely]]` on the two E0 TX-queue status checks in `sender_channel_try_send` (pre-send status, post-send status) and on the three E1 pending-atomic branches (in-step `if (pending_atomic_flush)`, end-of-step drain, end-of-pass drain) |

Stack paint was not in v2. Nothing reads `fabric_exp_*`, `EXP_FLAGS` or the delay args any more.

## 3. Policy constants

| Name | Where | Replaces | Value |
| --- | --- | --- | --- |
| `FABRIC_ERISC_ROUTER_V3` (macro) | first line of v3, before any include | v2's `FABRIC_ERISC_ROUTER_V2` define | Selects v3's policies in the shared headers. Must precede the first fabric include (checked by preprocessing: ct_args is first included from inside the router file) |
| `rx_batch_completion` | ct_args | `fabric_exp_rx_batch_completion` (bit 22) | v3: `true`. v1/v2: `fabric_exp_rx_batch_completion`, so they are unchanged. Each use keeps its own structural conditions: 16 trids needs no deadlock avoidance, one receiver channel, no VC1/VC2 (ct_args `NUM_TRANSACTION_IDS`); the batched completion step needs no first-level ack (`rx_batch_completion && !enable_first_level_ack`); the single global strict-flush check needs BH, no deadlock avoidance, and local NoC = downstream NoC = `noc_index` (transmission header) |
| `can_defer_local_atomic_flush()` | v3 | dropped its `fabric_exp_rx_batch_completion` term | Already requires `NUM_TRANSACTION_IDS == 16`, which only batching gives |
| `#if defined(FABRIC_ERISC_ROUTER_V2) \|\| defined(FABRIC_ERISC_ROUTER_V3)` | transmission `execute_local_semaphore_inc` | `#if defined(FABRIC_ERISC_ROUTER_V2)` | v3 stages flushed atomics and can defer them, as v2 does |
| `eth_txq_seq_update_timeout` | v3 | `fabric_exp_seq_update_timeout` | Keepalive: read from the existing named CT arg `EXP_SEQ_UPDATE_TIMEOUT` (host: `TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT`, 0 = keep firmware's 8000). Set at init by the teardown master, restored at teardown, exactly as v2. Helpers renamed `set_/restore_eth_txq_seq_update_timeouts` |

Why a macro rather than a template parameter: the bit-22 choice reaches namespace-scope constants (`NUM_TRANSACTION_IDS`,
`RX_CH_TRID_STARTS`) that several shared headers use, and `flush_write_to_noc_pipeline` is reached through the
pending-atomic and semaphore helpers from inside the shared local-delivery dispatch. A template parameter would have to
be threaded through all of them and still could not change ct_args' constants. The define-before-include pattern already exists (`FABRIC_ERISC_ROUTER_V2`).
Once v1/v2 are retired, the `#else` branches go and `rx_batch_completion` can go too.

Why a v3-owned speedy header: the shared one has 29 lines that call `phase_profiler` (a global the router defines),
read `fabric_exp_*` switches or spin the experiment delays. Making those conditional would scatter `#if`s through a file
v1 and v2 depend on. Main's
version is exactly the shared header without the experiment hooks, so v3 includes a copy of it. Cost: ~320 duplicated
lines until v1/v2 are retired.

## 4. Kept

Every production path v2 has: Wormhole `#if`s, 1D/ring/speedy, VC1/VC2 and intermesh, deadlock avoidance and bubble flow
control, first-level ack, stream-register credits, single-ERISC mode, UDM relay, tensix mux (`skip_src_ch_id_update`),
context switch and retrain, channel-trimming capture. Production telemetry as on main: BW telemetry (`update_bw_counters`,
`update_bw_cycles`, `update_telemetry`), perf telemetry (`PerfTelemetryRecorder`, recording windows), code profiling
(`NamedProfiler<RECEIVER_CHANNEL_FORWARD>`, `clear_code_profiling_buffer`), the heartbeat. All study changes that v2
hard-wires (bits 7, 8, 10, 12, 14, 19, 23, the §53–§74 E1 changes) are unchanged, including the fenced load groups.

## 5. Readability changes (all verified codegen-neutral)

- About 200 lines of main's comments that the v2 rewrite had dropped are back, wherever the code they describe is
  unchanged: stream-id and direction maps, the context-switch handshake states, 2D admit/forward and intermesh notes,
  connection-liveness and retrain notes, init/teardown steps. They were aligned automatically by the code line that follows
  them and then checked one by one. Three of main's comments that were stale where they sat were moved or dropped: the DPRINT
  note now sits on `ENABLE_STATEFUL_NOC_APIS`, the "value of 32" note on the inner loop, and "finer grain control over
  switching" was dropped (it described a removed switch).
- New comments on the study mechanisms (grouped reads, idle skip, overlapped send completion, `sender_channel_try_send`,
  batched completions, deferral encoding, keepalive, main loop). No RESULTS references in kernel code.
- Banner and naming cleanups ("Pure C++, Zero asm", `exp_set_*` → `set_eth_txq_*`). The receive step's generic path is
  no longer wrapped in a stray `{}` left by the DRISC `#else`.
- The batched-completion comment states what is checked (last slot; first slot too on an idle pass) and does not claim
  it is safe. Whether checking only the last slot while making progress is safe is still open (§26).

## 6. Semantic difference vs s79mc: the five hints

v3a ≡ s79mc minus the five `[[unlikely]]` (instruction-identical, 440/440 configs, -O3/-O2/-Os). What the hints did:

| | -O3 effect of dropping them |
| --- | --- |
| Hot E0 (E/W, EAST-facing) | Contiguous loop identical (226 insns, 33 stack loads). 12 out-of-line instructions reordered; codegen_diff's "35 differ" is those plus shifted branch targets |
| WEST/N/S-facing 2D E0 | +52 to +61 instructions overall, mostly out-of-line send tails (the TX-queue busy test with register moves duplicated). Contiguous loop +4 to +6 (e.g. 226 → 230: four `mv` from register shuffling of forwarding-channel counts), same stack-load count |
| Middle E1 (2D) | +17 instructions (1909 → 1926). The deferred-atomic bypass code (two-slot exact check, ~50 instructions) moved into the contiguous loop: 162 → 215 instructions, 18 → 20 stack loads. The common no-pending case now takes a forward jump over it. The completion path rematerialises the TXQ1 base and the trid-status base instead of holding them in registers. codegen_diff's "1375 differ" is mostly shifted addresses |
| 1D E0, ring E0, single-ERISC | -4 to +3 (the TX-queue status hint) |

This matches §77, where removing the same hints (s77sn vs s77s) gave E0 +42 to +52 instructions and measured E/W fused
−1.4, fused flushed −6.6, N/S small writes +4.2, N/S atomic +2.5. Expect v3a to differ from s79mc by a few cycles on E1-bound
and N/S rows on hardware. EAST-facing E0-bound rows should be unchanged.

Behaviour outside 13129088: v3 ignores `TT_FABRIC_EXP_FLAGS` and the delays, so with any flag value it behaves like
s79mc at 13129088. Host-side experiment settings that need device support v3 no longer has must not be used with it:
DRISC offload (bit 21, `TT_FABRIC_DRISC_OFFLOAD`), `TT_FABRIC_PHASE_PROFILE`, `TT_FABRIC_TXQ_COUNTERS`. The host would
allocate or pair resources that v3 never writes, and with DRISC offload the paired DRISC would never get its handshake.
Bit 9 (`EXP_TX_UPSTREAM_ON_E1`) changes the host's per-ERISC channel assignment, and neither v2 nor v3 implements it.

## 7. Kept from main that looks like a hint or oddity

- `[[unlikely]]` on `NOC_SPARSE_MCAST_WRITE` in `receiver_forward_packet` (1D write-and-forward). In shared headers v3
  uses: `[[unlikely]]` on the sparse branch and `[[likely]]` on `chunk_count == 4` in local delivery, `__builtin_unreachable()`
  above the dense send-type range, `[[unlikely]]` on the 1D route refill.
- `__attribute__((optimize("no-jump-tables")))` on `admit_2d_dispatch`/`forward_2d_dispatch`;
  `optimize("jump-tables")` on local delivery (shared).
- `optimize("Os")` on init/teardown helpers; `noinline` on `execute_pause_command`,
  `wait_for_static_connection_to_ready` and `init_local_sender_channel_worker_interfaces` (2D); `always_inline` on the
  `execute_main_loop` lambda; steps `FORCE_INLINE` except under `FABRIC_2D_VC1_ACTIVE`; `#pragma GCC unroll 1` in
  sender-slot header zeroing (shared).
- Main's comment "Keep hot routing coordinates in address-untaken scalars so they remain in registers" (it describes
  allocation behaviour; kept as main's).
- Oddities, all as on main: `port_direction_table` is never initialized or read but is passed through every step;
  `establish_edm_connection` ignores its stream id; `populate_local_sender_channel_free_slots_stream_id_ordered_map`
  ignores its first argument; `run_drain_step` is `ASSERT(false)` (not implemented); the speedy receiver sends its
  completion with `CHECK_BUSY=false`.
- Differences from main that v2 already had and v3a keeps: main's three `asm volatile("nop")` in `kernel_main` init are
  gone (v2 has no inline asm).

## 8. How it was checked (offline only)

Broker jobs swap variants into the live tree during the day (the tree's router files changed at 19:34 while this was
built), so every build here overlays onto a frozen snapshot: `git archive HEAD tt_metal/fabric` plus s79mc's v2. The
wrapper `ob.py` swaps the overlay source in `offline_build.py` and is otherwise the same tool. Scratch:
`/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/v3a/`.

| Build (all 440 each) | Result |
| --- | --- |
| v3a at -O3, -O2, -Os | all build (-Werror) |
| v3a vs s79mc minus the 5 hints, at -O3/-O2/-Os | 0/160 E0, 0/280 E1 changed at every level |
| s79mc (v2) with v3a's three shared headers, -O3 | 0/440 changed vs s79mc |
| v1 with v3a's three shared headers vs v1, -O3 (v1 built in v2's place) | 0/440 changed |
| probes `hook`, `statefield` (perturb.py on v3a), -O3 | both build; markers found (1 DECL, 1 PRELOOP/LOOP pair, 1 STATE) |

codegen_diff vs s79mc (changed configs, median/max instruction delta, max frame delta):

| Level | E0 | E1 | Hot E0 E/W | Hot E1 mid-W |
| --- | --- | --- | --- | --- |
| -O3 | 120/160, +0/+61, frame −16 | 82/280, +1/−36 | 2523 (+0), 35 differ | 1926 (+17), 1375 differ |
| -O2 | 120/160, +7/+57 | 120/280, +1/−26, frame −32 | 2210 (+12), 428 differ | 1839 (+16), 1425 differ |
| -Os | 40/160, −2/−2 | 40/280, +6/+6, frame −16 | 2130 (+0), 0 differ | 1677 (+6), 718 differ |

-O3 by router type (changed/total, instruction delta range): E0 1D 40/40 (−1/+3), E0 ring 40/40 (−4/−2), E0 speedy 0/40;
E0 2D E/W 24/24 (+0/+54: EAST-facing +0, WEST-facing +53/+54), E0 2D N/S 16/16 (+52/+61); E1 2D E/W 24/24 (+17/+25), E1 2D N/S
16/16 (−36/−2); E1 1D, ring, speedy 0/120; single-ERISC ring 36/40 (+0/+1), single-ERISC 2D N/S 6/16 (+0), other
single-ERISC 0/64.

Probes against v3a (-O3): `hook` changes every config (E0 +43 to +73 instructions, hot E0 +57 and frame −16; E1 +16 to
+72). `statefield` changes every E0 (+16 B frame, no instruction-count change; hot E0 66 instructions differ, stack
offsets) and the 120 single-ERISC E1 configs, but not the 160 RX-only E1s.

## 9. Not done, for later designs

- `ReceiverChannelPointers::exp_prefetched_pkts_received` (shared header, v1's bit 15) is still in E1's state and zeroed
  at reset. Removing it changes E1 layout, so it waits until v1 is retired.
- ct_args still declares v1/v2's experiment constants and static_asserts (v3 reads none). It can be cleaned once v1/v2 go.
- Unused parameters threaded through the steps (`port_direction_table`, `remote_receiver_channels`, the speedy step's
  relay/routing-table arguments): removing them changes signatures of non-inlined steps in VC1 builds, so they were left.
- Whether the E1 hint is wanted back in some other form (it keeps the deferral bypass out of the per-packet path) is a
  phase-2 question for hardware.
