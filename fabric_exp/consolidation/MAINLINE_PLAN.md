# Mainline plan: the v4 router (v4txa11) into tenstorrent/tt-metal

Audit for the user's review (2026-11-03). Read-only: no code changed, no builds, no device jobs, no commits.

**Bases.**
- **Final router:** `fabric_exp/variants/v4txa11/`. Line numbers marked `rc:`, `rd:`, `rr:`, `rs:`, `ri:`, `rro:`, `rld:`,
  `rrt:`, `rsi:`, `rte:` are its `router/router_{common,downstream,receiver,sender,init,round,local_delivery,routing,single,telemetry}.hpp`;
  `k:` is `fabric_erisc_router_v4.cpp`; `ct:` its `fabric_erisc_router_ct_args.hpp`.
- **Main:** merge-base `6d8a1032` (2026-10-28), unless marked origin/main (`91b4da52da0`, 320 commits later).
- **Study changes:** `git diff 6d8a1032 HEAD` plus the uncommitted host builder (`erisc_datamover_builder.cpp`, which
  already carries `host_per_vc_args.patch`).
- **Tree state.** A device job swapped a variant into the tree during this audit (`tt_metal/fabric/hw/inc/edm_fabric/router/`
  appeared at 19:31). Every comparison below uses git objects and the variant directory, never the live device files.
- **What v4 needs** comes from the kernel's dependency list (the `.dephash` of a v4txa11 device build in
  `~/.tt-metal-cache-val-v4txa11-two`, 139 tree files), then grep of every changed symbol against the router files
  and main's `fabric_erisc_router.cpp`.

---

## 0. Summary: the biggest items

1. **The port is a fresh series on origin/main, not a rebase of the study branch.**
   - v4 needs hunks from 13 of the study's 33 changed `tt_metal/` files: 4 fabric library headers, the packet header,
     3 NoC API files and 5 host files. Quasar's three NoC files need different content. On top of that:
     `dataflow_api_raw.h`, 11 new router headers and the kernel.
   - Everything else is study-only: main's router edits (+1,643), the v2 router (3,610), DRISC (1,032), the phase
     profiler and microbench headers, and most of the host diff. 7,056 test lines change too; none are needed to build
     or run v4.
2. **Two shared-header changes in the variant would break main's router as they stand.**
   - `ct:971` renames `is_receiver_channel_forwarding_disabled` to `is_vc_forwarding_disabled`. Main reads the old
     name (`fabric_erisc_router.cpp:1523` at the merge-base).
   - The variant's `dataflow_api.h` passes an 8th template argument (`skip_cmdbuf_chk`) to
     `noc_fast_write_dw_inline_with_state`. That function has 7 parameters on Quasar, which has no copy in the variant.
     Any Quasar kernel that uses `noc_inline_dw_write_with_state` would stop compiling (found by reading; no Quasar
     build was run).
3. **The host selects v4 through an env var read in four places** (`TT_FABRIC_KERNEL_VERSION=2`). It installs v4 in
   main's v2 kernel slot.
   - Mainline needs one owner, the "router kind". The kernel path, credit plan, opt level, transaction-ID count and
     slot rows all follow from it.
   - The 28 host-side `TT_FABRIC_*` study switches are not ported (table in §3.7).
4. **The two hardcoded-VC cases the user found are both removable** (§5.1).
   - Forward-command presets belong to buffer pairs, not to VC0.
   - Downstream "names" and "entries" need one mapping pair instead of a literal `0x1u` in 1D (in two places).
   - The same kind of domain mix appears twice more. `CommandBuffers::forward_data[vc]` indexes a per-receiver-channel
     host array by VC. `receiver_channel_by_vc` (ct) and `receiver_channel_of(vc)` (rr) state one fact twice.
5. **Wormhole does not compile today.**
   - Beyond the Blackhole-only `static_assert` (`rc:294`), `FullPresetForwards::forward` uses `sync_ready` outside the
     `#if defined(ARCH_BLACKHOLE)` that declares it (`rd:243`/`rd:260`).
   - The TX-queue status test (`rc:159-163`) has Blackhole semantics.
   - The host gives Wormhole stream-register credits on its single TX queue, which v4 rejects.
   - §6 lists all of it.
6. **origin/main moved under v4's routing.**
   - `Routing2DCodec::decode_action` now takes the express axis. The router-facing wrapper is
     `Fabric2DRoutingConfig::decode_action`.
   - The cmd-buf ids moved to `NocCmdBuf::`, and the config gained `multi_txq_enabled()`.
   - A new mode, `TT_METAL_FABRIC_DISABLE_INTERMESH_VC`, lands inter-mesh traffic on VC0. v4 has never run it.

---

## 1. File inventory and disposition

Line deltas are against origin/main after cleanup (estimated "~" where the cleanup is not written yet).

### 1.1 New files

| Path | Why | Disposition | Lines |
| --- | --- | --- | ---: |
| `tt_metal/fabric/hw/inc/edm_fabric/router/router_{common,utils,downstream,receiver,sender,init,round,local_delivery,routing,single,telemetry}.hpp` | The router: facts, sender and receiver programs, init/teardown, routing | new | 4,217 (before §5 fixes) |
| `tt_metal/fabric/impl/kernels/edm_fabric/<kernel>.cpp` (§1.6) | Phase skeleton, `kernel_main` | new | 103 |

### 1.2 Fabric device library (shared with main's router)

| Path | Why v4 needs it | Disposition | Variant vs main | Target |
| --- | --- | --- | ---: | ---: |
| `fabric_erisc_router_ct_args.hpp` | New arrays and args: `sender_free_slots_streams` (`ct:133`), `local_downstream_credit_streams` (`ct:147`), the per-VC tables (`ct:234-269`), `SENDER_CH_<i>_CONNECTION_SEMAPHORE_ADDR` (`ct:338-371`), `SENDER_CH_<i>_IS_WORKER_CHANNEL` (`ct:394`), `SKIP_SRC_CH_ID_UPDATE` (`ct:227`), `ENABLE_FIRST_LEVEL_ACK_VC2` | modify. Keep main's `is_receiver_channel_forwarding_disabled` name working; replace `rx_batch_completion` / `FABRIC_ERISC_ROUTER_V3/V4` (`ct:822-840`) with a host `NUM_TRANSACTION_IDS` | +136/-3 | ~+125/-5 |
| `fabric_erisc_datamover_channels.hpp` | `EdmChannelWorkerConnection<NOC>` (`:457`): one connection type for every slot count, with `close_connection(read_counter)`; `EdmChannelWorkerInterface` derives from it | modify; drop the router-v3 stride members and `// STABILITY_PROBE_STATE` (`:115-154`) | +89/-31 | ~+60/-30 |
| `fabric_router_flow_control.hpp` | Counter credit sender: `accumulate_completion_credit` / `accumulate_ack_credit` / `flush_completion_credits` (`:37-50`); `sender_channel_counter_addr` (`:152`) | modify; drop the stream-register twins (`:87-91`) and the EXP_TX_LEAN members (`:185-195`, `:222-225`) | +40/-0 | ~+25 |
| `fabric_edm_packet_transmission.hpp` | `flush_write_to_noc_pipeline`'s Blackhole one-check flush (`:105-118`), selected today by `rx_batch_completion` | modify (§2.1): the choice becomes a template parameter; drop the rest of the study's +273 | +273/-25 | ~+12/-2 |
| `fabric_stream_regs.hpp` | Home for the router's stream helpers (§4) | modify (placement); drop the study's `COMPILE_FOR_DRISC` guards | +4 (DRISC) | ~+20 |
| `fabric_txq_setup.h`, `fabric_router_eth_handshake.hpp`, `fabric_erisc_router_transaction_id_tracker.hpp`, `edm_fabric_utils.hpp`, `router_data_cache.hpp`, `telemetry/*`, `tt_fabric_utils.h`, `tt_fabric_api.h` | Used as main has them | revert study change (tracker +8, utils +11, api +2) | | 0 |

### 1.3 Hardware API files

| Path | Why | Disposition | Variant vs main | Target |
| --- | --- | --- | ---: | ---: |
| `tt_metal/hw/inc/hostdev/fabric_edm_packet_header.hpp` | `PackedPayloadAndSendType::load_word` / `unpack` / `src_ch_id` (one header word kept until handle, `rr:463`, `rr:384`) | modify; drop `NocFlushType` and the `bool` -> `uint8_t` flush fields (relaxed flush, v2/v3) | +35/-10 | ~+20/-3 |
| `tt_metal/hw/inc/api/dataflow/dataflow_api.h` | `noc_inline_dw_write(..., cmd_buf)` + `emulated_write_on_cmd_buf` (`rld:93`); `skip_cmdbuf_chk` on `noc_inline_dw_write_with_state` (`rd:263`); new `noc_inline_dw_write_to_node_with_state` (`rd:323`) | modify; drop `noc_semaphore_inc_set_state/with_state` (`:2457-2481`, v2/v3 only) and the "fabric router study" wording (`:2395`) | +70/-5 | ~+43/-5 |
| `tt_metal/hw/inc/internal/dataflow/dataflow_api_raw.h` | Carries `cmd_buf` / `emulated_write_on_cmd_buf` | modify | +15/-4 | +15/-4 |
| `.../tt-1xx/blackhole/noc_nonblocking_api.h` | `noc_write_to_node_with_state` (`:2367`), `noc_fast_write_dw_inline_to_node_with_state` (`:1957`), `skip_cmdbuf_chk` on `ncrisc_noc_set_transaction_id` and `noc_fast_write_dw_inline_with_state`, emulated write on `cmd_buf` | modify; drop `noc_fast_atomic_increment_set_state/with_state` (`:1322-1379`) | +162/-12 | ~+103/-12 |
| `.../tt-1xx/wormhole/noc_nonblocking_api.h` | Same forms (`:1558`, `:1944`; skip flags) | modify; drop the atomic set/with_state (`:1012-1064`) | +126/-3 | ~+72/-3 |
| `.../tt-2xx/quasar/noc_nonblocking_api_v{1,2,3}.h` | `dataflow_api.h` must compile for Quasar: the same forms and `skip_cmdbuf_chk` (§6.4) | modify (missing in the variant); revert the study's atomic set/with_state (+58/+44/+40 at HEAD) | 0 | ~+35 each |
| `tt_metal/hw/inc/internal/ethernet/tunneling.h`, `tt_eth_api.h`, WH `eth_l1_address_map.h` | Named TX-queue status, keepalive and accept-ahead helpers; a name for Wormhole's heartbeat word (§4) | modify (new in this plan) | 0 | ~+30 |
| `tt_metal/hw/inc/api/dataflow/buf_rw_note.h` | The variant's copy equals main's | nothing | 0 | 0 |

### 1.4 Host

| Path | Why | Disposition | Study diff | Target |
| --- | --- | --- | ---: | ---: |
| `tt_metal/fabric/erisc_datamover_builder.cpp` | New named args, worker-credit buffers by role, keepalive arg, router-kind opt level, the capture consistency `TT_FATAL` | modify; port ~50 lines, drop ~245 (§3) | +294 | ~+55 |
| `tt_metal/fabric/erisc_datamover_builder.hpp` | `set_eth_txq_seq_update_timeout` + field | modify; drop the DRISC / phase-profile declarations | +10 | +4 |
| `tt_metal/fabric/compute_mesh_router_builder.cpp` | Blackhole keepalive 1000; kernel path from the router kind | modify; drop the DRISC kernel, `TT_FABRIC_EXP_DUMP_ARGS`, the env read | +78 | ~+12 |
| `tt_metal/fabric/fabric_builder_context.{cpp,hpp}` | Counter credits for every VC when the router is v4 (v4 `static_assert`s counters, `rs:94`, `rr:173`) | modify; drop the env reads | +14 | ~+6 |
| `tt_metal/fabric/builder/fabric_static_sized_channels_allocator.cpp` | v4's slot rows (1D split; VC1/VC2 rows) | first push: tier table (0); later its own commit keyed by router kind (~+60); drop every `TT_FABRIC_EXP_*` knob | +264 | 0 / ~+60 |
| `tt_metal/llrt/rtoptions.{hpp,cpp}` | The router selection option (§1.6) | modify (new) | 0 | ~+12 |
| `tt_metal/fabric/fabric_init.cpp` | DRISC region zeroing | revert | +28 | 0 |

### 1.5 Study-only: revert to main or delete

| Path | Study lines | Disposition |
| --- | ---: | --- |
| `impl/kernels/edm_fabric/fabric_erisc_router.cpp` | +1,643 | revert to main |
| `impl/kernels/edm_fabric/fabric_erisc_router_v2.cpp` | 3,610 | delete (study v2; also v4's install slot) |
| `impl/kernels/edm_fabric/fabric_drisc_datamover.cpp` | 1,032 | delete |
| `hw/inc/edm_fabric/telemetry/fabric_{erisc_microbench,phase_profile_types,phase_profiler}.hpp` | 714 | delete |
| `fabric_router_adapter.hpp` (DRISC + `update_local_free_slots`), `edm_fabric_flow_control_helpers.hpp` (EXP_RX_GROUPED_READS), `fabric_erisc_router_speedy_path.hpp`, `api_common.h` (FLUSH_RELAXED) | +135, +12, +51, +3 | revert |
| `hostdevcommon/fabric_common.h` (`decode_action_y_first` loads both bytes) | +6/-4 | revert. LEDGER.md:121 says it was "not measured alone on the ERISC", so its effect is unattributed. If wanted, it goes in its own commit with an A/B. |
| `edm_fabric_worker_adapters.hpp` (benchmark send for VC2) | +15 | leave it to the test-streams branch |
| `tests/tt_metal/tt_fabric/**` (DRISC microbench, erisc_load_ubench, eth_txq_probe, phase/bandwidth profiler, streams) | 41 files, +7,056 | leave out of this series |
| `tt_metal/fabric/erisc_datamover_builder.cpp.orig` (untracked, git-ignored) | | delete before any commit |

### 1.6 Kernel name, location and selection

- **Location:** `tt_metal/fabric/impl/kernels/edm_fabric/`, next to `fabric_erisc_router.cpp`. The headers stay in
  `tt_metal/fabric/hw/inc/edm_fabric/router/`.
- **Name (user's call).**
  - Proposed: `fabric_erisc_role_router.cpp`. Each ERISC runs a role, and the code already says `is_sender_role` /
    `is_receiver_role`.
  - Alternative: `fabric_erisc_router_v4.cpp`, the team's word for it, but reviewers will ask where v2 and v3 are.
  - Not `fabric_erisc_router_v2.cpp`: every study tool and note means the study's v2 by that name.
- **The `FABRIC_ERISC_ROUTER_V4` define (`k:20`) goes.** It only picks 16 transaction IDs in `ct:834`. The host passes
  `NUM_TRANSACTION_IDS` instead, which also closes `ct:832`'s "TODO: Pass this value from host".
- **How the host selects it.**
  - Today `TT_FABRIC_KERNEL_VERSION=2` is read with `getenv` at `compute_mesh_router_builder.cpp:985`,
    `erisc_datamover_builder.cpp:2045`, `fabric_builder_context.cpp:58` and `fabric_static_sized_channels_allocator.cpp:802`.
  - Proposed: one rtoption, for example `TT_METAL_FABRIC_ROLE_ROUTER=1` beside `TT_METAL_FABRIC_OPT_LEVEL`.
  - Read it once into one owner, e.g. `FabricBuilderContext::router_kernel()` returning an enum
    `{EDM, ROLE_ROUTER}`. Everything that differs by router asks that owner (§3.6).
  - Default off at first. On Wormhole, fail with a clear message until §6 is done.

---

## 2. Dead and study-only code in the shared files

Evidence: for each item, a grep over the 12 v4 files and main's merge-base router. "Main" means main's
`fabric_erisc_router.cpp` plus the other kernels that include the header.

### 2.1 `fabric_edm_packet_transmission.hpp` (variant)

v4 calls only `execute_chip_unicast_to_local_chip_impl` (`rld:113`, `rr:363`), `flush_write_to_noc_pipeline`
(`rld:85`) and `shift_to_next_chunk` (`rld:141`). Includers: main's router, v2 and DRISC only.

| Code | Lines | v4 | Main | Action |
| --- | --- | :-: | :-: | --- |
| `flush_write_to_noc_pipeline`: Blackhole branch, one `ncrisc_noc_nonposted_writes_flushed` check when `rx_batch_completion && local NoC == downstream NoC == noc_index` | 105-118 | yes | no | keep the behavior; select it with a template parameter (`flush_all_local_writes`) that v4's callers pass, not with a kernel define |
| same function: `COMPILE_FOR_DRISC` arm | 108-110 | no | no | delete |
| `complete_pending_atomic_flush`, `complete_relaxed_atomic_flush`, `drain_pending_atomic_flush` | 152-179 | no | no | delete (v2/v3 deferred atomics) |
| `execute_local_semaphore_inc` (v2/v3 arm, `pending_atomic_flush`, `last_write_trid`) | 181-216 | no (its `#else` arm equals main's inline code) | no | delete; restore main's inline flush + `noc_semaphore_inc` at the three call sites |
| `drisc_local_write_init_cmd_buf`, `drisc_local_write_one_packet`, `drisc_local_semaphore_inc` (raw `NOC_CMD_BUF_WRITE_REG` sequences) | 218-287 | no | no | delete |
| `COMPILE_FOR_DRISC` arms inside `execute_chip_unicast_to_local_chip_impl` | 336, 354, 380, 390, 417-480 | no | no | delete |
| `pending_atomic_flush` parameter of `execute_chip_unicast_to_local_chip_impl` | 299 | no | no | delete |
| FLUSH_RELAXED re-encoding of the scatter atomic | 553-557 | no | no | delete |
| `update_local_free_slots` parameter of `forward_payload_to_downstream_edm` | 705, 730 | no | no (DRISC only) | delete |
| `forward_payload_to_downstream_edm`, `execute_chip_unicast_to_local_chip`, `print_pkt_*`, `update_packet_header_for_next_hop` | | no (`rrt` only names the last in a comment) | yes | keep |

**Observation (by reading, not run on a device).** Main's non-deadlock-avoidance flush (`:134-145`) loops over IDs
`0..NUM_TRANSACTION_IDS-1` whatever `rx_channel_id` is. Receiver channel r's IDs start at `r * NUM_TRANSACTION_IDS`
(`compile_time_arg_tmp.hpp:26`), so a flushed atomic on VC1's channel would not wait for VC1's own writes there. On
Blackhole v4 takes the one-check branch instead. This is worth raising upstream separately.

### 2.2 `fabric_erisc_datamover_channels.hpp`

| Code | Lines | v4 | Main | Action |
| --- | --- | :-: | :-: | --- |
| `EdmChannelWorkerConnection` (moved members, `close_connection(read_counter)`, ctor) | 457-516 | yes (`rs:320-338`, `rs:447`, `rs:1089-1091`) | through the base | keep |
| `first_slot_addr` / `last_slot_addr` / `slot_stride` and `advance_to_next_cached_buffer_slot_addr_by_stride` ("router v3") | 115-117, 138-152 | no | no | delete |
| `// STABILITY_PROBE_STATE` | 154 | | | delete |
| Blank lines left by the member move | 549, 589-591 | | | delete |
| `get_worker_semaphore_address` | in the base | no | no (no user at the merge-base either) | delete or leave (main's dead code) |

### 2.3 `fabric_router_flow_control.hpp`

| Code | Lines | v4 | Main | Action |
| --- | --- | :-: | :-: | --- |
| Counter sender `accumulate_completion_credit`, `accumulate_ack_credit`, `flush_completion_credits` | 37-50 | yes (`rr:292`, `rr:554-566`, `rr:655`) | no | keep |
| Stream-register sender `accumulate_completion_credit`, `flush_completion_credits` | 87-91 | no (v4 `static_assert`s counter credits) | no | delete |
| `sender_channel_counter_addr` | 152-154 | yes (`rs:70-73`) | no | keep |
| `note_packet_sent`, `has_completions_pending`, `has_acks_pending`, `packets_sent` (EXP_TX_LEAN_COMPLETIONS) | 185-195, 222-225 | no | no | delete |

### 2.4 `fabric_erisc_router_ct_args.hpp`

| Code | Lines | v4 | Main | Action |
| --- | --- | :-: | :-: | --- |
| `is_vc_forwarding_disabled` (renamed from `is_receiver_channel_forwarding_disabled`) | 964-972 | yes | **main uses the old name** | keep both names, or fix main's call site (see the note below) |
| `rx_batch_completion` + `FABRIC_ERISC_ROUTER_V3/V4` | 822-829 | only for the ID count and the flush | no | delete; host `NUM_TRANSACTION_IDS` |
| Conditional `NUM_TRANSACTION_IDS` | 832-840 | yes | yes | host arg |
| `static_assert(MAX_NUM_VCS == 3)` + per-VC arrays | 234-269 | yes | no | keep, reshaped (§3.2) |

**Note on the rename.** The host's `DISABLE_RX_CH<n>_FORWARDING` is per VC (`erisc_datamover_builder.cpp:1609-1611`).
Main indexes it by receiver channel (`fabric_erisc_router.cpp:1523`). On a VC0+VC2 router, VC2's channel 1 therefore
reads VC1's flag in main. By reading, the cost is a skipped optimization (VC2 still decodes routes), not wrong data.

### 2.5 Host files v4 touches

| File | Study-only code | Keep for v4 |
| --- | --- | --- |
| `erisc_datamover_builder.cpp` | `fabric_experiment_env` and `fabric_drisc_phase_profile_level` (51-60); phase-profile L1 allocation (268-279); `PHASE_PROFILE_*` / `EXP_*` / `PEER_ERISC_*` named args (925-975); `has_drisc_offload`, `get_drisc_*` (981-1049); `EXP_TX_UPSTREAM_ON_E1` lambda (1189-1196); the `TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT` override (1340-1342); `PEER_SENDER_CHANNEL_<i>_SERVICED` (1404-1409); `!has_drisc_offload()` in `compute_disable_rx_forwarding`; `TT_FABRIC_EXP_RECEIVER_AMORT` (1660-1664); the `TT_FABRIC_EXP_LOG_CONFIG` log (1693-1719); `TT_FABRIC_KERNEL_VERSION` in `get_kernel_opt_level` (2045) | `is_worker_channel` (1201) and its use in `should_force_internal_sender_skip`; `ETH_TXQ_SEQ_UPDATE_TIMEOUT` (1338); `ENABLE_FIRST_LEVEL_ACK_VC2` and `VC2_CROSSOVER_TO_VC1` (1360-1363); `SENDER_CH_<i>_CONNECTION_SEMAPHORE_ADDR` (1384-1389); `SENDER_CH_<i>_IS_WORKER_CHANNEL` (1504-1510); the worker-credit buffer choice (1521-1531); the capture consistency `TT_FATAL` (1612-1629); `set_eth_txq_seq_update_timeout` |
| `compute_mesh_router_builder.cpp` | `FABRIC_DRISC_OFFLOAD` define; `TT_FABRIC_KERNEL_VERSION`; `TT_FABRIC_EXP_DUMP_ARGS`; the DRISC kernel | Blackhole keepalive 1000 (576-580) |
| `fabric_builder_context.cpp` | `TT_FABRIC_KERNEL_VERSION`, `TT_FABRIC_EXP_STREAM_CREDITS` | counters on every VC for v4 |
| `fabric_static_sized_channels_allocator.cpp` | Every `TT_FABRIC_EXP_*` knob (`MAX_SENDER/RECEIVER_SLOTS`, `WORKER_SLOTS`, `LINEAR_SPARE_SLOTS`, `SLOTS{,_1D,_2D,_VC1,_VC2}`, `TIER_SLOTS`, `LOG_SLOTS`) | the 1D split (794-829) and the study rows (874-989) as v4's slot policy, keyed by router kind; wording cleaned (no §98, job ids, "study") |

---

## 3. Host-side changes and their shape

### 3.1 New named compile-time args (v4 reads 244; 24 are new vs main)

| Arg | Today | Clean? | Proposed shape |
| --- | --- | --- | --- |
| `SENDER_CH_<i>_IS_WORKER_CHANNEL` | from `is_worker_channel` (VC0's worker channel, VC2's first channel) | yes; comment says "Study (fabric_exp, v4)" | keep; plain comment. Long term part of a per-channel record (§3.2) |
| `SENDER_CH_<i>_CONNECTION_SEMAPHORE_ADDR` | `config.sender_channels_connection_semaphore_address[i]` | duplicate of runtime arg i (v4 `ASSERT`s they agree, `rs:293`) | keep the CT one for v4; document that main reads the RT copy |
| `ENABLE_FIRST_LEVEL_ACK_VC2`, `VC2_CROSSOVER_TO_VC1` | constant 0 (patch) | v4 `static_assert`s both off (`rc:304-306`) | fold into the per-VC loop (§3.2) |
| `ETH_TXQ_SEQ_UPDATE_TIMEOUT` | builder setter; Blackhole sets 1000; `TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT` override | clean but for the override; read in `ri:113`, outside ct args | drop the override; read it in ct args with the other TXQ args |
| `NUM_TRANSACTION_IDS` | (new; replaces the kernel define) | | host computes by router kind |
| `SKIP_SRC_CH_ID_UPDATE` | main's arg; v4 now reads it instead of deriving it from mux mode | yes | keep |

### 3.2 Per-VC and per-channel args: proposed shape

**Today.** The same per-VC fact is spread over three mechanisms:
- defines: `FABRIC_2D_VC1_ACTIVE`, `FABRIC_2D_VC1_SERVICED`, `FABRIC_2D_VC2_SERVICED`, `FABRIC_2D_VC0_CROSSOVER_TO_VC1`;
- hand-named args: `ENABLE_FIRST_LEVEL_ACK_VC0/1/2`, `VC0/1_DOWNSTREAM_EDM_SIZE`, `NUM_DOWNSTREAM_SENDERS_VC0/1`,
  `OPEN_DOWNSTREAM_VC0/1_CONNECTIONS`, `PACKED_DOWNSTREAM_VC0/1_SENDER_CHANNEL_IDS`, `DISABLE_RX_CH<n>_FORWARDING`
  (per VC despite the name);
- positional ct args: `DOWNSTREAM_SENDER_NUM_BUFFERS_ARRAY`, sized by receiver channel but read by VC (`ct:311-315`).

On the device, `ct:234-269` re-lists each as `{vc0, vc1, vc2}` behind `static_assert(MAX_NUM_VCS == 3)`. v4 adds
`DOWNSTREAM_EDM_COUNT_BY_VC` (`rd:46`) and `local_downstream_credit_streams[2][4]` (`ct:147`).

**Proposed host side.** A `RouterVcArgs` record per VC with one `emit(named_args, vc)` that writes `VC{vc}_<FACT>` in a
loop:
- `SERVICED`, `FIRST_LEVEL_ACK`, `FORWARDING_DISABLED`, `FORWARDS_ON_VC` (replaces "crossover to VC1"),
  `RECEIVER_CHANNEL`;
- `DOWNSTREAM_COUNT`, `DOWNSTREAM_NUM_BUFFERS`, `OPEN_DOWNSTREAMS`, `PACKED_DOWNSTREAM_CHANNELS`;
- `FABRIC_POSITION_START`, `HAS_WORKER_POSITION`.

The same for sender channels: a `RouterSenderChannelArgs` emitting `SENDER_CH_{i}_<FACT>`.

**Proposed device side.** `ct` builds `std::array<VcArgs, MAX_NUM_VCS>` once, with a small macro (`VC_ARG(n, FACT)` ->
`NAMED_CT_ARG("VC" #n "_" #FACT)`), so no file hand-lists VC numbers. Then:
- `vc_forwards_to_routers(vc)` (`rc:178`) becomes `vc_args[vc].downstream_count > 0`;
- `has_vc1` / `is_vc_serviced[1]` / `has_vc2` collapse to `vc_args[vc].serviced`.

**Scope.** The defines feed main's router too, so keep them while main ships. The first push can keep today's names
and add the loop later.

### 3.3 Runtime args

- v4 reads main's layout unchanged (`ri:220-302`, matching `get_runtime_args` at merge-base 1492-1540):
  - 10 connection semaphores, then 10 "buffer-index semaphores" (v4 zeroes them as producer cursors, `rs:1044-1057`);
  - per VC: `mask, first_slot[N], noc_x, noc_y, registration[N], location_info[N], buffer_index[N]`;
  - 10 teardown semaphores, then the UDM relay block.
- Study shortcuts to clean:
  - v4 skips `buffer_index[N]` by count (`ri:266`, "unused by router-to-router connections");
  - masks `mask & 0xF` (`ri:247`);
  - unpacks NoC x/y as one byte per dense index (`rd:186-191`).
- Proposed: the pack and unpack of a downstream block live together in one hostdev header
  (`DownstreamConnectionArgs` with `MAX_DOWNSTREAMS` and `NOC_COORD_BITS = 8`). The router stops repeating the format.
- No layout change is needed for the first push.

### 3.4 Slot allocation

- **v4's rows** come from the allocator under `TT_FABRIC_KERNEL_VERSION=2` (VC1 19,9,16,5,4; VC2 17,14,16 + 2,4; 1D
  split), on Blackhole at 4352 B only.
- **Proposed:** a slot policy chosen by router kind (`SlotPolicy::for_router(kind, arch, topology, vc_shape,
  packet_size)`), with the rows as data, not branches. Port the measured rows and none of the knobs.
- **Open:** SLOT_DECISIONS.md's open items (200G default 3072 B, VC1-first row) are the user's decisions and are not
  part of this port.
- **First push:** the tier table (main's default) is enough to build and run. Per memory, the older VC1 rows were
  window-bound on it, so VC1 throughput will be lower than the study's rows.

### 3.5 Command buffers, NoCs, TX queues

| Fact | Today | Shape for mainline |
| --- | --- | --- |
| Worker-channel credit buffer | Overridden inside `get_compile_time_args` (1521-1531): VC0's worker on `WR_CMD_BUF`, VC2's on `WR_REG_CMD_BUF`, others `AT`. Comment: "Study"; "main's router hands the arg to the worker interface, which does not use it" (verified: `sender_sync_noc_cmd_buf` is stored, never read) | Required: on `AT` the router's `static_assert` fails (`rc:258-260`). Set it where the config sets every other buffer id (`FabricEriscDatamoverConfig`, using origin/main's `NocCmdBuf::`), by role, with the router's rule as the comment |
| Forward data / credit buffers | Main's, the same for every receiver channel | Clean. The device indexes them by VC (§5.3) |
| TXQ ids | Main's: receiver TXQ 1 on two-ERISC Blackhole (merge-base 270-273) | Clean. Drop v4's `static_assert(receiver_txq_id == 1)` (`ri:149`); nothing depends on the number |
| Local / forward / ack NoCs | Main's | Clean |

### 3.6 Everything that differs by router: one owner

| Decision | Today | Proposed |
| --- | --- | --- |
| Kernel path | `getenv` at `compute_mesh_router_builder.cpp:985` | `router_kernel()` |
| Credit transport | `getenv` at `fabric_builder_context.cpp:58`: counters on all VCs, also one TXQ | `CreditTransportPlan` from `router_kernel()` |
| Opt level | `getenv` at `erisc_datamover_builder.cpp:2045` | v4: -O3 unless VC1 and VC2 are both active (memory: O3, not Os, to fit) |
| Transaction IDs | kernel define + `ct:822-840` | host `NUM_TRANSACTION_IDS` |
| Slot rows | `getenv` at `fabric_static_sized_channels_allocator.cpp:802` | `SlotPolicy` (§3.4) |
| Keepalive | Blackhole only, all routers (main ignores the arg) | unchanged |

### 3.7 Every `TT_FABRIC_*` study switch and how it goes

None is ported. The port builds from origin/main, so "removal" means not copying the hunk. The study branch keeps them.

| Switch | Read at (working tree) | Goes with |
| --- | --- | --- |
| `TT_FABRIC_KERNEL_VERSION` | allocator:802, mesh builder:985, builder:2045, builder ctx:58 | the `router_kernel()` owner |
| `TT_FABRIC_EXP_FLAGS` | builder:271, 934, 948, 982, 1192; fabric_init:90 | DRISC, phase profiler, EXP_* paths in main's router |
| `TT_FABRIC_EXP_TX_DELAY`, `_RX_DELAY`, `_DRISC_DELAY` | builder:935-937 | delay A/Bs |
| `TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT` | builder:938, 1340-1341 | keepalive override (the host value stays) |
| `TT_FABRIC_EXP_STACK_PAINT`, `TT_FABRIC_TXQ_COUNTERS`, `TT_FABRIC_PHASE_PROFILE`, `TT_FABRIC_PHASE_MASK`, `TT_FABRIC_PHASE_NO_ITER`, `TT_FABRIC_DRISC_PHASE_PROFILE` | builder:268-274, 925-940, 58-59 | phase profiler |
| `TT_FABRIC_DRISC_OFFLOAD` | builder:983, fabric_init:91 | DRISC |
| `TT_FABRIC_EXP_LOG_CONFIG`, `TT_FABRIC_EXP_LOG_SLOTS`, `TT_FABRIC_EXP_DUMP_ARGS` | builder:958, 1693; allocator:144; mesh builder:1007 | logging / shape inventory |
| `TT_FABRIC_EXP_RECEIVER_AMORT` | builder:1662 | main's speedy path |
| `TT_FABRIC_EXP_STREAM_CREDITS` | builder ctx:59 | credit plan |
| `TT_FABRIC_EXP_MAX_SENDER_SLOTS`, `_MAX_RECEIVER_SLOTS`, `_WORKER_SLOTS`, `_LINEAR_SPARE_SLOTS`, `_SLOTS`, `_SLOTS_1D`, `_SLOTS_2D`, `_SLOTS_VC1`, `_SLOTS_VC2`, `_TIER_SLOTS` | allocator:542-918 | slot sweeps |
| Test-side: `TT_FABRIC_NO_BW_TELEMETRY`, `TT_FABRIC_PHASE_PROFILE_CSV`, `TT_FABRIC_TEST_DIAG_LAUNCH_DELAY_MS`, the ubench CSV vars | tests/tt_fabric | stay on the study / streams branches |

---

## 4. Placement: what belongs in which file

| Item (today) | Where | Proposed home | API name (proposal) |
| --- | --- | --- | --- |
| `read_txq_status(q)` (CMD read, then STATUS: BH-55) + `txq_busy(status)` | `rc:159-163` | `internal/ethernet/tunneling.h`, beside `internal_::eth_txq_is_busy`, which becomes `status_busy(read_status(q))`; per-arch bodies (Wormhole: CMD != 0) | `internal_::eth_txq_read_status(q)`, `internal_::eth_txq_status_busy(status)` |
| Keepalive: `eth_txq_reg_read/write(q, ETH_TXQ_LOCAL_SEQ_UPDATE_TIMEOUT)` | `ri:117-135` | `tt_eth_api.h` | `eth_txq_get_seq_update_timeout(q)`, `eth_txq_set_seq_update_timeout(q, cycles)` |
| `eth_txq_reg_write(q, ETH_TXQ_DATA_PACKET_ACCEPT_AHEAD, n)` (main does the same raw write) | `ri:147`, `ri:157` | `tt_eth_api.h` | `eth_txq_set_data_packet_accept_ahead(q, n)` |
| Wormhole heartbeat word `0x1F80` (also main, `fabric_erisc_router.cpp:1904`) | `rc:75-79` | WH `eth_l1_address_map.h`, next to Blackhole's `MEM_SYSENG_ETH_HEARTBEAT` | one name for both arches, e.g. `MEM_SYSENG_ETH_HEARTBEAT` on WH too |
| UMD's heartbeat signature `0xDCBA0000` (also main :2256) | `rc:81` | a hostdev header shared with the host/UMD side (unverified where UMD defines it) | `FABRIC_HEARTBEAT_SIGNATURE` |
| ERISC phase-sync encoding `0x0A00 \| phase`, `0x0B00 \| phase`, mask `0x1FFF` (main uses the same at :2505-2516) | `ri:53-62` | `fabric_stream_regs.hpp` (or a small `erisc_sync.hpp`) shared by both routers | `erisc_sync_start(phase)`, `erisc_sync_answer(phase)`, `ERISC_SYNC_VALUE_MASK` |
| `init_stream_if_used`, `read_stream_count`, `read_stream_raw`, `stream_count_mask` | `rc:68-71`, `rc:107-126` | `fabric_stream_regs.hpp`, next to `get_ptr_val` / `init_ptr_val` | `init_ptr_val_if_used<ID>()`, `read_stream_credit_raw<ID>()`, `stream_credit_count(raw)` |
| `noc_write_to_node_with_state`, `noc_fast_write_dw_inline_to_node_with_state` (+ dataflow `noc_inline_dw_write_to_node_with_state`), `skip_cmdbuf_chk` on `ncrisc_noc_set_transaction_id` / `noc_fast_write_dw_inline_with_state` | variant API copies, Blackhole and Wormhole only | NoC API files for **every** arch: Blackhole, Wormhole, Quasar v1/v2/v3 | as named; drop the "Additive (fabric router study; for upstream review)" banners |
| Arch `#if` in `FullPresetForwards::forward` (Blackhole: raw ready reads + `ncrisc_noc_write_with_state<skip>`; else `noc_async_write_one_packet_with_trid_with_state`) | `rd:237-257` | dataflow API: give `noc_async_write_one_packet_with_trid_with_state` a `skip_cmdbuf_chk` parameter on every arch, so the router has one form | `noc_async_write_one_packet_with_trid_with_state<update_counter, posted, skip_cmdbuf_chk>` |
| Command-buffer ready test, read in pairs | `rd:242-246`, `rd:308-313` | already `noc_cmd_buf_ready` | none needed |
| Software write counters touched directly (`noc_nonposted_writes_num_issued/acked += n`) | `rld:77-80` | NoC API, beside the counters | `ncrisc_noc_count_nonposted_writes(noc, n)` |
| NoC coordinate encoding (`get_noc_addr(x, y, 0) >> NOC_ADDR_COORD_SHIFT`; `(coord << NOC_ADDR_COORD_SHIFT) \| addr`) | `rd:142-144`, `rd:515`, `rd:576` | dataflow API (the `_to_node_` forms take this coordinate) | `get_noc_node_coordinate(x, y, noc)`, `get_noc_addr_at_node(coordinate, addr)` |
| `INLINE_WRITE_ALL_BYTES = 0xF` | `rc:166` | dataflow API (the default `be` of `noc_inline_dw_write` is a bare `0xF`) | `NOC_INLINE_WRITE_ALL_BYTES` |
| `CHUNK_ENCODING_MASK = 0b11` copied from the library | `rld:131-132` | origin/main now has `tt::tt_fabric::NOC_SCATTER_CHUNK_ENCODING_MASK` (packet header) | use it |
| `CommandBuffers::kinds_on` sized `4` | `rc:219` | NoC API already has `NUM_NOC_CMD_BUFS` (BH :73, WH :57) | use it |
| Routing table, termination signal, telemetry block accessors | `rc:89-101` | fine where they are (router facts) | |
| `fabric_set_2d_intermesh_landing_route`, codec | `rrt:74-89` | fine (`tt_fabric_api.h`, `fabric_common.h`) | |

---

## 5. Straggler audit (router/*.hpp, kernel, ct args)

### 5.1 The two cases the user found

**(a) `CommandBuffers::forwards_on_vc0_bufs(vc)` (`rc:243-246`) and `PartialPresetForwards::preset`'s
`static_assert(VC == 0, ...)` (`rd:292-296`).**

Hardcoding VC0 is not needed. A partial preset writes only VC-independent fields (`rd:331-341`):
- posted, the static NoC VC (`forward_and_local_write_noc_vc`), the NoC (`edm_to_downstream_noc`);
- the credit value -1, byte enables, high address words.

What differs between VCs is only which two buffers they use. Proposed:
- In `CommandBuffers`, define the forward buffer pair of a VC, `forward_pair(vc)`, and
  `first_vc_on_forward_pair(vc)`: the lowest VC that forwards to routers on the same pair.
- `PartialPresetForwards<VC>::preset` presets its pair `if constexpr (first_vc_on_forward_pair(VC) == VC)` and is a
  no-op otherwise. Both `static_assert`s go.
- `receiver::preset_forward_commands` (`rr:828`, today `ForwardCommands<0>::preset(args.vc0())`) calls `preset` for
  every VC that forwards, by mask recursion.
- `can_share_vc0_destination<1>()` (`rd:118-125`, `rd:133`) becomes `all_forwarding_vcs_share_one_destination`:
  one pair, the same one live downstream.
- `credit_register_varies = enable_deadlock_avoidance || vc_forwards_to_routers(1)` (`rd:278`) becomes "more than one
  forwarding VC on this pair".

The rule then reads "VCs that share a buffer pair share a preset", and a VC given its own buffers needs no edit.
Byte identity on today's configs is expected but must be checked with the coverage build.

**(b) `live_downstreams<VC>()` (`rd:89-106`).**
- In 1D it returns `0x1u`; in 2D it iterates entries and sets `1u << direction` (a name).
- `Routing1D::Route::forward_to()` returns its own literal `0x1u` (`rrt:145`).
- `downstream_entry(name)` (`rc:149-151`) maps any 1D name to entry 0. That is why
  `downstream_entry(lowest_bit(live_downstreams<VC>()))` in `FullPresetForwards::preset` (`rd:216`) is right in 1D:
  by definition, not by a shared mapping.

Proposed: one pair in `router_common.hpp`.
- `downstream_name(entry)`: 2D, the entry's direction; 1D, 0.
- `downstream_entry(name)`, with `static_assert(downstream_entry(downstream_name(e)) == e)` for every entry.
- `live_downstreams` uses `1u << downstream_name(entry)` in both branches, as one loop.
- `Routing1D::Route::forward_to()` returns `forwards() ? Downstreams<VC>::live : 0`.

The same mapping is reimplemented in `record_forwarded` (`rr:337-339`, with raw `__builtin_ctz`) and should call the
pair.

### 5.2 Hardcoded VC numbers where a property is meant

| file:line | Today | Property meant | Fix |
| --- | --- | --- | --- |
| `rc:178` | `vc == 0 \|\| (vc == 1 && has_vc1)` | VC has downstream routers here | per-VC host fact (§3.2) |
| `rc:53-61`, `ct:236-249` | `has_vc1` (define `FABRIC_2D_VC1_ACTIVE`), `is_vc_serviced[]` (`*_SERVICED` defines), `has_vc2` (`ACTUAL_VC2_SENDER_CHANNELS`) | VC present / serviced, derived three ways | one per-VC record |
| `rd:46`, `ct:147` | `{VC0_DOWNSTREAM_EDM_SIZE, VC1_DOWNSTREAM_EDM_SIZE}`, `[2][4]` | per forwarding VC | per-VC args |
| `rd:69` | `first = VC == 0 ? 1 : VC1_FABRIC_POSITION_START` | the VC's first forwarding position (VC0's position 0 is its worker) | `fabric_position_start(VC) + vc_has_worker_position(VC)` |
| `rd:131-133`, `rd:278`, `rd:292-296`, `rc:243-246` | VC0 / VC1 in the forward forms | §5.1(a) | §5.1(a) |
| `rd:354-381` | Relay uses `DownstreamTraits<0>` ("UDM routers have VC0 only") | the relay's VC | `udm_vc` constant with that `static_assert` |
| `rd:478-491`, `rr:783-789`, `rr:816-821`, `ri:236-242` | `vc0` / `vc1` members, `args.vc0()` / `vc1()`, `open_downstream_vc0/vc1_connections` | forwarding VCs | `Each<forwarding_vcs, Downstreams>` (the receiver's own pattern) and per-VC `open` flags |
| `rr:131` | `(runs_vc(0) ? 1 : 0) \| (runs_vc(1) ? 2 : 0) \| (runs_vc(2) ? 4 : 0)` | mask over all VCs | `constexpr` loop to `MAX_NUM_VCS` |
| `rr:152` | `is_vc_crossover_to_vc1[VC] ? 1 : VC` | the VC a channel forwards on | host `VC<n>_FORWARDS_ON_VC` |
| `rr:333-335` | `{PACKED_DOWNSTREAM_VC0_..., PACKED_DOWNSTREAM_VC1_...}` | per forwarding VC | per-VC args |
| `rr:659-660` | `vc0_register_empty()` for the RX heartbeat | "the receiver's heartbeat VC" (VC0 can be trimmed while VC1 runs) | use `ReceiverChannelStates::loop_local_vc`, or state the choice |
| `rr:734-753`, `rs:135-147`, `rs:887-888`, `rs:910-912`, `rs:940-945` | `static_assert(MAX_NUM_VCS == 3)` + `<0>`, `<1>`, `<2>` written out | every VC | per pass: mask recursion over `receiver_vcs` / `SenderChannels::serviced_vcs` (no closure, as the files already do for channels); compile-time masks: a fold over `index_sequence<MAX_NUM_VCS>` |
| `rc:284-286`, `rc:306` | `is_vc_forwarding_disabled[2]`, `is_vc_crossover_to_vc1[2]` | VC2 as neighbour exchange | acceptable in the supported-config list (it names VC2 on purpose) |

### 5.3 Positional channel / entry numbers and domain mixes

| file:line | Today | Problem | Fix |
| --- | --- | --- | --- |
| `rc:191-194`, `rd:56-57` | `forward_data[vc]`, `forward_credits[vc]` | Indexes the host's per-**receiver-channel** arrays (`RX_CH_<r>_FWD_*`) by VC. With crossover, VC0's channel forwards with VC1's buffers. Correct only because the host sets every channel the same | index by the forwarding receiver channel (`ReceiverChannel<VC>::rx`), or have the host emit per-VC forward buffers |
| `ct:254-256` vs `rr:114-121` | `receiver_channel_by_vc[]` (sender) and `receiver_channel_of(vc)` (receiver) | one fact, two owners | keep one (ct) |
| `rr:190-196` | `source_channel()` returns `0` in mux mode | "the far router's worker channel" as a number | the fabric position of VC0's worker channel, from the host |
| `rr:344` | 1D `set_sender_channel_forwarded_to(rx, 1)` ("the downstream's forwarding channel, 1") | position literal | `DownstreamTraits<VC>::forwarding_position<0>()` |
| `rd:72` | `my_direction < facing ? my_direction : my_direction - 1` | `downstream_compact_index` written again from the downstream's side | `compact_index_on(facing, my_direction)` beside `downstream_compact_index` |
| `rd:94-104` | E/W-facing routers never turn onto N/S unless intermesh / edge | the codec's dimension order, hardcoded. origin/main's express axis (`Fabric2DRoutingConfig::express_on(axis)`, `cardinal_actions(axis)`) is the owner now | derive from the codec |
| `rd:95` vs `rd:131-132` | `turns = is_intermesh_router \|\| is_intramesh_router_on_edge` vs `!is_intermesh_router && !is_intermesh_router_on_edge && !is_intramesh_router_on_edge` | the "router picks a downstream at run time" fact written twice, with different terms | one `constexpr bool picks_downstream_at_run_time` |
| `rr:843`, `rsi:31` | `Credits credits(0)` / `credits{0}` | the ctor's `receiver_channel_index` is unused (`fabric_router_flow_control.hpp:19-29`) | a ctor without the argument |
| `rs:298` | bare `0` argument to `StaticSizedSenderChannelWorkerInterface` | unnamed | `/*name=*/0` or a named constant |
| `ri:115-132` | `firmware_keepalive[0]` / `[1]` | queues by index | two named members (`sender_queue`, `receiver_queue`) |
| `rr:808-813` | comment "channel 0 with crossover, else channel 1" | positional wording | by VC / property |
| `k:7` | "a worker on this chip (a VC's worker channel)" | fine after C4 | none |

### 5.4 Magic constants

| file:line | Constant | Fix |
| --- | --- | --- |
| `rc:78` | `0x1F80` (Wormhole heartbeat) | §4 |
| `rc:81` | `0xDCBA0000` | §4 (named, but owned by the wrong file) |
| `rc:219` | `std::array<uint32_t, 4>` | `NUM_NOC_CMD_BUFS` |
| `ri:53-62` | `0x0A00`, `0x0B00`, `0x1FFF` | §4 |
| `ri:185-187` | `fabric_config = 0`, `0xFFFF`, `0xFF` (main's, :2640-2645) | `FabricTelemetry` constants (`NO_NEIGHBOR_MESH_ID`, ...) |
| `ri:247` | `mask & 0xF` | `(1u << max_downstreams) - 1` |
| `rd:186-191` | byte packing `>> (dense * 8) & 0xFF` | §3.3 |
| `rs:197-228` | slot word: `/16`, `>> 12`, `<< 16`, `0xFFFF` | named `SLOT_ADDR_UNIT`, `FREE_SLOTS_BITS` (the `static_assert`s already state the limits) |
| `rr:560-566` | `0xFF`, `0x01010101` | `BYTE_MASK`, `ONE_PER_BYTE`; minor |
| `rr:174` | `num_slots <= 255` | ties to `SlotSources`' byte per slot: name it |
| `ri:149` | `receiver_txq_id == 1` | delete (§3.5) |
| `rd:529`, `rd:587` | `posted = !is_blackhole_arch` | fine (arch fact, commented) |

### 5.5 Study names and pointers left in shipped code

| Where | Text | Fix |
| --- | --- | --- |
| `k:95`; `ri:80-81`, `87`, `90`, `310`; `rc:78`, `288`; `rte:15`, `146`; `rr:327`, `352`, `605`, `808`; `rro:130`; `rd:276-277`, `353`, `563` | "main", "main's", "as main does" (18 places) | name the file (`fabric_erisc_router.cpp`), or state the reason without the comparison |
| `rd:563` | "main passes no word, so its open zeroes L1 address 0" | a finding about main's UDM relay open; report it upstream separately and keep only the why here |
| `ct:822` ("router v3"), `ct:825` (`FABRIC_ERISC_ROUTER_V3/V4`), channels `:138` ("router v3"), `:154` (`STABILITY_PROBE_STATE`) | study names | delete with the code (§2) |
| flow control `:185`, `:222` | "Experiment (EXP_TX_LEAN_COMPLETIONS)" | delete with the code |
| BH NoC API `:1957`, `:2367`; WH `:1558`, `:1944`; `dataflow_api.h:2395` | "Additive (fabric router study; for upstream review)" | plain API docs |
| `erisc_datamover_builder.cpp:1384`, `1504`, `1521`, `2038-2044`; allocator `:794-802`, `:874-899` | "Study (fabric_exp, v4)", "RESULTS §98", "job 367", "TT_FABRIC_KERNEL_VERSION=2" | rewrite with the code |
| `rsi:7` | "(Wormhole; Blackhole with two-ERISC mode off)" | true only after §6; keep but align with the supported list |
| `rte:16-18` | "dead inline calls alone shifted register allocation in the spill-heavy loops (VC1 senders, VC2 and speedy receivers)" | a study observation; keep the rule (`if constexpr` at call sites), drop the history |
| `rr:556-557` | "wrapped in helpers ... they change the -Os code (production VC1 builds)" | pins source shape to compiler output (house rule: no compiler-fragile assumptions). Keep the code, say why in terms of the operation, or accept helpers |
| `rs:76-77` | untrimmed array because "GCC rejects ... in a constant expression" | a documented compiler workaround; acceptable, but tell reviewers |

### 5.6 Other

- `ri:70-74`: `postcode()` drops **every** postcode on VC1 builds. Main drops only `INITIALIZATION_STARTED`
  (merge-base :2668). Neither states why. It needs a reason or removal.
- `ri:113`: reads `ETH_TXQ_SEQ_UPDATE_TIMEOUT` with `get_named_compile_time_arg_val` directly; every other arg comes
  through ct args.
- `ri:342`, `345`, `379`: C-style `(uint32_t)` casts of pointers.
- `rd:137-139`, `rd:554-560`: comment alignment off (clang-format).
- `rd:497-498`, `rrt:46-47`: double blank lines.
- `rs:503`, `rs:937`: `32 - __builtin_clz(MASK)`.
- `rr:338`: `__builtin_ctz`. Use `router_utils.hpp` helpers (add `mask_width`).
- `rte:133`, `rr:856`, `rs:978`: `round_end<false, true>` / `<true, RECEIVER>` take unnamed bools. A
  `Sides{sender, receiver}` value or named constants would read better.
- `ri:312`: `optimize("Os")` on `run_handshakes`. Allowed by the user (2026-11-03) for once-run init only; it is
  commented in place.

---

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

Router kind allowed on Wormhole; credit plan "counters everywhere" for v4; worker-credit buffers by role; opt level by
code size (measure first); slot policy: the tier table until measured.

### 6.4 Quasar (API only; v4 does not target it)

- The variant has no Quasar NoC copies.
- `dataflow_api.h`'s `noc_inline_dw_write_with_state` passes `skip_cmdbuf_chk` as an 8th template argument.
  Quasar's `noc_fast_write_dw_inline_with_state` has 7 (`noc_nonblocking_api_v3.h:840-848`). Callers exist (e.g.
  `fabric_router_adapter.hpp`, `tt_fabric_mux_v2_sender.hpp`, test kernels).
- Add the parameter and the `_to_node_` forms to v1/v2/v3, or the API PR breaks Quasar builds (by reading; not built).

---

## 7. Proposed branch shape

Built fresh on origin/main from the hunks listed above, never by merging the study branch. After each of commits 1-3,
main's router images should be byte-identical on the coverage set: the study's bytecmp tooling, a cheap and decisive
check.

### 7.1 Series (each a reviewable PR, or commits in one PR)

| # | Content | Files |
| --- | --- | --- |
| 1 | **NoC API forms** for every arch: `_to_node_with_state` writes and inline writes, `skip_cmdbuf_chk` on set-trid / inline-with-state / `noc_async_write_one_packet_with_trid_with_state`, `noc_inline_dw_write(cmd_buf, emulated_write_on_cmd_buf)`, coordinate helpers, the write-counter helper | `dataflow_api.h`, `dataflow_api_raw.h`, BH/WH `noc_nonblocking_api.h`, Quasar v1/v2/v3 |
| 2 | **Ethernet API**: TXQ status read and busy test, keepalive get/set, accept-ahead set, Wormhole heartbeat name. Main's router switched to them in the same commit | `tunneling.h`, `tt_eth_api.h`, WH `eth_l1_address_map.h`, `fabric_erisc_router.cpp` (call sites only) |
| 3 | **Fabric library**: packet-header `load_word` / `unpack` / `src_ch_id`; `EdmChannelWorkerConnection`; counter credit `accumulate_*` + `sender_channel_counter_addr`; stream helpers; flush choice as a template parameter; ct-args additions with `is_receiver_channel_forwarding_disabled` kept | `fabric_edm_packet_header.hpp`, `fabric_erisc_datamover_channels.hpp`, `fabric_router_flow_control.hpp`, `fabric_stream_regs.hpp`, `fabric_edm_packet_transmission.hpp`, `fabric_erisc_router_ct_args.hpp` |
| 4 | **Host**: router-kind option and owner; new named args; worker-credit buffers by role in the config; keepalive; credit plan, opt level and `NUM_TRANSACTION_IDS` by kind; the capture `TT_FATAL` | `rtoptions.{hpp,cpp}`, `fabric_builder_context.{hpp,cpp}`, `erisc_datamover_builder.{hpp,cpp}`, `compute_mesh_router_builder.cpp` |
| 5 | **Router**: `router/*.hpp` + kernel, with §5's fixes. §5.1 and the §5.3 domain fixes are the ones that may change code: check byte identity per change, as the study did | 12 new files |
| 6 | **Slot policy** for the router kind (§3.4) | `fabric_static_sized_channels_allocator.cpp` |
| 7 | **Wormhole bring-up**: lift the arch assert; the Wormhole host paths; IRAM size check | `router_common.hpp`, host |
| 8 | **Tests / CI**: move the validation yamls (`fabric_exp/v4/val_ring_p150.yaml`, `val_torus_y_p150.yaml`, `val_mux.yaml`, `val_nbr.yaml`) under `tests/tt_metal/tt_fabric/test_infra/test_yamls/`; a CI job running the mesh / 1D / ring / torus / mux / VC2 / NeighborExchange / trimming / inline-write / UDM suites with the router selected | tests |

Per-VC arg reshaping (§3.2) and runtime-arg pack/unpack sharing (§3.3) can follow as cleanups once main's router no
longer needs the old names.

### 7.2 Minimal first push (build and run on Blackhole, try Wormhole)

One branch on origin/main with commits 1, 3, 4 and 5 in their minimal form, plus a small commit 7.

- **Commit 1:** only the forms v4 calls, Blackhole + Wormhole + the Quasar `skip_cmdbuf_chk` parameter (so nothing
  else breaks). No banners.
- **Commit 3:** as listed. Keep main's `is_receiver_channel_forwarding_disabled` as an alias of
  `is_vc_forwarding_disabled`.
- **Commit 4:** the rtoption; named args as the study emits them (existing names); worker-credit buffers by role;
  counters for the router kind; -O3 unless VC1 + VC2; `NUM_TRANSACTION_IDS`. Tier-table slots.
- **Commit 5:** v4txa11's router files with only the edits the port forces:
  - `decode_action` via `Fabric2DRoutingConfig`;
  - `NOC_SCATTER_CHUNK_ENCODING_MASK`;
  - `NUM_TRANSACTION_IDS` from the host;
  - kernel name.

  The §5 cleanups follow as their own commits.
- **Commit 7 (Wormhole try):**
  - arch assert relaxed when `ARCH_WORMHOLE`;
  - the `sync_ready` fix;
  - TXQ status through the commit-2 API (bring commit 2 in, or a local per-arch `txq_busy` as a stopgap);
  - host counter credits on one TXQ.

  Expect an IRAM size question first.

Validation for that push: `variant_val_runs.sh`' list on Blackhole (both ERISC modes), the inline-write + flushed-atomic
fixtures, UDM unit and nightly, and main's router unchanged (bytecmp).

### 7.3 What origin/main changed under the port (adapt while porting)

| Upstream change | Affects |
| --- | --- |
| `Routing2DCodec::decode_action<MY_DIR, EXPRESS_AXIS>`; router-facing `Fabric2DRoutingConfig::decode_action<MY_DIR>`; express on X; `cardinal_actions(axis)` (`fabric_common.h`, +231 lines) | `rrt:86`; `live_downstreams`' turn rule (`rd:94-104`) must follow the codec |
| `handshake_info_t` moved to `edm_handshake_types.hpp` (same namespace) | `ri:325`, through the include |
| `connection_interface` values became `enum ConnectionValue : uint32_t` | `rs:247-248`, `rs:326-337` (implicit conversions; expected to compile) |
| Cmd-buf ids are `NocCmdBuf::*`; config `multi_txq_enabled()` | host ACK override; `fabric_builder_context.cpp` |
| `NOC_SCATTER_CHUNK_ENCODING_BITS / _MASK`; `shift_to_next_chunk` uses them | `rld:131-132` |
| `TT_METAL_FABRIC_DISABLE_INTERMESH_VC`: inter-mesh traffic on VC0 | v4 never ran it; VC0 intermesh ingress depends on `receiver_channel_is_intermesh_ingress[rx]`. Add to validation or reject in the supported-config list |
| `compute_mesh_router_builder.cpp` `create_kernel` region, `erisc_datamover_builder.cpp` (+25), main's router (+145) | port by hand; no semantic overlap found |

---

## 8. Unverified (marked in place)

- Quasar compile breakage from the 8-argument call (§6.4): by reading.
- Wormhole compile failure at `rd:260` (§6.1): by reading. No Wormhole build of v4 exists in the study.
- Main's non-deadlock flush covering channel 0's IDs only (§2.1), and main's VC2 forwarding flag (§2.4 note): by
  reading, not run.
- Wormhole IRAM limit for a 16.7 KB one-ERISC image (§6.2).
- Where UMD defines the heartbeat signature (§4).
- Byte identity of §5.1's preset-by-pair and §5.3's buffer indexing on today's configs: expected, not built.
