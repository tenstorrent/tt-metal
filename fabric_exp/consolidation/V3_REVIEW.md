# v3 router review: v3l2be1b, and the cleaned v3r

Reviewed: `fabric_exp/variants/v3l2be1b/` (the lead candidate, RESULTS §93), against v3a, main's router
(`main:tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp`, 453ee7bdfa7), main's headers, and the
study-tree headers the build takes from the overlay base (`~/.tt-metal-p1-base/tt_metal/fabric`).

References: `R:n` is line n of v3l2be1b's `fabric_erisc_router_v3.cpp`; `CT`, `TX`, `CH`, `SP` are its
`fabric_erisc_router_ct_args.hpp`, `fabric_edm_packet_transmission.hpp`, `fabric_erisc_datamover_channels.hpp` and
`fabric_erisc_router_v3_speedy_path.hpp`; `base/<file>:n` is a study-tree header under the overlay base. In v3r the ERISC 0
program is `sender_erisc` (was `e0`) and the ERISC 1 program `receiver_erisc` (was the `e1_*` functions); names below
are v3l2be1b's unless they say v3r.

Tags: **must-fix** (before a mainline PR), **should-fix** (in the PR or right after), **nit**.

## Top findings

1. **Unstated invariant with a latent bug, must-fix.** A deferred flushed atomic is staged in `write_at_cmd_buf` and
   fired later without a ready check, so nothing may use that command buffer in between. The bypass predicates encode
   this only implicitly (R:1623-1627, R:3282-3286), and the generic loop's copy gets it wrong. An intermesh-exit packet
   (`action == ACTION_LOCAL_DELIVER`) passes its test, and its stateless egress forward then writes its credit with
   `noc_inline_dw_write`, which uses `write_at_cmd_buf`. That overwrites the staged atomic, and the deferred fire then
   reissues whatever the credit write left in the buffer. E1's copy has the missing `!intermesh_egress`. Not reachable on logged configs: no edge router there
   defers. Fixed in v3r, codegen-neutral. v3e1c is changing this code.
2. **Two init paths must stay in lockstep, must-fix.** `kernel_main` and `e1_kernel_main` must make the same six
   `wait_for_other_local_erisc()` calls (R:5176-5822 vs R:4837-5085). A sync added to one path hangs every two-ERISC
   router at init. Only a comment (R:4794) says so.
3. **Study host interface, must-fix.** The router reads `EXP_SEQ_UPDATE_TIMEOUT` (R:4612-4638). Its ct_args reads about
   20 more study-only named args (CT:628-731, 983-1008). A name the host does not pass fails to compile
   (`get_named_ct_arg` hits `__builtin_unreachable()`), so v3l2be1b does not build against main's host. All §93 numbers
   were taken with a keepalive timeout of 1000 cycles; the firmware default is 8000.
   **Status (RESULTS §95, v3tk):** done. The keepalive is a host setting:
   - `FabricEriscDatamoverBuilder::set_eth_txq_seq_update_timeout`, passed as `ETH_TXQ_SEQ_UPDATE_TIMEOUT`;
   - 1000 in the Blackhole block of `compute_mesh_router_builder.cpp`, 0 (firmware value) elsewhere.
   v3tk reproduces the 1000-cycle numbers without the study override. By a name check, v3tk reads no named arg that
   main's host lacks, apart from this new one. That is not tested by building against main's host.
4. **Coverage gap, must-fix.** Every logged config is Blackhole: Wormhole, UDM, telemetry, profiling, VC2, data cache,
   spin-wait and fuse=0 builds were never compiled in this study. 168/208 E0 and 128/328 E1 logged configs run the
   specialized programs; the rest run the generic loop.
5. **Init ownership gap, should-fix.** Neither ERISC runs `remote_receiver_channels.init` any more: v3l2 counted it as
   receiver side, v3e1b as sender side. It is harmless only because both ends of every logged link share a layout.
   E0 still builds and opens the UDM relay, which is receiver-side state.
6. **Compiler-behavior assumptions.** The grouped-read fences promise less than their comments said. Further: the
   E1 1D path adds a second `[[unlikely]]` (it has no effect; v3r drops it), `optimize("no-jump-tables"/"Os")` attributes are inherited from main, the
   bit tests are written as sign tests to suit the ISA, and the `0x7FF` mask is chosen to fit an immediate.
7. **Fragility measured while making v3r.** Some semantically null edits changed codegen. Removing an unused,
   inlined parameter renamed registers in 113 E1 configs at -O2 and 88 at -Os (0 at -O3). A FORCE_INLINE heartbeat
   helper permuted E0 spill slots in the hot pass. Turning the `template <bool UNUSED>` helpers into plain functions
   reordered `.text`.
8. **Duplication.** The deferred-atomic resolution, the 2D decode and intermesh exit, the init sequence (TX queues,
   stream registers, a 57-line handshake block, runtime args), the outer-loop housekeeping (3 copies), the send path
   (2 copies) and the worker credit writes (E0 restates the interface's) are each written more than once.
9. **Structure.** One 5,879-line file holds three programs with inconsistent naming (`namespace e0` vs `e1_` prefixes),
   and comments refer to v3a and "as before". The shipped headers still carry DRISC offload code and experiment flags.

---

## a. Correctness and robustness

### a1. Protocol invariants the code relies on but does not state or check

| # | Tag | Invariant | Where | State in v3l2be1b | v3r |
| --- | --- | --- | --- | --- | --- |
| 1 | must-fix | While a deferred flushed atomic is pending, `write_at_cmd_buf` holds its address/data/control and is fired by `noc_semaphore_inc_with_state<true, skip_cmdbuf_chk=true>`; no other NoC command may use that buffer until then. `noc_inline_dw_write` uses it: the stateless forward credit writes of E1's cursor (R:2791) and of the generic adapter (`base/hw/inc/edm_fabric/fabric_router_adapter.hpp:339-343`), and inline-write packets. Atomics use it too. | R:1611-1670 (generic), R:3270-3328 (E1), TX:181-213 | Encoded only by `can_bypass` (plain write, and either a stateful single-live forward or local only). An intermesh exit is `action == ACTION_LOCAL_DELIVER` (`fabric_common.h:338`) with a stateless egress forward. E1's copy excludes it (`!intermesh_egress`, R:3286); **the generic copy does not (R:1623-1627): latent bug**, reachable on a 2D edge router without VC1 whose ERISC 1 runs the generic loop with deferral on (none logged). | Generic copy fixed (`!intermesh_egress`; identical PT_LOAD on every logged config: `intermesh_egress` is constant false off edge routers, and no logged edge router defers). Rule stated at `can_defer_local_atomic_flush` and `forward_payload`. After v3e1c: one shared function with an explicit "uses write_at_cmd_buf" predicate. |
| 2 | must-fix | Both ERISCs pass the same `wait_for_other_local_erisc()` sequence: 6 in init, 3 in teardown. The sync protocol (`0x0fed`/`0x1bad` in one scratch register, R:4645-4662) cannot detect a phase mismatch. | R:5176, 5632, 5694, 5753, 5805, 5822 vs R:4837, 4997, 5055, 5067, 5078, 5085 | Comment at R:4794 only | Stated in the file comment and at `run_kernel`. Fix: one init skeleton with per-ERISC hooks, or a phase number in the sync value (a mismatch then hangs with a readable value, or ASSERTs under watcher). |
| 3 | should-fix | Stream credit counts stay in [0, slots]. The `& 0x7FF` masks (R:1031, R:2567) and the VC0 single-bit idle test (`skip_idle_bit_test`, R:1016-1044) are exact only inside that range. A register driven out of range by a credit error reads as idle (bit test: the §92 hang mechanism) or, on E1, a wrapped local credit copy (0x1FFFF) reads as 2047 free downstream slots (over-send). | R:1016-1044, R:2565-2572, R:2774, R:2948 | Range assumed; the bit test's failure mode is not mentioned. v3l2b already restored exact predicates on E0's VC1 path. | Documented at `stream_count_mask` and `skip_idle_bit_test`. Consider a watcher-only ASSERT on the range. |
| 4 | should-fix | The deferral's 4-bit trid arithmetic (`(trid_s - trid) & 0xF`, `(transaction_id - 1) & 0xF`) assumes channel 0's trids start at 0, number 16 and follow slot order. | R:1622-1635, R:3281-3294, TX:366 | `NUM_TRANSACTION_IDS == 16` is in `can_defer_local_atomic_flush`; the start is not checked. | `static_assert(RX_CH_TRID_STARTS[0] == 0)` added. |
| 5 | should-fix | E1's slot records pack slot 4w+k into byte k of word w (little-endian), so `four_from` can shift. | R:2694-2724 | Unstated | `static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)` and comment added. |
| 6 | should-fix | E1's exact completion check holds only if every NoC command that reads a slot uses the local-write or the forwarding data command buffer (one record bit each), and each buffer's writes leave the NIU in issue order. | R:3038-3118 | The ordering premise is stated; the "only two command buffers read slots" premise is not. A new send type that reads the slot on a third buffer (v3e1c's area) would break it silently. | Both stated in v3r's comment. |
| 7 | should-fix | 1D route field bits: E1 tests `routing & WRITE_ONLY` and `routing & FORWARD_ONLY` separately, which needs `WRITE_AND_FORWARD == WRITE_ONLY \| FORWARD_ONLY` and `NOOP == 0`. | R:3375-3377 | Unchecked | `static_assert` added. |
| 8 | should-fix | The scatter packet's relaxed-flush flag sits in the 2-bit encoding slot after the last chunk: host side `base/fabric_edm_packet_header.hpp:778` and `base/hw/inc/api_common.h:425`, router side `(packet_encoding >> 2) & 1`. | TX:556 | Unstated on both sides | Post-port (header in v3e1c's area): name the bit on both sides. |
| 9 | nit | E0 restates the worker interface's credit writes for channel 0. It writes `credits_returned` to `cached_worker_semaphore_address` with `notify_worker_of_read_counter_update_src_address`, or to `local_sender_channel_0_connection_info_addr`. This holds only because `init_sender_channel_worker_interface` builds channel 0 with exactly those values. | R:4116-4152 | Unchecked | Documented in the sender_erisc comment. Fix: an interface method that takes the count. |
| 10 | nit | E1 rebuilds the NIU per-trid status address itself (`NOC_STATUS(NIU_MST_WRITE_REQS_OUTGOING_ID(0))` + trid words) instead of the tracker's `transaction_sent_reg_addr`. | R:3093-3103 | Duplicated knowledge | Proposal (codegen-neutral expected, not tried). |
| 11 | nit | `kernel_main` on every ERISC zeroes `edm_read_counter` of every sender channel (R:5379-5383). In the same init phase, ERISC 0's worker interface constructors set it to `SENDER_NUM_BUFFERS` for forwarding channels. When ERISC 1 runs `kernel_main` (generic two-ERISC builds) the two race, as on main. | R:5379-5383, R:4546-4553 | Inherited from main. Benign only if no peer reads a forwarding channel's `edm_read_counter`. | Not changed. The E1 program (v3e1b) no longer writes it. |

### a2. The E0/E1 init-ownership contract

What ERISC 0 skips when `e0::runs` (R:5409-5413, R:5578-5583): the VC0/VC1 downstream adapters (teardown-semaphore
zeroing, `init_ptr_val` of the local downstream free-slots copies), `local_receiver_channels.init` (receiver header
zeroing), and `remote_receiver_channels.init`. ERISC 1 does these writes either in `kernel_main` (generic path,
`sets_up_receiver_side`) or in `e1_kernel_main` (`e1_init_downstreams`, `local_receiver_channels.init`), between the
same two syncs. E1 skips the sender side (`edm_read_counter`, sender header zeroing, worker interfaces, connection
semaphores, `initialize_state_for_txq1_active_mode_sender_side`), which E0 does.

| Tag | Finding |
| --- | --- |
| should-fix | **`remote_receiver_channels.init` is done by neither ERISC** in the v3l2b + v3e1b pairing (V3L_NOTES §9). It zeroes headers in *local* L1 at the *far* receiver's slot addresses. That is redundant where both ends share a layout (those are the local receiver headers, which E1 zeroes; verified for the VC1 configs), and wrong where they do not (it would overwrite other local L1). The far receiver always zeroes its own headers. Nothing needs it. v3r documents this at `run_kernel` and in `kernel_main`, which still does it on the generic path. Decide for mainline: drop the L1 writes (keep the address setup), or `static_assert` that the two layouts match. |
| should-fix | **UDM relay.** E0 under `e0::runs` still constructs the local relay adapter and opens it (R:5541-5572, R:5733-5739), as main does on both ERISCs; `e0::runs` does not exclude `udm_mode`. The relay is receiver-side (local delivery). No UDM config was logged, so the double open is untested. Give it to ERISC 1 only, or exclude UDM from `e0::runs`. |
| should-fix | **Adapter mirror.** `e1_init_downstreams`/`e1_open_downstreams`/`e1_setup_downstream_cmd_bufs` repeat the writes of `RouterToRouterSender`'s constructor, `open()` and `setup_edm_noc_cmd_buf()` by hand (R:3418-3577). Any change to the adapter must be made in both places. v3r states this as a maintenance contract at `DownstreamCursor`. Better: have the adapter expose its init writes as a function both paths call. |
| nit | **`fuse_receiver_flush_and_completion_ptr`**: the generic loop has a separate flush counter when it is 0 (R:1722-1745), and `e1_receive_program` neither honours nor excludes it (R:2542-2562). Moot: the compile coverage below shows 0 is unsupported. `ct_args` `static_assert`s 1, the builder never emits 0, and the 0 branch does not compile, as on main. Delete the branch. |
| nit | E1's comment says ERISC 0 "initializes ... the far receiver's channel view" (R:4796); it does not (above). Fixed in v3r. |
| ok | Same CT and RT args on both ERISCs (V3L §7, checked per config pair), and E1 parses RT args with kernel_main's layout (R:4872-4927). Retrain sync init (R:5170) is master-only and E1 is never master. POSTCODEs and waypoints match. |

### a3. Assumptions that hold only for the logged configs

Logged set: 536 configs at -O3/-O2 (534 at -Os). Program per config (from symbols of the base build):

| ERISC | Program | Configs |
| --- | --- | --- |
| E0 | e0 (sender_erisc) | 1D line 40, 1D ring with deadlock avoidance + first-level ack 40, 2D VC0 40, 2D VC1 crossover 16, 2D VC1 serviced 32 |
| E0 | generic | speedy 40 |
| E1 | e1 (receiver_erisc) | 1D line 40, 2D VC0 40, 2D VC1 crossover 16, 2D VC1 serviced 32 |
| E1 | generic | single-ERISC 1D 40, single-ERISC 1D ring 40, single-ERISC 2D 40, two-ERISC 1D ring (deadlock avoidance) 40, speedy 40 |

| Tag | Assumption | Where |
| --- | --- | --- |
| must-fix | Keepalive: `EXP_SEQ_UPDATE_TIMEOUT` = 1000 in all 536 logged configs, read through a study-only named CT arg. Main's host does not provide it. With 0 the router leaves the firmware value (8000), which is not what was measured. Make it a host setting (or a constant) and re-measure. | R:4612-4638 |
| must-fix (now done offline, see "Compile coverage") | Only Blackhole was built. Never compiled in this study: `#if ARCH_WORMHOLE`, `is_blackhole_arch == false`, `enable_trid_flush_check_on_noc_txn`, telemetry (bandwidth, heartbeats, router state), perf telemetry, code profiling, channel-trimming capture, `ENABLE_RISC_CPU_DATA_CACHE`, both spin-wait flags, VC2, UDM, first-level ack on VC1, `fuse_receiver_flush_and_completion_ptr == 0`. (`enable_interrupts` is read but used nowhere, as on main.) V3L §7 compiled telemetry and trimming capture for E0 only. Before the PR: build every production config type (CI's fabric tests) for v3 on both architectures, and run the production-path tests on hardware. | |
| should-fix | Link layout symmetry (remote receiver entries equal local ones). See a2. | R:5582 |
| should-fix | E1 never has context switching enabled and is never the retrain master (`!enable_context_switch` is in `e1_receive_program`). E0 always has `IDLE_CONTEXT_SWITCHING`. The non-idle context-switch branch is untested. | R:2549, R:4470-4486 |
| nit | VC1 sender channels have 2 slots and VC1 receiver rings 4 trids for 8 slots. E1's completion check is conservative with shared trids (correct, waits longer). | R:3042-3049 |
| ok | Both specialized programs gate themselves on explicit compile-time conditions (`e0::runs` with `shape_supported`, `e1_receive_program`) and fall back to the generic loop, so unlogged shapes run v3a's generic code rather than a wrong specialization. |

### a4. Feature parity with main

Every identifier in main's `if constexpr` conditions also appears in v3's (checked mechanically), and v3's generic loop
is main's plus the study's changes (v3a). Per feature:

| Feature (main) | v3 generic loop | sender_erisc (E0) | receiver_erisc (E1) |
| --- | --- | --- | --- |
| 1D line / ring (low-latency routing, sparse multicast) | yes | yes (channels 0-1) | yes (1D path; ring with deadlock avoidance falls back) |
| 2D action routing, intermesh landing, intermesh exit, boundary-dir check | yes | n/a | yes |
| VC1 active / serviced, VC0→VC1 crossover | yes | yes | yes (channel 1 via shared bodies) |
| VC2 | yes (speedy-style steps) | falls back | falls back |
| Speedy path | yes (`SP`, a copy of main's file) | falls back | falls back |
| Deadlock avoidance / bubble flow control | yes | yes (`bubble_channels`) | falls back |
| First-level ack | yes | yes | falls back |
| Stream-register (non-counter) credits | yes | falls back | falls back |
| Single-ERISC mode | yes | n/a | n/a |
| skip_src_ch_id_update | yes | yes | falls back |
| UDM relay | yes | runs, and opens the relay (a2) | falls back |
| Tensix mux / downstream tensix wait | yes | yes (kernel_main) | yes (R:5060-5063) |
| Context switch, retrain sync, router commands (run/pause/drain/retrain) | yes | yes (master) | non-master handshake only |
| Telemetry: bandwidth, TX/RX heartbeat, router state | yes | yes | yes |
| Perf telemetry recorder | yes | yes | n/a (sender-only, as main) |
| Code profiling (RECEIVER_CHANNEL_FORWARD timer) | yes | n/a | yes |
| Channel-trimming capture | yes | yes | yes |
| RISC data cache | yes | falls back | runs; invalidates per step as the generic loop does |
| Spin-wait TX queue flags | yes | falls back (send) | yes (completion ack) |
| FORCE_ALL_PATHS_TO_USE_SAME_NOC, trid flush check | yes | n/a | falls back |
| `!fuse_receiver_flush_and_completion_ptr` | does not compile (as on main); `static_assert`ed off | n/a | n/a (unsupported) |
| `update_write_counter_for_send` on every send | removed in v3a: it only fed `local_write_counter`, which nothing reads (the speedy copy still calls it) | removed | n/a |
| Main's four `asm volatile("nop")` in init (#35346) | removed in v2/v3a. They were added on main to undo a perf regression that POSTCODEs caused through code layout: the same fragility class as goal 1. | | |

Missing: nothing that main's logged or unlogged paths need, apart from the `fuse == 0` gap and the UDM ownership
question above.

---

## b. Compiler-behavior assumptions

Measured while making v3r (exact PT_LOAD comparison against v3l2be1b):

| Edit (semantically null) | Effect |
| --- | --- |
| Remove the unused `stream_id` parameter of the FORCE_INLINE `establish_edm_connection` (its argument was a dead array load) | -O3: none. -O2: 113 E1 ELFs differ, -Os: 88. All are single-ERISC or generic E1s: registers renamed (s1↔s2), same instruction count. Kept out of v3r. |
| A FORCE_INLINE `update_fabric_heartbeat()` helper replacing the 3-line heartbeat update in the outer loops | -O3: 11/13 sampled E0s change. Three spill slots are permuted and one branch's operands swap in sender_erisc's pass. Kept out of v3r; v3r shares only the constant. |
| `template <bool UNUSED = true>` / `template <int = 0>` removed from 7 noinline helpers | Same instructions; those functions move earlier in `.text` (plain functions are emitted in definition order, template instantiations after), so every later address shifts. Kept out of v3r. |
| Renaming identifiers and namespaces, moving the two programs into headers, line shifts, comments, removing unused templates and constants, removing `optimize("no-jump-tables")` on two FORCE_INLINE functions, removing E1's second `[[unlikely]]`, rewriting the generic sign-bit tests, adding `!intermesh_egress` to the generic bypass | none (all 536/536/534) |
| The same sign-bit rewrite in sender_erisc's `send_packet` (v3r2) | -O3: all 168 sender_erisc E0s change; the hot pass is reallocated (344 of 832 instructions differ, 2 fewer). -O2: likewise; -Os: none. The same expression change is neutral in the generic loop. |

So "byte-identical" is reachable for text, naming and file structure, but not for small refactors of inlined code:
allocation tie-breaks move with the IR. These moves are benign (same instruction counts). Whether they move cycles is
what v3r2 measures.

Inventory, most fragile first:

| # | Tag | Construct | Where | Fragile? | Replacement or justification |
| --- | --- | --- | --- | --- | --- |
| 1 | should-fix | Grouped reads: volatile loads between two `std::atomic_signal_fence(seq_cst)`. The comment at R:987 says the fence "keeps all N stream-register loads back-to-back". The fence only keeps memory accesses from crossing it; arithmetic (including a use of the first value) may still be scheduled between the loads, which would serialize them. `read_grouped_free_slots` has no opening fence. | R:974-989, R:1112-1116, R:1162-1166, R:2983-2988, R:3110-3113, R:3300-3304, R:4037-4041, R:4062-4066, R:4276-4284 | Moderate: correct by construction, overlap depends on the scheduler. The brief allows this form ("v3a's fences"). | v3r states the real contract once ("Grouped reads", above `grouped_packet_check_channels`). Durable fix: one primitive, e.g. `load_group<N>(addrs) -> values` as a single `asm volatile` block of N `lw` with a memory clobber, which fixes issue order and adjacency. One place to audit, codegen changes on the hot paths: measure. |
| 2 | should-fix | Scheduling fences that order no reads: after the `src_ch_id` stamp (R:1122, R:4047) and after the free-slot bookkeeping (R:1170, R:4070). Volatile accesses are already ordered among themselves, so these only pin the position of the stamp and bookkeeping relative to the branch and the TX-queue writes. | same | Steering by intent (overlap with read latency) | Measure without them (a separate proposal, not in v3r2). If they matter, make them part of the documented primitive in #1; otherwise drop. |
| 3 | should-fix | `static_cast<int32_t>(x << (31 - ETH_TXQ_STATUS_CMD_ONGOING_BIT)) < 0`: a bit test written as a sign test so that it compiles to `slli; bltz` | R:1124, R:1172, R:4049, R:4072 | Not fragile, but obscure | `(x >> BIT) & 1`. Codegen-neutral for the generic loop (v3r applied it there); E0's two are in v3l3's area (v3r2 applies them). |
| 4 | should-fix | Second `[[unlikely]]`, on E1's 1D sparse write-and-forward. The brief allows only main's (R:429). | R:3380 | None in practice: removing it changes no ELF at any level | Removed in v3r. |
| 5 | should-fix | `__attribute__((optimize("Os")))` on seven cold init/teardown functions (from main) | R:4561, 4620, 4631, 4645, 4664, 4744, 4752 | Size steering, and an inlining barrier: GCC does not inline across differing `optimize` options. Without it, E1's hot `kernel_main` absorbs the helpers (+115 instructions in the mid-W E1). GCC documents `optimize` as not meant for production code. | Remove (v3r2). VC1 alone: E0 −120 B, E1 +276 B at -O3 (25,496 B, fits). |
| 6 | nit | `__attribute__((optimize("no-jump-tables")))` on FORCE_INLINE `admit_2d_dispatch`/`forward_2d_dispatch` (from main) | R:560, R:654 | None: verified no effect (always inlined, so the caller's options apply) | Removed in v3r. |
| 7 | nit | `optimize("jump-tables")` on local delivery and the relay (from main) | TX:288, 581, 602 | Same situation as #6 for the FORCE_INLINE ones | Remove after v3e1c (its area); expect no effect. |
| 8 | should-fix | Steps `FORCE_INLINE` except under `FABRIC_2D_VC1_ACTIVE` (from main) | R:1254, 1318, 1386 | Code-size shaping: inline vs out of line by config | Acceptable as a documented size policy; in v3 the specialized programs cover all logged VC1 builds, so it only affects unlogged single-ERISC VC1 builds. |
| 9 | should-fix | Main's comment "Keep hot routing coordinates in address-untaken scalars so they remain in registers" and the "Workaround the perf regression in RingAsLinear test" copies of the channel pointers out of tuples | R:2009-2015, R:2028-2033 | Allocation assumptions (SRA, register residency) | v3r rewrites the comments to say what they are. Structural fix: give the generic loop per-channel locals by construction, as the specialized programs do. |
| 10 | nit | Main's "manual unroll ... weirdness of codegen" in `init_local_sender_channel_worker_interfaces` | R:4569-4572 | Cold code | Keep or simplify with an index sequence. |
| 11 | nit | `#pragma GCC unroll 1` (header zeroing) and `unroll 16` (route refill), from main | CH:59, 107, 280; TX:660, 668 | Size steering in cold or one-time code | Acceptable, documented. The study-added `unroll 1` in `LeanStrideSenderEthChannel` (CH:177) is dead code, removed in v3r. |
| 12 | nit | `[[likely]]`/`[[unlikely]]`/`__builtin_unreachable()` in the local-delivery dispatch | TX:309, 331, 446, 518, 655 | Main's except TX:446 (dead DRISC branch) | Keep main's, remove the DRISC branch after v3e1c. |
| 13 | nit | `volatile auto p = reinterpret_cast<volatile T*>(...)`: the pointer variable itself is volatile, so every use reloads it from the stack (from main) | R:4866-4870, R:5209-5212 | Unintended; cold | Plain `auto` (v3r2). |
| 14 | nit | `0x7FF` count mask "fits an immediate" | R:1030-1031, R:2565-2567 | Chosen for the ISA; exact under invariant a1.3 | v3r: one `stream_count_mask` with a semantic comment. |
| 15 | nit | `template <bool UNUSED = true>` / `template <int = 0>` on noinline helpers and on `e1_kernel_main` | R:2802, 2825, 3477, 3514, 3546, 4006, 4288, 4799 | Not fragile. On `e1_kernel_main` it is needed: it defers instantiation, hence its `static_assert`s, to builds that run it. On the noinline helpers it only changes their placement (above). | v3r renames the parameter `INSTANTIATED_ON_USE`. v3r2 makes the helpers plain functions. |
| ok | | `PackedHeaderWord40` (one L1 load for size, type, src) and E1's record words: semantic, fewer uncached loads, layout `static_assert`ed | R:471-487 | | |
| ok | | `noinline` on cold or rarely taken bodies (pause, static-connection wait, connection events, upstream credit, TX-queue spin, VC1 shared bodies); `always_inline` lambdas | | Structural: they keep cold code out of the pass, and the code-size rule asks for that | |
| ok | | Struct-of-records state (`ForwardingState`, `E1RxState`) | R:3966-4002, R:2727-2735 | Performance depends on SRA, correctness does not; `statefield` leaves the hot loops unchanged (V3L §10) | |

---

## c. Structure and readability

| Tag | Finding | v3r |
| --- | --- | --- |
| should-fix | **One file, three programs.** E0 lives in `namespace e0`, E1 in `e1_`/`E1` prefixed globals, the generic loop in the global namespace. | Moved into `namespace sender_erisc` / `namespace receiver_erisc` (renames via `v3r_renames.py`). The two programs live in `tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_v3_sender_erisc.hpp` and `..._receiver_erisc.hpp`, included where they were, as main does with the speedy path. `receiver_erisc::run_kernel` (ERISC 1's init) stays in the .cpp next to `kernel_main`, so the two init sequences can be compared side by side. The .cpp drops from 5,879 to 3,945 lines. |
| should-fix | **Duplication between the specialized and generic paths.** Deferred-atomic resolution (R:1611-1670 vs R:3270-3328, ~60 lines, already diverged); 2D decode + intermesh exit + boundary check (R:1536-1595 vs R:3199-3253); the send path (`sender_channel_try_send` R:1077-1199 vs `send_packet` R:4019-4087, same sequence); outer-loop housekeeping (telemetry, heartbeat, context switch, perf window: R:2415-2464, R:3673-3691, R:4452-4497); init (TX queue, stream registers, the 57-line handshake block R:4998-5053 = R:5634-5691, RT-arg parsing). | Heartbeat address deduplicated. The init handshake helper is in v3r2. After the ports: one deferred-atomic function and a shared decode. |
| should-fix | **Function sizes.** `kernel_main` 765 lines (R:5115-5879), `run_fabric_edm_main_loop` 525 (R:1981-2505), `run_receiver_channel_step_impl` 380 (R:1450-1828), `e1_kernel_main` 320, `e1_rx_step` 250, `e0::run_main_loop` 200. | Not split (each split risks codegen, as b showed). Proposal: split `kernel_main` into arg parsing, receiver-side setup, sender-side setup and connection phases, the same skeleton for both ERISCs (fixes a1.2). |
| should-fix | **Names.** `e1_*`/`E1*` (study jargon for "ERISC 1"), `grouped_free_slots_imm_mask` (names an ISA detail), `run_e1_receive_program`, `UNUSED` template parameters, `generic_rx_local_delivery_impl` (a pass-through). | Renamed (map in `v3r_renames.py`): `receiver_erisc::{RxChannel, RxChannelState, SlotRecords, DownstreamConnections, DownstreamCursor, rx_channel_step, send_completions, retire_slot, local_delivery, forward_to, admit_2d, run_main_loop, run_kernel, runs, ...}`, `sender_erisc`, `stream_count_mask`, `INSTANTIATED_ON_USE`; the pass-through is gone. |
| should-fix | **Leftover scaffolding.** Comments citing v3a, "E0/E1", "as before", "unchanged"; the ct_args experiment block (CT:628-731, CT:983-1008: `EXP_FLAGS`, phase profiler, DRISC layout, `PEER_SENDER_CHANNEL_*`, ~20 named args); `LeanStrideSenderEthChannel`, `LeanSenderChannelBuffersFromAllocs`, `advance_..._by_stride_ct` (CH:139-198, unused); the `fabric_phase_profiler.hpp` include; DRISC branches in TX and in study-tree headers (section e); `EXP_SEQ_UPDATE_TIMEOUT`; the speedy copy's comment about v1/v2. | Removed or rewritten, except: TX's DRISC branches and the deferral comments in E1's step (v3e1c's area), and `EXP_SEQ_UPDATE_TIMEOUT` (host interface). The `STABILITY_PROBE_*` markers are kept as asked; drop them in the PR unless the team adopts `perturb.py`. |
| nit | **Dead parameters.** `port_direction_table` (never written or read, threaded through every receiver step and the speedy copy), `remote_receiver_channels` of `run_sender_channel_step`, `populate_..._ordered_map`'s first argument, `establish_edm_connection`'s stream id (all from main). | The two middle ones removed. `establish_edm_connection`'s is marked `[[maybe_unused]]`: removing it changes codegen (b). `port_direction_table` kept: it runs through the speedy copy, which should stay identical to main's file. |
| nit | `run_drain_step` is `ASSERT(false)` (main); `FABRIC_KERNEL_HEARTBEAT_ADDR` repeated in three loops. | Heartbeat address shared (with its meaning: the Ethernet firmware's heartbeat word). |
| nit | `kernel_main` returns early into `e1_kernel_main` (R:5118-5121) but the generic init and loop after it still get instantiated and compiled for those ERISC 1 builds (dead code, then removed). An `else`, or moving the generic body into its own function, would avoid that and make the three programs visibly disjoint. | Not changed (a 700-line re-indent). |
| nit | The E1 block is not clang-formatted: 9 lines exceed 120 columns (R:2534-3600). | Left for one clang-format pass after the ports (it touches v3e1c's lines). |

---

## d. Comments

Density is about right for the generic code (main's) and the two program overviews. Some comments are essays about
measurements, others are wrong.

| Tag | Where | Problem | v3r |
| --- | --- | --- | --- |
| must-fix | file top (R:5-9) | Main's 240-line EDM overview was replaced by 5 lines about a macro. Nothing tells a newcomer that one file holds three programs, or how they are selected. | A 25-line overview: channels, credits, the three programs and who runs which, init ownership and the sync contract. |
| must-fix | missing | Invariants a1.1, a1.2, a1.6 (deferral owns `write_at_cmd_buf`; sync count; two command buffers) | Added at `can_defer_local_atomic_flush`, `forward_payload`, `send_completions`, file top, `run_kernel`. |
| should-fix | R:987 | "keeps all N stream-register loads back-to-back before ch0 masks raw[0]": overclaims (b1) | Rewritten; contract stated once. |
| should-fix | R:1612-1614, R:3271-3273 | "Relaxed: 0x10 \| the trid of its own packet": for a plain atomic it is the previous packet's trid (TX:366 vs TX:394) | Encoding documented at `can_defer_local_atomic_flush`. R:1612 fixed; E1's copy is post-port (v3e1c's area). |
| should-fix | R:4794-4798, R:3769-3771 | Claim E0 initializes the far receiver view and E1 does "the same writes"; neither runs `remote_receiver_channels.init` (a2) | Fixed. |
| should-fix | R:1015, R:1037 | The bit test's failure mode is not mentioned (a1.3) | Fixed. |
| should-fix | R:2009-2015, R:2028-2033 | Allocation claims; "the RISC data cache is disabled for this build" is not true of every build | Rewritten. |
| nit | R:4601 | "copy ... (in L1) to local memory for performance": it is a constexpr table | Fixed. |
| nit | R:2514, 2525, 2527, 2529, 2532, 2539, 2867, 3043, 3048-3049 | "unchanged", "as before", "v3a's" | Fixed. |
| nit | R:4004, R:4104 | Measurements in comments ("never seen in practice", "0.2-0.5 per packet") | Left (v3l3's area). |
| nit | SP:14-16 | "Must be included AFTER ... helpers such as send_next_data": v3 has no `send_next_data` | Left: the copy should stay identical to main's file. |
| nit | TX:289-291, 582-584 | Commented-out `// #ifndef FABRIC_2D` around FORCE_INLINE (main) | Post-port. |

---

## e. Header changes vs main

The router also depends on study-tree headers that are not in the variant dir; they ship with it too.

| Tag | Header | Change vs main | Finding |
| --- | --- | --- | --- |
| must-fix | `fabric_erisc_router_ct_args.hpp` | Experiment block, ~130 lines and ~20 study-only named args; `rx_batch_completion`; 16 trids for the batched receiver | v3r removes the experiment block and the phase-profiler include. What remains vs main is `rx_batch_completion` (true for v3, false otherwise) and `NUM_TRANSACTION_IDS`. This ct_args no longer builds the study's v1/v2 routers or the DRISC datamover, so v3r can only replace v2 in v3 runs, as in every Phase 1 run. |
| must-fix | `fabric_edm_packet_transmission.hpp` | DRISC offload code (TX:108-118, 218-283, 336-478 under `COMPILE_FOR_DRISC`); the deferral (TX:152-213); the BH global flush (TX:104-123); `FLUSH_RELAXED` decode; `forward_and_local_write_noc_vc` on scatter chunks; an `update_local_free_slots` template parameter (never false outside DRISC: remove) | DRISC branches are dead for ERISC builds: remove (after v3e1c, whose area this is). `#if defined(FABRIC_ERISC_ROUTER_V2) \|\| defined(...V3)` (TX:188) becomes V3 only when v2 goes. Name the deferral encodings (a1.1) and the scatter relaxed bit (a1.8). `noc_semaphore_inc_set_state<true>` passes `posted = true`, which `noc_fast_atomic_increment_set_state` overrides to false on the line after: misleading, use `false`. The global-flush semantics change (prior atomics no longer waited for) is documented in its comment and limited to the batched receiver. Fine for v3; call it out in the PR. |
| should-fix | `fabric_erisc_datamover_channels.hpp` | Stride-advance fields and method; `LeanStride*` (unused); `STABILITY_PROBE_STATE` | v3r removes the unused classes and the `_ct` advance; the stride fields and method stay (the generic loop uses them). |
| nit | `fabric_erisc_router_v3_speedy_path.hpp` | Identical to main's `fabric_erisc_router_speedy_path.hpp` but for a comment | Delete the copy at mainline and include main's file. v3r only rewrites the comment to say so. |
| should-fix | `base/hw/inc/edm_fabric/fabric_router_flow_control.hpp:37-84,171-210` | `accumulate_completion_credit`/`flush_completion_credits` (batched credits), `note_packet_sent`/`has_*_pending` | Needed by v3. Comments call them "Experiment (EXP_TX_LEAN_COMPLETIONS)": reword. |
| should-fix | `base/hw/inc/edm_fabric/fabric_router_adapter.hpp:113-423` | DRISC branches; `update_local_free_slots` template parameter | Remove the DRISC branches. Keep the parameter only if a non-DRISC caller passes false (none in v3). |
| should-fix | `base/hw/inc/edm_fabric/edm_fabric_flow_control_helpers.hpp:233,253,270` | `advance_remote_receiver_buffer_pointer_ct` (unused by v3); `exp_prefetched_pkts_received` (dead state in `ReceiverChannelPointers`, zeroed at reset) | Remove both. Removing the field changes the generic E1 layout, so it comes with codegen changes (V3A §9). |
| should-fix | `base/hw/inc/edm_fabric/edm_fabric_utils.hpp:47-62` | BH fast path in `send_chunk_from_address_with_trid` (`ncrisc_noc_write_with_state<..., skip_cmdbuf_chk=true>` after its own ready wait) | Needed by v3's forwards. Keep it; drop the empty `#else` left behind. |
| nit | `base/hw/inc/edm_fabric/fabric_erisc_router_transaction_id_tracker.hpp:100-107` | `transaction_sent_reg_addr`, labelled Experiment | Useful: make E1 use it (a1.10), reword. `transaction_flushed()` really means "sent" (no write left in the NIU); the name misleads. |
| should-fix | `base/hw/inc/edm_fabric/fabric_stream_regs.hpp:9,98`, `base/hw/inc/tt_fabric_api.h:11` | `#if !defined(COMPILE_FOR_DRISC)` guards | DRISC scaffolding: remove. |
| must-fix | `base/fabric_edm_packet_header.hpp:252-310,776-780`, `base/hw/inc/api_common.h:425` | `NocFlushType {NONE, STRICT, RELAXED}` replaces `bool flush` in three command headers; scatter encodes relaxed in a spare chunk-encoding bit | A public packet-header API change (binary compatible: true == STRICT). It needs its own PR with host docs and tests. The router depends on it. |
| should-fix | `tt_metal/hw/inc/api/dataflow/dataflow_api.h`, `internal/tt-1xx/{blackhole,wormhole}/noc_nonblocking_api.h`, `tt-2xx/quasar/*` | `noc_semaphore_inc_set_state` / `noc_semaphore_inc_with_state` and the fast-atomic set/with-state pair | Needed by the deferral. `posted` is accepted, then forced to false. Document, or drop the parameter. |

---

## f. Plan

Before a mainline PR, in order:

1. **Land the parallel work on v3r** (v3l3, v3e1c), using the procedure in `v3r_renames.py`: rename both sides, diff,
   `join` v3r, patch, `split`. Hunks in comment lines that v3r also edited may need a hand merge: the receiver_erisc
   overview steps 4/6 and the sender_erisc overview. If a ported variant adds `e1_`-prefixed names, `rename` lists
   them; add them to `RENAMES` and rename both sides again.
2. **Then, in their areas** (all codegen-changing except the comment edits): one deferred-atomic function shared by the
   generic loop and receiver_erisc, with the `write_at_cmd_buf` rule as a predicate (a1.1); E0's sign-bit tests;
   remove TX's DRISC branches and the `V2 ||`; name the deferral encodings and the scatter relaxed bit; fix E1's
   deferral comment.
3. **Host interface:** make the keepalive timeout a real host setting (or a constant), and re-measure (a3). Done: v3tk, RESULTS §95.
4. **Init:** one init skeleton for both ERISCs (sync count by construction, a1.2); settle `remote_receiver_channels.init`
   and the UDM relay owner (a2); add `fuse_receiver_flush_and_completion_ptr` to `receiver_erisc::runs`.
5. **Coverage:** compile every production config type for Blackhole and Wormhole (telemetry, profiling, UDM, VC2,
   data cache, spin-wait, trimming); run validation for the generic-loop builds on hardware.
6. **Headers:** remove DRISC and experiment remnants from the study-tree headers (e), split `NocFlushType` into its own
   PR, delete the speedy copy and include main's file.
7. **Measure v3r2** (below) and adopt what does not cost cycles; decide on the grouped-read primitive (b1) and the
   scheduling fences (b2) by measurement.
8. **For the PR itself:** drop the `STABILITY_PROBE_*` markers (or adopt `perturb.py` in CI), and add the
   `TT_FABRIC_KERNEL_VERSION=3` selector.

**Applied in v3r** (deliverable 2, codegen-neutral on every logged config): renames and namespaces; the split into
two program headers; the file overview and invariant comments; scaffolding comments removed; the ct_args experiment
block and phase-profiler include; the unused `LeanStride*`/`_ct` code; the `generic_rx_local_delivery_impl`
pass-through; two dead parameters; the shared heartbeat address; `optimize("no-jump-tables")`; E1's second
`[[unlikely]]`; semantic bit tests in the generic loop; three `static_assert`s (trid start, little-endian records, 1D route bits). One semantic change: the
generic loop's deferral bypass now also excludes intermesh egress (a1.1, a latent bug fix that changes no logged
config's code).

## v3r

`fabric_exp/variants/v3r/`, the same layout as v3l2be1b plus two mirrored headers:

| File | Lines (v3l2be1b → v3r) | Change |
| --- | ---: | --- |
| `fabric_erisc_router_v3.cpp` | 5,879 → 3,945 | overview, generic loop, init; includes the two program headers where the code was |
| `tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_v3_receiver_erisc.hpp` | new, 1,209 | `namespace receiver_erisc` (was the E1 block) |
| `tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_v3_sender_erisc.hpp` | new, 781 | `namespace sender_erisc` (was `namespace e0`) |
| `fabric_erisc_router_ct_args.hpp` | 1,014 → 880 | experiment block removed; differs from main only by `rx_batch_completion` and `NUM_TRANSACTION_IDS` |
| `fabric_erisc_datamover_channels.hpp` | 697 → 649 | unused study classes removed |
| `fabric_edm_packet_transmission.hpp` | unchanged | v3e1c's area |
| `.../fabric_erisc_router_v3_speedy_path.hpp` | comment only | |

Derivation: `v3r = split(rename(v3l2be1b) + cleanup)`; `v3r_renames.py join fabric_exp/variants/v3r` gives the
single-file form that the cleanup diff and later ports apply to.

**Verification** (offline builds of every logged config, PT_LOAD segments compared by address, sizes and bytes,
keyed by config and ERISC, against v3l2be1b built by the same tool):

| Level | ELFs built | E0 identical | E1 identical | Differ |
| --- | ---: | ---: | ---: | ---: |
| -O3 | 536 | 208 / 208 | 328 / 328 | 0 |
| -O2 | 536 | 208 / 208 | 328 / 328 | 0 |
| -Os | 534 (the 2 malformed logged commands fail, as for v3l2be1b) | 207 / 207 | 327 / 327 | 0 |

So VC1 size is v3l2be1b's: 25,340 / 24,304 / 21,340 B at -O3 / -O2 / -Os.

Probes: `perturb.py v3r OUT hook statefield` applies (1 DECL, 3 PRELOOP/LOOP pairs, 3 STATE structs) and both probe
variants build. At -O3 each probe build of v3r is PT_LOAD-identical to the same probe applied to v3l2be1b on all 536
configs: `hook` changes every config of both against its base, and `statefield` changes the 40 speedy E0s and 120
generic E1s and nothing on the specialized programs, as in V3L §10.

## v3r2 (codegen-changing cleanups to measure)

`fabric_exp/variants/v3r2/` is v3r plus the six changes below, all semantically null. Each was built alone on v3r to
attribute its codegen changes: every logged config at -O3, and the 32 sampled configs at -O2 and -Os (one per config
class, including both hot configs). "Hot" means E0 E/W `10733501891889981293` (`sender_erisc::run_main_loop`, 832
instructions at -O3) and E1 mid-W `10856749301170935304` (`kernel_main`, init plus the inlined main loop, 1,258).

| Change | What | -O3, all configs changed | -O2 / -Os, sampled | Hot code |
| --- | --- | --- | --- | --- |
| templates | the 7 noinline helpers lose `template <bool INSTANTIATED_ON_USE = true>` and become plain functions | E0 168 (all sender_erisc), E1 128 (all receiver_erisc) | 11/13 E0, 9/19 E1 at both | E0 loop: same instructions, moved. E1 `kernel_main`: −6 instructions, re-laid out |
| os | `optimize("Os")` removed from 7 init/teardown functions | all 536 | all at -O2; none at -Os | E0 loop unchanged. E1 `kernel_main` +115 instructions: the helpers are now inlined (the attribute blocked inlining) |
| heartbeat | one FORCE_INLINE `update_fabric_heartbeat()` for the three outer loops | E0 168, E1 38 | O2: 11 E0, 1 E1; Os: 9 E0, 2 E1 | E0 loop: three spill slots permuted, same count. E1 hot: unchanged |
| param | `establish_edm_connection`'s unused stream id removed | none | 5/19 E1 at both (single-ERISC and generic E1s) | none |
| bittest | sender_erisc's two TX-queue busy tests as `(x >> BIT) & 1` | E0 168 | 11/13 E0 at -O2; none at -Os | E0 loop reallocated: 344 lines differ, 2 fewer instructions |
| init | `volatile auto` pointer variables become plain pointers; one `handshake_with_link_peer_and_local_routers()` replaces the two 57-line copies in `kernel_main` and `run_kernel` | all 536 | all | E0 loop unchanged. E1 `kernel_main` −3 instructions (init part) |

**v3r2 against v3l2be1b** (`codegen_diff.py`, every logged config; "differ" counts include instructions that only
moved):

| Level | E0 changed, Δ instructions med/max, Δ frame | E1 changed, Δ instructions med/max, Δ frame | Hot E0 E/W | Hot E1 mid-W |
| --- | --- | --- | --- | --- |
| -O3 | 208/208, +4/−142, −32/−64 | 328/328, +21/−203, −16/−128 | 2,486 → 2,507 insns, frame 384 → 368; loop 832 → 830 | 1,905 → 1,902, frame 432 → 400 |
| -O2 | 208/208, +8/−119, −32/−48 | 328/328, +15/−249, −16/−128 | 2,212 → 2,231, frame 416 → 400; loop 812 → 807 | 1,876 → 1,879, frame 432 → 400 |
| -Os | 207/207, −21/−22, −16/−32 | 327/327, −10/−46, −16/−64 | 1,973 → 1,952, frame 384 → 368; loop 591 → 591 (16 lines differ) | 1,692 → 1,681, frame 400 → 384 |

VC1 E0 + E1 (largest of each): -O3 12,912 + 12,428 = 25,340 B (as v3r; E0 −288, E1 +288), -O2 24,328 B (+24),
-Os 21,236 B (−104). All fit the 26,624 B buffer.

What to read from a hardware run of v3r2 against v3r: on sender_erisc E0s the loop is out of line, and only `bittest`
and `heartbeat` touch it. Wherever the loop is inlined into `kernel_main` (receiver_erisc and every generic-loop
build), `os`, `templates` and `init` re-lay out the loop as well; `heartbeat` also touches 38 E1s. If v3r2 is level with v3r at all three levels, all six are safe to adopt. A
difference beyond noise shows that layout or allocation alone moves cycles: rule 1's failure mode, and a reason to
bisect with the per-change variants: `fabric_exp/variants/v3r2/derive_v3r2.py` (a build-ignored helper in the variant
dir) applies any subset of the six to `v3r_renames.py join` of v3r, and `split` turns the result back into a variant.

Not in v3r2, measure separately if wanted:
- **Scheduling fences (b2):** drop the post-stamp and post-bookkeeping fences in the two send paths. This is the
  one candidate that could cost cycles by design (it may break overlap), and including it would mask the rest.
- Grouped-read primitive (b1), deferred-atomic unification (after v3e1c), shared outer-loop housekeeping,
  `exp_prefetched_pkts_received` removal (changes the generic E1's state layout).

## Ports onto v3r: v3s and v3sb

`fabric_exp/variants/v3s/` is v3l3e1c on v3r, and `fabric_exp/variants/v3sb/` is v3l3be1c on v3r. Both were made with the
`v3r_renames.py` procedure. Two renames were added to the map: `e1_local_write_uncounted` → `local_write_uncounted`
and `COMBINED_TEST` → `SKIP_IF_ALL_IDLE`. Each port changes only the two program headers.

- **Hand merges.** Each port had the same two hunks rejected:
  - `local_delivery`'s comment, which v3r had edited, plus the new helper above it;
  - `ForwardingState::attempted`'s comment and `wait_for_sender_txq_idle`, where v3r had renamed the dummy template
    parameter. Now that the function is FORCE_INLINE, the template is dropped.
  - One more hunk applied with fuzz 1, because of v3r's shared heartbeat constant. It landed correctly.
- **Comments.** These follow v3r's style: the receiver overview's step 6 and `local_delivery` describe the new arms,
  and the DRISC and "unchanged" references are gone.
- **Identity.** Each port is PT_LOAD-identical to its merge on every config: -O3 208/208 E0 and 328/328 E1, -O2 the
  same, -Os 207/207 and 327/327. The probe markers apply.

**v3e1c's arms and the `write_at_cmd_buf` rule (a1.1).**
- Only a plain write may bypass a pending atomic, so a scatter, atomic or fused packet reaches `local_delivery` with
  nothing staged.
- In each arm the staged increment (`execute_local_semaphore_inc`) is the packet's last NoC command:
  - The writes before it go to `local_chip_data_cmd_buf`.
  - In 2D, the forward runs before local delivery.
  - After it come only the record store, the packets-received register update and the Ethernet credit frame.
- The rule holds.

**Scatter counters.** BH's `noc_async_write_one_packet_with_trid` always adds one to `noc_nonposted_writes_num_issued`
and to `_acked` per write (#28758). E1 adds `num_write_chunks` once, after the last write and before the optional
increment. The count is identical for every chunk count and last-chunk encoding:

| Chunk count | Writes counted |
| --- | --- |
| 2 | 1 |
| 3 | 2 |
| 4 | 3 |

Each row is +1 when the last chunk is a write. A semaphore increment (with or without flush) or NOP adds nothing.
- On NOP both paths reach `ASSERT(false)` with the same counters.
- The flushed increment's check reads counters that already include every write, as in the impl.
- Nothing reads the counters between the writes; that matters for #28758's inline-write check.
- The issued commands are the same: the same cmd buffer, trid, VC, size and order.

**v3l3's idle test.**
- The decision is unchanged. Every attempt either yields (and sets `attempted`) or sends (and increments a counted
  `sent`). The count cannot wrap within a round.
- Should-fix:
  - The test now relies on every send path incrementing a counter that `packets_sent()` sums. A future send site
    outside the worker and `vc0/vc1_forwarding` would make a busy router look idle. Write this invariant next to
    `packets_sent`.
  - `attempted` now means "yielded": rename it `send_yielded`.
- Nit: `round_packets_sent` is computed every round even when the idle test is not compiled in. The compiler removes
  it, but the source should compute it under the same `if constexpr`.
- Inlining `credit_upstream_router` and the TX-queue spin is consistent with the review: those calls were only there
  to save code size.

## Compile coverage of production config types (v3sb, and v3a for attribution)

**Method** (`fabric_exp/consolidation/coverage.py`):
- **Mock runs (CPU-only).** `test_tt_fabric` runs with `TT_METAL_MOCK_CLUSTER_DESC_PATH`, with no device. It initializes
  the fabric and JIT-builds the router for every link, logging each compile and link command; it then fails in result
  validation (mock reads), which does not matter.
  - Settings are the logged suites': flags 13129088, keepalive 1000, BW telemetry off, plus each case's own env or
    yaml.
  - Clusters: this host's descriptor (Blackhole P150x8) and upstream's T3K descriptor (Wormhole, 8 chips).
  - VC1: the two-rank split of `vc1_mock.sh` (BH) and upstream's T3K 2x2 multi-mesh bindings (WH).
  - The runtime root is a directory of symlinks into the tree with its own `generated/`, so nothing is written into
    the tree.
  - **Check:** the mock BH 1D configs have exactly the 24 distinct named-arg signatures of the logged hardware 1D
    configs.
- **Synthesized configs.** These are types the builder never emits; each is a mock config with the named args edited
  as the builder would set them:
  - `ENABLE_RISC_CPU_DATA_CACHE`, the two spin-wait flags and `FUSE_RECEIVER_FLUSH_AND_COMPLETION_PTR = 0` are
    constants in `erisc_datamover_builder.cpp` (:1076, :1077, :1134, .hpp:545).
  - UDM comes only from `FabricTensixConfig::UDM`, which `test_tt_fabric`'s yaml cannot select. It is built from the 2D
    mux config: `UDM_MODE`, `LOCAL_RELAY_NUM_BUFFERS = 8` and `NUM_DS_OR_LOCAL_TENSIX_CONNECTIONS = 1`, plus
    `-DUDM_MODE=1` (`connect_to_local_tensix_builder`, `fabric_context.cpp:347`).
- **Builds.** One build per distinct config (ERISC, named args except per-link values, defines): 1,227 configs over 39
  cases, at -O3, -O2 and -Os.

Counts are distinct configs built / attempted:

| Config type | Arch | How | v3sb -O3 / -O2 / -Os | v3a -O3 / -O2 / -Os | Status |
| --- | --- | --- | --- | --- | --- |
| 1D line, 1D ring (deadlock avoidance + first-level ack), 2D mesh, 2D torus XY (deadlock avoidance + first-level ack in 2D) | BH | mock | all build (20, 20, 56, 24) | all build | builds |
| Telemetry (all stats), 2D and 1D | BH | mock, `TT_METAL_FABRIC_TELEMETRY=1` | 56/56, 20/20 at each level | same | builds |
| Perf (BW) telemetry, 2D | BH | mock, `TT_METAL_FABRIC_BW_TELEMETRY=1` | 56/56 | same | builds |
| Code profiling, 2D | BH | mock, `TT_FABRIC_PROFILE_RX_CH_FWD=1` | 56/56 | same | builds |
| Channel-trimming capture, 2D and 1D | BH | mock, yaml `enable_channel_trimming` | 56/56, 20/20 | same | builds |
| All instrumentation together (telemetry + BW + profiling + capture), 2D / 1D / ring / VC1 | BH | mock | 56, 20, 20, 44: all build | same | builds |
| VC2 (`TT_METAL_ENABLE_FABRIC_VC2=1`), 2D | BH | mock | **0/56 at each level** | 56/56 | **fails: new in v3** |
| Tensix mux (single ERISC), 2D and 1D | BH | mock | 28/28, 10/10 | same | builds |
| Single ERISC (`TT_METAL_DISABLE_FABRIC_TWO_ERISC=1`), 2D and 1D | BH | mock | 28/28, 10/10 | same | builds |
| VC1 (two ranks): two-ERISC, single-ERISC, with all instrumentation | BH | mock | 43, 22, 44: all build | same | builds |
| RISC data cache, 2D / 1D / VC1 | BH | synth | 56, 20, 43: all build | same | builds (never emitted by the host) |
| Spin-wait send and completion ack, 2D / 1D | BH | synth | 56, 20: all build | same | builds (never emitted) |
| UDM, 2D | BH | synth | 28/28 | same | builds |
| `fuse_receiver_flush_and_completion_ptr == 0`, 2D / 1D / VC1 | BH | synth | 0 at each level | 0 | **fails: inherited from main** (unsupported) |
| 1D line, ring, 2D mesh | WH | mock (T3K) | 12, 12, 28: all build | same | builds |
| All instrumentation, 1D and 2D | WH | mock | 12/12, 28/28 | same | builds |
| Tensix mux, 1D and 2D | WH | mock | 12/12, 28/28 | same | builds |
| VC1 (T3K 2x2 split; WH is single-ERISC) | WH | mock | 15/20, 16/20, **20/20** | same | -Os builds; -O3/-O2 overflow IRAM: **inherited** |
| `fuse == 0`, 1D | WH | synth | 0/12 | 0/12 | fails: inherited |
| Channel-trimming capture + replay (trimmed routers: 0–2 serviced sender channels, some speedy), 2D | BH | device log (job 271) | 120/120 at each level | 120/120 | builds |
| Channel-trimming replay, 1D | BH | device only | – | – | `cov_device_runs.sh` (ready, not run) |
| Speedy path on new types; Z / express (five-wide VC0) routers | BH, WH | – | – | – | not reachable here: speedy comes from trimming replay; Z needs a multi-galaxy topology |

**Failures and the smallest fixes.**
1. **VC2 (new in v3sb).**
   - The error is `too few arguments to function 'void teardown(...)' [-Wtemplate-body]`, at `receiver_erisc::run_kernel`'s
     call to `teardown`.
   - `run_kernel` is a template that VC2 builds never instantiate (`receiver_erisc::runs` is false there). GCC 15
     still checks its non-dependent calls, and VC2 builds give `teardown` a third tracker parameter.
   - **Fix:** guard `run_kernel`'s definition and its call in `kernel_main` with `runs`'s own condition,
     `#if defined(ARCH_BLACKHOLE) && !defined(FABRIC_2D_VC2_SERVICED)`. That is two `#if` lines. Tested on a scratch
     copy: VC2 builds at all levels, and the logged configs cannot change (only VC2 builds lose code). It belongs to
     the E1 restructuring of `run_kernel`.
2. **`fuse_receiver_flush_and_completion_ptr == 0` (inherited from main; not a supported config).**
   - `ct_args` `static_assert`s that the flag is 1, with the message "must be 0", which says the opposite. The builder
     never emits 0.
   - The generic receiver's non-fused branch (R:1722-1745) calls `is_caught_up_to` with two arguments and does not
     compile. Main has the same branch and call.
   - **Fix:** delete the dead branch and the flag (or fix the call and the message). This also settles a2's fuse row:
     receiver_erisc need not handle it.
3. **WH VC1 at -O3 / -O2 (inherited):** `.text` overflows `ERISC_APP_IRAM_CODE` on 5 of 20 (-O3) and 4 of 20 (-O2)
   configs. Production builds VC1 routers at -Os (`get_kernel_opt_level`), and -Os builds all 20. No fix needed. If WH
   VC1 at -O2 is ever wanted, the size has to come down.

**Device run for trimming replay.**
- The 2D replay builds come from job 271's log (`fabric_exp/results/cov_trim_2d.log`, 160 router configs: 80 capture,
  80 trimmed replay), ingested as `bh_2d_trimreplay`.
- `fabric_exp/consolidation/cov_device_runs.sh` now runs only the 1D suite: deterministic traffic, a 240 s timeout,
  and the stale profile removed first. `coverage.py ingest` takes its log afterwards.

**Trimmed-replay testing with random traffic hangs by design** (for the trimming track).
- `enable_channel_trimming` expands a test into a capture run and a replay run (`expand_channel_trimming`). The replay
  builds routers that service only the channels the capture saw.
- `full_device_random_pairing` (and the other random patterns) draw from the config builder's generator, which
  advances per expanded test (`tt_fabric_test_config.cpp:1522`, `:2208`). So the replay pairs devices differently
  from the capture. In job 271 it sent on channels the capture had trimmed (152 of 160 sender channels removed) and
  hung at "Waiting for programs", wedging the device.
- Capture and replay must use deterministic traffic: `all_to_all`, `neighbor_exchange` or `unidirectional_linear`,
  with `size` and `num_packets` set. Neither `--master-seed` nor a per-test seed helps: one generator serves both runs
  in sequence.
- A trimmed router that receives traffic for a trimmed channel waits forever. This is worth a host-side check or a
  router-side ASSERT under watcher.
