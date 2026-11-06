# Device code: study router (v4txa11) against the mainline port (22cdb1e55c3)

2026-11-04. Offline only: no device jobs, no router source edits in any tree. A cause is a hypothesis unless its row
names an isolating A/B. Host decisions (slot counts, transaction IDs, flags, opt levels) are in RECONCILE_HOST.md.

**Sides**

- Study: /home/aagarwal/tt-metal (aagarwal/fabric-router-study at e569c5deb20 plus uncommitted changes) with the
  fabric_exp/variants/v4txa11 overlay (23 files; the kernel is fabric_erisc_router_v4.cpp in the v2 slot). Main base
  6d8a1032acc.
- Port: 22cdb1e55c3 (local aagarwal/fabric-router-mainline-cleanup; pushed origin/aagarwal/fabric-router-mainline is
  7cd3c13a0db): main d2bd12fcf1f plus 33 commits. Main moved 391 commits between the two bases. Later cleanup commits
  (3860614b218 onward) are not covered.

**Method** (scripts and data in reconcile_device/)

- Include closure. The study's comes from the .d files of its v4txa11_O3 coverage build: 158 files over 83 cases.
  inc_scan.py also scans `#include` statically on both sides: 176 files for the study (a superset of the .d closure),
  181 for the port. 42 files differ in content (differ_files.txt), and 10 are reached by one side only (§6).
- Every diff hunk (hunks.py, giving hunks.json and hunks_code.txt): 407 hunks, of which 89 are comment-only, 36 are
  format-only and 282 change code. Each hunk's origin comes from git blame of its port lines at 22cdb1e55c3 (a port
  commit or a main PR). The study's lines are also checked against main at 6d8a1032acc: present there means main's
  old code, absent means a study change.
- Image facts come from each port commit's own offline gate (named per row) unless the row says otherwise.
  - One new offline check compares the route decode site in both sides' bh_2d receiver images (row P1).
    - Tools: decode_sites.py, and rebuild_port.py, which rebuilds the port's E1 images with debug info from the
      c5 coverage case's logged commands.
    - The port side was built against /home/aagarwal/tt-metal-mainline, which is clean at 22cdb1e55c3 (read only).
    - Outputs: decode_port_bh_2d.txt, decode_study_bh_2d.txt.
  - No other study-to-port image comparison was made. The two hosts pass different arguments (slot table, transaction
    IDs, bubble threshold), so whole-loop counts would mix host and device differences.
- Origin: (a) main's newer code, with its PR; (b) a port commit, with its sha; (c) a study change the port does not
  carry, with the MAINLINE_PLAN section where it is listed; (d) unexplained.
- Effect: naming (comments, names or format; identical images where a gate says so); codegen (same behavior, the code
  may differ); behavior.
- Evidence: port (measured on a port build or a port-side A/B, with the job); study (LEDGER / RESULTS §); gate (an
  offline image identity check); none.

## 0. Summary

44 rows, plus one naming line per file in §5. By origin: (a) 12, (b) 22, (c) 10, (d) 0.

Per-packet rows, in order of action:

1. **P1, the 2D route decode: A/B on the port.** The port runs main's branchy `decode_action_y_first`; the study reads
   both bytes. Offline, on bh_2d at -O3:
   - The port's N/S receivers load the X byte after the branch on the Y byte.
   - The study's load both bytes back to back.
   - The v3 A/B on this shape (LEDGER E1-18 / E1-19) cost N/S rows +5.7 to +6.7 cycles per packet.
   - It is unmeasured on v4 and on the port.
2. **P7, the cold multicast landing (cb42eb72501): A/B candidate.** The commit changes receiver::run's register
   allocation on 44 Galaxy mesh pass-through receivers without a source change. Unmeasured on device.
3. **Keep the port's:**
   - Measured on the port: P2 (bubble injection threshold), P3 (read groups dropped), P4 (pins dropped), P5 (worker
     credit preset whole).
   - Validated, no perf row isolates it: P6 (inline register write on the atomics buffer).
   - Identical images: P11, P12.
4. **Same code or behavior for v4txa11 configurations, by reading or by gate:** P8, P9, P10, P13, P14, P15. No action
   unless the user wants an offline identity check.

## 1. Rows that can move per-packet cycles

| # | Where: study / port | What differs | Origin | Effect | Evidence | Disposition |
| --- | --- | --- | --- | --- | --- | --- |
| P1 | Study fabric_common.h:278-283 (`decode_action_y_first`), :291-299, router_routing.hpp:86 (`Routing2DCodec::decode_action`). Port fabric_common.h:356-363, :371-381 (`decode_action<MY_DIR, EXPRESS_AXIS>`), :484-486 (`Fabric2DRoutingConfig::decode_action`), router_routing.hpp:86 | **Study:** loads the Y and X bytes unconditionally, then selects. **Port:** loads Y, branches, and loads X only when Y is 0. **Offline check** (bh_2d -O3; receivers that decode Y first: port 16 of 40, study 12 of 28; one sequence per side in all of them): **port** `add; lbu Y; zext.b; bnez; add; lbu X; zext.b`, so the X load issues after the branch on the Y load; **study** `add; add; lbu Y; lbu X; zext.b; beqz; zext.b`, both loads back to back. **Net:** the port loads one byte fewer when Y != 0 and chains the two loads when Y = 0. | (c): a study change, reverted on purpose (MAINLINE_PLAN §1.5; that row's "unattributed" is marked WRONG). The port's code is main's (#57785 shape; #57264 adds EXPRESS_AXIS, with the same logic when express is off). | Codegen, per packet on N/S receivers, and on Z receivers unless the express axis is X. | Study (v3): LEDGER E1-18 / E1-19 (RESULTS §98). v3vd (X load behind the Y test) cost NS_PureAtomic +6.7 and N/S write 16 B +5.7; v3ve (both bytes) removed it. None on v4 or the port; on the port the cost is a hypothesis. **Measured on the port 2026-11-05 (lb2 176226, SLOT_ALLOCATOR §12.13):** both bytes make the 2x4 mesh N/S rows faster (atomic -7.6, scatter -7.5, writes -1.0 to -1.7 cycles/packet; isolating) and the 4x2 torus N/S rows faster (-10.0 to -11.9; codegen confounded, bound about 2 cycles). | **A/B on the port** (§7 A/B-1). |
| P2 | Study router_sender.hpp:76-81; ct_args :638. Port router_sender.hpp:76-83, :571-577; ct_args :652-656 | An injection channel sends while the far VC0 receiver has at least 2 free slots (study), or max(2, R − 16) from the host arg `BUBBLE_FLOW_CONTROL_INJECTION_SENDER_CHANNEL_MIN_FREE_SLOTS` (port; static_assert keeps it between 2 and R). | (b) df64d0ad677 | Behavior, only on rings with 32-slot receivers: 170 images change one compare immediate (2 to 16); 5,345 are identical. | Port: full-ring multicast 8-ring 4352 B 375 to 254; 4-ring and half ring −21 to −24%; P150 ring rows at most +0.9% (Galaxy 172983, P150 758 / 759; PARITY "Bubble-flow-control"). | Keep the port's. Host half: RECONCILE_HOST. |
| P3 | Study router_sender.hpp:504 / :507 (the pass's first reads), :543 / :546 (the send's first reads), router_receiver.hpp:743 / :750 (the receiver pass's first reads). Port: removed (router_sender.hpp:536-538, :580-582; router_receiver.hpp:746-753) | Three `begin_read_group` / `end_read_group` pairs dropped. Both sides keep the send's second group, the visit counter group and the downstream credit re-read group. | (b) 993c99a564b, aa11a8786df, 82ea91f44a4 | Codegen: scheduling; on one ERISC the receiver pass is no longer inlined. | Port: per-site arms 779 / 780 / 785 / 786 / 788; joint arm drop5 789: no row slower. 2D and torus Y 3-hop −5.8 / −3.0, one-ERISC 2D −3.9..−4.0, one-ERISC multicast −5.0 (TODO "Placement-primitive audit"; bfc/audit/drop5_F.txt). Gate: each commit changes only its own site's functions (5,010 images). | Keep the port's. |
| P4 | Study router_sender.hpp:776 (`keep_in_memory(send_state)` after a later VC's visit), router_receiver.hpp:696-703, :725 (`after_step`). Port: removed (:814; :707, :728) | Two memory pins on a later VC's state dropped. Both sides keep the channel-state pin after a visit. | (b) 5fb378a7880, bc0108869e2 | Codegen, VC1 / VC2 routers only. | Port: site 816 had no effect on one-ERISC VC2 rows; site 713: one-ERISC VC2 −1.4..−2.0, two-ERISC VC2 1-hop −0.5..−0.7. Both are inside drop5 (789). | Keep the port's. |
| P5 | Study router_sender.hpp:384-416. Port :400-450 | **Port:** a worker channel's credit buffer is preset with the whole 4-byte posted write at connect (again after base firmware); a credit stores the count into the read counter and sends the preset command. **Study:** rewrites source and destination per credit. Without presets (one ERISC, DPRINT, watcher) both sides send a whole command. | (b) 7cd3c13a0db | Behavior (what is programmed per credit; the same write) and codegen: 2,466 E0 images, the credit from 19 to 13 instructions. | Port 768: 1-hop unidirectional 4352 B −0.5 (1D) / −0.4 (2D), other rows 0.0..−0.2. Validation 764-767, 775-776. | Keep the port's. |
| P6 | Study router_local_delivery.hpp:90-100 (`noc_inline_dw_write<…, emulated_write_on_cmd_buf=true>(…, cmd_buf)`), dataflow_api.h:2222-2237, dataflow_api_raw.h:360-398, BH noc_nonblocking_api.h:1070-1244. Port router_local_delivery.hpp:90-99 (the stock `noc_inline_dw_write`, plus a static_assert on the local-write buffer); the API files are main's again | An inline write to a register goes out on the atomics command buffer (as the stock call issues register writes), not on the local-write buffer. L1 inline writes are unchanged (NCRISC_WR_CMD_BUF on both sides). Both restore the local-write preset afterwards. | (b) d2343171a56 (removes the study forms that acc4656caaf had added) | Behavior (command buffer) in deliver_other, which is out of line: 2,488 receiver and single-ERISC images, same instruction count. | Port validation: inline / UDM 659, 724, 765 / 776. No perf row isolates it. | Keep the port's. |
| P7 | Study tt_fabric_api.h:236-260 (inline in `fabric_set_2d_intermesh_landing_route`). Port tt_fabric_api.h:196-230 (`fabric_set_2d_intermesh_landing_mcast_route`, noinline cold), :269 | The multicast landing's route rebuild moves out of line and is marked cold. | (b) cb42eb72501 | Codegen. 227 inter-mesh ingress E1 images change in the landing functions and `_start`. 44 Galaxy mesh pass-through E1 images also change receiver::run's register allocation without a source change: frame 240 to 224 B, stack loads and stores −2..−6, length −4 / −3 / +15 instructions (cause unknown; not IPA-RA). | Port: the unicast landing A/B 726 / 727 (study arm v4txa11land against v4txa11) was within 0.4 on every row. Multicast landings and the 44 pass-through receivers are unmeasured (no traffic for them on the P150). The commit is needed for the -O3 fit (2x8 pass-through: 736 B left). | **A/B candidate** on Galaxy pass-through rows if those rows matter (§7 A/B-2); otherwise keep. |
| P8 | Study fabric_edm_packet_header.hpp:437-457. Port :431-463 | Both routers use `load_word` / `unpack`. **Study:** `PackedPayloadAndSendType` gains `src_ch_id`, and `load` = `unpack(load_word)`. **Port:** keeps main's `load` and returns a separate `Unpacked` struct from `unpack`. | (b) a9150295798 | Codegen at most: the same load and fields by reading. Not compared by image. | Both sides keep the early header word. Removing it cost E/W scatter +6.8, ring1 +5.9, Y-torus N/S +6.4 (A3, P150 717-720, study arms; PARITY "NoC API against main"). | Keep the port's. |
| P9 | Study fabric_edm_packet_transmission.hpp:96-118 (one-check flush when `rx_batch_completion`), ct_args :822-829, kernel :18-21 (`#define FABRIC_ERISC_ROUTER_V4`). Port :96-118 (template `all_writes_counted`), router_local_delivery.hpp:85 (`<true>`) | **Study:** a kernel define selects the Blackhole one-check flush for every caller. **Port:** only the router's local delivery passes `<true>`. The other callers (the seminc / scatter arms of `execute_chip_unicast_to_local_chip_impl`) are not reached by v4txa11: deliver_other reaches that function only for sparse multicast, which has no flush. | (b) a9150295798, f4779d68dc5. The study's define path is (c), MAINLINE_PLAN §2.4. | The same flush code runs, by reading. | None (by reading). | Keep the port's. |
| P10 | Study fabric_edm_packet_header.hpp:252-257 (`NocFlushType`), uint8_t flush fields :265, :272, :296, :301, :305-308, :314, scatter re-encoding :789-794; packet_transmission :354-369, :535-566 (`execute_local_semaphore_inc`). Port: main's `bool flush` and the inline flush + `noc_semaphore_inc` | The relaxed (deferred) flush of v2 / v3. The v4txa11 local delivery tests flush as nonzero on both sides, so packets with flush 0 or 1 behave the same. FLUSH_RELAXED (2) is set only by the study's worker API (api_common.h). | (c), MAINLINE_PLAN §1.3, §1.5, §2.1. The port's lines are main's (#37751, #31016, #19765, #19877, #21450). | Naming for v4txa11 traffic; behavior only for FLUSH_RELAXED senders, which the port has none of. | None (by reading). | Keep the port's. |
| P11 | Study router_downstream.hpp:315-327. Port :318-339 | **Study → port:** `noc_write_to_node_with_state` becomes `noc_wwrite_with_state<update_addr_mid=false>`; `noc_inline_dw_write_to_node_with_state` becomes `noc_inline_dw_write_with_state`, which now takes both address halves plus `dst_noc_addr`. | (b) 25d65133347, 288afd237a7 | Naming: images identical (4,816 and 5,007 builds). | Gate. A1 / A2 / A5 (P150 717-720, Galaxy 172435) showed that main's stock forms cost cycles or lose credits. | Keep the port's. |
| P12 | Study router_downstream.hpp:91-137, :275-296; router_common.hpp:148-151, :242-246; router_receiver.hpp:827-829. Port router_downstream.hpp:90-137, :279-298; router_common.hpp:148-164, :254-269; router_receiver.hpp:829-838 | Forward presets per pair of command buffers (`vcs_on_forward_pair`); `forwards_have_one_destination` is a fold over the VCs; `downstream_name`. | (b) 3a9486d6da0 | Naming on every host configuration today (4,480 images identical). Behavior only if a later VC forwards on buffers of its own (no such host configuration). | Gate. | Keep the port's. |
| P13 | Port fabric_common.h:195-198, :215-234, :254-277, :304-335, :366-381, :455-501; tt_fabric_api.h:301-309; fabric_2d_route_interface.h:80-87, :163-183 | Express axis. Z routers decode X only when the express axis is X; the Z landing writes the X byte; `widen_packed_byte<Axis>` drops Z on the axis without chords. Without express, EXPRESS_AXIS defaults to Y and the logic is the plain-mesh decode. | (a) #57264, #63452 | Behavior only with FABRIC_EXPRESS_ENABLED, which no study configuration has. | Port coverage row 12 builds (PARITY). No device run. | Keep the port's. |
| P14 | Study ct_args :832-840 (chosen in the kernel; 16 on a lone receiver channel without bubble flow control under `rx_batch_completion`, else 8 / 4). Port ct_args :375-384, :841-842 (host arg `NUM_TRANSACTION_IDS`) | Where the transaction-ID count is chosen. The port's host applies the same rule (f4779d68dc5 message). It also stops with a TT_FATAL for VC1 + VC2 on ring or torus routers (d7a9ced282f). | (b) 400d5bd0473, f4779d68dc5 | Naming if the values match per configuration. | Host check: RECONCILE_HOST. | Keep the port's. |
| P15 | 5206b9a57da's naming edits across router_*.hpp | In 14 channel-trimming capture receiver images (bh_2d_trim 6, bh_trim_gtest 8), the route decode's register allocation moves: receiver::run +12 B, three instructions. No route-decode source changes. | (b) 5206b9a57da | Codegen, capture builds only. The other 4,466 images are identical. | Gate. | Keep the port's. |

The coordinator described TRID_ALLOCATION.md §6 as finding "a per-packet spill inside that decode on the study tree".
The spill is real but needs precision: it appears in the study tree's `id` arm, not in v4txa11. That arm uses
per-slot transaction IDs, and on its 4x2 N/S receivers one chip coordinate the decode reads is reloaded from the stack
on every packet (`lw a4,12(sp)` at fabric_common.h:281). It shows that the decode site is sensitive to register
pressure (hypothesis C, untested). It is not evidence for or against P1's shape.

## 2. Other rows that change code or behavior (off the per-packet path)

| # | Where: study / port | What differs | Origin | Effect | Evidence | Disposition |
| --- | --- | --- | --- | --- | --- | --- |
| B1 | Port router_init.hpp:396 (`teardown`), router_round.hpp:156 (`run_router_command`); router_sender.hpp study :1071 / port :1103 (`wait_for_connect_request`) | Cold attribute on three control paths. | (b) 3b4ccab4dd7 | Codegen: those functions, `_start` immediates and layout. | Gate (PARITY "CI multi-mesh program sizes"). Needed for the -O3 fit. | Keep the port's. |
| B2 | Port router_round.hpp:18-20, :120-125 | The streaming profiler's link clock sync. ERISC 0 of a synced link steps the sync port, checking every 16th round. | (b) 58a9a1e1a72 | Behavior with PROFILE_STREAMING only; compiled out otherwise (images unchanged). | Gate. | Keep the port's. |
| B3 | Study router_sender.hpp:1033. Port :1063-1065 | Sender stream registers are initialized for the router's own channels only. | (b) f4779d68dc5 | Init only. Fixes a compile error on mesh routers next to a Z-port boundary router. | Port coverage row 9 (PARITY). | Keep the port's. |
| B4 | Study router_common.hpp:281-286, :303. Port :306-308, :326 | VC2 accepted with or without VC1. | (b) f4779d68dc5 | Configuration acceptance. | Study probe v4txa11v (570-572); port 674 (vc1_vc2 at -Os, 12/12). | Keep the port's. The -Os rule for VC1 + VC2 is host (RECONCILE_HOST). |
| B5 | Study fabric_erisc_datamover_channels.hpp:509-539. Port :479-509 | `EdmChannelWorkerConnection` has default member initializers and brace construction. | (b) f4779d68dc5 | Construction only. | None. | Keep the port's. |
| B6 | Study router_utils.hpp:122-125, router_sender.hpp:577. Port: removed | `order_point` and its one use. | (b) 3cbc209a497 | Naming: 5,010 images identical. | Gate. | Keep the port's. |
| B7 | ct_args port :470-471; router_init.hpp study :149 (removed), :247-248 / port :251-258 | Helpers `mask_width` / `lowest_bit`, `reinterpret_cast`; the downstream mask is cut to `max_downstreams` bits (was 0xF); the receiver TX-queue static_assert is removed; the keepalive comes through the ct args. | (b) e44ee14e634 | Naming: images unchanged. | Gate. | Keep the port's. |
| B8 | ct_args study :226-227. Port :226 | `skip_src_ch_id_update`: the study reads the host's SKIP_SRC_CH_ID_UPDATE, the port reads FABRIC_TENSIX_EXTENSION_MUX_MODE. The host gives both the same value. | (b) bd290f3246c | Naming: images identical. | Gate. | Keep the port's. |
| B9 | ct_args study :235-236 / port :234-238; router_receiver.hpp:126 / :127 | `is_vc_serviced` is renamed `is_vc_configured`. | (b) 4db4f45790b | Naming: images identical. | Gate (4,816 builds). | Keep the port's. |
| B10 | Study router_common.hpp:74-81, router_init.hpp:87, :90, router_round.hpp:130 (literals). Port router_common.hpp:74-79, router_init.hpp:87, :90, router_round.hpp:143 (`FABRIC_KERNEL_HEARTBEAT_*`) | Heartbeat address, period and signature. Same values: 0x7CC70, 64, 0xDCBA0000. | (b) fe645101fcd, with constants from (a) #59086 | Naming: images unchanged. | Gate. | Keep the port's. |

## 3. Study-only code the port does not carry (dead for v4txa11)

| # | Where (study) | What | Origin | Effect | Disposition |
| --- | --- | --- | --- | --- | --- |
| C1 | fabric_edm_packet_transmission.hpp:108-110, :152-287, :298-299, :336-480 (DRISC arms), :701-731 | DRISC arms, deferred atomic flush helpers, `drisc_*` raw NoC sequences, the `pending_atomic_flush` and `update_local_free_slots` parameters. The relaxed-flush pieces are P10. | (c) §2.1 | None for v4txa11 (not called). | Keep the port's. |
| C2 | fabric_erisc_datamover_channels.hpp:115-117, :138-152, :458-472 | v3 stride members and `advance_to_next_cached_buffer_slot_addr_by_stride`; STABILITY_PROBE. | (c) §2.2 | None. | Keep the port's. |
| C3 | fabric_router_flow_control.hpp:87-92, :185-195, :221-225 | Stream-register credit twins; EXP_TX_LEAN_COMPLETIONS. | (c) §2.3 | None (v4 static_asserts counter credits). | Keep the port's. |
| C4 | fabric_router_adapter.hpp:113-425 (27 hunks) | DRISC and `update_local_free_slots`. The port's lines are main's (#28068, #33222, #57785). | (c) §1.5 | None: v4txa11 does not use the adapter. | Keep the port's. |
| C5 | edm_fabric_flow_control_helpers.hpp:231-270; fabric_erisc_router_transaction_id_tracker.hpp:100-107 | EXP_RX_GROUPED_READS. | (c) §1.5 | None. | Keep the port's. |
| C6 | edm_fabric_utils.hpp:54-66 | Blackhole fast path in `send_chunk_from_address_with_trid`. Only the adapter uses it. | (c), not listed in MAINLINE_PLAN | None for v4txa11. | Keep the port's. |
| C7 | fabric_stream_regs.hpp:9-11, :98-109; tt_fabric_api.h:11-13 | COMPILE_FOR_DRISC guards. | (c) §1.5 (goes with the DRISC kernel) | None. | Keep the port's. |
| C8 | dataflow_api.h:2395-2417, :2453-2479; dataflow_api_raw.h:390-396; BH noc_nonblocking_api.h:1322-1380, :1957-1997, :2359-2388; WH noc_nonblocking_api.h:1012-1065, :1558-1598, :1940-1960 | Study-only atomic and semaphore set / with_state forms, and the node-addressed forms that P6 and P11 replaced. | (c) §1.3 | None: v4txa11 calls only the forms in P6 / P11. | Keep the port's. |

## 4. Main's newer code off the router's per-packet path

All are (a). None changes what the v4 router runs per packet, by reading. Disposition for every row: keep the port's,
since it is main's.

| # | Port file:lines | PR | What |
| --- | --- | --- | --- |
| A1 | fabric_common.h:26-33, :42-58, :150-156, :304-335 | #59086 | `CoordinatedEriscContextSwitchState`, the `HopAction` enum, multicast tree edge constants, `FABRIC_KERNEL_HEARTBEAT_*` (used by B10). |
| A2 | fabric_edm_packet_header.hpp:184-187, :195; packet_transmission :143-145, :314-315 | #59086 | `NOC_SCATTER_CHUNK_ENCODING_BITS` / `_MASK` (2 / 0b11). The port router uses the mask (router_local_delivery.hpp:160) where the study had a local `CHUNK_ENCODING_MASK` (study :131-132, :156). Same value. |
| A3 | edm_handshake.hpp:10, :45 (study :45-56 moved); fabric_connection_interface.hpp:14-18; new edm_handshake_types.hpp | #59086 | Handshake struct moved into its own header; constants became an enum. |
| A4 | eth_fw_api.h:87-117, :125-127, :375-376 | #59626 | Firmware-stage breadcrumb in `eth_status_t` (spare words 24 to 22). Firmware writes it; the router headers do not. |
| A5 | dev_mem_map.h:67, :119-124; dev_msgs.h:388-406 | #61568 | BRISC firmware size and an L1 ECC halt slot (Tensix L1 map). |
| A6 | device_print_structures.h:69-70, :81 | #62697 | DPRINT buffer bit fields (in the router only with DPRINT). |
| A7 | tensor_accessor.h:130-139, :718-723; tensor_binding_token.h:118-174 | #61952 | Per-node tensor binding token (unused by the router). |
| A8 | api/debug/ring_buffer.h:23, :29-52 | #62910 | Watcher ring buffer saturating semaphore (watcher builds only). |
| A9 | api/debug/kernel_profiler.hpp:642-661, :741-745 | #63524 | Device profiler room checks (profiler builds only). |
| A10 | api/kernel_thread_globals.h:103-128 | #60188 | Tensix global semaphore helpers (unused by the router). |
| A11 | hw/inc/api/circular_buffer_constants.h (new), circular_buffer_interface.h | #62814 | Constants split into host-only and device-only headers. |

## 5. Naming, comment and format hunks, one line per file

Counts are hunks: comment-only / format-only. Rows named after a file cover its other hunks.

- Kernel (study fabric_erisc_router_v2.cpp ← v4 / port fabric_erisc_router.cpp): 1 / 0. 80ec66916c8 restores main's
  162-line design-notes block. The file name and slot follow MAINLINE_PLAN §1.6. The define is in P9.
- router_common.hpp: 3 / 3. 5206b9a57da names; the rest is P12, B4, B10.
- router_utils.hpp: 1 / 1. e44ee14e634 / f4779d68dc5 helpers; B6.
- router_downstream.hpp: 11 / 8. 5206b9a57da (`compact_index_on`), f4779d68dc5 renames; the rest is P11, P12.
- router_receiver.hpp: 11 / 3. 5206b9a57da (`receiver_vcs` loop, the relay's named VC), e44ee14e634; the rest is P3,
  P4, P12, B9.
- router_sender.hpp: 20 / 6. 5206b9a57da (field widths of the progress word), e44ee14e634, f4779d68dc5 renames; the
  rest is P2, P3, P4, P5, B1, B3, B6.
- router_init.hpp: 3 / 6. 5206b9a57da (TX-queue names), f4779d68dc5 casts; the rest is B1, B7, B10.
- router_round.hpp: 3 / 1. The rest is B1, B2, B10.
- router_local_delivery.hpp: 2 / 1 (c7a4d0fb29f: what deliver_other delivers). The rest is P6, P9, A2.
- router_routing.hpp: 2 / 0. Routing1D's `forward_to` uses `downstream_name` (3a9486d6da0). The rest is P1.
- router_single.hpp: 2 / 1. router_telemetry.hpp: 5 / 4 (e44ee14e634 names SENDER / RECEIVER). Nothing else.
- fabric_erisc_router_ct_args.hpp: 1 / 2. The rest is P2, P14, B7, B8, B9.
- fabric_erisc_datamover_channels.hpp: 4 / 0. The rest is B5, C2.
- fabric_router_flow_control.hpp: 2 / 0. The rest is C3.
- tt_fabric_api.h: 2 / 0. The rest is P7, P13, C7.
- fabric_common.h: 2 / 0. The rest is P1, P13, A1.
- fabric_edm_packet_header.hpp: 1 / 0. The rest is P8, P10, A2.
- dataflow_api.h: 4 / 0. BH noc_nonblocking_api.h: 1 / 0. The rest is P6, P11, C8.
- kernel_thread_globals.h: 4 / 0. circular_buffer_constants.h: 4 / 0. Main's comments (A10, A11).

## 6. Files only one side's router includes

- Study only:
  - fabric_erisc_router_v2.cpp (the study's kernel slot);
  - api/tt-metalium/circular_buffer_constants.h (moved by #62814; A11);
  - hostdev/profiler_zone_id.h (renamed by #63392).
- Port only:
  - fabric_erisc_router.cpp (the kernel);
  - hw/inc/api/circular_buffer_constants.h (#62814);
  - hostdev/observability_event_descriptor.h (#63392);
  - edm_fabric/edm_handshake_types.hpp (#59086; A3);
  - internal/ethernet/eth_ptp.hpp, streaming_profiler/kernels/eth_clock.hpp and link_sync.hpp (#58270, reached through
    B2's hook);
  - tt-2xx/quasar/noc/att/configs/quasar_aether_2x3_uncapped_att_config.h (#62723; Quasar only, not compiled for
    Blackhole).

## 7. Proposed A/Bs on the port

The design goes to the user before any arm is built (feedback-discuss-design-before-variant).

**A/B-1, the route decode (P1).**

- Arms: the port (22cdb1e55c3, or the cleanup tip at the time) against the port with `decode_action_y_first` reading
  both bytes, the study's three lines.
- Scope: one function in main's hostdevcommon/fabric_common.h, used by the router and by the intermesh landing.
- Offline first:
  - Check that only the receivers that decode Y first change at the decode site (decode_sites.py over a coverage
    build; this report shows both shapes).
  - Run loopvars `--headroom` for spills the change moves. The both-bytes form keeps two loaded values live at the
    site where TRID §6 saw a reload.
- Device rows:
  - E1-18's rows: N/S unicast write at 16 B and 4352 B, N/S atomic increment.
  - Galaxy N/S rows.
  - The core census, to see that no other row moves.
  - Passes reversed.
- Prediction (a hypothesis): N/S rows get faster if the chained load sits on the bounding ERISC's path. No one-side A/B
  has shown whether E1 bounds those rows on v4.
- Upstream: this changes main's #57785 shape, so the PR needs the numbers.

**A/B-2, the cold multicast landing's side effect (P7).** Only if Galaxy pass-through rates matter.

- Arms: the port against the port with the multicast landing inline again.
- Configurations: the 2x8 pass-through does not fit at -O3 without cb42eb72501, so only configurations that still fit
  can run the arm (for example G-PORT P10-mm-4x4).
- Rows: the 44 receivers' pass-through rows.

**Not proposed.**

- P8 (header struct shape) and P9 (flush selection) are the same code by reading. An offline identity check would
  need a port build with the study's form of each, which is cheap if wanted.
- No same-host census of v4txa11 against the port exists. Once this list and RECONCILE_HOST are settled, one such
  census would show whether anything beyond these rows moved.

## 8. The port's remaining write gap: static per-packet diff (2026-11-05, offline)

**Question.** After the both-byte decode (port 6dad5fec638), the allocator agent's same-host runs with matched slot
rows (lb2 175939 / 176226, pvs_runs.sh) still show a write gap:
- the port slower: 2x4 mesh N/S write 4352 B +0.9-1.0%, N/S write 16 B +2.8%, E/W write 16 B +3.0%;
- the port faster: 3-hop E/W write −3.0%, multicast write −1.8%;
- atomic and scatter rows match.

What follows is a static comparison only. Every difference is a candidate until a one-change A/B.

**Method** (tools and outputs in reconcile_device/write_gap/):
- **Port side.** A worktree at 6dad5fec638 (/home/aagarwal/tt-metal-port-gap), its own host built. A mock capture of
  the P150 2D mesh (cov.py, case bh_2d), and its 80 router images rebuilt with debug info.
- **Study side.** v4txa11's bh_2d coverage build (cov_m builds/v4txa11_O3).
- **Slot rows.** The port's tier row on these routers is the study's row, 17,16,16,16 / rx 16, read from the compile
  arguments by the slot probe. No override was needed.
- **Pairing** by router class (ERISC, direction, Ethernet channel, live VC0 downstreams, VC0 forwarding off/on): 45
  classes on both sides, 23 E0 and 22 E1. One class paired across different downstream variants (study 564 / port
  626 instructions) is a pairing artifact and is left out.
- **Compile inputs.** The paired builds' named compile-time args are identical except the study-only EXP_* /
  PHASE_PROFILE_* / PEER_* args (unread by these images) and the port's NUM_TRANSACTION_IDS (same value). The 40
  positional args are identical. The compile flags are identical except the port's `-DL1_ECC_HALT` (below).
- **Comparison.** Each run() function (sender::run on E0, receiver::run on E1) annotated with addr2line inline chains
  (annot.py) and diffed after normalizing registers and addresses (rundiff.py), then counted per path piece by inline
  chain (sites.py). Every out-of-line function was compared by size.

**Per-packet pieces** (instructions, L1 loads, stack loads, NoC command-register writes; study → port):

| ERISC, piece | write / forward rows run it | study | port | difference | RECONCILE row |
| --- | --- | --- | --- | --- | --- |
| E0 worker send (send<0u>) | the writer, every row | 66 i, 10 ld, 2 sp-ld, 0 cmd | 65 i, 10 ld, 2 sp-ld, 0 cmd | −1 instruction; **placement**: the study reaches the send through a taken branch to an out-of-line block and jumps back, the port falls through into it | P3 (aa11a8786df: no read group around the send's first reads) |
| E0 worker credit (WorkerChannel<0>::credit, out of line) | the writer, per credit event (a group, not a packet) | 19 i | 13 i | the credit command is preset whole; a credit is one store and one send request | P5 (7cd3c13a0db) |
| E0 forwarding sends ch1-3, visits, pass-start reads | the 3-hop forwarders' E0 | 61 / 61 / 62; 142; 28 | same | order only (two TX-queue status loads swap with a `lui`) | P3 |
| E1 local write (preset), plain write path | the destination | 42 i, 6 ld, 0 sp-ld, 14 cmd | identical, instruction for instruction | none | P6 / P9 not on this path |
| E1 forward (one / two downstreams) | the 3-hop forwarders' E1 | 52 / 96 i | identical | none | P11 / P12: none left |
| E1 route decode | every E1 | 3 (E/W) / 7 (N/S) i | identical | none (6dad5fec638's decode = the study's) | P1, closed |
| E1 admission, deliver dispatch, retire, completion proof, credit accumulate, credit frame | every E1 | | identical | none | P8, P9, P14: no effect on these images |
| E1 fused write + atomic (deliver line 179/183) | not a write row | 42 i | 42 i | order only: the port's local write updates the software write counters before the command's go, the study after | none (scheduling; same source) |
| `-DL1_ECC_HALT` (compile flag on every port kernel) | | absent | present | its only uses are in brisc.cc (Tensix BRISC firmware); no router header reads it | main #61568 |
| Out-of-line, off the packet path | | run_router_command 744 B, wait_for_connect_request 752 B, preset_credit 72 B | 400 / 580 / 96 B | cold attribute, smaller; preset at connect | B1, P5 |

**Reading.**
- On these 2D routers the router code the write rows run is identical on E1, at the destination and at the
  forwarders.
- On E0 it differs in two places only, both changes the port measured on its own:
  - aa11a8786df: 2D 3-hop −5.8 / −3.0 at 256 / 4352 B, other rows within 0.3 (placement audit, jobs 779-789);
  - 7cd3c13a0db: 1-hop 4352 B −0.5 / −0.4 (job 768).
- So the static router difference predicts the port faster on 3-hop. That fits the measured −3.0%, but only as a
  candidate: the port-side A/B was on another commit and host.
- It does not predict the port slower on any 1-hop row. **Nothing in the router code explains the 1-hop write rows
  being slower on the port.**

**Candidate outside the router: the test kernel.** pvs_runs.sh runs each arm with its own tree's test_tt_fabric.
- The study tree carries the streams work (fabric_exp/streams/port, streams_p3f.patch): a single-pattern benchmark
  sender runs the stream loop (`RUNS_STREAM_LOOP = STREAM_MODE || (BENCHMARK_MODE && NUM_TRAFFIC_CONFIGS == 1)`).
  That loop keeps a local copy of the connection, checks free slots in batches, and times only waits.
- The port tree has main's sender, `send_packets_stateful`. It reads the L1-resident connection per packet, and a
  free-slot check invalidates Blackhole's L1 cache.
- So on the 1-hop rows, where the worker feeding E0 is close to the bound, the two arms inject through different
  worker loops. They may also time them differently.
- The port's `-DL1_ECC_HALT` reaches the Tensix firmware the worker runs on, a second, smaller candidate on the same
  side.
- All of this is a hypothesis: the router code is equal; the worker and its firmware are not.

**Cheapest isolating A/B (for the user's decision; nothing submitted).** One lb2 job, both passes on one host:
- **arm 1:** 6dad5fec638 as built (main's test infra);
- **arm 2:** the same tree plus only the test-infra part of fabric_exp/port_tune/port_tune.patch (the files under
  tests/; router and host untouched, so router images are identical, checkable offline);
- **rows:** pvs_ns (E/W and N/S 1-hop writes at 16 and 4352 B) and pvs_c2d (3-hop, multicast). About 10 minutes.

If arm 2 lands on the study's numbers for the 1-hop writes, the gap is the test kernel, not the router. If not, the
next candidate is E0 send placement, which needs a variant.

### 8.1 Result: the quick test (a09u32 broker job 798, 2026-11-05)

The user approved a smaller test than the A/B above: only the sender kernel's single-config benchmark branch.
- **Arms** (fabric_exp/port_tune/gap_loop_ab.sh). Both use one host binary, 6dad5fec638's build.
  - **asbuilt:** main's sender, `send_packets_stateful`.
  - **loop:** that branch runs the study's one-segment stream-benchmark loop instead, copied from the study's
    `send_stream_benchmark`: a local copy of the connection, free-slot checks only when the known count runs out,
    header replay by the credit-only update, the slot table and flat lap, and the copy written back at the end.
    It adds no new compile or runtime args (stream_repeat = num_packets, count 1) and leaves out the stream-counter
    write. It is uncommitted in /home/aagarwal/tt-metal-port-gap.
  - Each arm has its own runtime root (/home/aagarwal/tt-metal-port-gap-roots/<arm>). The only file that differs is
    tt_fabric_test_sender.cpp.
- **Router images.** They are byte-identical between the arms by loadable bytes, checked in a mock build and again
  from the job's caches (fabric_exp/port_tune/elf_load_cmp.py).
- **Rows and order.** The 6dad5fec638 default slots are TX 17,16,16,16 and RX 16 (mock probe), the same as the
  study's row. Rows are pvs_ns. Pass a ran asbuilt then loop; pass b the reverse. One host.
- **Results** in fabric_exp/results/gaploop_20261105-023834/.

Cycles/packet. asbuilt and loop show passes a / b; "loop vs asbuilt" is the mean of the two passes. Study and
portdec are lb2 176226's means (a07u28).

| row | asbuilt a / b | loop a / b | loop vs asbuilt | study | portdec | loop vs study |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EW_Write 16 | 120.1 / 120.1 | 114.8 / 114.7 | -5.3 (-4.5%) | 116.7 | 120.1 | -1.7% |
| NS_Write 16 | 120.2 / 120.2 | 115.1 / 115.1 | -5.1 (-4.2%) | 117.0 | 120.3 | -1.7% |
| EW_Write 4352 | 128.2 / 128.2 | 126.8 / 126.9 | -1.3 (-1.0%) | 127.2 | 128.4 | -0.2% |
| NS_Write 4352 | 128.5 / 128.5 | 127.1 / 127.1 | -1.4 (-1.1%) | 127.4 | 128.6 | -0.3% |
| EW_Atomic 4352 | 130.2 / 130.2 | 129.9 / 129.9 | -0.3 (-0.2%) | 130.1 | 130.2 | -0.1% |
| NS_Atomic 4352 | 132.7 / 132.6 | 132.3 / 132.3 | -0.4 (-0.3%) | 132.3 | 132.7 | -0.0% |

**Reading.**
- **The 1-hop write gap to the study is the test sender kernel, not the router.** The A/B changes only the worker
  kernel, and that alone removes the gap. The atomics, the control, move 0.3-0.4.
- **With main's sender, these rows are bound by the worker that feeds E0.** That holds fully at 16 B and partly at
  4352 B. So §8's port-vs-study write comparisons, and SLOT_ALLOCATOR.md §12.13's "write rows keep the same gap",
  compared two different worker loops.
- **Repeatability.** Passes agree within 0.1. The asbuilt arm matches 176226's portdec within 0.2 on another host.
- **-DL1_ECC_HALT.** Both arms carry it; it is not needed to close the gap, and its own effect was not measured.
- **Hypothesis: loop vs study at 16 B.** The loop runs 1.7% faster than the study at 16 B. The cause could be the
  port's cheaper E0 send path (§8 table), or the port-built loop compiling differently from the study's sender. The
  comparison is cross-host and the two senders are not byte-identical. A same-host study arm would settle it.

### 8.2 The three senders' per-packet loops (static, 2026-11-05)

These are the device ELFs of the job 798 arms (its caches) and the study's sender. The study's sender was rebuilt from
176226's logged study commands against the study tree and its own pre-compiled firmware. The table counts the
steady-state packet: the slot already holds this config's header, so the packet is a credit-only update.

| sender | instructions / packet | L1 loads / stores (NoC issue counters, the same in all three, left out) | L1 cache invalidation (fence) | NoC issue |
| --- | ---: | --- | --- | --- |
| port asbuilt (main's `send_packets_stateful`) | 50 | 14 / 4: the connection is re-read from L1, and its slot index, slot address and write counter are written back, every packet | every packet (`wait_for_empty_write_slot`) | stateful credit write (1 go) |
| port loop | 25 | 3 / 0: the connection is a local copy; the loads are the slot-table byte and the lap entry (segment, header) | only when the known free slots run out | same |
| study (`send_stream_benchmark`) | 27 | as the port loop | as the port loop | same |

- **asbuilt vs loop** is the work job 798 removed: 25 instructions, all the connection's L1 traffic, and the fence
  on every packet. Measured: -5.1 to -5.4 cycles/packet at 16 B, -1.3 to -1.4 at 4352 B (§8.1).
- **The port loop and the study loop come from the same source.** The study's build has two more instructions per
  packet, both register allocation:
  - a stack reload of the buffer size for the slot address (`lw t1,36(sp)`; core-local, 1-2 cycles);
  - a rematerialized `li t4,1` before the credit go.
- **Hypothesis, not tested:** those 2 instructions explain the loop arm's -1.9 cycles against the study at 16 B
  (cross-host). If so, the 16 B rows are still bound by the worker, even with the stream loop.
  - A same-host study arm would settle the cross-host part.
  - Whether the loop arm is still worker-bound needs a one-side A/B, e.g. a fixed delay added to the worker loop.

## 9. deliver_other out of line or inlined (2026-11-05; a09u32 broker job 799)

**The change.** In router_local_delivery.hpp, deliver_other's `__attribute__((noinline))` became FORCE_INLINE.
Nothing else changed. Port 6dad5fec638. Scripts and tools are in fabric_exp/port_tune/dio/: dio_ab.sh,
dio_report.py, pair_images.py, cov_summary.py, dio_paths.py.

**Which images change.** Router-only rebuilds of 13 mocked configs (cov.py, 1,020 builds per arm):
- every two-ERISC E0 image is identical;
- every image that runs the receiver changes: two-ERISC E1 and one-ERISC images.
- Everything links at -O3 in both arms, so everything fits.
- Size changes run from -116 to +676 B. The largest image per class is in port_tune/dio/cov_summary.md:

| class | largest image, noinline -> inline |
| --- | --- |
| VC1 E1 | 10,020 -> 10,544 B |
| VC2 E1 | 7,808 -> 8,204 B |
| 2D E1 | 6,348 -> 6,380 B |
| Galaxy 2D E1 | 6,788 -> 6,892 B |
| Galaxy torus E1 | 6,676 -> 6,804 B |
| 1-ERISC torus | 11,152 -> 11,116 B |

The largest E0 is VC1 at 11,332 B, unchanged.

**Static per-packet delivery paths** (dio_paths.py). Each path runs from the type dispatch to retire, one per send
type. Instructions are listed with stack loads in brackets.

| path | noinline | inline, the device's 2D E1 | inline, other configs (2D, torus, VC1, VC2, Galaxy 2D) |
| --- | ---: | ---: | ---: |
| write | 28 [0] | 32 [0] | 33-35 [1-4] |
| atomic, no flush / flush | 42 / 46 [0] | 44 / 48 [0] | 43-46 / 49-50 [1-3] |
| fused write + atomic | 66 / 69 [0] | 65 / 68 [0] | 64-69 / 68-72 [1-5] |
| scatter, 2 chunks, last a write | 57-58 [0] | 61 [0] | 62-65 [1-3] |
| inline write | 53-65 [1-3 ld + 1-3 st], with call / ret | 45 / 53 [0-1] | 46-58 [1-4] |

- **The inlined dispatch is a jump table.** In every inline image the type dispatch becomes `li, bltu, sh2add, lw`
  (the table, in .data in local memory) then `add, jr`: one indirect jump per packet of every type. It replaces the
  compare chain, which costs a write 1 branch, a scatter 3 instructions, an atomic 5 and a fused 7.
- **New stack reloads on the common paths.** Outside the device's own config, the table base or offset is spilled and
  reloaded from the stack every packet, and more values spill on the inlined paths.
- **Outside delivery** (device config), inline rematerializes constants that noinline kept in registers: the stream
  register base in the packet poll and in retire, 0x01010101 in the completion-credit accumulate, and a TXQ constant
  in the credit frame.
- **What inlining removes on the inline-write path:** the call and ret, one caller-saved spill (t1: one stack store,
  one reload), the argument move, the callee's re-test of the type (li + beq), and four instructions after the call
  rematerializing a6 / a7. deliver_other is a leaf, so it had no prologue or epilogue. In total that is 11-12
  instructions fewer per inline write.

**Device** (fabric_exp/results/dio_20261105-031309/report.md): 2x4 mesh 1-hop E/W rows, both arms under main's test
sender and under the stream loop (job 798's).
- Two passes, arms reversed; one host. Passes agree within 0.1.
- Job checks: E0 images identical; E1 changed; test kernels identical.

Cycles/packet, inline minus noinline (mean of the two passes):

| row | main's sender | stream loop |
| --- | ---: | ---: |
| write 16 B | +0.49 (+0.4%) | +1.12 (+1.0%) |
| write 4352 B | +0.79 (+0.6%) | +0.37 (+0.3%) |
| atomic | +3.78 (+2.9%) | +3.69 (+2.8%) |
| scatter (2 chunks) | +5.61 (+3.9%) | +5.51 (+3.9%) |
| fused write + atomic | +2.60 (+1.7%) | +2.41 (+1.6%) |

- test_tt_fabric has no inline-write send type, so the device runs measure no inline-write row.

**Reading.**
- **Inlining deliver_other makes every common row slower** on this config, by 0.4 to 5.6 cycles/packet. Only E1's
  code changed, so these rows' rates depend on E1's per-packet cost.
- **Mechanism, hypothesis (not tested):** the jump-table dispatch (a local-memory load feeding an indirect jump on
  every packet), plus the constants rematerialized in the loop. The static path deltas (+2 to +4 instructions;
  fused -1) are smaller than the measured costs, so instruction counts alone do not account for the slowdown.
