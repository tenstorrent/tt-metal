# N/S fix 1: partial presets on forwards (design note)

Status (2026-11-03): **built as `variants/v4rxnsf2`** (the compile-time ForwardCommands design, §9; byte-identical to
v4rxnsf and to (a) v4rxnsa; validated 13/13 + inline/atomic + UDM, §9.2); the Galaxy N/S rows measured (§8.9: N/S
floor gone at 2 KiB slots, 172.9 -> 136.8). The sender consolidation (V4_SENDER_CONSOLIDATION.md) takes v4rxnsf2 as its
receiver side; router_downstream.hpp is frozen here. §0-§6 are the original design note (2026-11-02, before any run).
Labels: code = read from code; static = from the draft builds; meas = measured on a device; hyp = hypothesis.

**User decisions (2026-11-02, on §6; the drafts were rebuilt with them, §7):**
1. Bundle the NoC VC change (forwards on the link's EDM_NOC_VC) and the posted credits on buffer 1; attribute any gain
   to the bundle; isolating arms only if a result surprises.
2. Draft the two trims into (b) before the A/B, each readable (a named helper with a one-line why), else left out.
   Both are in (§1.2); being in the shared forward, (a)'s multi-downstream forwards have them too.
3. Accept the Y-torus delivery spill for the A/B (the torusns rows measure it).
4. The UDM relay stays stateless and unchanged: no partial branch. The drafts keep UDM routers wholly as today (no
   presets), so the relay's stateless writes never meet a preset buffer.
5. The additive NoC API is acceptable, documented next to the existing stateful helpers, variant copies only, flagged
   for upstream review (§1.1).
The (a)/(b) rule: (b) if it costs at most 1-2 cycles/packet on the single-downstream decision rows; if more, the
numbers go to the user, who decides (no fallback to (a) without that).

Background: V4_NS_FLOOR.md §2-§3 (N/S routers forward statelessly; an E/W forwarder made stateless rises by ~37 E1
cycles/packet, meas), V4_BOUNDS.md §4 (the command-buffer audit), V4_RECEIVER_OWNERSHIP.md "Planned: N/S floor stages"
(the fix as planned). The route preload (N/S fix 2) is parked: V4_ROUTE_PRELOAD.md §8 (draft v4rxrp3 kept).

The user's direction (2026-11-02):
- Data on command buffer 2, partially preset: the invariant fields preset; per forward only source, destination
  address, coordinate, length and go.
- Credit on its own preset buffer 1 (control, AT_DATA -1, AT_LEN_BE, TARG_ADDR_MID preset); per forward only
  TARG_ADDR_LO, TARG_ADDR_COORDINATE and go. Its ready read pairs with the data buffer's.
- Where an API lacks a needed form: the smallest additive API, in the variant's copy only.
- (a) partial presets only on routers with several live downstreams (single-downstream routers keep full presets), or
  (b) partial presets on every forward. "Draft (b); the A/B decides. Take (b) if its E/W cost is at most 1-2
  cycles/packet, else (a)."
- Must stay intact: the buffer-2 fix (c) (VC0 and VC1 share buffers 1-2 on N/S routers), buffer 0 for local writes with
  the inline-write fix (ii'), buffer 3's firmware-preset atomic return address (never used for forwards), the UDM
  relay forwards, teardown, the one-ERISC router.

## 0. Summary

- **Drafts.** `variants/v4rxnsb` = (b), `variants/v4rxnsa` = (a), both v4rxg3 + one shared forward helper + one additive
  API (§1). `variants/v4rxnsbml` = (b) on the multi-downstream proxy v4rxg3ml (every 2D router's entries live; the
  proxy shape is the same under (a) and (b)). Stage scripts reproduce each draft from v4rxg3 (checked by diff).
- **Files.** Only `router/router_downstream.hpp` and the pinned API headers (`noc_nonblocking_api.h` Blackhole and
  Wormhole, `dataflow_api.h`), in the variants' copies. Not router_common.hpp, router_single.hpp or router_sender.hpp
  (the sender agent's files); not E0: every E0 image is byte-identical to the lead in both drafts (static).
- **Per forward (static, -O3, executed path when both buffers are ready):**
  - today's stateless forward (N/S, edge, intermesh, every VC1 forward): 53 instructions, 14 command-buffer register
    writes, 2 serial ready polls, 3 counter stores;
  - today's full preset (1D, E/W inside a mesh): 23-24 instructions, 6 register writes (26 / 7 with deadlock
    avoidance);
  - partial preset (both drafts): 27-28 instructions, 9 register writes, 2 paired ready reads, 1 counter store.
  - So on multi-downstream routers both drafts remove 26 instructions and 5 register writes per forward; on
    single-downstream routers (b) adds 4 instructions and 3 register writes per forward (on deadlock-avoiding rings
    and tori, +1 instruction and +2 register writes). (a) adds nothing there: those images are byte-identical to the
    lead.
- **Identity (static).** (a) changes exactly the E1 images whose forwards are stateless today (394 of 2,863 at -O3
  and at -Os). (b) changes those 394 identically (byte-identical to (a)) plus the 610 single-downstream forwarding
  E1 images (1,004). One-ERISC, UDM (one ERISC on Blackhole) and every E0 image are byte-identical in both.
- **Guard.** The first draft held the credit buffer's ready bit across the data command, as the lead's E/W forward
  does; that bit spilled on the Galaxy express and torus routers (headroom check failed on 52 images, §3.3). The drafts
  now test both ready bits together before either command: every Galaxy family passes, and loops shrink. One
  production family still fails: the P150 Y-torus N/S routers (4 of 12 bh_torus E1 images), where two local-delivery
  values move to the stack (§3.3, §6 Q3). At -Os the fix removes 10-28 stack accesses per multi-downstream loop
  (today's stateless forwards spill there).
- **Predictions** (§4): the Galaxy N/S rows and the multi-downstream proxy are the rows to gain (counted work -26
  instructions per forward; cycles not estimated); under (b) the single-downstream rows (1D lines, rings, E/W inside a mesh)
  can only cost, by up to the counted +4 instructions per forward where E1 binds, and the decision rests on them.
- **Open questions** (§6): the NoC VC and posting of today's stateless forwards change with the fix (VC 1 to the
  link's EDM_NOC_VC; non-posted credits on buffer 3 to posted on buffer 1); two trims for (b)'s E/W cost; the torus
  headroom failure; the untested UDM relay branch; the additive API.

## 1. Code shape

### 1.1 The additive API (variant copies only; `api_patch.py`)

**For upstream review:** three additive functions, each documented in place as "Additive (fabric router study; for
upstream review)" and placed next to its existing stateful sibling. Nothing existing changes.

The data write: `noc_write_with_state<DM_DEDICATED_NOC, cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_wait, update_counter,
posted>` writes TARG_ADDR_LO (source), AT_LEN_BE, RET_ADDR_LO, RET_ADDR_COORDINATE and CMD_CTRL, as local writes already
do (router_local_delivery.hpp), but takes the destination as a 64-bit address and extracts the coordinate from it
(two shifts per write). Added next to `noc_wwrite_with_state` (Blackhole) / `noc_write_with_state` (Wormhole):

```cpp
template <uint8_t noc_mode = DM_DEDICATED_NOC, uint32_t cmd_buf, enum CQNocFlags flags, enum CQNocSend send = CQ_NOC_SEND,
          enum CQNocWait wait = CQ_NOC_WAIT, bool update_counter = true, bool posted = false>
inline __attribute__((always_inline)) void noc_write_to_node_with_state(
    uint32_t noc, uint32_t src_addr, uint32_t node_coordinate, uint32_t dst_addr, uint32_t size = 0, uint32_t ndests = 1);
    // Blackhole: as noc_write_with_state, with RET_ADDR_COORDINATE = node_coordinate as given; MID stays preset.
    // Wormhole: a thin wrapper of noc_wwrite_with_state, which already takes the coordinate separately.
```

Its init is the existing `noc_async_write_one_packet_with_trid_set_state` (CTRL, RET_ADDR_MID, RET_ADDR_COORDINATE).
(The cov_m coverage set has no Wormhole router configs, so the Wormhole forms are compiled only as templates.)

The credit lacks a form: `noc_fast_write_dw_inline_with_state` updates the low address or the coordinate, never both
(a static_assert). Added, next to it, for Blackhole and Wormhole:

```cpp
template <uint8_t noc_mode = DM_DEDICATED_NOC, bool posted = false, bool update_counter = true,
          bool skip_cmdbuf_chk = false>
inline __attribute__((always_inline)) void noc_fast_write_dw_inline_to_node_with_state(
    uint32_t noc, uint32_t cmd_buf, uint32_t node_coordinate, uint32_t addr_lo);
    // counters as noc_fast_write_dw_inline_with_state; then TARG_ADDR_LO, TARG_ADDR_COORDINATE, CMD_CTRL send
```

and its dataflow wrapper `noc_inline_dw_write_to_node_with_state<update_counter, posted, skip_cmdbuf_chk>(
node_coordinate, addr, cmd_buf, noc)`, which, as `noc_inline_dw_write_with_state`, always updates the counter on
Blackhole (issue 28758). The value (AT_DATA), byte enables, control and TARG_ADDR_MID stay as
`noc_inline_dw_write_set_state` left them, so only registers and word-aligned destinations qualify (documented).

### 1.2 The forward (shared by (a) and (b))

```cpp
template <uint8_t NOC, uint8_t DATA_CMD_BUF, uint8_t CREDIT_CMD_BUF>
FORCE_INLINE void forward_partially_preset(
    const DownstreamLink& link,
    uint32_t credit_register,
    uint32_t dst,
    uint32_t src_addr,
    uint32_t size_bytes,
    uint8_t trid) {
    // Read once: the register writes below store through uint32_t pointers, which may alias the field.
    const uint32_t node = link.noc_coord;
    const bool data_ready = noc_cmd_buf_ready(NOC, DATA_CMD_BUF);
    const bool credit_ready = noc_cmd_buf_ready(NOC, CREDIT_CMD_BUF);
    if (!(data_ready & credit_ready)) {
        while (!noc_cmd_buf_ready(NOC, DATA_CMD_BUF));
        while (!noc_cmd_buf_ready(NOC, CREDIT_CMD_BUF));
    }
    ncrisc_noc_set_transaction_id</*skip_cmdbuf_chk=*/true>(NOC, DATA_CMD_BUF, trid);
    noc_write_to_node_with_state<DM_DEDICATED_NOC, DATA_CMD_BUF, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_wait,
                                 /*update_counter=*/is_blackhole_arch, /*posted=*/true>(
        NOC, src_addr, node, dst, size_bytes);
    noc_inline_dw_write_to_node_with_state</*update_counter=*/false, /*posted=*/true, /*skip_cmdbuf_chk=*/true>(
        node, credit_register, CREDIT_CMD_BUF, NOC);
}
```

- The two trims (user decision 2): the coordinate is read once into `node` (the one-line why is in the code), and the
  data write takes it as a coordinate (`noc_write_to_node_with_state`) instead of a 64-bit address it would shift apart
  again. First-round drafts used `noc_write_with_state(..., link.noc_addr(dst), ...)` and read `link.noc_coord` twice.

- The two ready reads are issued together (they overlap, RESULTS §17) and tested together. When either buffer is busy
  (rare), both are waited for before either command, so nothing is held across the data command. The first draft
  tested the credit's bit after the data command, as the lead's E/W forward does; that bit spilled (§3.3).
- `credit_register` is a compile-time constant (`get_stream_reg_write_addr<remote_credit_stream<entry>>()`).
- Init, once, after the opens (`preset_downstream_cmd_bufs<0>`, out of line as in the lead):

```cpp
template <uint8_t NOC, uint8_t DATA_CMD_BUF, uint8_t CREDIT_CMD_BUF>
__attribute__((noinline)) void preset_forward_cmd_bufs() {
    constexpr uint8_t noc_vc = tt::tt_fabric::forward_and_local_write_noc_vc;
    noc_async_write_one_packet_with_trid_set_state</*posted=*/true>(0, DATA_CMD_BUF, NOC, noc_vc);
    noc_inline_dw_write_set_state</*posted=*/true, /*set_val=*/true>(
        0, pack_value_for_inc_on_write_stream_reg_write(-1), INLINE_WRITE_ALL_BYTES, CREDIT_CMD_BUF, NOC, noc_vc);
}
```

  The destination node in the presets is a placeholder (0); every forward writes its own. VC0's pair is preset, and
  VC1's where its buffers differ from VC0's (today they never do: the host gives every receiver channel buffers 2 and 1,
  the buffer-2 fix (c)).
- A compile-time consistency rule guards the presets (`forward_cmd_bufs_consistent<VC>`): a VC's data and credit
  buffers differ, neither is VC0's other buffer, and neither is the local-write buffer (0) or `write_at_cmd_buf` (3).
  Otherwise forwards fall back to the stateless path. Image-neutral today (checked: a rebuild with and without it is
  byte-identical); it states the ownership table (§2) in code.

### 1.3 Option (b): partial presets on every forward (`v4rxnsb`)

The lead's eight preset predicates (`command_buffers_can_keep_presets`, `uses_both/uses_a_vc0_forward_cmd_buf`,
`rides/clobbers_vc0_preset`, `vc0_keeps_forward_preset`, `forwards_are_stateful`, `another_vc_rides_vc0_preset`,
`forwards_set_credit_register`) go. One remains:

```cpp
constexpr bool forwards_use_presets =
#if !defined(DEBUG_PRINT_ENABLED) && !defined(WATCHER_ENABLED)
    command_buffer_presets && !FORCE_ALL_PATHS_TO_USE_SAME_NOC && !udm_mode && forward_cmd_bufs_consistent<0>() &&
    forward_cmd_bufs_consistent<1>();
#else
    false;
#endif
```

(`!udm_mode`: a UDM router's relay forwards stay stateless on the same data buffer, so a UDM router presets nothing,
as today.) `Downstreams<VC>::forward<NAME>` becomes one branch (`Relay::forward` is unchanged):

```cpp
if constexpr (forwards_use_presets) {
    forward_partially_preset<Traits::noc, Traits::data_cmd_buf, Traits::sync_cmd_buf>(
        links[entry], get_stream_reg_write_addr<Traits::template remote_credit_stream<entry>>(), dst, src_addr,
        size_bytes, trid);
} else {
    forward_stateless<Traits::noc, Traits::data_cmd_buf>(links[entry], dst, src_addr, size_bytes, trid);
}
```

The lead's inline stateful block (its own paired ready reads, the `#if ARCH_BLACKHOLE` write, the credit with
`update_addr_lo = forwards_set_credit_register`) and the per-destination init (`dense` index arithmetic) go. One shape
for one downstream or several, VC0 and VC1. router_downstream.hpp: 532 -> 499 lines (132
removed, 99 added, comments included; 503 as run, §7).

### 1.4 Option (a): partial presets where the lead keeps none (`v4rxnsa`)

The lead's predicates stay verbatim; two are added:

```cpp
constexpr bool forward_cmd_bufs_can_be_preset = /* not DPRINT/watcher */ command_buffer_presets &&
                                                !FORCE_ALL_PATHS_TO_USE_SAME_NOC && !udm_mode;
constexpr bool command_buffers_can_keep_presets = forward_cmd_bufs_can_be_preset && !is_intermesh_router &&
                                                  !is_intermesh_router_on_edge && !is_intramesh_router_on_edge;  // the lead's, split
constexpr bool forwards_partially_preset = forward_cmd_bufs_can_be_preset && !vc0_keeps_forward_preset() &&
                                           forward_cmd_bufs_consistent<0>() && forward_cmd_bufs_consistent<1>();
```

The forward has three paths:

```cpp
if constexpr (forwards_are_stateful<VC>()) {
    /* the lead's full-preset block, unchanged */
} else if constexpr (forwards_partially_preset) {
    forward_partially_preset<...>(links[entry], get_stream_reg_write_addr<...>(), dst, src_addr, size_bytes, trid);
} else {
    forward_stateless<...>(links[entry], dst, src_addr, size_bytes, trid);
}
```

and `preset_downstream_cmd_bufs<VC>` keeps the lead's per-destination full preset, with an `else if constexpr
(forwards_partially_preset)` branch that presets every forwarding VC's buffers partially. A router is all full or
all partial: VC1 rides VC0's full preset only where the lead lets it; anywhere else both VCs take partial presets, so
nothing overwrites a full preset. Single-downstream images are byte-identical to the lead (§3.4). router_downstream.hpp:
532 -> 631 lines (both forward shapes and both inits are kept; 632 as run, §7).

### 1.5 What does not change

- **Local writes, buffer 0** (`router_local_delivery.hpp`): untouched, including the inline-write fix (ii').
- **Buffer 3** (`write_at_cmd_buf`): the firmware's preset atomic return address is never written. After the fix only
  local atomics use it on a two-ERISC E1; today's stateless credits leave it (§2, §6 Q6).
- **One ERISC** (`command_buffer_presets` false; on Blackhole also FORCE_ALL_PATHS_TO_USE_SAME_NOC, which puts forward
  data on buffer 0 with the local writes): stateless as today. Every one-ERISC image is byte-identical to the lead in
  both drafts (static).
- **DPRINT / watcher builds**: stateless as today.
- **UDM** (user decision 4): `Relay::forward` is the lead's, stateless, and UDM routers preset nothing (`!udm_mode`
  in both drafts), so the relay's stateless data write never overwrites a preset. Every UDM router the host builds on
  Blackhole is one-ERISC anyway (all 28 bh_2d_udm_gtest configs have NUM_ACTIVE_ERISCS 1): no UDM image changes.
- **Teardown**: after the loop; it issues whole commands, so the presets do not matter there. E1 does not run the base
  firmware (the context switch runs on E0 with E1 parked, router_round.hpp `end_round`), so nothing disturbs E1's
  presets mid-run, as for the lead's E/W presets on the same buffers.
- **E0** (router_sender.hpp, router_single.hpp, router_common.hpp): untouched.

## 2. Command-buffer ownership per router shape (two-ERISC E1 unless marked; code + host args)

Host ids (erisc_datamover_builder.cpp): FWD_DATA 2, FWD_SYNC 1 and LOCAL_WRITE 0 for every receiver channel (VC0 and
VC1 alike); one-ERISC Blackhole: FWD_DATA 0. VC2 does not forward (its ids are unused). "Full" = destination preset
(lead's stateful forward); "partial" = §1.2; "stateless" = every field per forward.

| router shape | buffer 2 (forward data) | buffer 1 (forward credit) | buffer 3 | buffer 0 |
| --- | --- | --- | --- | --- |
| 1D line / ring (one downstream) | lead: full; (a): full; (b): partial | lead: full (ring: LO per forward, deadlock avoidance); (a): same; (b): partial | atomics | local writes (preset), inline-write fix (ii') |
| 2D E/W inside a mesh (one live downstream), VC0 and VC1 riding | lead/(a): full; (b): partial | lead/(a): full (LO per forward when VC1 rides); (b): partial | atomics | local writes |
| 2D N/S inside a mesh (2-3 live) | lead: stateless (posted, NoC VC 1); (a)/(b): partial | lead: unused in the loop; (a)/(b): partial (posted) | lead: atomics + stateless credits (non-posted, VC 1); (a)/(b): atomics | local writes |
| edge / intermesh (exit picks a downstream at run time), VC0 + VC1 | as N/S | as N/S | as N/S | local writes |
| torus (Y): N/S routers 2 live, E/W one | as N/S / as E/W | as N/S / as E/W | as N/S / as E/W | local writes |
| VC2 on | VC0 as its shape; VC2 never forwards | VC0 as its shape | as its shape | local writes (VC0 and VC2) |
| UDM (one ERISC on Blackhole) | - (data on buffer 0) | - | stateless credits (relay and routers) + atomics | local writes + forward data, stateless |
| one ERISC (FORCE_ALL_PATHS) | - (data on buffer 0) | - | stateless credits + atomics | local writes + forward data, stateless |
| DPRINT / watcher | stateless | - | stateless credits + atomics | local writes, stateless |

After either fix, on every two-ERISC E1 that forwards: buffer 2 = forward data only, buffer 1 = forward credits only,
buffer 3 = atomics only, buffer 0 = local writes only. Under (b) that holds for every such router; under (a) the
single-downstream ones keep their full presets on the same two buffers.

## 3. Static numbers (offline drafts on v4rxg3; -O3 unless marked)

Tools (`loopvars_maps/ns_presets/`): `fwdcount.py` (every inlined forward copy in the receiver loop, split by the innermost frame:
NoC register writes, ready reads, other loads, counter stores; the not-ready wait loops counted apart), `funccmp.py`
(per image, which functions' code changed), `guard_builds.py`, `loopvars.py --headroom`, `size_report.py`.

### 3.1 Per forward

The first forward copy of each image (all copies in an image are alike). Cells: shape, then executed instructions /
command-buffer register writes / loads / counter stores, with both buffers ready. The not-ready wait loops are off the
path in the full and partial forms; the stateless form's two polls run once each, so its count includes them. (xN) =
forward copies in the receiver loop. Loop = receiver loop size (largest strongly connected component of
`receiver::run`).

| E1 image | shape | lead: forward (instr / reg writes / loads / counter st) | (a) | (b) | loop lead / (a) / (b) |
| --- | --- | --- | --- | --- | --- |
| bh_1d 117711 | 1D line (x1) | full 23 / 6 / 2 / 2 | full 23 / 6 / 2 / 2 | partial 27 / 9 / 3 / 1 | 488 / 488 / 489 |
| bh_ring 104444 | 1D ring (deadlock avoidance) (x1) | full 26 / 7 / 2 / 2 | full 26 / 7 / 2 / 2 | partial 27 / 9 / 3 / 1 | 447 / 447 / 443 |
| bh_2d 938563 | 2D E/W in a mesh (3 entries, 1 live) (x1) | full 24 / 6 / 2 / 2 | full 24 / 6 / 2 / 2 | partial 28 / 9 / 3 / 1 | 492 / 492 / 492 |
| bh_2d 972491 | 2D N/S, 1 downstream (x1) | full 23 / 6 / 2 / 2 | full 23 / 6 / 2 / 2 | partial 27 / 9 / 3 / 1 | 488 / 488 / 488 |
| bh_2d 483794 | 2D N/S, 2 live (x2) | stateless 53 / 14 / 6 / 3 | partial 27 / 9 / 3 / 1 | partial 27 / 9 / 3 / 1 | 596 / 556 / 556 |
| bh_2d_vc2 104913 | 2D E/W, VC2 on (x1) | full 24 / 6 / 2 / 2 | full 24 / 6 / 2 / 2 | partial 28 / 9 / 3 / 1 | 799 / 799 / 799 |
| bh_2d_vc2 126334 | 2D N/S 2 live, VC2 on (x2) | stateless 53 / 14 / 6 / 3 | partial 27 / 9 / 3 / 1 | partial 27 / 9 / 3 / 1 | 931 / 891 / 891 |
| bh_torus 116442 | Y-torus E/W (deadlock avoidance) (x1) | full 26 / 7 / 2 / 2 | full 26 / 7 / 2 / 2 | partial 27 / 9 / 3 / 1 | 453 / 453 / 450 |
| bh_torus 118805 | Y-torus N/S, 2 live (deadlock avoidance) (x2) | stateless 53 / 14 / 6 / 3 | partial 27 / 9 / 3 / 1 | partial 27 / 9 / 3 / 1 | 568 / 538 / 538 |
| bh_vc1 115738 | VC1 edge router, 3+3 downstreams (x6) | stateless 53 / 14 / 6 / 3 | partial 27 / 9 / 3 / 1 | partial 27 / 9 / 3 / 1 | 1317 / 1197 / 1197 |
| bh_vc1 144490 | VC1 intermesh router, 2+2 (x2) | stateless 53 / 14 / 6 / 3 | partial 27 / 9 / 3 / 1 | partial 27 / 9 / 3 / 1 | 583 / 544 / 544 |
| glx_2d 105806 | Galaxy E/W (x1) | full 24 / 6 / 2 / 2 | full 24 / 6 / 2 / 2 | partial 28 / 9 / 3 / 1 | 492 / 492 / 492 |
| glx_2d 135682 | Galaxy N/S, 2 live (x2) | stateless 53 / 14 / 6 / 3 | partial 27 / 9 / 3 / 1 | partial 27 / 9 / 3 / 1 | 596 / 556 / 556 |
| glx_express 122107 | Galaxy express N, 4 downstreams (x4) | stateless 53 / 14 / 6 / 3 | partial 27 / 9 / 3 / 1 | partial 27 / 9 / 3 / 1 | 717 / 639 / 639 |
| glx_torus 141635 | Galaxy torus N, 3 downstreams (x3) | stateless 53 / 14 / 6 / 3 | partial 27 / 9 / 3 / 1 | partial 27 / 9 / 3 / 1 | 639 / 589 / 589 |

Where the instructions go (from the listings, `fwdcount.py` + the disassembly):
- **Stateless -> partial** (multi-downstream, both drafts), -26:
  - register writes 14 -> 9: the data write drops its control word (7 -> 6: tag, source, length, destination low,
    coordinate, go); the credit drops control, AT_DATA, TARG_ADDR_MID and AT_LEN_BE (7 -> 3: register, coordinate, go);
  - loads 6 -> 3: the credit's 64-bit address (2 words) and the non-posted counters go; the coordinate is loaded twice;
  - counter stores 3 -> 1: the non-posted credit's two (issued, acked) go, and the posted data and credit updates merge
    into one;
  - ALU 26 -> 12: the credit's byte-enable shift, MID mask, control and value constants;
  - two serial polls (each a load, a branch and its address) become two paired reads, one `or` and one branch.
- **Full -> partial** (single-downstream, (b) only), +4: +3 register writes (the data's RET_ADDR_COORDINATE, the
  credit's TARG_ADDR_LO and TARG_ADDR_COORDINATE), +2 loads of `link.noc_coord` (the compiler reloads it after the
  volatile register writes), +2 ALU masking the coordinate back out of the 64-bit address (`slli` / `srli`), +2 ALU
  for the credit register's address, +1 `or` for the joint test; -3 from one counter update instead of two, -1 branch,
  -2 from addressing buffer 1 off buffer 2's base and reusing one `li 1` for both go writes.
  On deadlock-avoiding routers the lead already writes the credit register: +1 instruction, +2 register writes.
- **The proxy** (v4rxg3ml -> v4rxnsbml, every 2D router multi-downstream): each forward 53 -> 27, the same as the
  production multi-downstream images; the E/W proxy loop 685 -> 625 (three forward copies), N/S 596 -> 556. bh_2d and
  glx_2d: all 76 E1 images differ, every E0 identical.

### 3.2 Loop size (guard_builds.py, receiver loop instructions, draft - lead)

| family | -O3 (a) | -O3 (b) | -Os (a) | -Os (b) |
| --- | --- | --- | --- | --- |
| 1D line (bh_1d, glx_1d) | 0 | +1 | 0 | +5 |
| ring (bh_ring, glx_ring) | 0 | -4 | 0 | +2 |
| 2D E/W inside a mesh, N/S with one downstream (bh_2d 20, glx_2d 28, VC2 on) | 0 | 0 | 0 | +5 |
| 2D N/S, 2 live (bh_2d 8, glx_2d 8, VC2 on) | -40 | -40 | -72 to -76 | -72 to -76 |
| Galaxy 2D, 3 live (glx_2d 12) | -70 | -70 | -106 | -106 |
| torus E/W / N/S (bh_torus) | 0 / -30 | -3 / -30 | 0 / -77 | +2 / -77 |
| Galaxy express (multi / single) | -50 to -78 / 0 | -50 to -78 / -3 | -97 to -128 / 0 | -97 to -128 / +2 |
| VC1 (every E1) | -39 to -120 | -39 to -120 | -71 to -207 | -71 to -207 |

Stack accesses in the loop (same tool): unchanged everywhere at -O3 except bh_torus N/S (+4 stores, +5 loads; §3.3),
Galaxy express / torus multi (+3 / +3, not hot: the headroom check passes them), glx_2d 3-live (+1 / +1) and VC1
(4-6 images +5 / +3, classed designed memory or around a call by the headroom check). At -Os the multi-downstream
loops lose 10-28 stack accesses (today's stateless forwards spill `dst`, `size_bytes` and the write's values there).

### 3.3 Guard and headroom (`loopvars.py --headroom`, v4rxg3 -> draft, -O3, every image)

| family | first draft (credit's ready bit tested after the data command) | drafts as built (joint test) |
| --- | --- | --- |
| glx_express | 8/28 pass: `credit_ready` moved to the stack in 20 | 28/28 |
| glx_express_line | 16/44: `credit_ready` in 28 | 44/44 |
| glx_torus | 4/8: `credit_ready` in 4 | 8/8 |
| bh_torus | 12/12 | **8/12: in the 4 N/S (2 live) images, `deliver`'s value and `local_write`'s destination move to the stack (+4 stores, +5 loads)** |
| bh_2d_codeprof, bh_2d_allinstr (instrumented, not production) | codeprof 20/28 | 20/28 each: `deliver`'s value and source |
| every other family (2D, 1D, ring, VC1, VC2, one ERISC, UDM, Galaxy 2D / 1D / ring / express mesh, trimming) | pass | pass |

(a) and (b) give the same results (the multi-downstream images are byte-identical between them; the single-downstream
images pass in (b) and are the lead's in (a)). A third form, a serial poll before each command (the API's own wait),
was tried on the failing families: the same bh_torus result, and the most counted instructions of the three (31 on the
path, both polls executed). No form passes everywhere; the joint test is kept (fewest counted instructions, every
Galaxy family passes). The torus spill is on the delivery path of the P150 Y-torus N/S routers, not on the forward
(§6 Q3).

### 3.4 Identity (full coverage, 2,863 buildable configs; bytecmp of loadable bytes, then per function)

| | -O3 | -Os |
| --- | --- | --- |
| (a): images differing from v4rxg3 | 394 | 394 |
| (b): images differing from v4rxg3 | 1,004 | 1,004 |
| (a) vs (b) | 610 differ (the single-downstream forwarders); the 394 multi-downstream images byte-identical | |

- (a)'s 394 = every two-ERISC E1 image whose forwards are stateless today: bh_2d 8 (N/S, 2 live), bh_torus 4, every
  bh_vc1 E1 (22), glx_2d 20, glx_express 20, glx_express_line 28, glx_express_mesh 20, glx_torus 4, the trimming
  gtests' 20, and the same shapes in the instrumented / VC2 / trimmed families. In all but the receiver-trimmed ones
  (bh_2d_vc2_trim_rx0 8, bh_vc1_trim_rx0 8: no forwards, init only) the receiver loop changes; nothing else does.
- (b)'s 1,004 = (a)'s 394 + every single-downstream forwarding E1 (1D, ring, E/W, torus E/W, express single; the
  receiver loop changes) + two-ERISC E1 images with nothing to forward, which now preset buffers 1-2 at init (init
  only: bh_neighbor 4, bh_2d_vc2_trim_rx0 28, bh_vc1_trim_rx0 8, bh_trim_gtest 6, and their instrumented copies).
- Byte-identical in both: every E0 image (1,054 two-ERISC E0 configs) and every one-ERISC image (757, incl. UDM:
  bh_2d_udm_gtest is all one-ERISC, and mux). Only two-ERISC E1 images change: (a) 394, (b) 1,004 of 1,052.
  (b) first made `preset_downstream_cmd_bufs` FORCE_INLINE, which re-allocated registers in one-ERISC `kernel_main`
  (init only); it is out of line again, as in the lead, and the one-ERISC images are identical.
- The consistency rule (§1.2) is image-neutral (a rebuild with and without it: identical).
- Ok-set: 2,863 / 3,124 build in all three (the 261 failures are the known non-buildable configs, the same set).

### 3.5 Code size and stack (size_report.py; worst over the 54 cases every build has)

| | -O3 E0+E1 worst (bh_vc1) | -O3 least free E0 / E1 | -Os E0+E1 worst (bh_vc1) | -Os least free |
| --- | --- | --- | --- | --- |
| v4rxg3 | 22,996 | 1,604 / 3,700 | 22,156 | 1,556 / 3,556 |
| (a) v4rxnsa | 22,608 (-388) | unchanged | 21,452 (-704) | unchanged |
| (b) v4rxnsb | 22,608 (-388) | unchanged | 21,452 (-704) | unchanged |

Per case (-O3, largest E1): bh_2d 6,392 -> 6,352 (a) / 6,328 (b); bh_1d 6,204 -> 6,204 / 6,160; bh_ring 6,040 ->
6,040 / 5,960; glx_2d 7,004 -> 6,828 / 6,828; glx_express 7,440 -> 7,224 / 7,224; bh_vc1 10,588 -> 10,200 / 10,200.

## 4. Predictions per row (written before any run; counted work only)

Rules: counted work only, no stall-time estimate; confidence marked; earlier estimates on this core missed by 2-4x.

Counted (§3.1), per forward on the forwarding E1:
- multi-downstream routers (N/S, edge, intermesh, VC1), both drafts: -26 instructions, -5 register writes, -2 counter
  stores, -3 loads, and the two serial polls become one paired read. That is 26 of the 29 instructions that separate
  today's stateless forward from the full preset; that 29-instruction gap measured ~37 E1 cycles/packet (V4_NS_FLOOR
  §2, sl vs lead). The multi-downstream bookkeeping (admission over the live downstreams, the dispatch; ~17-19 cycles
  in the same study) stays.
- single-downstream routers under (b): +4 instructions (+3 register writes: the data coordinate, the credit register
  and coordinate; +2 loads of the coordinate; +4 ALU for the coordinate mask and the credit address; +1 `or`; -3 from
  one counter update instead of two; -1 branch; -2 shared constants); on deadlock-avoiding routers (rings, torus E/W), where the lead already
  writes the credit register, +1 instruction and +2 register writes. Under (a): nothing (images byte-identical).

| row (P150 unless marked) | forwarders | bound side (earlier A/Bs) | (a) | (b) | confidence |
| --- | --- | --- | --- | --- | --- |
| multi-downstream proxy, 3-hop E/W and turn (v4rxg3ml vs v4rxnsbml) | 3 live | E1 (174.5 / 167.3, V4_NS_FLOOR §2) | gain | same as (a) | direction high, size low |
| Galaxy N/S rows (3-hop, 7-hop N/S, corner, turn, column merge) | N/S, 2-3 live | E1 (172.5-173.3 at 2 KiB slots; 182-187 at 4352 B) | gain; the bound may move to E0 (~140) only if E1 falls by more than ~32 | same as (a) | direction medium-high, size low |
| Galaxy E/W rows (1-hop, 3-hop E/W) | E/W, one live | E0 (E1 slack ~30) | 0 (images identical) | 0 (±1) | high / medium |
| hop3 (E/W 3-hop) | E/W, one live | E0 (lead E1 ~118 against 144.7) | 0 | 0 (±1): +4 instructions on a side with ~27 cycles of slack | high / medium |
| ew4352, ewscatter, VC2-on E/W rows (1 hop) | none (delivery only) | E0 | 0 | 0 (±0.5; loop layout only) | high |
| lin3 (1D 3-hop) | 1D | wire / both (118.0) | 0 | 0 to +4 | low |
| ring1, halfring | ring, deadlock avoidance | E1 / first-level acks | 0 | 0 to +2 (+1 instruction, +2 register writes) | low |
| census codec rows (forwarded atomic / fused / scatter, multicast scatter) | E/W, one live | E1-sensitive (codec costs showed there) | 0 | 0 to +4 | low |
| VC1 rows (census_vc1) | edge / intermesh, all stateless today | window-bound inter-mesh (job 360); others E1 | 0 to a gain where E1 binds | same as (a) | low |
| torusns, torusns_scatter (one N/S hop) | delivery only, on the 2-live N/S routers whose delivery values now spill (§3.3) | - | 0 to +2 | 0 to +2 | low |
| one ERISC | images byte-identical | - | 0 | 0 | high (identity) |
| latency (exp_latency.yaml) | per hop | path | 0 per E/W hop; a gain per multi-downstream hop | +0 to +4 per single-downstream hop | low |

Rows that can only cost under (b) (relative to (a)): every row whose forwarders are single-downstream and whose E1 is at
or near the bound: lin3 and the 1D lines, ring1 / halfring, the census codec rows; on the Galaxy, glx_1d / glx_ring
and the E/W legs. They decide the (a)/(b) rule. E/W rows with E1 slack (hop3) can hide (b)'s cost; §5 step 3 reads it
from E1 directly.

## 5. Measurement plan (after the user's review; nothing queued)

All P150 A/Bs single-change, same job for both arms, two passes in reversed router order, TT_FABRIC_NO_BW_TELEMETRY=1,
fresh kernel cache per router, Δ = draft - lead (cycles/packet).

1. **Correctness first** (one job, both drafts): mesh sanity 96/96 and `rxo0_inline_runs.sh` (inline write then flushed
   atomic, 1D and 2D), as `rp_probe_runs.sh val`. A failure or hang stops the plan (a hang rules the design out).
2. **The multi-downstream proxy** (the N/S path on the P150): v4rxg3ml against v4rxnsbml, the 3-hop E/W and turn rows
   (`sls_sl_runs.sh` base arm, 2D and 1D sets), plus PC samples of the forwarding E1 on the 3-hop E/W row
   (`bnd_pcs_runs.sh`): where the forward's samples went (NOC_CMD_BUF_WRITE_REG, the stateless credit's lines), within
   E1 only.
3. **The (a)/(b) decision rows**, lead v4rxg3 against v4rxnsb, with v4rxnsa as a third arm (byte-identical to the lead
   on every single-downstream E1, so on these rows it is a noise control):
   - E1-near-bound single-downstream rows: lin3 (1D 3-hop), ring1, halfring;
   - the census 2D 4352 and mix suites (forwarded atomic / fused / scatter, multicast scatter; `rp_probe_runs.sh
     codec`);
   - the E/W controls ew4352, hop3, ewscatter; VC2 on (vc2ew4352, vc02bew4352);
   - one ERISC (ew4352, ewscatter, mcast): images byte-identical, a control;
   - PC samples of hop3's forwarding E1, lead vs (b): the forward lines' cycles within E1, which a flat E0-bound row
     cannot show.
   Rule (user): (b) if its cost is at most 1-2 cycles/packet on these rows, else (a). If the rows are flat but the E1
   split shows a cost (E1 has slack on the P150 E/W rows), an isolating A/B makes E1 the bound: a one-side receive
   spin on both arms (the v4bdd delay hooks, ported to the drafts' copies), so Δ is (b)'s E1 cost.
4. **Multi-downstream production shapes on the P150**: VC1 (`census_vc1_runs.sh`; every VC1 E1 forward changes in both
   drafts), torus (torusns, torusns_scatter: one N/S hop, so delivery only, on the Y-torus N/S routers whose delivery
   values now spill, §3.3: these rows measure that spill).
5. **Validation on the pick**: `variant_val_runs.sh` 13/13 (includes the channel-trimming gtests on two and one
   ERISC), `rxo0_inline_runs.sh`, `rxo_udm_runs.sh` (UDM images are byte-identical; run anyway, as asked).
6. **Galaxy, after the VC2 line agent releases the slot**: Slurm, dry run first; **only bh-glx-110-a\* nodes
   (`bh_sc36_6` or the 110a pods), never bh_sc36_2**. Lead against the pick on V4_NS_FLOOR §1's rows (1-hop E/W, 3-hop
   E/W, 3-hop N/S, 7-hop N/S, 10-hop corner, turn, column merge) at both slot sets (4352 B 17,16,16 and 2 KiB 41,32,32),
   two passes reversed, with the base / tx32 / rx32 arms on the N/S rows, so the A/B also shows whether the bound
   moved from E1 to E0.

## 6. Open questions for the user

Answered by the user on 2026-11-02 (Q1 bundle; Q2 draft both trims; Q3 accept; Q4 relay stays stateless; Q5
acceptable, flagged for upstream review): see the decisions at the top. Q6-Q8 stand as notes.

1. **NoC VC and posting of today's stateless forwards.** The partial presets take the E/W forwards' control words:
   data and credit on the link's `EDM_NOC_VC` (2 + link index mod NUM_EDM_NOC_VCS), credit posted. Today's stateless
   forwards (N/S, edge, intermesh, VC1) use the API defaults: NoC VC 1, credit non-posted on buffer 3 (as main;
   V4_BOUNDS §4's audit). So on those routers the fix bundles four changes: register programming, credit buffer 3 ->
   1, credit non-posted -> posted, NoC VC 1 -> EDM_NOC_VC. Keep the E/W words (one command shape everywhere), or preset
   VC 1 / non-posted credits on the routers that are stateless today, so the A/B isolates the register work? If a gain
   needs attribution, that variant is the isolating arm.
2. **Two trims for (b)'s single-downstream cost (counted, not drafted).** In the listing (§3.1) the compiler reloads
   `link.noc_coord` after the data command's register writes (volatile stores), and the data write masks the coordinate
   it gets back out of a 64-bit address (`slli` / `srli`). Copying the coordinate into a local once (-1 load) and an
   additive data-side form that takes the node coordinate, as the credit's does (-2 ALU), would leave (b)'s
   single-downstream cost at +1 instruction net (the 3 register writes, offset by the merged counter update and the
   single branch). Draft them before the A/B?
3. **The joint ready test.** It replaces the lead's pattern (credit's bit tested after the data command) in the new
   forward only; the lead's full preset in (a) keeps its pattern. With it every Galaxy family passes the headroom
   check, but the P150 Y-torus N/S routers (4 bh_torus images) move two local-delivery values to the stack (§3.3);
   with the first draft's pattern those pass and 52 Galaxy express / torus images fail instead. Accept the torus
   spill for the A/B (torusns and torusns_scatter deliver on exactly those routers, so they measure it), or look for a
   structural fix before any adoption build?
4. **UDM relay.** Its partial branch compiles in no coverage image (UDM is one-ERISC on Blackhole). Keep one shape for
   every forward (as drafted), or keep the relay stateless until a two-ERISC UDM config exists to validate it?
5. **The API addition** (`noc_fast_write_dw_inline_to_node_with_state` in both architectures' noc_nonblocking_api.h,
   `noc_inline_dw_write_to_node_with_state` in dataflow_api.h): additive only. Acceptable for adoption, or kept
   router-local behind a named helper?
6. **Buffer 3 after the fix** holds only local atomics on a forwarding two-ERISC E1. FLUSH_DEFER_DESIGN's audit ("atomics
   and stateless forward doorbells are non-posted on write_at_cmd_buf (buffer 3) with tag 0") then loses its second
   half; its step B (move buffer 3's tag to a spare ID) concerns atomics only. For the flush-defer pass.
7. **Credit tag.** Buffer 1's packet tag is never set (as for today's E/W credits); posted doorbells count under
   whatever transaction ID it holds (the firmware's). Fine as today?
8. **Readability.** (b) removes eight predicates and one forward shape (499 lines); (a) keeps both shapes (631). If the
   A/B leaves (b) within the rule, (b) is also the readable one.

## 7. The drafts as run (decisions applied; offline, -O3 unless marked)

Changes from §1-§3's first-round drafts: the two trims in the shared forward (§1.2), the relay unchanged and UDM
routers without presets, the additive data-write form. Rebuilt from v4rxg3 by `stage_a.py` / `stage_b.py` (same names:
v4rxnsa, v4rxnsb, v4rxnsbml); full coverage at -O3 and -Os, swap guard clean before and after.

**Per forward** (the same method as §3.1):

| E1 image | shape | lead: forward (instr / reg writes / loads / counter st) | (a) | (b) | loop lead / (a) / (b) |
| --- | --- | --- | --- | --- | --- |
| bh_1d 117711 | 1D line (x1) | full 23 / 6 / 2 / 2 | full 23 / 6 / 2 / 2 | partial 24 / 9 / 2 / 1 | 488 / 488 / 486 |
| bh_ring 104444 | 1D ring (deadlock avoidance) (x1) | full 26 / 7 / 2 / 2 | full 26 / 7 / 2 / 2 | partial 24 / 9 / 2 / 1 | 447 / 447 / 440 |
| bh_2d 938563 | 2D E/W in a mesh (3 entries, 1 live) (x1) | full 24 / 6 / 2 / 2 | full 24 / 6 / 2 / 2 | partial 25 / 9 / 2 / 1 | 492 / 492 / 489 |
| bh_2d 972491 | 2D N/S, 1 downstream (x1) | full 23 / 6 / 2 / 2 | full 23 / 6 / 2 / 2 | partial 24 / 9 / 2 / 1 | 488 / 488 / 485 |
| bh_2d 483794 | 2D N/S, 2 live (x2) | stateless 53 / 14 / 6 / 3 | partial 24 / 9 / 2 / 1 | partial 24 / 9 / 2 / 1 | 596 / 550 / 550 |
| bh_2d_vc2 104913 | 2D E/W, VC2 on (x1) | full 24 / 6 / 2 / 2 | full 24 / 6 / 2 / 2 | partial 25 / 9 / 2 / 1 | 799 / 799 / 796 |
| bh_2d_vc2 126334 | 2D N/S 2 live, VC2 on (x2) | stateless 53 / 14 / 6 / 3 | partial 24 / 9 / 2 / 1 | partial 24 / 9 / 2 / 1 | 931 / 885 / 885 |
| bh_torus 116442 | Y-torus E/W (deadlock avoidance) (x1) | full 26 / 7 / 2 / 2 | full 26 / 7 / 2 / 2 | partial 24 / 9 / 2 / 1 | 453 / 453 / 447 |
| bh_torus 118805 | Y-torus N/S, 2 live (deadlock avoidance) (x2) | stateless 53 / 14 / 6 / 3 | partial 24 / 9 / 2 / 1 | partial 24 / 9 / 2 / 1 | 568 / 532 / 532 |
| bh_vc1 115738 | VC1 edge router, 3+3 downstreams (x6) | stateless 53 / 14 / 6 / 3 | partial 24 / 9 / 2 / 1 | partial 24 / 9 / 2 / 1 | 1317 / 1179 / 1179 |
| bh_vc1 144490 | VC1 intermesh router, 2+2 (x2) | stateless 53 / 14 / 6 / 3 | partial 24 / 9 / 2 / 1 | partial 24 / 9 / 2 / 1 | 583 / 538 / 538 |
| glx_2d 105806 | Galaxy E/W (x1) | full 24 / 6 / 2 / 2 | full 24 / 6 / 2 / 2 | partial 25 / 9 / 2 / 1 | 492 / 492 / 489 |
| glx_2d 135682 | Galaxy N/S, 2 live (x2) | stateless 53 / 14 / 6 / 3 | partial 24 / 9 / 2 / 1 | partial 24 / 9 / 2 / 1 | 596 / 550 / 550 |
| glx_express 122107 | Galaxy express N, 4 downstreams (x4) | stateless 53 / 14 / 6 / 3 | partial 24 / 9 / 2 / 1 | partial 24 / 9 / 2 / 1 | 717 / 627 / 627 |
| glx_torus 141635 | Galaxy torus N, 3 downstreams (x3) | stateless 53 / 14 / 6 / 3 | partial 24 / 9 / 2 / 1 | partial 24 / 9 / 2 / 1 | 639 / 580 / 580 |

- Partial forward 24-25 instructions (was 27-28): the second coordinate load and the coordinate's two shifts are gone.
- (b) against the full preset: +1 instruction and +3 register writes on 1D and E/W inside a mesh; -2 instructions and
  +2 register writes on deadlock-avoiding rings and torus E/W. Against the stateless forward: -29 instructions, -5
  register writes.

**Identity:** (a) 394 and (b) 1,004 images differ from v4rxg3, at -O3 and -Os, all two-ERISC E1; (a) and (b) are
byte-identical on the 394 multi-downstream images (610 differ, the single-downstream forwarders). Every E0 and every
one-ERISC image (incl. UDM) is byte-identical to the lead. The proxy: all 76 bh_2d / glx_2d E1 images differ from
v4rxg3ml, every E0 identical.

**Loop size (draft - lead):**

| family | -O3 (a) | -O3 (b) | -Os (a) | -Os (b) |
| --- | --- | --- | --- | --- |
| 1D line | 0 | -2 | 0 | +2 |
| ring | 0 | -7 | 0 | -1 |
| 2D E/W inside a mesh (bh_2d 20, glx_2d 28) | 0 | -3 | 0 | +2 |
| 2D N/S, 2 live | -46 | -46 | -78 | -78 |
| Galaxy 2D, 3 live | -79 | -79 | -115 | -115 |
| torus E/W / N/S | 0 / -36 | -6 / -36 | 0 / -83 | -1 / -83 |
| Galaxy express multi / single | -59 to -90 / 0 | -59 to -90 / -6 | -106 to -140 / 0 | -106 to -140 / -1 |
| VC1 | -45 to -138 | -45 to -138 | -77 to -225 | -77 to -225 |

**Guard and headroom:** as §3.3's joint-test column: every Galaxy family passes; the P150 Y-torus N/S images fail (4 of
12 bh_torus: the delivery spill the user accepted, decision 3); bh_2d_codeprof / _allinstr (instrumented) 20/28. At -Os
the multi-downstream loops lose 10-28 stack accesses.

**Size (size_report.py, worst over the 54 cases):** -O3 E0+E1 22,996 -> 22,536 (-460, bh_vc1); -Os 22,156 -> 21,380
(-776); least free stack unchanged (E0 1,604 / E1 3,700 at -O3; 1,556 / 3,556 at -Os). Both drafts alike.

## 8. Device results (P150; jobs 481-; TT_FABRIC_NO_BW_TELEMETRY=1; two passes in reversed router order)

### 8.1 Correctness (job 481)

v4rxnsb and v4rxnsa: mesh sanity 96/96 each; inline write then flushed atomic (rxo0_inline_runs.sh) 1D 2/2, 2D 2/2
each. No failure, no hang, tree restored.

### 8.2 The multi-downstream proxy (job 481; v4rxg3ml vs v4rxnsbml; `slsab.py nfpa,nfpb v4rxg3ml,v4rxnsbml`)

| row (sls_2d_p150 / sls_1d_p150) | lead ml 1024 | (b) ml 1024 | Δ | lead ml 4352 | (b) ml 4352 | Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3-hop E/W | 178.3 | 144.8 | -33.5 | 179.9 | 148.9 | -31.0 |
| turn (SlsTurn4) | 171.0 | 144.7 | -26.3 | 171.7 | 150.9 | -20.8 |
| multicast | 463.1 | 424.0 | -39.1 | 472.4 | 429.0 | -43.4 |
| row merge | 136.4 | 125.2 | -11.2 | 145.2 | 134.5 | -10.7 |
| 1-hop E/W, E/W bidir, N/S | | | 0.0 | | | 0.0 |
| 1D 3-hop (single-downstream: (b) vs the lead) | 118.2 | 118.9 | +0.7 | 124.5 | 131.0 | **+6.6** |
| 1D row merge | 108.9 | 108.9 | 0.0 | 125.8 | 128.7 | +2.9 |
| 1D 1-hop | 95.5 | 95.5 | 0.0 | 124.5 | 124.5 | 0.0 |

Passes agree within 0.1. The 3-hop E/W proxy at 1024 B lands at 144.8, where the production E/W lead sits (E0-bound,
V4_NS_FLOOR §2). Attribution: the bundle (user decision 1).

PC samples of the forwarding E1 (chip 2 / channel 4, hop3 at 4352 B, one sampled run each: lead ml 179.67, (b) ml
148.73; `pcs_lines.py`; cycles/packet = share x rate, within E1 only):

| inline frame | lead ml | (b) ml |
| --- | ---: | ---: |
| command-buffer register writes (NOC_CMD_BUF_WRITE_REG) | 21.6 | 12.6 |
| the stateless credit (noc_fast_default_write_dw_inline) | 22.5 | - |
| the stateless data write (write_one_packet_with_trid_impl + ncrisc_noc_fast_write) | 14.5 | - |
| forward_partially_preset (6.8 of it at the joint ready test's branch, where the paired reads' latency lands) | - | 7.6 |
| the new inline credit (noc_fast_write_dw_inline_to_node_with_state) | - | 2.1 |
| admission (any_without_free_slot) | 17.2 | 18.7 |

**Surprise (single-downstream 1D at 4352 B, +6.6):** a size-dependent cost, not the counted +1 instruction (at 1024 B
it is +0.7). Hypothesis (not tested): the joint ready test makes the data command wait for the credit buffer too;
with 4352-B writes saturating the NoC path, the previous doorbell's command buffer can stay busy under backpressure,
and the next data write is serialized behind it (the lead waits for the credit buffer only just before the credit).
Proposed isolating A/B (reported, not built): (b) with only the ready test changed back to the lead's pattern.

### 8.3 Probe designs for the 1D 4352-B cost (for the user's review; nothing built)

The +6.6 on the 1D 3-hop row at 4352 B (§8.2) is over the agreed 1-2 cycles for (b), so it goes to the user. The
hypothesis (untested; the 2D proxy profile does not show the slow-path loops, and there is no 1D profile yet): the
joint ready test makes each data write wait for the credit buffer as well; under the NoC backpressure of 4352-B
writes the previous doorbell's command buffer can still be busy, so the data write is serialized behind it. The
lead's full preset waits for the credit buffer only just before the credit. Two probes, each a single change on
v4rxnsb (only `forward_partially_preset`'s ready handling; everything else, the trims included, identical):

**nsbr: the lead's ready-test pattern.** Both reads still issued together; each buffer waited for just before its own
command.

```diff
     const bool data_ready = noc_cmd_buf_ready(NOC, DATA_CMD_BUF);
     const bool credit_ready = noc_cmd_buf_ready(NOC, CREDIT_CMD_BUF);
-    if (!(data_ready & credit_ready)) {
-        while (!noc_cmd_buf_ready(NOC, DATA_CMD_BUF));
-        while (!noc_cmd_buf_ready(NOC, CREDIT_CMD_BUF));
-    }
+    if (!data_ready) {
+        while (!noc_cmd_buf_ready(NOC, DATA_CMD_BUF));
+    }
     ncrisc_noc_set_transaction_id</*skip_cmdbuf_chk=*/true>(NOC, DATA_CMD_BUF, trid);
     noc_write_to_node_with_state<...>(NOC, src_addr, node, dst, size_bytes);
+    if (!credit_ready) {
+        while (!noc_cmd_buf_ready(NOC, CREDIT_CMD_BUF));
+    }
     noc_inline_dw_write_to_node_with_state<...>(node, credit_register, CREDIT_CMD_BUF, NOC);
```

- Isolates: whether the data write waiting for the credit buffer is the cost (the coupling), with the paired reads
  kept.
- Static expectation (from §3's first-round counts, where this pattern measured 32-33 executed instructions against the
  joint test's 27-28, both before the trims): about +5 per forward over (b) as built, so about 29-30, i.e. +6 over the
  lead's full preset on single-downstream routers; register writes unchanged (9); the two counter updates no longer
  merge (+3 of the 5). To be confirmed by a build if approved.
- Known headroom risk (§3.3, first-draft column): `credit_ready` moved to the stack on 52 Galaxy images (glx_express
  20/28, glx_express_line 28/44, glx_torus 4/8); bh_torus passed. Expect the same. **WRONG (§8.6, built): with the
  trims, this pattern passes the headroom check on every production family, the Galaxy express / torus ones and
  bh_torus included (the first draft spilled `credit_ready` with the 64-bit address and the second coordinate load
  still live).**

**nsbs: serial polls.** No ready state read ahead; each command polls its own buffer (the API's built-in wait).

```diff
-    const bool data_ready = noc_cmd_buf_ready(NOC, DATA_CMD_BUF);
-    const bool credit_ready = noc_cmd_buf_ready(NOC, CREDIT_CMD_BUF);
-    if (!(data_ready & credit_ready)) {
-        while (!noc_cmd_buf_ready(NOC, DATA_CMD_BUF));
-        while (!noc_cmd_buf_ready(NOC, CREDIT_CMD_BUF));
-    }
-    ncrisc_noc_set_transaction_id</*skip_cmdbuf_chk=*/true>(NOC, DATA_CMD_BUF, trid);
+    ncrisc_noc_set_transaction_id</*skip_cmdbuf_chk=*/false>(NOC, DATA_CMD_BUF, trid);
     noc_write_to_node_with_state<...>(NOC, src_addr, node, dst, size_bytes);
-    noc_inline_dw_write_to_node_with_state</*update_counter=*/false, /*posted=*/true, /*skip_cmdbuf_chk=*/true>(
+    noc_inline_dw_write_to_node_with_state</*update_counter=*/false, /*posted=*/true, /*skip_cmdbuf_chk=*/false>(
         node, credit_register, CREDIT_CMD_BUF, NOC);
```

- Isolates: the same coupling with nothing held across the data write, but also gives up the paired reads (each
  poll's latency lands at its own first use, serially). Read together with nsbr: if both remove the 4352-B cost, the
  coupling is the cause; if only nsbr does, the serialized second read costs about as much as it saves.
- Static expectation (the quick build in §3.3, before the trims: 31 executed instructions, both polls executed once,
  against the joint test's 27): about +3-4 over (b) as built, so about 27-28, +3-4 over the lead's full preset;
  register writes 9; one counter update.
- Known headroom risk (§3.3's quick build of this form): every Galaxy family passes; bh_torus N/S fails 4/12 as with
  the joint test (the accepted delivery spill). **WRONG for bh_torus (§8.6, built with the trims): 12/12 pass.**

**Rows, if approved (after the sender's round):** lead / nsb / nsbr / nsbs on lin3, the sls 1D set (Lin3HopEW, row
merge at 1024 and 4352 B), hop3 and ring1, two passes reversed; plus one PC sample of lin3's forwarding E1 on nsb and
on the lead (`bnd_pcs_runs.sh ... 1D lin3`), the 1D profile the hypothesis lacks. A lin3 PC sample could ride in the
held nfka job when it is resubmitted (about +90 s: `bash $C/bnd_pcs_runs.sh fabric_exp/variants/v4rxnsb nfslb 1D lin3
&& bash $C/bnd_pcs_runs.sh fabric_exp/variants/v4rxg3 nfslg3 1D lin3`); jobs 493 / 494 cannot take it (already
queued with fixed commands).

### 8.4 What (b) changes on a 1D router (so what the probes can isolate)

From the code (§1.2, §1.3 against the lead's stateful forward and `preset_downstream_cmd_bufs`), on a 1D line router:
- **Destination node per forward:** the data write adds RET_ADDR_COORDINATE, the credit adds TARG_ADDR_LO and
  TARG_ADDR_COORDINATE (the lead writes neither on a line: `forwards_set_credit_register` is false without deadlock
  avoidance), plus the load of `link.noc_coord`.
- **The joint ready test** (the data write also waits for the credit buffer; its slow path).
- **Code generation only:** the two posted-counter updates merge into one; the receiver loop is 2 instructions shorter
  (-O3); register allocation differs.
- **The same as the lead:** command buffers (data 2, credit 1), NoC VC (EDM_NOC_VC) and posting (posted data and
  credit) in the control words, AT_DATA (-1), byte enables, RET_ADDR_MID / TARG_ADDR_MID (0), the transaction ID write,
  the counters' totals, the presets' timing (after the opens). The init presets carry a placeholder node until the
  first forward writes its own.
So nsbr (only the ready test changed back) separates the joint test from the per-forward node writes; nsbs separates
it again with nothing held.

### 8.5 The decision rows (jobs 493 / 494, passes a / b in reversed router order; lead / (a) / (b); REPS 1)

| row (4352 B) | v4rxg3 | v4rxnsa | v4rxnsb |
| --- | ---: | ---: | ---: |
| lin3 (1D 3-hop) | 124.4 | 124.4 | **130.3** |
| ring1 | 140.9 / 140.8 | 140.8 | 140.8 |
| halfring | 173.0 | 173.0 | 172.4 |
| hop3 (2D E/W 3-hop) | 146.7 | 146.7 | 147.0 |
| ew4352 | 127.0 | 127.0 | 127.0 |
| ewscatter | 142.2 | 142.3 | 142.3 |
| VC2 on: hop3 / ew4352 | 173.3 / 134.7 | 173.3 / 134.7 | 173.5 / 134.7 |

Both passes give the same value on every row (ring1's lead 140.9 / 140.8 aside). lin3 reproduces the sls 1D cost
(+5.9 here, +6.6 in §8.2); every other row is within ±0.6 for (b) and ±0.1 for (a). PC samples of hop3's forwarding E1 (chip 2 / channel 4,
within E1 only; lead 146.70, (b) 147.02): in (b) the joint test's branch holds 5.0 % (7.4 cycles/packet), its
data-buffer wait loop 1.3 % (1.9) and its credit-buffer wait loop 1.3 % (1.9): the data write does wait for the
credit buffer at 4352 B on a single-downstream 2D router, which fits the hypothesis (not a test of it: hop3 is
E0-bound, and the wait could be absorbed elsewhere).

### 8.6 The probes, built (offline; user approval 2026-11-02; device rows after the sender's round)

`variants/v4rxnsbr`, `variants/v4rxnsbs` = v4rxnsb with §8.3's single change each (`loopvars_maps/ns_presets/
probes.py` reproduces both; checked by diff). Full coverage -O3 and -Os: 2,863 / 2,863 build; against v4rxnsb, 954
images differ in each (the forwarding E1 loops; (b)'s 50 init-only images are identical); every E0 and one-ERISC
image identical to the lead. Swap guard clean.

| | executed per forward (1D / 2D E/W / N/S / express) | register writes | counter stores | loads | expectation (§8.3) |
| --- | --- | ---: | ---: | ---: | --- |
| (b) v4rxnsb | 24 / 25 / 24 / 24 | 9 | 1 | 2 | |
| nsbr | 29 / 30 / 29 / 31 | 9 | 2 (3 on express) | 3 (4 on express) | 29-30: met |
| nsbs | 28 / 28 / 28 / 28 (both polls run once) | 9 | 2 | 3 | 27-28: met |
| lead full preset | 23 / 24 / - / - | 6 | 2 | 2 | |

Headroom (`loopvars.py --headroom`, v4rxg3 -> probe, every image): **nsbr passes on every family**, bh_torus 12/12 and
the Galaxy express / torus ones included, even bh_2d_codeprof; **nsbs passes on every production family**, bh_torus
12/12 included (bh_2d_codeprof 20/28). Both are better than (b) as built (bh_torus 8/12). On the express images nsbr
keeps one value in memory (counter stores 3, loads 4; +1 / +1 stack accesses in the loop, not classed hot).

Loop size (-O3, probe - lead): nsbr 1D +6, 2D E/W +6, ring +2, N/S 2 live -18, VC1 -28 to -90, Galaxy express multi
-42 / -54; nsbs 1D -5, 2D E/W -5, ring -9, N/S -50, VC1 -50 to -155, express multi -69 / -92.
Size (-O3, worst E0+E1 over the 54 cases, bh_vc1): lead 22,996, (b) 22,536, nsbr 22,780, nsbs 22,492.

**Rows (ready to queue; the coordinator queues them after the sender's round):**

```
# pass a
cd /home/aagarwal/tt-metal && C=fabric_exp/consolidation && R='v4rxg3 v4rxnsb v4rxnsbr v4rxnsbs' && ROUTERS="$R" REPS=1 bash $C/row_ab_runs.sh nfra lin3 hop3 ring1 && ARMS=base bash $C/sls_sl_runs.sh nfra v4rxg3 v4rxnsb v4rxnsbr v4rxnsbs
# pass b (reversed)
cd /home/aagarwal/tt-metal && C=fabric_exp/consolidation && R='v4rxnsbs v4rxnsbr v4rxnsb v4rxg3' && ROUTERS="$R" REPS=1 bash $C/row_ab_runs.sh nfrb lin3 hop3 ring1 && ARMS=base bash $C/sls_sl_runs.sh nfrb v4rxnsbs v4rxnsbr v4rxnsb v4rxg3
```
Each about 650 s (row_ab: 12 runs + 4 compiles; sls: the 2D and 1D sets, ~60 s per router). Reports:
`row_ab_report.py nfra` / `nfrb`, `slsab.py nfra,nfrb v4rxg3,v4rxnsb,v4rxnsbr,v4rxnsbs`.

### 8.7 The probes on the device (jobs 502 / 503, passes a / b reversed; lead / (b) / nsbr / nsbs)

Rows at 4352 B (`row_ab_report.py nfra` / `nfrb`; both passes identical):

| row | lead | (b) nsb | nsbr | nsbs |
| --- | ---: | ---: | ---: | ---: |
| lin3 (1D 3-hop) | 124.4 | 130.3 | 126.4 | 132.4 |
| hop3 (2D E/W 3-hop, E0-bound) | 146.7 | 147.0 | 146.9 | 147.0 |
| ring1 | 140.9 | 140.8 | 140.8 | 140.8 |

sls sets (`slsab.py nfra,nfrb v4rxg3,v4rxnsb,v4rxnsbr,v4rxnsbs`; Δ against the lead, mean of the passes, which agree
within 0.3):

| row | size | lead | Δ (b) | Δ nsbr | Δ nsbs |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1D 3-hop | 1024 | 118.1 | +0.7 | **+5.8** | **+13.9** |
| 1D 3-hop | 4352 | 124.5 | **+6.5** | +2.2 | **+7.9** |
| 1D row merge | 1024 | 108.9 | 0.0 | +2.0 | +4.6 |
| 1D row merge | 4352 | 125.8 | +2.9 | +1.2 | +3.4 |
| 2D multicast | 1024 | 419.2 | +0.4 | +2.6 | +5.8 |
| 2D multicast | 4352 | 427.0 | +0.6 | 0.0 | +1.8 |
| 2D turn | 4352 | 147.8 | +1.7 | +0.8 | +2.9 |
| 2D 3-hop E/W | 4352 | 145.4 | +0.5 | +0.2 | +0.6 |
| 1-hop rows (1D, 2D) | both | | 0.0 | 0.0 | 0.0 |

What these single-change A/Bs say (nsbr and nsbs each differ from nsb in the ready handling only):
- **The joint ready test causes most of (b)'s 4352-B cost on 1D** (A/B: nsb -> nsbr, 131.0 -> 126.7 on the 1D 3-hop at
  4352 B, -4.3; the hypothesis of §8.2-§8.3 holds for that part). Why the coupling costs at 4352 B and not at 1024 B
  (backpressure on the credit buffer) stays a hypothesis.
- **The lead's pattern costs at 1024 B** (A/B: nsb -> nsbr, 118.9 -> 124.0 on the 1D 3-hop, +5.1; multicast +2.2). The
  1D 3-hop at 1024 B is E1-bound, and nsbr executes 5 more instructions per forward than nsb (29 against 24: the two
  counter updates no longer merge, two more branches): about one cycle per counted instruction there (hypothesis on
  the mechanism; the size of the effect is measured).
- **Serial polls cost most** (A/B: nsb -> nsbs, +13.2 on the 1D 3-hop at 1024 B, +1.4 at 4352 B): each poll's read
  latency lands at its first use, in series (hypothesis on the mechanism).
- **Against the lead, no wait form keeps (b) within 1-2 cycles on 1D at both sizes:** (b) +0.7 / +6.5, nsbr +5.8 /
  +2.2, nsbs +13.9 / +7.9 (1024 / 4352). What all three share is the per-forward node writes on a single-downstream
  router (3 register writes, 1 load); on the E1-bound 1D 3-hop at 1024 B nsbr's +5.8 matches its +6 counted
  instructions over the lead (hypothesis: the node writes plus the lead pattern's own cost).
- 2D single-downstream rows mostly hide it (E0-bound): hop3 +0.3 at most; the turn row at 4352 B +0.8 to +2.9.

### 8.8 The probes on the multi-downstream proxy, and the production rows for (a) / (b) (jobs 504-506, 511)

Proxy (job 506, both passes in one job, reversed; `slsab.py nfqa,nfqb v4rxg3ml,v4rxnsbml,v4rxnsbrml,v4rxnsbsml`; Δ
against the lead ml, passes within 0.2):

| row | size | lead ml | Δ nsb ml (joint) | Δ nsbr ml | Δ nsbs ml |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3-hop E/W | 1024 | 178.3 | **-33.5** | -30.4 | -23.4 |
| 3-hop E/W | 4352 | 179.9 | **-31.0** | -28.7 | -23.1 |
| turn | 1024 | 171.0 | **-26.4** | -26.2 | -22.6 |
| turn | 4352 | 171.7 | **-20.8** | -19.9 | -16.9 |
| multicast | 1024 | 463.2 | **-38.9** | -37.2 | -34.4 |
| multicast | 4352 | 472.4 | **-43.1** | -41.0 | -33.4 |
| row merge | 1024 / 4352 | 136.4 / 145.2 | **-11.2 / -10.7** | -9.7 / -9.5 | -7.6 / -7.5 |

On multi-downstream routers the joint ready test is the best of the three at both sizes (A/B: nsb ml -> nsbr ml,
+2.3 to +3.1 on the 3-hop row; -> nsbs ml, +8 to +10). The proxy's 4352-B rows sitting ~4 above its 1024-B rows is
not the coupling: the lead's pattern does not close that gap.

Production rows, (a) and (b) against the lead (jobs 504 / 511 for passes a / b, reversed; census codec suites,
VC1 census, torus rows):
- census 4352 suite mean: lead 143.3, (b) 143.6, (a) 143.4; largest: multicast scatter 173.1 -> (b) 175.5, (a) 174.0;
  3-hop E/W write 146.9 -> (b) 147.2, (a) 146.8. mix suite mean 127.2 for all three. FWD_MixCmd is bimodal run to run
  (172.2-172.3 or 212.5-212.7 for the lead and (b) alike, 212.5 for (a) in both passes): not a router effect.
- VC1 (every VC1 forwarder is multi-downstream; (a) = (b)): 2-hop 239.2 -> 231.7 (16 B) and 258.1 -> 254.7 (4352 B);
  inter-mesh to mesh 1 chip 3: 375.4 -> 359.0 (16 B), 413.2 -> 397.7 (4352 B); vc1i mean 336.8 -> 331.6; vc1 mean
  159.1 -> 158.1.
- torusns 149.2 / torusns_scatter 191.0: unchanged for both (the accepted delivery spill does not show here).
- PC samples of lin3's forwarding E1 (chip 2 / channel 4; lead 124.41, (b) 130.30; within E1 only): (b)'s joint
  test's branch 5.6 % (7.3 cycles/packet), its data and credit wait loops 1.0 % each; the lead waits for the credit
  buffer too (its wait line 1.3 %), but after the data command was issued.

### 8.9 The Galaxy N/S rows (Slurm job 169615, bh-glx-110-a10u14, both passes on that node)

`consolidation/ns_glx.sbatch` (a trimmed copy of the slot study's sls_glx2k.sbatch: node guard, duplicate-label check,
a main_fw arm; dry-run with stubs first: normal, hang, node-guard and duplicate-label paths). Trees from base_1102 via
mk_glx_tree.sh: nsf_g3 (v4rxg3), nsf_nsa (v4rxnsa; its multi-downstream images are v4rxnsf2's, byte-identical);
main_fw on base_1102 (kernel 1, flags 0, firmware keepalive, the builder's opt level). The whole Galaxy as one 8x4 mesh,
VC0, default slots (2 KiB: 41,32,32; 4352 B: 17,16,16; logged). Passes a (lead, (a), main) and b (main, (a), lead); every
step 0 failed; largest pass spread 0.9 (multicast), N/S rows within 0.3. `sls2k_report.py fabric_exp/results/glx169615
2d`. Cycles/packet:

| row | slots | payload | main_fw | lead | (a) / v4rxnsf2 | Δ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 3-hop N/S | 2 KiB | 1024 / 2048 | 227.2 / 226.7 | 172.9 / 172.9 | **136.8 / 136.8** | -36.1 / -36.1 |
| 7-hop N/S | 2 KiB | 1024 / 2048 | 227.1 / 226.7 | 173.5 / 173.7 | **137.4 / 137.5** | -36.1 / -36.2 |
| 10-hop corner | 2 KiB | 1024 / 2048 | 227.2 / 226.6 | 173.6 / 173.7 | **140.2 / 139.8** | -33.5 / -33.9 |
| turn (1 N/S + 3 E/W) | 2 KiB | 1024 / 2048 | 227.1 / 226.6 | 170.9 / 171.1 | **140.0 / 139.6** | -31.0 / -31.5 |
| column merge | 2 KiB | 1024 / 2048 | 199.3 / 198.9 | 159.1 / 163.6 | **133.2 / 137.6** | -25.9 / -25.9 |
| 3-hop E/W (control) | 2 KiB | 1024 / 2048 | 227.4 / 226.6 | 140.3 / 139.6 | 140.3 / 139.5 | 0.0 |
| 1-hop E/W, N/S, bidir, row merge | 2 KiB | both | | | | ±0.3 |
| 3-hop N/S | 4352 B | 1024 / 2048 | 222.8 / 224.3 | 182.7 / 183.2 | 152.6 / 157.2 | -30.1 / -26.0 |
| 7-hop N/S | 4352 B | 1024 / 2048 | 222.7 / 224.3 | 186.0 / 189.8 | 166.8 / 173.1 | -19.2 / -16.7 |
| 10-hop corner | 4352 B | 1024 / 2048 | 222.7 / 224.3 | 186.7 / 188.8 | 167.1 / 171.4 | -19.5 / -17.4 |
| turn | 4352 B | 1024 / 2048 | 222.7 / 224.3 | 178.6 / 179.2 | 145.1 / 146.6 | -33.5 / -32.6 |
| column merge | 4352 B | 1024 / 2048 | 197.8 / 197.3 | 166.8 / 170.4 | 146.3 / 150.2 | -20.5 / -20.2 |
| every row at 4352 B payload | 4352 B | 4352 | 251.1 | 252.1 | 252.1 | 0.0 (the wire) |
| E/W and 1-hop rows | 4352 B | 1024 / 2048 | | | | ±0.1 |

- At 2 KiB slots the N/S rows (136.8-140.2) now sit at the 3-hop E/W row's level (139.5-140.3): the N/S floor of
  V4_NS_FLOOR §1 (172.5-173.3) is gone. Which side bounds them now is not measured here (no one-side arms: the g3 and
  (a) trees carry no delay hooks); that they match the E0-bound E/W row is a hypothesis for E0.
- At 4352 B slots (16 forwarding slots) the 7-hop and corner rows stay ~14-21 above the 3-hop N/S row: the window
  remnant V4_NS_FLOOR §1 described (a slot-allocation question, SLOT_STUDY), now on top of a lower E1 floor.
- Against main_fw (the shipped router, same trees' host): N/S 2 KiB-slot rows 227 -> 137-140.

## 9. Design sketch: one forward type chosen at compile time (user question, 2026-11-02; no build)

The downstream count is a compile-time fact, so the forward strategy can be a type picked once per VC, with one
interface, instead of the lead's predicates threaded through `forward` and `preset_downstream_cmd_bufs`. In effect
(a), in a readable shape.

```cpp
// How a VC's forwards use their command buffers, fixed at compile time by the router's downstreams.
//   FullPresetForwards: one destination for the whole run (one live downstream, never picked at run time). The buffers
//     keep the destination; a forward writes its slot, length and go (the lead's stateful forward, verbatim).
//   PartialPresetForwards: several destinations. The buffers keep what no forward changes; a forward writes its
//     destination too (§1.2).
//   StatelessForwards: no presets (one ERISC, DPRINT / watcher, UDM). Every forward writes its whole commands.
// One interface: preset(args), once after the opens; forward<ENTRY>(link, dst, src_addr, size_bytes, trid).
template <size_t VC>
using ForwardCommands = std::conditional_t<
    !forwards_can_preset,
    StatelessForwards<VC>,
    std::conditional_t<forwards_have_one_destination, FullPresetForwards<VC>, PartialPresetForwards<VC>>>;
```

Each form is a struct of two static functions; `Downstreams<VC>::forward` becomes
`ForwardCommands<VC>::template forward<entry>(links[entry], dst, src_addr, size_bytes, trid)`, init calls
`ForwardCommands<0>::preset(args.vc0())`, and `Relay::forward` stays `forward_stateless` (decision 4).

**Predicates that remain** (named facts, against the lead's nine: `command_buffers_can_keep_presets`,
`uses_both_vc0_forward_cmd_bufs`, `uses_a_vc0_forward_cmd_buf`, `rides_vc0_preset`, `clobbers_vc0_preset`,
`vc0_keeps_forward_preset`, `forwards_are_stateful`, `another_vc_rides_vc0_preset`, `forwards_set_credit_register`):
- `forwards_can_preset`: not DPRINT / watcher, two ERISCs, not FORCE_ALL_PATHS_TO_USE_SAME_NOC, not UDM, the buffers
  consistent (§1.2's rule);
- `forwards_have_one_destination`: VC0 has one live downstream, the router is neither intermesh nor on an edge, and
  VC1 (where present) shares VC0's buffers and live downstreams (the lead's "rides"); otherwise every VC is partial,
  so no form overwrites another's preset;
- inside `FullPresetForwards`: `credit_register_varies` = deadlock avoidance or VC1 present (the lead's
  `forwards_set_credit_register`).

**Size (estimate, not built):** about 570-590 lines (the lead's full-preset forward and per-destination preset kept
verbatim inside one struct, (b)'s partial forward and preset in another, the selection and three facts), against (a)
632, (b) 503, the lead 532 (as run, §7). The code of both shapes is inherent; what changes is that it is chosen in one place.

**Byte identity (expected; a build would confirm):** single-downstream images can be byte-identical to the lead if
`FullPresetForwards` keeps the lead's forward block and its out-of-line preset verbatim, as (a) showed (0 of its 610
single-downstream images changed). Multi-downstream images would equal (a)'s and (b)'s (identical to each other).

**Does the joint-wait question still matter?** For single-downstream routers no (they keep the lead's pattern, so
lin3's cost is gone by construction, as in (a)). For multi-downstream routers yes: the partial form's ready handling
is the open piece. The proxy's 4352-B rows sit ~4 above its 1024-B rows (3-hop 148.9 vs 144.8; the production E/W lead
differs by +0.9, 145.6 vs 144.7), so multi-downstream routers might carry the same coupling cost (hypothesis). The
probes test it on single-downstream rows; the direct test for multi-downstream routers is the proxy at 4352 B with
the probe forms (v4rxnsbml with nsbr's / nsbs's ready handling), the sls 2D set, two arms.

### 9.1 Built: v4rxnsf (user approval 2026-11-03; offline)

`variants/v4rxnsf` = v4rxg3 + the API patch + `loopvars_maps/ns_presets/stage_f.py` (reproduces it): the three forms
as structs (`StatelessForwards`, `FullPresetForwards`, `PartialPresetForwards`), each with `preset(args)` (out of line)
and `forward<ENTRY>(link, dst, src_addr, size_bytes, trid)`; `ForwardCommands<VC>` chooses one with
`std::conditional_t`; `Downstreams<VC>::forward` makes one call, `ForwardCommands<VC>::template forward<entry>(...)`,
with no branch; `fabric_erisc_router_v4.cpp` calls `ForwardCommands<0>::preset(args.vc0())`; `Relay::forward` stays
`forward_stateless`. `DownstreamArgs` and its two coordinate helpers moved up, ahead of the forms whose presets read
them.

The open details, as settled:
- **The VC1 check:** `can_share_vc0_destination<VC>()`: true when VC forwards to no router here, or uses VC0's two
  buffers and VC0's live downstreams. A template, so the absent VC1's traits are never instantiated.
  `forwards_have_one_destination` = not intermesh, not on an edge, VC0 one live downstream, and
  `can_share_vc0_destination<1>()`.
- **`credit_register_varies`** is a private static member of `FullPresetForwards` (= deadlock avoidance or VC1
  present: within the full form, a present VC1 always shares the preset, so its credit registers differ).
- The facts in the file: `vc_forwards_to_routers` (existing), `forward_cmd_bufs_consistent<VC>`,
  `forwards_can_preset`, `can_share_vc0_destination<VC>`, `forwards_have_one_destination`, and the private
  `credit_register_varies`, against the lead's nine predicates.

Results (full coverage, 2,863 buildable configs, -O3 and -Os; swap guard clean):
- **Byte-identical to v4rxnsa on every image** at -O3 and at -Os (2,863 / 2,863 each). So, against v4rxg3: 2,469
  identical (every single-downstream image, every E0, every one-ERISC and UDM image) and the same 394 multi-downstream
  E1 images as (a) differ.
- **VC1-can't-share images:** none in the coverage set. Where (a) and this design would differ (VC1 present, VC0 with one
  live downstream, not on an edge or intermesh, VC1 not sharing VC0's destination: (a) keeps VC0 full and VC1 stateless,
  this design makes both partial), the identity above would show it; every VC1 router the coverage captures is an
  edge or intermesh router, already partial in both.
- Guard, headroom, size: equal to (a)'s by construction (byte-identical images): §7 / §8.6's (a) numbers (headroom:
  bh_torus N/S 8/12, the accepted delivery spill; -O3 worst E0+E1 22,536, -Os 21,380).
- router_downstream.hpp: **618 lines** (the lead 532, (a) 632, (b) 503). Forward shapes: three behind one interface;
  the lead had two shapes threaded through `forward` and `preset_downstream_cmd_bufs` by nine predicates, (a) three
  shapes the same way.

### 9.2 Validation of v4rxnsf (jobs 513 / 514, queued by the coordinator; P150)

`variant_val_runs.sh v4rxnsf` 13/13, failed: none (two ERISCs: mesh 96/96, val1d 143/143, ring 19/19, Y-torus 12/12, mux
19/19, VC2 10/10, NeighborExchange 7/7, channel-trimming gtests 63 passed (3 skipped); one ERISC: mesh 96/96, ring 19/19,
VC2 10/10, NeighborExchange 7/7, channel trimming 63 passed (3 skipped)); `rxo0_inline_runs.sh` inline write then
flushed atomic 1D 2/2, 2D 2/2; `rxo_udm_runs.sh` UDM unit 8/8, nightly 68/68. No hang, tree restored.
Performance: every image is byte-identical to v4rxnsa, so (a)'s measured rows (§8.5-§8.8) are v4rxnsf's.

### 9.3 Trim 1: v4rxnsf2 (user, 2026-11-03)

`variants/v4rxnsf2` = v4rxnsf with `PartialPresetForwards::preset` presetting VC0's pair only, a `static_assert` that
VC1 is absent or on VC0's two buffers (the host's buffer ids own that; the comment says a VC given buffers of its own
would be preset there, so the 1D multi-line VC work starts at that assert if it gives VC1 its own buffers), and the
`preset_vc<FV>` helper gone; a small helper `uses_vc0_forward_cmd_bufs<VC>()` (a template, so an absent VC1's traits
are never instantiated) serves the assert and `can_share_vc0_destination`. `stage_f2.py` reproduces it.
- **Byte-identical to v4rxnsf on all 2,863 images at -O3 and at -Os**; v4rxnsf's validation covers it.
- router_downstream.hpp: **621 lines** (v4rxnsf 618): the trim removes the 12-line `preset_vc` but the shared helper
  (+11), the assert's two-line why (+3) and the per-argument wrap of the preset's inline set_state (+5, clang-format)
  add more. It reads better (the assumption is stated, one preset) but is not shorter.

## 10. Files

- Variants: `v4rxnsf2` (the result: ForwardCommands + trim 1); `v4rxnsf` (before trim 1; validated; byte-identical);
  `v4rxnsa` (a), `v4rxnsb` (b), `v4rxnsbml` ((b) on the v4rxg3ml proxy); probes `v4rxnsbr`, `v4rxnsbs` and their proxy
  forms `v4rxnsbrml`, `v4rxnsbsml`. All but v4rxnsf2 are records, not for adoption.
- `loopvars_maps/ns_presets/`: `api_patch.py`, `stage_a.py`, `stage_b.py`, `probes.py`, `stage_f.py`, `stage_f2.py`
  (each reproduces its variant), `fwdcount.py`, `funccmp.py`, `cfgprops.py`, `table.py`, `slsab.py` (the sls_sl_runs
  proxy/1D tables); outputs `fwd_final_*`, `fc_*`, `guard_*`, `headroom_*`, `cmp_*`; `round1/` = §3's numbers;
  `first_draft/` = the first draft's (credit bit tested after the data command).
- `consolidation/ns_glx.sbatch` (the Galaxy N/S job); yamls `/data/aagarwal/frs-glx/yamls/nsfix/`; trees
  `/data/aagarwal/frs-glx/trees/nsf_g3`, `nsf_nsa` (base_1102 + the variant's files); results `fabric_exp/results/
  glx169615/`.
- Builds (cov_m): `v4rxnsa`, `v4rxnsb`, `v4rxnsbr`, `v4rxnsbs`, `v4rxnsf`, `v4rxnsf2` at -O3 / -Os; `v4rxnsbml`,
  `v4rxnsbrml`, `v4rxnsbsml` -O3 (bh_2d, glx_2d).
- P150 jobs: 481 (correctness, proxy), 493 / 494 (decision rows), 502 / 503 (probes), 504 / 511 (codec, VC1, torus, lin3
  PC samples), 506 (proxy probes), 513 / 514 (v4rxnsf validation). Galaxy: Slurm 169615.
