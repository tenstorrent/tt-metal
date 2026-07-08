# TT-RDMA on Blackhole — Second WH-side Target (Port Design)

**Status:** design / not yet implemented. Scope-setting for bringing TT-RDMA
to Blackhole (BH) as a second chip-side target alongside Wormhole (WH).
**Assumption (from the program direction):** BH is incorporated into TT-RDMA for
**external, Ethernet-based fabrics with the same gateway concept** — external
RoCEv2 ↔ TT-RDMA translated by a BF3/FPGA gateway (`bf3-gateway-design.md`),
*not* wire-native RoCEv2 from the chip. The **wire protocol is unchanged**
(32 B header, opcode set, ethertype `0x1AF6`; `tt-rdma-wire-protocol-v1.md`).
What changes is everything the v1 docs quietly assume is "the WH
`erisc_cmac_simple` path": the FW binary, the L1 map, the MAC register map, and
the reliability layer.

**Grounding:** all BH facts below are from the `bh-erisc` firmware tree
(`bh-erisc` v1.12.0) — `docs/eth_arch_spec.md`, `src/common/api/eth_ss.h`,
`src/common/api/packet_header.h`, `src/common/api/eth_init.cpp`. Where a claim is
an *opportunity* rather than shipped behavior it is marked as such.

---

## 1. Why this is a port, not a recompile

WH TT-RDMA v1 runs as **`erisc_cmac_simple`** — a bespoke, standalone erisc FW
that owns the whole eth core, drives a DWC-XLGMAC "external CMAC" in raw L2 mode,
and implements the RX ring / opcode dispatch / MR table / WQE ring / software
reliability itself. The two **genuine** BH breaks are the **FW ownership model**
and the **MAC IP**; the raw-mode / software-reliability choices actually carry
over unchanged (see the reliability rows — that was my initial error, corrected
below). The full comparison:

| Assumption on WH | Reality on BH |
|---|---|
| One RISC owns the eth core; FW is standalone | **Two RISCs per ETH SS** sharing 512 KB L1; a **`bh-erisc` base FW** owns RISC0 for link training + a mailbox/api-table runtime that must keep running to maintain the link (`eth_arch_spec.md §1.4, §3`). |
| MAC is DWC XLGMAC; PFC via regs 0x70/0x90 | MAC/PCS is a **Rianta RSm410** IP; the flow-control register map is different, and PFC config is still a **`TODO`** in base FW (`eth_init.cpp:538`). |
| TT-RDMA reliability is software (Phase R cumulative-ACK + host retx) | **Same** — TT-RDMA uses **raw mode** on BH too, so software/gateway reliability still applies. HW packet-resend exists but is a TT-proprietary chip-to-chip protocol (see note below). |
| Raw L2 send via `ETH_TXQ_CMD_START_RAW` on the CMAC | Two named paths: **`eth_send_raw`** (no HW resend) and **`eth_send_packet`** (HW resend), both in `eth_ss.h`. WH has the *same* two paths (`ETH_TXQ_CMD[0]`=raw, `[1]`=packet, `eth_ss_regs.h`). |

> **Not a BH-only feature — corrected.** Hardware sequence-number packet-resend
> ARQ exists on **both** WH and BH. On WH it is `ETH_TXQ_CMD[1]` "packet transfer
> (with resend)" + `ETH_TXQ_CTRL[0]` resend-enable + `ETH_TXQ_REMOTE_SEQ_TIMEOUT`
> (`budabackend-master/.../erisc/src/api/eth_ss_regs.h`); on BH it is
> `eth_send_packet` + the 128-entry Go-back-N window (`eth_arch_spec.md §1.5.3`).
> **TT-RDMA does not use it on either chip** because it deliberately runs in
> **raw mode** (`ETH_TXQ_CMD_START_RAW` + RXQ raw + BUF_WRAP) — HW resend is a
> both-sides-must-be-TT protocol that a Mellanox NIC or FPGA gateway cannot
> speak. Raw mode is what makes TT-RDMA NIC-agnostic; the price is that
> reliability lives in software (Phase R) or the gateway. The genuine BH deltas
> are the ones in the other rows (FW model, Rianta MAC, 400 G, offload), **not**
> ARQ availability.
| ~100 GbE wire; L1 map squeezed into a small erisc SRAM | **400 Gbps** (4×100 SerDes, AlphaWave) into a **512 KB** L1 (`eth_arch_spec.md §1.3`). |

So a BH TT-RDMA "FW" is not `erisc_cmac_simple` rebuilt — it is a **tt-metal
active-eth kernel** (`active_erisc.cc` on RISC0) that coexists with the
`bh-erisc` base FW and drives the Ethernet Controller. The gateway, the SDK
surface, and the wire bytes are the parts that carry over unchanged.

## 2. Architecture delta (BH ETH SS vs WH external CMAC)

| Dimension | WH (`erisc_cmac_simple`) | BH ETH SS (`bh-erisc` + tt-metal active-eth) |
|---|---|---|
| ETH instances / chip | eth cores (external CMAC on a subset) | **14 ETH SS**, each independent, each with own RISCs/L1/MAC/SerDes (`§1.1`) |
| RISCs per instance | 1 (for the CMAC path) | **2** (RISC0 base FW / RISC1 SW), shared L1 (`§1.4`) |
| L1 size | small erisc SRAM (v1 map ends ~0x2B000) | **512 KB**; SW region `0x00000–0x70000`, base FW `0x70000–0x80000` (`§1.4, §3.2`) |
| MAC/PCS | DWC XLGMAC | **Rianta RSm410**, 10G–400G, cut-through RX (`§1.6`) |
| SerDes | (WH SerDes) | **AlphaWave**, 8-lane block shared by 2 tiles, 4 lanes each (`§1.7`) |
| Wire rate | ~100 GbE | **400 Gbps** / link (`§1.3`) |
| TX paths | raw L2 write to TX_BUF | **raw** (`eth_send_raw`, no ARQ) + **packet** (`eth_send_packet`, HW ARQ) + remote-reg-write (`§1.5`) |
| RX | CMAC BUF_WRAP ring, FW-parsed | 3 RX channels, **cut-through**, HW **RX classifier (TCAM)** + packet-edit table (`§1.5.2`, `eth_rx_flow.cpp`) |
| HW packet-resend ARQ | **Yes** — `ETH_TXQ_CMD[1]` packet mode + `ETH_TXQ_CTRL[0]` resend (`eth_ss_regs.h`); unused by TT-RDMA (raw mode) | **Yes** — `eth_send_packet`, 128-entry Go-back-N (`§1.5.3`); also unused by TT-RDMA (raw mode). *Not a BH-vs-WH difference.* |
| Flow control | FW writes DWC regs (PFC doc) | Rianta + controller; PFC currently **TODO** (`eth_init.cpp:538`) |
| Advanced offload | none | HW insert/strip of **VLAN / IPv4 / IPv6 / UDP/TCP / RoCE iCRC**, PTP, 802.3br preemption (`§1.3`) |

## 3. Firmware / integration model on BH

On WH the FW is the whole story. On BH the data path is **layered on top of a
resident base FW that must not be disturbed** (`eth_arch_spec.md §3.7–§3.10`):

1. **Base FW (`bh-erisc` `eth_init`, RISC0)** trains the link and runs a
   `while(true)` mailbox-service loop, exposing `boot_results` (`0x7CC00`),
   `eth_api_table` (`0x7CF00`), and 4 mailboxes (`0x7D000`).
2. **tt-metal active-eth take-over:** host loads `active_erisc.cc` at
   `MEM_AERISC_FIRMWARE_BASE` (inside the SW area below `0x70000`); it takes
   RISC0 and thereafter **periodically yields** to the base FW via
   `service_eth_msg()` through the api-table pointer, so the link stays alive.
   RISC1 runs `subordinate_erisc.cc`.
3. **TT-RDMA data path = an active-eth kernel** that:
   - **TX:** builds the 32 B v1 header + payload in the SW L1 region and issues
     `eth_send_raw(q, src, nbytes, dst_mac)` for external/gateway traffic (raw
     path → no HW ARQ, so software/gateway reliability still applies), or
     `eth_send_packet(...)` for chip-to-chip (HW ARQ).
   - **RX:** points an RX channel's landing region into SW L1 and consumes it;
     uses the **RX classifier TCAM** to match ethertype `0x1AF6` (and drop the
     rest) instead of WH's FW-side opcode filter.
   - **Opcode dispatch / MR table / WQE ring:** the same logic as WH, but
     **re-laid-out inside `0x00000–0x70000`** (512 KB is far larger than the WH
     map, so no address pressure) and **must not touch `0x70000–0x80000`**
     (base FW / boot_results / mailboxes — writing there bricks the link,
     `§3.10`).

**Rule inherited from BH (corrected — it is *not* a 1000 ms hard watchdog):**
under tt-metal, the base-FW main loop is dormant; its services run only when the
active kernel calls `internal_::risc_context_switch()`
(`erisc.h`), which does `service_eth_msg()` **every call** plus a
`update_boot_results_eth_link_status_check()` that is **self-debounced to
`ETH_UPDATE_LINK_STATUS_INTERVAL_MS = 1000 ms`** (`eth_fw_api.h:229,338`). Three
things follow, none of which is a deadline that kills a healthy link:

- **`service_eth_msg()` is not debounced** → host/CMFW/RISC1 mailbox calls (e.g.
  a host `is_link_up()`) stall until the next yield. Yield often enough that no
  mailbox client blocks past *its* timeout.
- **The link-status check self-rate-limits to 1000 ms**, and its own code warns
  *"calling this too many times can result in link instability"* — so 1000 ms is
  a **floor, not a ceiling**; yielding faster is fine (the debounce drops the
  extra calls), yielding slower just refreshes telemetry / drives recovery more
  slowly. Native recovery cadence is 200 ms / 5-poll detection
  (`eth_link_recovery_spec.md §8`), but on active-eth it runs at the debounced
  rate via the api-table.
- **Profile L keeps the peer alive in hardware** — packet mode auto-emits
  seq-update keepalives (`LOCAL_SEQ_UPDATE_TIMEOUT`), so the *peer* link is
  maintained regardless of FW yield cadence. Profile E (raw) has no HW keepalive;
  its upper bound is whatever the gateway/switch partner's link timer is.

**No hard watchdog exists — verified on both sides.** Inside bh-erisc, link
maintenance is a poll, not a deadline. In **CMFW** (`tt-system-firmware`,
Zephyr/SMC), the eth heartbeat (`0x7CC70`) and PCS link are read once per
**100 ms** telemetry tick (`lib/tenstorrent/bh_arc/{eth,telemetry}.c`) and packed
into `TAG_ETH_LIVE_STATUS` — **telemetry only; nothing resets, downs, or recovers
an eth core on a stale heartbeat.** CMFW's link-up judgement comes from the **PCS
hardware** (`GetEthLinkStatus`), not the FW heartbeat, so it is unaffected by a
busy RDMA loop. The only remaining bounds are therefore **soft**: mailbox-client
timeouts (host control calls), Profile-E partner link timers, and any **host-side
health policy** that watches the `TAG_ETH_LIVE_STATUS` heartbeat bit (an ops-layer
choice above firmware, not a FW watchdog). See `§10` open-Q #1.

### 3.1 RISC budget (concrete — verified against the tt-metal BH FW in this repo)

The tt-metal active-eth model already **consumes both RISCs** of the ETH tile
(`tt_metal/hw/firmware/src/tt-1xx/{active,subordinate}_erisc.cc`,
`dev_mem_map.h`). TT-RDMA has to fit *inside* that, not replace it:

| Resource | Value (BH active ETH) | Source | Implication for TT-RDMA |
|---|---|---|---|
| RISCs / tile | 2 — **RISC0 = `active_erisc`**, **RISC1 = `subordinate_erisc`** | `eth_arch_spec §3.1` | both already used; RDMA shares them |
| RISC0 role | go-message loop + dispatches eth kernels + **yields to base FW** via `internal_::risc_context_switch()` → `service_eth_msg()` (`active_erisc.cc:102,282`) | code | RISC0 **must keep servicing base FW** — the link dies otherwise |
| RISC1 role | 2nd data mover ("DM1"); runs kernel, drives NoC, **does not** service base FW (`subordinate_erisc.cc:134`) | code | good home for a tight RDMA loop |
| FW code / RISC | **24 KB** (`MEM_ERISC_FIRMWARE_SIZE`) | `dev_mem_map.h:276` | RDMA data-path code must fit in 24 KB |
| Local data / RISC | **8 KB**, RISC0 minus `0x700` for base FW (`MEM_AERISC_LOCAL_SIZE`) | `dev_mem_map.h:329` | keep RDMA state in **L1**, not local mem |
| Shared L1 / tile | **512 KB** (`MEM_ETH_SIZE`); SW region `0x0…MEM_SYSENG_RESERVED_BASE` | `dev_mem_map.h:37,237` | huge vs WH — RX ring / MR table / WQE pool relayout is unconstrained |
| Reserved (do not touch) | top region to `0x80000` (base FW code/data, `boot_results` `0x7CC00`, api-table `0x7CF00`, mailboxes `0x7D000`, NoC counters `0x7D040`) | `eth_arch_spec §3.10`, `dev_mem_map.h:339` | writing here bricks the link |

**The load-bearing consequence:** WH's `erisc_cmac_simple` was a **persistent loop
that owned the whole core**. A BH RDMA data path cannot be — a persistent loop on
RISC0 starves the base-FW link maintenance, and a persistent loop on RISC1 can't
service base FW at all (only RISC0 can). Two viable shapes:

1. **RDMA tight TX/RX loop on RISC1** (`subordinate`), RISC0 (`active`) keeps the
   go-message + base-FW-yield duty and shuttles doorbells/completions — mirrors
   the WH single-loop design while offloading link housekeeping to the peer RISC.
2. **RDMA loop on RISC0** with a **mandatory periodic `service_eth_msg()`** woven
   into the drain loop (every ≤~1 ms). Simpler, but every RDMA stall now also
   risks the link-health timer.

**NoC contention (answering the original open-Q):** the RDMA path is NoC-heavy
(inbound WRITE → `noc_write` to a NoC address; TX DMA-reads from L1/DRAM). On BH
it shares the tile NIU **three ways** — Tensix workload, the RDMA path, *and* the
base FW's periodic dynamic-NoC/link-recovery use — versus WH's two-way (erisc +
Tensix). Not a blocker (NoC is arbitrated), but the base-FW share is new and the
RDMA loop must assume the NIU can be busy with link-maintenance traffic it
doesn't control.

## 4. What carries over unchanged

- **Wire protocol** — 32 B header, opcode family ranges, `0x1AF6`
  (`tt-rdma-wire-protocol-v1.md`). A BH endpoint and a WH endpoint are wire-
  compatible with the same gateway.
- **Gateway** — `bf3-gateway-design.md` + the outbound/mesh work
  (`tt-rdma-mesh-addressing-spec.md`) are chip-agnostic; the gateway translates
  RoCEv2 ↔ v1 regardless of which TT chip terminates the `0x1AF6` link.
- **Host SDK surface** — `TtRdmaEndpoint` (`tt-rdma-host-sdk.md`) and the
  `QpHandle`/`RemoteMr` model (`mesh-addressing-spec §5`) are the same API; only
  the `bring_up()` path and the RCB/MR addresses behind it are BH-specific.
- **rkey / MR semantics, addressing model** — rkey-routes-one-sided,
  tag-as-QPN-routes-two-sided (`mesh-addressing-spec §2`) is a wire/gateway
  property, not a chip property.

## 5. The Blackhole opportunity (offload) — and its limits

BH's Ethernet Controller can, **in hardware**, insert/strip VLAN, IPv4/IPv6,
UDP/TCP and **RoCE iCRC**, and the RX classifier already enumerates `ROCEV2`,
`IPV4_UDP`, and `PFC` packet types (`packet_header.h` `eth_pkt_type_e`); the FW
tree even carries rxe-derived RoCE header/opcode definitions (`rxe_bth`,
`rxe_reth`, `rxe_aeth`, `RXE_*_MASK`, `IB_UDP_DPORT 4791`). Two things follow:

- **Opportunity:** on BH the gateway could be *thinner*, or a BH node could emit
  frames much closer to RoCEv2 (HW-inserted UDP/IP/iCRC) — shrinking the
  translation the gateway does today. The `header_cksum`→gateway-CRC concession
  in the `0x1AF7` plan (`mesh-addressing-spec §7.3`) is less pressing when the
  MAC can compute iCRC in hardware.
- **Limit — the offload is a 3-tier reality (verified in bh-erisc v1.12.0):**
  1. **Real silicon** — the registers exist: `ETH_CTRL_A_ROCE_ICRC_{CTRL,TX_INIT,
     RX_INIT,RX_RECEIVED,RX_CALCULATED}` + 21× `ROCE_ICRC_HDR_MASK`, the TX
     packet-edit table `ETH_TXPKT_CFG_A_{0..9}__{INSERT_CTL,VLAN1/2,CUSTOM_HDR}`,
     `TX_CHECKSUM_STAT`, and the full `ETH_RXCLASS_A_*` flow-director/TCAM.
  2. **Uncompiled reference driver** — `eth_rx_flow.cpp/.h` (64-row TCAM, actions
     queue/drop/rmhdr/prepend) is **absent from every Makefile**, `#include`s a
     non-existent `udp.h`, and has syntax errors. It is intent, not shipped code.
  3. **Dead spec scaffolding** — `packet_header.h` (574 lines) is copy-pasted from
     the Linux `rxe` soft-RoCE driver (BTH/RETH/AETH, `IB_UDP_DPORT 4791`) and is
     `#include`d by nothing.
  So the HW *hooks* are real, but there is **no shipped RoCE stack** — tiers 2–3
  must be re-derived against the register headers, not compiled as-is. PFC is
  likewise an un-implemented `TODO` (`eth_init.cpp:538`). The **locked program
  assumption remains "gateway always exists"** (`tt-rdma-bidirectional-mesh-gap.md`).
  Treat wire-native / thin-gateway RoCE on BH as a *future track gated on tiers
  2–3 maturing*, **not** the MVP. The MVP keeps ethertype `0x1AF6` + gateway,
  exactly as on WH. (BH is also **not a drop-in RoCE NIC** — QP/PSN/MR-translation/
  DCQCN/CQ are all FW+host software, there is no netdev / kernel-verbs / PCIe NIC
  function; it is an on-die-accelerated peer that still needs the gateway+SDK.)

## 6. Reliability: HW packet-resend is available on BOTH chips (and unused today)

The correction from §1 drives this section: hardware sequence-number resend is
**not** a BH novelty. Both WH (`ETH_TXQ_CMD[1]` + `ETH_TXQ_CTRL[0]` resend +
`ETH_TXQ_REMOTE_SEQ_TIMEOUT`) and BH (`eth_send_packet`, 128-entry Go-back-N)
have it — and TT-RDMA uses **neither**, because it runs raw mode for
third-party interop. What actually varies is the *link partner*, not the chip:

| Link | Path (both WH & BH) | Reliability |
|---|---|---|
| TT ↔ gateway / Mellanox / FPGA (external) | **raw** (`START_RAW` / `eth_send_raw`) | HW resend is unusable (partner isn't a TT chip) → **software (Phase R) or gateway deferred-ACK** (`mesh-addressing-spec §6`) |
| TT ↔ TT chip-to-chip (TT-Link) | **packet** (`ETH_TXQ_CMD[1]` / `eth_send_packet`) | **HW resend usable** — no software retx needed, on WH *or* BH |

Consequences for the roadmap:

- The software Phase R / selective-ACK work is for the **external raw path** and
  is chip-independent — it is not made obsolete by BH.
- **Chip-to-chip TT-Link could use HW packet mode on WH today**, not just BH;
  it was simply never wired into the TT-RDMA (raw-mode) FW. BH's Go-back-N is
  better *documented* (128-entry window, `§1.5.3`) but not categorically new.

### 6.1 Can the raw path "borrow" the seq counters? — No (resolved)

Checked in code on both chips. The sequence-number machinery is a property of
the **packet** command, not a register you can bolt onto raw:

- `eth_send_raw()` (`transfer_start_raw`) programs only `TRANSFER_START_ADDR`,
  `TRANSFER_SIZE_BYTES`, and the dest MAC — **no `DEST_ADDR`, no seq register**
  (WH `eth_ss.cpp:16`, BH `eth_ss.cpp:16`). RX raw mode consumes via
  `BUF_PTR`/`BUF_WRAP` (`ETH_RXQ_CTRL[2]`), not `LOCAL_RX_SEQ_NUM`.
- The BH arch spec states it outright: *"Only the 'Raw' path bypasses the
  sequence-number / retransmission protocol; everything else uses it"*
  (`eth_arch_spec §1.5`).
- Even if the counters *could* be forced in raw mode, the ack half
  (`REMOTE_RX_SEQ_NUM`, resend-on-drop) requires a **TT peer** to emit seq acks
  — a Mellanox NIC / FPGA gateway never will. That is the whole reason TT-RDMA
  chose raw mode. **So: not usable for the external path, on either chip.**
  Software Phase R / gateway deferred-ACK remains the answer there.

### 6.2 The positive finding: packet mode *is* native reliable remote-WRITE

`eth_send_packet(q, src_word_addr, dest_word_addr, num_words)` programs a
**remote `DEST_ADDR`** (`ETH_TXQ_DEST_ADDR`) and uses HW seq+resend — i.e. the
controller already does a **reliable one-sided write from local L1 to a remote
NoC address** (WH `eth_ss.cpp:28`, BH `eth_ss.cpp:43`; there is even
`eth_write_remote_reg` for remote register writes). For **chip-to-chip TT-Link**,
a TT-RDMA `WRITE` maps almost 1:1 onto `eth_send_packet` — the reliability,
sequencing, and remote-address delivery TT-RDMA builds in software for the raw
path are already in hardware here. This is the strongest argument for a
**two-profile FW**: raw-mode + software/gateway reliability for external
(RoCE-interop) links, packet-mode + HW ARQ for TT-native links. It applies
equally to WH and BH.

## 7. Two-profile firmware — the load-bearing BH bring-up decision

BH bring-up is committed, so this is now a design directive, not a hypothesis:
**ship one BH RDMA FW that runs two link profiles, selected per erisc rail at
bring-up.** The profiles differ in TX/RX mode, reliability, addressing/fan-out,
the WRITE data path, and link training. They map cleanly onto the two paths in
`tt-rdma-switch-interop-architecture.md` (Profile E ≈ Path A gateway building
block; Profile L ≈ the TT-Link / TTFabric mesh).

| Aspect | **Profile E — External / RoCE-interop** | **Profile L — TT-Link / chip-to-chip** |
|---|---|---|
| Link partner | gateway / Mellanox / FPGA (non-TT) | another TT chip (WH or BH) |
| TX | `eth_send_raw` (`transfer_start_raw`) | `eth_send_packet` (`transfer_start_data`) |
| RX | RXQ **raw** + `BUF_WRAP`; FW parses the ring | RXQ **packet** mode; HW lands payload at `DEST_ADDR` |
| Reliability | **software** (Phase R) + gateway deferred-ACK | **HW** seq + Go-back-N resend |
| WRITE data path | payload → RX ring → FW `rkey` lookup → `noc_write` | `eth_send_packet(src, remote DEST_ADDR)` → **HW writes the remote NoC address directly**; no remote FW, no remote MR lookup |
| Remote access control | **rkey/access/bounds enforced** on the target FW | **none on the wire** — HW writes where told (trusted TT fabric) |
| Addressing / fan-out | **1 `DEST_MAC`** (gateway) + in-band `rkey`/`tag`-QPN → **~65 k endpoints** (`switch-interop §3.1`) | per-queue **fixed `DEST_ADDR`** → **≤3 dests/erisc** on BH (`switch-interop §1`) |
| Completion signal | opcode `0x40` ACK frame (software) | HW `ETH_RXQ_REMOTE_RX_SEQ_NUM` advancing |
| Link training | no-AN raw (CMAC-simplified); external peer fails the `chip_info_exchange` handshake (`LINK_TRAIN_TIMEOUT_CHIP_INFO`) | AN + `chip_info_exchange` (mission FW) |
| Wire rate ceiling | host-PCIe-bound (~25 Gbps/rail today) | line rate (400 G on BH) |

### 7.1 The three decisions that fall out

1. **The WRITE asymmetry is the whole point of Profile L.** A one-sided `WRITE`
   over TT-Link is *not* "land in a ring, look up rkey, `noc_write`" — it is a
   single `eth_send_packet` whose `DEST_ADDR` **is** the remote NoC address, so
   the receiving chip's hardware writes it with **no FW involvement and no MR
   check**. The initiator resolves `rkey → remote NoC addr` locally (the imported
   `RemoteMr` must carry a NoC address, not just an rkey) and the wire does the
   rest. This is line-rate and zero-CPU on the target — but it means Profile L
   **drops the rkey/access/bounds enforcement** that Profile E keeps. That is
   acceptable *only* because TT-Link is a trusted on-fabric interconnect; it is
   **not** acceptable toward an external peer, which is exactly why external
   traffic stays on Profile E with full MR validation.

2. **Profile L fan-out is capped at ~3 peers per erisc** — the HW seq/retry
   context is bound to the queue's `DEST_ADDR`, and there are only 3 TXQ on BH
   (`switch-interop §1`, 4 on Grendel). So TT-Link is a **point-to-point mesh**
   (hop through neighbors / more rails), never a star to many peers over one
   erisc. Reaching many TT peers = many of the 14 ETH SS rails + mesh routing,
   *not* many flows on one queue. Profile E has the opposite shape: one queue,
   one MAC, tens of thousands of endpoints via in-band addressing — because raw
   mode carries no per-queue retry state to bind.

3. **Completion + reliability plumbing forks, but the SDK stays uniform.**
   Profile E completions come from software `0x40` ACKs (or the gateway's
   deferred ACK); Profile L completions come from watching
   `REMOTE_RX_SEQ_NUM`. The `QpHandle` carries which profile a connection uses
   (`mesh-addressing-spec §5.1` already has `rail()`); `post_write`/`post_read`
   look identical to the app, and the FW/SDK pick the path.

### 7.2 What is shared vs forked in the one FW

- **Shared:** the 32 B v1 header build, the opcode enum, the host SDK surface,
  the MR table (Profile E uses it on the *target* for validation; Profile L uses
  it on the *initiator* to resolve `rkey → remote NoC addr`), and the
  SEND→host-ring path.
- **Forked by profile:** TX call (`eth_send_raw` vs `eth_send_packet`), RXQ mode
  (raw+BUF_WRAP vs packet), reliability engine (software Phase R vs HW seq), and
  the WRITE handler (ring+lookup+`noc_write` vs direct `DEST_ADDR`).

### 7.3 Bring-up ordering

**Profile E first** — it is the direct WH port: same raw mode, same software
reliability, same gateway, reuses the entire v1 stack, and delivers multi-vendor
interop on stock silicon (`switch-interop` Path A). Phases BH.1–BH.4 below are
Profile E. **Profile L second** (BH.5) — higher value (line-rate, host-bypassed,
the TT-Link vision) but it needs packet-mode bring-up, the `chip_info_exchange`
handshake, and the trust decision in 7.1(1). Do not block external interop on it.

## 8. Reconciliation checklist — what each existing v1 doc must caveat

These docs currently read as WH-universal; each needs a BH note (or a BH
variant) once this port lands:

| Doc | WH-specific claim | BH correction |
|---|---|---|
| `README.md` | "WH FW `erisc_cmac_simple` unchanged" NIC-agnostic table | add a BH column: FW = tt-metal active-eth kernel over `bh-erisc` base FW; 400 Gbps; HW ARQ/offload |
| `tt-rdma-pfc-lossless.md` | DWC XLGMAC regs 0x70/0x90, `mac_tx_rx_enable` | **entirely WH-specific.** BH PFC is Rianta + controller, currently `TODO` — needs its own procedure |
| `tt-rdma-fw-arch-rx.md` | single-RISC, BUF_WRAP ring, L1 map `0x4000…0x2B000` | BH: two RISCs, cut-through RX + TCAM classifier, SW L1 region `0x0…0x70000`, must avoid `0x70000+` |
| `tt-rdma-host-sdk.md` | `bring_up()`, RCB/MR L1 addresses | BH addresses differ; `bring_up` must coexist with base FW (mailbox/api-table) |
| `runtime-workload-coexistence.md` | Qwen3 + `erisc_cmac_simple` on one WH die | BH coexistence is *three*-way: Tensix workload + tt-metal active-eth RDMA kernel + `bh-erisc` base FW, all sharing the ETH SS |
| `tt-rdma-verbs-provider.md` | "FW unchanged (md5 43bdcb46)" | provider is host-side and chip-agnostic; the FW-version line needs a BH entry |

## 9. Phase plan (proposed)

Mirrors the WH phase style; each gate is binary. Assumes a BH board with a
trained ETH link and the tt-metal active-eth path working. **Profile** column
ties each phase to §7 (E = external/raw, L = TT-Link/packet).

| Phase | Profile | Deliverable | Base-FW change? | Validation gate |
|---|---|---|---|---|
| **BH.0** | — | Active-eth RDMA kernel skeleton: RISC1 tight loop + RISC0 base-FW yield (§3.1), link stays up | no | `port_status` UP for 10 min with kernel resident + periodic `service_eth_msg()` |
| **BH.1** | E | Raw TX: build v1 header + payload in SW L1, `eth_send_raw` to a partner MAC | no | partner (MLX/FPGA) receives a valid `0x1AF6` frame; bytes match |
| **BH.2** | E | RX: classifier TCAM matches `0x1AF6` → SW L1 landing; opcode dispatch (SEND/WRITE) | no | inbound WRITE lands at the right NoC offset; 100% admit at 4 KB |
| **BH.3** | E | MR table + WQE ring re-laid-out in SW L1; host `register_mr` / `post_send_write` | no | SDK loopback WRITE/READ; `TtRdmaEndpoint` runs unmodified against BH addresses |
| **BH.4** | E | Gateway interop: BF3/FPGA translates RoCEv2 ↔ v1 to a BH node | no (gateway only) | `ib_write_bw` from a remote CX-5 into a BH MR; bytes match |
| **BH.5** | L | TT-Link: packet mode (`eth_send_packet` → remote `DEST_ADDR`), HW ARQ, WRITE-without-remote-MR (§7.1); BH↔BH / BH↔WH | no | sustained WRITE, 0 drops, no software retx (`RESEND_CNT`/`REMOTE_RX_SEQ_NUM` move); line rate to ≤3 peers/erisc |
| **BH.6** | E | PFC / lossless on Rianta (replaces the WH DWC procedure); implements the `eth_init.cpp:538` TODO path | **maybe** (base FW) | pause counters non-zero under load, 0 buffer-discard |
| **BH.7** (stretch) | E | RoCE offload investigation: HW iCRC/UDP/IP insert; measure how much gateway translation it removes | TBD | a frame emitted with HW-inserted UDP/IP/iCRC is accepted by a stock RoCE peer through a reduced-function gateway |

## 10. Open questions

1. **RISC budget — RESOLVED (§3.1).** Both RISCs are used: RISC0 (`active_erisc`)
   owns the go-loop + base-FW yield; RISC1 (`subordinate_erisc`) is a free data
   mover. Recommended: RDMA tight loop on RISC1, base-FW housekeeping on RISC0.
   NoC is a **three-way** share (Tensix + RDMA + base-FW dynamic-NoC), arbitrated,
   not a blocker. Budgets: 24 KB code/RISC, 8 KB local/RISC, 512 KB shared L1.
   **Yield-interval sub-question — mostly answered (§3, corrected):** not a hard
   1000 ms watchdog. `service_eth_msg()` must run often enough that no mailbox
   client (host/CMFW/RISC1) blocks past its timeout; the link-status check
   self-debounces to 1000 ms and must *not* be forced faster; Profile L is
   HW-keepalive'd, Profile E's ceiling is the partner's link timer. **CMFW
   watchdog — RESOLVED: none.** `tt-system-firmware` (Zephyr/SMC) samples the eth
   heartbeat + PCS link every 100 ms into `TAG_ETH_LIVE_STATUS` telemetry and
   takes **no action** on staleness; link-up is PCS-hardware-derived, not
   heartbeat-derived (`lib/tenstorrent/bh_arc/{eth,telemetry}.c`). **Only two soft
   items remain:** (a) whether any **host-side** health manager acts on the
   `TAG_ETH_LIVE_STATUS` heartbeat bit going stale (ops policy, not FW —
   deployment-specific); (b) the empirical flap-recovery tolerance at a chosen
   yield interval — the BH.0 rig gate.
2. **Cut-through RX + no app back-pressure** (`§1.6`) — the WH RX-ring-overflow
   mitigations (backpressure assert) don't exist the same way; what absorbs a
   burst on BH, PFC or a deep SW ring?
3. **Raw vs packet path — RESOLVED (§6.1–6.2).** Raw cannot borrow the seq
   counters (they run only in packet mode, and the ack half needs a TT peer). But
   packet mode *is* native reliable remote-WRITE (`eth_send_packet` → remote
   `DEST_ADDR` + HW resend) → use it for TT-Link, raw+software/gateway for
   external. Motivates a two-profile FW.
4. **Host bus ceiling** — WH's ~25 Gbps host-RDMA cap is PCIe Gen3 x4; what is
   BH's, and does it move the host-hugepage MR ceiling relative to the 400 Gbps
   wire?
5. **How much of the RoCE offload is real** vs. exploratory `packet_header.h`
   scaffolding (§5) — needs a HW/IP-doc audit before any thin-gateway claim.
6. **Multi-rail is native on BH** — 14 ETH SS × 400 Gbps is a large rail matrix;
   does `mesh-addressing-spec §8` (per-QP rail pinning, rkey rail-agnostic) hold
   unchanged? (Expected yes — NoC is chip-global — but validate.)

## 11. Companion docs

- `tt-rdma-switch-interop-architecture.md` — gateway (Path A) vs native-switch
  (Path B); the 3-TXQ / 10-header-row HW limits that cap Profile L fan-out (§7).
- `tt-rdma-wire-protocol-v1.md` — the wire format, unchanged on BH.
- `bf3-gateway-design.md` / `tt-rdma-mesh-addressing-spec.md` — the gateway +
  addressing, chip-agnostic; this doc says the BH side terminates the same link.
- `tt-rdma-fw-arch-rx.md` / `tt-rdma-pfc-lossless.md` / `tt-rdma-host-sdk.md` —
  the WH-specific docs this port must produce BH variants of (§7).
- `bh-erisc` `docs/eth_arch_spec.md` — the authoritative BH ETH SS reference
  this port builds on (external repo).
