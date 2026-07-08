# TT-RDMA Mesh Addressing + Remote-MR Import — Spec (gap items 1 & 2)

**Status:** design / build plan. Not yet implemented.
**Scope:** the two critical-path items that turn a TT node from a *passive RDMA target* into a *symmetric mesh endpoint* reachable by RoCEv2 verbs **in both directions**, through a translator gateway (BF3 or FPGA-NIC — never wire-native RoCEv2). See `tt-rdma-bidirectional-mesh-gap.md` for the full gap list; this doc specs items **1 (endpoint addressing)** and **2 (remote-MR import + QP object)** in detail.

**Headline result of the investigation:** the MVP needs **no WH FW change and no wire change.** The v1 header already carries everything required to route both directions, *if* we fix the addressing *interpretation* rather than the format. The work is host-SDK + gateway-side.

---

## 1. The problem, precisely

A TT chip's CMAC has exactly **one physical wire partner** — the gateway. The TX destination MAC is a single hardcoded register pair (`main_cmac.cc:204-205`, `ETH_TXQ_DEST_MAC_ADDR_HI/LO`), programmed once at init. Every frame WH emits goes to that one MAC. Therefore:

> **The identity of a *remote* endpoint reachable through the gateway cannot be expressed by L2 addressing. It must be carried in-band in the 32 B TT-RDMA header.**

The v1 header has no dedicated QPN field (`tt-rdma-wire-protocol-v1.md:14-36`). The verbs-provider doc proposed reusing the 16-bit `tag` as a QPN (`tt-rdma-verbs-provider.md:126-143`, "Option B"). Investigating the shipped FW showed that proposal is **half-right and half-broken**:

- `tag` is **16 bits**, at frame offset +2 (`main_cmac.cc` reads `*(uint16_t*)(frame+2)`), not the 32 bits the verbs-provider `post_send` pseudocode assumed (`tt-rdma-verbs-provider.md:316-317`). It cannot hold both a QPN and a completion cookie.
- `tag` is **already opcode-scoped**: for `READ_REQ`/`READ_RESP` it is the READ correlation key (`cslot = tag & (READ_CORR_N-1)`, `main_cmac.cc:762-764`); for `SEND`/`WRITE` it is an opaque cookie the FW never inspects. So a blanket "`tag` = QPN" **collides with READ correlation.**

The fix is to stop treating one field as the universal endpoint id and instead route each opcode by the field that already implies the endpoint.

## 2. The addressing model (no wire change, no FW change)

Two facts unlock a clean split:

1. **A remote MR belongs to exactly one remote endpoint.** So for one-sided ops (`WRITE`, `WRITE_IMM`, `READ_REQ`), the `rkey` in the header (frame+12) *already* identifies the target endpoint — the gateway just needs an `rkey → endpoint` table, which it must maintain anyway to translate `rkey`/`remote_offset` into a RoCE `RETH{vaddr, rkey}`.
2. **Two-sided ops (`SEND`, `SEND_IMM`) carry `rkey = 0`** — no MR, so no implied endpoint. These are the *only* opcodes that need an explicit endpoint id, and they don't use `tag` for anything else. So `tag = QPN` for SEND-family is free and non-colliding.

### 2.1 Per-opcode routing table (this is the whole model)

| Opcode | Endpoint identified by | `tag` means | Gateway routing action |
|---|---|---|---|
| `0x10 WRITE` | `rkey` → endpoint | opaque (unused) | `mr_fwd[rkey]` → RoCE `RDMA_WRITE_ONLY`, `RETH{vaddr=base+remote_offset, rkey=roce_rkey}` |
| `0x11 WRITE_IMM` | `rkey` → endpoint | opaque | as WRITE + `ImmDt = imm_data` |
| `0x20 READ_REQ` | `rkey` → endpoint | **correlation** (TT-local) | `mr_fwd[rkey]` → RoCE `RDMA_READ_REQUEST`; remember `tag` for the RESP |
| `0x01 SEND` | **`tag` = QPN** | **QPN** | `qp_fwd[tag]` → RoCE `SEND_ONLY` on that QP |
| `0x02 SEND_IMM` | **`tag` = QPN** | **QPN** | as SEND + `ImmDt` |
| `0x21 READ_RESP` | (TT-local landing) | correlation echo | inbound only; gateway built it from a RoCE read response |
| `0x40 ACK` | cumulative `seq` | — | see §6 |

**Why this needs no FW change:** the WH FW already (a) routes inbound one-sided ops by `rkey` into its own MR table, (b) treats `tag` as opaque for SEND, (c) uses `tag` only for READ correlation, and (d) DMA-pushes the full 32 B header (including `tag`) to the host RxWqeRing (`main_cmac.cc:744-747`), so the host can read the QPN on RX. Nothing above asks the FW to interpret a byte differently than it does today.

### 2.2 QPN namespace

- 16-bit `tag` ⇒ **65 536 QPs per CMAC port** for SEND-family. That is far above the single-erisc-per-chip reality; not a near-term limit.
- One-sided ops don't consume QPN space at all (routed by `rkey`), so a node doing pure WRITE/READ traffic to many peers needs zero QPNs.
- When a real per-QP dedicated field is eventually needed (per-QP ordering, >64k QPs, strict verbs semantics), that is the **Option C wire change** (ethertype `0x1AF7`, 36 B header) — deferred, and specced as gap-item 5 / verbs-provider Phase 13. This spec deliberately does **not** need it for the MVP.

## 3. Gateway binding tables

The gateway is a bidirectional proxy: it impersonates each TT node to the RoCE fabric, and impersonates each remote peer to the TT node. It holds:

```
qp_fwd[tt_qpn]            → { roce_qp_handle, remote_gid, remote_qpn }      // SEND routing (TT→remote)
qp_rev[roce_qpn]          → { tt_qpn }                                       // SEND demux  (remote→TT)
mr_fwd[rkey]              → { endpoint, remote_base_vaddr, roce_rkey }       // one-sided TT→remote
mr_rev[roce_rkey]         → { tt_mr_slot, tt_rkey }                          // one-sided remote→TT
seq_map[tt_qpn]           → { next_psn, psn_base ↔ seq_base }                // ACK correlation (§6)
read_pending[tt_tag]      → { roce_qp, roce_psn, length }                    // READ_RESP construction
```

**rkey pass-through (recommended default).** A TT rkey is `(slot<<24)|(rand16<<8)|gen` — 32 bits, fits a RoCE `rkey` exactly. When a single gateway fronts a single TT node, advertise the TT rkey *verbatim* as the RoCE rkey (`roce_rkey == tt_rkey`), so `mr_fwd`/`mr_rev` are identity maps and there is no per-frame rkey rewrite. Fall back to a translating map only when **multiple TT nodes behind one gateway** could collide in rkey space (then namespace rkeys per node).

## 4. Connection establishment — symmetric

The BF3 doc only specs the **passive** case (`bf3-gateway-design.md:296-317`: "compute side calls `rdma_connect`, BF3 listens"). Symmetric access needs both:

### 4.1 Inbound (TT as target) — already covered
External app `rdma_connect(gateway_ip)`. Gateway `rdma_accept`, allocates a `tt_qpn`, allocates/binds a TT MR slot for any memory the app will target, records `qp_rev`/`mr_rev`. This is the existing BF3 flow.

### 4.2 Outbound (TT as initiator) — the missing half
The TT host must be able to say "connect me to remote endpoint E and give me a handle." Since WH has no IP stack, the gateway proxies the active connect:

```
TT host SDK                      wh_mr_agent            gateway (BF3)                 remote peer
───────────                      ───────────            ─────────────                 ───────────
ep.connect(remote_uri)  ──UNIX/mgmt──▶  proxy_connect(remote_uri)
                                                │ rdma_resolve_addr/route
                                                │ rdma_create_qp  (gateway-owned RC QP)
                                                │ rdma_connect ──────────────────────▶ rdma_accept
                                                │ ◀───────────── QP params (qpn/psn) ──┘
                                                │ alloc tt_qpn; qp_fwd[tt_qpn]={roce_qp,...}
   ◀───── QpHandle{ tt_qpn, peer_mrs[] } ───────┘
```

The gateway holds an RC QP **per (TT node, remote peer)** pair. Remote-MR advertisements the peer exchanges at connect time are returned to the TT SDK as importable `RemoteMr`s (§5.2).

Control-plane transport is the same `wh_mr_agent` UNIX-socket + mgmt-link path the BF3 doc already introduces for MR registration (`bf3-gateway-design.md:167-177`). We extend its message set, not its topology.

## 5. Host-SDK surface (gap item 2)

The current SDK (`tt-rdma-host-sdk.md`, `external_iface_sender.hpp`) **conflates local and remote MRs**: `register_mr` returns an `MrHandle` for a *local* region, but `post_send_write(buf, MrHandle remote, offset)` then takes an `MrHandle` as the *remote* target (the example at `tt-rdma-host-sdk.md:302-308` is really a loopback). There is no way to construct a handle to a *peer's* MR, and no QP/connection object at all. Item 2 fixes both.

### 5.1 New types

```cpp
// A connection to one remote endpoint through the gateway.
// tt_qpn is the value the SDK stamps into `tag` for SEND-family ops.
class QpHandle {
public:
    uint16_t qpn() const;                 // wire tag for SEND/SEND_IMM on this QP
    bool     connected() const;
    std::span<const RemoteMr> peer_mrs() const;   // MRs the peer advertised at connect
    uint8_t  rail() const;                // erisc/CMAC port this QP is pinned to (§8)
private:
    /* gateway-assigned qpn, peer identity, per-QP seq base, pinned rail */
};

// A handle to a REMOTE peer's memory region. Distinct from MrHandle (local).
// Constructed from peer-advertised {rkey, base_vaddr} — never touches the
// local FW MR table.
struct RemoteMr {
    uint32_t rkey        = 0;   // peer's rkey, goes on the wire at frame+12
    uint64_t base_vaddr  = 0;   // peer's MR base; SDK computes remote_offset = addr - base
    uint64_t length      = 0;
    uint32_t access      = 0;   // advertised access flags (for local sanity checks)
    bool valid() const { return rkey != 0; }
};
```

`MrHandle` (existing, `external_iface_sender.hpp` / `tt-rdma-host-sdk.md:35-42`) stays as the **local** registration handle. The type split is the load-bearing API change: local vs remote MRs are no longer interchangeable.

### 5.2 New / revised endpoint methods

```cpp
// ── Connection management (new) ───────────────────────────────────────────
// Active connect (TT as initiator): proxied through wh_mr_agent → gateway.
// remote_uri is resolvable by the gateway (IP or GID:port).
QpHandle connect(std::string_view remote_uri, uint32_t timeout_ms = 5000);

// Passive accept (TT as target): register interest; the gateway allocates the
// tt_qpn when a remote peer connects. Returns handles as peers arrive.
void                    listen(uint16_t service_id);
std::optional<QpHandle> accept(uint32_t timeout_ms);

// Import a peer's MR advertisement so it can be used as a one-sided target.
// Normally populated automatically from connect()/accept() exchange; this is
// the explicit form for out-of-band rkey exchange.
RemoteMr import_remote_mr(uint32_t peer_rkey, uint64_t peer_base_vaddr,
                          uint64_t length, uint32_t access);

// ── One-sided ops — now take a QP + RemoteMr + absolute remote addr ────────
// (verbs-faithful: caller passes the peer's virtual address, SDK derives
//  remote_offset = remote_addr - remote.base_vaddr, stamps rkey = remote.rkey)
std::optional<uint32_t> post_write(QpHandle qp, std::span<const uint8_t> buf,
                                   RemoteMr remote, uint64_t remote_addr,
                                   uint32_t cookie = 0);
std::optional<uint32_t> post_write_imm(QpHandle qp, std::span<const uint8_t> buf,
                                       RemoteMr remote, uint64_t remote_addr,
                                       uint32_t immediate, uint32_t cookie = 0);
std::optional<uint32_t> post_read(QpHandle qp, RemoteMr remote, uint64_t remote_addr,
                                  size_t len, MrHandle local_landing,
                                  uint64_t local_offset, uint32_t cookie = 0);

// ── Two-sided ops — carry qpn in tag ───────────────────────────────────────
std::optional<uint32_t> post_send(QpHandle qp, std::span<const uint8_t> buf,
                                  uint32_t cookie = 0);
```

The pre-existing offset-based `post_send_write(buf, MrHandle, remote_offset)` stays as the low-level primitive on `ExternalIfaceSender`; `TtRdmaEndpoint` adds the QP-scoped, absolute-address forms above.

### 5.3 What actually changes in the frame builder

Tiny. Compared to today's `post_send_read` (`external_iface_sender.cpp:764-777`), which already writes `frame[0]=opcode`, `frame[2..3]=tag`, `frame[12..15]=rkey`, `frame[16..23]=remote_offset`:

- SEND-family: set `tag = qp.qpn()` instead of an opaque cookie.
- One-sided: set `rkey = remote.rkey`, `remote_offset = remote_addr - remote.base_vaddr` — that's it. Routing falls out at the gateway from `rkey`.

### 5.4 RX demux (`RxCqe.qpn`)

Add `uint16_t qpn` to `RxCqe` (`tt-rdma-host-sdk.md:62-71`). The FW already DMA-pushes the 32 B header into the host slot (`main_cmac.cc:744-747`); the SDK reads `tag` from that pushed header and surfaces it as `qpn` so a receiver can steer an inbound SEND to the right QP/CQ. No FW change.

## 6. End-to-end ACK correctness (why the gateway must *defer*)

A TT-initiated WRITE's `TxCqe` must complete **only after the remote peer has acknowledged** — not when the gateway received the frame. RDMA's reliability contract is end-to-end; a hop-by-hop ACK at the gateway would let the TT app observe "write complete" before the bytes are safe at the remote, silently corrupting on gateway/remote failure.

Required gateway behavior:
- On a TT one-sided/SEND request, the gateway forwards it as a RoCE op **and withholds the TT `0x40` ACK** until the real RoCE ACK (matching PSN) returns from the remote.
- It then emits a TT `0x40` with `seq` = the TT seq that maps to that PSN range (`seq_map`), advancing `rcb.acked_idx` on WH (`main_cmac.cc:402`, Phase R cumulative ACK).
- For READ, the gateway builds the TT `0x21 READ_RESP` from the RoCE read response, echoing the original `tag` (from `read_pending`) so FW's correlation table (`main_cmac.cc:757-793`) lands it in the right local MR.

**Caveat (inherited, gap-item 5):** on shipped FW, WH's ACK is *port-level cumulative*, not per-QP. With multiple outstanding QPs from one TT node, a slow remote head-of-lines the cumulative ACK for all of them. The MVP (M0–M6) ships with this limitation. §7 promotes the per-QP fix into an **accepted FW track** — it is latency/BW-neutral and is the right long-term answer.

## 7. FW-accepted enhancements (latency/BW-neutral)

FW changes are acceptable **provided they add nothing to the per-frame data loops** — the RX dispatch inner walk and the TX WQE-ring drain / DMA-pull pipeline. The three below clear that bar and are worth doing; they promote what §6 and the gap doc previously deferred. The test applied to each: *does it add per-frame work on the hot path?*

### 7.1 Per-QP ACK / sequencing — do this (highest value, zero hot-path cost)
Replace the single `rcb.acked_idx` word (`main_cmac.cc:402`) with a small array indexed by QPN. ACK processing is **already off the bulk data path** (one ACK per N data frames), so the added index touches only ACK receipt — the WRITE/READ/SEND loops are untouched. Fixes the §6 head-of-line caveat: a slow remote no longer stalls unrelated QPs' completions. **Cost:** ~2 B × N_QP of L1 (cap 256 QPs ≈ 1 KB); **0 ns on the data path.**

### 7.2 MR table 16 → 64 — do this (genuinely free)
The MR lookup is **already O(1) direct-index**: `slot = (rkey >> 24) & 0x0F` (`main_cmac.cc:596`), not a linear scan. Bump the mask to `& 0x3F` and grow the table (64 × 32 B = 2 KB L1). Removes the NCCL/UCX MR-pool wall (gap-item 6). **Cost:** 2 KB L1; **0 ns added** (index stays O(1)).

### 7.3 Dedicated QPN field — do this *if* per-QP seq must cover READ (it should)
Per-QP sequencing (7.1) has to attribute each frame to a QP. On SEND/WRITE `tag` is free for QPN, but on **READ** `tag` is the correlation key (`main_cmac.cc:762-764`) — so QPN can't live in `tag` uniformly. That forces a dedicated QPN field, which is a wire change. **The header size is the load-bearing decision, because payload must stay 16-byte aligned** so the FW's `ncrisc_noc_fast_write` from `PACKET_BUF+off+hdr` isn't misaligned (this is why the header is 32 B today; see P2 round-to-16). Alignment-safe sizes are **32 B or 48 B only** — a 36 B or 40 B header misaligns the payload and forces a bounce copy, which *does* cost bandwidth.

| Option | Header | Payload align | Small-frame BW cost (256 B) | Jumbo BW cost (4080 B) | Trade-off |
|---|---|---|---|---|---|
| **A — CHOSEN** | 32 B, QPN reclaimed from `header_cksum` at +28 | 16 B ✓ | 0% | 0% | drops the on-wire header CRC → move header-integrity to a gateway-side software check (gateway is not latency-critical); Ethernet FCS still covers the header per-link |
| ~~B~~ | ~~48 B, keep `header_cksum`, add 32-bit QPN~~ | 16 B ✓ | ~6% | ~0.4% | keeps end-to-end header CRC; costs 16 B/frame — rejected (open-Q #6 decided A) |
| ~~C~~ | ~~36/40 B~~ | **broken** | — | — | rejected: misaligns payload → NoC bounce copy → BW hit |

**Decision (open-Q #6): Option A.** The `0x1AF7` header stays 32 B; the field at +28 changes meaning from `header_cksum` to `qpn`. Concrete layout:

```
Offset  Size  0x1AF6 (v1)      0x1AF7 (extended)   Notes
------  ----  --------------   -----------------   ---------------------------------
 0      1     opcode           opcode              unchanged
 1      1     version_flags    version_flags       ver bits distinguish the format
 2      2     tag              tag                 correlation/cookie (freed at M9)
 4      4     length           length              unchanged
 8      4     seq              seq                 per-QP after M7 (was port-cumulative)
12      4     rkey             rkey                unchanged — still routes one-sided
16      8     remote_offset    remote_offset       unchanged
24      4     imm_data         imm_data            unchanged
28      4     header_cksum     qpn                 ← reclaimed. low 24b = QPN (RoCE-width),
                                                     high 8b reserved (0)
32      L     payload          payload             16-B aligned ✓
```

Ethertype `0x1AF7` for the extended format; dual-stack with `0x1AF6` during transition (the FW branches on ethertype *before* the header walk, so this is off the per-frame *inner* loop). Parse cost of the `qpn` word: one extra load, single-digit ns.

**Gateway responsibility created by this decision:** since the wire no longer carries a header CRC, the gateway must compute and verify a software CRC-32C over the 28 header bytes on every frame it forwards in each direction, to catch corruption introduced *inside* its own store-and-forward path (Ethernet FCS only protects each link, not gateway memory). This runs on the gateway's translation path, which is not latency-critical (`bf3-gateway-design.md §7`). WH FW does **not** compute or check it — that keeps the reclaim free on the hot path.

### 7.4 READ correlation off `tag` → onto `seq` — do this after 7.3
Once QPN has its own field, move READ correlation from `tag & 31` to a `seq`-keyed table (seq is already unique per outstanding op). Frees `tag` entirely and makes QPN uniform across all opcodes. Correlation lookup already runs per READ_RESP — just re-keyed, no new hot-path work.

### 7.5 Still genuinely out of scope here
- **ATOMIC (0x30 CAS/FAA)** — additive handlers, off the common path; scheduled when an app needs it (gap doc / README v1.2).
- **>2³² QPs, multi-erisc per chip** — v1.2 scaling, separate work.

## 7a. Revised recommendation

Ship **M0–M6 on stock FW** (proves the addressing model + both directions end-to-end), then land the **FW-accepted track 7.1 + 7.2 + 7.3(Option A)** as M7–M9 to make per-QP reliability/ordering correct and remove the MR wall — all inside the latency/BW envelope. 7.4 follows 7.3.

## 8. Multi-rail / multi-gateway topology (N erisc × M gateways)

Production is not one wire to one gateway. A WH chip has multiple external-CMAC-capable erisc cores (Wormhole: up to 16 eth cores), and a deployment fields multiple gateways (BF3 / FPGA) for bandwidth aggregation, route diversity, and no single point of failure. The addressing model above is **compositional** over this matrix — one property makes it work, one rule keeps it correct.

### 8.1 The enabler: NoC-global ⇒ MRs are rail-agnostic
A one-sided op resolves `rkey → mr.base_noc_addr` and does `ncrisc_noc_fast_write(base + offset)` (`main_cmac.cc:620`). That NoC address is reachable from **every** erisc core on the die. Therefore:

> An inbound WRITE/READ for a given MR may arrive on **any** erisc rail, via **any** gateway, and lands at the same memory — the rkey→address resolution is identical on every port.

**One-sided traffic load-balances and fails over across rails for free** — no per-rail MR state, no reassembly. This falls straight out of rkey-routing (§2).

### 8.2 The rule: pin a QP to one rail; aggregate with many QPs
Reliability (seq / ACK / retx) and ordering are **per-CMAC-port**. Per-packet striping one QP across rails reorders frames and splits its seq space → ARQ and ordering break. Correct pattern (identical to how NCCL / UCX already do multi-rail RoCE):

- **Pin a whole QP to one rail** — `(erisc port ↔ gateway ↔ remote)`.
- **Open multiple QPs**, spread across rails, for aggregate bandwidth.
- A flow larger than one rail is striped **per-message across QPs**, reassembled at the app/provider layer. TT exposes N rails as N ports/QPs; the striping lives above the wire.

So: **per-QP rail pinning, per-connection rail aggregation.** No new wire-level reliability protocol.

### 8.3 Why the earlier decisions pay off
- **rkey pass-through (§3) makes multi-gateway coordination-free.** Because the rkey *is* the TT chip's own key, every gateway shares the same rkey→MR meaning with **zero shared state** — no distributed rkey table to keep consistent across M gateways. (A translating-rkey scheme would have required exactly that.)
- **Per-QP seq (§7.1 / M7) is a prerequisite here, not just a nicety.** Port-cumulative ACK conflates all QPs on a rail; per-QP seq decouples them, which is what lets a QP be rebalanced or migrated to another rail without seq-space collisions.
- **The gateway is the natural reliability anchor.** It already defers end-to-end ACKs (§6); centralizing per-QP ARQ/deferred-ACK state there lets a QP migrate rails while keeping its window. Each erisc ring still drains per-port; migration = drain-then-reconnect.

### 8.4 What must be added
| Layer | Work |
|---|---|
| **FW** | One `erisc_cmac_simple` instance per external rail (gap-item 6 / README v1.2 multi-erisc). Per-QP state for a rail-K QP lives in erisc-K's L1. |
| **SDK** | A `Port`/rail abstraction; `QpHandle.rail()` binding (§5.1); a rail-selection policy (hash / round-robin / explicit) that assigns QPs to rails at `connect`/`accept`. |
| **Gateway cluster** | Shared rkey namespace (free via pass-through); per-QP RC state is per-gateway (the RC QP to the remote terminates on one gateway's HCA). |
| **HA** | Failover = drain the affected QPs, re-establish (RDMA-CM reconnect) on a surviving rail; other QPs keep running. |

### 8.5 Failure & bandwidth behavior
- **Graceful degradation:** a QP survives while any rail on its path survives; losing a gateway or link reconnects only the affected QPs, others are untouched. This is the fix for the single-point-of-failure risk (`bf3-gateway-design.md §11.3`).
- **Bandwidth:** aggregate ≈ Σ rails (N × ~25 Gbps PCIe, or ~100 Gbps FPGA each), linear until chip PCIe / NoC limits. Gateways are never the throughput bottleneck (BF3 200/400 GbE ≫ per-rail TT ceiling).
- **Ordering:** concurrent one-sided WRITEs to the same MR from *different* QPs/rails have no mutual ordering — already true in verbs (no cross-QP ordering), so not a regression. Overlapping-write last-writer-wins is an app concern, as on any RDMA fabric.

## 9. Phase plan + validation gates

| Phase | Deliverable | FW change? | Validation gate |
|---|---|---|---|
| **M0** | SDK type split: `RemoteMr` + `QpHandle`; `import_remote_mr`; offset math | no | unit test: `post_write` frame bytes == hand-computed (rkey, remote_offset) |
| **M1** | `tag = qpn` for SEND-family; `RxCqe.qpn` surfaced from pushed header | no | loopback: two QPNs, SENDs demux to correct QP; no cross-talk |
| **M2** | Gateway `mr_fwd`/`qp_fwd` + one-sided routing by rkey (TT→remote WRITE) | no | `ib_write_bw` *from a TT node* into a remote CX-5 MR; bytes match |
| **M3** | Gateway active-connect proxy via `wh_mr_agent`; `ep.connect()` | no | TT node initiates RC connection to remote; QP reaches RTS |
| **M4** | Deferred end-to-end ACK + PSN↔seq map | no | kill remote mid-WRITE → TT `TxCqe` reports failure, not success |
| **M5** | TT-initiated READ (rkey-routed REQ + gateway-built RESP) | no | `ib_read_bw` from a TT node; matches Phase I ceiling |
| **M6** | Bidirectional soak: external app WRITEs to TT *and* TT WRITEs to it, same QP set | no | 1 h, 0 drops, no counter drift |
| **M7** | Per-QP ACK/seq array (§7.1) + MR table 16→64 (§7.2) | **yes** (off hot path) | slow remote on QP-A does not stall QP-B completions; ≥64 concurrent MRs; TX/RX Gbps unchanged vs M6 |
| **M8** | Dedicated QPN field, ethertype `0x1AF7`, 32 B Option-A layout (§7.3) | **yes** | dual-stack `0x1AF6`/`0x1AF7`; per-frame data-loop latency and jumbo BW within noise of M6 |
| **M9** | READ correlation re-keyed to `seq`; uniform QPN all opcodes (§7.4) | **yes** | >32 outstanding READs per QP; no correlation collisions |
| **M10** | Multi-rail (§8): per-rail erisc FW, SDK `Port`/rail abstraction, QP→rail policy, gateway-cluster HA | **yes** (multi-erisc) | 4 rails aggregate ≈ 4× single-rail BW; kill one gateway → only its QPs reconnect, others uninterrupted |

M0–M1 are pure host-SDK (no gateway, testable on the WH↔MLX rig with a loopback partner). M2+ need the gateway (or the software-only two-NIC stand-in, `bf3-gateway-design.md:502-516`). M7–M9 are the FW-accepted track (§7) — each gate includes a **BW/latency non-regression check against M6** as the acceptance bar. M10 depends on M7 (per-QP seq) and the v1.2 multi-erisc FW.

## 10. Code touchpoints

**Host SDK (this repo):**
- `tt_metal/llrt/tt_rdma_endpoint.{hpp,cpp}` (new, per host-sdk doc) — `QpHandle`, `RemoteMr`, `connect`/`accept`/`import_remote_mr`, QP-scoped `post_write/read/send`; `Port`/rail abstraction + QP→rail policy (§8, M10).
- `tt_metal/llrt/external_iface_sender.cpp` — frame builder already stamps `tag`/`rkey`/`remote_offset` (`:764-777`); add `tag = qpn` for SEND-family; expose pushed-header `tag` on RX.
- `tests/tt_metal/llrt/test_tt_rdma_qp_addressing.cpp` (new) — M0/M1 gates.

**Gateway (BF3 / software stand-in, separate tree per `bf3-gateway-design.md`):**
- `qp_table.c` / `mr_table.c` — add `mr_fwd`/`mr_rev`/`qp_fwd`/`qp_rev` + rkey pass-through.
- `translate.c` — the §2.1 per-opcode routing; deferred-ACK state machine (§6); **software CRC-32C over the 28 header bytes each direction** for `0x1AF7` frames (§7.3 Option A — WH FW no longer carries it).
- `control_plane.c` + `wh_mr_agent` — active-connect proxy (§4.2); gateway-cluster membership + QP failover/reconnect (§8.4).

**FW (`budabackend-master`, `main_cmac.cc`):** **no change for M0–M6.** FW-accepted track (§7):
- M7: `acked_idx` → per-QP array indexed by QPN; MR-table mask `& 0x0F` → `& 0x3F` + table grow (`:596`, `:114`).
- M8: header walk gains the QPN word; ethertype branch adds `0x1AF7` alongside `0xF41A`/`0x1AF6` (`:337`) — off the inner loop.
- M9: READ correlation re-keyed from `tag & 31` to `seq` (`:762-764`).
- Acceptance bar for every FW change: soak BW/latency within noise of the M6 (stock-FW) baseline.

## 11. Open questions

1. **Multi-node rkey namespace.** Pass-through rkey (§3) assumes one TT node per gateway port. For N nodes behind one gateway, either namespace rkeys per node (bits stolen from the `slot` byte) or keep a translating `mr_rev`. Which do we standardize?
2. **`connect()` addressing on the TT side.** What URI does the TT SDK use — the remote app's IP, a GID, or a gateway-local service id? Recommend gateway-resolvable IP:port to match `librdmacm` habits.
3. **RemoteMr lifetime vs peer dereg.** If the remote peer deregs an MR mid-flight, the TT SDK's `RemoteMr` goes stale. Need a gateway→SDK invalidation message (mirror of the generation-bump the local path already uses, `tt-rdma-host-sdk.md:172-174`).
4. **SEND correlation vs QPN reuse of `tag`.** Confirmed no conflict today (SEND ignores `tag`), but if we ever want a SEND completion cookie *and* a QPN, we're out of 16-bit `tag` — forces Option C. Acceptable?
5. **Inline READ correlation across the gateway.** The gateway must hold `read_pending[tag]` for the RESP; if a TT node reuses `tag` (only 32 correlation slots, `READ_CORR_N`) faster than RoCE read RTT, entries collide. Cap outstanding TT-initiated READs at `READ_CORR_N` per node (SDK already assumes <32, `external_iface_sender.hpp:185-187`). §7.4 (M9) removes this cap by re-keying on `seq`.
6. **Header integrity vs frame size (§7.3). — RESOLVED: Option A.** The `0x1AF7` header stays 32 B; +28 `header_cksum` becomes `qpn` (low 24 b). On-wire header CRC is dropped (FCS covers each link); the gateway takes on a software CRC-32C over the 28 header bytes to catch in-gateway corruption. WH FW neither computes nor checks it, keeping the reclaim free on the hot path. Layout table in §7.3.
7. **Rail-selection policy + live QP migration (§8).** Hash QPN→rail, round-robin, or explicit pinning? And can a QP migrate rails *without* teardown (drain + reattach window at the gateway), or is failover always a full RDMA-CM reconnect (simpler, but a visible transient)? MVP: static hash + reconnect-on-failure; live migration deferred.

---

## 12. Companion docs

- `tt-rdma-bidirectional-mesh-gap.md` — the full both-directions gap list; this spec closes its items 1 & 2.
- `bf3-gateway-design.md` — the gateway this spec extends (adds the outbound/initiator half).
- `tt-rdma-verbs-provider.md` — the native-verbs alternative; §3 "QPN" is superseded by §2 here (rkey-routes-one-sided, tag-as-QPN-routes-two-sided).
- `tt-rdma-wire-protocol-v1.md` — the header this spec reinterprets without changing.
- `tt-rdma-host-sdk.md` — the SDK surface item 2 extends.
