# TT-RDMA-v1 on Blackhole + BlueField-3 DOCA Gateway — Implementation Plan

**Status:** design / implementation plan. Not yet started.
**Scope:** bring TT-RDMA-v1 to a Blackhole chip (chip-side FW port) and stand up a
BlueField-3 DOCA gateway that translates RoCEv2 ↔ TT-RDMA-v1, so unmodified
`libibverbs`/MPI/UCX/`ib_*_bw` apps reach the BH node.
**Consumes (unchanged):** the wire protocol (`tt-rdma-wire-protocol-v1.md`, 32 B
header, 8 opcodes, ethertype `0x1AF6`) and the host SDK model (`tt-rdma-host-sdk.md`).
**Companion design docs:** `tt-rdma-blackhole-port.md` (chip-side), `bf3-gateway-design.md`
(gateway), `tt-rdma-mesh-addressing-spec.md` (outbound/bidirectional), `tt-rdma-fw-arch-rx.md`
(RX FW), `tt-rdma-eswitch-bypass.md` / `tt-rdma-pfc-lossless.md` (rig), `tt-rdma-verbs-provider.md`
(the host-side alternative to a gateway).

---

## 0. Foundation — the hard part is already done

The physical layer this all rests on — a **trained, stable 200G_4X Blackhole ↔ BlueField-3
link** — is **working** (`bh-erisc-fpga`, `topology-config`; Rianta MAC, AlphaWave SerDes,
link-training + the tt-smi/UMD hang fixes). Every spec in this directory *assumes* a trained
link and starts from there. TT-RDMA is **L2-payload-and-up on top of that link.** The link we
stabilized becomes the gateway's **"tt" port ↔ BH erisc** leg.

What we already have and reuse: the `bh-erisc` base FW, the trained link, the blob-swap flash
flow, and `scripts/erisc_ports.sh` + local `tt-exalens` for L1/boot_results inspection.

---

## 1. Architecture and the three workstreams

```
Compute node — unchanged RoCE app (ib_write_bw / MPI / UCX / custom libibverbs)
      │  standard RoCEv2 (UDP/4791)
      ▼
BF3 GATEWAY  ── workstream ② ───────────────────────────────────
  "wan" port : ConnectX-7 HW terminates the RC QP (via doca_rdma)
  translate  : qp_table / mr_table / psn↔seq / tt_v1_codec
  "tt"  port : doca_eth raw-L2, emits ethertype 0x1AF6
      │  TT-RDMA-v1 wire (32 B hdr)      ◀── OUR TRAINED 200G LINK
      ▼
BLACKHOLE CHIP ── workstream ① ─────────────────────────────────
  bh-erisc base FW (RISC0)              ◀── DONE: trains link + mailbox loop
  + tt-metal active-eth RDMA kernel (Profile E, raw mode):
      RX classifier(TCAM 0x1AF6)→SW L1 ring → opcode dispatch → NoC write/read
      MR table + rkey validation + WQE ring
      │
      ▼ NoC → on-chip L1/DRAM (line rate)  |  host hugepage (PCIe-bound)  |  other chips
Host SDK ── workstream ③ ── TtRdmaEndpoint (register_mr / post_send_* / poll_cq)
```

Three tracks, one integration point (the `0x1AF6` wire we already carry). ① and ② are
independent until the interop gate.

| # | Workstream | Repo | Phases |
|---|---|---|---|
| ① | BH chip-side FW (tt-metal active-eth RDMA kernel) | `tt-metal` (`active/subordinate_erisc.cc`) + `bh-erisc` facts | BH.0–BH.7 |
| ② | BF3 DOCA gateway (`tt_rdma_gw`) + `bh_mr_agent` | new | G.1–G.7 |
| ③ | Host SDK (`TtRdmaEndpoint`) retargeted to BH | `tt-metal/llrt` | folds into BH.3 |

---

## 2. Workstream ① — Blackhole chip-side

### 2.1 FW model (non-negotiable)

The RDMA path is **not** a standalone FW like WH's `erisc_cmac_simple`. It is a **tt-metal
active-eth kernel** coexisting with the resident `bh-erisc` base FW:

- **RISC0 (`active_erisc`)** keeps the go-message loop and **must periodically yield to the
  base FW** via `internal_::risc_context_switch()` → `service_eth_msg()`, or the link dies.
- **RISC1 (`subordinate_erisc`)** is a free data mover — **home for the tight RDMA TX/RX loop.**
- No hard watchdog: yield often enough that no mailbox client times out; the link-status check
  self-debounces to `ETH_UPDATE_LINK_STATUS_INTERVAL_MS = 1000 ms` and must **not** be forced
  faster. CMFW only samples the eth heartbeat every 100 ms as telemetry and takes no action.
- Budgets: **24 KB code / RISC**, 8 KB local / RISC (keep RDMA state in L1), **512 KB shared L1**.

### 2.2 The L1 relayout — WH map → BH (the meat of BH.2/BH.3)

WH packs its map into a small SRAM. BH has 512 KB — relay out **freely inside `0x0–0x70000`**.
The one rule: **never write `0x70000+`** (base FW code/data, `boot_results 0x7CC00`, api-table
`0x7CF00`, mailboxes `0x7D000`, NoC counters `0x7D040`) — writing there **bricks the link**
(this is the region `erisc_ports.sh` *reads*).

| WH structure | WH addr | Size | BH action |
|---|---|---|---|
| RX ring (BUF_WRAP) | `0x4000` | 14 KB | **DONE:** RXQ2 raw BUF_WRAP, 128 KB ring. NB the RX-classifier TCAM does **not** exist in shipped FW (`eth_rx_flow.cpp` is dead code) — reception is the base-FW dst-MAC router (unicast→RXQ2); the kernel SW-filters/parses. See `tt-rdma-rx-dispatch-spec.md` |
| MR table | `0x8500` | 16×32 B | re-place; `rkey=(slot<<24)|(rand16<<8)|gen`, O(1) direct-index lookup; **grow to 64 slots** (free; NCCL/UCX need it) |
| RCB header + doorbells | `0x8400` | 64 B | re-place; ctrl doorbell `+0x28`, cumulative-ACK `rcb[2]` |
| WQE descr + payload pool | `0x8000` / `0x9000` | ~128 KB | re-place |
| TX_BUF0/1 | `0x29000/0x2A000` | — | re-place; must match SDK `kTxBuf0/1` |
| READ correlation table | `0x8800` | — | re-place |
| RxWqeRing (SEND landing) | **host hugepage +128 KB** | 96 KB | **not in L1** — DMA-pushed to host; unchanged |

MR entry: `+0 base_noc(u64) +8 length(u64) +16 rkey(u32) +20 access_flags +24 pd +28 rsvd`.
Validation per inbound WRITE/READ: `rkey_miss / rkey_access / rkey_bounds / rkey_wrap` → drop+log.

### 2.3 Two-profile FW (committed BH design directive)

Ship one BH RDMA FW that runs two link profiles, selected per erisc rail at bring-up:

| Aspect | **Profile E — external / RoCE-interop** | **Profile L — TT-Link / chip-to-chip** |
|---|---|---|
| Partner | gateway / Mellanox / FPGA (non-TT) | another TT chip |
| TX | `eth_send_raw` | `eth_send_packet` (remote `DEST_ADDR`) |
| RX | RXQ **raw** + BUF_WRAP; FW parses | RXQ **packet**; HW lands at `DEST_ADDR` |
| Reliability | **software** (Phase R) / gateway deferred-ACK | **HW** seq + Go-back-N |
| WRITE path | ring → `rkey` lookup → `noc_write` | HW writes remote NoC addr directly, **no remote FW, no MR check** |
| Access control | **rkey/access/bounds enforced** on target | none on wire (trusted fabric) |
| Fan-out | 1 DEST_MAC + in-band rkey/tag-QPN → ~65 k endpoints | fixed `DEST_ADDR`, **≤3 peers/erisc** (3 TXQ) |
| Ceiling | host-PCIe-bound per rail (see §5) | line rate (400 G) |

**Profile E is the gateway path and the MVP. Profile L (line-rate, host-bypassed TT-Link) is a
separate higher-value follow-on (BH.5) with a trust decision (drops rkey/bounds enforcement) —
do not block external interop on it.**

### 2.4 Phase plan (chip-side)

| Phase | Prof | Deliverable | Base-FW change | Gate |
|---|---|---|---|---|
| **BH.0** | — | Active-eth RDMA skeleton: RISC1 tight loop + RISC0 base-FW yield; link stays up | no | `port_status` UP 10 min with kernel resident + periodic `service_eth_msg()` |
| **BH.1** | E | Raw TX: build 32 B v1 hdr + payload in SW L1, `eth_send_raw` to BF3 "tt" MAC | no | BF3 receives valid `0x1AF6` frame; bytes match |
| **BH.2** ✅ **DONE** | E | RX: raw RXQ2 (dst-MAC router, **NOT** a TCAM — `eth_rx_flow.cpp` is dead code) → SW L1 landing; opcode dispatch (SEND/WRITE); BUF_WRAP streaming + 128 KB ring | no | **Met:** inbound WRITE byte-exact; jumbo dispatch 264k frames/s, bad=0, 8.7 Gbps lossless. See `tt-rdma-rx-dispatch-spec.md` |
| **BH.3 (core)** ✅ **DONE** | E | MR table (64 slots, rkey lookup + bounds) + WRITE via `noc_async_write` to off-core MR target (RISC off the copy) | no | **Met:** WRITE lands byte-exact on Tensix L1 via NoC; **0.02 → 8.48 Gbps (~128×)**, lossless. Remaining BH.3: host `TtRdmaEndpoint` retarget + SEND→host RxWqeRing |
| **BH.4** | E | Gateway interop | gateway only | `ib_write_bw` from remote CX into a BH MR; bytes match |
| **BH.5** | L | TT-Link packet mode, HW ARQ, WRITE-without-remote-MR; BH↔BH / BH↔WH | no | sustained WRITE, 0 drops, no SW retx; line rate to ≤3 peers/erisc |
| **BH.6** | E | PFC / lossless on **Rianta** (the `eth_init.cpp:538` TODO; WH DWC procedure does NOT port) | maybe | pause counters non-zero under load, 0 buffer-discard |
| **BH.7** | E | (stretch) RoCE HW-offload probe (iCRC/UDP/IP insert) — HW hooks exist, **no shipped stack** (`eth_rx_flow.cpp` uncompiled) → future track, not MVP | TBD | HW-inserted frame accepted by a stock RoCE peer via reduced gateway |

---

## 3. Workstream ② — BF3 DOCA gateway

### 3.1 De-risk first: x86 software stand-in

Before committing to BF3 Arm + the DOCA learning curve, run the **identical** translation logic
on a **plain x86 box with two ConnectX NICs** ("wan" + "tt") — libibverbs/DPDK, ~2× slower but
functionally identical. Validate the architecture there, then port to BF3. (Our BF3 is already
wired as the "tt"-side partner.)

### 3.2 Components

Daemon **`tt_rdma_gw`** (per `bf3-gateway-design.md §4`):
`doca_rdma_endpoint.c` (terminate RoCE QPs) · `doca_eth_endpoint.c` (raw-L2 "tt") ·
`translate.c` · `qp_table.c` (RoCE QPN→TT seq/MR) · `mr_table.c` (RoCE rkey→TT slot) ·
`psn_seq.c` (PSN↔seq) · `control_plane.c` (gRPC/REST) · `tt_v1_codec.{c,h}`.

Plus **`bh_mr_agent`** on the BH host: BF3 has no PCIe path to the chip, so MR registration
traverses BF3 → mgmt link → BH host → `TtRdmaEndpoint::register_mr` → chip.

DOCA usage (grounded):
- **"wan" (RoCE) = `doca_rdma`**, RC transport, RDMA-CM (`doca_rdma_start_listen_to_port` /
  `connection_accept`). Incoming RDMA_WRITE lands in a `doca_mmap` (RDMA-write perm); READ
  sources from one. **Must use `doca_rdma_*`, not raw verbs.** HW QP termination also *is* the
  eSwitch-bypass answer (no `tc`/slow-path).
- **"tt" side = `doca_eth`** (or DPDK) raw L2 — emits/receives `0x1AF6`.
- **`doca_flow`** = coarse steering / RSS across translation workers + VLAN/PCP=3 tt-egress tag.
  **Not** the full protocol rewrite (see §5/§6).

### 3.3 Per-opcode translation (from `bf3-gateway-design.md §5`)

`RC_SEND_ONLY→0x01` (rkey=0) · `RC_RDMA_WRITE_ONLY + RETH→0x10` with
`tt_remote_off = RETH.vaddr − mr_table[rkey].vaddr_base + tt_base_off` · `WRITE_WITH_IMM→0x11` ·
`RC_RDMA_READ_REQUEST→0x20`, correlate `(PSN,QPN,len)`, on `0x21` build `RC_RDMA_READ_RESPONSE`
+ AETH · multi-segment READ chunked at MTU (MVP caps at 4080 B) · **ACK: RoCE per-QP PSN ↔ TT
cumulative seq** · **ATOMIC: MVP NAKs** (v1.2 adds TT 0x30-range).
MR models: **A = explicit pre-registration (MVP)**, B = app-transparent `ibv_reg_mr` intercept (later).

### 3.4 Phase plan (gateway)

| Phase | Scope | Eng-wks |
|---|---|---|
| G.1 | BF3 (or x86) up; toy `doca_rdma` + `doca_eth` endpoints, no translation | 1 |
| G.2 | RC SEND → TT SEND (one pre-registered MR) | 2 |
| G.3 | RC RDMA_WRITE → TT WRITE | 1 |
| G.4 | RC RDMA_READ → TT READ_REQ/RESP (multi-segment reassembly) | 2 |
| G.5 | ACK round-trip (PSN ↔ cumulative seq) | 1 |
| G.6 | RDMA-CM connection setup, multi-QP | 1 |
| G.7 | App integration: MPI/UCX hello-world | 2 |

---

## 4. Workstream ③ — Host SDK

`TtRdmaEndpoint` (`register_mr` / `post_send_write[_imm]` / `post_send_read` / `post_recv[_any]` /
`poll_completion` / `poll_rx_completion`) composes the existing `ExternalIfaceSender`. It is
**chip-agnostic by design — only `bring_up()` and the RCB/MR L1 addresses are BH-specific**, and
`bring_up` must coexist with the base FW (mailbox/api-table), not own the core. Key facts:
`register_mr` needs **hugepage-backed** memory (`map_hugepage_to_noc`, IOMMU-pinnable);
reliability is **auto-on and non-negotiable for one-sided WRITE/READ**; the FW MR table holds
only *local* regions a peer targets (outbound-only nodes need 0 slots). Folds into BH.3.

---

## 5. Throughput — the road to ~200 Gbps line rate

**The BF3 is not the bottleneck** (200/400 GbE ports, PCIe Gen5 x16). The shipped v1 number
(~25 Gbps) comes from ceilings elsewhere. Line rate = clearing all three:

| Stage | Ceiling | Fix (lever) |
|---|---|---|
| wan RoCE ingress | line rate ✓ | — |
| **BF3 translation** | **~25 Gbps** (per-packet Arm C) | §6 tiering |
| tt → BH (our 200G link) | line rate ✓ | — |
| **BH erisc RX (raw dispatch)** | per-frame FW cost | jumbo + multi-rail |
| **NoC write → MR target** | **~25 Gbps** host-hugepage / **line rate on-chip** | on-chip targets |

Levers, in impact order:
1. **On-chip MR targets (DRAM/L1), not host hugepage** — the shipped 25 Gbps ceiling *is* the
   host-PCIe (Gen3 x4) hugepage case. `mr.base_noc_addr` pointing on-chip is NoC line rate,
   no host. Biggest single lever.
2. **HW-terminate the RoCE QP + zero-copy payload + translate headers only** (see §6) — kills the
   translation ceiling and doubles as eSwitch-bypass.
3. **Jumbo frames (4080 B)** — amortizes per-frame cost on BF3 and the BH erisc FW. (4096 works,
   9216 hangs.)
4. **Multi-rail aggregation** — a single Profile-E rail's FW dispatch may not sustain 200; BH has
   **14 ETH SS**. MRs are **rail-agnostic** (NoC is chip-global) → stripe *QPs* across rails
   (NCCL/UCX pattern: pin a QP to one rail, aggregate with many QPs; never per-packet stripe a QP).
5. **PFC-lossless (priority 3)** — no drops at line rate. Needs the BH Rianta PFC bring-up (BH.6).

What DOCA Flow does/doesn't do: **yes** to line-rate steering / RSS / VLAN-PCP tagging; **no** to
the full RoCE→TT translation — it can't match the BTH opcode deep in the UDP payload, and
**per-packet monotonic `seq` stamping is not a flow action.** So the translation lives in §6, not
in Flow.

> **Honest bottom line:** v1 today = ~25 Gbps. 200 Gbps is a real engineering push: on-chip
> targets + HW-QP-termination/zero-copy + jumbo + multi-rail. Profile L (TT↔TT, packet mode) is
> effortlessly line-rate but is **not** the gateway path.

---

## 6. Translation-engine tiering — Arm → zero-copy-Arm → DPA

The gateway's translation stage is the throughput crux. Three tiers, escalate only as needed:

| Tier | What | Rate | Latency | Complexity | When |
|---|---|---|---|---|---|
| **T1 — Arm software** | DPDK/DOCA polls, per-packet C on A78 cores builds TT hdr | ~25 Gbps | ~1–5 µs | low | **MVP (G.1–G.7)** — correctness first |
| **T2 — zero-copy Arm** | ConnectX HW terminates the RC QP; Arm touches **only the 32 B header**; payload DMA'd port-to-port (never `memcpy`'d) | approaches line rate for **jumbo** (cost is O(headers), not O(bytes)) | ~µs | medium | first optimization; **may suffice — try before DPA** |
| **T3 — DPA** | translation runs as a **DPA kernel in the NIC datapath** (`doca_dpa`) | datapath speed, no PCIe hop to Arm | ~hundreds ns | high | when T2 isn't enough (small-frame/high-pps or lowest latency), and for the **outbound** path |

**Why DPA is the right tool where Flow can't reach.** DPA is the missing middle tier: fully
**programmable per-packet C (like the Arm)** running **in the datapath (like Flow)**. The two
things that blocked full DOCA-Flow offload — (a) parsing the RoCE BTH/RETH and (b) per-packet
`seq` stamping + `rkey→tt_rkey` + `remote_offset` computation — are trivial DPA-kernel code, and
the DPA runs them far faster than the Arm.

**DPA-Verbs — the standout fit for the outbound / bidirectional (TT-as-initiator) path.**
`doca_dpa_dev_verbs.h` lets the DPA kernel issue RDMA **directly** (host configures the QP, DPA
uses it) with **no Arm/PCIe round-trip** — exactly what the mesh spec's hard half needs (active
connect, deferred end-to-end ACK, outbound routing), where an Arm round-trip would dominate latency.

**DPA caveats (why it's phase-2, not MVP):**
- Constrained env + `dpacc` toolchain + **DOCA-must-match-DPACC** version coupling.
- DPA kernels are **bounded, not free-running** — there's a max-kernel-time-alive cap
  (`doca_dpa_cap_get_max_kernel_time_alive_supported`); a DPA gateway is **event/completion-driven**
  (wake on RQ/CQ, process, yield), not a `while(true)` poll.
- DPA does **not** terminate the RoCE QP — the RC state machine (PSN/reassembly/ACK) stays in
  ConnectX HW; DPA-Verbs *uses* a host-configured QP.
- 200 Gbps on DPA is a **per-packet-cycle-budget question to measure**: jumbo → ~6 Mpps
  (plausible across DPA threads); small frames harder.

**De-risk DPA early:** a micro-benchmark DPA kernel (parse synthetic RoCE frame → emit TT frame,
measure Mpps/thread) tells you whether the cycle budget closes at 200 Gbps *before* committing the
translation engine to DPA.

**Composed line-rate data path:**
```
wan ─▶ ConnectX HW terminates RC QP  (PSN/ACK/reassembly at line rate)
       DOCA Flow: RSS/steer flows → workers; tag tt-egress (PCP=3)
       [T2 Arm header-only | T3 DPA kernel]: parse RoCE → 32B TT hdr + seq/rkey/off; payload zero-copy
       ── tt raw-L2 0x1AF6 ─▶ BH erisc → on-chip MR target (NoC line rate)
                                        + jumbo + multi-rail across 14 ETH SS
```

---

## 7. Scope boundaries

1. **MVP is *inbound only*** (external app = requester, BH = passive target). Making BH a
   *symmetric* endpoint that also **initiates** RDMA (real mesh / disaggregated-inference push) is
   a **separate track** — the mesh-addressing spec's M0–M10: per-opcode routing (one-sided by
   `rkey`, two-sided by `tag`=QPN), host-side `RemoteMr` import, a `QpHandle` object, gateway
   active-connect proxy, and **deferred end-to-end ACK** (don't ACK the TT sender until the real
   RoCE ACK returns). **No WH/BH FW change** for the MVP subset — all host-SDK + gateway — but do
   not fold it into the inbound MVP. DPA-Verbs (§6) is the natural accelerator for this track.
2. **The gateway is a deliberate choice, not the only door.** The `libtt_rdma` **verbs provider**
   (native `rdma-core` provider) lets unmodified apps hit the BH chip over `0x1AF6` with **no BF3
   at all** — better for a *direct host↔TT (FPGA-TT-Link)* path where a BF3 just re-adds a PCIe
   bottleneck. The **BF3 gateway wins when you need a standards-native NIC fronting a multi-vendor
   RoCE fabric/switch** (the stated goal here).

---

## 8. Sequencing, critical path, MVP

**Critical path = ① BH.0→BH.3 ∥ ② G.1→G.3, meeting at BH.4/G.4.** Independent until the interop
gate (they share only the already-working wire).

**MVP =** `ib_write_bw` from a stock ConnectX → gateway → bytes land in a BH MR at the right
offset, ACKs working. That is **BH.0–BH.4 + G.1–G.5 + SDK retarget + `bh_mr_agent`**.

| Weeks | Chip ① | Gateway ② |
|---|---|---|
| 0–2 | BH.0 (skeleton, link stays up) | G.1 on the **x86 de-risk box** (endpoints, loopback) |
| 2–5 | BH.1–BH.2 (raw TX/RX + dispatch) | G.2–G.3 (SEND, WRITE) — first cross-wire frame BH.1↔G.1 |
| 5–8 | BH.3 (MR table + SDK) | G.4–G.5 (READ + ACK) + `bh_mr_agent` |
| 8–10 | BH.4 interop | G.6–G.7 (`ib_*_bw`, MPI); port x86→BF3 |

Team (per docs): ~1 DOCA dev + ~1 BH FW dev + 0.5 host/integration. **~10 weeks to MVP.**
Post-MVP: T2/T3 throughput (§6), Profile L (BH.5), PFC (BH.6), the outbound track (§7), multi-rail.

---

## 9. Key decisions & risks

1. **Profile E only for MVP** — raw + software/gateway reliability. Profile L is a separate track
   with a trust decision. Don't conflate.
2. **RISC1 for the RDMA loop; RISC0 keeps yielding** `service_eth_msg()`. A persistent loop on
   RISC0 starves base-FW link maintenance (directly related to the link-liveness behavior debugged
   during bring-up). Load-bearing FW-shape decision.
3. **Reliability is software/gateway on the external path** — HW ARQ can't be used toward a non-TT
   partner. Cumulative-ACK first; selective-ACK (v1.1) once over lossy fabrics.
4. **BH PFC is unimplemented** (`eth_init.cpp:538` TODO, Rianta) — its own bring-up (BH.6). MVP can
   run without it on a point-to-point link that won't congest at ~25 Gbps.
5. **RoCE HW-offload is HW-hooks-only, no shipped stack** — "thin gateway"/wire-native RoCE is a
   future track, not MVP. MVP keeps `0x1AF6` + full gateway.
6. **Line rate is a stack, not a flag** (§5/§6) — de-risk the T2 zero-copy path and a DPA
   micro-benchmark early if 200 Gbps is a hard requirement.
7. **Single BF3 = SPOF** → the production answer is a rail matrix (many erisc rails × gateways);
   rkey is rail-agnostic so failover = drain + RDMA-CM reconnect of affected QPs (mesh spec §8).

---

## 10. First concrete step (this/next week)

- **Chip:** stand up **BH.0** — a minimal `active_erisc` kernel that loops + calls
  `service_eth_msg()` on RISC0 and idles RISC1, and prove `port_status` stays UP for 10 min with
  it resident (reuse the flash + `erisc_ports.sh` poll). Validates the coexistence model before
  any RDMA logic.
- **Gateway:** on the x86 de-risk box (or BF3 Arm), **G.1** — a `doca_rdma` loopback + a
  `doca_eth` raw-L2 send, no translation. Start from `/opt/mellanox/doca/samples/doca_rdma/` +
  `.../doca_eth/` and adapt (do not hand-roll; do not fall back to raw verbs / `tc`).

---

## 11. Fabric ↔ external integration — TT↔TT stays stock tt-fabric

The chip runs a **mix of firmware/kernels across its 14 independent ETH SS**, selected per rail:
- **TT↔TT rails → stock tt-fabric (unchanged).** tt-fabric / the EDM *is* tt-metal's chip-to-chip
  mechanism — reliable, line-rate, HW-sequenced mesh routing, with its own reserved L1 region
  (`MEM_ERISC_FABRIC_ROUTER_RESERVED_BASE`). No custom code, no risk.
- **External rails (erisc↔BF3/Mellanox) → the custom TT-RDMA active-eth kernel** (Profile E).

This works because tt-fabric only routes between TT chips; it will never route over a link whose
partner is a non-TT BF3/Mellanox. So external cores are inherently *not* fabric cores — free for the
RDMA kernel — and the fabric cores are untouched. Dispatch is per-`CoreCoord` (`CreateKernel`), and
the **`board_topology.yaml` role (EXTERNAL vs TT_INTERNAL)** is the selector for which rails get the
RDMA kernel.

**This supersedes the custom "Profile L" of §2.3/§7.** Profile L was re-inventing reliable TT↔TT,
which is exactly what tt-fabric already does — so **TT↔TT uses stock tt-fabric, and the custom
surface area shrinks to the external rails + the BF3 gateway.** Keep Profile L in mind only as a
fallback if a deployment can't run tt-fabric on the TT↔TT rails.

### 11.1 How much does the external rail integrate *under* tt-fabric?
Stock tt-fabric **cannot** route packets over the external link (foreign wire — TT-RDMA-raw/RoCE to
a non-TT partner). Integration is via a **bridge**, at increasing cost:

| Level | What | Status |
|---|---|---|
| Stock fabric routes over the external link | — | **No** — protocol/partner mismatch; external rail is not a fabric port |
| **Loose (NoC hand-off)** | workload moves fabric-delivered data (in L1/DRAM) to the RDMA rail; the two mechanisms meet only at the chip-global NoC | **Available now**, no fabric change — the near-term path |
| **Tight bridge (edge node)** | an on-chip edge that *terminates* fabric and *re-originates* TT-RDMA (egress), and *injects* inbound TT-RDMA into fabric (ingress) | **Custom** — the on-chip half of the gateway; real, novel work |
| **Transparent** (`fabric_send(external_dest)` "just works") | fabric control plane carries an external-egress destination type + the tight bridge | **Frontier** — the uncovered mesh-egress / bidirectional gap (`tt-rdma-mesh-egress-multicast.md`, `tt-rdma-bidirectional-mesh-gap.md`) |

**The composition (loose, near-term):** interior mesh chip ──tt-fabric (TT↔TT)──▶ edge chip L1/DRAM
──chip-global NoC──▶ edge chip's EXTERNAL erisc ──custom TT-RDMA (`0x1AF6`)──▶ BF3 gateway ──▶ RoCE.
Fabric and RDMA meet at the NoC; no custom TT↔TT protocol needed.

### 11.2 Coexistence verification (a gate, not an assumption)
Architecturally sound (independent SS; matches `runtime-workload-coexistence.md`), but verify in the
tt-metal runtime: **a custom user active-eth kernel on the external cores running concurrently with
tt-fabric active on the TT↔TT cores** in the same mesh — confirm the external cores are *not* in the
fabric reserved set and that enabling fabric doesn't claim them. Add this as an explicit BH-phase
gate before relying on the mix.

## 12. Bootstrap — starting the actual build

**Principle: get one frame across the real link, both ways, before any RDMA semantics.** The
wire is frozen; everything else layers onto a working spine. Do **not** start with `doca_rdma`/
RoCE, MR tables, or the x86 de-risk box — the BF3 (`mt41692`) is already on our trained link, so
the fastest proof is tcpdump both ways, then DOCA-ify.

**The one spec artifact (written first, committed with this plan):**
- `tt_metal/hw/inc/internal/ethernet/tt_rdma_wire.h` — the 32 B header struct, opcodes, MR entry,
  and golden test vectors (frozen from `tt-rdma-wire-protocol-v1.md`). **Both** the BH kernel and
  the DOCA gateway build against it (the gateway vendors a copy).
- `tt_metal/hw/inc/internal/ethernet/tt_rdma_l1_layout.h` — the chip-side SW-L1 region map +
  compile-time no-touch guards (`0x70000+` bricks the link; `0x40000..0x42000` is the RISC0 reset
  save). `TT_RDMA_L1_BASE` is the one value BH.0 pins against the real build.

**Two dev inner loops to stand up:**
- **TT:** base FW (link) = `bh-erisc-fpga` blob-swap flash (done); active-eth kernel = build in
  `tt-metal-external-eth`, load onto the eth core's **RISC1** via the runtime/llrt path, read state
  with `tt-exalens brxy` + `erisc_ports.sh`. Standing this loop up = BH.0 (the first new infra).
- **BF3:** DOCA sample loop from `/opt/mellanox/doca/samples/doca_eth/` (meson + `pkg-config doca`;
  fix `PKG_CONFIG_PATH` or build on the BF3 Arm — route to `doca-setup` if `pkg-config doca` fails).

**Bootstrap milestones (the spine):**

| M | What | Validate | DOCA? |
|---|---|---|---|
| **M-1a** | TT RISC1 kernel `eth_send_raw`s one golden `0x1AF6` frame | `tcpdump -i <ttport> ether proto 0x1af6 -xx` on BF3 | none (smoke) |
| **M-1b** | Frame into TT: `scapy sendp()` a `0x1AF6` frame from BF3 → RX-classifier lands it at `TT_RDMA_RX_RING_ADDR` | `tt-exalens brxy`, byte-match golden | none |
| **M-2** | Replace tcpdump/scapy with a **`doca_eth`** RX+TX program on the BF3 | doca_eth RX dump matches golden | **yes** |
| **M-3** | **Loopback contract test:** TT→BF3→TT echo of a full 32 B-header frame, byte-exact | automated (SDK sends, tt-exalens confirms) | yes |

**M-3 green = spine works on real hardware.** Then layer opcode dispatch + MR table (BH.2/BH.3) on
the chip and `doca_rdma` wan-side termination + translation (G.2+) on the BF3.

**First week:** Day 1 — DOCA build env + `doca_caps`/`mlxlink` to identify the tt-port device;
tcpdump the link. Days 2–3 — **BH.0** (heartbeat RISC1 kernel, `port_status` UP 10 min while
`dbg_heartbeat` advances). Days 3–4 — `tt_rdma_wire.h` wired in + **M-1a**. Days 4–5 — **M-1b** →
**M-2** → **M-3**.

**Repo / branch layout:** `bh-erisc-fpga` = base FW/link (unchanged); **`tt-metal-external-eth`
branch `aperezvicente/tt-rdma-bh-bf3`** = BH active-eth kernel + SDK + the two headers + this plan;
new **`tt_rdma_gw/`** = DOCA gateway, seeded from `doca_eth`, vendors `tt_rdma_wire.h`.

**Why this order de-risks:** it hits the three riskiest new things first — active-eth *coexistence*
(BH.0: does the link survive a resident kernel?), *raw-L2 on real silicon* (does `eth_send_raw` /
the RX-classifier TCAM move a `0x1AF6` frame?), and *DOCA build/bind on the tt port* — **before** any
RoCE/MR/QP complexity, and reuses everything from the link bring-up (trained link, flash flow,
`erisc_ports.sh`, `tt-exalens`, the BF3 already on the wire). The shared headers + M-3 test mean
both sides build against identical bytes from day one — the spec is executable, not prose.

## 13. References

- Wire: `tt-rdma-wire-protocol-v1.md` · SDK: `tt-rdma-host-sdk.md` · RX FW (WH, older): `tt-rdma-fw-arch-rx.md`
- **BH chip-side, validated on silicon:** TX `tt-rdma-tx-ring-spec.md` (200G/rail, 397G aggregate) ·
  RX `tt-rdma-rx-dispatch-spec.md` (dispatch + MR WRITE via noc_async_write, 8.5 Gbps, push model + perf headroom)
- Chip-side: `tt-rdma-blackhole-port.md` · Gateway: `bf3-gateway-design.md`
- Outbound/bidirectional: `tt-rdma-mesh-addressing-spec.md`, `tt-rdma-bidirectional-mesh-gap.md`
- Switch interop: `tt-rdma-switch-interop-architecture.md` · Alt: `tt-rdma-verbs-provider.md`
- Rig: `tt-rdma-eswitch-bypass.md`, `tt-rdma-pfc-lossless.md`
- BH ETH SS: `bh-erisc` `docs/eth_arch_spec.md` · Physical link: `bh-erisc-fpga` (`topology-config`)
- DOCA: RDMA / Eth / Flow / DPA samples under `/opt/mellanox/doca/samples/`

---

## §14. Performance mandate — max BW + min latency, RISC-V OFF the datapath (2026-07-23)

**Standing design principle for every part of this spec:** design for maximum bandwidth AND minimum
latency, and keep the RISC-V eth cores (and the host CPU) OUT of the per-packet datapath. Data
movement is a job for hardware / programmable datapath engines; the RISC-V FW and host only set up
rings + descriptors and manage control/completions.

**Why (measured + spec-grounded):**
- Raw-mode ethernet is NOT inherently low-BW/high-latency (same MAC/SerDes as TT-link; lower latency
  — no seq/ACK handshake). Measured raw TX on the BF3 rail: 0.86 Gbps @78 B → **43 Gbps aggregate
  @4 KB (2 rails)**. Raw CAN go fast.
- BUT a RISC-driven send loop caps at the RISC command rate (~650k cmd/s/rail). Per-frame at 4 KB =
  ~21 Gbps/rail = ~11% of 200 G. A per-command "burst" attempt (one big START_RAW, HW auto-split via
  ETH_TXQ_MAX_PKT_SIZE_BYTES @TXQ+0x0C) regressed — the RISC-in-the-loop busy-wait per big command
  serializes. **The ceiling is RISC-on-the-datapath, not raw mode and not the wire.**
- How stock TT-TT hits 400 G line rate (tt-isa EthernetTile + bh-erisc-orig-1.12.0): the NoC-overlay
  stream engine tunnels data over the eth link autonomously (RISC programs the stream once/phase),
  and TT-link packet mode does HW seq#/resend + accept-ahead. **The RISC is not in the inner loop.**
  This path is TT-proprietary (needs a TT peer) → unavailable to a BF3 gateway.

**Datapath architecture (both ends off the CPU/RISC per-packet path):**
- **TT (chip) side:** RISC1 owns a descriptor/WQE ring + control only. Payload is streamed into the
  eth TX buffer by a DMA/overlay producer; the eth TXQ drains to the wire. Use accept-ahead
  (multiple outstanding), MAX_PKT auto-split, jumbo frames, all 3 TXQ, and all 14 ETH SS in
  parallel. The RISC never busy-waits a transfer to completion in the fast path.
- **BF3 (gateway) side:** **DOCA DPA is the datapath engine** — the RoCEv2 ↔ TT-RDMA-v1 translation
  (BTH parse, per-packet seq/rkey stamp, MR lookup, reorder) runs on the DPA's programmable cores at
  line rate, NOT on the BF3 Arm CPU and NOT on the host. Arm/host only manage QP/MR tables + control.
  This is the plan's T3 tier promoted to the DEFAULT target (T1 Arm-software is a bring-up crutch).
- **Reliability without HW resend (raw mode):** lossless via PFC on the direct-attach link (no
  drops in steady state) + a lightweight DPA/SW ARQ for the rare loss — not a per-packet CPU tax.

**Consequence:** the gateway BW target is "best RISC-off-datapath raw pipeline + DPA translation,"
not TT-TT's HW-tunnel number. Every milestone (BH.2+ TX ring, gateway G.x) is specced to keep both
CPUs off the per-packet path from the start.
