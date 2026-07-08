# TT-RDMA Switch & RDMA-Fabric Interop: Gateway Path vs Native-Switch Path

**Status:** architecture note / decision framing. Reconciles the TT-RDMA-v1
gateway design with the Exabox "TT Ethernet can't attach to an L2 switch"
finding (design review 2025-05-13).

**One-line answer:** to make a Tenstorrent node interoperable on an L2 switch or
a RoCEv2 fabric *today*, you do **not** fix the TT silicon — you keep TT
point-to-point to a **gateway** that is itself a standards-native RDMA NIC, and
carry remote-endpoint identity **in-band** in the 32 B TT-RDMA header. The
silicon fix (parallel queues + linked-list ARQ + a bigger header table) is the
*other* path — native switch attach without a gateway — and is a separate,
later, hardware track.

---

## 1. Why TT cannot be a switch endpoint directly (the HW limits)

Verified in `bh-erisc` v1.12.0 register map
(`src/common/registers/eth_core_a_reg.h`); the same shape holds on Wormhole and
Grendel. Two independent limits, both properties of **TT-Link mode** (HW
DMA + sequence-number + retry):

1. **Per-queue HW N-ARQ binds retransmit to one fixed destination.** Only
   **3 TXQ / 3 RXQ** (`ETH_TXQ_{0,1,2}`), each TXQ with its **own `DEST_ADDR` +
   `REMOTE_SEQ_TIMEOUT`**. The retry/seq context is bound to a single
   destination MAC per queue ⇒ the HW model is `{Source → one fixed Dest}`, the
   opposite of switching's `{Source, Any Destination}`. "3 endpoints for
   Blackhole, 4 for Grendel."
2. **Header Configuration Table is 10 rows.** The TX packet-config table
   `ETH_TXPKT_CFG_A_{0..9}` (each row `{INSERT_CTL, VLAN1, VLAN2, CUSTOM_HDR}`)
   is the set of prependable headers a TXQ can select ⇒ ~10 reachable DA/MAC
   templates. This is the "one or ten destinations" figure.

The original architecture was point-to-point mesh (2D mesh, N-D torus; Device-ID
within a ≤1024-device mesh, Mesh-ID across meshes via exit nodes), so neither
limit was a defect *for that use*. They only bite when you try to attach TT
directly to a flat, switched, multi-vendor network.

---

## 2. The two paths

```
        ┌─────────────────────── PATH A: GATEWAY (available now) ───────────────────────┐
        │                                                                                │
  TT chip ──raw L2, 1 DEST_MAC──▶ GATEWAY ─────RoCEv2, real MAC/GID table────▶ L2 SWITCH ─▶ fabric
  (stock FW,                      (ConnectX-7 / FPGA:                          (any vendor)
   point-to-point,                 HW QP term, PSN, iCRC,
   in-band endpoint id)            DCQCN, big MAC table)

        └────────────────────────────────────────────────────────────────────────────────┘

        ┌─────────────────── PATH B: NATIVE SWITCH ATTACH (Exabox A0 / Keraunos 2) ───────┐
        │                                                                                  │
  TT chip ─────────────────standard Ethernet / RoCEv2─────────────────▶ L2 SWITCH ─▶ fabric
  (NEW SILICON: parallel queues + linked-list ARQ decoupled from a
   queue-static DEST_ADDR, + DA-indexed / memory-resident header table)

        └──────────────────────────────────────────────────────────────────────────────────┘
```

These are **complementary, not competing**: Path A is the near-term multi-vendor
interop story on shipped hardware; Path B is the long-term "gateway-less" native
switching capability that needs the A0 minimum-switch or Keraunos-2 silicon.

---

## 3. Path A — the gateway, and why the HW limits don't bind

The load-bearing move: **TT never attaches to the switch; the gateway does.**
The TT↔gateway link is strictly point-to-point (`bf3-gateway-design.md:95`),
and the gateway's *other* port is the genuine switch/fabric endpoint.

| Exabox HW limit | Blocks direct-on-switch because… | Doesn't bind via gateway because… |
|---|---|---|
| 1 hardcoded `DEST_MAC` + 10-row header table | can't address >~10 peers at L2 | TT emits **one** frame to **one** MAC (the gateway) — it needs a MAC table of size **1**. The large MAC/GID table lives on the gateway's ConnectX. |
| per-queue HW ARQ (3 TXQ, one `DEST_ADDR` each) | retry binds to one dest ⇒ ≤3 flows | TT runs **raw mode** — no HW ARQ. Reach is bounded by the 16-bit tag QPN space (65 536) + rkey table, **not** by TXQ count. |
| raw mode "not performant" | true for the TTFabric mesh-compute stack | the external TT-RDMA path already **accepted** raw + software/gateway reliability as its shipped model; header gen is done by the gateway + ConnectX offload. |

### 3.1 What decouples reach from the 3-queue / 10-MAC HW

Because L2 can't express remote identity (one `DEST_MAC`), the **32 B header**
carries it, per-opcode (`tt-rdma-mesh-addressing-spec.md §2`):

- **One-sided (`WRITE`/`WRITE_IMM`/`READ_REQ`) → routed by `rkey`.** A remote MR
  belongs to exactly one endpoint, so `rkey → endpoint` is a table the gateway
  keeps anyway. One-sided traffic to many peers consumes **zero** QPNs.
- **Two-sided (`SEND`/`SEND_IMM`) → routed by `tag` = QPN.** These carry
  `rkey = 0`, so `tag` is free and non-colliding ⇒ 65 536 QPs/port.

⇒ **one wire, one DEST_MAC, tens of thousands of remote endpoints** — fully
independent of the 3 TXQ and 10 header rows. **No FW change, no wire change.**

### 3.2 Building blocks (what you use)

- **Gateway:** BlueField-3 (ConnectX-7 HW QP termination + DOCA), or an FPGA-NIC,
  or — to de-risk first — a plain x86 box with two ConnectX NICs running
  libibverbs + DPDK (`bf3-gateway-design.md §12`, ~2× slower, same code).
- **Translation:** per-opcode RoCE↔TT mapping (`bf3-gateway-design.md §5`);
  **rkey pass-through** (TT rkey is 32 b, fits a RoCE rkey verbatim ⇒ identity
  map, zero per-frame rewrite, zero cross-gateway shared state).
- **Reliability:** **deferred end-to-end ACK** — the gateway withholds TT's
  `0x40` ACK until the remote's real RoCE ACK returns, so a `TxCqe` never
  completes before bytes are safe at the peer (`tt-rdma-mesh-addressing-spec.md §6`).
- **Wire:** TT-RDMA-v1, ethertype `0x1AF6`, 32 B header, ≤4080 B payload; genuine
  RoCEv2 (BTH/RETH/AETH, UDP-4791) on the fabric side.
- **Lossless / PCIe:** PFC on the TT↔gateway link (`tt-rdma-pfc-lossless.md`);
  DCQCN/PFC on the switched fabric handled by ConnectX + switch as normal;
  SR-IOV/switchdev per `tt-rdma-eswitch-bypass.md`.
- **Standard tooling, unchanged:** `ib_{write,read,send}_bw`, MPI, UCX, NVMe-oF.
  (NCCL/SHARP do not work — GPU-only / closed.)

### 3.3 Scale-out (production shape)

Not one wire to one gateway: **N erisc rails × M gateways**
(`tt-rdma-mesh-addressing-spec.md §8`).
- **NoC-global ⇒ MRs are rail-agnostic:** a one-sided op for any MR lands at the
  same memory via any rail/gateway ⇒ one-sided traffic load-balances + fails over
  for free.
- **Rule: pin a QP to one rail, aggregate with many QPs** (the NCCL/UCX multi-rail
  RoCE pattern). Never per-packet stripe a QP (splits seq space, breaks ordering).
- **HA:** lose a gateway/link ⇒ only its QPs reconnect (RDMA-CM); others
  untouched ⇒ graceful degradation, fixes the single-point-of-failure risk.

### 3.4 Honest caveats

- **Atomics** (CAS/FetchAdd): not in v1 ⇒ gateway NAKs or emulates-via-lease.
- **Port-cumulative ACK** until the per-QP FW fix (M7) ⇒ a slow remote
  head-of-lines all QPs on that rail.
- **MR table = 16** until M7 (→64) ⇒ NCCL/UCX MR-pool wall.
- **~25 Gbps per WH rail** (PCIe-bound) ⇒ 100 Gbps+ needs multiple rails/chips.
- **Multi-node-per-gateway rkey namespace** still open
  (`tt-rdma-mesh-addressing-spec.md §11.1`).

---

## 4. Path B — native switch attach (the Exabox silicon track)

To let TT attach to a switch **without a gateway**, the silicon must lift *both*
§1 limits. Options from the Exabox issue:

| Option | Scope | Schedule | Note |
|---|---|---|---|
| **PoR** | keep existing 4 Q | Freeze Aug 31, T/O Nov 30 | stays point-to-point; no switch interop |
| **Min Switch Support A0** (recommended) | add parallel queues **+ HW packet linked lists** | Freeze mid-Oct, T/O late-Jan | medium risk, emulation essential |
| **Keraunos 2** | ideal, 224G SERDES (N3P) | T/O Dec 2026 | ~15 mo; the real long-term answer |

**Scoping the A0 decision correctly (the trap):** "add parallel queues" alone
still binds ARQ to a queue-static `DEST_ADDR` — it moves 3→N, not 3→*any*. The
load-bearing parts are:
1. **Linked-list ARQ indirection** — decouple retry/seq state from a fixed
   per-queue `DEST_ADDR` (a per-flow list keyed by DA), and
2. **DA-indexed / memory-resident header table** — replace the 10-row static
   `ETH_TXPKT_CFG_A` with a lookup, lifting the ~10-MAC wall (this half is
   cheaper and independently useful even if the ARQ work slips).

**Recommended A0 success criterion:** *N reachable destinations through a switch
at TT-Link performance*, with N and the BW target stated explicitly — otherwise
"basic switch support" quietly collapses back to "a few more point-to-point
queues."

**Also correct the framing "raw mode is not an option":** that is true for the
**TTFabric mesh-compute stack** (needs TT-Link BW). It is *not* absolute — the
external TT-RDMA path ships raw + software/gateway reliability today (Path A).

---

## 5. Choosing a path

- **Need multi-vendor RDMA interop now, on shipped WH/BH:** Path A. Cost = one
  gateway hop + software reliability; benefit = works on stock FW, standard
  tooling, no silicon dependency.
- **Need gateway-less native switching (cost/latency of the hop unacceptable, or
  flat-topology scale beyond what N gateways give):** Path B — A0 minimum-switch
  as the bridge, Keraunos 2 as the destination.
- **Most deployments: both.** Path A for near-term Exabox interop; Path B planned
  in parallel (K2 planning must start now given ~15 mo design + PNR).

---

## 6. Companion docs

- `bf3-gateway-design.md` — the gateway build guide (inbound/passive translation).
- `tt-rdma-mesh-addressing-spec.md` — in-band endpoint addressing + remote-MR
  import + the outbound/symmetric half + multi-rail §8 (the core of Path A).
- `tt-rdma-wire-protocol-v1.md` — the 32 B header this all reinterprets.
- `tt-rdma-pfc-lossless.md` / `tt-rdma-eswitch-bypass.md` — lossless + PCIe path.
- `tt-rdma-blackhole-port.md` — BH variant (same wire + gateway; different FW/MAC).
- HW evidence for §1: `bh-erisc` v1.12.0 `src/common/registers/eth_core_a_reg.h`
  (`ETH_TXQ_{0,1,2}_DEST_ADDR`, `ETH_TXPKT_CFG_A_{0..9}`).
