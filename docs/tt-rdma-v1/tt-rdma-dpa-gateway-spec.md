# TT-RDMA v1 — BlueField-3 DPA Gateway Datapath (RoCEv2 ↔ TT-RDMA-v1)

Status: design draft (2026-07-23). The BF3 half of the §14 performance mandate
(`tt-rdma-bh-bf3-impl-plan.md`): **the DOCA DPA is the datapath engine; the BF3 Arm CPU and the x86
host CPU are OFF the per-packet path.** Pairs with the chip-side `tt-rdma-tx-ring-spec.md`.

Scope note: this specifies the **architecture + DOCA component decomposition + data flow**. The
DPA-side kernel *source* (the code `dpacc` compiles) is out of scope here — write it against the
public DOCA DPA / DPA-Verbs / DPA-Comms guides and the shipped `/opt/mellanox/doca/samples/doca_dpa/`.
This doc says *what the DPA kernels must do and how they bind to the host control plane*, not their C.

Golden references: DOCA DPA (host `doca_dpa_*`), DPA-Verbs (`libdoca_dpa_dev_verbs.a` /
`doca_dpa_dev_verbs.h`, in-kernel RDMA), DPA-Comms (`libdoca_dpa_dev_comm.a`), `doca_rdma` (WAN RoCE
QP), `doca_eth` (raw L2 to TT), `doca_flow` (steering), `tt-rdma-wire-protocol-v1.md` (frame).

---

## 1. Principle: DPA runs the datapath, Arm/host run control only

A gateway that translates on the Arm CPU (plan tier T1, ~25 Gbps, PCIe/Arm-bound) is a bring-up
crutch. The line-rate design puts the **per-packet translation on the DPA processor** — the NIC's
programmable datapath cores sitting on the wire — so neither the BF3 Arm nor the host x86 touches a
packet in steady state. This is the plan's **T3 tier promoted to the default target.**

| Plane | Runs on | Touches each packet? |
|---|---|---|
| **Control** (QP/MR/flow setup, DPA image load, thread create, MR-register CONTROL ops, exceptions, ARQ policy) | BF3 Arm (host `doca_*`) + x86 host `bh_mr_agent` | No |
| **Datapath** (BTH/TT-RDMA parse, table lookup, header rewrite, forward, seq/PSN stamp) | **DPA processor** (`doca_dpa_app` kernels) | **Yes — this is the gateway** |
| **Wire I/O** | NIC HW: RoCE QP (WAN), raw-L2 eth queue (TT), DOCA Flow steering | Yes (HW) |

## 2. DOCA component decomposition

**Control plane (BF3 Arm, host-side C — set up once, then hands off):**
- `doca_dpa` context per DPA instance; load the `doca_dpa_app` image (`dpacc`-compiled) with the
  gateway kernels; create N `doca_dpa_thread` execution contexts (one per flow-group / core).
- **WAN side:** `doca_rdma` — create/connect the RoCEv2 QP(s) that terminate the remote initiator's
  RC connection. **Host configures the QP; the DPA *uses* it** (the DPA-Verbs coupling rule): export
  the QP to a DPA-Verbs handle the kernel drives.
- **TT side:** `doca_eth` raw-L2 TX/RX queue bound so the DPA can post/consume frames (see §7 open
  item on the exact DPA↔eth binding).
- **Steering:** `doca_flow` — RSS/5-tuple steer incoming WAN flows to the right DPA thread/QP, and
  PCP-tag for PFC. Flow does *steering only*, never translation (it can't match BTH deep nor stamp
  per-packet seq).
- **Tables:** QP-table, MR-table (rkey ↔ {base, len, access}), PSN↔seq state — in DPA-reachable
  memory, written by the Arm control plane, read/updated by the DPA kernels.
- **`bh_mr_agent` (x86 host):** registers on-chip / host MR targets and publishes rkeys (the BF3 has
  no PCIe path to the TT chip; the host bridges MR registration). Control-plane only.

**Datapath (DPA processor — the `doca_dpa_app` kernels):**
- Ingress/egress translation kernels, **event-driven** (see §5), using **DPA-Verbs** for WAN RDMA
  ops and the eth queue for TT frames, **DPA-Comms** for inter-thread coordination (e.g. shared
  seq/PSN counters, work hand-off).

## 3. Datapath — inbound (WAN RoCEv2 → TT-RDMA)

Remote initiator does an RDMA WRITE/SEND to the gateway QP:

1. NIC HW receives the RoCEv2 packet into the DPA-owned QP; a **completion** wakes the DPA ingress
   kernel (no Arm involvement).
2. DPA kernel parses **BTH/RETH** (opcode, PSN, rkey, virtual addr, length) — data already in NIC
   memory (zero-copy).
3. Table lookup: RoCE `{qp, rkey, vaddr}` → TT-RDMA `{opcode, rkey', remote_offset, seq}` via the
   QP/MR maps. Build the 32 B `tt_rdma_hdr_t` (CRC-32C stamped).
4. **Forward to TT:** post a raw-L2 send on the eth TXQ referencing the payload in place — the DPA
   writes the header + points the send at the payload; the NIC serializes it to the TT chip at
   ethertype `0x1AF6`. The chip-side TX ring (`tt-rdma-tx-ring-spec.md`) is the mirror on the TT end.
5. Stamp/advance the per-flow `seq`; PFC keeps it lossless.

Arm/host packet touches: **zero.**

## 4. Datapath — outbound (TT-RDMA → WAN RoCEv2)

Chip sends a TT-RDMA frame to the gateway (raw L2, `0x1AF6`):

1. NIC eth RXQ delivers the frame; a completion wakes the DPA egress kernel.
2. DPA parses the 32 B TT-RDMA header (opcode/rkey/remote_offset/seq/imm), validates CRC + MR.
3. Map to the RoCE QP + remote MR; **issue the RDMA op directly from the DPA kernel via DPA-Verbs**
   (WRITE/WRITE_IMM/READ/SEND) — *no host round-trip* (the canonical DPA-Verbs use case: the host
   round-trip is the latency bottleneck, so the verb is issued in-kernel).
4. Map TT-RDMA `seq` → RoCE PSN; handle READ_REQ by issuing a WAN READ and later returning READ_RESP.

## 5. Event-driven structure (respect `max_kernel_time_alive`)

DPA kernels **must not** be infinite loops — the DPA has a **max-kernel-time-alive** cap
(`doca_dpa_cap_get_max_kernel_time_alive_supported`), and DPA-Comms returns `_AGAIN` meaning *the
kernel must yield*. So the datapath is a **completion handler**, not a spin loop:

- A kernel is (re)launched / woken by a `doca_dpa_completion` on the RoCE CQ or eth RXQ.
- It drains a **batch** of ready completions, does the translations, posts forwards, then **yields**.
- Re-arm on the next completion. This is the same "arms, doesn't drain/spin" discipline as the chip
  side — and the same reason we avoid busy-poll for power/latency.

Throughput comes from **many DPA threads** (one per flow-group) each handling its CQ batch in
parallel, fed by DOCA Flow RSS — scale out, not a hot loop.

## 6. Reliability + ordering (raw TT link has no HW resend)

- **PFC lossless** on the TT↔BF3 direct link → no steady-state drops (primary).
- The DPA owns the **seq ↔ PSN** mapping: it tracks per-flow TT-RDMA `seq` and RoCE PSN, detects
  gaps, and drives **retransmit** (re-issue from the retained WQE / request re-send) — entirely on
  the DPA, no Arm per-packet. RoCE's own reliability covers the WAN side; the DPA bridges the two
  reliability domains.
- Golden-vector CRC-32C (`tt_rdma_wire.h`) is validated on the DPA for every inbound TT frame; the
  DPA and the chip build identical bytes (the M-3 loopback contract).

## 7. Zero-copy + the DPA↔queue bindings (open items)

- **Zero-copy:** payload stays in NIC/DPA memory across the translation — the DPA rewrites only the
  32 B header and re-points the descriptor; it never copies the payload. Header-only touch = the T2→
  T3 progression in plan §6.
- **OPEN — DPA↔eth binding:** confirm, against the installed DOCA version, how the DPA posts/consumes
  **raw-L2** frames on the TT side — whether via a `doca_eth` queue exposed to the DPA, or a raw QP
  driven through DPA-Verbs. Gate the gateway MVP on this (route via the public DOCA guides;
  `doca_dpa_cap_is_supported` + the shipped `doca_dpa` samples are the truth on this install).
- **OPEN — capability + version gates:** `doca_dpa_cap_is_supported` on the active `doca_devinfo`
  (BF3 has a DPA; confirm the specific features), and **DOCA must match the DPACC compiler** version
  (DOCA compatibility policy). Record both in the bring-up log.

## 8. Scaling to line rate

- **N DPA threads** across the DPA cores, one per flow-group; DOCA Flow RSS fans WAN flows to them.
- **Multiple QPs / multiple TT rails** (the chip exposes 1–2 external rails today; the design
  generalizes to more). Aggregate = per-thread rate × threads, bounded by DPA core count + wire.
- **Batching** per completion event amortizes per-packet fixed cost; **jumbo** on the TT side.
- Translation is header-only + table lookup (O(1)) — no payload work — so the DPA core budget, not
  the payload size, sets the ceiling.

## 9. Interfaces / contracts

- **Wire (both directions):** `tt-rdma-wire-protocol-v1.md` — ethertype `0x1AF6`, 32 B header,
  CRC-32C. The DPA `tt_v1_codec` and the chip `tt_rdma_hdr_build.h` MUST produce identical bytes
  (shared golden vectors).
- **Chip side:** `tt-rdma-tx-ring-spec.md` (RISC-off-datapath TX ring) is the mirror; the gateway's
  inbound forward lands in the chip RX ring, the chip TX ring feeds the gateway's egress kernel.
- **MR model:** rkey namespace shared host↔chip↔gateway (`tt_rdma_mr_entry_t`); `bh_mr_agent`
  publishes on-chip MR rkeys the WAN initiator uses.

## 10. Milestones (gateway G.x, under the DPA-default mandate)

- **G.1** — control plane on Arm: `doca_dpa` context + load a trivial `doca_dpa_app`; stand up one
  RoCE QP (`doca_rdma`) + one raw-L2 eth queue (`doca_eth`); confirm caps + DOCA/DPACC version match.
- **G.2** — resolve the §7 DPA↔eth binding; a DPA kernel that, on a WAN CQE, emits one raw `0x1AF6`
  frame to the TT chip (inbound one-packet path, DPA-only). Confirm on the chip RX ring.
- **G.3** — outbound: DPA kernel consumes a TT `0x1AF6` frame and issues an RDMA op via DPA-Verbs
  (no host round-trip). M-3 loopback byte-match both directions.
- **G.4** — tables + seq/PSN reliability + PFC lossless; sustained bidirectional translation.
- **G.5** — scale-out (N DPA threads + DOCA Flow RSS + multi-rail); measure line-rate approach.
- **G.6** — verbs-provider / MPI/UCX interop on the WAN side (stock apps reach a BH node).

## 11. What stays OFF the datapath (the invariant to preserve at every milestone)
- BF3 Arm: QP/MR/flow setup, DPA image load, MR-register CONTROL ops, exception/slow-path, ARQ
  policy. Never per-packet.
- x86 host: `bh_mr_agent` MR registration + rkey publish. Never per-packet.
- If any milestone needs the Arm or host in the per-packet loop, that's a design regression — climb
  back (the same discipline as the chip-side RISC-off-datapath rule).
