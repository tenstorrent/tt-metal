# TT-RDMA Bidirectional Mesh Gap Analysis

**Status:** analysis. Scope-setting for the "TT node as a symmetric RoCEv2-reachable mesh endpoint" work.
**Assumption (locked):** there is **always a gateway/translator** (BF3 SmartNIC or FPGA-NIC) between the RoCEv2 fabric and the TT chip. WH does **not** speak wire-native RoCEv2. Dropping wire-native RoCEv2 removes the three hardest gaps — IP/UDP+BTH encapsulation, ICRC, and DCQCN/ECN congestion control — which all become the gateway's problem, not TT's.

**Goal:** a TT cluster mesh node must be accessible by RoCEv2 verbs **in both directions** — external apps reach TT (inbound), *and* TT reaches external apps (outbound).

---

## 1. The reframe: symmetry, not wire format

Given "gateway always exists," the remaining gap is not protocol translation — it's **direction symmetry and endpoint addressing.** TT-RDMA-v1 is a point-to-point, target-centric protocol. A both-directions mesh needs it to behave as an endpoint-addressed, symmetric one.

| Direction | Requester | Gateway translates | Current design coverage |
|---|---|---|---|
| **Inbound** — TT is target/responder | external RoCE app | RoCE request → TT-RDMA request; WH FW is passive target | **~90%** — `bf3-gateway-design.md §5` is written entirely this way |
| **Outbound** — TT is initiator/requester | TT node (host SDK or Tensix) | TT-RDMA request → RoCE request toward a chosen remote peer | **~1 bullet** — this is the real hole |

Everything below is about closing the **outbound** direction and the addressing model both directions share in a mesh.

## 2. What already works (so the list is honest)

- WH **can** initiate: Phase I shipped host `post_send_read` (`external_iface_sender.hpp:182-188`), and the TX WQE ring already emits WRITE/SEND.
- WH-as-target is genuinely done: WRITE / READ_REQ / SEND / WRITE_IMM handlers with MR validation, 100% wire-verified (`README.md` shipped table).
- Gateway **inbound** translation is fully specced (`bf3-gateway-design.md`).

So the gap is not "can TT emit RDMA ops" — it's "can TT emit them **to a named remote endpoint in a mesh, and get correct completions back.**"

## 3. Ranked gaps

### 1. Remote-endpoint identity on the wire — the root blocker 🔴
The v1 header names a memory region (`rkey`+`remote_offset`) and a correlation (`tag`) but **no destination node**. The CMAC TX dest MAC is a single hardcoded register (`main_cmac.cc:204-205`) — one physical wire partner (the gateway) — so an outbound request has no way to say *which* remote peer it's for. **Fully specced in `tt-rdma-mesh-addressing-spec.md`:** route one-sided ops by `rkey` (a remote MR belongs to one endpoint) and two-sided ops by `tag`-as-QPN. **No FW change, no wire change** for the MVP.

### 2. TT-side "remote MR handle" + QP object — SDK model gap 🔴
The SDK conflates local and remote MRs: `register_mr` returns a *local* `MrHandle`, but `post_send_write(buf, MrHandle remote, offset)` wants a *remote* one — the host-sdk example (`tt-rdma-host-sdk.md:302-308`) is a loopback. There is no way to import a peer's `{rkey, base_vaddr}`, and no QP/connection object at all. **Fully specced in `tt-rdma-mesh-addressing-spec.md §5`** (`RemoteMr`, `QpHandle`, `import_remote_mr`, QP-scoped ops).

### 3. TT as active connector — one-sided control plane 🔴
BF3 §5.8 assumes WH is always the **passive** side. Outbound needs the gateway to proxy an **active** RC connect on TT's behalf (WH has no IP stack), and to hold a gateway-owned RC QP per (TT node, remote peer). Specced in `tt-rdma-mesh-addressing-spec.md §4.2`; reuses the `wh_mr_agent` mgmt path.

### 4. End-to-end (not hop-by-hop) completion 🔴
A TT-initiated WRITE's completion must fire only after the *remote* ACKs, not when the gateway receives the frame — otherwise the app sees success before bytes are safe. The gateway must **defer** the TT `0x40` ACK until the real RoCE ACK returns. Specced in `tt-rdma-mesh-addressing-spec.md §6`.

### 5. Per-QP sequencing — bites the initiator hardest 🟡 → **FW-accepted track**
v1's ACK is **port-level cumulative**, not per-QP (`main_cmac.cc:402`; admitted to violate RC ordering, `tt-rdma-verbs-provider.md` open-Q #4). One TT node fanning out to N peers over one port: a slow remote head-of-lines every outbound op. **FW changes are on the table (and this one is latency/BW-neutral):** the ACK path is already off the bulk data loop, so a per-QP `acked_idx` array costs 0 ns on the data path. It pairs with a dedicated QPN field (32 B Option-A layout — reclaim `header_cksum`, keep payload 16-B aligned). Specced as the FW-accepted track M7–M9 in `tt-rdma-mesh-addressing-spec.md §7`. Ship the MVP (M0–M6) without it, then land it.

### 6. Scale ceilings both directions inherit 🟡
- **16 MR slots** (`main_cmac.cc:114`, `tt-rdma-host-sdk.md:143`, "no automatic eviction") — a mesh node with many peers × (local + imported remote) MRs blows this. Bump to 64 is **free**: the lookup is already O(1) direct-index (`slot = rkey>>24 & 0x0F`, `main_cmac.cc:596`), so `& 0x3F` + 2 KB more L1 adds no latency (spec §7.2, M7).
- **Single shared 32-slot WQE ring**, host-mutex arbitrated → HoL blocking across QPs. Per-QP rings are v1.2 / multi-erisc.
- **One CMAC/erisc port per chip** — whole node funnels through one wire until multi-erisc. Production wants an **N erisc × M gateway rail matrix** for BW aggregation + HA; the addressing model composes over it (MRs are rail-agnostic because the NoC is chip-global; pin a QP to one rail, aggregate with many). Full design: `tt-rdma-mesh-addressing-spec.md §8` (M10).
- **ATOMIC absent** — gateway must NAK CAS/FAA (many MPI/UCX/NCCL paths use them).
- **4096 MTU must be guaranteed** (`README.md` open-Q #1: hangs at 9216, 4096-exact unproven) to match RoCE path-MTU cleanly.

## 4. Critical path

Items **1 → 2 → 3 → 4** (spec phases M0–M6) are the minimum for symmetric mesh access, and **none require a WH FW change** — host-SDK + gateway work. Then the **FW-accepted track (M7–M9)** lands items 5 and 6: per-QP ACK/seq, MR table 16→64, and a dedicated QPN field (32 B Option-A). FW changes are welcome as long as they stay latency/BW-neutral; §7 of the spec shows all three clear that bar (they touch ACK receipt and L1 sizing, not the per-frame data loop), with a **non-regression-vs-M6** acceptance gate on each.

The BF3 doc as written yields a working **inbound-only** demo (`ib_write_bw` from a compute node into a TT node) with none of the above. It's the **outbound** direction — TT node initiating RDMA into the RoCE fabric — that items 1–4 unlock, and that's what makes it a *symmetric mesh node* rather than a passive target.

## 5. Companion docs

- `tt-rdma-mesh-addressing-spec.md` — detailed spec for items 1 & 2 (the critical path); includes the gateway binding tables, symmetric connection setup, SDK surface, and phase plan.
- `bf3-gateway-design.md` — the gateway (inbound-complete; this analysis adds the outbound half).
- `tt-rdma-verbs-provider.md` — the native-verbs alternative to a gateway.
- `README.md` — v1.0 shipped status + planned backlog (items 5 & 6 live there).
