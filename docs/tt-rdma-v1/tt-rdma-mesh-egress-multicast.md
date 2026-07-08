# Interior-Mesh Egress & Multicast to External RDMA Endpoints

**Status:** architecture note / design. Composes two already-documented halves
that were never connected: **tt-fabric** (intra-mesh, `tech_reports/TT-Fabric/
TT-Fabric-Architecture.md`) and **TT-RDMA external interop** (`bf3-gateway-design.md`,
`tt-rdma-mesh-addressing-spec.md`). Neither half documents the composition, and
**one-to-many fan-out to external NICs is not covered anywhere** — this doc fills
that.

**Scenario (the question this answers):** a TT chip **buried inside the
tt-fabric mesh** — no external CMAC link of its own, surrounded by other TT chips
— wants to push a **tensor / burst transfer** and **multicast it to several GPUs**
that hang off NICs on an L2/RoCE switch. Is that possible, and how?

**Short answer:** yes, as **three legs**. Legs 1–2 are shipped-or-designed;
**leg 3 (external multicast) is the genuine gap** and lands on the gateway.

---

## 1. The three legs

```
 interior TT chip ─(1) tt-fabric──▶ edge TT chip ─(2) TT-RDMA raw──▶ gateway ─(3) RoCEv2──▶ switch ─▶ GPU NICs
  (no ext link,       route to        (has CMAC,       0x1AF6          replicate/           (fan-out
   tensor in L1/DRAM)  exit node)      RDMA initiator)                  UD-mcast)             to N GPUs)
```

| Leg | What it is | Status |
|---|---|---|
| **1. Interior → edge** | tt-fabric routing across the mesh to a gateway-connected "exit" chip | **shipped** (tt-fabric) |
| **2. Edge → gateway** | TT-RDMA-v1 raw egress (Profile E, `0x1AF6`) | **designed** (M2/M5, not impl.) |
| **3. Gateway → N GPUs** | one-to-many fan-out on the RoCE fabric | **gap** — this doc |

## 2. Leg 1 — interior → edge chip (tt-fabric), and where the tensor lives

This leg is **tt-fabric, not TT-RDMA.** The interior chip's tensor is routed
hop-by-hop to an edge chip that has an external CMAC link. Routing to a chip that
isn't in the local mesh is exactly tt-fabric's **inter-mesh / exit-node routing**
(`TT-Fabric-Architecture.md §2.2.1.2`): the packet targets an *exit node* in the
direction of the destination, via the L0/L1 routing tables. Chip-to-chip delivery
is a **remote NoC write** — the fabric writes the payload to a NoC address on the
destination chip (`budabackend .../erisc/src/eth_routing_v2.cpp:1747 noc_write`,
plus a specialized ordered `CMD_DATA_BLOCK_DRAM` bulk path).

**Which memory does it land in? (the load-bearing detail)**

The fabric writes to **any NoC address**, so the tensor can land directly in
**Tensix L1** — that is the *fast* path, not DRAM:

| Tier | BW | Size | Use |
|---|---|---|---|
| **Tensix L1** | ~TB/s | ~1.5 MB / core | tensor fits (tiled / EDM-streamed) → **lowest latency, preferred** |
| **DRAM** | ~512 GB/s | ~12 GB | tensor too big for L1 (weights, large activations) |
| **host hugepage** (PCIe tile) | ~25 Gbps | host RAM | only if the *host* is the consumer — slowest; avoid for chip-to-chip |

Two corrections to the naive "stage in DRAM/host" intuition:

1. **Land in Tensix L1 when it fits — the fabric supports NoC→L1 directly and
   it's fastest.** The modern data mover (**EDM**, `tt_metal/fabric/.../edm_fabric/`)
   streams **L1→L1** across chips, pipelined with flow control. DRAM is a
   *capacity* fallback, not a fabric requirement.
2. **It is never *erisc* L1.** The eth core's L1 holds the RDMA FW working set
   (RX ring / TX WQE / MR table; on BH shared with the `bh-erisc` base FW). The
   tensor lands in **Tensix L1 or DRAM**. When leg 2 egresses it, the **CMAC TX
   DMA-pulls from wherever it lives over the NoC** (`mr.base_noc_addr` = any NoC
   address) — there is **no copy into erisc L1**. That is the whole point of the
   DMA-pull design (`README.md` "Key insight").

**Caveat that makes the L1-vs-DRAM choice mostly moot end-to-end:** legs 2–3 are
**wire/PCIe-bound** (100 GbE on WH, 400 G on BH, then the gateway's link). So
L1-vs-DRAM at the edge chip changes *intra-mesh latency*, not the *egress
throughput* to the GPUs. Pick by tensor size: fits → EDM L1-streaming (lowest
mesh latency); too big → DRAM. Either way the edge CMAC reads it in place.

## 3. Leg 2 — edge chip → gateway (TT-RDMA egress)

The edge chip is the **RDMA initiator**. Its CMAC runs **Profile E** (raw mode,
`0x1AF6`) to one gateway MAC; the tensor egresses as WRITE/SEND, DMA-pulled from
the Tensix-L1/DRAM address it landed at in leg 1. This is the standard designed
path — per-opcode translation in `bf3-gateway-design.md §5`, addressing/completion
in `tt-rdma-mesh-addressing-spec.md §2,§6`. Nothing new for a **single**
destination.

## 4. Leg 3 — gateway → multiple GPUs (the gap)

**TT-RDMA-v1 has no external multicast.** It is reserved-only on the wire
(`tt-rdma-wire-protocol-v1.md §2`, `0x50–0xEF` future range) and the transport is
RC-like (one QP → one peer) / one-sided-by-rkey. tt-fabric *does* have
multicast/scatter (`TT-Fabric-Architecture.md §4`), but that replicates to **TT
nodes inside the mesh** — the external GPUs are **not** fabric nodes, they are
beyond the gateway. So the one-to-many fan-out **must happen at the gateway**, two
ways:

| Option | How | Pros | Cons |
|---|---|---|---|
| **A. Gateway N-unicast replication** (recommended) | TT sends the tensor to the gateway **once**; gateway issues **N RoCE unicast WRITEs**, one per GPU MR | reliable (per-GPU end-to-end ACK); works on any switch; **no wire/FW change** | N× bandwidth on gateway↔switch; gateway needs an MR per GPU |
| **B. RoCE UD multicast** | gateway joins a RoCE **multicast group**, one UD send → **switch replicates** to members | bandwidth-efficient (1× on gateway link) | **unreliable datagram** (no ACK/delivery guarantee); needs multicast-group setup on switch + GPU NICs; UD not in the gateway design today |

**Completion semantics (new, either way):** the TT initiator posts one op but wants
one completion meaning "delivered." For **A**, the gateway must **aggregate N RoCE
ACKs** before releasing the single TT `0x40` ACK (extends the deferred-ACK rule,
`mesh-addressing-spec §6`) — a `TxCqe` then means "all N GPUs have the bytes." For
**B**, there are no ACKs: completion means "sent," not "delivered."

**Gateway state added:** a fan-out group table — `mcast_group[gid] → [ {gpu_mr,
gpu_gid, roce_qp}, … ]` — plus a replicate-on-forward action in `translate.c`.
This is the multicast analogue of the unicast `mr_fwd`/`qp_fwd` tables
(`mesh-addressing-spec §3`).

### 4.1 Why Option B is a trap for tensors (would BlueField handle it?)

**Literally, yes** — BF3 is a ConnectX-7, which has native hardware UD +
multicast, and its Arm cores run stock OFED (`ibv_*`). So BF3 can be the multicast
*sender*: it joins the group (MGID → IP-mcast → Ethernet mcast MAC), posts UD
sends on a UD QP, and the **switch replicates** to every GPU NIC that joined
(IGMP snooping). Transmit once, fabric fans out — that is the bandwidth win.

**But UD multicast is the *only* RoCE multicast that exists (no RC multicast), and
it strips four things the RC/unicast path gives for free** — all of which land on
BF3 or the GPU app:

1. **Unreliable + unordered.** No ACK, no retransmit; dropped datagrams vanish
   silently. For a tensor (every byte matters) BF3 would have to bolt on an
   **app-level NACK/repair** layer — which erases the efficiency that motivated B.
2. **MTU-fragmented.** A UD message is **one datagram ≤ path MTU** (~4 KB); there
   is no RC-style HW segmentation. A multi-MB tensor becomes **thousands of UD
   sends**, and BF3 must add a fragment/seq header so receivers can reassemble.
3. **SEND, not WRITE — no memory semantics.** UD multicast delivers into each
   receiver's **recv queue**, not to a known GPU address. The "RDMA writes
   straight into GPU HBM at `addr`" property is **gone**; each GPU app must
   pre-post buffers and copy into place.
4. **Fabric config.** IP-multicast group + IGMP snooping (or static mcast
   forwarding) on the switch, and every GPU NIC must join.

This is why collective libraries (NCCL/UCX) **do not** use RoCE multicast for bulk
tensor data — they use RC unicast ring/tree, or HW collectives (SHARP). **Reserve
Option B for small, loss-tolerant, one-to-very-many broadcast** (a control message,
a re-sendable parameter push) — never for a tensor you cannot afford to lose.

### 4.2 Scale-out — the recommended *reliable* multicast (two reliable stages)

Instead of one unreliable UD stage on the switch, do **two reliable stages**: use
**tt-fabric's own HW multicast** (reliable, in-mesh; `TT-Fabric-Architecture.md
§4`) to deliver the tensor to **K edge chips**, each on its own rail/gateway; each
gateway then does **Option A (reliable RC unicast WRITE)** to its **subset** of the
GPUs.

```
                 ┌─ edge chip 1 ─▶ gateway 1 ─▶ RC WRITE ─▶ GPUs 1..m
interior chip ──fabric multicast──┼─ edge chip 2 ─▶ gateway 2 ─▶ RC WRITE ─▶ GPUs m+1..2m
 (one send)      (reliable HW)    └─ edge chip 3 ─▶ gateway 3 ─▶ RC WRITE ─▶ GPUs 2m+1..N
```

Why this beats single-gateway A *and* UD B:

- **Reliability preserved end-to-end** — fabric multicast is reliable HW; every
  egress is RC (ACKed, HW-segmented, one-sided WRITE straight into GPU memory).
  No fragmentation layer, no app NACK, no lost memory semantics.
- **Bandwidth spread** — each gateway unicasts to only N/K GPUs, cutting Option
  A's N× cost to **(N/K)× per gateway** (the rail-matrix payoff,
  `mesh-addressing-spec §8`).
- **BF3 does nothing special** — each gateway runs the ordinary unicast egress it
  already does for a single GPU, just times its subset. No UD, no IGMP, no
  reassembly.

So the fan-out happens as **fabric HW multicast (reliable) → RC unicast at the
edges (reliable)**, not one unreliable UD hop. **This is the recommended path for
multicasting a tensor you care about**; single-gateway Option A is the K=1 case;
Option B is only for loss-tolerant broadcast.

## 5. Shipped / designed / missing

| Piece | State |
|---|---|
| Interior→edge routing (inter-mesh exit node) | **shipped** — tt-fabric |
| NoC→Tensix-L1 / DRAM landing, EDM L1-streaming | **shipped** — tt-fabric / EDM |
| Edge-chip DMA-pull egress from any NoC addr | **shipped** primitive (WH v1.0); BH per `tt-rdma-blackhole-port.md` |
| Edge→gateway→**single** external GPU (unicast RoCE) | **designed** — `bf3-gateway-design.md` (M2/M5) |
| **External multicast / N-way fan-out** | **missing** — Option A/B above; no doc until this one |
| Aggregated N-ACK completion | **missing** — extends `mesh-addressing-spec §6` |

## 6. Open questions

1. **A vs B by default?** Recommend **A (N-unicast)** for the MVP — reliable, no
   new wire/FW, stock switch. B (UD multicast) only if gateway↔switch bandwidth
   is the bottleneck *and* unreliable delivery is acceptable for the tensor.
2. **Partial-failure semantics** for Option A: if 1 of N GPUs NAKs/times out, does
   the TT `TxCqe` fail, or complete with a per-GPU status vector? (RDMA has no
   native "partial multicast" completion — this is a new contract.)
3. **Group membership control plane:** who populates `mcast_group[]` — the TT SDK
   (`ep.create_mcast_group([RemoteMr…])`) or gateway-side config? Mirror the
   `import_remote_mr` path (`mesh-addressing-spec §5.2`).
4. **Ordering across the fan-out:** concurrent multicasts to the same GPU set from
   different interior sources have no mutual ordering (as in verbs) — app concern.
5. **Interior-source completion latency:** with tt-fabric transit + aggregated
   N-ACK, the interior chip's completion is `fabric-RTT + max(per-GPU RoCE RTT)`.
   Acceptable for bulk tensors; not for fine-grained sync.

## 7. Companion docs

- `tech_reports/TT-Fabric/TT-Fabric-Architecture.md` — leg 1 (routing §2.2,
  inter-mesh exit nodes §2.2.1.2, multicast/scatter APIs §4). Authoritative fabric ref.
- `bf3-gateway-design.md` / `tt-rdma-mesh-addressing-spec.md` — legs 2–3 unicast
  translation, binding tables, deferred ACK.
- `tt-rdma-switch-interop-architecture.md` — why the external side needs the
  gateway at all (Path A).
- `tt-rdma-wire-protocol-v1.md` — multicast is reserved-only in the opcode map.
- `runtime-workload-coexistence.md` — the `mr.base_noc_addr` = L1/DRAM/host/remote
  memory-tier picture this doc's §2 builds on.
- `tt-rdma-blackhole-port.md` — on BH, leg-2 egress + the 400 G / rail-matrix scale.
```
