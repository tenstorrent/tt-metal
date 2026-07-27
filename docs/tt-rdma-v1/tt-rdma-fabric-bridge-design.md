# ext-TT-RDMA ↔ TT-fabric Bridge — Design (the inbound cross-chip landing)

**Status:** design (2026-07-26). Closes the gap that `tt-rdma-bidirectional-mesh-gap.md`, `tt-rdma-mesh-egress-multicast.md`, and `tt-rdma-bh-bf3-impl-plan.md §11.1` all identify but none builds: an inbound external RDMA op (from a BlueField/RoCE gateway over QSFP `0x1AF6`) landing on an MR that lives on a **different** TT chip than the edge chip that owns the QSFP link. Grounded in two investigations: the RX line-rate work (`tt-rdma-rx-linerate-research.md` — the Tensix drainer pool) and a read of the TT-fabric EDM (`tt_metal/fabric/`).

## 1. The problem

A production TT-NIC endpoint is **BF3 + N TT cards meshed over TT-fabric**, exposing the node's aggregate MR space. An inbound WRITE/READ enters at the **edge** chip (the one cabled to the gateway) but the target MR may be on an **interior** chip. Today the receiving eth path only issues a **chip-local** `noc_async_write` (`main_cmac.cc` MR handler; `mesh-addressing-spec §8.1` — "reachable from every erisc core **on the die**"). There is no chip-id on the wire and no component assigned the cross-chip hop. So inbound-to-interior-chip has no path.

## 2. The insight — the two halves already meet at the NoC

- **TT-fabric already does cross-chip landing.** The EDM (Ethernet Data Mover, `tt_metal/fabric/`) is a push transport: a worker emits a packet header carrying `(dst_mesh_id, dst_chip_id)` + a 64-bit destination NoC address `(x,y,local)`, TT-fabric routes it hop-by-hop, and the **terminal EDM eth-RISC issues the final `noc_async_write`** into the target chip's L1/DRAM. It has the completion primitives we need — atomic-inc, fused write+atomic-inc, and UDM **acked** writes/reads with `fabric_write_barrier()` (`hw/inc/udm/tt_fabric_udm.hpp`).
- **Our RX drainer workers are Tensix cores** (`tt-rdma-rx-linerate-research.md` §7), and **TT-fabric's worker-inject API is designed to be called from a Tensix worker** (`WorkerToFabricEdmSender`; `udm::fabric_fast_write_any_len(dst_dev_id, dst_mesh_id, src_addr, dest_addr, len, …, trid, posted)`, which auto-splits into ≤4352 B fabric packets).
- **TT-fabric does not ingest from an external Ethernet source** — it "picks up from the edge-chip L1 inward." That boundary is exactly what the ext-RDMA edge path bridges.

So the bridge is not new transport — it is **one branch in the RX drainer**: for a remote MR, issue a fabric write instead of a local NoC write.

## 3. The design

### 3.1 MR entries carry a fabric-global address (no wire change)
Extend the MR-table entry's target from a chip-local `base_noc_addr` to a **fabric-global address**: `{ mesh_id, chip_id, noc_addr }`. The wire header is unchanged — the edge chip still resolves `rkey → MR` locally (`slot = rkey>>24`), exactly as today; only the *stored* target becomes fabric-global. The control plane (MR register, Phase 1.6) populates it — an MR on chip B, registered through the gateway, stores `{B.mesh, B.chip, B.noc_addr}` in the edge chip's table. `mesh_id/chip_id == self` marks a local MR (the common, fast case).

### 3.2 The drainer worker: one branch (extends the Phase-3.1 pool)
Per frame, after `rkey`→MR resolve + access/bounds validate (the logic already prototyped in `bh_rdma_rx_worker.cpp`):
```
if (mr.is_local)   noc_async_write(payload -> mr.noc_addr + roff);              // edge-chip MR (200G, proven)
else               udm::fabric_fast_write_any_len(mr.mesh, mr.chip,             // interior-chip MR
                       worker_scratch, mr.noc_addr + roff, len, trid, posted);  // TT-fabric routes + lands
```
The worker read the frame into local L1 (the compute-local landing of exp 3); for a remote MR that local copy becomes the fabric-write source. The worker is a fabric client (`WorkerToFabricEdmSender.open()` once at start).

### 3.3 Completion
- **WRITE (no imm):** silent; for reliability the gateway defers the RoCE ACK until safe (`mesh-addressing-spec §6`). For a remote MR, "safe" now means the **fabric write completed at the interior chip** — use `udm` acked mode + `fabric_write_barrier()` (or fused write+atomic-inc to a completion counter) before the edge signals done.
- **WRITE_IMM / SEND:** the worker posts a CQE to the host RxWqeRing (extends Phase 1.2a). For a remote MR the CQE is posted after the fabric barrier confirms landing.

## 4. Honest caveats / open items

- **Cross-chip is NOT line-rate per hop.** TT-fabric is **RISC store-and-forward** (each intermediate EDM eth-RISC copies the packet L1→eth→L1 and re-injects) — no HW cut-through. So a remote-MR write is slower than a local one and consumes eth-RISC bandwidth on every intermediate chip. **Single-link 200 G *to an interior chip's MR* is bounded by the fabric path, not the edge link.** Local-MR (edge chip) stays at the measured 200 G; remote-MR throughput is a fabric-BW question (per-link fabric bandwidth is not quantified in the fabric code — **measure it**). Mitigation: place hot MRs on the edge chip; treat interior-chip MRs as the lower-BW tier; consider multi-plane fabric routes.
- **Reliability spanning the fabric is undefined.** The deferred end-to-end ACK (`§6`) stops at the edge node. A cross-chip op needs the ACK to reflect the *interior* landing — hence the §3.3 fabric-barrier gating. Retransmit semantics on a fabric-hop failure are unspecified.
- **MR registration across chips.** The control plane must register an MR that physically lives on chip B into the edge chip A's `rkey` table with B's fabric address — a cluster-level MR service, not per-chip. (The gateway already holds the `rkey→endpoint` map; extend "endpoint" to `(chip, addr)`.)
- **Coexistence.** The custom external-rail RX kernel + stock TT-fabric EDM run on *different* eth cores of the same chip (external CMAC rails vs TT↔TT rails); the reserved-core split must be proven not to collide (`impl-plan §11.2`, `production-plan §5.3`) — an unverified gate, not an assumption.

## 5. Staged plan (Phase 4 track)

Depends on: Phase 1.6 (MR lifecycle), Phase 3.1 (Tensix drainer pool). Gated by: exps 1–3 (green).

- **4F.1** MR entry = fabric-global `{mesh, chip, noc_addr}`; `is_local` fast path. Cluster MR-registration service maps an interior-chip MR into the edge `rkey` table.
- **4F.2** Drainer worker fabric-write branch (`udm::fabric_fast_write_any_len`); worker opens a `WorkerToFabricEdmSender`. Validate byte-exact landing on a **2-chip** TT-fabric mesh (edge + one interior) via the DOCA sender → interior chip's L1.
- **4F.3** Completion: fabric-barrier-gated CQE + deferred ACK reflecting interior landing.
- **4F.4** **Measure cross-chip inbound BW** (DOCA sender → interior MR, 1..k hops) — establishes the remote-MR tier ceiling vs the local 200 G.
- **4F.5** Reliability: fabric-hop failure handling + retransmit; MR-across-chips consistency; coexistence gate (custom RX kernel + stock EDM on one chip).

**Acceptance:** an inbound RoCE WRITE from the gateway lands byte-exact in an MR on an interior chip reached over TT-fabric, with a completion that reflects the interior landing — and a measured cross-chip BW number.

## 6. One-line summary

TT-fabric already provides cross-chip routing + arbitrary-address landing + completions; our Tensix drainer workers are fabric clients. The bridge is **MR entries that carry a fabric-global address + a one-branch fabric-write in the drainer** — reusing both halves, closing the "tight bridge" gap. Local-MR stays 200 G; remote-MR is fabric-BW-bound (to be measured).
