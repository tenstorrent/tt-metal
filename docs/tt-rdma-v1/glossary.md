# TT-RDMA — Glossary

Acronyms and terms used across the `tt-rdma-v1` docs. TT-specific meaning is
given where it differs from the industry-standard one. Canonical values (opcode
numbers, header layout) live in `tt-rdma-wire-protocol-v1.md`; this file only
expands names.

---

## Verbs / RDMA semantics

| Term | Expansion | Meaning in these docs |
|---|---|---|
| **RDMA** | Remote Direct Memory Access | Read/write a peer's memory without involving its CPU. |
| **TT-RDMA** | Tenstorrent RDMA | This project's verbs-shaped, NIC-agnostic RDMA layer over 100 GbE (ethertype `0x1AF6`). |
| **verb** | — | An RDMA operation primitive (SEND, WRITE, READ, ATOMIC) as defined by the IBTA `libibverbs` API. |
| **one-sided op** | — | WRITE / READ / ATOMIC — the target CPU is not involved; the initiator names a memory region directly. |
| **two-sided op** | — | SEND / RECV — the receiver must pre-post a buffer; both sides participate. |
| **SEND** | — | Unsolicited message into the peer's next pre-posted recv slot (TT opcode `0x01`). |
| **WRITE** | RDMA WRITE | Place bytes at `base[remote_offset]` in a peer's registered MR (opcode `0x10`). |
| **READ** | RDMA READ | `READ_REQ` (`0x20`) → target fetches → `READ_RESP` (`0x21`) carries bytes back. |
| **ATOMIC** | — | CAS / Fetch-Add on 8 B at a remote offset (opcodes `0x30`-range; deferred to v1.2). |
| **IMM** | **Immediate** (data) | A small inline value carried with the op and delivered to the receiver's completion. `SEND_IMM` (`0x02`), `WRITE_IMM` (`0x11`). Same as IB `*_WITH_IMM`. |
| **MR** | Memory Region | A registered, access-controlled span of memory. On TT, `mr.base_noc_addr` is a 64-bit NoC address (host hugepage, L1, DRAM, or remote chip). |
| **rkey** / **lkey** | remote / local key | Keys returned by `reg_mr`. `rkey` goes on the wire (frame+12) and authorizes remote access; `lkey` is local-only. TT `rkey = (slot<<24)\|(rand16<<8)\|generation`. |
| **PD** | Protection Domain | Groups QPs + MRs that may interoperate. `pd=0` = default (deferred on TT). |
| **QP** | Queue Pair | A connection endpoint (a send + recv queue). |
| **QPN** | Queue Pair Number | Identifies a QP. On TT-RDMA v1, carried in the 16-bit `tag` for SEND-family (see mesh-addressing-spec). |
| **WQE** | Work Queue Entry | A posted work request (a "descriptor"). TT TX ring is 32 × 4096 B WQE slots at L1:0x9000. |
| **CQ / CQE** | Completion Queue / Entry | Where op completions land; one CQE per completed WQE. |
| **RC / UD / SRQ / XRC** | Reliable Connected / Unreliable Datagram / Shared Receive Queue / eXtended RC | IB QP service types. TT-RDMA v1 is RC-like only. |
| **SGE / sg_list** | Scatter-Gather Element / list | The buffer(s) a WQE reads from; the provider gathers them into one contiguous payload. |
| **ODP** | On-Demand Paging | Lazy MR pinning via page faults. Not in v1 (pin-only). |

## Reliability / sequencing

| Term | Expansion | Meaning |
|---|---|---|
| **seq** | sequence number | TT-RDMA per-op reliability counter (frame+8). |
| **ACK / NAK** | acknowledge / negative-acknowledge | ACK opcode `0x40`; `seq` field carries the cumulative `ack_seq`. |
| **cumulative-ACK** | — | One ACK confirms everything up to `seq`. TT v1's model; **port-level**, not per-QP. |
| **selective-ACK** | SACK | Per-seq ACK + targeted retransmit. Planned v1.1 (replaces re-firing the whole window). |
| **ARQ** | Automatic Repeat reQuest | The retransmit-on-loss reliability scheme. |
| **retx** | retransmit | Re-sending un-acked frames. |
| **PSN** | Packet Sequence Number | RoCE's per-QP sequence number. The gateway maps PSN ↔ TT `seq`. |
| **window** | — | The set of in-flight un-acked ops; "window-full" back-pressures the sender. |

## On-chip / firmware

| Term | Expansion | Meaning |
|---|---|---|
| **WH** / **BH** | Wormhole / Blackhole | Tenstorrent chip generations. |
| **erisc** | Ethernet RISC | The RISC-V core on a WH eth tile that runs `erisc_cmac_simple` FW. |
| **CMAC** | 100G MAC | The Ethernet MAC block driven by the erisc; "external CMAC" = the wire-facing port. |
| **Tensix** | — | WH's compute core grid (runs the inference workload, peer to the erisc FW). |
| **NoC** | Network-on-Chip | On-die interconnect. `mr.base_noc_addr` resolves to L1 / DRAM / PCIe-tile / ETH-tile. |
| **L1** | — | Per-core SRAM (~1.5 MB). Holds the RX ring, TX WQE pool, RCB, MR table. |
| **DRAM** | — | Per-chip off-die memory (~12 GB); bulk MR target. |
| **RCB** | Ring Control Block | The host↔FW mailbox at L1:0x8400 (producer/consumer/acked idx, mode, ctrl doorbell at +0x28). |
| **PACKET_BUF** | — | CMAC RX DMA ring base (L1:0x4000). |
| **TX_BUF0/1** | — | Staging buffers for FW-built TX frames (READ_RESP, ACK), at L1:0x29000/0x2A000. |
| **BUF_WRAP** | — | CMAC RX mode where frames wrap a single ring; frame boundaries come from the header `length`. |
| **BUF_PTR** | — | CMAC's RX write pointer register; the pre-v1 length-inference hack read it (source of the `mixed_size` leak). |
| **DMA-pull / DMA-push** | — | FW reads TX payload from host hugepage (pull); FW writes RX descriptors to host hugepage (push). |
| **admit / admit rate** | — | Fraction of wire-received frames the FW successfully parses and dispatches. |
| **soak** | — | Sustained-traffic stress run. |
| **stride-walk** | — | The RX drain that steps through the ring by `length` per frame. |
| **rail** | — | One (erisc port ↔ gateway ↔ remote) path. Multi-rail = BW aggregation + HA. |
| **aiclk** | AI clock | WH core clock; drops to 500 MHz when idle, capping cold-burst TX. |
| **md5** | — | Used here as the FW-build checkpoint fingerprint (e.g. `43bdcb46` = v1.0). |

## Wire / Ethernet / QoS

| Term | Expansion | Meaning |
|---|---|---|
| **ethertype** | — | L2 protocol id. `0x1AF6` = TT-RDMA v1; `0x1AF4/5` = legacy soak; `0x1AF7` = planned QPN-extended (mesh-spec §7.3). |
| **FCS** | Frame Check Sequence | Ethernet's per-link CRC; covers the whole frame. |
| **CRC-32C** | — | Castagnoli CRC; the (currently unused-on-wire) `header_cksum`, moved to a gateway software check in the `0x1AF7` plan. |
| **MTU / jumbo** | Maximum Transmission Unit | Max L2 payload. TT jumbo = 4080 B (CMAC max-pkt 4096). |
| **PFC** | Priority Flow Control | Per-priority pause (802.1Qbb) for a lossless class. TT uses priority 3. |
| **PCP** | Priority Code Point | The 3-bit priority in an 802.1Q VLAN tag. |
| **VLAN / VID / TCI** | Virtual LAN / VLAN ID / Tag Control Info | 802.1Q tagging; TT-RDMA rides VID 100, PCP 3. |
| **DSCP** | Differentiated Services Code Point | IP-layer QoS marking (RoCEv2 alt to PCP; needs IP wrap). |
| **PCS / FEC** | Physical Coding Sublayer / Forward Error Correction | PHY-layer link bring-up state ("PCS lock"). |
| **CTLE** | Continuous-Time Linear Equalizer | SerDes RX equalization; swept per-cable during bring-up. |

## RoCE / gateway / verbs stack

| Term | Expansion | Meaning |
|---|---|---|
| **RoCE / RoCEv2** | RDMA over Converged Ethernet | Industry-standard RDMA-on-Ethernet. v2 = routable (UDP/4791). The gateway translates RoCEv2 ↔ TT-RDMA-v1. |
| **BTH** | Base Transport Header | RoCE's transport header (opcode, QPN, PSN). |
| **RETH** | RDMA Extended Transport Header | RoCE WRITE/READ header (vaddr, rkey, length). |
| **AETH** | ACK Extended Transport Header | RoCE ACK header (syndrome, credit, PSN). |
| **ICRC** | Invariant CRC | RoCE end-to-end CRC (gateway's problem, not TT's). |
| **DCQCN / ECN** | Datacenter QCN / Explicit Congestion Notification | RoCE congestion control (gateway-side). |
| **GID / PKey** | Global Identifier / Partition Key | RoCE addressing; provider synthesizes a MAC-derived GID. |
| **MAD** | Management Datagram | IB fabric management; not applicable to TT-RDMA. |
| **libibverbs / rdma-core** | — | The userspace RDMA API / its upstream repo. |
| **librdmacm** | RDMA Connection Manager | TCP-based QP-info exchange (`rdma_connect`, `rdma_listen`). |
| **provider** | — | A userspace `.so` implementing verbs for one device class (`libtt_rdma`). |
| **uverbs / ib_core** | — | Kernel RDMA subsystem entry points. |
| **rxe / siw / efa** | — | Existing rdma-core providers used as templates (soft-RoCE, soft-iWARP, AWS EFA). |

## Hardware / partners / hosts

| Term | Expansion | Meaning |
|---|---|---|
| **BF3** | BlueField-3 | NVIDIA SmartNIC; one gateway option (runs DOCA on Arm cores). |
| **DOCA** | — | NVIDIA's BlueField SDK (doca_rdma, doca_flow, doca_dma). |
| **ConnectX (CX-5/7)** | — | Mellanox/NVIDIA HCAs. Today's test partner is a CX-5 (Gen3 x4). |
| **FPGA-TT-Link** | — | The FPGA-NIC partner path (open-nic-shell + QDMA); the not-PCIe-bound option. |
| **QDMA** | Queue DMA | Xilinx/AMD FPGA DMA engine used by the FPGA-TT-Link partner. |
| **eSwitch** | embedded switch | The Mellanox NIC's internal switch; its slow-path is the ~6 Gbps small-frame ceiling. |
| **switchdev** | — | `devlink` mode that HW-offloads the eSwitch fast-path (bypasses the host CPU). |
| **SR-IOV / PF / VF / representor** | Single-Root I/O Virtualization / Physical / Virtual Function | NIC virtualization; switchdev needs ≥1 VF; the representor is the PF's slow-path netdev. |
| **vfio-pci** | — | Kernel driver that hands a whole PCI device to userspace (full DPDK bind). |
| **DPDK / PMD** | Data Plane Development Kit / Poll-Mode Driver | Userspace packet I/O; the partner app uses it on the "tt" side. |
| **hugepage** | — | 2 MB pinned pages; required for NoC-DMA-able host MRs. |
| **PCIe Gen3 x4 / BAR** | — | The WH host bus (~25 Gbps ceiling for host-RDMA); BAR = the MMIO window into the chip. |
| **TLB** | Translation Lookaside Buffer | Here: the UMD PCIe address-window used to reach chip L1 (cached for speed). |

## Workloads / benchmarks / tooling

| Term | Meaning |
|---|---|
| **MPI / UCX / NCCL / NVMe-oF** | RDMA-consuming frameworks the gateway/provider aim to support unchanged. |
| **ib_write_bw / ib_read_bw / ib_send_bw / ib_*_lat** | OFED `perftest` micro-benchmarks (the acceptance oracles). |
| **osu_bw / osu_latency** | OSU MPI micro-benchmarks. |
| **tt-metal / tt-exalens / tt-smi** | TT SDK / on-chip debug readback / device management CLI. |
| **Qwen3-0.6B** | The persistent Tensix inference workload used in the coexistence diagram. |
