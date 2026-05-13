# BlueField-3 → TT-RDMA-v1 Gateway: Build Guide

**Status:** design / build plan. Not yet implemented.

A SmartNIC gateway that translates between standard RoCEv2 and TT-RDMA-v1.
Lets industry-standard RDMA tooling (`ib_*_bw`, MPI, UCX, NVMe-oF, custom
`libibverbs` apps) reach Tenstorrent nodes without any app-side changes.

The TT-RDMA-v1 firmware on Wormhole stays untouched (still the v1.0
shipped FW, md5 `43bdcb46`). All protocol translation lives in the gateway.

---

## 1. End-to-end architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│ COMPUTE NODE                                                              │
│ ┌──────────────────────────┐     ┌─────────────────────────────────────┐ │
│ │ Application (PyTorch,    │     │ OFED libibverbs / librdmacm         │ │
│ │ MPI, UCX, ib_*_bw,       │────▶│ Standard, unchanged                 │ │
│ │ custom)                  │     └────────────────┬────────────────────┘ │
│ └──────────────────────────┘                      │ verbs / RDMA-CM      │
│                                  ┌─────────────────▼────────────────────┐ │
│                                  │ ConnectX HCA on compute side         │ │
│                                  └─────────────────┬────────────────────┘ │
└────────────────────────────────────────────────────┼────────────────────┘
                                                     │ RoCEv2 wire
                                                     │ ethertype 0x8915 (v1)
                                                     │ or UDP/4791 (v2)
              ┌──────────────────────────────────────▼────────────────────┐
              │ BLUEFIELD-3 GATEWAY                                       │
              │                                                            │
              │  ConnectX-7 NIC port "wan"  (RoCE-side)                   │
              │   - HW QP termination, PSN tracking, ACK gen               │
              │   - Exposes ibv_ devices to BF3 Arm Linux                  │
              │            │                                                │
              │            ▼ RX                                             │
              │   DOCA Flow rules                                          │
              │    - Match BTH opcode → redirect bulk path to HW pipeline  │
              │    - Tail path → Arm cores                                 │
              │            │                                                │
              │            ▼                                                │
              │   Arm A78AE cores (8×)                                      │
              │    - QP→TT mapping table                                   │
              │    - rkey translation table                                │
              │    - PSN → TT seq mapping                                  │
              │    - ATOMIC handling (NAK or emulate)                      │
              │            │                                                │
              │            ▼                                                │
              │   TT-RDMA-v1 ENCODER (lib)                                 │
              │    - Build 32 B v1 hdr                                     │
              │    - Pack as ethertype 0x1AF4 frame                        │
              │            │                                                │
              │            ▼                                                │
              │   ConnectX-7 NIC port "tt"  (TT-RDMA-v1 side)              │
              │    - Raw L2 ethernet                                       │
              └────────────┼─────────────────────────────────────────────────┘
                           │
                           ▼ TT-RDMA-v1 wire
                           ▼ ethertype 0x1AF4
              ┌──────────────────────────────────────────────────────────┐
              │ WORMHOLE CHIP                                            │
              │  erisc_cmac_simple.elf (TT-RDMA-v1 FW, v1.0)             │
              │   - 32 B header dispatch by opcode                       │
              │   - MR table at L1:0x8500 (16 entries × 32 B)            │
              │   - 4-deep NoC PCIe read TX pipeline                     │
              │   - NoC-direct WRITE/READ to MR base                     │
              │            │                                              │
              │            ▼                                              │
              │  NoC fabric                                              │
              │   → Local L1 (per-core 1.5 MB)                           │
              │   → DRAM (per-chip 12 GB)                                │
              │   → Other chips via tt-fabric                            │
              └──────────────────────────────────────────────────────────┘
```

---

## 2. Hardware bill of materials

**BlueField-3:**
- SKU: B3220 (200 GbE, 16 cores, 16 GB) at minimum. B3240 (400 GbE) preferred for headroom.
- PCIe Gen5 x16 slot for BF3 itself
- Two physical ports: "wan" (talks to compute cluster), "tt" (talks to WH)
- Optional: passive copper DAC to WH for the "tt" port (cheap, low-latency); fiber for longer distances

**Wormhole chip + host:**
- Standard WH n300 in a Gen3 x16 PCIe slot
- Hugepages: 1024 × 2 MB minimum (`echo 1024 > /sys/devices/system/node/node0/hugepages/hugepages-2048kB/nr_hugepages`)
- WH's CMAC port wired to BF3's "tt" port via QSFP28 (100 GbE)

**Network topology:**
- Compute cluster ↔ BF3 "wan": existing RoCE fabric (whatever the cluster runs)
- BF3 "tt" ↔ WH CMAC: dedicated 100 GbE link, point-to-point, no switch needed

**Sizing rule:** one BF3 per WH-chip-cluster. The BF3 is the "shadow NIC" for TT capacity. If you have 8 WH chips on one host, one BF3 with 8 internal channels is sufficient — BF3's 200/400 GbE has plenty of bandwidth above v1's ~25 Gbps ceiling.

---

## 3. Software stack

| Layer | Component | Version | Source |
|---|---|---|---|
| BF3 OS | DOCA OS Linux | 2.7+ (Ubuntu 22.04 / RHEL 9 based) | NVIDIA NGC |
| DOCA | DOCA SDK | 2.7+ | https://developer.nvidia.com/doca |
| DOCA libs | doca_rdma, doca_flow, doca_dma | bundled | with DOCA SDK |
| OFED (BF3) | MLNX_OFED | 5.8+ (bundled with DOCA OS) | NVIDIA |
| DPDK | Mellanox DPDK | 22.11+ (bundled) | with DOCA |
| Build tools | meson, ninja, gcc-aarch64 | latest | apt |
| Compute-side OFED | MLNX_OFED on the compute nodes | match HCA fw rev | NVIDIA |
| **WH FW** | erisc_cmac_simple.elf | **md5 43bdcb46 (v1.0)** | this repo |
| **WH SDK** | external_iface_sender.{cpp,hpp} | v1.0 tag | this repo |

---

## 4. Components to build

### 4.1 BF3 gateway daemon (`tt_rdma_gw`)

Single Linux process running on BF3 Arm cores. Owns:
- The "wan" RoCE endpoint (using `doca_rdma` to terminate QPs)
- The "tt" raw-Ethernet endpoint (DPDK or DOCA Eth)
- Translation engine
- Control-plane gRPC/REST server for MR registration

```
tt_rdma_gw/
├── meson.build
├── src/
│   ├── main.c                 ← daemon entry, signal handling, CLI
│   ├── doca_rdma_endpoint.c   ← terminates RoCE QPs via doca_rdma
│   ├── doca_eth_endpoint.c    ← raw L2 send/recv on "tt" port
│   ├── translate.c            ← bidirectional translation engine
│   ├── qp_table.c             ← RoCE QPN → TT seq# / MR mapping
│   ├── mr_table.c             ← RoCE rkey → TT slot index
│   ├── psn_seq.c              ← per-QP PSN ↔ TT seq tracking
│   ├── control_plane.c        ← gRPC/REST for app integration
│   └── tt_v1_codec.{c,h}      ← encode/decode TT-RDMA-v1 32 B header
├── proto/
│   └── tt_rdma_gw.proto       ← control plane API
└── tests/
    ├── translate_test.c       ← unit tests with synthetic frames
    └── e2e_test.sh            ← end-to-end on real hardware
```

### 4.2 Control plane (`tt_rdma_gw_ctl`)

CLI + gRPC client for apps to register MRs:

```bash
# Register a TT MR slot for an app
tt_rdma_gw_ctl register-mr \
    --slot 2 \
    --base-noc-addr 0x418000050000 \
    --length 0x8000 \
    --access REMOTE_RW

# List active mappings
tt_rdma_gw_ctl list-mappings
```

Internally calls the WH's `register_mr_slot()` via PCIe doorbell (through
`wh_mr_agent`), then records the (RoCE-rkey, TT-rkey) mapping in BF3's
qp_table.

### 4.3 WH-side helper (`wh_mr_agent`)

Tiny daemon on the WH-host that owns the PCIe connection to the WH chip.
Receives MR-registration requests over a UNIX socket from BF3 (via OOB
management channel), calls `ExternalIfaceSender::register_mr_slot()`,
returns the actual TT rkey.

Why a separate process: BF3 has no PCIe path to WH directly. WH's PCIe
goes to the host CPU. So MR registration must traverse:
BF3 → mgmt link → WH-host → ExternalIfaceSender → WH chip.

---

## 5. Translation logic (per-opcode)

### 5.1 RC SEND → TT 0x01 SEND

```
RoCEv2 receive (wan port):
  L2 + IP + UDP + BTH(opcode=0x00 RC_SEND_ONLY) + payload + ICRC
       ↓ ConnectX-7 hardware terminates the QP, hands payload to DOCA
  DOCA delivers payload buffer to gateway

Gateway:
  qp_table.lookup(BTH.QPN) → target_tt_slot
  Build TT v1 header:
    opcode = 0x01
    version_flags = 0x01
    tag = seq & 0xFFFF
    length = payload_size
    seq = next_outgoing_seq[QPN]++
    rkey = 0      (SEND doesn't use MR on target)
    remote_offset = 0
    imm = 0
  Emit on "tt" port: [L2 hdr ethertype 0x1AF4] + [32 B TT hdr] + [payload]

WH FW:
  case 0x01 → NoC-push to host RxWqeRing slot

Return path:
  WH emits ACK frame (opcode 0x40) when it processes the SEND
  Gateway intercepts, translates to RC ACKNOWLEDGE (AETH with same PSN)
  ConnectX-7 sends ACK back to compute-node HCA
```

### 5.2 RC RDMA_WRITE → TT 0x10 WRITE

```
RoCEv2 receive:
  BTH(opcode=0x06 RC_RDMA_WRITE_ONLY) + RETH(vaddr, rkey, length) + payload

Gateway:
  qp_table.lookup(QPN) → tt_slot context
  mr_table.lookup(RETH.rkey) → tt_rkey, tt_base_off
  tt_remote_off = RETH.vaddr - mr_table[RETH.rkey].vaddr_base + tt_base_off
  Build TT v1 header:
    opcode = 0x10
    length = RETH.length
    seq = next_outgoing_seq[QPN]++
    rkey = tt_rkey
    remote_offset = tt_remote_off
  Emit + payload

WH FW: case 0x10 → ncrisc_noc_fast_write to MR_TABLE[tt_rkey].base + tt_remote_off
```

### 5.3 RC RDMA_WRITE_WITH_IMM → TT 0x11 WRITE_IMM

Same as 5.2 + carry RETH-following ImmediateData into TT header `imm` at +24.

### 5.4 RC RDMA_READ_REQUEST → TT 0x20 READ_REQ

```
RoCEv2 receive:
  BTH(opcode=0x0C RC_RDMA_READ_REQUEST) + RETH(vaddr, rkey, length)

Gateway:
  - Look up tt_rkey, tt_remote_off (same as WRITE)
  - Allocate a local "pending read" buffer on BF3 to receive the response
  - Build TT v1 READ_REQ frame
  - Record (PSN, QPN, length) so we can construct the RC READ_RESPONSE
  - Emit READ_REQ
  - Wait for TT 0x21 READ_RESP arrival

WH FW: case 0x20 → ncrisc_noc_fast_read from MR, stage 0x21 in TX_BUF1, fire on TXQ1

Gateway receives 0x21:
  Lookup by tag → original PSN/QPN
  Build RC RDMA_READ_RESPONSE_ONLY (or _FIRST/_MIDDLE/_LAST if multi-segment)
  AETH credit field
  Send via doca_rdma_send back to compute-node HCA
```

### 5.5 RC RDMA_READ multi-segment

RoCE chunks large READs at MTU boundaries. WH READ_REQ is one-shot up to
4080 B. Gateway:
- If `RETH.length > MTU`, gateway issues multiple TT READ_REQs with different tags
- Re-assembles responses, sends RC READ_RESPONSE_FIRST/MIDDLE/LAST
- MVP: cap RC READ length at 4080 B and let the app retry larger ones in chunks

### 5.6 RC ACK ↔ TT 0x40 ACK

RoCE ACK is per-QP, PSN-based. TT ACK is cumulative seq.

```
TT 0x40 ACK from WH arrives:
  ack_seq = (seq from the v1 header)
  Look up which QPN was associated with this seq range
  Generate RC ACKNOWLEDGE with corresponding PSN(s)
  Send to compute HCA

Reverse: RoCE ACK from compute HCA arrives:
  Map PSN → corresponding TT seq
  Emit TT 0x40 with that seq
  WH FW updates rcb.acked_idx
```

### 5.7 RC ATOMIC ops (FetchAdd, CmpSwap)

Not in TT v1. Two choices:
1. **NAK**: Send RC RNR-NAK or remote-access-error. App sees atomic-not-supported.
2. **Emulate on BF3**: Treat as `READ + local CAS in Arm + WRITE`. Defeats
   atomicity guarantees unless we have a per-MR lock. Only safe with
   single writer per MR.

MVP: option 1. v1.2 fix: add TT 0x30-range ATOMIC opcodes, then translate properly.

### 5.8 Connection setup (RDMA-CM)

RoCEv2 connections set up via `librdmacm` (TCP-based 3-way handshake
exchanging QP info).

```
Compute side: rdma_create_qp + rdma_connect("bf3-ip", port)
BF3 listens: rdma_listen — accepts QP creation
BF3 also runs: tt_rdma_gw allocates a TT slot via wh_mr_agent for this QP
```

QP table entry on BF3:

```c
struct qp_context {
  uint32_t roce_qpn;
  uint32_t next_outgoing_seq;
  uint32_t expected_inbound_psn;
  struct mr_binding mrs[N];   // RoCE rkey → TT slot
  ...
};
```

---

## 6. Memory model + MR binding

RoCE expects:
- Apps register memory regions via `ibv_reg_mr(addr, length, access)` → get `rkey, lkey`
- Use that `rkey` in WRITE/READ ops

TT-RDMA needs:
- A pre-allocated TT MR slot (0..15), each with a NoC-encoded base address
- WH FW knows nothing about RoCE rkeys

**Two MR registration models:**

### Model A: explicit pre-registration (MVP)

```bash
# App or admin calls control plane BEFORE running RDMA traffic
tt_rdma_gw_ctl register-mr \
  --slot 3 \
  --base-noc-addr 0x...  \
  --length 64MB \
  --access REMOTE_RW
# Returns: roce_rkey to use in ibv_post_send
```

App's `ibv_reg_mr` returns a pre-allocated rkey that BF3 already knows
maps to TT slot 3.

Tradeoff: clunky for dynamic workloads, simple for steady-state.

### Model B: app-transparent (better, post-MVP)

BF3 intercepts `ibv_reg_mr` calls from the compute node:
1. App calls `ibv_reg_mr(host_addr, len, REMOTE_WRITE)` on compute node
2. ConnectX HCA generates a RoCE rkey
3. BF3's gateway notices the new MR via control channel
4. BF3 → wh_mr_agent → register TT MR slot

The question for model B is **where does the data actually land on the TT side?** Options:
- A pre-allocated WH L1 buffer per MR
- A pre-allocated WH DRAM buffer
- Host hugepage RAM mapped via NoC-PCIe encoding

The "host hugepage" option mirrors how GPUDirect works (peer DMA to GPU memory).
Tenstorrent's equivalent: TT compute reads from host hugepage via PCIe NoC
after the WRITE lands.

**Recommendation:** Start with Model A for MVP. Move to Model B once you
understand the actual app patterns. Most HPC apps register MRs upfront
and reuse them.

---

## 7. Fast path vs slow path

BF3 has two ways to handle translation:

### Software fast path (Arm cores)

- DPDK app polls "wan" RX queue
- Per-packet C code in Arm cores parses BTH, calls translation, builds TT frame, TXes on "tt" port
- Latency: ~1–5 µs per frame
- Throughput: ~25 Gbps is fine (target ceiling)
- Easier to debug, more flexible

### Hardware fast path (DOCA Flow)

- Match BTH opcode in hardware
- Rewrite header in hardware (BTH → TT 32B)
- Bypass Arm cores for the bulk path
- Latency: ~hundreds of ns
- Throughput: 100s of Gbps possible
- Setup complexity: high; requires careful DOCA Flow rule definition

**Recommendation:** MVP uses software fast path on Arm. Get correctness
first. Profile. Add DOCA Flow acceleration for the common cases
(RC_RDMA_WRITE_ONLY most importantly) only if needed.

---

## 8. Phased implementation plan

| Phase | Scope | Calendar | Eng-weeks |
|---|---|---|---|
| **G.1** | Bootstrap: BF3 OS up, toy DOCA RDMA + Eth endpoints, no translation | 1 wk | 1 |
| **G.2** | RC SEND → TT SEND (with one pre-registered MR) | 2 wks | 2 |
| **G.3** | RC RDMA_WRITE → TT WRITE | 1 wk | 1 |
| **G.4** | RC RDMA_READ → TT READ_REQ/RESP | 2 wks | 2 |
| **G.5** | ACK round-trip (PSN ↔ seq) | 1 wk | 1 |
| **G.6** | RDMA-CM connection setup, multi-QP | 1 wk | 1 |
| **G.7** | App integration: MPI hello-world over UCX | 1-2 wks | 2 |
| **MVP total** | End-to-end, single QP, single MR | **~10 wks** | **~10** |
| G.8+ | Production hardening (multi-QP, error inj, telemetry, DOCA Flow accel) | 3+ months | ~12 |

**Team:** 1 DOCA dev (BF3 side) + 0.5 WH dev (PCIe agent + integration tests) + 0.25 sysadmin (rig setup).

---

## 9. Testing strategy

### Unit tests (BF3, no hardware)

- `translate_test.c`: hand-craft RoCE frames, run through translation,
  assert TT output bytes
- `rkey_table_test.c`: stress-test the rkey mapping under concurrent registrations
- `psn_seq_test.c`: PSN wrap, out-of-order handling

### Integration tests (BF3 + WH)

`e2e_test.sh`:
1. Bring up WH with v1.0 FW
2. Bring up BF3 with tt_rdma_gw
3. Register MR slot 3 via control plane
4. Run `ib_write_bw` from compute side, verify bytes land at correct WH offset
5. Run `ib_send_bw`, `ib_read_bw`
6. Run a small MPI test
7. Verify no leaked QPs, MR slots, etc.

### Soak tests

- 24 h `ib_write_bw -D 86400` to detect any state-machine drift
- Mixed read/write/send workload
- Connection churn: open/close 1000 QPs per second

### Compatibility matrix

| RDMA tool | Expected to work? | Phase |
|---|---|---|
| `ib_send_bw` | yes | G.2 |
| `ib_write_bw` | yes | G.3 |
| `ib_read_bw` | yes | G.4 |
| `rping` | yes | G.6 |
| MPI (MVAPICH, Open MPI) point-to-point | yes | G.7 |
| UCX | yes | G.7 |
| NVMe-oF | yes (BF3 already does this) | post-G.7 |
| NCCL | **no** (GPU-only, closed-source) | never |
| HW collectives (SHARP) | no | never |

---

## 10. Observability + diagnostics

Critical for production. Build in from G.1:

- DOCA Telemetry plugin: per-QP TX/RX bytes, per-opcode counters
- syslog: all rkey lookups, translation failures, PSN skips
- gRPC metrics endpoint: rkey table size, QP count, frame rate
- Per-direction Gbps + retx counter (from WH dbg)
- Easy way to dump qp_table and mr_table state for debugging
- Prometheus + Grafana for cluster operators

---

## 11. Risks + open questions

1. **DOCA app sandboxing.** BF3 runs Linux; you can SSH in and debug.
   Production deployments might require signed images. Plan with the
   NVIDIA DOCA team.

2. **WH-host management channel.** How does BF3 talk to wh_mr_agent?
   Options: Ethernet management LAN, OOB UART, dedicated PCIe link.
   Each has its own latency + reliability tradeoffs.

3. **Single point of failure.** BF3 down = TT cluster unreachable from
   RDMA peers. Plan for HA: dual BF3, active/standby with floating MR table.

4. **Performance under load.** Need careful profiling. The 25 Gbps WH
   ceiling is fine for many workloads; for KV-cache transfer at
   100 Gbps+ you'd need multiple WHs in parallel.

5. **Atomics.** v1 doesn't support them. Many HPC workloads use them
   (CAS for locks, FetchAdd for counters). Either implement on WH or
   do an "atomic via lease" workaround on BF3.

6. **MTU + fragmentation.** RoCEv2 typically uses MTU 1024 or 4096.
   TT can go to 4080. Match them or fragment.

7. **Multi-tenant isolation.** If multiple users want to share the WH
   cluster via the gateway, you need per-tenant rkey namespaces, MR
   slot allocation, QoS. Real but solvable.

---

## 12. Half-step alternative: software-only gateway

If 10 weeks of DOCA dev is too much commitment for the initial experiment:

**Use a regular server with two ConnectX NICs (one "wan", one "tt")
instead of BF3.**

- Same architecture, Linux process with libibverbs + DPDK does the translation
- Same code, runs on x86 Linux instead of BF3 Arm
- ~2× slower (no DOCA Flow HW acceleration) but functionally identical

**Use this for the MVP/PoC.** Port to BF3 once the architecture is
validated. This de-risks the project before committing to BF3 hardware +
the DOCA learning curve.

---

## 13. What this unlocks

After MVP (G.7):

- **Existing RDMA-aware HPC/AI workloads run on Tenstorrent clusters with no
  app changes.** MPI, UCX, framework-level RDMA backends.
- **Disaggregated inference becomes possible** — GPU node WRITEs KV cache
  to TT node via RDMA, TT runs inference layer.
- **Industry-standard tooling for bring-up and debugging.** `ib_*_bw`
  works for performance characterization.
- **Multi-vendor cluster narrative** is real: "we're an RDMA endpoint,
  treat us like one."

After production hardening:

- **Mixed-vendor production deployments** with monitoring, failover,
  multi-tenancy.
- **Standard SLA story** — same RDMA fabric guarantees as any other
  endpoint on the wire.

---

## 14. Companion / related docs in this repo

- `README.md` — TT-RDMA v1.0 overview, shipped + planned
- `tt-rdma-wire-protocol-v1.md` — wire format spec
- `tt-rdma-fw-arch-rx.md` — FW RX architecture
- `tt-rdma-host-sdk.md` — `ExternalIfaceSender` API surface
- `tt-rdma-pfc-lossless.md` — PFC config for lossless fabric
- `tt-rdma-eswitch-bypass.md` — SR-IOV/switchdev for PCIe-bound paths

The BF3 gateway built per this doc consumes the **wire protocol** + **host
SDK** from those docs. It does not require any changes to the WH FW
beyond v1.0.
