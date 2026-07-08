# TT-RDMA-over-Ethernet — v1.0 — SHIPPED 2026-05-13

**Status (2026-05-13):** v1.0 shipped. All eight v1 opcodes implemented end-to-end. Host SDK exposes `post_send` / `post_send_read` / `register_mr_slot` / `wait_completion`. TX 22–25.8 Gbps, RX 24.8 Gbps (Mellanox Gen3 x4 ceiling). 100 % wire-verified.

**Repo tags:**
- `tt-metal-external-eth`: `tt-rdma-v1.0` on branch `aperezvicente/external-mac-port-interop`
- `budabackend-master`: `tt-rdma-v1.0` on branch `master` → pushed to `aperezvicente-TT/wh-erisc-fpga` as `tt-rdma-v1-import`
**Target audience:** anyone touching the WH external CMAC path, the wire protocol, or the host SDK.

This directory holds the design contract for **TT-RDMA**: a verbs-shaped, NIC-agnostic RDMA layer that runs over standard 100 GbE between a Wormhole erisc core's CMAC and any compliant partner (Mellanox NIC, FPGA-based NIC, or another WH chip).

## Document map

| File | Scope |
|---|---|
| `README.md` (this file) | Status, architecture overview, phase plan |
| `tt-rdma-wire-protocol-v1.md` | 32-byte header, opcode set, hex examples |
| `tt-rdma-fw-arch-rx.md` | WH FW RX path (single-ring BUF_WRAP, opcode dispatch, MR validation) |
| `tt-rdma-host-sdk.md` | `TtRdmaEndpoint` API: `register_mr`, `post_send_write`, `post_recv`, etc. |
| `tt-rdma-pfc-lossless.md` | Mellanox PFC config + WH CMAC pause enable |
| `tt-rdma-eswitch-bypass.md` | Mellanox switchdev / vfio-pci paths to remove the eSwitch bottleneck |
| `bf3-gateway-design.md` | BlueField-3 + DOCA gateway design for RoCEv2 ↔ TT-RDMA-v1 translation — lets `libibverbs`/MPI/UCX apps reach WH nodes |
| `tt-rdma-verbs-provider.md` | Native `rdma-core` provider (`libtt_rdma`) — the software alternative to the BF3 gateway for FPGA-TT-Link |
| `tt-rdma-bidirectional-mesh-gap.md` | What's missing to make a TT node a *symmetric* RoCEv2-reachable mesh endpoint (accessible in both directions via a gateway) |
| `tt-rdma-mesh-addressing-spec.md` | Detailed spec for the two critical-path gaps: endpoint addressing (rkey/QPN routing) + remote-MR import + QP object — no FW change |
| `tt-rdma-switch-interop-architecture.md` | Gateway path vs native-switch path: how TT interoperates on L2 switches / RoCE fabrics today (gateway + in-band addressing) vs the Exabox A0/Keraunos-2 silicon fix (parallel queues + linked-list ARQ + bigger header table) |
| `tt-rdma-mesh-egress-multicast.md` | Composition: an **interior** tt-fabric mesh chip pushing a tensor out through an edge chip + gateway, incl. **one-to-many multicast** to external GPUs (memory-tier staging + gateway fan-out — the piece neither the fabric nor the RDMA docs cover) |
| `runtime-workload-coexistence.md` | "Where do bytes move" diagram: a Tensix inference workload (Qwen3) sharing one chip with the erisc CMAC FW over FPGA-TT-Link |
| `tt-rdma-blackhole-port.md` | Bringing TT-RDMA to **Blackhole** as a 2nd chip-side target: same wire + same gateway, but a different FW model (tt-metal active-eth over `bh-erisc` base FW), Rianta MAC, 400 G links, RoCE HW-offload hooks (note: HW packet-resend ARQ exists on WH too — not a BH delta) |
| `glossary.md` | Acronym + term key (IMM, RETH, PSN, RCB, MR, QPN, WQE/CQE, eswitch, …) |

## Architecture, one diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│ APPLICATION  — host CPU  OR  tensix kernel on this/other WH chip     │
├──────────────────────────────────────────────────────────────────────┤
│ TtRdmaEndpoint SDK  (verbs API; see tt-rdma-host-sdk.md)             │
│   register_mr, post_send_write/read/atomic, post_recv, poll_cq       │
├──────────────────────────────────────────────────────────────────────┤
│ WH erisc FW dispatch  (see tt-rdma-fw-arch-rx.md)                    │
│   WRITE  ─→ MR-table lookup ─→ noc_write(mr.base_noc + offset)       │
│   READ   ─→ MR-table lookup ─→ noc_read + CMAC-TX READ_RESP          │
│   SEND   ─→ host RxWqeRing slot (DMA-push via NoC→PCIe)              │
│   ACK    ─→ Phase 2 reliability state                                │
├──────────────────────────────────────────────────────────────────────┤
│ CMAC raw-mode RX (single 14 KB BUF_WRAP ring; length-prefixed frames)│
│ CMAC TX  (DMA-pull from any source MR via NoC PCIe or NoC fabric)    │
├──────────────────────────────────────────────────────────────────────┤
│ PFC priority 3 lossless                                              │
├──────────────────────────────────────────────────────────────────────┤
│ 100 GbE wire ↔ Mellanox (host RDMA)  /  FPGA NIC (chip-to-chip)      │
│                                       /  another WH (TT-Link)       │
└──────────────────────────────────────────────────────────────────────┘
```

**Key insight: `mr.base_noc_addr` is a 64-bit NoC-encoded address.** It can point at:
- a host hugepage (via the PCIe NoC tile) — ~25 Gbps host-RDMA ceiling, bounded by host PCIe
- a tensix core's L1 / DRAM (local NoC) — **line rate** for inference scratchpads, KV-cache
- a remote WH chip's L1 / DRAM (via NoC fabric over Ethernet) — **line rate, no host involvement** (the TT-Link vision)

The same FW handler services all three cases. The host's `register_mr` call decides which NoC address goes in the table; FW just looks it up + does an NoC write.

## NIC-agnostic design

The WH-side FW + wire protocol is **NIC-agnostic**. What differs across deployments is the partner.

| Component | Mellanox path | FPGA path | TT-RDMA WH↔WH path |
|---|---|---|---|
| WH FW (`erisc_cmac_simple`) | unchanged | unchanged | unchanged |
| Wire protocol | unchanged | unchanged | unchanged |
| Host SDK (WH side) | unchanged | unchanged | unchanged |
| Partner driver | DPDK + `test_external_cmac_smoke_mlx` | open-nic-shell + QDMA | another WH erisc + CMAC |
| Partner ingress | eSwitch → DPDK queue | QDMA DMA → host RAM | NoC fabric → tensix/L1/DRAM |
| Sustained Gbps | ~5–6 Gbps legacy / ~25 Gbps switchdev | ~100 Gbps wire (FPGA-dependent) | **~line rate**, no host touch |
| PFC config | `mlnx_qos` | FPGA-programmable | CMAC native (Phase A) |
| Switchdev / SR-IOV | required for high-throughput | N/A | N/A |

Today's testbed: **WH ↔ Mellanox ConnectX-5 (Gen3 x4)**. Every Gbps number quoted below is from this rig.

> **The table above varies the *partner*; the *chip side* is a second axis.** All
> shipped numbers are Wormhole. **Blackhole is a second chip-side target** — same
> wire (`0x1AF6`) and same gateway concept, but a different FW model (a tt-metal
> active-eth kernel coexisting with the `bh-erisc` base FW, not standalone
> `erisc_cmac_simple`), a Rianta MAC, 400 Gbps links, and RoCE HW-offload hooks.
> (Hardware packet-resend ARQ is *not* a BH delta — WH has it too via `ETH_TXQ`
> packet mode; TT-RDMA uses raw mode on both, so it's unused either way.) The
> WH-specific docs (PFC register map, RX FW arch, L1 layout) need BH variants.
> Full port design: `tt-rdma-blackhole-port.md`.

## Where we are (shipped vs planned)

### Shipped (v1.0)

| Phase | Description | FW md5 (cumulative) |
|---|---|---|
| **Phase A** | CMAC PFC pause enable (RFE+PFCE + per-priority TFE) | b9b826ce |
| **Phase B** | Wire opcode dispatch; fixed `/4 vs /16` BUF_START_WORD_ADDR bug | c14246df |
| **Phase C** | Single-ring BUF_WRAP RX (eliminated ping-pong race) | b5f184f0 |
| **TX jumbo** | 32-slot × 4096 B WQE ring | 883b6286 |
| **P1** | L1 relayout: TX_BUF→0x29000, RXQ1 disabled, RX ring 14 KB | b697d5e7 |
| **P2** | MR table at L1:0x8500, strict v1 opcode filter, 16 B alignment | e2525165 |
| **P3** | WRITE opcode 0x10 (local memcpy) | 7116e420 |
| **P4** | WRITE → `ncrisc_noc_fast_write` (off-fabric) | 377d5a57 |
| **P5** | Host RxWqeRing — SEND DMA-pushed to host hugepage at +128 KB | 024982b8 |
| **P6** | READ_REQ target-side handler + RESP staging | 088c139d |
| **P6.1** | READ_RESP wire emission via CMAC TXQ1 | 6021e2b2 |
| **P7** | Host `register_mr_slot` doorbell at RCB+0x28 | ccfd826e |
| **Q(a)** | MAC JE=1 + MLX MTU 9000 + unicast → 100 % FW admit at 4080 B jumbo | b06cecae |
| **Q(b)** | SEND_IMM (0x02) + WRITE_IMM (0x11) handlers | 790cbd15 |
| **Q(c)** | READ_REQ uses `ncrisc_noc_fast_read` from full 64-bit MR base | cf8eb752 |
| **Phase R** | ACK opcode 0x40 reception; receiver auto-ACKs; cumulative-ACK validated at 0 % drop | (no FW change) |
| **Phase 3.3** | 4-deep NoC PCIe read pipeline (TX_BUF0_A/B/C/D + TRIDs 4–7) | 13e160cf |
| **Phase I** | Host `post_send_read` + correlation table at L1:0x8800; round-trip with `--rx-echo-read` | **43bdcb46** |

### Validated performance (sustained, warm chip)

| Direction | Frame size | Number | What it means |
|---|---|---|---|
| **WH → MLX (TX)** | 4080 B jumbo | **22–25.8 Gbps** | 4-deep PCIe-read pipeline; 100 % wire-verified via Mellanox PHY counter |
| **MLX → WH (RX)** | 4080 B jumbo | **24.8 Gbps** | 100 % CMAC admit, 99.997 % FW admit (28 ppm unknown-opcode noise) |
| **Reliability** | 500 frames @ 0 % drop | **100 % success** | retx_runs=23 (normal cumulative-ACK pacing) |
| **READ initiator** | 4 × 128 B | **4/4 landed** | host post_send_read → wire → MLX echo → local MR write — exact byte-match |

Both directions are within ~1 Gbps of the Mellanox PCIe Gen3 x4 ceiling.

### Locked-in design decisions

- **Wire ethertype**: 0x1AF6 (legacy 0x1AF4/5 stay during transition).
- **Header**: 32 bytes, fixed; `length` field at +4 (kills the BUF_PTR-inference race).
- **Opcode encoding**: 0x01 SEND, 0x02 SEND_IMM, 0x10 WRITE, 0x11 WRITE_IMM, 0x20 READ_REQ, 0x21 READ_RESP, 0x40 ACK, 0xF0 CONTROL.
- **Reserved opcode ranges**: 0x0X SEND-family, 0x1X WRITE-family, 0x2X READ-family, 0x3X ATOMIC-family, 0x4X control-plane, 0xF0 CONTROL.
- **Frame alignment**: payload must be 16-byte aligned (matches CMAC L1 placement).
- **MR table**: 16 entries × 32 B at L1:0x8500. `rkey = (slot_idx << 24) | (rand16 << 8) | generation`. Holds only *local* regions exposed as inbound targets — remote MRs you initiate against are host-side only and cost no slot. Sizing per workload (1 for smoke, ~12 for a verbs app, 64 for NCCL/UCX): `tt-rdma-host-sdk.md §2 "How many MRs do I actually need?"`.
- **Reliability**: auto-on for one-sided ops (WRITE/READ); off for plain SEND.
- **RX ring**: single 14 KB at L1:0x4000–0x7800; CMAC BUF_WRAP enabled.
- **TX layout**: 32 slots × 4096 B at WQE_PAYLOAD_BASE = 0x9000 (ends at 0x29000); TX_BUF0/1 at 0x29000/0x2A000.

### Planned (post-v1, in priority order)

| # | Deliverable | Effort | Note |
|---|---|---|---|
| **v1.1 selective-ACK** | Per-seq ACK + rate-limited retx (replaces re-fire-whole-window) | ~1 week | v1 ships with cumulative-ACK + PFC-lossless wire; selective-ACK matters once we run over lossy fabrics |
| **v1.1 PFC e2e** | Mellanox `mlnx_qos` + VLAN PCP=3 tagging in DPDK | rig-config-heavy | FW side already done (Phase A) |
| **v1.1 SR-IOV bypass** | mlxconfig SRIOV_EN=1 → switchdev → vfio-pci | ½ day after rig config | unblocks the ~6 Gbps eSwitch ceiling for small frames |
| **v1.2 ATOMIC** | FAA / CAS opcodes (0x30-range) | 3 days | wire format already reserved |
| **v1.2 multi-chip** | Multi-NIC mesh, 8+ erisc per chip | ~2 weeks | scales the SDK API beyond one port |
| **v1.2 1 h+ soak** | Long-run stability under sustained traffic | 1 day rig time | confirms no drift in dbg counters or admit rate |
| **v1.2 dynamic ARQ** | tune retx_timeout based on RTT measurement | 2 days | currently fixed at 10 ms |

## Critical files in the repo

**FW (in `budabackend-master`):**
- `src/firmware/riscv/targets/erisc_cmac_simple/src/main_cmac.cc` — dispatcher + WRITE handler + MR table init
- `src/firmware/riscv/targets/erisc_cmac_simple/src/api/eth_cmac_init.cpp` — RX ring size, RXQ1 disable, PFC enable, TX_BUF address
- `src/firmware/riscv/targets/erisc_cmac_simple/src/api/eth_cmac_init.h` — `TX_BUF0_ADDR`, `TX_BUF1_ADDR`

**Host (in this repo):**
- `tt_metal/llrt/external_iface_sender.{hpp,cpp}` — WQE-ring API, hugepage mapping, post_send
- `tests/tt_metal/llrt/test_external_cmac_soak.cpp` — TX throughput harness
- `tests/tt_metal/llrt/test_external_cmac_rx_idle.cpp` — RX harness (admit + dbg counter readback)
- `tests/tt_metal/llrt/test_external_cmac_smoke_mlx.cpp` — DPDK partner app (SEND, WRITE, ACK)

**Test harness:**
- `tests/tt_metal/llrt/setup_hugepages_mlx_smoke.sh` — 2MB hugepages for DPDK
- `scripts/install_sudoers_mlx_smoke.sh` — passwordless sudo for `mlnx_qos`, `dcb`, `devlink`, etc.

## FW build flow

The cmac_simple FW lives in a separate repo (`budabackend-master`) and uses an older RV32 toolchain. Required env:

```bash
cd budabackend-master/src/firmware/riscv/targets/erisc_cmac_simple
SFPI=/home/alex/mpi-shfs/tenstorrent/sfpi/build/sfpi \
SFPI_VANILLA=1 \
ARCH_NAME=wormhole_b0 \
make
# Install into tt-metal:
cp .../build/.../erisc_cmac_simple.elf tt_metal/hw/firmware/bin/
```

`SFPI_VANILLA=1` is **required** — the default Makefile flags use `-mwormhole`/`-march=rv32imw` which aren't recognized by the newer GCC 15.1.0 in our SFPI install.

## Backups

Every milestone's FW elf is preserved at `/tmp/erisc_cmac_simple.elf.*`:

```
/tmp/erisc_cmac_simple.elf.phase3_2.af1f0aad  # pre-Phase-A baseline
/tmp/erisc_cmac_simple.elf.phaseC.b5f184f0    # pre-jumbo
/tmp/erisc_cmac_simple.elf.jumbo.883b6286     # validated 20.84 Gbps TX
```

(Note: `/tmp/` is rebooted away. Rebuild from source if needed; commits in `budabackend-master` are the authoritative source.)

## Open questions

1. **CMAC TX max-pkt ceiling**: 4096 works, 9216 hangs. Real ceiling in `(4096, 9216]` — binary-search left for future.
2. **CMAC RX alignment beyond 16 B**: 1450B test showed 16% admit; the round-up-to-16 in P2 partly helped. Some larger alignment (32? 64?) may apply for sizes farther from 2^N. Needs an empirical sweep.
3. **Ring overflow at sustained small-frame bursts**: 14 KB ring + ~1 µs FW iter handles up to ~50 frames buffered. Bursts of 1000+ × 256 B still overflow ~30%. Mitigations: ARQ (Phase 2 extension), faster outer loop, or "drain to NoC immediately" inside the walk.
4. **PCS link state under high TX**: not yet stressed beyond brief soaks. Long-soak (1 hour+) link stability is unproven.
5. **Multi-MR + multi-tenant**: only slot 0 tested. Concurrent MRs with different rkeys should work by design but no regression test yet.

## Migration / compatibility

- v0 (legacy soak at ethertype 0x1AF4, no opcode header) still receivable by FW's `op_unknown` drop path — counts noise, doesn't propagate as RDMA.
- v1 (0x1AF6 or 0x1AF4 + RDMA header) is the production format.
- The phase plan was designed to be roll-forward-only: every shipped FW retains backward compatibility with prior tests.
