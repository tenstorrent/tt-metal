# TT-RDMA-over-Ethernet — v1 Specification

**Status:** living document. v1 architecture locked; phases P1–P4 shipped, P5+ in progress.
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

## Where we are (shipped vs planned)

### Shipped

| Phase | Description | Memory note | FW md5 |
|---|---|---|---|
| **Phase A** | CMAC PFC pause enable (RFE+PFCE + per-priority TFE) | `project_phase_a_pfc_fw_fix.md` | b9b826ce |
| **Phase B** | Wire opcode header at 0x1AF6; uncovered `/4 vs /16` ping-pong bug | `project_phase_b_buf_start_word_addr_bug.md` | c14246df |
| **Phase C** | Single-ring BUF_WRAP RX (eliminated ping-pong race; 0 mixed_size) | `project_phase_c_single_ring.md` | b5f184f0 |
| **TX jumbo** | WQE ring 64→32 / payload 1536→4096 → **20.84 Gbps TX** | `project_tx_jumbo_20gbps.md` | 883b6286 |
| **P1** | L1 relayout: TX_BUF→0x29000, RXQ1 disabled, RX ring grew to 14 KB | `project_p1_l1_relayout.md` | b697d5e7 |
| **P2** | MR table at L1:0x8500; strict v1-opcode filter; 16-byte alignment | `project_p2_mr_table_align.md` | e2525165 |
| **P3** | WRITE opcode 0x10 (local-L1 memcpy); MR lookup + rkey/access/bounds validation | `project_p3_write_opcode.md` | 7116e420 |
| **P4** | **WRITE handler uses `ncrisc_noc_fast_write` → off-fabric WRITE**; 1000/1000 @ 1024B | `project_p4_noc_write.md` | 377d5a57 |

### Validated performance (this rig)

| Direction | Frame size | Number | What's measured |
|---|---|---|---|
| WH → wire TX | 4080 B | **20.84 Gbps** | sustained, PCIe-post-bound |
| MLX → WH RX | 256 B / 1024 B (16-aligned) | **100% admit** | FW serviced every CMAC-admitted frame |
| MLX → WH RX | 1450 B | (alignment-quirk; see P2 note) | requires post-strip size ≡ 0 mod 16 |
| MLX → WH WRITE | 1000 × 1024 B | **1000/1000, 0 drops** | P4 NoC-direct WRITE, fire-and-forget |

### Locked-in design decisions

- **Wire ethertype**: 0x1AF6 (legacy 0x1AF4/5 stay during transition).
- **Header**: 32 bytes, fixed; `length` field at +4 (kills the BUF_PTR-inference race).
- **Opcode encoding**: 0x01 SEND, 0x02 SEND_IMM, 0x10 WRITE, 0x11 WRITE_IMM, 0x20 READ_REQ, 0x21 READ_RESP, 0x40 ACK, 0xF0 CONTROL.
- **Reserved opcode ranges**: 0x0X SEND-family, 0x1X WRITE-family, 0x2X READ-family, 0x3X ATOMIC-family, 0x4X control-plane, 0xF0 CONTROL.
- **Frame alignment**: payload must be 16-byte aligned (matches CMAC L1 placement).
- **MR table**: 16 entries × 32 B at L1:0x8500. `rkey = (slot_idx << 24) | (rand16 << 8) | generation`.
- **Reliability**: auto-on for one-sided ops (WRITE/READ); off for plain SEND.
- **RX ring**: single 14 KB at L1:0x4000–0x7800; CMAC BUF_WRAP enabled.
- **TX layout**: 32 slots × 4096 B at WQE_PAYLOAD_BASE = 0x9000 (ends at 0x29000); TX_BUF0/1 at 0x29000/0x2A000.

### Planned (in dependency order)

| # | Deliverable | Effort | Memory hook |
|---|---|---|---|
| **P5** | Host `RxWqeRing` (mirror of TX WQE-ring) for SEND/SEND_IMM/WRITE_IMM completion delivery to host hugepage via NoC→PCIe DMA-push | 2-3 days | TBD |
| **P6** | READ_REQ / READ_RESP opcodes with FW-side request-correlation table | 2 days | TBD |
| **P7** | Host `register_mr` API + CONTROL/CTRL_REG_MR FW handler (replaces hardcoded MR slot 0) | 1 day | TBD |
| **P8** | End-to-end PFC validation (Mellanox `mlnx_qos` + VLAN PCP=3 tagging in DPDK app) | requires partner config | `project_phase_a_pfc_fw_fix.md` (FW side done) |
| **P9** | Multi-queue TX (TXQ0+TXQ1 parallel) — pushes TX past 20.84 Gbps | 2 days | TBD |
| **P10** | ATOMIC_CAS / FAA opcodes + READ-side ARQ | 3 days | TBD |
| **(rig)** | Mellanox SR-IOV + switchdev — requires `mlxconfig SRIOV_EN=1` + reboot | ½ day after rig config | `project_switchdev_blocked_sriov.md` |

P5–P7 finish the production WRITE/SEND/READ surface. P8–P10 are throughput + completeness polish. Switchdev unblocks host-side RDMA past the 5–6 Gbps eSwitch ceiling.

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
