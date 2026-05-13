# Plan B — Bypassing the Mellanox eSwitch for TX

## Recommendation up front

Try in this order:
1. **Switchdev mode** — lowest-risk, highest-information. Reuses existing DPDK binary; rollback is one `devlink` command.
2. **Full DPDK bind via vfio-pci** — only if switchdev doesn't unblock throughput. Loses kernel coexistence.
3. **FPGA TT-Link NIC takeover** — last resort. Use only if both above plateau below 15 Gbps.

**Hypothesis:** Phase 3.2's 5.9 Gbps ceiling comes from CX-5's eSwitch slow-path on the legacy (non-switchdev) RX pipeline, where every flow-matched frame traverses the host-visible representor before landing in the DPDK queue. Switchdev moves that match into the eSwitch's offloaded fast-path (TC flower HW offload), and pure data-plane traffic bypasses the host CPU entirely.

## Path 1 — `devlink mode switchdev` (recommended first)

### Enable

```bash
sudo pkill -f test_external_cmac_smoke_mlx

# Allocate at least one VF (required for switchdev on CX-5).
echo 1 | sudo tee /sys/class/net/enp2s0np0/device/sriov_numvfs

# Unbind VF from mlx5_core so DPDK can grab it later.
VF_PCI=$(readlink /sys/class/net/enp2s0np0/device/virtfn0 | sed 's|.*/||')
echo "$VF_PCI" | sudo tee /sys/bus/pci/drivers/mlx5_core/unbind

# Switch PF eswitch to switchdev. PF must be DOWN first.
sudo ip link set enp2s0np0 down
sudo devlink dev eswitch set pci/0000:02:00.0 mode switchdev
sudo ip link set enp2s0np0 up

# Rebind VF.
echo "$VF_PCI" | sudo tee /sys/bus/pci/drivers/mlx5_core/bind
```

### Verify

```bash
sudo devlink dev eswitch show pci/0000:02:00.0     # expect "mode switchdev"
ip link show                                        # expect representor: enp2s0npf0vf0
ls /sys/class/net | grep -E '^en'
```

### DPDK app changes

In switchdev, PF netdev becomes slow-path; data-plane targets a VF. Update DPDK app to bind to VF PCI ID — `--allow 0000:02:00.1` (or whatever VF enumerates as). `port_init()` in `test_external_cmac_smoke_mlx.cpp:176-229` works unchanged; `rte_flow` rules installed by `install_flow()` now hardware-offloaded.

### Rollback

```bash
sudo pkill -f test_external_cmac_smoke_mlx
sudo ip link set enp2s0np0 down
sudo devlink dev eswitch set pci/0000:02:00.0 mode legacy
sudo ip link set enp2s0np0 up
echo 0 | sudo tee /sys/class/net/enp2s0np0/device/sriov_numvfs
```

### Expected throughput

Phase 3.2 bottlenecked at 5.9 Gbps. Removing eSwitch slow-path typically yields 60–80 Gbps RX on CX-5 (per mlx5 driver doc switchdev section). TT side won't feed that much, but host-side bottleneck should disappear; **if we still cap at ~6 Gbps after this, the bottleneck is on WH TX side, not the NIC** — itself a critical data point.

### Risks

- Switchdev requires kernel mlx5 driver with `CONFIG_MLX5_ESWITCH=y` (default in Ubuntu HWE). If `devlink` returns `Operation not supported`, fall back to path 2.
- Brief link flap during mode switch (any ssh on `enp2s0np0` IP affected).
- DPDK invocation must change from `--proc-type=primary` to `--allow $VF_PCI`.

## Path 2 — Full DPDK bind (vfio-pci)

### Enable

```bash
sudo pkill -f test_external_cmac_smoke_mlx
sudo ip link set enp2s0np0 down

echo "mlx5_core" > /tmp/tt-rdma-mlx-driver-backup
sudo modprobe vfio-pci
sudo /usr/share/dpdk/usertools/dpdk-devbind.py --bind=vfio-pci 0000:02:00.0
```

### Verify

```bash
sudo /usr/share/dpdk/usertools/dpdk-devbind.py --status | grep -A2 0000:02:00.0
# expect: drv=vfio-pci unused=mlx5_core
```

### Rollback

```bash
sudo /usr/share/dpdk/usertools/dpdk-devbind.py --bind=mlx5_core 0000:02:00.0
sudo ip link set enp2s0np0 up
```

### Expected throughput

CX-5 with mlx5 PMD in pure DPDK ownership reaches ~95 Gbps line-rate for 1500B frames. If switchdev got us to >25 Gbps, this is unnecessary.

### Risks

- Loses `ip`/`ethtool` on `enp2s0np0` — no management, no `mlnx_qos`, no PFC config via netdev. PFC must instead be configured via DPDK `rte_eth_dev_set_pfc_ctrl` (DPDK 22.11+). Couples Plan A and Plan B: if this route, Plan A's `mlnx_qos` stops working; need DPDK-native PFC enablement.
- Any rig user expecting kernel netdev loses it.
- Rollback ~30s but DPDK teardown isn't always clean (EAL hang on secondary).

## Path 3 — FPGA TT-Link NIC

The `83a98004b1 test(smoke): fix FPGA BAR path` commit references the existing `open-nic-shell-tt-link` FPGA setup. Use FPGA as dedicated TT-RDMA endpoint; leave Mellanox for management.

### Enable (sketch)

- Bring up FPGA's QDMA driver per existing test harness
- Re-target WH CMAC's destination MAC to FPGA's MAC
- Build new DPDK app (or vendor PMD shim) for QDMA path

### Verify

- Link up on FPGA; CMAC PCS lock against FPGA (may need different CTLE override per per-cable sweep in `eth_cmac_init.cpp:404-421`)
- DMA throughput baseline via FPGA loopback test

### Rollback

Physical: swap cable back. Software: re-target WH dst MAC; revert FPGA driver load.

### Expected throughput

Bounded by QDMA + PCIe Gen4 x16 (~25 GB/s = 200 Gbps host-side); wire 100 Gbps line rate. No eSwitch, no embedded firmware bottleneck. Best-case path to 25+ Gbps.

### Risks

- Bringing up FPGA-based NIC is days, not hours.
- Loses Linux netdev tooling entirely.
- Cable / SFP+ compatibility — per-lane CTLE sweep currently Mellanox-specific.

## Coordinated test plan (Plan A + Plan B together)

After (Plan B path 1) + (Plan A enabled):

```bash
# 1. Re-flash WH FW with PFC RX/TX enables in mac_tx_rx_enable.
# 2. Host setup:
sudo ./tests/tt_metal/llrt/setup_hugepages_mlx_smoke.sh
echo 1 | sudo tee /sys/class/net/enp2s0np0/device/sriov_numvfs
sudo ip link set enp2s0np0 down
sudo devlink dev eswitch set pci/0000:02:00.0 mode switchdev
sudo ip link set enp2s0np0 up
sudo mlnx_qos -i enp2s0np0 --pfc 0,0,0,1,0,0,0,0 --trust pcp
sudo mlnx_qos -i enp2s0np0 --prio_tc 0,0,0,3,0,0,0,0

# 3. Run Phase 3.2 TX soak, DPDK bound to VF, 5 minutes.
# 4. Capture:
#    - sustained TX Gbps (host counters + WH 0x814/0x818 octet regs)
#    - rx_prio3_buffer_discard (must be 0)
#    - rx_prio3_pause / tx_prio3_pause (must be > 0 under load)
#    - p99 latency from gw kernel timestamping
```

**Target: 15 Gbps sustained TX, 0 frame drops, non-zero pause counters.**

**If sustained TX is still ~6 Gbps after switchdev + PFC, the bottleneck is on WH TX side** (DMA-pull engine, NoC, or FW pacing) — not the NIC. Next investigation moves away from networking into TX-side firmware. This is the single most valuable signal the combined experiment produces.

---

### Critical Files

- `budabackend-master/src/firmware/riscv/targets/erisc_cmac_simple/src/api/eth_cmac_init.cpp` (PFC RX/TX enable in `mac_tx_rx_enable`)
- `budabackend-master/src/firmware/riscv/targets/erisc_cmac_simple/src/main_cmac.cc` (TX_PAUSE counter at 0x894)
- `budabackend-master/src/firmware/riscv/targets/erisc/src/api/dwc_xlgmac_regs.h` (`MAC_PRI0_TX_FLOW_CTRL`, `MAC_RX_FLOW_CTRL`)
- `tests/tt_metal/llrt/test_external_cmac_smoke_mlx.cpp` (`port_init` — VLAN insert offload + `--allow` for VF PCI; `install_flow` — VLAN match)
- `tests/tt_metal/llrt/setup_hugepages_mlx_smoke.sh` (template for `setup_pfc_mlx_smoke.sh` and `setup_switchdev_mlx_smoke.sh`)
