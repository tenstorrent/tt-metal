# Plan A — TT-RDMA Lossless Wire Layer via PFC

## Recommendation up front

Run TT-RDMA traffic in 802.1Q VLAN (VID 100) with PCP=3. Enable PFC on priority 3 on both Mellanox and WH CMAC. Simplest path on ConnectX-5 (no RoCE/IP/DSCP plumbing needed). DPDK app needs one-line change: add VLAN tag insertion. CMAC pause-honoring is OFF by default — needs two register writes to enable.

## 1. Mellanox side — enable PFC on priority 3

Confirm driver:
```
ethtool -i enp2s0np0          # driver: mlx5_core
modinfo mlx5_core | head -3
```

Enable PFC (works with MLNX_OFED or upstream):
```
sudo mlnx_qos -i enp2s0np0 --pfc 0,0,0,1,0,0,0,0 --trust dscp
```

Alternative on newer iproute2:
```
sudo dcb pfc set dev enp2s0np0 prio-pfc 0:off 1:off 2:off 3:on 4:off 5:off 6:off 7:off
```

Map priority 3 to dedicated TC:
```
sudo mlnx_qos -i enp2s0np0 --prio_tc 0,0,0,3,0,0,0,0 --tcbw 0,0,0,100,0,0,0,0
```

Verify:
```
sudo mlnx_qos -i enp2s0np0          # PFC line: "0 0 0 1 0 0 0 0"
sudo dcb pfc show dev enp2s0np0
```

Counters under load:
```
ethtool -S enp2s0np0 | grep -E 'tx_pause|rx_pause|prio3_(tx|rx)_pause|prio3_buffer|tx_pause_storm'
```

Non-zero `tx_prio3_pause` under load = PFC working from NIC end.

Rollback (single command):
```
sudo mlnx_qos -i enp2s0np0 --pfc 0,0,0,0,0,0,0,0 --prio_tc 0,0,0,0,0,0,0,0
```

## 2. Map TT-RDMA traffic to priority 3

**Option 1 (recommended): VLAN + PCP=3.** Tag every TT-RDMA frame with 802.1Q VID 100, PCP=3.

- ConnectX-5 ingress maps PCP→prio when `trust pcp` set
- DPDK side: `mbuf->ol_flags |= RTE_MBUF_F_TX_VLAN; mbuf->vlan_tci = (3<<13)|100;` and enable `RTE_ETH_TX_OFFLOAD_VLAN_INSERT`
- CMAC side: build 802.1Q tag in L1 payload buffer the gw kernel hands to `eth_send_packet`. Frame becomes `[DA][SA][0x8100][TCI][0x1AF4][payload][FCS]`. `ETH_TXQ_ETH_TYPE=0x8100`; 802.1Q TCI + inner ethertype become first 4 payload bytes.

**Option 2: DSCP.** Requires UDP/IP wrap — significant FW work. Skip unless RoCEv2 destination.

**Option 3: Egress-port-default priority.** Only works on RX. Skip.

## 3. WH CMAC side — pause-frame config

DWC XLGMAC at 0x70 (MAC_PRI0_TX_FLOW_CTRL) and 0x90 (MAC_RX_FLOW_CTRL):
- `MAC_RX_FLOW_CTRL` bit 0 = RFE (honor incoming pause); bit 1 = UP (unicast pause); bit 8 = PFCE (PFC vs legacy 802.3x)
- `MAC_PRI0_TX_FLOW_CTRL`…`MAC_PRI7_TX_FLOW_CTRL` (0x70, 0x74, …0x8C) per-priority. Bit 1 = TFE; bits [31:16] = PT (pause time)

**Current state:** `eth_cmac_init.cpp:191-231 (mac_tx_rx_enable)` does NOT touch 0x70 or 0x90. **CMAC ignores incoming pause frames (RFE=0) and generates none (TFE=0).** Mellanox PFC will fire but WH won't throttle. Must fix.

Required additions in `mac_tx_rx_enable()`:
```c
// Honor incoming PFC pause frames
eth_mac_reg_write(MAC_RX_FLOW_CTRL,
    (1u << 0) |   // RFE = 1
    (1u << 8));   // PFCE = 1 (PFC mode)

// Generate PFC pause for priority 3 when RX FIFO fills
eth_mac_reg_write(MAC_PRI0_TX_FLOW_CTRL + (3 * 4),
    (0xFFFFu << 16) | (1u << 1));

// Arm PFC mode in Q0
eth_mac_reg_write(MAC_PRI0_TX_FLOW_CTRL,
    eth_mac_reg_read(MAC_PRI0_TX_FLOW_CTRL) | (1u << 8));
```

Also add TX pause stat in `main_cmac.cc:155-168`:
```c
test_results->rxstatus[14] = eth_mac_reg_read(0x894);  // TX pause frames
```

## 4. Validation plan

```bash
# 1. Reload erisc_cmac_simple FW with PFC enable writes.
# 2. Host:
sudo ./tests/tt_metal/llrt/setup_hugepages_mlx_smoke.sh
sudo mlnx_qos -i enp2s0np0 --pfc 0,0,0,1,0,0,0,0 --trust pcp
sudo mlnx_qos -i enp2s0np0 --prio_tc 0,0,0,3,0,0,0,0
sudo ip link set enp2s0np0 up

# 3. DPDK receiver with VLAN/PCP=3:
sudo /tmp/mlx_smoke_bin/test_external_cmac_smoke_mlx \
    --proc-type=primary -- --count 0 --ethertype 0x1AF4

# 4. WH TX soak at 100% line rate.

# 5. Observe:
watch -n1 "ethtool -S enp2s0np0 | grep -E 'rx_prio3_pause|tx_prio3_pause|rx_prio3_buffer_discard'"
watch -n1 "python3 tests/tt_metal/llrt/check_erisc_memory.py --print-rxstatus"
```

**Pass criteria, PFC on:**
- `rx_prio3_pause > 0` OR `tx_prio3_pause > 0` (at least one direction non-zero at 100% load)
- `rx_prio3_buffer_discard == 0`
- CMAC `rxstatus[13]` + `rxstatus[14]` > 0 if symmetric congestion
- Phase 3.2 retry counter doesn't climb

**Pass criteria, PFC off (control):**
- `rx_prio3_buffer_discard` climbs, retries fire on WH. Proves PFC is the variable.

## 5. Risks and mitigations

- **Pause storms / HOL blocking.** If WH RX stalls (FW bug, NoC backpressure), pauses MLX indefinitely, blocking ALL prio 3 traffic. Mitigation: PFC watchdog (`--pfc_storm`), keep prio 3 dedicated to TT-RDMA.
- **Config drift across reboots.** Volatile. Stage in `tests/tt_metal/llrt/setup_pfc_mlx_smoke.sh`; bake into `/etc/network/if-up.d/tt-rdma-pfc` only after validation.
- **Other traffic on prio 3.** Verify default qdisc isn't remarking: `tc -s qdisc show dev enp2s0np0`.
- **Pause frames not counted on CMAC side.** Add stat slot 0x894 (§3).
- **DPDK app currently does no VLAN insert.** Must add to `port_init` before any of this matters.
