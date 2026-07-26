# TT-RDMA v1 — DOCA HW-TX Gateway Sender (BF3 → BH)

Status: working on silicon (2026-07-24). The first BF3 DOCA building block (impl-plan G.1/G.2's
`doca_eth_endpoint.c`): a **hardware-TX** raw-L2 sender that emits TT-RDMA-v1 `0x1AF6` frames to the
Blackhole at near line rate — replacing the userspace `tt_rdma_bf3_send`, which is eSwitch/CPU-bound.
It is both the true-ceiling-finder for the BH RX and the gateway's TX leg.

## Why DOCA HW-TX

| Sender | Rate | Limit |
|---|---|---|
| host userspace `sendmmsg` | ~11 Gbps | BlueField eSwitch slow path |
| DPU-Arm userspace `sendmmsg` on `p0` | ~16.7 Gbps | Arm CPU |
| **DOCA `doca_eth_txq` HW-TX (pipelined) on `mlx5_0`** | **~143 Gbps (PHY-measured), near line rate** | the BH RX, not the sender |

The DOCA path posts send-task *batches* to the NIC's HW TX rings — the Arm never touches per-frame
egress. Pipelined (multiple batches in flight) it saturates the 200 G uplink.

## Deploy + build (one command)

The source is vendored in the repo (`tt_metal/tt_rdma/gw/ttblast_sample.c`, the NVIDIA BSD-3 sample +
TT-RDMA mods) so it survives DPU reboots (the DPU's `/tmp` is wiped). Deploy from the host:

```sh
tt_metal/tt_rdma/gw/deploy_doca_sender.sh          # scp source + build -> /tmp/doca_ttblast on the DPU
tt_metal/tt_rdma/gw/deploy_doca_sender.sh --run    # ... then run (mlx5_0 -> BH ext idx2, dst 02:..:02)
```
It ensures the host↔DPU tmfifo IP, scp's `ttblast_sample.c`, and builds it on the BlueField Arm against
the DPU's stock DOCA sample sources. Env overrides: `DPU`, `DPU_PASS`, `DEV`, `DMAC`.

## Build (manual, on the BlueField Arm — DOCA 3.4, no meson needed)

Base sample: `/opt/mellanox/doca/samples/doca_eth/eth_txq_batch_send_ethernet_frames`. Build the
(modified) sample with gcc + pkg-config (meson is not required):

```sh
export PKG_CONFIG_PATH=/opt/mellanox/doca/lib/aarch64-linux-gnu/pkgconfig
S=/opt/mellanox/doca/samples/doca_eth/eth_txq_batch_send_ethernet_frames
CM=/opt/mellanox/doca/samples/doca_eth
gcc -O2 -w -I$S -I$CM -I/opt/mellanox/doca/samples \
  $(pkg-config --cflags doca-eth doca-common doca-argp doca-flow) \
  ttblast_sample.c $S/eth_txq_batch_send_ethernet_frames_main.c \
  $CM/eth_common.c $CM/eth_flow_common.c /opt/mellanox/doca/samples/common.c \
  $(pkg-config --libs doca-eth doca-common doca-argp doca-flow) -lpthread -o /tmp/doca_ttblast
```

## Modifications to `..._sample.c` (turn the demo into a line-rate TT-RDMA blaster)

Minimal-diff against the shipped sample (do not vendor the NVIDIA source; re-derive these edits):

1. **Defines:** `TASKS_IN_TASK_BATCH 32→64`; `REGULAR_PKT_SIZE 1500→4126` (14 L2 + 32 TT hdr + 4080
   jumbo payload); `TASK_BATCHES_NUM 1→4` (pipeline depth; 4×64 = 256 = `MAX_BURST_SIZE`).
2. **Frame builder** (`create_eth_txq_packet_buffers`): set `ether_type = htobe16(0x1AF6)`, then write a
   32 B little-endian `tt_rdma_hdr_t` at the payload — opcode `0x10` (WRITE), ver 1, `length = 4080`,
   `rkey = 0x00CAFE42`, `remote_offset/imm/cksum = 0` — then the payload ("TTWR" + fill).
3. **Send callback:** drop the per-packet logging and the `doca_buf_dec_refcount` — keep the bufs alive
   so they are re-submitted every iteration without re-filling (frames are constant). Just decrement
   `inflight_task_batches`.
4. **Run loop:** replace the one-shot submit with a pipelined loop —
   ```c
   for (iter = 0; iter < NUM_SEND_ITERS; iter++) {
       while (state.inflight_task_batches >= TASK_BATCHES_NUM)     // throttle to pipeline depth,
           (void)doca_pe_progress(state.core_resources.core_objs.pe);  // busy-poll (no sleep)
       create_eth_txq_task_batch(&state);   // same pre-filled bufs
       submit_eth_txq_task_batch(&state);   // HW TX; inflight++
   }
   while (state.inflight_task_batches != 0) doca_pe_progress(...);  // final drain
   ```
   Free the bufs at the end (the callback no longer does). NB the *non*-pipelined form (drain each
   batch fully before the next, with the sample's 10 µs `nanosleep`) only reaches ~3.5 Gbps — the
   pipeline (busy-poll, N in flight) is what gets to line rate.

## Run

```sh
# on the BlueField Arm; mlx5_0 = uplink port0 -> BH ext rail idx2 (mlx5_1 = port1 -> idx5)
sudo /tmp/doca_ttblast -d mlx5_0 -m 02:00:00:00:00:02   # dst unicast -> BH RXQ2
```

Prereqs (redo after any reboot): BF3↔BH link trained (forced 200G); DPU-side MTU 9000 on
`p0/p1/pf0hpf/pf1hpf`; host PF MTU 9000. See `tt-rdma-rx-dispatch-spec.md` §8 and the port-mapping notes.

## Result (validated on silicon)

- **~143 Gbps wire** (PHY `tx_bytes_phy`), 19.2 M jumbo frames — near the 200 G line rate.
- End-to-end **byte-exact**: DOCA HW-TX → wire → BH RXQ2 (raw) → dispatch → MR lookup → `noc_async_write`
  → Tensix L1 (`52575454` = "TTWR" + payload). The full inbound WRITE path at line-rate ingress.
- **Confirms the BH RX ceiling (~15.8 Gbps/rail) is the binding limit**, not the sender: at 143 Gbps in,
  the RX is swamped and resync-thrashes. Closing the 16 → 143 Gbps gap is RX-side work (per-frame parse
  offload / NoC-descriptor coalescing / multi-rail / PFC), per `tt-rdma-rx-dispatch-spec.md` §8.

## Next (toward the real gateway)

This is the TX *leg* only — it emits a fixed WRITE frame. The gateway (`bf3-gateway-design.md`,
`tt-rdma-dpa-gateway-spec.md`) adds the WAN side: terminate a RoCEv2 RC QP (`doca_rdma`), translate
BTH/RETH → the 32 B TT header per packet (Arm T1 → zero-copy T2 → DPA T3), and the MR/QP/PSN↔seq
tables. This sender proves the tt-side `doca_eth` HW-TX path it will feed.
