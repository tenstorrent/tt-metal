# TT-RDMA v1 — RISC-off-datapath TX Ring (Blackhole chip side)

Status: design draft (2026-07-23). Implements the §14 performance mandate of
`tt-rdma-bh-bf3-impl-plan.md`: **maximum BW + minimum latency, with the RISC-V eth cores and the host
CPU OFF the per-packet datapath.** This is the chip-side TX engine for the external (NIC/BF3) rail,
in *raw* Ethernet mode (the NoC-overlay tunnel + TT-link packet mode are TT-proprietary and cannot be
used to a non-TT peer — see plan §14).

Golden references: `../../BlackholeA0/EthernetTile/EthernetTxRx.md` (HW), `tt-rdma-wire-protocol-v1.md`
(frame), `tt_metal/hw/inc/internal/ethernet/tt_rdma_l1_layout.h` (L1 map), and the measured baseline
(43 Gbps aggregate raw @4 KB; the ceiling was RISC-in-the-loop, not the wire).

---

## 1. Principle: the RISC arms, hardware moves

The naive loop had the RISC-V core in the per-frame path (build header → write TXQ regs → busy-wait
TX drain → repeat), capping at the RISC command rate (~650k cmd/s/rail). The line-rate design removes
the RISC from the per-*packet* path entirely, using three HW levers that exist in raw mode:

1. **`MAX_PKT` auto-split** (`ETH_TXQ_MAX_PKT_SIZE_BYTES` @ `TXQ+0x0C`): one TX command over a large L1
   region is split by hardware into many MTU frames. So the RISC arms **once per message**, not once
   per frame — message payloads of many KB become one arm.
2. **Accept-ahead, not drain-wait** (`ETH_TXQ_STATUS.CMD_ONGOING` @ `TXQ+0x08`): `CMD_ONGOING` clears
   when HW has *accepted* the command (read the descriptor / started reading L1), **not** when the
   frame has drained to the wire. The fast path polls for *accept* and immediately arms the next —
   the wire drains in the background. It never waits for completion.
3. **Zero-copy payload**: the RISC never copies payload bytes. Payload arrives in L1 by DMA — host
   PCIe write, or NoC pull from Tensix/DRAM/another chip — placed there by the *producer*. The RISC
   only reads a doorbell and writes 3 TXQ registers.

Result: RISC work per message = O(1) tiny (a few register writes), independent of message size; BW is
bounded by the wire + DMA, not by the RISC.

## 2. Actors

| Actor | Role | On the per-packet datapath? |
|---|---|---|
| **Producer** (host over PCIe, or a Tensix/DRAM NoC source, or the BF3 gateway via DMA) | writes frames into the payload ring + descriptors into the WQE ring; rings a doorbell | No (bulk DMA only) |
| **RISC1 (subordinate erisc)** | drains the WQE ring: arm TXQ (3 regs), advance consumer, pipeline accept-ahead; manage completions | **Control only** — arms, never copies, never drains |
| **ETH TXQ hardware** | prepend L2 (TX header table), auto-split at `MAX_PKT`, serialize to the wire | Yes (this is the datapath) |
| **RISC0 (active erisc)** | unchanged: base-FW link maintenance; RISC1 must keep yielding (BH.0 coexistence) | No |

## 3. L1 data structures (in the RDMA region, `tt_rdma_l1_layout.h`)

All within `[TT_RDMA_L1_BASE .. TT_RDMA_L1_END)`, clear of `0x70000+` (base FW) and the `0x40000`
reset save.

- **WQE descriptor ring** — `TT_RDMA_WQE_DESCR_ADDR`, 64 × 16 B. Each descriptor (one **message**):
  ```
  +0  u32  frame_l1_off   // offset (from TT_RDMA_WQE_PAYLOAD_ADDR) of the [32B hdr][payload] frame
  +4  u32  frame_len      // total bytes to transmit (header + payload); HW splits at MAX_PKT
  +8  u32  flags_txq      // bit0..1 = TXQ index (round-robin 0..2), bit8 = OWNED_BY_FW (producer sets last)
  +12 u32  cookie         // opaque (completion correlation / seq echo)
  ```
- **Payload ring** — `TT_RDMA_WQE_PAYLOAD_ADDR`, 32 × 4096 B (128 KB). The producer writes each
  message's full frame (`tt_rdma_hdr_t` + payload) here via DMA. For messages > 4 KB, use a larger
  contiguous carve (the slot stride is a config knob; payload may span multiple slots).
- **Ring control block (RCB)** — `TT_RDMA_RCB_ADDR` (doorbells + indices, all u32):
  ```
  +0x00 hb              // RISC1 heartbeat / frames-armed counter (observability)
  +0x04 stop            // host->RISC1 graceful-stop doorbell (BH.0 lifecycle)
  +0x10 prod_idx        // producer's next-free WQE index (producer writes; RISC1 reads) = the doorbell
  +0x14 cons_idx        // RISC1's next-to-arm WQE index (RISC1 writes; producer reads for backpressure)
  +0x18 compl_idx       // RISC1's last-completed WQE index (RISC1 writes; producer reclaims slots)
  +0x1C txq_inflight[3] // per-TXQ outstanding count (RISC1 internal; optional telemetry)
  ```

The ring is SPSC (single producer, single consumer) — lock-free via the index words + a release
fence on `OWNED_BY_FW`.

## 4. RISC1 fast path (control only — never copies, never drains)

```c
// one-time setup
for q in {0,1,2}: tt_rdma_set_max_pkt_size(q, MAX_PKT);          // e.g. 4080 (jumbo) or link MTU
configure TX header table row per TXQ: DA=BF3, SA, ethertype 0x1AF6, INSERT_CTL=0  // §wire
uint32_t next_q = 0;

// steady state — NO busy-wait for drain, NO payload copy
for (;;) {
    uint32_t prod = RCB.prod_idx;                 // doorbell (producer's DMA'd frames are ready)
    while (RCB.cons_idx != prod) {
        wqe = WQE_DESCR[cons_idx & MASK];
        if (!(wqe.flags_txq & OWNED_BY_FW)) break; // release-fence guard: frame not fully written yet
        uint32_t q = wqe.flags_txq & 3;            // (or next_q round-robin for balancing)

        // pipeline: only wait until the queue ACCEPTED the previous arm (fast), not drained
        while (TT_ETH_REG32(TXQ(q)+STATUS) & CMD_ONGOING) { /* brief; wire drains in background */ }
        tt_rdma_send_raw(q, WQE_PAYLOAD_ADDR + wqe.frame_l1_off, wqe.frame_len);  // 3 reg writes, fires

        RCB.cons_idx = ++cons_idx;                 // reclaim-visible to producer
        RCB.compl_idx = cons_idx;                  // arm==complete for fire-and-forget; see §7 for ACKed
        RCB.hb = ++armed;
    }
    if (RCB.stop) break;                           // graceful stop (BH.0)
    // if empty: light pause or WFI on a doorbell interrupt (see §8 latency) — no spin-burn
}
```

Per message the RISC does: 1 ring read + 1 status poll + 3 register writes + 2 index writes. **No
data copy. No completion wait.** Message size is irrelevant to RISC cost — a 4 KB or 64 KB message is
the same ~handful of instructions, and `MAX_PKT` fans it out to many wire frames in hardware.

## 5. Getting payload into L1 without the RISC (the producer side)

The RISC never generates or copies payload. Options for the producer, in priority order:

1. **Host DMA-pull / PCIe write** (gateway inbound): the BF3/host writes frames straight into the
   payload ring over PCIe (BAR/hugepage), rings `prod_idx`. This is the shipped-v1 `ExternalIfaceSender`
   model — reuse it. Zero chip-CPU involvement.
2. **NoC pull from Tensix/DRAM** (chip-origin traffic): a NoC async read (`noc_async_read`) issued
   *once per message* streams the payload from DRAM/Tensix L1 into the payload ring; the RISC issues
   the NoC read (control) but the NoC engine moves the bytes. Overlap read(N+1) with TX(N).
3. **Header build**: the 32 B `tt_rdma_hdr_t` is small; the producer (host/DPA) builds it, OR the RISC
   stamps only the header (32 B) into a pre-DMA'd payload — still O(32 B), not O(payload).

## 6. Parallelism — scale out, not up

- **3 TXQ per tile**: round-robin messages across all three TX queues so three transfers are
  in-flight per eth tile; `MAX_PKT` auto-split keeps each queue busy.
- **14 ETH SS**: the gateway rail is one tile, but the design generalizes — one WQE ring + RISC1
  drainer per active external tile, driven by a per-tile producer. Aggregate BW scales with tiles.
- **Jumbo frames**: larger `MAX_PKT` → fewer per-frame HW overheads. Bounded by peer MTU / lossless
  config; validate the large-frame framing (see §9).

## 7. Reliability (raw mode has no HW resend)

Raw mode has no sequence-number/resend (that's TT-link, TT-only). Layered, off the per-packet CPU path:

- **PFC lossless** on the direct-attach link → no drops in steady state (primary mechanism).
- **Lightweight ARQ** for the rare loss: carried in the `tt_rdma_hdr_t` `seq` field + `REQ_ACK`; the
  **BF3 DPA** tracks per-flow seq and requests retransmit — the retransmit is just re-arming the WQE
  (the frame is still in the ring). The chip RISC does not run a per-packet reliability state machine.
- `compl_idx` only advances past a WQE once it is ACK-eligible to be reclaimed (fire-and-forget:
  immediately; reliable: after DPA ACK covers its seq).

## 8. Latency path (small messages)

For latency-sensitive small messages, bypass batching: the producer writes one WQE + rings the
doorbell; the RISC arms immediately. To avoid spin-polling burn while idle, the RISC1 drainer can
**WFI on a doorbell** (host writes `prod_idx` → wake) rather than busy-poll — sub-µs arm latency with
no idle power cost (matches the "no busy-poll" note from the BH.0 power discussion). End-to-end small
-message latency = doorbell + 3 reg writes + MAC serialize + wire + BF3 RX — no handshake, no copy.

## 9. Open questions / validation gates

- **Big-transfer `STATUS` semantics.** A first burst attempt (one large `START_RAW`, `MAX_PKT` split)
  regressed to 1.7 Gbps with tiny frames — the RISC busy-waited the big command and `MAX_PKT` didn't
  fan out as expected. **Gate BH.2 on:** confirm `CMD_ONGOING` clears on *accept* for a large raw
  transfer, confirm `MAX_PKT` split granularity, and confirm each split frame carries the TX-header
  L2 correctly. Measure arm-rate vs frame-rate to prove the RISC is off the per-frame path.
- **Large-frame framing / MTU.** The 4 KB "fragmentation" seen earlier is `MAX_PKT` auto-split working;
  confirm the BF3/DPA reassembles split frames of one message correctly (one TT-RDMA header at the
  start, payload chunks following).
- **Ring backpressure.** Producer must honor `cons_idx`/`compl_idx` to not overwrite in-flight slots.
- **TXQ vs base FW.** Use TXQ ≥ 1 (base FW owns TXQ0 for link maintenance on TT rails; on the pure
  NIC rail TXQ0 may be free, but keep the RDMA queues separate for safety).

## 10. Milestones

- **BH.2a** — single-TXQ WQE ring: producer writes N frames + doorbell; RISC1 drains with accept-ahead
  (no drain-wait); measure arm-rate and BW vs the per-frame baseline. Gate on §9 big-transfer check.
- **BH.2b** — 3-TXQ round-robin + `MAX_PKT` auto-split large messages; target ≫ 43 Gbps/rail.
- **BH.2c** — NoC-pull producer (chip-origin) + host-DMA producer (gateway inbound, reuse
  `ExternalIfaceSender`); confirm zero RISC payload copies.
- **BH.2d** — WFI doorbell latency path; measure small-message RTT.
- Reliability (PFC + DPA ARQ) and multi-rail fold in with the gateway (G.x) milestones.

---

## §11. BH.2a implementation results (2026-07-23, on silicon)

First cut of the ring drainer (`bh_rdma_tx_ring.cpp` + `bh1_tx_ring`). Findings:

- **Arm-rate win confirmed.** Removing the per-frame header-build/CRC/copy, the RISC arm-rate hits
  **~1.75 M arms/s** (unpaced) vs the M-1a probe's ~650k/s — ~2.7×. The RISC-arms-only design works.
- **Raw `START_RAW` has no deep accept-ahead → over-arming WEDGES the TXQ.** Unpaced, `CMD_ONGOING`
  eventually sticks and the arm loop freezes (host `Finish` hangs; needs a board reset). The M-1a
  probe never wedged only because its per-frame CRC/copy *accidentally paced* it. **Fix: pace the
  ring drainer** (added `pace` arg); paced (`pace≳500`, ~333k/s) runs stable, no wedge, clean stop.
- **RESOLVED (2026-07-23) — the "0 wire bytes" symptom was a large-single-raw-frame egress ceiling,
  NOT a ring/zero-copy defect.** Adding the ISA TXQ counters (`ETH_TXQ_PKT_START_CNT` `TXQ+0x34`,
  `PKT_END_CNT` `TXQ+0x3C`, `WORD_CNT` `TXQ+0x40`) as an on-core before/after snapshot (read back over
  the RCB debug region) flipped the diagnosis. Measured on TT dev1 ext rail idx2 → BF3 pciconf1, with
  `ethtool -S <ttport> rx_bytes_phy` as the wire cross-check:
  - **4080 B frame, `max_pkt` = 4080 or 0:** `PKT_START`/`PKT_END` advance == armed (1.66M) and
    `WORD_CNT` advances, **yet BF3 wire rx_bytes = 0.** So the counters count *internal command
    processing, not PHY egress* — which is exactly what made the accepted-`CMD`/advancing-arm-counter
    look like success in the original investigation. A single raw transfer at/above ~4080 B does not
    egress the BH Rianta MAC.
  - **64 B frame, no split:** wire rx_pkts == armed, rx_bytes correct (82 B/frame incl. L2+FCS). **The
    descriptor-driven, zero-copy, no-RISC-touch ring transmits fine.** `risc_touch` (RISC writes the
    source before each arm) was a red herring — not needed, and `risc_touch=1` actually hangs (its
    `send_raw` busy-waits a stuck `CMD_ONGOING`).
  - **4080 B frame, `max_pkt` = 1024 (auto-split):** ~4 wire frames per arm, **4.15 GB in 3 s
    (~11 Gbps *paced*), confirmed at the BF3 PHY.** MAX_PKT auto-split — the §1 mechanism — works and
    is the fix.
  **Consequence:** the RISC-off-datapath / zero-copy premise is validated end-to-end. The rule is
  **always set `MAX_PKT` <= the working egress ceiling**; arming a single frame above it silently
  drops.
- **Egress ceiling — the 1518 B wall was the BF3 NIC RX MTU (1500), NOT the BH side.** The binary
  search first found single wire frames > 1518 B blocked, but that was the BlueField's default
  MTU 1500 silently dropping oversize frames at PHY ingress — mlx5 drops them *before* `rx_bytes_phy`
  (they'd surface only in `rx_oversize_pkts_phy`), which is why the wire byte-delta read 0 and misled
  the diagnosis. **The BH Rianta MAC is already configured for 4096 B frames:** `eth_init.cpp:500/505`
  set `MAC_RS_TX/RX_SLAVE.*_max_pkt_len = 4204` (4096 + 104 B packet-mode header + 4). Proof: after
  `ip link set dev <ttport> mtu 9000`, a single 4080 B frame (no split) egresses — wire rx_bytes
  +2.73 GB, `rx_oversize` delta 0. **So BH does jumbo with NO FW change** — just raise the BF3 tt-port
  MTU (`ip link set mtu 9000`; not persistent across reboot → bake into a setup script).
  **Recommended `MAX_PKT ≈ 4080`** for single-frame jumbo (far more efficient than 1500-chunk split);
  BH single-frame TX ceiling ≈ the MAC's 4200, BF3 RX now 9000.
- **Register-stickiness gotcha:** `tt_rdma_set_max_pkt_size` writes `ETH_TXQ_MAX_PKT_SIZE_BYTES` only
  when its arg != 0, so `MAX_PKT` **persists across kernel runs** — a run passing `max_pkt=0` inherits
  the previous run's value. The drainer must **always set `MAX_PKT` explicitly** (<= 1500); never rely
  on 0/default.
- **LINE RATE ACHIEVED — one rail, one TXQ, paced: 198.8 Gbps wire / 197.4 Gbps goodput (~200 G).**
  Both rails at MTU 9000, arming 16384 B messages at `MAX_PKT` 4080 (5 jumbo frames/arm), pace=100.
  The decisive measurement: **arm rate held at ~1.5 M msg/s independent of message size** (4 KB → 16 KB
  dropped only 1.577 → 1.509 M/s). This resolves the §9 open question — **`CMD_ONGOING` clears on
  *accept*, not drain**: the RISC arms at a fixed ~1.5 M/s regardless of transfer size, the HW does all
  per-frame work, and wire BW = arm_rate × message_size. The RISC-off-datapath premise (§1) is proven
  on silicon: bigger messages scale straight to line rate at constant RISC cost. (At equal paced arm
  rate a 4 KB message gave identical ~51 Gbps goodput for jumbo vs 1500-split — jumbo just uses 1/3 the
  wire frames; the throughput lever is message size per arm, not chunk size.)
- **Remaining follow-ups:** (a) persist the BF3 tt-port MTU 9000 (`tt_metal/tt_rdma/bh0/set_tt_rail_mtu.sh`,
  both rails; not reboot-persistent); (b) 3-TXQ (BH.2b) + 2nd rail for aggregate > 200 G; (c) the `pace`
  is now just a safety throttle — with accept-ahead confirmed, characterize the unpaced max and whether
  the old over-arming wedge is gone now that TX actually egresses.
- **CAUTION logged:** an early diagnostic touched `payload_base + 128 KB`, which from a high base
  overruns past `0x70000` into the link-bricking base-FW region. Any payload staging must stay within
  the RDMA L1 window; validate `TT_RDMA_L1_BASE`/size vs `MEM_SYSENG_RESERVED_BASE`.
