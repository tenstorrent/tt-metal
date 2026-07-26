# TT-RDMA v1 — RX Dispatch (Blackhole chip side)

Status: validated on silicon (2026-07-24). The receive counterpart to `tt-rdma-tx-ring-spec.md`.
Implements BH.2-RX + the core of BH.3 (MR table + WRITE): inbound TT-RDMA-v1 frames from a NIC/gateway
(BF3) are received, parsed, dispatched by opcode, and WRITE payloads land at their MR target — with the
RISC-V core OFF the per-byte datapath (the §14 performance mandate of `tt-rdma-bh-bf3-impl-plan.md`,
applied to RX).

Golden references: `tt-rdma-wire-protocol-v1.md` (frame), `bh-erisc` `docs/eth_arch_spec.md` (ETH SS
RX), `tt_metal/hw/inc/internal/ethernet/tt_rdma_wire.h` / `tt_rdma_l1_layout.h` / `tt_rdma_eth_rx.h`.
Code: `tt_metal/tt_rdma/bh0/kernels/bh_rdma_rx_dispatch.cpp` + `bh1_rx_dispatch_host.cpp`; BF3 sender
`tt_rdma_bf3_send.c`.

---

## 1. Principle: everything is a PUSH; the RISC routes, the NoC moves the bytes

The inbound path is push end-to-end — nobody pulls across the wire:

1. **BF3 → wire:** the NIC/gateway *transmits* (pushes) raw `0x1AF6` frames. This is inherent to RDMA
   WRITE/SEND — the initiator drives the transfer. (Even READ is push-of-REQ + push-of-RESP; there is
   no wire-level pull in this protocol.)
2. **wire → L1:** the Rianta MAC RX engine *pushes* received frames into an L1 ring (RXQ raw mode,
   BUF_WRAP). The RISC never pulls from the wire — it reads the write pointer and walks what landed.
3. **L1 → MR target:** the eth core is the NoC *initiator* and *pushes* the payload to the MR's
   `base_noc_addr` via `noc_async_write` (Tensix L1 / DRAM / another chip). The RISC issues one
   descriptor per message; the NoC engine moves the bytes.

So the RISC does O(1) work per frame — parse the 32 B header, look up the rkey, issue one NoC-write
descriptor — and **never copies a payload byte**. This is the RX analog of the TX ring's "RISC arms,
HW moves." Consequence of push: **no inherent back-pressure to the sender** — the BH absorbs bursts
with a deep ring and throttles the sender only via out-of-band PFC (BH.6), never by pulling.

## 2. Actors

| Actor | Role | On the per-byte datapath? |
|---|---|---|
| **BF3 / gateway** | pushes `0x1AF6` frames onto the wire (RoCE→TT translation, or the userspace `tt_rdma_bf3_send` today) | Yes (it is the source) |
| **Rianta MAC RX + RXQ2** | cut-through receive; pushes L2-stripped frames into the L1 ring; advances BUF_PTR | Yes (this is the ingress datapath) |
| **RISC1 (subordinate erisc)** | walk ring → parse header → opcode dispatch → rkey lookup → issue `noc_async_write` | **Control/routing only** — parses headers, never copies payload |
| **NoC engine** | moves the payload from the L1 ring to the MR target off-core | Yes (this is the landing datapath) |
| **RISC0 (active erisc)** | unchanged base-FW yield (BH.0 coexistence) | No |

## 3. RX mechanism (BH ETH SS specifics)

- **No HW ethertype classifier.** `eth_rx_flow.cpp` (the TCAM/flow-director that could match `0x1AF6`)
  is dead/uncompiled in `bh-erisc`. So the `blackhole-port.md` "classifier TCAM → landing region"
  premise is **wrong**. Reception uses the base-FW **dst-MAC router** instead (`eth_init.cpp:266-280`):
  broadcast→RXQ0, multicast→RXQ1, **unicast/"other"→RXQ2**. A unicast `0x1AF6` frame lands on RXQ2,
  which the base FW never drains (it only reads RXQ2's drop counter for link telemetry) — so RXQ2 is
  free for this kernel. The MAC is **accept-all** (no dst-address filter, no station address
  programmed), so the sender may use any unicast dst MAC.
- **Raw mode, kernel-configured.** Base FW leaves all RXQ in *packet* mode; the kernel reconfigures
  RXQ2 to **raw** (`ETH_RXQ_CTRL`: packet_mode = bit 1 / 0x2, **buf_wrap = bit 2 / 0x4** — NB not bit
  1). Raw mode strips the 14 B L2, so frames land as `[32 B tt_rdma_hdr][payload]`, contiguous — the
  opcode is at `rxbuf[0]`. RXQ block base `0xFFB94000`; `BUF_PTR` @ +0x08 is in **bytes** (it pegs at
  the ring size, not ring-size words); `BUF_START_WORD_ADDR` @ +0x0C is in 16 B words;
  `BUF_SIZE_WORDS` @ +0x10.
- **MAC RX max frame = 4204** (`eth_init.cpp:505`, committed base FW) → jumbo RX up to ~4200 B needs
  **no FW change**; only the sender's egress MTU matters (a DPU-side config on the BF3, not the BH).

## 4. L1 data structures (`tt_rdma_l1_layout.h`)

- **RX ring** — a large BUF_WRAP streaming ring. Default is `TT_RDMA_RX_RING_BIG` = **128 KB**, which
  reuses the (TX-only) WQE payload region since the RX-dispatch kernel never transmits. The 16 KB
  `TT_RDMA_RX_RING` (~4 jumbo frames) laps at ~9 Gbps and is retained only for small tests. Size must
  be a multiple of 16 so every wrap-straddling header word / frame stride stays word-aligned.
- **MR table** — `TT_RDMA_MR_TABLE_ADDR`, 64 × 32 B (`tt_rdma_mr_entry_t`). `rkey = (slot<<24) |
  (rand16<<8) | gen`; O(1) lookup on `rkey>>24`. Validation per WRITE: `rkey` match, `REMOTE_WRITE`
  access flag, `remote_offset + len ≤ mr.length` → else drop.
- **Stats** — 8 u32 in the RCB dbg region: `total, send, write, write_ok, unknown, bad, last_op,
  read_pos` (host observability).

## 5. The dispatch loop (RISC1, control/routing only)

```c
tt_rdma_rxq_init(RXQ2, ring, ring_size, /*wrap=*/1);      // raw BUF_WRAP; L2-stripped landing
read_pos = 0;
for (;;) {
    wp = BUF_PTR(RXQ2);                                    // bytes pushed by HW (wraps 0..ring_size)
    avail = (wp + ring_size - read_pos) % ring_size;
    while (avail >= 32) {                                  // 32 B header
        op, len, rkey, roff = ring_rd(read_pos ...);       // wrap-aware header reads
        if (len > MAX_PAYLOAD) { bad++; break; }           // framing lost / ring lapped
        frame = align16(32 + len);
        if (frame > avail) break;                          // not fully landed yet
        switch (op) {
          SEND/SEND_IMM: send++;                            // (real impl: DMA-push to host RxWqeRing)
          WRITE/WRITE_IMM:
            mr = MR[rkey>>24];
            if (mr.rkey==rkey && (mr.access&REMOTE_WRITE) && roff+len<=mr.len) {
                dst = get_noc_addr(mr.noc_x, mr.noc_y, mr.base + roff);
                noc_async_write(ring + payload_off, dst, len);   // straddle-split at wrap; RISC issues,
                write_ok++;                                        // NoC moves — no byte copy
            }
          default: unknown++;
        }
        read_pos = (read_pos + frame) % ring_size;
        avail -= frame;
    }
    if (use_noc) noc_async_write_barrier();                // per-batch: bound outstanding NoC writes
    publish stats; if (stop) break; light pause;
}
```

The straddle split: when a frame wraps the ring end, the payload is issued as two `noc_async_write`s
(to `dst` and `dst+first`); each 4 B word stays aligned because `ring_size % 16 == 0`. `noc_base == 0`
selects a local L1 copy fallback (Stage 1/2a) instead of the NoC path.

## 6. Push model and flow control

Because the sender pushes and the BH can't pull, overrun is handled by, in order:
1. **Deep ring** (128 KB) — absorbs bursts; at 9.4 Gbps jumbo the 16 KB ring lapped (bad exploded), the
   128 KB ring kept up losslessly.
2. **Fast drain** — `noc_async_write` (not a RISC copy) keeps the consumer ahead of the producer.
3. **PFC (BH.6)** — the only true back-pressure: the BH pauses the sender when the ring fills. Not yet
   implemented (Rianta PFC is the `eth_init.cpp:538` TODO); the ~ppm discards seen without it are the
   push-with-no-flow-control tax.

## 7. Validated results (on silicon, BF3 → BH ext rail idx2, 128 KB ring)

| Path | Result |
|---|---|
| **RX dispatch (parse+route), jumbo SEND** | **264k frames/s, bad=0, 8.7 Gbps** — lossless, matches wire |
| **RX WRITE via `noc_async_write` (jumbo)** | **256k frames/s, bad=0, 8.48 Gbps** — lossless |
| RX WRITE via RISC copy (before Stage 2b) | ~2k frames/s, ~0.02 Gbps → **~128× slower** |
| Off-core landing | WRITE lands **byte-exact on a Tensix worker L1** via the NoC |
| Correctness | opcode counts exact, `read_pos` byte-exact, payload byte-exact ("TTWR"+incr) |

All numbers are **sender-limited** — the userspace `tt_rdma_bf3_send` caps the wire at ~9 Gbps (3.4 at
1500 B, 9.2 at jumbo). The BH RX has not been stressed to its own ceiling.

## 8. Performance (measured 2026-07-24)

**Sender is the hard part.** The BF3 is a BlueField DPU in embedded mode, so host-PF raw traffic goes
through the eSwitch slow path:
- **Host userspace sender:** ~11 Gbps and ~⅔ of frames dropped after `sendmmsg` accepts them —
  eSwitch-capped, multi-thread/`sendmmsg` does NOT help (`tt-rdma-eswitch-bypass.md`).
- **DPU-Arm sender on the uplink `p0`** (eSwitch-bypassed): ~16.7 Gbps, ~98 % delivered — now capped by
  the Arm userspace CPU, not the eSwitch.
- **DOCA `doca_eth_txq` HW-TX (pipelined) on `mlx5_0`**: **~143 Gbps (PHY-measured), near line rate** —
  the Arm is off the per-frame egress path. This is the real gateway TX leg (`tt-rdma-gateway-sender.md`)
  and it **proves the sender is no longer the limit: the BH RX is.** At 143 Gbps in, the RX is swamped
  and resync-thrashes (drops to ~0.5 Gbps) — confirming the ~15.8 Gbps/rail drain ceiling below is the
  binding constraint, and the ~16 → 143 Gbps gap is RX-side work.

**BH RX WRITE (via `noc_async_write` to Tensix L1), jumbo 4080 B, single rail:**

| Sender rate | BH RX WRITE | loss |
|---|---|---|
| 9.4 Gbps (host) | 8.48 Gbps | 0 (sender-limited) |
| **15.8 Gbps (DPU 2 thr)** | **15.8 Gbps, 477k frames/s** | **0 — lossless** |
| ~16.9 Gbps (DPU 4 thr) | ~16 Gbps | a few k drops/s (at the edge) |

**→ Measured BH RX drain ceiling ≈ 15.8 Gbps/rail lossless** (SEND parse ≈ same) — ~2× the earlier
host-limited 8.5 Gbps. It sits right at the DPU userspace sender's own ~16.7 Gbps ceiling, so the two
are **co-limited**; a DOCA HW-TX sender is needed to see if the chip goes higher. **~2 rails → ~32 Gbps
aggregate** by extension.

**Overload behavior (fixed):** past the drain rate the ring laps; a single bad-length frame used to
`break` the walk forever → catastrophic collapse to ~0. Now the walk **resyncs to the write head on a
bad frame** (drop the lapped span, resume) → graceful degradation. But under heavy overload the
lap/resync work competes with draining, so throughput still *degrades* (e.g. 2.2 Gbps at 16.7 Gbps in) —
the real fix is **PFC (BH.6)** to keep the sender ≤ drain, staying in the lossless regime.

**Levers to raise the ~16 Gbps/rail ceiling toward line rate**, in order:
1. **DOCA/DPA HW-TX sender** — required even to push past ~16.7 Gbps and find the true chip ceiling.
2. **Multi-rail** — ~2× (as TX did for its 397 G aggregate).
3. **Lower per-poll / per-frame overhead** — coalesce `noc_async_write`s for contiguous same-MR frames,
   batch the barrier, trim the stats/pace in the hot loop.
4. **Bigger frames** — more bytes per header (needs MTU > 4080, another DPU-side step).
5. **PFC** — makes ≤-ceiling operation lossless and removes the resync tax.
6. **Get the RISC off the per-frame parse** — revive the HW RX classifier (dead code) or an
   overlay/DMA-driven path; some per-frame RISC work is fundamental unless HW-assisted (routing needs
   the header). This is where the gap to per-rail line rate ultimately lives.

## 9. Open questions / follow-ups

- **SEND landing** — currently counted only; the real path DMA-pushes SEND payloads to the host
  RxWqeRing (host hugepage via NoC→PCIe). Not yet built.
- **CRC-32 validation** — **Done (SW).** The header `header_cksum` (CRC-32, poly 0x04C11DB7 =
  reflected 0xEDB88320) is validated on RX; mismatches are dropped + counted (`crc_err`). The kernel
  computes it bit-serially today. **Measured cost (2026-07-26, DPU sender, eSwitch-bypassed):** the
  bit-serial CRC is a real fast-path bottleneck, not free — with CRC **off** the RISC drained a jumbo
  WRITE stream losslessly at ≥560k fps (≥18 Gbps, sender-limited — ceiling not even reached); with CRC
  **on** the RISC ceiling drops to ~408k fps (~13 Gbps) and beyond that it laps into resync collapse.
  So the SW CRC costs **≥25% of jumbo-WRITE throughput** (≥0.5 µs/frame added; SEND small-frame showed
  ~0.18 µs/frame lower bound). This is the concrete justification for the HW offload below. **Follow-up (HW offload):** the ETH-CTRL `ROCE_ICRC` engine (regs @
  `0xFFB98100`) implements this exact polynomial with an inline RX check (`RX_CHECK_EN` + a
  `RX_CALCULATED` vs `RX_RECEIVED` compare) and a 64-byte header bytemask. Wiring it removes the CRC
  from the RISC hot path entirely (zero cycles/frame). This is why the wire cksum was moved onto the
  0x04C11DB7 polynomial. On-silicon step: calibrate the CTRL bit-order/reflection selects + init so
  HW `RX_CALCULATED` == the SW `tt_rdma_crc32` for a known frame, *then* trust `RX_CHECK_EN`.
  - **Probe tool:** `bh1_icrc_probe` (`kernels/bh_rdma_icrc_probe.cpp`, regs in `tt_rdma_eth_icrc.h`).
    It puts RXQ2 in raw mode, snapshots the POR ICRC config, optionally programs `CTRL`/`RX_INIT` from
    args, and on each landed frame reports the engine's `RX_CALCULATED`/`RX_RECEIVED` next to the SW
    `tt_rdma_crc32` of the same header — printing a verdict (`HW_engaged`, `rx_calc==sw_crc`). Run it,
    fire a known frame from the BF3, and read the verdict. Sweep the 16×16 bit-order combos from a shell
    loop over the `ctrl_hex` arg (base off POR `0x30700000`); `RX_CHECK_EN` stays off until HW==SW is
    proven, so a mis-tuned engine can never drop live traffic. **Two possible outcomes, both informative:**
    (a) some config yields `rx_calc==sw_crc` → bake that config into the RX kernel + flip `RX_CHECK_EN`,
    CRC leaves the hot path; (b) the engine never engages on raw `0x1AF6` framing (it is built for
    IPv4/UDP:4791/BTH RoCE packets) → HW offload needs RoCE-shaped frames, so keep the SW check but drop
    it to a slice-by-4 table (~7 ops/byte vs ~32 bit-serial). The probe tells us which before we commit.
- **ACK / READ** — `0x40` ACK reception and `0x20/0x21` READ_REQ/RESP not yet implemented on BH RX.
- **MR carries the full NoC address** — Stage 2b passes the target `(noc_x, noc_y, base)` as kernel
  args; the productized form stores the NoC-encoded `base_noc_addr` in the MR entry (host builds it).
- **Overflow handling** — a single bad-length frame currently `break`s the whole walk; on a lapped ring
  this stalls. Needs resync-on-bad + a drop counter, plus PFC to avoid lapping at all.

## 10. Milestones

- **RX.0 (M-1b)** — RXQ2 raw reception; BF3 unicast `0x1AF6` → L1 byte-exact. **Done.**
- **RX.1 (BH.2-RX)** — frame walk + opcode dispatch (SEND/WRITE); NOWRAP one-shot. **Done.**
- **RX.2 (Stage 2a)** — BUF_WRAP streaming ring (continuous RX). **Done.**
- **RX.3** — 128 KB ring (lossless jumbo absorb). **Done.**
- **RX.4 (Stage 2b, core of BH.3)** — MR table + WRITE via `noc_async_write` off-core (8.5 Gbps). **Done.**
- **RX.5** — CRC-32 header validation (SW). **Done.** HW `ROCE_ICRC` offload — follow-up.
- **RX.5b** — SEND→host RxWqeRing; ACK/READ. Pending.
- **RX.6** — PFC-lossless (BH.6) + resync-on-bad; fast gateway sender to find the real ceiling. Pending.
