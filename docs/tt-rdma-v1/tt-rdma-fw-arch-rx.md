# fw-arch-rx-rdma.md — TT-RDMA-over-Ethernet RX Path on WH erisc CMAC FW

## 0. Premise

Replace the current 2-slot (later 4-slot probe) ping-pong PACKET_BUF / PACKET_BUF1 scheme in `run_wqe_ring_loop`'s RX block with a **single ring** sized `N` bytes at `PACKET_BUF_ADDR=0x4000`, exploiting CMAC's already-enabled `ETH_RXQ_CTRL_BUF_WRAP` mode (`eth_cmac_init.cpp:745`). Frame boundaries come from the wire-protocol length field:

```
[u8 opcode][u8 flags][u16 reserved][u32 length][u32 seq][u32 rkey]
[u64 remote_offset][... payload ...]
```

`length` = total frame size in bytes including the header. Header is 24B aligned to 16B (matches CMAC RX DMA write width).

ACK frames reuse the same header with `opcode=ACK` and `length=24` (header-only, `seq` carries `ack_seq`).

---

## 1. L1 Layout Overhaul

### Constraint recap

- Earlier experiment: PACKET_BUF2 at `0x21000` collapsed admit to 0.12% — CMAC RX DMA can't address that region on this config. **Hard cap: stay inside 0x4000–0x7FFF for RX DMA targets.**

### Proposed new map

```
0x0000-0x021F  startup + boot params
0x1F40-0x1FFF  GW mailbox (some fields repurposed)
0x2000-0x3AD0  cmac_simple .text / .data
0x4000-0x7BFF  PACKET_BUF (RX ring, single, N=14 KB)  ← NEW
0x7C00-0x7FFF  reserved scratch (256 B; for ring-overflow staging)
0x8000-0x83FF  WQE descriptor table (unchanged, 1 KB)
0x8400-0x84BF  RCB header (unchanged through +0x3F)
0x84C0-0x84FF  RCB DBG counters (unchanged, dbg[0..15])
0x8500-0x85FF  MR table (16 entries × 16 B = 256 B)  ← NEW
0x8600-0x8FFF  RxWqeRing (RX→host descriptors, 10 slots × 256B — see L5 doc)
0x9000-0x20FFF WQE_PAYLOAD (TX side, unchanged 96 KB)
0x21000-0x21800 TX_BUF0 (relocated from 0x6000)  ← NEW
0x21800-0x22000 TX_BUF1 (relocated from 0x6600)  ← NEW
0x22000+        free / FW scratch (not RX-DMA-addressable)
```

### Why N = 14 KB

- Phase 3.2 sustained ~909k fps wire RX. 1500 B frame at 909k fps = ~1.36 GB/s = **1.36 KB/µs**. A 14 KB ring gives ~10 µs head-room before producer catches consumer.
- TX_BUF0/1 must move out of `0x6000–0x6BFF` to free contiguous space. CMAC **TX** (unlike RX) has no addressability problem with upper L1 — Phase 3.2 already writes WQE payload up to 0x21000. Safe to relocate.

> Punt: 14 vs 16 KB ring. If we drop the spill scratch at 0x7C00 we get 16 KB (power-of-two-friendly modulo). 14 KB is the safer default for v1.

---

## 2. BUF_WRAP-Based RX Consumption

### FW state

```c
const uint32_t RX_RING_BASE = PACKET_BUF_ADDR;        // 0x4000
const uint32_t RX_RING_N    = 14 * 1024;              // 14336 bytes
uint32_t last_rx_ptr = eth_rxq_reg_read(0, ETH_RXQ_BUF_PTR);  // byte offset
```

### Inner drain loop

```c
static uint32_t rx_drain(uint32_t* last_rx_ptr_io) {
    uint32_t last = *last_rx_ptr_io;
    uint32_t frames = 0;

    while (frames < RX_INNER_DRAIN_MAX) {  // §5: 32
        uint32_t cur = eth_rxq_reg_read(0, ETH_RXQ_BUF_PTR);
        if (cur == last) break;

        uint32_t avail = (cur >= last) ? (cur - last)
                                       : (RX_RING_N - last + cur);
        if (avail < RDMA_HDR_BYTES) break;

        uint32_t len = read_u32_with_wrap(RX_RING_BASE, last, RX_RING_N, 4);
        uint8_t  op  = read_u8_at (RX_RING_BASE, last, RX_RING_N, 0);

        if (len < RDMA_HDR_BYTES || len > RDMA_FRAME_MAX) {
            dbg[26]++;       // corrupt-length
            last = cur;      // resync: conservative drop
            break;
        }
        if (len > avail) break;  // frame not fully arrived yet

        // Tail-edge fence: wait for in-flight DMA to commit
        if ((last + len) % RX_RING_N == cur) {
            for (int w = 0; w < 64; w++) {
                if (eth_rxq_reg_read(0, 0x50 /*OUTSTANDING_WR_CNT*/) == 0) break;
            }
        }

        dispatch_opcode(op, RX_RING_BASE, last, RX_RING_N, len);

        last = (last + len) % RX_RING_N;
        frames++;
    }

    *last_rx_ptr_io = last;
    return frames;
}
```

### Edge-case rules

| Case | Behavior |
|---|---|
| Header straddles wrap | `read_u32_with_wrap` returns from two reads at `ring[last]` and `ring[0]` |
| Payload straddles wrap | Dispatcher does two NoC writes (split at wrap) |
| Length > N or > MTU | Increment `dbg[26]`; resync `last = cur` |
| Length < HDR or 0 | Same — corrupt |
| Overflow (producer laps consumer) | Detect via heuristic; `dbg[25]++`; resync; Phase 2 ACK retx recovers |

---

## 3. Opcode Dispatch

```c
enum {
    OP_SEND      = 0x01,
    OP_WRITE     = 0x02,
    OP_READ_REQ  = 0x03,
    OP_READ_RESP = 0x04,
    OP_ACK       = 0x05,
};

static void dispatch_opcode(uint8_t op, uint32_t base, uint32_t off,
                            uint32_t N, uint32_t len) {
    uint32_t seq  = read_u32_with_wrap(base, off, N, 8);
    uint32_t rkey = read_u32_with_wrap(base, off, N, 12);
    uint64_t roff = read_u64_with_wrap(base, off, N, 16);
    dbg[15 + op]++;

    switch (op) {
    case OP_WRITE: {
        const mr_entry_t* mr = mr_lookup(rkey);
        if (!mr || !(mr->access & MR_ACCESS_WRITE) ||
             roff + (len - RDMA_HDR_BYTES) > mr->length) {
            dbg[23]++;
            return;
        }
        uint32_t pay_off  = (off + RDMA_HDR_BYTES) % N;
        uint32_t pay_len  = len - RDMA_HDR_BYTES;
        uint64_t dst_noc  = mr->base_noc_addr + roff;
        if (pay_off + pay_len <= N) {
            noc_fast_write(0, base + pay_off, dst_noc, pay_len);
        } else {
            uint32_t first = N - pay_off;
            noc_fast_write(0, base + pay_off, dst_noc,         first);
            noc_fast_write(0, base,           dst_noc + first, pay_len - first);
        }
        break;
    }
    case OP_SEND:
        rxwqe_ring_publish(base, off, N, len, seq);
        break;
    case OP_READ_REQ: {
        const mr_entry_t* mr = mr_lookup(rkey);
        if (!mr || !(mr->access & MR_ACCESS_READ)) { dbg[23]++; return; }
        build_read_resp_and_tx(mr->base_noc_addr + roff,
                               len - RDMA_HDR_BYTES, seq);
        break;
    }
    case OP_ACK: {
        uint32_t cur_acked = rcb[2];
        if ((int32_t)(seq - cur_acked) > 0) rcb[2] = seq;
        dbg[1]++;
        break;
    }
    default:
        dbg[22]++;
        break;
    }
}
```

`noc_fast_write` non-blocking on NoC 0. (NoC 1 reserved for TX-side DMA-pull, avoids NoC head-of-line blocking.)

---

## 4. Memory Region Table

### Layout (16 entries × 16 B = 256 B at 0x8500)

```c
typedef struct {
    uint32_t base_noc_lo;     // +0x00
    uint32_t base_noc_hi;     // +0x04  (PCIe/NoC tile encoding)
    uint32_t length;          // +0x08  bytes
    uint8_t  access_flags;    // +0x0C  bit0=READ, bit1=WRITE, bit2=ATOMIC
    uint8_t  rkey;            // +0x0D  must match entry index for v1
    uint16_t reserved;        // +0x0E
} mr_entry_t;

#define MR_TABLE_ADDR  0x8500
#define MR_TABLE_N     16
```

For v1, `rkey == table_index`. Lookup is single-cycle indexed read; no hash.

### Host registration mechanism — **recommend (a)**

**(a) Control opcode over the TX WQE ring.** Define `OP_REGISTER_MR` as a local WQE (`flags |= WQE_FLAG_CTRL_LOCAL`). FW services it between retx handling and TX drain, reads `{rkey, base_lo, base_hi, length, access}` from desc + payload slot, writes to MR_TABLE, never emits a wire frame. Host gets completion via `cq_head`.

**(b) New RCB field at `RCB+0x80..0xFF`.** Trivial; no flow control, no completion, capped at 8 entries.

Choose (a); (b) as fallback if descriptor-table changes are too invasive.

> Punt: MR table location 0x8500. If RxWqeRing wants more room, move MR to `0x83C0` (RCB header has 64 B free at +0xC0..+0xFF).

---

## 5. Concurrency with TX

```c
#define RX_INNER_DRAIN_MAX 32   // frames per outer iter
```

Rationale: 32 × ~1 µs/frame (NoC write) = ~32 µs worst case before returning to TX. Under steady-state Phase 3.2 TX cadence (~1.1 µs/frame), this injects at most ~30 TX frames of latency, absorbable by 64-slot ring.

Outer loop order (unchanged shape):

```
while (true) {
    handle_retx();
    tx_drain_one();
    rx_drain(&last);   // §2, bounded at 32
}
```

Single TX WQE per outer iter (existing convention). Yields 1:32 TX:RX worst-case ratio when RX is hot — matches typical one-sided WRITE workload asymmetry.

---

## 6. Phase 2 Reliability Integration

**Current:** ACK frames have `WQE_ACK_MAGIC=0xACE5C0DE` at payload offset 0; offset 4 = `ack_seq`. RX block matches the magic and updates `rcb[2]`.

**New:** ACK has `opcode=OP_ACK=0x05`; `seq` field carries `ack_seq`. Dispatch arm updates `rcb[2]` identically. Magic-word check becomes single opcode comparison.

**Transition:**
1. Land FW with both detection paths active (opcode==0x05 *or* first word == WQE_ACK_MAGIC) for one-build overlap.
2. Host emits new header.
3. Drop magic check.

retx-pending walk (`rcb[5]`) unchanged.

---

## 7. Concrete Diff vs Current Code

### `main_cmac.cc`

| Location | Change |
|---|---|
| L41–50 (PACKET_BUF1/2/3 defines) | Delete; add `RX_RING_BASE`, `RX_RING_N` |
| L52–122 (layout doc) | Append MR doc; replace ACK-magic doc with opcode-header doc |
| L149–182 (`update_counters`) | Drop `rxstatus[10]` (RXQ1 unused); drop first-4-word peek |
| L309–445 (TX block) | Unchanged through `rcb[5]` retx + TX drain |
| L445–573 (RX block) | **Replaced wholesale** with §2 + §3 |
| New helpers | `rx_drain`, `dispatch_opcode`, `mr_lookup`, `read_u32_with_wrap`, `read_u64_with_wrap`, `noc_fast_write_wrap_safe` (static, same TU) |
| L344–352 (dbg init) | Extend dbg[0..27] |

### `eth_cmac_init.cpp` (~L740)

```c
uint32_t rx0_buf_word_addr  = RX_RING_BASE / 16;   // = 0x400
uint32_t rx0_buf_size_words = RX_RING_N / 16;      // = 0x380
eth_rxq_init(0, rx0_buf_word_addr, rx0_buf_size_words, /*wrap=*/true, /*packet_mode=*/false);
// RXQ1: disable or stub to 0x7C00 (256 B)
```

Relocate `TX_BUF0_ADDR=0x21000`, `TX_BUF1_ADDR=0x21800` in `eth_cmac_init.h:37-38`.

---

## 8. Debugging Hooks (extended dbg[] at 0x8440)

| dbg[] | Offset | Meaning |
|---|---|---|
| 0 | +0x00 | rx_frames |
| 1 | +0x04 | ack_matches (compat; == dbg[20]) |
| 2 | +0x08 | retx_runs |
| 3..7 | +0x0C–0x1F | first 20 B of most-recent header |
| 8 | +0x20 | last_avail |
| 9 | +0x24 | mac_rx_cnt |
| 10 | +0x28 | rxq0_drop |
| 11 | +0x2C | last_cur |
| 12 | +0x30 | last_last_rx_ptr |
| 14 | +0x38 | inner_drain_count |
| 16..21 | +0x40–0x57 | per-opcode counters (SEND, WRITE, READ_REQ, READ_RESP, ACK, +1 spare) |
| 22 | +0x58 | unknown_opcode |
| 23 | +0x5C | rkey_or_bounds_fail |
| 24 | +0x60 | ring_wrap_events |
| 25 | +0x64 | ring_overflow_events |
| 26 | +0x68 | corrupt_length_events |
| 27 | +0x6C | outstanding_wr_wait_timeouts |

Total: 28 × 4 B = 112 B; fits in existing 0x8440–0x84FF window.

---

## 9. Top 3 Risks + Mitigations

### Risk 1 — Wrap-mid-frame data corruption from CMAC's L1 write queue

With BUF_WRAP, the in-flight L1 write race exists at the tail edge. **Mitigation:** at `last + len == cur` mod N, gate dispatch on `OUTSTANDING_WR_CNT == 0` with 64-poll bound (proven fix from `project_v2_ring_rx_100pct_admit.md`).

### Risk 2 — Ring overflow under sustained burst

14 KB ring × 1.36 KB/µs wire = ~10 µs head-room. NoC write ~1 µs; 32-frame drain ~32 µs. Will overrun under sustained pure-WRITE burst.

**Mitigation, ordered by effort:**
1. Detect: `dbg[25]` overflow + `last = cur` resync; Phase 2 retx recovers.
2. Backpressure: assert `ETH_RXQ_CTRL_FORCE_BPRESSURE` when `last_avail > 3N/4`, clear at `< N/2`.
3. Architectural: split RX onto erisc1.

### Risk 3 — Mid-frame header bytes split across wrap break parsing

`*(uint32_t*)(ring + N - 2)` reads past ring → adjacent L1 (MR garbage).

**Mitigation:** `read_u32_with_wrap` byte-composes on straddle. Costs 4 byte loads in rare wrap-tail case. Unit test with fault-injection packet sized to land header at `N-2`.

---

### Critical Files

- `budabackend-master/src/firmware/riscv/targets/erisc_cmac_simple/src/main_cmac.cc`
- `budabackend-master/src/firmware/riscv/targets/erisc_cmac_simple/src/api/eth_cmac_init.cpp`
- `budabackend-master/src/firmware/riscv/targets/erisc_cmac_simple/src/api/eth_cmac_init.h`
- `budabackend-master/src/firmware/riscv/targets/erisc_cmac_simple/src/api/eth_hw_api.cpp`
- `tt-metal-external-eth/tests/tt_metal/llrt/test_external_cmac_soak.cpp` (dbg offset/labels)
