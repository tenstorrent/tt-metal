# TT-RDMA-over-Ethernet — Wire Protocol v1

Status: design draft. Target: WH ext-CMAC ↔ Mellanox CX-5 / FPGA-NIC, 100GbE L2.
Supersedes: ad-hoc soak format (ethertype 0x1AF4, seq# at byte 14, no opcode field).

---

## 0. Why a protocol, not more workarounds

The 2% `dbg_mixed_size_evts` leak on packed pile-ups isn't a bug in the stride-walk — it's a *consequence* of having no length field on the wire. FW infers length from `BUF_PTR / pkt_end_cnt` because the wire doesn't tell it. The fix isn't another fence; it's putting `length` in the header where every other reliable transport puts it. Everything else (rkey, remote_offset, opcode multiplexing) costs ~24 B more and unlocks all four RDMA verbs.

## 1. Frame layout (after CMAC L2-strip)

Fixed 32-byte header, little-endian, payload appended:

```
Offset  Size  Field            Notes
------  ----  --------------   --------------------------------------------
 0      1     opcode           1-byte enum (§2). 0 = reserved/invalid.
 1      1     version_flags    bits[3:0]=ver (=1), bit4=IMM_PRESENT,
                               bit5=REQ_ACK, bit6=SOLICITED, bit7=reserved
 2      2     tag              request-response correlation (READ/ATOMIC);
                               opaque cookie for SEND/WRITE
 4      4     length           payload byte length (NOT including header).
                               FW uses this for single-ring BUF_WRAP walk —
                               this is the field that kills mixed_size_evts.
 8      4     seq              reliability seq# (Phase 2 cumulative-ACK).
                               Replaces today's frame[14] position; see §6.
12      4     rkey             memory region key (0 = unused / SEND).
16      8     remote_offset    byte offset within rkey'd region (WRITE/READ).
                               Ignored when rkey=0.
24      4     imm_data         immediate value (valid iff version_flags.IMM).
28      4     header_cksum     CRC-32C over bytes [0..27]. Inner integrity
                               for header only — payload is covered by FCS.
32      L     payload          L = length bytes; absent for READ_REQ/ACK.
```

Total header = 32 B. Header offsets are 4-byte aligned so the erisc can use word loads through the whole walk. The header CRC is over the 28 bytes of header *only* — wire FCS covers the payload byte-for-byte already, and a payload-CRC would force the entire frame to be in L1 before validation (we want to stream).

**Why this order:**
- `opcode` first so FW can branch in 1 load + 1 compare before reading anything else.
- `length` at +4 (not buried) so the stride walk can read it in the same word as opcode/version/tag and skip ahead — fixes the current single-ring walk.
- `seq` keeps its bit-width but moves from offset 14 to offset 8 (4-byte aligned, no overlap with MAC bytes after L2 strip). Phase 2 receiver/FW ACK code only needs to update one constant.
- `rkey` + `remote_offset` together = 12 B, contiguous so a future zero-copy path can DMA them as one word-pair into a region-lookup register.
- `imm_data` separate from payload so receivers polling for completion don't need to look inside the payload buffer.

## 2. Opcode set (1-byte enum, room for 256)

```
0x00  RESERVED
0x01  SEND               unsolicited; goes into next pre-posted recv slot
0x02  SEND_IMM           SEND + imm_data delivered to recv CQE
0x10  WRITE              place at base[remote_offset] (rkey checked)
0x11  WRITE_IMM          WRITE + imm_data raises a completion on receiver
0x20  READ_REQ           initiator → target; payload empty, length=request_len
0x21  READ_RESP          target → initiator; payload=fetched bytes
0x30  ATOMIC_CAS         compare-and-swap on 8B at remote_offset (deferred)
0x31  ATOMIC_FAA         fetch-and-add on 8B at remote_offset (deferred)
0x32  ATOMIC_RESP        result delivery for CAS/FAA (deferred)
0x40  ACK                cumulative ACK; payload empty; ack_seq=seq field
0xF0  CONTROL            sub-opcode in imm_data (MR register/dereg, etc.)
0xFE  PROBE              link-layer keepalive; loopback contents in payload
0xFF  RESERVED
```

Reserve 0x50-0xEF for future verbs (multicast, unreliable-datagram). Sub-opcode under 0xF0 keeps the main switch single-byte.

**Recommendation:** ship v1 with SEND, SEND_IMM, WRITE, WRITE_IMM, READ_REQ, READ_RESP, ACK, CONTROL. Defer ATOMIC.

## 3. Memory regions

FW maintains a **fixed-size table of 16 MRs** at a new RCB-extension address (proposal: `kRcbMrTableAddr = 0x8500`, 16 entries × 32 B = 512 B). Each entry:

```
+0   u64  base_noc_addr    target NoC address (region origin)
+8   u64  length           bytes
+16  u32  rkey             host-assigned, 0 = entry-not-valid
+20  u32  access_flags     bit0=LOCAL_WRITE, bit1=REMOTE_WRITE,
                           bit2=REMOTE_READ, bit3=REMOTE_ATOMIC
+24  u32  pd               protection domain id (defer; 0 = default)
+28  u32  reserved
```

**Validation rule (FW, per inbound WRITE/READ_REQ):**
```
mr = lookup_by_rkey(rkey)
if (!mr || mr.rkey != rkey)               -> drop, log "rkey_miss"
if ((opcode is WRITE) && !REMOTE_WRITE)   -> drop, log "rkey_access"
if ((opcode is READ_REQ) && !REMOTE_READ) -> drop, log "rkey_access"
if (remote_offset + length > mr.length)   -> drop, log "rkey_bounds"
if (remote_offset + length < remote_offset) -> drop, log "rkey_wrap"
```

**Registration path:** new CONTROL sub-opcode `CTRL_REG_MR` (sub-op 0x01) over the existing host→FW RCB mailbox path used by Phase 3's `host_noc_addr` publish. Host writes a candidate MR entry into a staging slot at RCB+0x60, then writes `CTRL_REG_MR` into a "control_doorbell" word; FW copies into the MR table and clears the doorbell.

## 4. Ethertype

**Recommendation: allocate 0x1AF6 for TT-RDMA v1.** Keep 0x1AF4 (data) and 0x1AF5 (ACK) reserved for the legacy soak path during transition.

## 5. Reliability interaction

- **Single seq# space**, all data-carrying opcodes draw from it.
- **READ correlation** uses `tag`, not seq#.
- **ACK frames** carry `ack_seq` in the `seq` field itself (cumulative ack value reuses the slot). `length=0`, `rkey=0`, `tag=ack_flags`. Replaces the current `WQE_ACK_MAGIC=0xACE5C0DE` check.
- **REQ_ACK bit** (version_flags.bit5) lets the sender force an immediate ACK on a specific frame.

## 6. Backward compat / migration

1. **v0 keeps running** at ethertype 0x1AF4.
2. **v1 ships at 0x1AF6** with `version_flags.ver = 1`. Both can flow simultaneously.
3. **v0 deprecation** when v1 hits feature-parity with the soak loop's perf.

The `version_flags.ver` field is the safety net: `ver` mismatch detectable in the first byte after opcode.

## 7. Wire-format examples (hex, post L2-strip)

### 7.1 SEND, 64 B payload, seq=0x1234, tag=0xCAFE

```
01 01 FE CA  40 00 00 00  34 12 00 00  00 00 00 00
00 00 00 00 00 00 00 00  00 00 00 00  C3 A1 9B 7E
<64 B payload>
```

### 7.2 WRITE_IMM, 1408 B payload, rkey=0xDEADBEEF, remote_offset=0x10000, imm=0xAB

```
11 11 01 00  80 05 00 00  78 56 34 12  EF BE AD DE
00 00 01 00 00 00 00 00  AB 00 00 00  4F 22 D1 06
<1408 B payload>
```

### 7.3 READ_REQ, request 4096 B at rkey=0x42, remote_offset=0x2000

```
20 01 07 00  00 10 00 00  9A 00 00 00  42 00 00 00
00 20 00 00 00 00 00 00  00 00 00 00  17 8E 33 BC
<no payload>
```

### 7.4 READ_RESP, carrying the 4096 B back, tag echoes 7

```
21 01 07 00  00 10 00 00  03 00 00 00  00 00 00 00
00 00 00 00 00 00 00 00  00 00 00 00  92 4C 5F 71
<4096 B payload>
```

### 7.5 ACK, cumulative ack=0x12345677

```
40 01 00 00  00 00 00 00  77 56 34 12  00 00 00 00
00 00 00 00 00 00 00 00  01 00 00 00  88 11 EE FF
<no payload>
```

---

## Tradeoffs considered and rejected

- **Variable-length header**: not worth it — fixed 32 B header lets FW issue single 32-B word-block read.
- **Inner payload CRC**: FCS already gets BER ~10⁻¹². Skip until we see an actual integrity bug.
- **2-byte opcode**: 256 is plenty. Saved byte goes to `version_flags`.
- **Per-frame rkey lookup cost**: 16-entry linear scan ~16 cycles, immeasurable next to the NoC transaction.

## What this fixes (concretely)

- The 2% `mixed_size_evts` leak: `length` in header replaces BUF_PTR/pkt_end_cnt division.
- ACK dispatch: opcode-based, no payload-magic match.
- Path to one-sided WRITE: drop-in once MR registration lands.
