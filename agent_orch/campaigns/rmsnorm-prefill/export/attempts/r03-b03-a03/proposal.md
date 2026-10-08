# r03-b03-a03: output drain round-robins its DRAM tile writes over the 4 unicast request VCs (0-3) instead of the single static VC 1, so a core keeps several 2 KB packets injecting at once instead of each one waiting for the previous packet to clear VC 1

## Motivation
On every shape the drain is now the tail of the kernel, and its limit is **per core**:
- Parent report (`drain.py`, medians over calls x chips): after the first POST tile, the writer drains at **100-109
  ns/tile** (h3584 3.06 µs/28, h4096 3.50/32, h6144 5.19/48, h7168 5.62/56). Compute packs at ~64-68 ns/tile, so the
  drain ends 1.1-1.8 µs after the pack's POST end.
- The rate is the same on h3584 (rows alternate DRAM-bank quads, because 28 % 8 = 4) and on the bank-lockstepped
  shapes (32/48/56 % 8 = 0). r03-b04-a02 also saw the same per-core rate with only 10 writers on the chip. So the limit
  is not DRAM-bank or aggregate-link bandwidth. It is how fast one core gets its packets onto the NoC.
- Software issue cost is not the limit either. I disassembled the JIT-built writer (`brisc.elf`): the drain loop
  body is ~50 instructions per tile, then a spin on `NOC_CMD_CTRL` (cmd-buf ready). At ~147 cycles/tile, most of each
  tile's time is spent waiting for the command buffer.
- The NoC ISA docs (tt-isa-documentation, NIU programming): a request's `NOC_CMD_CTRL` returns to ready only **once a
  virtual channel has been assigned** to it. Every drain write uses `NOC_CMD_VC_STATIC` with VC 1
  (NOC_UNICAST_WRITE_VC), so tile k+1 can't get its VC until tile k's 2 KB packet has drained out of that VC's
  injection buffer. Whenever a packet stalls downstream (DRAM NIU ingest, a hot link into a DRAM column), the
  whole per-core stream stalls behind it: one packet in injection at a time per NoC.

## Mechanism
`dit_rmsnorm_fused_worker_writer.cpp`, drain loop only: issue each output tile with
`noc.async_write<NocOptions::CUSTOM_VC>(..., {.vc = v})`, where v rotates over 0,1,2,3 per tile. These are the four
unicast request VCs: class bits 0b00/0b01 x buddy bit. 4-5 are multicast and 6-7 are responses, so none of them is
touched. The dual-NoC rule (which tiles go on NoC0) is unchanged. Each NoC gets its own VC rotation. Flush/barrier
semantics are unchanged: the counters are per NIU, not per VC. Only the order of arrival at different banks can
change, and nothing depends on it. Kernel-only (JIT), no host change.

## Why this is not a repeat
Earlier drain work changed *which* NoC or *which path* a tile takes (r01-b02-a04, r01-b03-a04, r02-b01-a01,
r02-b02-a01, r02-b03-a01), the bank order (r01-b03-a03), the flush depth (r01-b02-a02), or the core count/timing
(r01-b03-a0x, r03-b04-a02). Every one of them kept all writes on one static VC per NoC. None looked at VC-level
injection concurrency, which is a per-core limit. That is the kind of limit r03-b04-a02 measured.

## Expected effect and risk
- If VC serialization is the limit, the per-tile drain time drops toward the pack rate (~65-70 ns/tile). The drain end
  then moves 0.8-1.7 µs earlier (more on the wide shapes), about +5-8% score. If the limit is a shared link or the DRAM
  NIU, the effect is nil. The per-tile drain rate in `drain.py` will show which.
- Risks: VC 0/2/3 traffic shares buffers with other unicast users (fabric EDM local writes, dispatch). That costs
  bandwidth at worst, with no deadlock: all of these are sink writes. Correctness is unaffected, because every tile
  goes to a distinct address.
