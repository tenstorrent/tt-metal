# eth_credit: credit flow control across a direct Ethernet link

`x280_eth_credit` measures read/write-pointer credit flow control on the path

```text
X280 forwarder (chip A) → E0 (chip A) → Ethernet → E1 (chip B) → X280 forwarder (chip B)
```

and finds which hop bounds sustained throughput. It covers one edge, with
traffic in either or both directions.

The two chips are direct Ethernet neighbours. The Ethernet send names a remote
ERISC L1 address and carries no chip ID, so the design cannot reach a
non-neighbour; multi-hop transport is tt-fabric's problem.

## Protocol

Every FIFO lives in its consumer's memory. Every pointer has one writer: the
producer owns the write pointer, the consumer owns the read pointer, and each
pushes its pointer into a shadow copy held by the other side. A consumer polls
the shadow write pointer; a producer compares its write pointer with the shadow
read pointer.

### Forwarders

Each of P X280 harts per chip is a forwarder, P ≤ `EC_MAX_P = 4`. Forwarder `p`
runs one loop that sends through its own `tx_fifo[p]` on the local E0 and
drains its own inbound FIFO `in[p]`. The path mirrors per hart: forwarder `p` on
chip A feeds link FIFO `rx[p]` on chip B's E1, which delivers into forwarder
`p`'s `in[p]` on chip B. Received packets are consumed, not re-forwarded. A
one-direction run uses the same code, and the idle side's FIFOs stay empty.

### The nine writes

For one direction, chip A to chip B:

| # | Writer → target | What | Transport |
|---|---|---|---|
| W1 | forwarder `p` (A) → E0 (A) `tx_fifo[p]` slot | packet header | NoC |
| W2 | forwarder `p` (A) → E0 (A) `wr_sh[p]` | new write pointer | NoC |
| W3 | E0 (A) → E1 (B) `rx[p]` slot | slot contents | Ethernet TXQ 0, TT-link L1 write |
| W4 | E0 (A) → E1 (B) pending counter `pkts[p]` | +1 | Ethernet TXQ 0, TT-link MMIO write |
| W5 | E1 (B) → forwarder `p` (B) `in[p]` slot | header | NoC 1 |
| W6 | E1 (B) → forwarder `p` (B) `wr_sh` | new write pointer | NoC 1 |
| R1 | forwarder `p` (B) → E1 (B) `out_rd_sh[p]` | new read pointer | NoC |
| R2 | E1 (B) → E0 (A) `crd[p]` | new link read pointer | Ethernet TXQ 1, TT-link L1 write |
| R3 | E0 (A) → forwarder `p` (A) `rd_sh[p]` | `crd[p]`, unchanged | NoC 0 |

`tx_fifo[p]` and `rx[p]` have the same depth, `N_LINK`, and link slot `k`
pairs with `tx_fifo[p]` slot `k`. E1's read pointer for `rx[p]` is therefore
also the read pointer for `tx_fifo[p]`, and E0 forwards R2's value as R3 without
translation.

E0 sends a `tx_fifo[p]` slot directly as the Ethernet source. That slot stays
busy until R2 returns it: TT-link rereads source L1 on every resend and gives no
send-complete. The sender's credit therefore includes E1's forwarding. The
fabric router does the same: with first-level ack off (it is on only where a
direction needs deadlock avoidance, bubble flow control on ring and torus
topologies), credits go to upstream workers on completion acks
(`erisc_datamover_builder.cpp`, `fabric_erisc_router.cpp`).

E1 frees an `rx[p]` slot after W5's write acknowledgement, then sends R2.

### Ethernet cores and queues

The two Ethernet roles run on the two baby-RISCVs of the Ethernet tile:

- **ERISC0 (E0)** is the outbound role. It polls its forwarders' shadows and its
  `crd[p]` in L1, calling `invalidate_l1_cache()` before each reread, and sends
  R3. TXQ 0 carries its W3 and W4.
- **ERISC1 (E1)** is the inbound role. It reads `pkts[p]` (a register read, no
  fence), writes −1 per packet it consumes, then does W5,
  `noc_async_write_barrier()`, W6. It sends R2 on TXQ 1.

E0 and E1 each round-robin over `p`, skipping a FIFO that is empty or whose
downstream has no credit. E1 keeps its delivery state per `p`.

TXQ 0's TT-link association is established by base firmware. E0 calls
`eth_enable_packet_mode()` (`tt_metal/fabric/hw/inc/edm_fabric/fabric_txq_setup.h`)
once, before any traffic, on both tiles. It programs both queues whatever queue
number it is given:

- puts TXQ 0 and TXQ 1 in TT-link mode (`ETH_TXQ_CTRL` bit 0);
- sets TX header-table entry 1's destination MAC for TXQ 1, and both queues'
  `TXPKT_CFG_SEL`;
- puts RXQ 0 and RXQ 1 in packet mode;
- reprograms `MAC_RX_ADDR_ROUTING` so the receiver sorts TXQ 0 and TXQ 1 traffic
  into RXQ 0 and RXQ 1.

This is both a sender-side and a receiver-side setting, so it runs on both
tiles. It writes TXQ 0 registers, so it runs before any traffic.

It writes whole registers, with these side effects:

- `ETH_TXQ_CTRL` is written as 1 on TXQ 0 and TXQ 1, which clears `DIS_DROP`
  (bit 3). `EthernetTxRx.md` names bit 0 `ETH_TXQ_CTRL_KEEPALIVE`; tt-metal
  calls it `packet_resend_mode_active`.
- `ETH_RXQ_CTRL` is written as bit 1 only, which clears `BUF_WRAP`.
- `MAC_RX_ADDR_ROUTING` bits 0–7 become broadcast → RXQ 0 and multicast →
  RXQ 1, for data and MMIO alike; the unicast bits 8–11 are preserved.
  `MAC_RX_ROUTING` is set to 0.
- TX header-table entry 1's destination MAC becomes `01:00:00:00:00:01`, and
  `TXPKT_CFG_SEL_{SW,HW}` are set for both queues.

The state is left set after the run, as production fabric leaves it, and
`tt-smi -r` is the clean slate. Nothing restores it: neither the router's
`teardown()` nor Metal's active-ERISC firmware, which writes no TXQ, RXQ or
classifier register (`tt_metal/hw/firmware/src/tt-1xx/active_erisc.cc`).
Whether base firmware re-initialises it is unknown, because base firmware is a
binary. What breaks if it is left set is undocumented; later Metal sessions
work after production fabric runs, which is evidence and not proof. This is an
accepted risk.

`EthernetTxRx.md` § Associating TX queues with RX queues says firmware
configures the TXQ n ↔ RXQ n association, destination MAC and classifier
included. That contradicts tt-metal, which reprograms the association on every
fabric initialisation.

Base firmware does not time-share ERISC0 while E0 runs. Metal's `active_erisc`
firmware disables interrupts and owns the loop, and base firmware code runs only
when the kernel yields: `risc_context_switch`, `run_routing`, or the
`ctx_switch` variants of `eth_send_packet` and `eth_write_remote_reg`
(`tunneling.h`). The unsafe sends E0 uses never yield. E0 therefore sees no
base-firmware interference, and also gets no link-status check (debounced to
once per 1000 ms, `eth_fw_api.h`) and no mailbox service for the run. The
consequence of skipping them is undocumented. Runs are kept to seconds and the
benchmark adds no yields; this is an accepted risk. A benchmark that yields
copies the router's coordinated switch through stream 30's scratch register
(`eth_retrain_link_sync_stream_id` in `erisc_datamover_builder.hpp`), and
budgets stream 30.

Every send on either queue waits for `eth_txq_is_busy()` on that queue to
clear. The check is mandatory on Blackhole, where reads of `ETH_TXQ_CMD` return
zero, so the assertion inside the unsafe send does not establish readiness.
TXQ 2 cannot send MMIO writes and is not used.

Each ERISC uses a fixed NoC. In dual-ERISC mode tt-metal requires an ERISC0
kernel in dedicated NoC mode to use NoC 0 and an ERISC1 kernel to use NoC 1, and
fails kernel creation otherwise ("In dual ERISC mode, ERISC0 in dedicated mode
must use NOC0"). E0's R3 therefore travels on NoC 0, and E1's W5 and W6 on
NoC 1.

E0 and E1 address the X280 with the same translated coordinate.
`Coordinates.md` gives both NoCs one coordinate system under translation except
where Y < 2 and X ≠ 8; the host refuses that region on NoC 1. The X280's TLB
windows to E0 and E1 use the Ethernet tile's virtual (translated) coordinate. A
mode-1 strict X280 window at an active Ethernet tile's translated coordinate
reaches ERISC L1 on silicon, and an active ERISC1 on NoC 1 reaches LIM at the
L2CPU's translated coordinate. With translation enabled (`NIU_CFG_0` bit 14), a
NIU always passes the X/Y of `NOC_TARG_ADDR_HI` through its translation tables
(`BlackholeA0/NoC/Coordinates.md`), so a NOC0 coordinate reaches an Ethernet
tile only through translation.

### Pre-run ordering

Each chip runs one `core_sync_barrier` (`kernels/shared/core_sync_barrier.hpp`).
It is a star: workers do local setup, announce arrival, and wait on their
release slot, while the director waits for every arrival and then releases.
Each participant calls `sync` once per launch, and the host zeroes the barrier
storage.

- **Director:** one forwarder hart per chip.
- **Workers:** E0, E1, and the other forwarder harts.

Before arriving, E0:

1. resets its tile's `pkts[p]` stream counters (below);
2. calls `eth_enable_packet_mode()`;
3. handshakes with the peer tile's E0 over TXQ 0;
4. arrives.

The handshake is a bounded copy of `erisc::datamover::handshake`'s sender and
receiver sides, in `kernels/eth_credit/metal_common.hpp`, with the same wire
protocol and addresses as tt-metal's multi-TXQ Ethernet unit test
(`eth_multi_txq_rxq_bidirectional.cpp`). The original has no timeout; on expiry
the copy fails with `STALL_TIMEOUT`, site `HANDSHAKE`.

The overlay streams belong to the tile, so E0 resets the `pkts[p]` counters that
its local E1 reads. It does so in kernel init, before the handshake, with the
router's `init_ptr_val`, which writes `STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX`
(`fabric_stream_regs.hpp`), then reads each counter back with `get_ptr_val` and
fails with `BAD_ARGS` if any is non-zero: a stale counter is otherwise a silent
two-ERISC hang, and before the handshake is the only point where the counters
are quiescent. No host path writes an Ethernet tile's stream registers; the
watcher skips Ethernet cores when it reads them. The peer E0 finishes the
handshake only after this tile's E0 reaches it, the peer E0 arrives only after
its handshake, and the peer's first W4 follows its chip's release. No +1
therefore lands before the reset. E1 does not touch the counters before
release.

E0 is launched on both tiles in every run. On a tile whose outbound direction is
idle, it does its setup, handshake and barrier, then exits.

All pre-run ordering goes through the barrier:

- E1 touches TXQ 1 only after release.
- Forwarders finish setup before arriving.
- Every participant starts traffic only after its chip's barrier releases.

The barrier's tokens carry its own per-launch run nonce; the FIFOs carry none.
Metal-side barrier waits are bounded by the transport's `iteration_cap`, 1e8
polls by default, which the host deadline covers; `bounded_poll` never yields.
X280-side waits end on host STOP. A barrier `StopRequested` fails with
`STALL_TIMEOUT`, site `BARRIER`, and `ConfigError` with `BAD_ARGS`.

Active ERISCs are barrier workers under an X280 director with Word arrival,
because an X280 director cannot host a counter. `MetalTransport<ACTIVE_ETH>`
(`kernels/shared/core_sync_transport_metal.hpp`) covers them: its semaphore
base, per-processor scratch for the inline write, and roster mapping
(`harness/regions/core_sync_roster.hpp`).
`X280CoreSyncDirector.ActiveEriscWordWorkersOnBothNocs` exercises both ERISCs of
one active core (NoC 0 and NoC 1) as workers.

### W4: stream-register counter over TXQ 0

W4 is a TT-link MMIO write (`ETH_TXQ_CMD_START_REG`) of +1 into the peer tile's
`pkts[p]` stream register, through its increment-on-write port
(`STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX`). It arrives through
the peer's RXQ 0. The sequence per packet is the router's `send_next_data`: wait
for `eth_txq_is_busy()`, `eth_send_packet_bytes_unsafe` for the data, wait
again, then the counter update.

This is correct under the documented TT-link behaviour:

- the receiver accepts packets only in sequence, so W4 is never accepted ahead
  of W3;
- the MMIO write carries its value inline and has no L1 source to reread on
  resend;
- the receiver discards duplicates, so a resent +1 is not counted twice.

The remaining assumption is that the RX queue's L1 write for W3 is visible
before the MMIO write for W4 takes effect. The Ethernet documentation does not
state it; the admission test fails to refute it on silicon (§ Cases). It is the
same ordering tt-metal fabric's router relies on (data, then
`remote_update_ptr_val`, on one TXQ). E0's wait between W3 and W4 orders enqueue
only: `ETH_TXQ_STATUS_CMD_ONGOING_BIT` clears once the packet metadata is latched
(`BlackholeA0/EthernetTile/EthernetTxRx.md` § `ETH_TXQ_CMD`). Wormhole states "no
guarantees are provided between different kinds of packet"
(`WormholeB0/EthernetTile/EthernetTxRx.md` § RX Queue Memory Ordering). The
admission test's tail marker checks only each slot's last 16 bytes, so it
detects an early counter but not a partly landed slot. A guarantee would need E1
to wait for `ETH_RXQ_OUTSTANDING_WR_CNT == 0` before reading the slot.

### R2: link read pointer over TXQ 1

R2 is a 16-byte TT-link L1 write of E1's read pointer for `rx[p]` into the peer
E0's `crd[p]`.

R2 is an L1 write because MMIO writes can travel only on TXQ 0. At most one RX
queue per tile accepts TT-link MMIO writes (`EthernetTxRx.md` § RX Queues), and
TXQ n pairs with remote RXQ n (§ Associating TX queues with RX queues), so MMIO
reaches the peer only through TXQ 0 → RXQ 0. The L1 write is safe for a read
pointer and not for a write pointer. A value reread on resend is only ever the
same or newer, and a newer read pointer is still a valid credit; a newer write
pointer would publish a slot whose data has not landed. R2 needs no ordering
against any data write.

### Pointers

Every pointer is a `uint32` over `0 … 2N−1` for its FIFO's depth N, and every
FIFO depth is a power of two:

```text
slot(x)  = x & (N − 1)
used     = (wr − rd) mod 2N
empty    = rd == wr
full     = used == N
```

Comparisons are equality or `(a − b) mod 2N`, so pointers are correct across
wrap. `N_LINK` is the depth of `tx_fifo[p]` and of its link FIFO `rx[p]`;
`N_DELIVER` is the depth of each forwarder's inbound FIFO `in[p]`.

Every producer and consumer checks distance: `(wr − rd) mod 2N ≤ depth` for each
pointer pair it reads, and `pkts[p] ≤ N_LINK` at E1. E0 and E1 also check that
the forwarder's W2 and R1 pointers only advance from the value they last
accepted, by at most the room E0 had granted or the headers E1 had delivered
(`credit_pair_ok`, as packet_injection's producer checks its credit), so an
older pointer landing after a newer one is refused rather than used. A
violation fails with `COUNTER_DISTANCE`.

### Ordering on the NoC

The NoC does not order two writes to non-overlapping L1 ranges at the
destination, even on the same route and virtual channel. Every NoC hop
therefore waits for its data writes to be acknowledged before writing the
pointer that publishes them:

- E1: W5, `noc_async_write_barrier()`, then W6.
- Forwarder: W1 through a strict TLB window, whose store completes only when the
  NoC write has, then W2.

`BATCH` amortises the wait: a producer writes up to `BATCH` slots, waits once,
then publishes one pointer covering them. A final batch may be partial.

Readers order their loads: a fence before the shadow load, and a fence between
the shadow load and the slot loads it covers, as `ordered_produced_load` and
the slot-load fence in `kernels/packet_injection/x280_consumer.hpp` do.

### Bidirectional operation

On each tile E0 serves the outbound direction and E1 the inbound one; each
forwarder sends and receives in one loop. The two ERISC kernels on a tile share
its L1, but every pointer still has one writer, so they need no cross-core
synchronisation beyond the barrier. Stall counts are per core; instrumenting
bidirectional runs stays open because both directions share the link.

### Layout rules

- Each pointer occupies its own aligned 16-byte unit. The NoC writes L1 in
  atomic 16-byte units, and Blackhole requires a NoC write's source and
  destination to be congruent mod 16, or the write never lands.
- A local pointer and its peer's shadow copy sit at the same offset mod 16. The
  host checks every pair before launch.
- X280 FIFOs and shadows live in LIM.
- ERISC poll loops call `invalidate_l1_cache()` (a `fence` on Blackhole) before
  rereading remotely written L1, because the baby-RISCV L0 data cache is not
  coherent with remote writes.

### Stream-ID budget

An Ethernet tile has 32 overlay streams (`ETH_NOC_NUM_STREAMS`). With no fabric
running, IDs 1–31 are available. Stream 0 is left alone because tt-metal uses
its registers as scratch (`get_cq_finish_ptr`, `get_sync_register_ptr`). The
design needs P stream counters per tile, the `pkts[p]` counters W4 targets; R2's
read pointers live in L1. That base firmware claims no stream is inferred, not
documented: the router assigns IDs 0–31 while base firmware runs.

## ABI and layout

The benchmark's symbols use the prefix `EC_`/`ec_`.

### Header

Packets carry the shared shadow header `fs_hybrid_mesh_packet_header_t`
(`kernels/shared/fabric_shadow_header.hpp`), x280's 96-byte copy of the
hybrid-mesh packet header:

| Field | Offset | Use |
|---|---|---|
| `seq` | 8–11 | sequence number per `p` |
| `payload_size_bytes` | 40 | as in production |
| `src_ch_id` | 43 | producer `p` |

`seq` occupies the unused tail of `NocCommandFields` after the 8-byte unicast
command header. Production does not read these bytes for unicast writes; the
`seq` accessor in `kernels/eth_credit/eth_credit_header.hpp` says so, and a
`static_assert` pins the offset. `fabric_shadow_header.hpp` carries the
production fields' offsets (`FS_PAYLOAD_SIZE_BYTES_OFFSET`,
`FS_SRC_CH_ID_OFFSET`) and accessors.

`fs_seed_unicast_header` sets the destination and rewrites all 96 bytes, so the
forwarder writes `src_ch_id` and `seq` after it. The router's ERISC also writes
`src_ch_id` (`send_next_data`), as a sender-channel index; that is precedent for
using the field, not for this meaning of it.

A slot is the 96-byte header followed by the payload, so `SLOT_SIZE = 96 + 16k`,
with the payload at most 65,520 bytes because `payload_size_bytes` is a
`uint16`. W1 and W5 are 96-byte writes. W2 and W6 follow the acknowledgement
wait, so a header is never read before it lands.

### Pointers and counters

- Every pointer is a 2N-wrap `uint32` (§ Pointers).
- `pkts[p]` holds the pending count: +1 remotely from E0, −1 locally by E1. It
  never exceeds `N_LINK`, well within the 17-bit register.
- `crd[p]` in E0's L1 is E1's read pointer for `rx[p]`, forwarded unchanged as
  R3.
- Each inbound FIFO has one source, so the receiver checks sequence numbers per
  `p`. It checks `src_ch_id < P` before indexing by it.

### Memory map

Each core holds a control area of 16-byte units followed by its slot arrays.

Each Ethernet tile has one control area, 512-byte aligned, grouped by accessor
with hot units first. Under a low-order 16-byte bank interleave, hot unit i lands
in bank i. The interleave function `bank = (addr >> 4) & 31` is inferred from
the Tensix L1 interleave, not documented for Blackhole Ethernet L1 (32 banks of
1024 × 128 bits); the comment at the control-area layout in
`kernels/eth_credit/eth_credit_abi.h` says so. E0's and E1's slot regions follow
one after the other, each sized to need.

| Core | Control units | Slots |
|---|---|---|
| E0 | `wr_sh[p]`, `crd[p]`, `rd_src[p]`, result | `tx_fifo[p]`, `N_LINK` each |
| E1 | `out_rd_sh[p]`, `crd_src[p]`, `out_wr_src[p]`, result | `rx[p]`, `N_LINK` each |
| Forwarder `p` (LIM) | `rd_sh[p]` (R3), `wr_src` (W2 source), `wr_sh` (W6), `rd_src` (R1 source), `seq_expected`, result | W1 header staging; `in[p]`, `N_DELIVER` × 96 B |

E0 and E1 each have 3P hot units, so an Ethernet tile has 6P, at most 24, and
every hot unit has its own bank at every P.

Cold units follow the hot units: the core_sync release slots, one 32-byte
`handshake_info_t` unit at the same address on both tiles, which admission
checks, and the two result blocks.

The X280 LIM arena is, in order: the per-hart argument records, the ready
table, and the P forwarder areas. It is placed in the runtime's available LIM
window (`available_lim()`), the same base on both chips, so it moves with the
runtime's LIM reservation and no LIM address is compiled into the image. The X280 TLB reservation is
`X280_TLB_ETH_CREDIT_BASE = 161` in `include/rt/x280_noc_map.h`, one Ethernet
window per hart, with `static_assert`s that it sits above the packet-injection
and core_sync blocks and below the host block.

Fit: a bidirectional run holds `tx_fifo` and `rx` for P on each tile, so tile
slots take `2 · P · N_LINK · SLOT_SIZE`. At P = 4 and `N_LINK = 8` that is
64 slots: about 262 KiB at 4 KiB payload and about 134 KiB at 2 KiB, both within
the 359 KiB window. `N_LINK = 16` at 4 KiB payload needs about 524 KiB and does
not fit.

### Terminal status

Each kernel records its terminal state only in its own result block, as a
16-byte `{uint32 state, uint16 origin, uint16 code, uint32 detail, uint32
reserved}`. `state` is 0 (running, host-initialised), 1 (`DONE`) or 2
(`FAILED`); `origin` is the kernel's own id. No kernel writes or reads another
kernel's status: a failed kernel's neighbours exit on their own stall budgets,
and the host attributes the failure (§ Failure and termination).

- Kernel ids: `X280_FWD_p`, `E0`, `E1`, plus a chip bit. Each chip has an E0,
  an E1 and P forwarders, so an id names which chip's core it is.
- Error codes: `NONE`, `STALL_TIMEOUT` (detail = stall site), `BAD_SEQUENCE`
  (detail = expected), `COUNTER_DISTANCE`, `LINK_DOWN`, `BAD_ARGS`.
  - A barrier `StopRequested` or a host STOP is `STALL_TIMEOUT` with the site as
    detail; a barrier `ConfigError` is `BAD_ARGS`.
  - A `src_ch_id` that is out of range or names another forwarder is
    `BAD_SEQUENCE`, and so is a tail-marker mismatch, with detail = the expected
    ordinal.
  - Besides the per-loop stall sites, the one-shot sites are `BARRIER`,
    `HANDSHAKE`, `DRAIN` and `E0_R3_SOURCE` (E0 waiting for its R3 source unit
    to be read out before rewriting it).
- Stall loops check their budgets and link state at a poll cadence, following
  `packet_injection`'s `amortized_safety_check`.

### Result block

One per kernel, and one per forwarder hart. Each is written only by its kernel,
in the cold part of its control area, with a fixed maximum size. It holds
aggregates only:

- `{magic, kernel id, terminal status}`;
- packet counts in and out per `p`;
- per stall site, a stall count and total ticks, where a stall is a pass with no
  progress (the router's progress flag). `FWD_W1_STORE`, `E0_TXQ0_BUSY`,
  `E1_TXQ1_BUSY` and `E1_NOC_BARRIER` are cost sites instead: each runs right
  after its own issue, so nearly every batch finds work outstanding (a strict
  store always completes). They count batches and the ticks spent.
  `ec_site_is_cost()` in `eth_credit_abi.h` names them, and the host's
  `stall_ticks()` sums stall sites only;
- first and last timestamps on the kernel's own clock, stamped per batch (after
  a batch completes), not per packet;
- failure detail: for `BAD_SEQUENCE`, `p`, expected and seen; for timeouts, the
  last local and shadow pointers.

| Kernel | Stall sites |
|---|---|
| X280_FWD | outbound FIFO full, W1 store time, inbound FIFO empty |
| E0 | inputs empty, link credits exhausted, TXQ 0 busy |
| E1 | link empty, `in[p]` full, NoC barrier, TXQ 1 busy |

### Arguments

- **Shape** (P, `N_LINK`, `N_DELIVER`, `SLOT_SIZE`, `BATCH`):
  - ERISC kernels take it at compile time, JIT-built per configuration. The host
    also passes it as runtime arguments, and the kernel compares them with its
    compile-time values, failing with `BAD_ARGS` on a mismatch.
  - Forwarders take it at run time, bounded by `EC_MAX_P = 4`, and validate it
    before their first send.
  - Admission requires `1 ≤ BATCH ≤ min(N_LINK, N_DELIVER)`, depths at most
    `EC_MAX_DEPTH = 1024`, and a poll cadence at most 2^20. The admission test
    (`EC_TAIL_MARKER`) additionally requires at least one slot above
    `ETH_TXQ_MAX_PKT_SIZE` (4114) and a nonzero register reading.
- **Placement and control** are runtime: addresses, per-role peer NoC
  coordinates, packet counts, stall budgets and kernel id. A packet count of
  zero per `p` is admitted, which runs the barrier and handshake with no
  traffic.
- **Bring-up modes** are E0/E1 compile-time defines, off by default and JIT-built
  per case: `EC_E0_SINK_LOCAL`, `EC_E0_SOURCE_SELF`, `EC_E1_SINK_LOCAL`,
  `EC_TAIL_MARKER` and `EC_E0_COUNTER_FIRST_DELAY`.
- **One struct per role** (`ec_x280_fwd_args_t`, `ec_e0_args_t`,
  `ec_e1_args_t`), each starting `{magic, kernel_id}`. There is no schema word:
  any change to an args or result record changes its magic (`EC_ABI_MAGIC`
  'ECB3', `EC_RESULT_MAGIC` 'ECR3'), so a host and an image built from different
  layouts reject each other's records. Each kernel derives its tx_fifo from
  `ctrl_base` and the layout offset, and E0 range-checks the derived base.
  - E0 and E1 embed one shared prefix, `ec_erisc_args_t`, first in their
    records; the role-specific fields follow it. Every Metal record is all
    `uint32_t` with no padding, so runtime word i is its i-th field: the host
    serializes by copying bytes (`pack_runtime_args`) and a kernel reads the
    words straight into its record (`read_args`). Size and offset asserts in
    `eth_credit_abi.h` pin that boundary. Metal's portable runtime-argument
    ceiling is 341 words (`tt_metal/api/tt-metalium/kernel_types.hpp`).
  - Barrier worker indices are not arguments: each ERISC kernel arrives at its
    role's fixed index (`ec_barrier_index_e0/e1()`), and its role is checked
    against `kernel_id`.
  - The device validators refuse stream ids outside 1–31 and duplicates within a
    role's `stream_id` or E0's `local_stream_id`, as host admission does.
  - Every record carries `lim_end`, the end of the available LIM window the host
    placed the X280 layout in. `ec_narrow_lim` lowers each kernel's compiled LIM
    bound (`X280_LIM_END`, the reset shape) to it, so a LIM way the active shape
    does not back is refused.
- One host function derives every role's arguments from a single layout result.

### Files

- `kernels/eth_credit/eth_credit_abi.h` (C11/C++17, no Metal includes):
  constants, maxima, enums, status/result/args structs, checked sizing formulas,
  `static_assert`s, the banking comment.
- `kernels/eth_credit/eth_credit_header.hpp`: the `seq` accessor over the shared
  shadow header.
- `kernels/eth_credit/{e0,e1}_kernel.cpp`, and `kernels/eth_credit/metal_common.hpp`,
  which they share (bounded handshake, status and result writes).
- `kernels/eth_credit/x280_fwd.hpp` and `src/eth_credit_x280_fwd.cpp`: the X280
  forwarder and its image, `firmware-eth-credit-x280-fwd.bin`.
- `tests/device/suites/eth_credit/regions.hpp`, admission and layout:
  power-of-two depths, `SLOT_SIZE` form, maxima, `BATCH` bounds; fit per window,
  reporting the first failing constraint and bytes over; 512-byte control
  alignment and mod-16 congruence; stream IDs 1–31 and arguments ≤ 341 words;
  per-role peer NoC coordinates.
- `tests/device/suites/eth_credit/{run,records}.hpp` and
  `tests/device/test_eth_credit_hw.cpp`: the host sequence and the cases.
- `tests/host/test_eth_credit.cpp`: host-only tests of congruence, fit and
  rejection, argument serialization (each runtime word is the field its
  declaration order gives), 2N-wrap arithmetic across many wraps, header
  offsets, bank placement (every hot control unit on a distinct
  `(addr >> 4) & 31` at every P ≤ 4, tied to the inferred mapping), per-role peer
  coordinates and `BATCH` bounds.
- `tools/eth_credit_jit_check.sh`: the CTest `x280_eth_credit_jit` (below).

`CreateKernel` takes `kernels/eth_credit/{e0,e1}_kernel.cpp` directly.
`EthernetConfig` carries no `compiler_include_paths`, so an Ethernet kernel's
only extra include root is its source directory. Headers reached from these
kernels therefore include every other x280 header by relative path:
`kernels/shared/core_sync_transport_x280.hpp` includes
`../../include/rt/x280_noc_map.h`, and `eth_credit_abi.h` includes
`../shared/core_sync_layout.h`.

`X280_ETH_CREDIT_KERNEL_ROOT` points the JIT at another tree holding
`kernels/eth_credit/`, so a mutation check compiles an edited copy without
editing this one.

## Host sequence

The ERISC kernels run under Metal and the forwarders under the X280 runtime, in
one host process, under slow dispatch (the executable links
`x280_slow_dispatch`). No production fabric runs alongside; the process owns the
whole Metal session.

1. Require Blackhole, two directly connected chips in one `MeshDevice`, and two
   `ACTIVE_ETH` processors reported by the HAL. That needs Ethernet firmware
   1.7.0 or later (below it tt-metal falls back to single-ERISC mode with a
   warning, `llrt/firmware_capability.cpp`) and `TT_METAL_DISABLE_MULTI_AERISC`
   unset. The suite uses chips 0 and 1 (`kChipPair`).
2. Find a live link: `get_active_ethernet_cores(true)` per device,
   `Cluster::is_ethernet_link_up`, then `get_connected_ethernet_core`, checking
   that the reverse mapping returns the original endpoint and excluding reserved
   tunnel cores. Metal's `get_connected_ethernet_core` throws for a core whose
   peer is a chip outside the visible cluster, so on a host where chip 0 or 1
   also links to another card, run with every card visible.
3. Choose the X280 tile on each chip (L2CPU 0), its P forwarder harts, and the
   director hart.
4. Check that both tiles' Ethernet queues are idle: `ETH_TXQ_STATUS` cmd-ongoing
   clear on TXQ 0 and TXQ 1, and `ETH_RXQ_OUTSTANDING_WR_CNT` zero, read through
   `Cluster::read_reg`. Fail the run before launch otherwise. The check is
   necessary, not sufficient: a TXQ keeps resending unacknowledged packets after
   `CMD_ONGOING` clears (`EthernetTxRx.md` § ETH_TXQ_CMD, § TX Queues), and no
   Blackhole register counts them. So a run that ends undrained (a Metal throw,
   an X280 deadline the case did not plan, an E0/E1 without a finished result,
   `LINK_DOWN`, or a retrain-count change) makes the suite refuse every later
   launch in the process; a planned negative control does not. Differing LIM
   bases between the chips and a missing tile also fail before launch.
5. Lay out every FIFO, shadow pointer, result block and the barrier storage.
   Check alignment, `BATCH` bounds, and that every base (Ethernet control, LIM)
   and stride (FIFO, forwarder area, slot) is a multiple of 16
   (`check_congruence`, `EC_CHECK_CONGRUENCE` naming the offending value): every
   transfer address is a base plus those strides and 16-byte-unit offsets, so
   every source and destination is then congruent mod 16. A window whose
   placement cannot end at or below 2^32 is refused on its base constraint
   before any alignment arithmetic. Derive each role's peer NoC coordinates for
   its NoC.
6. Initialise everything a peer writes (shadows, slots, barrier storage), and
   read it back. A read-back that is not all zero, on either chip's ERISC control
   area or LIM arena, fails the run before any Metal or X280 enqueue. Kernels
   initialise only state they alone write, and E0 resets its tile's stream
   counters (§ Pre-run ordering). A counter or pointer left stale from a previous
   run hangs two-ERISC operation; initialising and reading back every
   peer-written counter and pointer, `crd[p]` included, guards against it.
7. Compile every chip's program (`Program::impl().compile`, idempotent) before
   any enqueue. Slow dispatch otherwise compiles each chip's program at its own
   launch. In one direction chip A launches E0 alone and chip B launches E0 and
   E1, so a fresh E1 build would delay chip B's E0 past the 2 s handshake budget
   and both E0s would fail `STALL_TIMEOUT(HANDSHAKE)`.
8. Enqueue the Metal `MeshWorkload` (E0 on both tiles, E1 where a direction is
   active) without blocking, enqueue both X280 sessions, then wait on all of
   them. Enqueue and `Finish` are wrapped: a Metal failure is recorded as
   `RunResult::metal_failure` and in the JSONL, and result harvesting still runs.
9. Read and validate every result block, and attribute any failure (§ Failure
   and termination). Record the firmware link retrain count
   (`eth_live_status_t::retrain_count`, `hal.get_eth_fw_mailbox_val`) on both
   ERISCs before launch and after `Finish`, in an `eth_credit_link` JSONL record
   with E0's terminal status. It is one word,
   `MEM_SYSENG_BOOT_RESULTS_BASE` 0x7CC00 + 128 + 256 + 128 = 0x7CE00 =
   `MEM_RETRAIN_COUNT_ADDR` (`HalL1MemAddrType::RETRAIN_COUNT`), sizes
   static_asserted in `eth_fw_api.h`; the suite asserts the two HAL addresses
   coincide.

## Failure and termination

- Each kernel bounds every phase, the timed loop included, with a stall budget
  in its own clock: ticks without progress, reset on progress. Stall budgets are
  equal across kernels. The host's X280 wait is derived from the stall budget
  and admission-checked before launch, as in `packet_injection`:
  `Config::host_wait_ms` (0 derives it) must be at least `kHostWaitPhases` (4:
  E0's handshake, the timed loop, E0's final drain, and a neighbour that notices
  only on its own budget) × `stall_budget_ms` + `kHostWaitSlackMs` (5000, for
  launch and a healthy run's traffic), else `BAD_PARAM(HOST_WAIT)`. A budget
  reset on progress does not bound a healthy run's total time, so the default is
  a no-progress floor plus slack, not a total-run deadline: a case whose traffic
  can exceed the 5 s slack passes `host_wait_ms` explicitly.
- A failing kernel records `FAILED` with its own error in its own result block
  and exits; it tells no neighbour. Neighbours exit on their own stall budgets,
  and the host attributes the failure from the result blocks.
- The kernels' own ERISC waits are bounded: TXQ busy, NoC barrier, and the R3
  source-unit wait. Two Metal-internal waits are not: every `noc_async_write`
  issue spins on `noc_cmd_buf_ready`, and the core_sync arrival's `notify_write`
  spins in the write barrier. A wedged NoC therefore hangs an ERISC past its
  budget; the host's wait in `Finish` is then bounded by the operation timeout
  (below).
- The in-loop link check reads `PCS_STATUS` directly on both ERISCs. tt-metal's
  `is_link_up()` on physical ERISC0 waits about 5.4e9 cycles whenever
  `PCS_STATUS != 1` (`eth_fw_api.h`), which would overrun the stall budget on a
  flapping link; ERISC1 takes the cheap arm.
- Every validation failure is `BAD_ARGS` in the kernel's own result block, which
  is initialised before the first early return, once `ctrl_base` is bounded.
  Every argument-derived write destination is bounded against its window.
- A forwarder in the barrier waits for host STOP when a peer fails there, and
  the host attributes the failure.
- A strict-window (mode 1) X280 store blocks in hardware until its NoC write
  completes. A forwarder store to a dead tile therefore hangs that hart, and
  only the host's X280 wait timeout bounds it.
- `Finish` takes no timeout argument. Under slow dispatch it waits once per
  device for the cores to finish, and `TT_METAL_OPERATION_TIMEOUT_SECONDS`
  bounds each of those waits from when it starts waiting, not from launch; the
  suite sets it to 120 itself. On expiry `Finish` throws `Timeout (N ms) waiting
  for physical cores to finish`. A trapped or wedged ERISC keeps running after
  the throw, and the device then needs `tt-smi -r`.
- The host passes a packet count per `p`. Each kernel stops once it has sent,
  forwarded or consumed that count for every `p`. E1 issues its final R2 for
  every `p` before exiting, and E0 waits for every R2 before exiting. A resend of
  a `tx_fifo` slot after E0 exits is harmless, because TT-link discards duplicate
  sequence numbers. The host checks that sent, forwarded at E0, forwarded at E1
  and consumed agree for every `p`.
- Each forwarder checks that each sequence number on `in[p]` is the previous plus
  one and fails with `BAD_SEQUENCE(expected, seen)` otherwise.
- Link-down, trap, timeout or partial drain is a failed run, never an empty or
  default result.

The host attributes a failure (`attribute()`, `RunResult::attribution`, printed
as one `attribution:` log line and written as an `eth_credit_attribution` JSONL
record):

1. Every kernel whose code is not `STALL_TIMEOUT` is an origin.
2. Without one, the host reports the timed-out kernels' stall sites.
3. A launched kernel that wrote no result is listed as missing, the prime
   suspect.

The host keeps the device's own recorded failure over a generic host timeout, as
the `packet_injection` host does.

## Instrumentation

- Throughput: (n − k1)/(`last_ts` − `first_ts`) on the receiving forwarder's
  clock, where k1 is the packets the forwarder had received when it stamped
  `first_ts` (its result block's `first_batch_packets`). The timestamps are per
  batch, so the window covers the packets after the first batch. The JSONL
  records `packets` = n and `first_batch_packets` = k1, and the offline tool
  derives n − k1. A request's useful bytes are the 96-byte header that reaches
  the X280, and the record names the payload bytes separately.
- Per kernel, on its own clock: stall counts and stall time per stall site, and
  per-packet service time, written as one `eth_credit_kernel` JSONL record per
  launched kernel: case, chip, role, kernel id, state, code, detail, packets in
  and out per `p`, each kept site as `{site, cost, count, ticks}`, `first_ts`,
  `last_ts` and failure detail. The `cost` flag lets a reader apply the "stalls
  least" rule from the record alone: the kernel that stalls least sits behind
  the bottleneck, and cost sites are excluded from that comparison and read as
  time spent.
- E0 classifies a pass that sends nothing by credit before input
  (`ec_e0_idle_site`): with no link credit it is `LINK_CREDITS`, else
  `INPUTS_EMPTY`. Zero credit implies zero available input, since the forwarder
  refills `tx_fifo` only as credit returns, so testing input first would charge
  every credit wait to `INPUTS_EMPTY`.
- Single-clock round trips: the ping-pong case's E0 window over n packets is n
  round trips.
- No timestamp from one core is subtracted from another's.
- Each record names both chips, both ERISC logical coordinates, the X280 tile,
  P, `N_LINK`, `N_DELIVER`, `SLOT_SIZE` and `BATCH`.
- The ERISC wall clock (`eth_read_wall_clock`) counts AICLK. It sits at the
  Tensix live-timer address (`ETH_RISC_REGS_START` 0xFFB10000 +
  `ETH_RISC_WALL_CLOCK_0` 0x21F0 = `TENSIX_TIMER_LOCAL_LOW_ADDR`,
  `tt_eth_ss_regs.h`), so the AICLK sampler of `docs/measurement.md`
  (`measurement::assess_aiclk_run`) aimed at the Ethernet core brackets it. Stall
  budgets use each chip's programmed AICLK, every run records per-chip ERISC
  clock evidence (`reference_counter: erisc_wall_clock`), and ERISC ticks
  convert to time only when that evidence is clean
  (`EriscWallClockTicksAtProgrammedAiclk`).

## Cases

`x280_eth_credit` runs its cases in bring-up order; each step carries a run that
must fail:

| Case | Checks |
|---|---|
| `BarrierAndHandshakeWithNoTraffic`, `BarrierAndHandshakeAtAFreshlyCompiledShape` | the per-chip barrier with every participant and no traffic, E0 and E1 as active-ERISC Word-arrival workers, plus `eth_enable_packet_mode()` and the cross-link handshake |
| `RefusedE0ArgumentIsAttributedToE0`, `WithheldPeerE0TimesOutTheHandshakeAndLeavesTheProcessClean`, `ReadbackMismatchOnEitherChipStopsBeforeEnqueue`, `UndrainedRunRefusesTheNextLaunch` | attribution and the pre-launch refusals; with one chip's E0 withheld, the other E0 fails `STALL_TIMEOUT(HANDSHAKE)` and the forwarders and E1 fail `STALL_TIMEOUT(BARRIER)` |
| `NocHopAlone` | forwarder → E0 with W1, wait, W2 and R3 credits (`EC_E0_SINK_LOCAL`), every header's `seq` checked at E0 |
| `EthernetHopAlone`, `SingleSlotPingPong` | E0 → E1 data with W4 on TXQ 0 and R2 on TXQ 1 (`EC_E0_SOURCE_SELF`, `EC_E1_SINK_LOCAL`), and single-slot round trips |
| `AdmissionTailMarkerHoldsWhenCounterFollowsData` | each W3 slot carries a per-packet marker in its last 16 bytes (`EC_TAIL_MARKER`), which E1 checks when `pkts[p]` shows the packet, including slots TXQ 0 splits into several packets; a pass fails to refute W3's L1 write taking effect before W4's MMIO write |
| `AdmissionPositiveControlCatchesPrematureCounter` | E0 sends W4 before W3 (`EC_E0_COUNTER_FIRST_DELAY`) and E1 fails `BAD_SEQUENCE`: the check observes a premature counter. It cannot force hardware to reorder an in-order W3/W4 pair |
| `FullPathForcedWrap`, `FourForwarders`, `BothDirections` | the full path with forced wrap of `N_LINK` and `N_DELIVER`, P forwarders, and both directions, validated by per-hop count conservation and per-`p` sequence checks |
| `EriscWallClockTicksAtProgrammedAiclk` | the ERISC clock evidence over 100k packets |
| `ThroughputBySlotBatchAndP` | the throughput sweep over slot size, `BATCH` and P |

The E0/E1 kernels are built by the Metal JIT at launch. `x280_eth_credit_jit`
(`tools/eth_credit_jit_check.sh`) compiles them without a device the way the JIT
does for an active ERISC: tt-metal's `active_erisck.cc` wrapper around the
kernel source, the JIT's flags and include roots, and the kernel's own directory
as its only extra root, for ERISC0 (NoC 0) and ERISC1 (NoC 1) at three shapes
and every bring-up mode. When Metal's runtime cache (`TT_METAL_CACHE`, else
`~/.cache`) holds active-ERISC firmware built from this checkout, each object is
also linked against it; the check reports E0 and E1 separately as "linked" or
"compile only". A cache entry matches by checkout path, not revision, so a link
result counts only after a Metal program has run from this build.

## Hardware facts

From the Tenstorrent ISA documentation:

- `BlackholeA0/EthernetTile/EthernetTxRx.md`
  - **TX Queues**: in TT-link mode the TX queue resends until acknowledged and
    rereads the packet body from L1 for every resend; the recipient discards
    packets with duplicate sequence numbers.
  - **RX Queues**: at most one RX queue per tile accepts TT-link MMIO writes.
  - **Associating TX queues with RX queues**: local TXQ n communicates with
    remote RXQ n.
  - **`ETH_TXQ_CMD`**: value 2 (L1 write) gives no indication of when L1 reads
    finish; value 4 (MMIO write, TXQ 0/1 only) takes its data from
    `ETH_TXQ_REMOTE_REG_DATA`, which software may change once metadata is
    latched.
  - **`ETH_TXQ_STATUS`**: `ETH_TXQ_STATUS_CMD_ONGOING_BIT` is set while a
    command is being accepted.
  - **RX Queue Memory Map**: `ETH_RXQ_LOCAL_RX_SEQ_NUM` is the next expected
    sequence number, and `ETH_RXQ_PACKET_DROP_CNT` counts packets discarded for
    sequence-number mismatch, so packets are accepted in order.
    `ETH_RXQ_BYTE_CNT`, `ETH_RXQ_PKT_END_CNT` and `ETH_RXQ_OUTSTANDING_WR_CNT`
    state that receipt runs ahead of the L1 commit.
- `BlackholeA0/EthernetTile/BabyRISCV/README.md`, **Memory map**: L1 is
  reachable by raw and TT-link L1 writes only; TT-link MMIO writes reach the
  register ranges from `0xFFB1_2000`, which include the overlay streams.
- `BlackholeA0/TensixTile/BabyRISCV/README.md`, **L0 Data Cache**: 64 bytes in
  four 16-byte lines, not coherent with other L1 writers, flushed by `fence`.
- `BlackholeA0/L2CPUTile/TLBWindows.md`: the `ordering` field names modes 0
  Default, 1 Strict AXI, 2 Posted Writes, 3 Counted Writes. The forwarder relies
  on mode 1, where a store's write response waits for NoC completion.
- `WormholeB0/NoC/Ordering.md`: two same-VC writes to non-overlapping L1 ranges
  have no commit-order guarantee; an MMIO write may race ahead of an earlier L1
  write; stronger ordering requires waiting for the response.
- `WormholeB0/NoC/Overlay/AsGeneralUse.md`: an idle stream's
  `STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX` is incremented by `j` by
  writing `(j << 6) + i` to the update register.

Measured on a Blackhole p150b pair:

- `ETH_TXQ_MAX_PKT_SIZE_BYTES` reads 4114 on TXQ 0 on both tiles. A larger W3
  leaves as several TT-link packets.
- Host MMIO reads of `ETH_TXQ_STATUS` and `ETH_RXQ_OUTSTANDING_WR_CNT` work
  through `Cluster::read_reg` on the Ethernet core.
- The ERISC wall clock counts AICLK (1350 MHz on both chips).
- W1 is 12 strict 64-bit window stores per 96-byte header, each one NoC round
  trip, so it is the forwarder's largest cost site.

Documentation gaps:

- Blackhole has no `NoC/Ordering.md`, no `NoC/Overlay/` (which
  `EthernetTxRx.md` links to), and no `Alignment.md`. The NoC and overlay facts
  above are Wormhole's, assumed to hold on Blackhole.
- The Ethernet documentation does not state that consecutive TT-link packets'
  L1 and MMIO writes take effect in order. The admission test checks it.
- `TLBWindows.md` names the four X280 ordering modes but defines none of them.

## Reuse

tt-metal:

- `tt_metal/hw/inc/internal/ethernet/tunneling.h`: `eth_txq_is_busy`,
  `eth_send_packet_bytes_unsafe`, `eth_write_remote_reg`.
- `tt_metal/fabric/hw/inc/edm_fabric/fabric_stream_regs.hpp`:
  `remote_update_ptr_val`, `increment_local_update_ptr_val`, `get_ptr_val`,
  `init_ptr_val`.
- `tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp`:
  `send_next_data`, the production data-then-counter sequence.
- `tt_metal/hw/inc/internal/tt-1xx/risc_common.h`: `invalidate_l1_cache`.
- `tests/tt_metal/tt_metal/eth/test_eth_multi_txq_rxq.cpp` with its
  `eth_multi_txq_rxq_bidirectional.cpp` kernel: active-link discovery and ERISC
  kernel launch, the cross-link handshake, data on one TXQ followed by
  `remote_update_ptr_val`, and 16-byte L1 acks on a second TXQ after
  `eth_enable_packet_mode`.
- `tt_metal/fabric/hw/inc/edm_fabric/fabric_txq_setup.h`: `eth_enable_packet_mode`.
- `tt_metal/fabric/erisc_datamover_builder.cpp`: the router's two-ERISC mode,
  which on Blackhole with two `ACTIVE_ETH` processors sets `receiver_txq_id = 1`
  and uses free-running L1 counters whenever the sender and receiver TXQs differ.
- `tt_metal/llrt/hal/tt-1xx/blackhole/bh_hal_active_eth.cpp`:
  `configure_for_2erisc()` gives `ACTIVE_ETH` two processors, ERISC0 and ERISC1.

This tree:

- `kernels/shared/core_sync_barrier.hpp`, with `core_sync_transport_metal.hpp`
  and `core_sync_transport_x280.hpp`: the per-chip barrier.
- `tests/device/suites/p2p_latency/metal_peer_runner.hpp` and
  `erisc_peer_tile.hpp`: an X280 session and a Metal-launched ERISC peer in one
  process, zero-and-read-back of peer regions, mod-16 congruence checks.
- `harness/execution/execution.hpp`: `enqueue_x280`, `wait_x280` and
  `cleanup_x280` as separate steps, so both chips' X280 sessions are enqueued
  before either is waited on.
- `tests/device/suites/packet_injection/regions.hpp`: phase and host-budget
  admission; `kernels/packet_injection/x280_consumer.hpp`: ordered loads.
- `kernels/shared/fabric_shadow_header.hpp`: `fs_hybrid_mesh_packet_header_t`
  and `fs_seed_unicast_header`.
- `harness/regions/regions.hpp`: `write_args`.
- `harness/regions/metal_l1_region.hpp`: `query_unreserved_window`, which
  accepts `HalProgrammableCoreType::ACTIVE_ETH`.
- `harness/session/session.hpp`: `OpenConfig::mesh_device_ids`, which builds a
  1×N `MeshDevice` with physical ids; the session sets the PLL on every chip in
  the mesh.

`MeshSocket` is not reusable for these endpoints. Its config and data buffers
are allocated on Tensix worker cores, and its coordinate conversion is
hard-wired to `worker_core_from_logical_core`. The fabric worker and EDM channel
classes carry routing, connection handshakes and fabric credit state that a
one-link benchmark does not need.

## Parameters

- P (≤ `EC_MAX_P`): forwarder harts per chip, each with its own `tx_fifo[p]` and
  `in[p]`
- `N_LINK` (power of two): depth of each `tx_fifo[p]` and its link FIFO `rx[p]`
- `N_DELIVER` (power of two): depth of each forwarder's inbound FIFO `in[p]`
- `SLOT_SIZE = 96 + 16k`
- `BATCH`, `1 ≤ BATCH ≤ min(N_LINK, N_DELIVER)`: up to this many slots written
  per wait and per pointer update, and packets per FIFO per round-robin turn
- Direction: one direction or both

## Running it

`x280_eth_credit` measures read/write-pointer credit flow control over one live
Ethernet link between UMD chips 0 and 1: X280 forwarders on each chip, E0 on
ERISC0 (NoC 0) and E1 on ERISC1 (NoC 1) of the linked active Ethernet core. Its
cases run in bring-up order, from the barrier and handshake with no traffic to
both directions, and each step carries a run that must fail. E0 and E1 are
JIT-built per shape from `kernels/eth_credit/{e0,e1}_kernel.cpp`. An Ethernet
kernel's only extra include root is its source directory, so they reach every
x280 header by relative path; the `x280_eth_credit_jit` CTest compiles them the
same way with no device. ERISC stall budgets
scale by each chip's programmed AICLK, and every run brackets both ERISC wall
clocks with the AICLK sampler of [measurement.md](../measurement.md); ERISC ticks convert to time
only when that evidence is clean. Each run also writes an `eth_credit_link`
record per chip with the base firmware's link retrain count before launch and
after `Finish`, so a failure can be checked against a retrain, and the chip's
ERISC clock evidence; and an `eth_credit_kernel` record per launched kernel
with its result block: status, packets per `p`, every stall and cost site's
count and ticks, and failure detail. The console prints raw packet counts and
ticks and, for a failed run, the attribution line and every chip's kernel
results; rates come from the tabulator. The manifest's `eth_credit.throughput`
configuration runs `ThroughputBySlotBatchAndP` with every card visible
(`TT_VISIBLE_DEVICES` unset), for the reason below. The design is above.

```bash
env -u TT_VISIBLE_DEVICES TT_METAL_RUNTIME_ROOT="$PWD" ./build/test/tt_metal/x280/x280_eth_credit
```

The suite uses L2CPU 0 of both chips; the harness boots resident idle on each
chip whose L2CPUs are all held in reset ([harness.md](../harness.md#running-benchmarks)). It links
`x280_slow_dispatch`, and it sets `TT_METAL_OPERATION_TIMEOUT_SECONDS=120`
itself, which bounds each device's wait inside a slow-dispatch `Finish`,
counted from when `Finish` starts waiting, not from launch; every E0/E1 wait's
own stall budget bounds the kernels themselves. Finding the link asks Metal for
each active Ethernet core's peer, and Metal refuses a peer outside the visible
cluster, so on a host where chip 0 or 1 also links to another card, run with
every card visible, as above.
A Metal call that throws is recorded as
`metal_failure` in the run's `eth_credit_link` records, and the result blocks
are still harvested. `X280_ETH_CREDIT_KERNEL_ROOT` points the E0/E1
sources at another x280 tree, which is how a mutation check compiles an edited
copy.

A run that ends undrained (a Metal call threw, an X280 wait expired that the
case expected to finish, an E0/E1 left no result, or the link went down or
retrained) may leave TT-link resends in flight: a TX queue keeps resending
unacknowledged packets after `CMD_ONGOING` clears, and no Blackhole register
counts them, so the idle-queue check before each run cannot rule them out. The
suite then refuses every later run in the process. A fresh process cannot see
that an earlier one ended undrained, so after one, or after a process killed
mid-run, reset both chips before the next run; the next run boots their
L2CPUs again:

```bash
env -u TT_VISIBLE_DEVICES tt-smi -r 0,1   # nothing may hold the cards
```
