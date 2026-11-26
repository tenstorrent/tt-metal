# NoC interface counters

Reference for the event counters of the NoC interface unit (NIU) of a tile and for `TT_METAL_PROFILE_NOC_COUNTERS`, the tt-metal runtime option that captures them per launch in the device profiler. [hardware_reference.md](hardware_reference.md) covers the Tensix MMIO counters and [riscv_pmc.md](riscv_pmc.md) the monitor inside the RISC-V cores; the NIU counters are a third, separate set. The option reads hardware counter totals per launch; `TT_METAL_DEVICE_PROFILER_NOC_EVENTS` is a different tool that records each NoC call a kernel makes.

## 1. Support matrix

| Architecture | Tile | NoCs | Capturing core | Captured by `TT_METAL_PROFILE_NOC_COUNTERS` |
|---|---|---|---|---|
| Blackhole | Tensix | 2 | BRISC | yes |
| Wormhole | Tensix | 2 | BRISC | yes |
| Quasar | Tensix | 1 | DM0 | yes |
| Wormhole, Blackhole | Ethernet | 2 | none | no |

Ethernet tiles are left out: their firmware has no launch window that spans every core of the tile, and fabric
and dispatch traffic through the NIU would swamp the kernel's own transfers. Quasar has one NoC in the register
headers (`NIU_NOCS` is 1).

## 2. What the counters are

Each NIU has a file of 32-bit counters per NoC, read as plain loads from the tile's own register space; any core of
the tile can read them. Nothing resets them, so a capture is a delta modulo 2^32. They count events, requests and
data words, and never stalls: a counter that does not move says the NIU saw no such event, not how long a core
waited.

Master side counters (`MST_*`) count what the tile issues, slave side counters (`SLV_*`) what the tile serves for
other tiles. A read request issued by this tile moves data in as `MST_RD_DATA_WORD_RECEIVED`; a read of this tile's
L1 by another tile moves data out as `SLV_RD_DATA_WORD_SENT`.

| Group (mask bit) | Counters |
|---|---|
| master (1) | `MST_RD_REQ_SENT`, `MST_RD_DATA_WORD_RECEIVED`, `MST_NONPOSTED_WR_REQ_SENT`, `MST_NONPOSTED_WR_DATA_WORD_SENT`, `MST_POSTED_WR_REQ_SENT`, `MST_POSTED_WR_DATA_WORD_SENT` |
| slave (2) | `SLV_RD_REQ_RECEIVED`, `SLV_RD_DATA_WORD_SENT`, `SLV_NONPOSTED_WR_REQ_RECEIVED`, `SLV_NONPOSTED_WR_DATA_WORD_RECEIVED`, `SLV_POSTED_WR_REQ_RECEIVED`, `SLV_POSTED_WR_DATA_WORD_RECEIVED` |
| atomic (4) | `MST_NONPOSTED_ATOMIC_SENT`, `MST_POSTED_ATOMIC_SENT`, `SLV_NONPOSTED_ATOMIC_RECEIVED`, `SLV_POSTED_ATOMIC_RECEIVED` |

The register indices are the same on all three architectures; the register address is
`NIU_STATUS_BASE + noc * NIU_NOC_STRIDE + index * 4` (`0xFFB20200` on Wormhole and Blackhole, `0x02000200` on Quasar,
stride `0x10000`).

When each counter moves is in the ISA documentation ([Blackhole](https://github.com/tenstorrent/tt-isa-documentation/blob/main/BlackholeA0/NoC/Counters.md), [Wormhole](https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/NoC/Counters.md)). A data word is one
flit at the destination: a transfer of `n` bytes to address `a` is `ceil(((a % F) + n) / F)` words, with `F` 64 bytes on
Blackhole and 32 on Wormhole (256 on Quasar, checked on the Quasar emulator). Words times `F` is the byte count of an
aligned transfer and an upper bound otherwise. The ISA documentation marks reading `MST_RD_DATA_WORD_RECEIVED` as
unsupported; on a Blackhole p100a it followed the flit count exactly for aligned, unaligned and one-byte reads, eight reads
in flight and reads from DRAM.

The metal firmware reads only the barrier counters (`RD_RESP_RECEIVED`, `NONPOSTED_WR_REQ_SENT`,
`WR_ACK_RECEIVED`, `ATOMIC_RESP_RECEIVED`, `REQS_OUTSTANDING_ID`); this capture adds the data word counters next to them.

## 3. Where the definitions live

The headers are in tt-llk (`tools/include/perf_counters/noc_counters_*.h`, namespace `llk::perf::noc`) and the host
side is `tools/python/tt_llk_perf/`; the two READMEs there list what each file holds. The metal side is
`tt_metal/hw/inc/internal/profiler/noc_counters.hpp`: marker ids, payload layout and emission.

Adding a counter: append the enumerator to `NocCounter`, add one `{NocCounter::NAME, index, NocGroup::GROUP}` entry to
the `noc_counters` table and bump the array size. Keep the counters of every group an even number per NoC (they are
emitted in pairs). The host parsers read the headers, so nothing else changes.

## 4. Using it in tt-metal

```bash
export TT_METAL_PROFILE_NOC_COUNTERS=3     # master and slave counters
```

The value is a group mask, hex or decimal (1 master, 2 slave, 4 atomic), or `on` for master and slave. `0` or unset
leaves the capture out entirely. Setting it forces the device profiler on. The mask is a compile-time
constant of the kernel and firmware builds and part of the build key, so changing it rebuilds both; the key is
unchanged when the option is off.

The counters belong to the tile, so one core captures once per launch. On Wormhole and Blackhole the BRISC firmware
reads the start values before it starts the TRISCs and NCRISC, and the BRISC kernel entry closes the window once NCRISC
and the TRISCs have finished, so it covers the whole launch; a launch in which BRISC runs no kernel, or runs a dispatch
kernel, publishes no capture. On Quasar the DM0 firmware opens it before the other cores are started and closes it
after they have finished. The launch message, run-time arguments and go signal arrive before the window opens; the
profiler's own writes to DRAM when its buffer fills, and host reads of the tile during the launch, are not separated
from the kernel's transfers.

Each launch emits one header marker and one marker per pair of counters, all `TS_DATA` with the payload in the
`data` column of `profile_log_device.csv` and the same timestamp; `tools/python/tt_llk_perf/noc_counters.py` has the
layout. The room for them is held back from the kernel's own zones, so a kernel that fills the profiler buffer still
gets them.

## 5. Reading the markers on the host

```python
from tt_llk_perf import noc_counters

launch = noc_counters.unpack_launch(header_9100, {0: pair_9101, 1: pair_9102})
noc_counters.decode_noc_launch("blackhole", launch)   # counts per NoC, bytes and request totals
```

`noc_metrics.compute_noc_metrics(counts, word_bytes)` gives, per NoC: `noc_issued_read_bytes`,
`noc_issued_write_bytes` (this tile's own transfers), `noc_served_read_bytes`, `noc_served_write_bytes` (other
tiles' transfers with this tile), `noc_bytes_in` (data arriving: issued reads plus served writes), `noc_bytes_out` (data leaving: issued writes plus served reads),
the issued request counts and bytes per issued request. A metric whose group was off reads `None`.
`perf_counter_analysis.decode_noc_counter_markers(rows, arch)` returns one row per launch and NoC.

For example, `metal_example_eltwise_binary` (64 tiles of 2 KiB per buffer, core 0,0) reads 2 x 128 KiB on NoC 0,
`MST_RD_REQ_SENT` 128 and `MST_RD_DATA_WORD_RECEIVED` 4096, and writes 128 KiB on NoC 1, `MST_NONPOSTED_WR_REQ_SENT` 64
and `MST_NONPOSTED_WR_DATA_WORD_SENT` 2048.

## 6. Open questions

- Capturing Ethernet tiles.
- Wiring `decode_noc_counter_markers` into the ops report.
