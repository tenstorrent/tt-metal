# NoC interface counters

Reference for the event counters of the NoC interface unit (NIU) of a tile and for `TT_METAL_PROFILE_NOC_COUNTERS`, the tt-metal runtime option that captures them per launch in the device profiler. [hardware_reference.md](hardware_reference.md) covers the Tensix MMIO counters and [riscv_pmc.md](riscv_pmc.md) the monitor inside the RISC-V cores; the NIU counters are a third, separate set.

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

One data word is a fixed number of bytes per architecture:

| Architecture | Bytes per data word | Status |
|---|---|---|
| Blackhole | 64 | measured on a card: a 128 KiB read is 2048 data words |
| Wormhole | 32 | assumed from the 256-bit NoC payload width, not yet checked on a card |
| Quasar | 256 | assumed from the 2048-bit NoC payload width, not yet checked on an emulator |

The metal firmware reads only the barrier counters (`RD_RESP_RECEIVED`, `NONPOSTED_WR_REQ_SENT`,
`WR_ACK_RECEIVED`, `ATOMIC_RESP_RECEIVED`, `REQS_OUTSTANDING_ID`); this capture adds the data word counters next to them.

## 3. Where the definitions live

The hardware description is in tt-llk (`tools/include/perf_counters/`, namespace `llk::perf::noc`, self-contained):

| Header | Holds |
|---|---|
| `noc_counters_types.h` | `NocCounter` (names, append only), `NocGroup`, the group masks, `Entry` = `{counter, status register index, group}` |
| `noc_counters_blackhole.h`, `noc_counters_wormhole.h`, `noc_counters_quasar.h` | `noc_counters` table in capture order, `NIU_STATUS_BASE`, `NIU_NOC_STRIDE`, `NIU_NOCS`, `NIU_DATA_WORD_BYTES` |
| `noc_counters_inventory.h` | Picks the arch header from `ARCH_*`; `selected_regs<GroupMask>()` is the compile-time list of registers of the selected groups |
| `noc_counters_hw.h` | `read(noc, reg)` |

The metal consumer is `tt_metal/hw/inc/internal/profiler/noc_counters.hpp`: marker ids, payload layout and emission.
The hooks are in `tt_metal/hw/firmware/src/tt-1xx/brisc.cc` and `tt-2xx/dm.cc`; the option is parsed in
`tt_metal/llrt/rtoptions.cpp` and becomes `-DPROFILE_NOC_COUNTERS=<mask>` in `tt_metal/jit_build/build.cpp`. The host
side is `tools/python/tt_llk_perf/`: `headers.py` parses the tables, `noc_counters.py` decodes the markers and
`noc_metrics.py` derives byte and request totals; `tools/tracy/perf_counter_analysis.py` has
`decode_noc_counter_markers`.

Adding a counter: append the enumerator to `NocCounter`, add one `{NocCounter::NAME, index, NocGroup::GROUP}` entry to
each arch table and bump the array size. Keep the counters of every group an even number per NoC (they are emitted in
pairs). The host parsers read the headers, so nothing else changes.

## 4. Using it in tt-metal

```bash
export TT_METAL_PROFILE_NOC_COUNTERS=3     # master and slave counters
```

The value is a group mask, hex or decimal (1 master, 2 slave, 4 atomic), or `on` for master and slave. `0` or unset
leaves the capture out of the firmware entirely. Setting it forces the device profiler on. The mask is a compile-time
constant of the firmware and part of the firmware build key, so changing it rebuilds the firmware; the key is
unchanged when the option is off.

The counters belong to the tile, so one core captures once per launch: BRISC on Wormhole and Blackhole, DM0 on
Quasar. The window opens before the other cores of the tile are started and closes after they have finished, so it
covers the whole launch of the tile. The profiler's own writes to DRAM when its buffer fills, and the dispatch
traffic around the launch, are not separated from the kernel's transfers.

Each launch emits one header marker and one marker per pair of counters, all `TS_DATA` with the payload in the
`data` column of `profile_log_device.csv`, all with the same timestamp:

| Marker id | `data` |
|---|---|
| 9100 | `(noc count << 32) \| group mask` |
| 9101 + k | `(counter[2k + 1] << 32) \| counter[2k]` |

`counter[]` holds the per-NoC deltas, NoC 0 first, each NoC in the table order of the selected groups, so a mask of 3
on Blackhole is 24 counters in 12 pair markers (ids 9101 to 9112) behind the header. The most any mask gives is 16
pairs (9101 to 9116). The host keeps one marker per timestamp, core, RISC and id, which is why every pair has an id of
its own. At mask 3 on a two-NoC tile the 13 markers take 52 of the 512 words of the BRISC profiler buffer, and are dropped as
a whole when the buffer has no room.

## 5. Reading the markers on the host

```python
from tt_llk_perf import noc_counters

launch = noc_counters.unpack_launch(header_9100, {0: pair_9101, 1: pair_9102})
noc_counters.decode_noc_launch("blackhole", launch)   # counts per NoC, bytes and request totals
```

`noc_metrics.compute_noc_metrics(counts, word_bytes)` gives, per NoC: `noc_issued_read_bytes`,
`noc_issued_write_bytes` (this tile's own transfers), `noc_served_read_bytes`, `noc_served_write_bytes` (other
tiles' transfers with this tile), `noc_read_bytes` and `noc_write_bytes` (both sides), `noc_bytes_in`
(data arriving: issued reads plus served writes), `noc_bytes_out` (data leaving: issued writes plus served reads),
the issued request counts and bytes per issued request. A metric whose group was off reads `None`.
`perf_counter_analysis.decode_noc_counter_markers(rows, arch)` returns one row per launch and NoC.

## 6. Measured on Blackhole

`metal_example_eltwise_binary` (64 tiles of 2 KiB per buffer, two read buffers, one write buffer, core 0,0) read from
the counters directly: NoC 0 `MST_RD_REQ_SENT` +128 and `MST_RD_DATA_WORD_RECEIVED` +4096 (2 x 128 KiB / 64 B), NoC 1
`MST_NONPOSTED_WR_REQ_SENT` +64 and `MST_NONPOSTED_WR_DATA_WORD_SENT` +2048.

## 7. Firmware size

The capture is a read loop at the start of the launch and one emission routine at the end, so its size does not
depend on the group mask. Measured with the firmware built offline, same toolchain, profiler on, option on versus
option off:

| Core | Extra bytes | Image before | Region |
|---|---|---|---|
| Blackhole BRISC | 316 to 332 | 7372 to 7396 B | 9984 B |
| Wormhole BRISC | 332 | 5960 B | 7712 B |
| Quasar DM0 | 444 | 7308 B | links, also with the RISC-V monitor and Tensix counter groups (9868 to 10308 B) |

The other cores do not change. The image sizes are those of the runtime firmware build with the profiler's DRAM buffer
(`PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC` nonzero), over every DRAM bank count a card can have. Blackhole BRISC with
all five Tensix counter groups and the RISC-V monitor is 9420 to 9468 B; adding this capture makes it 9736 to 9784 B,
so the region is 9984 B (it was 9728 B before this capture), a margin of 200 B. With the option off the firmware images
are byte-identical to a build without the hooks.

## 8. Open questions

- The data word size on Wormhole and Quasar.
- How multicast and broadcast writes are counted on the issuing side.
- Capturing Ethernet tiles.
