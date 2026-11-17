# NoC router counters

Reference for the packet and flit counters inside the NoC router of a tile and for `TT_METAL_PROFILE_ROUTER_COUNTERS`, the tt-metal runtime option that captures them per launch in the device profiler. [noc_counters.md](noc_counters.md) covers the counters of the tile's NoC interface unit (NIU), [riscv_pmc.md](riscv_pmc.md) the monitor inside the RISC-V cores and [hardware_reference.md](hardware_reference.md) the Tensix MMIO counters. The router counters are a fourth, separate set, and the only one that sees traffic passing through a tile.

## 1. Support matrix

| Architecture | Counters | Ports | Count | Capturing core | Captured by `TT_METAL_PROFILE_ROUTER_COUNTERS` |
|---|---|---|---|---|---|
| Blackhole | 64-bit, 16 virtual channels per port | 2 output ports, no input side | head flits, so one count is one packet | BRISC | yes |
| Quasar | 32-bit, 16 virtual channels per port | 4 output and 4 input ports | every flit (head, body, tail) | DM0 | yes, A0 silicon behaviour unproven |
| Wormhole | none in the register map | none | none | none | no, the option is refused |

Ethernet tiles are left out for the reason given in [noc_counters.md](noc_counters.md). Quasar has one NoC in the register headers.

## 2. What the counters are

A router has one input and one output port per direction it connects to, plus the port into the tile's own NIU. The counters sit on the ports that face other tiles. Each has one register per virtual channel (VC), 16 per port, free running; nothing in tt-metal resets them, so a capture is a delta. They are plain loads from the tile's own register space, so any core of the tile can read them.

| | Blackhole | Quasar |
|---|---|---|
| Counted event | a head flit leaving through the output port on that VC | a valid flit of any type on that VC, leaving an output port or entering an input port |
| One count is | one packet | one flit |
| Register width | 64-bit, low word first (two words per VC) | 32-bit |
| Ports | output port 1 at `0x500`, output port 2 at `0x580` | output ports 1 to 4 at `0x600`, `0x640`, `0x680`, `0x6C0`; input ports 1 to 4 at `0x700`, `0x740`, `0x780`, `0x7C0` |
| Evidence | register map and design description; reset by bit 0 of `DEBUG_COUNTER_RESET` | register map and design description only; A0 silicon not checked |

The address of a counter is `ROUTER_REGS_BASE + noc * 0x10000 + bank offset + vc * stride`, with the base `0xFFB20000` on Blackhole and `0x02000000` on Quasar. The capture reads the low word of each Blackhole counter, so a delta is modulo 2^32.

What this means for what a core sees:

- Only traffic that leaves the tile through a port is counted. A packet that ejects into the tile's own NIU is not counted on Blackhole, and neither is a transfer from a tile to itself.
- Packets that pass through the tile on the way between two other tiles are counted, together with the packets the tile injects. The counters do not tell the two apart.
- A read counts once at the tile that issues the request, and its response counts at the tiles the response crosses, which includes the tile that serves the data and not the tile that issued the read. A write counts at the issuing tile's router, and a non-posted write's acknowledgement counts at the tiles it crosses on the way back, not at the issuer.
- Blackhole counts packets, so it has no byte count: one packet moves up to `NOC_MAX_BURST_SIZE` bytes of payload.
- The two Blackhole ports are called 1 and 2 here. The hardware description names them after the dimension they route along, and tt-metal's `NOC_PORT_X` and `NOC_PORT_Y` constants disagree with that description about which one is port 1, so use the number. On a quiet device the port a transfer uses is found by sending to a tile in the same row or the same column.

Virtual channels in tt-metal: Blackhole unicast writes and read requests use VC 1 (`NOC_UNICAST_WRITE_VC`), multicast writes VC 4, and the dispatch cores' multicast writes VC 5. Quasar uses VC 1, 8 and 9. Which channels responses and acknowledgements use is not documented here.

## 3. Where the definitions live

The hardware description is in tt-llk (`tools/include/perf_counters/`, namespace `llk::perf::router`, self-contained):

| Header | Holds |
|---|---|
| `router_counters_types.h` | `RouterPort` (names, append only), `CountKind`, `Bank` = `{port, byte offset}`, `NUM_VCS`, the layout of the option mask |
| `router_counters_blackhole.h`, `router_counters_wormhole.h`, `router_counters_quasar.h` | `router_banks` table in capture order, `ROUTER_REGS_BASE`, `ROUTER_NOC_STRIDE`, `ROUTER_NOCS`, `ROUTER_VC_STRIDE`, `COUNTER_BITS`, `COUNT_KIND`. The Wormhole table is empty |
| `router_counters_inventory.h` | Picks the arch header from `ARCH_*`; `selected_bank_offsets<Option>()`, `selected_vcs<Option>()`, `value_count(option)` |
| `router_counters_hw.h` | `read(noc, bank_offset, vc)` |

The metal consumer is `tt_metal/hw/inc/internal/profiler/router_counters.hpp`: marker ids, payload layout and emission. The hooks are in `tt_metal/hw/firmware/src/tt-1xx/brisc.cc` and `tt-2xx/dm.cc`; the option is parsed in `tt_metal/llrt/rtoptions.cpp` and becomes `-DPROFILE_ROUTER_COUNTERS=<mask>` in `tt_metal/jit_build/build.cpp`. The host side is `tools/python/tt_llk_perf/`: `headers.py` parses the tables, `router_counters.py` decodes the markers and `router_metrics.py` derives the totals; `tools/tracy/perf_counter_analysis.py` has `decode_router_counter_markers`.

Adding a port: append the enumerator to `RouterPort` if it is new and add one `{RouterPort::NAME, offset}` entry to the arch table, bump the array size. The host parsers read the headers, so nothing else changes. The table has at most 8 entries: the option has 8 port bits.

## 4. Using it in tt-metal

```bash
export TT_METAL_PROFILE_ROUTER_COUNTERS=on          # every port, the 16 channels added up per port
export TT_METAL_PROFILE_ROUTER_COUNTERS=0x30002     # both Blackhole ports, VC 1 only
export TT_METAL_PROFILE_ROUTER_COUNTERS=0x1030012   # both ports, VC 1 and VC 4 kept apart
```

The value is `on` or a mask, hex or decimal:

| Bits | Meaning |
|---|---|
| 0 to 15 | virtual channels to count |
| 16 to 23 | ports to capture; bit 16 is the first entry of the architecture's table (Blackhole: output port 1, 2; Quasar: output ports 1 to 4, then input ports 1 to 4). Bits beyond the table are ignored |
| 24 | keep every selected channel apart; without it the selected channels of a port are added up into one value |

`on` is every channel and every port, added up per port: 4 values on Blackhole (2 NoCs x 2 ports) and 8 on Quasar. The sum is the default because one value per port answers the question of how much traffic a tile's router carried, and a capture reads 16 registers per port and NoC at each end of the window, whatever the sum. Keeping channels apart is for telling request traffic from multicast or dispatch traffic; it multiplies the values by the number of channels, and a capture is at most 32 values, so on Blackhole 8 channels on both ports and both NoCs fill it. A mask that selects no channel or no port is refused, `0` or unset leaves the capture out of the firmware entirely, and a mask that gives more than 32 values fails the firmware build with a message. Setting the option forces the device profiler on. On Wormhole the option is refused when the firmware is built. The mask is a compile-time constant of the firmware and part of the firmware build key, so changing it rebuilds the firmware; the key is unchanged when the option is off.

The router belongs to the tile, so one core captures once per launch: BRISC on Blackhole, DM0 on Quasar, the same core that captures the NoC interface counters. The window opens before the other cores of the tile are started and closes after they have finished. The profiler's own writes to DRAM and the dispatch traffic around the launch that cross this tile's router are not separated from the kernel's transfers.

Each launch emits one header marker and one marker per pair of values, all `TS_DATA` with the payload in the `data` column of `profile_log_device.csv`, all with the same timestamp:

| Marker id | `data` |
|---|---|
| 9120 | `(noc count << 32) \| option mask` |
| 9121 + k | `(value[2k + 1] << 32) \| value[2k]` |

The values are the per-launch deltas, NoC 0 first, then the selected ports in table order, and inside a port one value (channels added up) or one value per selected channel in ascending order. An odd value count leaves the high half of the last pair at 0. The ids are above the Tensix counters (9090, 9091), the RISC-V monitor (9092 to 9096) and the NoC interface counters (9100 to 9116); the host keeps one marker per timestamp, core, RISC and id, which is why every pair has an id of its own. A capture is at most 17 markers, 68 words of the 512 of the profiler buffer, and is dropped as a whole when the buffer has no room.

## 5. Reading the markers on the host

```python
from tt_llk_perf import router_counters

launch = router_counters.unpack_launch(header_9120, {0: pair_9121, 1: pair_9122})
router_counters.decode_router_launch("blackhole", launch)   # counts per NoC and port, totals, what a count is
```

Labels are the port name (`OUT_PORT1`) or the port name with the channel (`OUT_PORT1_VC1`) when the channels were kept apart. `router_metrics.compute_router_metrics(counts)` gives `router_out_count` and `router_in_count`, the sums over the captured output and input ports; a side that was not captured reads `None`. `decode_router_launch` also returns `count_kind` (`HEAD_FLIT` or `ALL_FLIT`) so that a count is never read as bytes. `perf_counter_analysis.decode_router_counter_markers(rows, arch)` returns one row per launch and NoC.

## 6. Checking on a card

`metal_example_eltwise_binary` runs on core 0,0 with reads on NoC 0 and non-posted writes on NoC 1, 64 tiles of 2 KiB per buffer, two read buffers and one write buffer. Run with the NoC interface counters on as well (`TT_METAL_PROFILE_NOC_COUNTERS=3`), so the request counts are known:

- NoC 0: `MST_RD_REQ_SENT` is 128, so the router of core 0,0 must show at least 128 output counts on NoC 0, all of them on VC 1 (requests). The responses return to this tile and eject, so they do not add to this tile's counters.
- NoC 1: `MST_NONPOSTED_WR_REQ_SENT` is 64, so at least 64 output counts on NoC 1 on VC 1. The write acknowledgements eject here.
- Anything above those numbers is traffic from other tiles passing through the router, which a single-core run on a quiet device should keep small.
- The sum over all channels is at least the VC 1 total.

## 7. Firmware size

The capture is one scan routine and one emission routine, so its size hardly depends on the option mask. Measured with the firmware built offline, same toolchain, profiler on with the runtime's DRAM buffer size (`PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC` nonzero), over every DRAM bank count a card can have (7 and 8 banks, 36 Blackhole build keys), option on versus option off:

| Core | Image before | Extra bytes | Region |
|---|---|---|---|
| Blackhole BRISC | 7372 to 7396 B | 476 | 9984 B |
| Quasar DM0 | 7308 B | 444 | 12288 B |

The other cores do not change, and with the option off the firmware images are byte-identical to a build without the hooks, profiler on or off.

The Blackhole BRISC region was grown twice for the counters already in it and is not grown again, so not every combination of counter options fits:

| NoC interface counters (`3`) | RISC-V monitor | Tensix groups (`47`) | Router counters | BRISC image | Fits 9984 B |
|---|---|---|---|---|---|
| on | on | on | off | 9736 to 9784 B | yes |
| off | on | on | on | 9876 to 9924 B | yes |
| on | on | off | on | 8352 to 8376 B | yes |
| on | off | off | on | 8136 to 8160 B | yes |
| on | off | on | on | 9992 B or more | no |
| on | on | on | on | about 10.2 KB | no |

When the firmware does not fit, the firmware build fails with a message that names the option; turn off one of the others. Quasar DM0 with every option on, router counters included, is 10204 B of its 12288 B region. The BRISC capture is kept for the window it gives: NCRISC has room (1836 B of 2560 B with every option on), but a capture there would cover the NCRISC kernel only and would be missing for a launch without one.

## 8. Open questions

- Quasar: the counts, the width and the direction of every port are read from the design description. Nothing has been measured on A0 silicon or on the emulator.
- Blackhole: which of the two ports is the row direction and which the column direction, confirmed by a transfer to a tile in the same row.
- Whether the Blackhole counters are touched by anything else on the tile: nothing in tt-metal resets them.
- Reading the upper word of a Blackhole counter is left out; a delta of the low word is exact for launches below 2^32 packets per port.
