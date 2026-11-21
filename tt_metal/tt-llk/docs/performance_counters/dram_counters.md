# DRAM tile traffic counters

Reference for the counters that sit in a DRAM channel tile and for `TT_METAL_PROFILE_DRAM_COUNTERS`, the tt-metal runtime option that turns them into a measured DRAM traffic number per program. The Tensix counters ([hardware_reference.md](hardware_reference.md)) and the RISC-V monitor ([riscv_pmc.md](riscv_pmc.md)) live in worker cores; these belong to the DRAM tile, so a core cannot read them with a CSR or an MMIO load, the host reads them over PCIe.

## 1. Support matrix

| Architecture | NIU slave counters | Captured |
|---|---|---|
| Blackhole | yes, host reads of the tile NIU, one data word is 64 B | yes |
| Wormhole | NIU registers of a DRAM tile are in NoC address space at `0x100080000` plus an instance per tile, one data word is 32 B (assumed) | yes, address from the exalens register map, not yet confirmed to count |
| Quasar | NIU status offsets equal the other architectures, but how a host reaches a DRAM tile, the NoC count and the word width are not known | no, the table is empty and the option logs a warning |

On Blackhole the memory controller behind the tiles of a channel has its own activity monitor (DRAM commands, row activations, power states, AXI port handshakes and stalls). `TT_METAL_PROFILE_DRAM_CONTROLLER` captures it per program next to the tile counters, see section 7. Wormhole and Quasar have no controller map here.

## 2. Where the definitions live

Headers in `tools/include/perf_counters/`, namespace `llk::perf::dram`, self-contained:

| Header | Holds |
|---|---|
| `dram_counters_types.h` | `DramCounter` (names, append only), `Arch`, `Entry` = `{counter, register offset}`, `Layout`, and for the controller `ControllerEntry` = `{name, address}` and `ControllerLayout` |
| `dram_counters_blackhole.h`, `dram_counters_wormhole.h`, `dram_counters_quasar.h` | `layout` (NIU base, NoC instance stride, word bytes, NoC count) and the `niu_counters` table; Blackhole also `controller_layout` and `controller_counters` |
| `dram_counters_inventory.h` | `niu_table(arch)`, `layout_for(arch)`, `niu_address(arch, noc, offset)`, `controller_layout_for(arch)`, `controller_table(arch)`; the host picks the architecture at run time, so there is no `ARCH_*` switch |

Every counter is a free-running 32-bit register: deltas are taken with wrapping subtraction. The NIU offsets are `0x200 + 4 * index` inside the NoC register space of the tile. On Blackhole that space is `0xFFB20000` for NoC 0 and `0x10000` further for NoC 1. On Wormhole a DRAM tile has no worker-style register window: its NIU registers are at `0x100080000`, `0x100090000` or `0x1000A0000` for NoC 0 (instance 0, 1, 2 for noc0 y modulo 3 equal to 1, 2, 0) and `0x8000` above that for NoC 1, so the host needs the noc0 y of each tile (`Layout` holds base, strides, instance count and bias; `niu_address(arch, noc, noc0_y, offset)`):

| Counter | Index | Meaning |
|---|---|---|
| `NIU_SLV_RD_DATA_WORD_SENT` | 0x33 | data words the tile sent as read responses |
| `NIU_SLV_RD_REQ_RECEIVED` | 0x35 | read requests received |
| `NIU_SLV_NONPOSTED_WR_DATA_WORD_RECEIVED` | 0x38 | data words of non-posted writes |
| `NIU_SLV_POSTED_WR_DATA_WORD_RECEIVED` | 0x39 | data words of posted writes |
| `NIU_SLV_NONPOSTED_WR_REQ_RECEIVED` | 0x3A | non-posted write requests |
| `NIU_SLV_POSTED_WR_REQ_RECEIVED` | 0x3B | posted write requests |

The counters count NoC traffic into and out of the DRAM channel tile, not what the memory controller does behind it. Bytes are `words * word_bytes`. Every DRAM tile has them, and a bank is reachable through several tiles, so traffic to one bank can land on a tile other than its preferred one: the totals over all tiles are the robust number, the per-tile split follows the NoC endpoint each core used.

The runtime reads three of the six: `NIU_SLV_RD_DATA_WORD_SENT` and the two write data word counters, on both NoCs. Request counters are in the table for completeness and are not read, to keep the number of register reads (and their own traffic, below) small.

## 3. Capture

`TT_METAL_PROFILE_DRAM_COUNTERS=1` makes the runtime read the NIU counters of every DRAM tile on both NoCs through host register reads before a program is launched and after it is done, and write the differences. `raw` does the same without the idle calibration (the capture's own reads are still removed). The option does not need the device profiler, nothing is compiled into firmware, and with the option unset no code of the feature runs.

Hooks sit in the slow-dispatch launch (after the host writes are fenced, right before the go signal) and where the host observes the cores idle: a blocking enqueue, `Finish`, or the wait before a later program on the same cores. Before the end snapshot the write counters are re-read until two consecutive readings agree (at most 8) so that posted writes still in flight when the cores report done are counted; the read counters are never waited on, because other traffic keeps them moving. A program whose write counters never settled is flagged.

Limits:

- Fast dispatch is not captured. Programs are queued and run asynchronously, the host has no instant at which one program starts and the next has not, so a host snapshot cannot be attributed to a program. A fast-dispatch mode would have to drain the queue before and after every program; the option logs a warning instead.
- Any other traffic in the window lands in the number: another host thread that reads or writes device DRAM, other processes on the chip, a second program in flight on the same device (rows are flagged `overlapped`). Measure on an otherwise quiet device.
- With a non-blocking enqueue the end snapshot is taken when the host next waits, so the window can run past the end of the program; enqueue blocking, or call `Finish` right after the program, for the tightest window. A program launched while another is still in flight closes the earlier one at that moment (flag `overlapped` on both).
- Each snapshot is one PCIe register read per counter (tiles x NoCs x 3), so expect on the order of milliseconds per snapshot and three or more snapshots per program.

### The capture's own reads

On the UMD register path a host read of any register of a DRAM tile NIU is itself counted by the tile as one read word (and one read request) on one NoC: back-to-back reads of one register increment it by exactly 1 per read there, and not on the other path (exalens). The runtime therefore counts the register reads it issues to every tile and removes them exactly: the correction of a read counter between its start and end reading is the number of reads issued to that tile between those two readings. This holds whether a read sees its own increment or not, because it is a difference of two read indices. Which NoC the reads count on is measured once per device (64 back-to-back reads of the read counter of the first tile on each NoC, a slope near 1 means coupled); the correction applies to the read counters of that NoC only, never to write counters. `dram_traffic.csv` keeps the removed words in `self_words`.

### Other agents' traffic

With nothing running on the card, the read counters still advanced at several thousand words per second in a bursty way, writes did not; the same rate shows with the reader on either path, so it is another agent. The runtime measures it once per device before the first program (option value `1`): three back-to-back snapshot pairs and three pairs 20 ms apart, per register, medians, after the own reads are removed. The rate per second is the slope between the two (words and durations of both kinds of pair), the cost per pair is what the back-to-back pairs leave above that rate. It writes `dram_idle.csv`; with `raw` the file holds only its header, so the decoder never applies an older calibration to a raw capture. The decoder subtracts `per_pair + per_second * elapsed` from every register of every program and clamps at zero. Being bursty, the rate is only a mean: the precision of the read number is bounded by the variance of that traffic over the program's window, so short programs on a busy card have the largest relative error.

## 4. Output

Both files are written to the profiler log directory (`TT_METAL_PROFILER_DIR` or `generated/profiler/.logs`), recreated by the first program of a process.

`dram_traffic.csv`, one row per program launch, DRAM tile, NoC and counter with a nonzero corrected delta (`NONE` row when nothing moved):

| Column | Meaning |
|---|---|
| `device_id`, `program_id` | chip and the program runtime id, the id the device profiler logs as the op identifier |
| `launch_seq` | counter of launches on the device in this process |
| `flags` | bit 0: another program was in flight; bit 1: write counters were still moving after 8 reads |
| `elapsed_ns` | host time from the start snapshot to the final snapshot |
| `dram_x`, `dram_y`, `noc` | translated coordinates of the tile and the NoC (0 or 1) |
| `counter`, `delta_words`, `self_words` | `DramCounter` name, its wrapped difference with the capture's own reads removed, and the words removed |

`dram_idle.csv`: `device_id, dram_x, dram_y, noc, counter, words_per_snapshot_pair, words_per_second`.

`dram_controller.csv` (with `TT_METAL_PROFILE_DRAM_CONTROLLER`), one row per launch, channel and nonzero monitor counter: `device_id, program_id, launch_seq, flags, elapsed_ns` as in `dram_traffic.csv` (same `launch_seq`), then `channel`, `dram_x, dram_y` of the tile the registers were read through, `counter` and `value`. Flag bit 2 means the controller interval ran out inside the window.

## 5. Python decode

`tt_llk_perf.dram_counters` (stdlib only) reads both files:

```python
from tt_llk_perf import dram_counters
rows = dram_counters.read_traffic("generated/profiler/.logs/dram_traffic.csv")
idle = dram_counters.read_idle("generated/profiler/.logs/dram_idle.csv")
for p in dram_counters.per_program(rows, "blackhole", idle):
    print(p.program, p.read_bytes, p.write_bytes, p.read_gbps)
```

`per_program` returns per launch: read and write bytes per DRAM tile, totals, GB/s over `elapsed_ns`, and the flags. `read_controller` and `per_program_controller` do the same for `dram_controller.csv`: counters per channel, read and write commands and activations over all channels, the row hit rate and an upper bound of the bytes. `python -m tt_llk_perf.dram_counters <log dir> --arch blackhole` prints both tables. Join with the op table on `program_id`.

## 6. What has to be checked on hardware

The NIU write counters matched a kernel's output exactly on a Blackhole card. Open:

1. The own-read correction and the idle rate inside the runtime: `tools/dram_counters_probe/dram_validate.py` runs a program that moves a known number of bytes with the option off and on and compares `bytes / 64` with the measured words; expect writes exact and reads within 2 percent, no flags.
2. `tools/dram_counters_probe/dram_idle_drift.py` measures the drift and the per-read increment on the UMD and exalens paths.
3. The controller capture on Wormhole and Quasar: no map yet. On a Blackhole p100a the eltwise binary example (slow dispatch, 2 x 128 KiB in, 128 KiB out) gave 4096 read and 2048 write commands over the seven channels, exactly its 262144 and 131072 bytes and the tile write words; read data handshakes 3456 + 640 on ports 1 and 3, write data 2048 on port 2; 96 percent row hits; the same counts with `raw`, and every controller back on its firmware interval and gating afterwards.
4. `tools/dram_counters_probe/dram_niu_probe.py --arch wormhole --mode umd|exalens` reads the read counter of every DRAM tile around a known host read of the tile and prints the change against the expected words; run it before trusting Wormhole numbers. Quasar: not tried.

## 7. Memory controller activity monitor (Blackhole)

Every Blackhole DRAM channel has one GDDR6 controller behind its three NoC tiles. The controller counts what it does over a capture interval and holds the counts of the last completed interval in 32-bit registers. The host reaches them through the register window of a tile: the controller address with bit 40 set. Without that bit the same address reads DRAM contents. Only some tiles of a channel answer register reads, so the runtime reads the AXI control word through each tile of the channel and uses the first one whose three AXI port resets read released.

| Counter | Meaning, per interval |
|---|---|
| `RD_CMD`, `RD_AP_CMD`, `WR_CMD`, `WR_AP_CMD` | read and write commands, without and with auto precharge; their sum is the total |
| `ACT_CMD`, `PRECHARGE_CMD`, `PRECHARGE_ALL_CMD` | row activations and precharges; `1 - ACT / (reads + writes)` is the row hit rate |
| `REFRESH_CMD`, `POWER_DOWN_CYCLES`, `SELF_REFRESH_CYCLES` | refreshes, and memory clocks the DRAM spent powered down or in self refresh |
| `RD_TO_WR_SWITCH`, `REORDER_AGE_LIMIT`, `RMW_CYCLE` | read to write turnarounds, reorder queue entries that hit their age limit, read-modify-write cycles |
| `QUEUE_SUM_LO`, `QUEUE_SUM_HI` | sum of the controller queue level over the interval's memory clocks (39 bits) |
| `AXIn_CLOCKS` | AXI clocks of port n in the interval |
| `AXIn_AR`, `AXIn_R`, `AXIn_AW`, `AXIn_W` | handshakes of the read address, read data, write address and write data channels of port n |
| `AXIn_AR_STALL`, ... `AXIn_W_STALL` | clocks a channel had valid data that the controller did not accept |

The AXI ports are numbered 1 to 3, as in the controller's own map. A command moves 64 B. A host access narrower than that is still one command, so `commands * 64` is an upper bound of the bytes.

### Capture

`TT_METAL_PROFILE_DRAM_CONTROLLER=1` works only together with `TT_METAL_PROFILE_DRAM_COUNTERS`, and uses its windows. Before the tile snapshot that starts a program, the runtime saves each controller's interval and AXI control registers, turns off the dynamic clock gating of the AXI ports, and starts a fresh interval of the longest length (2^32 - 1 memory clocks, about 2.1 s at 16 Gbps). After the tile snapshot that ends the program, it writes a zero interval, which stops counting and latches the counts, reads the counters, and writes the saved interval and AXI control back. The controller writes stay outside the tile window, so the tile counters never see them.

### Limits

- The DRAM firmware owns the monitor. It runs its own interval and polls the completion flag to decide when idle DRAM goes into self refresh. A window replaces that interval, so the DRAM does not enter self refresh while a program is measured, and the firmware interval restarts when it is written back. Nothing else is changed; with the option unset no register is touched.
- A window longer than one interval wraps: the counts then cover only its last part and the rows carry flag bit 2. Split such a program, or read the counts as a rate.
- A dynamically gated AXI port's monitor stops with its clock, so its handshake counts are stale. The capture turns the gating off for the window. Expect slightly higher idle power for that time.
- Like the tile counters, everything else that touches DRAM in the window is counted: host reads and writes, other processes, an overlapped program.
