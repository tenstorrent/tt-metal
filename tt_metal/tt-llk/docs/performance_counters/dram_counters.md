# DRAM tile traffic counters

Reference for the counters that sit in a DRAM channel tile and for `TT_METAL_PROFILE_DRAM_COUNTERS`, the tt-metal runtime option that turns them into a measured DRAM traffic number per slow-dispatch program. The Tensix counters ([hardware_reference.md](hardware_reference.md)) and the RISC-V monitor ([riscv_pmc.md](riscv_pmc.md)) live in worker cores; these belong to the DRAM tile, so a core cannot read them with a CSR or an MMIO load, the host reads them over PCIe.

## 1. Support matrix

| Architecture | NIU slave counters | Captured |
|---|---|---|
| Blackhole | yes, one data word is 64 B | yes |
| Wormhole | yes, one data word is 32 B | yes |
| Quasar | how a host reaches a DRAM tile is not known | no, the option logs a warning |

On Blackhole the memory controller behind the tiles of a channel has its own activity monitor. `TT_METAL_PROFILE_DRAM_CONTROLLER` captures it per program next to the tile counters, see section 6.

## 2. The counters

The counters of a DRAM tile are the slave side NoC interface counters of [noc_counters.md](noc_counters.md): data words and requests the tile served, per NoC, free running 32-bit registers read as wrapping deltas. They count NoC traffic into and out of the tile, not what the memory controller does behind it, and a data word is a flit at the destination, so bytes are exact for aligned transfers and an upper bound otherwise. A bank is reachable through several tiles, so the totals over all tiles are the robust number; the per-tile split follows the NoC endpoint each core used.

The runtime reads three of them on every tile and NoC: `SLV_RD_DATA_WORD_SENT` and the two write data word counters. The headers are in `tools/include/perf_counters/dram_counters_*.h` (namespace `llk::perf::dram`): `Layout` says where the host reads the registers of a tile on each architecture (`niu_address(arch, noc, tile_noc0_y, reg)`), and `dram_counters_types.h` holds the file names and flag bits of the capture. The READMEs there list the rest.

## 3. Capture

`TT_METAL_PROFILE_DRAM_COUNTERS=1` makes the runtime read the counters of every DRAM tile before a program is launched and after it is done, and write the differences. `raw` does the same without the idle calibration. The option does not need the device profiler, nothing is compiled into firmware, and with the option unset no code of the feature runs.

The window opens in the slow-dispatch launch, after the host writes are fenced and right before the go signal, and closes where the host observes the program's cores idle: a blocking enqueue, `Finish`, or the wait before a later program on the same cores. Before the end snapshot the write counters are re-read until two consecutive readings agree (at most 8), so posted writes still in flight when the cores report done are counted.

Two corrections:

- **Own reads.** A host read of a DRAM tile register counts as one read word on one NoC of that tile. The runtime counts the reads it issues to each tile and subtracts them from that tile's read counter on the NoCs where reads count (measured per tile at start-up); `self_words` holds what was removed.
- **Other agents.** On an idle card the read counters still move, a few thousand words per second in bursts. With the option value `1` the runtime fits `per_pair + per_second * seconds` per register from back-to-back and 20 ms snapshot pairs before the first program and writes it to `dram_idle.csv`; the decoder subtracts it and clamps at zero. The rate is a mean, so short programs on a busy card have the largest relative error.

Limits:

- The option is a slow dispatch tool. Fast dispatch runs programs asynchronously, so a host snapshot cannot be attributed to one; the option logs a warning there.
- Anything else that touches DRAM in the window is counted: another host thread, other processes, a second program in flight on the device (`overlapped`) or a service workload (`service_active`). Measure on an otherwise quiet device.
- With a non-blocking enqueue the window ends when the host next waits; enqueue blocking, or call `Finish` right after the program, for the tightest window.
- A snapshot is one PCIe read per counter (tiles x NoCs x 3), milliseconds per snapshot, and devices are captured independently.

## 4. Output

The files go to the profiler log directory (`TT_METAL_PROFILER_DIR` or `generated/profiler/.logs`), recreated by the first program of a process.

`dram_traffic.csv`, one row per launch, DRAM tile, NoC and counter with a nonzero corrected delta (`NONE` row when nothing moved):

| Column | Meaning |
|---|---|
| `device_id`, `program_id` | chip and the program runtime id, the id the device profiler logs as the op identifier |
| `launch_seq` | launch number, unique in the process |
| `flags` | bit 0 `overlapped`: another program was in flight; bit 1 `unsettled`: write counters were still moving after 8 reads; bit 2 `interval_wrapped` (controller rows); bit 3 `not_finished`: nobody waited for the program, closed when the device closed; bit 4 `service_active`: a service workload ran on the device |
| `elapsed_ns` | host time from the start snapshot to the final snapshot |
| `dram_x`, `dram_y`, `noc` | translated coordinates of the tile and the NoC |
| `counter`, `delta_words`, `self_words` | counter name, its wrapped difference with the capture's own reads removed, and the words removed |

`dram_idle.csv`: `device_id, dram_x, dram_y, noc, counter, words_per_snapshot_pair, words_per_second`.

`dram_controller.csv`, one row per launch, channel and nonzero monitor counter: the first five columns of `dram_traffic.csv` (same `launch_seq`), then `channel`, the `dram_x, dram_y` the registers were read through, `counter` and `value`.

## 5. Python decode

`tt_llk_perf.dram_counters` (stdlib only):

```python
from tt_llk_perf import dram_counters
rows = dram_counters.read_traffic("generated/profiler/.logs/dram_traffic.csv")
idle = dram_counters.read_idle("generated/profiler/.logs/dram_idle.csv")
for p in dram_counters.per_program(rows, "blackhole", idle):
    print(p.program, p.read_bytes, p.write_bytes, p.read_gbps)
```

`per_program` returns per launch the read and write bytes per DRAM tile, totals, GB/s over `elapsed_ns` and the flags. `read_controller` and `per_program_controller` do the same for `dram_controller.csv`: counters per channel, commands and activations over all channels, an approximate row hit rate (`1 - ACT / commands`, which also counts the activations after a refresh or a power-down exit) and an upper bound of the bytes. `python -m tt_llk_perf.dram_counters <log dir> --arch blackhole` prints both tables. Join with the op table on `program_id`. `tools/dram_counters_probe/dram_validate.py` checks a known workload end to end.

## 6. Memory controller activity monitor (Blackhole)

Every Blackhole DRAM channel has one GDDR6 controller behind its NoC tiles. Its activity monitor counts what the controller does over an interval and holds the counts of the last interval; the runtime reads them through one tile of the channel.

| Counter | Meaning, per interval |
|---|---|
| `RD_CMD`, `RD_AP_CMD`, `WR_CMD`, `WR_AP_CMD` | read and write commands, without and with auto precharge |
| `ACT_CMD`, `PRECHARGE_CMD`, `PRECHARGE_ALL_CMD` | row activations and precharges |
| `REFRESH_CMD`, `POWER_DOWN_CYCLES`, `SELF_REFRESH_CYCLES` | refreshes, and memory clocks spent powered down or in self refresh |
| `RD_TO_WR_SWITCH`, `REORDER_AGE_LIMIT`, `RMW_CYCLE` | read to write turnarounds, queue entries that hit their age limit, read-modify-write cycles |
| `QUEUE_SUM_LO`, `QUEUE_SUM_HI` | sum of the queue level over the interval's memory clocks |
| `AXIn_CLOCKS`, `AXIn_AR`, `AXIn_R`, `AXIn_AW`, `AXIn_W` | clocks and channel handshakes of AXI port n (1 to 3) |
| `AXIn_AR_STALL`, ... `AXIn_W_STALL` | clocks a channel had valid data the controller did not accept |

A command moves 64 B; a narrower host access is still one command, so `commands * 64` is an upper bound of the bytes.

`TT_METAL_PROFILE_DRAM_CONTROLLER=1` works only together with `TT_METAL_PROFILE_DRAM_COUNTERS` and uses its windows. Before the tile snapshot that starts a program the runtime saves each controller's interval and AXI control values, turns off the dynamic clock gating of the AXI ports and starts an interval of the longest length; after the end snapshot it latches and reads the counts and writes the saved values back. The controller writes stay outside the tile window.

Limits:

- The capture takes over the monitor's interval and the AXI clock gating for the window. The values it replaced wait in a file in the temp directory until they are written back, so if a process dies with a window open, the next capture on that card puts them back (a card reset does too). DRAM power management that relies on the monitor pauses during a window, and idle power is slightly higher.
- An interval lasts at least about 2.1 s; a longer window may have wrapped, and its rows carry `interval_wrapped`.
- With the option unset no controller register is touched.
