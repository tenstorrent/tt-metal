# Ethernet link counters

Reference for the counters of an ethernet core and for `TT_METAL_PROFILE_ETH_COUNTERS`, the tt-metal runtime option that turns them into measured link traffic per slow-dispatch program and per device lifetime. The Tensix counters ([hardware_reference.md](hardware_reference.md)), the RISC-V monitor ([riscv_pmc.md](riscv_pmc.md)), the NoC interface counters ([noc_counters.md](noc_counters.md)) and the DRAM tile counters ([dram_counters.md](dram_counters.md)) cover other blocks. These belong to the ethernet core: the TX and RX queue controllers in front of the MAC, and values the base firmware keeps in the L1 of the core. The host reads them over PCIe or the NoC.

## 1. Support matrix

| Architecture | Queue registers | Firmware link counters in L1 | Captured |
|---|---|---|---|
| Blackhole | 3 TX and 3 RX queues; packets, words, transfers, resends, drops, payload bytes; one TX or RX word is 96 B | frames, bytes, bad FCS, dropped, corrected and uncorrected codewords, retrain, flap, TX resend and RX drop copies; ethernet firmware 1.5.0 or newer | yes |
| Wormhole | 2 TX and 2 RX queues; packets, words, transfers, resends, drops; one word is 16 B; not read from a host on silicon yet | retrain count, CRC errors, corrected and uncorrected codewords; ethernet firmware 6.14.0 or newer | yes, untested |
| Quasar | register map not known | not validated | no, the tables are empty and the option logs a warning |

Not captured, on any architecture:

- The MAC/PCS statistics block (frame size histograms, unicast, multicast and broadcast counts, pause and PFC frame counts and durations, per-lane FEC). Reading it is not proven safe, section 3.
- Back-pressure and flow-control durations: the only such counters are in that block.
- Per-lane SerDes error counters: the base firmware keeps bit error counts of its own bist runs only.

## 2. Where the definitions live

Headers in `tools/include/perf_counters/`, namespace `llk::perf::eth`, self-contained:

| Header | Holds |
|---|---|
| `eth_counters_types.h` | `EthCounter` (names, append only), `Arch`, `QueueEntry` = `{counter, offset in the queue register space}`, `L1Entry` = `{counter, queue, words, word order, L1 address}`, `Layout` |
| `eth_counters_blackhole.h`, `eth_counters_wormhole.h`, `eth_counters_quasar.h` | `layout` (TX and RX queue base, queue stride, queue counts, word bytes) and the tables `tx_queue_counters`, `rx_queue_counters`, `l1_counters` |
| `eth_counters_inventory.h` | `tx_queue_table(arch)`, `rx_queue_table(arch)`, `l1_table(arch)`, `layout_for(arch)`, `tx_queue_address(arch, queue, offset)`, `rx_queue_address(...)`; the host picks the architecture at run time, so there is no `ARCH_*` switch |

### Queue registers

Every queue register is a free-running 32-bit counter; reading it has no side effect and the host reads them without involving the ERISC. Deltas use wrapping subtraction. Queue `q` of a kind is at `base + q * 0x1000`:

| | Blackhole | Wormhole |
|---|---|---|
| TX queues | `0xFFB90000`, 3 queues | `0xFFB90000`, 2 queues |
| RX queues | `0xFFB94000`, 3 queues | `0xFFB92000`, 2 queues |

| Counter | Offset | Meaning |
|---|---|---|
| `TXQ_TRANSFER_CNT` | 0x30 | transfers (commands) the ERISC issued to the queue |
| `TXQ_PKT_START_CNT` | 0x34 | packets started |
| `TXQ_PKT_END_CNT` | 0x3C | packets sent |
| `TXQ_WORD_CNT` | 0x40 | words sent to the MAC, headers and padding included |
| `TXQ_RESEND_CNT` | 0x68 | resend start events: a drop notification from the receiver or an ack timeout while packets were unacknowledged |
| `RXQ_WORD_CNT` | 0x14 | words received, frame overhead included |
| `RXQ_PKT_START_CNT` | 0x24 | frames started |
| `RXQ_PKT_END_CNT` | 0x28 | frames ended |
| `RXQ_PKT_DROP_CNT` | 0x4C | packets dropped: a sequence number mismatch or a full receive buffer |
| `RXQ_BYTE_CNT` | 0x04 | Blackhole only: payload bytes written to L1, counted in 16 B steps |

The word size is the width of the MAC data path: 96 B on Blackhole, where a keepalive is one TX word and a packet of up to 64 B of payload is one word, and 16 B on Wormhole. A resend is the retransmission signal of the reliable packet mode: `TXQ_RESEND_CNT` on the sender and `RXQ_PKT_DROP_CNT` on the receiver of the same link move together. The queues count what a program sends and also keepalives and sequence number updates, so an idle link moves them too, section 4.

### Values the base firmware keeps in L1

On Blackhole the base firmware refreshes a live status block, `boot_results_t::eth_live_status` at `0x7CE00`, about once a second from the MAC/PCS block, and the capture reads it as one block per core. Its 64-bit counters are cumulative and stored low word first. The queue counts in it are 64-bit copies of the queue registers, useful to check the registers and to see counts that wrapped. `FW_LINK_POLL_ALIVE_COUNT` moves each time the firmware polls; a delta of 0 over a window means the block was not refreshed in it, and then every other `FW_` value of the window is stale: the runtime drops those rows and sets flag bit 2 instead of reporting zeros. `FW_` values are exact over windows of several seconds, not per program. On a Blackhole QuietBox with a 19.x firmware bundle the block did not refresh at all in 20 s, so there every window is flagged stale and only the queue registers are reported.

On Wormhole the base firmware keeps four values: `FW_RETRAIN_COUNT` (`0x1EDC`), `FW_CRC_ERR` (`0x1F7C`), `FW_CORR_CW` (`0x1F90`) and `FW_UNCORR_CW` (`0x1F98`), the 64-bit ones as high word, then low word.

The addresses are checked against the firmware headers in `test_perf_eth_counters.py` when run from a metal checkout.

## 3. The MAC/PCS statistics block is not read

The Blackhole MAC/PCS block at `0xFFBA0000` holds 81 64-bit statistics counters (37 TX, 44 RX): frames and octets good, bad, dropped, by unicast, multicast and broadcast, eight frame size ranges, pause and PFC frame counts per class, and fragment and assembly errors. Each exists twice, as an active counter and as a snapshot copy. The block base agrees with what the firmware and tt-exalens use. Its offsets come from a SoC register description that is not the silicon pin: the TX and RX statistics bases are `0xFFBA2800` and `0xFFBA2C00` for the active counters and `0xFFBA2600` and `0xFFBA2A00` for the snapshots; only the TX configuration register at `0xFFBA2200` is cross-checked against metal's own definitions.

The block is not safe to read from a host, so the runtime does not touch it:

- Snapshots copy the active counters when a trigger register is written, either by software or periodically. A snapshot with the clear option set zeroes the active counters. The description's reset values enable that clear for both directions.
- The reset values also make a read destructive in places: reading the TX snapshot copy clears it, and reading the RX active counters clears them. The TX active and RX snapshot copies do not clear on read.
- The base firmware already uses this block: its live status block holds "snapshot" values, so it triggers snapshots and reads the copies. A host read of a clearing copy takes counts away from it, and active counters that its snapshots zero move backwards under a host delta.
- Which bits the firmware sets is not known without reading the two configuration registers on a running card.

`tools/eth_counters_probe/eth_mib_config.py` reads exactly those two registers (read-only) and prints the clear and periodic bits; tt-exalens in safe mode refuses to read the block, so the script reports that and stops. A host snapshot trigger is not written by anything in this stack. If the probe shows the active TX counters are not cleared by snapshots and not cleared on read, a non-destructive read of that copy would be possible; the RX active counters are the open case.

## 4. Capture

`TT_METAL_PROFILE_ETH_COUNTERS=1` makes the runtime read every queue register and the L1 values of every active ethernet core of a device through host reads and write the differences. `raw` does the same without the idle measurement. The option does not need the device profiler, nothing is compiled into firmware, and with the option unset no code of the feature runs. Active ethernet cores are the ones of `get_active_ethernet_cores`: cores whose link is connected to another chip. Ethernet firmware older than the first one with the L1 values keeps only the queue registers, with a warning.

Two kinds of window:

- Per program (slow dispatch): the hooks of the DRAM capture, right before the go signal and where the host observes the cores idle. Meaningful for programs that use ethernet: CCL and fabric kernels, ethernet kernels of the unit tests.
- Per run: one snapshot when a device is opened and one when it is closed. This works under fast dispatch, where per-program windows do not exist, and is the one to use for CCL ops of a model: the total includes the ethernet traffic of everything between open and close, and fabric start-up.

Limits:

- Fast dispatch is not captured per program (the option logs a warning); the run totals are written.
- Other traffic in the window lands in the number, including links that carry a second program of another process or traffic of other chips that cross this chip.
- Receive counters on the far chip of a link are read when that chip's own program ends, so a transfer is complete on both sides only in the run totals, or when both chips' programs end together.
- No settle loop is used, unlike the DRAM capture: the receiving side of a link is on another chip and keeps moving.
- Each snapshot reads one L1 block and, per queue, five TX and four or five RX registers for every active core: tens of milliseconds on a 16-link chip, more over a remote chip.

### Idle rate

Keepalives, sequence number updates and fabric heartbeats move the queue registers with no program running. With option value `1` the runtime measures that rate once per device before the first program (three snapshot pairs 50 ms apart, median per register) and writes `eth_idle.csv`; the decoder subtracts `rate * elapsed` from each queue register of each program and clamps at zero. `FW_` values are not corrected. Measured on a cabled Blackhole QuietBox, an idle link moved every queue's packet start, packet end and word counters, TX and RX, by about 100000 per second (about 1e5 packets per second of background per queue). A per-program window on a quiet link is therefore dominated by this background: a 1 ms program sees about 100 counts per queue from it, and bytes derived from words are only meaningful after the subtraction and when the program sends well above that. Use `1`, not `raw`, for per-program numbers, and expect the residual to be limited by how steady the background rate is. A link that carries traffic of another agent is not idle: measure on a quiet link, or pass `raw` and subtract your own baseline.

### Firmware-side capture

A snapshot taken by the ERISC around its kernel is not built: the queue registers are reachable from the host, the kernels of fabric routers outlive a single op, and any firmware hook would change the ERISC image with the option off. Host reads add no instructions to the firmware.

## 5. Output

All files are written to the profiler log directory (`TT_METAL_PROFILER_DIR` or `generated/profiler/.logs`), recreated by the first write of a process.

`eth_traffic.csv`, one row per program launch, ethernet core and counter with a nonzero delta (a `NONE` row when nothing moved):

| Column | Meaning |
|---|---|
| `device_id`, `program_id` | chip and the program runtime id, the id the device profiler logs as the op identifier |
| `launch_seq` | counter of launches on the device in this process |
| `flags` | bit 0: another program was in flight; bit 1: a 64-bit counter went down in the window (a link reset), its row is dropped; bit 2: the firmware poll counter did not move, so no `FW_` row is written |
| `elapsed_ns` | host time from the start snapshot to the final snapshot |
| `eth_x`, `eth_y`, `channel` | translated coordinates and ethernet channel of the core |
| `counter`, `queue`, `delta` | `EthCounter` name, the queue (`-1` when the value has none), its wrapped difference |

`eth_run.csv`: `device_id, run_id, flags, elapsed_ns, eth_x, eth_y, channel, counter, queue, delta`, one set of rows per device open and close. `eth_idle.csv`: `device_id, eth_x, eth_y, channel, counter, queue, counts_per_second`.

## 6. Python decode

`tt_llk_perf.eth_counters` (stdlib only) reads the files:

```python
from tt_llk_perf import eth_counters
rows = eth_counters.read_traffic("generated/profiler/.logs/eth_traffic.csv")
idle = eth_counters.read_idle("generated/profiler/.logs/eth_idle.csv")
for p in eth_counters.per_program(rows, "blackhole", idle):
    print(p.program, p.tx_bytes, p.rx_bytes, p.resends, p.drops)
for r in eth_counters.per_run(eth_counters.read_run("generated/profiler/.logs/eth_run.csv"), "blackhole"):
    print(r.device, r.tx_bytes, r.cores[(25, 25)].get("FW_UNCORR_CW"))
```

`per_program` returns per launch: TX and RX bytes (queue words times the word size, summed over queues) per ethernet core and in total, GB/s over `elapsed_ns`, resend and drop totals, every counter per core and per queue, and the flags. `python -m tt_llk_perf.eth_counters <log dir> --arch blackhole` prints both tables. Join with the op table on `program_id`.

## 7. What has to be checked on hardware

Nothing here has run on a device. `tools/eth_counters_probe/README.md` lists the scripts; each must show:

1. Single Blackhole card, any connected port: `eth_idle_drift.py` shows the queue registers readable (never `0xFFFFFFFF`), monotonic, and the idle rate of `TXQ_WORD_CNT` and `RXQ_WORD_CNT` per queue; the L1 block refreshes (`FW_LINK_POLL_ALIVE_COUNT` moves about once a second; on a QuietBox with a 19.x firmware bundle it did not move, which the runtime reports as stale) and its 64-bit queue copies agree with the registers over a window of several seconds.
2. `eth_mib_config.py`: the two MAC configuration registers, which decides whether any statistics counter can be read without disturbing the firmware.
3. Any multi-chip system, slow dispatch: `eth_validate.py --arch blackhole|wormhole --binary <unit_tests_eth>` runs `MeshDeviceFixture.ActiveEthKernelsSendInterleavedBufferAllConnectedChips`, which sends about one unreserved L1 over every connected link and erisc four times. It must show: option off writes no `eth_*.csv`; option on at least four programs with sender bytes, receiver bytes of the same order, no drops (resends are reported: a healthy link retransmits now and then), no reset flag, and per-device run totals at least the sum of that device's program rows. The two-chip fixtures of the other ethernet tests (`TwoDeviceBlackholeFixture`, `N300MeshDeviceFixture`) skip under slow dispatch and their test bodies skip without it, so no existing test validates the per-program path with an exactly known packet count; the check is sender and receiver bytes against the buffer sizes in the printed table, and the 96 B and 16 B word sizes are read off `payload / words`.
4. Per-run totals under fast dispatch are validated only by the CCL run below.
5. A fabric or CCL run with known bytes (an all-gather of a known tensor under fast dispatch) in `eth_run.csv`: the bytes summed over the ring match the expected transfer within the idle and fabric start-up traffic.
