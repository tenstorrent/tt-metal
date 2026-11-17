# Keraunos CCE microbenchmarks: handoff

**Status:** built and run on the `mmk` emulator (run s1, 2026-11-17, at `2d6552fb1ea`): 43 of 47 cases pass, every data check included. The four `CpuStore64_*_65536B` cases were not run; each needs about 2 h of emulator time (see Results of run s1).
**Branch:** `kzhao/keraunos-ubench`, on top of `kzhao/keraunos-three-device`.
**Owner:** Kevin Zhao (`kzhaoTT`). Keraunos/Mimir runtime support comes from Kevin Stevens.

## Goal

Port the X280 bandwidth and latency microbenchmarks (tt-metal PR #60692, `kzhao/x280-port-benchmarks`) to Keraunos CCE harts.
- They are built natively as metal JIT kernels, using the Keraunos support in metal.
- They cover on-chip transport and latency, and the same over Ethernet links. On the emulator, the Ethernet links run in loopback between separate HSIO tiles.
- The harness was written from scratch at the owner's request. It does not reuse the X280 harness.

## Branch decisions

- **Base:** `kzhao/keraunos-three-device`.
  - It has the CCE Ethernet send (`tt_metal/hw/inc/internal/tt-2xx/quasar/eth_cce_send.h`), the CCE NoC shim (`cce_noc.h`), the Ethernet-link slot, and the two- and three-chip HSIO descriptors.
  - It sits on Kevin Stevens's "Initial Keraunos support" commit (`2c1b43e591`), so it supports **CCE0 only**.
- **Planned rebase:** onto Kevin Stevens's *next* branch once he publishes it. His 11-13 message said it would have CCE1 and the Keraunos→Mimir path working.
  - His current tip, `kstevens/enable_keraunos` at `a0038c7521`, is WIP: it splits each CCE into its own core with a 0 or 4 MiB base, and is not verified on the emulator.
  - Expect conflicts around `ETH_CCE_LOCAL_BASE`, `MEM_L1_UNCACHED_BASE` (8 MiB on CCE0, 12 MiB on CCE1 after the split) and hart indexing.
  - The benchmark code keeps every address in `keraunos_ubench.h` helpers, and the host-side `UbenchSession` builds them, so a rebase should only touch those.

## Files

| Path | What it holds |
|---|---|
| `tests/tt_metal/tt_metal/test_kernels/misc/keraunos_ubench/keraunos_ubench.h` | SRAM layout, the `Result` block, enums, address helpers, and device helpers (`now()` fenced rdcycle, `uncached()`, `gate()`, `wait_eq()`) |
| `.../keraunos_ubench/timer_check.cpp` | rdcycle sanity checks: monotonic, bracket cost, add chain, uncached load chains |
| `.../keraunos_ubench/latency.cpp` | Ping-pong initiator and responder; transport is CpuStore, Idma or Eth |
| `.../keraunos_ubench/transport_bw.cpp` | Per-hart bandwidth: IdmaMetal, IdmaLean or CpuStore64; start gate across harts |
| `.../keraunos_ubench/eth_bw_sender.cpp`, `eth_bw_receiver.cpp` | Streamed Ethernet sends with a handshake and a closing flag |
| `tests/tt_metal/tt_metal/context/keraunos_session.hpp` | `Session`, shared with `KeraunosEthSend`: device open, CCE0 lookup per HSIO tile, SRAM read and write, the Ethernet-link slot, and launch with a hang watchdog. Fails if the descriptor has no Keraunos `tile_stride` |
| `tests/tt_metal/tt_metal/perf_microbenchmark/keraunos/test_keraunos_ubench.cpp` | Host side: `UbenchSession` (the shared `Session` plus result blocks and address helpers; one device open for the whole binary), one parameterized test per sweep point, statistics, data checks and the result log. Its header comment lists the run filters |
| `tests/tt_metal/tt_metal/perf_microbenchmark/sources.cmake`, `CMakeLists.txt` | Builds the `test_keraunos_ubench` executable under `TT_UMD_BUILD_GRENDEL_JTAG`, with the quasar include dir for `dev_mem_map.h` |

## Tests

The tests build into their own executable, `build/test/tt_metal/perf_microbenchmark/keraunos/test_keraunos_ubench`, like the other metal perf microbenchmarks. Each sweep point is its own gtest case (`INSTANTIATE_TEST_SUITE_P`), so `--gtest_filter` picks what runs; there are no opt-in variables. Parameterized names carry the `KeraunosUbench/` prefix, e.g. `KeraunosUbench/KeraunosUbenchTransportBw.Run/IdmaLean_Tile_8Harts_4096B`.

| Suite | Cases | Measures | Needs |
|---|---|---|---|
| `KeraunosUbench.TimerCheck` | 1 | rdcycle monotonicity, bracket min and max, cycles per add, cycles per dependent uncached load (own SRAM and another tile) | CCE0 on HSIO 0 (1 for the remote chain) |
| `KeraunosUbenchLatency.PingPong` | `{SameCore,CrossTile}{CpuStore,Idma}` | RTT and TURN at p50/p90/p99 | Cross-tile needs the HSIO firewalls open (see below) |
| `KeraunosUbenchTransportBw.Run` | `{IdmaMetal,CpuStore64}_{Local,Tile}_{1,8}Harts_{256,4096,65536}B`, `IdmaLean_{Local,Tile}_{1,8}Harts_{256,4096}B` | INJECTION and DRAINED per hart (RETIRED for CpuStore64), and the run rate over the slowest hart | Same as above for `Tile` |
| `KeraunosUbenchEthLatency.PingPong` | `{0,64,1024,4096}B` | RTT and TURN over HSIO0.eth0↔HSIO1.eth0 | Loopback up in packet mode |
| `KeraunosUbenchEthBandwidth.Run` | `{256,4096,65536}B_{1,4}Txq` | INJECTION, ACKED and DELIVERED, the idle handshake RTT, and TX-queue busy polls | Same as above |

The `KeraunosUbenchEth*` suites have no loopback check: without the loopback their kernels hang until the 1800 s host timeout. Leave them out of the filter unless bring-up started it.

**Measurement rules** (taken from the X280 suite's `design.md`):
- A kernel reads only its own hart's rdcycle, and never subtracts another hart's timestamps.
- The device writes raw cycles; the host computes every statistic, using floor-index percentiles.
- A round trip is never halved into a one-way time.
- Every bandwidth case checks the moved data with a host readback. This matters because of the NIU write-ack issue: an ack does not prove the data landed.

**Results:**
- Logged with `log_info`, one line per case: p50/p90/p99 for latency, rates for bandwidth. Raw samples are not written out.
- Values are emulator RTL cycles. Clock ratios between domains on the emulator (CCE, NoC, D2D, Ethernet) need not match silicon.

## How to build and run

Run s1 used the cce-metal Phase 0a setup (`cce-metal/runs/RUNS.md` rows p0a-400g and s64-v3 in the docs repo):
- **Model:** `mmk` `26ww45.eth-pcs-output-inter`, which has the SerDes cross-tile loopbacks HSIO0↔HSIO1 and HSIO2↔HSIO3. Server `test_metal_server.py +prod_rom_disabled=1 +serdes_data_rate=400` from `grendelemulation-eth-bringup` (`kstevens/metal_bringup` `057ad21` plus the 26ww45 model-selection patch).
- **Bring-up:** `bringup_mmk.py` from chippy `c498fe024` with `MMK_ETH_RATE=400g MMK_ETH_LPBK=none`. It brings both links up at 400G and opens the HSIO fabric firewalls (`disable_ker_hsio_filters` per tile); without the firewalls, cross-tile stores fail with DECERR.
- **Build:** `build_Release` with `-DTT_UMD_BUILD_GRENDEL_JTAG=ON` and the `CHIPPY_*` dirs, in the `metal-emu-dev` container, targets `test_keraunos_ubench` and `unit_tests_context`.
- **Scripts** (soc-l-10): `/localdev/kzhao/ubench/emu-session.sh <tag>` starts the server and runs bring-up; `/localdev/kzhao/ubench/run.sh <tag> <host> <binary> <filter>` runs one gtest invocation with the variables below. Release the emulator afterwards with `emu kill -d <zrun-dir>`.

```
export TT_METAL_EMU_SERVER=<host>:<port>
export TT_METAL_EMU_SOC_DESC=$TT_METAL_HOME/tt_metal/third_party/umd/tests/soc_descs/mmk.yaml
export TT_METAL_SLOW_DISPATCH_MODE=1
B=./build/test/tt_metal/perf_microbenchmark/keraunos/test_keraunos_ubench
$B --gtest_filter='KeraunosUbench.TimerCheck'     # first
$B --gtest_filter='-*KeraunosUbenchEth*'          # every on-chip case
$B --gtest_filter='*KeraunosUbenchEth*'           # Ethernet, once the loopback is up
```

If the host times out, a kernel is hung. The test prints every result block and exits. In run s1 the next device open reloaded the cores and recovered without restarting the server.

## Results of run s1

Server soc-zebu-02 (zrun.31), 2026-11-17. Values are emulator cycles of the measuring CCE hart; rates are bytes per CCE cycle. Logs: `/localdev/kzhao/ubench/runs/*.log` on soc-l-10. `KeraunosEmu.*` and `KeraunosEthSend/*` (`unit_tests_context`) also passed 6/6 on the same server, after the shared `Session` change.

**TimerCheck:** bracket 3..27 cycles; 4.08 cycles per add-loop iteration; dependent uncached loads 33.8 cycles (own SRAM) and 175.5 (HSIO1 global address); 0 of 1024 reads backwards, 0 zero deltas.

| Latency (p50 / p99) | RTT | TURN |
|---|---|---|
| SameCore CpuStore (harts 0↔1) | 182 / 210 | 25 / 25 |
| SameCore Idma | 411 / 442 | 108 / 113 |
| CrossTile CpuStore (HSIO0↔1) | 344 / 382 | 24 / 24 |
| CrossTile Idma | 551 / 576 | 108 / 109 |
| Eth, token only | 1507 / 1547 | 145 / 149 |
| Eth, 64 B payload | 1800 / 1838 | 296 / 301 |
| Eth, 1024 B | 1803 / 1831 | 294 / 301 |
| Eth, 4096 B | 1983 / 1996 | 297 / 298 |

TransportBw run rate over the slowest hart, B/cyc, INJECTION / DRAINED (CpuStore64: RETIRED):

| Case | 256 B | 4096 B | 65536 B |
|---|---|---|---|
| IdmaMetal Local, 1 hart | 4.05 / 3.97 | 65.0 / 63.2 | 122.9 / 105.2 |
| IdmaMetal Local, 8 harts | 31.9 / 31.3 | 117.0 / 112.2 | 115.8 / 113.7 |
| IdmaMetal Tile, 1 hart | 4.05 / 3.94 | 64.8 / 62.1 | 90.2 / 77.0 |
| IdmaMetal Tile, 8 harts | 28.6 / 25.4 | 76.4 / 72.6 | 78.8 / 77.2 |
| IdmaLean Local, 1 hart | 5.20 / 5.08 | 83.4 / 80.6 | — |
| IdmaLean Local, 8 harts | 41.3 / 39.4 | 117.3 / 112.2 | — |
| IdmaLean Tile, 1 hart | 5.21 / 5.04 | 83.3 / 69.4 | — |
| IdmaLean Tile, 8 harts | 29.1 / 25.4 | 76.4 / 72.6 | — |
| CpuStore64 Local, 1 hart | 0.561 | 0.612 | not run |
| CpuStore64 Local, 8 harts | 4.00 | 4.00 | not run |
| CpuStore64 Tile, 1 hart | 0.561 | 0.612 | not run |
| CpuStore64 Tile, 8 harts | 0.834 | 0.833 | not run |

| EthBandwidth | INJECTION | ACKED | DELIVERED | Idle RTT | Busy-queue polls |
|---|---|---|---|---|---|
| 256 B, 1 TX queue | 1.56 | 1.56 | 1.44 | 978 | 0 |
| 256 B, 4 | 1.54 | 1.54 | 1.43 | 966 | 0 |
| 4096 B, 1 | 24.8 | 24.7 | 22.7 | 962 | 0 |
| 4096 B, 4 | 24.9 | 24.8 | 22.8 | 981 | 0 |
| 65536 B, 1 | 31.3 | 30.6 | 30.2 | 1906 | 502 |
| 65536 B, 4 | 31.7 | 30.6 | 30.3 | 1601 | 0 |

**Emulator time.** The chip advances only while the server serves a host access (CM-107), so a case's wall time follows the host's polling and SRAM traffic, not the cycles it measures. The 8-hart 65536 B iDMA cases took about 30 min each; the CpuStore64 4096 B cases 5 to 30 min. At 0.1 to 0.6 B/cyc, a CpuStore64 65536 B case would need about 2 h, so run s1 skipped all four.

**Two bugs found and fixed in s1:** an iDMA copy to another tile used the chiplet-local view (`CrossTileIdma` hung, assumption 3), and the Ethernet bandwidth flag was sent to the bare `kEthFlagOffset` instead of `base + kEthFlagOffset` (`EthBandwidth` hung at its handshake).

## Assumptions to check first

1. **rdcycle** keeps counting through uncached load and iDMA stalls. Run s1 confirmed this: dependent uncached loads cost 33.8 and 175.5 cycles, against 4.08 per add-loop iteration. That iteration is the `addi` plus the loop's own counter and branch, so it is not 1 cycle per add; whether rdcycle runs at the core clock is still open (see the clock question below).
2. **`IdmaLean`** assumes command buffer 0 keeps its VC, TRID and copy-mode settings across issues, so that only src, dst and len need setting per transfer. Run s1 confirmed this: every IdmaLean case moved the right data, at 5.2 B/cyc against IdmaMetal's 4.05 at 256 B.
3. **On-chip iDMA addressing** uses the chiplet-local view `0x2000_0000 + hsio*tile_stride + 0x280_0000 + off` for the copying core's own tile, the same as `eth_cce_send.h`'s `sram_local_addr()`, and the system address `0x12_0680_0000 + hsio*tile_stride + off` for another tile, as metal's `quasar_cce_noc_routes()` does. Run s1 on the emulator (2026-11-17) confirmed this: with the local view for another tile, `CrossTileIdma` hung at its first exchange. **CPU addressing** uses the global window `0x12_0680_0000 + hsio*tile_stride + off` for another tile, and the uncached alias (`+0x80_0000`) on the same core.
4. **Ethernet.** RX writes are not snooped, so mailboxes and flags are read through the uncached alias. Payloads land at the same flat offset on the link peer. The closing flag in `EthBandwidth` goes out on TX queue 0 only after every send is acked, so DELIVERED is an upper bound and includes one return path.
5. **SRAM region:** `base = MEM_IERISC_ROUTER_UNRESERVED_BASE` (rounded up to 64) plus `kRegionBytes = 0x110000`, which must fit in 4 MiB. The shared `Session` constructor checks this with `TT_FATAL`.

## Next steps

1. Record the CCE clock under `test_metal_server.py`, so cycles can be converted.
2. Rebase onto Kevin Stevens's next branch. Then add CCE1 and CCE0↔CCE1 pairs (same tile, sibling-CCE view at +4 MiB).
3. Add Keraunos→Mimir destinations (M1 CCE SRAM, M1 GDDR, and M0 GDDR, which is multi-hop) to TransportBandwidth and Latency, using the `bringup_mmk_fw.py` hybrid bring-up.
4. The CpuStore64 65536 B cases: run them only on a server of their own, or drop them; 4096 B already shows the steady rate.
5. Later: atomic_bench, core_sync and pipeline_stage ports. A packet_injection variant with CCE producers. eth_credit once fabric on Keraunos is stable. Quasar Tensix peers when `qsr.s1` support lands.

## Open questions for Kevin Stevens

- When does the next branch land? Is one core per CCE the final model?
- Does the CCE have a fixed-rate timer (CLINT `mtime` or a NEO wall clock)? What is the CCE clock on `mmk`?
- What are the gotchas for cached local atomics? Is there an ETA on the iDMA atomic encodings?
- Will `keraunos_cce.h` (the draft device API) be the supported surface?

## Context sources

- Slack DM between Kevin Zhao and Kevin Stevens, Oct 22 to Nov 13 2026: CCE0-only limits, D2D emulation fix, uncached-offset guidance, mmk vs Grendel addressing.
- `keraunos_plan.md`, `keraunos_worklog.md`, `grendel_keraunos.md` and `mimir_setup.md` on `kstevens/enable_keraunos`.
- PR #60692 `tt_metal/fabric/experimental/x280/docs/` (`design.md`, `measurement.md`, `benchmarks/*.md`) for methodology.
