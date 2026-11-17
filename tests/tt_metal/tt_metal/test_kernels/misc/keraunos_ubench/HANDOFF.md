# Keraunos CCE microbenchmarks: handoff

**Status:** the code is written and pushed, but has never been built or run. It needs a first build and a run on the `mmk` emulator.
**Branch:** `claude/glean-access-ggmie0`, one commit on top of `kzhao/keraunos-three-device`.
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
  - The benchmark code keeps every address in `keraunos_ubench.h` helpers, and the host-side `Session` builds them, so a rebase should only touch those.

## Files

| Path | What it holds |
|---|---|
| `tests/tt_metal/tt_metal/test_kernels/misc/keraunos_ubench/keraunos_ubench.h` | SRAM layout, the `Result` block, enums, address helpers, and device helpers (`now()` fenced rdcycle, `uncached()`, `gate()`, `wait_eq()`) |
| `.../keraunos_ubench/timer_check.cpp` | rdcycle sanity checks: monotonic, bracket cost, add chain, uncached load chains |
| `.../keraunos_ubench/latency.cpp` | Ping-pong initiator and responder; transport is CpuStore, Idma or Eth |
| `.../keraunos_ubench/transport_bw.cpp` | Per-hart bandwidth: IdmaMetal, IdmaLean or CpuStore64; start gate across harts |
| `.../keraunos_ubench/eth_bw_sender.cpp`, `eth_bw_receiver.cpp` | Streamed Ethernet sends with a handshake and a closing flag |
| `tests/tt_metal/tt_metal/perf_microbenchmark/keraunos/test_keraunos_ubench.cpp` | Host side: `Session` (one device open for the whole binary), one parameterized test per sweep point, statistics, data checks, JSONL records. Its header comment lists the run filters and the environment variables |
| `tests/tt_metal/tt_metal/perf_microbenchmark/sources.cmake`, `CMakeLists.txt` | Builds the `test_keraunos_ubench` executable under `TT_UMD_BUILD_GRENDEL_JTAG`, with the quasar include dir for `dev_mem_map.h` |

## Tests

The tests build into their own executable, `build/test/tt_metal/perf_microbenchmark/keraunos/test_keraunos_ubench`, like the other metal perf microbenchmarks. Each sweep point is its own gtest case (`INSTANTIATE_TEST_SUITE_P`), so `--gtest_filter` picks what runs; there are no opt-in variables. Parameterized names carry the `KeraunosUbench/` prefix, e.g. `KeraunosUbench/KeraunosUbenchTransportBw.Run/IdmaLean_Tile_8Harts_4096B`.

| Suite | Cases | Measures | Needs |
|---|---|---|---|
| `KeraunosUbench.TimerCheck` | 1 | rdcycle monotonicity, bracket min and max, cycles per add, cycles per dependent uncached load (own SRAM and another tile) | CCE0 on HSIO 0 (1 for the remote chain) |
| `KeraunosUbenchLatency.PingPong` | `{SameCore,CrossTile}{CpuStore,Idma}` | RTT and TURN at p50/p90/p99, plus the rendezvous exchange | Cross-tile needs the HSIO firewalls open (see below) |
| `KeraunosUbenchTransportBw.Run` | `{IdmaMetal,IdmaLean,CpuStore64}_{Local,Tile}_{1,8}Harts_{256,4096,65536}B` | INJECTION and DRAINED per hart, and the run rate over the slowest hart | Same as above for `Tile` |
| `KeraunosUbenchEthLatency.PingPong` | `{0,64,1024,4096}B` | RTT and TURN over HSIO0.eth0↔HSIO1.eth0 | Loopback up in packet mode |
| `KeraunosUbenchEthBandwidth.Run` | `{256,4096,65536}B_{1,4}Txq` | INJECTION, ACKED and DELIVERED, the idle handshake RTT, and TX-queue busy polls | Same as above |

The `KeraunosUbenchEth*` suites have no loopback check: without the loopback their kernels hang until the 1800 s host timeout. Leave them out of the filter unless bring-up started it.

**Measurement rules** (taken from the X280 suite's `design.md`):
- A kernel reads only its own hart's rdcycle, and never subtracts another hart's timestamps.
- The device writes raw cycles; the host computes every statistic, using floor-index percentiles.
- A round trip is never halved into a one-way time.
- Every bandwidth case checks the moved data with a host readback. This matters because of the NIU write-ack issue: an ack does not prove the data landed.

**Results:**
- Appended to `$KERAUNOS_UBENCH_OUTPUT`, default `keraunos_ubench.jsonl`, as one JSON object per configuration.
- Values are emulator RTL cycles. Clock ratios between domains on the emulator (CCE, NoC, D2D, Ethernet) need not match silicon.
- Set `KERAUNOS_UBENCH_CCE_MHZ` to also log ns and GB/s.

## How to build and run

The environment follows `mimir_setup.md` and Kevin Stevens's Keraunos instructions:
- `grendelemulation`: `kstevens/metal_bringup`, using `test_metal_server.py`
- `chippy`: `kstevens/metal_bringup`
- `tt-umd`: `kstevens/grendel-emu-additions`
- Build with `-DTT_UMD_BUILD_GRENDEL_JTAG=ON` and the `CHIPPY_*` dirs.

Bring-up must also open the HSIO fabric firewalls (`disable_ker_hsio_filters` per tile). Without that, cross-tile stores fail with DECERR. Kevin Zhao's `bringup_mmk.py` / `firewall.py` patch was sent to Kevin Stevens on 11-04.

```
export TT_METAL_EMU_SERVER=<host>:<port>
export TT_METAL_EMU_SOC_DESC=$TT_METAL_HOME/tt_metal/third_party/umd/tests/soc_descs/mmk.yaml
export TT_METAL_SLOW_DISPATCH_MODE=1
B=./build/test/tt_metal/perf_microbenchmark/keraunos/test_keraunos_ubench
$B --gtest_filter='KeraunosUbench.TimerCheck'     # first
$B --gtest_filter='-*KeraunosUbenchEth*'          # every on-chip case
$B --gtest_filter='*KeraunosUbenchEth*'           # Ethernet, once the loopback is up
```

If the host times out, a kernel is hung. The test prints every result block and exits, and the emulator job needs a reset before the next run.

## Verification so far

- The kernels pass a riscv64 clang `-fsyntax-only -Wall -Wextra` check, but against **stub** headers. The real SFPI build has not been run, so the signatures of the overlay, cce_noc and cce_eth calls are not checked.
- `keraunos_ubench.h` compiles in host mode.
- `test_keraunos_ubench.cpp` was reviewed by hand only. It has never been compiled.

## Assumptions to check first

1. **rdcycle** keeps counting through uncached load and iDMA stalls, and is a real cycle count. `TimerCheck` tests both: the add chain should come out at about 1 cycle per add, and the load chains should be much slower than that.
2. **`IdmaLean`** assumes command buffer 0 keeps its VC, TRID and copy-mode settings across issues, so that only src, dst and len need setting per transfer. No existing test confirms this. If it misbehaves, compare it against `IdmaMetal`.
3. **On-chip iDMA addressing** uses the chiplet-local view `0x2000_0000 + hsio*tile_stride + 0x280_0000 + off`, the same as `eth_cce_send.h`'s `sram_local_addr()`. **CPU addressing** uses the global window `0x12_0680_0000 + hsio*tile_stride + off` for another tile, and the uncached alias (`+0x80_0000`) on the same core.
4. **Ethernet.** RX writes are not snooped, so mailboxes and flags are read through the uncached alias. Payloads land at the same flat offset on the link peer. The closing flag in `EthBandwidth` goes out on TX queue 0 only after every send is acked, so DELIVERED is an upper bound and includes one return path.
5. **SRAM region:** `base = MEM_IERISC_ROUTER_UNRESERVED_BASE` (rounded up to 64) plus `kRegionBytes = 0x110000`, which must fit in 4 MiB. The `Session` constructor checks this with `TT_FATAL`.

## Next steps

1. Build, run `TimerCheck`, then run everything. Fix compile errors against the real headers.
2. Record the CCE clock under `test_metal_server.py`, so cycles can be converted.
3. Rebase onto Kevin Stevens's next branch. Then add CCE1 and CCE0↔CCE1 pairs (same tile, sibling-CCE view at +4 MiB).
4. Add Keraunos→Mimir destinations (M1 CCE SRAM, M1 GDDR, and M0 GDDR, which is multi-hop) to TransportBandwidth and Latency, using the `bringup_mmk_fw.py` hybrid bring-up.
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
