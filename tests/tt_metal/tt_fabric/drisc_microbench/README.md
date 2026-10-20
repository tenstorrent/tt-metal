# DRISC microbenchmark

Measures what it costs a DRISC to move fabric router packets, and what it costs an ERISC to hand packets to it. It
replaces the estimated DRISC costs in the placement model (`fabric_exp/model/catalog.yaml`). No fabric runs: one
program puts a kernel on a DRISC and one on ERISC1 of an active Ethernet core (the receiving router's ERISC, on NoC1
as today), and the two step through phases together. See the header of `test_drisc_microbench.cpp` for the phases.

Fabric must stay disabled, because its routers occupy the ERISCs. The test never sets a fabric config and launches
both kernels as an ordinary program. Metal still turns fabric on for dispatch when fast dispatch reaches remote chips,
or on a Galaxy; the test then fails up front. The test needs slow dispatch anyway, which never does this.

## Build and run

```bash
cmake --build build --target test_drisc_microbench
export TT_METAL_SLOW_DISPATCH_MODE=1    # fast dispatch cannot launch DRISC programs
./build/test/tt_metal/tt_fabric/test_drisc_microbench                      # 4 KiB payload, mcycle timers
DRISC_MB_CSV=fabric_exp/results/drisc_mb.csv ./build/test/tt_metal/tt_fabric/test_drisc_microbench
```

Environment: `DRISC_MB_PAYLOAD` (4096), `DRISC_MB_N` (64 back-to-back ops), `DRISC_MB_N_RT` (32 round trips),
`DRISC_MB_PIPE_PKTS` (512), `DRISC_MB_DEPTHS` (`1,2,4,8,12,16`), `DRISC_MB_DRISC_INDEX`,
`DRISC_MB_ETH_INDEX`, `DRISC_MB_WALL_CLOCK` (1: time the DRISC with its wall clock instead of mcycle),
`DRISC_MB_OPT` (O3, as the router builds), `DRISC_MB_CSV`.

The host picks a Metal DRISC whose NIU on the ERISC's NoC is in stream mode, preferring one with both NIUs in stream
mode so NoC0 and NoC1 are both measured. Each kernel gives up after ~1 s in any wait and reports the phase; the test
then fails instead of hanging.

## Requirements and what the first runs showed

- **Firmware bundle >= 19.12.0.** Older bundles run syseng firmware on a DRAM core, so Metal does not enable DRISCs
  and the test skips. Do not force them on with `TT_METAL_ENABLE_BLACKHOLE_DRAM_PROGRAMMABLE_CORES=1` there.
- **Slow dispatch.** Fast dispatch cannot launch DRISC programs (`Invalid core type get_dispatch_write_offset DRAM`);
  the test skips without `TT_METAL_SLOW_DISPATCH_MODE=1`.
- **ERISC1 uses NoC1.** Metal requires it in dual-ERISC mode, as the receiving router does.
- **Code size.** A DRISC kernel's code must fit ~11 KB (0x2bd0 bytes on this firmware). The benchmark runs as three
  programs (`DRISC_MB_PART` 1: per-operation costs, NoC operations, round trips; 2: the pipelines; 3: the DRISC's
  command buffers and NoC ports) so each fits at O3. Changing `DRISC_MB_OPT` does not rebuild a cached kernel (the optimization level is not part of
  Metal's kernel hash for DRISC or Ethernet kernels); clear `~/.cache/tt-metal-cache/*/kernels/drisc_mb_*` first.
- **Timer.** The DRISC's mcycle works and ticks 1.000 per wall-clock tick; `DRISC_MB_WALL_CLOCK=1` gives the same
  results. The O3 DRISC kernel fits.
- Results are repeatable to 0.1 cycle across runs, DRISCs and Ethernet cores, except round-trip latency, which
  depends on placement. Rebuilding after unrelated kernel edits moves back-to-back figures by up to ~7 cycles
  (code layout), so compare figures from one build. Results: `fabric_exp/RESULTS.md` §14.
- "issue" has no barrier: a payload write waits for the command buffer, busy until the previous write has left,
  so back-to-back 4 KiB issue is the NoC's rate. "alone" times one operation with the NoC idle, its values
  computed before the timer starts: the CPU cost with the port free (includes one ~7-cycle timer read).

- **Other Ethernet cores.** Part 3 also writes into the L1 landing area of up to two other active Ethernet cores
  (the next ones after `DRISC_MB_ETH_INDEX`), as a router forwards downstream; they run no kernel, so fabric must
  be off (it is).
- **Part 4 hang.** Running both DRISCs at once on different NoCs (part 4 variants 3 and 4: pull NoC0 + write NoC1,
  pull NoC1 + write NoC0) hung the chip; they are left out unless `DRISC_MB_TWO_MASK` includes bits 3-4. Recover
  with `tt-smi -r all`: resetting chip 0 alone once left its card off the PCIe bus. `DRISC_MB_PARTS` (`1,2,3,4`)
  and `DRISC_MB_TWO_SPIN=1` (an ERISC kernel polls during part 4) help narrow problems down.
- **Command-buffer state persists across kernels.** Firmware sets the NoC command buffers once at boot (`noc_init`)
  and the NoC API relies on that state: the read buffer's control word, and the source coordinate of the write and
  register-write buffers. The DRISC kernel restores it on exit (`noc_init_one`); writes never use the read buffer.

## What each number feeds in the model

| Output | Model parameter |
| --- | --- |
| DRISC `forward copy ..., alone` (stateless / stateful + trid / lean) | `core_types.drisc.forward_per_copy`: DRISC CPU per copy with its port free |
| DRISC `forward copy ..., issue` / `drain` per NoC | the DRISC's copy rate when back to back (the NoC's rate, not CPU) |
| DRISC `write payload -> ERISC, drain` B/cycle per NoC | `core_types.drisc.egress_bytes_per_cycle`, `noc_ports` |
| ERISC `push ..., alone` (stateless / stateful + trid / lean) | `connectors.erisc_to_drisc_payload.producer` (+ `producer_per_batch` for the doorbell) |
| ERISC `push ..., issue` / `drain` | the Ethernet tile's port time per push (`noc.egress_bytes_per_cycle`) |
| DRISC `stream register read (doorbell poll)` | `polls.handoff` on a DRISC |
| DRISC `doorbell -> ERISC stream reg, issue` | the DRISC side of staging-credit returns (`connectors.drisc_to_erisc`) |
| `ping` round trips, `write payload + wait for ack` | `noc.write_latency_cycles`; latency added by the DRISC hop |
| pipeline cycles/packet vs depth, stateless and lean; DRISC work/packet | staging slots needed; DRISC load per forwarded packet |
| `DRISC/ERISC` load and stream-read ratios | `core_types.drisc.alpha` for load-bound work |
| ERISC `stream register read, 4 independent (per read)` vs one | whether polling several channels at once overlaps the reads (`polls.fused_per_channel`) |

The pipeline forwards back into the same ERISC, so DRISC -> ERISC and ERISC -> DRISC traffic share that tile's NIU;
a router would forward to another Ethernet core.
