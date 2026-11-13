# Power Virus

A C++ gtest (`unit_tests_didt_power_virus`, Blackhole only) that holds constant load in four domains, each turned on and throttled on its own, until a duration elapses or Ctrl-C.
It needs slow dispatch.

| Domain | Generator | Knob |
|---|---|---|
| TRISC (FPU) | MVMUL loop from pre-filled L1 on every worker | `PV_TRISC` |
| DRAM | BRISC on every worker reads round-robin across all DRAM banks | `PV_DRAM` |
| NOC | NCRISC posted L1-to-L1 writes along long row/column ring routes | `PV_NOC` |
| Ethernet | Bidirectional traffic on every link between chips in the same run | `PV_ETH` |

```bash
export TT_METAL_SLOW_DISPATCH_MODE=1
# Everything at 100% on every chip for 60 s
PV_TRISC=100 PV_DRAM=100 PV_NOC=100 PV_ETH=100 PV_SUPER_SYNC=1 PV_DURATION_S=60 \
  ./build/test/didt/unit_tests_didt_power_virus --gtest_filter="*AllDevices*"
# One chip, FPU at 50% until Ctrl-C
PV_TRISC=50 PV_DURATION_S=0 ./build/test/didt/unit_tests_didt_power_virus --gtest_filter="*SingleDevice*"
```

Tests:
- `PowerVirus_Smoke`: device 0 with TRISC, DRAM and NOC at 100% and super-sync, for 2 s.
- `PowerVirus_SingleDevice`: device 0.
- `PowerVirus_AllDevices`: every chip. Ethernet only runs here, because both ends of a link must run together. Links that leave the cluster are skipped.

Run knobs:

| Knob | Meaning | Default |
|---|---|---|
| `PV_DURATION_S` | Run time in seconds; 0 runs until Ctrl-C | 10 (Smoke: 2) |
| `PV_SUPER_SYNC` | 1 starts every Tensix generator on a chip in the same window | 0 (Smoke: 1) |
| `PV_DUTY_CYCLE_PCT` | Chip-wide hot/quiet duty cycle over every generator | 100 (off) |
| `PV_DUTY_PERIOD_512` | Duty-cycle period in units of 512 AICLK ticks, a power of two | 4096 (~1.6 ms at 1.35 GHz) |

Per-domain knobs, where `<X>` is `PV_TRISC`, `PV_DRAM`, `PV_NOC` or `PV_ETH`:

| Knob | Meaning | Default |
|---|---|---|
| `<X>` | Utilization as a percentage of the domain's own maximum; 0 turns it off | TRISC 100, the others 0 |
| `<X>_MODE` | Below 100: `smooth` spreads the load evenly; `phase` runs at full rate for that share of every period, with every core on the chip switching together | `smooth` |
| `<X>_PERIOD_512` | Period of `phase` mode in units of 512 ticks, a power of two | 256 (~97 µs) |
| `<X>_BURST`, `<X>_IDLE_512` | Burst mode: units of work, then idle (units of 512 ticks). Overrides `<X>` and `<X>_MODE` | unset |
| `<X>_XFER_BYTES` | Transfer size, a power of two (not TRISC). DRAM 64–16384, NOC 64–65536, ETH 64–8192 | DRAM 8192, NOC 65536, ETH 8192 |

How the smooth mode throttles each domain:
- TRISC: MATH pads every tile with a fixed number of NOPs, interpolated from the FPU calibration. Below about 11% it falls back to `phase`.
- NOC: each route times every transfer and then idles for `transfer time × (100 − pct) / pct`.
- DRAM and Ethernet: each core or link issues one transfer at a fixed interval. The interval is set from the full rate (64 GB/s per DRAM bank shared across all cores, 48 GB/s per Ethernet link) at the nominal 1.35 GHz AICLK. When the chip throttles, these rates scale down with AICLK.

At 100% the FPU runs at about 93% utilization, which is the maximum for this loop.

The test fails if:
- any enabled generator does no work on some core;
- the FPU output on a core is missing, non-finite or all zero;
- an Ethernet link does not shut down cleanly.

When a Tensix generator runs without the other Tensix generators, the test also checks it against what its knobs should give:
- FPU utilization within ±4 points of the expected value;
- DRAM and NOC bandwidth above 60% of the expected rate.

Ethernet bandwidth is always checked the same way. With all generators on, the chip throttles AICLK to its floor and the generators compete for L1 and the NOC, so the test only reports their numbers.
