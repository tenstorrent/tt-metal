# RISC-V performance counters

The reference moved next to the Tensix counter documentation:
[tt_metal/tt-llk/docs/performance_counters/riscv_pmc.md](../../tt_metal/tt-llk/docs/performance_counters/riscv_pmc.md).

```bash
export TT_METAL_PROFILE_RISCV_PMC=0x10110111,0xA0000000   # mhpmevent3, mhpmevent4
export TT_METAL_PROFILE_RISCV_PMC_DM=0x201,0x801,0x202,0x402  # Quasar data-movement cores only, one to four selects
```

The option captures the RISC-V performance monitor of the Blackhole baby cores and ERISC0 of the active and idle ethernet cores and the Quasar TRISC and
data-movement cores into the device profiler (markers 9092 to 9096). Wormhole has no counters.
