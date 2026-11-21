# Tensix performance counters

Header-only description of the hardware performance counters, namespace `llk::perf`.

- `types.h`: `PerfCounterType` (counter names, ordinal is the wire format), `Bank`, `Entry`.
- `blackhole.h`, `wormhole.h`, `quasar.h`: per-bank `{name, select}` tables and the L1 mux width.
- `inventory.h`: picks the arch table from `ARCH_*` and exposes `table_for(bank, l1_mux)`.
- `registers.h`: `BankRegs`, `bank_regs(bank)`, shared register addresses and bit constants.
- `hw.h`: register primitives (`configure`, `start`, `stop`, `select`, `read_table`, `set_l1_mux`).

RISC-V performance monitor (the counters inside the cores, not the Tensix banks above), namespace
`llk::perf::riscv`:

- `riscv_pmc_types.h`: `CoreKind` (select encoding of a core family), `PmcEvent` (event names), `Entry`.
- `riscv_pmc_blackhole.h`, `riscv_pmc_wormhole.h`, `riscv_pmc_quasar.h`: per-core-kind `{event, select}` tables
  (`baby_events`, `trisc_events`, `dm_events`); Wormhole's is empty because those cores have no counters.
- `riscv_pmc_inventory.h`: picks the arch table from `ARCH_*` and exposes `events_for(CoreKind)`.
- `riscv_pmc_hw.h`: the CSR addresses and `select`, `zero_counters`, `read_mcycle`, `read_minstret`,
  `read_counter3`, `read_counter4` and the counter 5 and 6 variants used by the Quasar data-movement cores.

NoC interface (NIU) event counters, namespace `llk::perf::noc`:

- `noc_counters_types.h`: `NocCounter` (names), `NocGroup`, group masks, `Entry` = `{counter, register index, group}`.
- `noc_counters_blackhole.h`, `noc_counters_wormhole.h`, `noc_counters_quasar.h`: the `noc_counters` table in capture
  order plus `NIU_STATUS_BASE`, `NIU_NOC_STRIDE`, `NIU_NOCS` and `NIU_DATA_WORD_BYTES`.
- `noc_counters_inventory.h`: picks the arch table from `ARCH_*`, `selected_regs<GroupMask>()`.
- `noc_counters_hw.h`: `read(noc, reg)`.

Reference: `docs/performance_counters/noc_counters.md`.

A profiler record carries the raw mhpmevent select, not an ordinal, so `PmcEvent` is only for names: append new
names, the host decodes a select by looking it up in the arch table. Reference: `docs/performance_counters/riscv_pmc.md`.

Two consumers use these headers and nothing else describes the counters:

- the metal profiler (`tt_metal/hw/inc/internal/profiler/perf_counters.hpp`), which adds its record format,
  group bit contract and emission into the profiler buffer;
- the LLK perf harness (`tt_metal/tt-llk/tests/helpers/include/counters.h`), which adds its L1 ABI,
  zones and barriers.

Adding a counter: append the name to the end of `PerfCounterType` (never reorder or remove, the
ordinals are decoded by position on the host), add one `{PerfCounterType::NAME, select}` entry to the
right bank table in the arch header, and bump the size in that table declaration. Adding a RISC-V event: append the name to `PmcEvent`, add `{PmcEvent::NAME, select}` to the right table and
bump its size. The Python side
(`tt_metal/tt-llk/tools/python/tt_llk_perf`) parses names and tables from these headers, so nothing
else has to change.

Quasar has one debug block per NEO and two ways to reach it: a TRISC uses the local window
(`LOCAL_REGS_WINDOW`, the default), the DM cores use the NoC window (`neo_window(n)`); every register
primitive takes the window as its last argument. There is no L1 counter bank (`table_for(Bank::L1)` is
empty, `L1_MUX_POSITIONS` is 0); instead one clear-on-read l1_client CSR counts one `subport*8 + event`
selection (`l1_client_regs`, `l1_client_start/stop/read`, validity in `l1_client_selection_is_valid`).
Metal defines `LLK_PERF_TABLES_IN_TEXT` before including the headers so the DM firmware keeps the
tables out of its 2 KB data region.
