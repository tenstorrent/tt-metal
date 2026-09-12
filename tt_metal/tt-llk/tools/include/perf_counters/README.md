# Tensix performance counters

Header-only description of the hardware performance counters, namespace `llk::perf`.

- `types.h`: `PerfCounterType` (counter names, ordinal is the wire format), `Bank`, `Entry`.
- `blackhole.h`, `wormhole.h`, `quasar.h`: per-bank `{name, select}` tables and the L1 mux width.
- `inventory.h`: picks the arch table from `ARCH_*` and exposes `table_for(bank, l1_mux)`.
- `registers.h`: `BankRegs`, `bank_regs(bank)`, shared register addresses and bit constants.
- `hw.h`: register primitives (`configure`, `start`, `stop`, `select`, `read_table`, `set_l1_mux`).

Two consumers use these headers and nothing else describes the counters:

- the metal profiler (`tt_metal/tools/profiler/perf_counters.hpp`), which adds its record format,
  group bit contract and emission into the profiler buffer;
- the LLK perf harness (`tt_metal/tt-llk/tests/helpers/include/counters.h`), which adds its L1 ABI,
  zones and barriers.

Adding a counter: append the name to the end of `PerfCounterType` (never reorder or remove, the
ordinals are decoded by position on the host), add one `{PerfCounterType::NAME, select}` entry to the
right bank table in the arch header, and keep the `NUM_*_COUNTERS` constant in step. The Python side
(`tt_metal/tt-llk/tools/python/tt_llk_perf`) parses names and tables from these headers, so nothing
else has to change.

Quasar has one debug block per NEO and two ways to reach it: a TRISC uses the local window
(`LOCAL_REGS_WINDOW`, the default), the DM cores use the NoC window (`neo_window(n)`); every register
primitive takes the window as its last argument. There is no L1 counter bank (`table_for(Bank::L1)` is
empty, `L1_MUX_POSITIONS` is 0); instead one clear-on-read l1_client CSR counts one `subport*8 + event`
selection (`l1_client_regs`, `l1_client_start/stop/read`, validity in `l1_client_selection_is_valid`).
Metal defines `LLK_PERF_TABLES_IN_TEXT` before including the headers so the DM firmware keeps the
tables out of its 2 KB data region.
