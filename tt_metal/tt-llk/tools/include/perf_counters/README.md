# Tensix performance counters

Header-only description of the hardware performance counters, namespace `llk::perf`.

- `types.h`: `PerfCounterType` (counter names, ordinal is the wire format), `Bank`, `Entry`.
- `blackhole.h`, `wormhole.h`: per-bank `{name, select}` tables and the L1 mux width.
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
else has to change. Quasar has no tables here yet; both arch-selecting headers `#error` on `ARCH_QUASAR`.
