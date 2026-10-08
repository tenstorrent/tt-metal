# Raw LREG lifetime experiment — 2026-10-07

## Question and controlled intervention

Do independent read/write identity pairs preserve a raw L0 value across a typed
SFPI temporary? Does explicitly threading a C++ value change the result?

`sources/sfpu_raw_lreg_device.cpp` produces 1.0 in raw L0, loads/stores a typed
2.0 temporary, then stores raw L0 to the output. The independent oracle is 1.0
in every output element; the input tile must remain 2.0. Both TT (MMIO) and TTI
instruction forms run. Only the annotation scheme changes:

0. No annotations (diagnostic control).
1. `writelreg(readlreg(0), 0)` after producer and before consumer.
2. Raw effect metadata after producer and consumer.
3. `saved = readlreg(0)` after producer; `writelreg(saved, 0)` before consumer.

Scheme 3 is an explicit use-def dependency. It is **not** an implementation of
Nathan's proposed `sfpvalue` builtin, and does not test that proposed API.

## Measured results

Physical Blackhole on `ttuser@tt-quietbox-0.local`, runtime checkout
`b8c16a2977d1674c49980f1802097e5205ac5187` plus these test files;
standalone PR #22 compiler `bc27e710ef47be5ac403476230dd8b2e8db8255a`.

| Scheme | O2 TT wrong elements | O2 TTI wrong elements |
|---|---:|---:|
| No annotation | 1024/1024 | 1024/1024 |
| Independent pairs | 1024/1024 | 1024/1024 |
| Effect metadata | 0/1024 | 0/1024 |
| Threaded value | 0/1024 | 0/1024 |

Additionally, scheme 3 passed with `-fdisable-rtl-rvtt_lreg_livein` at O2 and O3,
each with default scheduling and with `-fschedule-insns -fschedule-insns2`:
eight hardware cases, zero wrong elements. Thus this explicit region does not
need the new live-in pass. The standalone O2 scheduled TTI assembly with the
pass disabled has READ L0, temporary SFPLOAD/SFPSTORE L1, WRITE L0, and no extra
move. This is an instruction observation, **not** a latency measurement.

O0 hardware validation did not run: the shared BRISC harness fails compilation
in `t6_semaphore_init`, with an impossible immediate constraint for TTI_SEMINIT.
Do not count that as a pass or as a device failure of scheme 3.

## Re-run

Use the matching SFPI compiler/headers with `sfprawlreg_effect` support. From
`tt_metal/tt-llk/tests/python_tests`, on a Blackhole device with the LLK test
environment installed:

```sh
CHIP_ARCH=blackhole TT_LLK_EXTRA_COMPILER_OPTIONS='-O2' \
  ../.venv/bin/python -m pytest test_raw_lreg_device.py -s -q

for opt in O2 O3; do
  for scheduling in '' '-fschedule-insns -fschedule-insns2'; do
    CHIP_ARCH=blackhole \
    TT_LLK_EXTRA_COMPILER_OPTIONS="-$opt $scheduling -fdisable-rtl-rvtt_lreg_livein" \
      ../.venv/bin/python -m pytest -s -q \
      'test_raw_lreg_device.py::test_raw_lreg_device[0-3]' \
      'test_raw_lreg_device.py::test_raw_lreg_device[1-3]' || exit
  done
done
```

For a separately built compiler, prepend `-B/path/to/compiler/backend/` and
`-I/path/to/matching/sfpi/include` to the options, as in this experiment.
Do not use `--compile-producer`: that does not execute the hardware assertions.
XFAIL for schemes 0/1 means wrong output was observed, not correctness success.
Schemes 2/3 are required to produce correct output.

The companion `raw_lreg_full_annotation.cpp` is an assembly-only comparator;
`selftest_raw_lreg_metadata.sh --target-cxx /path/to/riscv-tt-elf-g++` exercises
all four schemes. Allocation scans alone are not a correctness oracle because
a compiler may legally preserve a value with moves.

## Decision and remaining scope

Independent identity pairs are insufficient in this experiment. Explicit
threading is a working alternative for a producer/consumer region whose value
can be carried in C++. This does not establish that independent macro wrappers
can communicate that lifetime, nor that the effect pass can be removed globally.

No production wrapper policy is changed on this evidence alone. Before choosing
a general replacement, test partial predicates/inactive lanes, multiple live
registers, pressure/spills, control flow, calls, and representative affected LLKs.
Other chips, arbitrary raw opcodes, formal equivalence, exhaustive input coverage,
and performance are not established here. The proposed unknown-value builtin
remains a design alternative, not an experimentally evaluated implementation.
