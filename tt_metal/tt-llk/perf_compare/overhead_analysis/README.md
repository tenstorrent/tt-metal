# Performance Counter Overhead Analysis

Reference artifacts for the `mvlahovic/perf_counters_overhead` tracking issue.

## Variant under analysis

A single eltwise_binary_fpu test variant was chosen because it shows the largest
absolute total |Δ WC-NC| across all four `PerfRunType`s in the current design:

```
formats:   Bfp8_b -> Bfp8_b
mathop:    Elwmul
tile_cnt:  16
fidelity:  HiFi2
dest_acc:  Yes
```

Sum of |Δ WC-NC| across run_types for the current STRICT design = **257 cycles/tile**.

| run_type        | Δ WC-NC (cyc/tile) |
|-----------------|-------------------:|
| L1_TO_L1        |               -7   |
| UNPACK_ISOLATE  |               -7   |
| MATH_ISOLATE    |             -242   |
| PACK_ISOLATE    |               -1   |

MATH_ISOLATE dominates the overhead and is the focus of the open question.

## Directory layout

```
overhead_analysis/
├── README.md                    — this file
├── nc/                          — NC (no-counters) build artifacts
│   ├── L1_TO_L1/{unpack,math,pack}.asm + .insns.txt + build.h
│   ├── UNPACK_ISOLATE/...
│   ├── MATH_ISOLATE/...
│   ├── PACK_ISOLATE/...
│   ├── brisc.asm + .insns.txt
├── wc/                          — WC (--enable-perf-counters) build artifacts
│   └── (same structure as nc/)
└── diff/                        — unified diffs WC vs NC
    ├── L1_TO_L1/{unpack,math,pack}_{asm,insns}.diff + build_h.diff
    ├── UNPACK_ISOLATE/...
    ├── MATH_ISOLATE/...
    ├── PACK_ISOLATE/...
    ├── brisc_asm.diff
    └── brisc_insns.diff
```

## File flavors

* `*.asm`         — source-interleaved disassembly (`riscv-tt-elf-objdump -dC --source --no-show-raw-insn`)
* `*.insns.txt`   — pure instruction stream, addresses stripped, comments removed
                    (for clean instruction-level diff)
* `build.h`       — auto-generated per-variant build configuration
* `*_asm.diff`    — unified diff of the .asm files (NC vs WC)
* `*_insns.diff`  — unified diff of the .insns.txt files (cleaner — only shows
                    real opcode changes, not address/comment noise)
* `build_h.diff`  — unified diff of the build.h files (per-variant config)

## Reproducing locally

From `tt_metal/tt-llk/tests/python_tests/`:

```bash
# NC build (no perf counters)
CHIP_ARCH=blackhole pytest -q --compile-producer -n 4 -x \
  "perf_eltwise_binary_fpu.py::test_perf_eltwise_binary_fpu[formats:Bfp8_b->Bfp8_b-mathop:Elwmul-tile_count:16-math_fidelity:HiFi2-dest_acc:Yes]" \
  --speed-of-light

# WC build (with perf counters)
CHIP_ARCH=blackhole pytest -q --compile-producer -n 4 -x --enable-perf-counters \
  "perf_eltwise_binary_fpu.py::test_perf_eltwise_binary_fpu[formats:Bfp8_b->Bfp8_b-mathop:Elwmul-tile_count:16-math_fidelity:HiFi2-dest_acc:Yes]" \
  --speed-of-light
```

ELFs are emitted to `/tmp/tt-llk-build/sources/eltwise_binary_fpu_perf.cpp/<HASH>/elf/`
where `<HASH>` is derived from the variant's `build.h`. BRISC ELF is in
`/tmp/tt-llk-build/shared/elf/brisc.elf`.

Disassemble with:
```bash
OBJDUMP=/opt/tenstorrent/sfpi/compiler/bin/riscv-tt-elf-objdump
$OBJDUMP -dC --source --no-show-raw-insn /tmp/tt-llk-build/sources/eltwise_binary_fpu_perf.cpp/<HASH>/elf/math.elf > math.asm
```

## Run-type → active-thread mapping (STRICT design)

| run_type        | active TRISC (arms + freezes + reads) |
|-----------------|---------------------------------------|
| L1_TO_L1        | pack                                  |
| L1_CONGESTION   | pack                                  |
| PACK_ISOLATE    | pack                                  |
| UNPACK_ISOLATE  | unpack                                |
| MATH_ISOLATE    | math                                  |

Non-active TRISCs block on a `pc_buf`-mapped semaphore barrier at both
`perf_counter_scoped::ctor` (entry barrier) and `~perf_counter_scoped`
(exit barrier), so cross-thread wall-clock stays symmetric and the counter
window is precisely `active.arm → active.freeze`.

## What the diffs show

* `*_insns.diff` confirms WC bodies inside the zone (between zone_ctor's
  wall_clock read and zone_dtor's wall_clock read) are byte-identical to NC,
  apart from register renames and a 1-instruction swap (`lui+lw` for bss
  load in NC ↔ `gp-relative lw` in WC).
* Almost all instruction count growth (~150-290 extra insns per thread) is
  **outside the zone window**: it lives in `perf_counter_scoped::ctor`,
  `perf_counter_scoped::dtor`, the freeze/read loop, and the `get_zone_id`
  scan.
* Despite identical zone-internal code, MATH_ISOLATE measures
  ~-242 cyc/tile in WC. The body itself runs slightly faster in WC
  than NC — the open question is **why** (see the tracking issue for
  speculation and open lines of investigation).
