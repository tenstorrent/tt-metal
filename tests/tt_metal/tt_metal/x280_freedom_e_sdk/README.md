<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

SPDX-License-Identifier: Apache-2.0
-->

# freedom-e-sdk on Quasar DM cores (SiFive X280)

Can [sifive/freedom-e-sdk](https://github.com/sifive/freedom-e-sdk) be integrated
with tt-metal's existing X280 code?

**Yes.** `./build.sh` produces a single ELF that contains SiFive's freedom-metal
(entry code, libc glue, hart and cache HAL) *and* tt-metal's unmodified
`risc_common.h` X280 cache primitives, built for `-mcpu=tt-qsr64-rocc` and linked
at the address tt-metal loads Quasar DM kernels to. 14 verification checks pass.

It is **not executed**: there is no Quasar silicon and no Quasar simulator
available, so this demonstrates build, link and ISA compatibility, not runtime
behaviour. See [Not executed](#not-executed) below.

---

## What "the existing X280 code" is

There is no directory called `x280` in tt-metal. The X280 lineage lives in the
Quasar (`tt-2xx`) data-movement core support:

| Where | What |
| --- | --- |
| [risc_common.h:170-330](../../../../tt_metal/hw/inc/internal/tt-2xx/risc_common.h#L170-L330) | The X280 code proper: `flush_l1_dcache`, `invalidate_l1_dcache`, `invalidate_l1_icache`, `flush_l2_cache_*`, `invalidate_l2_cache*`. Its own comments cite the *SiFive X280 Core Manual* §3.4.2/6.1.1/6.1.2 for `CFLUSH.D.L1` / `CDISCARD.D.L1`, name rocket-chip as the basis for the Quasar DM cores, and point at **freedom-metal's `src/cache.c` as the reference implementation**. |
| [quasar_dm_cache_management.md](../data_movement/quasar_cache/quasar_dm_cache_management.md) | The cache hierarchy those primitives drive: per-core 4 KB 2-way L1 D$/I$, a shared 128 KB 4-way L2, backed by Tensix L1. |
| [qa_hal.cpp:348](../../../../tt_metal/llrt/hal/tt-2xx/quasar/qa_hal.cpp#L348) | The DM cores are compiled `-mcpu=tt-qsr64-rocc`. |
| [dev_mem_map.h](../../../../tt_metal/hw/inc/internal/tt-2xx/quasar/dev_mem_map.h) | `MEM_KERNEL_BASE` = 0x400000, `MEM_DM_KERNEL_SIZE` = 48 KB — where DM kernels are linked. |

So the integration question is really: *does freedom-e-sdk work against the
Quasar DM core's toolchain, ISA and memory map?*

## What the demo proves

Each of these is checked mechanically by `build.sh` stage 7.

**1. The toolchain accepts freedom-e-sdk's build model.** freedom-metal's own
autoconf build, driven by freedom-e-sdk's `scripts/libmetal.mk`, produces
`libmetal.a` (1.7 MB) and `libmetal-gloss.a` (300 KB) for `-mcpu=tt-qsr64-rocc`.
No source changes to either upstream repo.

**2. Stock upstream code builds unmodified.** freedom-e-sdk's own
`software/hello` — untouched program, untouched build system — links for the
`tt-quasar-dm` BSP. sfpi ships a full newlib for the `qsr64-lp64` multilib
(`libc.a`, `libm.a`, `libgloss.a`), so `printf` works.

**3. tt-metal's X280 header compiles inside a freedom-e-sdk program.**
`src/x280_cache_tt.cc` includes `internal/tt-2xx/risc_common.h` unmodified, using
the same include set and defines `qa_hal.cpp` hands the JIT compiler for a DM
kernel.

**4. The two codebases emit the same instruction.** This is the sharpest result.
Both implementations of X280 L1 D$ flush end up in the same ELF:

```
# tt-metal, risc_common.h:  __asm__("tt.cache.cflush.d.l1 %0")
00000000004013e8 <tt_x280_flush_l1_dcache>:
  4013ec:  fc050073   tt.cache.cflush.d.l1  a0      # flush one line
  4013f8:  fc000073   tt.cache.cflush.d.l1  zero    # rs1=x0: flush all of L1 D$

# freedom-metal, src/cache.c:  .insn i 0x73, 0, x0, %2, -0x40
00000000004030f8 <metal_dcache_l1_flush>:
  403154:  fc068073   tt.cache.cflush.d.l1  a3
```

freedom-metal hand-encodes the instruction with `.insn`; tt-metal spells it as a
mnemonic that sfpi's assembler knows. Same opcode, same operand form — sfpi's
objdump decodes freedom-metal's `.insn` back to the same mnemonic. That is the
X280 heritage in `risc_common.h`'s comments turning out to be literally true at
the encoding level.

**5. It lands where tt-metal puts DM kernels.** Entry point 0x00400000
(`MEM_KERNEL_BASE`), image 36244 B inside the 48 KB `MEM_DM_KERNEL_SIZE` window.
`build.sh` reads both numbers out of `dev_mem_map.h` with the preprocessor rather
than hardcoding them.

**6. The ISA is the real one.** `Tag_RISCV_arch` on the linked ELF:

```
rv64i2p0_m2p0_a2p0_b1p0_zihpm2p0_zmmul1p0_zaamo1p0_zalrsc1p0_zba1p0_zbb1p0_zbc1p0_zbs1p0_xttcache1p0_xttroccqsr1p0
```

`xttcache` (the `CFLUSH.D.L1` family) and `xttroccqsr` (Quasar's RoCC interface)
are present. `f`, `d`, `c` and `v` are absent — everything in the image,
including newlib's `printf`, came out soft-float and uncompressed.

## What the hello world actually does

`src/hello_x280.c` prints its own build provenance, then performs a cache handoff
between the two APIs on one buffer:

```c
metal_dcache_l1_flush(hartid, addr);            // SiFive freedom-metal
tt_x280_flush_l1_dcache(addr);                  // tt-metal risc_common.h
tt_x280_flush_l2_cache_range(addr, sizeof(handoff));   // L2 -> Tensix L1
tt_x280_invalidate_l1_icache();                 // FENCE.I
```

Layer by layer, nothing reimplemented:

| Layer | Comes from |
| --- | --- |
| entry / crt0 / stdio plumbing | freedom-metal (freedom-e-sdk) |
| libc | newlib, from tt-metal's sfpi toolchain |
| hart identity, cache availability | freedom-metal `metal/cpu.h`, `metal/cache.h` |
| cache maintenance | tt-metal `risc_common.h` |
| console | `src/quasar_tty.c` (this demo) |
| link addresses | tt-metal `dev_mem_map.h` |

`src/quasar_tty.c` is where the integration earns its keep. freedom-metal routes
all stdio through one hook, `metal_tty_putc()`, implemented against the BSP's
`stdout-path` UART — and left *weak* when a BSP has no UART. A Quasar DM core has
no UART; the way data leaves a DM core is by landing in Tensix L1 and being made
visible to the NoC, which is what tt-metal's DPRINT does. So `quasar_tty.c`
replaces the UART shim with a TL1 ring buffer and flushes it with tt-metal's X280
cache primitives. **freedom-metal `printf` on top of Tenstorrent cache
management** is the integration, concretely.

## What had to be adapted

Three things, none of them a source change to either upstream repo:

1. **Toolchain triple.** freedom-metal configures with autoconf, and `config.sub`
   rejects `riscv-tt-elf` ("machine `riscv-tt` not recognized"). `build.sh`
   symlinks sfpi's binaries as `riscv64-unknown-elf-*`, the triple freedom-e-sdk
   already expects. Target and multilibs are baked into the compiler, so the name
   it is invoked under changes nothing about the output. The alternative — a
   one-line `config.sub` patch — was avoided to keep upstream pristine.

2. **Multilib selection.** freedom-e-sdk derives `-march`/`-mabi` from
   `RISCV_ARCH`/`RISCV_ABI` in the BSP's `settings.mk`. `-march` alone selects
   sfpi's *default* multilib, whose libc is built for a different ISA; only
   `-mcpu=tt-qsr64-rocc` selects `qsr64-lp64`. `bsp/settings.mk` appends the
   `-mcpu` to `RISCV_CFLAGS` etc., which standalone.mk already supports.

3. **Link address.** The nearest upstream rv64 BSP links at 0x80000000, which is
   exactly out of range for `medlow`-compiled newlib (`R_RISCV_HI20` reaches
   ±2 GiB around 0) — the first attempt failed with a wall of "relocation
   truncated to fit". Quasar's low TL1 addresses have no such problem, so
   retargeting the linker script at `MEM_KERNEL_BASE` fixed the ISA problem and
   the memory-map problem at once.

## Limits and honest gaps

**The BSP is derived, not generated.** The right way to produce a freedom-e-sdk
BSP is to run `freedom-devicetree-tools` over a `design.dts`. Those tools need
`dtc`, and neither is installed here, so `build.sh` starts from
`bsp/qemu-sifive-u54` and retargets the ISA and the memory map.
`bsp/quasar-dm.dts` is a hand-written devicetree recording what a
generated-from-scratch BSP should describe (8 DM cores, `compatible =
"sifive,x280"`, TL1, cache geometry, no UART), so that step is a matter of
running the tools rather than working out the hardware description. **What is
left over from the u54 BSP is its peripheral description** — a CLINT at
0x2000000, a PLIC at 0xc000000, a UART at 0x10013000. None of those exist on a
Quasar DM core. Nothing in this demo touches them, but `metal_cpu_get_timer()`
and anything interrupt-related would read garbage on real hardware until the BSP
is regenerated.

**A Quasar DM core is not a stock X280.** A stock X280 is RV64GCV. This target is
`rv64imab_...` plus TT custom extensions: no hardware float, no compressed
instructions, no vector unit. Any freedom-metal or freedom-e-sdk code assuming
F/D or V does not apply — that rules out most of freedom-e-sdk's floating-point
benchmarks (`dhrystone`, `coremark` will build; anything V-based will not).

**Peripheral drivers do not transfer.** freedom-metal's `gpio.c`, `spi.c`,
`i2c.c`, `pwm.c`, `rtc.c`, `uart.c` and friends drive SiFive Freedom E/S
peripherals that Quasar does not have. What transfers is the core-level HAL:
`cpu.c`, `cache.c`, `atomic.c`, `lock.c`, `csr`, `pmp.c`, `time.c`, the entry
code and the gloss syscall layer.

**L2 needs a driver either way.** freedom-metal's `metal_l2cache_*` dispatch to a
`sifive,ccache0`-style driver. Quasar's L2 controller is its own, driven through
`L2_FLUSH_ADDR` / `L2_INVALIDATE_ADDR` / `L2_FULL_INVALIDATE_ADDR` in
`quasar/overlay/overlay_addresses.h`. tt-metal's `risc_common.h` already
implements those operations, which is why the demo calls the tt-metal side for L2
and can call either side for L1 D$.

**Fabric-level integration is a separate, larger job.** See [wip/](wip/).

## Not executed

The demo builds and inspects; it does not run. There is no Quasar silicon (this
host has Blackhole), no Quasar simulator, and no `spike` or `qemu-system-riscv64`
installed. The `xttcache` and `xttroccqsr` instructions in the image would not be
understood by a stock RISC-V simulator anyway.

Two routes to actually seeing "Hello, World" on a screen, in increasing order of
fidelity:

1. **Retarget the same sources at a generic rv64 core** (`-march=rv64imac`, the
   `qemu-sifive-u54` BSP as-is) and run under `qemu-system-riscv64`. The
   freedom-metal side and `hello_x280.c`'s structure survive; the X280 cache
   calls have to be stubbed, because that is precisely what a generic core lacks.
   This proves the program logic, not the integration.
2. **Run the existing ELF on a Quasar RTL/emulation target**, loading it into the
   DM kernel window and reading the TL1 console buffer (magic `0x28028028`) back
   over the NoC. This needs no changes to what `build.sh` already produces.

## Building

```bash
cd tests/tt_metal/tt_metal/x280_freedom_e_sdk
./build.sh              # clones freedom-e-sdk, builds, verifies
./build.sh --clean      # remove build/
```

Requires network access on first run (clones freedom-e-sdk plus its
freedom-metal and BSP-generator submodules into `build/`), `python3`, and sfpi.
sfpi is picked up from `$TT_METAL_HOME/runtime/sfpi/compiler`; if this checkout
has not been built, point at another one:

```bash
SFPI=/path/to/tt-metal/runtime/sfpi/compiler ./build.sh
```

Artifacts land in `build/out/`: `hello_x280.elf`, `.lst`, `.map`, and the
function-scoped disassembly the cache-instruction checks compare.

## Layout

```
build.sh                 8-stage build + 14 verification checks
bsp/settings.mk          freedom-e-sdk BSP settings for -mcpu=tt-qsr64-rocc
bsp/quasar-dm.dts        reference devicetree for a properly generated BSP
src/hello_x280.c         the hello world
src/quasar_tty.c         freedom-metal stdio -> Tensix L1, via X280 cache ops
src/x280_cache_tt.cc     C wrappers over tt-metal's risc_common.h primitives
wip/                     earlier, unfinished attempt at fabric-level integration
```
