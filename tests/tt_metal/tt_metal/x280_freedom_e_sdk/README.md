<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

SPDX-License-Identifier: Apache-2.0
-->

# Is sifive/freedom-e-sdk integratable with X280?

**Yes — and for the real X280 it is close to trivial. The hello world runs:
see [It runs](#it-runs) for the actual output.**

There are two different things at Tenstorrent that the phrase "X280 code" can mean,
and the answer is much stronger for the first one:

| | **[`l2cpu/`](l2cpu/) — the real X280** | [`quasar_dm/`](quasar_dm/) — X280-*derived* |
| --- | --- | --- |
| Hardware | SiFive X280 cluster in Blackhole's L2CPU tiles | Quasar DM cores (rocket-chip/X280 heritage) |
| Code lives in | **tt-llm-engine `x280/`** | tt-metal `tt_metal/hw/inc/internal/tt-2xx/` |
| Toolchain | **stock `riscv64-unknown-elf-gcc`** (vendored) | tt-metal's `sfpi` fork, `riscv-tt-elf-gcc` |
| ISA | **`rv64gc` / `lp64d`** (real core is rv64gcv, VLEN=512) | `rv64imab_…_xttcache_xttroccqsr` / `lp64` |
| Link base | LIM `0x08001000` | TL1 `0x00400000` |
| Silicon | **Blackhole — available on this host** | none exists yet |
| Work to integrate | **2 weak hooks + 1 link flag, 0 patches** | shim + multilib flag + relink |

`l2cpu/` is the answer to the question as asked. `quasar_dm/` was written first,
before the tt-llm-engine code was located; it is kept because it is a real result
about a real (future) target, not because it is the better answer.

---

## The short version for the real X280

freedom-e-sdk assumes a stock SiFive core built by `riscv64-unknown-elf-gcc`.
A Blackhole L2CPU X280 **is** a stock SiFive core built by
`riscv64-unknown-elf-gcc`. There is essentially nothing to bridge:

- **Same toolchain.** tt-llm-engine vendors
  `riscv64-unknown-elf-gcc 15.2.0` at `x280/toolchain/` (fetched from
  riscv-collab). That is the exact triple freedom-e-sdk defaults to
  (`CROSS_COMPILE ?= riscv64-unknown-elf`). No shim, no `config.sub` patch, no
  multilib juggling — all three of which the Quasar port needed.
- **Same ISA and ABI.** tt-llm-engine's `x280/Makefile` uses
  `-march=rv64gc -mabi=lp64d -mcmodel=medany`. freedom-e-sdk's rv64 BSPs use
  the same string spelled `rv64imafdc` / `lp64d`. The vendored toolchain's
  newlib is built for exactly that, non-multilib.
- **Same memory region — literally.** freedom-e-sdk's stock
  `bsp/sifive-hifive-unmatched` linker scripts already declare

  ```
  lim (airwx) : ORIGIN = 0x8000000, LENGTH = 0x1e0000
  ```

  which is byte for byte the X280's LIM region on Blackhole
  (`[0x08000000, 0x081E0000)`, 1.875 MiB, per `x280/include/x280.h`). Not a
  coincidence: it is the same SiFive core-complex LIM convention.
- **Same boot steps.** freedom-metal's `src/entry.S` and tt-llm-engine's
  hand-written `x280/boot/entry.S` do nearly the same things in the same order,
  because both are bringing up a SiFive core:

  | `x280/boot/entry.S` | freedom-metal `_enter` |
  | --- | --- |
  | 1. init `gp` | yes (`.option norelax; la gp, __global_pointer$`) |
  | 2. set `mtvec` | yes (`early_trap_vector`) |
  | 3. clear Feature Disable CSR `0x7c1` | **yes — `csrwi 0x7C1, 0`**, gated on `__metal_chicken_bit` |
  | 3b. `mstatus.VS` = Initial | **no** → one of the two gaps |
  | *(neither does this)* `mstatus.FS` enable | **no** → the other gap, found by running |
  | 4. per-hart stack pointer | yes (`sp -= hartid * __stack_size`) |
  | 5. zero `.bss` | yes — *and* copies `.data`, which `x280/boot/entry.S` lists as a "Level 3 / future" item |
  | 6. `call main` | yes, via `_start` → `__libc_init_array` → `main` |
  | 7. spin if `main` returns | yes (`__metal_after_main`) |

  freedom-metal's boot path is a **superset** of the existing one except for the
  vector-unit enable.

So `l2cpu/` needed two weak hooks freedom-metal's `entry.S` already calls if you
define them — see [`l2cpu/src/x280_bringup.c`](l2cpu/src/x280_bringup.c) — plus one
link-time setting:

1. `__metal_before_start` → enable `mstatus.FS` (FP unit) and `mstatus.VS`
   (vector unit). Both sit at `Off` after reset, and any F/D or vector
   instruction traps until they are on. Only the FP half is strictly required
   here, and it was found by *running* the thing — see below.
2. `__metal_after_main` → `CEASE` (`0x30500073`, SiFive's halt instruction) instead
   of spinning. freedom-metal's `metal_shutdown()` drives a `sifive,test0` block,
   which an L2CPU tile does not have.
3. `-Wl,--defsym=__stack_size=0x8000` → the BSP defaults to 1 KiB per hart, which
   is not enough for newlib's `printf`. 32 KiB matches tt-llm-engine's own
   `x280/ld/x280.ld`.

**No patches to freedom-e-sdk, freedom-metal, or tt-llm-engine.**

## It runs

`l2cpu/qemu/run_qemu.sh` executes the integrated program under
`qemu-system-riscv64 -machine sifive_u`. Real output:

```
=====================================================================
 hello_x280_lim -- freedom-e-sdk on a Blackhole L2CPU SiFive X280
=====================================================================

[1] core identity (CSRs)
  mhartid                             = 1
  misa                                = 0x800000000014112d
  misa.V (vector)                     = no (not an X280 -- emulated core)
  build flavor                        = X280_QEMU (emulation harness)

[2] freedom-metal (freedom-e-sdk)
  metal_cpu_get_current_hartid()      = 1

[3] LIM layout (tt-llm-engine x280/include/x280.h)
  _enter (image first instruction)    = 0x08001000
  console block                       = 0x08101000 (magic 0x2800c0ffee000280)
  sentinel                            = 0x08100000

[4] hardware floating point (rv64gc / lp64d)
  1.0 / 3.0                           = 0.3333333333

---------------------------------------------------------------------
 Hello, World!
---------------------------------------------------------------------

--- LIM readback via qemu monitor ---
0000000008100000: 0xdeadbeefcafebabe
0000000008101000: 0x2800c0ffee000280 0x0000000000000889
```

The last two lines are the host-visible contract, read back out of emulated LIM
exactly as `x280/host/loader.py` would read it over the NOC: the sentinel
`0xDEADBEEFCAFEBABE` at `0x08100000` and the console block's magic plus a length
of 0x889 bytes at `0x08101000`.

Why `sifive_u` is a fair harness: its L2 LIM is at `0x08000000` — the same address
as Blackhole's X280 LIM — so the firmware runs at its **real link address,
unmodified**. `run_qemu.sh` asserts that from `info mtree` rather than trusting
it. qemu's mask ROM jumps to a fixed `0x80000000`, so a three-instruction
trampoline there jumps into LIM; that is a fair analogue of the real thing, where
the host writes the image into LIM over the NOC and releases the L2CPU from reset.

The emulated flavor differs from the hardware build by exactly one define,
`-DX280_QEMU`: it mirrors `metal_tty_putc` to the `sifive_u` UART so output is
visible on a terminal, and drops `CEASE` (which a generic U54 does not implement).
Same sources, same BSP, same toolchain otherwise.

**What this proves:** freedom-metal's `_enter` boots, sets up `gp`/`mtvec`/per-hart
stacks, zeroes `.bss`, copies `.data`, runs the C runtime and reaches `main()`;
newlib's `printf` works end to end through our `metal_tty_putc` hook; hardware
double-precision arithmetic works under `lp64d`; and the LIM console and sentinel
land at the addresses the existing host tooling reads.

**What it does not prove:** anything X280-specific. `misa.V = no` in the output is
the harness being honest — qemu models a generic U54 with no vector unit, no
`CEASE`, and none of Blackhole's NOC, DMA or DDR front port. Those need silicon.

### Two real bugs that only running found

Both were invisible to the static checks, and **both would have hit hardware.**

1. **`__stack_size` defaults to 1 KiB.** The BSP's linker script says
   `PROVIDE(__stack_size = 0x400)`. newlib's `vfprintf` needs far more; with 1 KiB
   the stack silently runs off the bottom, corrupts a saved return address, and
   the first `printf()` "returns" to address 0. Symptom: a `fault_fetch` at
   `epc=0x0` and total silence. Fixed with
   `-Wl,--defsym=__stack_size=0x8000` — 32 KiB per hart, which is exactly what
   tt-llm-engine's own `x280/ld/x280.ld` allocates (`STACK_SIZE = 0x8000`). That
   the hand-rolled script already picked 32 KiB is a good sign the number is
   right.
2. **`mstatus.FS` is Off at reset.** This build is `-mabi=lp64d`, so doubles live
   in FP registers and even a prologue that spills `fs0` traps until FS is
   enabled. freedom-metal's `_enter` never touches FS, and neither does
   tt-llm-engine's `x280/boot/entry.S` — fine for firmware that never uses FP, a
   hang the moment it does. Now set alongside `mstatus.VS` in
   `__metal_before_start`.

A third, cosmetic: `nano.specs` printf has no `%ll` and no float support, so
`%016llx` printed a literal `lx` and `%f` printed nothing until the format
specifiers were narrowed to `%l` (`long` is 64-bit here anyway) and
`-u _printf_float` was added.

This is the argument for actually running it rather than stopping at 21 static
checks: the static checks all passed on a binary whose first `printf` jumped to
address 0.

## What `l2cpu/build.sh` produces

A 23424-byte raw binary that tt-llm-engine's existing loader can take as-is,
with 21 checks passing:

```
entry 0x08001000 == X280_ACTIVE_FW_LOAD_ADDR
freedom-metal's _enter is the image's first instruction
image ends 0x08006ed8, below the sentinel at 0x08100000
freedom-metal clears SiFive Feature Disable CSR 0x7c1
__metal_before_start writes mstatus (VS enable)
CEASE (0x30500073) present
Tag_RISCV_arch: rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_zicsr2p0_zifencei2p0_…
```

The `_enter`-is-first check matters: the loader writes a raw `.bin` to
`X280_ACTIVE_FW_LOAD_ADDR` and releases reset, so the first byte of the image
*is* the entry point. freedom-metal's linker script puts `.init` (containing
`_enter`) first, so this works with no intervention — but it is a contract worth
asserting rather than assuming.

The program itself ([`l2cpu/src/hello_x280_lim.c`](l2cpu/src/hello_x280_lim.c))
prints core identity CSRs (`mvendorid`, `marchid`, `mimpid`, `misa`), `vlenb`
(→ VLEN, which only reads back if the `mstatus.VS` hook ran), freedom-metal's
hart API, the LIM layout it was linked into, and a real double-precision divide
(this target has hardware F/D, unlike Quasar DM). Output goes to a LIM block at
`0x08101000` — an L2CPU tile has no UART, so
[`x280_lim_console.c`](l2cpu/src/x280_lim_console.c) points freedom-metal's
`metal_tty_putc()` hook at memory the host NOC-reads instead. It finishes by
writing `0xDEADBEEFCAFEBABE` to `0x08100000`, the sentinel
`x280/host/loader.py` already polls, so existing tooling can confirm it ran
without knowing anything about it.

## Building

```bash
cd tests/tt_metal/tt_metal/x280_freedom_e_sdk/l2cpu
./build.sh              # clones freedom-e-sdk, builds the hardware flavor, verifies
./qemu/run_qemu.sh      # builds the X280_QEMU flavor and RUNS it
./build.sh --clean
```

`run_qemu.sh` needs `qemu-system-riscv64`; if there is no system one it unpacks
the Ubuntu packages into `build/qemu/` with `dpkg-deb -x` (no root required).

Needs network on first run and a `riscv64-unknown-elf` toolchain. It prefers
tt-llm-engine's vendored one so the demo and the production firmware share a
compiler; point at it explicitly if autodetection misses:

```bash
X280_TOOLCHAIN=/path/to/tt-llm-engine/x280/toolchain ./build.sh
# or
TT_LLM_ENGINE=/path/to/tt-llm-engine ./build.sh
```

## Running it on hardware — read this first

`build.sh` does not touch hardware. Unlike the Quasar target, this one *can*
actually run: Blackhole silicon is present on this host. But there is a real
hazard, quoted from `tt-llm-engine/x280/README.md` §2.1:

> This toolkit is designed and validated for **single-card Blackhole hosts**
> (e.g. P150). On a Galaxy / multi-chip chassis the pyluwen boot path
> **WILL hang the chip and require a host PSU power cycle to recover** —
> SSH and IPMI both go down.

**This host is a Blackhole Galaxy** (32 chips, `tt-galaxy-*` board type), so the
default pyluwen path is exactly the unsafe one. The documented Galaxy-safe route
is the tt-exalens backend, which does all register access over the NOC debug
path:

```bash
cd /path/to/tt-llm-engine
X280_BACKEND=exalens TT_DEVICE=<n> python3 x280/host/boot_idle_x280.py
```

and then, for this binary specifically, `x280/host/loader.py`:
`detect_l2cpu` → `assert_l2cpu_reset` → `prime_lim_ecc` →
`load_binary_at(chip, x, y, "hello_x280_lim.bin", 0x08001000)` →
`release_l2cpu_reset` → `poll_flag(… 0x08100000, 0xDEADBEEFCAFEBABE)`, then read
the console block at `0x08101000` (check `magic == 0x2800C0FFEE000280`, then read
`len` bytes of `data`).

I have not run any of that. Booting an L2CPU on a shared Galaxy chassis is not
something to do unannounced, and tt-llm-engine's own `CLAUDE.md` adds a hard rule
that pyluwen and UMD/tt-metal must never open the same device in one process —
violating it corrupts ARC firmware and needs a physical power cycle. Say the word
and I will drive the exalens path on a specific device.

## Limits and honest gaps

**The BSP is derived, not generated** (same caveat as the Quasar port). Producing
one properly means running `freedom-devicetree-tools` over a `design.dts`; that
needs `dtc`, which is not installed. `build.sh` starts from
`bsp/qemu-sifive-u54` — chosen because its linker script is single-region, which
is how a LIM-resident firmware runs — and retargets the ISA and the memory
window. **What is left over is the u54 peripheral description**: a CLINT at
`0x2000000`, a PLIC at `0xc000000`, a UART at `0x10013000`. The CLINT address
happens to be right (the X280 memory map has CLINT at `0x02000000`), but the PLIC
and UART are fiction on an L2CPU tile. Nothing in this demo touches them;
`metal_cpu_get_timer()` reads the CLINT and so is plausibly correct, while
anything interrupt-driven needs the BSP regenerated first.

**The BSP describes one hart, so `metal_cpu_get()` returns NULL.** Visible in the
run above: `metal_cpu_get_num_harts() = 1` and `metal_cpu_get() = NULL`, because
`qemu-sifive-u54` is the *single*-core variant and only hart 0 is in its cpu
table. Everything that does not need the cpu table works; `metal_cpu_get_timer()`
and the CLINT API need a BSP that describes the harts. Basing the BSP on
`qemu-sifive-u54mc` (multicore) would be the quick improvement, and a properly
generated BSP would describe all four X280 harts per tile.

**Which hart runs `main` is a link-time constant.** `PROVIDE(__metal_boot_hart = 0)`
in the BSP linker script; freedom-metal's `gloss/crt0.S` parks every other hart in
`secondary_main()`. The emulation harness overrides it to 1 with `--defsym`
because qemu's `sifive_u` hart 0 is an E51 with no FP. On Blackhole all four X280
harts in a tile are identical, so this stays 0 unless the host releases a
different one.

**Single hart.** `main()` runs on whichever hart is released from reset.
freedom-metal's `_enter` gives every hart its own stack and only hart 0 zeroes
`.bss`, so the multi-hart structure is there, but the demo does not exercise the
CLINT `msip` IPI protocol that `x280/boot/entry.S`'s `_trap_ipi` implements.
freedom-metal has a `riscv_clint0` driver and `metal_cpu_software_set_ipi()`, so
this is a port, not a redesign.

**Vector unit enabled but unused.** `mstatus.VS` is set and `vlenb` is read back,
but no vector instructions are issued: the vendored toolchain's newlib is
non-multilib `rv64gc`, so `-march=rv64gcv` would apply per translation unit only.
tt-llm-engine's RVV paths do exactly that.

**Peripheral drivers do not transfer.** freedom-metal's `gpio.c`, `spi.c`,
`i2c.c`, `pwm.c`, `rtc.c`, `uart.c` drive SiFive Freedom E/S peripherals Blackhole
does not have. What transfers is the core-level HAL — `cpu.c`, `cache.c`,
`atomic.c`, `lock.c`, `pmp.c`, `time.c`, CSR access, entry code, gloss syscalls —
plus the build system and linker-script machinery.

**Nothing here uses tt-llm-engine's own drivers.** Its `dma_engine.h` (Synopsys
DW_ahb_dmac), `noc.h` (2 MiB / 128 GiB NOC TLBs) and `socket_api.h` are
freestanding headers that would compile in this program unchanged; wiring them up
is the obvious next step and needs no freedom-metal changes.

## Layout

```
README.md              this assessment
l2cpu/                 Blackhole L2CPU SiFive X280 -- the real X280
  build.sh             6-stage build + 21 verification checks (hardware flavor)
  qemu/run_qemu.sh     ACTUALLY RUNS IT under qemu -machine sifive_u + 8 checks
  qemu/trampoline.S    stands in for "host releases the L2CPU from reset"
  bsp/settings.mk      freedom-e-sdk BSP settings (rv64gc/lp64d, sifive-7-series)
  src/hello_x280_lim.c the hello world
  src/x280_lim_console.c  freedom-metal stdio -> LIM, read back over the NOC
  src/x280_bringup.c   the weak hooks: mstatus.FS/VS enable, CEASE halt
quasar_dm/             Quasar DM cores (X280-derived); see quasar_dm/README.md
wip/                   unfinished fabric-level integration attempt
```
