<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

SPDX-License-Identifier: Apache-2.0
-->

# freedom-e-sdk on Quasar DM (X280-derived)

> Prefer [`../l2cpu/`](../l2cpu/) for the **real** Blackhole X280 (stock toolchain, silicon exists). This directory targets Quasar DM cores — X280-derived, custom sfpi ISA, no silicon yet.

`./build.sh` links freedom-metal with tt-metal's unmodified `risc_common.h` X280 cache primitives for `-mcpu=tt-qsr64-rocc` at `MEM_KERNEL_BASE`. Build/link/ISA checks pass; **not executed** (no Quasar silicon/simulator).

## tt-metal X280 lineage

| Where | What |
| --- | --- |
| [risc_common.h](../../../../../tt_metal/hw/inc/internal/tt-2xx/risc_common.h) | X280 cache ops (`CFLUSH.D.L1` / `CDISCARD.D.L1`); cites freedom-metal `cache.c` |
| [quasar_dm_cache_management.md](../../data_movement/quasar_cache/quasar_dm_cache_management.md) | L1/L2 hierarchy |
| [qa_hal.cpp](../../../../../tt_metal/llrt/hal/tt-2xx/quasar/qa_hal.cpp) | `-mcpu=tt-qsr64-rocc` |
| [dev_mem_map.h](../../../../../tt_metal/hw/inc/internal/tt-2xx/quasar/dev_mem_map.h) | `MEM_KERNEL_BASE` = 0x400000, 48 KB window |

## What is verified

`build.sh` stage 7 checks:

1. freedom-metal builds for `-mcpu=tt-qsr64-rocc` (no upstream source patches).
2. Stock `software/hello` links for the derived BSP.
3. `risc_common.h` compiles inside the freedom-e-sdk program.
4. Both L1 D$ flush paths emit the same `tt.cache.cflush.d.l1` encoding.
5. Entry at `MEM_KERNEL_BASE`, image fits `MEM_DM_KERNEL_SIZE`.
6. ELF arch tag has `xttcache` / `xttroccqsr`; no f/d/c/v.

Hello world handoff: `metal_dcache_l1_flush` then `tt_x280_flush_*` on one buffer. Console is TL1 via `quasar_tty.c` (no UART).

## Adaptations (no upstream patches)

1. **Toolchain triple** — symlink sfpi as `riscv64-unknown-elf-*` (`config.sub` rejects `riscv-tt-elf`).
2. **Multilib** — append `-mcpu=tt-qsr64-rocc` in `bsp/settings.mk` (`-march` alone picks the wrong libc).
3. **Link address** — retarget linker script to `MEM_KERNEL_BASE` (upstream 0x80000000 breaks medlow).

## Gaps

- BSP derived from `qemu-sifive-u54`; leftover CLINT/PLIC/UART are fiction until regenerated from `bsp/quasar-dm.dts`.
- Not a stock X280 (no F/D/C/V) — FP/V freedom-e-sdk code does not apply.
- SiFive peripheral drivers do not transfer; use tt-metal for L2 (`L2_FLUSH_ADDR` etc.).
- Fabric-level work: see [wip/](../wip/).

## Build

```bash
cd tests/tt_metal/tt_metal/x280_freedom_e_sdk/quasar_dm
./build.sh
./build.sh --clean
```

Needs network (first run), `python3`, sfpi (`$TT_METAL_HOME/runtime/sfpi/compiler` or `SFPI=...`).

## Layout

```
build.sh                 build + verify
bsp/settings.mk          -mcpu=tt-qsr64-rocc BSP settings
bsp/quasar-dm.dts        reference DTS for a real generate
src/hello_x280.c         hello world
src/quasar_tty.c         metal_tty_putc → Tensix L1 + cache flush
src/x280_cache_tt.cc     wrappers over risc_common.h
```
