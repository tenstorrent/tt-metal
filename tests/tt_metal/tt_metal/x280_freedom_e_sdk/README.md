<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

SPDX-License-Identifier: Apache-2.0
-->

# freedom-e-sdk on X280

Yes — Blackhole's L2CPU X280 is a stock SiFive core, so freedom-e-sdk integrates with two weak hooks and one link flag (no patches).

| | [`l2cpu/`](l2cpu/) — real X280 | [`quasar_dm/`](quasar_dm/) — X280-derived |
| --- | --- | --- |
| Hardware | SiFive X280 in Blackhole L2CPU | Quasar DM cores |
| Code | tt-llm-engine `x280/` | tt-metal `tt_metal/hw/inc/internal/tt-2xx/` |
| Toolchain | stock `riscv64-unknown-elf-gcc` | sfpi `riscv-tt-elf-gcc` |
| ISA | `rv64gc` / `lp64d` (silicon: rv64gcv) | `rv64imab_…_xttcache_xttroccqsr` / `lp64` |
| Link base | LIM `0x08001000` | TL1 `0x00400000` |
| Silicon | Blackhole | none yet |
| Integration | 2 weak hooks + `__stack_size` | shim + multilib + relink |

`l2cpu/` is the real answer. `quasar_dm/` is kept for the future Quasar target.

## Integration gaps (`l2cpu/`)

freedom-metal's `_enter` already covers gp/mtvec/chicken-bit/stacks/.bss/.data. Fill the rest in [`l2cpu/src/x280_bringup.c`](l2cpu/src/x280_bringup.c):

1. `__metal_before_start` — enable `mstatus.FS` and `mstatus.VS` (both Off at reset; FP traps under `-mabi=lp64d` until FS is on).
2. `__metal_after_main` — `CEASE` instead of spinning (`metal_shutdown()` needs a `sifive,test0` block L2CPU lacks).
3. `-Wl,--defsym=__stack_size=0x8000` — BSP default 1 KiB is too small for newlib `printf` (match tt-llm-engine `x280.ld`).

## Build and run (emulation)

```bash
cd tests/tt_metal/tt_metal/x280_freedom_e_sdk/l2cpu
./build.sh              # clones freedom-e-sdk, builds hardware flavor, verifies
./qemu/run_qemu.sh      # builds X280_QEMU flavor and runs under sifive_u
./build.sh --clean
```

Needs network on first run and `riscv64-unknown-elf`. Prefers tt-llm-engine's vendored toolchain:

```bash
X280_TOOLCHAIN=/path/to/tt-llm-engine/x280/toolchain ./build.sh
# or
TT_LLM_ENGINE=/path/to/tt-llm-engine ./build.sh
```

`run_qemu.sh` uses system qemu or unpacks Ubuntu packages into `build/qemu/` (no root). qemu's L2 LIM is at `0x08000000` (same as Blackhole), so the binary runs at its real link address. `-DX280_QEMU` adds a UART mirror and drops `CEASE`.

## Hardware (Galaxy caution)

`build.sh` does not touch hardware. From tt-llm-engine `x280/README.md` §2.1:

> Validated for **single-card Blackhole**. On Galaxy / multi-chip, the pyluwen boot path **WILL hang the chip and need a host PSU power cycle**.

Use the exalens backend. Read-only probe, then least-invasive runner (no WayEnable, no AICLK ramp, no PLL — LIM only):

```bash
./build.sh
<python_env>/bin/python3 hw/probe_l2cpu.py
TT_DEVICE=0 X280_L2CPU=0 <python_env>/bin/python3 hw/run_on_hardware.py
```

Needs `ttexalens`; do not open the same device with UMD/tt-metal or pyluwen in one process.

Destructive reference path (tt-llm-engine): `X280_BACKEND=exalens` → `boot_idle_x280.py` / `loader.py` at `0x08001000`, poll sentinel `0xDEADBEEFCAFEBABE` at `0x08100000`, read console at `0x08101000`.

## Known gaps

- **BSP is derived** from `qemu-sifive-u54` (needs `dtc` for a real generate). Leftover u54 PLIC/UART are fiction; regenerate before interrupt work. Prefer `qemu-sifive-u54mc` for multi-hart `metal_cpu_get()`.
- **Boot hart** is `PROVIDE(__metal_boot_hart = 0)`; qemu overrides to 1 (E51 has no FP). Blackhole harts are identical — keep 0 unless host releases another.
- **Vector enabled, unused** — newlib is non-multilib `rv64gc`; use `-march=rv64gcv` per TU for RVV.
- **SiFive peripheral drivers** (gpio/spi/uart/…) do not apply; core HAL does.
- **tt-llm-engine drivers** (`dma_engine.h`, `noc.h`, `socket_api.h`) not wired yet — next step, no freedom-metal changes needed.

## Layout

```
README.md              this file
l2cpu/                 Blackhole L2CPU SiFive X280
  build.sh             build + verify (hardware flavor)
  qemu/run_qemu.sh     run under qemu -machine sifive_u
  qemu/trampoline.S    mask-ROM → LIM jump (stands in for host release)
  bsp/settings.mk      rv64gc/lp64d, sifive-7-series
  src/hello_x280_lim.c hello world
  src/x280_lim_console.c  metal_tty_putc → LIM
  src/x280_bringup.c   FS/VS enable + CEASE hooks
  hw/                  read-only probe + least-invasive runner
quasar_dm/             Quasar DM (X280-derived); see quasar_dm/README.md
wip/                   unfinished fabric-level attempt
```
