<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
SPDX-License-Identifier: Apache-2.0
-->

# l2cpu: bring-up, restart and control of the Blackhole L2CPU (x280) tile

A Blackhole chip has four L2CPU tiles, each with four SiFive x280 RISC-V cores (RV64GCV, VLEN 512) that share an L3
cache and a directly attached DRAM channel. This directory brings the harts of one or several tiles (tile 0: CPUs
0-3, NoC0 (8,3), local DRAM channel D5 = tt-metal DRAM bank 5; tiles 1-3 below) from a fresh chip reset to a running
bare-metal firmware, and then stops, reloads and restarts that firmware **without another chip reset**. It
provides:

- a small bare-metal firmware runtime (M-mode, no OS, no libc): boot, per-hart idle loops in `wfi`, heartbeats, a
  host mailbox for debug commands, trap records, a log ring, a work-dispatch service for applications, and park /
  WARM or COLD restart; applications plug in through `fw/app.h` (`examples/heartbeat` is the default image);
- a resident page (`fw/resident.S`) with per-hart RNMI entry stubs and a park loop, written once per chip reset;
- a host Python package (`host/l2cpu`) and a CLI (`scripts/l2cpuctl.py`): `start / stop / restart / status / log`,
  with a clock guard in front of every access to the tile;
- QEMU tests of the firmware logic and a chip smoke test.

The x280 sees its local DRAM directly; a firmware and the host share a **region** of it (layout in
`include/l2cpu_boot.h`). tt-metal kernels and the host reach the same bytes coherently through the L2CPU tile.

## Hardware facts

Measured on one P300 chip (tile 0, x280 at 1750 MHz unless noted). "Expected" = documented in the Blackhole ISA
documentation (`tt-isa-documentation`, `BlackholeA0/L2CPUTile`) or in tt-bh-linux and consistent with what we saw;
"not established" = not tested.

| Fact | Status |
|---|---|
| **Clock holder.** The L2CPU clock runs only while some process holds the chip open (the kernel driver's L2CPU power flag, OR over all open file descriptors). A plain `open("/dev/tenstorrent/N")`, a UMD device handle or a tt-metal device all hold it. With no holder the clock stops (PLL4 CNTL_5 = 0) and the harts freeze with their state intact; they resume when a holder opens the chip. CLINT `mtime` stops too. | measured |
| `ttnn.close_device` does not release the power flag: the clock stays on until the process exits. | measured |
| Every power-state change (e.g. a device open) reprograms the L2CPU PLL to 800 MHz; `ctl.ensure_clock()` re-applies 1750 MHz. | measured |
| **A NoC access to the tile while its clock is off hangs the chip** (every later NoC read, ARC included, times out; a chip reset recovers). `L2cpuHw(guard=True)` reads the PLL through ARC (safe) before every tile access and refuses when the clock is off. | measured |
| **Release once per chip reset.** The harts leave reset on a 0 -> 1 transition of `L2CPU_RESET` (ARC 0x8003_0014) bit 4 + tile. A second release in the same chip epoch does not restart the tile: in 4 attempts harts 0-2 never ran again, hart 3 restarted in 3, and a Memory Port access afterwards hung the chip. This matches the ISA statement ("only once, due to a hardware bug"). Restart is therefore done in software (park + jump) and with RNMIs, never with the reset bit. | measured |
| Out of reset: M-mode, MMU off, `mstatus.MIE = 0`, `FS = VS = 0` (FP/vector instructions trap until set), `mnstatus` = 0x1808 (NMIE = 1, as documented). | measured |
| **Coherent address.** The tile's local DRAM is at x280 PA `0x4000_3000_0000 + offset` (cached Memory Port). From the NoC the same bytes are at `(8,3) 0x4000_3000_0000 + offset` (a 47-bit address: a Tensix kernel must program the command buffer's MID register itself), coherent with the x280 caches in both directions without flushes. The low aliases are **not** coherent: `(8,3) 0x3000_0000 + offset` (System Port, uncached) and the DRAM tile itself (bank 5 at `offset`). DRAM offset = tt-metal bank-5 address (bank base 0). | measured |
| ttnn DRAM buffers are only 64 B aligned: the region base is rounded up to 64 KiB inside the buffer. | measured |
| **Doorbell.** A NoC write of a non-zero u32 to `(8,3) 0x2006_0000` (MSI catcher, 16-entry FIFO) raises PLIC source 6; the firmware wakes hart 0 from `wfi` with it. Writes from the host and from Tensix kernels both work; host write to firmware-visible ~4 us. | measured |
| **PLIC enable bits are not reset** by a chip reset: they come up random in every context (seen in hart 1's M context: 0x96d35eca 0x30c1883a 0x68c40b4c 0x6180a588, the doorbell source 6 included). The boot clears every enable word of the 8 contexts of harts 0-3, then enables only source 6 for hart 0's M context. | measured |
| **Reset vectors** at PA `0x2001_0000 + 8*h` (host: through the external-peripherals high alias, PA + 0xFFFF_F7FE_DFF0_0000); default 0xd000_0000. | measured |
| **RNMI.** Trigger bits at PA `0x2001_0414` (bit h = hart h; level-triggered; the host sets and clears), handler addresses at `0x2001_0418 + 16*h` (+8: exception handler). Both reset to **0** (a stray RNMI before they are written jumps to PA 0): `ctl.start` writes them before the release. One entry per pulse (1000/1000), `mncause` 0x8000000000000002, `mnepc` of a hart in `wfi` = the instruction after the `wfi`; taken in `wfi` with interrupts disabled, in an interrupts-off spin and in an exception loop. CSRs `mnscratch/mnepc/mncause/mnstatus` = 0x350-0x353; `mnret` = `.word 0x70200073` (binutils 2.42 has no mnemonic). | measured |
| Status word PA `0x2001_0400` (u16): bits 8-11 = harts 0-3 in `wfi`. Other bits not established. | measured (wfi bits) |
| The tile's own NIU registers are at PA `0x2005_6000` (NoC0) and `0x2005_7000` (NoC1) (node and endpoint IDs read back correctly), but a request programmed into the NIU initiator is never issued: the x280 cannot use it as a DMA. PA `0x2008_0000` is a Synopsys DW_ahb_dmac (ID 0x44571110, 2 channels, disabled at reset); what its master can address is not established. | measured |
| CSR 0x7c1 (feature disable) must be written 0 by each hart; the L3 is configured to 2 MiB cache (CCACHE0_WAYENABLE = 15) and the L2 prefetchers to the SiFive values. `cflush.d.l1` is an illegal instruction; `CCACHE0_FLUSH64` (PA 0x0201_0200) works. | measured |
| Caches: L1I 32 KiB and L1D 32 KiB per hart, L2 128 KiB per hart, L3 2 MiB shared, 64 B lines; L1D handles one outstanding miss at a time. | expected (ISA) |
| TLB windows (224 x 2 MiB, 32 x 128 GiB) map any NoC tile into the x280 address space; they take raw NoC0 coordinates (DRAM bank 0 = (0,11)). | measured |
| **Tiles 1-3.** Tile t is at NoC0 (8,3) / (8,9) / (8,5) / (8,7) for t = 0..3 (NIU node id and `NOC_ENDPOINT_ID` 0x0901_0000 + t read back); its local DRAM is tt-metal bank 5 / 6 / 7 / 7, at the same PAs as tile 0 (Memory Port 0x4000_3000_0000 + bank address, System Port 0x3000_0000 + bank address). **Tiles 2 and 3 share D7**: their regions must not overlap (take them from two buffers). Reset vectors, scratch, RNMI registers, status word, MSI catcher (0x2006_0000 at the tile's own (x,y), PLIC source 6 of its hart 0), WAYENABLE and L2 prefetchers are per tile at the same addresses. | measured (chip 0, all four tiles) |
| **Several tiles, one release.** `L2CPU_RESET` reads 0x0f after a chip reset (no tile harvested); one read-modify-write setting bits 4..7 inside one PLL 200 -> 1750 MHz dance releases all four tiles (READY within 1 ms of the release); one PLL clocks all four tiles (each runs at 1750 MHz). The once-per-reset rule holds per tile bit. | measured |
| **Reader slots are per tile.** The uncached-read collapse (3+ harts reading at once) is per tile; two harts on each of the four tiles read concurrently at 69-71 us per 303,872 B row (tiles 2 and 3, sharing D7, ~2 % slower): 8 reader slots per chip, 33.9 GB/s. | measured |
| `mcycle` does not advance while a hart sleeps in `wfi`; use `mtime` for wall clock. | measured |
| **PMP.** 8 entries (`pmpaddr0-7`; `pmpaddr8-15` read 0, `pmpaddr16+`, `pmpcfg4+` and odd `pmpcfg` numbers are illegal instructions), granule 4 KiB (G = 10, NAPOT minimum 4 KiB), `pmpaddr` bits 44..0 (NAPOT all-ones = 0x1fff_ffff_ffff covers every PA). Every `pmpcfg` bit is writable. **No Smepmp** (`mseccfg` 0x747 is an illegal instruction). A LOCKED entry also checks M-mode (load fault 5, store fault 7, fetch fault 1, precise, `mtval` = address), ignores later writes to its `pmpcfg` byte and `pmpaddr`, a locked TOR entry locks the `pmpaddr` below it, and the entries stay across a software restart. A chip reset clears the L and A fields (the locked entries are gone) but not `pmpaddr` and R/W/X (random after power-up, the previous epoch's values afterwards). | measured (hart 0; every hart reads its locked set back at boot) |
| other boards (P150), multi-chip opens | not established |

Read speed of one 303,872 B buffer (one Qwen3 logits row), one hart unless noted:

| Path | Time |
|---|---|
| uncached TLB window to another DRAM bank, scalar 8 B loads (~400 cycles each) | 12.16 ms |
| uncached TLB window, RVV `vle64` (one 64 B NoC read per instruction; m8 is not faster) | 0.366 ms |
| cached TLB window + `CCACHE0_FLUSH64` per line (without the flush the data is stale) | 0.73 ms |
| local DRAM, uncached alias (`0x3000_0000 + off`), RVV | 0.059 ms (2 harts in parallel: 0.069 ms each; 3-4 harts: 1.1-1.7 ms each) |
| local DRAM, cached, line not in L3 (DRAM-resident), RVV | 0.284 ms |
| local DRAM, cached, L3 hit, RVV / scalar | 0.011 ms / 0.101 ms |

The first one-hart bare-metal example on this tile is tt-metal PR 52981 (LIM-resident hello world); this component
runs from DRAM, uses the documented boot sequence of tt-bh-linux and adds restart, control and a host package.

## Build

Toolchain (Ubuntu 24.04): `sudo apt-get install gcc-riscv64-unknown-elf binutils-riscv64-unknown-elf
qemu-system-misc` (gcc 13.2, QEMU 8.2). Flags: `-march=rv64gc -mabi=lp64d -mcmodel=medany -ffreestanding -nostdlib
-fno-builtin -fno-tree-loop-distribute-patterns -fno-jump-tables`, linked against `-lgcc` only. Every address is
pc-relative, so one flat image runs at any load address (`make pic-check` links at two addresses and compares).
RVV code (applications): gcc 13.2 miscompiles `vl = __riscv_vsetvl_e32m8(n - i); ... i += vl;` into
`i += n - i`; compute `vl = min(remaining, vlmax)` in scalar code.

    make -C tools/l2cpu/fw                 # bh-irq, bh-poll, QEMU test images, resident blobs, PIC check, md5s
    make -C tools/l2cpu/fw clean all qemu-test
    make -C tools/l2cpu/fw image PLATFORM=bh NOTIFY=irq APP=<dir with app.c, or a name such as sampling>

Flavours: `bh-irq` (default image: hart 0 sleeps in `wfi`, woken by the doorbell or the 1 ms timer), `bh-poll`
(hart 0 polls), `qemu-{irq,poll}-test` (QEMU, with the test driver on a fifth hart). Outputs:
`fw/build/<flavour>/fw.bin` (flat, first byte = entry), `fw.elf`, `fw.lst`; `fw/build/resident/resident-bh.bin`.
An application directory may add an `app.mk` (extra include paths and flags, sources built with their own flags such
as V, its own QEMU test driver); its images get the suffix `-<APP_NAME>` (e.g. `bh-irq-sampling`, see `sampling/`).

## Run

    source tools/l2cpu/scripts/l2cpu_env.sh          # TT_METAL_HOME, PYTHONPATH, PY; chip selection stays in your env
    export L2CPU_LOCK=/tmp/<the lock every device user on this machine takes>
    export L2CPU_RESET_CMD="tt-smi -r"                # e.g. "tt-smi -r 0,1,2,3" when a board's chips reset together
    tools/l2cpu/scripts/l2cpu_run.sh "smoke" $PY tools/l2cpu/scripts/bringup_smoke.py --restarts 200
    # or, with a region you own (x280 PA, 64 KiB aligned, >= 8 MiB, nothing else using it):
    tools/l2cpu/scripts/l2cpu_run.sh "start" $PY tools/l2cpu/scripts/l2cpuctl.py --region 0x... start tools/l2cpu/fw/build/bh-irq/fw.bin

`l2cpu_run.sh` takes the lock, resets the chips (without `TT_VISIBLE_DEVICES`, the reset addresses the board) and
runs the command: the harts of a tile leave reset once per chip reset, so every run that starts the firmware gets
its own reset. Keep the clock holder alive for as long as the firmware should run (the CLI's `hold SECONDS`). The
PMP policy (section "PMP policy") is on by default; `L2CPU_PMP=0` (or the CLI's `--pmp off`) starts without it.

## API (`host/l2cpu`)

    from l2cpu import L2cpuHw, L2cpuCtl, L2cpuMonitor, TtnnClusterBackend, make_backend, layout
    hw = L2cpuHw(TtnnClusterBackend(0), tile=0, guard=True)   # or make_backend("umd") without a tt-metal device
    ctl = L2cpuCtl(hw, region_pa, region_size=8 << 20)   # region_size: what the firmware + application use (PMP)
    ctl.start(image, pmp=None)       # fresh chip reset only; image, resident page, RNMI handlers, PMP flag, release
                                     # pmp: None = L2CPU_PMP (default on), False = off, True / pmp.build(...) = policy
    start_tiles([ctl0, ctl1, ...], image)   # several tiles (one L2CPU_RESET write); then one ctl per tile as above
    ctl.stop()                       # L1 (mailbox PARK), then L2 (RNMI) for harts that did not park -> records
    ctl.restart(image=None, warm=True, slot=None, pmp=None)   # stop, (load), go; ~0.6-1.2 ms; refuses another policy
    ctl.rnmi(mask, mode)             # raw RNMI: COUNT (resume) or PARK
    ctl.status(), ctl.is_alive(), ctl.log_text(), ctl.error(), ctl.hart_state(h), ctl.record(h), ctl.mb(cmd, ...)
    ctl.inject(hart, kind)           # test hook: illegal instruction, or a hart only an RNMI can reach
    ctl.mb(L2CPU_MB_INJECT, hart, L2CPU_INJECT_LOAD|STORE|JUMP, pa)   # PMP fault tests (policy on only)
    ctl.ensure_clock()               # re-apply the target clock after a power-state change
    L2cpuMonitor(ctl).start()        # heartbeat / error watch thread

`bringup.region_base_pa(buffer_address)` turns a ttnn interleaved buffer into a region: its page in the tile's local
bank (5 for tile 0, 6 / 7 / 7 for tiles 1 / 2 / 3), the same PA for every tile. The CLI takes `--tile` (and lists:
`--tile 0,1 --region PA0,PA1 start IMAGE` releases both with one write).

## Region layout and boot record

See `include/l2cpu_boot.h` (Python mirror `host/l2cpu/layout.py`, generated by `tools/gen_boot_py.py`). Generic part:
header 0x0000 (ident with `boot_epoch`, `boot_mode`, `restart_count`, `app_id`, `app_layout_version`; fw_status;
first-error word; heartbeats; hart states with trap records; counters; inject / work_seq / work_done lines), mailbox
0x4000, scratch 0x6000, log ring 0x100000, image slots A 0x200000 and B 0x280000, stacks 0x300000, trap stacks
0x380000, resident page 0x3A0000, heap 0x400000. Application windows: APP_CTRL 0x1000-0x3FFF, APP_LOW
0x8000-0xFFFFF and APP_HIGH from 0x800000 to the end of the region; applications version them with
`app_layout_version` and never edit the generic header. A COLD boot zeroes [0, 2 MiB); the host writes nothing into
the region before `fw_status == READY`.

Boot record (external-peripherals scratch, PA 0x2001_0100, written by the loader): +0x00 image PA, +0x08
"L2CPBOOT", +0x10 region PA, +0x18 PMP flag ("PMP1" = apply the policy, anything else = no PMP), rest 0. The image
takes the region from it, so it can run from either slot. The PMP table (8 x {pmpaddr, pmpcfg}) sits in the
resident page at +0xA00, inside its locked R+X half.

## Restart protocol

1. `start`: before the release the host writes the resident page (code + control: `go_epoch` 0, `boot_epoch` 1,
   COLD, `entry`) and the 8 RNMI handler addresses.
2. Park: L1 = mailbox `PARK` (16): hart 0 replies, IPIs the workers (they jump to the resident soft park), waits up
   to 50 ms for them, parks itself. L2 = the host sets trigger bits (resident `rnmi_mode` = PARK), waits for the
   records, clears the bits. Every park records kind, `mnepc`/`mcause`/`mepc`. A firmware's fatal-error park also
   enters the resident park loop, so a trapped hart is restarted like any other.
3. Park loop: state PARKED, wait for the hart's trigger bit to clear, NMIE := 1, spin until `go_epoch` changes.
4. Go: the host (optionally) writes an image into a slot, sets `fw_status` 0, `entry`, `boot_mode` (WARM: keep the
   application windows, mailbox counters, log, counters, heartbeats, error word; COLD: zero [0, 2 MiB)), `boot_epoch`
   + 1, then `go_epoch` + 1 last. Each hart: `fence.i`, jump to `entry`. The image releases its harts on the new
   epoch (stale values in .bss never match), starts a new work-dispatch generation, completes PLIC source 6 and
   clears its IPI, and reports READY with the new `boot_epoch`. A WARM boot acknowledges a mailbox command still
   pending from before the restart with `L2CPU_MB_ERR_STALE` instead of running it (e.g. the L1 PARK that `stop()`
   sent to a hung hart 0 before falling back to RNMI would otherwise park the new image right after READY).

## PMP policy

Goal: a wild x280 access (bad pointer, overrun, a stray NoC access through an unprogrammed TLB window) traps
precisely (mcause 1 / 5 / 7, `mtval` = address) and parks the hart in the resident error park, where a restart
revives it, instead of hanging the NoC / AHB or overwriting what restart and recovery depend on.

Design: one fixed policy per chip epoch. The host computes it (`host/l2cpu/pmp.py`, `pmp.build(region,
region_size)`), writes the table into the resident page with the page and sets the boot record flag before the
release. Every hart, first thing in `fw_main` (`fw/pmp.c`, before it touches anything outside the region), writes
the 8 entries with the L bit, then reads them back and compares with the table. Locked entries check M-mode and
stay until the next chip reset, so a restart finds them in place: the writes are ignored, the read-back proves the
table still describes them (hart record `pmp` = APPLIED at the first boot, REUSED after a restart, 0x100 | entry and
an error park if they differ). The x280 has no Smepmp (no `mseccfg`, so no RLB to edit locked entries and no MMWP
for a default deny): the locked set with an explicit last deny-all entry is the only way to constrain M-mode.

The x280 has 8 entries, so the ranges are merged (priority = index; every entry locked):

| # | Range | Perm | Covers |
|---|---|---|---|
| 0 | TOR [0, 0x2000_1000) | R W | core complex: CLINT 0x0200_0000, L3 controller 0x0201_0000 (WAYENABLE, FLUSH64), L2 prefetchers 0x0203_0000, PLIC 0x0C00_0000 (internal buses, not the NoC), and the TLB window config page 0x2000_0000 |
| 1 | TOR [0x2000_1000, 0x2008_0000) | R | external peripherals read only: control page (reset vectors, boot record, hart status, RNMI trigger and handler addresses), watchdogs, NIUs, MSI catcher (the doorbell drain is a read); the register survey read every word here without a fault or a hang |
| 2 | NAPOT resident page [+0, +4 KiB) | R X | resident code, control words and the PMP table (the per-hart records in the upper 4 KiB stay writable) |
| 3, 4 | TOR [region, region + size) | R W X | the region (cached Memory Port): header, mailbox, log, image slots, stacks, heaps, application windows |
| 5 | NAPOT around the uncached alias of the region | R | the uncached reads of the applications (sampling: logits rows); smallest aligned block that contains the alias, required to stay in the local DRAM |
| 6 | NAPOT TLB windows 0..31 (uncached) | R W | the runtime's 16 mapping slots (`plat_noc_map`: tensor reads, token writes, mailbox NoC commands) |
| 7 | NAPOT all-ones | none | everything else: other TLB windows, cached TLB windows, the DMA controller 0x2008_0000 and every unlisted PA |

Rule: **the region, its size, the TLB windows and the on/off state are fixed for the chip epoch.** `ctl.restart`
refuses a policy that differs from the one locked at `start` (and an entry outside the region's image slots) before
it touches the tile, with "PMP policy is locked for this chip epoch"; a different region or window set needs a chip
reset and a new `start`. `ctl.start(..., pmp=False)` or `L2CPU_PMP=0` starts without a policy: no CSR is written and
the firmware behaves as before (the flag word is 0, as every earlier loader wrote it).

The mailbox checks PEEK / POKE / COPY / FILL / MEMCMP against the same table (`fw_pmp_allows`, the hardware's
first-match rule) and answers `L2CPU_MB_ERR_DENIED` without an access; `CSR_WRITE` (PMP CSRs only, for the probe)
cannot change a locked entry. Fault-injection kinds `L2CPU_INJECT_LOAD / STORE / JUMP` (address in the inject line)
are refused while the policy is off.

Left open by the policy (8 entries): the region itself (code, header, stacks and records are writable by any hart),
the core-complex devices (a wild store can disturb the timer, interrupts or cache configuration, but these sit on
the core's internal buses and answer), the TLB window config page (a wild store can re-point one of windows 0..31,
and an access through it then goes wherever it points), the NoC targets behind windows 0..31 (`NOC_READ32 /
NOC_WRITE32` take any tile), reads of the external peripherals (a read pops the MSI FIFO) and reads of local DRAM
around the region through the uncached alias. The Tensix link responder is a separate image without the policy.

Cost: the checks are in the core's access path; measured with the policy on and off (same commands, two chip
epochs) in the table below.

| P300 chip 0, medians (`L2CPU_PMP=1` / `=0`, one chip epoch each) | policy on | policy off |
|---|---|---|
| batch 32 replay, uncached zone, read phase (8 rows of 303,872 B per hart, 4 harts) | 591.2 us | 591.2 us |
| batch 32 replay x280 step, 1 / 4 tiles (bench setting) | 1137.4 / 342.3 us | 1137.6 / 342.3 us |
| batch 1 replay x280 total, greedy / T0.7 k50 p0.9 | 32.3 / 90.9 us | 32.2 / 90.9 us |
| Qwen3-8B batch 32, 4 tiles: x280 read / wait for the Tensix push per step | 458.8 / 584.0 us | 459.1 / 584.5 us |
| Qwen3-8B batch 32, 4 tiles: ms/token | 30.311 | 30.312 |
| WARM restart, L1 / L2 (median of 200) | 0.71 / 0.62 ms | 0.69 / 0.61 ms (PR 1 smoke) |

No measurable cost: the check is part of the access path, and Tensix pushes into the region are NoC writes that PMP
does not see.

## Waits and their bounds

Every wait in the firmware, the resident page and the host package, with its bound and what happens at the bound.
Numbers are the defaults in `include/l2cpu_boot.h` and `host/l2cpu`.

| Wait (where) | Bound | At the bound | Host sees it by | Recovery |
|---|---|---|---|---|
| worker waits for hart 0's release flag (`start.S`) | 2^28 polls (~1 s on the chip) | hart enters the resident error park (kind ERROR, mcause 0) | READY timeout; record PARKED / ERROR | L2 park + restart; chip reset |
| worker waits for hart 0's region preparation (`main.c`) | 1 s | `L2CPU_ERR_BOOT_TIMEOUT` (arg = hart), hart parks | error word; READY timeout | restart |
| hart 0 waits for the workers to report IDLE before READY | 100 ms | log line, READY anyway | hart_state status, heartbeat stall (monitor) | restart |
| hart 0 waits for the workers to park on `PARK` | 50 ms | hart 0 parks anyway | `stop()` sees unparked harts | L2 RNMI for those harts (automatic in `stop()`) |
| `fw_wait_workers` (application work items) | parked worker: immediate; else 1 s (`fw_wait_workers_us`) | `L2CPU_ERR_WORKER_DEAD` / `L2CPU_ERR_WORK_TIMEOUT` (arg = hart), returns -1, hart 0 stays alive | error word (monitor) | restart; L2 for a spinning worker |
| hart 0 MSI FIFO drain (boot and every doorbell) | 64 pops | rest on the next wake (level stays high) | n/a | n/a |
| PLIC claim / complete | single access each | n/a | n/a | n/a |
| idle `wfi` (every hart) | timer wake every 1 ms | heartbeat | heartbeat (monitor, 2 s stall) | restart |
| log ring lock | 10^6 tries (10^5 in a trap) | logs without the lock | n/a | n/a |
| mailbox COPY / FILL / MEMCMP | 16 MiB per command | `L2CPU_MB_ERR_ARG` | status | n/a |
| mailbox access to a bad address | precise fault | `L2CPU_MB_ERR_FAULT` + mcause, hart 0 alive | status | n/a |
| mailbox / NoC access to a target that never answers | **not bounded** (a load that never returns has no instruction boundary; RNMI cannot take it). With the PMP policy, PEEK / POKE / COPY / FILL / MEMCMP outside the policy are refused before any access (`L2CPU_MB_ERR_DENIED`) and a wild x280 access outside it traps; what stays reachable is the policy's own ranges, i.e. NoC targets behind the allowed TLB windows (`NOC_READ32` / `NOC_WRITE32`, applications) | host mailbox timeout | `ctl.mb` raises after 1 s; heartbeat stall | chip reset |
| resident park loop waits for `go_epoch` | **intentionally unbounded** (the host decides when to go) | n/a | record PARKED | RNMI still reaches the hart (NMIE = 1) |
| resident waits for its trigger bit to clear (RNMI) | until the host clears it | n/a | record IN_RNMI | `ctl.rnmi` always clears after its own timeout |
| host: `bringup` READY poll | 5 s | `RuntimeError` | exception | chip reset + start |
| host: `ctl.wait_ready` after a restart | 5 s | `L2cpuCtlError` with the records | exception | `stop()` + restart; chip reset |
| host: `ctl.mb` | 1 s | `L2cpuCtlError` | exception | restart (L2) |
| host: `ctl.stop` | L1: mailbox PARK 0.2 s + 50 ms for the records; then L2 100 ms | `L2cpuCtlError` ("chip reset needed") | exception | chip reset |
| host: `ctl.rnmi` handler wait | 100 ms (default) | clears the trigger, returns the counts | return value | n/a |
| host: clock guard | no wait | raises before touching the tile | exception | start a holder |
| host: `L2cpuMonitor` | 0.1 s sampling, 2 s heartbeat stall | `on_fail(reason)`; optionally exits the process | callback | restart; chip reset |
| `l2cpu_run.sh` lock | waits for the lock (other users) | n/a | n/a | n/a |

Environment-dependent waits, bounded only by timeouts: the clock holder (no holder = frozen harts and a hung chip on
access: the guard turns it into an exception), any NoC target addressed by the mailbox, and producers outside the
tile (Tensix kernels, see the applications). Tests forcing the bounds (QEMU): stalled worker -> `WORK_TIMEOUT`
after 1000 ms with hart 0 alive; mailbox command never acknowledged -> host timeout and a stopped heartbeat; trapped
worker -> `WORKER_DEAD`, resident error park, WARM restart revives it.

## Tests

- `make -C tools/l2cpu/fw clean all qemu-test`: header mirror test (host gcc, rv64, and the tt-metal riscv32 JIT
  compiler when found) and the QEMU suite in the irq and poll flavours (QEMU `virt`, 5 harts: 4 firmware harts and a
  test-driver hart that plays the host): boot from a garbage-filled region, mailbox incl. guarded faults, 200 work
  items, WARM restart into another slot and at the same address, a work item published while parked and served
  after the restart, COLD restart, 1000 restart cycles, trap + worker-dead + restart, the forced bounds above.
  QEMU has no RNMI: the L2 path is tested on the chip only. PMP (QEMU `virt` implements it; the driver hart plays
  the host and is not bound by the harts' entries): 8 locked entries applied at the first boot with the flag,
  mailbox DENIED outside the policy, locked CSRs ignore writes, a wild load (mcause 5), a store into the resident
  page (7), a jump into a data range (1) and a hart 0 store each trap and take the error park, restart + 100
  restarts re-use the locked set, a changed table is refused by every hart; the trap and bound tests after it run
  under the policy.
- `tests/test_boot_mirror.py`, `tests/test_qemu.py` (pytest, skipped without the toolchain),
  `tests/test_chip_smoke.py` (pytest, runs only with `L2CPU_CHIP_TESTS=1`; resets the chips).
- Chip, PMP: `scripts/pmp_probe.py` (policy off: entry count, granule, Smepmp, lock behaviour through the mailbox
  CSR read / write), `scripts/pmp_faults.py` (policy on: section "PMP policy").
- Chip: `scripts/bringup_smoke.py` (bring-up, heartbeats, mailbox, echo work items, trap test, N restarts
  alternating L1 / L2).

## Limits

- Verified on one P300 chip opened as a single chip: tile 0 throughout, tiles 1-3 with the bring-up, the link and
  the sampling firmware (one release for all four). P150, multi-chip opens and other kernel-driver / firmware
  versions are not established. Which PLL4 postdivider feeds which tile is not established (all four are set equal).
- PMP: 8 entries cover the ranges, not their contents (section "PMP policy": the region itself, the core-complex
  devices and the allowed TLB windows stay open to a wild access). The Tensix link test responder
  (`tensix/responder`) is a separate image without the policy (unprotected).
- A restart needs a live clock holder and a cooperative or RNMI-reachable hart; a hart stalled inside a NoC access
  that never completes needs a chip reset.
- The region must stay reserved for the whole session: with a ttnn buffer, keep the tensor alive.
