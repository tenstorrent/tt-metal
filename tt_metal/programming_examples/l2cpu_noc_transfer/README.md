<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# L2CPU NOC transfer

Bring-up examples for the Blackhole **L2CPU** tile — the on-die x280 cluster (four
RV64IMAC+Zicbom "harts"). These prove out, step by step, using the L2CPU as a NOC
participant, building toward the end goal: **a Tensix kernel hands a payload to the
L2CPU, and the L2CPU forwards it off-chip to another device.**

Everything here targets **Blackhole** (L2CPU tiles exist only there). The stages so
far were measured on a single-chip p100a.

## The L2CPU as a NOC participant — the one key fact

The x280 has **no NOC command interface**. It cannot issue `noc_async_write` the way a
Tensix RISC does. Its only NOC access is a **TLB window**: write the destination tile
`(x, y)` + address bits into a config register, then ordinary RISC-V loads/stores
through the window's aperture become NOC reads/writes. Everything the firmware does on
the NOC (`x280/fw.c`, `x280/fw_bw.c`) goes through `set_window()` + plain load/store.

Two properties of this path shape the forward-path design:

1. **The TLB window does loads/stores.** When the x280 *drives* the NOC, everything it
   does is a plain load or store through the aperture. The fabric worker->router
   protocol needs exactly that: plain writes for payload/header, one 32-bit store to a
   stream register for the credit, and a plain inbound write for the router's credit
   return (see `../l2cpu_fabric_forward/`).
2. **Inbound NOC atomics are measured, not assumed.** `metal_example_l2cpu_atomic_probe`
   seeds a word in L2CPU memory with a plain write, fires one `noc_semaphore_inc` at it,
   waits a bounded time for the NIU atomic-response counter instead of blocking on
   `noc_async_atomic_barrier()`, and reads the word back. Run it whenever a design
   would like to lean on atomics against the L2CPU, and treat a miss as a probe or
   encoding question first (address class, `NOC_AT` encoding, alignment) — the tool
   exists so the question can be re-asked cheaply.

## Stages

### Stage 1 — L2CPU as a passive NOC target
`l2cpu_noc_transfer.cpp` + `kernels/l2cpu_rw.cpp` — **built by default.**

A Tensix data-movement kernel round-trips a pattern through the L2CPU tile's LIM
scratchpad (x280 physical `0x0800_0000`), with the harts held in reset:

```
host -> DRAM -> Tensix L1 --noc_async_write--> L2CPU LIM
                Tensix L1 <--noc_async_read--- L2CPU LIM -> DRAM -> host
```

plus an `noc_inline_dw_write` word patch. Inbound NOC access works with the harts in
reset. Env: `L2CPU_X`/`L2CPU_Y` (default NOC0 `8,3`). `TT_L2CPU_TEST_ATOMIC=1` adds a
`noc_semaphore_inc` + blocking `noc_async_atomic_barrier()` to the kernel; prefer the
bounded probe below for measuring atomics.

```
./build/programming_examples/metal_example_l2cpu_noc_transfer
```

### NOC atomic probe
`l2cpu_atomic_probe.cpp` + `kernels/l2cpu_atomic_probe.cpp` — **built by default.**

One `noc_semaphore_inc` per target word (LIM `0x0801_0100` and uncached GDDR alias
`0x3010_2000` by default), bounded wait on `NIU_MST_ATOMIC_RESP_RECEIVED`, plain
read-back, and a resync of the software ack counter so nothing hangs afterwards. Prints,
per target, whether the increment landed and whether a response arrived. Env:
`DEVICE_ID`, `L2CPU_X/Y`, `PROBE_ADDR_A/B` (0 disables B), `PROBE_SPIN`, `PROBE_INCR`.
Pair it with `metal_example_l2cpu_x280_boot --chip N niu`, which reads the L2CPU tile's
own NIU slave counters (`NIU_SLV_NONPOSTED_ATOMIC_RECEIVED`, `NIU_SLV_ATOMIC_RESP_SENT`)
before/after. Once the x280 has been booted the L3 is a cache and the LIM address class
no longer exists — probe the GDDR alias then.

```
./build/programming_examples/metal_example_l2cpu_atomic_probe
```

### NOC alignment probe
`l2cpu_align_probe.cpp` + `kernels/l2cpu_align_probe.cpp` — **built by default.**

Writes 64 B from Tensix L1 into L2CPU memory and reads it back for a matrix of source and
destination offsets within 64 B, and prints where the data landed. Result (2026-09-11):
writes are correct at any offset pair; reads fetch from `(src & ~63) + (landing & 63)`,
i.e. they are correct only when both sides share the same offset within 64 B.

```
./build/programming_examples/metal_example_l2cpu_align_probe
```

### Stage 2 — L2CPU as an active NOC initiator (echo)
`x280/fw.c`, `x280/start.S`, `x280/x280_boot.cpp`, `x280/x280_echo_test.cpp` +
`kernels/x280_echo_poll.cpp`.

The x280 hart is booted and runs firmware that, on a mailbox request from a Tensix
kernel, aims its TLB window at that Tensix and **writes a result back into the Tensix's
L1** — the first x280-initiated NOC write into a Tensix. This is the proof the L2CPU
can *drive* the NOC, which is what off-chip forwarding needs.

### Stage 3 — TLB-window store bandwidth
`x280/fw_bw.c`. Times x280 stores through the window (NOC loopback to its own GDDR) —
the go/no-go number for hosting a fabric-mux-style forwarder on the L2CPU. Sweeps
posted vs. default ordering and 64- vs 32-bit stores; `x280_boot` prints MB/s.

## Building

**Host programs** (four CMake targets, built with the normal metalium build):

| target | source | links |
|---|---|---|
| `metal_example_l2cpu_noc_transfer` | `l2cpu_noc_transfer.cpp` | `TT::Metalium` |
| `metal_example_l2cpu_atomic_probe` | `l2cpu_atomic_probe.cpp` | `TT::Metalium` |
| `metal_example_l2cpu_x280_echo`    | `x280/x280_echo_test.cpp` | `TT::Metalium` |
| `metal_example_l2cpu_x280_boot`    | `x280/x280_boot.cpp` | `umd::tt-umd` (raw UMD) |

**x280 firmware** (bare-metal RV64; clang, no riscv64 GCC needed):

```
cd x280 && ./build_fw.sh      # -> build/echo.bin (stage 2), build/bw.bin (stage 3)
```

The firmware is freestanding (`start.S` + `dram.ld`, no libc). Both images link
`.text` at `0x4000_3000_0000` (cached GDDR), matching the reset vector `x280_boot`
programs. `lim.ld` is the older LIM-resident layout, kept for reference.

## Running stage 2 / 3

```
# One-shot per chip reset: releases the x280 harts and loads firmware.
# (--chip N selects the chip; default 0)
./build/programming_examples/metal_example_l2cpu_x280_boot boot x280/build/echo.bin   # (or build/bw.bin)
./build/programming_examples/metal_example_l2cpu_x280_boot status                      # PLL / reset / mailbox

# Stage 2: drive the echo from a Tensix kernel.
./build/programming_examples/metal_example_l2cpu_x280_echo
```

> **Hart release is one-shot per chip reset.** A crashed/wedged firmware needs a
> `tt-smi -r` before you can boot again. `x280_boot` refuses to boot if the tile's
> reset bit is already set.

## Next: kernel -> L2CPU -> another device

Implemented in `../l2cpu_fabric_forward/`: the x280 acts as a fabric worker feeding the
standard EDM router over one ethernet link, with a small freestanding library
(`x280/l2cpu_fabric.h`) that gives L2CPU firmware `open / send / close` against the
router, and L2CPU-to-L2CPU messaging in both directions. See that README for the
protocol, the bring-up sequence and the measured results.
