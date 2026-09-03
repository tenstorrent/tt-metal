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

Two hard constraints discovered here, both shaping the forward-path design:

1. **No NOC atomics against the L2CPU (inbound).** A Tensix `noc_semaphore_inc` into
   L2CPU memory hangs — the bridge doesn't implement atomics. Protocols against the
   L2CPU must be built from plain reads/writes. (Recoverable: the next device init
   resets the wedged Tensix.)
2. **The TLB window does loads/stores only — no atomic op.** So when the x280 *drives*
   the NOC, any credit/semaphore handshake it participates in must also be
   plain-write-based, not atomic-increment-based.

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
reset. Env: `L2CPU_X`/`L2CPU_Y` (default NOC0 `8,3`), `TT_L2CPU_TEST_ATOMIC=1` to
probe the (unsupported) atomic path.

```
./build/programming_examples/metal_example_l2cpu_noc_transfer
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

**Host programs** (three CMake targets, built with the normal metalium build):

| target | source | links |
|---|---|---|
| `metal_example_l2cpu_noc_transfer` | `l2cpu_noc_transfer.cpp` | `TT::Metalium` |
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
./x280/x280_boot boot x280/build/echo.bin      # (or build/bw.bin for the bw sweep)
./x280/x280_boot status                         # inspect PLL / reset / mailbox

# Stage 2: drive the echo from a Tensix kernel.
./build/programming_examples/metal_example_l2cpu_x280_echo
```

> **Hart release is one-shot per chip reset.** A crashed/wedged firmware needs a
> `tt-smi -r` before you can boot again. `x280_boot` refuses to boot if the tile's
> reset bit is already set.

## Next: kernel -> L2CPU -> another device

Target topology first: **two Blackhole chips over a single ethernet link** (fabric
mesh later). The forward path extends stage 2:

1. A Tensix kernel drops a payload into L2CPU memory (stage-1 write path).
2. It signals the x280 firmware via the mailbox (stage-2 handshake).
3. The firmware points its TLB window at a **local ethernet/fabric endpoint** and
   streams the payload out (stage-3 bandwidth path), using **plain-write** credit
   updates — never atomics (constraints 1 & 2 above).
4. A Tensix kernel on the peer chip reads the delivered payload out of its ethernet
   core / L1.

De-risk step 3 as a single-chip loopback (x280 -> local ethernet core, verified
in-chip) before wiring up the second device. See `x280/fw.c`'s `set_window()` for the
window-targeting primitive the forwarder is built on.
