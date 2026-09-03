<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# HANDOFF — l2cpu_fabric_forward (kernel → L2CPU → another device)

**Written:** 2026-09-03, on a machine with **no Tenstorrent device attached**.
**For:** the next agent/engineer on a machine with **two fabric-connected
Blackhole chips**. Everything here compiles; **nothing has been run on
hardware**. Your job is to bring it up on real silicon.

## What this is

A Tensix kernel on **Device A** hands a payload to the on-die **L2CPU (x280)**,
and the x280 — acting as a **from-scratch fabric worker** feeding the standard,
unmodified EDM router — forwards it over one ethernet link (`FABRIC_1D`,
`num_hops=1`) to a Tensix **receiver on Device B**. No forwarder Tensix in the
data path; the x280 does the off-chip send.

- **Design rationale & decisions:** `docs/superpowers/specs/2026-09-03-x280-fabric-worker-design.md`
- **Step-by-step plan (Tasks 1–9):** `docs/superpowers/plans/2026-09-03-x280-fabric-worker.md`
- **Why this is possible:** the fabric worker→EDM protocol uses **no NOC
  atomics** — credits are plain writes to stream registers, and the EDM→worker
  credit is a plain write to a plain L1 address (which stage 1 of the sibling
  `l2cpu_noc_transfer` example proved works inbound to the L2CPU). The one
  genuinely unproven primitive is the **x280→EDM stream-register credit write
  through the TLB window** (see Risk #1).

## Status of each piece

| File | State | Verified |
|---|---|---|
| `x280/fabric_mbox.h` | shared host/firmware mailbox + param contract | — |
| `x280/fw_fabric.c` | x280 fabric-worker firmware (open/send/close) | **compiles** (clang, entry `0x4000_3000_0000`), 15 `CONFIRM ON HARDWARE` markers |
| `x280/build_fw.sh` | firmware build (clang, no riscv64 GCC) | **builds** `build/fw_fabric.bin` (1688 B) |
| `l2cpu_fabric_forward.cpp` | host orchestrator (both chips) | **compiles + links**, 4 `TODO(hardware)` markers |
| `kernels/producer.cpp` | Tensix chip A: payload→LIM, poke x280 | consistent w/ sibling (JIT — built at runtime) |
| `kernels/receiver.cpp` | Tensix chip B: delivered L1→DRAM | consistent w/ sibling (JIT) |
| `CMakeLists.txt` (+ parent) | target `metal_example_l2cpu_fabric_forward` | wired, builds |

The x280 **boot tool is reused** from the sibling example
(`../l2cpu_noc_transfer/x280/x280_boot.cpp`, target
`metal_example_l2cpu_x280_boot`) — note that tool had a UMD-API fix on this
branch (`read_from_arc`→`read_from_arc_apb`); make sure it builds.

## How to build

```bash
cd <repo>
# 1. host + boot tool (metal build with examples on)
cmake -DBUILD_PROGRAMMING_EXAMPLES=ON build && \
  ninja -C build metal_example_l2cpu_fabric_forward metal_example_l2cpu_x280_boot
# 2. x280 firmware (clang; produces x280/build/fw_fabric.bin)
cd tt_metal/programming_examples/l2cpu_fabric_forward/x280 && ./build_fw.sh
```

## How to run (the intended sequence — UNVALIDATED)

```bash
# from a clean state:
tt-smi -r                     # x280 hart release is ONE-SHOT per chip reset
# the host program orchestrates boot + fabric + kernels itself:
./build/programming_examples/metal_example_l2cpu_fabric_forward
```
Env knobs in the host: payload size, L2CPU coords, `FF_SKIP_BOOT=1` (skip x280
boot), `FF_CREDITS_STREAM_ID` (see Risk #2). Expected success: chip-B DRAM ==
staged pattern, "Test Passed".

## Order to bring it up (do these first)

1. **Confirm the fabric trains.** Before anything else, verify two chips see each
   other and `FABRIC_1D` comes up (the host aborts early if not). **Known risk on
   the original box:** its Blackhole showed a single chip and 1D sublines did not
   train fabric. If fabric doesn't train, STOP — the whole approach is blocked at
   the hardware layer, not the code.
2. **Run the stream-register probe (plan Task 8) BEFORE trusting the full path.**
   Write a ~10-line x280 firmware that does one `credit_write_minus_one` to
   `get_stream_reg_write_addr(stream_id)` and reads it back. This isolates Risk
   #1 — the single unproven primitive — from everything else. If it fails, jump
   to Plan B (raw eth) rather than debugging the full pipeline.
3. Then work the plan Tasks 4→7 order (boot+params → open → send → close),
   checking the x280 mailbox status/fault words at each step.

## Top risks & unknowns (priority order — resolve in this order)

1. **Stream-register credit write through the x280 window** (THE unproven
   primitive). Firmware resolved the address as
   `0xFFB40000 + sid*0x1000 + (270<<2)` with packed value `0xFFFFFFC0`
   (`-1 << 6`) — **verify these against the live NOC-overlay headers on target**
   (`fabric_stream_regs.hpp` → `STREAM_REG_ADDR` / `STREAM_REMOTE_DEST_BUF_
   SPACE_AVAILABLE_UPDATE_REG_INDEX` / `REMOTE_DEST_BUF_WORDS_FREE_INC`). If a
   window store doesn't reach the overlay reg with inc-on-write semantics →
   **Plan B: raw ethernet** (`eth_send_bytes`, atomics-free, x280 feeds a minimal
   erisc kernel; see spec "Plan B" and the raw-eth primitives in
   `tt_metal/hw/inc/internal/ethernet/dataflow_api.h`).

2. **Real `sender_channel_credits_stream_id`.** The host sources most EDM params
   live via `append_fabric_connection_rt_args(..., CoreType::ETH)` (args order
   from `erisc_datamover_builder.cpp:542-554`), but the ETH path does **not**
   emit the credits stream id — it defaults to a compile-time `STREAM_ID`. The
   real value is `conn->worker_free_slots_stream_id` from the device-init L1 conn
   table (VC0 path, `edm_fabric_worker_adapters.hpp:112-128`). Host placeholders
   it (default 0, `FF_CREDITS_STREAM_ID` env). **Get the real value on HW.**

3. **Raw-UMD ↔ open-MeshDevice coexistence.** The host pokes the x280 mailbox via
   a second in-process UMD `Cluster` (`X280Mailbox`) while tt-metal owns the
   device — **unvalidated, may throw "device busy"** (construction failure is
   caught, non-fatal). Fallback: route the conn-param + request writes through a
   one-time Tensix setup kernel (plan Task 4 fallback), or sequence device
   open/close around the UMD poke.

4. **CONTRACT GAP — x280 self NOC coords.** The mailbox has **no field** for the
   L2CPU tile's own NOC coords, which the EDM needs to push credits back to the
   worker. Firmware placeholders `worker_xy = (0,0)`. **First packet works**
   (credit sink seeded from `edm_read_counter` at open); **multi-packet flow
   control does not** until real coords are supplied. Fix: add
   `FF_CONN_WORKER_NOC_X/Y` to `fabric_mbox.h`, have the host fill them (the
   L2CPU tile is NOC0 `(8,3)` for tile 0), and read them in `edm_open`.

5. **Packet header size / layout.** Firmware assumes `PKT_HEADER_SIZE=64`
   (`FABRIC_1D_PKT_HDR_EXTENSION_WORDS=1`, host default). If the fabric was built
   with 0, header is 48 B — flip the define (fields `0x00`–`0x2C` are identical
   either way). Also assumes **raw NOC0-identity** dest coords; if fabric uses
   virtual/translated coords, the header targets the wrong chip-B L1.

6. **Open-ack race.** The adapter has no explicit open-ack; firmware `edm_open`
   writes the connection value and returns optimistically (store-visibility
   read-back only), so the first send could race the EDM accepting the
   connection. Add a real ack spin if the first packet is dropped.

7. **Discovery vs. boot sequencing.** Host runs link discovery + early-abort
   BEFORE booting the x280, and boots BEFORE `create_unit_meshes()` (so the boot
   subprocess owns the device). If the control-plane query needs a created mesh
   on HW, move discovery after mesh creation (commented in the host).

## Firmware protocol map (what to check in `fw_fabric.c`)

`fw_main` (hart 0): stamp hartid → `FF_STATE_ALIVE` → wait `FF_MBOX_CONN` params
→ sanity check → `FF_STATE_PARAMS_READY` → `edm_open` → `FF_STATE_OPENED` →
request loop: on new `FF_MBOX_REQ`, `edm_send` one packet → `FF_STATE_SENT` →
`edm_close` → `FF_STATE_CLOSED`. Every spin bounded by `SPIN_CAP` → set
`FF_MBOX_FAULT_CODE` (`FF_FAULT_*`) and park with heartbeat (never wedge — hart
release is one-shot). Window MMIO (`set_window`, aperture) reused verbatim from
the sibling `fw.c`. Resolved constants and their source citations are in-source;
grep `CONFIRM ON HARDWARE`.

## Debugging aids

- The x280 mailbox (uncached GDDR alias `0x3010_0000`, layout in
  `fabric_mbox.h`) carries heartbeat, `fw_state`, `fault_code`, and a status
  block (`slots_seen`, `credit_writes`, `last_free_slots`). Read it with the
  sibling `metal_example_l2cpu_x280_boot status` or raw UMD.
- `fw_state` tells you how far the firmware got; `fault_code` tells you why it
  stopped. Cross-reference the `FF_STATE_*` / `FF_FAULT_*` enums.
- If the receiver gets nothing but the x280 reached `FF_STATE_SENT`: suspect
  Risk #1 (credit write) or #5 (header/coords) — run the Task 8 probe.
