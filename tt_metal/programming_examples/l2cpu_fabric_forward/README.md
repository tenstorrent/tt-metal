<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# L2CPU (x280) as a fabric worker — L2CPU ↔ L2CPU over the fabric

The Blackhole **L2CPU** tile (an x280 RISC-V cluster at NOC0 `(8,3)`) has no NOC
command interface, only a TLB window through which plain loads/stores become NOC
reads/writes. This example shows that this is enough to make the L2CPU a first-class
**fabric worker**: firmware on the x280 opens a connection to the standard,
unmodified fabric router (EDM) on a local ethernet core and pushes packets to any
core on any chip the fabric reaches — including the L2CPU of another chip — exactly
like a Tensix worker does with `WorkerToFabricEdmSender`.

```
CHIP A                                                        CHIP B
[host] --Tensix kernel--> x280 mailbox (request + payload)
                          x280 fw: l2f_send()  ── slot writes + credit ──> [EDM, eth core A]
                                                                              ║ FABRIC_1D
                          [x280 inbox] <── echo (chip B fw: l2f_send()) ── [EDM, eth core B] ──> [x280 inbox on chip B]
                                                                                            (or a Tensix core's L1)
```

**Measured 2026-09-11** on two ethernet-linked p150b Blackhole chips (bh-32, tt-metal
main @ `7b5ae30865b`, `FABRIC_1D`, 48 B packet header, 16 × 4400 B router slots):

| test | result |
|---|---|
| chip A x280 → fabric → chip B **L2CPU inbox**, 4096 B + 16 B header (2 packets) | all words match; 793 µs in-firmware |
| chip B x280 **auto-echo** → chip A inbox (firmware-initiated, no host/Tensix) | all words match |
| 16000 B (4 payload packets + header) both directions | all words match; 3.0 ms in-firmware |
| chip A x280 → chip B **Tensix L1** (`FF_DEST=tensix`), 4096 B | all words match; 770 µs |
| `l2f_close()`: router releases the channel (handshake word → 0) | 11 µs, no fault |
| reconfigure a running firmware for a new host run (`FF_SKIP_BOOT=1`) | works; no chip reset needed |

In-firmware time is dominated by 8-byte window stores (~5 MB/s); see *Next steps*.

## Files

| file | role |
|---|---|
| `x280/l2cpu_fabric.h` | **The reusable piece.** Freestanding C header: `l2f_open / l2f_send / l2f_send_packet / l2f_free_slots / l2f_close` — the worker→router protocol on top of TLB-window loads/stores. No tt-metal headers; every constant cites its source. |
| `x280/fw_fabric.c` | Firmware app on top of the library: waits for a connection block, probes window coordinates, opens, then serves requests and echoes inbox messages. Reconfigures itself when the host publishes a new connection block. |
| `x280/fabric_mbox.h` | Mailbox contract (host ↔ Tensix kernels ↔ firmware), in the tile's uncached GDDR alias `0x3010_0000`. |
| `x280/build_fw.sh` | Builds `x280/build/fw_fabric.bin` with clang (reuses `../l2cpu_noc_transfer/x280/{start.S,dram.ld,build_fw.sh}`). |
| `kernels/conn_setup.cpp` | Tensix kernel: `WorkerToFabricEdmSender::build_from_args()` on the args `append_fabric_connection_rt_args(..., CoreType::WORKER)` emits → writes the resolved router parameters into the x280 mailbox. Also plants a probe word in the router's cursor pad. |
| `kernels/l2cpu_mem_write.cpp`, `kernels/l2cpu_mem_read.cpp` | Generic host ↔ L2CPU memory movers (DRAM ↔ L1 ↔ L2CPU tile). All host access to x280 memory goes through these. |
| `kernels/receiver.cpp` | `FF_DEST=tensix` variant: copies the delivered L1 bytes to DRAM for verification. |
| `l2cpu_fabric_forward.cpp` | Host orchestrator (`metal_example_l2cpu_fabric_forward`). |

The x280 boot tool (`metal_example_l2cpu_x280_boot`, `--chip N status|niu|boot <fw>`)
lives in the sibling `../l2cpu_noc_transfer/` example.

## Build

```bash
./build_metal.sh --release --enable-ccache --build-programming-examples   # host + boot tool
tt_metal/programming_examples/l2cpu_fabric_forward/x280/build_fw.sh       # -> x280/build/fw_fabric.bin
```

Firmware toolchain: clang-20 with `--target=riscv64-unknown-elf`, lld, llvm-objcopy
(no riscv64 GCC needed). `-march=rv64ima_zicsr_zicbom`, no C extension (the trap
handler in `start.S` skips faulting instructions by 4 bytes).

## Run

```bash
tt-smi -r                                            # x280 hart release is ONE-SHOT per chip reset
export TT_METAL_HOME=$PWD
./build/programming_examples/metal_example_l2cpu_fabric_forward                # boots both x280s, A->B->A
FF_SKIP_BOOT=1 FF_DEST=tensix ./build/programming_examples/metal_example_l2cpu_fabric_forward
FF_SKIP_BOOT=1 FF_PAYLOAD_SIZE=16000 ./build/programming_examples/metal_example_l2cpu_fabric_forward
FF_SKIP_BOOT=1 FF_CLOSE=1 ./build/programming_examples/metal_example_l2cpu_fabric_forward
```

After the first run the firmware keeps running; every later run (`FF_SKIP_BOOT=1`)
publishes a new connection block and the firmware re-probes and re-opens against
that run's routers. A chip reset is only needed to load a *different firmware binary*.

Env knobs: `FF_CHIP_A`/`FF_CHIP_B` (0/1), `L2CPU_X`/`L2CPU_Y` (8,3), `FF_PAYLOAD_SIZE`
(4096; multiple of 16, ≤ 16320), `FF_DEST` (`l2cpu`|`tensix`), `FF_TWO_WAY` (1: boot
chip B's x280 too and expect the echo), `FF_CLOSE` (0), `FF_SKIP_BOOT` (0),
`FF_PROBE_NOC0` (0, see *Facts established here*), `FF_TIMEOUT_S` (20), `FF_BOOT_TOOL`, `FF_FW_BIN`.

## How it works

### Connection parameters (host → x280)
A Tensix worker learns its router connection from the L1 connection table device-init
writes (`tensix_fabric_connections_l1_info_t`). `kernels/conn_setup.cpp` runs that exact
code path (`WorkerToFabricEdmSender::build_from_args<TENSIX>`) and copies the resolved
fields into the mailbox connection block (`FF_CONN_*`): EDM eth core (translated
coords), channel buffer base / slot count / slot size, handshake and worker-location
addresses, the producer-cursor address, the free-slots **stream register** write/read
addresses (stream id 22 = sender channel 0), plus what the x280 cannot know by itself:
its own NOC coords (where the router pushes credits), the packet header size
(`get_tt_fabric_packet_header_size_bytes()`), two local words for the router's credit
return and teardown ack, the peer L2CPU's inbox for firmware-initiated replies, and a
per-run nonce in `FF_CONN_VALID`.

### Worker protocol (`x280/l2cpu_fabric.h`)
Port of `edm_fabric_worker_adapters.hpp` to window MMIO:

* **open** — read the producer cursor and `edm_read_counter` from the router, seed the
  local credit sink, write `worker_semaphore_address` / `worker_teardown_semaphore_address`
  / `worker_xy` into `EDMChannelWorkerLocationInfo`, write `1` to the handshake word.
* **send** — wait for a free slot (`num_buffers − (write_counter − edm_read_counter)`),
  store payload then the 48/64 B header into the slot (`8 B` window stores), then one
  `32-bit` store of `0xFFFFFFC0` (`-1 << 6`) to the router's free-slots stream register
  (`STREAM_REG_ADDR(22, 270)`), advance the cursor. Payloads larger than a slot are
  chunked; packets on one connection arrive in order.
* **credits back** — the router writes its read counter into the local sink word with a
  plain `noc_inline_dw_write` (no atomics involved).
* **close** — persist the cursor, write `2` to the handshake word, wait for the router to
  clear it to `0` (its "connection released" signal). The router additionally acks with
  `noc_semaphore_inc(teardown_word, 1)`; whether that landed is reported in
  `FF_DIAG_TEARDOWN_ACK` but nothing depends on it.

### Messages between L2CPUs
A message is *data packets into the peer's inbox data area* followed by *one 16 B
header packet* `{len, seq, tag, 0}` into the peer's inbox header (same connection ⇒
in order; the receiver polls `seq`). `FF_TAG_ORIGINAL` messages are echoed back by a
firmware with `FF_CFLAG_AUTO_ECHO` as `FF_TAG_ECHO` (which is never echoed again).
Host-side "send" = write payload to `FF_MBOX_OUTBOX`, write a request block, write
`FF_REQ_SEQ` last, poll `FF_RESP_SEQ`.

## Facts established here (all measured on hardware)

* **The L2CPU's NOC port translates coordinates.** The window probe matched the
  router's *translated* eth coordinates `(25,25)` for eth channel 5 on both chips, and
  `x280_boot niu` reads the tile's `NIU_CFG_0` with `NOC_ID_TRANSLATE_EN` set (register
  block at the 64-bit base `0xFFFFFFFFFF000000` UMD uses for NOC2AXI tiles — **not**
  `0xFFB2_0000`). L2CPU translated coords are identity (`(8,3)`), so `(8,3)` is right in
  both the window config and packet headers. A NOC0-*physical* eth coordinate is an
  unmapped translated coordinate here; the fallback probe is opt-in (`FF_PROBE_NOC0=1`).
* **A 32-bit window store to a router stream register performs the inc-on-write credit.**
  (Risk #1 of the original design.) The router's free-slot count moved `0x10 → …` per
  packet and every slot was returned.
* **The receiving router writes into L2CPU memory over NOC1** (`edm_to_local_chip_noc = 1`)
  with the same coordinates — Blackhole `NOC_0_X/NOC_0_Y` are identity.
* **Window reads of the 0xFFB4xxxx stream-register page work** (used for diagnostics).
* **Firmware `.bss` is not zeroed by loading** — `start.S` now clears `_bss_start.._bss_end`.
* **Never truncate an x280 pointer to 32 bits.** `.data/.bss` live in the *cached* GDDR
  alias ≥ 4 GiB; the truncated address is the *uncached* alias of the same DRAM and
  returns stale bytes (this produced a garbage inbox header on the first run).
* **Sender channel 0 is exclusive**: while the x280 holds the connection, no Tensix worker
  can connect to that router. Close it (`FF_CLOSE=1`) if Tensix fabric traffic is needed
  on the same link.
* **NOC atomics into the L2CPU (measured, not a verdict)**: with the `noc_semaphore_inc`
  encoding (`NOC_AT_INS_INCR_GET`), two probes moved the tile's
  `NIU_SLV_NONPOSTED_ATOMIC_RECEIVED` counter by +2 while `NIU_SLV_ATOMIC_RESP_SENT`
  stayed 0 and the target word was unchanged; the router's teardown
  `noc_semaphore_inc` shows the same signature (`FF_DIAG_TEARDOWN_ACK = 0`). Untried:
  other AT opcodes (SWAP/CAS/ACC), posted atomics, other L2CPU NOC ports. Re-measure
  with `metal_example_l2cpu_atomic_probe` + `x280_boot niu`; nothing in this path
  needs atomics. (With the hart booted the L3 is a cache, so probe the GDDR alias,
  not LIM.)

## Debugging

`FF_MBOX` line 0 (`heartbeat`, `fw_state`, `fault_code`, `trap_count`, `mcause`,
`boot_marker`), the diagnostics block (`FF_DIAG_*`: probe result, window coords, values
adopted at open, free slots, inbox/echo counters, config generation, teardown ack) and
the response block are printed by the host; `x280_boot --chip N status` shows line 0
and `niu` the tile's NIU config/counters from the host. Every wait in the firmware is a
2 s `mcycle` deadline; a fault sets `fault_code` and the loop keeps serving.

## Next steps

* Bandwidth: 8 B window stores give ~5 MB/s; try wider stores / the window's ordering
  bits (layout of the property word beyond `x_end/y_end` is unverified) or stage
  payload in eth L1 with fewer, larger transactions.
* Interrupt/doorbell instead of polling on the x280.
* Multi-hop (`FF_REQ_NUM_HOPS > 1`; the 1D routing encoder is in place) and 2D fabric.
* Hosting a fabric mux on the x280 (multiple upstream producers).
