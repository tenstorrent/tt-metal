<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# The fabric mux, hosted on the L2CPU

Worker kernels use the **V1 fabric-mux client interface** (`build … / wait ready /
connect / fabric_async_write / disconnect / terminate`), but the mux core they talk to is
the Blackhole **L2CPU** tile: its x280 firmware (`x280/fw_mux.c`) owns the per-channel
slot rings in L2CPU memory, forwards every committed slot into its own router
connection (`../l2cpu_fabric_forward/x280/l2cpu_fabric.h`), and returns credits into each
worker's L1 through its TLB window. The kernel-side change is the set of addresses it
is given and one header (`kernels/tt_fabric_l2cpu_mux_interface.hpp`); the call sequence
and semantics are unchanged.

```
chip A                                                             chip B
[worker 0] ─┐  slot writes + commit        ┌─────────────── L2CPU (8,3) ───────────────┐
[worker 1] ─┼─────────────────────────────▶│ ch0..7: 8 slots each, handshake, cursor,   │  fabric   [Tensix L1]
[worker 2] ─┤                              │ location info, producer write counter      ├──router──▶  [c][p] regions
[worker 3] ─┘◀── read counter into L1 ─────│ x280: forward committed slots, credits back │
                                           └────────────────────────────────────────────┘
```

**Measured 2026-09-11** (bh-32, two p150b, FABRIC_1D, 48 B header, 4400 B router slots):

| test (`LM_ROUNDS` client programs per mux run; every client also sends one header-only atomic-inc) | result |
|---|---|
| 4 clients × 8 × 2048 B, 1 round | all 16384 words match; graceful TERMINATED |
| 4 × 5 × **1040 B** (not a 64 B multiple), **3 rounds** (channels reconnect, cursor wraps: idx 6 → 4 → 2) | all rounds match; persisted cursors correct; 12 atomic incs → semaphore 60 |
| 8 × 24 × **4352 B** (max payload, 835 KB/round), 2 rounds | all match; mux accepted 835 KB in 30 ms, delivered in 47 ms; semaphore 384 |
| 1 × **200 × 16 B** tiny packets, 2 rounds | all match; cursors 201 → 402 |
| default run twice in a row (`FF_SKIP_BOOT=1`, firmware re-arms after TERMINATED) | both pass |
| `l2cpu_fabric_forward` two-way after the shared-header refactor | still passes |

Delivery is verified only after the atomic-inc semaphore on chip B reaches the expected
count: clients return once the mux has *accepted* their packets, forwarding to the
router and delivery complete afterwards.

## What is the same, what changed

| | Tensix V1 mux | L2CPU mux |
|---|---|---|
| Client type | `WorkerToFabricMuxSender<N>` | `WorkerToL2cpuMuxSender<N>` (same methods) |
| Build handle | `build_connection_to_fabric_endpoint(...)` | `build_connection_to_l2cpu_mux(...)` — same 13 args in the same order |
| Wait ready / connect / send / disconnect / terminate | `wait_for_fabric_endpoint_ready`, `fabric_client_connect`, `fabric_async_write`, `fabric_atomic_inc`, `fabric_client_disconnect`, `fabric_endpoint_terminate` | same names; `wait_for_l2cpu_mux_ready` replaces the ready wait (16 B read) |
| Handshake, location info, cursor, teardown, read-counter credits | `EDMChannelWorkerLocationInfo`, 1/2 handshake, `SenderChannelProducerCursor`, mux writes read counter into worker L1 | identical, byte for byte |
| **Commit** | decrement stream register `channel_id` on the mux core | **plain inline write of the producer's write counter** into `LM_WRITE_COUNTER(ch)` — the L2CPU tile has no stream registers |
| Teardown ack | `noc_semaphore_inc` into the worker | plain write of 1 into the worker's teardown word (satisfies the same `!= 1` spin) |
| Channel kinds | full-size + header-only pools | uniform channels (a header-only packet is a 48 B payload) |
| Shutdown | client writes termination signal | same; firmware re-arms when the host clears the word, so no chip reset between runs |

Host side, `l2cpu_mux_layout.h` plays the role of `FabricMuxConfig`'s getters:
`LM_CHANNEL_BASE(ch, slot)`, `LM_CONN_INFO(ch)`, `LM_HANDSHAKE(ch)`, `LM_WRITE_COUNTER(ch)`
(= flow-control address), `LM_CURSOR(ch)` (= buffer-index address), `LM_STATUS`,
`LM_TERMINATION`. The mux's own router connection is set up with the same
`conn_setup.cpp` kernel and mailbox as the fabric_forward example.

## The one rule that bit: 64-byte offsets on READS

Measured with `metal_example_l2cpu_align_probe` (in `../l2cpu_noc_transfer/`):

* **Writes Tensix L1 → L2CPU memory are fine at any source/destination offset.** All ten
  offset combinations tested (0/16/32/48 mixes) landed exactly where asked. Payloads from
  16 B-aligned circular buffers and headers from the 144 B-strided packet header pool
  are therefore valid write sources.
* **Reads L2CPU memory → Tensix L1 fetch from `(src & ~63) + (landing & 63)`.** The
  source's offset within 64 B is replaced by the L1 landing address's offset; the data is
  right only when `src % 64 == landing % 64`, and wrong silently otherwise (this is how
  the first mux client spun forever on a status word that read back as a neighbour).

Only the client's own three reads are affected (status word, cursor block, read-counter
seed), so `build_connection_to_l2cpu_mux` derives their landing zones from one 64 B-aligned
local scratch (it aligns the address it is given). Consequences in this example:

* `LM_CURSOR(ch)` and `LM_CONN_INFO(ch)` are 64 B strided so their offsets are fixed
  (cursor `+0`, `edm_read_counter` `+0x30`);
* the client takes one ≥ 192 B local scratch (the "buffer index" argument), aligns it to
  64 B and places the cursor landing at `+0x00`, the status landing at `+0x40` and the
  flow-control word at `+0x70`; the flow-control address argument is honoured only if it
  already sits at `+0x30 mod 64`;
* mux slots are strided to `LM_SLOT_STRIDE(slot_bytes)` (4400 → 4416); not required for
  writes, kept so slot addresses stay 64 B aligned for the x280's 8 B copies.

Inline 4 B writes and the x280's own window stores are unaffected either way.

## Two more things the tests taught

* **A close request may arrive before the mux ever saw the open.** A client can open,
  send its packets and request close between two firmware polls. The Tensix mux handles
  this because `connect_is_requested()` is true for both handshake values; the x280
  firmware now does the same: on `2` it takes the location info from the block and acks
  regardless of whether it observed the `1`. Before the fix the client spun forever on
  the missing ack.
* **A single-page interleaved L1 `MeshBuffer` lives in one bank, i.e. one core's L1.**
  `EnqueueRead/WriteMeshBuffer` on it reach that core, not core (0,0) where the fabric
  delivers to `buffer->address()`; the allocation only reserves the address range on
  every core. The example zeroes and reads chip B's L1 through kernels on core (0,0)
  (`kernels/l1_fill.cpp`, `receiver.cpp`).

## Build and run

```bash
./build_metal.sh --release --enable-ccache --build-programming-examples
tt_metal/programming_examples/l2cpu_mux/x280/build_fw.sh            # -> x280/build/fw_mux.bin
tt-smi -r                                                            # once per firmware binary
export TT_METAL_HOME=$PWD
./build/programming_examples/metal_example_l2cpu_mux                 # boots the x280, 4 clients x 8 packets
FF_SKIP_BOOT=1 LM_NUM_CLIENTS=8 LM_NUM_PACKETS=32 LM_PAYLOAD=1024 ./build/programming_examples/metal_example_l2cpu_mux
```

Env: `LM_NUM_CLIENTS` (≤ 8), `LM_NUM_PACKETS`, `LM_PAYLOAD` (multiple of 16, ≤ slot − header),
`FF_SKIP_BOOT`, `FF_CHIP_A/B`, `L2CPU_X/Y`, `FF_TIMEOUT_S`, `FF_BOOT_TOOL`, `FF_FW_BIN`.
Diagnostics: `metal_example_l2cpu_x280_boot --chip 0 rd 0x30200100 16` dumps the
per-channel stats; `rd 0x30201400 16` the handshakes; `rd 0x30201500 16` the write
counters; `status`/`niu` the firmware mailbox and NIU counters.

## Files

| file | role |
|---|---|
| `kernels/tt_fabric_l2cpu_mux_interface.hpp` | client class + V1-named helper overloads |
| `kernels/mux_sender_client.cpp` | worker kernel: the V1 test client's sequence against the L2CPU mux |
| `x280/fw_mux.c`, `x280/build_fw.sh` | mux firmware (uses the fabric_forward library and mailbox) |
| `l2cpu_mux_layout.h` | shared memory map (host + firmware) |
| `l2cpu_mux.cpp` | host: boot, router setup, N clients, verify, stats |
| `../l2cpu_fabric_forward/l2cpu_host_utils.hpp` | shared host helpers (X280Mem, setup_connection, status) |

## Using it from `all_gather_minimal_matmul_async` (assessment, 2026-09-11)

Of the ttnn `*matmul*` collectives only `all_gather_minimal_matmul_async` uses the Tensix
V1 mux (`FabricMuxConfig` + `tt_fabric_mux.cpp`, one mux per (link, direction), one
full-size channel per worker, no header-only channels). Its writer kernels
(`matmul_dataflow_common.hpp`, `dm_in0_sender.cpp`, `dm_in1_sender_out.cpp`) call:
`build_connection_to_fabric_endpoint` (13 args), `wait_for_fabric_endpoint_ready`,
`fabric_client_connect`, the linear API `fabric_unicast_noc_{scatter_write,unicast_write,
unicast_atomic_inc}_with_state` (which only need `wait_for_empty_write_slot`,
`send_payload_without_header_non_blocking_from_address`,
`send_payload_flush_non_blocking_from_address`), `fabric_client_disconnect`,
`fabric_endpoint_terminate`. Every one of those is provided by
`WorkerToL2cpuMuxSender`, and the `is_mux_sender` trait specialization in the client
header lets the `_with_state` templates accept it.

What has to change:

| piece | change |
|---|---|
| `matmul_dataflow_common.hpp` | under `USE_L2CPU_MUX`: `WorkerToL2cpuMuxSender<N>` + `build_connection_to_l2cpu_mux`, `mux.wait_for_ready(status, LM_CONFIG_GEN, nonce)`; RT 14 becomes a ≥192 B L1 scratch address instead of a semaphore id |
| program factory `fabric_mux_connection_*_args` | `FabricMuxConfig` getters → `LM_*` macros; CT arg 0 = `LM_NUM_BUFFERS` (8), CT arg 1 = `LM_SLOT_STRIDE(slot)`; `mux_x/y` = L2CPU tile (8,3)/(8,5)/(8,7)/(8,9) per (link, direction) |
| program factory mux placement | drop `CreateKernel(tt_fabric_mux.cpp)` and `get_fabric_mux_run_time_args`; instead add one `conn_setup.cpp` kernel per mux to the op program (it delivers the router connection block with a fresh nonce; the firmware reconfigures and re-publishes `LM_CONFIG_GEN`, which the clients wait for) |
| device bring-up | boot `fw_mux.bin` on each chip's L2CPU before the device opens (subprocess call of the boot tool in the test fixture); `tt-smi -r` first |
| termination | either keep the op's graceful terminate and clear `LM_TERMINATION` in the conn_setup kernel of the next launch, or leave the mux running |

Alignment is not a blocker: the op's payloads (16 B-aligned CB pages) and headers
(144 B-strided packet header pool) are NOC **writes**, which the probe showed land
correctly at any offset. Real limits: 4 L2CPU tiles per chip → at most 4 muxes
(Linear `num_links ≤ 4`, no FSDP fusion), fixed 8 slots per channel, and x280
forwarding bandwidth far below a Tensix mux. A correctness-first port is one op, one
link, one direction; the files above are the complete list.

## Limits and next steps

* Throughput is bounded by the x280's 8 B window stores (~5 MB/s per forwarded byte
  stream). Wider window bursts or DMA-style copies are the next lever.
* One router connection per L2CPU; the mux holds sender channel 0 of that router.
* Header-only (credit) channels are emulated with full slots; a smaller-stride pool is a
  layout change only.
* The V2 client (`FabricMuxV2Sender`) could be ported the same way: replace its stream
  register commit with the write-counter write and its manager's `noc_semaphore_inc` ack
  with a plain write.
