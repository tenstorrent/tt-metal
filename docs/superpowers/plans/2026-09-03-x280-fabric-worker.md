# x280 Fabric Worker — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Tensix kernel hands a payload to the on-die L2CPU (x280) on Device A, and the x280 — acting as a from-scratch fabric worker feeding the standard EDM router — forwards it over one ethernet link to a Tensix receiver on Device B.

**Architecture:** Reuse the unmodified fabric EDM router and `FABRIC_1D` config. The novelty is confined to x280 firmware that reimplements the `WorkerToFabricEdmSender` open/send/close protocol using only TLB-window MMIO (plain loads/stores + one stream-register write) — no NOC atomics, no forwarder Tensix. The host boots the x280, sources the EDM connection parameters, and delivers them to the x280 mailbox.

**Tech Stack:** C++ host (tt-metalium + raw UMD), Tensix data-movement kernels, bare-metal RV64 x280 firmware (clang, `build_fw.sh`), tt-metal fabric (`FABRIC_1D`).

**Spec:** `docs/superpowers/specs/2026-09-03-x280-fabric-worker-design.md`

## Global Constraints

- **Hardware required:** two fabric-connected Blackhole chips that actually train `FABRIC_1D`. No device is attached during planning; every "run on HW" step is gated on that. Confirm the A↔B link trains before Task 5+.
- **x280 hart release is one-shot per chip reset.** Firmware must never wedge: every poll loop is bounded with a timeout that faults to the mailbox and parks with a heartbeat. Recovery is `tt-smi -r`.
- **No NOC atomics anywhere** (inbound to L2CPU is unsupported; the fabric worker path needs none). Credits use stream-register writes / plain L1 writes only.
- **No forwarder Tensix in the data path.** The producer Tensix only stages the payload + pokes the x280; a Tensix never touches the fabric.
- **Single packet first cut:** payload ≤ one EDM buffer slot (`buffer_size_bytes`). One worker, one link, one direction (A→B), one packet in flight.
- **License header** on every new file: `// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC` / `// SPDX-License-Identifier: Apache-2.0`.
- **Firmware build:** clang recipe in `tt_metal/programming_examples/l2cpu_noc_transfer/x280/build_fw.sh` (`rv64imac_zicbom`, `-mcmodel=medany`, `dram.ld`, entry `0x4000_3000_0000`). Extend it; do not fork the flags.
- **Location:** new example lives in `tt_metal/programming_examples/l2cpu_fabric_forward/`, reusing `../l2cpu_noc_transfer/x280/{x280_boot,start.S,dram.ld,build_fw.sh}` where possible.

---

## File Structure

- `tt_metal/programming_examples/l2cpu_fabric_forward/CMakeLists.txt` — targets: host program + guarded UMD boot reuse.
- `tt_metal/programming_examples/l2cpu_fabric_forward/l2cpu_fabric_forward.cpp` — host orchestrator (both chips).
- `tt_metal/programming_examples/l2cpu_fabric_forward/kernels/producer.cpp` — Tensix (chip A): payload → LIM, poke x280.
- `tt_metal/programming_examples/l2cpu_fabric_forward/kernels/receiver.cpp` — Tensix (chip B): delivered L1 → DRAM.
- `tt_metal/programming_examples/l2cpu_fabric_forward/x280/fw_fabric.c` — x280 fabric-worker firmware.
- `tt_metal/programming_examples/l2cpu_fabric_forward/x280/build_fw.sh` — thin wrapper calling the noc_transfer builder on `fw_fabric.c`.
- `tt_metal/programming_examples/l2cpu_fabric_forward/README.md` — build/run/verify.
- `tt_metal/programming_examples/CMakeLists.txt` — add the subdirectory.

Mailbox layout (x280 LIM, uncached GDDR alias `0x3010_0000`, extends stage-2 layout):
- `0x000` heartbeat u64 | `0x008` fw_state u64 | `0x010` hartid u64 | `0x018` trap_count u64 | `0x020` fault_code u64
- `0x080` request: `req_seq u32 | payload_lim_addr u32 | size u32 | dest_noc_x u32 | dest_noc_y u32 | dest_l1_addr u32`
- `0x100` conn params block (written by host): the `build_from_args` VC2 set — `edm_noc_x, edm_noc_y, edm_buffer_base_addr, num_buffers_per_channel, buffer_size_bytes, edm_connection_handshake_l1_addr, edm_worker_location_info_addr, edm_copy_of_wr_counter_addr, sender_channel_credits_stream_id, worker_free_slots_l1_addr` (u32 each)
- `0x180` status: `state u32 | slots_seen u32 | credit_writes u32 | last_free_slots u32`

---

## Task 1: Example scaffold + firmware build wiring (no device)

**Files:**
- Create: `tt_metal/programming_examples/l2cpu_fabric_forward/CMakeLists.txt`
- Create: `tt_metal/programming_examples/l2cpu_fabric_forward/l2cpu_fabric_forward.cpp` (stub `main` returning 0, prints "scaffold")
- Create: `tt_metal/programming_examples/l2cpu_fabric_forward/x280/fw_fabric.c` (stub: sets `fw_state=ALIVE`, heartbeats forever — copy the park loop from `../l2cpu_noc_transfer/x280/fw.c`)
- Create: `tt_metal/programming_examples/l2cpu_fabric_forward/x280/build_fw.sh`
- Modify: `tt_metal/programming_examples/CMakeLists.txt` (add `add_subdirectory(l2cpu_fabric_forward)`)

**Interfaces:**
- Produces: build target `metal_example_l2cpu_fabric_forward`; firmware `x280/build/fw_fabric.bin`.

- [ ] **Step 1:** Write `build_fw.sh` — a thin wrapper that invokes the verified builder in the sibling example on `fw_fabric.c` with `dram.ld`:
```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
SIB=../../l2cpu_noc_transfer/x280
CLANG="${CLANG:-clang}"
OBJCOPY="${OBJCOPY:-$(command -v llvm-objcopy || ls /usr/bin/llvm-objcopy-* | head -1)}"
mkdir -p build
"${CLANG}" --target=riscv64-unknown-elf -march=rv64imac_zicbom -mabi=lp64 -mcmodel=medany \
  -mno-relax -nostdlib -ffreestanding -fno-pic -Os -fuse-ld=lld \
  -I"${SIB}" -T "${SIB}/dram.ld" "${SIB}/start.S" fw_fabric.c -o build/fw_fabric.elf
"${OBJCOPY}" -O binary build/fw_fabric.elf build/fw_fabric.bin
echo "built build/fw_fabric.bin ($(stat -c%s build/fw_fabric.bin) bytes)"
```

- [ ] **Step 2:** Write the CMakeLists mirroring `../l2cpu_noc_transfer/CMakeLists.txt` (host target links `TT::Metalium`; reuse the sibling `x280_boot` target rather than rebuilding it). Add `.gitignore` with `x280/build/`.

- [ ] **Step 3:** Write stub `fw_fabric.c` (MBOX defines from `fw.c`; `fw_main` sets `fw_state=0xA11FE`, loops incrementing heartbeat).

- [ ] **Step 4: Verify firmware compiles** (no device):
Run: `cd tt_metal/programming_examples/l2cpu_fabric_forward/x280 && ./build_fw.sh`
Expected: `built build/fw_fabric.bin (... bytes)`, entry `0x400030000000` via `llvm-readelf-20 -h build/fw_fabric.elf`.

- [ ] **Step 5: Verify host compiles** (no device):
Run: `cmake -DBUILD_PROGRAMMING_EXAMPLES=ON build && ninja -C build metal_example_l2cpu_fabric_forward` (restore `-DBUILD_PROGRAMMING_EXAMPLES=OFF` after if that was the tree's prior state).
Expected: binary at `build/programming_examples/metal_example_l2cpu_fabric_forward`.

- [ ] **Step 6: Commit** `git add ... && git commit -m "l2cpu_fabric_forward: scaffold example + firmware build wiring"`

---

## Task 2: Host fabric bring-up + link discovery (HW)

**Files:**
- Modify: `l2cpu_fabric_forward.cpp`

**Interfaces:**
- Produces: `struct Link { chip_id_t a, b; CoreCoord eth_a, eth_b; uint32_t link_idx; }` discovered A↔B; a helper `bring_up_fabric()` that returns two unit-mesh devices with `FABRIC_1D` enabled.

- [ ] **Step 1:** Implement, mirroring `tests/tt_metal/tt_fabric/fabric_data_movement/test_basic_fabric_smoke.cpp` and `tests/tt_metal/tt_fabric/common/fabric_fixture.hpp:80-129`:
```cpp
tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::FABRIC_1D);
auto meshes = distributed::MeshDevice::create_unit_meshes(ids /* 2 chip ids */);
auto& cp = MetalContext::instance().get_control_plane();
auto src = cp.get_fabric_node_id_from_physical_chip_id(ids[0]);
auto dst = cp.get_fabric_node_id_from_physical_chip_id(ids[1]);
auto chans = cp.get_forwarding_eth_chans_to_chip(src, dst);   // must be non-empty
```
Abort early (before any x280 boot) with a clear message if `ids.size() < 2`, the link is down, or `chans` is empty.

- [ ] **Step 2: Run on HW:** `./build/programming_examples/metal_example_l2cpu_fabric_forward`
Expected: prints the two chip ids, the eth channel/link between them, "fabric FABRIC_1D up"; exits 0. If it aborts "fabric not trained", STOP — Global Constraints hardware prerequisite is unmet (known risk on this box).

- [ ] **Step 3:** On teardown call `device->close()` then `SetFabricConfig(FabricConfig::DISABLED)`.

- [ ] **Step 4: Commit** `git commit -m "l2cpu_fabric_forward: host fabric bring-up + link discovery"`

---

## Task 3: Producer Tensix + receiver Tensix, host-verified without the x280 (HW)

This task builds the two Tensix kernels and proves the non-fabric halves in isolation: producer writes to LIM (readback by host), receiver moves a host-injected L1 blob to DRAM.

**Files:**
- Create: `kernels/producer.cpp`, `kernels/receiver.cpp`
- Modify: `l2cpu_fabric_forward.cpp`

**Interfaces:**
- `producer.cpp` args: `{l1_src, dram_src, size, lim_addr, l2cpu_x, l2cpu_y, mbox, seq, dest_noc_x, dest_noc_y, dest_l1_addr}` — reads DRAM→L1, writes L1→LIM (mirror `../l2cpu_noc_transfer/kernels/l2cpu_rw.cpp` steps 1–2), then writes the request block to `mbox+0x80`.
- `receiver.cpp` args: `{dst_l1_addr, dram_dst, size}` — reads its own L1 at `dst_l1_addr` (invalidate cache first), writes to DRAM.

- [ ] **Step 1:** Write `producer.cpp` (payload → LIM + request-block NOC write to the x280 mailbox, mirroring `x280_echo_poll.cpp:38-45`).
- [ ] **Step 2:** Write `receiver.cpp`.
- [ ] **Step 3:** Host: stage a known pattern to chip-A DRAM; run producer; read LIM back via a second Tensix read or host UMD; assert LIM == pattern.
- [ ] **Step 4:** Host: inject a blob into chip-B receiver L1 (UMD `write_to_device`); run receiver; read chip-B DRAM; assert == blob.
- [ ] **Step 5: Run on HW**, expect both asserts pass.
- [ ] **Step 6: Commit** `git commit -m "l2cpu_fabric_forward: producer/receiver Tensix kernels, verified standalone"`

---

## Task 4: Host boots x280 + delivers connection params (HW)

**Files:**
- Modify: `l2cpu_fabric_forward.cpp`

**Interfaces:**
- Produces: a populated conn-params block at `mbox+0x100` (the ten u32 values from Global Constraints mailbox layout), plus a booted x280 running the Task-1 stub firmware (heartbeat visible).

- [ ] **Step 1:** Boot the x280 on chip A by invoking the sibling boot path — reuse `x280_boot`'s logic (load `fw_fabric.bin` to GDDR, set reset vectors, PLL, release). Either shell out to the built `metal_example_l2cpu_x280_boot` or link its boot routine. Confirm heartbeat advances (mirror `x280_boot.cpp:222-235`).
- [ ] **Step 2:** Source the EDM connection parameters for the A→B link. **Primary (host-side):** locate the values `append_fabric_connection_rt_args(src, dst, link_idx, program, {core}, args)` would emit (see `tt_metal/api/tt-metalium/experimental/fabric/fabric.hpp:66`) and the device-init fabric connection table (`tensix_fabric_connections_l1_info_t` at `FABRIC_CONNECTIONS_BASE`, read per eth channel — adapter L112-128). Extract `edm_noc_x/y, edm_buffer_base_addr, num_buffers_per_channel, buffer_size_bytes, edm_connection_handshake_l1_addr, edm_worker_location_info_addr, edm_copy_of_wr_counter_addr, worker_free_slots_stream_id`. Pick a LIM address for `worker_free_slots_l1_addr` (where the EDM will push credits to the x280) and a `sender_channel_credits_stream_id`.
  - **If host-side extraction is impractical:** fall back to a one-time Tensix setup kernel that calls `build_from_args` and copies the resolved fields into LIM. Note which path was used in the README.
- [ ] **Step 3:** Write the ten u32 params into `mbox+0x100` via UMD before signaling the producer.
- [ ] **Step 4: Run on HW**, read the conn block back, print all ten values; sanity-check `edm_noc_x/y` equals the discovered eth core A and addresses are L1-range.
- [ ] **Step 5: Commit** `git commit -m "l2cpu_fabric_forward: host boots x280 and delivers EDM connection params"`

---

## Task 5: x280 firmware — EDM connection open (HW)

**Files:**
- Modify: `x280/fw_fabric.c`

**Interfaces:**
- Consumes: conn params at `mbox+0x100`.
- Produces: an opened connection (EDM sees the worker); `status.state=OPENED` at `mbox+0x180`.

This ports `WorkerToFabricEdmSender::open_start`/`open_finish` (`edm_fabric_worker_adapters.hpp:445-` and the `open()` at ~L558) to window MMIO. The x280 aims TLB window 0 at `(edm_noc_x, edm_noc_y)` (reuse `set_window()` from `fw.c`) and:

- [ ] **Step 1:** Read the EDM producer cursor / read counter: window-read `SenderChannelProducerCursor` from `edm_worker_location_info_addr + offsetof(...edm_read_counter)` (adapter L457-472) into LIM. (Read `EDMChannelWorkerLocationInfo` layout from `tt_metal/fabric/hw/inc/edm_fabric/...`; reproduce the field offsets in the firmware.)
- [ ] **Step 2:** Write the worker location info the EDM needs — the LIM address (`worker_free_slots_l1_addr`) where the EDM should push free-slot credits, into `edm_worker_location_info_addr + offsetof(...worker_semaphore_address)` (adapter L474-479) via window store.
- [ ] **Step 3:** Poke `edm_connection_handshake_l1_addr` on the EDM (window store of the connect value) to signal "connected".
- [ ] **Step 4:** Bounded-spin until the EDM acks the open (per `open_finish`); on timeout write `fault_code=OPEN_TIMEOUT` and park. Set `status.state=OPENED`.
- [ ] **Step 5: Run on HW:** boot this firmware, run the host through Task 4, read `mbox+0x180`; expect `state=OPENED`, `fault_code=0`.
- [ ] **Step 6: Commit** `git commit -m "l2cpu_fabric_forward: x280 firmware opens EDM connection over window MMIO"`

---

## Task 6: x280 firmware — send one packet (HW, first end-to-end)

**Files:**
- Modify: `x280/fw_fabric.c`

**Interfaces:**
- Consumes: request block at `mbox+0x80` (payload in LIM, dest), opened connection from Task 5.
- Produces: one fabric packet delivered to `dest_l1_addr` on chip B.

Ports the send sequence in `tt_metal/fabric/hw/inc/linear/api.h:81-94` (`fabric_unicast_noc_unicast_write`): `to_chip_unicast(num_hops=1)` + `to_noc_unicast_write` → wait for slot → write payload → write header → credit. Translated to window MMIO:

- [ ] **Step 1:** Wait for a free slot: bounded-spin reading `worker_free_slots_l1_addr` in LIM (the EDM pushes the count here — plain inbound write, stage-1 proven) until `> 0`. Record `slots_seen`.
- [ ] **Step 2:** Build the packet header in LIM. Reproduce `PACKET_HEADER_TYPE` (from `tt_metal/fabric/hw/inc/.../packet_header.hpp`): set chip-unicast routing with `num_hops=1` and a NOC-unicast-write command targeting `get_noc_addr(dest_noc_x, dest_noc_y, dest_l1_addr)` with `size`. Copy the exact field encodings from `to_chip_unicast` / `to_noc_unicast_write`.
- [ ] **Step 3:** Window-store the payload from LIM into the EDM slot at `edm_buffer_base_addr + slot*buffer_size_bytes + sizeof(PACKET_HEADER_TYPE)`, then window-store the header to `edm_buffer_base_addr + slot*buffer_size_bytes` (payload before header — the header's valid bit publishes the slot; mirror `send_current_slot_non_blocking` adapter L352-367).
- [ ] **Step 4:** Update the EDM credit: window-store `pack_value_for_inc_on_write_stream_reg_write(-1)` (= `-1 << REMOTE_DEST_BUF_WORDS_FREE_INC`) to `get_stream_reg_write_addr(sender_channel_credits_stream_id)` = `STREAM_REG_ADDR(stream_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX)` on the EDM tile (`fabric_stream_regs.hpp:88-93`; resolve `STREAM_REG_ADDR` from the NOC overlay headers and reproduce the constant in firmware). Advance the local write counter. Bump `credit_writes`.
- [ ] **Step 5:** Run the receiver (Task 3) on chip B into `dest_l1_addr`.
- [ ] **Step 6: Run full end-to-end on HW:** host stages pattern → producer → x280 send → receiver → chip-B DRAM. Assert chip-B DRAM == pattern.
Expected: match. If the receiver never sees data, isolate with Task 8 (stream-reg probe) before deeper debugging.
- [ ] **Step 7: Commit** `git commit -m "l2cpu_fabric_forward: x280 sends one fabric packet, end-to-end delivery"`

---

## Task 7: Close/teardown + fault hardening (HW)

**Files:**
- Modify: `x280/fw_fabric.c`, `l2cpu_fabric_forward.cpp`

- [ ] **Step 1:** After send, port `close()` (adapter L623): window-poke the teardown/handshake so the EDM releases the connection cleanly.
- [ ] **Step 2:** Audit every spin loop in `fw_fabric.c` for a bounded timeout → `fault_code` + park (Global Constraint). Enumerate fault codes in the mailbox layout comment.
- [ ] **Step 3:** Host: on any nonzero `fault_code`, print it and the full mailbox diagnostics (mirror `x280_echo_test.cpp:106-120`).
- [ ] **Step 4: Run on HW**, confirm clean close (re-running the host without a chip reset still works, i.e. the connection wasn't left dangling — note the one-shot reset caveat still applies to re-boots).
- [ ] **Step 5: Commit** `git commit -m "l2cpu_fabric_forward: connection teardown + fault hardening"`

---

## Task 8: Fallback — standalone stream-register probe (HW, only if Task 6 credits misbehave)

**Files:**
- Create: `x280/fw_probe.c`

- [ ] **Step 1:** Minimal firmware: aim window at eth core A, write `pack_value_for_inc_on_write_stream_reg_write(-1)` to `get_stream_reg_write_addr(stream_id)`, then window-read the stream reg's value back (`get_stream_reg_read_addr`) into the mailbox.
- [ ] **Step 2: Run on HW:** confirm the register reflects the increment. This isolates "does an x280 window store reach a NOC stream register with inc-on-write semantics" from the full protocol.
- [ ] **Step 3:** If it fails: the stream-reg-through-window primitive is unsupported → escalate to the spec's raw-eth Plan B (`eth_send_bytes`, atomics-free) as a separate design change. Document the finding in the README and a memory note.
- [ ] **Step 4: Commit** `git commit -m "l2cpu_fabric_forward: standalone x280 stream-register probe"`

---

## Task 9: README + final end-to-end verification

**Files:**
- Create: `tt_metal/programming_examples/l2cpu_fabric_forward/README.md`

- [ ] **Step 1:** Document: prerequisites (two `FABRIC_1D` chips), build (`build_fw.sh` + cmake), run order (boot x280 → host), the mailbox layout, the data flow diagram from the spec, and the fault codes.
- [ ] **Step 2: Full run on HW** from a clean `tt-smi -r`; capture the pass output.
- [ ] **Step 3: Commit** `git commit -m "l2cpu_fabric_forward: README + verified end-to-end"`
- [ ] **Step 4:** Update memory `project_l2cpu_noc_transfer` with the outcome (worked / stream-reg failed → raw-eth).

---

## Self-Review

**Spec coverage:** Goal → Tasks 1–9. x280 worker protocol (open/send/close) → Tasks 5–7. Connection-param sourcing (host-side + Tensix fallback) → Task 4. Standard EDM reuse / `FABRIC_1D` → Task 2. Producer/receiver → Task 3. Single-packet assumption → Task 6 Step 3. Error handling / bounded spins → Task 7. Stream-reg risk + raw-eth Plan B → Task 8. Testing (2-chip end-to-end) → Task 6/9. README → Task 9. All spec sections mapped.

**Placeholder scan:** Firmware Tasks 5–6 instruct the implementer to reproduce exact struct offsets / macros from named source files (`EDMChannelWorkerLocationInfo`, `PACKET_HEADER_TYPE`, `STREAM_REG_ADDR`) rather than inlining bytes — this is deliberate: those layouts must be mirrored from live headers, not fabricated in a plan, and each step names the exact file and function to copy. Verified constants (packed credit value, stream-reg index, header ops, build recipe, mailbox layout) are given concretely.

**Type consistency:** Mailbox offsets and the ten conn-param fields are named identically across Tasks 4/5/6. `worker_free_slots_l1_addr` (EDM→x280 credit sink) and `sender_channel_credits_stream_id` (x280→EDM credit) are used consistently. `num_hops=1` throughout.

**Note on TDD:** this is firmware + multi-chip hardware; there is no host unit-test harness for the device path. "Tests" are hardware end-to-end runs (compile checks where a device isn't needed). Each task still ends in an independently verifiable deliverable.
