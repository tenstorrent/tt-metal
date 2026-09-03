<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Design: x280 (L2CPU) as a fabric worker — kernel → L2CPU → another device

**Branch:** `vsureshTT/L2CPU`
**Date:** 2026-09-03
**Status:** design approved (direction), spec under review

## Goal

A Tensix kernel on **Device A** hands a payload to the on-die **L2CPU** (x280),
and the x280 forwards that payload **off-chip to Device B** over one ethernet
link, where a Tensix kernel receives it. The x280 does the off-chip send itself,
acting as a **fabric worker** feeding the standard, unmodified EDM router — the
existing fabric handshake stays intact. There is **no forwarder Tensix** in the
data path.

This is the end-to-end (two-chip) target directly; no separate single-chip
milestone. A tiny stream-register probe is the fallback if credits misbehave.

## Background — what the branch already proves

Three stages exist on this branch (see `tt_metal/programming_examples/l2cpu_noc_transfer/README.md`):

- **Stage 1** — Tensix ↔ L2CPU LIM round-trip. Established: inbound **plain**
  reads/writes and inline-dw writes to the L2CPU work; inbound **NOC atomics**
  to the L2CPU hang (unsupported).
- **Stage 2** — x280 hart booted, runs firmware, and does an **x280-initiated
  NOC write** back into a Tensix via a **TLB window** (config regs at
  `0x2000_0000`, aperture at `0x0430_0000_00`). The x280 has no NOC command
  interface; window loads/stores are its only NOC egress.
- **Stage 3** — x280 TLB-window store bandwidth benchmark.

Firmware builds with clang (`x280/build_fw.sh`), links `.text` at
`0x4000_3000_0000` (cached GDDR), booted by `x280/x280_boot` (raw UMD; one-shot
reset release per chip).

## Key feasibility result (why Option 1 works)

The `WorkerToFabricEdmSender` worker→EDM protocol
(`tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp`) uses **zero
NOC atomics** (`grep noc_semaphore_inc` → none). Every step maps to a primitive
the L2CPU has already been shown to do:

| Protocol step | Mechanism | x280 capability | Proven? |
|---|---|---|---|
| Write payload+header into EDM buffer slot | plain NOC writes → eth-core L1 | outbound window stores | stage-1 style ✅ |
| Decrement EDM free-slots (credit) | plain write of packed value to a **stream register** on the EDM tile (`get_stream_reg_write_addr`, `pack_value_for_inc_on_write_stream_reg_write(-1)`, adapter L208/L380–384) | outbound window store | **unproven — experiment #1 primitive** |
| Learn EDM free-slots (EDM→worker) | in the **worker** config (`IS_WORKER = !I_USE_STREAM_REG_FOR_CREDIT_RECEIVE`), EDM writes the count to a **plain L1 address on the worker's tile** (adapter L203–207) | inbound plain write to LIM | **stage-1 proved** ✅ |
| Read EDM producer cursor at open | plain NOC reads ← eth-core L1 | window reads | stage-2 proved ✅ |

The EDM→x280 direction — where "no inbound atomics" could have bitten — is a
**plain write to a plain L1 address** in the worker configuration, exactly what
stage 1 validated. The only genuinely unproven primitive is the **x280→EDM
stream-register credit write**.

## The x280's input contract (connection parameters)

From `build_from_args` VC2 runtime-arg path (adapter L133–147), a worker needs:
`edm_noc_x`, `edm_noc_y`, `edm_buffer_base_addr`, `num_buffers_per_channel`,
`buffer_size_bytes`, `edm_connection_handshake_l1_addr`,
`edm_worker_location_info_addr`, `edm_copy_of_wr_counter_addr`, the
**sender-channel credits stream id**, and a worker-side **L1 address** where the
EDM pushes free-slot credits. These are the values the host fabric API normally
bakes into a Tensix kernel's runtime args; here they are delivered to the x280
firmware instead (see Component 5).

## Architecture

```
CHIP A                                                      CHIP B
──────                                                      ──────
[Producer Tensix]  payload → L2CPU LIM  +  mailbox poke     [Receiver Tensix]
        │                                                     reads delivered payload
        ▼                                                     → DRAM → host verifies
[x280 fabric-worker firmware]  (new)                               ▲
   • open:  handshake with local EDM                              │
   • send:  payload+header → EDM slot; credit stream-reg write    │
   • close: teardown                                              │
        │                                                         │
        ▼   (standard, unmodified EDM)                            │
[EDM router, eth core A] ═══ FABRIC_1D, num_hops=1 ═══▶ [EDM router, eth core B] ─▶ receiver L1
```

### Components

1. **Producer Tensix kernel (chip A)** — reads the host payload from DRAM into
   L1, writes it into L2CPU LIM (stage-1 write path), then pokes the x280
   mailbox with `{ready, payload_lim_addr, size, dest_noc_addr}`. Never touches
   the fabric — not a forwarder.

2. **x280 fabric-worker firmware (chip A)** — *new*, extends stage-2 firmware.
   Reimplements the worker→EDM push using TLB-window MMIO only:
   - **open**: read the EDM producer cursor / read counter (window reads);
     write the x280's worker-location-info — including the LIM address the EDM
     should push free-slot credits to — into `edm_worker_location_info_addr`
     (window writes); poke `edm_connection_handshake_l1_addr` to signal
     connected.
   - **send**: poll the local LIM free-slot counter (window read of own tile)
     until a slot is free; construct a `PACKET_HEADER_TYPE`
     (`to_chip_unicast(1)` + `to_noc_unicast_write(dest_noc_addr, size)`);
     window-store payload then header into
     `edm_buffer_base_addr + slot*buffer_size_bytes` on eth core A; window-store
     the packed `-1` credit value to the sender-channel credits stream register;
     advance the local write counter.
   - **close**: poke teardown/handshake.

3. **Standard EDM routers** — unmodified, on eth cores of both chips. Enabled by
   `SetFabricConfig(FabricConfig::FABRIC_1D)` before device creation. This is the
   "fabric handshake intact" requirement.

4. **Receiver Tensix kernel (chip B)** — the EDM delivers the payload into a
   known L1 address (the `dest_noc_addr` the header carried); the kernel moves
   it to chip-B DRAM for host readback.

5. **Host program** — orchestrates:
   `SetFabricConfig(FABRIC_1D)` → `MeshDevice::create_unit_meshes(ids)` →
   discover the A↔B link (control-plane fabric node ids + forwarding link
   indices) → boot the x280 on chip A (`x280_boot boot fabric_worker.bin`) →
   **source the EDM connection params and write them into the x280 mailbox via
   UMD** → stage payload to chip-A DRAM → launch producer (chip A) + receiver
   (chip B) → read back chip-B DRAM → verify → `SetFabricConfig(DISABLED)`.

### Connection-parameter sourcing (decision)

**Chosen: host-side via UMD.** The host extracts the same values
`append_fabric_connection_rt_args` would emit for a Tensix worker and writes them
into the x280 LIM mailbox, keeping Tensix out of both data path and setup.
**Fallback:** a one-time Tensix setup kernel that writes the params into LIM at
init (still not a data-path forwarder) — used only if host-side extraction of
these values proves impractical against the current fabric API surface.

### Simplifying assumptions (first cut)

- **Single packet**: payload ≤ one EDM buffer slot (`buffer_size_bytes`), so the
  x280 issues exactly one slot write + one credit update. Multi-packet chunking
  is a later extension.
- One x280 worker, one ethernet link, one direction (A → B), one packet in
  flight at a time.

## Data flow

host → chip-A DRAM → producer Tensix L1 → **L2CPU LIM** → (x280 reads LIM) →
**EDM buffer slot, eth core A** → eth link → eth core B → **receiver Tensix L1**
→ chip-B DRAM → host verify.

## Error handling

- **x280 hart is one-shot per chip reset**: firmware must not wedge on a failed
  open. Bounded spin loops with timeouts on every EDM poll (open cursor read,
  free-slot wait); on timeout, write a fault code to the mailbox and park with a
  heartbeat (stage-2 pattern) so the host can diagnose without a hang. Recovery
  is `tt-smi -r`.
- **Fabric not trained**: host checks the A↔B link is up before boot; abort
  early (before consuming the x280 one-shot reset) if not. (Known risk on this
  box — see Risks.)
- **Credit/stream-reg failure**: detected as the free-slot wait timing out or the
  receiver never seeing data; isolate with the standalone stream-reg probe.
- **Host readback mismatch**: report first N mismatching words (stage-1 pattern).

## Testing strategy

- **Primary: two-chip end-to-end.** Host stages a known pattern, runs the full
  path, verifies chip-B DRAM equals the input. Closest references to copy:
  `tests/tt_metal/tt_fabric/fabric_data_movement/test_basic_fabric_smoke.cpp`
  (host, 2 devices, link discovery, `FABRIC_1D`) and
  `tt_metal/fabric/hw/inc/edm_fabric/.../edm_fabric_writer.cpp` +
  `tt_metal/fabric/hw/inc/linear/api.h` (`fabric_unicast_noc_unicast_write`, the
  reference worker send sequence the x280 mirrors).
- **Fallback probe (only if credits misbehave):** a ~10-line firmware that does
  one x280 stream-register write and reads it back, to isolate the single
  unproven primitive without the full path.
- Reuse stage-1/stage-2 verification patterns (mailbox diagnostics, first-N
  mismatch reporting).

## Risks

1. **Stream-register write through the x280 window** (highest) — the one
   unproven primitive. Mitigation: fallback probe; if it fails, the raw-eth path
   (x280 → minimal erisc kernel via `eth_send_bytes`, atomics-free, all plain L1
   + TXQ pokes) is a documented Plan B, though it bypasses the fabric.
2. **Fabric training on this box** — prior notes record this box's Blackhole
   showing a single chip and 1D sublines not training fabric. Two
   fabric-connected BH chips that actually train `FABRIC_1D` are a hard
   prerequisite; confirm before building.
3. **Connection-param sourcing** — the fabric API targets Tensix kernels;
   extracting the raw param set host-side may need new plumbing. Fallback: Tensix
   setup kernel.
4. **Packet-header / slot-layout fidelity** — the x280 must byte-match the EDM's
   expected header and slot layout. Mitigation: mirror `linear/api.h` exactly;
   compare against a Tensix-worker capture.

## Out of scope (YAGNI)

- Fabric 2D / mesh routing, multi-hop, the fabric mux (`FabricTensixConfig::MUX`).
- Multiple concurrent workers, multiple links, TXQ1.
- x280 hosting the full mux (the longer-term endgame; not this example).
- Performance tuning — correctness first.
