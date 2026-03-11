# Phase 2: Silicon Data-Transfer Validation - Research

**Researched:** 2026-03-11
**Domain:** TT-Fabric silicon test infrastructure (host-side C++ test runners + device kernels)
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- Rewrite unicast_runner.cpp and multicast_runner.cpp to use `BaseFabricFixture` from `fabric_fixture.hpp`
- Drop dependency on deleted `test_common.hpp` -- define minimal types inline or in a new lightweight header
- Test functions take `BaseFabricFixture*` (not `MeshDeviceFixtureBase*`)
- Follow the `addrgen_write/unicast_runner.cpp` pattern exactly: allocate buffers, fill with pattern, dispatch kernel, wait, compare
- Reuse existing addrgen receiver kernel (`rx_addrgen.cpp`) for completion signaling -- it's protocol-agnostic (waits on semaphore, validates buffer)
- `TEST_F` cases registered in `test_auto_packetization.cpp` (not in runner files)
- Runner files added to `sources.cmake` for `fabric_unit_tests` binary
- Run both `Fabric2DFixture` (mesh APIs) and `Fabric1DFixture` (linear APIs) silicon tests
- Multiple sizes per test: size < MAX (single packet), size = 2*MAX+512 (loop + remainder), size = 1 MiB (stress)
- FABRIC_MAX_PACKET_SIZE is a runtime value -- sizes computed at test time
- Byte-for-byte data comparison: fill source with incrementing byte pattern, compare destination buffer exactly
- All 9 families tested on silicon (chunking families + scatter passthroughs + sparse multicast)
- Multi-hop multicast to all reachable neighbors on the 4-device system
- Test MeshMcastRange with non-zero hops in all active directions

### Claude's Discretion

- Exact buffer allocation strategy (L1 vs DRAM)
- Test pattern (incrementing bytes, random, etc.)
- How to parameterize tests across the 9 families (GTest parameterized vs separate TEST_F per family)
- Whether to create new kernel files per family or extend existing kernels with runtime-arg-based dispatch

### Deferred Ideas (OUT OF SCOPE)

- Connection manager variant silicon tests (multi-connection breadth-first verification) -- would need multi-connection test setup
- Performance benchmarking of auto-packetization overhead vs manual chunking
</user_constraints>

---

## Summary

This phase converts the existing (broken) auto-packetization host-side runners from `MeshDeviceFixtureBase` to `BaseFabricFixture`, adds silicon data-validation TEST_F cases for all 9 API wrapper families, and verifies byte-for-byte correctness at multiple payload sizes. The key challenge is that `BaseFabricFixture` creates per-chip `MeshDevice` objects (via `create_unit_meshes`) rather than a single shared `MeshDevice` spanning all chips. This fundamentally changes how buffers are allocated, data is written/read, and programs are dispatched compared to the addrgen test pattern which uses `MeshDeviceFixtureBase` with a single multi-chip mesh.

The existing device kernels (`unicast_tx_writer_raw.cpp`, `multicast_tx_writer_raw.cpp`, `linear_unicast_tx_writer_raw.cpp`) are already functional for the basic unicast/multicast write families. New device kernels are needed for the remaining 6 families (scatter, fused atomic inc, fused scatter+atomic inc, sparse multicast). The runner infrastructure can be shared across all families by parameterizing on family type.

**Primary recommendation:** Fix the two existing runners to use `BaseFabricFixture` APIs (per-chip MeshDevice, direct L1/DRAM buffer allocation, single-chip program dispatch), add kernel variants for all 9 families, and register parameterized TEST_F cases in `test_auto_packetization.cpp`. Use `get_tt_fabric_max_payload_size_bytes()` on the host side to compute test payload sizes.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Google Test | (bundled) | TEST_F / TEST_P test registration and assertions | Standard TT-Metal test framework |
| BaseFabricFixture | N/A | Device setup with fabric enabled, per-chip MeshDevice | Mandated by CONTEXT.md; used by all fabric_data_movement tests |
| Fabric2DFixture / Fabric1DFixture | N/A | Concrete fixture subclasses for 2D mesh and 1D linear fabric configs | Pre-existing, tested, provides `SetUpTestSuite`/`TearDownTestSuite` |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `tt::tt_metal::distributed::*` | N/A | `MeshBuffer`, `MeshWorkload`, `WriteShard`, `ReadShard`, `EnqueueMeshWorkload` | Buffer I/O and program dispatch on MeshDevice |
| `tt::tt_fabric::append_fabric_connection_rt_args` | N/A | Pack fabric sender runtime args for device kernels | Every sender kernel needs fabric connection setup |
| `tt::tt_metal::CreateGlobalSemaphore` | N/A | Create L1 semaphore visible across chips | Completion signaling from sender to receiver |
| `rx_addrgen.cpp` | N/A | Protocol-agnostic receiver kernel: waits on semaphore, resets it | Reusable for all 9 families |

---

## Architecture Patterns

### Recommended Project Structure

```
tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/
  kernels/
    unicast_tx_writer_raw.cpp          # Existing: mesh unicast write
    multicast_tx_writer_raw.cpp        # Existing: mesh multicast write
    linear_unicast_tx_writer_raw.cpp   # Existing: linear unicast write
    scatter_tx_writer_raw.cpp          # NEW: unicast scatter write (mesh + linear via defines)
    fused_atomic_inc_tx_writer_raw.cpp # NEW: unicast fused write+atomic_inc
    mcast_scatter_tx_writer_raw.cpp    # NEW: multicast scatter write
    mcast_fused_tx_writer_raw.cpp      # NEW: multicast fused write+atomic_inc
    fused_scatter_atomic_inc_tx_writer_raw.cpp  # NEW: unicast fused scatter+atomic_inc
    mcast_fused_scatter_atomic_inc_tx_writer_raw.cpp  # NEW: multicast fused scatter+atomic_inc
    sparse_mcast_tx_writer_raw.cpp     # NEW: sparse multicast (linear-only)
  unicast_runner.cpp                   # REWRITE: BaseFabricFixture, handles all unicast families
  multicast_runner.cpp                 # REWRITE: BaseFabricFixture, handles all multicast families
test_auto_packetization.cpp            # ADD: TEST_F silicon data tests (existing compile-only tests stay)
```

### Pattern 1: BaseFabricFixture Buffer and Dispatch (CRITICAL DIFFERENCE from addrgen tests)

**What:** `BaseFabricFixture` creates per-chip `MeshDevice` objects via `create_unit_meshes`, NOT a single multi-chip mesh. This changes the entire I/O pattern.

**When to use:** All tests in this phase.

**Key differences from addrgen `MeshDeviceFixtureBase` pattern:**

| Concern | MeshDeviceFixtureBase (addrgen) | BaseFabricFixture (this phase) |
|---------|-------------------------------|-------------------------------|
| Mesh device | Single `MeshDevice` spanning all chips | Per-chip `MeshDevice` from `devices_map_` |
| Buffer allocation | `MeshBuffer::create(rcfg, local, mesh.get())` on shared mesh | `tt::tt_metal::CreateBuffer(device, config)` on individual device |
| Data write | `Dist::WriteShard(mcq, buf, data, coord, true)` | `tt::tt_metal::EnqueueWriteBuffer(cq, buf, data, true)` |
| Data read | `Dist::ReadShard(mcq, rx, buf, coord, true)` | `tt::tt_metal::EnqueueReadBuffer(cq, buf, rx, true)` |
| Program dispatch | `Dist::EnqueueMeshWorkload(mcq, workload, blocking)` | `fixture->RunProgramNonblocking(device, prog)` + `fixture->WaitForSingleProgramDone(device, prog)` |
| Device lookup | `view.impl().get_device(coord)->id()` | `fixture->get_device(chip_id)` or `fixture->get_devices()[i]` |
| Semaphore | `CreateGlobalSemaphore(mesh.get(), ...)` | `CreateGlobalSemaphore(per_chip_mesh.get(), ...)` |

**Example (unicast runner with BaseFabricFixture):**
```cpp
// Source: Derived from existing fabric_data_movement tests using BaseFabricFixture
void run_raw_unicast_test(BaseFabricFixture* fixture, uint32_t payload_size) {
    const auto& cp = tt::tt_metal::MetalContext::instance().get_control_plane();

    // Get per-chip mesh devices
    auto src_mesh = fixture->get_device(src_phys);  // MeshDevice for one chip
    auto dst_mesh = fixture->get_device(dst_phys);
    auto* src_dev = src_mesh->get_devices()[0];
    auto* dst_dev = dst_mesh->get_devices()[0];

    // Allocate buffers on individual devices
    auto src_buf = tt::tt_metal::CreateBuffer(src_dev, {payload_size, payload_size, BufferType::DRAM});
    auto dst_buf = tt::tt_metal::CreateBuffer(dst_dev, {payload_size, payload_size, BufferType::DRAM});

    // Write data directly to device
    auto& src_cq = src_mesh->mesh_command_queue();
    tt::tt_metal::EnqueueWriteBuffer(src_cq, src_buf, tx_data, true);
    tt::tt_metal::EnqueueWriteBuffer(dst_cq, dst_buf, zeros, true);

    // Create programs for individual devices (not mesh workloads)
    // ... (see full pattern below)
}
```

**However** - looking more carefully at how existing `BaseFabricFixture` tests actually work (e.g., the `RunTestUnicastRaw`, `RunTestUnicastConnAPI` functions declared in `fabric_fixture.hpp`), they use the `BaseFabricFixture` primitives directly. The existing broken auto-packetization runners use `MeshDeviceFixtureBase` because they were copied from the addrgen test pattern. The rewrite must align with `BaseFabricFixture` patterns.

### Pattern 2: Existing BaseFabricFixture Test Flow (from test_basic_fabric_apis.cpp)

The proven flow for `BaseFabricFixture` silicon tests is:

1. `fixture->get_devices()` to get list of per-chip MeshDevice objects
2. Use `generate_worker_mem_map()` for L1 address layout
3. Allocate L1 buffers at known offsets (source_l1_buffer_address, target_address)
4. Write test data directly via `tt::tt_metal::detail::WriteToBuffer`
5. Create sender program with fabric sender kernel
6. Create receiver program with wait kernel
7. `fixture->RunProgramNonblocking(dst_mesh, rx_prog)` (non-blocking)
8. `fixture->RunProgramNonblocking(src_mesh, tx_prog)` (non-blocking)
9. `fixture->WaitForSingleProgramDone(src_mesh, tx_prog)`
10. `fixture->WaitForSingleProgramDone(dst_mesh, rx_prog)`
11. Read back via `tt::tt_metal::detail::ReadFromBuffer` and compare

### Pattern 3: Receiver Kernel Reuse

**What:** `rx_addrgen.cpp` is a minimal kernel that waits for a semaphore to reach `expected_value`, then resets it. It takes 2 runtime args: `sem_addr` and `expected_value`.

**When to use:** Every test in this phase. For non-fused-atomic-inc families, set `expected_value=1` (a separate atomic_inc is sent after all data). For fused families, set `expected_value=1` (the auto-packetizing wrapper fires atomic_inc only on the final chunk).

**Source:** `tests/tt_metal/tt_fabric/fabric_data_movement/addrgen_write/kernels/rx_addrgen.cpp`

```cpp
void kernel_main() {
    const uint32_t sem_addr = get_arg_val<uint32_t>(0);
    const uint32_t expected_value = get_arg_val<uint32_t>(1);
    volatile tt_l1_ptr uint32_t* sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
    noc_semaphore_wait(sem_ptr, expected_value);
    noc_semaphore_set(sem_ptr, 0);  // Reset for next test
}
```

### Pattern 4: Kernel Family Selection Strategy (Claude's Discretion)

**Recommendation: Separate kernel files per family group, NOT runtime-arg dispatch.**

Rationale:
1. The existing device kernel code uses different API calls per family (e.g., `fabric_unicast_noc_unicast_write` vs `fabric_unicast_noc_scatter_write` vs `fabric_unicast_noc_fused_unicast_with_atomic_inc`). These have different parameter types (`NocUnicastCommandHeader` vs `NocUnicastScatterCommandHeader` vs `NocUnicastAtomicIncFusedCommandHeader`).
2. Runtime-arg dispatch would require `#ifdef` or switch-case in the kernel, bloating code size and making the test less representative of real usage.
3. Phase 1 already established the separate-kernel pattern with `unicast_tx_writer_raw.cpp` and `multicast_tx_writer_raw.cpp`.
4. The host-side runner can still be shared -- it selects which kernel file to compile based on the family being tested.

**Grouping:**
- Unicast families (4): basic write, scatter write, fused atomic inc, fused scatter+atomic inc
- Multicast families (4): basic write, scatter write, fused atomic inc, fused scatter+atomic inc
- Sparse multicast (1): linear-only

### Pattern 5: Host-Side Payload Size Computation

**What:** `FABRIC_MAX_PACKET_SIZE` is device-side only (reads from L1 config). On the host, use `get_tt_fabric_max_payload_size_bytes()` from `<tt-metalium/experimental/fabric/fabric.hpp>`.

```cpp
#include <tt-metalium/experimental/fabric/fabric.hpp>

// In test setup:
const uint32_t max_payload = (uint32_t)tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();
const uint32_t size_small = max_payload / 2;                  // < MAX, single packet
const uint32_t size_medium = 2 * max_payload + 512;           // exercises chunking loop + remainder
const uint32_t size_large = 1u << 20;                         // 1 MiB stress test
```

**Note:** `get_tt_fabric_max_payload_size_bytes()` returns the max PAYLOAD size (excludes header). This matches `FABRIC_MAX_PACKET_SIZE` on the device side, which is also the max payload size.

### Pattern 6: Multicast Direction Setup (MeshMcastRange)

For multicast on 2D mesh, each direction needs its own sender and packet header. The existing `multicast_tx_writer_raw.cpp` kernel already implements this pattern:

```cpp
// Per-direction senders: W, E, N, S
// MeshMcastRange{e_hops, w_hops, n_hops, s_hops}
// W direction: MeshMcastRange{0, w_hops, 0, 0}
// E direction: MeshMcastRange{e_hops, 0, 0, 0}
// N direction: MeshMcastRange{e_hops, w_hops, n_hops, 0}  (includes E/W coverage)
// S direction: MeshMcastRange{e_hops, w_hops, 0, s_hops}  (includes E/W coverage)
```

The completion signaling uses `fabric_set_mcast_route` + `to_noc_unicast_atomic_inc` + manual `send_payload_flush_non_blocking_from_address` (not the auto-packetizing API).

### Pattern 7: Scatter Write Device Kernel Pattern

**What:** Scatter writes send data to TWO destinations per packet. The `NocUnicastScatterCommandHeader` carries two NOC addresses.

**For auto-packetization scatter tests:** The auto-packetizing scatter wrappers are passthroughs (no chunking for raw scatter -- per Phase 1 decisions). So scatter tests should use payloads that fit in a single packet (size <= MAX_PACKET_SIZE). The test validates the passthrough works correctly, not chunking.

```cpp
// Scatter kernel pattern:
const uint64_t dst_noc_addr0 = safe_get_noc_addr(rx_noc_x, rx_noc_y, dst_base_addr, 0);
const uint64_t dst_noc_addr1 = safe_get_noc_addr(rx_noc_x, rx_noc_y, dst_base_addr + scatter_offset, 0);

fabric_unicast_noc_scatter_write(
    &sender, packet_header,
    dst_dev_id, dst_mesh_id,
    src_l1_addr, scatter_size,
    NocUnicastScatterCommandHeader{dst_noc_addr0, dst_noc_addr1});
```

**Scatter payload semantics:** The source buffer is split in half -- first half goes to `noc_addr0`, second half goes to `noc_addr1`. Each half must be <= MAX_PACKET_SIZE/2.

### Pattern 8: Fused Atomic Inc Device Kernel Pattern

**What:** Fused write+atomic_inc sends data AND increments a semaphore in a single fabric operation. For auto-packetized large payloads, intermediate chunks are regular writes and only the final chunk is fused.

```cpp
// The auto-packetizing wrapper handles the split automatically:
fabric_unicast_noc_fused_unicast_with_atomic_inc(
    &sender, packet_header,
    dst_dev_id, dst_mesh_id,
    src_l1_addr, total_size,
    NocUnicastAtomicIncFusedCommandHeader{dst_noc_addr, sem_noc_addr, /*inc=*/1, /*width_bits=*/32});

// NO separate atomic_inc needed -- the fused wrapper handles it.
// Receiver waits for sem == 1 (fires once on final chunk).
```

### Anti-Patterns to Avoid

- **Using MeshDeviceFixtureBase with BaseFabricFixture tests:** These fixtures are incompatible. BaseFabricFixture uses per-chip MeshDevices; MeshDeviceFixtureBase uses a single multi-chip MeshDevice. Mixing them causes crashes.
- **Allocating MeshBuffer on BaseFabricFixture:** MeshBuffer requires a multi-chip MeshDevice. Use `CreateBuffer` on individual devices instead.
- **Sending atomic_inc after fused write APIs:** The auto-packetizing fused wrappers already send the atomic_inc on the final chunk. Sending another would double-increment the semaphore.
- **Testing scatter with payloads > MAX:** Raw scatter wrappers are passthrough (no chunking). Large scatter payloads will fail or corrupt data.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Completion signaling | Custom semaphore polling kernel | `rx_addrgen.cpp` kernel | Protocol-agnostic, proven on silicon, handles reset |
| Fabric sender setup | Manual L1 address packing | `WorkerToFabricEdmSender::build_from_args` + `append_fabric_connection_rt_args` | Complex multi-field packing with device-specific offsets |
| NOC address computation (device) | Manual coordinate encoding | `safe_get_noc_addr(noc_x, noc_y, addr, noc_index)` | Handles NOC coordinate translation correctly |
| Max packet size (host) | Hardcoded constant | `get_tt_fabric_max_payload_size_bytes()` | Runtime-configured, varies by system |
| Multicast direction routing | Manual hop computation | Use `get_forwarding_link_indices` + bounding box pattern from multicast_runner | Proven pattern handles all topology shapes |
| Test data pattern | Random generator | `0xA5A50000u + i` incrementing word pattern | Deterministic, easy to debug mismatches, used throughout codebase |

---

## Common Pitfalls

### Pitfall 1: BaseFabricFixture vs MeshDeviceFixtureBase API mismatch

**What goes wrong:** The existing broken runners call `fixture->get_mesh_device()` which returns a multi-chip MeshDevice. BaseFabricFixture has `get_devices()` (vector of per-chip MeshDevices) and `get_device(ChipId)` instead.

**Why it happens:** The original runners were scaffolded from addrgen test code which uses MeshDeviceFixtureBase.

**How to avoid:** Use ONLY BaseFabricFixture APIs:
- `fixture->get_devices()` for device enumeration
- `fixture->get_device(chip_id)` for specific device access
- `fixture->RunProgramNonblocking(mesh, prog)` for dispatch
- `fixture->WaitForSingleProgramDone(mesh, prog)` for completion
- Direct buffer I/O via `EnqueueWriteBuffer`/`EnqueueReadBuffer` on per-chip command queues

**Warning signs:** Compilation errors about `get_mesh_device()` not being a member of `BaseFabricFixture`.

### Pitfall 2: Semaphore not reset between test runs

**What goes wrong:** After a successful test, the semaphore is at 1. The next test run sees it immediately and passes without waiting for data.

**Why it happens:** `GlobalSemaphore` persists across TEST_F cases in the same fixture.

**How to avoid:** `rx_addrgen.cpp` already resets the semaphore to 0 after the wait completes. Verify the receiver program runs to completion before starting the next test. Also consider using `static` semaphore carefully -- the addrgen tests use `static std::optional<GlobalSemaphore>` which persists across tests.

### Pitfall 3: Buffer alignment requirements

**What goes wrong:** DRAM buffers on Blackhole require 32-byte alignment. L1 buffers require 16-byte alignment. Misaligned buffers cause silent data corruption or hangs.

**Why it happens:** Test payload sizes like `2*MAX+512` may not be aligned.

**How to avoid:** Always round up buffer allocation size to alignment boundary. The addrgen tests compute `aligned_page_size = ((page_size + alignment - 1) / alignment) * alignment`. For this phase (raw-size, single-page buffers), ensure `CreateBuffer` size is aligned. The payload data itself can be any size -- alignment applies to the buffer allocation.

### Pitfall 4: Multicast semaphore accounting

**What goes wrong:** With multicast, each direction sends its own completion atomic_inc. If a chip receives data from 2 directions (e.g., corner chip in the mesh), it gets 2 atomic_inc bumps but the receiver only expects 1.

**Why it happens:** The existing multicast kernel sends one atomic_inc per active direction, and the receiver waits for `sem == 1`. A chip at the intersection of two multicast trees gets bumped twice.

**How to avoid:** For this test system (4 chips, likely 2x2), each receiver chip is reachable from the sender via exactly one direction. The existing pattern (sem_wait_value=1) works. But if testing on larger meshes, this would need adjustment. The N/S direction MeshMcastRange includes E/W hops, so the N/S multicast covers the corner chips -- only one direction reaches each chip.

### Pitfall 5: Source data must be in sender's L1 for fabric send

**What goes wrong:** The fabric sender reads from L1 and sends over fabric. If source data is only in DRAM, the kernel must DMA it to L1 first.

**Why it happens:** The existing `unicast_tx_writer_raw.cpp` takes `src_l1_addr` as an argument, implying data is already in L1.

**How to avoid:** Either:
1. Allocate source buffer in L1 directly (simpler for small/medium payloads)
2. Use a reader kernel (RISCV_0) to DMA DRAM->L1 via circular buffer, and a writer kernel (RISCV_1) to send from L1 (like addrgen tests)

For this phase, option 1 is simpler since test payloads are manageable in L1 (up to 1 MiB fits in L1 on Blackhole). However, the existing broken runner uses DRAM for source buffer (`BufferType::DRAM`) and passes `src_buf->address()` as `src_l1_addr`. This is WRONG -- DRAM addresses are not in L1 space.

**Correct approach:** Either use `BufferType::L1` for source, OR add a reader kernel that copies DRAM->CB->L1 before the writer sends (matching the addrgen 2-kernel pattern).

### Pitfall 6: Topology awareness -- 4-device system

**What goes wrong:** Tests assume hop counts or directions that don't exist on the test system.

**How to avoid:** The 4-device TT system is typically a 2x2 mesh or 1x4 linear chain. For 2D tests, assume 2x2 with all 4 cardinal directions having at most 1 hop. For 1D tests, chips 0-1-2-3 in a line with max 3 hops. Detect actual topology at runtime using `get_forwarding_link_indices`.

---

## Code Examples

### Verified: rx_addrgen.cpp receiver kernel (completion wait)
```cpp
// Source: tests/tt_metal/tt_fabric/fabric_data_movement/addrgen_write/kernels/rx_addrgen.cpp
void kernel_main() {
    const uint32_t sem_addr = get_arg_val<uint32_t>(0);
    const uint32_t expected_value = get_arg_val<uint32_t>(1);
    volatile tt_l1_ptr uint32_t* sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
    noc_semaphore_wait(sem_ptr, expected_value);
    noc_semaphore_set(sem_ptr, 0);
}
```

### Verified: Existing unicast sender kernel (mesh API)
```cpp
// Source: auto_packetization/kernels/unicast_tx_writer_raw.cpp
// Uses mesh::experimental namespace, builds fabric sender from RT args,
// calls fabric_unicast_noc_unicast_write with full payload size,
// then sends atomic_inc for completion signaling.
// This kernel is CORRECT for the basic unicast write family.
```

### Verified: Existing multicast sender kernel (mesh API, per-direction fanout)
```cpp
// Source: auto_packetization/kernels/multicast_tx_writer_raw.cpp
// Uses per-direction senders (W, E, N, S), each with own packet header.
// Sends auto-packetizing multicast write per direction using MeshMcastRange.
// Completion: sends atomic_inc via manual send_payload_flush_non_blocking_from_address per direction.
```

### Verified: Host-side max payload size query
```cpp
// Source: tt_metal/fabric/fabric.cpp:71-74
size_t get_tt_fabric_max_payload_size_bytes() {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    return control_plane.get_fabric_context().get_fabric_max_payload_size_bytes();
}
```

### Verified: BaseFabricFixture device access pattern
```cpp
// Source: tests/tt_metal/tt_fabric/common/fabric_fixture.hpp
// fixture->get_devices() returns vector<shared_ptr<MeshDevice>>
// fixture->get_device(ChipId) returns shared_ptr<MeshDevice> for that chip
// Each MeshDevice wraps a single chip: mesh->get_devices()[0] is the IDevice*
```

### Verified: BaseFabricFixture program dispatch
```cpp
// Source: fabric_fixture.hpp lines 136-163
fixture->RunProgramNonblocking(device_mesh, program);  // Enqueue, don't wait
fixture->WaitForSingleProgramDone(device_mesh, program);  // Block until done
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| MeshDeviceFixtureBase (multi-chip mesh) | BaseFabricFixture (per-chip mesh) | Pre-existing in repo | Different buffer/dispatch APIs |
| test_common.hpp with RawTestParams | File deleted, must define inline | Phase 1 cleanup | Runner must define own types |
| Single kernel for all families | Separate kernel per family | This phase (recommended) | Cleaner test coverage |

---

## Open Questions

1. **L1 vs DRAM for source buffer in BaseFabricFixture tests**
   - What we know: The existing broken runner uses DRAM source buffers, but the device kernel expects L1 addresses. The addrgen tests solve this with a reader-kernel DMA pipeline.
   - What's unclear: Whether L1 is large enough for all test payload sizes (1 MiB) on the target system, or whether a 2-kernel DRAM->L1->fabric pipeline is needed.
   - Recommendation: Use DRAM source + reader kernel (2-kernel pattern) for consistency with addrgen tests and to handle 1 MiB payloads safely. Alternatively, use DRAM source buffer directly since the fabric sender may be able to DMA from DRAM (needs verification).

2. **Exact topology of 4-device test system**
   - What we know: 4 devices available. The 2D fixture configures FABRIC_2D, the 1D fixture configures FABRIC_1D.
   - What's unclear: Whether it's a 2x2 grid, 1x4 line, or T3K (4 of 8 chips).
   - Recommendation: Detect topology dynamically using `get_forwarding_link_indices`. The multicast pattern already handles arbitrary topologies via bounding-box computation.

3. **How to handle fused scatter + atomic_inc testing (families 7 and 8)**
   - What we know: These combine scatter write semantics with atomic_inc signaling. The scatter writes to two addresses per packet, and the atomic_inc fires on the final chunk.
   - What's unclear: Whether `fabric_unicast_noc_fused_scatter_write_atomic_inc` and its multicast variant are tested elsewhere or truly need new test kernels.
   - Recommendation: Create dedicated kernels. The API shape is: `NocUnicastScatterAtomicIncFusedCommandHeader{noc_addr0, noc_addr1, sem_noc_addr, inc, width_bits}`.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Google Test (C++ GTest) |
| Config file | `tests/tt_metal/tt_fabric/sources.cmake` |
| Quick run command | `./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="Fabric2DFixture.AutoPacketization*"` |
| Full suite command | `./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="*AutoPacketization*"` |

### Phase Requirements to Test Map

| Behavior | Test Type | Automated Command |
|----------|-----------|-------------------|
| Unicast write chunking delivers correct data | silicon integration | `--gtest_filter="*AutoPacket*Unicast*"` |
| Multicast write chunking delivers correct data to all destinations | silicon integration | `--gtest_filter="*AutoPacket*Multicast*"` |
| Scatter write passthrough delivers correct data | silicon integration | `--gtest_filter="*AutoPacket*Scatter*"` |
| Fused atomic_inc fires exactly once for large payloads | silicon integration | `--gtest_filter="*AutoPacket*FusedAtomic*"` |
| Fused scatter+atomic_inc fires exactly once | silicon integration | `--gtest_filter="*AutoPacket*FusedScatter*"` |
| Sparse multicast (linear-only) delivers correct data | silicon integration | `--gtest_filter="*AutoPacket*Sparse*"` |
| Multiple payload sizes (< MAX, 2*MAX+512, 1 MiB) all pass | silicon integration | Each family tested at all 3 sizes |
| Both 2D and 1D fixtures work | silicon integration | Separate TEST_F for Fabric2DFixture and Fabric1DFixture |

### Sampling Rate
- **Per task commit:** Compile check (`./build_metal.sh -e -c --build-tests`)
- **Per wave merge:** Full auto-packetization test suite on silicon
- **Phase gate:** All silicon tests green before verify-work

### Wave 0 Gaps
- [ ] `auto_packetization/test_common.hpp` -- new RawTestParams struct and family enum (file was deleted, must recreate)
- [ ] Runner rewrite: `unicast_runner.cpp` adapted for BaseFabricFixture
- [ ] Runner rewrite: `multicast_runner.cpp` adapted for BaseFabricFixture
- [ ] `sources.cmake` -- runner .cpp files added to UNIT_TESTS_FABRIC_SRC
- [ ] New kernel files for 6 remaining families (scatter, fused, fused_scatter, sparse_multicast)
- [ ] TEST_F silicon cases in `test_auto_packetization.cpp`

---

## Sources

### Primary (HIGH confidence)
- Direct source inspection: `tests/tt_metal/tt_fabric/common/fabric_fixture.hpp` -- BaseFabricFixture class, all methods, Fabric2DFixture/Fabric1DFixture
- Direct source inspection: `tests/tt_metal/tt_fabric/fabric_data_movement/addrgen_write/unicast_runner.cpp` -- proven silicon test pattern (711 lines)
- Direct source inspection: `tests/tt_metal/tt_fabric/fabric_data_movement/addrgen_write/multicast_runner.cpp` -- multicast silicon pattern (535 lines)
- Direct source inspection: `tests/tt_metal/tt_fabric/fabric_data_movement/addrgen_write/test_main.cpp` -- TEST_P registration pattern
- Direct source inspection: `tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/unicast_runner.cpp` -- existing broken runner (reference for what needs fixing)
- Direct source inspection: `tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/multicast_runner.cpp` -- existing broken multicast runner
- Direct source inspection: `tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/*.cpp` -- 6 existing kernel files
- Direct source inspection: `tests/tt_metal/tt_fabric/fabric_data_movement/addrgen_write/kernels/rx_addrgen.cpp` -- receiver kernel
- Direct source inspection: `tests/tt_metal/tt_fabric/fabric_data_movement/addrgen_write/test_common.hpp` -- AddrgenTestParams struct, enum
- Direct source inspection: `tt_metal/fabric/fabric.cpp` -- `get_tt_fabric_max_payload_size_bytes()` implementation
- Direct source inspection: `tests/tt_metal/tt_fabric/sources.cmake` -- build configuration

### Secondary (MEDIUM confidence)
- Phase 1 RESEARCH.md -- API inventory, chunking patterns, command header structures
- Phase 2 CONTEXT.md -- locked decisions from user

---

## Metadata

**Confidence breakdown:**
- Test infrastructure (BaseFabricFixture): HIGH -- fully inspected fixture source code
- Runner adaptation pattern: HIGH -- addrgen runners provide complete reference
- Device kernel patterns: HIGH -- existing kernels for 3 families are verified working (compile probes pass)
- Remaining 6 kernel families: MEDIUM -- API signatures known from Phase 1, but no existing kernel code to reference
- Topology/hop counts: MEDIUM -- runtime detection pattern is proven, but exact test system topology not confirmed

**Research date:** 2026-03-11
**Valid until:** 30 days (stable C++ test infrastructure)
