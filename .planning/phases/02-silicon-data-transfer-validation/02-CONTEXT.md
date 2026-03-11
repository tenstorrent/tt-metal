# Phase 2: Silicon Data-Transfer Validation - Context

**Gathered:** 2026-03-11
**Status:** Ready for planning

<domain>
## Phase Boundary

Validate all 9 auto-packetizing wrapper families deliver correct data on silicon. The compile-only tests from Phase 1 confirm the headers build — this phase confirms data actually arrives correctly when payloads exceed FABRIC_MAX_PACKET_SIZE. Runners are rewritten for BaseFabricFixture, registered in test_auto_packetization.cpp, and compiled into fabric_unit_tests.

</domain>

<decisions>
## Implementation Decisions

### Runner code adaptation
- Rewrite unicast_runner.cpp and multicast_runner.cpp to use `BaseFabricFixture` from `fabric_fixture.hpp`
- Drop dependency on deleted `test_common.hpp` — define minimal types inline or in a new lightweight header
- Test functions take `BaseFabricFixture*` (not `MeshDeviceFixtureBase*`)
- Follow the `addrgen_write/unicast_runner.cpp` pattern exactly: allocate buffers, fill with pattern, dispatch kernel, wait, compare
- Reuse existing addrgen receiver kernel (`rx_addrgen.cpp`) for completion signaling — it's protocol-agnostic (waits on semaphore, validates buffer)
- `TEST_F` cases registered in `test_auto_packetization.cpp` (not in runner files)
- Runner files added to `sources.cmake` for `fabric_unit_tests` binary
- Run both `Fabric2DFixture` (mesh APIs) and `Fabric1DFixture` (linear APIs) silicon tests

### Test payload sizes
- Multiple sizes per test:
  1. size < MAX (single packet — regression test, no chunking exercised)
  2. size = 2 * MAX + 512 (exercises loop + remainder)
  3. size = 1 MiB (many chunks — stress test)
- FABRIC_MAX_PACKET_SIZE is a runtime value — sizes computed at test time
- Byte-for-byte data comparison: fill source with incrementing byte pattern, compare destination buffer exactly

### API variant coverage — all 9 families on silicon
- **Chunking families (oversized payloads):**
  - `fabric_unicast_noc_unicast_write` (mesh + linear)
  - `fabric_multicast_noc_unicast_write` (mesh + linear)
  - `fabric_unicast_noc_fused_unicast_with_atomic_inc` (mesh + linear) — verify atomic_inc fires exactly once
  - `fabric_multicast_noc_fused_unicast_with_atomic_inc` (mesh + linear)
- **Scatter passthroughs (small payloads, verify passthrough works):**
  - `fabric_unicast_noc_scatter_write` (mesh + linear)
  - `fabric_multicast_noc_scatter_write` (mesh + linear)
  - `fabric_unicast_noc_fused_scatter_write_atomic_inc` (mesh + linear)
  - `fabric_multicast_noc_fused_scatter_write_atomic_inc` (mesh + linear)
- **Sparse multicast (linear-only, small payload):**
  - `fabric_sparse_multicast_noc_unicast_write`

### Multicast topology
- Multi-hop multicast to all reachable neighbors on the 4-device system
- Test MeshMcastRange with non-zero hops in all active directions

### Claude's Discretion
- Exact buffer allocation strategy (L1 vs DRAM)
- Test pattern (incrementing bytes, random, etc.)
- How to parameterize tests across the 9 families (GTest parameterized vs separate TEST_F per family)
- Whether to create new kernel files per family or extend existing kernels with runtime-arg-based dispatch

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `tests/tt_metal/tt_fabric/fabric_data_movement/addrgen_write/unicast_runner.cpp` — proven silicon test pattern to mirror
- `tests/tt_metal/tt_fabric/fabric_data_movement/addrgen_write/multicast_runner.cpp` — multicast silicon pattern
- `tests/tt_metal/tt_fabric/common/fabric_fixture.hpp` — `BaseFabricFixture`, `Fabric2DFixture`, `Fabric1DFixture`
- `tests/tt_metal/tt_fabric/common/utils.hpp` — test utilities
- Existing device kernels in `auto_packetization/kernels/` — unicast_tx_writer_raw.cpp, multicast_tx_writer_raw.cpp, compile probes
- addrgen receiver kernel (`rx_addrgen.cpp` or equivalent) — reusable for completion signaling

### Established Patterns
- `BaseFabricFixture::DoSetUpTestSuite(FabricConfig)` for device setup
- `RunProgramNonblocking` for dispatch
- `append_fabric_connection_rt_args` for kernel runtime args
- `detail::CompileProgram` for compile-only validation (Phase 1)
- `FABRIC_2D=1` define for mesh kernel compilation

### Integration Points
- `tests/tt_metal/tt_fabric/sources.cmake` — add runner .cpp files to UNIT_TESTS_FABRIC_SRC
- `tests/tt_metal/tt_fabric/fabric_data_movement/test_auto_packetization.cpp` — add TEST_F cases for silicon tests

</code_context>

<specifics>
## Specific Ideas

- Follow addrgen_write runner pattern exactly — it's proven on silicon
- Reuse addrgen receiver kernel for completion signaling
- Device test serialization constraint still applies — only one test at a time on hardware

</specifics>

<deferred>
## Deferred Ideas

- Connection manager variant silicon tests (multi-connection breadth-first verification) — would need multi-connection test setup
- Performance benchmarking of auto-packetization overhead vs manual chunking

</deferred>

---

*Phase: 02-silicon-data-transfer-validation*
*Context gathered: 2026-03-11*
