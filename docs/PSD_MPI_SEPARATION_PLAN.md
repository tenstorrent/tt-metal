# PSD and MPI/Discovery Separation Plan

## Goal
Isolate the Physical System Descriptor (PSD) from MPI utilities and discovery logic so that:
1. **PSD** can move to a public header as a pure data structure + query API (no MPI dependency)
2. **Discovery** lives in a private utility that takes input arguments and returns a populated PSD object
3. PSD objects have **no inherent** MPI or discovery functionality

---

## Current State Analysis

### What PSD Currently Contains

| Component | MPI/Discovery? | Hardware Dep? | Location |
|-----------|----------------|---------------|----------|
| **Data structures** (ASICDescriptor, EthConnection, PhysicalConnectivityGraph, etc.) | No | No | Header |
| **Constructors** | Yes – discovery ctors take DistributedContext, call run_discovery | Yes – cluster, hal | .hpp/.cpp |
| **run_discovery()**, run_local_discovery(), run_global_discovery() | **Yes** – MPI barrier, send/recv | Yes – cluster, UMD | .cpp |
| **resolve_hostname_uniqueness()** | **Yes** – MPI collective | No | .cpp |
| **exchange_metadata()** | **Yes** – MPI gather/scatter | No | .cpp |
| **generate_cross_host_connections()** | No | No | .cpp |
| **merge()** | No | No (validation only) | .cpp |
| **Query APIs** (get_asic_neighbors, get_tray_id, etc.) | No | No | .cpp |
| **my_host_name()** | **Yes** – uses distributed_context_ for disambiguation | No | .cpp |
| **Serialization** (YAML, proto) | No | No | serialization/ |
| **query_local_ethernet_metrics()** | No | **Yes** – cluster, hal | .cpp |
| **get_chip_id_for_asic()** | No | **Yes** – cluster_desc_ | .cpp (private) |

### Key Dependencies to Remove from PSD
- `distributed_context_` member
- `cluster_`, `hal_` members (used for discovery and live metrics)
- `cluster_desc_` (derived from cluster during discovery)
- All MPI calls: barrier, send, recv
- Discovery methods: run_discovery, run_local_discovery, run_global_discovery, resolve_hostname_uniqueness, exchange_metadata

### PSD Members That Stay (Pure Data)
- `system_graph_` (PhysicalConnectivityGraph)
- `asic_descriptors_`
- `host_to_mobo_name_`
- `host_to_rank_` (populated by discovery, but it's just data)
- `exit_node_connection_table_`
- `all_hostnames_unique_`
- `ethernet_firmware_version_`
- `pcie_devices_per_tray_`
- `pcie_id_to_asic_location_`
- `target_device_type_`

---

## Proposed Architecture

### 1. Public PSD (pure data + queries)

**Location:** `tt_metal/api/tt-metalium/experimental/fabric/physical_system_descriptor.hpp` (or similar under public API)

**Contents:**
- All data structs: `EthernetMetrics`, `ASICDescriptor`, `EthConnection`, `ExitNodeConnection`, `PhysicalConnectivityGraph`, type aliases
- `PhysicalSystemDescriptor` class with:
  - **Constructors:**
    - Default / move (for internal use)
    - `PhysicalSystemDescriptor(const std::string& proto_path)` – parse from file (already exists, no MPI)
    - `PhysicalSystemDescriptor(PhysicalSystemDescriptor&& other)` – move
  - **No constructors** that take cluster, DistributedContext, or hal
  - **Query APIs only** – all existing getters that read from stored data
  - **Serialization helpers** – `dump_to_yaml()`, `generate_yaml_node()`, `emit_to_text_proto()`
  - **merge()** – for combining PSDs (used internally by discovery)
- **No** `run_discovery()`, `run_local_discovery()`, `run_global_discovery()`
- **No** `distributed_context_`, `cluster_`, `hal_`, `cluster_desc_` members
- **No** `query_local_ethernet_metrics()` on PSD – becomes a separate utility

**my_host_name() behavior:** When hostnames are unique, return `get_host_name()`. When not unique (from discovery), the discovery utility will have populated `host_to_rank_` and the caller can derive `hostname + "_" + rank`. For a pure PSD, we need a way to know "my" host. Options:
- (a) Store `local_hostname` in PSD during discovery (set by discovery utility)
- (b) Require `my_host_name()` to take an optional rank parameter for disambiguation
- (c) Keep `all_hostnames_unique_` and have `my_host_name()` require the caller to pass rank when hostnames aren't unique – but then PSD would need rank. Cleaner: discovery sets `host_to_rank_`, and we add a separate concept of "this process's hostname" – e.g. discovery utility returns `std::pair<PhysicalSystemDescriptor, std::string> my_hostname` or we add a small `DiscoveryResult` struct.

Simpler approach: **Store `local_hostname_`** in PSD. Discovery sets it. For file-parsed PSD, it can be empty or set from first host. The existing `my_host_name()` can read `local_hostname_` when available; otherwise fall back to `get_host_name()` when `all_hostnames_unique_` is true. When we have no distributed context, we cannot do the disambiguation – so for file-based PSD, we assume hostnames are unique or the user provides context elsewhere.

Refined: **PSD remains a dumb data container.** `my_host_name()` is problematic because it uses MPI rank. We have two choices:
1. Remove `my_host_name()` from PSD. Callers that need "my host" get it from the discovery utility's return value (e.g. `DiscoveryResult{psd, my_hostname, my_rank}`).
2. Add optional `local_hostname_` and `local_rank_` that discovery populates. For file-based PSD they stay unset. `my_host_name()` returns `local_hostname_` when set.

Option 2 preserves backward compatibility. Discovery will set `local_hostname_` and the PSD will have it. File-based PSD won't have it – `my_host_name()` could TT_FATAL or require the caller to use a different API. Actually the current `my_host_name()` when `all_hostnames_unique_` uses `get_host_name()` – no MPI. When not unique it uses rank from distributed_context_. So we need rank from somewhere. Storing `local_rank_` and `local_hostname_` in PSD (set by discovery) keeps PSD self-contained without holding a DistributedContext reference.

### 2. Discovery Utility (private)

**Location:** `tt_metal/fabric/physical_system_discovery.hpp` (private, in fabric impl) + `physical_system_discovery.cpp`

**API:**
```cpp
namespace tt::tt_metal {

// Run discovery and return a populated PSD. No MPI or cluster references in the returned PSD.
PhysicalSystemDescriptor run_physical_system_discovery(
    tt::umd::Cluster& cluster,
    std::shared_ptr<distributed::multihost::DistributedContext> distributed_context,
    const Hal* hal,
    tt::TargetDevice target_device_type,
    bool run_global_discovery = true,
    bool run_live_discovery = false);

}  // namespace tt::tt_metal
```

**Note:** Call sites typically use `const_cast<tt::umd::Cluster&>(*cluster.get_driver())` since `get_driver()` returns `const std::unique_ptr<Cluster>&`.

**Responsibilities:**
- Call barrier, resolve_hostname_uniqueness (MPI)
- run_local_discovery logic (query UMD cluster, build local PSD data)
- run_global_discovery logic (exchange_metadata, merge, validate)
- Populate `local_hostname_` and `local_rank_` in the returned PSD
- Return a `PhysicalSystemDescriptor` with no MPI/cluster/hal references

### 3. Live Metrics Utility (private)

**Location:** Same or adjacent private header

```cpp
LocalEthernetMetrics query_local_ethernet_metrics(
    const PhysicalSystemDescriptor& psd,
    tt::umd::Cluster& cluster,
    const Hal* hal);
```

Takes PSD + hardware context, returns metrics. Used by cluster validation.

### 4. Serialization (stays as-is, can be public or private)

- `deserialize_physical_system_descriptor_from_text_proto_file()` – returns PSD, no MPI
- `deserialize_physical_system_descriptor_from_bytes()` – same
- `serialize_physical_system_descriptor_to_bytes()` – takes PSD, no MPI
- `emit_physical_system_descriptor_to_text_proto()` – same

These depend only on PSD and proto. They can live with the public PSD or in a separate serialization module that includes the public PSD header.

### 5. Proto-based Constructor

Current: `PhysicalSystemDescriptor(const std::string& mock_proto_desc_path)` – uses deserialization, no MPI. This stays. The deserialization currently constructs PSD with `null_cluster, nullptr, nullptr, target_device_type, false` and then merge. We need a new constructor that doesn't require those – e.g. a private constructor that takes the result of `proto_to_physical_system_descriptor()` or a factory `PhysicalSystemDescriptor::from_proto_file(path)`.

---

## File Layout

```
tt_metal/api/tt-metalium/experimental/fabric/
  physical_system_descriptor.hpp    # NEW – public PSD (data types + class, no MPI)

tt_metal/fabric/
  physical_system_descriptor.cpp   # Slim – only PSD methods that don't need discovery (queries, merge, yaml, etc.)
  physical_system_discovery.hpp                # NEW – private, run_physical_system_discovery() declaration
  physical_system_discovery.cpp                 # NEW – all discovery logic moved here
  serialization/
    physical_system_descriptor_serialization.hpp/cpp  # Unchanged, but includes public PSD header
```

The existing `physical_system_descriptor.hpp` in `tt_metal/fabric/` would be deprecated or removed once the public one is in place. Alternatively, the public header could live at `tt_metal/api/tt-metalium/experimental/fabric/physical_system_descriptor.hpp` and the fabric impl keeps a thin include or uses the public one.

---

## Implementation Progress

### Current Status Summary

| Phase | Status | Completion |
|-------|--------|------------|
| Phase 1: Introduce New Types and Discovery API | ✅ COMPLETED | 100% |
| Phase 2: Slim Down PSD | ✅ COMPLETED | 100% |
| Phase 3: Update Call Sites | ✅ COMPLETED | 100% |
| Phase 4: Move PSD to Public Header | ✅ COMPLETED | 100% |

**Key Achievements:**
- ✅ Discovery logic successfully separated into `physical_system_discovery` module
- ✅ New `run_physical_system_discovery()` API implemented and working
- ✅ All call sites updated to use new API
- ✅ PSD class cleaned up: removed MPI/hardware dependencies, old constructors, and discovery methods
- ✅ `clear()` and `merge()` made public for re-discovery scenarios
- ✅ `query_local_ethernet_metrics` moved to free function
- ✅ Build compiles successfully with all changes
- ✅ Fixed segmentation fault in `exchange_metadata` with empty check

**Remaining Work:**
- ❌ Public header migration not started (pending decision on when to expose PSD publicly)
- ⚠️ Old members (`distributed_context_`, `cluster_`, `hal_`, `cluster_desc_`) removed from PSD
- ⚠️ Old constructors removed from PSD (only minimal constructor remains)

---

## Migration Steps

### Phase 1: Introduce New Types and Discovery API ✅ COMPLETED
1. ✅ Create `physical_system_discovery.hpp` with `run_physical_system_discovery()` declaration.
2. ✅ Create `physical_system_discovery.cpp` and move discovery logic from `physical_system_descriptor.cpp`:
   - `run_discovery`, `run_local_discovery`, `run_global_discovery` → moved to `discovery_impl` namespace
   - `resolve_hostname_uniqueness`, `exchange_metadata` → moved to `discovery_impl` namespace
   - Helper functions used only by discovery (`get_tray_id_for_chip`, `get_asic_position`, etc.) → moved to `discovery_helpers` namespace
3. ✅ Implement `run_physical_system_discovery()` to build and return a PSD (no MPI/cluster stored in result).
4. ✅ Add `local_hostname_` and `local_rank_` to PSD for discovery-populated identity.
5. ✅ Move `query_local_ethernet_metrics` to a free function in `physical_system_discovery.cpp`.

**Status:** Phase 1 is complete. All discovery logic has been moved to the new discovery module, and the new API is functional.

### Phase 2: Slim Down PSD ✅ COMPLETED
**Completed Tasks:**
- ✅ `my_host_name()` updated to use `local_hostname_` and `local_rank_` (set by discovery)
- ✅ `query_local_ethernet_metrics(psd, cluster, hal)` implemented as free function
- ✅ `cluster_validation_utils` updated to use new `query_local_ethernet_metrics` signature
- ✅ Removed all old constructors taking cluster, DistributedContext, hal
- ✅ Removed `run_discovery()`, `run_local_discovery()`, `run_global_discovery()` methods from PSD
- ✅ Removed `distributed_context_`, `cluster_`, `hal_`, `cluster_desc_` members from PSD
- ✅ Moved all discovery helper functions (`resolve_hostname_uniqueness`, `exchange_metadata`, `get_chip_id_for_asic`, etc.) to `physical_system_discovery.cpp`
- ✅ Made `clear()` and `merge()` public methods for re-discovery scenarios
- ✅ Added `local_hostname_` and `local_rank_` private members (set by discovery via friend access)
- ✅ Updated serialization to use minimal constructor
- ✅ Fixed all compilation errors and build succeeds

**Key Changes:**
- PSD now only contains data structures and query APIs
- All MPI and hardware dependencies removed from PSD class
- Discovery logic completely isolated in `physical_system_discovery` module
- Friend declarations allow discovery functions to set `local_hostname_`, `local_rank_`, and `all_hostnames_unique_`

### Phase 3: Update Call Sites ✅ COMPLETED
1. ✅ **Control plane:** Updated both call sites to use `run_physical_system_discovery()`:
   ```cpp
   auto& driver = const_cast<tt::umd::Cluster&>(*cluster.get_driver());
   this->physical_system_descriptor_ = std::make_unique<PhysicalSystemDescriptor>(
       run_physical_system_discovery(driver, distributed_context, &this->hal_.get(), rtoptions.get_target_device()));
   ```

2. ✅ **Serialization:** Updated to use minimal constructor - verified working.

3. ✅ **cluster_validation_utils:** Updated to use `query_local_ethernet_metrics(psd, cluster, hal)`.

4. ✅ **cluster_validation_utils:** Updated re-discovery call site to use `clear()` + `merge()`:
   ```cpp
   auto& driver = const_cast<tt::umd::Cluster&>(*cluster.get_driver());
   auto new_psd = run_physical_system_discovery(driver, distributed_context, &hal, rtoptions.get_target_device(), true, true);
   ctx.physical_system_descriptor.clear();
   ctx.physical_system_descriptor.merge(std::move(new_psd));
   ```

5. ✅ **Tests:** All test files updated to use `run_physical_system_discovery()`:
   - `test_physical_system_descriptor.cpp` - all test cases updated
   - `test_link_retraining.cpp` - updated to use `clear()` + `merge()` for re-discovery
   - `test_send_recv_pipeline.cpp` - updated
   - `multi_host_pipeline.cpp` - updated
   - `test_topology_mapper.cpp` - updated
   - `test_routing_tables.cpp` - updated

6. ✅ **All call sites:** Updated to properly dereference `get_driver()` (returns `const std::unique_ptr<Cluster>&`, function expects `Cluster&`)

**Status:** Phase 3 is complete. All call sites have been migrated to use `run_physical_system_discovery()` with proper `clear()` and `merge()` usage for re-discovery scenarios.

### Phase 4: Move PSD to Public Header ✅ COMPLETED
**Prerequisites:**
- ✅ Complete Phase 2 (remove MPI/cluster/hal dependencies from PSD)
- ✅ Verify all call sites are migrated (Phase 3 complete)

**Tasks:**
1. ✅ Create `tt_metal/api/tt-metalium/experimental/fabric/physical_system_descriptor.hpp` with:
   - ✅ Data types moved from current header (ASICDescriptor, EthConnection, ExitNodeConnection, etc.)
   - ✅ `PhysicalSystemDescriptor` class declaration (minimal constructor, queries, serialization)
   - ✅ Removed all forward declarations of MPI/DistributedContext/Cluster/Hal
   - ✅ Included `EthernetMetrics` and related types
   - ✅ Added `set_discovery_data()` method to replace friend declarations with MPI types
   - ✅ Added `get_all_hostnames_unique()` getter method

2. ✅ Update includes across the codebase:
   - ✅ Updated key internal files to use public header
   - ✅ Old header (`tt_metal/fabric/physical_system_descriptor.hpp`) forwards to public header for backward compatibility
   - ✅ Discovery files updated to use public header

3. ✅ Ensure no public header pulls in MPI or DistributedContext:
   - ✅ Verified `physical_system_discovery.hpp` remains private
   - ✅ Public header has no friend declarations requiring MPI types
   - ✅ Used `set_discovery_data()` method instead of friend declarations

4. ✅ Old header handling:
   - ✅ Kept `tt_metal/fabric/physical_system_descriptor.hpp` as forwarding include for backward compatibility
   - ✅ Old header forwards to new public header and includes forward declarations for discovery functions

**Implementation Notes:**
- Created `set_discovery_data(local_hostname, local_rank, all_hostnames_unique)` public method
- Discovery functions call `set_discovery_data()` instead of using friend access
- Public header is completely free of MPI and hardware dependencies
- All tests pass with new public header structure

---

## Next Steps

### Completed ✅
- ✅ All call sites migrated to use `run_physical_system_discovery()`
- ✅ PSD cleaned up (removed MPI/hardware dependencies)
- ✅ Discovery logic isolated in separate module
- ✅ Build compiles successfully
- ✅ All compilation errors fixed

### Immediate (Testing)
1. Run CPU-only tests to verify functionality
2. Run dual-host mock cluster tests
3. Verify serialization/deserialization works correctly

### Long-term (Phase 4)
1. Create public header for PSD (when ready to expose publicly)
2. Migrate includes across codebase
3. Remove old private header or convert to forwarding include
4. Consider adding timeout mechanism for discovery (as noted in Future Improvements)

---

## Dependency Checklist for Public PSD Header

Public PSD header must NOT include or depend on:
- `distributed_context.hpp` / `DistributedContext`
- `cluster.hpp` / `Cluster` (tt::umd)
- `hal.hpp` / `Hal`
- Any MPI headers
- `yaml-cpp` (optional – can keep dump_to_yaml in impl if it pulls heavy deps)

Public PSD header may include:
- `fabric_types.hpp` (AsicID, TrayID, etc.)
- Standard library, tt_stl, umd/semver if needed
- Optional: `YAML::Node` forward decl for `generate_yaml_node()` – or move to impl

---

## Open Questions

1. **YAML dependency:** Is `generate_yaml_node()` / `dump_to_yaml()` part of the public API? If yes, we need yaml-cpp in the public surface or we move these to an extension module.
2. **get_chip_id_for_asic:** Used only by query_local_ethernet_metrics. With that as a free function, we could have an internal helper that takes (psd, cluster_desc) to resolve chip_id. No need to store chip_id in PSD.
3. **cluster_desc_ for get_chip_id_for_asic:** We need chip_id for local ASICs when querying metrics. The cluster_desc has get_chip_unique_ids(). The inverse map (unique_id -> chip_id) could be passed into `query_local_ethernet_metrics` or we could require the cluster (and derive cluster_desc inside the function).
4. **Backward compatibility:** Do we need to preserve the old PSD constructors during a transition period, or can we switch all call sites in one change?

---

## Future Improvements / Testing Considerations

### Timeout for Physical Discovery

**Issue:** When physical discovery fails (e.g., MPI communication errors, network issues, or validation failures), the discovery process can hang indefinitely, making it difficult to diagnose issues and causing tests to appear stuck.

**Recommendation:** Add a timeout mechanism to `run_physical_system_discovery()` to prevent indefinite hangs. Options:
- Add a timeout parameter (e.g., `std::chrono::seconds timeout = std::chrono::seconds(300)`) to the discovery function
- Use MPI's built-in timeout mechanisms if available
- Implement watchdog timers for critical MPI operations (barrier, send/recv)
- Return a `std::optional<PhysicalSystemDescriptor>` or use a result type that can indicate timeout vs. failure

This is especially important for:
- CI/CD test environments where hanging tests block pipelines
- Multi-host scenarios where one rank failure can cause others to hang
- Debugging discovery issues where failures are not immediately obvious

**Implementation Note:** Consider adding this in Phase 1 or Phase 2, as it improves robustness of the discovery utility.

---

## Implementation Summary

### ✅ Completed Work

**Phase 1: Discovery API Separation** - 100% Complete
- ✅ Created `physical_system_discovery.hpp` and `.cpp` modules
- ✅ Moved all discovery logic (run_discovery, run_local_discovery, run_global_discovery, resolve_hostname_uniqueness, exchange_metadata)
- ✅ Implemented `run_physical_system_discovery()` free function API
- ✅ Added `local_hostname_` and `local_rank_` members to PSD
- ✅ Moved `query_local_ethernet_metrics` to free function
- ✅ All helper functions organized into anonymous namespace and `discovery_impl` namespace
- ✅ Fixed segmentation fault in `exchange_metadata` with empty `asic_descriptors_` check

**Phase 2: Slim Down PSD** - 100% Complete
- ✅ Removed all old constructors taking cluster, DistributedContext, hal
- ✅ Removed `run_discovery()`, `run_local_discovery()`, `run_global_discovery()` methods
- ✅ Removed `distributed_context_`, `cluster_`, `hal_`, `cluster_desc_` members
- ✅ Moved all discovery helper functions to `physical_system_discovery.cpp`
- ✅ Made `clear()` and `merge()` public methods
- ✅ Updated `my_host_name()` to use `local_hostname_` and `local_rank_`
- ✅ Updated serialization to use minimal constructor
- ✅ Fixed all compilation errors (forward declarations, friend access, duplicate functions)

**Phase 3: Call Site Updates** - 100% Complete
- ✅ Control plane updated (2 call sites)
- ✅ `cluster_validation_utils` updated for `query_local_ethernet_metrics` and re-discovery
- ✅ `run_cluster_validation.cpp` updated
- ✅ All test files updated (6+ test files)
- ✅ All call sites properly handle `get_driver()` dereferencing

**Testing**
- ✅ Code compiles successfully
- ✅ All build errors resolved
- ✅ Ready for test execution

### ✅ Completed Work

**Phase 4: Public Header Migration** - 100% Complete
- ✅ Created public header at `tt_metal/api/tt-metalium/experimental/fabric/physical_system_descriptor.hpp`
- ✅ Moved all PSD class declaration and data types to public header
- ✅ Removed MPI/hardware forward declarations from public header
- ✅ Created `set_discovery_data()` method to replace friend declarations with MPI types
- ✅ Updated discovery implementation to use `set_discovery_data()` instead of direct member access
- ✅ Added `get_all_hostnames_unique()` getter method
- ✅ Updated CMakeLists.txt to include new public header in TT_METAL_PUBLIC_API
- ✅ Updated old header to forward include new public header for backward compatibility
- ✅ Updated key internal includes to use new public header
- ✅ Build compiles successfully
- ✅ Tests pass with new public header

### Key Files Modified

**New Files:**
- `tt_metal/fabric/physical_system_discovery.hpp` - Discovery API declaration
- `tt_metal/fabric/physical_system_discovery.cpp` - Discovery implementation
- `tt_metal/api/tt-metalium/experimental/fabric/physical_system_descriptor.hpp` - Public header for PSD

**Modified Files:**
- `tt_metal/fabric/physical_system_descriptor.hpp` - Now forwards to public header, kept for backward compatibility
- `tt_metal/fabric/physical_system_descriptor.cpp` - Updated to include public header, added `set_discovery_data()` implementation
- `tt_metal/fabric/physical_system_discovery.hpp` - Updated to include public header
- `tt_metal/fabric/physical_system_discovery.cpp` - Updated to use `set_discovery_data()` instead of friend access
- `tt_metal/CMakeLists.txt` - Added public header to TT_METAL_PUBLIC_API
- `tt_metal/fabric/control_plane.cpp` - Updated to use run_physical_system_discovery() (2 call sites)
- `tools/scaleout/validation/utils/cluster_validation_utils.cpp` - Updated query_local_ethernet_metrics call and re-discovery logic
- `tools/scaleout/validation/run_cluster_validation.cpp` - Updated to use run_physical_system_discovery() (2 call sites)
- `tools/tests/scaleout/test_link_retraining.cpp` - Updated constructor and re-discovery to use run_physical_system_discovery()
- `tests/ttnn/unit_tests/gtests/multiprocess/test_send_recv_pipeline.cpp` - Updated to use run_physical_system_discovery()
- `tests/tt_metal/multihost/socket_pipeline/multiprocess/multi_host_pipeline.cpp` - Updated to use run_physical_system_discovery()
- `tests/tt_metal/tt_fabric/physical_discovery/test_physical_system_descriptor.cpp` - Updated all tests
- `tests/tt_metal/tt_fabric/fabric_router/test_topology_mapper.cpp` - Updated to use run_physical_system_discovery()
- `tests/tt_metal/tt_fabric/fabric_router/test_routing_tables.cpp` - Updated to use run_physical_system_discovery()
- `tt_metal/fabric/CMakeLists.txt` - Added physical_system_discovery.cpp to build

**Key Implementation Details:**
- Discovery functions use `set_discovery_data()` method instead of friend access (avoids MPI types in public header)
- All call sites properly dereference `get_driver()` using `const_cast<tt::umd::Cluster&>(*cluster.get_driver())`
- Helper functions (`get_host_name()`, `get_mobo_name()`, etc.) made inline to avoid redefinition in unity builds
- Fixed empty container access in `exchange_metadata()` with `TT_FATAL` check
- Public header has no MPI or hardware dependencies - pure data structure + query API
- Old header (`tt_metal/fabric/physical_system_descriptor.hpp`) forwards to public header for backward compatibility

---

## Summary

| Component | Location | MPI? | Notes |
|-----------|----------|------|-------|
| PhysicalSystemDescriptor (data + queries) | Public header | No | Pure container |
| run_physical_system_discovery() | Private header | Yes | Returns PSD |
| query_local_ethernet_metrics() | Private | No (takes cluster+hal) | Free function |
| Serialization (proto, yaml) | Can be public | No | Depends only on PSD |
| Control plane / TopologyMapper | Unchanged | Uses PSD + discovery | Call run_physical_system_discovery() |
