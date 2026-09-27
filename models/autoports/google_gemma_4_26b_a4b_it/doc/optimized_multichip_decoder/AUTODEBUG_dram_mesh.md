# AutoDebug: DRAM-sharded multi-reader matmul on MeshDevice

## Finding

The failing Stage05 command is a host-side descriptor construction failure, not a device kernel failure. The failing run is `dram_qkv_r2_sliding.command.json`, which runs the multichip decoder with `--attention-dram qkv --dram-readers 2 --dram-storage-cores 4` and exits before launch with:

`TT_FATAL @ tt_metal/impl/device/experimental/device.cpp:19: get_worker_noc_hop_distance() is only supported on unit MeshDevice.`

The direct cause is the two-reader DRAM-sharded matmul path passing the multi-device `MeshDevice` pointer into the public hop-distance helper. That helper explicitly accepts raw `Device` and unit `MeshDevice` only:

- `tt_metal/impl/device/experimental/device.cpp:17-22` dynamic-casts to `distributed::MeshDevice` and fatals when `mesh->num_devices() != 1`.
- `ttnn/cpp/ttnn/operations/matmul/device/utilities/matmul_utilities.cpp:446-447` calls this helper while choosing secondary DRAM-bank reader cores.

The first erroneous boundary is earlier than the fatal. `MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory::create_descriptor` gets `tt::tt_metal::IDevice* device = a.device()` at `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:979` and passes that mesh-wide device into `create_program_dram_sharded_descriptor`. The generic mesh descriptor adapter does not pass a mesh coordinate to this factory, because the factory's fourth parameter is `std::optional<CoreRangeSet>`, not `std::optional<MeshCoordinate>` (`...program_factory.hpp:14-18`). For uniformly placed tensors, `create_and_cache_mesh_workload` builds a single coordinate range for the full mesh (`ttnn/api/ttnn/device_operation.hpp:318-331`), and `DescriptorMeshWorkloadAdapter` invokes `create_descriptor` once for that range with no coordinate (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:607-615`). The descriptor factory therefore has only the mesh pointer available.

## Why one reader passes and two readers fails

`get_dram_bank_reader_assignments` always obtains the primary DRAM-bank readers first:

- `ttnn/cpp/ttnn/operations/matmul/device/utilities/matmul_utilities.cpp:411` calls `device->get_optimal_dram_bank_to_logical_worker_assignment(noc)`.
- For a `MeshDevice`, that method delegates to the first local/reference device at `tt_metal/distributed/mesh_device.cpp:1215-1216`.

When `workers_per_bank == 1`, the helper just emits those primary readers and returns at `matmul_utilities.cpp:415-420`. That matches the passing contrast run `dram_qkv_r1_sliding.command.json`: the command exits 0 and records `passed: True`, cache PCCs near 0.999997, and `min_pcc 0.997589630569692`.

When `workers_per_bank > 1`, the helper must place secondary readers. It validates NOC0, scans candidate logical cores, and scores candidates by NOC hop distance to the primary reader (`matmul_utilities.cpp:422-459`). That is the first time the mesh-wide `IDevice*` reaches the public hop-distance helper, and a 1x4 mesh triggers the exact unit-mesh assertion.

The parent host probe confirms this boundary without hardware:

- `dram_mesh_host_before.log`: one-reader mesh4 agrees with raw-device behavior.
- Two-reader and three-reader mesh4 reproduce the exact public API failure.
- Raw device and unit mesh agree for two and three readers.

This is a host-boundary result only. It proves the failing assertion and the reader-count trigger, but it does not prove final silicon correctness after a source fix.

## Minimal fix recommendation

Do not relax `tt_metal::experimental::Device::get_worker_noc_hop_distance(IDevice*, ...)` to accept arbitrary multi-device meshes. Its current guard is intentional, and there is already a coordinate-specific overload for mesh use.

Instead, keep the DRAM-sharded matmul descriptor range-level behavior and make the DRAM-reader placement helper use the same representative raw device that `MeshDeviceImpl::get_optimal_dram_bank_to_logical_worker_assignment(noc)` already uses. In `ttnn/cpp/ttnn/operations/matmul/device/utilities/matmul_utilities.cpp`, resolve a placement device once near the start of `get_dram_bank_reader_assignments`:

```cpp
tt::tt_metal::IDevice* placement_device = device;
if (auto* mesh = dynamic_cast<ttnn::distributed::MeshDevice*>(device)) {
    const auto local_devices = mesh->get_devices();
    TT_FATAL(!local_devices.empty(), "DRAM reader placement requires at least one local device");
    placement_device = local_devices.front();
}
```

Then use `placement_device` for:

- `get_optimal_dram_bank_to_logical_worker_assignment(noc)`,
- `compute_with_storage_grid_size()`,
- `experimental::Device::get_worker_noc_hop_distance(...)`.

That is the smallest source change that makes the one-reader and multi-reader branches internally consistent. It matches the existing representative-device convention for mesh primary-reader placement, keeps raw-device and unit-mesh behavior unchanged, avoids changing public experimental API semantics, and avoids making this factory coordinate-aware unless heterogeneous per-device placement becomes a separate project.

The likely sibling copy under `ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/utilities/matmul_utilities.cpp` has the same `get_device_for_dram_banks` helper pattern but did not show the multi-reader assignment helper in the inspected range. It does not need to be changed for this failure unless a matching multi-reader helper is added or found there.

## Discriminating experiment

Before touching hardware, keep the parent host probe or add a small host test with fake `Device` and `MeshDevice` objects around the actual `get_dram_bank_reader_assignments` logic:

1. Pre-fix expectations:
   - `workers_per_bank == 1`: raw device, unit mesh, and mesh4 all agree.
   - `workers_per_bank == 2` and `3`: raw device and unit mesh agree; mesh4 throws the exact public API unit-mesh error.
2. Post-fix expectations:
   - `workers_per_bank == 1`, `2`, and `3`: raw device, unit mesh, and mesh4 all agree on assignments.
   - Reader cores remain unique and excluded storage cores are not selected.
   - The public hop-distance helper still rejects a direct multi-device mesh call.
   - The existing NOC1 guard for multi-reader DRAM-sharded matmul still rejects.

After the source patch builds, rerun the original failing stage command from `dram_qkv_r2_sliding.command.json`. The local build wrapper is not usable on this host because `dram_mesh_wrapper_build.log` reports Docker unavailable, but the parent found `/usr/bin/clang++-20` and an existing Release Ninja build available for focused host/build checks.

## Coverage gap and remaining uncertainty

Existing nightly reader-count coverage is parameterized for `num_workers_per_dram_bank` values 1, 2, and 3 in `tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_dram_sharded.py:261-306`, but it uses the `device` fixture, so it exercises raw/unit-device behavior and misses the multi-device `MeshDevice` descriptor path.

The main remaining hardware uncertainty after the minimal fix is not the host assertion. It is whether representative-device DRAM-reader placement is good enough for all devices in a mesh with heterogeneous harvesting. The current one-reader mesh path already uses the front local device's assignment, and `mesh_device.hpp:119-135` documents that this deprecated no-coordinate overload can be wrong on heterogeneously harvested meshes. The minimal fix preserves that existing convention to unblock homogeneous/local 1x4 runs; it does not solve per-coordinate optimal placement or heterogeneous-harvest performance.
