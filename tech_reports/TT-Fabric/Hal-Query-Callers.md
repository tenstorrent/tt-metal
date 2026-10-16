# Callers of the `hal::` query functions

Every call to a free function declared in `tt_metal/api/tt-metalium/hal.hpp`, and whether the calling function already has a handle (`MetalEnv` or `MeshDevice`) that could replace the ambient-cluster lookup. The audit found 374 call sites on `3e6669b4b28` (2026-10-08). All counts below are for the current branch, `ruizhang/hal-query-tier2` rebased on `origin/main` at `c0b2affd083`: 38 sites have been migrated, 336 remain.

**336 call sites in 177 files (226 enclosing functions).** No call site has a `MetalEnv` in scope. 171 (51%) have a device in scope, 106 (32%) are one `.device()` hop away, and 59 (18%) have no handle.

By effort (see [Low-hanging fruit](#low-hanging-fruit)): 6 sites can be swapped in place, 1 (a test) needs a `t.device()`-style handle added first, 260 need an env route (mostly alignments), and 69 need more work.

Handle columns:

- **In scope**: the enclosing function already has a `MeshDevice`/`IDevice` as a parameter, member, gtest fixture member, or a local extracted from a tensor/buffer before the call.
- **One hop**: no device name in the function, but a `Tensor`, `Buffer` or sibling object is there, so `t.device()` is one call away.
- **No handle**: nothing device-bearing is reachable without adding a parameter or moving the call.

## Findings

- **Only 17 remaining sites are outside tests, ttnn and Python**: 2 in `tt_metal/`, 7 in `tt-train/` sources, 8 in `tools/`. 14 of them already have a `MeshDevice` or `IDevice` (or a `Hal`) in scope; the other 3 are argv parsing in `run_cluster_validation.cpp`. No site has a `MetalEnv` in scope today. `MeshDevice` has no public env accessor yet, though it exposes `arch()` and `allocator()`, and `Allocator::get_alignment(BufferType)` exists.
- **Most callers are ttnn op factories and validators.** 269 of 336 remaining sites are in `ttnn/cpp/ttnn/operations`; 79 of those are in `experimental/quasar` forks of mainline ops. 122 in-scope sites already hold a local `device` extracted from a tensor, buffer or fixture in the same function.
- **24 remaining sites have no `MetalDeviceProperties` getter to move to**: `get_pcie_alignment` (4), `get_noc_max_burst_size_bytes` (2), `get_erisc_l1_unreserved_base` (14), `get_erisc_l1_unreserved_size` (4). The other 312 map to one of the nine getters on `MetalDeviceProperties` (reached through `MetalEnv::get_device_properties()`, declared in `metal_env.hpp`).
- **Python bindings are a separate surface.** `ttnn.device.get_l1_alignment`, `get_dram_alignment`, `get_max_worker_l1_unreserved_size` and `ttnn.bfp_utils.get_*_alignment` bind the free functions with no device argument (5 C++ sites, 15 Python calls in `tests/` and `models/`). The `MeshDevice.sfpu_eps/nan/inf` lambdas already take a device and ignore it.

## Low-hanging fruit

Each site is put in the easiest tier that applies. Tier 1 is 6 sites, tier 2 is 1, tier 3 is 260, and 69 need more work.

| Tier | What the change takes | Sites | Files |
|---|---|--:|--:|
| 1 | Swap in place: a `Hal` object is already in the function. Every `arch` / `arch_name` site with a device in scope has been migrated, so only the `Hal` sites are left (6). | 6 | 5 |
| 2 | Add a handle that is not in scope yet, then swap. Every `arch` site that needed only `tensor.device()` has been migrated; what is left is the test-config site below (1). | 1 | 1 |
| 3 | Needs the env route: the query is on `MetalDeviceProperties` (reached from `MetalEnv`) and not on `MeshDevice` (16), or it is an alignment, which `MeshDevice` exposes only through `allocator()` (244; 144 with a device in scope, 100 one hop away). | 260 | 139 |
| 4 | Everything else; grouped below with a suggested path for each group. | 69 | 36 |

### Where each query is available today

| Query | On `MeshDevice` | On `MetalDeviceProperties` (via `MetalEnv::get_device_properties()`) | On `Hal` |
|---|---|---|---|
| `get_arch` | `device->arch()` | yes | yes |
| `get_arch_name` | `device->arch()`, then a string. The lowercase helper `tt::get_string_lowercase` is in `tt_metal/common`, not the public API. Most callers compare against `"blackhole"` or `"wormhole_b0"` and could compare the enum instead. | yes | yes |
| `get_l1_alignment`, `get_dram_alignment` | `device->allocator()->get_alignment(BufferType::L1 / DRAM)`. The values match `Hal` (the allocator config is filled from `hal.get_alignment(...)`), but `allocator()` throws on a remote-only `MeshDevice`, and `launch()` runs validation and output-spec code on such meshes before its `is_remote_only()` short-circuit. Do not use it as a drop-in. | yes | yes |
| `get_max_worker_l1_unreserved_size`, `get_eps`, `get_nan`, `get_inf`, `get_num_dataflow_buffers` | no | yes | yes |
| `get_pcie_alignment`, `get_noc_max_burst_size_bytes`, `get_erisc_l1_unreserved_base/size` | no | no | yes |

From a `MeshDevice`, `mesh_device->impl().metal_env()` returns the `MetalEnvImpl` (which also has `get_hal()`). `impl()` is public, but `MeshDeviceImpl` and `MetalEnvImpl` are internal headers, so ttnn op code cannot use this path today. tt_metal, tools and tests can; several ttnn gtests already include `impl/context/metal_context.hpp`. A planned follow-up adds `MeshDevice::get_device_properties()`, returning the same `MetalDeviceProperties` handle through this path, which would give ttnn code a public, remote-only-safe route to these queries (see `MetalEnv-API-Review.md` §3).

`MeshDevice::arch()` is safe on remote-only meshes: it reads the arch from the env's cluster, not from local devices.

Of the 6 remaining `Hal`-in-scope sites in tier 1, 1 holds a `Hal` reference in a fixture (`test_kernel_config_buffer.cpp`, initialized from `MetalContext::instance()`); the 3 in `emule_descriptor_builder.cpp` were migrated. The other 5 call `MetalContext::instance().hal()` inline. Swapping to a `Hal` removes the `hal::` call but not the singleton.

### Done on branch `ruizhang/hal-query-tier1`

- 5 `hal::get_arch()` sites became `device->arch()`: `sparse_sdpa_program_factory.cpp`, `conv2d_op_sharded_program_factory.cpp` (mainline and quasar), and two in `ccl_common.cpp`.
- The 3 sites in `emule_descriptor_builder.cpp` use the `Hal` already in the function (`hw.get_alignment(...)`, `hw.get_num_dataflow_buffers()`). It is only built in emulation configurations and was not compiled.
- A first attempt that replaced 98 alignment and arch sites with `device->allocator()->get_alignment(...)` was dropped because of the remote-only problem above.

### Done on branch `ruizhang/hal-query-tier2` (stacked on tier 1)

- 11 of the 12 tier 2 sites now read `tensor.device()->arch()`: the sparse SDPA validator and MSA validator and factory, mainline and quasar `binary_ng` `create_descriptor`, `topk_large_indices`, `prepare_chunk_recurrence`, `upsample`, and `apply_halo_padding` in mainline and quasar `fold.cpp`. All 10 files compile.
- `prepare_chunk_recurrence` now checks the tensor is allocated before reading its device, and `upsample` has an explicit `TT_FATAL` that its input is a device tensor, because `Tensor::device()` returns null for host or unallocated tensors. Inside `launch()`, the inputs are already checked to be allocated device tensors.
- Not done: `TestConfigBuilder::should_skip_test_on_platform` in `tt_fabric_test_config.cpp`. `IDeviceInfoProvider` has no arch accessor, so this needs a new virtual on that interface.

### Done on branch `ruizhang/hal-query-tier2`, continued

- **Relabeled to tier 2 and migrated (12):** the ttnn `arch` sites that were first labeled tier 1 but have no local device variable, only an inline `x.device()` in the function: conv2d sharded factory, conv3d (device op and factory), `attn_res_gather_softmax`, `attn_res_weighted_reduce_nc`, `pack_scaled_fp8_kv_cache`, `indexer_score`, mainline and quasar `generic_pools`, and the `sparse_sdpa` and `sparse_sdpa_msa` frontends. Each already dereferences the same tensor's device elsewhere, so no new null case.
- **`arch_name` string compares (4):** `all_gather_async`, `all_gather_concat`, `all_reduce_async` and `rms_allgather` validators now compare `tensor.device()->arch()` against `tt::ARCH::BLACKHOLE` instead of `hal::get_arch_name() != "blackhole"`.
- **Fabric test infra (3):** added `TestFixture::get_arch()`; `get_max_connections_per_device()` and the two frequency-expectation helpers use it (the helpers are no longer `static`).
- **Reclassified, not migrated:** `test_pgm_dispatch.cpp` (`pgm_dispatch`) reads `arch_name` to pick the trace region size that is then passed to `create_unit_mesh`, so there is no device yet. It is now tier 4, "runs before a device exists".
- **Reclassified, not migrated (2):** `LatencyResultsManager::initialize_results_csv_file` was first migrated to `fixture_.get_arch()` and then reverted after review: it runs from `initialize_latency_results_csv_file()` before `open_devices()` creates `mesh_device_`, so the fixture has no device yet. It is back in the "runs before a device exists" group with the other results managers.

### Tiers per function

| Function | Tier 1 | Tier 2 | Tier 3 | Tier 4 |
|---|--:|--:|--:|--:|
| `hal::get_l1_alignment` | 0 | 0 | 142 | 17 |
| `hal::get_dram_alignment` | 0 | 0 | 102 | 7 |
| `hal::get_arch` | 0 | 0 | 0 | 14 |
| `hal::get_arch_name` | 0 | 1 | 0 | 11 |
| `hal::get_erisc_l1_unreserved_base` | 5 | 0 | 0 | 9 |
| `hal::get_max_worker_l1_unreserved_size` | 1 | 0 | 4 | 1 |
| `hal::get_pcie_alignment` | 0 | 0 | 0 | 4 |
| `hal::get_num_dataflow_buffers` | 0 | 0 | 4 | 0 |
| `hal::get_inf` | 0 | 0 | 4 | 0 |
| `hal::get_erisc_l1_unreserved_size` | 0 | 0 | 0 | 4 |
| `hal::get_noc_max_burst_size_bytes` | 0 | 0 | 0 | 2 |
| `hal::get_nan` | 0 | 0 | 2 | 0 |
| `hal::get_eps` | 0 | 0 | 2 | 0 |
| `hal::get_arch_num_circular_buffers` | 0 | 0 | 0 | 0 |

### Tiers per area

| Area | Tier 1 | Tier 2 | Tier 3 | Tier 4 |
|---|--:|--:|--:|--:|
| `ttnn ops` | 0 | 0 | 234 | 35 |
| `ttnn core/bindings` | 0 | 0 | 9 | 10 |
| `tests` | 5 | 1 | 9 | 16 |
| `tt-train` | 0 | 0 | 7 | 0 |
| `tools` | 1 | 0 | 0 | 7 |
| `tt_metal` | 0 | 0 | 1 | 1 |

### Tier 4, grouped

| Group | Sites | Suggested path |
|---|--:|---|
| Dead code | 2 | `CreateReadKernel` and `CreateWriteKernel` in `moreh_helper_functions.cpp` are `[[maybe_unused]]` and have no callers. Delete them. |
| Handle one level up | 17 | The helper takes no device, but its callers have a tensor or device. Pass the value in from the caller. For alignments, a `const MeshDevice&` alone is not enough because of the remote-only issue; the handle has to reach the env (tier 3). Callers have tensors: `has_subaligned_shard_row`, `page_alignment`, `is_llk_bcast` (binary_ng and ternary `create_descriptor`). Callers have a device: `shrink_shard_for_rm_page_alignment`, `compute_reshape_mapping_host_tensor`, `aligned_page_size`, `default_c_in_block`, `get_pool_input_memory_config`. Most callers have a device, the rest are in the conv L1-usage chain: `get_input_channels_alignment` (9 of 10), `should_coalesce_1d_depthwise_conv_reads` (7 of 9). Test helper: `cursor_align_elems`. |
| Handle two or more levels up | 13 | The conv/pool L1-usage estimators (`get_cb_info`, the three `*_noc_transfer_rate` helpers, `calculate_L1_usage`, `calculate_L1_usage_for_pool2d_slice`, `get_factory_parameters`, and `align_buffer` via `PoolCBSizes::global_cb_total`), `hits_bh_col_bcast_bf16_to_fp32_hang` (via `is_llk_bcast`), and untilize codegen `aligned_tile_page_size` (via `make_tile_cb`). The roots hold a device: `Conv2dSliceAttr` and `Pool2dSliceAttr` have a `MeshDevice* device` member. Instead of passing a device through every level, compute a small struct of arch facts (arch, L1 and DRAM alignment) once at the root and pass that down. |
| Python bindings | 5 | Free-function bindings `ttnn.get_l1_alignment`, `get_dram_alignment`, `get_max_worker_l1_unreserved_size` and `bfp_utils.get_*_alignment` (15 Python calls in `tests/` and `models/`). Either bind them as `MeshDevice` methods, like the existing `sfpu_eps/nan/inf`, or bind `MetalDeviceProperties`. |
| Runs before a device exists | 8 | `ttnn/core/device.cpp` dispatch-config defaulting (2), `test_pgm_dispatch.cpp` arg parsing and trace-region sizing (3), and `run_cluster_validation.cpp` argv defaults (3, ERISC size). Take a `const MetalEnv&`, or move the query after the device is opened. The ERISC ones also need a getter (see the last group). |
| Fabric test result managers | 8 | `ResultsManager`, `BandwidthResultsManager` and `LatencyResultsManager` choose their CSV names in `initialize_*_results_csv_file()`, which runs before `open_devices()` creates the mesh device. They need the arch before any device exists; the fixture is expected to get a `MetalEnv`, which would cover this. |
| No public API for the query; handle in scope; code may use internal headers | 8 | A device is in scope, but no public API has the query: `host_d2h_leg.cpp` (PCIe), `cluster_validation_utils.cpp` (ERISC base, 4), `test_erisc_data_mover_with_workers.cpp` (ERISC base), and `tt_fabric_telemetry_manager.cpp` (ERISC base, 2). These files can include internal headers, so they can read `Hal` via `mesh_device->impl().metal_env().get_hal()` without a new public getter. |
| No public API for the query; ttnn code | 7 | No public API has the query, and ttnn cannot reach `Hal`. H2D/D2H service constructors and `make_completion_layout` need PCIe alignment (3). `concat_s2s_tiled_program_factory.cpp` and `matmul_utilities.cpp` need the NoC max burst (2). `EriscDatamoverConfig` default member initializers need the ERISC region (2). These need new `MetalDeviceProperties` getters. `EriscDatamoverConfig` is only built by `create_erisc_datamover_builder`, and that function's only callers are 3 tests. |
| Static initializer | 1 | `telemetry_addr` is a namespace-scope global in a test header. Make it a function that takes the device; it then joins the internal-headers group. |

## Callers per function

Counts are call sites (a line with two calls counts twice). "Original" is the count found on `3e6669b4b28`; "Remaining" is what is left on this branch. `get_arch_num_circular_buffers` is deprecated (removal after 2026-10-23); its only caller, `emule_descriptor_builder.cpp`, has been migrated.

| Function | Original | Migrated | Remaining | Files | Outside tests/ttnn | In scope | One hop | No handle | Replacement (`MetalDeviceProperties`) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|---|
| `hal::get_l1_alignment` | 160 | 1 | 159 | 112 | 4 | 76 | 66 | 17 | `MetalDeviceProperties::l1_alignment()` |
| `hal::get_dram_alignment` | 110 | 1 | 109 | 74 | 2 | 68 | 34 | 7 | `MetalDeviceProperties::dram_alignment()` |
| `hal::get_arch` | 45 | 31 | 14 | 7 | 0 | 0 | 0 | 14 | `MetalDeviceProperties::arch()` |
| `hal::get_arch_name` | 16 | 4 | 12 | 5 | 0 | 0 | 1 | 11 | `MetalDeviceProperties::arch_name()` |
| `hal::get_erisc_l1_unreserved_base` | 14 | 0 | 14 | 8 | 5 | 11 | 0 | 3 | none |
| `hal::get_max_worker_l1_unreserved_size` | 6 | 0 | 6 | 6 | 0 | 3 | 1 | 2 | `MetalDeviceProperties::max_worker_l1_unreserved_size()` |
| `hal::get_pcie_alignment` | 4 | 0 | 4 | 3 | 1 | 3 | 0 | 1 | none |
| `hal::get_num_dataflow_buffers` | 4 | 0 | 4 | 3 | 0 | 4 | 0 | 0 | `MetalDeviceProperties::num_dataflow_buffers()` |
| `hal::get_inf` | 4 | 0 | 4 | 4 | 2 | 3 | 1 | 0 | `MetalDeviceProperties::sfpu_inf()` |
| `hal::get_erisc_l1_unreserved_size` | 4 | 0 | 4 | 2 | 3 | 0 | 0 | 4 | none |
| `hal::get_noc_max_burst_size_bytes` | 2 | 0 | 2 | 2 | 0 | 1 | 1 | 0 | none |
| `hal::get_nan` | 2 | 0 | 2 | 2 | 0 | 1 | 1 | 0 | `MetalDeviceProperties::sfpu_nan()` |
| `hal::get_eps` | 2 | 0 | 2 | 2 | 0 | 1 | 1 | 0 | `MetalDeviceProperties::sfpu_eps()` |
| `hal::get_arch_num_circular_buffers` | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | `MetalDeviceProperties::num_dataflow_buffers()` |

## Callers by area

| Area | Call sites | Files | In scope | One hop | No handle |
|---|--:|--:|--:|--:|--:|
| `ttnn ops` | 269 | 141 | 132 | 104 | 33 |
| `ttnn core/bindings` | 19 | 7 | 10 | 1 | 8 |
| `tests` | 31 | 20 | 15 | 1 | 15 |
| `tt-train` | 7 | 5 | 7 | 0 | 0 |
| `tools` | 8 | 2 | 5 | 0 | 3 |
| `tt_metal` | 2 | 2 | 2 | 0 | 0 |

## The 59 sites with no handle

| Reason | Sites | Files | Functions |
|---|--:|--:|---|
| no device-bearing argument; a parameter would have to be added | 29 | 18 | arch, dram_alignment, l1_alignment |
| `ResultsManager` / `BandwidthResultsManager` hold no fixture or device member | 7 | 2 | arch_name |
| Python module-level binding of the free function; no device arg in the Python API | 5 | 2 | dram_alignment, l1_alignment, max_worker_l1_unreserved_size |
| no device-bearing argument; caller already passes `tt::ARCH arch` | 3 | 1 | l1_alignment |
| argv parsing, before any device/env exists (natural MetalEnv case) | 3 | 1 | erisc_l1_unreserved_size |
| default member initializer of `EriscDatamoverConfig`; a handle would have to arrive via its constructor | 2 | 1 | erisc_l1_unreserved_base, erisc_l1_unreserved_size |
| dispatch-config defaulting before a MeshDevice exists (natural MetalEnv case) | 2 | 1 | arch |
| runs during arg parsing / init, before the device is opened | 2 | 1 | arch_name |
| free helper called from the ctor; takes only a counter count | 1 | 1 | pcie_alignment |
| runs before the device is created: the arch picks the trace region size passed to `create_unit_mesh` (natural MetalEnv case) | 1 | 1 | arch_name |
| fixture default member initializer; runs before `SetUp()` creates `devices_` | 1 | 1 | max_worker_l1_unreserved_size |
| `LatencyResultsManager` has `fixture_`, but the CSV name is chosen in `initialize_latency_results_csv_file()`, before `open_devices()` creates `mesh_device_` | 1 | 1 | arch_name |
| anonymous-namespace free function; its callers are `CodeProfiler` members with `eth_readback_` | 1 | 1 | erisc_l1_unreserved_base |
| namespace-scope global initializer, runs at static-init time | 1 | 1 | erisc_l1_unreserved_base |

## Every call site

<details><summary><code>hal::get_l1_alignment</code> (159)</summary>

| Location | Enclosing function | Handle | Tier | Evidence |
|---|---|---|---|---|
| `tests/fabric/fabric_data_movement/test_basic_1d_fabric.cpp:632` | run_unicast_test_bw_chips | In scope | 3 | already extracted: `auto sender_device = fixture->get_device(src_physical_device_id);` |
| `tests/fabric/test_infra/tt_fabric_test_common.hpp:351` | get_l1_alignment | In scope | 3 | `TestFixture` member `mesh_device_` |
| `tests/tt_metal/dispatch/dispatch_program/test_sub_device.cpp:86` | TEST_F | In scope | 3 | gtest fixture member (`devices_` / `mesh_device_` / `this->device()`) |
| `tests/ttnn/unit_tests/gtests/tensor/test_h2d_stream_service.cpp:231` | run_h2d_stream_service_case | In scope | 3 | param `tt::tt_metal::distributed::MeshDevice>& mesh_device` |
| `tt-train/sources/ttml/metal/ops/moe_group/device/moe_group_device_operation.cpp:169` | ttml_moe_group | In scope | 3 | already extracted: `auto grid = dispatched.device()->compute_with_storage_grid_size();` |
| `tt-train/sources/ttml/metal/ops/moe_group/device/moe_group_program_factory.cpp:178` | MoeGroupProgramFactory::create | In scope | 3 | already extracted: `auto* device = args.dispatched.device();` |
| `tt-train/sources/ttml/metal/ops/moe_ungroup/device/moe_ungroup_program_factory.cpp:113` | MoeUngroupProgramFactory::create | In scope | 3 | already extracted: `auto* device = args.expert_out.device();` |
| `tt_metal/tools/mem_bench/context.hpp:65` | Context | In scope | 3 | ctor param `devices_` (map of MeshDevice) |
| `ttnn/core/services/d2h_socket_service.cpp:499` | cfg_ | In scope | 3 | ctor param `mesh_device` / member `mesh_device_` |
| `ttnn/core/services/d2h_socket_service.cpp:525` | cfg_ | In scope | 3 | ctor param `mesh_device` / member `mesh_device_` |
| `ttnn/core/services/h2d_socket_service.cpp:484` | cfg_ | In scope | 3 | ctor param `mesh_device` / member `mesh_device_` |
| `ttnn/core/services/h2d_socket_service.cpp:605` | cfg_ | In scope | 3 | ctor param `mesh_device` / member `mesh_device_` |
| `ttnn/core/tensor/d2d_stream_service.cpp:293` | derive_common_plan | One hop | 3 | `const Tensor& backing` -> `.device()` |
| `ttnn/core/tensor/d2d_stream_service.cpp:984` | build_lease_workload | In scope | 3 | param `distributed::MeshDevice>& mesh` |
| `ttnn/ops/ccl/all_gather/all_gather.cpp:108` | use_composite_all_gather | In scope | 3 | already extracted: `const auto& mesh_shape = input_tensor.device()->shape();` |
| `ttnn/ops/ccl/all_to_all_combine/device/all_to_all_combine_program_factory.cpp:79` | build_combine_program_descriptor | In scope | 3 | already extracted: `auto* mesh_device = input_tensor.device();` |
| `ttnn/ops/ccl/all_to_all_dispatch/device/all_to_all_dispatch_program_factory.cpp:317` | build_dispatch_program_descriptor | In scope | 3 | already extracted: `auto* mesh_device = input_tensor.device();` |
| `ttnn/ops/ccl/reduce_to_root/device/reduce_to_root_op.cpp:81` | ReduceToRootOp::validate | In scope | 3 | already extracted: `auto* mesh_device = input_l.device();` |
| `ttnn/ops/ccl/reduce_to_root/device/reduce_to_root_program.cpp:200` | reduce_to_root_program_factory | In scope | 3 | already extracted: `auto* mesh_device = tensor_args.input_tensor_l.device();` |
| `ttnn/ops/copy/typecast/device/typecast_device_op.cpp:125` | TypecastDeviceOperation::validate_on_program_cache_miss | In scope | 3 | already extracted: `!(input_tensor.device()->arch() == tt::ARCH::QUASAR &&` |
| `ttnn/ops/copy/typecast/device/typecast_device_op.cpp:42` | can_use_sharded_optimized_factory | One hop | 3 | `const TypecastInputs& tensor_args` -> `.device()` |
| `ttnn/ops/copy/typecast/device/typecast_sharded_program_factory.cpp:73` | TypecastShardedProgramFactory::create_program_artifacts | In scope | 3 | already extracted: `const auto* device = input.device();` |
| `ttnn/ops/copy/typecast/device/typecast_sharded_program_factory.cpp:75` | TypecastShardedProgramFactory::create_program_artifacts | In scope | 3 | already extracted: `const auto* device = input.device();` |
| `ttnn/ops/data_movement/concat/device/concat_block_sharded_program_factory.cpp:87` | ConcatBlockShardedProgramFactory::create_descriptor | In scope | 3 | already extracted: `auto* device = input_tensors[0].device();` |
| `ttnn/ops/data_movement/fold/device/fold_multi_core_program_factory.cpp:58` | Fold::MultiCore::create_program_artifacts | One hop | 3 | `const tensor_args_t& tensor_args` -> `.device()` |
| `ttnn/ops/data_movement/fold/device/fold_multi_core_program_factory.cpp:59` | Fold::MultiCore::create_program_artifacts | One hop | 3 | `const tensor_args_t& tensor_args` -> `.device()` |
| `ttnn/ops/data_movement/fold/fold.cpp:526` | fold | One hop | 3 | `const ttnn::Tensor& input_tensor_` -> `.device()` |
| `ttnn/ops/data_movement/moe_expert_token_remap/device/moe_expert_token_remap_program_factory.cpp:38` | MoeExpertTokenRemapDeviceOperation::Multicore::create_descriptor | In scope | 3 | already extracted: `auto* mesh_device = topk_tensor.device();` |
| `ttnn/ops/data_movement/moe_routing_remap/device/moe_routing_remap_program_factory.cpp:44` | MoeRoutingRemapDeviceOperation::SingleCore::create_descriptor | One hop | 3 | `const tensor_args_t& tensor_args` -> `.device()` |
| `ttnn/ops/data_movement/move/device/move_overlap_program_factory.cpp:102` | MoveOverlapProgramFactory::create_descriptor | In scope | 3 | already extracted: `const tt::tt_metal::distributed::MeshDevice* device = output.device();` |
| `ttnn/ops/data_movement/move/device/move_overlap_program_factory.cpp:92` | MoveOverlapProgramFactory::create_descriptor | In scope | 3 | already extracted: `const tt::tt_metal::distributed::MeshDevice* device = output.device();` |
| `ttnn/ops/data_movement/move/move.cpp:92` | move_impl | In scope | 3 | already extracted: `auto output_tensor = create_device_tensor(output_tensor_spec, ghost_input_tensor.device());` |
| `ttnn/ops/data_movement/pad/device/pad_device_operation.cpp:188` | PadDeviceOperation::validate_on_program_cache_miss | One hop | 3 | `const tensor_args_t& tensor_args` -> `.device()` |
| `ttnn/ops/data_movement/pad/device/pad_device_operation.cpp:203` | PadDeviceOperation::validate_on_program_cache_miss | One hop | 3 | `const tensor_args_t& tensor_args` -> `.device()` |
| `ttnn/ops/data_movement/pad/device/pad_rm_reader_writer_multi_core_default_program_factory.cpp:64` | PadRmReaderWriterMultiCoreDefaultProgramFactory::create_program_artifacts | In scope | 3 | already extracted: `MeshDevice* device = a.device();` |
| `ttnn/ops/data_movement/pad/device/pad_rm_sharded_width_only_program_factory.cpp:122` | PadRmShardedWidthOnlyProgramFactory::create_program_artifacts | One hop | 3 | `const PadInputs& tensor_args` -> `.device()` |
| `ttnn/ops/data_movement/permute/device/permute_rm_program_factory.cpp:25` | page_size | One hop | 3 | `const ttnn::Tensor& input_tensor` -> `.device()` |
| `ttnn/ops/data_movement/permute/device/permute_tiled_program_factory.cpp:87` | get_buffer_alignment | One hop | 3 | `const ttnn::Tensor& tensor` -> `.device()` |
| `ttnn/ops/data_movement/permute/permute.cpp:248` | permute | One hop | 3 | `const ttnn::Tensor& input_tensor` -> `.device()` |
| `ttnn/ops/data_movement/repeat/device/repeat_utils.cpp:239` | generate_repeat_shard_spec | In scope | 3 | already extracted: `auto* device = input_tensor.device();` |
| `ttnn/ops/data_movement/reshape_on_device/device/reshape_tile_program_factory.cpp:61` | ReshapeTileProgramFactory::create_descriptor | One hop | 3 | `ReshapeOnDeviceInputs& tensor_args` -> `.device()` |
| `ttnn/ops/data_movement/reshape_view/device/reshape_rm_program_factory.cpp:134` | ReshapeViewRMProgramFactory::create_program_artifacts | In scope | 3 | already extracted: `MeshDevice* device = input.device();` |
| `ttnn/ops/data_movement/reshape_view/device/reshape_rm_program_factory.cpp:135` | ReshapeViewRMProgramFactory::create_program_artifacts | In scope | 3 | already extracted: `MeshDevice* device = input.device();` |
| `ttnn/ops/data_movement/roll/device/roll_program_factory.cpp:220` | compute_roll_plan | In scope | 3 | already extracted: `auto* device = input.device();` |
| `ttnn/ops/data_movement/sharded/interleaved_to_sharded/device/interleaved_to_sharded_op.cpp:88` | InterleavedToShardedDeviceOperation::validate_inputs | In scope | 3 | already extracted: `if (output_tensor.device() != input_tensor.device()) {` |
| `ttnn/ops/data_movement/sharded/interleaved_to_sharded/device/interleaved_to_sharded_program_factory.cpp:113` | InterleavedToShardedProgramFactory::create_program_artifacts | In scope | 3 | already extracted: `bool is_blackhole = (input.device()->arch() == tt::ARCH::BLACKHOLE);` |
| `ttnn/ops/data_movement/sharded/interleaved_to_sharded/device/interleaved_to_sharded_program_factory.cpp:165` | InterleavedToShardedProgramFactory::create_program_artifacts | In scope | 3 | already extracted: `bool is_blackhole = (input.device()->arch() == tt::ARCH::BLACKHOLE);` |
| `ttnn/ops/data_movement/sharded/reshard/device/reshard_device_operation.cpp:202` | ReshardDeviceOperation::validate_inputs | One hop | 3 | `const tensor_args_t& tensor_args` -> `.device()` |
| `ttnn/ops/data_movement/sharded/sharded_to_interleaved/device/sharded_to_interleaved_device_operation.cpp:62` | ShardedToInterleavedDeviceOperation::validate_inputs | In scope | 3 | already extracted: `if (output_tensor.device() != input_tensor.device()) {` |
| `ttnn/ops/data_movement/sharded/sharded_to_interleaved/device/sharded_to_interleaved_program_factory.cpp:244` | ShardedToInterleavedProgramFactory::create_program_artifacts | In scope | 3 | already extracted: `bool is_blackhole = (input.device()->arch() == tt::ARCH::BLACKHOLE);` |
| `ttnn/ops/data_movement/sharded_partial/interleaved_to_sharded_partial/device/interleaved_to_sharded_partial_program_factory.cpp:130` | InterleavedToShardedPartialProgramFactory::create_descriptor | In scope | 3 | already extracted: `bool is_blackhole = (input.device()->arch() == tt::ARCH::BLACKHOLE);` |
| `ttnn/ops/data_movement/sharded_partial/interleaved_to_sharded_partial/device/interleaved_to_sharded_partial_program_factory.cpp:171` | InterleavedToShardedPartialProgramFactory::create_descriptor | In scope | 3 | already extracted: `bool is_blackhole = (input.device()->arch() == tt::ARCH::BLACKHOLE);` |
| `ttnn/ops/data_movement/sharded_partial/sharded_to_interleaved_partial/device/sharded_to_interleaved_partial_program_factory.cpp:200` | ShardedToInterleavedPartialProgramFactory::create_descriptor | In scope | 3 | already extracted: `bool is_blackhole = (input.device()->arch() == tt::ARCH::BLACKHOLE);` |
| `ttnn/ops/data_movement/slice/device/slice_device_operation.cpp:329` | SliceDeviceOperation::select_program_factory | One hop | 3 | `const tensor_args_t& tensor_args` -> `.device()` |
| `ttnn/ops/data_movement/slice/device/slice_program_factory_rm.cpp:190` | compute_dfb_size | One hop | 3 | `const Tensor& input` -> `.device()` |
| `ttnn/ops/data_movement/slice/device/slice_program_factory_rm.cpp:193` | compute_dfb_size | One hop | 3 | `const Tensor& input` -> `.device()` |
| `ttnn/ops/data_movement/slice/device/slice_program_factory_rm.cpp:77` | get_slice_runtime_args_rm | One hop | 3 | `const Tensor& input_tensor` -> `.device()` |
| `ttnn/ops/data_movement/slice/device/slice_program_factory_rm.cpp:80` | get_slice_runtime_args_rm | One hop | 3 | `const Tensor& input_tensor` -> `.device()` |
| `ttnn/ops/data_movement/slice/device/slice_program_factory_rm_sharded.cpp:282` | SliceRmShardedProgramFactory::create_program_artifacts | One hop | 3 | `const SliceInputs& tensor_args` -> `.device()` |
| `ttnn/ops/data_movement/tilize/device/tilize_device_operation.cpp:249` | TilizeDeviceOperation::validate_on_program_cache_miss | In scope | 3 | already extracted: `!(input_tensor_a.device()->arch() == tt::ARCH::QUASAR && input_tensor_a.dtype() == DataType::FLOAT32 &&` |
| `ttnn/ops/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_device_operation.cpp:147` | TilizeWithValPaddingDeviceOperation::validate_on_program_cache_miss | One hop | 3 | `const Tensor& input_tensor` -> `.device()` |
| `ttnn/ops/experimental/ccl/all_to_all_dispatch_metadata/device/all_to_all_dispatch_metadata_program_factory.cpp:506` | AllToAllDispatchMetadataDeviceOperation::AllToAllDispatchMetadataSparse::create_at | In scope | 3 | already extracted: `auto* mesh_device = input_tensor.device();` |
| `ttnn/ops/experimental/ccl/moe/selective_reduce_combine/device/selective_reduce_combine_device_operation.cpp:60` | SelectiveReduceCombineDeviceOperation::validate_on_program_cache_miss | One hop | 3 | `const tensor_args_t& tensor_args` -> `.device()` |
| `ttnn/ops/experimental/ccl/moe/selective_reduce_combine/device/selective_reduce_combine_device_operation.cpp:89` | SelectiveReduceCombineDeviceOperation::validate_on_program_cache_miss | One hop | 3 | `const tensor_args_t& tensor_args` -> `.device()` |
| `ttnn/ops/experimental/ccl/moe/selective_reduce_combine/device/selective_reduce_combine_device_operation.cpp:93` | SelectiveReduceCombineDeviceOperation::validate_on_program_cache_miss | One hop | 3 | `const tensor_args_t& tensor_args` -> `.device()` |
| `ttnn/ops/experimental/ccl/moe/selective_reduce_combine/device/selective_reduce_combine_program_factory.cpp:416` | build_selective_reduce_combine_program_artifacts | In scope | 3 | already extracted: `auto* mesh_device = input_tensor.device();` |
| `ttnn/ops/experimental/ccl/moe_compute/device/moe_compute_device_operation.cpp:224` | MoEComputeDeviceOperation::compute_output_specs | In scope | 3 | already extracted: `auto* mesh_device = tilize_input_tensor.device();` |
| `ttnn/ops/experimental/ccl/moe_compute/device/moe_compute_program_factory.cpp:239` | MoEComputeMeshWorkloadFactory::create_at | In scope | 3 | already extracted: `auto* mesh_device = tilize_input_tensor.device();` |
| `ttnn/ops/experimental/ccl/moe_gpt/device/moe_gpt_device_operation.cpp:111` | MoEGPTDeviceOperation::compute_output_specs | In scope | 3 | already extracted: `auto* device = tensor_args.w0_w1_tensor.device();` |
| `ttnn/ops/experimental/ccl/moe_gpt/device/moe_gpt_program_factory.cpp:437` | MoEGPTMeshWorkloadFactory::create_at | In scope | 3 | already extracted: `auto* device = tensor_args.input_tensor.device();` |
| `ttnn/ops/experimental/cnn/convert_to_hwc/device/convert_to_hwc_program_factory.cpp:356` | compute_alignment_requirement_in_elements | One hop | 3 | `const Tensor& input_tensor` -> `.device()` |
| `ttnn/ops/experimental/conv3d/device/conv3d_device_operation.cpp:229` | Conv3dDeviceOperation::validate_on_program_cache_miss | In scope | 3 | already extracted: `const auto& device_grid = input_tensor_a.device()->compute_with_storage_grid_size();` |
| `ttnn/ops/experimental/conv3d/device/conv3d_device_operation.cpp:66` | Conv3dDeviceOperation::validate_on_program_cache_miss | In scope | 3 | already extracted: `const auto& device_grid = input_tensor_a.device()->compute_with_storage_grid_size();` |
| `ttnn/ops/experimental/deepseek_prefill/combine/device/combine_program_factory.cpp:172` | build_program_for_coord | In scope | 3 | already extracted: `auto* mesh_device = dispatched_buffer.device();` |
| `ttnn/ops/experimental/deepseek_prefill/dispatch/device/dispatch_program_factory.cpp:268` | create_dispatch_program | In scope | 3 | already extracted: `auto* mesh_device = input_tensor.device();` |
| `ttnn/ops/experimental/deepseek_prefill/hybrid_routed_expert_ffn/device/hybrid_half_merge.cpp:88` | overlay_circular_buffers | One hop | 3 | `Buffer* l1_arena` -> `->device()` (check for null; arena is optional when the fused pass is off) |
| `ttnn/ops/experimental/deepseek_prefill/per_token_cast_back/device/per_token_cast_back_program_factory.cpp:242` | PerTokenCastBackProgramFactory::create | In scope | 3 | already extracted: `auto* device = input_e4m3.device();` |
| `ttnn/ops/experimental/padded_slice/device/padded_slice_rm_program_factory.cpp:156` | get_padded_slice_runtime_args_rm_sharded_output | One hop | 3 | `const Tensor& input_tensor` -> `.device()` |
| `ttnn/ops/experimental/padded_slice/device/padded_slice_rm_program_factory.cpp:295` | PaddedSliceRMProgramFactory::create_descriptor | One hop | 3 | `const PaddedSliceInputs& tensor_args` -> `.device()` |
| `ttnn/ops/experimental/padded_slice/device/padded_slice_rm_program_factory.cpp:298` | PaddedSliceRMProgramFactory::create_descriptor | One hop | 3 | `const PaddedSliceInputs& tensor_args` -> `.device()` |
| `ttnn/ops/experimental/padded_slice/device/padded_slice_rm_program_factory.cpp:57` | padded_slice_rm_reader_address | One hop | 3 | `const Tensor& input_tensor` -> `.device()` |
| `ttnn/ops/experimental/padded_slice/device/padded_slice_rm_program_factory.cpp:59` | padded_slice_rm_reader_address | One hop | 3 | `const Tensor& input_tensor` -> `.device()` |
| `ttnn/ops/experimental/padded_slice/device/padded_slice_tile_program_factory.cpp:435` | PaddedSliceTileProgramFactory::create_descriptor | One hop | 3 | `const PaddedSliceInputs& tensor_args` -> `.device()` |
| `ttnn/ops/experimental/plusone/device/plusone_program_factory.cpp:50` | PlusOneProgramFactory::create_program_artifacts | One hop | 3 | `const Tensor& input` -> `.device()` |
| `ttnn/ops/experimental/quasar/fold/device/fold_multi_core_program_factory.cpp:48` | Fold::MultiCore::create_program_artifacts | One hop | 3 | `const tensor_args_t& tensor_args` -> `.device()` |
| `ttnn/ops/experimental/quasar/fold/device/fold_multi_core_program_factory.cpp:49` | Fold::MultiCore::create_program_artifacts | One hop | 3 | `const tensor_args_t& tensor_args` -> `.device()` |
| `ttnn/ops/experimental/quasar/interleaved_to_sharded/device/interleaved_to_sharded_op.cpp:91` | InterleavedToShardedDeviceOperation::validate_inputs | In scope | 3 | already extracted: `if (output_tensor.device() != input_tensor.device()) {` |
| `ttnn/ops/experimental/quasar/interleaved_to_sharded/device/interleaved_to_sharded_program_factory.cpp:134` | InterleavedToShardedProgramFactory::create_program_artifacts | In scope | 3 | already extracted: `bool is_blackhole = (input.device()->arch() == tt::ARCH::BLACKHOLE);` |
| `ttnn/ops/experimental/quasar/interleaved_to_sharded/device/interleaved_to_sharded_program_factory.cpp:144` | InterleavedToShardedProgramFactory::create_program_artifacts | In scope | 3 | already extracted: `bool is_blackhole = (input.device()->arch() == tt::ARCH::BLACKHOLE);` |
| `ttnn/ops/experimental/quasar/move/device/move_overlap_program_factory.cpp:111` | MoveOverlapProgramFactory::create_program_artifacts | In scope | 3 | already extracted: `const tt::tt_metal::distributed::MeshDevice* device = output.device();` |
| `ttnn/ops/experimental/quasar/move/device/move_overlap_program_factory.cpp:115` | MoveOverlapProgramFactory::create_program_artifacts | In scope | 3 | already extracted: `const tt::tt_metal::distributed::MeshDevice* device = output.device();` |
| `ttnn/ops/experimental/quasar/move/move.cpp:92` | move_impl | In scope | 3 | already extracted: `auto output_tensor = create_device_tensor(output_tensor_spec, ghost_input_tensor.device());` |
| `ttnn/ops/experimental/quasar/pad/device/pad_device_operation.cpp:188` | PadDeviceOperation::validate_on_program_cache_miss | One hop | 3 | `const tensor_args_t& tensor_args` -> `.device()` |
| `ttnn/ops/experimental/quasar/pad/device/pad_device_operation.cpp:203` | PadDeviceOperation::validate_on_program_cache_miss | One hop | 3 | `const tensor_args_t& tensor_args` -> `.device()` |
| `ttnn/ops/experimental/quasar/pad/device/pad_rm_reader_writer_multi_core_default_program_factory.cpp:74` | PadRmReaderWriterMultiCoreDefaultProgramFactory::create_program_artifacts | In scope | 3 | already extracted: `MeshDevice* device = a.device();` |
| `ttnn/ops/experimental/quasar/pad/device/pad_rm_sharded_width_only_program_factory.cpp:97` | PadRmShardedWidthOnlyProgramFactory::create_program_artifacts | One hop | 3 | `const PadInputs& tensor_args` -> `.device()` |
| `ttnn/ops/experimental/quasar/padded_slice/device/padded_slice_rm_program_factory.cpp:226` | PaddedSliceRMProgramFactory::create_program_artifacts | One hop | 3 | `const PaddedSliceInputs& tensor_args` -> `.device()` |
| `ttnn/ops/experimental/quasar/padded_slice/device/padded_slice_rm_program_factory.cpp:229` | PaddedSliceRMProgramFactory::create_program_artifacts | One hop | 3 | `const PaddedSliceInputs& tensor_args` -> `.device()` |
| `ttnn/ops/experimental/quasar/padded_slice/device/padded_slice_rm_program_factory.cpp:86` | get_padded_slice_runtime_args_rm_sharded_output | One hop | 3 | `const Tensor& input_tensor` -> `.device()` |
| `ttnn/ops/experimental/quasar/padded_slice/device/padded_slice_rm_program_factory.cpp:89` | get_padded_slice_runtime_args_rm_sharded_output | One hop | 3 | `const Tensor& input_tensor` -> `.device()` |
| `ttnn/ops/experimental/quasar/padded_slice/device/padded_slice_tile_program_factory.cpp:420` | PaddedSliceTileProgramFactory::create | In scope | 3 | already extracted: `tt::tt_metal::distributed::MeshDevice* device = a.device();` |
| `ttnn/ops/experimental/quasar/reshape_view/device/reshape_rm_program_factory.cpp:131` | ReshapeViewRMProgramFactory::create_descriptor | In scope | 3 | already extracted: `MeshDevice* device = input.device();` |
| `ttnn/ops/experimental/quasar/reshape_view/device/reshape_rm_program_factory.cpp:132` | ReshapeViewRMProgramFactory::create_descriptor | In scope | 3 | already extracted: `MeshDevice* device = input.device();` |
| `ttnn/ops/experimental/quasar/reshard/device/reshard_device_operation.cpp:174` | ReshardDeviceOperation::validate_inputs | One hop | 3 | `const tensor_args_t& tensor_args` -> `.device()` |
| `ttnn/ops/experimental/quasar/sharded_to_interleaved/device/sharded_to_interleaved_device_operation.cpp:62` | ShardedToInterleavedDeviceOperation::validate_inputs | In scope | 3 | already extracted: `if (output_tensor.device() != input_tensor.device()) {` |
| `ttnn/ops/experimental/quasar/sharded_to_interleaved/device/sharded_to_interleaved_program_factory.cpp:331` | ShardedToInterleavedProgramFactory::create_program_artifacts | In scope | 3 | already extracted: `bool is_blackhole = (input.device()->arch() == tt::ARCH::BLACKHOLE);` |
| `ttnn/ops/experimental/quasar/slice/device/slice_device_operation.cpp:283` | SliceDeviceOperation::select_program_factory | One hop | 3 | `const tensor_args_t& tensor_args` -> `.device()` |
| `ttnn/ops/experimental/quasar/slice/device/slice_program_factory_rm.cpp:187` | compute_cb_size | One hop | 3 | `const Tensor& input` -> `.device()` |
| `ttnn/ops/experimental/quasar/slice/device/slice_program_factory_rm.cpp:190` | compute_cb_size | One hop | 3 | `const Tensor& input` -> `.device()` |
| `ttnn/ops/experimental/quasar/slice/device/slice_program_factory_rm.cpp:93` | get_slice_runtime_args_rm | One hop | 3 | `const Tensor& input_tensor` -> `.device()` |
| `ttnn/ops/experimental/quasar/slice/device/slice_program_factory_rm.cpp:96` | get_slice_runtime_args_rm | One hop | 3 | `const Tensor& input_tensor` -> `.device()` |
| `ttnn/ops/experimental/quasar/slice/device/slice_program_factory_rm_sharded.cpp:228` | SliceRmShardedProgramFactory::create_program_artifacts | One hop | 3 | `const SliceInputs& tensor_args` -> `.device()` |
| `ttnn/ops/experimental/quasar/slice_write/device/slice_write_rm_interleaved_program_factory.cpp:103` | get_slice_write_runtime_args_rm | One hop | 3 | `const Tensor& input_tensor` -> `.device()` |
| `ttnn/ops/experimental/quasar/slice_write/device/slice_write_rm_interleaved_program_factory.cpp:222` | SliceWriteRMInterleavedProgramFactory::create | In scope | 3 | already extracted: `tt::tt_metal::distributed::MeshDevice* device = input.device();` |
| `ttnn/ops/experimental/quasar/slice_write/device/slice_write_rm_interleaved_program_factory.cpp:225` | SliceWriteRMInterleavedProgramFactory::create | In scope | 3 | already extracted: `tt::tt_metal::distributed::MeshDevice* device = input.device();` |
| `ttnn/ops/experimental/quasar/slice_write/device/slice_write_rm_sharded_input_program_factory.cpp:109` | get_slice_write_runtime_args_rm_sharded_input | One hop | 3 | `const Tensor& input_tensor` -> `.device()` |
| `ttnn/ops/experimental/quasar/slice_write/device/slice_write_rm_sharded_input_program_factory.cpp:241` | SliceWriteRMShardedInputProgramFactory::create_program_artifacts | One hop | 3 | `const SliceWriteInputs& tensor_args` -> `.device()` |
| `ttnn/ops/experimental/quasar/tilize/device/tilize_device_operation.cpp:146` | TilizeDeviceOperation::validate_on_program_cache_miss | One hop | 3 | `tensor_args_t& tensor_args` -> `.device()` |
| `ttnn/ops/experimental/quasar/tilize_with_val_padding/device/tilize_with_val_padding_device_operation.cpp:143` | TilizeWithValPaddingDeviceOperation::validate_on_program_cache_miss | One hop | 3 | `const Tensor& input_tensor` -> `.device()` |
| `ttnn/ops/experimental/quasar/typecast/device/typecast_device_op.cpp:113` | TypecastDeviceOperation::validate_on_program_cache_miss | One hop | 3 | `const TypecastInputs& tensor_args` -> `.device()` |
| `ttnn/ops/experimental/quasar/typecast/device/typecast_device_op.cpp:42` | can_use_sharded_optimized_factory | One hop | 3 | `const TypecastInputs& tensor_args` -> `.device()` |
| `ttnn/ops/experimental/quasar/typecast/device/typecast_sharded_program_factory.cpp:70` | TypecastShardedProgramFactory::create_descriptor | One hop | 3 | `const TypecastInputs& tensor_args` -> `.device()` |
| `ttnn/ops/experimental/quasar/typecast/device/typecast_sharded_program_factory.cpp:72` | TypecastShardedProgramFactory::create_descriptor | One hop | 3 | `const TypecastInputs& tensor_args` -> `.device()` |
| `ttnn/ops/experimental/reduction/deepseek_moe_fast_reduce_nc_fused/device/deepseek_moe_fast_reduce_nc_fused_program_factory.cpp:164` | build_program_descriptor | In scope | 3 | already extracted: `auto* device = tensor_args.input_tensor.device();` |
| `ttnn/ops/experimental/reduction/deepseek_moe_fast_reduce_nc_fused/device/deepseek_moe_fast_reduce_nc_fused_program_factory.cpp:168` | build_program_descriptor | In scope | 3 | already extracted: `auto* device = tensor_args.input_tensor.device();` |
| `ttnn/ops/experimental/reduction/deepseek_moe_fast_reduce_nc_fused/device/deepseek_moe_fast_reduce_nc_fused_program_factory.cpp:170` | build_program_descriptor | In scope | 3 | already extracted: `auto* device = tensor_args.input_tensor.device();` |
| `ttnn/ops/experimental/rgb_to_yuv/device/rgb_to_yuv_program_factory.cpp:156` | RgbToYuvProgramFactory::create | In scope | 3 | already extracted: `const auto* device = input.device();` |
| `ttnn/ops/experimental/slice_write/device/slice_write_rm_interleaved_program_factory.cpp:123` | get_slice_write_runtime_args_rm | One hop | 3 | `const Tensor& input_tensor` -> `.device()` |
| `ttnn/ops/experimental/slice_write/device/slice_write_rm_interleaved_program_factory.cpp:234` | SliceWriteRMInterleavedProgramFactory::create_descriptor | In scope | 3 | already extracted: `tt::tt_metal::IDevice* device = input.device();` |
| `ttnn/ops/experimental/slice_write/device/slice_write_rm_interleaved_program_factory.cpp:236` | SliceWriteRMInterleavedProgramFactory::create_descriptor | In scope | 3 | already extracted: `tt::tt_metal::IDevice* device = input.device();` |
| `ttnn/ops/experimental/slice_write/device/slice_write_rm_sharded_input_program_factory.cpp:136` | get_slice_write_runtime_args_rm_sharded_input | One hop | 3 | `const Tensor& input_tensor` -> `.device()` |
| `ttnn/ops/experimental/slice_write/device/slice_write_rm_sharded_input_program_factory.cpp:263` | SliceWriteRMShardedInputProgramFactory::create_descriptor | One hop | 3 | `const SliceWriteInputs& tensor_args` -> `.device()` |
| `ttnn/ops/point_to_point/device/host/local_copy_program_factory.cpp:25` | local_copy_program_factory | In scope | 3 | already extracted: `auto* mesh_device = dynamic_cast<MeshDevice*>(input_tensor.device());` |
| `ttnn/ops/point_to_point/device/host/point_to_point_device_op.cpp:135` | PointToPointOp::validate | In scope | 3 | already extracted: `auto* mesh_device = input_tensor.device();` |
| `ttnn/ops/point_to_point/device/host/point_to_point_device_op.cpp:168` | PointToPointOp::compute_output_specs | One hop | 3 | `const tensor_args_t& tensor_args` -> `.device()` |
| `ttnn/ops/point_to_point/device/host/receive_program_factory.cpp:29` | receive_program_factory | In scope | 3 | already extracted: `auto* mesh_device = dynamic_cast<MeshDevice*>(output_tensors.at(0).device());` |
| `ttnn/ops/point_to_point/device/host/send_program_factory.cpp:30` | send_program_factory | In scope | 3 | already extracted: `auto* mesh_device = dynamic_cast<MeshDevice*>(tensor_args.input_tensor.device());` |
| `ttnn/ops/pool/grid_sample/device/grid_sample_utils.cpp:72` | get_aligned_stick_size | One hop | 3 | `const Tensor& tensor` -> `.device()` |
| `ttnn/ops/pool/pool_utils.cpp:614` | get_aligned_stick_size | One hop | 3 | `const Tensor& tensor` -> `.device()` |
| `ttnn/ops/prefetcher/prefetcher/device/dram_prefetcher_program_factory.cpp:133` | DramPrefetcherOperation::create_descriptor | One hop | 3 | `const tensor_args_t& tensor_args` -> `.device()` |
| `ttnn/ops/prefetcher/prefetcher/device/dram_prefetcher_program_factory.cpp:144` | DramPrefetcherOperation::create_descriptor | One hop | 3 | `const tensor_args_t& tensor_args` -> `.device()` |
| `ttnn/ops/reduction/argmax/device/argmax_multi_core_program_factory.cpp:224` | ArgMaxMultiCoreProgramFactory::create_program_artifacts | In scope | 3 | already extracted: `const tt::tt_metal::distributed::MeshDevice& device = output.device();` |
| `tt-train/tests/ops/moe_group_op_test.cpp:43` | cursor_align_elems | No handle | 4: Handle one level up | no device-bearing argument; a parameter would have to be added |
| `ttnn/cpp/ttnn-nanobind/bfp_utils.cpp:127` | py_module | No handle | 4: Python bindings | Python module-level binding of the free function; no device arg in the Python API |
| `ttnn/cpp/ttnn-nanobind/device.cpp:734` | device_module | No handle | 4: Python bindings | Python module-level binding of the free function; no device arg in the Python API |
| `ttnn/ops/conv/conv2d/conv2d_op_program_factory_common.cpp:383` | get_local_l1_noc_transfer_rate | No handle | 4: Handle two or more levels up | no device-bearing argument; caller already passes `tt::ARCH arch` |
| `ttnn/ops/conv/conv2d/conv2d_op_program_factory_common.cpp:421` | get_all_dram_noc_transfer_rate | No handle | 4: Handle two or more levels up | no device-bearing argument; caller already passes `tt::ARCH arch` |
| `ttnn/ops/conv/conv2d/conv2d_op_program_factory_common.cpp:459` | get_mcast_many_l1_linked_noc_transfer_rate | No handle | 4: Handle two or more levels up | no device-bearing argument; caller already passes `tt::ARCH arch` |
| `ttnn/ops/conv/conv2d/conv2d_utils.cpp:120` | get_input_channels_alignment | No handle | 4: Handle one level up | no device-bearing argument; a parameter would have to be added |
| `ttnn/ops/data_movement/common/synthesize_output_shard_spec.cpp:168` | shrink_shard_for_rm_page_alignment | No handle | 4: Handle one level up | no device-bearing argument; a parameter would have to be added |
| `ttnn/ops/data_movement/repeat_interleave/codegen/repeat_interleave_codegen_supported.cpp:25` | page_alignment | No handle | 4: Handle one level up | no device-bearing argument; a parameter would have to be added |
| `ttnn/ops/data_movement/reshape_view/device/reshape_tiled_program_factory.cpp:232` | compute_reshape_mapping_host_tensor | No handle | 4: Handle one level up | no device-bearing argument; a parameter would have to be added |
| `ttnn/ops/data_movement/slice/slice.cpp:53` | has_subaligned_shard_row | No handle | 4: Handle one level up | no device-bearing argument; a parameter would have to be added |
| `ttnn/ops/experimental/conv3d/prepare_conv3d_weights.cpp:33` | default_c_in_block | No handle | 4: Handle one level up | no device-bearing argument; a parameter would have to be added |
| `ttnn/ops/experimental/multi_scale_deformable_attn/device/multi_scale_deformable_attn_program_factory.cpp:30` | aligned_page_size | No handle | 4: Handle one level up | no device-bearing argument; a parameter would have to be added |
| `ttnn/ops/experimental/quasar/pool_generic/generic_pools.cpp:62` | get_pool_input_memory_config | No handle | 4: Handle one level up | no device-bearing argument; a parameter would have to be added |
| `ttnn/ops/experimental/quasar/reshape_view/device/reshape_tiled_program_factory.cpp:233` | compute_reshape_mapping_host_tensor | No handle | 4: Handle one level up | no device-bearing argument; a parameter would have to be added |
| `ttnn/ops/pool/generic/generic_pools.cpp:59` | get_pool_input_memory_config | No handle | 4: Handle one level up | no device-bearing argument; a parameter would have to be added |
| `ttnn/ops/pool/pool_utils.cpp:713` | calculate_L1_usage_for_pool2d_slice | No handle | 4: Handle two or more levels up | no device-bearing argument; a parameter would have to be added |

</details>

<details><summary><code>hal::get_dram_alignment</code> (109)</summary>

| Location | Enclosing function | Handle | Tier | Evidence |
|---|---|---|---|---|
| `tt-train/sources/ttml/metal/ops/moe_group/device/moe_group_program_factory.cpp:150` | MoeGroupProgramFactory::create | In scope | 3 | already extracted: `auto* device = args.dispatched.device();` |
| `tt-train/sources/ttml/metal/ops/moe_ungroup/device/moe_ungroup_program_factory.cpp:159` | MoeUngroupProgramFactory::create | In scope | 3 | already extracted: `auto* device = args.expert_out.device();` |
| `ttnn/ops/ccl/all_gather/all_gather.cpp:107` | use_composite_all_gather | In scope | 3 | already extracted: `const auto& mesh_shape = input_tensor.device()->shape();` |
| `ttnn/ops/ccl/all_to_all_combine/device/all_to_all_combine_program_factory.cpp:80` | build_combine_program_descriptor | In scope | 3 | already extracted: `auto* mesh_device = input_tensor.device();` |
| `ttnn/ops/core/to_memory_config/to_memory_config_op.cpp:105` | can_use_interleaved_to_sharded | In scope | 3 | already extracted: `MeshDevice* device = input_tensor.device();` |
| `ttnn/ops/data_movement/common/common.cpp:992` | make_block_plan | In scope | 3 | already extracted: `const tt::tt_metal::CoreCoord grid_size = input_tensor.device()->compute_with_storage_grid_size();` |
| `ttnn/ops/data_movement/fold/device/fold_multi_core_dram_program_factory.cpp:324` | fold_multi_core_row_major_interleaved | In scope | 3 | already extracted: `auto* device = input_tensor.device();` |
| `ttnn/ops/data_movement/fold/device/fold_multi_core_dram_program_factory.cpp:331` | fold_multi_core_row_major_interleaved | In scope | 3 | already extracted: `auto* device = input_tensor.device();` |
| `ttnn/ops/data_movement/fold/fold.cpp:526` | fold | One hop | 3 | `const ttnn::Tensor& input_tensor_` -> `.device()` |
| `ttnn/ops/data_movement/pad/device/pad_rm_reader_writer_multi_core_default_program_factory.cpp:130` | PadRmReaderWriterMultiCoreDefaultProgramFactory::create_program_artifacts | In scope | 3 | already extracted: `MeshDevice* device = a.device();` |
| `ttnn/ops/data_movement/pad/device/pad_rm_reader_writer_multi_core_default_program_factory.cpp:65` | PadRmReaderWriterMultiCoreDefaultProgramFactory::create_program_artifacts | In scope | 3 | already extracted: `MeshDevice* device = a.device();` |
| `ttnn/ops/data_movement/permute/device/permute_rm_program_factory.cpp:24` | page_size | One hop | 3 | `const ttnn::Tensor& input_tensor` -> `.device()` |
| `ttnn/ops/data_movement/permute/device/permute_tiled_program_factory.cpp:86` | get_buffer_alignment | One hop | 3 | `const ttnn::Tensor& tensor` -> `.device()` |
| `ttnn/ops/data_movement/reshape_on_device/device/reshape_tile_program_factory.cpp:61` | ReshapeTileProgramFactory::create_descriptor | One hop | 3 | `ReshapeOnDeviceInputs& tensor_args` -> `.device()` |
| `ttnn/ops/data_movement/reshape_view/device/reshape_rm_program_factory.cpp:134` | ReshapeViewRMProgramFactory::create_program_artifacts | In scope | 3 | already extracted: `MeshDevice* device = input.device();` |
| `ttnn/ops/data_movement/roll/device/roll_program_factory.cpp:103` | compute_roll_plan | In scope | 3 | already extracted: `auto* device = input.device();` |
| `ttnn/ops/data_movement/roll/device/roll_program_factory.cpp:225` | compute_roll_plan | In scope | 3 | already extracted: `auto* device = input.device();` |
| `ttnn/ops/data_movement/sharded/interleaved_to_sharded/device/interleaved_to_sharded_program_factory.cpp:164` | InterleavedToShardedProgramFactory::create_program_artifacts | In scope | 3 | already extracted: `bool is_blackhole = (input.device()->arch() == tt::ARCH::BLACKHOLE);` |
| `ttnn/ops/data_movement/sharded_partial/interleaved_to_sharded_partial/device/interleaved_to_sharded_partial_program_factory.cpp:170` | InterleavedToShardedPartialProgramFactory::create_descriptor | In scope | 3 | already extracted: `bool is_blackhole = (input.device()->arch() == tt::ARCH::BLACKHOLE);` |
| `ttnn/ops/data_movement/slice/device/slice_program_factory_rm.cpp:189` | compute_dfb_size | One hop | 3 | `const Tensor& input` -> `.device()` |
| `ttnn/ops/data_movement/slice/device/slice_program_factory_rm.cpp:192` | compute_dfb_size | One hop | 3 | `const Tensor& input` -> `.device()` |
| `ttnn/ops/data_movement/slice/device/slice_program_factory_rm.cpp:76` | get_slice_runtime_args_rm | One hop | 3 | `const Tensor& input_tensor` -> `.device()` |
| `ttnn/ops/data_movement/slice/device/slice_program_factory_rm.cpp:79` | get_slice_runtime_args_rm | One hop | 3 | `const Tensor& input_tensor` -> `.device()` |
| `ttnn/ops/data_movement/tilize/device/tilize_multi_core_block_program_factory.cpp:55` | TilizeMultiCoreBlockProgramFactory::create_program_artifacts | In scope | 3 | already extracted: `IDevice* device = a.device();` |
| `ttnn/ops/data_movement/tilize/tilize.cpp:78` | tilize | In scope | 3 | already extracted: `return create_device_tensor(spec, input_tensor.device());` |
| `ttnn/ops/data_movement/tilize_with_val_padding/device/factories/tilize_with_val_padding_multi_core_block_interleaved_program_factory.cpp:84` | TilizeWithValPaddingMultiCoreBlockInterleavedFactory::create_program_artifacts | In scope | 3 | already extracted: `MeshDevice* device = a.device();` |
| `ttnn/ops/data_movement/tilize_with_val_padding/tilize_with_val_padding.cpp:108` | tilize_with_val_padding | In scope | 3 | already extracted: `return create_device_tensor(spec, input_tensor.device());` |
| `ttnn/ops/experimental/ccl/all_to_all_dispatch_metadata/device/all_to_all_dispatch_metadata_program_factory.cpp:507` | AllToAllDispatchMetadataDeviceOperation::AllToAllDispatchMetadataSparse::create_at | In scope | 3 | already extracted: `auto* mesh_device = input_tensor.device();` |
| `ttnn/ops/experimental/ccl/moe/selective_reduce_combine/device/selective_reduce_combine_device_operation.cpp:61` | SelectiveReduceCombineDeviceOperation::validate_on_program_cache_miss | One hop | 3 | `const tensor_args_t& tensor_args` -> `.device()` |
| `ttnn/ops/experimental/ccl/moe_compute/device/moe_compute_program_factory.cpp:240` | MoEComputeMeshWorkloadFactory::create_at | In scope | 3 | already extracted: `auto* mesh_device = tilize_input_tensor.device();` |
| `ttnn/ops/experimental/ccl/moe_gpt/device/moe_gpt_program_factory.cpp:438` | MoEGPTMeshWorkloadFactory::create_at | In scope | 3 | already extracted: `auto* device = tensor_args.input_tensor.device();` |
| `ttnn/ops/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_program.cpp:550` | build_ring_reduce_scatter_minimal_async_program_artifacts | In scope | 3 | already extracted: `auto* mesh_device = input_tensor.device();` |
| `ttnn/ops/experimental/ccl/reduce_scatter_minimal_direct/device/reduce_scatter_minimal_direct_factory.cpp:281` | build_program_descriptor_at | In scope | 3 | already extracted: `auto* mesh_device = input_tensor.device();` |
| `ttnn/ops/experimental/conv3d/device/conv3d_program_factory.cpp:365` | Conv3dProgramFactory::create_descriptor | In scope | 3 | already extracted: `const uint32_t device_num_dram_banks = static_cast<uint32_t>(input_tensor.device()->num_dram_channels());` |
| `ttnn/ops/experimental/deepseek_prefill/hybrid_routed_expert_ffn/device/hybrid_program_factory.cpp:477` | append_to_descriptor | In scope | 3 | already extracted: `auto* device = tensor_arguments.activations.device();` |
| `ttnn/ops/experimental/deepseek_prefill/moe_fused_swiglu/device/moe_fused_swiglu_program_factory.cpp:417` | create_moe_fused_swiglu_program_descriptor | In scope | 3 | already extracted: `auto* device = tensor_arguments.activations.device();` |
| `ttnn/ops/experimental/deepseek_prefill/moe_hash_gate/device/moe_hash_gate_program_factory.cpp:87` | MoeHashGateDeviceOperation::ProgramFactory::create_descriptor | In scope | 3 | already extracted: `auto* device = scores.device();` |
| `ttnn/ops/experimental/padded_slice/device/padded_slice_rm_program_factory.cpp:155` | get_padded_slice_runtime_args_rm_sharded_output | One hop | 3 | `const Tensor& input_tensor` -> `.device()` |
| `ttnn/ops/experimental/padded_slice/device/padded_slice_rm_program_factory.cpp:294` | PaddedSliceRMProgramFactory::create_descriptor | One hop | 3 | `const PaddedSliceInputs& tensor_args` -> `.device()` |
| `ttnn/ops/experimental/padded_slice/device/padded_slice_rm_program_factory.cpp:297` | PaddedSliceRMProgramFactory::create_descriptor | One hop | 3 | `const PaddedSliceInputs& tensor_args` -> `.device()` |
| `ttnn/ops/experimental/padded_slice/device/padded_slice_rm_program_factory.cpp:57` | padded_slice_rm_reader_address | One hop | 3 | `const Tensor& input_tensor` -> `.device()` |
| `ttnn/ops/experimental/padded_slice/device/padded_slice_rm_program_factory.cpp:59` | padded_slice_rm_reader_address | One hop | 3 | `const Tensor& input_tensor` -> `.device()` |
| `ttnn/ops/experimental/padded_slice/device/padded_slice_tile_program_factory.cpp:434` | PaddedSliceTileProgramFactory::create_descriptor | One hop | 3 | `const PaddedSliceInputs& tensor_args` -> `.device()` |
| `ttnn/ops/experimental/plusone/device/plusone_program_factory.cpp:50` | PlusOneProgramFactory::create_program_artifacts | One hop | 3 | `const Tensor& input` -> `.device()` |
| `ttnn/ops/experimental/quasar/fold/device/fold_multi_core_dram_program_factory.cpp:314` | fold_multi_core_row_major_interleaved | In scope | 3 | already extracted: `auto* device = input_tensor.device();` |
| `ttnn/ops/experimental/quasar/fold/device/fold_multi_core_dram_program_factory.cpp:321` | fold_multi_core_row_major_interleaved | In scope | 3 | already extracted: `auto* device = input_tensor.device();` |
| `ttnn/ops/experimental/quasar/interleaved_to_sharded/device/interleaved_to_sharded_program_factory.cpp:143` | InterleavedToShardedProgramFactory::create_program_artifacts | In scope | 3 | already extracted: `bool is_blackhole = (input.device()->arch() == tt::ARCH::BLACKHOLE);` |
| `ttnn/ops/experimental/quasar/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp:1212` | process_mcast_in1_program_and_create_override_variables | In scope | 3 | param `tt_metal::distributed::MeshDevice& device` |
| `ttnn/ops/experimental/quasar/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp:176` | process_mcast_in0_program_and_create_override_variables | In scope | 3 | param `tt_metal::distributed::MeshDevice& device` |
| `ttnn/ops/experimental/quasar/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp:3005` | create_program_mcast_in0_descriptor | In scope | 3 | param `tt_metal::distributed::MeshDevice& device` |
| `ttnn/ops/experimental/quasar/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp:4037` | create_program_mcast_in1_descriptor | In scope | 3 | param `tt_metal::distributed::MeshDevice& device` |
| `ttnn/ops/experimental/quasar/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp:5440` | create_program_mcast_in0_artifacts | In scope | 3 | param `tt_metal::distributed::MeshDevice& device` |
| `ttnn/ops/experimental/quasar/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp:6509` | create_program_mcast_in1_artifacts | In scope | 3 | param `tt_metal::distributed::MeshDevice& device` |
| `ttnn/ops/experimental/quasar/matmul/device/factory/matmul_multicore_reuse_mcast_2d_program_factory.cpp:145` | create_program_mcast_in0_in1_descriptor | In scope | 3 | param `tt::tt_metal::distributed::MeshDevice& device` |
| `ttnn/ops/experimental/quasar/matmul/device/factory/matmul_multicore_reuse_mcast_2d_program_factory.cpp:1667` | create_program_mcast_in0_in1 | In scope | 3 | param `tt::tt_metal::distributed::MeshDevice& device` |
| `ttnn/ops/experimental/quasar/matmul/device/factory/matmul_multicore_reuse_mcast_2d_program_factory.cpp:3364` | create_program_mcast_in0_in1_artifacts | In scope | 3 | param `tt_metal::distributed::MeshDevice& device` |
| `ttnn/ops/experimental/quasar/matmul/device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:198` | create_program_dram_sharded_descriptor | In scope | 3 | param `tt::tt_metal::distributed::MeshDevice& device` |
| `ttnn/ops/experimental/quasar/matmul/device/factory/matmul_multicore_reuse_optimized_program_factory.cpp:157` | MatmulMultiCoreReuseOptimizedProgramFactory::create_program_artifacts | In scope | 3 | already extracted: `const tt_metal::distributed::MeshDevice& device = in0_buffer.device();` |
| `ttnn/ops/experimental/quasar/matmul/device/sparse/factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp:96` | SparseMatmulMultiCoreReuseMcast1DProgramFactory::create | In scope | 3 | already extracted: `auto* const device = a.device();` |
| `ttnn/ops/experimental/quasar/pad/device/pad_rm_reader_writer_multi_core_default_program_factory.cpp:133` | PadRmReaderWriterMultiCoreDefaultProgramFactory::create_program_artifacts | In scope | 3 | already extracted: `MeshDevice* device = a.device();` |
| `ttnn/ops/experimental/quasar/pad/device/pad_rm_reader_writer_multi_core_default_program_factory.cpp:75` | PadRmReaderWriterMultiCoreDefaultProgramFactory::create_program_artifacts | In scope | 3 | already extracted: `MeshDevice* device = a.device();` |
| `ttnn/ops/experimental/quasar/padded_slice/device/padded_slice_rm_program_factory.cpp:225` | PaddedSliceRMProgramFactory::create_program_artifacts | One hop | 3 | `const PaddedSliceInputs& tensor_args` -> `.device()` |
| `ttnn/ops/experimental/quasar/padded_slice/device/padded_slice_rm_program_factory.cpp:228` | PaddedSliceRMProgramFactory::create_program_artifacts | One hop | 3 | `const PaddedSliceInputs& tensor_args` -> `.device()` |
| `ttnn/ops/experimental/quasar/padded_slice/device/padded_slice_rm_program_factory.cpp:85` | get_padded_slice_runtime_args_rm_sharded_output | One hop | 3 | `const Tensor& input_tensor` -> `.device()` |
| `ttnn/ops/experimental/quasar/padded_slice/device/padded_slice_rm_program_factory.cpp:88` | get_padded_slice_runtime_args_rm_sharded_output | One hop | 3 | `const Tensor& input_tensor` -> `.device()` |
| `ttnn/ops/experimental/quasar/padded_slice/device/padded_slice_tile_program_factory.cpp:419` | PaddedSliceTileProgramFactory::create | In scope | 3 | already extracted: `tt::tt_metal::distributed::MeshDevice* device = a.device();` |
| `ttnn/ops/experimental/quasar/reshape_view/device/reshape_rm_program_factory.cpp:131` | ReshapeViewRMProgramFactory::create_descriptor | In scope | 3 | already extracted: `MeshDevice* device = input.device();` |
| `ttnn/ops/experimental/quasar/slice/device/slice_program_factory_rm.cpp:186` | compute_cb_size | One hop | 3 | `const Tensor& input` -> `.device()` |
| `ttnn/ops/experimental/quasar/slice/device/slice_program_factory_rm.cpp:189` | compute_cb_size | One hop | 3 | `const Tensor& input` -> `.device()` |
| `ttnn/ops/experimental/quasar/slice/device/slice_program_factory_rm.cpp:92` | get_slice_runtime_args_rm | One hop | 3 | `const Tensor& input_tensor` -> `.device()` |
| `ttnn/ops/experimental/quasar/slice/device/slice_program_factory_rm.cpp:95` | get_slice_runtime_args_rm | One hop | 3 | `const Tensor& input_tensor` -> `.device()` |
| `ttnn/ops/experimental/quasar/slice_write/device/slice_write_rm_interleaved_program_factory.cpp:102` | get_slice_write_runtime_args_rm | One hop | 3 | `const Tensor& input_tensor` -> `.device()` |
| `ttnn/ops/experimental/quasar/slice_write/device/slice_write_rm_interleaved_program_factory.cpp:221` | SliceWriteRMInterleavedProgramFactory::create | In scope | 3 | already extracted: `tt::tt_metal::distributed::MeshDevice* device = input.device();` |
| `ttnn/ops/experimental/quasar/slice_write/device/slice_write_rm_interleaved_program_factory.cpp:224` | SliceWriteRMInterleavedProgramFactory::create | In scope | 3 | already extracted: `tt::tt_metal::distributed::MeshDevice* device = input.device();` |
| `ttnn/ops/experimental/quasar/slice_write/device/slice_write_rm_sharded_input_program_factory.cpp:108` | get_slice_write_runtime_args_rm_sharded_input | One hop | 3 | `const Tensor& input_tensor` -> `.device()` |
| `ttnn/ops/experimental/quasar/slice_write/device/slice_write_rm_sharded_input_program_factory.cpp:240` | SliceWriteRMShardedInputProgramFactory::create_program_artifacts | One hop | 3 | `const SliceWriteInputs& tensor_args` -> `.device()` |
| `ttnn/ops/experimental/quasar/tilize/device/tilize_multi_core_block_program_factory.cpp:114` | TilizeMultiCoreBlockProgramFactory::create_program_artifacts | In scope | 3 | already extracted: `MeshDevice* device = a.device();` |
| `ttnn/ops/experimental/quasar/tilize_with_val_padding/device/factories/tilize_with_val_padding_multi_core_block_interleaved_program_factory.cpp:139` | TilizeWithValPaddingMultiCoreBlockInterleavedFactory::create_descriptor | In scope | 3 | already extracted: `MeshDevice* device = a.device();` |
| `ttnn/ops/experimental/quasar/to_memory_config/to_memory_config_op.cpp:105` | can_use_interleaved_to_sharded | In scope | 3 | already extracted: `MeshDevice* device = input_tensor.device();` |
| `ttnn/ops/experimental/rgb_to_yuv/device/rgb_to_yuv_program_factory.cpp:156` | RgbToYuvProgramFactory::create | In scope | 3 | already extracted: `const auto* device = input.device();` |
| `ttnn/ops/experimental/slice_write/device/slice_write_rm_interleaved_program_factory.cpp:122` | get_slice_write_runtime_args_rm | One hop | 3 | `const Tensor& input_tensor` -> `.device()` |
| `ttnn/ops/experimental/slice_write/device/slice_write_rm_interleaved_program_factory.cpp:233` | SliceWriteRMInterleavedProgramFactory::create_descriptor | In scope | 3 | already extracted: `tt::tt_metal::IDevice* device = input.device();` |
| `ttnn/ops/experimental/slice_write/device/slice_write_rm_interleaved_program_factory.cpp:235` | SliceWriteRMInterleavedProgramFactory::create_descriptor | In scope | 3 | already extracted: `tt::tt_metal::IDevice* device = input.device();` |
| `ttnn/ops/experimental/slice_write/device/slice_write_rm_sharded_input_program_factory.cpp:135` | get_slice_write_runtime_args_rm_sharded_input | One hop | 3 | `const Tensor& input_tensor` -> `.device()` |
| `ttnn/ops/experimental/slice_write/device/slice_write_rm_sharded_input_program_factory.cpp:262` | SliceWriteRMShardedInputProgramFactory::create_descriptor | One hop | 3 | `const SliceWriteInputs& tensor_args` -> `.device()` |
| `ttnn/ops/experimental/transformer/nlp_create_qkv_heads_decode/device/nlp_create_qkv_heads_decode_interleaved_program_factory.cpp:81` | NLPCreateQKVHeadsDecodeInterleavedProgramFactory::create_program_artifacts | In scope | 3 | already extracted: `const bool scratch_as_scratchpad = input_tensor.device()->arch() == tt::ARCH::QUASAR;` |
| `ttnn/ops/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp:1322` | process_mcast_in1_program_and_create_override_variables | In scope | 3 | param `tt_metal::distributed::MeshDevice& device` |
| `ttnn/ops/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp:194` | process_mcast_in0_program_and_create_override_variables | In scope | 3 | param `tt_metal::distributed::MeshDevice& device` |
| `ttnn/ops/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp:3323` | create_program_mcast_in0_artifacts | In scope | 3 | param `tt_metal::distributed::MeshDevice& device` |
| `ttnn/ops/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp:4691` | create_program_mcast_in1_artifacts | In scope | 3 | param `tt_metal::distributed::MeshDevice& device` |
| `ttnn/ops/matmul/device/factory/matmul_multicore_reuse_mcast_2d_program_factory.cpp:2059` | create_program_mcast_in0_in1 | In scope | 3 | param `tt::tt_metal::distributed::MeshDevice& device` |
| `ttnn/ops/matmul/device/factory/matmul_multicore_reuse_mcast_2d_program_factory.cpp:396` | create_program_mcast_in0_in1_spec | In scope | 3 | param `tt::tt_metal::distributed::MeshDevice& device` |
| `ttnn/ops/matmul/device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:283` | create_program_dram_sharded_spec | In scope | 3 | param `tt::tt_metal::distributed::MeshDevice& device` |
| `ttnn/ops/matmul/device/factory/matmul_multicore_reuse_optimized_program_factory.cpp:147` | MatmulMultiCoreReuseOptimizedProgramFactory::create_program_artifacts | In scope | 3 | already extracted: `const tt_metal::distributed::MeshDevice& device = in0_buffer.device();` |
| `ttnn/ops/matmul/device/sparse/factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp:147` | SparseMatmulMultiCoreReuseMcast1DProgramFactory::create_program_artifacts | In scope | 3 | already extracted: `auto* const device = a.device();` |
| `ttnn/ops/moreh/moreh_fold/device/fold_program_factory_rm.cpp:89` | MorehFoldOperation::MultiCore::create_program_artifacts | In scope | 3 | already extracted: `MeshDevice* device = input.device();` |
| `ttnn/ops/pool/grid_sample/device/grid_sample_utils.cpp:71` | get_aligned_stick_size | One hop | 3 | `const Tensor& tensor` -> `.device()` |
| `ttnn/ops/pool/pool_utils.cpp:613` | get_aligned_stick_size | One hop | 3 | `const Tensor& tensor` -> `.device()` |
| `ttnn/ops/pool/upsample/device/upsample_program_factory_multicore_interleaved.cpp:100` | UpsampleMultiCoreInterleavedProgramFactory::create_program_artifacts | In scope | 3 | already extracted: `MeshDevice* const device = output.device();` |
| `ttnn/ops/reduction/argmax/device/argmax_multi_core_program_factory.cpp:224` | ArgMaxMultiCoreProgramFactory::create_program_artifacts | In scope | 3 | already extracted: `const tt::tt_metal::distributed::MeshDevice& device = output.device();` |
| `ttnn/ops/transformer/sdpa/device/sparse_sdpa_device_operation.cpp:191` | SparseSDPAOperation::validate_on_program_cache_miss | One hop | 3 | `const SparseSDPAInputs& t` -> `.device()` |
| `ttnn/ops/transformer/sdpa/device/sparse_sdpa_msa_device_operation.cpp:181` | SparseSDPAMsaOperation::validate_on_program_cache_miss | One hop | 3 | `const SparseSDPAMsaInputs& t` -> `.device()` |
| `ttnn/cpp/ttnn-nanobind/bfp_utils.cpp:129` | py_module | No handle | 4: Python bindings | Python module-level binding of the free function; no device arg in the Python API |
| `ttnn/cpp/ttnn-nanobind/device.cpp:729` | device_module | No handle | 4: Python bindings | Python module-level binding of the free function; no device arg in the Python API |
| `ttnn/ops/data_movement/repeat_interleave/codegen/repeat_interleave_codegen_supported.cpp:26` | page_alignment | No handle | 4: Handle one level up | no device-bearing argument; a parameter would have to be added |
| `ttnn/ops/data_movement/slice/slice.cpp:52` | has_subaligned_shard_row | No handle | 4: Handle one level up | no device-bearing argument; a parameter would have to be added |
| `ttnn/ops/data_movement/untilize/codegen/untilize_codegen_program_factory.cpp:57` | aligned_tile_page_size | No handle | 4: Handle two or more levels up | no device-bearing argument; a parameter would have to be added |
| `ttnn/ops/experimental/multi_scale_deformable_attn/device/multi_scale_deformable_attn_program_factory.cpp:29` | aligned_page_size | No handle | 4: Handle one level up | no device-bearing argument; a parameter would have to be added |
| `ttnn/ops/sliding_window/sliding_window.cpp:1280` | align_buffer | No handle | 4: Handle two or more levels up | no device-bearing argument; a parameter would have to be added |

</details>

<details><summary><code>hal::get_arch</code> (14)</summary>

| Location | Enclosing function | Handle | Tier | Evidence |
|---|---|---|---|---|
| `ttnn/core/device.cpp:27` | default_dispatch_core_axis | No handle | 4: Runs before a device exists | dispatch-config defaulting before a MeshDevice exists (natural MetalEnv case) |
| `ttnn/core/device.cpp:44` | validate_dispatch_core_config | No handle | 4: Runs before a device exists | dispatch-config defaulting before a MeshDevice exists (natural MetalEnv case) |
| `ttnn/ops/conv/conv2d/conv2d_op_program_factory_common.cpp:118` | get_cb_info | No handle | 4: Handle two or more levels up | no device-bearing argument; a parameter would have to be added |
| `ttnn/ops/conv/conv2d/conv2d_op_program_factory_common.cpp:224` | get_cb_info | No handle | 4: Handle two or more levels up | no device-bearing argument; a parameter would have to be added |
| `ttnn/ops/conv/conv2d/conv2d_op_program_factory_common.cpp:91` | get_cb_info | No handle | 4: Handle two or more levels up | no device-bearing argument; a parameter would have to be added |
| `ttnn/ops/conv/conv2d/conv2d_utils.cpp:581` | should_coalesce_1d_depthwise_conv_reads | No handle | 4: Handle one level up | no device-bearing argument; a parameter would have to be added |
| `ttnn/ops/eltwise/binary_ng/device/binary_ng_program_factory.cpp:334` | hits_bh_col_bcast_bf16_to_fp32_hang | No handle | 4: Handle two or more levels up | no device-bearing argument; a parameter would have to be added |
| `ttnn/ops/eltwise/binary_ng/device/binary_ng_program_factory.cpp:361` | is_llk_bcast | No handle | 4: Handle one level up | no device-bearing argument; a parameter would have to be added |
| `ttnn/ops/experimental/quasar/binary_ng/device/binary_ng_program_factory.cpp:328` | hits_bh_col_bcast_bf16_to_fp32_hang | No handle | 4: Handle two or more levels up | no device-bearing argument; a parameter would have to be added |
| `ttnn/ops/experimental/quasar/binary_ng/device/binary_ng_program_factory.cpp:355` | is_llk_bcast | No handle | 4: Handle one level up | no device-bearing argument; a parameter would have to be added |
| `ttnn/ops/moreh/moreh_helper_functions.cpp:108` | CreateReadKernel | No handle | 4: Dead code | no device-bearing argument; a parameter would have to be added |
| `ttnn/ops/moreh/moreh_helper_functions.cpp:125` | CreateWriteKernel | No handle | 4: Dead code | no device-bearing argument; a parameter would have to be added |
| `ttnn/ops/pool/pool_utils.cpp:138` | get_factory_parameters | No handle | 4: Handle two or more levels up | no device-bearing argument; a parameter would have to be added |
| `ttnn/ops/pool/pool_utils.cpp:389` | calculate_L1_usage | No handle | 4: Handle two or more levels up | no device-bearing argument; a parameter would have to be added |

</details>

<details><summary><code>hal::get_arch_name</code> (12)</summary>

| Location | Enclosing function | Handle | Tier | Evidence |
|---|---|---|---|---|
| `tests/fabric/test_infra/tt_fabric_test_config.cpp:1188` | TestConfigBuilder::should_skip_test_on_platform | One hop | 2 | member `device_info_provider_` (IDeviceInfoProvider&, implemented by TestFixture) |
| `tests/fabric/test_infra/tt_fabric_test_bandwidth_results.cpp:289` | BandwidthResultsManager::set_comparison_statistics_csv_file_path | No handle | 4: Fabric test result managers | `ResultsManager` / `BandwidthResultsManager` hold no fixture or device member |
| `tests/fabric/test_infra/tt_fabric_test_bandwidth_results.cpp:64` | BandwidthResultsManager::initialize_results_csv_file | No handle | 4: Fabric test result managers | `ResultsManager` / `BandwidthResultsManager` hold no fixture or device member |
| `tests/fabric/test_infra/tt_fabric_test_latency_results.cpp:557` | LatencyResultsManager::initialize_results_csv_file | No handle | 4: Fabric test result managers | `LatencyResultsManager` has `fixture_`, but the CSV name is chosen in `initialize_latency_results_csv_file()`, before `open_devices()` creates `mesh_device_` |
| `tests/fabric/test_infra/tt_fabric_test_results.cpp:171` | ::populate_upload_metadata_fields | No handle | 4: Fabric test result managers | `ResultsManager` / `BandwidthResultsManager` hold no fixture or device member |
| `tests/fabric/test_infra/tt_fabric_test_results.cpp:195` | ::generate_summary_csv | No handle | 4: Fabric test result managers | `ResultsManager` / `BandwidthResultsManager` hold no fixture or device member |
| `tests/fabric/test_infra/tt_fabric_test_results.cpp:209` | ::generate_summary_upload_csv | No handle | 4: Fabric test result managers | `ResultsManager` / `BandwidthResultsManager` hold no fixture or device member |
| `tests/fabric/test_infra/tt_fabric_test_results.cpp:230` | ::init_diff_csv_file | No handle | 4: Fabric test result managers | `ResultsManager` / `BandwidthResultsManager` hold no fixture or device member |
| `tests/fabric/test_infra/tt_fabric_test_results.cpp:246` | ::get_golden_csv_filename | No handle | 4: Fabric test result managers | `ResultsManager` / `BandwidthResultsManager` hold no fixture or device member |
| `tests/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch.cpp:106` | get_core_count | No handle | 4: Runs before a device exists | runs during arg parsing / init, before the device is opened |
| `tests/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch.cpp:1243` | main | No handle | 4: Runs before a device exists | runs during arg parsing / init, before the device is opened |
| `tests/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch.cpp:705` | pgm_dispatch | No handle | 4: Runs before a device exists | runs before the device is created: the arch picks the trace region size passed to `create_unit_mesh` (natural MetalEnv case) |

</details>

<details><summary><code>hal::get_erisc_l1_unreserved_base</code> (14)</summary>

| Location | Enclosing function | Handle | Tier | Evidence |
|---|---|---|---|---|
| `tests/fabric/test_infra/tt_fabric_test_code_profiler.cpp:19` | get_code_profiling_buffer_addr | No handle | 1 | Hal in scope (function already calls `MetalContext::instance().hal()`) |
| `tests/tt_metal/api/test_dram.cpp:299` | TEST_F | In scope | 1 | Hal in scope (function already calls `MetalContext::instance().hal()`) |
| `tests/tt_metal/api/test_dram.cpp:336` | TEST_F | In scope | 1 | Hal in scope (function already calls `MetalContext::instance().hal()`) |
| `tests/tt_metal/dispatch/dispatch_program/test_EnqueueProgram.cpp:1534` | TEST_F | In scope | 1 | Hal in scope (function already calls `MetalContext::instance().hal()`) |
| `tools/scaleout/validation/utils/cluster_validation_utils.cpp:710` | dump_link_stats | In scope | 1 | Hal in scope (function already calls `MetalContext::instance().hal()`) |
| `tests/fabric/test_infra/tt_fabric_telemetry.hpp:47` | (file/class scope) | No handle | 4: Static initializer | namespace-scope global initializer, runs at static-init time |
| `tests/fabric/test_infra/tt_fabric_telemetry_manager.cpp:32` | TelemetryManager::read_telemetry | In scope | 4: No public API for the query; handle in scope; code may use internal headers | member `fixture_` (TestFixture&) -> `mesh_device_` |
| `tests/fabric/test_infra/tt_fabric_telemetry_manager.cpp:95` | TelemetryManager::clear_telemetry | In scope | 4: No public API for the query; handle in scope; code may use internal headers | member `fixture_` (TestFixture&) -> `mesh_device_` |
| `tests/ttnn/unit_tests/gtests/ccl/test_erisc_data_mover_with_workers.cpp:446` | RunWriteBWTest | In scope | 4: No public API for the query; handle in scope; code may use internal headers | param `tt::tt_metal::distributed::MeshDevice>& sender_mesh_device` |
| `tools/scaleout/validation/utils/cluster_validation_utils.cpp:182` | configure_local_kernels | In scope | 4: No public API for the query; handle in scope; code may use internal headers | `ClusterContext& ctx` holds `devices` (map of MeshDevice) |
| `tools/scaleout/validation/utils/cluster_validation_utils.cpp:183` | configure_local_kernels | In scope | 4: No public API for the query; handle in scope; code may use internal headers | `ClusterContext& ctx` holds `devices` (map of MeshDevice) |
| `tools/scaleout/validation/utils/cluster_validation_utils.cpp:333` | configure_cross_host_kernels | In scope | 4: No public API for the query; handle in scope; code may use internal headers | `ClusterContext& ctx` holds `devices` (map of MeshDevice) |
| `tools/scaleout/validation/utils/cluster_validation_utils.cpp:334` | configure_cross_host_kernels | In scope | 4: No public API for the query; handle in scope; code may use internal headers | `ClusterContext& ctx` holds `devices` (map of MeshDevice) |
| `ttnn/ops/ccl/ccl_host_datastructures.hpp:21` | (file/class scope) | No handle | 4: No public API for the query; ttnn code | default member initializer of `EriscDatamoverConfig`; a handle would have to arrive via its constructor |

</details>

<details><summary><code>hal::get_max_worker_l1_unreserved_size</code> (6)</summary>

| Location | Enclosing function | Handle | Tier | Evidence |
|---|---|---|---|---|
| `tests/tt_metal/dispatch/dispatch_program/test_kernel_config_buffer.cpp:35` | (file/class scope) | No handle | 1 | Hal in scope (fixture member `hal_`, declared just above) |
| `tests/fabric/test_infra/tt_fabric_test_common.hpp:349` | get_l1_unreserved_size | In scope | 3 | `TestFixture` member `mesh_device_` |
| `ttnn/ops/data_movement/common/fold_common.hpp:72` | tile_native_fold_rejection_reason | One hop | 3 | `const Tensor& input_tensor` -> `.device()` |
| `ttnn/ops/experimental/conv3d/device/conv3d_program_factory.cpp:415` | Conv3dProgramFactory::create_descriptor | In scope | 3 | already extracted: `const uint32_t device_num_dram_banks = static_cast<uint32_t>(input_tensor.device()->num_dram_channels());` |
| `ttnn/ops/experimental/deepseek_prefill/moe_fused_swiglu/device/moe_fused_swiglu_program_factory.cpp:396` | create_moe_fused_swiglu_program_descriptor | In scope | 3 | already extracted: `auto* device = tensor_arguments.activations.device();` |
| `ttnn/cpp/ttnn-nanobind/device.cpp:724` | device_module | No handle | 4: Python bindings | Python module-level binding of the free function; no device arg in the Python API |

</details>

<details><summary><code>hal::get_pcie_alignment</code> (4)</summary>

| Location | Enclosing function | Handle | Tier | Evidence |
|---|---|---|---|---|
| `tt_metal/distributed/host_d2h_leg.cpp:110` | D2HLeg::create | In scope | 4: No public API for the query; handle in scope; code may use internal headers | param `dist::MeshDevice>& mesh` |
| `ttnn/core/services/d2h_socket_service.cpp:423` | cfg_ | In scope | 4: No public API for the query; ttnn code | ctor param `mesh_device` / member `mesh_device_` |
| `ttnn/core/services/h2d_socket_service.cpp:277` | make_completion_layout | No handle | 4: No public API for the query; ttnn code | free helper called from the ctor; takes only a counter count |
| `ttnn/core/services/h2d_socket_service.cpp:484` | cfg_ | In scope | 4: No public API for the query; ttnn code | ctor param `mesh_device` / member `mesh_device_` |

</details>

<details><summary><code>hal::get_num_dataflow_buffers</code> (4)</summary>

| Location | Enclosing function | Handle | Tier | Evidence |
|---|---|---|---|---|
| `tests/tt_metal/api/dataflow_buffer/test_dataflow_buffer_configs.cpp:2339` | TEST_F | In scope | 3 | gtest fixture member (`devices_` / `mesh_device_` / `this->device()`) |
| `tests/tt_metal/api/metal2_host_api/unit_tests/invariant_tests/program_spec/dfbs_per_node.cpp:102` | TEST_F | In scope | 3 | gtest fixture member (`devices_` / `mesh_device_` / `this->device()`) |
| `tests/tt_metal/api/metal2_host_api/unit_tests/invariant_tests/program_spec/dfbs_per_node.cpp:125` | TEST_F | In scope | 3 | gtest fixture member (`devices_` / `mesh_device_` / `this->device()`) |
| `tests/tt_metal/api/metal2_host_api/unit_tests/program/compute_config_lowering.cpp:143` | TEST_F | In scope | 3 | gtest fixture member (`devices_` / `mesh_device_` / `this->device()`) |

</details>

<details><summary><code>hal::get_inf</code> (4)</summary>

| Location | Enclosing function | Handle | Tier | Evidence |
|---|---|---|---|---|
| `tt-train/sources/ttml/metal/ops/sdpa_bw/device/sdpa_bw_kv_program_factory.cpp:267` | SDPABackwardKVProgramFactory::create | In scope | 3 | already extracted: `auto* device = grad_output.device();` |
| `tt-train/sources/ttml/metal/ops/sdpa_bw/device/sdpa_bw_q_program_factory.cpp:279` | SDPABackwardQProgramFactory::create | In scope | 3 | already extracted: `auto* device = grad_output.device();` |
| `ttnn/core/distributed/distributed_nanobind.cpp:528` | py_module | In scope | 3 | lambda takes `MeshDevice*` (currently unused) |
| `ttnn/ops/eltwise/unary_backward/unary_backward.cpp:628` | acosh_bw | One hop | 3 | `const Tensor& grad` -> `.device()` |

</details>

<details><summary><code>hal::get_erisc_l1_unreserved_size</code> (4)</summary>

| Location | Enclosing function | Handle | Tier | Evidence |
|---|---|---|---|---|
| `tools/scaleout/validation/run_cluster_validation.cpp:229` | parse_validation_args | No handle | 4: Runs before a device exists | argv parsing, before any device/env exists (natural MetalEnv case) |
| `tools/scaleout/validation/run_cluster_validation.cpp:231` | parse_validation_args | No handle | 4: Runs before a device exists | argv parsing, before any device/env exists (natural MetalEnv case) |
| `tools/scaleout/validation/run_cluster_validation.cpp:233` | parse_validation_args | No handle | 4: Runs before a device exists | argv parsing, before any device/env exists (natural MetalEnv case) |
| `ttnn/ops/ccl/ccl_host_datastructures.hpp:20` | (file/class scope) | No handle | 4: No public API for the query; ttnn code | default member initializer of `EriscDatamoverConfig`; a handle would have to arrive via its constructor |

</details>

<details><summary><code>hal::get_noc_max_burst_size_bytes</code> (2)</summary>

| Location | Enclosing function | Handle | Tier | Evidence |
|---|---|---|---|---|
| `ttnn/ops/data_movement/concat/device/concat_s2s_tiled_program_factory.cpp:200` | ConcatS2STiledProgramFactory::create_program_artifacts | One hop | 4: No public API for the query; ttnn code | `const ConcatInputs& tensor_args` -> `.device()` |
| `ttnn/ops/matmul/device/utilities/matmul_utilities.cpp:364` | get_max_page_size_and_num_pages | In scope | 4: No public API for the query; ttnn code | unnamed param `const MeshDevice&` (already passed, unused) |

</details>

<details><summary><code>hal::get_nan</code> (2)</summary>

| Location | Enclosing function | Handle | Tier | Evidence |
|---|---|---|---|---|
| `ttnn/core/distributed/distributed_nanobind.cpp:524` | py_module | In scope | 3 | lambda takes `MeshDevice*` (currently unused) |
| `ttnn/ops/eltwise/unary_backward/unary_backward.cpp:627` | acosh_bw | One hop | 3 | `const Tensor& grad` -> `.device()` |

</details>

<details><summary><code>hal::get_eps</code> (2)</summary>

| Location | Enclosing function | Handle | Tier | Evidence |
|---|---|---|---|---|
| `ttnn/core/distributed/distributed_nanobind.cpp:520` | py_module | In scope | 3 | lambda takes `MeshDevice*` (currently unused) |
| `ttnn/ops/experimental/quasar/binary/device/binary_composite_op.cpp:35` | nextafter | One hop | 3 | `const Tensor& input_a` -> `.device()` |

</details>

<details><summary><code>hal::get_arch_num_circular_buffers</code> (0)</summary>

| Location | Enclosing function | Handle | Tier | Evidence |
|---|---|---|---|---|

</details>

## Method and limits

Found with ripgrep for any `hal::get_*` spelling of the 14 declared functions. Matches in comments (10) and in markdown or Python files (6) are excluded. No file calls them unqualified or through `using namespace`; inside `tt_metal/hal.cpp` the deprecated alias calls `get_num_dataflow_buffers` (not counted).

Handle classification is by script over the enclosing function's parameters and body, then spot-checked by hand. It was not compiled. For the tier 4 helpers, direct callers were found with ripgrep and checked for a tensor or device in the calling function; chains deeper than two levels were followed by hand only for the conv/pool estimators.
