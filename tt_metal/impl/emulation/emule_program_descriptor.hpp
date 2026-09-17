// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
//
// The emule program descriptor: a flat POD snapshot of everything the emulator
// needs from a tt-metal Program/IDevice. build_emule_descriptor (the marshaller,
// emule_descriptor_builder.cpp) is the ONLY code that reads private tt-metal types;
// it fills this POD, and the interpretation modules (device_map, kernel_defines,
// program_model, cb_dfb_setup, metal2_emit, jit) consume ONLY this — never a
// private tt-metal type. That is the API boundary that makes the modules movable.
//
// All types live in namespace tt_emule and are deliberately distinct from tt-metal's
// own descriptor family (tt-metalium/program_descriptors.hpp): the top-level here is
// EmuleProgramDescriptor, not ProgramDescriptor, to avoid that name collision.
//
// Depends on nothing but <std> + primitives: every tt-metal enum/handle/value is
// flattened here. Target home after the cross-repo move is tt-emule-blaze
// include/tt_emule/program_descriptor.hpp; it lives here in-place for now.
//
// Sits ABOVE the runtime sync structs (tt_emule::CBSyncState in cb_sync_state.hpp,
// tt_emule::DFBSyncState in dfb_sync_state.hpp): it carries the RAW config; the
// engine derives page_mask, capacity, the atomics, and the tile/face precedence.

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace tt_emule {

// ── mirrored enums (kept in lockstep with tt-metal; the marshaller static_casts) ──
enum class AccessPattern : uint8_t { STRIDED = 0, ALL = 1 };  // DataflowBufferConfig::cap
enum class SemScope : uint8_t { CORE = 0 /* … mirror tt_metal SemScope … */ };

using KernelHandle = uint32_t;

// ─────────────────────────────── SocView (once per device) ───────────────────────────────
struct DramView {  // one per metal_SocDescriptor DRAM view
    uint32_t noc_xy[2] = {
        0, 0};  // [NOC0,NOC1] = (y<<NOC_NODE_ID_BITS)|x  get_preferred_worker_core_for_dram_view(v,noc)
    uint32_t address_offset = 0;  // metal_SocDescriptor::get_address_offset(v)
    // The umd LOGICAL channel is resolved on the consumer side (build_core_map) from the chip's own
    // umd descriptor, so it is not carried here.
};
struct L1Bank {
    uint32_t logical_x = 0, logical_y = 0;  // allocator->get_logical_core_from_bank_id(b)
    uint32_t noc_xy = 0;                    // IDevice::virtual_core_from_logical_core(logical, WORKER)
};
struct PctInfo {                           // per programmable-core-type (HAL)
    uint32_t core_type = 0;                // hal.get_programmable_core_type(pct)
    uint32_t kernel_config_addr = 0;       // hal.get_dev_addr(pct, KERNEL_CONFIG)
    uint32_t kernel_config_size = 0;       // hal.get_dev_size(pct, KERNEL_CONFIG)
    uint32_t default_unreserved_addr = 0;  // hal.get_dev_addr(pct, DEFAULT_UNRESERVED)  (TENSIX dynamic window)
    uint32_t routing_table_addr = 0;       // hal.get_dev_addr(pct, ROUTING_TABLE)
};
struct SocView {
    uint32_t arch = 0;                              // Cluster::arch()  -> ARCH_{WORMHOLE,BLACKHOLE,QUASAR}
    uint32_t fabric_config = 0;                     // MetalContext::get_fabric_config()
    bool fabric_2d = false;                         // is_2d_fabric_config(...)
    std::vector<DramView> dram_views;               // size == get_num_dram_views()
    std::vector<L1Bank> l1_banks;                   // size == allocator->get_num_banks(L1)
    uint32_t worker_grid_x = 0, worker_grid_y = 0;  // compute_with_storage_grid_size()
    std::array<uint32_t, 64> worker_col_to_virt{};  // logical col -> virtual x
    std::array<uint32_t, 64> worker_row_to_virt{};  // logical row -> virtual y
    uint32_t dram_alignment = 0, l1_alignment = 0;  // hal::get_{dram,l1}_alignment()
    uint32_t arch_num_circular_buffers = 0;         // hal.get_arch_num_circular_buffers()
    bool has_tile_counter_registers = false;        // hal.has_tile_counter_registers()  (Quasar)
    std::vector<PctInfo> pcts;                      // size == hal.get_programmable_core_type_count()
    uint32_t mesh_id = 0, chip_id = 0;              // ControlPlane::get_fabric_node_id_from_physical_chip_id()
    // host_alignment_requirement(size) is size-dependent -> resolve on the emule side
    // (inject a small callback or a value table), not a scalar field.
};

// ─────────────────────────────── Kernel ───────────────────────────────
struct SourceRef {  // Kernel::kernel_source()
    bool is_file = true;
    std::string path;        // when is_file
    std::string inline_src;  // when !is_file
};
struct NamedRtEntry {  // NamedRuntimeArgEntry (jit_build_settings.hpp)
    std::string field;
    uint32_t index = 0, length = 0;
    uint32_t dispatch = 0;  // RuntimeArgDispatch enum value (kept numeric for cache-key parity)
};
using NamedCtNamespaces =
    std::map<std::string, std::vector<std::pair<std::string, uint32_t>>>;    // process_named_ct_arg_namespaces
using NamedRtNamespaces = std::map<std::string, std::vector<NamedRtEntry>>;  // process_named_runtime_args

// Metal-2.0 binding handles — flattened from the four PULL-BASED process_*_binding_handles callbacks.
struct DfbBinding {
    std::string name;
    uint16_t dfb_id = 0;
    bool is_relay = false;
    uint8_t prefetcher_pipe = 0;
};
struct SemBinding {
    std::string name;
    uint16_t sem_id = 0;
    SemScope scope = SemScope::CORE;
    uint32_t total_binder_harts = 0;
};
struct TensorBinding {
    std::string name;
    uint32_t cta_offset = 0;
    uint32_t addr_crta_offset = 0;
};  // num_rt_words asserted 0 today
struct ScratchBinding {
    std::string name;
    uint32_t size_bytes = 0;
    uint32_t addr_crta_word = 0;
};
struct Bindings {
    bool is_metal2 = false;                          // Kernel::is_metal2_kernel()
    std::vector<std::string> rta_names, crta_names;  // get_runtime_arg_names / get_common_runtime_arg_names
    std::vector<DfbBinding> dfb;
    std::vector<SemBinding> sem;
    std::vector<TensorBinding> tensor;
    std::vector<ScratchBinding> scratch;
};

struct CoreRange4 {
    uint32_t sx = 0, sy = 0, ex = 0, ey = 0;
};  // an inclusive logical core-range box (KernelGroup / kernel core_range_set)

struct KernelDescriptor {
    KernelHandle id = 0;
    SourceRef source;                                                   // kernel_source()
    std::vector<std::string> include_paths;                             // process_include_paths()
    std::vector<uint32_t> compile_time_args;                            // compile_time_args()
    std::unordered_map<std::string, uint32_t> named_compile_time_args;  // named_compile_time_args()
    NamedCtNamespaces named_ct_arg_namespaces;                          // process_named_ct_arg_namespaces()
    NamedRtNamespaces named_runtime_arg_namespaces;                     // process_named_runtime_args()
    std::unordered_map<std::string, std::string> defines;               // process_defines()
    uint32_t programmable_core_type = 0;                                // get_kernel_programmable_core_type()
    uint32_t processor_class = 0;                                       // get_kernel_processor_class()
    uint32_t processor_type = 0;                                        // get_kernel_processor_type(0)
    bool is_compute = false;                                            // processor_class == COMPUTE
    bool is_quasar_compute = false;        // is_compute && dynamic_cast<QuasarComputeKernel>
    uint32_t dm_processor = 0;             // DataMovementKernel::config().processor (RISCV_0/1->BRISC/NCRISC)
    bool is_data_movement = false;         // dynamic_cast<DataMovementKernel> succeeds
    uint32_t compile_processor_index = 0;  // hal.get_processor_index(core_type, class, COMPILE_FOR idx)
    bool has_compute_config = false;       // config() held a ComputeConfig
    bool fp32_dest_acc_en = false, dst_full_sync_en = false;  // ComputeKernel::config()
    std::vector<uint32_t> proc_ids;
    uint32_t num_threads = 1;                   // Quasar get_dm/compute_processors, else single
    std::vector<uint32_t> common_runtime_args;  // common_runtime_args()  (program-wide)
    Bindings bindings;                          // build_metal2_snapshot()
    std::vector<CoreRange4> core_ranges;        // core_range_set().ranges()  (per-core placement bound)
    // Per-core launch offsets + unique runtime args live in CoreKernel below.
};

// Per (kernel, logical-core): the launch offsets + unique runtime-arg values on this core.
// The marshaller resolves the KernelGroup launch_msg here so the consumer needs no KG/firmware read.
struct CoreKernel {
    KernelHandle kernel = 0;
    uint32_t kernel_config_base = 0;                     // KG launch_msg kernel_config()[pct]
    uint16_t rta_offset = 0xFFFF, crta_offset = 0xFFFF;  // KG rta_offset[processor_index] (0xFFFF = no args)
    std::vector<uint32_t> unique_rt_args;                // Kernel::runtime_args(core)  (empty => common only)
};

// ─────────────────────────────── Circular buffers ───────────────────────────────
// Silicon keeps TWO stores; the POD mirrors them and NEVER folds them:
//   RING  = the four L1 config words -> tt_emule::CBSyncState.
//   GEOM  = the compile-time tile/face descriptor -> build_kernel_defines / EMULE_TILE_*.
// tile_size derives from the Tile, never from page_size. The tile/face precedence
// (explicit FaceGeometry > CB Tile > full-tile) is EMULE logic that needs the live
// tt-metal Tile, so the MARSHALLER applies resolve_tile_geometry and stores the
// RESOLVED primitives here; consumers read them directly and never rebuild a Tile.
struct ResolvedGeom {        // resolve_tile_geometry(tile, unpack_face) applied on the marshaller side
    uint32_t tile_size = 0;  // ResolvedTileGeometry::tile.get_tile_size(data_format)
    uint32_t tile_r_dim = 32, tile_c_dim = 32;   // effective tile height / width
    uint32_t face_r_dim = 16, num_faces = 4;     // resolved face_r_dim / num_faces
    uint32_t partial_face = 0, narrow_tile = 0;  // effective tile partial_face / narrow_tile flags
};
struct CbBuffer {              // one local buffer index
    uint8_t index = 0;         // local_buffer_indices()
    uint32_t page_size = 0;    // page_size(index)    -- RING
    uint32_t num_pages = 0;    // num_pages(index)    -- RING
    uint32_t data_format = 0;  // data_format(index)  (tt::DataFormat raw) -- GEOM
    ResolvedGeom geom;         // tile(index) + unpack_face_geometry(index), resolved -- GEOM
};
struct CbDescriptor {
    uint32_t address = 0;             // address()            -- RING
    uint32_t total_size = 0;          // size()               -- RING (persistent range end)
    bool globally_allocated = false;  // globally_allocated()-- RING
    std::vector<CbBuffer> buffers;    // per local index
};

// ─────────────────────────────── Dataflow buffers (Quasar) ───────────────────────────────
struct DfbDescriptor {
    uint32_t device_slot = 0;                      // DataflowBufferImpl::device_slot
    uint32_t entry_size = 0, num_entries = 0;      // config.entry_size / num_entries  (capacity derived in emule)
    uint8_t num_producers = 0, num_consumers = 0;  // config.num_{producers,consumers}
    uint16_t producer_risc_mask = 0, consumer_risc_mask = 0;  // config.{producer,consumer}_risc_mask
    AccessPattern cap = AccessPattern::STRIDED;               // config.cap  (CONSUMER access pattern, NOT a size)
    bool has_finalize = false;
    uint32_t finalize_l1_offset = 0;  // core_lookup_[core].second.second
    uint32_t data_format = 0;         // config.data_format  (feeds the CB tables)
    ResolvedGeom geom;                // config.tile / .unpack_face_geometry, resolved (valid data_format only)
};

// ─────────────────────────────── Semaphores ───────────────────────────────
struct SemaphoreDescriptor {
    uint32_t id = 0;             // Semaphore::id()  (addr = kernel_config_base + sem_offset + id*EMULE_SEM_ALIGN)
    uint32_t initial_value = 0;  // Semaphore::initial_value()
    // per-core coverage lives in CoreDescriptor.semaphore_ids (from initialized_on_logical_core(core)).
};

// ─────────────────────────────── Kernel groups & launch offsets ───────────────────────────────
// The one hard read: kernel_config_base[pct] and per-processor rta_offset/crta_offset come
// from the firmware launch_msg.view().kernel_config() layout — no plain accessor
// (emulated_program_runner.cpp:2051-2055). The marshaller reads launch_msg directly (it is
// in-tree, has access). A cleaner accessor is a later concern, not a boundary this POD forces.
struct ProcLaunchOffset {
    uint32_t processor_index = 0, rta_offset = 0, crta_offset = 0;
};
struct KernelGroupDescriptor {
    uint32_t pct = 0;                            // programmable-core-type index
    uint32_t kernel_config_base = 0;             // kc.kernel_config_base()[pct]
    std::vector<ProcLaunchOffset> proc_offsets;  // kc.rta_offset()[processor_index]
    std::vector<KernelHandle> kernel_ids;        // KernelGroup::kernel_ids
    std::vector<CoreRange4> core_ranges;         // KernelGroup::core_ranges
};

// Per logical core: which CBs/DFBs/semaphores/kernels are live there.
struct CoreDescriptor {
    uint32_t logical_x = 0, logical_y = 0;
    std::vector<CbDescriptor> cbs;        // circular_buffers_on_core(core)
    std::vector<DfbDescriptor> dfbs;      // dataflow_buffers_on_core(core)
    std::vector<uint32_t> semaphore_ids;  // sems with initialized_on_logical_core(core)
    std::vector<CoreKernel> kernels;      // per-kernel unique runtime args on this core
};

// ─────────────────────────────── Top level ───────────────────────────────
struct ProgramConfig {                   // ProgramImpl-level
    uint64_t program_id = 0;             // get_id()
    uint32_t context_id = 0;             // get_context_id()
    std::vector<uint32_t> config_sizes;  // get_program_config_sizes()          (per pct)
    std::vector<uint32_t> sem_offset;    // get_program_config(pct).sem_offset  (per pct)
};

struct EmuleProgramDescriptor {
    ProgramConfig config;
    std::vector<KernelGroupDescriptor> kernel_groups;
    std::unordered_map<KernelHandle, KernelDescriptor> kernels;  // union of get_kernels(pct)
    std::vector<KernelHandle> kernel_order;  // get_kernels(pct) order across pcts (collect_kernels drives on it)
    std::vector<CoreDescriptor> cores;                           // logical_cores()
    std::vector<SemaphoreDescriptor> semaphores;                 // semaphores()
    // SocView is device-scoped (program-invariant) -> build once, pass alongside.
};

}  // namespace tt_emule
