// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <bit>
#include <functional>
#include <set>

#include <tt-metalium/hal.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "impl/metal2_host_api/test_utils.hpp"

namespace tt::tt_metal::experimental::metal2_host_api {

// ============================================================================
// Constants
// ============================================================================

// TODO: These constants should be queriable from the public API (currently HAL, for consistency)
//       They are currently also hardcoded in the temporary Quasar host_api.hpp. Need to clean this up.
static constexpr uint32_t QUASAR_DM_CORES_PER_NODE = 8;
static constexpr uint32_t QUASAR_TENSIX_CORES_PER_NODE = 4;

// ============================================================================
// Type Definitions
// ============================================================================

// Data structure built up from ProgramSpec to enable fast lookups.
struct CollectedSpecData {
    // Name -> spec lookups
    std::unordered_map<KernelSpecName, const KernelSpec*> kernel_by_name;
    std::unordered_map<DFBSpecName, const DataflowBufferSpec*> dfb_by_name;
    std::unordered_map<SemaphoreSpecName, const SemaphoreSpec*> semaphore_by_name;

    // DFB endpoint info (derived from kernel bindings)
    struct DFBEndpointInfo {
        const KernelSpec* producer = nullptr;
        const KernelSpec* consumer = nullptr;
        const KernelSpec::DFBBinding* producer_binding = nullptr;
        const KernelSpec::DFBBinding* consumer_binding = nullptr;
    };
    std::unordered_map<DFBSpecName, DFBEndpointInfo> dfb_endpoints;
};

// Bitmask for tracking processor allocation on a node.
// (Factory functions are defined with processor assignment helper functions.)
template <uint8_t NUM_CORES>
struct ProcessorMask {
    static_assert(NUM_CORES > 0 && NUM_CORES <= 8, "ProcessorMask supports 1-8 processors");
    static constexpr uint8_t VALID_BITS_MASK = (NUM_CORES == 8) ? 0xFF : ((1 << NUM_CORES) - 1);

    uint8_t bits = 0x00;

    // Operators
    ProcessorMask operator|(ProcessorMask other) const { return {uint8_t(bits | other.bits)}; }
    ProcessorMask operator&(ProcessorMask other) const { return {uint8_t(bits & other.bits)}; }
    ProcessorMask operator~() const { return {uint8_t(~bits & VALID_BITS_MASK)}; }
    ProcessorMask& operator|=(ProcessorMask other) {
        bits |= other.bits;
        return *this;
    }
    ProcessorMask& operator&=(ProcessorMask other) {
        bits &= other.bits;
        return *this;
    }

    // Queries
    uint8_t num_in_use() const { return std::popcount(bits); }
    uint8_t num_available() const { return NUM_CORES - num_in_use(); }
    bool is_idx_available(uint8_t idx) const { return (bits & (1 << idx)) == 0; }
    bool is_idx_in_use(uint8_t idx) const { return (bits & (1 << idx)) != 0; }
    bool conflicts_with(ProcessorMask other) const { return (bits & other.bits) != 0; }
};

using DMProcessorMask = ProcessorMask<QUASAR_DM_CORES_PER_NODE>;
using ComputeEngineMask = ProcessorMask<QUASAR_TENSIX_CORES_PER_NODE>;

// Kernel -> ProcessorMask maps
using DMProcessorMaskMap = std::unordered_map<const KernelSpec*, DMProcessorMask>;
using ComputeEngineMaskMap = std::unordered_map<const KernelSpec*, ComputeEngineMask>;

// DFB name -> ID map (for unpack_to_dest_mode indexing)
using DFBNameToIdMap = std::unordered_map<DFBSpecName, uint32_t>;

// ============================================================================
// Test Hook: Architecture Override (Implementation)
// ============================================================================
// See test_utils.hpp for documentation and usage.

std::optional<tt::ARCH>& arch_override() {
    thread_local std::optional<tt::ARCH> override_value = std::nullopt;
    return override_value;
}

ArchOverrideGuard::ArchOverrideGuard(tt::ARCH arch) { arch_override() = arch; }
ArchOverrideGuard::~ArchOverrideGuard() { arch_override() = std::nullopt; }

// ============================================================================
// Helper Function Forward Declarations
// ============================================================================

// Utilities
inline tt::ARCH get_arch();  // Returns override if set, otherwise HAL
inline bool is_gen2_arch();
inline bool is_gen1_arch();
NodeRangeSet to_node_range_set(const std::variant<NodeCoord, NodeRange, NodeRangeSet>& nodes);
bool nodes_intersect(
    const std::variant<NodeCoord, NodeRange, NodeRangeSet>& a,
    const std::variant<NodeCoord, NodeRange, NodeRangeSet>& b);
bool nodes_contains(
    const std::variant<NodeCoord, NodeRange, NodeRangeSet>& superset,
    const std::variant<NodeCoord, NodeRange, NodeRangeSet>& subset);

// Phase 1: Collection & Validation
CollectedSpecData CollectSpecData(const ProgramSpec& spec);
void ValidateProgramSpec(const ProgramSpec& spec, const CollectedSpecData& collected);

// Phase 2: Processor Assignment
template <uint8_t NUM_CORES>
ProcessorMask<NUM_CORES> CreateMask(uint8_t mask);
template <uint8_t NUM_CORES>
std::optional<ProcessorMask<NUM_CORES>> ReserveProcessors(uint8_t n, const ProcessorMask<NUM_CORES>& already_in_use);
std::pair<DMProcessorMaskMap, ComputeEngineMaskMap> SolveKernelToProcessorAssignments(
    const ProgramSpec& spec, const CollectedSpecData& collected);

// Phase 3: Program Building
experimental::dfb::DataflowBufferConfig MakeDataflowBufferConfig(
    const DataflowBufferSpec* dfb_spec,
    const CollectedSpecData::DFBEndpointInfo& dfb_endpoint_info,
    const DMProcessorMaskMap& kernel_to_dm_processor_mask_map,
    const ComputeEngineMaskMap& kernel_to_compute_processor_mask_map);
tt::tt_metal::KernelSource MakeKernelSource(const KernelSpec& kernel_spec);
tt::tt_metal::experimental::quasar::QuasarDataMovementConfig MakeQuasarDataMovementConfig(
    const KernelSpec& kernel_spec, const DFBNameToIdMap& dfb_name_to_id);
tt::tt_metal::experimental::quasar::QuasarComputeConfig MakeQuasarComputeConfig(
    const KernelSpec& kernel_spec, const DFBNameToIdMap& dfb_name_to_id);
std::set<tt::tt_metal::DataMovementProcessor> GetDMProcessorSet(DMProcessorMask mask);
std::set<tt::tt_metal::experimental::quasar::QuasarComputeProcessor> GetComputeProcessorSet(ComputeEngineMask mask);

// ============================================================================
//  PUBLIC ENTRY POINT: MakeProgramFromSpec
// ============================================================================

Program MakeProgramFromSpec(const ProgramSpec& spec, bool skip_validation) {
    // Collect derived data (builds lookup tables, checks structural invariants)
    CollectedSpecData collected = CollectSpecData(spec);

    // Validate semantic rules (can be skipped for trusted inputs)
    if (!skip_validation) {
        ValidateProgramSpec(spec, collected);
    }

    // Solve kernel-to-core assignments
    // NOTE: Current solver assumes that a given DM kernel uses the _same_ set of DM cores on every node/cluster.
    auto [kernel_to_dm_processor_mask_map, kernel_to_compute_processor_mask_map] =
        SolveKernelToProcessorAssignments(spec, collected);

    // Build the Program
    auto program_impl = std::make_shared<detail::ProgramImpl>();

    // Create DataflowBuffers and build name -> ID map for unpack_to_dest_mode
    DFBNameToIdMap dfb_name_to_id;
    for (const auto& [dfb_name, dfb_endpoint_info] : collected.dfb_endpoints) {
        const DataflowBufferSpec* dfb_spec = collected.dfb_by_name.at(dfb_name);
        const experimental::dfb::DataflowBufferConfig config = MakeDataflowBufferConfig(
            dfb_spec, dfb_endpoint_info, kernel_to_dm_processor_mask_map, kernel_to_compute_processor_mask_map);

        // Add the DFB to the ProgramImpl, and register the name -> handle mapping
        uint32_t dfb_id = program_impl->add_dataflow_buffer(to_node_range_set(dfb_spec->target_nodes), config);
        program_impl->register_dfb_spec_name(dfb_name, dfb_id);
        dfb_name_to_id[dfb_name] = dfb_id;
    }

    // Create Kernels
    for (const KernelSpec& kernel_spec : spec.kernels) {
        KernelSource kernel_src = MakeKernelSource(kernel_spec);
        NodeRangeSet node_ranges = to_node_range_set(kernel_spec.target_nodes);

        std::shared_ptr<Kernel> kernel;

        if (kernel_spec.is_dm_kernel()) {
            auto config = MakeQuasarDataMovementConfig(kernel_spec, dfb_name_to_id);
            auto processors = GetDMProcessorSet(kernel_to_dm_processor_mask_map.at(&kernel_spec));
            kernel = std::make_shared<experimental::quasar::QuasarDataMovementKernel>(
                kernel_src, node_ranges, config, processors);
        } else {
            auto config = MakeQuasarComputeConfig(kernel_spec, dfb_name_to_id);
            auto processors = GetComputeProcessorSet(kernel_to_compute_processor_mask_map.at(&kernel_spec));
            kernel = std::make_shared<experimental::quasar::QuasarComputeKernel>(
                kernel_src, node_ranges, config, processors);
        }

        // Add the kernel to the ProgramImpl and register the name -> handle mapping
        KernelHandle handle = program_impl->add_kernel(kernel, HalProgrammableCoreType::TENSIX);
        program_impl->register_kernel_spec_name(kernel_spec.unique_id, handle);

        // Register the RTA schema for validation
        const auto& schema = kernel_spec.runtime_arguments_schema;
        std::unordered_map<CoreCoord, size_t> num_rtas_per_node;
        for (const auto& [node_coord, num_args] : schema.num_runtime_args_per_node) {
            num_rtas_per_node[node_coord] = num_args;
        }
        program_impl->register_kernel_rta_schema(
            kernel_spec.unique_id, num_rtas_per_node, schema.num_common_runtime_args);
    }

    return Program(std::move(program_impl));
}

// ============================================================================
// IMPLEMENTATION: Utilities
// ============================================================================

inline tt::ARCH get_arch() {
    if (auto& override = arch_override(); override.has_value()) {
        return *override;
    }
    return tt::tt_metal::hal::get_arch();
}

inline bool is_gen2_arch() { return get_arch() == tt::ARCH::QUASAR; }

inline bool is_gen1_arch() {
    tt::ARCH arch = get_arch();
    return arch == tt::ARCH::WORMHOLE_B0 || arch == tt::ARCH::BLACKHOLE;
}

// TODO: Are these NodeRangeSet helpers really needed?
//       I'm sure a convert-to-CoreRangeSet helper already exists.
//       Everything is converted to CoreRangeSet upfront in the iterative API.

NodeRangeSet to_node_range_set(const std::variant<NodeCoord, NodeRange, NodeRangeSet>& nodes) {
    return std::visit(
        [](const auto& n) -> NodeRangeSet {
            using T = std::decay_t<decltype(n)>;
            if constexpr (std::is_same_v<T, NodeRangeSet>) {
                return n;
            } else if constexpr (std::is_same_v<T, NodeRange>) {
                return NodeRangeSet(n);
            } else {
                // NodeCoord case
                return NodeRangeSet(NodeRange(n, n));
            }
        },
        nodes);
}

bool nodes_intersect(
    const std::variant<NodeCoord, NodeRange, NodeRangeSet>& a,
    const std::variant<NodeCoord, NodeRange, NodeRangeSet>& b) {
    NodeRangeSet a_set = to_node_range_set(a);
    NodeRangeSet b_set = to_node_range_set(b);
    return a_set.intersects(b_set);
}

bool nodes_contains(
    const std::variant<NodeCoord, NodeRange, NodeRangeSet>& superset,
    const std::variant<NodeCoord, NodeRange, NodeRangeSet>& subset) {
    NodeRangeSet superset_node_range_set = to_node_range_set(superset);
    NodeRangeSet subset_node_range_set = to_node_range_set(subset);
    return superset_node_range_set.contains(subset_node_range_set);
}

// ============================================================================
// IMPLEMENTATION: Collection & Validation
// ============================================================================

// ----------------------------------------------------------------------------
// CollectSpecData: Build derived data structures from a ProgramSpec
// ----------------------------------------------------------------------------
//
// Indexes the ProgramSpec into lookup tables for efficient access.
// Function enforces STRUCTURAL invariants only:
//   - No duplicate names (would corrupt map lookups)
//   - No dangling references (would cause .at() failures later)
//   - Complete endpoint info (DFBs have both producer and consumer)
//
// If this function returns, the CollectedSpecData is internally consistent,
// and the ProgramSpec is structurally well-formed.
// Semantic validation (thread limits, architecture rules, etc.) is separate.

CollectedSpecData CollectSpecData(const ProgramSpec& spec) {
    CollectedSpecData collected;

    // Collect KernelSpecs
    for (const auto& kernel : spec.kernels) {
        auto [it, inserted] = collected.kernel_by_name.try_emplace(kernel.unique_id, &kernel);
        TT_FATAL(inserted, "Duplicate KernelSpec name '{}'", kernel.unique_id);
    }

    // Collect DataflowBufferSpecs
    for (const auto& dfb : spec.dataflow_buffers) {
        auto [it, inserted] = collected.dfb_by_name.try_emplace(dfb.unique_id, &dfb);
        TT_FATAL(inserted, "Duplicate DataflowBufferSpec name '{}'", dfb.unique_id);
    }

    // Build DFB endpoint info from kernel bindings
    for (const auto& kernel : spec.kernels) {
        for (const auto& dfb_binding : kernel.dfb_bindings) {
            // Referential integrity: the DFB must exist
            TT_FATAL(
                collected.dfb_by_name.contains(dfb_binding.dfb_spec_name),
                "Kernel '{}' references unknown DFB '{}'",
                kernel.unique_id,
                dfb_binding.dfb_spec_name);

            CollectedSpecData::DFBEndpointInfo& endpoint_info = collected.dfb_endpoints[dfb_binding.dfb_spec_name];

            if (dfb_binding.endpoint_type == KernelSpec::DFBEndpointType::PRODUCER) {
                TT_FATAL(
                    endpoint_info.producer == nullptr,
                    "DFB '{}' has multiple producers (second: '{}')",
                    dfb_binding.dfb_spec_name,
                    kernel.unique_id);
                endpoint_info.producer = &kernel;
                endpoint_info.producer_binding = &dfb_binding;
            } else if (dfb_binding.endpoint_type == KernelSpec::DFBEndpointType::CONSUMER) {
                TT_FATAL(
                    endpoint_info.consumer == nullptr,
                    "DFB '{}' has multiple consumers (second: '{}')",
                    dfb_binding.dfb_spec_name,
                    kernel.unique_id);
                endpoint_info.consumer = &kernel;
                endpoint_info.consumer_binding = &dfb_binding;
            } else {
                TT_FATAL(false, "RELAY endpoints are only used for remote DFB, which is not supported yet");
            }
        }
    }

    // Completeness: every DFB must have exactly one producer and one consumer
    for (const auto& [dfb_name, endpoint_info] : collected.dfb_endpoints) {
        TT_FATAL(endpoint_info.producer != nullptr, "DFB '{}' has no producer", dfb_name);
        TT_FATAL(endpoint_info.consumer != nullptr, "DFB '{}' has no consumer", dfb_name);
    }

    // Referential integrity: every declared DFB must be bound by some kernel
    for (const auto& dfb : spec.dataflow_buffers) {
        TT_FATAL(
            collected.dfb_endpoints.contains(dfb.unique_id),
            "DFB '{}' is defined but not bound by any kernel",
            dfb.unique_id);
    }

    // Collect SemaphoreSpecs
    for (const auto& semaphore : spec.semaphores) {
        auto [it, inserted] = collected.semaphore_by_name.try_emplace(semaphore.unique_id, &semaphore);
        TT_FATAL(inserted, "Duplicate SemaphoreSpec name '{}'", semaphore.unique_id);
    }

    return collected;
}

// ----------------------------------------------------------------------------
// ValidateProgramSpec: Semantic validation
// ----------------------------------------------------------------------------
//
// This function checks SEMANTIC rules (that don't affect the CollectedSpecData structure):
//   - Architecture requirements
//   - Resource limits
//   - Feature support
//   - Target node constraints (worker overlap, node coverage, node validity)
//
// Assumes CollectedSpecData is already built.

void ValidateProgramSpec(const ProgramSpec& spec, const CollectedSpecData& collected) {
    //////////////////////////////
    // Architecture checks
    //////////////////////////////

    TT_FATAL(
        tt::tt_metal::hal::get_arch() == tt::ARCH::QUASAR,
        "Metal 2.0 API is currently only implemented for Quasar. WH/BH support coming soon.");

    //////////////////////////////
    // Validate KernelSpecs
    //////////////////////////////

    // A Program needs at least one kernel
    TT_FATAL(!spec.kernels.empty(), "A ProgramSpec must have at least one KernelSpec");

    // Validate no unimplemented compiler options are used
    for (const auto& kernel : spec.kernels) {
        TT_FATAL(
            kernel.compiler_options.include_paths.empty(),
            "KernelSpec '{}' specifies include_paths -- this feature is not yet implemented. (Coming soon!)",
            kernel.unique_id);
    }

    // Validate no per-node thread maps are used (not yet implemented)
    for (const auto& kernel : spec.kernels) {
        TT_FATAL(
            !kernel.thread_node_map.has_value(),
            "KernelSpec '{}' specifies thread_node_map, but per-node thread counts are not implemented.",
            kernel.unique_id);
    }

    // Validate kernel thread counts
    for (const auto& kernel : spec.kernels) {
        TT_FATAL(kernel.num_threads > 0, "KernelSpec '{}' has no threads!", kernel.unique_id);
        if (kernel.is_compute_kernel()) {
            TT_FATAL(
                kernel.num_threads <= QUASAR_TENSIX_CORES_PER_NODE,
                "KernelSpec '{}' has too many threads. The architecture supports up to {} for compute kernels.",
                kernel.unique_id,
                QUASAR_TENSIX_CORES_PER_NODE);
        }
        if (kernel.is_dm_kernel()) {
            TT_FATAL(
                kernel.num_threads <= QUASAR_DM_CORES_PER_NODE,
                "KernelSpec '{}' has too many data movement threads. The architecture supports up to {} for data "
                "movement kernels.",
                kernel.unique_id,
                QUASAR_DM_CORES_PER_NODE);
        }
    }

    // Validate DM configs
    for (const auto& kernel : spec.kernels) {
        if (kernel.is_dm_kernel()) {
            const auto& data_movement_config = std::get<DataMovementConfiguration>(kernel.config_spec);

            // Both Gen1 and Gen2 configs are optional. But at least one must be specified.
            TT_FATAL(
                data_movement_config.gen1_data_movement_config.has_value() ||
                    data_movement_config.gen2_data_movement_config.has_value(),
                "KernelSpec '{}' must specify a DM config for Gen1, Gen2, or both.",
                kernel.unique_id);

            // The config for the current target architecture must be specified.
            if (is_gen2_arch()) {
                TT_FATAL(
                    data_movement_config.gen2_data_movement_config.has_value(),
                    "KernelSpec '{}' must specify a Gen2 DM config when targeting Quasar.",
                    kernel.unique_id);
            } else if (is_gen1_arch()) {
                TT_FATAL(
                    data_movement_config.gen1_data_movement_config.has_value(),
                    "KernelSpec '{}' must specify a Gen1 DM config when targeting WH or BH.",
                    kernel.unique_id);
            } else {
                TT_FATAL(false, "Unknown architecture");
            }
        }
    }

    // Validate compute kernel unpack_to_dest_mode references
    for (const auto& kernel : spec.kernels) {
        if (kernel.is_compute_kernel()) {
            const auto& compute_config = std::get<ComputeConfiguration>(kernel.config_spec);
            for (const auto& [dfb_name, mode] : compute_config.unpack_to_dest_mode) {
                TT_FATAL(
                    collected.dfb_by_name.contains(dfb_name),
                    "Kernel '{}' unpack_to_dest_mode references unknown DFB '{}'",
                    kernel.unique_id,
                    dfb_name);
            }
        }
    }

    //////////////////////////////////
    // Validate DataflowBufferSpecs
    //////////////////////////////////

    // Remote DFBs are not supported yet
    for (const auto& dfb : spec.dataflow_buffers) {
        TT_FATAL(!dfb.is_remote_dfb, "Remote DFBs are not supported yet");
        TT_FATAL(!dfb.producer_consumer_map || dfb.producer_consumer_map->empty(), "Remote DFBs are not supported yet");
    }

    // Borrowed memory is not supported yet
    for (const auto& dfb : spec.dataflow_buffers) {
        TT_FATAL(
            !dfb.uses_borrowed_memory,
            "DFB '{}' uses borrowed memory, but this feature is not yet implemented",
            dfb.unique_id);
    }

    // DFB aliasing is not supported yet
    for (const auto& dfb : spec.dataflow_buffers) {
        TT_FATAL(
            !dfb.alias_with.has_value() || dfb.alias_with->empty(),
            "DFB '{}' specifies alias_with, but DFB aliasing is not yet implemented",
            dfb.unique_id);
    }

    // Data format metadata (optional param) MUST be specified for a DFB with a compute endpoint
    for (const auto& [dfb_name, endpoint_info] : collected.dfb_endpoints) {
        if ((endpoint_info.producer && endpoint_info.producer->is_compute_kernel()) ||
            (endpoint_info.consumer && endpoint_info.consumer->is_compute_kernel())) {
            const DataflowBufferSpec* dfb_spec = collected.dfb_by_name.at(dfb_name);
            TT_FATAL(
                dfb_spec->data_format_metadata.has_value(),
                "DFB '{}' is used by a compute kernel, but no data_format_metadata is specified",
                dfb_name);
        }
    }

    //////////////////////////////////
    // Validate SemaphoreSpecs
    //////////////////////////////////

    // Semaphores aren't supported yet for Quasar
    TT_FATAL(spec.semaphores.empty(), "Semaphores are not supported yet");

    // Validate no semaphore bindings are used (semaphores not yet implemented)
    for (const auto& kernel : spec.kernels) {
        TT_FATAL(
            kernel.semaphore_bindings.empty(),
            "KernelSpec '{}' has semaphore bindings, but semaphores are not yet implemented",
            kernel.unique_id);
    }

    //////////////////////////////
    // Validate WorkerSpecs
    //////////////////////////////

    // WorkerSpecs are required on Gen2+
    // NOTE: WorkerSpec data is strictly redundant, but improves clarity and messaging.
    //       If it's hated, we can make it optional on Gen2 and derive the WorkerSpec if absent.
    TT_FATAL(spec.workers.has_value(), "Workers are required on Gen2+");
    const auto& workers = spec.workers.value();
    TT_FATAL(!workers.empty(), "At least one WorkerSpec is required");

    // WorkerSpecs may not overlap in their target nodes
    for (const auto& worker : workers) {
        for (const auto& other_worker : workers) {
            if (worker.unique_id == other_worker.unique_id) {
                continue;
            }
            if (nodes_intersect(worker.target_nodes, other_worker.target_nodes)) {
                TT_FATAL(
                    false,
                    "WorkerSpecs '{}' and '{}' overlap in target nodes",
                    worker.unique_id,
                    other_worker.unique_id);
            }
        }
    }

    // A WorkerSpec must have at least one kernel
    for (const auto& worker : workers) {
        TT_FATAL(!worker.kernels.empty(), "WorkerSpec '{}' has no kernels", worker.unique_id);
    }

    // Does the Worker have enough cores to run all its kernels?
    for (const auto& worker : workers) {
        uint32_t dm_cores_needed = 0;
        uint32_t compute_cores_needed = 0;
        for (const auto& kernel_name : worker.kernels) {
            const auto& kernel_spec = collected.kernel_by_name.at(kernel_name);
            if (kernel_spec->is_compute_kernel()) {
                compute_cores_needed += kernel_spec->num_threads;
            }
            if (kernel_spec->is_dm_kernel()) {
                dm_cores_needed += kernel_spec->num_threads;
            }
        }
        TT_FATAL(
            compute_cores_needed <= QUASAR_TENSIX_CORES_PER_NODE,
            "WorkerSpec '{}' needs {} compute cores, but only {} are available",
            worker.unique_id,
            compute_cores_needed,
            QUASAR_TENSIX_CORES_PER_NODE);
        TT_FATAL(
            dm_cores_needed <= QUASAR_DM_CORES_PER_NODE,
            "WorkerSpec '{}' needs {} data movement cores, but only {} are available",
            worker.unique_id,
            dm_cores_needed,
            QUASAR_DM_CORES_PER_NODE);
    }

    // A worker can have at most one compute kernel
    for (const auto& worker : workers) {
        uint32_t num_compute_kernels = 0;
        for (const auto& kernel_name : worker.kernels) {
            const auto& kernel_spec = collected.kernel_by_name.at(kernel_name);
            if (kernel_spec->is_compute_kernel()) {
                num_compute_kernels++;
            }
        }
        TT_FATAL(num_compute_kernels <= 1, "WorkerSpec '{}' has more than one compute kernel", worker.unique_id);
    }

    // Check that WorkerSpec target nodes are valid nodes
    // (TODO once legality rules on Quasar are defined)
    // (For WH/BH, this check should catch attempts to assign kernels to dispatch nodes.)

    // Kernels in a WorkerSpec must contain the WorkerSpec's target nodes
    for (const auto& worker : workers) {
        for (const auto& kernel_name : worker.kernels) {
            const auto& kernel_spec = collected.kernel_by_name.at(kernel_name);
            TT_FATAL(
                nodes_contains(kernel_spec->target_nodes, worker.target_nodes),
                "Kernel '{}' target nodes must contain WorkerSpec '{}' target nodes",
                kernel_name,
                worker.unique_id);
        }
    }

    // DFBs in a WorkerSpec must contain the WorkerSpec's target nodes
    for (const auto& worker : workers) {
        for (const auto& dfb_name : worker.dataflow_buffers) {
            const auto& dfb_spec = collected.dfb_by_name.at(dfb_name);
            TT_FATAL(
                nodes_contains(dfb_spec->target_nodes, worker.target_nodes),
                "DFB '{}' target nodes must contain WorkerSpec '{}' target nodes",
                dfb_name,
                worker.unique_id);
        }
    }

    // Semaphores in a WorkerSpec must contain the WorkerSpec's target nodes
    for (const auto& worker : workers) {
        for (const auto& semaphore_name : worker.semaphores) {
            const auto& semaphore_spec = collected.semaphore_by_name.at(semaphore_name);
            TT_FATAL(
                nodes_contains(semaphore_spec->target_nodes, worker.target_nodes),
                "Semaphore '{}' target nodes must contain WorkerSpec '{}' target nodes",
                semaphore_name,
                worker.unique_id);
        }
    }

    // Check that all kernel target nodes are "accounted for" by a WorkerSpec
    std::unordered_map<KernelSpecName, NodeRangeSet> kernel_node_ranges;
    for (const auto& worker : workers) {
        for (const auto& kernel_name : worker.kernels) {
            // A kernel may belong to multiple WorkerSpecs.
            // Merge the target nodes of all WorkerSpecs that contain this kernel.
            kernel_node_ranges[kernel_name].merge(to_node_range_set(worker.target_nodes));
        }
    }
    for (const auto& kernel : spec.kernels) {
        KernelSpecName kernel_name = kernel.unique_id;
        NodeRangeSet kernel_target_nodes = to_node_range_set(kernel.target_nodes);

        // The kernel must belong to at least one WorkerSpec
        TT_FATAL(kernel_node_ranges.contains(kernel_name), "Kernel '{}' is not part of any WorkerSpec", kernel_name);

        // All the kernel's target nodes must be accounted for by the WorkerSpecs that contain it
        NodeRangeSet worker_derived_target_nodes = kernel_node_ranges.at(kernel_name);
        TT_FATAL(
            worker_derived_target_nodes == kernel_target_nodes,
            "Kernel '{}' has target nodes that are not accounted for by any WorkerSpec",
            kernel_name);
    }

    // Check that all DFB target nodes are "accounted for" by a WorkerSpec
    // (May revisit this, depending on how remote DFBs are implemented)
    std::unordered_map<DFBSpecName, NodeRangeSet> dfb_node_ranges;
    for (const auto& worker : workers) {
        for (const auto& dfb_name : worker.dataflow_buffers) {
            // A DFB may belong to multiple WorkerSpecs.
            // Merge the target nodes of all WorkerSpecs that have this DFB.
            dfb_node_ranges[dfb_name].merge(to_node_range_set(worker.target_nodes));
        }
    }
    for (const auto& dfb : spec.dataflow_buffers) {
        DFBSpecName dfb_name = dfb.unique_id;
        NodeRangeSet dfb_target_nodes = to_node_range_set(dfb.target_nodes);

        // The DFB must belong to at least one WorkerSpec
        TT_FATAL(dfb_node_ranges.contains(dfb_name), "DFB '{}' is not part of any WorkerSpec", dfb_name);

        // All the DFB's target nodes must be accounted for by the WorkerSpecs that have it
        NodeRangeSet worker_derived_target_nodes = dfb_node_ranges.at(dfb_name);
        TT_FATAL(
            worker_derived_target_nodes == dfb_target_nodes,
            "DFB '{}' has target nodes that are not accounted for by any WorkerSpec",
            dfb_name);
    }
}

// ============================================================================
// IMPLEMENTATION: Processor Assignment
// ============================================================================

// ProcessorMask factory functions
template <uint8_t NUM_CORES>
ProcessorMask<NUM_CORES> CreateMask(uint8_t mask) {
    TT_FATAL(
        mask <= ProcessorMask<NUM_CORES>::VALID_BITS_MASK,
        "Mask specifies too many cores for ProcessorMask<{}>: {}",
        NUM_CORES,
        mask);
    return {mask};
}

template <uint8_t NUM_CORES>
std::optional<ProcessorMask<NUM_CORES>> ReserveProcessors(uint8_t n, const ProcessorMask<NUM_CORES>& already_in_use) {
    if (already_in_use.num_available() < n) {
        return std::nullopt;
    }

    ProcessorMask<NUM_CORES> newly_reserved;
    for (uint8_t i = 0; i < NUM_CORES && n > 0; i++) {
        if (already_in_use.is_idx_available(i)) {
            newly_reserved.bits |= (1 << i);
            n--;
        }
    }
    return newly_reserved;
}

// Reserve DM processors for a kernel on a WorkerSpec.
// Returns {this_kernel_mask, updated_cumulative_mask}
// Throws TT_FATAL on conflict or allocation failure (see simplifying assumption notes)
std::pair<DMProcessorMask, DMProcessorMask> ReserveDMProcessors(
    const KernelSpec* kernel_spec,
    std::optional<DMProcessorMask> existing_mask,
    DMProcessorMask cumulative_mask,
    const WorkerSpecName& worker_id) {
    // Was this kernel already assigned a mask from a previous WorkerSpec?
    if (existing_mask.has_value()) {
        DMProcessorMask existing = existing_mask.value();

        // Check for conflict with what's already allocated on the current WorkerSpec
        TT_FATAL(
            !existing.conflicts_with(cumulative_mask),
            "Kernel '{}' requires processors already in use on WorkerSpec '{}'. "
            "The \"common DM cores\" assumption has been violated!",
            kernel_spec->unique_id,
            worker_id);

        // Return existing mask and updated cumulative
        return {existing, cumulative_mask | existing};
    }

    // First time seeing this kernel - reserve new processors
    std::optional<DMProcessorMask> reserved = ReserveProcessors(kernel_spec->num_threads, cumulative_mask);
    TT_FATAL(
        reserved.has_value(),
        "Failed to reserve processors for DM kernel '{}' on WorkerSpec '{}'. "
        "The \"common DM cores\" assumption has been violated!",
        kernel_spec->unique_id,
        worker_id);

    DMProcessorMask mask = reserved.value();
    return {mask, cumulative_mask | mask};
}

// Assign compute processor mask for a kernel.
ComputeEngineMask AssignComputeProcessors(const KernelSpec* kernel_spec, const KernelSpecName& kernel_name) {
    auto reserved = ReserveProcessors(kernel_spec->num_threads, CreateMask<QUASAR_TENSIX_CORES_PER_NODE>(0x00));
    TT_FATAL(
        reserved.has_value(),
        "Compute kernel '{}' reservation failed. Condition should be unreachable after validation.",
        kernel_name);
    return reserved.value();
}

// Solve kernel-to-core assignments.
// NOTE: Despite the earlier legality checks, it is possible for the solver to fail! (with TT_FATAL)
//   The current implementation makes a (likely temporary) simplifying assumption:
//      A given DM kernel will run on the _same_ set of DM cores on every node/cluster.
//   If the input ProgramSpec passes legality checks but fails in the solver, the resulting error
//   message will make it clear what went wrong (i.e. overly restrictive "common DM cores" assumption).
std::pair<DMProcessorMaskMap, ComputeEngineMaskMap> SolveKernelToProcessorAssignments(
    const ProgramSpec& spec, const CollectedSpecData& collected) {
    DMProcessorMaskMap kernels_to_dm_processor_mask;
    ComputeEngineMaskMap kernels_to_compute_processor_mask;

    for (const WorkerSpec& worker : spec.workers.value()) {
        // Cumulative DM processor mask for this WorkerSpec
        // (If we decide to reserve DM cores for internal use, just update the initial mask here.)
        DMProcessorMask cumulative_dm_mask = CreateMask<QUASAR_DM_CORES_PER_NODE>(0x00);

        // Since we enforce (at most) one compute kernel per WorkerSpec, no need to track cumulative mask.

        for (const KernelSpecName& kernel_name : worker.kernels) {
            const KernelSpec* kernel_spec = collected.kernel_by_name.at(kernel_name);

            if (kernel_spec->is_dm_kernel()) {
                // Look up existing DM mask, if any (from previous WorkerSpec iterations)
                std::optional<DMProcessorMask> existing_mask =
                    kernels_to_dm_processor_mask.contains(kernel_spec)
                        ? std::optional(kernels_to_dm_processor_mask.at(kernel_spec))
                        : std::nullopt;

                auto [mask, new_cumulative] =
                    ReserveDMProcessors(kernel_spec, existing_mask, cumulative_dm_mask, worker.unique_id);

                kernels_to_dm_processor_mask[kernel_spec] = mask;
                cumulative_dm_mask = new_cumulative;
            } else {
                // Compute kernel
                kernels_to_compute_processor_mask[kernel_spec] = AssignComputeProcessors(kernel_spec, kernel_name);
            }
        }
    }

    return std::make_pair(kernels_to_dm_processor_mask, kernels_to_compute_processor_mask);
}

// ============================================================================
// IMPLEMENTATION: Program Building
// ============================================================================

// Create a DataflowBufferConfig from a DataflowBufferSpec and endpoint info.
experimental::dfb::DataflowBufferConfig MakeDataflowBufferConfig(
    const DataflowBufferSpec* dfb_spec,
    const CollectedSpecData::DFBEndpointInfo& dfb_endpoint_info,
    const DMProcessorMaskMap& kernel_to_dm_processor_mask_map,
    const ComputeEngineMaskMap& kernel_to_compute_processor_mask_map) {
    const KernelSpec* producer = dfb_endpoint_info.producer;
    const KernelSpec* consumer = dfb_endpoint_info.consumer;

    // Create the combined processor mask for producer and consumer
    // (uint16_t, where bits 0-7 = DM riscs, bits 8-15 = Tensix riscs)
    auto get_dfb_risc_mask = [&](const KernelSpec* kernel) -> uint16_t {
        if (kernel->is_dm_kernel()) {
            return kernel_to_dm_processor_mask_map.at(kernel).bits;
        } else {
            return kernel_to_compute_processor_mask_map.at(kernel).bits << 8;
        }
    };
    uint16_t producer_risc_mask = get_dfb_risc_mask(producer);
    uint16_t consumer_risc_mask = get_dfb_risc_mask(consumer);

    // Convert user-facing access pattern enum to hardware interface access pattern enum
    // (TODO: We should merge these enums; it's silly to have separate ones.)
    auto to_hw_access_pattern = [](DFBAccessPattern pattern) -> experimental::dfb::AccessPattern {
        switch (pattern) {
            case DFBAccessPattern::STRIDED: return experimental::dfb::AccessPattern::STRIDED;
            case DFBAccessPattern::BLOCKED: return experimental::dfb::AccessPattern::BLOCKED;
            case DFBAccessPattern::CONTIGUOUS: TT_FATAL(false, "CONTIGUOUS access pattern is not yet supported");
        }
        TT_FATAL(false, "Unknown DFBAccessPattern");
    };
    auto producer_access_pattern = to_hw_access_pattern(dfb_endpoint_info.producer_binding->access_pattern);
    auto consumer_access_pattern = to_hw_access_pattern(dfb_endpoint_info.consumer_binding->access_pattern);

    return experimental::dfb::DataflowBufferConfig{
        .entry_size = dfb_spec->entry_size,
        .num_entries = dfb_spec->num_entries,
        .producer_risc_mask = producer_risc_mask,
        .num_producers = dfb_endpoint_info.producer->num_threads,
        .pap = producer_access_pattern,
        .consumer_risc_mask = consumer_risc_mask,
        .num_consumers = dfb_endpoint_info.consumer->num_threads,
        .cap = consumer_access_pattern,
        .enable_implicit_sync = !dfb_spec->disable_implicit_sync,
        .data_format = dfb_spec->data_format_metadata.value_or(tt::DataFormat::Invalid),
        .tile = dfb_spec->tile_format_metadata};
}

// ----------------------------------------------------------------------------
// MakeKernelSource: Create a KernelSource from a KernelSpec
// ----------------------------------------------------------------------------

KernelSource MakeKernelSource(const KernelSpec& kernel_spec) {
    KernelSource::SourceType source_type = (kernel_spec.source_type == KernelSpec::SourceType::FILE_PATH)
                                               ? KernelSource::SourceType::FILE_PATH
                                               : KernelSource::SourceType::SOURCE_CODE;
    return KernelSource(kernel_spec.source, source_type);
}

// ----------------------------------------------------------------------------
// MakeQuasarDataMovementConfig: Create a QuasarDataMovementConfig from a KernelSpec
// ----------------------------------------------------------------------------

experimental::quasar::QuasarDataMovementConfig MakeQuasarDataMovementConfig(
    const KernelSpec& kernel_spec, const DFBNameToIdMap& dfb_name_to_id) {
    TT_FATAL(kernel_spec.is_dm_kernel(), "Expected a DM kernel");

    // Convert defines from vector<pair> to map
    std::map<std::string, std::string> defines_map;
    for (const auto& [key, value] : kernel_spec.compiler_options.defines) {
        defines_map[key] = value;
    }

    // Start with user-provided compile-time args, then add DFB accessor mappings
    // TODO: This is a TEMPORARY solution to pass DFB accessor names to the kernel.
    //   A follow-up PR that will introduce a more elegant kernel-side mechanism
    //    for creating DFBs from local accessor names.
    auto named_compile_args = kernel_spec.compile_time_arg_bindings;
    for (const auto& dfb_binding : kernel_spec.dfb_bindings) {
        uint32_t dfb_id = dfb_name_to_id.at(dfb_binding.dfb_spec_name);
        named_compile_args[dfb_binding.local_accessor_name] = dfb_id;
    }

    return experimental::quasar::QuasarDataMovementConfig{
        .num_threads_per_cluster = kernel_spec.num_threads,
        .compile_args = {},  // Compile args are passed via named_compile_args
        .defines = defines_map,
        .named_compile_args = named_compile_args,
        .is_legacy_kernel = false,
        .opt_level = kernel_spec.compiler_options.opt_level,
    };
}

// ----------------------------------------------------------------------------
// MakeQuasarComputeConfig: Create a QuasarComputeConfig from a KernelSpec
// ----------------------------------------------------------------------------

experimental::quasar::QuasarComputeConfig MakeQuasarComputeConfig(
    const KernelSpec& kernel_spec, const DFBNameToIdMap& dfb_name_to_id) {
    TT_FATAL(kernel_spec.is_compute_kernel(), "Expected a compute kernel");
    const auto& compute_config = std::get<ComputeConfiguration>(kernel_spec.config_spec);

    // Handle unpack_to_dest_mode:
    //  - The user-facing KernelSpec provides the modes keyed by DFB name.
    //  - The unpack_to_dest_mode vector in the QuasarComputeConfig is indexed by DFB ID.
    //  - DFB IDs are always issued sequentially from zero, so this works.

    // Size the vector to the number of DFBs.
    std::vector<UnpackToDestMode> unpack_modes(dfb_name_to_id.size(), UnpackToDestMode::Default);

    // Populate unpack_modes using DFB ID as the index
    for (const auto& [dfb_name, mode] : compute_config.unpack_to_dest_mode) {
        uint32_t dfb_id = dfb_name_to_id.at(dfb_name);
        unpack_modes[dfb_id] = mode;
    }

    // Handle defines:
    // Must convert from vector<pair> to map.)
    // (TODO: Consider changing this in the KernelSpec API to avoid unnecessary conversion?)
    // (The design motivation was consistency with the existing ProgramDescriptor API.)
    std::map<std::string, std::string> defines_map;
    for (const auto& [key, value] : kernel_spec.compiler_options.defines) {
        defines_map[key] = value;
    }

    // Start with user-provided compile-time args, then add DFB accessor mappings
    // TODO: This is a TEMPORARY solution to pass DFB accessor names to the kernel.
    //    A follow-up PR that will introduce a more elegant kernel-side mechanism
    //    for creating DFBs from local accessor names.
    auto named_compile_args = kernel_spec.compile_time_arg_bindings;
    for (const auto& dfb_binding : kernel_spec.dfb_bindings) {
        uint32_t dfb_id = dfb_name_to_id.at(dfb_binding.dfb_spec_name);
        named_compile_args[dfb_binding.local_accessor_name] = dfb_id;
    }

    return experimental::quasar::QuasarComputeConfig{
        .num_threads_per_cluster = kernel_spec.num_threads,
        .math_fidelity = compute_config.math_fidelity,
        .fp32_dest_acc_en = compute_config.fp32_dest_acc_en,
        .dst_full_sync_en = compute_config.dst_full_sync_en,
        .unpack_to_dest_mode = unpack_modes,
        .bfp8_pack_precise = compute_config.bfp8_pack_precise,
        .math_approx_mode = compute_config.math_approx_mode,
        .compile_args = {},  // Compile args are passed via named_compile_args
        .defines = defines_map,
        .named_compile_args = named_compile_args,
        .opt_level = kernel_spec.compiler_options.opt_level,
    };
}

// --------------------------------------------------------------------------------
// GetDMProcessorSet: Convert a DMProcessorMask to a set of DataMovementProcessor
// --------------------------------------------------------------------------------

std::set<DataMovementProcessor> GetDMProcessorSet(DMProcessorMask mask) {
    std::set<DataMovementProcessor> processors;
    for (uint8_t i = 0; i < QUASAR_DM_CORES_PER_NODE; ++i) {
        if (mask.is_idx_in_use(i)) {
            processors.insert(static_cast<DataMovementProcessor>(i));
        }
    }
    return processors;
}

// ------------------------------------------------------------------------------------------
// GetComputeProcessorSet: Convert a ComputeEngineMask to a set of QuasarComputeProcessor
// ------------------------------------------------------------------------------------------
//
// The ComputeEngineMask represents the active Tensix engines on a node.
// (Based on the number of compute kernel threads running on that node.)
// Each Tensix engine has 4 compute processors.
// So if bit i is set in the mask, we include all 4 processors for that engine:
//   Engine 0 -> NEO_0_COMPUTE_{0,1,2,3}
//   Engine 1 -> NEO_1_COMPUTE_{0,1,2,3}
//   etc.

std::set<experimental::quasar::QuasarComputeProcessor> GetComputeProcessorSet(ComputeEngineMask mask) {
    using QuasarComputeProcessor = experimental::quasar::QuasarComputeProcessor;
    constexpr uint8_t PROCESSORS_PER_ENGINE = experimental::quasar::QUASAR_NUM_COMPUTE_PROCESSORS_PER_TENSIX_ENGINE;

    std::set<QuasarComputeProcessor> processors;
    for (uint8_t engine = 0; engine < QUASAR_TENSIX_CORES_PER_NODE; ++engine) {
        if (mask.is_idx_in_use(engine)) {
            // Add all 4 compute processors for this engine
            for (uint8_t proc = 0; proc < PROCESSORS_PER_ENGINE; ++proc) {
                uint8_t processor_id = engine * PROCESSORS_PER_ENGINE + proc;
                processors.insert(static_cast<QuasarComputeProcessor>(processor_id));
            }
        }
    }
    return processors;
}

}  // namespace tt::tt_metal::experimental::metal2_host_api
