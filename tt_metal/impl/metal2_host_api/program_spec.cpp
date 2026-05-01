// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <bit>
#include <functional>
#include <limits>
#include <set>
#include <string_view>
#include <unordered_set>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>  // fmt::formatter<tt::DataFormat> for TT_FATAL messages
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/context/metal_env_accessor.hpp"
#include "impl/dispatch/dispatch_core_manager.hpp"
#include <core_descriptor.hpp>
#include <llrt/tt_cluster.hpp>

namespace tt::tt_metal::experimental::metal2_host_api {

// ============================================================================
// Constants
// ============================================================================

// TODO: These constants should be queriable from the public API (currently HAL, for consistency)
//       They are currently also hardcoded in the temporary Quasar host_api.hpp. Need to clean this up.
static constexpr uint32_t QUASAR_DM_CORES_PER_NODE = 8;
static constexpr uint32_t QUASAR_RESERVED_DM_CORES_PER_NODE = 2;  // DM0 and DM1 reserved for internal use
static constexpr uint32_t QUASAR_USER_DM_CORES_PER_NODE = QUASAR_DM_CORES_PER_NODE - QUASAR_RESERVED_DM_CORES_PER_NODE;
static constexpr uint32_t QUASAR_TENSIX_ENGINES_PER_NODE = 4;

// ============================================================================
// Type Definitions
// ============================================================================

// Data structure built up from ProgramSpec to enable fast lookups
struct CollectedSpecData {
    // Name -> spec lookups.
    // dfb_by_name covers BOTH local and remote DFBs.
    // For remote DFBs, the pointee is the inner dfb_spec.
    // To check if a DFB is remote, check the remote_dfb_by_name map.
    std::unordered_map<KernelSpecName, const KernelSpec*> kernel_by_name;
    std::unordered_map<DFBSpecName, const DataflowBufferSpec*> dfb_by_name;
    std::unordered_map<DFBSpecName, const RemoteDataflowBufferSpec*> remote_dfb_by_name;
    std::unordered_map<SemaphoreSpecName, const SemaphoreSpec*> semaphore_by_name;

    // DFB endpoint info (derived from kernel bindings).
    // Populated for both local and remote DFBs.
    struct DFBEndpointInfo {
        const KernelSpec* producer = nullptr;
        const KernelSpec* consumer = nullptr;
        const KernelSpec::DFBBinding* producer_binding = nullptr;
        const KernelSpec::DFBBinding* consumer_binding = nullptr;
    };
    std::unordered_map<DFBSpecName, DFBEndpointInfo> dfb_endpoints;

    // WorkUnit membership: a kernel may belong to multiple WorkUnitSpecs.
    std::unordered_map<KernelSpecName, std::vector<const WorkUnitSpec*>> kernel_work_units;

    // Derived node sets:
    //  - kernel_node_set: union of containing WorkUnitSpec target_nodes
    //  - dfb_node_set: union of binding-kernels' node sets (local DFBs only).
    std::unordered_map<KernelSpecName, NodeRangeSet> kernel_node_set;
    std::unordered_map<DFBSpecName, NodeRangeSet> dfb_node_set;
};

// Bitmask for tracking processor allocation on a node
template <uint8_t NUM_CORES>
struct ProcessorMask {
    static_assert(NUM_CORES > 0 && NUM_CORES <= 8, "ProcessorMask supports 1-8 processors");
    static constexpr uint8_t VALID_BITS_MASK = (NUM_CORES == 8) ? 0xFF : ((1 << NUM_CORES) - 1);

    uint8_t bits = 0x00;

    // Operators
    bool operator==(ProcessorMask other) const { return bits == other.bits; }
    bool operator!=(ProcessorMask other) const { return bits != other.bits; }
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
using ComputeEngineMask = ProcessorMask<QUASAR_TENSIX_ENGINES_PER_NODE>;

// Kernel -> ProcessorMask maps (Gen2/Quasar only)
using DMProcessorMaskMap = std::unordered_map<const KernelSpec*, DMProcessorMask>;
using ComputeEngineMaskMap = std::unordered_map<const KernelSpec*, ComputeEngineMask>;

// Kernel -> DFB risc mask (passed to MakeDataflowBufferConfig)
//   Gen1: bit 0 = RISCV_0 (BRISC), bit 1 = RISCV_1 (NCRISC), bit 2 = Tensix compute
//   Gen2: bits 0-7 = DM processors, bits 8-15 = Tensix compute engines
using KernelRiscMaskMap = std::unordered_map<const KernelSpec*, uint16_t>;

// DFB name -> DFB ID map (for unpack_to_dest_mode indexing)
using DFBNameToIdMap = std::unordered_map<DFBSpecName, uint32_t>;
using SemaphoreNameToIdMap = std::unordered_map<SemaphoreSpecName, uint32_t>;

// ============================================================================
// Basic Utility Helpers
// ============================================================================

inline tt::ARCH get_arch() { return tt::tt_metal::hal::get_arch(); }

inline bool is_gen2_arch() { return get_arch() == tt::ARCH::QUASAR; }

inline bool is_gen1_arch() {
    tt::ARCH arch = get_arch();
    return arch == tt::ARCH::WORMHOLE_B0 || arch == tt::ARCH::BLACKHOLE;
}

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

// Local accessor names for kernel resource bindings must be valid C++ identifiers
// They are used verbatim in the generated kernel source code.
// TODO: Move this to ttsl in a follow up PR
bool IsValidCppIdentifier(std::string_view s) {
    if (s.empty()) {
        return false;
    }
    // Reject names with non-identifier characters or an empty/leading-digit form.
    const unsigned char c0 = static_cast<unsigned char>(s[0]);
    if (!((c0 >= 'a' && c0 <= 'z') || (c0 >= 'A' && c0 <= 'Z') || c0 == '_')) {
        return false;
    }
    for (size_t i = 1; i < s.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_')) {
            return false;
        }
    }

    // Reject reserved identifier patterns per [lex.name]/3.
    // Names containing "__", or starting with "_" followed by an uppercase letter.
    if (s.size() >= 2 && s[0] == '_' && s[1] >= 'A' && s[1] <= 'Z') {
        return false;
    }
    if (s.find("__") != std::string_view::npos) {
        return false;
    }

    // Reject C++ keywords. Anything in this set would produce uncompilable code
    // when emitted as a variable identifier in kernel_bindings_generated.h.
    static const std::unordered_set<std::string_view> kCppKeywords = {
        "alignas",     "alignof",   "and",        "and_eq",    "asm",      "auto",         "bitand",
        "bitor",       "bool",      "break",      "case",      "catch",    "char",         "char8_t",
        "char16_t",    "char32_t",  "class",      "compl",     "concept",  "const",        "consteval",
        "constexpr",   "constinit", "const_cast", "continue",  "co_await", "co_return",    "co_yield",
        "decltype",    "default",   "delete",     "do",        "double",   "dynamic_cast", "else",
        "enum",        "explicit",  "export",     "extern",    "false",    "float",        "for",
        "friend",      "goto",      "if",         "inline",    "int",      "long",         "mutable",
        "namespace",   "new",       "noexcept",   "not",       "not_eq",   "nullptr",      "operator",
        "or",          "or_eq",     "private",    "protected", "public",   "register",     "reinterpret_cast",
        "requires",    "return",    "short",      "signed",    "sizeof",   "static",       "static_assert",
        "static_cast", "struct",    "switch",     "template",  "this",     "thread_local", "throw",
        "true",        "try",       "typedef",    "typeid",    "typename", "union",        "unsigned",
        "using",       "virtual",   "void",       "volatile",  "wchar_t",  "while",        "xor",
        "xor_eq",
    };

    // If we got this far, and the name doesn't match any keywords, it's valid.
    return !kCppKeywords.contains(s);
}

// ============================================================================
// Step 1: Spec Collection & Validation
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

    // Collect DataflowBufferSpecs (local DFBs)
    for (const auto& dfb : spec.dataflow_buffers) {
        auto [it, inserted] = collected.dfb_by_name.try_emplace(dfb.unique_id, &dfb);
        TT_FATAL(inserted, "Duplicate DataflowBufferSpec name '{}'", dfb.unique_id);
    }

    // Collect RemoteDataflowBufferSpecs (remote DFBs).
    // Remote DFBs share the DFB name space with local DFBs, since kernel bindings
    // refer to either kind by the same DFBSpecName.
    for (const auto& remote_dfb : spec.remote_dataflow_buffers) {
        const DFBSpecName& name = remote_dfb.dfb_spec.unique_id;
        auto [it1, inserted1] = collected.dfb_by_name.try_emplace(name, &remote_dfb.dfb_spec);
        TT_FATAL(inserted1, "Duplicate DataflowBufferSpec name '{}' (across local and remote DFBs)", name);
        auto [it2, inserted2] = collected.remote_dfb_by_name.try_emplace(name, &remote_dfb);
        TT_FATAL(inserted2, "Duplicate RemoteDataflowBufferSpec name '{}'", name);
    }

    // Build DFB endpoint info from kernel bindings
    for (const auto& kernel : spec.kernels) {
        // Check for duplicate local_accessor_names within this kernel
        std::unordered_set<std::string> accessor_names;
        for (const auto& dfb_binding : kernel.dfb_bindings) {
            auto [it, inserted] = accessor_names.insert(dfb_binding.local_accessor_name);
            TT_FATAL(
                inserted,
                "Kernel '{}' has duplicate local_accessor_name '{}'",
                kernel.unique_id,
                dfb_binding.local_accessor_name);
            TT_FATAL(
                IsValidCppIdentifier(dfb_binding.local_accessor_name),
                "Kernel '{}' DFB local_accessor_name '{}' must be a valid C++ identifier",
                kernel.unique_id,
                dfb_binding.local_accessor_name);

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

    // Referential integrity: every declared DFB (local or remote) must be bound by some kernel
    for (const auto& dfb : spec.dataflow_buffers) {
        TT_FATAL(
            collected.dfb_endpoints.contains(dfb.unique_id),
            "DFB '{}' is defined but not bound by any kernel",
            dfb.unique_id);
    }
    for (const auto& remote_dfb : spec.remote_dataflow_buffers) {
        const DFBSpecName& name = remote_dfb.dfb_spec.unique_id;
        TT_FATAL(
            collected.dfb_endpoints.contains(name),
            "RemoteDataflowBufferSpec '{}' is defined but not bound by any kernel",
            name);
    }

    // Collect SemaphoreSpecs
    for (const auto& semaphore : spec.semaphores) {
        auto [it, inserted] = collected.semaphore_by_name.try_emplace(semaphore.unique_id, &semaphore);
        TT_FATAL(inserted, "Duplicate SemaphoreSpec name '{}'", semaphore.unique_id);
    }

    // Validate semaphore bindings
    for (const auto& kernel : spec.kernels) {
        std::unordered_set<std::string> accessor_names;
        for (const auto& binding : kernel.semaphore_bindings) {
            auto [it, inserted] = accessor_names.insert(binding.accessor_name);
            TT_FATAL(
                inserted,
                "Kernel '{}' has duplicate semaphore accessor_name '{}'",
                kernel.unique_id,
                binding.accessor_name);
            TT_FATAL(
                IsValidCppIdentifier(binding.accessor_name),
                "Kernel '{}' semaphore accessor_name '{}' must be a valid C++ identifier",
                kernel.unique_id,
                binding.accessor_name);
            TT_FATAL(
                collected.semaphore_by_name.contains(binding.semaphore_spec_name),
                "Kernel '{}' references unknown semaphore '{}'",
                kernel.unique_id,
                binding.semaphore_spec_name);
        }
    }

    // Check for duplicate WorkUnitSpec unique_ids
    {
        std::unordered_set<WorkUnitSpecName> work_unit_names;
        for (const auto& work_unit : spec.work_units) {
            auto [it, inserted] = work_unit_names.insert(work_unit.unique_id);
            TT_FATAL(inserted, "Duplicate WorkUnitSpec name '{}'", work_unit.unique_id);
        }
    }

    // Build WorkUnitSpec membership for each kernel, validating references along the way.
    // A kernel may belong to multiple WorkUnitSpecs; its effective target node set is the union.
    for (const auto& work_unit : spec.work_units) {
        for (const auto& kernel_name : work_unit.kernels) {
            TT_FATAL(
                collected.kernel_by_name.contains(kernel_name),
                "WorkUnitSpec '{}' references unknown kernel '{}'",
                work_unit.unique_id,
                kernel_name);
            collected.kernel_work_units[kernel_name].push_back(&work_unit);
        }
    }

    // Every declared kernel must be referenced by at least one WorkUnitSpec
    // (otherwise, it has no node placement and would never run).
    for (const auto& kernel : spec.kernels) {
        TT_FATAL(
            collected.kernel_work_units.contains(kernel.unique_id),
            "Kernel '{}' is not referenced by any WorkUnitSpec",
            kernel.unique_id);
    }

    // Derive each kernel's effective target node set: union of containing WorkUnitSpec target_nodes.
    for (const auto& [kernel_name, work_units] : collected.kernel_work_units) {
        NodeRangeSet node_set;
        for (const WorkUnitSpec* work_unit : work_units) {
            node_set = node_set.merge(to_node_range_set(work_unit->target_nodes));
        }
        collected.kernel_node_set[kernel_name] = node_set;
    }

    // Derive each local DFB's allocation node set: union of binding-kernels' node sets.
    // (Collected, but unvalidated. Semantic integrity checks for DFB take place in ValidateProgramSpec.)
    for (const auto& dfb : spec.dataflow_buffers) {
        const auto& endpoints = collected.dfb_endpoints.at(dfb.unique_id);
        NodeRangeSet node_set = collected.kernel_node_set.at(endpoints.producer->unique_id);
        node_set = node_set.merge(collected.kernel_node_set.at(endpoints.consumer->unique_id));
        collected.dfb_node_set[dfb.unique_id] = node_set;
    }

    return collected;
}

// ----------------------------------------------------------------------------
// ValidateNodeBounds: Node coordinate bounds checking
// ----------------------------------------------------------------------------
//
// Validates that every NodeCoord referenced by a WorkUnitSpec or SemaphoreSpec
// is within the compute worker grid on this device.
// (Kernel and DFB placement is derived from WorkUnitSpec membership, so
// bounds-checking the WorkUnitSpecs covers them too.)
//
// NOTE: We're dealing in logical coordinates. (Harvesting is handled by UMD.)
//
// ASSUMPTION: All chips in a MeshDevice are identical, so chip 0 is
// representative of every device in the mesh.

void ValidateNodeBounds(const ProgramSpec& spec) {

    MetalEnvImpl& env_impl = MetalEnvAccessor(MetalContext::instance().get_env()).impl();

    // Handle the mock device case (for cheap unit testing)
    const bool is_mock = MetalContext::instance().get_cluster().get_target_device_type() == tt::TargetDevice::Mock;

    // A default DispatchCoreConfig and 1 CQ is sufficient to look up the compute grid size
    // from the YAML descriptor, and both are available in mock mode.
    DispatchCoreConfig dispatch_core_config{};
    uint8_t num_hw_cqs = 1;
    constexpr ChipId chip_id = 0;

    // But, best get the real dispatch_core_config and num_hw_cqs
    // (Makes no difference now, but hardbaking that assumption could be brittle)
    if (!is_mock) {
        auto& dispatch_mgr = MetalContext::instance().get_dispatch_core_manager();
        dispatch_core_config = dispatch_mgr.get_dispatch_core_config();
        num_hw_cqs = dispatch_mgr.get_num_hw_cqs();
    }

    // The compute_grid already accounts for the dispatch row/col
    // No need for dispatch-specific checks (and dispatch-specific error messages confuse users)
    const CoreCoord compute_grid = tt::get_compute_grid_size(env_impl, chip_id, num_hw_cqs, dispatch_core_config);

    auto check_target_nodes = [&](const std::variant<NodeCoord, NodeRange, NodeRangeSet>& target_nodes,
                                  std::string_view entity_type,
                                  std::string_view entity_name) {
        const NodeRangeSet range_set = to_node_range_set(target_nodes);
        for (const NodeRange& range : range_set.ranges()) {
            for (const NodeCoord& node : range) {
                TT_FATAL(
                    node.x < compute_grid.x && node.y < compute_grid.y,
                    "{} '{}' targets node ({},{}), which is out of bounds. "
                    "The compute worker grid on this device is {}x{}.",
                    entity_type,
                    entity_name,
                    node.x,
                    node.y,
                    compute_grid.x,
                    compute_grid.y);
            }
        }
    };

    for (const auto& work_unit : spec.work_units) {
        check_target_nodes(work_unit.target_nodes, "WorkUnitSpec", work_unit.unique_id);
    }
    for (const auto& sem : spec.semaphores) {
        check_target_nodes(sem.target_nodes, "SemaphoreSpec", sem.unique_id);
    }
}

// ValidateProgramSpec: Semantic validation
// ----------------------------------------------------------------------------
//
// This function checks SEMANTIC rules (that don't affect the CollectedSpecData structure):
//   - Architecture requirements
//   - Resource limits
//   - Feature support
//   - Target node constraints (work_unit overlap, node coverage, node validity)
//
// Assumes CollectedSpecData is already built.

void ValidateProgramSpec(const ProgramSpec& spec, const CollectedSpecData& collected) {
    // Sanity check for supported architecture.
    TT_FATAL(is_gen1_arch() || is_gen2_arch(), "Unsupported architecture.");

    //////////////////////////////
    // Node bounds checks
    //////////////////////////////

    ValidateNodeBounds(spec);

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
            !kernel.node_specific_thread_counts.has_value(),
            "KernelSpec '{}' specifies node_specific_thread_counts, but per-node thread counts are not implemented.",
            kernel.unique_id);
    }

    // Validate named RTA/CRTA schema and named CTAs
    for (const auto& kernel : spec.kernels) {
        // All three kinds share the args:: namespace — their names must be mutually unique.
        std::unordered_map<std::string, const char*> seen;  // name -> kind
        auto check_name = [&](const std::string& name, const char* kind) {
            TT_FATAL(
                IsValidCppIdentifier(name),
                "KernelSpec '{}' {} name '{}' is not a valid C++ identifier.",
                kernel.unique_id,
                kind,
                name);
            auto [it, inserted] = seen.try_emplace(name, kind);
            TT_FATAL(
                inserted,
                "KernelSpec '{}' has a naming collision: '{}' is declared as both a {} and a {}.",
                kernel.unique_id,
                name,
                it->second,
                kind);
        };
        for (const auto& name : kernel.runtime_arguments_schema.named_runtime_args) {
            check_name(name, "named RTA");
        }
        for (const auto& name : kernel.runtime_arguments_schema.named_common_runtime_args) {
            check_name(name, "named CRTA");
        }
        for (const auto& [name, value] : kernel.compile_time_arg_bindings) {
            (void)value;
            check_name(name, "named CTA");
        }
    }

    // Validate kernel thread counts
    for (const auto& kernel : spec.kernels) {
        TT_FATAL(kernel.num_threads > 0, "KernelSpec '{}' has no threads!", kernel.unique_id);
        if (kernel.is_compute_kernel()) {
            if (is_gen2_arch()) {
                TT_FATAL(
                    kernel.num_threads <= QUASAR_TENSIX_ENGINES_PER_NODE,
                    "KernelSpec '{}' has too many threads. The architecture supports up to {} for compute kernels.",
                    kernel.unique_id,
                    QUASAR_TENSIX_ENGINES_PER_NODE);
                // On Quasar, we're not allowing 3-thread compute kernels.
                TT_FATAL(
                    kernel.num_threads != 3,
                    "KernelSpec '{}' has 3 threads, which is not supported for compute kernels. Legal values are 1, 2, "
                    "and 4.",
                    kernel.unique_id);
            } else {
                TT_FATAL(
                    kernel.num_threads == 1,
                    "KernelSpec '{}' specifies {} compute threads, but the target architecture does not support "
                    "multi-threaded kernels.",
                    kernel.unique_id,
                    kernel.num_threads);
            }
        }
        if (kernel.is_dm_kernel()) {
            if (is_gen2_arch()) {
                TT_FATAL(
                    kernel.num_threads <= QUASAR_USER_DM_CORES_PER_NODE,
                    "KernelSpec '{}' has too many data movement threads. The maximum is {}.",
                    kernel.unique_id,
                    QUASAR_USER_DM_CORES_PER_NODE);
            } else {
                TT_FATAL(
                    kernel.num_threads == 1,
                    "KernelSpec '{}' specifies {} DM threads, but the target architecture does not support "
                    "multi-threaded kernels. "
                    "num_threads must be 1.",
                    kernel.unique_id,
                    kernel.num_threads);
            }
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

    // On Gen1 (WH/BH), check that no two DM kernels on the same node claim the same processor.
    // (The kernel's effective node set is derived from WorkUnitSpec membership.)
    if (is_gen1_arch()) {
        // Maps (node, processor) -> the kernel that already claimed it
        std::map<std::pair<NodeCoord, DataMovementProcessor>, KernelSpecName> claimed;
        for (const auto& kernel : spec.kernels) {
            if (!kernel.is_dm_kernel()) {
                continue;
            }
            const auto& dm_config = std::get<DataMovementConfiguration>(kernel.config_spec);
            const auto& gen1 = dm_config.gen1_data_movement_config.value();
            const NodeRangeSet& nodes = collected.kernel_node_set.at(kernel.unique_id);
            for (const auto& range : nodes.ranges()) {
                for (const auto& node : range) {
                    auto key = std::make_pair(node, gen1.processor);
                    auto [it, inserted] = claimed.try_emplace(key, kernel.unique_id);
                    TT_FATAL(
                        inserted,
                        "KernelSpec '{}' conflicts with '{}' on node ({}, {}): both claim the same DM processor. ",
                        kernel.unique_id,
                        it->second,
                        node.x,
                        node.y);
                }
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

    // Compute kernels cannot have any semaphore bindings.
    // (This may later change for Quasar.)
    for (const auto& kernel : spec.kernels) {
        TT_FATAL(
            !kernel.is_compute_kernel() || kernel.semaphore_bindings.empty(),
            "KernelSpec '{}' has semaphore bindings. "
            "Semaphore bindings are not currently supported for compute kernels.",
            kernel.unique_id);
    }

    //////////////////////////////////
    // Validate DataflowBufferSpecs
    //////////////////////////////////

    // Validate the total number of DFBs in the ProgramSpec:
    //  - For Gen1, there's a hard limit (hal::get_arch_num_circular_buffers())
    //  - For Gen2, the true DFB limit is configuration-dependent, based on the availability
    //    of tile counters. This won't actually get checked until the Program is enqueued :(
    //  - However, the Gen1 check actually DOES apply to Gen2 as a strict upper limit.
    //    In practice, we'll run out of tile counters long before we hit the HAL CB limit,
    //    but then runtime software sizes some buffers based on this.
    //
    // For Quasar, this is a partial validation only!
    // The true number of available DFBs depends on the tile counters, which are consumed in
    // a DFB configuration-dependent way.
    // Unfortunately, those checks won't trigger until the DFB code runs... not until the
    // Program is actually enqueued :(
    {
        const uint32_t max_dfbs = tt::tt_metal::hal::get_arch_num_circular_buffers();
        if (spec.dataflow_buffers.size() > max_dfbs) {
            if (is_gen1_arch()) {
                TT_THROW(
                    "ProgramSpec '{}' has too many DataflowBufferSpecs ({}). The target "
                    "architecture supports up to {}.",
                    spec.program_id,
                    spec.dataflow_buffers.size(),
                    max_dfbs);
            } else if (is_gen2_arch()) {
                TT_THROW(
                    "ProgramSpec '{}' has too many DataflowBufferSpecs ({}). The permitted "
                    "number of DFBs for the target architecture is configuration-dependent, "
                    "but {} is a hard upper limit.",
                    spec.program_id,
                    spec.dataflow_buffers.size(),
                    max_dfbs);
            } else {
                TT_FATAL(false, "Unknown architecture");
            }
        }
    }

    // Validate local DFB endpoint placement:
    // A local DFB's producer and consumer kernels must be on the same node (sharing SRAM memory).
    // For this to be true, the producer and consumer kernels need IDENTICAL WorkUnitSpec membership.
    auto kernel_work_unit_set = [&](const KernelSpecName& name) {
        std::set<const WorkUnitSpec*> work_units;
        for (const WorkUnitSpec* w : collected.kernel_work_units.at(name)) {
            work_units.insert(w);
        }
        return work_units;
    };
    for (const auto& dfb : spec.dataflow_buffers) {
        const auto& endpoints = collected.dfb_endpoints.at(dfb.unique_id);
        const auto producer_work_units = kernel_work_unit_set(endpoints.producer->unique_id);
        const auto consumer_work_units = kernel_work_unit_set(endpoints.consumer->unique_id);
        TT_FATAL(
            producer_work_units == consumer_work_units,
            "Local DFB '{}' is bound by producer kernel '{}' and consumer kernel '{}', but they "
            "do not share identical WorkUnitSpec membership. Local DFBs require both endpoints to "
            "live on the same set of WorkUnitSpecs; either refactor the placement, or model this as "
            "a RemoteDataflowBufferSpec.",
            dfb.unique_id,
            endpoints.producer->unique_id,
            endpoints.consumer->unique_id);
    }

    // Remote DFBs are not yet supported.
    //
    // TODO: When remote DFB is supported, add a validation checks. Enforce that
    //       each (producer_node, consumer_node) entry in producer_consumer_map has
    //       p_node != c_node.

    TT_FATAL(
        spec.remote_dataflow_buffers.empty(),
        "RemoteDataflowBufferSpec is part of the Metal 2.0 API surface but is not yet supported "
        "by the runtime. (ProgramSpec '{}' has {} remote DFB(s).)",
        spec.program_id,
        spec.remote_dataflow_buffers.size());

    // Borrowed memory is not yet implemented
    for (const auto& dfb : spec.dataflow_buffers) {
        TT_FATAL(
            !dfb.uses_borrowed_memory,
            "DFB '{}' uses borrowed memory, but this feature is not yet implemented",
            dfb.unique_id);
    }

    // DFB aliasing is not supported yet
    for (const auto& dfb : spec.dataflow_buffers) {
        TT_FATAL(
            dfb.alias_with.empty(),
            "DFB '{}' has a non-empty alias_with, but DFB aliasing is not yet implemented",
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

    // Data format must be valid for the architecture
    const tt::ARCH arch = get_arch();
    for (const auto& dfb : spec.dataflow_buffers) {
        if (dfb.data_format_metadata.has_value()) {
            TT_FATAL(
                tt::is_data_format_supported(dfb.data_format_metadata.value(), arch),
                "DFB '{}' has data format '{}' which is not supported on architecture {}",
                dfb.unique_id,
                dfb.data_format_metadata.value(),
                arch);
        }
    }

    //////////////////////////////////
    // Validate SemaphoreSpecs
    //////////////////////////////////

    for (const auto& sem : spec.semaphores) {
        TT_FATAL(
            sem.memory_type == SemaphoreSpec::SemaphoreMemoryType::L1,
            "SemaphoreSpec '{}' uses non-L1 memory type, which is not yet supported",
            sem.unique_id);
        if (is_gen2_arch()) {
            TT_FATAL(
                sem.initial_value == 0,
                "SemaphoreSpec '{}' has initial_value={} but only zero is supported on Quasar",
                sem.unique_id,
                sem.initial_value);
        }
    }

    //////////////////////////////
    // Validate WorkUnitSpecs
    //////////////////////////////

    // WorkUnitSpec is required: a valid ProgramSpec has at least one WorkUnitSpec.
    const auto& work_units = spec.work_units;
    TT_FATAL(!work_units.empty(), "At least one WorkUnitSpec is required");

    // WorkUnitSpecs may not overlap in their target nodes
    for (const auto& work_unit : work_units) {
        for (const auto& other_work_unit : work_units) {
            if (work_unit.unique_id == other_work_unit.unique_id) {
                continue;
            }
            if (nodes_intersect(work_unit.target_nodes, other_work_unit.target_nodes)) {
                TT_FATAL(
                    false,
                    "WorkUnitSpecs '{}' and '{}' overlap in target nodes",
                    work_unit.unique_id,
                    other_work_unit.unique_id);
            }
        }
    }

    // A WorkUnitSpec must have at least one kernel
    for (const auto& work_unit : work_units) {
        TT_FATAL(!work_unit.kernels.empty(), "WorkUnitSpec '{}' has no kernels", work_unit.unique_id);
    }

    // Does the WorkUnit have enough cores to run all of its kernels?
    for (const auto& work_unit : work_units) {
        uint32_t dm_cores_needed = 0;
        uint32_t compute_engines_needed = 0;
        for (const auto& kernel_name : work_unit.kernels) {
            const auto& kernel_spec = collected.kernel_by_name.at(kernel_name);
            if (kernel_spec->is_compute_kernel()) {
                compute_engines_needed += kernel_spec->num_threads;
            }
            if (kernel_spec->is_dm_kernel()) {
                dm_cores_needed += kernel_spec->num_threads;
            }
        }
        if (is_gen2_arch()) {
            TT_FATAL(
                compute_engines_needed <= QUASAR_TENSIX_ENGINES_PER_NODE,
                "WorkUnitSpec '{}' needs {} Tensix engines, but only {} are available",
                work_unit.unique_id,
                compute_engines_needed,
                QUASAR_TENSIX_ENGINES_PER_NODE);
            TT_FATAL(
                dm_cores_needed <= QUASAR_USER_DM_CORES_PER_NODE,
                "WorkUnitSpec '{}' requests {} data movement cores. This exceeds the permitted maximum of {}.",
                work_unit.unique_id,
                dm_cores_needed,
                QUASAR_USER_DM_CORES_PER_NODE);
        }
        if (is_gen1_arch()) {
            TT_FATAL(
                compute_engines_needed <= 1,
                "WorkUnitSpec '{}' has {} compute kernels. The target architecture supports at most one.",
                work_unit.unique_id,
                compute_engines_needed);
            TT_FATAL(
                dm_cores_needed <= 2,
                "WorkUnitSpec '{}' has {} data movement kernels. The target architecture supports at most two.",
                work_unit.unique_id,
                dm_cores_needed);
        }
    }

    // A work_unit can have at most one compute kernel
    for (const auto& work_unit : work_units) {
        uint32_t num_compute_kernels = 0;
        for (const auto& kernel_name : work_unit.kernels) {
            const auto& kernel_spec = collected.kernel_by_name.at(kernel_name);
            if (kernel_spec->is_compute_kernel()) {
                num_compute_kernels++;
            }
        }
        TT_FATAL(num_compute_kernels <= 1, "WorkUnitSpec '{}' has more than one compute kernel", work_unit.unique_id);
    }

    // NOTE:
    // Placement consistency between kernels, DFBs, and WorkUnitSpecs is now structural,
    // not validated:
    //  - Kernels' effective node sets ARE the union of their containing WorkUnitSpecs' target_nodes
    //  - DFBs' allocation node sets are the union of their binding kernels' node sets
}

// ============================================================================
// Step 2: Processor Assignment
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

// Reserve DM processors for a kernel on a WorkUnitSpec.
// Returns {this_kernel_mask, updated_cumulative_mask}
// Throws TT_FATAL on conflict or allocation failure (see simplifying assumption notes)
std::pair<DMProcessorMask, DMProcessorMask> ReserveDMProcessors(
    const KernelSpec* kernel_spec,
    std::optional<DMProcessorMask> existing_mask,
    DMProcessorMask cumulative_mask,
    const WorkUnitSpecName& work_unit_id) {
    // Was this kernel already assigned a mask from a previous WorkUnitSpec?
    if (existing_mask.has_value()) {
        DMProcessorMask existing = existing_mask.value();

        // Check for conflict with what's already allocated on the current WorkUnitSpec
        TT_FATAL(
            !existing.conflicts_with(cumulative_mask),
            "Kernel '{}' requires processors already in use on WorkUnitSpec '{}'. "
            "One of the following must be true: \n"
            " - The ProgramSpec is invalid, and the legality checks were bypassed. \n"
            " - A solution exists, but the greedy algorithm failed to find it. \n"
            " - The runtime's \"common DM cores\" assumption has been violated!",
            kernel_spec->unique_id,
            work_unit_id);

        // Return existing mask and updated cumulative
        return {existing, cumulative_mask | existing};
    }

    // First time seeing this kernel - reserve new processors
    std::optional<DMProcessorMask> reserved = ReserveProcessors(kernel_spec->num_threads, cumulative_mask);
    TT_FATAL(
        reserved.has_value(),
        "Failed to reserve processors for DM kernel '{}' on WorkUnitSpec '{}'. "
        "The \"common DM cores\" assumption has been violated!",
        kernel_spec->unique_id,
        work_unit_id);

    DMProcessorMask mask = reserved.value();
    return {mask, cumulative_mask | mask};
}

// Assign compute processor mask for a kernel.
ComputeEngineMask AssignComputeProcessors(const KernelSpec* kernel_spec, const KernelSpecName& kernel_name) {
    auto reserved = ReserveProcessors(kernel_spec->num_threads, CreateMask<QUASAR_TENSIX_ENGINES_PER_NODE>(0x00));
    TT_FATAL(
        reserved.has_value(),
        "Compute kernel '{}' reservation failed. Condition should be unreachable after validation.",
        kernel_name);
    return reserved.value();
}

// ----------------------------------------------------------------------------
// DM Processor Assignment
// ----------------------------------------------------------------------------
//
// Solves kernel-to-core assignments for DM kernels. Guaranteed to find a valid
// assignment if one exists under the "simplifying assumption" (each DM kernel
// uses the same processor cores on all nodes it targets).
//
// Approach:
//   1. (Optional) Sort kernels by "most constrained first" (more nodes, more threads)
//   2. Use greedy assignment: pick first available cores for each kernel
//   3. If greedy fails, backtrack by trying different kernel orderings
//
// Sorting step is optional. Not yet sure whether it's useful or not.
// It may be that straight greedy is sufficient in a majority of cases
// ----------------------------------------------------------------------------

namespace dm_solver {

// Map from kernel name to its derived effective node set (union of containing
// WorkUnitSpec target_nodes). Used by the solver to read each kernel's placement.
using KernelNodeSetMap = std::unordered_map<KernelSpecName, NodeRangeSet>;

// State for tracking per-node processor usage
class NodeUsageTracker {
public:
    DMProcessorMask& get_used_mask(const NodeCoord& node) {
        if (!node_used_masks_.contains(node)) {
            node_used_masks_[node] = CreateMask<QUASAR_DM_CORES_PER_NODE>(0x03);  // Reserve DM0, DM1
        }
        return node_used_masks_[node];
    }

    // Compute union of used masks across all target nodes
    DMProcessorMask get_combined_used_mask(const NodeRangeSet& target_nodes) {
        DMProcessorMask combined_used = CreateMask<QUASAR_DM_CORES_PER_NODE>(0x00);
        for (const auto& range : target_nodes.ranges()) {
            for (const auto& node : range) {
                combined_used = combined_used | get_used_mask(node);
            }
        }
        return combined_used;
    }

    // Mark cores as used on all target nodes
    void mark_used(const NodeRangeSet& target_nodes, DMProcessorMask mask) {
        for (const auto& range : target_nodes.ranges()) {
            for (const auto& node : range) {
                get_used_mask(node) |= mask;
            }
        }
    }

    // Unmark cores on all target nodes (for backtracking)
    void unmark_used(const NodeRangeSet& target_nodes, DMProcessorMask mask) {
        for (const auto& range : target_nodes.ranges()) {
            for (const auto& node : range) {
                get_used_mask(node) &= ~mask;
            }
        }
    }

    void reset() { node_used_masks_.clear(); }

private:
    std::map<NodeCoord, DMProcessorMask> node_used_masks_;
};

// Constraint score for sorting: higher = more constrained (RISC cores should be assigned earlier)
int ConstraintScore(const KernelSpec* k, const NodeRangeSet& kernel_nodes) {
    int node_count = static_cast<int>(kernel_nodes.num_cores());
    int thread_count = k->num_threads;
    return (node_count * 100) + thread_count;  // nodes dominate, threads break ties
}

void SortByConstraint(std::vector<const KernelSpec*>& kernels, const KernelNodeSetMap& kernel_node_set) {
    std::sort(kernels.begin(), kernels.end(), [&kernel_node_set](const KernelSpec* a, const KernelSpec* b) {
        int score_a = ConstraintScore(a, kernel_node_set.at(a->unique_id));
        int score_b = ConstraintScore(b, kernel_node_set.at(b->unique_id));
        if (score_a != score_b) {
            return score_a > score_b;  // Higher score first
        }
        // In the case of a tie, use unique_id as (deterministic) tiebreaker
        return a->unique_id < b->unique_id;
    });
}

// Try to assign all kernels in the given order using greedy selection
// Returns true if successful, populates result map
bool TryGreedyAssignment(
    const std::vector<const KernelSpec*>& kernel_order,
    const KernelNodeSetMap& kernel_node_set,
    NodeUsageTracker& tracker,
    DMProcessorMaskMap& result) {
    for (const KernelSpec* kernel : kernel_order) {
        const NodeRangeSet& target_nodes = kernel_node_set.at(kernel->unique_id);
        DMProcessorMask combined_used = tracker.get_combined_used_mask(target_nodes);

        auto selected = ReserveProcessors(kernel->num_threads, combined_used);
        if (!selected.has_value()) {
            return false;  // Can't assign this kernel
        }

        result[kernel] = selected.value();
        tracker.mark_used(target_nodes, selected.value());
    }
    return true;
}

// Backtracking solver over kernel orderings
// Note: In the worst case, this is O(N!) in the number of kernels.
//       In practice, I expect this will almost always solve in the first greedy attempt (if sorted).
//       The backtracking is just here for pathological cases.
//       Even then, it shouldn't be horrendous. We won't have a a huge number of kernels in a ProgramSpec.
//       And in the common case (traced), Program creation isn't on the critical path.
//       We can revisit if this ever becomes a problem.
bool SolveWithOrderingBacktrack(
    std::vector<const KernelSpec*> kernels,  // by value - we'll permute it
    const KernelNodeSetMap& kernel_node_set,
    NodeUsageTracker& tracker,
    DMProcessorMaskMap& result) {
    // Try current ordering
    if (TryGreedyAssignment(kernels, kernel_node_set, tracker, result)) {
        return true;
    }

    // Backtrack: try all permutations
    // (std::next_permutation requires sorted input)
    // Sort by unique_id for deterministic permutation order
    auto by_name = [](const KernelSpec* a, const KernelSpec* b) { return a->unique_id < b->unique_id; };
    std::sort(kernels.begin(), kernels.end(), by_name);
    do {
        tracker.reset();
        result.clear();
        if (TryGreedyAssignment(kernels, kernel_node_set, tracker, result)) {
            return true;
        }
    } while (std::next_permutation(kernels.begin(), kernels.end(), by_name));

    return false;
}

}  // namespace dm_solver

// Gen2 (Quasar) processor assignment: runs the backtracking DM solver and returns
// a KernelRiscMaskMap using the Gen2 bit encoding (DM: bits 0-7, compute: bits 8-15).
KernelRiscMaskMap SolveGen2KernelRiscMasks(const ProgramSpec& spec, const CollectedSpecData& collected) {
    DMProcessorMaskMap dm_assignments;
    ComputeEngineMaskMap compute_assignments;

    // Collect DM kernels and compute kernels separately
    std::vector<const KernelSpec*> dm_kernels;
    for (const KernelSpec& kernel : spec.kernels) {
        if (kernel.is_dm_kernel()) {
            dm_kernels.push_back(&kernel);
        } else {
            // Compute kernels: trivial assignment (one compute kernel per node assumption)
            compute_assignments[&kernel] = AssignComputeProcessors(&kernel, kernel.unique_id);
        }
    }

    // Sort by constraint score (most constrained first)
    constexpr bool kSortByConstraint = true;  // Toggle to disable upfront sorting
    if constexpr (kSortByConstraint) {
        dm_solver::SortByConstraint(dm_kernels, collected.kernel_node_set);
    }

    // Solve DM assignments
    dm_solver::NodeUsageTracker tracker;
    bool success =
        dm_solver::SolveWithOrderingBacktrack(dm_kernels, collected.kernel_node_set, tracker, dm_assignments);

    TT_FATAL(
        success,
        "Failed to find valid processor assignments for DM kernels. "
        "Either the ProgramSpec is invalid, or that the \"same DM cores on every node\" "
        "simplifying assumption has been violated.");

    // Convert to KernelRiscMaskMap using Gen2 bit encoding
    KernelRiscMaskMap result;
    for (const auto& [kernel, mask] : dm_assignments) {
        result[kernel] = mask.bits;  // DM processors in bits 0-7
    }
    for (const auto& [kernel, mask] : compute_assignments) {
        result[kernel] = static_cast<uint16_t>(mask.bits) << 8;  // Compute engines in bits 8-15
    }
    return result;
}

// Gen1 (WH/BH) processor assignment: just read the explicit processor from Gen1DataMovementConfig
// and returns a KernelRiscMaskMap using the Gen1 bit encoding (RISCV_0: bit 0, RISCV_1: bit 1, compute: bit 2).
KernelRiscMaskMap BuildGen1KernelRiscMasks(const ProgramSpec& spec) {
    static constexpr uint8_t GEN1_COMPUTE_RISC_BIT = 2;

    KernelRiscMaskMap result;
    for (const KernelSpec& kernel : spec.kernels) {
        if (kernel.is_dm_kernel()) {
            const auto& dm_config = std::get<DataMovementConfiguration>(kernel.config_spec);
            const auto& gen1 = dm_config.gen1_data_movement_config.value();
            result[&kernel] = static_cast<uint16_t>(1u << static_cast<uint8_t>(gen1.processor));
        } else {
            result[&kernel] = static_cast<uint16_t>(1u << GEN1_COMPUTE_RISC_BIT);
        }
    }
    return result;
}

// ============================================================================
// Step 3: Program Building Helpers
// ============================================================================

// Create map of local accessor name -> logical DFB id
tt::tt_metal::DataflowBufferLocalAccessorHandleMap MakeDataflowBufferLocalAccessorHandles(
    const KernelSpec& kernel_spec, const DFBNameToIdMap& dfb_name_to_id) {
    tt::tt_metal::DataflowBufferLocalAccessorHandleMap out;
    out.reserve(kernel_spec.dfb_bindings.size());
    for (const auto& dfb_binding : kernel_spec.dfb_bindings) {
        const uint32_t id = dfb_name_to_id.at(dfb_binding.dfb_spec_name);
        TT_FATAL(
            id <= std::numeric_limits<uint16_t>::max(),
            "Kernel '{}' DFB '{}' logical id {} does not fit uint16_t",
            kernel_spec.unique_id,
            dfb_binding.dfb_spec_name,
            id);
        out.emplace(dfb_binding.local_accessor_name, static_cast<uint16_t>(id));
    }
    return out;
}

// Create map of local accessor name -> logical Semaphore id
tt::tt_metal::SemaphoreLocalAccessorHandleMap MakeSemaphoreLocalAccessorHandles(
    const KernelSpec& kernel_spec, const SemaphoreNameToIdMap& semaphore_name_to_id) {
    tt::tt_metal::SemaphoreLocalAccessorHandleMap out;
    out.reserve(kernel_spec.semaphore_bindings.size());
    for (const auto& semaphore_binding : kernel_spec.semaphore_bindings) {
        const uint32_t id = semaphore_name_to_id.at(semaphore_binding.semaphore_spec_name);
        TT_FATAL(
            id <= std::numeric_limits<uint16_t>::max(),
            "Kernel '{}' semaphore '{}' id {} does not fit uint16_t",
            kernel_spec.unique_id,
            semaphore_binding.semaphore_spec_name,
            id);
        out.emplace(semaphore_binding.accessor_name, static_cast<uint16_t>(id));
    }
    return out;
}

// Create a DataflowBufferConfig from a DataflowBufferSpec and endpoint info.
experimental::dfb::DataflowBufferConfig MakeDataflowBufferConfig(
    const DataflowBufferSpec* dfb_spec,
    const CollectedSpecData::DFBEndpointInfo& dfb_endpoint_info,
    const KernelRiscMaskMap& kernel_to_risc_mask) {
    const KernelSpec* producer = dfb_endpoint_info.producer;
    const KernelSpec* consumer = dfb_endpoint_info.consumer;

    uint16_t producer_risc_mask = kernel_to_risc_mask.at(producer);
    uint16_t consumer_risc_mask = kernel_to_risc_mask.at(consumer);

    // Convert user-facing access pattern enum to hardware interface access pattern enum
    // (TODO: We should merge these enums; it's silly to have separate ones.)
    auto to_hw_access_pattern = [](DFBAccessPattern pattern) -> experimental::dfb::AccessPattern {
        switch (pattern) {
            case DFBAccessPattern::STRIDED: return experimental::dfb::AccessPattern::STRIDED;
            case DFBAccessPattern::ALL: return experimental::dfb::AccessPattern::ALL;
            case DFBAccessPattern::BLOCKED: TT_FATAL(false, "BLOCKED access pattern is not yet supported");
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
    return std::visit(
        [&](const auto& src) -> KernelSource {
            using T = std::decay_t<decltype(src)>;
            if constexpr (std::is_same_v<T, KernelSpec::SourceFilePath>) {
                TT_FATAL(!src.path.empty(), "KernelSpec '{}' has empty source file path", kernel_spec.unique_id);
                return KernelSource(src.path.string(), KernelSource::SourceType::FILE_PATH);
            } else if constexpr (std::is_same_v<T, KernelSpec::SourceCode>) {
                TT_FATAL(!src.code.empty(), "KernelSpec '{}' has empty inline source code", kernel_spec.unique_id);
                return KernelSource(src.code, KernelSource::SourceType::SOURCE_CODE);
            } else {
                static_assert(!sizeof(T*), "Unhandled KernelSpec::source alternative");
            }
        },
        kernel_spec.source);
}

// ----------------------------------------------------------------------------
// MakeGen1DataMovementConfig: Create a DataMovementConfig (WH/BH) from a KernelSpec
// ----------------------------------------------------------------------------

// (Temporary) Shims
// ProgramSpec APIs use vector<pair> for conceptually map-like data structures.
// This is deliberate, done so ProgramSpec stays hashable for TTNN's program caching.
// For now, just convert to the map types that the core runtime expects.
// TODO: Fix this inefficiency eventually.
std::unordered_map<std::string, uint32_t> to_named_compile_args_map(
    const KernelSpec::CompileTimeArgBindings& bindings) {
    return std::unordered_map<std::string, uint32_t>(bindings.begin(), bindings.end());
}
std::map<std::string, std::string> to_defines_map(const KernelSpec::CompilerOptions::Defines& defines) {
    return std::map<std::string, std::string>(defines.begin(), defines.end());
}

DataMovementConfig MakeGen1DataMovementConfig(const KernelSpec& kernel_spec) {
    TT_FATAL(kernel_spec.is_dm_kernel(), "Expected a DM kernel");
    const auto& dm_config = std::get<DataMovementConfiguration>(kernel_spec.config_spec);
    const auto& gen1 = dm_config.gen1_data_movement_config.value();

    return DataMovementConfig{
        .processor = gen1.processor,
        .noc = gen1.noc,
        .noc_mode = gen1.noc_mode,
        .compile_args = {},  // only named_compile_args is used
        .defines = to_defines_map(kernel_spec.compiler_options.defines),
        .named_compile_args = to_named_compile_args_map(kernel_spec.compile_time_arg_bindings),
        .opt_level = kernel_spec.compiler_options.opt_level,
    };
}

// ----------------------------------------------------------------------------
// BuildUnpackToDestModeVector:
// Translate the Metal 2.0 user-facing DFB-name->mode map into the (gross)
// CB-indexed vector that the JIT data-format machinery expects.
//
// This DFB/CB translation layer is confusing. The gory details:
//   - The JIT consumer (get_unpack_dst_formats) will read unpack_to_dest_mode at
//     index cb_id, where cb_id is the slot used by set_dfb_data_fmt_and_tile
//     in buf_dataformat_arr (aka, dfb->id).
//   - The unpack_mode for a DFB "d" needs to be at unpack_modes[d->id]
//   - The vector must be at least max_cbs long, or the consumer gets angry
//     (it iterates buf_formats up to max_cbs).
//   - This is true on WH, BH, and Quasar. (Yes, Quasar too.)
//
// What is the max CBs / DFBs?
//   - WH/BH: Hardcoded as max_cbs. Different number on WH vs. BH.
//   - Quasar has a variable cap, based on tile-counter registers.
//     In actual practice, we'll run out LONG before we get the HAL-reported
//     limit of 64.
// ----------------------------------------------------------------------------

std::vector<UnpackToDestMode> BuildUnpackToDestModeVector(
    const std::vector<ComputeConfiguration::UnpackToDestModeEntry>& user_modes, const DFBNameToIdMap& dfb_name_to_id) {
    const uint32_t max_cbs = tt::tt_metal::hal::get_arch_num_circular_buffers();
    std::vector<UnpackToDestMode> unpack_modes(max_cbs, UnpackToDestMode::Default);
    for (const auto& [dfb_name, mode] : user_modes) {
        uint32_t dfb_id = dfb_name_to_id.at(dfb_name);
        // This TT_FATAL is unreachable, provided that validation wasn't skipped.
        TT_FATAL(
            dfb_id < max_cbs,
            "Internal Error: DFB '{}' has id {} which exceeds the JIT data-format "
            "slot count ({}); compute kernels cannot reference DFBs past this limit",
            dfb_name,
            dfb_id,
            max_cbs);
        unpack_modes[dfb_id] = mode;
    }
    return unpack_modes;
}

// ----------------------------------------------------------------------------
// MakeGen1ComputeConfig: Create a ComputeConfig (WH/BH) from a KernelSpec
// ----------------------------------------------------------------------------

ComputeConfig MakeGen1ComputeConfig(const KernelSpec& kernel_spec, const DFBNameToIdMap& dfb_name_to_id) {
    TT_FATAL(kernel_spec.is_compute_kernel(), "Expected a compute kernel");
    const auto& compute_config = std::get<ComputeConfiguration>(kernel_spec.config_spec);

    std::vector<UnpackToDestMode> unpack_modes =
        BuildUnpackToDestModeVector(compute_config.unpack_to_dest_mode, dfb_name_to_id);

    return ComputeConfig{
        .math_fidelity = compute_config.math_fidelity,
        .fp32_dest_acc_en = compute_config.fp32_dest_acc_en,
        .dst_full_sync_en = compute_config.dst_full_sync_en,
        .unpack_to_dest_mode = unpack_modes,
        .bfp8_pack_precise = compute_config.bfp8_pack_precise,
        .math_approx_mode = compute_config.math_approx_mode,
        .compile_args = {},  // only named_compile_args is used
        .defines = to_defines_map(kernel_spec.compiler_options.defines),
        .named_compile_args = to_named_compile_args_map(kernel_spec.compile_time_arg_bindings),
        .opt_level = kernel_spec.compiler_options.opt_level,
    };
}

// ----------------------------------------------------------------------------
// MakeQuasarDataMovementConfig: Create a QuasarDataMovementConfig from a KernelSpec
// ----------------------------------------------------------------------------

experimental::quasar::QuasarDataMovementConfig MakeQuasarDataMovementConfig(const KernelSpec& kernel_spec) {
    TT_FATAL(kernel_spec.is_dm_kernel(), "Expected a DM kernel");

    return experimental::quasar::QuasarDataMovementConfig{
        .num_threads_per_cluster = kernel_spec.num_threads,
        .compile_args = {},  // only named_compile_args is used
        .defines = to_defines_map(kernel_spec.compiler_options.defines),
        .named_compile_args = to_named_compile_args_map(kernel_spec.compile_time_arg_bindings),
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

    std::vector<UnpackToDestMode> unpack_modes =
        BuildUnpackToDestModeVector(compute_config.unpack_to_dest_mode, dfb_name_to_id);

    return experimental::quasar::QuasarComputeConfig{
        .num_threads_per_cluster = kernel_spec.num_threads,
        .math_fidelity = compute_config.math_fidelity,
        .fp32_dest_acc_en = compute_config.fp32_dest_acc_en,
        .dst_full_sync_en = compute_config.dst_full_sync_en,
        .unpack_to_dest_mode = unpack_modes,
        .bfp8_pack_precise = compute_config.bfp8_pack_precise,
        .math_approx_mode = compute_config.math_approx_mode,
        .compile_args = {},  // Compile args are passed via named_compile_args
        .defines = to_defines_map(kernel_spec.compiler_options.defines),
        .named_compile_args = to_named_compile_args_map(kernel_spec.compile_time_arg_bindings),
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
    for (uint8_t engine = 0; engine < QUASAR_TENSIX_ENGINES_PER_NODE; ++engine) {
        if (mask.is_idx_in_use(engine)) {
            // Add all 4 compute processors for this engine
            for (uint8_t proc = 0; proc < PROCESSORS_PER_ENGINE; ++proc) {
                uint8_t processor_id = (engine * PROCESSORS_PER_ENGINE) + proc;
                processors.insert(static_cast<QuasarComputeProcessor>(processor_id));
            }
        }
    }
    return processors;
}

// ============================================================================
// Public Entry Point
// ============================================================================

Program MakeProgramFromSpec(const ProgramSpec& spec, bool skip_validation) {
    log_debug(tt::LogMetal, "Creating Program from ProgramSpec ({})", spec.program_id);

    // Step 1a: Collect derived data (builds lookup tables, checks structural invariants)
    CollectedSpecData collected = CollectSpecData(spec);

    // Step 1b: Validate semantic rules (can be skipped for trusted inputs)
    if (!skip_validation) {
        ValidateProgramSpec(spec, collected);
    }

    // Step 2: Build kernel risc masks (arch-specific)
    //  - Gen2: backtracking solver assigns DM cores automatically
    //  - Gen1: processor is user-specified in Gen1DataMovementConfig
    KernelRiscMaskMap kernel_to_risc_mask =
        is_gen2_arch() ? SolveGen2KernelRiscMasks(spec, collected) : BuildGen1KernelRiscMasks(spec);

    // Step 3: Build the Program
    auto program_impl = std::make_shared<detail::ProgramImpl>();

    // Create DataflowBuffers and build name -> ID map.
    // NOTE: Iterate over spec.dataflow_buffers (not collected.dfb_endpoints) to ensure
    //       deterministic DFB ID assignment based on user-specified order.
    DFBNameToIdMap dfb_name_to_id;
    for (const auto& dfb_spec : spec.dataflow_buffers) {
        const DFBSpecName& dfb_name = dfb_spec.unique_id;
        const auto& dfb_endpoint_info = collected.dfb_endpoints.at(dfb_name);
        const experimental::dfb::DataflowBufferConfig config =
            MakeDataflowBufferConfig(&dfb_spec, dfb_endpoint_info, kernel_to_risc_mask);

        // Add the DFB to the ProgramImpl, and register the name -> handle mapping.
        // Allocation nodes are derived from binding kernels' WorkUnitSpec membership.
        uint32_t dfb_id = program_impl->add_dataflow_buffer(collected.dfb_node_set.at(dfb_name), config);
        program_impl->register_dfb_spec_name(dfb_name, dfb_id);
        dfb_name_to_id[dfb_name] = dfb_id;
    }

    // Create Semaphores and build name -> ID map.
    // NOTE: Iterate over spec.semaphores to preserve user-provided deterministic ordering.
    SemaphoreNameToIdMap semaphore_name_to_id;
    for (const auto& semaphore_spec : spec.semaphores) {
        const SemaphoreSpecName& semaphore_name = semaphore_spec.unique_id;
        uint32_t sem_id = program_impl->create_semaphore(
            to_node_range_set(semaphore_spec.target_nodes), semaphore_spec.initial_value, CoreType::WORKER);
        program_impl->register_semaphore_spec_name(semaphore_name, sem_id);
        semaphore_name_to_id[semaphore_name] = sem_id;
    }

    // Create Kernels (arch-specific)
    for (const KernelSpec& kernel_spec : spec.kernels) {
        KernelSource kernel_src = MakeKernelSource(kernel_spec);
        const NodeRangeSet& node_ranges = collected.kernel_node_set.at(kernel_spec.unique_id);

        // Make the local accessor name -> DFB ID map for this kernel
        const tt::tt_metal::DataflowBufferLocalAccessorHandleMap dfb_handles =
            MakeDataflowBufferLocalAccessorHandles(kernel_spec, dfb_name_to_id);
        const tt::tt_metal::SemaphoreLocalAccessorHandleMap semaphore_handles =
            MakeSemaphoreLocalAccessorHandles(kernel_spec, semaphore_name_to_id);

        // Named-args schema fields passed to the Kernel ctor. The names are used at JIT time
        // to emit kernel_args_generated.h and factor into the kernel cache key.
        const auto& named_rtas = kernel_spec.runtime_arguments_schema.named_runtime_args;
        const auto& named_crtas = kernel_spec.runtime_arguments_schema.named_common_runtime_args;

        // Create the kernel object
        std::shared_ptr<Kernel> kernel;

        // Kernel creation APIs accept a "is_metal2_kernel" bool, which fences Metal 2.0 JIT machinery
        constexpr bool is_metal2_kernel = true;

        if (is_gen2_arch()) {
            uint16_t risc_mask = kernel_to_risc_mask.at(&kernel_spec);
            if (kernel_spec.is_dm_kernel()) {
                auto config = MakeQuasarDataMovementConfig(kernel_spec);
                auto processors = GetDMProcessorSet(DMProcessorMask{(uint8_t)(risc_mask & 0xFF)});
                kernel = std::make_shared<experimental::quasar::QuasarDataMovementKernel>(
                    kernel_src,
                    node_ranges,
                    config,
                    processors,
                    is_metal2_kernel,
                    dfb_handles,
                    semaphore_handles,
                    named_rtas,
                    named_crtas);
            } else {
                auto config = MakeQuasarComputeConfig(kernel_spec, dfb_name_to_id);
                auto processors = GetComputeProcessorSet(ComputeEngineMask{(uint8_t)(risc_mask >> 8)});
                kernel = std::make_shared<experimental::quasar::QuasarComputeKernel>(
                    kernel_src,
                    node_ranges,
                    config,
                    processors,
                    is_metal2_kernel,
                    dfb_handles,
                    semaphore_handles,
                    named_rtas,
                    named_crtas);
            }
        } else {  // gen1
            if (kernel_spec.is_dm_kernel()) {
                auto config = MakeGen1DataMovementConfig(kernel_spec);
                kernel = std::make_shared<DataMovementKernel>(
                    kernel_src,
                    node_ranges,
                    config,
                    is_metal2_kernel,
                    dfb_handles,
                    semaphore_handles,
                    named_rtas,
                    named_crtas);
            } else {
                auto config = MakeGen1ComputeConfig(kernel_spec, dfb_name_to_id);
                kernel = std::make_shared<ComputeKernel>(
                    kernel_src,
                    node_ranges,
                    config,
                    is_metal2_kernel,
                    dfb_handles,
                    semaphore_handles,
                    named_rtas,
                    named_crtas);
            }
        }

        // Add the kernel to the ProgramImpl and register the name -> handle mapping
        KernelHandle handle = program_impl->add_kernel(kernel, HalProgrammableCoreType::TENSIX);
        program_impl->register_kernel_spec_name(kernel_spec.unique_id, handle);

        // Register the RTA+CRTA schema (named lists + vararg counts) with the ProgramImpl.
        // Used by ValidateProgramRunParams and SetProgramRunParameters to validate and serialize
        // the user-provided values at dispatch time.
        //
        // User-facing vararg RTA specification (see kernel_spec.hpp):
        //   - num_runtime_varargs (scalar): default count applied to every node the kernel
        //     runs on.
        //   - num_runtime_varargs_per_node (optional): sparse per-node overrides on top of
        //     the scalar default. Unlisted nodes fall back to the scalar.
        // We apply the scalar first across target_nodes, then overlay each override entry.
        // An explicit override of 0 erases the scalar-default entry so run-params treats
        // that node as having no varargs (rather than requiring an "empty" value list).
        // Overlapping override entries (two entries covering the same node) are an error.
        const auto& user_schema = kernel_spec.runtime_arguments_schema;
        detail::ProgramImpl::KernelRTASchema runtime_schema;
        runtime_schema.named_runtime_args = user_schema.named_runtime_args;
        runtime_schema.named_common_runtime_args = user_schema.named_common_runtime_args;
        if (user_schema.num_runtime_varargs > 0) {
            for (const NodeRange& range : node_ranges.ranges()) {
                for (const NodeCoord& node : range) {
                    runtime_schema.num_runtime_varargs_per_node[node] = user_schema.num_runtime_varargs;
                }
            }
        }
        if (user_schema.num_runtime_varargs_per_node.has_value()) {
            std::unordered_set<NodeCoord> seen_overrides;
            for (const auto& [nodes_spec, num_varargs] : *user_schema.num_runtime_varargs_per_node) {
                const NodeRangeSet expanded = to_node_range_set(nodes_spec);
                for (const NodeRange& range : expanded.ranges()) {
                    for (const NodeCoord& node : range) {
                        const bool inserted = seen_overrides.insert(node).second;
                        TT_FATAL(
                            inserted,
                            "KernelSpec '{}' num_runtime_varargs_per_node has overlapping entries "
                            "for node {}",
                            kernel_spec.unique_id,
                            node.str());
                        if (num_varargs > 0) {
                            runtime_schema.num_runtime_varargs_per_node[node] = num_varargs;
                        } else {
                            // Explicit zero override: drop any scalar-default entry so
                            // run-params treats this node as missing (→ 0 expected).
                            runtime_schema.num_runtime_varargs_per_node.erase(node);
                        }
                    }
                }
            }
        }
        runtime_schema.num_common_runtime_varargs = user_schema.num_common_runtime_varargs;
        program_impl->register_kernel_rta_schema(kernel_spec.unique_id, runtime_schema);
    }

    return Program(std::move(program_impl));
}

}  // namespace tt::tt_metal::experimental::metal2_host_api
