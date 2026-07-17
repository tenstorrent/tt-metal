// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <bit>
#include <functional>
#include <limits>
#include <numeric>
#include <set>
#include <string_view>
#include <unordered_set>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>  // fmt::formatter<tt::DataFormat> for TT_FATAL messages
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <hostdevcommon/tensor_accessor/arg_config.hpp>
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/context/metal_env_accessor.hpp"
#include "impl/dispatch/dispatch_core_manager.hpp"
#include <core_descriptor.hpp>
#include <llrt/tt_cluster.hpp>
#include <variant>

namespace tt::tt_metal::experimental {

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
    // dfb_by_name covers BOTH local and cross-node DFBs.
    // For cross-node DFBs, the pointee is the inner dfb_spec.
    // To check if a DFB is cross-node, check the cross_node_dfb_by_name map.
    std::unordered_map<KernelSpecName, const KernelSpec*> kernel_by_name;
    std::unordered_map<DFBSpecName, const DataflowBufferSpec*> dfb_by_name;
    std::unordered_map<DFBSpecName, const CrossNodeDataflowBufferSpec*> cross_node_dfb_by_name;
    std::unordered_map<SemaphoreSpecName, const SemaphoreSpec*> semaphore_by_name;
    std::unordered_map<ScratchpadSpecName, const ScratchpadSpec*> scratchpad_by_name;
    std::unordered_map<TensorParamName, const TensorParameter*> tensor_parameter_by_name;

    // Tensor parameter usage (derived from kernel tensor bindings).
    // Tracks which kernels bind a given tensor parameter.
    std::unordered_map<TensorParamName, std::vector<const KernelSpec*>> tensor_parameter_users;

    // Scratchpad binders (derived from kernel scratchpad bindings).
    // Tracks which kernels bind a given ScratchpadSpec. More than one may, provided their node sets
    // are disjoint; the per-node placement census in ValidateProgramSpec enforces that (it needs the
    // kernel node sets, derived below). A kernel binds a given scratchpad at most once (enforced
    // during collection), so each kernel appears at most once in a spec's binder list.
    std::unordered_map<ScratchpadSpecName, std::vector<const KernelSpec*>> scratchpad_binders;

    // DFB endpoint info (derived from kernel bindings).
    // Populated for both local and cross-node DFBs.
    //
    // Multiple PRODUCER KernelSpecs (and multiple CONSUMER KernelSpecs) may bind the same DFB,
    // provided they have non-overlapping node coverage and matching binding-site parameters
    // (access_pattern, num_threads). This permits the canonical Metal 2.0 expression of the
    // legacy "two KernelDescriptors per work split, sharing CBs" pattern. The physical
    // invariant is local: at each node, exactly one producer kernel instance and one
    // consumer kernel instance.
    struct DFBEndpointInfo {
        struct EndpointRecord {
            const KernelSpec* kernel = nullptr;
            const DFBBinding* binding = nullptr;
        };
        std::vector<EndpointRecord> producers;
        std::vector<EndpointRecord> consumers;
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

// Kernel -> ProcessorMask map (Gen2/Quasar only).
// DM masks flow through KernelCouplingGroup (equivalence class) rather than per-KernelSpec, so the DM
// counterpart of this map is defined in the dm_solver namespace and keyed by KernelCouplingGroup*.
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

NodeRangeSet to_node_range_set(const Nodes& nodes) {
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

bool nodes_intersect(const Nodes& a, const Nodes& b) {
    NodeRangeSet a_set = to_node_range_set(a);
    NodeRangeSet b_set = to_node_range_set(b);
    return a_set.intersects(b_set);
}

// Helper: return a DFB's alias-with list.
const std::vector<DFBSpecName>& dfb_alias_with(const DataflowBufferSpec& dfb) {
    return dfb.advanced_options.alias_with;
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

    // Collect CrossNodeDataflowBufferSpecs (cross-node DFBs).
    // Cross-node DFBs share the DFB name space with local DFBs, since kernel bindings
    // refer to either kind by the same DFBSpecName.
    for (const auto& cross_node_dfb : spec.cross_node_dataflow_buffers) {
        const DFBSpecName& name = cross_node_dfb.dfb_spec.unique_id;
        auto [it1, inserted1] = collected.dfb_by_name.try_emplace(name, &cross_node_dfb.dfb_spec);
        TT_FATAL(inserted1, "Duplicate DataflowBufferSpec name '{}' (across local and cross-node DFBs)", name);
        auto [it2, inserted2] = collected.cross_node_dfb_by_name.try_emplace(name, &cross_node_dfb);
        TT_FATAL(inserted2, "Duplicate CrossNodeDataflowBufferSpec name '{}'", name);
    }

    // Build DFB endpoint info from kernel bindings
    for (const auto& kernel : spec.kernels) {
        // Track per-accessor-name signatures within this kernel. Reusing a single
        // accessor_name across two DFBBindings is permitted as a "self-loop pair":
        // both bindings target the same DFB with opposite endpoint types (one PRODUCER,
        // one CONSUMER). This lets a kernel that both produces and consumes the same DFB
        // use a single device-side accessor name instead of two aliasing wrappers.
        struct AccessorBindingInfo {
            DFBSpecName dfb_spec_name;
            bool has_producer = false;
            bool has_consumer = false;
        };
        std::unordered_map<std::string, AccessorBindingInfo> accessor_bindings;
        // Track, per DFB, which endpoint roles this kernel has already bound. Within a kernel a DFB
        // may be bound at most once per role; the only multi-binding form is the self-loop pair (one
        // PRODUCER + one CONSUMER, whose accessor names may differ). A second binding of the same
        // role under a different accessor name is the forbidden "one buffer, two names" aliasing
        // (see the check below). Scoped to the kernel so it resets per iteration — a DFB legitimately
        // carries different accessor names on different kernels (producer 'out', consumer 'in'), so
        // this must not be global.
        struct DFBBoundRoles {
            bool has_producer = false;
            bool has_consumer = false;
        };
        std::unordered_map<DFBSpecName, DFBBoundRoles> dfb_bound_roles;
        for (const auto& dfb_binding : kernel.dfb_bindings) {
            auto [it, inserted] = accessor_bindings.try_emplace(
                dfb_binding.accessor_name, AccessorBindingInfo{dfb_binding.dfb_spec_name});
            AccessorBindingInfo& info = it->second;
            if (inserted) {
                TT_FATAL(
                    IsValidCppIdentifier(dfb_binding.accessor_name),
                    "Kernel '{}' DFB accessor_name '{}' must be a valid C++ identifier",
                    kernel.unique_id,
                    dfb_binding.accessor_name);
            } else {
                TT_FATAL(
                    info.dfb_spec_name == dfb_binding.dfb_spec_name,
                    "Kernel '{}' uses accessor_name '{}' for two different DFBs ('{}' and '{}'). "
                    "Reusing a name is only permitted when both bindings target the same DFB (self-loop pair).",
                    kernel.unique_id,
                    dfb_binding.accessor_name,
                    info.dfb_spec_name,
                    dfb_binding.dfb_spec_name);
            }
            const bool is_producer = (dfb_binding.endpoint_type == DFBEndpointType::PRODUCER);
            bool& seen_this_type = is_producer ? info.has_producer : info.has_consumer;
            TT_FATAL(
                !seen_this_type,
                "Kernel '{}' has duplicate {} binding for accessor_name '{}'",
                kernel.unique_id,
                is_producer ? "PRODUCER" : "CONSUMER",
                dfb_binding.accessor_name);
            seen_this_type = true;

            // Forbid binding the same DFB twice in the same role within this kernel (e.g. two CONSUMER
            // bindings under different accessor names). The legitimate multi-binding form is the
            // self-loop pair — one PRODUCER + one CONSUMER — which this allows regardless of whether
            // the two bindings share an accessor name. The same-role same-name case is already caught
            // above (duplicate {PRODUCER,CONSUMER} binding for accessor_name); this closes the
            // different-name gap. "One buffer, two names" in kernel code must be a handle alias
            // (constexpr auto x = dfb::y) over a single binding, not a second binding — two accessors /
            // DataflowBuffer objects for one FIFO break the object<->DFB identity that device-side
            // debug tooling relies on.
            DFBBoundRoles& bound_roles = dfb_bound_roles[dfb_binding.dfb_spec_name];
            bool& role_already_bound = is_producer ? bound_roles.has_producer : bound_roles.has_consumer;
            TT_FATAL(
                !role_already_bound,
                "Kernel '{}' has two {} bindings to DFB '{}' under different accessor names. Within a "
                "kernel a DFB may be bound at most once per role (the only multi-binding form is the "
                "self-loop pair: one PRODUCER + one CONSUMER). To refer to one buffer by multiple names "
                "in kernel code, alias the handle (constexpr auto x = dfb::y) instead of adding a second binding.",
                kernel.unique_id,
                is_producer ? "PRODUCER" : "CONSUMER",
                dfb_binding.dfb_spec_name);
            role_already_bound = true;

            // Referential integrity: the DFB must exist
            TT_FATAL(
                collected.dfb_by_name.contains(dfb_binding.dfb_spec_name),
                "Kernel '{}' references unknown DFB '{}'",
                kernel.unique_id,
                dfb_binding.dfb_spec_name);

            CollectedSpecData::DFBEndpointInfo& endpoint_info = collected.dfb_endpoints[dfb_binding.dfb_spec_name];

            if (dfb_binding.endpoint_type == DFBEndpointType::PRODUCER) {
                endpoint_info.producers.push_back({&kernel, &dfb_binding});
            } else if (dfb_binding.endpoint_type == DFBEndpointType::CONSUMER) {
                endpoint_info.consumers.push_back({&kernel, &dfb_binding});
            }
        }
    }

    // Completeness: every DFB must have at least one producer and one consumer.
    // (Cross-role coverage matching and within-role binding-site uniformity are checked
    // later, after kernel node coverage is computed.)
    for (const auto& [dfb_name, endpoint_info] : collected.dfb_endpoints) {
        TT_FATAL(!endpoint_info.producers.empty(), "DFB '{}' has no producer", dfb_name);
        TT_FATAL(!endpoint_info.consumers.empty(), "DFB '{}' has no consumer", dfb_name);
    }

    // Referential integrity: every declared DFB (local or cross-node) must be bound by some kernel
    for (const auto& dfb : spec.dataflow_buffers) {
        TT_FATAL(
            collected.dfb_endpoints.contains(dfb.unique_id),
            "DFB '{}' is defined but not bound by any kernel",
            dfb.unique_id);
    }
    for (const auto& cross_node_dfb : spec.cross_node_dataflow_buffers) {
        const DFBSpecName& name = cross_node_dfb.dfb_spec.unique_id;
        TT_FATAL(
            collected.dfb_endpoints.contains(name),
            "CrossNodeDataflowBufferSpec '{}' is defined but not bound by any kernel",
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

    // Collect ScratchpadSpecs
    for (const auto& scratchpad : spec.scratchpads) {
        auto [it, inserted] = collected.scratchpad_by_name.try_emplace(scratchpad.unique_id, &scratchpad);
        TT_FATAL(inserted, "Duplicate ScratchpadSpec name '{}'", scratchpad.unique_id);
        TT_FATAL(
            scratchpad.size_per_node != 0,
            "ScratchpadSpec '{}' has size_per_node == 0; a scratchpad must reserve a non-zero number of bytes "
            "(did you forget to set size_per_node?).",
            scratchpad.unique_id);
    }

    // Collect scratchpad bindings (structural checks here; the node-set placement check is in
    // ValidateProgramSpec, which has the derived kernel node sets).
    // A scratchpad is private, node-local L1. More than one KernelSpec may bind the same
    // ScratchpadSpec, but only on disjoint node sets — the same node-local-resource discipline as a
    // local DFB. Same-node co-binding (true sharing) is gated behind a future AdvancedOption and is
    // rejected by the per-node census in ValidateProgramSpec.
    for (const auto& kernel : spec.kernels) {
        std::unordered_set<std::string> accessor_names;
        std::unordered_set<ScratchpadSpecName> bound_specs;
        for (const auto& binding : kernel.scratchpad_bindings) {
            auto [it, inserted] = accessor_names.insert(binding.accessor_name);
            TT_FATAL(
                inserted,
                "Kernel '{}' has duplicate scratchpad accessor_name '{}'",
                kernel.unique_id,
                binding.accessor_name);
            TT_FATAL(
                IsValidCppIdentifier(binding.accessor_name),
                "Kernel '{}' scratchpad accessor_name '{}' must be a valid C++ identifier",
                kernel.unique_id,
                binding.accessor_name);
            TT_FATAL(
                collected.scratchpad_by_name.contains(binding.scratchpad_spec_name),
                "Kernel '{}' references unknown scratchpad '{}'",
                kernel.unique_id,
                binding.scratchpad_spec_name);
            // A kernel may bind a given scratchpad at most once. Two bindings would request two
            // separate per-node allocations of the same spec under one kernel — muddy semantics, and
            // a node-level violation of "one binding instance per node". This is structural (no node
            // info needed), so it is caught here rather than in the placement census.
            auto [sit, sinserted] = bound_specs.insert(binding.scratchpad_spec_name);
            TT_FATAL(
                sinserted,
                "Kernel '{}' binds scratchpad '{}' more than once (latest under accessor_name '{}'). A "
                "kernel may bind a given scratchpad at most once.",
                kernel.unique_id,
                binding.scratchpad_spec_name,
                binding.accessor_name);
            collected.scratchpad_binders[binding.scratchpad_spec_name].push_back(&kernel);
        }
    }
    // Every declared scratchpad must be bound by some kernel: an unbound scratchpad would reserve L1
    // that no kernel can reach.
    for (const auto& scratchpad : spec.scratchpads) {
        TT_FATAL(
            collected.scratchpad_binders.contains(scratchpad.unique_id),
            "ScratchpadSpec '{}' is declared but not bound by any kernel.",
            scratchpad.unique_id);
    }

    // Collect TensorParameters
    for (const auto& tensor_parameter : spec.tensor_parameters) {
        auto [it, inserted] =
            collected.tensor_parameter_by_name.try_emplace(tensor_parameter.unique_id, &tensor_parameter);
        TT_FATAL(inserted, "Duplicate TensorParameter name '{}'", tensor_parameter.unique_id);
    }

    // Validate kernel tensor bindings
    for (const auto& kernel : spec.kernels) {
        // A tensor binding is legal on both DM and compute kernels:
        //   - a DM kernel can use the binding token to construct a TensorAccessor or LocalTensorAccessor
        //   - a compute kernel can only use LocalTensorAccessor (NOC-free, local-L1 only)

        std::unordered_set<std::string> accessor_names;
        for (const auto& binding : kernel.tensor_bindings) {
            auto [it, inserted] = accessor_names.insert(binding.accessor_name);
            TT_FATAL(
                inserted,
                "Kernel '{}' has duplicate tensor accessor_name '{}'",
                kernel.unique_id,
                binding.accessor_name);
            TT_FATAL(
                IsValidCppIdentifier(binding.accessor_name),
                "Kernel '{}' tensor accessor_name '{}' must be a valid C++ identifier",
                kernel.unique_id,
                binding.accessor_name);
            TT_FATAL(
                collected.tensor_parameter_by_name.contains(binding.tensor_parameter_name),
                "Kernel '{}' references unknown TensorParameter '{}'",
                kernel.unique_id,
                binding.tensor_parameter_name);

            collected.tensor_parameter_users[binding.tensor_parameter_name].push_back(&kernel);
        }
    }

    // A borrowed-memory DFB uses its backing TensorParameter via DataflowBufferSpec::borrowed_from
    // (the DFB resolves its L1 address from that parameter's TensorArgument at runtime) even when no
    // kernel binds the parameter directly. Count that as a use so the completeness check below doesn't
    // reject a borrowed-only parameter. Existence of the referent is validated separately in the
    // borrowed-DFB checks. Only local DFBs are walked here: borrowed memory is a local-L1 feature,
    // so spec.dataflow_buffers is the relevant set (cross-node DFBs are runtime-unsupported).
    for (const auto& dfb : spec.dataflow_buffers) {
        if (dfb.borrowed_from.has_value()) {
            collected.tensor_parameter_users[*dfb.borrowed_from];  // register as used (no kernel user)
        }
    }

    // Referential integrity: every declared TensorParameter must be referenced by some kernel
    // binding or a DFB borrowed_from. (Same usage requirement as DFBs; an unused tensor parameter
    // is a user error.)
    for (const auto& tensor_parameter : spec.tensor_parameters) {
        TT_FATAL(
            collected.tensor_parameter_users.contains(tensor_parameter.unique_id),
            "TensorParameter '{}' is defined but not bound by any kernel",
            tensor_parameter.unique_id);
    }

    // Build WorkUnitSpec membership for each kernel, validating references along the way.
    // (WorkUnitSpec.name is debug-only; no uniqueness invariant.)
    // A kernel may belong to multiple WorkUnitSpecs; its effective target node set is the union.
    for (const auto& work_unit : spec.work_units) {
        for (const auto& kernel_name : work_unit.kernels) {
            TT_FATAL(
                collected.kernel_by_name.contains(kernel_name),
                "WorkUnitSpec '{}' references unknown kernel '{}'",
                work_unit.name,
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
    // (Collected, but unvalidated. Semantic integrity checks for DFB take place in ValidateProgramSpec.
    //  Once those pass, producer and consumer coverages are guaranteed equal; here we union both
    //  sides for safety before that guarantee holds.)
    for (const auto& dfb : spec.dataflow_buffers) {
        const auto& endpoints = collected.dfb_endpoints.at(dfb.unique_id);
        NodeRangeSet node_set;
        for (const auto& rec : endpoints.producers) {
            node_set = node_set.merge(collected.kernel_node_set.at(rec.kernel->unique_id));
        }
        for (const auto& rec : endpoints.consumers) {
            node_set = node_set.merge(collected.kernel_node_set.at(rec.kernel->unique_id));
        }
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

    auto check_target_nodes = [&](const Nodes& target_nodes,
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
        check_target_nodes(work_unit.target_nodes, "WorkUnitSpec", work_unit.name);
    }
    for (const auto& sem : spec.semaphores) {
        check_target_nodes(sem.target_nodes, "SemaphoreSpec", sem.unique_id.get());
    }
}

// Whether a Gen2 DM kernel opts out of implicit sync for a particular DFB.
// Two routes lead to the same opt-out:
//   - disable_dfb_implicit_sync_for_all: the per-kernel hammer, covering every DFB the kernel binds.
//   - disable_dfb_implicit_sync_for: an explicit per-DFB list.
// Precondition: the caller has already established this is a DM kernel with a gen2_config.
bool DmKernelDisablesImplicitSync(const DataMovementGen2Config& gen2_config, const DFBSpecName& dfb_name) {
    if (gen2_config.disable_dfb_implicit_sync_for_all) {
        return true;
    }
    const auto& vec = gen2_config.disable_dfb_implicit_sync_for;
    return std::find(vec.begin(), vec.end(), dfb_name) != vec.end();
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
        for (const auto& name : kernel.runtime_arg_schema.runtime_arg_names) {
            check_name(name, "named RTA");
        }
        for (const auto& name : kernel.runtime_arg_schema.common_runtime_arg_names) {
            check_name(name, "named CRTA");
        }
        for (const auto& [name, value] : kernel.compile_time_args) {
            (void)value;
            check_name(name, "named CTA");
        }
    }

    // Validate enqueue-loop-invariant named-arg declarations: each invariant name must reference a
    // declared named RTA / CRTA. Invariance requires naming the argument, so positional varargs
    // cannot be marked invariant.
    for (const auto& kernel : spec.kernels) {
        const std::unordered_set<std::string> declared_rtas(
            kernel.runtime_arg_schema.runtime_arg_names.begin(), kernel.runtime_arg_schema.runtime_arg_names.end());
        for (const auto& name : kernel.advanced_options.enqueue_invariant_runtime_args) {
            TT_FATAL(
                declared_rtas.contains(name),
                "KernelSpec '{}' marks runtime arg '{}' enqueue-loop invariant, but it is not a declared named runtime "
                "arg (runtime_arg_schema.runtime_arg_names). Only named runtime args can be marked invariant; "
                "positional varargs cannot.",
                kernel.unique_id,
                name);
        }
        const std::unordered_set<std::string> declared_crtas(
            kernel.runtime_arg_schema.common_runtime_arg_names.begin(),
            kernel.runtime_arg_schema.common_runtime_arg_names.end());
        for (const auto& name : kernel.advanced_options.enqueue_invariant_common_runtime_args) {
            TT_FATAL(
                declared_crtas.contains(name),
                "KernelSpec '{}' marks common runtime arg '{}' enqueue-loop invariant, but it is not a declared named "
                "common runtime arg (runtime_arg_schema.common_runtime_arg_names). Only named common runtime args can "
                "be marked invariant; positional varargs cannot.",
                kernel.unique_id,
                name);
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
        if (kernel.is_data_movement_kernel()) {
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

    // Validate hardware configs: a kernel's config generation must match the target platform. There
    // is no implicit cross-generation substitution — supplying the wrong alternative results in direct error.
    for (const auto& kernel : spec.kernels) {
        if (kernel.is_data_movement_kernel()) {
            const auto& data_movement_config = std::get<DataMovementHardwareConfig>(kernel.hw_config);

            if (is_gen1_arch()) {
                TT_FATAL(
                    std::holds_alternative<DataMovementGen1Config>(data_movement_config),
                    "KernelSpec '{}' targets Gen1 (WH/BH) but its DataMovementHardwareConfig holds a "
                    "DataMovementGen2Config. Supply a Gen1 config (e.g. "
                    "CreateReader1xxDataMovementConfig()/CreateWriter1xxDataMovementConfig()).",
                    kernel.unique_id);

                // Gen1 has exactly two DM processors: RISCV_0 (BRISC) and RISCV_1 (NCRISC).
                // RISCV_2..RISCV_7 exist only on Gen2/Quasar. Reject them here, mirroring the legacy
                // CreateDataMovementKernel "DM0 or DM1 only" guard. Resolving is safe now: the check
                // above guarantees a role hint or an explicit Gen1 config is present.
                const DataMovementProcessor processor =
                    std::get<DataMovementGen1Config>(data_movement_config).processor;
                TT_FATAL(
                    processor == DataMovementProcessor::RISCV_0 || processor == DataMovementProcessor::RISCV_1,
                    "KernelSpec '{}' targets Gen1 (WH/BH) but requests DM processor RISCV_{}. Gen1 has only "
                    "RISCV_0 and RISCV_1; RISCV_2..RISCV_7 exist only on Gen2/Quasar.",
                    kernel.unique_id,
                    static_cast<int>(processor));
            } else if (is_gen2_arch()) {
                TT_FATAL(
                    std::holds_alternative<DataMovementGen2Config>(data_movement_config),
                    "KernelSpec '{}' targets Gen2 (Quasar) but its DataMovementHardwareConfig holds a "
                    "DataMovementGen1Config. Supply a Gen2 config (DataMovementGen2Config{{}}).",
                    kernel.unique_id);
            }
        }

        if (kernel.is_compute_kernel()) {
            const auto& compute_config = std::get<ComputeHardwareConfig>(kernel.hw_config);

            if (is_gen1_arch()) {
                TT_FATAL(
                    std::holds_alternative<ComputeGen1Config>(compute_config),
                    "KernelSpec '{}' targets Gen1 (WH/BH) but its ComputeHardwareConfig holds a "
                    "ComputeGen2Config. Supply a Gen1 config (ComputeGen1Config).",
                    kernel.unique_id);
            } else if (is_gen2_arch()) {
                TT_FATAL(
                    std::holds_alternative<ComputeGen2Config>(compute_config),
                    "KernelSpec '{}' targets Gen2 (Quasar) but its ComputeHardwareConfig holds a "
                    "ComputeGen1Config. Supply a Gen2 config (ComputeGen2Config).",
                    kernel.unique_id);
            }
        }
    }

    // On Gen1 (WH/BH), the DM kernels sharing a node must be mutually coherent:
    //   1. Distinct DM processors (RISCV_0 vs RISCV_1) — no two kernels may pin the same RISC.
    //   2. Agreeing NOC mode. noc_mode configures shared per-core NOC hardware (command-buffer
    //      partitioning + completion-counter location) and is compiled into each kernel binary as
    //      the NOC_MODE define (see kernel.cpp), so two kernels on a node with different modes are
    //      incoherent. Mirrors the KernelGroup-construction guard ("KernelGroup must have the same
    //      noc mode for all kernels"), surfaced here at spec-validation time with a clearer message.
    //   3. In DM_DEDICATED_NOC mode, distinct NOCs. Each DM kernel's NoC traffic is statically
    //      compiled to NOC_INDEX == config.noc (see kernel.cpp), so two dedicated-NOC kernels
    //      pinned to the same NOC deadlock the device. This enforces the NOC-distinctness invariant
    //      that KernelGroup finalize silently relies on (it writes brisc_noc_id = arg.noc for
    //      RISCV_0 vs 1 - arg.noc for RISCV_1, which agree only when the two NOCs differ -- "safe
    //      due to prior correctness validation"). The legacy CheckDataMovementConfig intended this
    //      check but did not reliably enforce it for the common reader+writer pair (it runs before
    //      the second DM kernel is registered). DM_DYNAMIC_NOC kernels are exempt: they may
    //      intentionally share a NOC, freeing the other NOC for fabric.
    // (Each kernel's effective node set is derived from WorkUnitSpec membership.)
    if (is_gen1_arch()) {
        // (node, processor) -> the kernel that already claimed it.
        std::map<std::pair<NodeCoord, DataMovementProcessor>, KernelSpecName> claimed_processor;
        // node -> (noc mode, the kernel that first set it) — all DM kernels on a node must agree.
        std::map<NodeCoord, std::pair<NOC_MODE, KernelSpecName>> node_noc_mode;
        // (node, noc) -> the kernel that already claimed it (dedicated-NOC kernels only).
        std::map<std::pair<NodeCoord, NOC>, KernelSpecName> claimed_noc;
        for (const auto& kernel : spec.kernels) {
            if (!kernel.is_data_movement_kernel()) {
                continue;
            }
            const auto& gen1 = std::get<DataMovementGen1Config>(std::get<DataMovementHardwareConfig>(kernel.hw_config));
            const NodeRangeSet& nodes = collected.kernel_node_set.at(kernel.unique_id);
            for (const auto& range : nodes.ranges()) {
                for (const auto& node : range) {
                    auto [proc_it, proc_inserted] =
                        claimed_processor.try_emplace(std::make_pair(node, gen1.processor), kernel.unique_id);
                    TT_FATAL(
                        proc_inserted,
                        "KernelSpec '{}' conflicts with '{}' on node ({}, {}): both claim the same DM processor. ",
                        kernel.unique_id,
                        proc_it->second,
                        node.x,
                        node.y);

                    // All DM kernels on a node must agree on NOC mode -- it configures shared per-core NOC
                    // hardware. Independent of the NOC-distinctness check below (which is gated per-kernel on
                    // DM_DEDICATED_NOC); their source order does not affect behavior.
                    auto [mode_it, mode_inserted] =
                        node_noc_mode.try_emplace(node, std::make_pair(gen1.noc_mode, kernel.unique_id));
                    TT_FATAL(
                        mode_inserted || mode_it->second.first == gen1.noc_mode,
                        "KernelSpec '{}' conflicts with '{}' on node ({}, {}): they set different NOC modes (one "
                        "DM_DEDICATED_NOC, the other DM_DYNAMIC_NOC). All data movement kernels on a node must use "
                        "the same NOC mode.",
                        kernel.unique_id,
                        mode_it->second.second,
                        node.x,
                        node.y);

                    // NOC-distinctness applies only to statically-pinned (dedicated-NOC) kernels.
                    if (gen1.noc_mode == NOC_MODE::DM_DEDICATED_NOC) {
                        auto [noc_it, noc_inserted] =
                            claimed_noc.try_emplace(std::make_pair(node, gen1.noc), kernel.unique_id);
                        TT_FATAL(
                            noc_inserted,
                            "KernelSpec '{}' conflicts with '{}' on node ({}, {}): both are dedicated-NOC data "
                            "movement kernels pinned to NOC_{}, which hangs the device. Give them distinct NOCs, or "
                            "use DM_DYNAMIC_NOC mode to intentionally share a NOC.",
                            kernel.unique_id,
                            noc_it->second,
                            node.x,
                            node.y,
                            static_cast<int>(gen1.noc));
                    }
                }
            }
        }
    }

    // Validate compute kernel unpack_modes entries against the per-DFB unpack legality table.
    //
    // "Unpack to Dest" means the unpacker writes a consumed DFB straight into the Dest register,
    // bypassing SrcA/B. Its legality depends on the Dest width (enable_32_bit_dest), the DFB's
    // element width, the binding role, and the generation:
    //
    //   UnpackToSrc                          → always accepted (the default path).
    //   UnpackToDest, producer-only binding  → inert (the DFB is never unpacked): tolerated.
    //   UnpackToDest, consumer, enable=true  → accepted (Dest is 32-bit; the choice is coherent).
    //   UnpackToDest, consumer, enable=false, 32-bit format (Float32/Int32/UInt32/RawUInt32)
    //                                        → REJECTED on every generation: a 32-bit datum cannot
    //                                          be unpacked into a 16-bit Dest register.
    //   UnpackToDest, consumer, enable=false, <=16-bit format, Gen1
    //                                        → REJECTED: bad for perf (bypasses SrcA/B for no gain).
    //   UnpackToDest, consumer, enable=false, <=16-bit format, Gen2
    //                                        → accepted: Gen2 has no unpack-to-Dest penalty.
    //   (A compute self-loop DFB binds both roles; the consumer rules govern it.)
    //
    // Separately, where the Src-vs-Dest choice is REAL an explicit entry is REQUIRED rather than
    // silently defaulting to UnpackToSrc: a consumed Float32 DFB with enable_32_bit_dest=true.
    //
    // INTENTIONAL INTERMEDIATE GAP — do not "fix" without the follow-up. The require-an-explicit-
    // entry rule is Float32-only. The choice is just as real for a consumed Int32/UInt32 DFB with
    // enable_32_bit_dest=true, and the end goal is to require an entry there too — but that is a
    // legality tightening that would reject roughly a dozen already-ported ops, so it is deferred to
    // a follow-up PR (see issue #49936). Until then, an unspecified int32/uint32 consumer silently
    // defaults to UnpackToSrc (its 32-bit value truncated to ~19 bits): wrong, but it preserves
    // existing behavior. (Some accepted UnpackToDest cases are also silently mishandled by the LLK
    // today — a codegen gap being fixed LLK-side, not a host-validation concern.)

    // A DataFormat whose elements are 32 bits wide, and so cannot be held by a 16-bit Dest register.
    // (Note: datum_size() throws on the block/MX formats.)
    auto is_32bit_element_format = [](tt::DataFormat fmt) {
        switch (fmt) {
            case tt::DataFormat::Float32:
            case tt::DataFormat::Int32:
            case tt::DataFormat::UInt32:
            case tt::DataFormat::RawUInt32: return true;
            default: return false;
        }
    };

    for (const auto& kernel : spec.kernels) {
        if (!kernel.is_compute_kernel()) {
            continue;
        }
        const auto& compute_config = std::get<ComputeHardwareConfig>(kernel.hw_config);
        const auto& unpack_modes =
            std::visit([](const auto& config) -> const auto& { return config.unpack_modes; }, compute_config);
        const bool enable_32_bit_dest =
            std::visit([](const auto& config) { return config.enable_32_bit_dest; }, compute_config);
        const bool is_gen2 = std::holds_alternative<ComputeGen2Config>(compute_config);

        // Index the kernel's DFB bindings: which it binds at all, and which it CONSUMES. A self-loop
        // DFB appears as two separate bindings (one PRODUCER, one CONSUMER — there is no BOTH endpoint
        // type); indexing by name into a set dedups them, and membership in consumed_dfbs makes the
        // consumer rules govern it.
        std::unordered_set<DFBSpecName> bound_dfbs;
        std::unordered_set<DFBSpecName> consumed_dfbs;
        for (const auto& binding : kernel.dfb_bindings) {
            bound_dfbs.insert(binding.dfb_spec_name);
            if (binding.endpoint_type == DFBEndpointType::CONSUMER) {
                consumed_dfbs.insert(binding.dfb_spec_name);
            }
        }

        // Validate each explicit entry, tracking which DFBs got one (to require one below where the
        // choice is real). Duplicate DFB entries are impossible: unpack_modes is a Table with unique
        // keys, so a repeated DFB overwrites the prior value.
        std::unordered_set<DFBSpecName> dfbs_with_entry;
        for (const auto& [dfb_name, mode] : unpack_modes) {
            dfbs_with_entry.insert(dfb_name);
            TT_FATAL(
                bound_dfbs.contains(dfb_name),
                "Kernel '{}' unpack_modes entry references DFB '{}', which the kernel does not bind",
                kernel.unique_id,
                dfb_name);

            if (mode == UnpackMode::UnpackToSrc) {
                continue;  // Always allowed.
            }
            //////////////////////////
            // mode == UnpackToDest
            //////////////////////////
            if (!consumed_dfbs.contains(dfb_name)) {
                continue;  // Compute kernel is bound as the DFB Producer: inert, tolerated.
            }

            // Compute kernel is the DFB's consumer.

            if (enable_32_bit_dest) {
                continue;  // UnpackTo Dest, with 32-bit Dest: always permitted
            }

            // UnpackToDest into a 16-bit Dest:
            // Legality checks are gen-specific, and depends on the element width.

            const DataflowBufferSpec* dfb_spec = collected.dfb_by_name.at(dfb_name);
            if (!dfb_spec->data_format_metadata.has_value()) {
                continue;  // Format unknown (deferred to the data_format-required check).
            }

            const tt::DataFormat fmt = dfb_spec->data_format_metadata.value();
            TT_FATAL(
                !is_32bit_element_format(fmt),
                "Compute kernel '{}' unpack_modes entry for DFB '{}' specifies UnpackToDest, but the DFB entries use a "
                "32-bit format ({}) and enable_32_bit_dest is false. A 32-bit datum cannot be unpacked into "
                "a 16-bit Dest register. Set enable_32_bit_dest=true, or use UnpackToSrc.",
                kernel.unique_id,
                dfb_name,
                fmt);
            TT_FATAL(
                is_gen2,
                "Compute kernel '{}' unpack_modes entry for DFB '{}' specifies UnpackToDest, but "
                "enable_32_bit_dest=false "
                "and the data type is not a 32-bit type. On Gen1 architectures, bypassing the SrcA/B path (with no "
                "precision benefit) is not permitted because it leads to worse performance. Use UnpackToSrc instead.",
                kernel.unique_id,
                dfb_name);
            // On Gen2, <=16-bit format + UnpackToDest + enable_32_bit_dest=false
            // is permitted. Unpacking to dest on Gen2 does not carry the performance penalty it does on Gen1.
        }

        // Require an explicit entry (i.e. don't assume a default) if the following conditions are all true:
        //  - the compute kernel is the DFB consumer
        //  - the data format is FP32
        //  - enable_32_bit_dest=true
        // NOTE: Int32/UInt32 are also 32-bit formats, but they are deliberately NOT required here yet.
        //       See the INTENTIONAL INTERMEDIATE GAP note above.
        //       This check should be extended to int32/uint32. (TODO: Issue #49936)
        if (enable_32_bit_dest) {
            for (const auto& binding : kernel.dfb_bindings) {
                if (binding.endpoint_type != DFBEndpointType::CONSUMER) {
                    continue;
                }
                const DataflowBufferSpec* dfb_spec = collected.dfb_by_name.at(binding.dfb_spec_name);
                if (!dfb_spec->data_format_metadata.has_value()) {
                    continue;  // Format unknown (deferred to the data_format-required check).
                }

                // FP32 only for now
                if (dfb_spec->data_format_metadata.value() != tt::DataFormat::Float32) {
                    continue;
                }
                TT_FATAL(
                    dfbs_with_entry.contains(binding.dfb_spec_name),
                    "Compute kernel '{}' consumes FP32 DFB '{}' with enable_32_bit_dest=true, but provides no "
                    "unpack_modes entry for this DFB. This configuration requires an explicit choice "
                    "between UnpackMode::UnpackToSrc and UnpackMode::UnpackToDest.",
                    kernel.unique_id,
                    binding.dfb_spec_name);
            }
        }
    }

    // Compute kernels cannot have any semaphore bindings.
    // (There's no use case for ever wanting this, so best just forbid it.)
    for (const auto& kernel : spec.kernels) {
        TT_FATAL(
            !kernel.is_compute_kernel() || kernel.semaphore_bindings.empty(),
            "KernelSpec '{}' has semaphore bindings. "
            "Semaphore bindings are not supported for compute kernels.",
            kernel.unique_id);
    }

    // Validate DM kernel disable_dfb_implicit_sync_for entries.
    //
    // Implicit sync is a Gen2-only, DM-only mechanism (ISR-based credit posting from NoC
    // transaction completion). A DM kernel can opt out per-DFB by listing the DFB's name in
    // its Gen2Config::disable_dfb_implicit_sync_for vector, or opt out of all the DFBs it binds at
    // once via Gen2Config::disable_dfb_implicit_sync_for_all. Either way the opt-out applies to the
    // side(s) of the DFB this kernel binds (producer, consumer, or both for a self-loop).
    //
    // Per-kernel rule: every listed name references a DFB the kernel binds (typo guard).
    //
    // Cross-kernel rule (per DFB): on each side independently, all DM kernels must agree on the
    // opt-out — either all disable it (by list or by _all), or none do. (Producer-side and
    // consumer-side are checked separately; the underlying hardware mechanism is per-side, with
    // one mask per side.)
    {
        // Per-kernel pass: typo guard.
        for (const auto& kernel : spec.kernels) {
            if (!kernel.is_data_movement_kernel()) {
                continue;
            }
            const auto& dm_config = std::get<DataMovementHardwareConfig>(kernel.hw_config);
            if (!std::holds_alternative<DataMovementGen2Config>(dm_config)) {
                continue;
            }
            std::unordered_set<DFBSpecName> bound_dfbs;
            for (const auto& binding : kernel.dfb_bindings) {
                bound_dfbs.insert(binding.dfb_spec_name);
            }
            for (const auto& dfb_name : std::get<DataMovementGen2Config>(dm_config).disable_dfb_implicit_sync_for) {
                TT_FATAL(
                    bound_dfbs.contains(dfb_name),
                    "Kernel '{}' disable_dfb_implicit_sync_for entry references DFB '{}', which the kernel does not "
                    "bind",
                    kernel.unique_id,
                    dfb_name);
            }
        }

        // Cross-kernel pass: per-DFB producer-side and consumer-side agreement.
        // Note: a single DFB can be bound by multiple producer KernelSpecs and multiple
        // consumer KernelSpecs — ops sometimes specialize the same kernel source by CTAs,
        // producing several KernelSpecs that share a DFB.
        auto check_side_agreement =
            [&](const std::vector<CollectedSpecData::DFBEndpointInfo::EndpointRecord>& endpoints,
                const DFBSpecName& dfb_name,
                std::string_view side_label) {
                const KernelSpec* canonical = nullptr;
                bool canonical_disables = false;
                for (const auto& ep : endpoints) {
                    if (!ep.kernel->is_data_movement_kernel()) {
                        continue;
                    }
                    const auto& dm_config = std::get<DataMovementHardwareConfig>(ep.kernel->hw_config);
                    if (!std::holds_alternative<DataMovementGen2Config>(dm_config)) {
                        // Gen1-only DM kernel — can't physically participate in Gen2 implicit sync; abstains.
                        continue;
                    }
                    const bool disables =
                        DmKernelDisablesImplicitSync(std::get<DataMovementGen2Config>(dm_config), dfb_name);
                    if (canonical == nullptr) {
                        canonical = ep.kernel;
                        canonical_disables = disables;
                        continue;
                    }
                    TT_FATAL(
                        disables == canonical_disables,
                        "DFB '{}' has disagreeing implicit-sync opt-out state on the {} side",
                        dfb_name,
                        side_label);
                }
            };
        for (const auto& [dfb_name, endpoint_info] : collected.dfb_endpoints) {
            check_side_agreement(endpoint_info.producers, dfb_name, "producer");
            check_side_agreement(endpoint_info.consumers, dfb_name, "consumer");
        }
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
                    spec.name,
                    spec.dataflow_buffers.size(),
                    max_dfbs);
            } else if (is_gen2_arch()) {
                TT_THROW(
                    "ProgramSpec '{}' has too many DataflowBufferSpecs ({}). The permitted "
                    "number of DFBs for the target architecture is configuration-dependent, "
                    "but {} is a hard upper limit.",
                    spec.name,
                    spec.dataflow_buffers.size(),
                    max_dfbs);
            } else {
                TT_FATAL(false, "Unknown architecture");
            }
        }
    }

    // Validate per-DFB sizing: entry_size and num_entries must be set to non-zero values.
    // (Sizes may still be overridden at runtime via ProgramRunArgs, but a ProgramSpec value is required.)
    for (const auto& dfb : spec.dataflow_buffers) {
        TT_FATAL(
            dfb.entry_size > 0,
            "DataflowBufferSpec '{}' has entry_size = 0. entry_size must be set to a non-zero value.",
            dfb.unique_id);
        TT_FATAL(
            dfb.num_entries > 0,
            "DataflowBufferSpec '{}' has num_entries = 0. num_entries must be set to a non-zero value.",
            dfb.unique_id);
    }

    // Validate local DFB endpoint placement and multi-binding consistency.
    //
    // The hardware invariant is local: a local DFB lives in shared SRAM on each node, so at every
    // node where the DFB is instantiated, exactly one producer kernel instance and exactly one
    // consumer kernel instance must run on that node. Metal 2.0 permits multiple PRODUCER
    // KernelSpecs (and multiple CONSUMER KernelSpecs) per DFB, so we enforce that invariant directly
    // as a per-node census, plus per-role uniformity of the binding-site config:
    //   1./2. Placement: every node hosting the DFB runs exactly one producer instance and exactly
    //         one consumer instance (the per-node census below). This subsumes both "no node has two
    //         same-role instances" and "producer and consumer node coverage coincide".
    //   3. All bindings on the same role have matching `access_pattern` (the DFB scheduler
    //      config is shared per role).
    //   4. All KernelSpecs on the same role have matching `num_threads` (the per-side
    //      credit-tracking config is shared per role).
    // Self-loop (a kernel that appears in both producers and consumers of a DFB) is currently
    // restricted to the simple single-producer-single-consumer case.
    for (const auto& dfb : spec.dataflow_buffers) {
        const auto& endpoints = collected.dfb_endpoints.at(dfb.unique_id);

        // (3) and (4): per-role uniformity of binding-site parameters, plus kernel kind.
        // Kind (compute vs DM) must agree because the DFB's hardware config carries a single
        // processor mask per role, and compute / DM masks live in disjoint bit ranges (bits
        // 0-7 vs 8-15 on Gen2; orthogonal RISC encodings on Gen1) — mismatched kinds cannot
        // share a mask.
        auto check_role_uniformity = [&](const auto& records, std::string_view role) {
            if (records.size() < 2) {
                return;
            }
            const auto first_pattern = records[0].binding->access_pattern;
            const auto first_threads = records[0].kernel->num_threads;
            const bool first_is_compute = records[0].kernel->is_compute_kernel();
            const auto& first_kernel = records[0].kernel->unique_id;
            for (size_t i = 1; i < records.size(); ++i) {
                TT_FATAL(
                    records[i].binding->access_pattern == first_pattern,
                    "DFB '{}' has multiple {} bindings with mismatched access_pattern (kernel '{}' vs kernel '{}')",
                    dfb.unique_id,
                    role,
                    first_kernel,
                    records[i].kernel->unique_id);
                TT_FATAL(
                    records[i].kernel->num_threads == first_threads,
                    "DFB '{}' has multiple {} KernelSpecs with mismatched num_threads (kernel '{}' = {} vs kernel '{}' "
                    "= {})",
                    dfb.unique_id,
                    role,
                    first_kernel,
                    first_threads,
                    records[i].kernel->unique_id,
                    records[i].kernel->num_threads);
                TT_FATAL(
                    records[i].kernel->is_compute_kernel() == first_is_compute,
                    "DFB '{}' has multiple {} KernelSpecs mixing compute and data-movement kinds "
                    "('{}' is a {} kernel; '{}' is a {} kernel). All KernelSpecs bound to the same "
                    "DFB role must be of the same kind — the DFB's hardware config carries a single "
                    "processor mask per role.",
                    dfb.unique_id,
                    role,
                    first_kernel,
                    first_is_compute ? "compute" : "data-movement",
                    records[i].kernel->unique_id,
                    first_is_compute ? "data-movement" : "compute");
            }
        };
        check_role_uniformity(endpoints.producers, "PRODUCER");
        check_role_uniformity(endpoints.consumers, "CONSUMER");

        // (1)/(2) Placement — per-node census. A local DFB lives in shared SRAM on each node, so
        // every node it is instantiated on must run exactly one producer instance and exactly one
        // consumer instance. Tally instances per node directly from the bindings: this subsumes the
        // old within-role disjointness check (a node with >1 same-role instance) and the cross-role
        // coverage check (a node with one role but not the other), and reports the offending node in
        // node terms rather than WorkUnitSpec terms. It counts actual node occupancy, so overlapping
        // same-role placements are caught regardless of how the WorkUnitSpec bookkeeping produced
        // them. (Self-loops fall out naturally: a self-looping kernel tallies as one producer AND
        // one consumer on each of its nodes.)
        std::unordered_map<NodeCoord, std::vector<const KernelSpec*>> producers_on_node;
        std::unordered_map<NodeCoord, std::vector<const KernelSpec*>> consumers_on_node;
        auto tally_role = [&](const auto& records, auto& on_node) {
            for (const auto& rec : records) {
                for (const NodeCoord& node : corerange_to_cores(collected.kernel_node_set.at(rec.kernel->unique_id))) {
                    on_node[node].push_back(rec.kernel);
                }
            }
        };
        tally_role(endpoints.producers, producers_on_node);
        tally_role(endpoints.consumers, consumers_on_node);

        // Footprint = every node hosting any instance of either role. A std::set gives deterministic
        // iteration order, hence deterministic error messages.
        std::set<NodeCoord> footprint;
        for (const auto& [node, kernels] : producers_on_node) {
            footprint.insert(node);
        }
        for (const auto& [node, kernels] : consumers_on_node) {
            footprint.insert(node);
        }

        auto names_at = [](const std::unordered_map<NodeCoord, std::vector<const KernelSpec*>>& on_node,
                           const NodeCoord& node) -> std::string {
            auto it = on_node.find(node);
            if (it == on_node.end() || it->second.empty()) {
                return "none";
            }
            std::string names;
            for (const KernelSpec* k : it->second) {
                names += (names.empty() ? "'" : ", '") + k->unique_id.get() + "'";
            }
            return names;
        };

        for (const NodeCoord& node : footprint) {
            auto p_it = producers_on_node.find(node);
            auto c_it = consumers_on_node.find(node);
            const size_t num_producers = p_it == producers_on_node.end() ? 0 : p_it->second.size();
            const size_t num_consumers = c_it == consumers_on_node.end() ? 0 : c_it->second.size();
            if (num_producers == 1 && num_consumers == 1) {
                continue;
            }
            std::string_view guidance;
            if (num_producers == 0) {
                guidance =
                    "This node has a consumer but no producer — ensure a producer kernel covers it "
                    "(via its WorkUnitSpec membership).";
            } else if (num_consumers == 0) {
                guidance =
                    "This node has a producer but no consumer — ensure a consumer kernel covers it "
                    "(via its WorkUnitSpec membership).";
            } else {
                guidance =
                    "Multiple same-role kernel instances land on this node — their placements overlap; "
                    "give each disjoint nodes.";
            }
            TT_FATAL(
                false,
                "Local DFB '{}' is malformed at node {}: {} producer instance(s) ({}) and {} consumer "
                "instance(s) ({}). A local DFB lives in shared SRAM on each node, so every node it is "
                "instantiated on must run exactly one producer and one consumer kernel instance. {}",
                dfb.unique_id,
                node.str(),
                num_producers,
                names_at(producers_on_node, node),
                num_consumers,
                names_at(consumers_on_node, node),
                guidance);
        }

        // Find a self-loop participant: a kernel bound to this DFB as both producer and consumer.
        // Stays nullptr if the DFB is not self-looped. Iterating producers in vector order keeps the
        // pick — and any resulting error message — deterministic across runs.
        const KernelSpec* self_loop_kernel = nullptr;
        for (const auto& p : endpoints.producers) {
            for (const auto& c : endpoints.consumers) {
                if (p.kernel == c.kernel) {
                    self_loop_kernel = p.kernel;
                    break;
                }
            }
            if (self_loop_kernel != nullptr) {
                break;
            }
        }

        if (self_loop_kernel != nullptr) {
            // A data-movement kernel may self-loop a DFB (bind it as both PRODUCER and CONSUMER) only
            // on Gen1 (WH/BH), where a DFB lowers to a plain circular buffer that a single DM RISC can
            // both fill and drain. On Gen2 the DFB's tile-counter credit machinery requires disjoint
            // producer/consumer RISCs, so a DM self-loop cannot be lowered. Catch it here (with a clear
            // message) rather than let it fall through to a confusing "producer_risc_mask and
            // consumer_risc_mask must not overlap" error in the DFB backend. (Compute self-loops are
            // always legal: they lower to the intra-Tensix packer->unpacker flow.)
            TT_FATAL(
                !(is_gen2_arch() && self_loop_kernel->is_data_movement_kernel()),
                "DataflowBuffer '{}' is self-looped by data-movement kernel '{}' (bound as both PRODUCER "
                "and CONSUMER). Self-loop DFBs are not supported for data-movement kernels on Gen2 "
                "architectures. Consider using a scratchpad or LocalTensorAccessor instead.",
                dfb.unique_id,
                self_loop_kernel->unique_id);

            // Self-loop interplay with multi-binding: the producer set must equal the consumer set
            // as sets of KernelSpec*. This permits the natural pattern of multiple same-source
            // KernelSpecs each self-looping the DFB on their disjoint node ranges, while rejecting
            // the case where a self-looping kernel shares the DFB with an unrelated kernel (which
            // would make the producer/consumer mask and lowering semantics ambiguous).
            std::unordered_set<const KernelSpec*> producer_kernels;
            std::unordered_set<const KernelSpec*> consumer_kernels;
            for (const auto& p : endpoints.producers) {
                producer_kernels.insert(p.kernel);
            }
            for (const auto& c : endpoints.consumers) {
                consumer_kernels.insert(c.kernel);
            }
            TT_FATAL(
                producer_kernels == consumer_kernels,
                "DFB '{}' is self-looped (some kernel appears as both producer and consumer), but "
                "the set of producer KernelSpecs differs from the set of consumer KernelSpecs. "
                "When a DFB is self-looped, every same-side binding must come from a self-loop "
                "participant (i.e. a kernel that appears on both sides).",
                dfb.unique_id);
        }
    }

    // Cross-node DFBs are not yet supported.
    //
    // TODO: When cross-node DFB is supported, add a validation checks. Enforce that
    //       each (producer_node, consumer_node) entry in producer_consumer_map has
    //       p_node != c_node.

    TT_FATAL(
        spec.cross_node_dataflow_buffers.empty(),
        "CrossNodeDataflowBufferSpec is part of the Metal 2.0 API surface but is not yet supported "
        "by the runtime. (ProgramSpec '{}' has {} cross-node DFB(s).)",
        spec.name,
        spec.cross_node_dataflow_buffers.size());

    // Scratchpad placement census (multi-binding rule).
    //
    // A scratchpad is private, node-local L1. More than one KernelSpec may bind the same
    // ScratchpadSpec, but only on disjoint nodes: each node hosting the scratchpad must run exactly
    // one binding kernel instance, so the per-node region stays private to that one kernel.
    // (Allocation and CRTA delivery are per-binding-kernel — allocate_scratchpads stacks each
    // kernel's scratchpad onto its own cores' allocators — so disjoint bindings never interact.) Two
    // binding kernels on the same node would be true sharing, which is deferred behind a future
    // AdvancedOption and rejected here. Mirrors the local-DFB per-node census above. (A kernel
    // binding the same scratchpad twice is already rejected during collection, so every binder here
    // is a distinct kernel.)
    for (const auto& scratchpad : spec.scratchpads) {
        auto binders_it = collected.scratchpad_binders.find(scratchpad.unique_id);
        if (binders_it == collected.scratchpad_binders.end() || binders_it->second.size() < 2) {
            continue;  // unbound (caught earlier) or single binder — always legal.
        }
        // Tally binding-kernel instances per node. A std::map keeps iteration (and error messages)
        // deterministic.
        std::map<NodeCoord, std::vector<const KernelSpec*>> binders_on_node;
        for (const KernelSpec* kernel : binders_it->second) {
            for (const NodeCoord& node : corerange_to_cores(collected.kernel_node_set.at(kernel->unique_id))) {
                binders_on_node[node].push_back(kernel);
            }
        }
        for (const auto& [node, kernels] : binders_on_node) {
            if (kernels.size() <= 1) {
                continue;
            }
            std::string names;
            for (const KernelSpec* kernel : kernels) {
                names += (names.empty() ? "'" : ", '") + kernel->unique_id.get() + "'";
            }
            TT_FATAL(
                false,
                "ScratchpadSpec '{}' is bound by {} kernel instances on node {} ({}). A scratchpad is "
                "private node-local L1; multiple kernels may bind the same scratchpad only on disjoint "
                "nodes, so each node's instance stays private to one kernel. Sharing one node's "
                "scratchpad across kernels is not yet supported — give each binding kernel disjoint nodes.",
                scratchpad.unique_id,
                kernels.size(),
                node.str(),
                names);
        }
    }

    // Validate borrowed-memory DFBs.
    //
    // A borrowed-memory DFB names a TensorParameter via DataflowBufferSpec::borrowed_from. The
    // backing MeshTensor flows through ProgramRunArgs::tensor_args at execution time.
    // We enforce only the safety-relevant checks:
    //  - the named parameter exists
    //  - the TensorSpec places storage in L1,
    //  - the spec is large enough
    // We don't validate any layout considerations (interleaved vs sharded, page / tile sizes, etc.)
    // That's on the user; this is an advanced feature.
    for (const auto& dfb : spec.dataflow_buffers) {
        if (!dfb.borrowed_from.has_value()) {
            continue;
        }
        const TensorParamName& tp_name = *dfb.borrowed_from;
        auto it = collected.tensor_parameter_by_name.find(tp_name);
        TT_FATAL(
            it != collected.tensor_parameter_by_name.end(),
            "DFB '{}' borrows memory from TensorParameter '{}', but no such TensorParameter is declared in the "
            "ProgramSpec.",
            dfb.unique_id,
            tp_name);
        const TensorSpec& tensor_spec = it->second->spec;
        TT_FATAL(
            tensor_spec.memory_config().is_l1(),
            "DFB '{}' borrows memory from TensorParameter '{}', but its TensorSpec is not L1-resident (L1 is "
            "required). Both L1 and L1_SMALL are accepted.",
            dfb.unique_id,
            tp_name);
        // Coarse spec-time sizing check against the TensorSpec's full packed size. No Buffer is
        // available at spec time, so we can't query the per-bank allocation; the precise per-bank
        // check fires at attach time in AttachBorrowedDFBBuffers (program_run_args.cpp), where
        // a Buffer is in hand. For sharded L1 tensors the two checks differ — a DFB can pass
        // here against the full-tensor size and still fail per-bank later. By design.
        const size_t dfb_bytes = static_cast<size_t>(dfb.entry_size) * static_cast<size_t>(dfb.num_entries);
        const size_t tensor_bytes = tensor_spec.compute_packed_buffer_size_bytes();
        TT_FATAL(
            dfb_bytes <= tensor_bytes,
            "DFB '{}' (entry_size {} * num_entries {} = {} bytes) is larger than its borrowed TensorParameter '{}' "
            "({} bytes).",
            dfb.unique_id,
            dfb.entry_size,
            dfb.num_entries,
            dfb_bytes,
            tp_name,
            tensor_bytes);
    }

    // Validate DFB alias groups.
    // Rules:
    //  1. Transitivity: every DFB in an alias group must list every other member in its
    //     alias_with field. This strict requirement is redundant by design. This is a
    //     "dangerous" feature that a kernel author should use deliberately.
    //  2. Same total size: entry_size * num_entries must match within a group.
    //  3. Same node coverage: each DFB in the group must cover the same set of nodes
    //  4. Consistent borrowed_from: either no member borrows, or all members borrow from
    //     the same TensorParameter. (Aliased borrows from the same memory object is a
    //     weird-but-valid scenario.)
    {
        // The "extended group" of a DFB is its alias_with plus the DFB itself. Two DFBs
        // are in the same alias group iff their extended groups are equal.
        auto extended_group = [](const DataflowBufferSpec& d) {
            std::set<DFBSpecName> s(dfb_alias_with(d).begin(), dfb_alias_with(d).end());
            s.insert(d.unique_id);
            return s;
        };

        // Pre-pass: every name in every alias_with must refer to a real DFB and must not
        // be self-referential.
        for (const auto& dfb : spec.dataflow_buffers) {
            for (const auto& alias_name : dfb_alias_with(dfb)) {
                TT_FATAL(
                    collected.dfb_by_name.contains(alias_name),
                    "DFB '{}' lists unknown alias '{}' in alias_with",
                    dfb.unique_id,
                    alias_name);
                TT_FATAL(alias_name != dfb.unique_id, "DFB '{}' lists itself in alias_with", dfb.unique_id);
            }
        }

        for (const auto& dfb : spec.dataflow_buffers) {
            if (dfb_alias_with(dfb).empty()) {
                continue;
            }
            const size_t total_size_a = static_cast<size_t>(dfb.entry_size) * static_cast<size_t>(dfb.num_entries);
            const auto group_a = extended_group(dfb);
            const auto& nodes_a = collected.dfb_node_set.at(dfb.unique_id);

            for (const auto& alias_name : dfb_alias_with(dfb)) {
                const DataflowBufferSpec* alias_spec = collected.dfb_by_name.at(alias_name);

                // Rule 1: full clique declaration.
                const auto group_b = extended_group(*alias_spec);
                if (group_a != group_b) {
                    TT_THROW(
                        "DFBs '{}' and '{}' do not declare the same alias group. Every DFB in an "
                        "alias group must list every other member in its alias_with field.",
                        dfb.unique_id,
                        alias_name);
                }

                // Rule 2: same total size.
                const size_t total_size_b =
                    static_cast<size_t>(alias_spec->entry_size) * static_cast<size_t>(alias_spec->num_entries);
                TT_FATAL(
                    total_size_a == total_size_b,
                    "Aliased DFBs '{}' and '{}' have different total sizes ({} vs {} bytes). "
                    "Aliased DFBs must have the same total size (entry_size * num_entries).",
                    dfb.unique_id, alias_name, total_size_a, total_size_b);

                // Rule 3: same node coverage.
                const auto& nodes_b = collected.dfb_node_set.at(alias_name);
                TT_FATAL(
                    nodes_a == nodes_b,
                    "Aliased DFBs '{}' and '{}' cover different sets of nodes. Aliased DFBs must "
                    "cover the same node coverage (their bound kernels' WorkUnitSpec membership "
                    "must yield identical target_nodes unions) — the shared L1 region must be "
                    "reserved at the same cores for all members.",
                    dfb.unique_id,
                    alias_name);

                // Rule 4: consistent borrowed_from.
                TT_FATAL(
                    dfb.borrowed_from == alias_spec->borrowed_from,
                    "Aliased DFBs '{}' and '{}' have inconsistent borrowed_from. Either no member "
                    "of an alias group borrows, or all members borrow from the same TensorParameter.",
                    dfb.unique_id,
                    alias_name);
            }
        }
    }

    // Data format metadata (optional param) MUST be specified for a DFB with a compute endpoint
    auto any_compute_endpoint = [](const auto& records) {
        for (const auto& rec : records) {
            if (rec.kernel->is_compute_kernel()) {
                return true;
            }
        }
        return false;
    };
    for (const auto& [dfb_name, endpoint_info] : collected.dfb_endpoints) {
        if (any_compute_endpoint(endpoint_info.producers) || any_compute_endpoint(endpoint_info.consumers)) {
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
        const uint32_t init_value = sem.advanced_options.initial_value;
        if (is_gen2_arch()) {
            TT_FATAL(
                init_value == 0,
                "SemaphoreSpec '{}' has initial_value={} but only zero is supported on Quasar",
                sem.unique_id,
                init_value);
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
            if (work_unit.name == other_work_unit.name) {
                continue;
            }
            if (nodes_intersect(work_unit.target_nodes, other_work_unit.target_nodes)) {
                TT_FATAL(
                    false,
                    "WorkUnitSpecs '{}' and '{}' overlap in target nodes",
                    work_unit.name,
                    other_work_unit.name);
            }
        }
    }

    // A WorkUnitSpec must have at least one kernel
    for (const auto& work_unit : work_units) {
        TT_FATAL(!work_unit.kernels.empty(), "WorkUnitSpec '{}' has no kernels", work_unit.name);
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
            if (kernel_spec->is_data_movement_kernel()) {
                dm_cores_needed += kernel_spec->num_threads;
            }
        }
        if (is_gen2_arch()) {
            TT_FATAL(
                compute_engines_needed <= QUASAR_TENSIX_ENGINES_PER_NODE,
                "WorkUnitSpec '{}' needs {} Tensix engines, but only {} are available",
                work_unit.name,
                compute_engines_needed,
                QUASAR_TENSIX_ENGINES_PER_NODE);
            TT_FATAL(
                dm_cores_needed <= QUASAR_USER_DM_CORES_PER_NODE,
                "WorkUnitSpec '{}' requests {} data movement cores. This exceeds the permitted maximum of {}.",
                work_unit.name,
                dm_cores_needed,
                QUASAR_USER_DM_CORES_PER_NODE);
        }
        if (is_gen1_arch()) {
            TT_FATAL(
                compute_engines_needed <= 1,
                "WorkUnitSpec '{}' has {} compute kernels. The target architecture supports at most one.",
                work_unit.name,
                compute_engines_needed);
            TT_FATAL(
                dm_cores_needed <= 2,
                "WorkUnitSpec '{}' has {} data movement kernels. The target architecture supports at most two.",
                work_unit.name,
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
        TT_FATAL(num_compute_kernels <= 1, "WorkUnitSpec '{}' has more than one compute kernel", work_unit.name);
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
    const std::string& work_unit_id) {
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

// Equivalence class of DM kernels coupled by shared DFB endpoint roles.
//
// All DM kernels bound to the same DFB on the same role (PRODUCER/CONSUMER) must end up
// with identical DM RISC masks — the DFB's hardware config carries a single mask per role.
// Membership is computed by union-find over DM kernels: two kernels are merged if they share
// any DFB endpoint role; the transitive closure yields KernelCouplingGroups.
//
// num_threads is uniform within a group (the per-DFB-side num_threads validator + transitive
// equality guarantees this for any chain of shared endpoints).
//
// The DM solver operates on KernelCouplingGroups instead of individual KernelSpecs: each group is
// assigned one DMProcessorMask, which then applies to every member. A non-multi-bound DM
// kernel ends up in a singleton group.
struct KernelCouplingGroup {
    std::vector<const KernelSpec*> members;  // ≥ 1; canonical member is members.front()
    NodeRangeSet merged_node_set;            // union of members' node sets
    uint8_t num_threads = 0;                 // shared across members
};

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

// Result map: one DMProcessorMask per KernelCouplingGroup (which expands to its member kernels).
using KernelCouplingGroupMaskMap = std::unordered_map<const KernelCouplingGroup*, DMProcessorMask>;

// Constraint score for sorting: higher = more constrained (assigned earlier)
int ConstraintScore(const KernelCouplingGroup* g) {
    int node_count = static_cast<int>(g->merged_node_set.num_cores());
    int thread_count = static_cast<int>(g->num_threads);
    return (node_count * 100) + thread_count;  // nodes dominate, threads break ties
}

// Deterministic tiebreaker: sort by the lexicographically-smallest member unique_id.
// Group::members is canonicalized at construction time so members.front() is the sort key.
const std::string& group_sort_key(const KernelCouplingGroup* g) { return g->members.front()->unique_id.get(); }

void SortByConstraint(std::vector<const KernelCouplingGroup*>& groups) {
    std::sort(groups.begin(), groups.end(), [](const KernelCouplingGroup* a, const KernelCouplingGroup* b) {
        int score_a = ConstraintScore(a);
        int score_b = ConstraintScore(b);
        if (score_a != score_b) {
            return score_a > score_b;  // Higher score first
        }
        return group_sort_key(a) < group_sort_key(b);
    });
}

// Try to assign all groups in the given order using greedy selection.
// Returns true if successful, populates result map.
bool TryGreedyAssignment(
    const std::vector<const KernelCouplingGroup*>& group_order,
    NodeUsageTracker& tracker,
    KernelCouplingGroupMaskMap& result) {
    for (const KernelCouplingGroup* group : group_order) {
        DMProcessorMask combined_used = tracker.get_combined_used_mask(group->merged_node_set);

        auto selected = ReserveProcessors(group->num_threads, combined_used);
        if (!selected.has_value()) {
            return false;  // Can't assign this group
        }

        result[group] = selected.value();
        tracker.mark_used(group->merged_node_set, selected.value());
    }
    return true;
}

// Backtracking solver over kernel coupling group orderings.
// Note: In the worst case, this is O(N!) in the number of groups.
//       In practice, I expect this will almost always solve in the first greedy attempt (if sorted).
//       The backtracking is just here for pathological cases.
//       Even then, it shouldn't be horrendous. We won't have a huge number of kernels in a ProgramSpec.
//       And in the common case (traced), Program creation isn't on the critical path.
//       We can revisit if this ever becomes a problem.
bool SolveWithOrderingBacktrack(
    std::vector<const KernelCouplingGroup*> groups,  // by value - we'll permute it
    NodeUsageTracker& tracker,
    KernelCouplingGroupMaskMap& result) {
    // Try current ordering
    if (TryGreedyAssignment(groups, tracker, result)) {
        return true;
    }

    // Backtrack: try all permutations.
    // (std::next_permutation requires sorted input.)
    auto by_name = [](const KernelCouplingGroup* a, const KernelCouplingGroup* b) {
        return group_sort_key(a) < group_sort_key(b);
    };
    std::sort(groups.begin(), groups.end(), by_name);
    do {
        tracker.reset();
        result.clear();
        if (TryGreedyAssignment(groups, tracker, result)) {
            return true;
        }
    } while (std::next_permutation(groups.begin(), groups.end(), by_name));

    return false;
}

// Build DM kernel groups via union-find over shared DFB endpoint roles.
//
// Two DM kernels are merged if they share a DFB endpoint role (both PRODUCER of the same DFB,
// or both CONSUMER of the same DFB). The transitive closure yields equivalence classes.
//
// Compute kernels are not eligible — they don't participate in the DM solver. (Compute kernels
// bound to the same DFB role share num_threads, which makes AssignComputeProcessors deterministic
// across them, so no equivalence-class machinery is needed for compute.)
std::vector<KernelCouplingGroup> BuildDMKernelCouplingGroups(
    const std::vector<const KernelSpec*>& dm_kernels,
    const CollectedSpecData& collected,
    const KernelNodeSetMap& kernel_node_set) {
    // Small N — flat union-find indexed by position in dm_kernels.
    std::unordered_map<const KernelSpec*, size_t> idx_of;
    idx_of.reserve(dm_kernels.size());
    for (size_t i = 0; i < dm_kernels.size(); ++i) {
        idx_of[dm_kernels[i]] = i;
    }

    std::vector<size_t> parent(dm_kernels.size());
    std::iota(parent.begin(), parent.end(), size_t{0});
    auto find = [&parent](size_t x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];  // path compression
            x = parent[x];
        }
        return x;
    };
    auto unite = [&](size_t a, size_t b) {
        size_t ra = find(a), rb = find(b);
        if (ra != rb) {
            parent[ra] = rb;
        }
    };

    // For each DFB, union all DM kernels on each side (PRODUCER, CONSUMER) independently.
    auto union_same_side = [&](const auto& endpoints) {
        std::optional<size_t> anchor;
        for (const auto& rec : endpoints) {
            if (!rec.kernel->is_data_movement_kernel()) {
                continue;
            }
            const size_t k = idx_of.at(rec.kernel);
            if (!anchor.has_value()) {
                anchor = k;
            } else {
                unite(anchor.value(), k);
            }
        }
    };
    for (const auto& [dfb_name, endpoint_info] : collected.dfb_endpoints) {
        union_same_side(endpoint_info.producers);
        union_same_side(endpoint_info.consumers);
    }

    // Collect classes: root index → group.
    // Iterate dm_kernels in given order to preserve a deterministic per-class member order.
    std::unordered_map<size_t, size_t> root_to_group_idx;
    std::vector<KernelCouplingGroup> groups;
    for (size_t i = 0; i < dm_kernels.size(); ++i) {
        const size_t r = find(i);
        auto [it, inserted] = root_to_group_idx.try_emplace(r, groups.size());
        if (inserted) {
            groups.emplace_back();
        }
        groups[it->second].members.push_back(dm_kernels[i]);
    }

    // Finalize each group: merged_node_set + num_threads + canonical member sort.
    for (auto& g : groups) {
        std::sort(g.members.begin(), g.members.end(), [](const KernelSpec* a, const KernelSpec* b) {
            return a->unique_id < b->unique_id;
        });
        for (const KernelSpec* k : g.members) {
            g.merged_node_set = g.merged_node_set.merge(kernel_node_set.at(k->unique_id));
        }
        g.num_threads = g.members.front()->num_threads;
    }
    return groups;
}

}  // namespace dm_solver

// Gen2 (Quasar) processor assignment: runs the backtracking DM solver and returns
// a KernelRiscMaskMap using the Gen2 bit encoding (DM: bits 0-7, compute: bits 8-15).
//
// The DM solver operates on KernelCouplingGroups (equivalence classes of DM kernels coupled by shared
// DFB endpoint roles), not individual kernels. Each group is assigned one DMProcessorMask
// which then applies to every member kernel. This ensures multi-bound same-role kernels end
// up with identical masks, matching the per-side single-mask shape of DataflowBufferConfig.
KernelRiscMaskMap SolveGen2KernelRiscMasks(const ProgramSpec& spec, const CollectedSpecData& collected) {
    ComputeEngineMaskMap compute_assignments;

    // Collect DM kernels and compute kernels separately.
    // Compute kernels get a deterministic per-kernel mask (one compute kernel per node assumption;
    // num_threads uniformity is enforced upstream, so same-role compute kernels get identical masks
    // without any coupling-group machinery).
    std::vector<const KernelSpec*> dm_kernels;
    for (const KernelSpec& kernel : spec.kernels) {
        if (kernel.is_data_movement_kernel()) {
            dm_kernels.push_back(&kernel);
        } else {
            compute_assignments[&kernel] = AssignComputeProcessors(&kernel, kernel.unique_id);
        }
    }

    // Build DM kernel groups (equivalence classes via shared DFB endpoint roles).
    // Each group's merged_node_set is the union of its members' node sets — the solver
    // will pick a mask that's free on every node in that union.
    std::vector<dm_solver::KernelCouplingGroup> groups =
        dm_solver::BuildDMKernelCouplingGroups(dm_kernels, collected, collected.kernel_node_set);

    std::vector<const dm_solver::KernelCouplingGroup*> group_ptrs;
    group_ptrs.reserve(groups.size());
    for (const auto& g : groups) {
        group_ptrs.push_back(&g);
    }

    // Sort by constraint score (most constrained first)
    constexpr bool kSortByConstraint = true;  // Toggle to disable upfront sorting
    if constexpr (kSortByConstraint) {
        dm_solver::SortByConstraint(group_ptrs);
    }

    // Solve DM assignments at the group level
    dm_solver::NodeUsageTracker tracker;
    dm_solver::KernelCouplingGroupMaskMap group_assignments;
    bool success = dm_solver::SolveWithOrderingBacktrack(group_ptrs, tracker, group_assignments);

    TT_FATAL(
        success,
        "Failed to find valid processor assignments for DM kernels. "
        "Either the ProgramSpec is invalid, or that the \"same DM cores on every node\" "
        "simplifying assumption has been violated.");

    // Convert to KernelRiscMaskMap using Gen2 bit encoding: expand each group's mask to all members.
    KernelRiscMaskMap result;
    for (const auto& [group, mask] : group_assignments) {
        for (const KernelSpec* member : group->members) {
            result[member] = mask.bits;  // DM processors in bits 0-7
        }
    }
    for (const auto& [kernel, mask] : compute_assignments) {
        result[kernel] = static_cast<uint16_t>(mask.bits) << 8;  // Compute engines in bits 8-15
    }
    return result;
}

// Gen1 (WH/BH) processor assignment: read the kernel's gen1_config processor and return a
// KernelRiscMaskMap using the Gen1 bit encoding (RISCV_0: bit 0, RISCV_1: bit 1, compute: bit 2).
KernelRiscMaskMap BuildGen1KernelRiscMasks(const ProgramSpec& spec) {
    static constexpr uint8_t GEN1_COMPUTE_RISC_BIT = 2;

    KernelRiscMaskMap result;
    for (const KernelSpec& kernel : spec.kernels) {
        if (kernel.is_data_movement_kernel()) {
            const auto& dm_config = std::get<DataMovementHardwareConfig>(kernel.hw_config);
            const auto gen1 = std::get<DataMovementGen1Config>(dm_config);
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

// Per-TensorParameter resolved layout.
//
// cta_payload: positional CTA words appended to the kernel's compile-time args for
// each binding of this TensorParameter. Mirrors what TensorAccessorArgs::append_to
// would produce on the legacy path, but is built from the TensorSpec + MeshDevice
// since no Buffer exists at spec-build time.
//
// extra_crta_words: additional CRTA words (beyond the always-present base address
// slot) that this binding occupies, used by the device-side accessor to read
// runtime-resolved fields. Non-zero when the TensorParameter opts into a dynamic
// field that lives in CRTAs: either sharded + dynamic_tensor_shape (which puts
// `rank` shape words in CRTAs), or interleaved row-major + dynamic_tensor_shape (one
// page-size word). The two are mutually exclusive per binding -- see runtime_field_is_page_size.
struct ResolvedTensorParameter {
    std::vector<uint32_t> cta_payload;

    // How many CRTA words (beyond the base address) does this binding consume?
    // This is only used if TensorParameter relaxations have been requested.
    uint32_t extra_crta_words = 0;

    // What info the runtime field CRTA words actually contain depends on the relaxation.
    // Currently, there are only two mutually exclusive possibilities (though more may be added):
    //  1. The interleaved row-major page-size (one CRTA only)
    //  2. The sharded dynamic_tensor_shape shape (one CRTA per tensor dim)
    // For now, since there are only two mutually exclusive possibilities, it's sufficient to
    // distinguish them with a boolean.
    bool runtime_field_is_page_size = false;
};

// Resolve a TensorParameter's static layout into a CTA payload + an extra CRTA word
// count for any runtime-resolved fields.
//
// CTA layout produced:
//  - word 0 is the args_config raw byte
//  - word 1 is aligned_page_size
// For sharded tensors only:
//  - word 2 is rank
//  - word 3 is num_banks
//  - remaining words: per-dim tensor_shape_in_pages (omitted if dynamic_tensor_shape),
//    per-dim shard_shape_in_pages, and packed bank coordinates (two per uint32)
//
// The tensor base address always lives in CRTAs (filled in per-enqueue from the
// corresponding TensorArgument). When dynamic_tensor_shape is set on a sharded tensor,
// the runtime tensor's shape is also written into CRTAs at enqueue time, in
// `extra_crta_words` slots immediately after the address slot.
// (See also ResolveTensorBindingsForKernel below.)
ResolvedTensorParameter ResolveTensorParameterStaticCTAs(
    const TensorParameter& tensor_parameter, const distributed::MeshDevice& mesh_device) {
    const TensorSpec& spec = tensor_parameter.spec;
    const MemoryConfig& memory_config = spec.memory_config();
    const BufferType buffer_type = memory_config.buffer_type();
    const bool is_dram = (buffer_type == BufferType::DRAM);
    const bool is_sharded = memory_config.is_sharded();
    // dynamic_tensor_shape is only meaningful on sharded tensors: for interleaved
    // tensors the CTA payload never carried tensor_shape in the first place (and
    // the device-side accessor doesn't read it), so the flag is a pure host-side
    // validation loosening and has no effect on the CTA/CRTA layout.
    const bool dyn_shape = tensor_parameter.advanced_options.dynamic_tensor_shape && is_sharded;
    // dynamic_tensor_shape lets the bound tensor's logical shape vary. For an interleaved ROW-MAJOR
    // tensor the page size (= last_dim_width * elem_size) is part of that varying shape, so it must
    // ride a runtime CRTA word too -- otherwise it goes stale on a program-cache hit and the
    // accessor strides by the wrong number of bytes. We fold that in here rather than expose a
    // separate flag: a useful page-size change is ALWAYS a shape change on row-major (you can't vary
    // the width without varying the logical shape), so there is no "page size varies but shape
    // doesn't" case to give a flag to. Tiled page size is dtype-fixed and sharded page size is
    // spec-fixed, so neither triggers this; sharded dynamic_tensor_shape carries shape-in-pages
    // words instead (dyn_shape above). dyn_shape and dyn_page are mutually exclusive by layout.
    const bool dyn_page =
        tensor_parameter.advanced_options.dynamic_tensor_shape && !is_sharded && spec.layout() == Layout::ROW_MAJOR;

    tensor_accessor::ArgsConfig args_config;
    if (is_sharded) {
        args_config.set(tensor_accessor::ArgConfig::Sharded);
    }
    if (is_dram) {
        args_config.set(tensor_accessor::ArgConfig::IsDram);
    }
    if (dyn_shape) {
        args_config.set(tensor_accessor::ArgConfig::RuntimeTensorShape);
    }
    if (dyn_page) {
        args_config.set(tensor_accessor::ArgConfig::RuntimePageSize);
    }

    // aligned_page_size: align the unaligned page size up to the buffer-type alignment.
    const size_t unaligned_page_size = spec.compute_page_size_bytes();
    const uint32_t alignment = mesh_device.allocator()->get_alignment(buffer_type);
    const size_t aligned_page_size = align(unaligned_page_size, static_cast<size_t>(alignment));
    TT_FATAL(
        aligned_page_size <= std::numeric_limits<uint32_t>::max(),
        "TensorParameter '{}' aligned page size {} exceeds uint32_t max {}",
        tensor_parameter.unique_id,
        aligned_page_size,
        std::numeric_limits<uint32_t>::max());

    ResolvedTensorParameter result;
    std::vector<uint32_t>& cta_payload = result.cta_payload;

    // Common header (always emitted, sharded or not):
    cta_payload.push_back(args_config.raw());
    // If the page size is static, it rides as a CTA.
    // (If it's dynamic, it will live in a CRTA word instead.)
    if (!dyn_page) {
        cta_payload.push_back(static_cast<uint32_t>(aligned_page_size));
    } else {
        TT_FATAL(!is_sharded, "Internal error: dynamic page size should not occur on a sharded tensor parameter");

        // One runtime field: the page size, re-derived from the bound buffer each dispatch
        // and emitted immediately after the base-address word (see EmitBindingCrtaValues).
        result.extra_crta_words = 1;
        result.runtime_field_is_page_size = true;
    }

    // The rest of the logic in this function pertains to sharded tensors only.
    // Early return for a non-sharded tensor.
    if (!is_sharded) {
        return result;
    }

    //////////////////////////////////
    // Sharded tensor handling
    //////////////////////////////////

    // Sharded: emit rank, num_banks, tensor_shape_in_pages (CTA only when static),
    // shard_shape_in_pages, bank_coords.
    const BufferShardingArgs sharding_args = spec.compute_buffer_sharding_args();
    const std::optional<BufferDistributionSpec>& bds_opt = sharding_args.buffer_distribution_spec();
    TT_FATAL(
        bds_opt.has_value(),
        "TensorParameter '{}' is sharded but TensorSpec produced no BufferDistributionSpec",
        tensor_parameter.unique_id);
    const BufferDistributionSpec& bds = *bds_opt;

    const Shape& tensor_shape = bds.tensor_shape_in_pages();
    const Shape& shard_shape = bds.shard_shape_in_pages();
    const std::vector<CoreCoord>& bank_coords = bds.cores();
    const size_t rank = tensor_shape.rank();
    const size_t n_banks = bank_coords.size();

    cta_payload.push_back(static_cast<uint32_t>(rank));
    cta_payload.push_back(static_cast<uint32_t>(n_banks));

    if (!dyn_shape) {
        for (size_t i = 0; i < rank; ++i) {
            cta_payload.push_back(static_cast<uint32_t>(tensor_shape[i]));
        }
    } else {
        // Shape lives in CRTAs (one word per dim, written at enqueue time from the bound MeshTensor).
        result.extra_crta_words = static_cast<uint32_t>(rank);
    }
    for (size_t i = 0; i < rank; ++i) {
        cta_payload.push_back(static_cast<uint32_t>(shard_shape[i]));
    }

    // Bank coords packed two-per-uint32.
    // Non-DRAM coords are virtualized; DRAM coords are kept logical (DRAM bank id == logical x).
    const CoreType core_type = is_dram ? CoreType::DRAM : CoreType::WORKER;
    auto resolve_coord = [&](const CoreCoord& logical) -> CoreCoord {
        if (is_dram) {
            return logical;
        }
        return mesh_device.virtual_core_from_logical_core(logical, core_type);
    };
    for (size_t i = 0; i < n_banks; i += 2) {
        const CoreCoord c1 = resolve_coord(bank_coords[i]);
        if (i + 1 < n_banks) {
            const CoreCoord c2 = resolve_coord(bank_coords[i + 1]);
            cta_payload.push_back(((c2.x & 0xFF) << 24) | ((c2.y & 0xFF) << 16) | ((c1.x & 0xFF) << 8) | (c1.y & 0xFF));
        } else {
            cta_payload.push_back(((c1.x & 0xFF) << 8) | (c1.y & 0xFF));
        }
    }

    return result;
}

// Per-kernel resolved tensor binding data:
//  - All the kernel's TensorBindingHandle (type is defined in kernel.hpp)
//  - The positional CTAs to append to the kernel's (unnamed) CTAs
//  - The full CRTA buffer layout (named CRTAs + binding section + vararg-section start),
//    precomputed here so consumers (headergen, runtime) don't have to re-derive section
//    boundaries by walking handles. See KernelCrtaLayout in jit_build_settings.hpp.
struct TensorBindingsForKernel {
    std::vector<TensorBindingHandle> handles;
    std::vector<uint32_t> cta_words;  // appended after any pre-existing positional CTAs
                                      // (currently, this is the only Metal 2.0 use of positional CTAs)
    KernelCrtaLayout crta_layout;
};

// Resolve the tensor bindings for a single kernel:
//  1. Walk the kernel's tensor_bindings in declaration order
//  2. Pack each binding's CTA payload into a contiguous positional buffer
//  3. Assign each binding a slot in the kernel's TensorBinding section of the CRTA buffer.
//     The section is structurally separate, immediately following the user-named CRTAs.
//     Each binding occupies (1 + extra_crta_words) words: the always-present base address
//     word, plus any runtime accessor field words (e.g. shape, when dynamic_tensor_shape
//     is set on a sharded TensorParameter).
//  4. Record the resulting CRTA buffer layout (the three section sizes) on the output, so
//     the headergen and runtime can consult it directly instead of re-summing the bindings.
//
// (SetProgramRunArgs will fill the address slots and runtime field slots at enqueue
// time, extracting info from the TensorArgs.)
TensorBindingsForKernel ResolveTensorBindingsForKernel(
    const KernelSpec& kernel,
    const std::unordered_map<TensorParamName, ResolvedTensorParameter>& resolved_tensor_parameters,
    size_t base_named_crta_count) {
    TensorBindingsForKernel out;
    out.handles.reserve(kernel.tensor_bindings.size());

    uint32_t cta_word_offset = 0;
    size_t crta_word_index = base_named_crta_count;
    uint32_t binding_section_words = 0;
    for (const auto& binding : kernel.tensor_bindings) {
        const ResolvedTensorParameter& resolved = resolved_tensor_parameters.at(binding.tensor_parameter_name);
        const std::vector<uint32_t>& binding_ctas = resolved.cta_payload;

        TensorBindingHandle handle;
        handle.accessor_name = binding.accessor_name;
        handle.tensor_parameter_name = binding.tensor_parameter_name.get();
        handle.cta_offset = cta_word_offset;
        handle.addr_crta_offset = static_cast<uint32_t>(crta_word_index * sizeof(uint32_t));
        handle.num_runtime_field_crta_words = resolved.extra_crta_words;
        handle.runtime_field_is_page_size = resolved.runtime_field_is_page_size;

        out.cta_words.insert(out.cta_words.end(), binding_ctas.begin(), binding_ctas.end());
        cta_word_offset += static_cast<uint32_t>(binding_ctas.size());
        const uint32_t binding_words = 1u + resolved.extra_crta_words;
        crta_word_index += binding_words;
        binding_section_words += binding_words;

        out.handles.push_back(std::move(handle));
    }

    out.crta_layout.num_named_words = static_cast<uint32_t>(base_named_crta_count);
    out.crta_layout.binding_section_words = binding_section_words;
    out.crta_layout.vararg_section_offset = static_cast<uint32_t>(base_named_crta_count) + binding_section_words;

    return out;
}

// Per-kernel resolved scratchpad bindings:
//  - one CRTA word per binding (the scratchpad's allocated L1 base address), in declaration order
//  - the scratchpad section sits immediately after the TensorBinding section and before varargs, so
//    each binding's absolute CRTA word index (and thus addr_crta_word) is fixed at codegen time
//    (varargs are open-ended / runtime-counted, so a section placed after them would not be).
// The allocated_address is left 0 here; allocate_scratchpads fills it once L1 is allocated.
struct ScratchpadBindingsForKernel {
    std::vector<ScratchpadBindingHandle> handles;
    uint32_t section_words = 0;  // == number of scratchpad bindings
};

ScratchpadBindingsForKernel ResolveScratchpadBindingsForKernel(
    const KernelSpec& kernel,
    const std::unordered_map<ScratchpadSpecName, const ScratchpadSpec*>& scratchpad_by_name,
    size_t scratchpad_base_crta_word) {
    ScratchpadBindingsForKernel out;
    out.handles.reserve(kernel.scratchpad_bindings.size());

    size_t crta_word_index = scratchpad_base_crta_word;
    for (const auto& binding : kernel.scratchpad_bindings) {
        const ScratchpadSpec* scratchpad_spec = scratchpad_by_name.at(binding.scratchpad_spec_name);

        ScratchpadBindingHandle handle;
        handle.accessor_name = binding.accessor_name;
        handle.size_bytes = scratchpad_spec->size_per_node;
        handle.addr_crta_word = static_cast<uint32_t>(crta_word_index);
        // handle.allocated_address stays 0 until allocate_scratchpads runs.
        out.handles.push_back(std::move(handle));
        crta_word_index += 1;  // one address word per scratchpad binding
    }

    out.section_words = static_cast<uint32_t>(kernel.scratchpad_bindings.size());
    return out;
}

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
        out.emplace(dfb_binding.accessor_name, static_cast<uint16_t>(id));
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
    // With multi-binding, all same-role KernelSpecs share kind (DM/compute), access_pattern,
    // num_threads, and risc_mask. The first three are enforced in ValidateProgramSpec; the
    // fourth is solver-guaranteed on Gen2 (the coupling-group equivalence-class constraint)
    // and user-validated on Gen1 (see Step 2b in MakeProgramFromSpec). So any
    // representative producer/consumer gives the correct DFB config — we take the first.
    const KernelSpec* producer = dfb_endpoint_info.producers.front().kernel;
    const KernelSpec* consumer = dfb_endpoint_info.consumers.front().kernel;
    const DFBBinding* producer_binding = dfb_endpoint_info.producers.front().binding;
    const DFBBinding* consumer_binding = dfb_endpoint_info.consumers.front().binding;

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
    auto producer_access_pattern = to_hw_access_pattern(producer_binding->access_pattern);
    auto consumer_access_pattern = to_hw_access_pattern(consumer_binding->access_pattern);

    // A compute kernel that self-loops a DFB (binds it as both producer and consumer) lowers to the
    // intra-Tensix packer->unpacker flow, so the lower-layer DFB API needs TensixScope::INTRA. The
    // Metal 2.0 surface does not expose a scope option — INTRA is the only supported topology, applied
    // automatically here. Self-loop is detected as any overlap between the producer and consumer kernel
    // sets — under the multi-binding regime the first-record pointers may differ even when the kernel
    // sets are identical (the overlap is what matters, not vector ordering). Upstream validation
    // guarantees producer set == consumer set whenever any overlap exists, so reading from the first
    // producer is safe and representative. A DM self-loop (Gen1-only) needs no tensix_scope.
    const bool is_self_loop = [&] {
        for (const auto& p : dfb_endpoint_info.producers) {
            for (const auto& c : dfb_endpoint_info.consumers) {
                if (p.kernel == c.kernel) {
                    return true;
                }
            }
        }
        return false;
    }();
    std::optional<experimental::dfb::TensixScope> tensix_scope;
    if (is_self_loop && producer->is_compute_kernel()) {
        tensix_scope = experimental::dfb::TensixScope::INTRA;
    }

    // Compute the per-side implicit-sync value by polling the bound DM kernels' Gen2 votes.
    // Sides with no DM endpoints get implicit_sync=false (no DM endpoint to enable it for).
    // Validator guarantees per-side agreement among DM kernels, so any DM kernel's vote works.
    auto side_implicit_sync_enabled =
        [&](const std::vector<CollectedSpecData::DFBEndpointInfo::EndpointRecord>& endpoints) -> bool {
        bool any_dm = false;
        bool disabled = false;
        for (const auto& ep : endpoints) {
            if (!ep.kernel->is_data_movement_kernel()) {
                continue;
            }
            any_dm = true;
            const auto& dm_config = std::get<DataMovementHardwareConfig>(ep.kernel->hw_config);
            if (!std::holds_alternative<DataMovementGen2Config>(dm_config)) {
                continue;
            }
            if (DmKernelDisablesImplicitSync(std::get<DataMovementGen2Config>(dm_config), dfb_spec->unique_id)) {
                disabled = true;
            }
        }
        return any_dm && !disabled;
    };
    return experimental::dfb::DataflowBufferConfig{
        .entry_size = dfb_spec->entry_size,
        .num_entries = dfb_spec->num_entries,
        .producer_risc_mask = producer_risc_mask,
        .num_producers = static_cast<uint8_t>(producer->num_threads),
        .pap = producer_access_pattern,
        .consumer_risc_mask = consumer_risc_mask,
        .num_consumers = static_cast<uint8_t>(consumer->num_threads),
        .cap = consumer_access_pattern,
        .enable_producer_implicit_sync = side_implicit_sync_enabled(dfb_endpoint_info.producers),
        .enable_consumer_implicit_sync = side_implicit_sync_enabled(dfb_endpoint_info.consumers),
        .data_format = dfb_spec->data_format_metadata.value_or(tt::DataFormat::Invalid),
        .tile = dfb_spec->tile_format_metadata,
        .unpack_face_geometry = dfb_spec->unpack_face_geometry_metadata,
        .tensix_scope = tensix_scope,
        // DFB borrowed memory mode is declared at program creation time.
        // The actual backing memory L1 address is attached at runtime.
        .borrows_memory = dfb_spec->borrowed_from.has_value()};
}

// ----------------------------------------------------------------------------
// MakeKernelSource: Create a KernelSource from a KernelSpec
// ----------------------------------------------------------------------------

KernelSource MakeKernelSource(const KernelSpec& kernel_spec) {
    return std::visit(
        [&](const auto& src) -> KernelSource {
            using T = std::decay_t<decltype(src)>;
            if constexpr (std::is_same_v<T, std::filesystem::path>) {
                TT_FATAL(!src.empty(), "KernelSpec '{}' has empty source file path", kernel_spec.unique_id);
                return KernelSource(src.string(), KernelSource::SourceType::FILE_PATH);
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
    const KernelSpec::CompileTimeArgs& bindings) {
    return std::unordered_map<std::string, uint32_t>(bindings.begin(), bindings.end());
}
std::map<std::string, std::string> to_defines_map(const KernelSpec::CompilerOptions::Defines& defines) {
    return std::map<std::string, std::string>(defines.begin(), defines.end());
}

DataMovementConfig MakeGen1DataMovementConfig(const KernelSpec& kernel_spec) {
    TT_FATAL(kernel_spec.is_data_movement_kernel(), "Expected a DM kernel");
    const auto& dm_config = std::get<DataMovementHardwareConfig>(kernel_spec.hw_config);
    const auto gen1 = std::get<DataMovementGen1Config>(dm_config);

    return DataMovementConfig{
        .processor = gen1.processor,
        .noc = gen1.noc,
        .noc_mode = gen1.noc_mode,
        .compile_args = {},  // only named_compile_args is used
        .defines = to_defines_map(kernel_spec.compiler_options.defines),
        .named_compile_args = to_named_compile_args_map(kernel_spec.compile_time_args),
        .opt_level = kernel_spec.compiler_options.opt_level,
        .compiler_include_paths = kernel_spec.compiler_options.include_paths,
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
    const ComputeUnpackModes& user_modes, const DFBNameToIdMap& dfb_name_to_id) {
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
        // Public UnpackMode -> internal UnpackToDestMode. UnpackToDest keeps full FP32 by
        // unpacking straight to Dest; UnpackToSrc is the SrcA/B path (the internal "Default").
        unpack_modes[dfb_id] =
            (mode == UnpackMode::UnpackToDest) ? UnpackToDestMode::UnpackToDestFp32 : UnpackToDestMode::Default;
    }
    return unpack_modes;
}

// ----------------------------------------------------------------------------
// MakeGen1ComputeConfig: Create a ComputeConfig (WH/BH) from a KernelSpec
// ----------------------------------------------------------------------------

ComputeConfig MakeGen1ComputeConfig(const KernelSpec& kernel_spec, const DFBNameToIdMap& dfb_name_to_id) {
    TT_FATAL(kernel_spec.is_compute_kernel(), "Expected a compute kernel");
    const auto& compute_config = std::get<ComputeHardwareConfig>(kernel_spec.hw_config);

    TT_FATAL(
        std::holds_alternative<ComputeGen1Config>(compute_config),
        "Trying to construct a Gen1 compute config but the kernel's ComputeHardwareConfig does not hold a "
        "ComputeGen1Config, generation mismatch, please provide the correctly typed hardware config.");
    const auto& gen1 = std::get<ComputeGen1Config>(compute_config);

    std::vector<UnpackToDestMode> unpack_dst_modes = BuildUnpackToDestModeVector(gen1.unpack_modes, dfb_name_to_id);

    return ComputeConfig{
        .math_fidelity = gen1.fpu_math_fidelity,
        .fp32_dest_acc_en = gen1.enable_32_bit_dest,
        .dst_full_sync_en = !gen1.double_buffer_dest,
        .unpack_to_dest_mode = unpack_dst_modes,
        .bfp8_pack_precise = (gen1.bfp_pack_precision_mode == Precision::Precise),
        .math_approx_mode = (gen1.sfpu_precision_mode == Precision::Approximate),
        .compile_args = {},  // only named_compile_args is used
        .defines = to_defines_map(kernel_spec.compiler_options.defines),
        .named_compile_args = to_named_compile_args_map(kernel_spec.compile_time_args),
        .opt_level = kernel_spec.compiler_options.opt_level,
        .compiler_include_paths = kernel_spec.compiler_options.include_paths,
    };
}

// ----------------------------------------------------------------------------
// MakeQuasarDataMovementConfig: Create a QuasarDataMovementConfig from a KernelSpec
// ----------------------------------------------------------------------------

experimental::quasar::QuasarDataMovementConfig MakeQuasarDataMovementConfig(const KernelSpec& kernel_spec) {
    TT_FATAL(kernel_spec.is_data_movement_kernel(), "Expected a DM kernel");

    return experimental::quasar::QuasarDataMovementConfig{
        .num_threads_per_cluster = kernel_spec.num_threads,
        .compile_args = {},  // only named_compile_args is used
        .defines = to_defines_map(kernel_spec.compiler_options.defines),
        .named_compile_args = to_named_compile_args_map(kernel_spec.compile_time_args),
        .is_legacy_kernel = false,
        .opt_level = kernel_spec.compiler_options.opt_level,
        .compiler_include_paths = kernel_spec.compiler_options.include_paths,
    };
}

// ----------------------------------------------------------------------------
// MakeGen2ComputeConfig: Create a QuasarComputeConfig from a KernelSpec
// ----------------------------------------------------------------------------

experimental::quasar::QuasarComputeConfig MakeGen2ComputeConfig(
    const KernelSpec& kernel_spec, const DFBNameToIdMap& dfb_name_to_id) {
    TT_FATAL(kernel_spec.is_compute_kernel(), "Expected a compute kernel");
    const auto& compute_config = std::get<ComputeHardwareConfig>(kernel_spec.hw_config);
    TT_FATAL(
        std::holds_alternative<ComputeGen2Config>(compute_config),
        "Trying to construct a Gen2 compute config but the kernel's ComputeHardwareConfig does not hold a "
        "ComputeGen2Config, generation mismatch, please provide the correctly typed hardware config.");
    const auto& gen2 = std::get<ComputeGen2Config>(compute_config);

    std::vector<UnpackToDestMode> unpack_dst_modes = BuildUnpackToDestModeVector(gen2.unpack_modes, dfb_name_to_id);

    return experimental::quasar::QuasarComputeConfig{
        .num_threads_per_cluster = kernel_spec.num_threads,
        .math_fidelity = gen2.fpu_math_fidelity,
        .fp32_dest_acc_en = gen2.enable_32_bit_dest,
        .dst_full_sync_en = !gen2.double_buffer_dest,
        .unpack_to_dest_mode = unpack_dst_modes,
        .math_approx_mode = (gen2.sfpu_precision_mode == Precision::Approximate),
        .enable_2x_src_format = gen2.enable_2x_src_register,
        .unpack_to_dest_en = gen2.unpack_to_dest_en,
        .compile_args = {},  // Compile args are passed via named_compile_args
        .defines = to_defines_map(kernel_spec.compiler_options.defines),
        .named_compile_args = to_named_compile_args_map(kernel_spec.compile_time_args),
        .opt_level = kernel_spec.compiler_options.opt_level,
        .compiler_include_paths = kernel_spec.compiler_options.include_paths,
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

Program MakeProgramFromSpec(const distributed::MeshDevice& mesh_device, const ProgramSpec& spec, bool skip_validation) {
    log_debug(tt::LogMetal, "Creating Program from ProgramSpec ({})", spec.name);

    // Step 1a: Collect derived data (builds lookup tables, checks structural invariants)
    CollectedSpecData collected = CollectSpecData(spec);

    // Step 1b: Validate semantic rules (can be skipped for trusted inputs)
    if (!skip_validation) {
        ValidateProgramSpec(spec, collected);
    }

    // Step 2a: Build kernel risc masks (arch-specific)
    //  - Gen2: backtracking solver assigns DM cores automatically
    //  - Gen1: processor is user-specified in Gen1Config
    KernelRiscMaskMap kernel_to_risc_mask =
        is_gen2_arch() ? SolveGen2KernelRiscMasks(spec, collected) : BuildGen1KernelRiscMasks(spec);

    // Step 2b: For multi-binding DFBs, all KernelSpecs on the same role must end up with
    // identical risc_masks. The DFB has a single producer_risc_mask / consumer_risc_mask in
    // its hardware config.
    //
    // Gen1: the mask is a deterministic function of the user's KernelSpec hw_config (compute
    //   placement is fixed; DM processor is user-specified via Gen1Config). A
    //   mismatch is a user error — incompatible processor placement across multi-bound kernels.
    // Gen2 (Quasar): the mask is solver-assigned, with the solver constrained to give every
    //   member of a DM coupling-group equivalence class the same DM mask. Compute kernel masks
    //   are deterministic from num_threads, which is uniform per role. So on Gen2 the uniformity
    //   property is guaranteed by construction; the check is retained as a defensive assertion.
    for (const auto& dfb : spec.dataflow_buffers) {
        const auto& endpoints = collected.dfb_endpoints.at(dfb.unique_id);
        auto check_uniform_mask = [&](const auto& records, std::string_view role) {
            if (records.size() < 2) {
                return;
            }
            const uint16_t first_mask = kernel_to_risc_mask.at(records[0].kernel);
            const auto* first_kernel = records[0].kernel;
            for (size_t i = 1; i < records.size(); ++i) {
                const uint16_t mask = kernel_to_risc_mask.at(records[i].kernel);
                if (mask == first_mask) {
                    continue;
                }
                if (is_gen2_arch()) {
                    TT_THROW(
                        "Internal error: Gen2 solver produced disagreeing risc_masks for DFB '{}' "
                        "{} bindings ('{}' = 0x{:x} vs '{}' = 0x{:x}). The coupling-group solver "
                        "extension should guarantee per-role mask uniformity by construction.",
                        dfb.unique_id,
                        role,
                        first_kernel->unique_id,
                        first_mask,
                        records[i].kernel->unique_id,
                        mask);
                } else {
                    TT_FATAL(
                        false,
                        "DFB '{}' has multiple {} KernelSpecs ('{}', '{}') with mismatched "
                        "processor placement (risc_mask 0x{:x} vs 0x{:x}). Multi-binding "
                        "requires all same-role kernels to share processor placement (for DM "
                        "kernels, check Gen1Config::processor; for compute, the "
                        "placement is determined by the KernelSpec's config type).",
                        dfb.unique_id,
                        role,
                        first_kernel->unique_id,
                        records[i].kernel->unique_id,
                        first_mask,
                        mask);
                }
            }
        };
        check_uniform_mask(endpoints.producers, "PRODUCER");
        check_uniform_mask(endpoints.consumers, "CONSUMER");
    }

    // Step 2c: Resolve TensorParameters against the MeshDevice into static CTA payloads.
    //
    // TensorBindings ride two existing kernel-arg channels:
    //   - Static layout (rank, shape, bank coords, ...) flows through the kernel's positional
    //     CTA buffer. (Empty in Metal 2.0 today; the binding payload is its first user.)
    //   - Per-enqueue base address flows through a reserved-prefix named CRTA, appended to
    //     the kernel's user-named CRTAs and filled by SetProgramRunArgs from the
    //     corresponding TensorArgument entry. TensorParameters that opt into a dynamic accessor
    //     field (currently: dynamic_tensor_shape, sharded only) carry additional CRTA words
    //     immediately after the address slot, also filled at enqueue time.
    std::unordered_map<TensorParamName, ResolvedTensorParameter> resolved_tensor_parameters;
    resolved_tensor_parameters.reserve(spec.tensor_parameters.size());
    for (const auto& tensor_parameter : spec.tensor_parameters) {
        resolved_tensor_parameters.emplace(
            tensor_parameter.unique_id, ResolveTensorParameterStaticCTAs(tensor_parameter, mesh_device));
    }

    // Step 3: Build the Program
    auto program_impl = std::make_shared<detail::ProgramImpl>();
    program_impl->set_program_spec_name(spec.program_id);

    // Register TensorParameters with the program for ValidateProgramRunArgs to consult at enqueue.
    for (const auto& tensor_parameter : spec.tensor_parameters) {
        const bool dyn_shape = tensor_parameter.advanced_options.dynamic_tensor_shape;
        const bool match_padded_only = tensor_parameter.advanced_options.match_padded_shape_only;
        const bool enqueue_invariant = tensor_parameter.advanced_options.enqueue_invariant;
        program_impl->register_tensor_parameter(
            tensor_parameter.unique_id.get(), tensor_parameter.spec, dyn_shape, match_padded_only, enqueue_invariant);
    }

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
        // (For borrowed-memory DFBs, config.borrows_memory was set in MakeDataflowBufferConfig;
        // the device-side runtime uses that to skip regular L1 allocation.)
        uint32_t dfb_id = program_impl->add_dataflow_buffer(collected.dfb_node_set.at(dfb_name), config);
        program_impl->register_dfb_spec_name(dfb_name.get(), dfb_id);
        dfb_name_to_id[dfb_name] = dfb_id;

        // Borrowed-memory DFB: record the dfb_id ↔ TensorParamName binding so that
        // SetProgramRunArgs / UpdateTensorArgs can resolve and attach the actual L1 Buffer
        // at runtime (analog of dynamic CB's UpdateDynamicCircularBufferAddress).
        if (dfb_spec.borrowed_from.has_value()) {
            program_impl->register_dfb_borrowed_binding(dfb_id, dfb_spec.borrowed_from->get());
        }
    }

    // Wire alias groups: for each DFB that has alias_with entries, make the first
    // encountered DFB in the group the primary and call set_dfb_alias for each secondary.
    // handled_as_secondary prevents a DFB from being treated as a primary when it was
    // already registered as a secondary by an earlier DFB in the group. Soundness relies
    // on the strict-clique invariant enforced by ValidateProgramSpec: every group member
    // lists every other member, so the primary's alias_with covers the whole group.
    {
        std::unordered_set<DFBSpecName> handled_as_secondary;
        for (const auto& dfb_spec : spec.dataflow_buffers) {
            if (handled_as_secondary.contains(dfb_spec.unique_id)) {
                continue;
            }
            if (dfb_alias_with(dfb_spec).empty()) {
                continue;
            }
            const uint32_t primary_id = dfb_name_to_id.at(dfb_spec.unique_id);
            for (const auto& alias_name : dfb_alias_with(dfb_spec)) {
                if (handled_as_secondary.contains(alias_name)) {
                    continue;
                }
                const uint32_t secondary_id = dfb_name_to_id.at(alias_name);
                program_impl->set_dfb_alias(primary_id, secondary_id);
                handled_as_secondary.insert(alias_name);
            }
        }
    }

    // Create Semaphores and build name -> ID map.
    // NOTE: Iterate over spec.semaphores to preserve user-provided deterministic ordering.
    SemaphoreNameToIdMap semaphore_name_to_id;
    for (const auto& semaphore_spec : spec.semaphores) {
        const SemaphoreSpecName& semaphore_name = semaphore_spec.unique_id;
        const uint32_t init_value = semaphore_spec.advanced_options.initial_value;
        uint32_t sem_id = program_impl->create_semaphore(
            to_node_range_set(semaphore_spec.target_nodes), init_value, CoreType::WORKER);
        program_impl->register_semaphore_spec_name(semaphore_name.get(), sem_id);
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

        // Resolve TensorBindings for this kernel:
        //  - pack each binding's pre-resolved CTA payload into the kernel's positional CTA buffer
        //  - assign each binding a slot in the kernel's CRTA buffer (TensorBinding address section)
        const auto& user_named_crtas = kernel_spec.runtime_arg_schema.common_runtime_arg_names;
        TensorBindingsForKernel ta_bindings = ResolveTensorBindingsForKernel(
            kernel_spec, resolved_tensor_parameters, /*base_named_crta_count=*/user_named_crtas.size());

        // Create TensorBindingHandles for this kernel
        const std::vector<TensorBindingHandle>& tensor_binding_handles = ta_bindings.handles;

        // Resolve scratchpad bindings for this kernel. The scratchpad section follows the TensorBinding
        // section and precedes varargs, so it begins at the tensor-binding resolution's (pre-scratchpad)
        // vararg offset; we then push the vararg section out by the scratchpad section's width so the
        // crta_layout that flows into the kernel ctor reflects all four sections.
        ScratchpadBindingsForKernel sp_bindings = ResolveScratchpadBindingsForKernel(
            kernel_spec,
            collected.scratchpad_by_name,
            /*scratchpad_base_crta_word=*/ta_bindings.crta_layout.vararg_section_offset);
        ta_bindings.crta_layout.scratchpad_section_words = sp_bindings.section_words;
        ta_bindings.crta_layout.vararg_section_offset += sp_bindings.section_words;

        // Named-args schema fields passed to the Kernel ctor. The names are used at JIT time to
        // emit kernel_args_generated.h and factor into the kernel cache key. The TensorBinding
        // address section is tracked separately (via tensor_binding_handles), so we pass the user
        // CRTA list through unchanged.
        const auto& named_rtas = kernel_spec.runtime_arg_schema.runtime_arg_names;

        // Create the kernel object
        std::shared_ptr<Kernel> kernel;

        // Kernel creation APIs accept a "is_metal2_kernel" bool, which fences Metal 2.0 JIT machinery
        constexpr bool is_metal2_kernel = true;

        if (is_gen2_arch()) {
            uint16_t risc_mask = kernel_to_risc_mask.at(&kernel_spec);
            if (kernel_spec.is_data_movement_kernel()) {
                auto config = MakeQuasarDataMovementConfig(kernel_spec);
                config.compile_args = ta_bindings.cta_words;  // populate positional CTAs from tensor bindings
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
                    user_named_crtas,
                    tensor_binding_handles,
                    ta_bindings.crta_layout);
            } else {
                auto config = MakeGen2ComputeConfig(kernel_spec, dfb_name_to_id);
                config.compile_args = ta_bindings.cta_words;
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
                    user_named_crtas,
                    tensor_binding_handles,
                    ta_bindings.crta_layout);
            }
        } else {  // gen1
            if (kernel_spec.is_data_movement_kernel()) {
                auto config = MakeGen1DataMovementConfig(kernel_spec);
                config.compile_args = ta_bindings.cta_words;
                kernel = std::make_shared<DataMovementKernel>(
                    kernel_src,
                    node_ranges,
                    config,
                    is_metal2_kernel,
                    dfb_handles,
                    semaphore_handles,
                    named_rtas,
                    user_named_crtas,
                    tensor_binding_handles,
                    ta_bindings.crta_layout);
            } else {
                auto config = MakeGen1ComputeConfig(kernel_spec, dfb_name_to_id);
                config.compile_args = ta_bindings.cta_words;
                kernel = std::make_shared<ComputeKernel>(
                    kernel_src,
                    node_ranges,
                    config,
                    is_metal2_kernel,
                    dfb_handles,
                    semaphore_handles,
                    named_rtas,
                    user_named_crtas,
                    tensor_binding_handles,
                    ta_bindings.crta_layout);
            }
        }

        // Attach the resolved scratchpad bindings to the kernel (set post-construction: their sizes are
        // part of the kernel cache key, so this must run before the kernel is compiled). allocate_scratchpads
        // will later fill each handle's allocated_address.
        kernel->set_scratchpad_binding_handles(std::move(sp_bindings.handles));

        // Add the kernel to the ProgramImpl and register the name -> handle mapping
        KernelHandle handle = program_impl->add_kernel(kernel, HalProgrammableCoreType::TENSIX);
        program_impl->register_kernel_spec_name(kernel_spec.unique_id.get(), handle);

        // Register the RTA+CRTA schema (named lists + vararg counts) with the ProgramImpl.
        // Used by ValidateProgramRunArgs and SetProgramRunArgs to validate and serialize
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
        const auto& user_schema = kernel_spec.runtime_arg_schema;
        detail::ProgramImpl::KernelRTASchema runtime_schema;
        runtime_schema.runtime_arg_names = user_schema.runtime_arg_names;

        // Pass the user CRTA list through.
        // NOTE: The TensorBinding address section is tracked separately on the Kernel
        // (via tensor_binding_handles) and its slot offsets are baked into each binding handle's
        // addr_crta_offset; SetProgramRunArgs uses BOTH to assemble the per-enqueue CRTA buffer.
        runtime_schema.common_runtime_arg_names = user_named_crtas;

        // Precompute the name -> slot-index maps so the hot UpdateProgramRunArgs path does O(1)
        // lookups rather than rebuilding a map per call.
        for (size_t i = 0; i < runtime_schema.runtime_arg_names.size(); ++i) {
            runtime_schema.runtime_arg_name_to_slot.emplace(runtime_schema.runtime_arg_names[i], i);
        }
        for (size_t i = 0; i < runtime_schema.common_runtime_arg_names.size(); ++i) {
            runtime_schema.common_runtime_arg_name_to_slot.emplace(runtime_schema.common_runtime_arg_names[i], i);
        }

        // Enqueue-loop-invariant named args. Each is a subset of the declared named RTAs/CRTAs and
        // may be omitted from a partial UpdateProgramRunArgs call (retaining its value). Legality
        // (each invariant name actually names a declared arg) is checked in ValidateProgramSpec.
        for (const auto& name : kernel_spec.advanced_options.enqueue_invariant_runtime_args) {
            runtime_schema.enqueue_invariant_runtime_arg_names.insert(name);
        }
        for (const auto& name : kernel_spec.advanced_options.enqueue_invariant_common_runtime_args) {
            runtime_schema.enqueue_invariant_common_runtime_arg_names.insert(name);
        }

        // Varargs schema now lives on KernelAdvancedOptions.
        const uint32_t num_runtime_varargs = kernel_spec.advanced_options.num_runtime_varargs;
        const uint32_t num_common_runtime_varargs = kernel_spec.advanced_options.num_common_runtime_varargs;
        const bool has_per_node_override = !kernel_spec.advanced_options.num_runtime_varargs_per_node.empty();

        if (num_runtime_varargs > 0) {
            for (const NodeRange& range : node_ranges.ranges()) {
                for (const NodeCoord& node : range) {
                    runtime_schema.num_runtime_varargs_per_node[node] = num_runtime_varargs;
                }
            }
        }
        if (has_per_node_override) {
            std::unordered_set<NodeCoord> seen_overrides;
            for (const auto& [nodes_spec, num_varargs] : kernel_spec.advanced_options.num_runtime_varargs_per_node) {
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
        runtime_schema.num_common_runtime_varargs = num_common_runtime_varargs;
        program_impl->register_kernel_rta_schema(kernel_spec.unique_id.get(), runtime_schema);
    }

    return Program(std::move(program_impl));
}

}  // namespace tt::tt_metal::experimental
