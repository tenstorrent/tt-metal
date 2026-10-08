// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string_view>
#include <type_traits>
#include <umd/device/types/arch.hpp>
#include <unordered_set>
#include <variant>

#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>
#include <tt-metalium/experimental/metal2_host_api/advanced_options.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/data_movement_hardware_config.hpp>

#include "llrt/hal.hpp"

namespace tt::tt_metal::experimental {

// TODO: Shouldn't be in helpers.
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
// Basic Utility Helpers
// ============================================================================

// TODO: This should be upstreamed.
inline NodeRangeSet to_node_range_set(const Nodes& nodes) {
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

// Local accessor names for kernel resource bindings must be valid C++ identifiers
// They are used verbatim in the generated kernel source code.
// TODO: Move this to ttsl in a follow up PR
inline bool IsValidCppIdentifier(std::string_view s) {
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

inline bool is_gen2_arch(tt::ARCH arch) { return arch == tt::ARCH::QUASAR; }

inline bool is_gen2_arch(const Hal& hal) { return is_gen2_arch(hal.get_arch()); }

inline bool is_gen1_arch(tt::ARCH arch) { return arch == tt::ARCH::WORMHOLE_B0 || arch == tt::ARCH::BLACKHOLE; }

inline bool is_gen1_arch(const Hal& hal) { return is_gen1_arch(hal.get_arch()); }

inline bool nodes_intersect(const Nodes& a, const Nodes& b) {
    NodeRangeSet a_set = to_node_range_set(a);
    NodeRangeSet b_set = to_node_range_set(b);
    return a_set.intersects(b_set);
}

// Set equality that is independent of how the two sets decompose into ranges (NodeRangeSet's
// operator== compares the range lists, so equal sets with different range splits compare unequal).
inline bool same_node_set(const NodeRangeSet& a, const NodeRangeSet& b) {
    return a.num_cores() == b.num_cores() && a.intersection(b).num_cores() == a.num_cores();
}

// TODO: Move this as prefetcher domain

// Role of a kernel binding a PrefetcherPipe accessor group, from its nodes and the group's receiver
// sets alone (the spec does not name senders; a pipe's sender is the pipe object's). The kernel is
// the group's receiver when its nodes are exactly the union of the receiver sets, and its sender
// when its nodes avoid every receiver and number one per pipe (which node hosts which pipe is
// settled when the pipes are supplied). ValidateProgramSpec and ReservePrefetcherPipeSlots both
// derive the role through these two definitions.
inline bool is_prefetcher_pipe_receiver_role(const NodeRangeSet& kernel_nodes, const NodeRangeSet& group_receivers) {
    return same_node_set(kernel_nodes, group_receivers);
}

inline bool is_prefetcher_pipe_sender_role(
    const NodeRangeSet& kernel_nodes, const NodeRangeSet& group_receivers, size_t num_pipes) {
    return !kernel_nodes.intersects(group_receivers) && kernel_nodes.num_cores() == num_pipes;
}

// Helper: return a DFB's alias-with list.
inline const std::vector<DFBSpecName>& dfb_alias_with(const DataflowBufferSpec& dfb) {
    return dfb.advanced_options.alias_with;
}

// Whether a DM kernel opts out of implicit sync for a particular DFB.
// Two routes lead to the same opt-out:
//   - disable_dfb_implicit_sync_for_all: the per-kernel hammer, covering every DFB the kernel binds.
//   - disable_dfb_implicit_sync_for: an explicit per-DFB list.
// If config_2xx is not engaged, implicit sync stays at its default (on for every bound DFB).
inline bool DmKernelDisablesImplicitSync(const DataMovementHardwareConfig& dm_config, const DFBSpecName& dfb_name) {
    if (!dm_config.config_2xx.has_value()) {
        return false;
    }
    const auto& gen2_config = *dm_config.config_2xx;
    if (gen2_config.disable_dfb_implicit_sync_for_all) {
        return true;
    }
    const auto& vec = gen2_config.disable_dfb_implicit_sync_for;
    return std::find(vec.begin(), vec.end(), dfb_name) != vec.end();
}

}  // namespace tt::tt_metal::experimental
