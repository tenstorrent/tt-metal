// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>

namespace tt::tt_metal::experimental::metal2_host_api {

// Forward-declare *Name typedefs that AdvancedOptions members reference.
// Each is also declared in its owning spec header; C++ permits redeclaration
// of identical typedefs in the same namespace.
using DFBSpecName = std::string;

//------------------------------------------------------------
// Advanced options for Metal 2.0 specs
//------------------------------------------------------------
//
// Each Metal 2.0 Spec (KernelSpec, DataflowBufferSpec, TensorParameter, ...) may
// carry a std::optional<*AdvancedOptions> field at the end of the struct. The
// *AdvancedOptions struct holds members that meet ONE OR MORE of the following:
//
//   - Niche use case (most users will never set it).
//   - Not safe by construction (footgun disguised as a feature).
//   - Placeholder for feedback (unstable; may move or disappear based on real usage).
//   - Slated for removal (kept only because a replacement does not yet exist;
//     should carry [[deprecated]] with a message stating the removal plan).
//
// Members that are merely "advanced but mainstream" — production-ready features
// most users will not need but that work safely and predictably — stay on the
// main Spec, NOT here.
//
// The std::optional wrapper + explicit type name at the use site
// (e.g. `.advanced_options = KernelSpecAdvancedOptions{.foo = bar}`) is
// intentional: it puts a small ergonomic speed bump in front of reaching into
// this bucket on autopilot.
//
// (TODO: comments in this header to be revisited with Audrey after structural
// changes land.)

struct KernelSpecAdvancedOptions {
    // (Optional) Per-node thread count specification.
    // The default threading is KernelSpec::num_threads. However, you may override
    // this on a per-node basis.
    // NOTE: This feature is currently unsupported. It's an open question if we EVER
    //       want to support it. Here as a placeholder; specifying it will trigger a
    //       runtime error.
    using NodeSpecificThreadCount = std::pair<Nodes, int>;  // {node_set, num_threads}
    using NodeSpecificThreadCounts = std::vector<NodeSpecificThreadCount>;
    std::optional<NodeSpecificThreadCounts> node_specific_thread_counts = std::nullopt;
};

struct DataflowBufferSpecAdvancedOptions {
    // Alias two or more DFBs.
    // Aliased DFBs are logically distinct, but physically share the same backing memory.
    // Aliased DFBs offer NO guarantees against data clobbering; kernel logic must ensure safety.
    //
    // Rules for aliased DFBs:
    //   - Every DFB in the alias group must list every other member as an alias
    //   - Aliased DFBs must have the same total size (num_entries * entry_size).
    //   - All members must target the same node set
    //     (derived from their bound kernels' WorkUnitSpecs).
    std::vector<DFBSpecName> alias_with;
};

struct TensorParameterAdvancedOptions {
    // By default, the MeshTensor argument provided at execution time must
    // EXACTLY match the TensorParameter's declared TensorSpec. The options
    // below relax this match requirement in particular ways.
    //
    // NOTE: These options are UNSAFE if set to true; most kernels will not function
    // correctly if the tensor argument's spec deviates from the declared spec.
    // Use with caution and ensure that your kernel logic is compatible.

    // Permit tensor arguments whose logical_shape differs from the declared shape.
    // The argument's padded_shape must still match exactly.
    // Effects:
    //  - Validation checks are relaxed
    //  - TensorAccessor configuration is completely unchanged
    bool match_padded_shape_only = false;

    // Permit tensor arguments with dynamic logical shape.
    // The argument's logical_shape AND padded_shape may differ from the declared shape.
    // Effects:
    //  - Validation checks are relaxed
    //  - For an interleaved tensor, TensorAccessor configuration is unchanged
    //  - For a sharded tensor, the TensorAccessor configuration dynamically reflects the
    //    argument's actual shape. (Shape becomes an implicit runtime argument.)
    bool dynamic_tensor_shape = false;
};

}  // namespace tt::tt_metal::experimental::metal2_host_api
