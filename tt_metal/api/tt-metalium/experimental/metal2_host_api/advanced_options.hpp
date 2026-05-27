// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt::tt_metal::experimental::metal2_host_api {

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
    // No fields yet — populated as features migrate in.
};

struct DataflowBufferSpecAdvancedOptions {
    // No fields yet — populated as features migrate in.
};

struct TensorParameterAdvancedOptions {
    // No fields yet — populated as features migrate in.
};

}  // namespace tt::tt_metal::experimental::metal2_host_api
