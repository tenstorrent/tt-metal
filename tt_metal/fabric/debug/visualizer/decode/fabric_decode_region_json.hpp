// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// A region's `decoded` in the decoded file: its reads, each element interpreted, put into the region's shape. The
// shape is picked from the manifest, never from the data:
//
//   L1, one element        { at, value }
//   L1, several elements   { at, elements: [ ... ] }
//   L1 of byte             { at, size_bytes, nonzero_bytes }
//   Stream register        { at, value }, and upper_bits only when the word's bits above value_bits are set
//   Multi-sampled          its shape above for the full read, and rounds: [ { round, t_ns, value | elements } ]
//   Inactive               { active: false, at, size_bytes, nonzero_bytes }
//   Unread                 { unread: "<code>" }
//
// `at` is where the full read's bytes of the region start in the .bin. A multi-sampled region the full read didn't
// reach has only its rounds.

#pragma once

#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_interpret.hpp"
#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_values.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_model.hpp"

namespace tt::tt_fabric::decode {

// The `decoded` of the L1 region at context.path, from its reads in `values`. Fails when `values` has nothing at that
// path, since every region the router names has reads, inactive bytes or an unread reason.
json make_decoded_region_json(
    const RouterValues& values, const manifest::content::L1& region, const InterpretContext& context);

// The `decoded` of the stream register at context.path, from its reads in `values`. Fails as the L1 overload does.
json make_decoded_region_json(
    const RouterValues& values, const manifest::content::StreamReg& stream_register, const InterpretContext& context);

}  // namespace tt::tt_fabric::decode
