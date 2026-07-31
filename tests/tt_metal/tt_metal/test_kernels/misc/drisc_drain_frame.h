// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Wire format for the DRISC -> host profiler drain, shared by the DRISC kernel and the host decoder.
//
// This header exists so the two sides cannot drift. The X280 readers rotted precisely that way: each
// hardcoded the control-vector offsets it needed, the layout moved underneath them, and every copy was
// self-consistent while being wrong (see the banners in tools/x280_bm/src/prof*.c). One definition,
// included by both.
//
// A page is a flat sequence of frames:
//
//   +--------------------------------+
//   | w0: kind | lane | nwords       |   DATA frame
//   | w1: core_xy = (y << 16) | x    |
//   | nwords of raw marker payload   |
//   +--------------------------------+
//   | w0: kind = PAD                 |   rest of the page is padding; stop parsing
//   +--------------------------------+
//
// Frames never straddle a page, so the host can decode each page independently and a dropped page
// costs exactly that page. The sender pads whenever the next frame would not fit.
//
// `nwords` is a run of contiguous ring words for one (core, lane). The sender splits reads at the ring
// wrap, so the payload here is already linear -- the host never needs the ring geometry.

#pragma once

#include <cstdint>

namespace drisc_drain {

constexpr uint32_t FRAME_HEADER_WORDS = 2;

constexpr uint32_t KIND_PAD = 0;
constexpr uint32_t KIND_DATA = 1;

constexpr uint32_t FRAME_KIND_SHIFT = 28;
constexpr uint32_t FRAME_LANE_SHIFT = 24;
constexpr uint32_t FRAME_LANE_MASK = 0xFu;
constexpr uint32_t FRAME_NWORDS_MASK = 0xFFFFFFu;

constexpr uint32_t frame_w0(uint32_t kind, uint32_t lane, uint32_t nwords) {
    return (kind << FRAME_KIND_SHIFT) | ((lane & FRAME_LANE_MASK) << FRAME_LANE_SHIFT) | (nwords & FRAME_NWORDS_MASK);
}

constexpr uint32_t frame_kind(uint32_t w0) { return w0 >> FRAME_KIND_SHIFT; }
constexpr uint32_t frame_lane(uint32_t w0) { return (w0 >> FRAME_LANE_SHIFT) & FRAME_LANE_MASK; }
constexpr uint32_t frame_nwords(uint32_t w0) { return w0 & FRAME_NWORDS_MASK; }

}  // namespace drisc_drain
