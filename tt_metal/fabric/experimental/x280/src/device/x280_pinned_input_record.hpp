// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The pinned-input record the runtime keeps in each tile's general-purpose
// scratch, and the pure decisions prepare_pinned_input makes over it. Private to
// the component.

#pragma once

#include "x280/device/x280_tile.hpp"
#include "x280_hw_addrs.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace x280 {

// Word layout. Words from kPinnedInputWordFirstReserved on are reserved and zero.
inline constexpr size_t kPinnedInputWordMagic = 0;
inline constexpr size_t kPinnedInputWordStatus = 1;
inline constexpr size_t kPinnedInputWordWayEnable = 2;
inline constexpr size_t kPinnedInputWordBaseLo = 3;
inline constexpr size_t kPinnedInputWordBaseHi = 4;
inline constexpr size_t kPinnedInputWordBytesLo = 5;
inline constexpr size_t kPinnedInputWordBytesHi = 6;
inline constexpr size_t kPinnedInputWordFirstReserved = 7;

/** "PIN" and record version 1. */
inline constexpr uint32_t kPinnedInputRecordMagic = 0x50494E01;
inline constexpr uint32_t kPinnedInputRecordAborted = 1;
inline constexpr uint32_t kPinnedInputRecordPrepared = 2;

using PinnedInputRecord = decltype(PinnedInputState::words);

static_assert(
    sizeof(PinnedInputRecord) <= kL2cpuScratchBytes, "the record must fit the L2CPU's general-purpose scratch");

/**
 * The record for `status`, which must be Aborted or Prepared, at the fixed
 * pinned-input region and X280_PINNED_INPUT_WAYENABLE. Throws for any other
 * status.
 */
PinnedInputRecord encode_pinned_input_record(PinnedInputStatus status);

/** Classifies sixteen words as read. Every word all ones is InvalidObservation;
 * all zero is None; any other all-ones word is Corrupt. */
PinnedInputState decode_pinned_input_record(const PinnedInputRecord& words);

/**
 * The refusal prepare_pinned_input returns before writing, or nullopt to
 * proceed from None. Checked in order: an all-ones `raw_wayenable`, the shape,
 * `session_live`, then the record.
 */
std::optional<PinnedInputPrepareOutcome> refuse_pinned_input_prepare(
    uint8_t runtime_lim_ways, uint32_t raw_wayenable, bool session_live, const PinnedInputState& state);

/**
 * The record a prepare commits once `result`'s run is over: None when the
 * submission attached no session, so the image never ran; Prepared when the
 * wait was ResultReady with hart 0 X280_PINNED_INPUT_RESULT_PREPARED and harts
 * 1-3 X280_PINNED_INPUT_RESULT_NOOP_HART, the cleanup returned to idle, and the
 * final state is a valid live resident idle; otherwise Aborted, including when
 * `result.enqueue` is unset.
 */
PinnedInputStatus pinned_input_status_after_run(const PinnedInputPrepareResult& result);

}  // namespace x280
