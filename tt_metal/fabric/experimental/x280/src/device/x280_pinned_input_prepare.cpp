// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "x280_tile_impl.hpp"

#include "x280_format.hpp"
#include "x280_hw_addrs.hpp"

#include "rt/x280_pinned_input.h"

#include <exception>
#include <string>

namespace x280::detail {

namespace {

constexpr uint64_t kPrepareRunValue = uint64_t{kPinnedInputRecordMagic} << 32;

}  // namespace

PinnedInputRecord TileImpl::read_pinned_input_record_unlocked() const {
    PinnedInputRecord words{};
    for (size_t i = 0; i < words.size(); ++i) {
        words[i] = transport().read_reg32(id().chip_id, core(), kL2cpuScratchBase + i * sizeof(uint32_t));
    }
    return words;
}

void TileImpl::write_pinned_input_words_unlocked(const PinnedInputRecord& words, size_t first, size_t end) {
    for (size_t i = first; i < end; ++i) {
        transport().write_reg32(id().chip_id, core(), kL2cpuScratchBase + i * sizeof(uint32_t), words[i]);
    }
}

PinnedInputState TileImpl::pinned_input_state() const {
    const std::lock_guard<std::mutex> pinned(pinned_input_mutex_);
    runtime_->require_mesh_live("pinned_input_state", &id());
    const std::lock_guard<std::mutex> serialise(chip_->lock());
    return decode_pinned_input_record(read_pinned_input_record_unlocked());
}

PinnedInputPrepareResult TileImpl::prepare_pinned_input(const PinnedInputPrepareRequest& request) {
    PinnedInputPrepareResult result;
    result.wait_timeout = request.wait_timeout;
    result.cleanup_timeout = request.cleanup_timeout;
    if (request.wait_timeout.count() <= 0 || request.cleanup_timeout.count() <= 0) {
        result.outcome = PinnedInputPrepareOutcome::InvalidRequest;
        return result;
    }

    const std::lock_guard<std::mutex> pinned(pinned_input_mutex_);
    try {
        runtime_->require_mesh_live("prepare_pinned_input", &id());
        const auto aborted = encode_pinned_input_record(PinnedInputStatus::Aborted);
        {
            const std::lock_guard<std::mutex> serialise(chip_->lock());
            result.runtime_lim_ways = runtime_->lim_shape().lim_ways;
            result.raw_wayenable = transport().read_reg32(id().chip_id, core(), kL3CacheWayEnable);
            result.state = decode_pinned_input_record(read_pinned_input_record_unlocked());
            const auto refusal = refuse_pinned_input_prepare(
                result.runtime_lim_ways, result.raw_wayenable, !runtime_->sessions().empty(), result.state);
            if (refusal) {
                result.outcome = *refusal;
                return result;
            }

            // The record read all zero, so the reserved words need no write.
            write_pinned_input_words_unlocked(aborted, 0, kPinnedInputWordFirstReserved);
            result.state = decode_pinned_input_record(read_pinned_input_record_unlocked());
            if (result.state.words != aborted) {
                result.error = "the preparing record did not read back";
                return result;
            }
        }

        ActiveRunRequest run;
        run.image_path = request.image_path;
        run.run_value = kPrepareRunValue | static_cast<uint64_t>(id().hw_index + 1);
        run.cleanup_timeout = request.cleanup_timeout;
        const auto& enqueued = result.enqueue.emplace(runtime_->enqueue(runtime_->tile(id()), run, /*exclusive=*/true));
        if (enqueued.session) {
            if (enqueued.enqueued()) {
                result.wait = runtime_->session(enqueued.session).wait_until_result_ready(request.wait_timeout);
            }
            result.cleanup = runtime_->cleanup(enqueued.session, request.cleanup_timeout);
            result.final_state = inspect();
        }

        const PinnedInputStatus next = pinned_input_status_after_run(result);
        if (next == PinnedInputStatus::Aborted) {
            return result;
        }
        const std::lock_guard<std::mutex> serialise(chip_->lock());
        PinnedInputRecord expected{};
        if (next == PinnedInputStatus::Prepared) {
            // One word moves Aborted to Prepared, so no torn write reads as Prepared.
            expected = encode_pinned_input_record(PinnedInputStatus::Prepared);
            write_pinned_input_words_unlocked(expected, kPinnedInputWordStatus, kPinnedInputWordStatus + 1);
        } else {
            write_pinned_input_words_unlocked(expected, 0, kPinnedInputRecordWords);
        }
        result.state = decode_pinned_input_record(read_pinned_input_record_unlocked());
        if (result.state.words != expected) {
            result.error = std::string("the ") + to_string(next) + " record did not read back";
        } else if (next == PinnedInputStatus::Prepared) {
            result.outcome = PinnedInputPrepareOutcome::Prepared;
        } else if (enqueued.outcome == EnqueueOutcome::Busy) {
            // Admission found a session the snapshot above missed.
            result.outcome = PinnedInputPrepareOutcome::Busy;
        }
    } catch (const std::exception& error) {
        result.error = error.what();
    } catch (...) {
        result.error = "unknown pinned-input prepare failure";
    }
    return result;
}

}  // namespace x280::detail
