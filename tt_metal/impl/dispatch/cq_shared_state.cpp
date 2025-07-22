// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "cq_shared_state.hpp"

namespace tt::tt_metal {
void CQOwnerState::take_ownership(SubDeviceId id, uint32_t cq_id) {
    if (cq_id_ != std::nullopt && cq_id_ != cq_id) {
        TT_FATAL(
            cq_id_ == cq_id,
            "Sub device id {} currently in use by cq {}. Can't enqueue program from cq {}. Finish or wait "
            "for an event to transfer ownership.",
            *id,
            *cq_id_,
            cq_id);
    }
    cq_id_ = cq_id;
    ownership_event_id_ = std::nullopt;
}

void CQOwnerState::finished(uint32_t cq_id) {
    if (cq_id_.has_value() && cq_id_ == cq_id) {
        cq_id_ = std::nullopt;
        ownership_event_id_ = std::nullopt;
    }
}

void CQOwnerState::recorded_event(uint32_t event_id, uint32_t event_cq) {
    if (cq_id_.has_value() && cq_id_ == event_cq) {
        if (ownership_event_id_.has_value()) {
            TT_ASSERT(*ownership_event_id_ < event_id, "Ownership event ID must be less than the current event ID");
        } else {
            ownership_event_id_ = event_id;
        }
    }
}

void CQOwnerState::waited_for_event(uint32_t event_id, uint32_t event_cq, uint32_t cq_id) {
    if (cq_id_.has_value() && event_cq == cq_id_ && cq_id_ != cq_id && ownership_event_id_.has_value() &&
        ownership_event_id_ <= event_id) {
        cq_id_ = std::nullopt;
        ownership_event_id_ = std::nullopt;
    }
}
}  // namespace tt::tt_metal
