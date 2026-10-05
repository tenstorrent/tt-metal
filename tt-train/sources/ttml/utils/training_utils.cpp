// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "utils/training_utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <tt_stl/assert.hpp>

namespace ttml::utils {

double steps_per_epoch(size_t corpus_tokens, uint32_t global_batch_size, uint32_t sequence_length) {
    TT_FATAL(
        global_batch_size > 0U && sequence_length > 0U,
        "global_batch_size and sequence_length must be positive, got global_batch_size={} sequence_length={}",
        global_batch_size,
        sequence_length);
    const size_t tokens_per_step = static_cast<size_t>(global_batch_size) * sequence_length;
    return static_cast<double>(corpus_tokens) / static_cast<double>(tokens_per_step);
}

uint32_t resolve_effective_max_steps(uint32_t max_steps, uint32_t num_epochs, double steps_per_epoch) {
    TT_FATAL(
        max_steps > 0U || num_epochs > 0U,
        "No stop condition: set max_steps > 0 or num_epochs > 0 in training_config.");
    if (num_epochs == 0U) {
        return max_steps;
    }

    const double epoch_steps = std::max(1.0, std::ceil(static_cast<double>(num_epochs) * steps_per_epoch));
    if (max_steps > 0U && epoch_steps >= static_cast<double>(max_steps)) {
        return max_steps;
    }
    TT_FATAL(
        epoch_steps <= static_cast<double>(std::numeric_limits<uint32_t>::max()),
        "num_epochs={} at {} steps per epoch needs {} steps, which exceeds the uint32 step counter",
        num_epochs,
        steps_per_epoch,
        epoch_steps);
    return static_cast<uint32_t>(epoch_steps);
}

uint32_t epochs_completed(size_t step, double steps_per_epoch) {
    if (steps_per_epoch <= 0.0) {
        return 0U;
    }
    const double completed = static_cast<double>(step) / steps_per_epoch;
    if (completed >= static_cast<double>(std::numeric_limits<uint32_t>::max())) {
        return std::numeric_limits<uint32_t>::max();
    }
    return static_cast<uint32_t>(completed);
}

}  // namespace ttml::utils
