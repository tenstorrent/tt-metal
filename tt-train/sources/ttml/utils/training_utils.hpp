// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

namespace ttml::utils {

/**
 * @brief Optimizer steps needed for one pass over the corpus tokens.
 * @param corpus_tokens Number of tokens in the training corpus.
 * @param global_batch_size Samples consumed per optimizer step (batch size x gradient accumulation steps).
 * @param sequence_length Tokens per sample.
 * @return corpus_tokens / (global_batch_size * sequence_length); fractional values are preserved.
 */
[[nodiscard]] double steps_per_epoch(size_t corpus_tokens, uint32_t global_batch_size, uint32_t sequence_length);

/**
 * @brief Run length in optimizer steps: whichever of the max_steps / num_epochs caps comes first.
 * @param max_steps Step cap; 0 disables it.
 * @param num_epochs Epoch cap; 0 disables it. A partial final epoch is rounded up to a full step.
 * @param steps_per_epoch Result of steps_per_epoch().
 * TT_FATAL if both caps are disabled, or if the epoch cap exceeds the uint32 step counter.
 */
[[nodiscard]] uint32_t resolve_effective_max_steps(uint32_t max_steps, uint32_t num_epochs, double steps_per_epoch);

/**
 * @brief Whole epochs finished after `step` optimizer steps.
 * @param step Optimizer steps completed so far.
 * @param steps_per_epoch Result of steps_per_epoch(). A non-positive value reports 0.
 * @return floor(step / steps_per_epoch).
 */
[[nodiscard]] uint32_t epochs_completed(size_t step, double steps_per_epoch);

}  // namespace ttml::utils
