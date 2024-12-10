// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "utils.hpp"

#include "autograd/tensor.hpp"
#include "ops/binary_ops.hpp"

void LossAverageMeter::update(float loss, size_t count) {
    m_sum += loss * static_cast<float>(count);
    m_count += count;
}

float LossAverageMeter::average() const {
    if (m_count == 0) {
        return 0.F;
    }
    return m_sum / static_cast<float>(m_count);
}

void LossAverageMeter::reset() {
    m_sum = 0.0F;
    m_count = 0;
}

std::string read_file_to_str(const std::string &file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + file_path);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

uint32_t round_up_to_tile(uint32_t value, uint32_t tile_size) {
    return (value + tile_size - 1) / tile_size * tile_size;
}

GradientAccumulator::GradientAccumulator(uint32_t accumulation_steps) : m_accumulation_steps(accumulation_steps) {
}

bool GradientAccumulator::should_zero_grad() const {
    return m_steps % m_accumulation_steps == 0;
}

bool GradientAccumulator::should_step() const {
    return m_steps % m_accumulation_steps == 0;
}

ttml::autograd::TensorPtr GradientAccumulator::scale(ttml::autograd::TensorPtr &tensor) {
    if (m_accumulation_steps > 1) {
        return ttml::ops::mul(tensor, 1.0F / static_cast<float>(m_accumulation_steps));
    }

    return tensor;
}

void GradientAccumulator::update(float loss, size_t samples) {
    m_total_loss += loss * samples * static_cast<float>(m_accumulation_steps);
    m_total_samples += samples;
    ++m_steps;
}

void GradientAccumulator::reset() {
    m_total_loss = 0.0F;
    m_total_samples = 0;
    m_steps = 0;
}

float GradientAccumulator::average_loss() const {
    return m_total_loss / static_cast<float>(m_total_samples);
}
