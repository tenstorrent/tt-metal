// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "utils.hpp"

#include "autograd/auto_context.hpp"
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

std::unique_ptr<ttml::schedulers::LRSchedulerBase> create_idendity_scheduler(
    ttml::optimizers::OptimizerBase *optimizer, [[maybe_unused]] size_t total_steps) {
    return std::make_unique<ttml::schedulers::LambdaScheduler>(optimizer, [](int epoch) { return 1.0F; });
}

std::unique_ptr<ttml::schedulers::LRSchedulerBase> create_warmup_with_linear_scheduler(
    ttml::optimizers::OptimizerBase *optimizer, size_t total_steps) {
    const float default_warmup_factor = 0.1F;
    const size_t warmup_steps = size_t(total_steps * default_warmup_factor);
    const size_t linear_decay_steps = total_steps - warmup_steps;

    std::vector<std::unique_ptr<ttml::schedulers::LRSchedulerBase>> schedulers;
    schedulers.push_back(std::make_unique<ttml::schedulers::LinearScheduler>(optimizer, 0.0F, 1.0F, warmup_steps));
    schedulers.push_back(
        std::make_unique<ttml::schedulers::LinearScheduler>(optimizer, 1.0F, 0.01F, linear_decay_steps));
    std::vector<size_t> steps = {warmup_steps, linear_decay_steps};
    return std::make_unique<ttml::schedulers::SequentialScheduler>(optimizer, std::move(schedulers), std::move(steps));
}

void initialize_device(const tt::tt_metal::distributed::MeshShape &mesh_shape, const std::vector<int> &device_ids) {
    ttml::autograd::ctx().open_device(mesh_shape, device_ids);
}
