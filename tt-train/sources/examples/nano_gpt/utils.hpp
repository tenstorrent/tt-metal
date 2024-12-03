// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>

#include "autograd/tensor.hpp"
#include "serialization/msgpack_file.hpp"
#include "serialization/serialization.hpp"

class LossAverageMeter {
    float m_sum = 0.0F;
    size_t m_count = 0;

public:
    void update(float loss, size_t count = 1);

    [[nodiscard]] float average() const;

    void reset();
};

std::string read_file_to_str(const std::string &file_path);

template <typename Model, typename Optimizer>
void save_model_and_optimizer(
    std::string &model_path,
    const std::shared_ptr<Model> &model,
    Optimizer &optimizer,
    const std::string &model_name,
    const std::string &optimizer_name) {
    ttml::serialization::MsgPackFile serializer;
    ttml::serialization::write_module(serializer, model_name, model.get());
    ttml::serialization::write_optimizer(serializer, optimizer_name, &optimizer);
    serializer.serialize(model_path);
}

template <typename Model, typename Optimizer>
void load_model_and_optimizer(
    std::string &model_path,
    const std::shared_ptr<Model> &model,
    Optimizer &optimizer,
    const std::string &model_name,
    const std::string &optimizer_name) {
    ttml::serialization::MsgPackFile deserializer;
    deserializer.deserialize(model_path);
    ttml::serialization::read_module(deserializer, model_name, model.get());
    ttml::serialization::read_optimizer(deserializer, optimizer_name, &optimizer);
}

uint32_t round_up_to_tile(uint32_t value, uint32_t tile_size = 32);

class GradientAccumulator {
public:
    explicit GradientAccumulator(uint32_t accumulation_steps);

    [[nodiscard]] bool should_zero_grad() const;
    [[nodiscard]] bool should_step() const;
    ttml::autograd::TensorPtr scale(ttml::autograd::TensorPtr &tensor_ptr);
    void update(float loss, size_t samples = 1);
    void reset();

    [[nodiscard]] float average_loss() const;

private:
    uint32_t m_accumulation_steps = 1;
    uint32_t m_steps = 0;

    float m_total_loss = 0.0F;
    size_t m_total_samples = 0;
};
