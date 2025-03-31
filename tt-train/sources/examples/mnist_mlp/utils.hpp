// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <fmt/core.h>

#include <cassert>
#include <chrono>
#include <cstddef>
#include <string>
#include <unordered_map>

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

class Timers {
public:
    void start(const std::string_view &name);

    long long stop(const std::string_view &name);

private:
    std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> m_timers;
};

template <typename Model, typename Optimizer>
void save_training_state(
    const std::string &model_path,
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
void load_training_state(
    const std::string &model_path,
    const std::shared_ptr<Model> &model,
    Optimizer &optimizer,
    const std::string &model_name,
    const std::string &optimizer_name) {
    ttml::serialization::MsgPackFile deserializer;
    deserializer.deserialize(model_path);
    ttml::serialization::read_module(deserializer, model_name, model.get());
    ttml::serialization::read_optimizer(deserializer, optimizer_name, &optimizer);
}
