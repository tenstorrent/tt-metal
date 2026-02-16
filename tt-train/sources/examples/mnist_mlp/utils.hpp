// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <fmt/core.h>

#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <unordered_map>

#include "serialization/flatbuffer_file.hpp"
#include "serialization/serialization.hpp"

// Expand ${TT_METAL_RUNTIME_ROOT} in a config path string.
// Falls back to inferring the value from the compile-time CONFIGS_FOLDER.
inline std::string expand_config_path(const std::string &path) {
    static const std::string kPlaceholder = "${TT_METAL_RUNTIME_ROOT}";
    auto pos = path.find(kPlaceholder);
    if (pos == std::string::npos) {
        return path;
    }
    std::string tt_metal_root;
    const char *env = std::getenv("TT_METAL_RUNTIME_ROOT");
    if (env != nullptr) {
        tt_metal_root = env;
    } else {
        tt_metal_root = std::filesystem::path(CONFIGS_FOLDER).parent_path().parent_path().string();
    }
    std::string result = path;
    result.replace(pos, kPlaceholder.length(), tt_metal_root);
    return result;
}

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
    ttml::serialization::FlatBufferFile serializer;
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
    ttml::serialization::FlatBufferFile deserializer;
    deserializer.deserialize(model_path);
    ttml::serialization::read_module(deserializer, model_name, model.get());
    ttml::serialization::read_optimizer(deserializer, optimizer_name, &optimizer);
}
