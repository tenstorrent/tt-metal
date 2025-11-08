// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <atomic>

namespace tt::tt_metal {

class CoreIDs {
public:
    // static CoreIDs& instance();

    // std::int64_t get_python_operation_id();
    // void set_python_operation_id(std::int64_t operation_id);
    // std::int64_t fetch_and_increment_python_operation_id();

    // std::int64_t get_tensor_id();
    // void set_tensor_id(std::int64_t tensor_id);
    // std::int64_t fetch_and_increment_tensor_id();

    // std::int64_t get_device_operation_id();
    // void set_device_operation_id(std::int64_t device_operation_id);
    // std::int64_t fetch_and_increment_device_operation_id();

    static CoreIDs& instance() {
        static CoreIDs instance;
        return instance;
    }

    std::int64_t get_python_operation_id() { return python_operation_id.load(); }
    void set_python_operation_id(std::int64_t python_operation_id_) { python_operation_id = python_operation_id_; }
    std::int64_t fetch_and_increment_python_operation_id() { return python_operation_id.fetch_add(1); }

    std::int64_t get_tensor_id() { return tensor_id.load(); }
    void set_tensor_id(std::int64_t tensor_id_) { tensor_id = tensor_id_; }
    std::int64_t fetch_and_increment_tensor_id() { return tensor_id.fetch_add(1); }

    std::int64_t get_device_operation_id() { return device_operation_id.load(); }
    void set_device_operation_id(std::int64_t device_operation_id_) { device_operation_id = device_operation_id_; }
    std::int64_t fetch_and_increment_device_operation_id() { return device_operation_id.fetch_add(1); }

private:
    CoreIDs() = default;
    ~CoreIDs() = default;
    std::atomic<std::int64_t> tensor_id;
    std::atomic<std::int64_t> python_operation_id;
    std::atomic<std::int64_t> device_operation_id = 1;
};
}  // namespace tt::tt_metal
