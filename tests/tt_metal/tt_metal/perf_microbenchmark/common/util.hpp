/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <vector>

#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/host_api.hpp"
#include "debug/dprint_buffer.h"
#include "llrt/hal.hpp"

inline uint64_t get_t0_to_any_riscfw_end_cycle(tt::tt_metal::Device *device, const tt::tt_metal::Program &program) {
#if defined(TRACY_ENABLE)
    // TODO: use enums from profiler_common.h
    enum BufferIndex { BUFFER_END_INDEX, DROPPED_MARKER_COUNTER, MARKER_DATA_START };
    enum TimerDataIndex { TIMER_ID, TIMER_VAL_L, TIMER_VAL_H, TIMER_DATA_UINT32_SIZE };
    auto worker_cores_used_in_program =
        device->worker_cores_from_logical_cores(program.logical_cores()[hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX)]);
    auto device_id = device->id();
    uint64_t min_cycle = -1;
    uint64_t max_cycle = 0;
    dprint_buf_msg_t *dprint_msg =
        hal.get_dev_addr<dprint_buf_msg_t *>(HalProgrammableCoreType::TENSIX, HalMemAddrType::DPRINT);

    // This works for tensix only, will need to be updated for eth
    vector<uint64_t> print_buffer_addrs = {
        reinterpret_cast<uint64_t>(dprint_msg->data[DPRINT_RISCV_INDEX_NC]),
        reinterpret_cast<uint64_t>(dprint_msg->data[DPRINT_RISCV_INDEX_BR]),
        reinterpret_cast<uint64_t>(dprint_msg->data[DPRINT_RISCV_INDEX_TR0]),
        reinterpret_cast<uint64_t>(dprint_msg->data[DPRINT_RISCV_INDEX_TR1]),
        reinterpret_cast<uint64_t>(dprint_msg->data[DPRINT_RISCV_INDEX_TR2]),
    };
    for (const auto &worker_core : worker_cores_used_in_program) {
        for (const auto &buffer_addr : print_buffer_addrs) {
            vector<std::uint32_t> profile_buffer;
            uint32_t end_index;
            uint32_t dropped_marker_counter;
            profile_buffer = tt::llrt::read_hex_vec_from_core(device_id, worker_core, buffer_addr, DPRINT_BUFFER_SIZE);

            end_index = profile_buffer[BUFFER_END_INDEX];

            TT_ASSERT(end_index < (DPRINT_BUFFER_SIZE / sizeof(uint32_t)));
            dropped_marker_counter = profile_buffer[DROPPED_MARKER_COUNTER];

            uint32_t step = (end_index - MARKER_DATA_START) / TIMER_DATA_UINT32_SIZE;
            uint32_t timer_id = 1;
            for (int i = MARKER_DATA_START; i < end_index; i += TIMER_DATA_UINT32_SIZE, timer_id++) {
                uint64_t cycle =
                    ((static_cast<uint64_t>(profile_buffer[i + TIMER_VAL_H]) << 32) | profile_buffer[i + TIMER_VAL_L]);

                if (timer_id == 1 && cycle < min_cycle) {
                    min_cycle = cycle;
                }

                if (timer_id == step && cycle > max_cycle) {
                    max_cycle = cycle;
                }
            }
        }
    }

    uint64_t t0_to_any_riscfw_end = max_cycle - min_cycle;
#else
    uint64_t t0_to_any_riscfw_end = 0;
#endif

    return t0_to_any_riscfw_end;
}

inline int get_tt_npu_clock(tt::tt_metal::Device *device) {
    return tt::Cluster::instance().get_device_aiclk(device->id());
}

template <typename T>
inline T calculate_average(const std::vector<T> &vec, bool skip_first_run = true) {
    if (vec.empty()) {
        return static_cast<T>(0);
    }

    int index = (skip_first_run && vec.size() != 1) ? (1) : (0);
    T sum = std::accumulate(vec.begin() + index, vec.end(), static_cast<T>(0));
    T average = sum / (vec.size() - index);
    return average;
}

enum class NOC_INDEX { NOC_RISCV_0, NOC_RISCV_1 };

enum class NOC_DIRECTION { X_PLUS_DIR, Y_MINUS_DIR, X_MINUS_DIR, Y_PLUS_DIR };

enum class ACCESS_TYPE { READ, WRITE };

enum class BUFFER_TYPE { DRAM, L1 };

std::string NOC_INDEXToString(NOC_INDEX enumValue) {
    switch (enumValue) {
        case NOC_INDEX::NOC_RISCV_0: return "NOC_RISCV_0";
        case NOC_INDEX::NOC_RISCV_1: return "NOC_RISCV_1";
        default: return "Unknown";
    }
}

std::string NOC_DIRECTIONToString(NOC_DIRECTION enumValue) {
    switch (enumValue) {
        case NOC_DIRECTION::X_PLUS_DIR: return "X_PLUS_DIR";
        case NOC_DIRECTION::Y_MINUS_DIR: return "Y_MINUS_DIR";
        case NOC_DIRECTION::X_MINUS_DIR: return "X_MINUS_DIR";
        case NOC_DIRECTION::Y_PLUS_DIR: return "Y_PLUS_DIR";
        default: return "Unknown";
    }
}

std::string ACCESS_TYPEToString(ACCESS_TYPE enumValue) {
    switch (enumValue) {
        case ACCESS_TYPE::READ: return "READ";
        case ACCESS_TYPE::WRITE: return "WRITE";
        default: return "Unknown";
    }
}

std::string BUFFER_TYPEToString(BUFFER_TYPE enumValue) {
    switch (enumValue) {
        case BUFFER_TYPE::DRAM: return "DRAM";
        case BUFFER_TYPE::L1: return "L1";
        default: return "Unknown";
    }
}
