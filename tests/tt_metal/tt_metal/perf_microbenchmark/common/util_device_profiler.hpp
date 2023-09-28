/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"

inline uint64_t get_t0_to_any_riscfw_end_cycle(
    tt::tt_metal::Device *device, const tt::tt_metal::Program &program) {
#if defined(PROFILER)
  // TODO: use enums from profiler_common.h
  enum BufferIndex {
    BUFFER_END_INDEX,
    DROPPED_MARKER_COUNTER,
    MARKER_DATA_START
  };
  enum TimerDataIndex {
    TIMER_ID,
    TIMER_VAL_L,
    TIMER_VAL_H,
    TIMER_DATA_UINT32_SIZE
  };
  auto worker_cores_used_in_program =
      device->worker_cores_from_logical_cores(program.logical_cores());
  auto cluster = device->cluster();
  auto device_id = device->id();
  uint64_t min_cycle = -1;
  uint64_t max_cycle = 0;
  vector<uint32_t> print_buffer_addrs = {PRINT_BUFFER_NC, PRINT_BUFFER_BR,
                                         PRINT_BUFFER_T0, PRINT_BUFFER_T1,
                                         PRINT_BUFFER_T2};
  for (const auto &worker_core : worker_cores_used_in_program) {
    for (const auto &buffer_addr : print_buffer_addrs) {
      vector<std::uint32_t> profile_buffer;
      uint32_t end_index;
      uint32_t dropped_marker_counter;
      profile_buffer = tt::llrt::read_hex_vec_from_core(
          cluster, device_id, worker_core, buffer_addr, PRINT_BUFFER_SIZE);

      end_index = profile_buffer[BUFFER_END_INDEX];
      TT_ASSERT(end_index < (PRINT_BUFFER_SIZE / sizeof(uint32_t)));
      dropped_marker_counter = profile_buffer[DROPPED_MARKER_COUNTER];

      uint32_t timer_id = 1;
      for (int i = MARKER_DATA_START; i < end_index;
           i += TIMER_DATA_UINT32_SIZE, timer_id++) {
        uint64_t cycle =
            ((static_cast<uint64_t>(profile_buffer[i + TIMER_VAL_H]) << 32) |
             profile_buffer[i + TIMER_VAL_L]);

        if (cycle < min_cycle) {
          min_cycle = cycle;
        }

        if (timer_id == 4 && cycle > max_cycle) {
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
  auto cluster = device->cluster();
  auto device_id = device->id();
  int ai_clk = 0;
  if (cluster->type == tt::TargetDevice::Silicon) {
    ai_clk = cluster->get_device_aiclk(device->id());
  }
  return ai_clk;
}
