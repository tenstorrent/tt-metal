#pragma once
#include <cstdint>
#include <stdint.h>

namespace host_mem {

struct address_map {

  // SYSMEM accessible via DEVICE-to-HOST MMIO

  static constexpr std::int32_t DEVICE_TO_HOST_MMIO_SIZE_BYTES    = 1024 * 1024 * 1024; // 1GB
  static constexpr std::int32_t DEVICE_TO_HOST_SCRATCH_SIZE_BYTES = 128 * 1024 * 1024;
  static constexpr std::int32_t DEVICE_TO_HOST_SCRATCH_START      = DEVICE_TO_HOST_MMIO_SIZE_BYTES - DEVICE_TO_HOST_SCRATCH_SIZE_BYTES;
  static constexpr std::int32_t DEVICE_TO_HOST_REGION_SIZE_BYTES  = DEVICE_TO_HOST_MMIO_SIZE_BYTES - DEVICE_TO_HOST_SCRATCH_SIZE_BYTES;
  static constexpr std::int32_t DEVICE_TO_HOST_REGION_START       = 0;

  static constexpr std::int32_t ETH_ROUTING_BLOCK_SIZE = 32 * 1024;
  static constexpr std::int32_t ETH_ROUTING_BUFFERS_START = DEVICE_TO_HOST_SCRATCH_START;
  static constexpr std::int32_t ETH_ROUTING_BUFFERS_SIZE = ETH_ROUTING_BLOCK_SIZE * 16 * 4;// 16 ethernet cores x 4 buffers/core

  // Concurrent perf trace parameters
  static constexpr std::int32_t HOST_PERF_SCRATCH_BUF_START = DEVICE_TO_HOST_SCRATCH_START + ETH_ROUTING_BUFFERS_SIZE;
  static constexpr std::int32_t HOST_PERF_SCRATCH_BUF_SIZE = 64 * 1024 * 1024;
  static constexpr std::int32_t NUM_THREADS_IN_EACH_DEVICE_DUMP = 1;
  static constexpr std::int32_t NUM_HOST_PERF_QUEUES = 6 * 64;
  static constexpr std::int32_t HOST_PERF_QUEUE_SLOT_SIZE = HOST_PERF_SCRATCH_BUF_SIZE / NUM_HOST_PERF_QUEUES / 32 * 32;
};
}
