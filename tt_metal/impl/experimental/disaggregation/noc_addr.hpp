// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// noc_addr decode helpers shared by the KV chunk readers.
//
// A KvCacheLocation.noc_addr is encoded by kv_cache_utils.py as
// (dram_view/bank_id << 32) | local_addr. Both the in-process reader
// (kv_chunk_address_table.cpp::read_device_chunk) and the device-less UMD
// reader (umd_dram_reader.cpp::read_dram_umd) split it the same way, so the
// decode lives in one place. Mirrors tt-llm-engine
// disaggregation/migration/src/worker/include/noc_addr.hpp.
namespace tt::tt_metal::experimental::disaggregation {

inline uint32_t addr_channel(uint64_t noc_addr) { return static_cast<uint32_t>(noc_addr >> 32); }
inline uint32_t addr_local(uint64_t noc_addr) { return static_cast<uint32_t>(noc_addr & 0xFFFFFFFFull); }

}  // namespace tt::tt_metal::experimental::disaggregation
