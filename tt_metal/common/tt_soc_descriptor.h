
#pragma once

#include <cstddef>
#include <string>
#include <map>
#include <unordered_map>
#include <vector>

#include <iostream>
#include <string>

#include "core_coord.h"
#include "common/tt_backend_api_types.hpp"

static constexpr std::size_t DEFAULT_L1_SIZE = 1 * 1024 * 1024;
static constexpr std::size_t DEFAULT_DRAM_SIZE_PER_CORE = 8 * 1024 * 1024;
static constexpr int NUM_TRISC = 3;

static constexpr tt::ARCH ArchNameDefault = tt::ARCH::GRAYSKULL;

//! SocCore type enumerations
/*! Superset for all chip generations */
enum class CoreType {
  ARC,
  DRAM,
  ETH,
  PCIE,
  WORKER,
  HARVESTED,
  ROUTER_ONLY,

};

//! SocNodeDescriptor contains information regarding the Node/Core
/*!
    Should only contain relevant configuration for SOC
*/
struct CoreDescriptor {
  CoreCoord coord = CoreCoord(0, 0);
  CoreType type;

  std::size_t l1_size = 0;
  std::size_t l1_bank_size = 0;
  std::size_t dram_size_per_core = 0;
};

//! tt_SocDescriptor contains information regarding the SOC configuration targetted.
/*!
    Should only contain relevant configuration for SOC
*/
struct tt_SocDescriptor {
  tt::ARCH arch;
  CoreCoord grid_size;
  CoreCoord worker_grid_size;
  CoreCoord compute_and_storage_grid_size;
  std::unordered_map<CoreCoord, CoreDescriptor> cores;
  std::vector<CoreCoord> arc_cores;
  std::vector<CoreCoord> workers;
  std::vector<CoreCoord> harvested_workers;
  std::vector<CoreCoord> compute_and_storage_cores;  // saved as CoreType::WORKER
  std::vector<CoreCoord> storage_cores;  // saved as CoreType::WORKER
  std::vector<CoreCoord> dispatch_cores; // saved as CoreType::WORKER
  std::vector<CoreCoord> pcie_cores;
  std::unordered_map<int, int> worker_log_to_routing_x;
  std::unordered_map<int, int> worker_log_to_routing_y;
  std::unordered_map<int, int> routing_x_to_worker_x;
  std::unordered_map<int, int> routing_y_to_worker_y;
  std::vector<std::vector<CoreCoord>> dram_cores;  // per channel list of dram cores
  std::vector<CoreCoord> preferred_worker_dram_core;  // per channel preferred worker endpoint
  std::vector<CoreCoord> preferred_eth_dram_core;  // per channel preferred eth endpoint
  std::vector<size_t> dram_address_offsets;  // starting address offset
  std::unordered_map<CoreCoord, std::tuple<int, int>> dram_core_channel_map;  // map dram core to chan/subchan
  std::vector<CoreCoord> ethernet_cores;  // ethernet cores (index == channel id)
  std::unordered_map<CoreCoord,int> ethernet_core_channel_map;
  std::vector<std::size_t> trisc_sizes;  // Most of software stack assumes same trisc size for whole chip..
  std::string device_descriptor_file_path = std::string("");
  bool has(CoreCoord input) { return cores.find(input) != cores.end(); }
  int overlay_version;
  int unpacker_version;
  int dst_size_alignment;
  int packer_version;
  int worker_l1_size;
  int storage_core_l1_bank_size;
  int eth_l1_size;
  uint32_t dram_bank_size;
  std::unordered_map<CoreCoord, std::vector<CoreCoord>> perf_dram_bank_to_workers;

  bool is_worker_core(const CoreCoord &core) const;
  CoreCoord get_worker_core(const CoreCoord& core) const;
  bool is_compute_and_storage_core(const CoreCoord &core) const;
  bool is_storage_core(const CoreCoord &core) const;
  bool is_dispatch_core(const CoreCoord &core) const;

  int get_num_dram_channels() const;
  CoreCoord get_core_for_dram_channel(int dram_chan, int subchannel) const;
  CoreCoord get_preferred_worker_core_for_dram_channel(int dram_chan) const;
  CoreCoord get_preferred_eth_core_for_dram_channel(int dram_chan) const;
  size_t get_address_offset(int dram_chan) const;
  int get_num_dram_subchans() const;
  int get_num_dram_blocks_per_channel() const;

  bool is_ethernet_core(const CoreCoord &core) const;
  bool get_channel_of_ethernet_core(const CoreCoord &core) const;
};

// Allocates a new soc descriptor on the heap. Returns an owning pointer.
std::unique_ptr<tt_SocDescriptor> load_soc_descriptor_from_yaml(std::string device_descriptor_file_path);
