
#pragma once

#include "core_coord.h"
#include "common/tt_backend_api_types.hpp"
#include "third_party/umd/device/tt_soc_descriptor.h"


//! tt_SocDescriptor contains information regarding the SOC configuration targetted.
/*!
    Should only contain relevant configuration for SOC
*/
struct metal_SocDescriptor : public tt_SocDescriptor {
  public:
  std::vector<CoreCoord> preferred_worker_dram_core;  // per channel preferred worker endpoint
  std::vector<CoreCoord> preferred_eth_dram_core;  // per channel preferred eth endpoint
  std::vector<size_t> dram_address_offsets;  // starting address offset
  CoreCoord compute_with_storage_grid_size;
  std::vector<RelativeCoreCoord> compute_with_storage_cores;  // saved as CoreType::WORKER
  std::vector<RelativeCoreCoord> storage_cores;  // saved as CoreType::WORKER
  std::vector<RelativeCoreCoord> dispatch_cores; // saved as CoreType::WORKER
  int l1_bank_size;

  metal_SocDescriptor(std::string device_descriptor_path);
  metal_SocDescriptor(const tt_SocDescriptor& other);
  metal_SocDescriptor() = default;

  CoreCoord get_preferred_worker_core_for_dram_channel(int dram_chan) const;
  CoreCoord get_preferred_eth_core_for_dram_channel(int dram_chan) const;
  size_t get_address_offset(int dram_chan) const;

  bool is_harvested_core(const CoreCoord &core) const;
  const std::vector<CoreCoord>& get_pcie_cores() const;
  const std::vector<CoreCoord> get_dram_cores() const;
  const std::vector<CoreCoord>& get_ethernet_cores() const;

  private:
  void init();
  void load_dram_metadata_from_device_descriptor();
  void map_workers_to_dram_banks();
};

void load_dispatch_and_banking_config(metal_SocDescriptor &soc_descriptor, uint32_t num_harvested_noc_rows);
