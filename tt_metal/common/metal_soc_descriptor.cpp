#include "metal_soc_descriptor.h"

#include <iostream>
#include <string>
#include <fstream>
#include "yaml-cpp/yaml.h"
#include "tt_metal/third_party/umd/device/tt_device.h"
#include "dev_mem_map.h"

#include "common/assert.hpp"

CoreCoord metal_SocDescriptor::get_preferred_worker_core_for_dram_channel(int dram_chan) const {
    tt::log_assert(dram_chan < this->preferred_worker_dram_core.size(), "dram_chan={} must be within range of preferred_worker_dram_core.size={}", dram_chan, this->preferred_worker_dram_core.size());
    return this->preferred_worker_dram_core.at(dram_chan);
};

CoreCoord metal_SocDescriptor::get_preferred_eth_core_for_dram_channel(int dram_chan) const {
    tt::log_assert(dram_chan < this->preferred_eth_dram_core.size(), "dram_chan={} must be within range of preferred_eth_dram_core.size={}", dram_chan, this->preferred_eth_dram_core.size());
    return this->preferred_eth_dram_core.at(dram_chan);
};

size_t metal_SocDescriptor::get_address_offset(int dram_chan) const {
    tt::log_assert(dram_chan < this->dram_address_offsets.size(), "dram_chan={} must be within range of dram_address_offsets.size={}", dram_chan, this->dram_address_offsets.size());
    return this->dram_address_offsets.at(dram_chan);
}

bool metal_SocDescriptor::is_harvested_core(const CoreCoord &core) const {
    for (const auto& core_it : this->harvested_workers) {
        if (core_it == core) {
            return true;
        }
    }
    return false;
}

const std::vector<CoreCoord>& metal_SocDescriptor::get_pcie_cores() const {
    return this->pcie_cores;
}

const std::vector<CoreCoord> metal_SocDescriptor::get_dram_cores() const {
    std::vector<CoreCoord> cores;

    // This is inefficient, but is currently not used in a perf path
    for (const auto& channel_it : this->dram_cores) {
        for (const auto& core_it : channel_it) {
            cores.push_back(core_it);
        }
    }

    return cores;
}

const std::vector<CoreCoord>& metal_SocDescriptor::get_ethernet_cores() const {
    return this->ethernet_cores;
}

void metal_SocDescriptor::load_dram_metadata_from_device_descriptor() {
  YAML::Node device_descriptor_yaml = YAML::LoadFile(this->device_descriptor_file_path);
  this->preferred_eth_dram_core.clear();
  for (const auto& core_node: device_descriptor_yaml["dram_preferred_eth_endpoint"]) {
    if (core_node.IsScalar()) {
      this->preferred_eth_dram_core.push_back(format_node(core_node.as<std::string>()));
    } else {
      tt::log_fatal ("Only NOC coords supported for dram_preferred_eth_endpoint cores");
    }
  }

  this->preferred_worker_dram_core.clear();
  for (const auto& core_node: device_descriptor_yaml["dram_preferred_worker_endpoint"]) {
    if (core_node.IsScalar()) {
      this->preferred_worker_dram_core.push_back(format_node(core_node.as<std::string>()));
    } else {
      tt::log_fatal ("Only NOC coords supported for dram_preferred_worker_endpoint");
    }
  }

  this->dram_address_offsets = device_descriptor_yaml["dram_address_offsets"].as<std::vector<size_t>>();
}

// Determines which core will write perf-events on which dram-bank.
// Creates a map of dram cores to worker cores, in the order that they will get dumped.
void metal_SocDescriptor::map_workers_to_dram_banks() {
  for (CoreCoord worker: this->workers) {
    TT_ASSERT(this->dram_cores.size() > 0, "No DRAM channels detected");
    // Initialize target dram core to the first dram.
    CoreCoord target_dram_bank = this->dram_cores.at(0).at(0);
    std::vector<std::vector<CoreCoord>> dram_cores_per_channel;
    if (this->arch == tt::ARCH::WORMHOLE || this->arch == tt::ARCH::WORMHOLE_B0) {
      dram_cores_per_channel = {{CoreCoord(0, 0)}, {CoreCoord(0, 5)}, {CoreCoord(5, 0)}, {CoreCoord(5, 2)}, {CoreCoord(5, 3)}, {CoreCoord(5, 5)}};
    } else {
      dram_cores_per_channel = this->dram_cores;
    }
    for (const auto &dram_cores : dram_cores_per_channel) {
      for (CoreCoord dram: dram_cores) {
        int diff_x = worker.x - dram.x;
        int diff_y = worker.y - dram.y;
        // Represents a dram core that comes "before" this worker.
        if (diff_x >= 0 && diff_y >= 0) {
          int diff_dram_x = worker.x - target_dram_bank.x;
          int diff_dram_y = worker.y - target_dram_bank.y;
          // If initial dram core comes after the worker, swap it with this dram.
          if (diff_dram_x < 0 || diff_dram_y < 0) {
            target_dram_bank = dram;
            // If both target dram core and current dram core come before the worker, choose the one that's closer.
          } else if (diff_x + diff_y < diff_dram_x + diff_dram_y) {
            target_dram_bank = dram;
          }
        }
      }
    }
    if (this->perf_dram_bank_to_workers.find(target_dram_bank) == this->perf_dram_bank_to_workers.end()) {
      this->perf_dram_bank_to_workers.insert(std::pair<CoreCoord, std::vector<CoreCoord>>(target_dram_bank, {worker}));
    } else {
      this->perf_dram_bank_to_workers[target_dram_bank].push_back(worker);
    }
  }
}

void metal_SocDescriptor::init() {
  this->trisc_sizes = {MEM_TRISC0_SIZE, MEM_TRISC1_SIZE, MEM_TRISC2_SIZE};  // TODO: Read trisc size from yaml
  this->load_dram_metadata_from_device_descriptor();
  this->map_workers_to_dram_banks();
}

metal_SocDescriptor::metal_SocDescriptor(std::string device_descriptor_path) : tt_SocDescriptor(device_descriptor_path) {
  this->init();
}

metal_SocDescriptor::metal_SocDescriptor(const tt_SocDescriptor& other) : tt_SocDescriptor(other) {
  this->init();
}

const std::string get_product_name(tt::ARCH arch, uint32_t num_harvested_noc_rows) {
  const static std::map<tt::ARCH, std::map<uint32_t, std::string> > product_name = {
      {tt::ARCH::GRAYSKULL, { {0, "E150"} } },
      {tt::ARCH::WORMHOLE_B0, { {0, "galaxy"}, {1, "nebula_x1"}, {2, "nebula_x2"} } }
  };

  return product_name.at(arch).at(num_harvested_noc_rows);
}

void load_dispatch_and_banking_config(metal_SocDescriptor &soc_descriptor, uint32_t num_harvested_noc_rows) {
  YAML::Node device_descriptor_yaml = YAML::LoadFile(soc_descriptor.device_descriptor_file_path);

  auto product_to_config  = device_descriptor_yaml["dispatch_and_banking"];
  auto product_name = get_product_name(soc_descriptor.arch, num_harvested_noc_rows);
  auto config = product_to_config[product_name];

  soc_descriptor.l1_bank_size = config["l1_bank_size"].as<int>();

  // TODO: Add validation for compute_with_storage, storage only, and dispatch core specification
  auto compute_with_storage_start = config["compute_with_storage_grid_range"]["start"];
  auto compute_with_storage_end = config["compute_with_storage_grid_range"]["end"];
  TT_ASSERT(compute_with_storage_start.IsSequence() and compute_with_storage_end.IsSequence());
  TT_ASSERT(compute_with_storage_end[0].as<size_t>() >= compute_with_storage_start[0].as<size_t>());
  TT_ASSERT(compute_with_storage_end[1].as<size_t>() >= compute_with_storage_start[1].as<size_t>());

  soc_descriptor.compute_with_storage_grid_size = CoreCoord({
    .x = (compute_with_storage_end[0].as<size_t>() - compute_with_storage_start[0].as<size_t>()) + 1,
    .y = (compute_with_storage_end[1].as<size_t>() - compute_with_storage_start[1].as<size_t>()) + 1,
  });

  // compute_with_storage_cores are a subset of worker cores
  // they have already been parsed as CoreType::WORKER and saved into `cores` map when parsing `functional_workers`
  for (auto x = 0; x < soc_descriptor.compute_with_storage_grid_size.x; x++) {
    for (auto y = 0; y < soc_descriptor.compute_with_storage_grid_size.y; y++) {
        const auto relative_coord = RelativeCoreCoord({.x = x, .y = y});
        soc_descriptor.compute_with_storage_cores.push_back(relative_coord);
    }
  }

  // storage_cores are a subset of worker cores
  // they have already been parsed as CoreType::WORKER and saved into `cores` map when parsing `functional_workers`
  for (const auto &core_node : config["storage_cores"]) {
    RelativeCoreCoord coord = {};
    if (core_node.IsSequence()) {
      // Logical coord
      coord = RelativeCoreCoord({.x = core_node[0].as<int>(), .y = core_node[1].as<int>()});
    } else {
      tt::log_fatal ("Only logical relative coords supported for storage_cores cores");
    }
    soc_descriptor.storage_cores.push_back(coord);
  }

  // dispatch_cores are a subset of worker cores
  // they have already been parsed as CoreType::WORKER and saved into `cores` map when parsing `functional_workers`
  for (const auto &core_node : config["dispatch_cores"]) {
    RelativeCoreCoord coord = {};
    if (core_node.IsSequence()) {
      // Logical coord
      coord = RelativeCoreCoord({.x = core_node[0].as<int>(), .y = core_node[1].as<int>()});
    } else {
      tt::log_fatal ("Only logical relative coords supported for dispatch_cores cores");
    }
    soc_descriptor.dispatch_cores.push_back(coord);
  }
}
