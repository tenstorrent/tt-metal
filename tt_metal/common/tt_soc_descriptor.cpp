#include "tt_soc_descriptor.h"

#include <iostream>
#include <string>
#include <fstream>
#include "yaml-cpp/yaml.h"
#include "l1_address_map.h"
#include "device/tt_device.h"

#include "common/assert.hpp"

bool tt_SocDescriptor::is_worker_core(const CoreCoord &core) const {
    return (
        routing_x_to_worker_x.find(core.x) != routing_x_to_worker_x.end() &&
        routing_y_to_worker_y.find(core.y) != routing_y_to_worker_y.end());
}
CoreCoord tt_SocDescriptor::get_worker_core(const CoreCoord &core) const {
    CoreCoord worker_xy = {
        static_cast<size_t>(routing_x_to_worker_x.at(core.x)), static_cast<size_t>(routing_y_to_worker_y.at(core.y))};
    return worker_xy;
}

// Compute and storage cores are worker cores that do compute and their L1 can be used for storage
bool tt_SocDescriptor::is_compute_and_storage_core(const CoreCoord &core) const {
    return std::find(this->compute_and_storage_cores.begin(), this->compute_and_storage_cores.end(), core) != this->compute_and_storage_cores.end();
}

// Storage cores do not do any compute, their L1 is used for storage
bool tt_SocDescriptor::is_storage_core(const CoreCoord &core) const {
    return std::find(this->storage_cores.begin(), this->storage_cores.end(), core) != this->storage_cores.end();
}

// Dispatch cores send work to compute and storage cores
bool tt_SocDescriptor::is_dispatch_core(const CoreCoord &core) const {
    return std::find(this->dispatch_cores.begin(), this->dispatch_cores.end(), core) != this->dispatch_cores.end();
}

int tt_SocDescriptor::get_num_dram_channels() const {
    int num_channels = 0;
    for (const auto& dram_core : dram_cores) {
        if (dram_core.size() > 0) {
            num_channels++;
        }
    }
    return num_channels;
}
CoreCoord tt_SocDescriptor::get_core_for_dram_channel(int dram_chan, int subchannel) const {
    tt::log_assert(dram_chan < this->dram_cores.size(), "dram_chan={} must be within range of num_dram_channels={}", dram_chan, this->dram_cores.size());
    tt::log_assert(subchannel < this->dram_cores.at(dram_chan).size(), "subchannel={} must be within range of num_subchannels={}", subchannel, this->dram_cores.at(dram_chan).size());
    return this->dram_cores.at(dram_chan).at(subchannel);
};
CoreCoord tt_SocDescriptor::get_preferred_worker_core_for_dram_channel(int dram_chan) const {
    tt::log_assert(dram_chan < this->preferred_worker_dram_core.size(), "dram_chan={} must be within range of preferred_worker_dram_core.size={}", dram_chan, this->preferred_worker_dram_core.size());
    return this->preferred_worker_dram_core.at(dram_chan);
};
CoreCoord tt_SocDescriptor::get_preferred_eth_core_for_dram_channel(int dram_chan) const {
    tt::log_assert(dram_chan < this->preferred_eth_dram_core.size(), "dram_chan={} must be within range of preferred_eth_dram_core.size={}", dram_chan, this->preferred_eth_dram_core.size());
    return this->preferred_eth_dram_core.at(dram_chan);
};
size_t tt_SocDescriptor::get_address_offset(int dram_chan) const {
    tt::log_assert(dram_chan < this->dram_address_offsets.size(), "dram_chan={} must be within range of dram_address_offsets.size={}", dram_chan, this->dram_address_offsets.size());
    return this->dram_address_offsets.at(dram_chan);
}
int tt_SocDescriptor::get_num_dram_subchans() const {
    int num_chan = 0;
    for (const std::vector<CoreCoord> &core : this->dram_cores) {
        num_chan += core.size();
    }
    return num_chan;
}
int tt_SocDescriptor::get_num_dram_blocks_per_channel() const {
    int num_blocks = 0;
    if (arch == tt::ARCH::GRAYSKULL) {
        num_blocks = 1;
    } else if (arch == tt::ARCH::WORMHOLE) {
        num_blocks = 2;
    } else if (arch == tt::ARCH::WORMHOLE_B0) {
        num_blocks = 2;
    }
    return num_blocks;
}


bool tt_SocDescriptor::is_ethernet_core(const CoreCoord &core) const {
    return this->ethernet_core_channel_map.find(core) != ethernet_core_channel_map.end();
}

bool tt_SocDescriptor::get_channel_of_ethernet_core(const CoreCoord &core) const {
    return this->ethernet_core_channel_map.at(core);
}

const char* ws = " \t\n\r\f\v";

// trim from end of string (right)
inline std::string& rtrim(std::string& s, const char* t = ws)
{
    s.erase(s.find_last_not_of(t) + 1);
    return s;
}

// trim from beginning of string (left)
inline std::string& ltrim(std::string& s, const char* t = ws)
{
    s.erase(0, s.find_first_not_of(t));
    return s;
}

// trim from both ends of string (right then left)
inline std::string& trim(std::string& s, const char* t = ws)
{
    return ltrim(rtrim(s, t), t);
}

void load_core_descriptors_from_device_descriptor(
    YAML::Node device_descriptor_yaml, tt_SocDescriptor &soc_descriptor) {
  auto worker_l1_size = device_descriptor_yaml["worker_l1_size"].as<int>();
  auto storage_core_l1_bank_size = device_descriptor_yaml["storage_core_l1_bank_size"].as<int>();
  auto eth_l1_size = device_descriptor_yaml["eth_l1_size"].as<int>();

  auto arc_cores = device_descriptor_yaml["arc"].as<std::vector<std::string>>();
  for (const auto &core_string : arc_cores) {
    CoreDescriptor core_descriptor;
    core_descriptor.coord = format_node(core_string);
    core_descriptor.type = CoreType::ARC;
    soc_descriptor.cores.insert({core_descriptor.coord, core_descriptor});
    soc_descriptor.arc_cores.push_back(core_descriptor.coord);
  }
  auto pcie_cores = device_descriptor_yaml["pcie"].as<std::vector<std::string>>();
  for (const auto &core_string : pcie_cores) {
    CoreDescriptor core_descriptor;
    core_descriptor.coord = format_node(core_string);
    core_descriptor.type = CoreType::PCIE;
    soc_descriptor.cores.insert({core_descriptor.coord, core_descriptor});
    soc_descriptor.pcie_cores.push_back(core_descriptor.coord);
  }

  int current_dram_channel = 0;
  for (const auto &channel_it : device_descriptor_yaml["dram"]) {
    soc_descriptor.dram_cores.push_back({});
    auto &soc_dram_cores = soc_descriptor.dram_cores.at(soc_descriptor.dram_cores.size() - 1);
    const auto &dram_cores = channel_it.as<std::vector<std::string>>();
    for (int i = 0; i < dram_cores.size(); i++) {
      const auto &dram_core = dram_cores.at(i);
      CoreDescriptor core_descriptor;
      core_descriptor.coord = format_node(dram_core);
      core_descriptor.type = CoreType::DRAM;
      soc_descriptor.cores.insert({core_descriptor.coord, core_descriptor});
      soc_dram_cores.push_back(core_descriptor.coord);
      soc_descriptor.dram_core_channel_map[core_descriptor.coord] = {current_dram_channel, i};
    }
    current_dram_channel++;
  }

  soc_descriptor.preferred_eth_dram_core.clear();
  for (const auto& channel_it: device_descriptor_yaml["dram_preferred_eth_endpoint"]) {
    soc_descriptor.preferred_eth_dram_core.push_back(format_node(channel_it.as<std::string>()));
  }
  soc_descriptor.preferred_worker_dram_core.clear();
  for (const auto& channel_it: device_descriptor_yaml["dram_preferred_worker_endpoint"]) {
    soc_descriptor.preferred_worker_dram_core.push_back(format_node(channel_it.as<std::string>()));
  }
  soc_descriptor.dram_address_offsets = device_descriptor_yaml["dram_address_offsets"].as<std::vector<size_t>>();

  auto eth_cores = device_descriptor_yaml["eth"].as<std::vector<std::string>>();
  int current_ethernet_channel = 0;
  for (const auto &core_string : eth_cores) {
    CoreDescriptor core_descriptor;
    core_descriptor.coord = format_node(core_string);
    core_descriptor.type = CoreType::ETH;
    core_descriptor.l1_size = eth_l1_size;
    core_descriptor.l1_bank_size = eth_l1_size;
    soc_descriptor.cores.insert({core_descriptor.coord, core_descriptor});
    soc_descriptor.ethernet_cores.push_back(core_descriptor.coord);

    soc_descriptor.ethernet_core_channel_map[core_descriptor.coord] = current_ethernet_channel;
    current_ethernet_channel++;
  }
  std::vector<std::string> worker_cores = device_descriptor_yaml["functional_workers"].as<std::vector<std::string>>();
  std::set<int> worker_routing_coords_x;
  std::set<int> worker_routing_coords_y;
  std::unordered_map<int,int> routing_coord_worker_x;
  std::unordered_map<int,int> routing_coord_worker_y;
  for (const auto &core_string : worker_cores) {
    CoreDescriptor core_descriptor;
    core_descriptor.coord = format_node(core_string);
    core_descriptor.type = CoreType::WORKER;
    core_descriptor.l1_size = worker_l1_size;
    core_descriptor.l1_bank_size = worker_l1_size;
    core_descriptor.dram_size_per_core = DEFAULT_DRAM_SIZE_PER_CORE;
    soc_descriptor.cores.insert({core_descriptor.coord, core_descriptor});
    soc_descriptor.workers.push_back(core_descriptor.coord);
    worker_routing_coords_x.insert(core_descriptor.coord.x);
    worker_routing_coords_y.insert(core_descriptor.coord.y);
  }

  int func_x_start = 0;
  int func_y_start = 0;
  std::set<int>::iterator it;
  for (it=worker_routing_coords_x.begin(); it!=worker_routing_coords_x.end(); ++it) {
    soc_descriptor.worker_log_to_routing_x[func_x_start] = *it;
    soc_descriptor.routing_x_to_worker_x[*it] = func_x_start;
    func_x_start++;
  }
  for (it=worker_routing_coords_y.begin(); it!=worker_routing_coords_y.end(); ++it) {
    soc_descriptor.worker_log_to_routing_y[func_y_start] = *it;
    soc_descriptor.routing_y_to_worker_y[*it] = func_y_start;
    func_y_start++;
  }

  soc_descriptor.worker_grid_size = CoreCoord(func_x_start, func_y_start);

  auto harvested_cores = device_descriptor_yaml["harvested_workers"].as<std::vector<std::string>>();
  for (const auto &core_string : harvested_cores) {
    CoreDescriptor core_descriptor;
    core_descriptor.coord = format_node(core_string);
    core_descriptor.type = CoreType::HARVESTED;
    soc_descriptor.cores.insert({core_descriptor.coord, core_descriptor});
  }

  std::set<int> compute_and_storage_coords_x;
  std::set<int> compute_and_storage_coords_y;
  auto compute_and_storage_cores = device_descriptor_yaml["compute_and_storage_cores"].as<std::vector<std::string>>();
  // compute_and_storage_cores are a subset of worker cores
  // they have already been parsed as CoreType::WORKER and saved into `cores` map when parsing `functional_workers`
  for (const auto &core_string : compute_and_storage_cores) {
    CoreCoord coord = format_node(core_string);
    compute_and_storage_coords_x.insert(coord.x);
    compute_and_storage_coords_y.insert(coord.y);
    soc_descriptor.compute_and_storage_cores.push_back(coord);
  }

  soc_descriptor.compute_and_storage_grid_size = CoreCoord(compute_and_storage_coords_x.size(), compute_and_storage_coords_y.size());

  auto storage_cores = device_descriptor_yaml["storage_cores"].as<std::vector<std::string>>();
  // storage_cores are a subset of worker cores
  // they have already been parsed as CoreType::WORKER and saved into `cores` map when parsing `functional_workers`
  for (const auto &core_string : storage_cores) {
    CoreCoord coord = format_node(core_string);
    auto &storage_core_desc = soc_descriptor.cores.at(coord);
    storage_core_desc.l1_bank_size = storage_core_l1_bank_size;
    soc_descriptor.storage_cores.push_back(coord);
  }

  auto dispatch_cores = device_descriptor_yaml["dispatch_cores"].as<std::vector<std::string>>();
  // dispatch_cores are a subset of worker cores
  // they have already been parsed as CoreType::WORKER and saved into `cores` map when parsing `functional_workers`
  for (const auto &core_string : dispatch_cores) {
    CoreCoord coord = format_node(core_string);
    soc_descriptor.dispatch_cores.push_back(coord);
  }

  auto router_only_cores = device_descriptor_yaml["router_only"].as<std::vector<std::string>>();
  for (const auto &core_string : router_only_cores) {
    CoreDescriptor core_descriptor;
    core_descriptor.coord = format_node(core_string);
    core_descriptor.type = CoreType::ROUTER_ONLY;
    soc_descriptor.cores.insert({core_descriptor.coord, core_descriptor});
  }
}

void load_soc_features_from_device_descriptor(YAML::Node &device_descriptor_yaml, tt_SocDescriptor *soc_descriptor) {
  soc_descriptor->overlay_version = device_descriptor_yaml["features"]["overlay"]["version"].as<int>();
  soc_descriptor->packer_version = device_descriptor_yaml["features"]["packer"]["version"].as<int>();
  soc_descriptor->unpacker_version = device_descriptor_yaml["features"]["unpacker"]["version"].as<int>();
  soc_descriptor->dst_size_alignment = device_descriptor_yaml["features"]["math"]["dst_size_alignment"].as<int>();
  soc_descriptor->worker_l1_size = device_descriptor_yaml["worker_l1_size"].as<int>();
  soc_descriptor->storage_core_l1_bank_size = device_descriptor_yaml["storage_core_l1_bank_size"].as<int>();
  soc_descriptor->eth_l1_size = device_descriptor_yaml["eth_l1_size"].as<int>();
  soc_descriptor->dram_bank_size = device_descriptor_yaml["dram_bank_size"].as<uint32_t>();
}

// Determines which core will write perf-events on which dram-bank.
// Creates a map of dram cores to worker cores, in the order that they will get dumped.
void map_workers_to_dram_banks(tt_SocDescriptor *soc_descriptor) {
  for (CoreCoord worker:soc_descriptor->workers) {
    TT_ASSERT(soc_descriptor->dram_cores.size() > 0, "No DRAM channels detected");
    // Initialize target dram core to the first dram.
    CoreCoord target_dram_bank = soc_descriptor->dram_cores.at(0).at(0);
    std::vector<std::vector<CoreCoord>> dram_cores_per_channel;
    if (soc_descriptor->arch == tt::ARCH::WORMHOLE || soc_descriptor->arch == tt::ARCH::WORMHOLE_B0) {
      dram_cores_per_channel = {{CoreCoord(0, 0)}, {CoreCoord(0, 5)}, {CoreCoord(5, 0)}, {CoreCoord(5, 2)}, {CoreCoord(5, 3)}, {CoreCoord(5, 5)}};
    } else {
      dram_cores_per_channel = soc_descriptor->dram_cores;
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
    if (soc_descriptor->perf_dram_bank_to_workers.find(target_dram_bank) == soc_descriptor->perf_dram_bank_to_workers.end()) {
      soc_descriptor->perf_dram_bank_to_workers.insert(std::pair<CoreCoord, std::vector<CoreCoord>>(target_dram_bank, {worker}));
    } else {
      soc_descriptor->perf_dram_bank_to_workers[target_dram_bank].push_back(worker);
    }
  }
}

std::unique_ptr<tt_SocDescriptor> load_soc_descriptor_from_yaml(std::string device_descriptor_file_path) {
  std::unique_ptr<tt_SocDescriptor> soc_descriptor = std::unique_ptr<tt_SocDescriptor>(new tt_SocDescriptor());

  std::ifstream fdesc(device_descriptor_file_path);
  if (fdesc.fail()) {
      throw std::runtime_error("Error: device descriptor file " + device_descriptor_file_path + " does not exist!");
  }
  fdesc.close();

  YAML::Node device_descriptor_yaml = YAML::LoadFile(device_descriptor_file_path);
  std::vector<std::size_t> trisc_sizes = {MEM_TRISC0_SIZE,
                                          MEM_TRISC1_SIZE,
                                          MEM_TRISC2_SIZE};  // TODO: Read trisc size from yaml

  auto grid_size_x = device_descriptor_yaml["grid"]["x_size"].as<int>();
  auto grid_size_y = device_descriptor_yaml["grid"]["y_size"].as<int>();

  load_core_descriptors_from_device_descriptor(device_descriptor_yaml, *soc_descriptor);
  soc_descriptor->grid_size = CoreCoord(grid_size_x, grid_size_y);
  soc_descriptor->device_descriptor_file_path = device_descriptor_file_path;
  soc_descriptor->trisc_sizes = trisc_sizes;

  std::string arch_name_value = device_descriptor_yaml["arch_name"].as<std::string>();
  arch_name_value = trim(arch_name_value);

  soc_descriptor->arch = tt::get_arch_from_string(arch_name_value);

  load_soc_features_from_device_descriptor(device_descriptor_yaml, soc_descriptor.get());

  map_workers_to_dram_banks(soc_descriptor.get());
  return soc_descriptor;
}
