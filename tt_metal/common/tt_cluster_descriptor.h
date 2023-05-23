
#pragma once

#include "core_coord.h"

#include <unordered_map>
#include <unordered_set>
#include <set>
#include <map>
#include <tuple>
#include <string>
#include <vector>
#include <memory>

using chip_id_t = int;
using ethernet_channel_t = int;
namespace YAML { class Node; }

class tt_ClusterDescriptor {
  private:

  std::unordered_map<chip_id_t, std::unordered_map<ethernet_channel_t, std::tuple<chip_id_t, ethernet_channel_t> > > ethernet_connections;
  std::unordered_map<chip_id_t, CoreCoord> chip_locations;
  std::unordered_set<chip_id_t> chips_with_mmio;
  std::unordered_set<chip_id_t> all_chips;

  std::unordered_set<chip_id_t> enabled_active_chips;

  static void load_ethernet_connections_from_connectivity_descriptor(YAML::Node &yaml, tt_ClusterDescriptor &desc);
  static void load_chips_from_connectivity_descriptor(YAML::Node &yaml, tt_ClusterDescriptor &desc);

 public:
  tt_ClusterDescriptor() = default;
  tt_ClusterDescriptor(const tt_ClusterDescriptor&)=default;

  /*
   * Returns the pairs of channels that are connected where the first entry in the pair corresponds to the argument ordering when calling the function
   * An empty result implies that the two chips do not share any direct connection
   */
  std::vector<std::tuple<ethernet_channel_t, ethernet_channel_t>> get_directly_connected_ethernet_channels_between_chips(const chip_id_t &first, const chip_id_t &second) const;
  bool is_chip_mmio_capable(const chip_id_t &chip_id) const;
  chip_id_t get_closest_mmio_capable_chip(const chip_id_t &chip) const;

  static std::unique_ptr<tt_ClusterDescriptor> create_from_yaml(const std::string &cluster_descriptor_file_path);
  static std::unique_ptr<tt_ClusterDescriptor> create_for_grayskull_cluster(
      const std::set<chip_id_t> &target_device_ids);
  // const CoreCoord get_chip_xy(const chip_id_t &chip_id) const;
  // const chip_id_t get_chip_id_at_location(const CoreCoord &chip_location) const;


  bool chips_have_ethernet_connectivity() const;

  std::unordered_map<chip_id_t, CoreCoord> get_chip_locations() const;
  std::unordered_map<chip_id_t, std::unordered_map<ethernet_channel_t, std::tuple<chip_id_t, ethernet_channel_t> > > get_ethernet_connections() const;
  std::unordered_set<chip_id_t> get_chips_with_mmio() const;
  std::unordered_set<chip_id_t> get_all_chips() const;
  std::size_t get_number_of_chips() const;

  bool ethernet_core_has_active_ethernet_link(chip_id_t local_chip, ethernet_channel_t local_ethernet_channel) const;
  std::tuple<chip_id_t, ethernet_channel_t> get_chip_and_channel_of_remote_ethernet_core(chip_id_t local_chip, ethernet_channel_t local_ethernet_channel) const;

  void specify_enabled_devices(const std::vector<chip_id_t> &chip_ids);
  void enable_all_devices();

};

std::set<chip_id_t> get_sequential_chip_id_set(int num_chips);
