// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tunnels_from_mmio_device.hpp"
#include <umd/device/cluster_descriptor.hpp>
#include <umd/device/cluster.hpp>


#include <tt_stl/assert.hpp>

namespace tt::llrt {

// TODO: Stop using these functions here and in PhysicalSystemDescriptor once UMD provides support for ASIC index/offset
// in WH systems
const std::unordered_set<ChipId>& get_devices_controlled_by_mmio_device(
    tt::umd::ClusterDescriptor& cluster_descriptor, ChipId mmio_device_id) {
    TT_ASSERT(
        cluster_descriptor.get_chips_grouped_by_closest_mmio().count(mmio_device_id),
        "Expected device {} to be an MMIO device!",
        mmio_device_id);
    return cluster_descriptor.get_chips_grouped_by_closest_mmio().at(mmio_device_id);
}

#define MAX_TUNNEL_DEPTH 4
std::map<ChipId, std::vector<std::vector<ChipId>>> discover_tunnels_from_mmio_device(
      tt::umd::ClusterDescriptor& cluster_desc) {

      std::map<ChipId, std::vector<std::vector<ChipId>>> tunnels_from_mmio_device = {};
      const auto& all_eth_connections = cluster_desc.get_ethernet_connections();

      for (const auto& [mmio_chip_id, _] : cluster_desc.get_chips_with_mmio()) {
          std::vector<std::vector<ChipId>> tunnels_from_mmio = {};
          TT_ASSERT(cluster_desc.is_chip_mmio_capable(mmio_chip_id));

          if (!all_eth_connections.contains(mmio_chip_id)) {
              tunnels_from_mmio_device.insert({mmio_chip_id, {}});
              continue;
          }

          auto device_ids = cluster_desc.get_chips_grouped_by_closest_mmio().at(mmio_chip_id);
          device_ids.erase(mmio_chip_id);

          if (device_ids.empty()) {
              tunnels_from_mmio_device.insert({mmio_chip_id, {}});
              continue;
          }

          for (const auto& [eth_chan, connected_chip_chan] : all_eth_connections.at(mmio_chip_id)) {
              const auto& other_chip_id = std::get<0>(connected_chip_chan);
              if (device_ids.contains(other_chip_id)) {
                  device_ids.erase(other_chip_id);
                  std::vector<ChipId> first_stop = {other_chip_id};
                  auto it = std::find(tunnels_from_mmio.begin(), tunnels_from_mmio.end(), first_stop);
                  TT_FATAL(it == tunnels_from_mmio.end(), "Duplicate first tunnel stop found.");
                  tunnels_from_mmio.push_back(first_stop);
              }
          }

          device_ids = cluster_desc.get_chips_grouped_by_closest_mmio().at(mmio_chip_id);
          device_ids.erase(mmio_chip_id);
          for (auto& tunnel : tunnels_from_mmio) {
              device_ids.erase(tunnel[0]);
          }

          bool tunneled_device_hit;
          while (!device_ids.empty()) {
              tunneled_device_hit = false;
              for (auto& dev_vec : tunnels_from_mmio) {
                  for (const auto& [eth_chan, connected_chip_chan] : all_eth_connections.at(dev_vec.back())) {
                      const auto& other_chip_id = std::get<0>(connected_chip_chan);
                      auto id_iter = device_ids.find(other_chip_id);
                      if (id_iter != device_ids.end()) {
                          device_ids.erase(id_iter);
                          dev_vec.push_back(other_chip_id);
                          tunneled_device_hit = true;
                          break;
                      }
                  }
              }
              TT_FATAL(tunneled_device_hit || device_ids.empty(),
                  "Detected ethernet connections did not match expected device connectivity, try re-running tt-topology.");
          }

          TT_FATAL(!tunnels_from_mmio.empty(), "Must have at least 1 tunnel from MMIO Device.");
          uint32_t tunnel_depth = tunnels_from_mmio[0].size();
          for (auto& dev_vec : tunnels_from_mmio) {
              TT_FATAL(dev_vec.size() == tunnel_depth, "All tunnels must have same depth.");
              if (dev_vec.size() > MAX_TUNNEL_DEPTH) {
                  dev_vec.resize(MAX_TUNNEL_DEPTH);
              }
              dev_vec.insert(dev_vec.begin(), mmio_chip_id);
          }
          tunnels_from_mmio_device.insert({mmio_chip_id, tunnels_from_mmio});
      }

      return tunnels_from_mmio_device;
  }

}  // namespace tt::llrt
