// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "../common/telemetry_table_v2.h"
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <tt-metalium/mesh_device.hpp>
#include "x280_metal_transport.hpp"

#include "x280_address_map.h"

#include <tt-metalium/mesh_device.hpp>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

struct TileX280 {
    int chip_id;
    tt::umd::CoreCoord core;
};

int main(int , char*[]) {
    auto mesh =
        tt::tt_metal::distributed::MeshDevice::create(tt::tt_metal::distributed::MeshDeviceConfig(std::nullopt));
    auto transport = x280::X280MetalTransport::bind(*mesh);

    // Every enabled L2CPU core on every chip we can reach over PCIe. We don't know which tile the
    // loader started, so check them all and report the ones that carry our magic.
    std::vector<TileX280> x280_tiles; 
    for (int chip_id : x280::sorted_usable_chip_ids(*mesh, *transport)) {
        if (!transport->is_mmio_local(chip_id)) {
            std::cout << "Chip " << chip_id << " is not PCIe-local, skipping.\n";
            continue;
        }
        for (const auto& core : transport->l2cpu_cores(chip_id)) {
            x280_tiles.push_back(TileX280{.chip_id = chip_id, .core = core});
        }
    }

  for (const auto& t : x280_tiles) {
      // Skip tiles not running our firmware.
      if (transport->read_u64(t.chip_id, t.core, X280_STANDALONE_HEARTBEAT_ADDR) !=
          static_cast<uint64_t>(X280_STANDALONE_HEARTBEAT_MAGIC)) continue;

      // table_version is written last by the firmware; wait for it to be non-zero.
      if (transport->read_reg32(t.chip_id, t.core, TABLE_ADDR) == 0) continue;

      std::vector<uint8_t> bytes(sizeof(MetricsTelemetryTable));
      transport->read_block(t.chip_id, t.core, TABLE_ADDR, bytes.data(), bytes.size());
      MetricsTelemetryTable tbl;
      std::memcpy(&tbl, bytes.data(), sizeof(tbl));

      for (uint32_t i = 0; i < tbl.num_metrics && i < 128; ++i) {
          const MetricLabel& l = tbl.labels[i];
          if (l.value_index >= 128) continue;
          std::cout << l.metric_name << " = ";
          if (l.metric_type == 0) std::cout << tbl.num_values[l.value_index].num_value;
          else                    std::cout << tbl.str_values[l.value_index].str_value;
          std::cout << "\n";
      }
  }


    return 0;    
}
