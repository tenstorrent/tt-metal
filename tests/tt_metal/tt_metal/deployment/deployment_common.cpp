// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "deployment_common.hpp"

#include <dirent.h>

#include <fstream>

std::atomic_bool g_stop_requested = false;
std::atomic_bool g_stop_message_printed = false;

void handle_sigint(int) {
    g_stop_requested.store(true);

    if (!g_stop_message_printed.exchange(true)) {
        const char msg[] = "\nSIGINT received, waiting to finish current test...\n";
        [[maybe_unused]] ssize_t written = write(STDERR_FILENO, msg, sizeof msg - 1); /* NOLINT */
    }
}

static std::map<uint32_t, std::string> id_to_bdf;
static void init_id_to_bdf() {
    if (id_to_bdf.empty()) {
        tt::Cluster& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
        tt::umd::ClusterDescriptor* desc = cluster.get_cluster_desc();
        const std::unordered_map<tt::ChipId, std::string>& bdfs = desc->get_chip_pci_bdfs();
        for (const auto& [id, bdf] : bdfs) {
            id_to_bdf[id] = bdf;
        }
    }
}

std::string pci_bdf_for_device_id(uint32_t device_id) {
    init_id_to_bdf();

    if (!id_to_bdf.contains(device_id)) {
        return "unknown";
    }

    return id_to_bdf[device_id];
}

std::string trim_copy(std::string s) {
    while (!s.empty() && std::isspace(s.front())) {
        s.erase(s.begin());
    }
    while (!s.empty() && std::isspace(s.back())) {
        s.pop_back();
    }
    return s;
}

std::string read_text_file_trimmed(const std::string& path) {
    std::ifstream file(path);
    std::string value;
    std::getline(file, value);
    return trim_copy(value);
}

std::string get_ubb_id_str(uint32_t chip_id) {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    tt::umd::ClusterDescriptor& cluster_desc = (*(cluster.get_cluster_desc()));
    auto ubb_id = tt::tt_fabric::get_ubb_id(cluster_desc, chip_id);
    return "UBB: " + std::to_string(ubb_id.tray_id) + " Chip: " + std::to_string(ubb_id.asic_id) +
           " BDF: " + pci_bdf_for_device_id(chip_id);
}

static std::vector<uint32_t> get_chip_ids() {
    init_id_to_bdf();

    std::vector<uint32_t> keys;
    for (auto& it : id_to_bdf) {
        keys.push_back(it.first); /* NOLINT */
    }

    sort(keys.begin(), keys.end());

    return keys;
}

std::vector<std::string> get_chip_physical_locations() {
    std::vector<std::string> ret;
    for (uint32_t id : get_chip_ids()) {
        ret.emplace_back(get_ubb_id_str(id));
    }

    return ret;
}
