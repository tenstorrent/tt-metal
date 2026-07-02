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
        write(2, msg, sizeof msg - 1);
    }
}

std::string pci_bdf_for_device_id(uint32_t device_id) {
    static std::map<uint32_t, std::string> id_to_bdf;

    if (!id_to_bdf.size()) {
        tt::Cluster& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
        tt::umd::ClusterDescriptor* desc = cluster.get_cluster_desc();
        const std::unordered_map<tt::ChipId, std::string>& bdfs = desc->get_chip_pci_bdfs();
        for (const auto& [id, bdf] : bdfs) {
            id_to_bdf[id] = bdf;
        }
    }

    if (!id_to_bdf.count(device_id)) {
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
