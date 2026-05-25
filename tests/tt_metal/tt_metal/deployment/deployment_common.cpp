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

static const std::vector<std::string>& get_tenstorrent_pci_bdfs_cached() {
    static const std::vector<std::string> bdfs = []() {
        std::vector<std::string> out;

        DIR* dir = opendir("/sys/bus/pci/devices");
        if (dir == nullptr) {
            return out;
        }

        while (auto* entry = readdir(dir)) {
            const std::string bdf = entry->d_name;
            if (bdf.empty() || bdf[0] == '.') {
                continue;
            }

            const std::string base = "/sys/bus/pci/devices/" + bdf;
            std::string vendor = read_text_file_trimmed(base + "/vendor");
            std::transform(
                vendor.begin(), vendor.end(), vendor.begin(), [](unsigned char c) { return std::tolower(c); });

            // Tenstorrent PCI vendor id. Sorted order gives a stable best-effort map:
            // runtime bdf={} device_id=0 -> first Tenstorrent BDF, bdf={} device_id=1 -> second, etc.
            if (vendor == "0x1e52") {
                out.push_back(bdf);
            }
        }

        closedir(dir);
        std::sort(out.begin(), out.end());
        return out;
    }();

    return bdfs;
}

std::string pci_bdf_for_device_id(uint32_t device_id) {
    const auto& bdfs = get_tenstorrent_pci_bdfs_cached();
    if (device_id < bdfs.size()) {
        return bdfs[device_id];
    }
    return "unknown";
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
