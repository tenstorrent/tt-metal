#include <string>
#include "tt_metal/host_api.hpp"

std::string bufferTypeToString(tt::tt_metal::BufferType type) {
    switch (type) {
        case tt::tt_metal::BufferType::DRAM:
            return "DRAM";
        case tt::tt_metal::BufferType::L1:
            return "L1";
        default:
            return "UNKNOWN";
    }
}

int main() {
    constexpr int device_id = 0;

    // Create Device
    tt::tt_metal::Device *device = CreateDevice(device_id);

    // Get Arch Info
    tt::ARCH arch = device->arch();
    log_info(tt::LogTest, "Arch {}", arch);

    // Get Core Grid Size
    CoreCoord logical_grid_size = device->logical_grid_size();
    CoreCoord compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    log_info(tt::LogTest, "logical grid size {}", logical_grid_size);
    log_info(tt::LogTest, "compute with storage grid size {}", compute_with_storage_grid_size);

    // Get Memory Info
    int num_dram_channels = device->num_dram_channels();
    uint32_t l1_size_per_core = device->l1_size_per_core();
    uint32_t dram_size_per_channel = device->dram_size_per_channel();
    log_info(tt::LogTest, "num_dram_channels {}, dram_size_per_channel {}, l1_size_per_core {}", num_dram_channels, dram_size_per_channel, l1_size_per_core);

    uint32_t num_dram_banks = device->num_banks(tt::tt_metal::BufferType::DRAM);
    uint32_t num_l1_banks = device->num_banks(tt::tt_metal::BufferType::L1);
    uint32_t dram_bank_size  = device->bank_size(tt::tt_metal::BufferType::DRAM);
    uint32_t l1_bank_size = device->bank_size(tt::tt_metal::BufferType::L1);
    log_info(tt::LogTest, "Number of banks for type {} is {} and each bank size is {} bytes ({:.3f} MBs)", bufferTypeToString(tt::tt_metal::BufferType::DRAM), num_dram_banks, dram_bank_size, static_cast<double>(dram_bank_size) / (1024 * 1024));
    log_info(tt::LogTest, "Number of banks for type {} is {} and each bank size is {} bytes ({:.3f} MBs)", bufferTypeToString(tt::tt_metal::BufferType::L1), num_l1_banks, l1_bank_size, static_cast<double>(l1_bank_size) / (1024 * 1024));

    #if 0
    for (int i = 0; i < num_dram_banks; ++i) {
        uint32_t dram_channel = device->dram_channel_from_bank_id(i);
        log_info(tt::LogTest, "dram_channel {} corresponds to bank id {}", dram_channel, i);
    }
    #endif

    #if 0
    for (int x = 0; x < logical_grid_size.x; ++x) {
        for (int y = 0; y < logical_grid_size.y; ++y) {
            CoreCoord logical_core {x, y};
            CoreCoord physical_core = device->worker_core_from_logical_core(logical_core);
            log_info(tt::LogTest, "Logical core ({}, {}) mapped to physical core ({}, {})",
                 logical_core.x, logical_core.y, physical_core.x, physical_core.y);
        }
    }
    #endif

    CloseDevice(device);
    return 0;
}
