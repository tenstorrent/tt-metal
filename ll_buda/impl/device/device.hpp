#pragma once

#include "ll_buda/impl/device/memory_manager.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt {

namespace ll_buda {

// Represents all cores within range specified by the two cores
using CoreRange = std::pair<tt_xy_pair, tt_xy_pair>;
using CoreBlocks = std::vector<std::variant<tt_xy_pair, CoreRange>>;

template<class... Ts> struct overloaded_core : Ts... { using Ts::operator()...; };
template<class... Ts> overloaded_core(Ts...) -> overloaded_core<Ts...>;

class DramBuffer;

// A physical PCIexpress Tenstorrent device
class Device {
   public:

    friend void tt_gdb(Device* device, int chip_id, const vector<tt_xy_pair> cores, vector<string> ops);
    Device(tt::ARCH arch, int pcie_slot) : arch_(arch), cluster_(nullptr), pcie_slot_(pcie_slot) {}

    tt::ARCH arch() const { return arch_; }

    int pcie_slot() const { return pcie_slot_; }

    tt_cluster *cluster() const { return cluster_; }  // Need to access cluster in llrt APIs

    int num_dram_banks() const;

    tt_xy_pair worker_core_from_logical_core(const tt_xy_pair &logical_core) const;

    std::vector<tt_xy_pair> worker_cores_from_logical_cores(const std::vector<tt_xy_pair> &logical_cores);

    uint32_t allocate_buffer(int dram_channel, uint32_t size_in_bytes);

    uint32_t allocate_buffer(int dram_channel, uint32_t size_in_bytes, uint32_t address);

   private:
    bool cluster_is_initialized() const { return cluster_ != nullptr; }
    bool is_initialized() const { return cluster_is_initialized() and not banked_dram_manager_.empty(); }

    // Checks that the given arch is on the given pci_slot and that it's responding
    // Puts device into reset
    bool initialize();
    friend bool InitializeDevice(Device *device);
    void initialize_cluster();
    void initialize_banked_dram_manager();
    
    // Puts device into reset
    bool close();
    friend bool CloseDevice(Device *device);

    friend std::vector<DramBuffer *> CreateInterleavedDramBuffers(
        Device *device, int num_bank_units, int num_entries_per_bank_unit, int num_bytes_per_entry);

    static constexpr TargetDevice target_type_ = TargetDevice::Silicon;
    tt::ARCH arch_;
    tt_cluster *cluster_;
    int pcie_slot_;
    std::unordered_map<int, MemoryManager *> banked_dram_manager_;
};

}  // namespace ll_buda

}  // namespace tt
