#pragma once

#include <memory>

#include "hostdevcommon/common_values.hpp"
#include "tt_metal/impl/allocator/basic_allocator.hpp"
#include "tt_metal/impl/allocator/l1_banking_allocator.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt {

namespace tt_metal {

// Fwd declares
enum class BufferType;
class CircularBuffer;
class Buffer;
class Program;

// A physical PCIexpress Tenstorrent device
class Device {
   public:

    friend void tt_gdb(Device* device, int chip_id, const vector<CoreCoord> cores, vector<string> ops);
    Device(tt::ARCH arch, int pcie_slot) : arch_(arch), cluster_(nullptr), pcie_slot_(pcie_slot), closed_(false), allocator_scheme_(MemoryAllocator::BASIC) {}

    ~Device();

    // TODO: Add copy/move semantics
    Device(const Device &other) { }
    Device& operator=(const Device &other) { return *this; }

    Device(Device &&other) { }
    Device& operator=(Device &&other) { return *this; }

    tt::ARCH arch() const { return arch_; }

    int pcie_slot() const { return pcie_slot_; }

    MemoryAllocator allocator_scheme() const { return this->allocator_scheme_; }

    tt_cluster *cluster() const;  // Need to access cluster in llrt APIs

    int num_dram_channels() const;

    uint32_t l1_size() const;

    CoreCoord logical_grid_size() const;

    CoreCoord compute_and_storage_grid_size() const;

    CoreCoord worker_core_from_logical_core(const CoreCoord &logical_core) const;

    std::vector<CoreCoord> worker_cores_from_logical_cores(const std::vector<CoreCoord> &logical_cores);

    uint32_t num_banks(const BufferType &buffer_type) const;

    uint32_t dram_channel_from_bank_id(uint32_t bank_id) const;

    CoreCoord core_from_dram_channel(uint32_t dram_channel) const;

    i32 l1_bank_offset_from_bank_id(uint32_t bank_id) const;

    CoreCoord logical_core_from_bank_id(uint32_t bank_id) const;

    std::vector<uint32_t> bank_ids_from_dram_channel(uint32_t dram_channel) const;

    std::vector<uint32_t> bank_ids_from_logical_core(const CoreCoord &logical_core) const;

   private:
    bool cluster_is_initialized() const { return cluster_ != nullptr; }
    void check_allocator_is_initialized() const;

    // Checks that the given arch is on the given pci_slot and that it's responding
    // Puts device into reset
    bool initialize(const MemoryAllocator &memory_allocator=MemoryAllocator::BASIC, const std::vector<uint32_t>& l1_bank_remap = {});
    friend bool InitializeDevice(Device *device, const MemoryAllocator &memory_allocator);
    void initialize_cluster();
    void initialize_allocator(const MemoryAllocator &memory_allocator=MemoryAllocator::BASIC, const std::vector<uint32_t>& l1_bank_remap = {});
    void initialize_harvesting_information();
    // Puts device into reset
    bool close();
    friend bool CloseDevice(Device *device);

    friend class Buffer;
    friend class CircularBuffer;

    static constexpr TargetDevice target_type_ = TargetDevice::Silicon;
    tt::ARCH arch_;
    tt_cluster *cluster_;
    int pcie_slot_;
    std::unique_ptr<Allocator> allocator_;
    MemoryAllocator allocator_scheme_;
    bool closed_;

    bool harvesting_initialized_ = false;
    CoreCoord post_harvested_worker_grid_size_ = {};
    std::unordered_map<CoreCoord, CoreCoord> logical_to_routing_coord_lookup_table_ = {};
    unsigned int num_harvested_rows_ = 0;
};

}  // namespace tt_metal

}  // namespace tt
