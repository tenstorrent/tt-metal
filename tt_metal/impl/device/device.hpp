#pragma once

#include <memory>

#include "tt_metal/impl/allocator/basic_allocator.hpp"
#include "tt_metal/impl/allocator/l1_banking_allocator.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt {

namespace tt_metal {

// Represents all cores within range specified by the two cores
using CoreRange = std::pair<CoreCoord, CoreCoord>;
using CoreBlocks = std::vector<std::variant<CoreCoord, CoreRange>>;

template<class... Ts> struct overloaded_core : Ts... { using Ts::operator()...; };
template<class... Ts> overloaded_core(Ts...) -> overloaded_core<Ts...>;

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

    CoreCoord logical_core_from_bank_id(uint32_t bank_id) const;

    std::vector<uint32_t> bank_ids_from_dram_channel(uint32_t dram_channel) const;

    std::vector<uint32_t> bank_ids_from_logical_core(const CoreCoord &logical_core) const;

   private:
    bool cluster_is_initialized() const { return cluster_ != nullptr; }
    void check_allocator_is_initialized() const;

    // Checks that the given arch is on the given pci_slot and that it's responding
    // Puts device into reset
    bool initialize(const MemoryAllocator &memory_allocator=MemoryAllocator::BASIC);
    friend bool InitializeDevice(Device *device, const MemoryAllocator &memory_allocator);
    void initialize_cluster();
    void initialize_allocator(const MemoryAllocator &memory_allocator=MemoryAllocator::BASIC);

    // Puts device into reset
    bool close();
    friend bool CloseDevice(Device *device);

    friend class Buffer;
    friend class CircularBuffer;
    friend std::vector<CircularBuffer *> CreateCircularBuffers(
        Program &program,
        Device *device,
        uint32_t buffer_index,
        const CoreRange &core_range,
        uint32_t num_tiles,
        uint32_t size_in_bytes,
        DataFormat data_format
    );

    static constexpr TargetDevice target_type_ = TargetDevice::Silicon;
    tt::ARCH arch_;
    tt_cluster *cluster_;
    int pcie_slot_;
    std::unique_ptr<Allocator> allocator_;
    MemoryAllocator allocator_scheme_;
    bool closed_;
};

}  // namespace tt_metal

}  // namespace tt
