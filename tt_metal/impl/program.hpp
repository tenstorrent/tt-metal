#pragma once

#include <bitset>

#include "tt_metal/impl/buffers/buffer.hpp"
#include "tt_metal/impl/buffers/circular_buffer.hpp"
#include "tt_metal/impl/buffers/semaphore.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/impl/kernels/kernel.hpp"
#include "common/tt_backend_api_types.hpp"
#include "hostdevcommon/common_values.hpp"

namespace tt {

namespace tt_metal {

struct KernelGroup {
    ComputeKernel *compute = nullptr;
    DataMovementKernel *riscv_0 = nullptr;
    DataMovementKernel *riscv_1 = nullptr;
};

class Program {
   public:
    Program();

    Program(const Program &other) = delete;
    Program& operator=(const Program &other) = delete;

    Program(Program &&other) = default;
    Program& operator=(Program &&other) = default;

    ~Program();

    void construct_core_range_set_for_worker_cores();

    const u64 get_id() const { return this->id; }

    std::vector<Kernel *> kernels() const { return kernels_; }

    const std::vector<CircularBuffer> &circular_buffers() const { return circular_buffers_; }

    const std::vector< Semaphore > & semaphores() const { return semaphores_; }

    std::vector<ComputeKernel *> compute_kernels() const;

    std::vector<DataMovementKernel *> data_movement_kernels() const;

    KernelGroup kernels_on_core(const CoreCoord &core) const;

    std::map<CoreCoord, KernelGroup> core_to_kernel_group() const;

    const std::vector<CircularBuffer> circular_buffers_on_core(const CoreCoord &core) const;

    auto semaphores_on_core(const CoreCoord &core) const;

    size_t num_semaphores ( const CoreCoord & core ) const;
    size_t num_semaphores () const;
    uint32_t semaphore_address ( uint32_t sem_index ) const;
    void init_semaphores ( const Device & device, const CoreCoord &logical_core ) const;
    std::vector<CoreCoord> logical_cores() const;

    CoreRangeSet get_worker_core_range_set() const { return worker_crs_; };

    std::vector<std::string> cores_to_ops() const;

    void validate_circular_buffer_region(const Device *device, const CoreCoord &logical_core) const;

    void validate_circular_buffer_region(const Device *device) const;

   private:
    struct CircularBufferConfig {
        // Tracks which circular buffer indices are being used
        std::bitset<NUM_CIRCULAR_BUFFERS> indices;

        // Holds vector of addresses where circular buffers are allocated [start, end)
        // There are multiple ranges because per core L1 regions are not in lockstep but circular buffers spanning multiple cores must share the same address
        // To enable this, circular buffer address is the maximum address amongst all of its target cores
        // This vector is sorted from lower to higher address spaces
        std::vector<std::pair<u64, u64>> l1_regions = {{UNRESERVED_BASE, UNRESERVED_BASE}};

        // Sets `indices` at position index
        void add_index(u32 index);

        // Returns address for next circular buffer
        // Circular buffers are placed sequentially on a core so the next available address gets appended to the last L1 region
        u64 get_address_candidate() const;

        // If address is the end of the last L1 region, the last region is extended by size bytes,
        //  otherwise address must be higher than existing regions and a new L1 region [address, size) is added
        void mark_address(u64 address, u64 size);
    };

    u64 id; // Need to make non-const due to move constructor
    static std::atomic<u64> program_counter;
    std::vector<Kernel *> kernels_;
    std::vector<CircularBuffer> circular_buffers_;
    std::map<CoreCoord, CircularBufferConfig> per_core_cb_config_;
    std::vector<Semaphore> semaphores_;
    CoreRangeSet worker_crs_;

    friend DataMovementKernel *CreateDataMovementKernel(
        Program &program,
        const std::string &file_name,
        const CoreCoord &core,
        const std::vector<uint32_t> &compile_args,
        DataMovementProcessor processor_type,
        NOC noc);

    friend DataMovementKernel *CreateDataMovementKernel(
        Program &program,
        const std::string &file_name,
        const CoreCoord &core,
        DataMovementProcessor processor_type,
        NOC noc);

    friend DataMovementKernel *CreateDataMovementKernel(
        Program &program,
        const std::string &file_name,
        const CoreRange &core_range,
        const std::vector<uint32_t> &compile_args,
        DataMovementProcessor processor_type,
        NOC noc);

    friend DataMovementKernel *CreateDataMovementKernel(
        Program &program,
        const std::string &file_name,
        const CoreRange &core_range,
        DataMovementProcessor processor_type,
        NOC noc);

    friend DataMovementKernel *CreateDataMovementKernel(
        Program &program,
        const std::string &file_name,
        const CoreRangeSet &core_range_set,
        const std::vector<uint32_t> &compile_args,
        DataMovementProcessor processor_type,
        NOC noc);

    friend DataMovementKernel *CreateDataMovementKernel(
        Program &program,
        const std::string &file_name,
        const CoreRangeSet &core_range_set,
        DataMovementProcessor processor_type,
        NOC noc);

    friend ComputeKernel *CreateComputeKernel(
        Program &program,
        const std::string &file_name,
        const CoreCoord &core,
        const std::vector<uint32_t> &compile_args,
        MathFidelity math_fidelity,
        bool fp32_dest_acc_en,
        bool math_approx_mode);

    friend ComputeKernel *CreateComputeKernel(
        Program &program,
        const std::string &file_name,
        const CoreRange &core_range,
        const std::vector<uint32_t> &compile_args,
        MathFidelity math_fidelity,
        bool fp32_dest_acc_en,
        bool math_approx_mode);

    friend ComputeKernel *CreateComputeKernel(
        Program &program,
        const std::string &file_name,
        const CoreRangeSet &core_range_set,
        const std::vector<uint32_t> &compile_args,
        MathFidelity math_fidelity,
        bool fp32_dest_acc_en,
        bool math_approx_mode);

    friend const CircularBuffer &CreateCircularBuffers(
        Program &program,
        const std::set<uint32_t> &buffer_indices,
        const CoreRangeSet &core_range_set,
        uint32_t num_tiles,
        uint32_t size_in_bytes,
        DataFormat data_format,
        std::optional<uint32_t> l1_address);

    friend uint32_t CreateSemaphore(Program &program, const CoreRange &core_range, uint32_t initial_value);

    friend uint32_t CreateSemaphore(Program &program, const CoreRangeSet &core_range_set, uint32_t initial_value);

    void add_kernel(Kernel *kernel) { kernels_.push_back(kernel); }

    const CircularBuffer &add_circular_buffer(const CoreRangeSet &core_range_set, const std::set<u32> &indices, u32 num_tiles, u32 size_bytes, const DataFormat &data_format, std::optional<u32> address);

    void add_semaphore(const CoreRangeSet & crs, uint32_t address, uint32_t init_value);
};

}  // namespace tt_metal

}  // namespace tt
