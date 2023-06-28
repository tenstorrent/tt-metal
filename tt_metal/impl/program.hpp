#pragma once

#include <optional>

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

template <typename T, uint32_t NUM>
using FixedSlots = std::array<std::optional<T>, NUM>;

class Program {
   public:
    Program();

    Program(const Program &other) = delete;
    Program& operator=(const Program &other) = delete;

    Program(Program &&other) = default;
    Program& operator=(Program &&other) = default;

    ~Program();

    const u64 get_id() const { return this->id; }

    std::vector<Kernel *> kernels() const { return kernels_; }

    std::vector<CircularBuffer *> circular_buffers() const { return circular_buffers_; }

    std::vector<Semaphore *> semaphores() const { return semaphores_; }

    std::vector<ComputeKernel *> compute_kernels() const;

    std::vector<DataMovementKernel *> data_movement_kernels() const;

    KernelGroup kernels_on_core(const CoreCoord &core) const;

    std::map<CoreCoord, KernelGroup> core_to_kernel_group() const;

    std::vector<CircularBuffer *> circular_buffers_on_core(const CoreCoord &core) const;

    std::vector<Semaphore *> semaphores_on_core(const CoreCoord &core) const;

    std::vector<CoreCoord> logical_cores() const;

    CoreRangeSet logical_core_range_set() const;

    std::vector<std::string> cores_to_ops() const;

   private:
    u64 id; // Need to make non-const due to move constructor
    static std::atomic<u64> program_counter;
    std::vector<Kernel *> kernels_;
    std::vector<CircularBuffer *> circular_buffers_;
    std::vector<Semaphore *> semaphores_;

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

    friend CircularBuffer *CreateCircularBuffers(
        Program &program,
        Device *device,
        const std::set<uint32_t> &buffer_indices,
        const CoreRangeSet &core_range_set,
        uint32_t num_tiles,
        uint32_t size_in_bytes,
        uint32_t l1_address,
        DataFormat data_format);

    friend CircularBuffer *CreateCircularBuffers(
        Program &program,
        Device *device,
        const std::set<uint32_t> &buffer_indices,
        const CoreRangeSet &core_range_set,
        uint32_t num_tiles,
        uint32_t size_in_bytes,
        DataFormat data_format);

    friend Semaphore *CreateSemaphore(Program &program, Device *device, const CoreRange &core_range, uint32_t initial_value);

    friend Semaphore *CreateSemaphore(Program &program, Device *device, const CoreRangeSet &core_range_set, uint32_t initial_value);

    void add_kernel(Kernel *kernel) { kernels_.push_back(kernel); }

    void add_circular_buffer(CircularBuffer *circular_buffer);

    void add_semaphore(Semaphore *semaphore) { semaphores_.push_back(semaphore); }
};

}  // namespace tt_metal

}  // namespace tt
