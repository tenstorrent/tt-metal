#pragma once

#include "tt_metal/impl/buffers/buffer.hpp"
#include "tt_metal/impl/buffers/circular_buffer.hpp"
#include "tt_metal/impl/buffers/semaphore.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/impl/kernels/kernel.hpp"

namespace tt {

namespace tt_metal {

struct KernelGroup {
    ComputeKernel *compute = nullptr;
    DataMovementKernel *riscv_0 = nullptr;
    DataMovementKernel *riscv_1 = nullptr;
};

class Program {
   public:
    Program() {}

    Program(const Program &other) = delete;
    Program& operator=(const Program &other) = delete;

    Program(Program &&other);
    Program& operator=(Program &&other);

    ~Program();

    std::vector<Kernel *> kernels() const { return kernels_; }

    std::vector<CircularBuffer *> circular_buffers() const { return circular_buffers_; }

    std::vector<ComputeKernel *> compute_kernels() const;

    std::vector<DataMovementKernel *> data_movement_kernels() const;

    KernelGroup kernels_on_core(const CoreCoord &core) const;

    std::map<CoreCoord, KernelGroup> core_to_kernel_group() const;

    std::vector<CircularBuffer *> circular_buffers_on_core(const CoreCoord &core) const;

    std::vector<Semaphore *> semaphores_on_core(const CoreCoord &core) const;

    std::vector<CoreCoord> logical_cores() const;

    std::string core_to_op(const CoreCoord &core) const;

    std::vector<std::string> cores_to_ops() const;

   private:
    std::vector<Kernel *> kernels_;
    std::vector<CircularBuffer *> circular_buffers_;
    std::unordered_map<CoreCoord, std::vector<Semaphore *>> logical_core_to_semaphores_;

    friend DataMovementKernel *CreateDataMovementKernel(
        Program &program,
        const std::string &file_name,
        const CoreCoord &core,
        const KernelArgs &kernel_args,
        DataMovementProcessor processor_type,
        NOC noc);

    friend DataMovementKernel *CreateDataMovementKernel(
        Program &program,
        const std::string &file_name,
        const CoreCoord &core,
        DataMovementProcessor processor_type,
        NOC noc);

    friend ComputeKernel *CreateComputeKernel(
        Program &program,
        const std::string &file_name,
        const CoreCoord &core,
        const KernelArgs &kernel_args,
        MathFidelity math_fidelity,
        bool fp32_dest_acc_en,
        bool math_approx_mode);

    friend DataMovementKernel *CreateDataMovementKernel(
        Program &program,
        const std::string &file_name,
        const CoreRange &core_range,
        const KernelArgs &kernel_args,
        DataMovementProcessor processor_type,
        NOC noc);

    friend DataMovementKernel *CreateDataMovementKernel(
        Program &program,
        const std::string &file_name,
        const CoreRange &core_range,
        DataMovementProcessor processor_type,
        NOC noc);

    friend ComputeKernel *CreateComputeKernel(
        Program &program,
        const std::string &file_name,
        const CoreRange &core_range,
        const KernelArgs &kernel_args,
        MathFidelity math_fidelity,
        bool fp32_dest_acc_en,
        bool math_approx_mode);

    friend CircularBuffer *CreateCircularBuffer(
        Program &program,
        Device *device,
        uint32_t buffer_id,
        const CoreCoord &core,
        uint32_t num_tiles,
        uint32_t size_in_bytes,
        uint32_t l1_address,
        DataFormat data_format);

    friend CircularBuffer *CreateCircularBuffer(
        Program &program,
        Device *device,
        uint32_t buffer_index,
        const CoreCoord &core,
        uint32_t num_tiles,
        uint32_t size_in_bytes,
        DataFormat data_format);

    friend std::vector<CircularBuffer *> CreateCircularBuffers(
        Program &program,
        Device *device,
        uint32_t buffer_index,
        const CoreRange &core_range,
        uint32_t num_tiles,
        uint32_t size_in_bytes,
        uint32_t l1_address,
        DataFormat data_format);

    friend std::vector<CircularBuffer *> CreateCircularBuffers(
        Program &program,
        Device *device,
        uint32_t buffer_index,
        const CoreRange &core_range,
        uint32_t num_tiles,
        uint32_t size_in_bytes,
        DataFormat data_format);

    friend std::vector<Semaphore *> CreateSemaphores(Program &program, Device *device, const CoreRange &core_range, uint32_t initial_value);

    void add_kernel(Kernel *kernel) { kernels_.push_back(kernel); }

    void add_circular_buffer(CircularBuffer *circular_buffer) { circular_buffers_.push_back(circular_buffer); }

    void add_semaphore(Semaphore *semaphore) { logical_core_to_semaphores_[semaphore->logical_core()].push_back(semaphore); }
};

}  // namespace tt_metal

}  // namespace tt
