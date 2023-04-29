#pragma once

#include "tt_metal/impl/buffers/buffer.hpp"
#include "tt_metal/impl/buffers/circular_buffer.hpp"
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

    ~Program();

    std::vector<Kernel *> kernels() const { return kernels_; }

    std::vector<L1Buffer *> l1_buffers() const { return l1_buffers_; }

    std::vector<CircularBuffer *> circular_buffers() const { return circular_buffers_; }

    std::vector<ComputeKernel *> compute_kernels() const;

    std::vector<DataMovementKernel *> data_movement_kernels() const;

    KernelGroup kernels_on_core(const tt_xy_pair &core) const;

    std::map<tt_xy_pair, KernelGroup> core_to_kernel_group() const;

    std::vector<CircularBuffer *> circular_buffers_on_core(const tt_xy_pair &core) const;

    std::vector<L1Buffer *> l1_buffers_on_core(const tt_xy_pair &core) const;

    std::vector<tt_xy_pair> logical_cores() const;

    std::string core_to_op(const tt_xy_pair &core) const;

    std::vector<std::string> cores_to_ops() const;

   private:
    std::vector<Kernel *> kernels_;
    std::vector<CircularBuffer *> circular_buffers_;
    std::vector<L1Buffer *> l1_buffers_;

    friend DataMovementKernel *CreateDataMovementKernel(
        Program *program,
        const std::string &file_name,
        const tt_xy_pair &core,
        DataMovementKernelArgs *kernel_args,
        DataMovementProcessor processor_type,
        NOC noc);

    friend DataMovementKernel *CreateDataMovementKernel(
        Program *program,
        const std::string &file_name,
        const tt_xy_pair &core,
        DataMovementProcessor processor_type,
        NOC noc);

    friend ComputeKernel *CreateComputeKernel(
        Program *program,
        const std::string &file_name,
        const tt_xy_pair &core,
        ComputeKernelArgs *kernel_args,
        MathFidelity math_fidelity,
        bool fp32_dest_acc_en,
        bool math_approx_mode);

    friend DataMovementKernel *CreateDataMovementKernel(
        Program *program,
        const std::string &file_name,
        const CoreRange &core_range,
        DataMovementKernelArgs *kernel_args,
        DataMovementProcessor processor_type,
        NOC noc);

    friend DataMovementKernel *CreateDataMovementKernel(
        Program *program,
        const std::string &file_name,
        const CoreRange &core_range,
        DataMovementProcessor processor_type,
        NOC noc);

    friend ComputeKernel *CreateComputeKernel(
        Program *program,
        const std::string &file_name,
        const CoreRange &core_range,
        ComputeKernelArgs *kernel_args,
        MathFidelity math_fidelity,
        bool fp32_dest_acc_en,
        bool math_approx_mode);

    friend L1Buffer *CreateL1Buffer(Program *program, Device *device, const tt_xy_pair &core, uint32_t size_in_bytes, uint32_t address);
    friend L1Buffer *CreateL1Buffer(Program *program, Device *device, const tt_xy_pair &core, uint32_t size_in_bytes);

    friend CircularBuffer *CreateCircularBuffer(
        Program *program,
        Device *device,
        uint32_t buffer_id,
        const tt_xy_pair &core,
        uint32_t num_tiles,
        uint32_t size_in_bytes,
        uint32_t l1_address,
        DataFormat data_format);

    friend CircularBuffer *CreateCircularBuffer(
        Program *program,
        Device *device,
        uint32_t buffer_index,
        const tt_xy_pair &core,
        uint32_t num_tiles,
        uint32_t size_in_bytes,
        DataFormat data_format);

    friend std::vector<CircularBuffer *> CreateCircularBuffers(
        Program *program,
        Device *device,
        uint32_t buffer_index,
        const CoreRange &core_range,
        uint32_t num_tiles,
        uint32_t size_in_bytes,
        uint32_t l1_address,
        DataFormat data_format);

    friend std::vector<CircularBuffer *> CreateCircularBuffers(
        Program *program,
        Device *device,
        uint32_t buffer_index,
        const CoreRange &core_range,
        uint32_t num_tiles,
        uint32_t size_in_bytes,
        DataFormat data_format);

    void add_kernel(Kernel *kernel) { kernels_.push_back(kernel); }

    void add_l1_buffer(L1Buffer *buffer) { l1_buffers_.push_back(buffer); }

    void add_circular_buffer(CircularBuffer *circular_buffer) { circular_buffers_.push_back(circular_buffer); }
};

}  // namespace tt_metal

}  // namespace tt
