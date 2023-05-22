#include "tt_metal/impl/program.hpp"

namespace tt {

namespace tt_metal {

Program::Program(Program &&other)
    : kernels_(other.kernels_), circular_buffers_(other.circular_buffers_), logical_core_to_semaphores_(other.logical_core_to_semaphores_) {
        other.kernels_.clear();
        other.circular_buffers_.clear();
        other.logical_core_to_semaphores_.clear();
}

Program &Program::operator=(Program &&other) {
    if (this != &other) {
        this->kernels_ = other.kernels_;
        this->circular_buffers_ = other.circular_buffers_;
        this->logical_core_to_semaphores_ = other.logical_core_to_semaphores_;
        other.kernels_.clear();
        other.circular_buffers_.clear();
        other.logical_core_to_semaphores_.clear();
    }
    return *this;
}

std::vector<ComputeKernel *> Program::compute_kernels() const {
    std::vector<ComputeKernel *> compute_kernels;
    for (auto kernel : kernels_) {
        if (auto compute_kernel = dynamic_cast<ComputeKernel *>(kernel)) {
            compute_kernels.push_back(compute_kernel);
        }
    }
    return compute_kernels;
}

std::vector<DataMovementKernel *> Program::data_movement_kernels() const {
    std::vector<DataMovementKernel *> data_movement_kernels;
    for (auto kernel : kernels_) {
        if (auto data_movement_kernel = dynamic_cast<DataMovementKernel *>(kernel)) {
            data_movement_kernels.push_back(data_movement_kernel);
        }
    }
    return data_movement_kernels;
}

void populate_kernel_group(KernelGroup &kernel_group, Kernel *kernel) {
    if (auto compute_kernel = dynamic_cast<ComputeKernel *>(kernel)) {
        kernel_group.compute = compute_kernel;
    } else if (auto dm_kernel = dynamic_cast<DataMovementKernel *>(kernel)) {
        if (dm_kernel->data_movement_processor() == DataMovementProcessor::RISCV_0) {
            kernel_group.riscv_0 = dm_kernel;
        } else {
            kernel_group.riscv_1 = dm_kernel;
        }
    }
}

KernelGroup Program::kernels_on_core(const tt_xy_pair &core) const {
    KernelGroup kernel_group;
    for (auto kernel : kernels_) {
        auto cores = kernel->logical_cores();
        if (std::find(cores.begin(), cores.end(), core) != cores.end()) {
            populate_kernel_group(kernel_group, kernel);
        }
    }
    return kernel_group;
}

std::map<tt_xy_pair, KernelGroup> Program::core_to_kernel_group() const {
    std::map<tt_xy_pair, KernelGroup> core_to_kernel_group;

    for (auto kernel : kernels_) {
        for (auto core : kernel->logical_cores()) {
            KernelGroup &kernel_group = core_to_kernel_group[core];
            populate_kernel_group(kernel_group, kernel);
        }
    }

    return core_to_kernel_group;
}

std::string Program::core_to_op(const tt_xy_pair &core) const {
    for (auto kernel : kernels_) {
        auto cores = kernel->logical_cores();
        if (std::find(cores.begin(), cores.end(), core) != cores.end()) {
            std::string bin_path = kernel->binary_path(core);
            size_t bin_path_size = bin_path.size();


            // The hash corresponds to string after last '/'
            int i;
            for (i = bin_path_size - 1; i > -1; i--) {
                if (bin_path.at(i) == '/') {
                    i--;
                    break;
                }
            }

            // The op name corresponds to string after second last '/'
            for (; i > -1; i--) {
                if (bin_path.at(i) == '/') {
                    std::string op = bin_path.substr(i + 1, bin_path_size - i - 1);
                    return op;
                }
            }
        }
    }
    return "";
}

std::vector<std::string> Program::cores_to_ops() const {
    std::vector<std::string> ops;

    for (const auto &core : this->logical_cores()) {
        ops.push_back(this->core_to_op(core));
    }
    return ops;
}

std::vector<CircularBuffer *> Program::circular_buffers_on_core(const tt_xy_pair &core) const {
    std::vector<CircularBuffer *> cbs_on_core;
    for (auto circular_buffer : circular_buffers_) {
        if (circular_buffer->logical_core() == core) {
            cbs_on_core.push_back(circular_buffer);
        }
    }
    return cbs_on_core;
}

std::vector<Semaphore *> Program::semaphores_on_core(const tt_xy_pair &core) const {
    if (this->logical_core_to_semaphores_.find(core) == this->logical_core_to_semaphores_.end()) {
        return {};
    }
    return this->logical_core_to_semaphores_.at(core);
}

std::vector<tt_xy_pair> Program::logical_cores() const {
    std::vector<tt_xy_pair> cores_in_program;
    std::set<tt_xy_pair> unique_cores;
    for (auto kernel : kernels_) {
        for (auto core : kernel->logical_cores()) {
            if (unique_cores.find(core) != unique_cores.end()) {
                continue;
            }
            unique_cores.insert(core);
            cores_in_program.push_back(core);
        }
    }
    return cores_in_program;
}

Program::~Program() {
    for (auto kernel : kernels_) {
        delete kernel;
    }
    for (auto circular_buffer : circular_buffers_) {
        delete circular_buffer;
    }
}

}  // namespace tt_metal

}  // namespace tt
