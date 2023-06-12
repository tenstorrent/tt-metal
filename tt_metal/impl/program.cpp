#include "tt_metal/impl/program.hpp"

namespace tt {

namespace tt_metal {

Program::Program(Program &&other)
    : kernels_(other.kernels_), circular_buffers_(other.circular_buffers_), semaphores_(other.semaphores_) {
        other.kernels_.clear();
        other.circular_buffers_.clear();
        other.semaphores_.clear();
}

Program &Program::operator=(Program &&other) {
    if (this != &other) {
        this->kernels_ = other.kernels_;
        this->circular_buffers_ = other.circular_buffers_;
        this->semaphores_ = other.semaphores_;
        other.kernels_.clear();
        other.circular_buffers_.clear();
        other.semaphores_.clear();
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

KernelGroup Program::kernels_on_core(const CoreCoord &core) const {
    KernelGroup kernel_group;
    for (auto kernel : kernels_) {
        auto cores = kernel->logical_cores();
        if (std::find(cores.begin(), cores.end(), core) != cores.end()) {
            populate_kernel_group(kernel_group, kernel);
        }
    }
    return kernel_group;
}

std::map<CoreCoord, KernelGroup> Program::core_to_kernel_group() const {
    std::map<CoreCoord, KernelGroup> core_to_kernel_group;

    for (auto kernel : kernels_) {
        for (auto core : kernel->logical_cores()) {
            KernelGroup &kernel_group = core_to_kernel_group[core];
            populate_kernel_group(kernel_group, kernel);
        }
    }

    return core_to_kernel_group;
}

std::vector<std::string> Program::cores_to_ops() const {
    std::vector<std::string> ops;

    for (const auto &core : this->logical_cores()) {
        for (auto kernel : kernels_) {
        auto cores = kernel->logical_cores();
            if (std::find(cores.begin(), cores.end(), core) != cores.end()) {
                ops.push_back(kernel->name());
            }
        }
    }
    return ops;
}

std::vector<CircularBuffer *> Program::circular_buffers_on_core(const CoreCoord &core) const {
    std::vector<CircularBuffer *> cbs_on_core;
    for (auto circular_buffer : circular_buffers_) {
        if (circular_buffer->is_on_logical_core(core)) {
            cbs_on_core.push_back(circular_buffer);
        }
    }
    return cbs_on_core;
}

std::vector<Semaphore *> Program::semaphores_on_core(const CoreCoord &core) const {
    std::vector<Semaphore *> semaphores;
    for (auto semaphore : this->semaphores_) {
        if (semaphore->initialized_on_logical_core(core)) {
            semaphores.push_back(semaphore);
        }
    }
    return semaphores;
}

std::vector<CoreCoord> Program::logical_cores() const {
    std::vector<CoreCoord> cores_in_program;
    std::set<CoreCoord> unique_cores;
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

void validate_runtime_args_size(const CoreCoord &logical_core, const RISCV &riscv, const std::vector<uint32_t> &runtime_args) {
    uint32_t runtime_args_size = runtime_args.size() * sizeof(uint32_t);
    uint64_t l1_arg_base;
    uint64_t result_base;
    switch (riscv) {
        case RISCV::BRISC:
            l1_arg_base = BRISC_L1_ARG_BASE;
            result_base = BRISC_L1_RESULT_BASE;
            break;
        case RISCV::NCRISC:
            l1_arg_base = NCRISC_L1_ARG_BASE;
            result_base = NCRISC_L1_RESULT_BASE;
            break;
        default:
            TT_ASSERT(false, "Only BRISC and NCRISC have runtime arg support");
    }

    std::stringstream identifier;
    identifier << riscv;
    if (l1_arg_base + runtime_args_size >= result_base) {
        TT_THROW(std::to_string(runtime_args_size / 1024) + "KB " + identifier.str()  + " runtime args targeting " + logical_core.str() + " are too large.\
            Cannot be written as they will run into memory region reserved for result. Max allowable size is " + std::to_string((result_base - l1_arg_base)/1024) + " KB.");
    }
}

void Program::set_runtime_args(const CoreCoord &logical_core, const RISCV &riscv, const std::vector<uint32_t> &runtime_args) {
    TT_ASSERT(riscv == RISCV::BRISC or riscv == RISCV::NCRISC, "Compute kernels do not support runtime args");
    std::vector<uint32_t> rt_args = runtime_args;
    // dumpDeviceProfiler needs to know core coordinates to know what cores to dump from
    rt_args.push_back(logical_core.x);
    rt_args.push_back(logical_core.y);

    auto validate_runtime_args_size = [&]() {
        uint32_t runtime_args_size = runtime_args.size() * sizeof(uint32_t);
        uint64_t l1_arg_base;
        uint64_t result_base;
        switch (riscv) {
            case RISCV::BRISC:
                l1_arg_base = BRISC_L1_ARG_BASE;
                result_base = BRISC_L1_RESULT_BASE;
                break;
            case RISCV::NCRISC:
                l1_arg_base = NCRISC_L1_ARG_BASE;
                result_base = NCRISC_L1_RESULT_BASE;
                break;
            default:
                TT_ASSERT(false, "Only BRISC and NCRISC have runtime arg support");
        }
        std::stringstream identifier;
        identifier << riscv;
        if (l1_arg_base + runtime_args_size >= result_base) {
            TT_THROW(std::to_string(runtime_args_size / 1024) + "KB " + identifier.str()  + " runtime args targeting " + logical_core.str() + " are too large.\
                Cannot be written as they will run into memory region reserved for result. Max allowable size is " + std::to_string((result_base - l1_arg_base)/1024) + " KB.");
        }
    };

    validate_runtime_args_size();
    this->core_to_runtime_args_[logical_core][riscv] = rt_args;
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
