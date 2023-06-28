#include "tt_metal/impl/program.hpp"

namespace tt {

namespace tt_metal {

std::atomic<u64> Program::program_counter = 0;

Program::Program(): id(program_counter++) {}

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

void Program::add_circular_buffer(CircularBuffer *circular_buffer) {
    std::map<CoreCoord, FixedSlots<CircularBuffer *, NUM_CIRCULAR_BUFFERS>> per_core_cb_config;
    for (auto core_range : circular_buffer->core_range_set().ranges()) {
        for (auto x = core_range.start.x; x <= core_range.end.x; x++) {
            for (auto y = core_range.start.y; y <= core_range.end.y; y++) {
                CoreCoord logical_core(x, y);

                auto cbs_on_core = this->circular_buffers_on_core(logical_core);
                for (auto cb_on_core : cbs_on_core) {
                    for (auto existing_buffer_index : cb_on_core->buffer_indices()) {
                        per_core_cb_config[logical_core][existing_buffer_index] = cb_on_core;
                    }
                }

                for (auto buffer_index : circular_buffer->buffer_indices()) {
                    log_assert(not per_core_cb_config[logical_core][buffer_index].has_value(), "Circular buffer index {} already exists on core {}", buffer_index, logical_core.str());
                }
            }
        }
    }

    circular_buffers_.push_back(circular_buffer);
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

CoreRangeSet Program::logical_core_range_set() const {
    CoreRangeSet s({});
    for (auto kernel : kernels_ )
    {
        s.merge ( kernel->core_range_set());
    }
    return s;
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
