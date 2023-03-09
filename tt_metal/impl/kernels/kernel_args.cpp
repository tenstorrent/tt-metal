#include "tt_metal/impl/kernels/kernel_args.hpp"

#include "common/utils.hpp"
#include "build_kernels_for_riscv/hlk_desc.hpp"

namespace tt {

namespace tt_metal {

DataMovementKernelArgs::DataMovementKernelArgs(const tt_xy_pair &logical_core, const std::vector<uint32_t> &compile_time_args) {
    core_to_compile_time_args_.insert({logical_core, compile_time_args});
}

void DataMovementKernelArgs::set_kernel_args_map(const CoreBlocks &core_blocks, const std::vector<std::vector<uint32_t>> &args_spec, bool set_compile_time_args) {
    for (auto index = 0; index < core_blocks.size(); index++) {
        auto core = core_blocks.at(index);
        auto args = args_spec.at(index);

        std::visit(overloaded_core {
            [this, args, set_compile_time_args](tt_xy_pair single_core) {
                if (set_compile_time_args) {
                    this->core_to_compile_time_args_.insert({single_core, args});
                } else {
                    this->core_to_runtime_args_.insert({single_core, args});
                }
            },
            [this, args, set_compile_time_args](CoreRange core_range) {
                auto start_core = core_range.first;
                auto end_core = core_range.second;
                for (auto x = start_core.x; x <= end_core.x; x++) {
                    for (auto y = start_core.y; y <= end_core.y; y++) {
                        auto core_in_range = tt_xy_pair(x, y);
                        if (set_compile_time_args) {
                            this->core_to_compile_time_args_.insert({core_in_range, args});
                        } else {
                            this->core_to_runtime_args_.insert({core_in_range, args});
                        }
                    }
                }
            }
        }, core);
    }
}

DataMovementKernelArgs::DataMovementKernelArgs(
    const CoreBlocks &core_blocks,
    const std::vector<std::vector<uint32_t>> &compile_time_args_spec) {
    TT_ASSERT(core_blocks.size() == compile_time_args_spec.size());
    set_kernel_args_map(core_blocks, compile_time_args_spec, /*set_compile_time_args=*/true);
}

std::vector<uint32_t> DataMovementKernelArgs::compile_time_args(const tt_xy_pair &logical_core) const {
    if (core_to_compile_time_args_.find(logical_core) != core_to_compile_time_args_.end()) {
        return core_to_compile_time_args_.at(logical_core);
    }
    return {};
}

std::vector<uint32_t> DataMovementKernelArgs::runtime_args(const tt_xy_pair &logical_core) const {
    if (core_to_runtime_args_.find(logical_core) != core_to_runtime_args_.end()) {
        return core_to_runtime_args_.at(logical_core);
    }
    return {};
}

void DataMovementKernelArgs::set_runtime_args(const tt_xy_pair &logical_core, const std::vector<uint32_t> &runtime_args) {
    core_to_runtime_args_.insert_or_assign(logical_core, runtime_args);
}

ComputeKernelArgs::ComputeKernelArgs(const tt_xy_pair &logical_core, const std::vector<uint32_t> &compile_time_args) {
    core_to_compile_time_args_.insert({logical_core, compile_time_args});
}

ComputeKernelArgs::ComputeKernelArgs(const CoreBlocks &core_blocks, const std::vector<std::vector<uint32_t>> &compile_time_args_spec) {
    TT_ASSERT(core_blocks.size() == compile_time_args_spec.size());
    for (auto index = 0; index < core_blocks.size(); index++) {
        auto core = core_blocks.at(index);
        auto &compile_time_args = compile_time_args_spec.at(index);

        std::visit(overloaded_core {
            [this, compile_time_args](tt_xy_pair single_core) {
                this->core_to_compile_time_args_.insert({single_core, compile_time_args});
            },
            [this, compile_time_args](CoreRange core_range) {
                auto start_core = core_range.first;
                auto end_core = core_range.second;
                for (auto x = start_core.x; x <= end_core.x; x++) {
                    for (auto y = start_core.y; y <= end_core.y; y++) {
                        auto core_in_range = tt_xy_pair(x, y);
                        this->core_to_compile_time_args_.insert({core_in_range, compile_time_args});
                    }
                }
            }
        }, core);
    }
}

vector<uint32_t> ComputeKernelArgs::compile_time_args(const tt_xy_pair &logical_core) const {
    if (core_to_compile_time_args_.find(logical_core) != core_to_compile_time_args_.end()) {
        return core_to_compile_time_args_.at(logical_core);
    }
    return {};
}

size_t DataMovementKernelArgsHash::operator()(const DataMovementKernelArgs& dmk_args) const {
    return tt::utils::vector_hash<uint32_t>{}(dmk_args.compile_time_args(logical_core));
}

size_t ComputeKernelArgsHash::operator()(const ComputeKernelArgs &ck_args) const {
        return tt::utils::vector_hash<uint32_t>{}(ck_args.compile_time_args(logical_core));
}

size_t ComputeKernelDefinesHash::operator()(const std::map<std::string, std::string> &c_defines) const {
    size_t hash_value = 0;
    for (auto it = c_defines.begin(); it != c_defines.end(); ++it)
        boost::hash_combine(hash_value, std::hash<std::string>{}(it->first + it->second));
    return hash_value;
}


}  // namespace tt_metal

}  // namespace tt
