#pragma once

#include "common/tt_xy_pair.h"
#include "ll_buda/impl/device/device.hpp"

namespace tt {

namespace ll_buda {

class DataMovementKernelArgs {
   public:
    DataMovementKernelArgs() {}

    DataMovementKernelArgs(const tt_xy_pair &logical_core, const std::vector<uint32_t> &compile_time_args);
    
    DataMovementKernelArgs(const CoreBlocks &core_blocks, const std::vector<std::vector<uint32_t>> &compile_time_args_spec);

    std::vector<uint32_t> compile_time_args(const tt_xy_pair &logical_core) const;

    std::vector<uint32_t> runtime_args(const tt_xy_pair &logical_core) const;

    void set_runtime_args(const tt_xy_pair &logical_core, const std::vector<uint32_t> &runtime_args);

   private:
    void set_kernel_args_map(const CoreBlocks &core_blocks, const std::vector<std::vector<uint32_t>> &args_spec, bool set_compile_time_args);

    std::unordered_map<tt_xy_pair, std::vector<uint32_t>> core_to_compile_time_args_;
    std::unordered_map<tt_xy_pair, std::vector<uint32_t>> core_to_runtime_args_;
};

class ComputeKernelArgs {
   public:
    ComputeKernelArgs() : compile_time_args_size_(0) {}

    ComputeKernelArgs(const tt_xy_pair &logical_core, void *compile_time_args, size_t compile_time_args_size) : compile_time_args_size_(compile_time_args_size) {
        core_to_compile_time_args_.insert({logical_core, compile_time_args});
    }

    ComputeKernelArgs(const CoreBlocks &core_blocks, const std::vector<void *> &compile_time_args_spec, size_t compile_time_args_size);

    void *compile_time_args(const tt_xy_pair &logical_core) const;

    size_t compile_time_args_size(const tt_xy_pair &logical_core) const { return compile_time_args_size_; }

   private:
    std::unordered_map<tt_xy_pair, void *> core_to_compile_time_args_;
    size_t compile_time_args_size_;
};

struct DataMovementKernelArgsHash {
    DataMovementKernelArgsHash(const tt_xy_pair &core) : logical_core{core} { }

    size_t operator()(const DataMovementKernelArgs& dmk_args) const;

    tt_xy_pair logical_core;
};

struct ComputeKernelArgsHash {
    ComputeKernelArgsHash(const tt_xy_pair &core) : logical_core{core} { }

    size_t operator()(const ComputeKernelArgs &c_args) const;

    tt_xy_pair logical_core;
};

struct ComputeKernelDefinesHash {
    ComputeKernelDefinesHash(const tt_xy_pair &core) : logical_core{core} { }

    size_t operator()(const std::map<std::string, std::string> &c_defines) const;

    tt_xy_pair logical_core;
};

}  // namespace ll_buda

}  // namespace tt
