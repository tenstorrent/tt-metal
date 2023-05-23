#pragma once

#include "common/core_coord.h"
#include "tt_metal/impl/device/device.hpp"

namespace tt {

namespace tt_metal {

// Fwd declares
class Device;
class DataMovementKernel;

/**
 *
 * @brief Kernel args are initialized with compile time arguments which are supplied to a kernel during compilation. @n
 * They need to be passed to APIs that create DataMovement or Compute kernels.
 *
 * */
class KernelArgs {
   public:
    KernelArgs() {}

    /**
     * @brief Construct a KernelArgs object to represent kernel args on a single core
     *
     * @param logical_core Logical Tensix core coordinate indicating where these args can be written
     * @param compile_time_args Arguments supplied to the kernel during compilation
     */
    KernelArgs(const CoreCoord &logical_core, const std::vector<uint32_t> &compile_time_args);

    /**
     * @brief Construct a KernelArgs object to represent shared kernel args across a range of cores
     *
     * @param core_range Range of logical Tensix core coordinates (inclusive) indicating where these args can be written
     * @param compile_time_args Arguments supplied to kernels in the core range during compilation
     */
    KernelArgs(const CoreRange &core_range, const std::vector<uint32_t> &compile_time_args);

    /**
     * @brief Construct a KernelArgs object to represent kernel args across multiple groups of cores that can have the same or different args
     *
     * @param core_blocks Vector holding CoreRanges and single logical Tensix core coordinates indicating where these args can be written
     * @param compile_time_args Arguments supplied to kernels in the core range during compilation. Order of compile time args indicates which core block the args apply to
     */
    KernelArgs(const CoreBlocks &core_blocks, const std::vector<std::vector<uint32_t>> &compile_time_args);

    KernelArgs(const KernelArgs &other);
    KernelArgs& operator=(const KernelArgs &other);

    KernelArgs(KernelArgs &&other);
    KernelArgs& operator=(KernelArgs &&other);

    std::vector<uint32_t> compile_time_args(const CoreCoord &logical_core) const;

    std::vector<uint32_t> runtime_args(const CoreCoord &logical_core) const;

   private:
    void set_kernel_args_map(const CoreBlocks &core_blocks, const std::vector<std::vector<uint32_t>> &args_spec, bool set_compile_time_args);

    void set_runtime_args(const CoreCoord &logical_core, const std::vector<uint32_t> &runtime_args);
    friend bool WriteRuntimeArgsToDevice(Device *device, DataMovementKernel *kernel, const CoreCoord &logical_core, const std::vector<uint32_t> &runtime_args);

    std::unordered_map<CoreCoord, std::vector<uint32_t>> core_to_compile_time_args_;
    std::unordered_map<CoreCoord, std::vector<uint32_t>> core_to_runtime_args_;
};

struct KernelArgsHash {
    KernelArgsHash(const CoreCoord &core) : logical_core{core} { }

    size_t operator()(const KernelArgs& args) const;

    CoreCoord logical_core;
};

struct KernelDefinesHash {
    KernelDefinesHash(const CoreCoord &core) : logical_core{core} { }

    size_t operator()(const std::map<std::string, std::string> &c_defines) const;

    CoreCoord logical_core;
};

}  // namespace tt_metal

}  // namespace tt
