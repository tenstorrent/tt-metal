#pragma once

#include <vector>
#include <map>

#include "build_kernels_for_riscv/build_kernel_options.hpp"
#include "common/base_types.hpp"
#include "tt_metal/impl/device/device.hpp"

namespace tt {

namespace tt_metal {

enum class DataMovementProcessor {
    RISCV_0 = 0,  // BRISC
    RISCV_1 = 1,  // NCRISC
};

enum NOC : uint8_t {
    RISCV_0_default = 0,
    RISCV_1_default = 1,
    NOC_0 = 0,
    NOC_1 = 1,
};

enum class KernelType {
    DataMovement = 0,  // reader / writer
    Compute = 1,       // unpack -> math -> pack
};

std::ostream& operator<<(std::ostream& os, const DataMovementProcessor& processor);
std::ostream& operator<<(std::ostream& os, const KernelType& type);

struct KernelGroup;

class Kernel {
   public:
    Kernel(const std::string &kernel_path_file_name, const CoreRangeSet &core_range_set, KernelType kernel_type) :
        kernel_path_file_name_(kernel_path_file_name), core_range_set_(core_range_set), kernel_type_(kernel_type) {}

    Kernel(const std::string &kernel_path_file_name, const CoreRangeSet &core_range_set, const std::vector<uint32_t> &compile_args, KernelType kernel_type) :
        kernel_path_file_name_(kernel_path_file_name), core_range_set_(core_range_set), compile_time_args_(compile_args), kernel_type_(kernel_type) {}

    virtual ~Kernel() {}

    std::string kernel_path_file_name() const { return kernel_path_file_name_; }

    std::string name() const;

    CoreRangeSet core_range_set() const { return core_range_set_; }

    std::set<CoreCoord> logical_cores() const;

    bool is_on_logical_core(const CoreCoord &logical_core) const;

    KernelType kernel_type() const { return kernel_type_; }

    std::string const &binary_path() const { return binary_path_; }

    std::vector<ll_api::memory> const &binaries() const;

    std::vector<uint32_t> compile_time_args() const { return compile_time_args_; }

    size_t compile_time_args_hash() const;

    std::map<std::string, std::string> defines() const { return defines_; }

    std::vector<uint32_t> runtime_args(const CoreCoord &logical_core) const;

    void set_runtime_args(const CoreCoord &logical_core, const std::vector<uint32_t> runtime_args);

    virtual bool configure(Device *device, const CoreCoord &logical_core) const = 0;

    // Will cause CompileProgram to emit a file hlk_defines_generated.h
    // Each unique combination of defines will produce a unique compiled instantiation
    // This file is then automatically included in the generated compiled kernel files
    void add_define(const std::string& name, const std::string& value) { defines_[name] = value; }
    void add_define(const std::string& name, int value) { defines_[name] = std::to_string(value); }
    size_t define_args_hash() const;

   protected:
    std::string kernel_path_file_name_;                 // Full kernel path and file name
    CoreRangeSet core_range_set_;
    KernelType kernel_type_;
    std::string binary_path_;
    std::vector<ll_api::memory> binaries_;    // DataMovement kernels have one binary each and Compute kernels have three binaries
    std::vector<uint32_t> compile_time_args_;
    std::map<CoreCoord, std::vector<uint32_t>> core_to_runtime_args_;
    std::map<std::string, std::string> defines_; // preprocessor defines. this is to be able to generate generic instances.

    void set_binary_path(const std::string &binary_path) { binary_path_ = binary_path; }
    void set_binaries(const std::string &binary_path);

    friend void CompileKernel(Device *device, Program &program, Kernel *kernel, bool profile_kernel);
};

class DataMovementKernel : public Kernel {
   public:
    DataMovementKernel(
        const std::string &kernel_path_file_name,
        const CoreRangeSet &core_range_set,
        DataMovementProcessor processor,
        NOC noc) :
        Kernel(kernel_path_file_name, core_range_set, KernelType::DataMovement),
        processor_(processor),
        noc_(noc) {}

    DataMovementKernel(
        const std::string &kernel_path_file_name,
        const CoreRangeSet &core_range_set,
        const std::vector<uint32_t> &compile_args,
        DataMovementProcessor processor,
        NOC noc) :
        Kernel(kernel_path_file_name, core_range_set, compile_args, KernelType::DataMovement),
        processor_(processor),
        noc_(noc) {}

    DataMovementKernel(
        const std::string &kernel_path_file_name,
        const CoreRange &core_range,
        DataMovementProcessor processor,
        NOC noc) :
        Kernel(kernel_path_file_name, CoreRangeSet({core_range}), KernelType::DataMovement),
        processor_(processor),
        noc_(noc) {}

    DataMovementKernel(
        const std::string &kernel_path_file_name,
        const CoreRange &core_range,
        const std::vector<uint32_t> &compile_args,
        DataMovementProcessor processor,
        NOC noc) :
        Kernel(kernel_path_file_name, CoreRangeSet({core_range}), compile_args, KernelType::DataMovement),
        processor_(processor),
        noc_(noc) {}

    DataMovementKernel(
        const std::string &kernel_path_file_name,
        const CoreCoord &core,
        DataMovementProcessor processor,
        NOC noc) :
        Kernel(kernel_path_file_name, CoreRangeSet({CoreRange{.start=core, .end=core}}), KernelType::DataMovement),
        processor_(processor),
        noc_(noc) {}

    DataMovementKernel(
        const std::string &kernel_path_file_name,
        const CoreCoord &core,
        const std::vector<uint32_t> &compile_args,
        DataMovementProcessor processor,
        NOC noc) :
        Kernel(kernel_path_file_name, CoreRangeSet({CoreRange{.start=core, .end=core}}), compile_args, KernelType::DataMovement),
        processor_(processor),
        noc_(noc) {}

    ~DataMovementKernel() {}

    DataMovementProcessor data_movement_processor() const { return processor_; }

    NOC noc() const { return noc_; }

    bool configure(Device *device, const CoreCoord &logical_core) const;

   private:
    void write_runtime_args_to_device(Device *device, const CoreCoord &logical_core) const;

    friend bool WriteRuntimeArgsToDevice(Device *device, DataMovementKernel *kernel, const CoreCoord &logical_core, const std::vector<uint32_t> &runtime_args);

    DataMovementProcessor processor_;  // For data transfer kernels: NCRISC & BRISC
    NOC noc_;
};

class ComputeKernel : public Kernel {
   public:
    ComputeKernel(
        const std::string &kernel_path_file_name,
        const CoreRangeSet &core_range_set,
        const std::vector<uint32_t> &compile_args,
        MathFidelity math_fidelity,
        bool fp32_dest_acc_en,
        bool math_approx_mode) :
        Kernel(kernel_path_file_name, core_range_set, compile_args, KernelType::Compute),
        math_fidelity_(math_fidelity),
        fp32_dest_acc_en_(fp32_dest_acc_en),
        math_approx_mode_(math_approx_mode) {}

    ComputeKernel(
        const std::string &kernel_path_file_name,
        const CoreRange &core_range,
        const std::vector<uint32_t> &compile_args,
        MathFidelity math_fidelity,
        bool fp32_dest_acc_en,
        bool math_approx_mode) :
        Kernel(kernel_path_file_name, CoreRangeSet({core_range}), compile_args, KernelType::Compute),
        math_fidelity_(math_fidelity),
        fp32_dest_acc_en_(fp32_dest_acc_en),
        math_approx_mode_(math_approx_mode) {}

    ComputeKernel(
        const std::string &kernel_path_file_name,
        const CoreCoord &core,
        const std::vector<uint32_t> &compile_args,
        MathFidelity math_fidelity,
        bool fp32_dest_acc_en,
        bool math_approx_mode) :
        Kernel(kernel_path_file_name, CoreRangeSet({CoreRange{.start=core, .end=core}}), compile_args, KernelType::Compute),
        math_fidelity_(math_fidelity),
        fp32_dest_acc_en_(fp32_dest_acc_en),
        math_approx_mode_(math_approx_mode) {}

    ~ComputeKernel() {}

    bool configure(Device *device, const CoreCoord &logical_core) const;

    MathFidelity math_fidelity() const { return math_fidelity_; }

    bool fp32_dest_acc_en() const { return fp32_dest_acc_en_; }

    bool math_approx_mode() const { return math_approx_mode_; }

   private:
    MathFidelity math_fidelity_;  // Math fidelity
    bool fp32_dest_acc_en_;       //
    bool math_approx_mode_;       // Run math in approx mode
};

struct KernelDefinesHash {
    KernelDefinesHash() {}

    size_t operator()(const std::map<std::string, std::string> &c_defines) const;
};

}  // namespace tt_metal

}  // namespace tt
