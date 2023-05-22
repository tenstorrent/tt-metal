#pragma once

#include <vector>
#include <map>

#include "common/base_types.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/impl/kernels/kernel_args.hpp"
#include "build_kernels_for_riscv/build_kernel_options.hpp"

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
    DataMovement = 0,  // reader / writter
    Compute = 1,       // unpack -> math -> pack
};

std::ostream& operator<<(std::ostream& os, const DataMovementProcessor& processor);
std::ostream& operator<<(std::ostream& os, const KernelType& type);

class Kernel;

void ConfigureForCompilation(Kernel *kernel, build_kernel_for_riscv_options_t &build_options, const tt_xy_pair &logical_core, const std::string &out_dir_path);

class Kernel {
   public:
    Kernel(const std::string &kernel_path_file_name, const CoreRange &core_range, KernelType kernel_type) :
        kernel_path_file_name_(kernel_path_file_name), start_core_(core_range.first), end_core_(core_range.second), kernel_type_(kernel_type) {}

    Kernel(const std::string &kernel_path_file_name, const tt_xy_pair &core, KernelType kernel_type) :
        kernel_path_file_name_(kernel_path_file_name), start_core_(core), end_core_(core), kernel_type_(kernel_type) {}

    Kernel(const std::string &kernel_path_file_name, const CoreRange &core_range, const KernelArgs &kernel_args, KernelType kernel_type) :
        kernel_path_file_name_(kernel_path_file_name), start_core_(core_range.first), end_core_(core_range.second), kernel_args_(kernel_args), kernel_type_(kernel_type) {}

    Kernel(const std::string &kernel_path_file_name, const tt_xy_pair &core, const KernelArgs &kernel_args, KernelType kernel_type) :
        kernel_path_file_name_(kernel_path_file_name), start_core_(core), end_core_(core), kernel_args_(kernel_args), kernel_type_(kernel_type) {}

    virtual ~Kernel() {}

    std::string kernel_path_file_name() const { return kernel_path_file_name_; }

    std::string name() const;

    std::vector<tt_xy_pair> logical_cores() const;

    bool is_on_logical_core(const tt_xy_pair &logical_core) const;

    KernelType kernel_type() const { return kernel_type_; }

    std::string binary_path(const tt_xy_pair &logical_core) const;

    KernelArgs &kernel_args() { return kernel_args_; }

    const KernelArgs &kernel_args() const { return kernel_args_; }

    std::vector<uint32_t> compile_time_args(const tt_xy_pair &logical_core) const;

    std::vector<uint32_t> runtime_args(const tt_xy_pair &logical_core) const;

    size_t compile_time_args_hash(const tt_xy_pair &logical_core) const;

    virtual bool configure(Device *device, const tt_xy_pair &logical_core) const = 0;

    // Will cause CompileProgram to emit a file hlk_defines_generated.h
    // Each unique combination of defines will produce a unique compiled instantiation
    // This file is then automatically included in the generated compiled kernel files
    void add_define(const std::string& name, const std::string& value) { defines_[name] = value; }
    void add_define(const std::string& name, int value) { defines_[name] = std::to_string(value); }
    size_t define_args_hash(const tt_xy_pair& logical_core) const;

   protected:
    std::string kernel_path_file_name_;                 // Full kernel path and file name
    tt_xy_pair start_core_;                             // First logical Tensix coordinates within core grid
    tt_xy_pair end_core_;                               // Last logical Tensix coordinates within core grid
    KernelType kernel_type_;
    std::map<tt_xy_pair, std::string> binary_path_;     //
    KernelArgs kernel_args_;
    std::map<std::string, std::string> defines_; // preprocessor defines. this is to be able to generate generic instances.

    void set_binary_path(const tt_xy_pair &logical_core, const std::string &binary_path) { binary_path_.insert({logical_core, binary_path}); }

    virtual void configure_for_compilation(build_kernel_for_riscv_options_t &build_options, const tt_xy_pair &logical_core, const std::string &out_dir_path) = 0;

    friend void ConfigureForCompilation(Kernel *kernel, build_kernel_for_riscv_options_t &build_options, const tt_xy_pair &logical_core, const std::string &out_dir_path);
};

class DataMovementKernel : public Kernel {
   public:
    DataMovementKernel(
        const std::string &kernel_path_file_name,
        const CoreRange &core_range,
        DataMovementProcessor processor,
        NOC noc) :
        Kernel(kernel_path_file_name, core_range, KernelType::DataMovement),
        processor_(processor),
        noc_(noc) {}

    DataMovementKernel(
        const std::string &kernel_path_file_name,
        const CoreRange &core_range,
        const KernelArgs &kernel_args,
        DataMovementProcessor processor,
        NOC noc) :
        Kernel(kernel_path_file_name, core_range, kernel_args, KernelType::DataMovement),
        processor_(processor),
        noc_(noc) {}

    DataMovementKernel(
        const std::string &kernel_path_file_name,
        const tt_xy_pair &core,
        DataMovementProcessor processor,
        NOC noc) :
        Kernel(kernel_path_file_name, core, KernelType::DataMovement),
        processor_(processor),
        noc_(noc) {}

    DataMovementKernel(
        const std::string &kernel_path_file_name,
        const tt_xy_pair &core,
        const KernelArgs &kernel_args,
        DataMovementProcessor processor,
        NOC noc) :
        Kernel(kernel_path_file_name, core, kernel_args, KernelType::DataMovement),
        processor_(processor),
        noc_(noc) {}

    ~DataMovementKernel() {}

    DataMovementProcessor data_movement_processor() const { return processor_; }

    NOC noc() const { return noc_; }

    bool configure(Device *device, const tt_xy_pair &logical_core) const;

   private:
    void configure_for_compilation(build_kernel_for_riscv_options_t &build_options, const tt_xy_pair &logical_core, const std::string &out_dir_path);

    void write_runtime_args_to_device(Device *device, const tt_xy_pair &logical_core) const;

    friend bool WriteRuntimeArgsToDevice(Device *device, DataMovementKernel *kernel, const tt_xy_pair &logical_core, const std::vector<uint32_t> &runtime_args);

    DataMovementProcessor processor_;  // For data transfer kernels: NCRISC & BRISC
    NOC noc_;
};

class ComputeKernel : public Kernel {
   public:
    ComputeKernel(
        const std::string &kernel_path_file_name,
        const CoreRange &core_range,
        const KernelArgs &kernel_args,
        MathFidelity math_fidelity,
        bool fp32_dest_acc_en,
        bool math_approx_mode) :
        Kernel(kernel_path_file_name, core_range, kernel_args, KernelType::Compute),
        math_fidelity_(math_fidelity),
        fp32_dest_acc_en_(fp32_dest_acc_en),
        math_approx_mode_(math_approx_mode) {}

    ComputeKernel(
        const std::string &kernel_path_file_name,
        const tt_xy_pair &core,
        const KernelArgs &kernel_args,
        MathFidelity math_fidelity,
        bool fp32_dest_acc_en,
        bool math_approx_mode) :
        Kernel(kernel_path_file_name, core, kernel_args, KernelType::Compute),
        math_fidelity_(math_fidelity),
        fp32_dest_acc_en_(fp32_dest_acc_en),
        math_approx_mode_(math_approx_mode) {}

    ~ComputeKernel() {}

    bool configure(Device *device, const tt_xy_pair &logical_core) const;

   private:
    void configure_for_compilation(build_kernel_for_riscv_options_t &build_options, const tt_xy_pair &logical_core, const std::string &out_dir_path);

    MathFidelity math_fidelity_;  // Math fidelity
    bool fp32_dest_acc_en_;       //
    bool math_approx_mode_;       // Run math in approx mode
};

}  // namespace tt_metal

}  // namespace tt
