#include "tt_metal/impl/kernels/kernel.hpp"

#include <set>
#include <unordered_set>

#include "build_kernels_for_riscv/build_kernels_for_riscv.hpp"
#include "llrt/llrt.hpp"

namespace tt {

namespace tt_metal {

std::string Kernel::name() const {
    auto pos_of_name = kernel_path_file_name_.rfind("/") + 1;
    auto pos_of_dot = kernel_path_file_name_.rfind(".");
    std::string kernel_name = kernel_path_file_name_.substr(pos_of_name, (pos_of_dot - pos_of_name));
    return kernel_name;
}

std::vector<tt_xy_pair> Kernel::logical_cores() const {
    std::vector<tt_xy_pair> cores;
    for (auto x = start_core_.x; x <= end_core_.x; x++) {
        for (auto y = start_core_.y; y <= end_core_.y; y++) {
            cores.push_back(tt_xy_pair(x, y));
        }
    }
    return cores;
}

bool Kernel::is_on_logical_core(const tt_xy_pair &logical_core) const {
    bool in_x_range = (logical_core.x >= start_core_.x) and (logical_core.x <= end_core_.x);
    bool in_y_range = (logical_core.y >= start_core_.y) and (logical_core.y <= end_core_.y);
    return in_x_range and in_y_range;
}

std::string Kernel::binary_path(const tt_xy_pair &logical_core) const {
    if (not is_on_logical_core(logical_core)) {
        TT_THROW("Cannot access binary for " + name() + " because it is not on core " + logical_core.str());
    }
    return binary_path_.at(logical_core);
}

std::vector<uint32_t> DataMovementKernel::compile_time_args(const tt_xy_pair &logical_core) const {
    if (not is_on_logical_core(logical_core)) {
        TT_THROW("Cannot access compile time args for " + name() + " because it is not on core " + logical_core.str());
    }
    return kernel_args_->compile_time_args(logical_core);
}

std::vector<uint32_t> DataMovementKernel::runtime_args(const tt_xy_pair &logical_core) const {
    if (not is_on_logical_core(logical_core)) {
        TT_THROW("Cannot access runtime args for " + name() + " because it is not on core " + logical_core.str());
    }
    return kernel_args_->runtime_args(logical_core);
}

size_t DataMovementKernel::compile_time_args_hash(const tt_xy_pair &logical_core) const {
    if (not is_on_logical_core(logical_core)) {
        TT_THROW("Cannot hash compile time args for " + name() + " because it is not on core " + logical_core.str());
    }
    return DataMovementKernelArgsHash{logical_core}(*kernel_args_);
}

std::vector<uint32_t> ComputeKernel::compile_time_args(const tt_xy_pair &logical_core) const {
    if (not is_on_logical_core(logical_core)) {
        TT_THROW("Cannot access compile time args for " + name() + " because it is not on core " + logical_core.str());
    }
    return kernel_args_->compile_time_args(logical_core);
}

size_t ComputeKernel::compile_time_args_hash(const tt_xy_pair &logical_core) const {
    if (not is_on_logical_core(logical_core)) {
        TT_THROW("Cannot hash compile time args for " + name() + " because it is not on core " + logical_core.str());
    }
    return ComputeKernelArgsHash{logical_core}(*kernel_args_);
}

size_t ComputeKernel::define_args_hash(const tt_xy_pair& logical_core) const {
    if (not is_on_logical_core(logical_core)) {
        TT_THROW("Cannot hash compile time args for " + name() + " because it is not on core " + logical_core.str());
    }
    return ComputeKernelDefinesHash{logical_core}(defines_);
}


void ConfigureForCompilation(Kernel *kernel, build_kernel_for_riscv_options_t *build_kernel_for_riscv_options, const tt_xy_pair &logical_core, const std::string &out_dir_path) {
    if (kernel == nullptr) {
        return;
    }
    kernel->configure_for_compilation(build_kernel_for_riscv_options, logical_core, out_dir_path);
}

void DataMovementKernel::configure_for_compilation(build_kernel_for_riscv_options_t *build_kernel_for_riscv_options, const tt_xy_pair &logical_core, const std::string &out_dir_path) {
    if (processor_ == DataMovementProcessor::RISCV_0) {
        build_kernel_for_riscv_options->brisc_kernel_file_name = kernel_path_file_name_;
    }
    if (processor_ == DataMovementProcessor::RISCV_1) {
        build_kernel_for_riscv_options->ncrisc_kernel_file_name = kernel_path_file_name_;
    }
    set_binary_path(logical_core, out_dir_path);
}

void ComputeKernel::configure_for_compilation(build_kernel_for_riscv_options_t *build_kernel_for_riscv_options, const tt_xy_pair &logical_core, const std::string &out_dir_path) {
    auto compute_kernel_args = kernel_args_->compile_time_args(logical_core);
    // build_kernel_for_riscv_options->set_hlk_args_all_cores(compute_kernel_args, compute_kernel_args_size);
    build_kernel_for_riscv_options->set_hlk_file_name_all_cores(kernel_path_file_name_);
    build_kernel_for_riscv_options->set_hlk_math_fidelity_all_cores(math_fidelity_);
    build_kernel_for_riscv_options->fp32_dest_acc_en = fp32_dest_acc_en_;
    build_kernel_for_riscv_options->hlk_defines = defines_;

    set_binary_path(logical_core, out_dir_path);
}

void init_test_mailbox(Device *device, const tt_xy_pair &core, uint64_t test_mailbox_addr) {
    std::vector<uint32_t> test_mailbox_init_val = {INIT_VALUE};
    tt::llrt::write_hex_vec_to_core(
        device->cluster(), device->pcie_slot(), core, test_mailbox_init_val, test_mailbox_addr);

    std::vector<uint32_t> test_mailbox_init_val_check;
    test_mailbox_init_val_check = tt::llrt::read_hex_vec_from_core(
        device->cluster(), device->pcie_slot(), core, test_mailbox_addr, sizeof(uint32_t));  // read a single uint32_t
    TT_ASSERT(test_mailbox_init_val_check[0] == INIT_VALUE);
}

void DataMovementKernel::write_runtime_args_to_device(Device *device, const tt_xy_pair &logical_core) const {
    auto cluster = device->cluster();
    auto pcie_slot = device->pcie_slot();
    auto worker_core = device->worker_core_from_logical_core(logical_core);

    auto runtime_args = kernel_args_->runtime_args(logical_core);
    uint32_t core_x = logical_core.x;
    uint32_t core_y = logical_core.y;
    runtime_args.push_back(core_x);
    runtime_args.push_back(core_y);

    uint64_t l1_arg_base;
    switch (processor_) {
        case DataMovementProcessor::RISCV_0:
            l1_arg_base = BRISC_L1_ARG_BASE;
            break;
        case DataMovementProcessor::RISCV_1:
            l1_arg_base = NCRISC_L1_ARG_BASE;
            break;
        default:
            TT_THROW("Unexpected data movement processor type");
    }

    tt::llrt::write_hex_vec_to_core(cluster, pcie_slot, worker_core, runtime_args, l1_arg_base);
}

bool DataMovementKernel::configure(Device *device, const tt_xy_pair &logical_core) const {
    bool pass = true;
    if (not is_on_logical_core(logical_core)) {
        TT_THROW("Cannot configure kernel because it is not on core " + logical_core.str());
    }
    auto cluster = device->cluster();
    auto pcie_slot = device->pcie_slot();
    auto worker_core = device->worker_core_from_logical_core(logical_core);
    auto binary_path = binary_path_.at(logical_core);

    // Always load brisc
    int brisc_id = 0;
    pass &= tt::llrt::test_load_write_read_risc_binary(
        cluster, binary_path + "/brisc/brisc.hex", pcie_slot, worker_core, brisc_id);
    init_test_mailbox(device, worker_core, TEST_MAILBOX_ADDR);

    if (processor_ == DataMovementProcessor::RISCV_1) {
        int ncrisc_id = 1;
        pass &= tt::llrt::test_load_write_read_risc_binary(
            cluster, binary_path + "/ncrisc/ncrisc.hex", pcie_slot, worker_core, ncrisc_id);
        init_test_mailbox(device, worker_core, TEST_MAILBOX_ADDR_NCRISC);
        tt::llrt::enable_ncrisc(cluster, pcie_slot, worker_core);
    }
    return pass;
}

bool ComputeKernel::configure(Device *device, const tt_xy_pair &logical_core) const {
    bool pass = true;
    if (not is_on_logical_core(logical_core)) {
        TT_THROW("Cannot configure kernel because it is not on core " + logical_core.str());
    }
    auto cluster = device->cluster();
    auto pcie_slot = device->pcie_slot();
    auto worker_core = device->worker_core_from_logical_core(logical_core);
    auto binary_path = binary_path_.at(logical_core);

    for (int trisc_id = 0; trisc_id <= 2; trisc_id++) {
        std::string trisc_id_str = std::to_string(trisc_id);
        pass &= tt::llrt::test_load_write_read_trisc_binary(
            cluster,
            binary_path + "/tensix_thread" + trisc_id_str + "/tensix_thread" + trisc_id_str + ".hex",
            pcie_slot,
            worker_core,
            trisc_id);
        init_test_mailbox(device, worker_core, trisc_mailbox_addresses[trisc_id]);
    }
    tt::llrt::enable_triscs(cluster, pcie_slot, worker_core);

    return pass;
}

}  // namespace tt_metal

}  // namespace tt
