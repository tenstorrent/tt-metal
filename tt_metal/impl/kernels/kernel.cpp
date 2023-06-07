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

std::set<CoreCoord> Kernel::logical_cores() const {
    std::set<CoreCoord> cores;
    for (auto core_range : this->core_range_set_.ranges()) {
        auto start = core_range.start;
        auto end = core_range.end;
        for (auto x = start.x; x <= end.x; x++) {
            for (auto y = start.y; y <= end.y; y++) {
                CoreCoord logical_core({.x=x, .y=y});
                cores.insert(logical_core);
            }
        }
    }
    return cores;
}

bool Kernel::is_on_logical_core(const CoreCoord &logical_core) const {
    return this->core_range_set_.core_coord_in_core_ranges(logical_core);
}

std::vector<ll_api::memory> const &Kernel::binaries() const {
    const static std::map<KernelType, int> kernel_type_to_expected_num_binaries = {
        {KernelType::Compute, 3},
        {KernelType::DataMovement, 1}
    };
    int expected_num_binaries = kernel_type_to_expected_num_binaries.at(this->kernel_type_);
    if (not this->binaries_.empty() and this->binaries_.size() != expected_num_binaries) {
        std::stringstream identifier;
        identifier << this->kernel_type_;
        TT_THROW("Expected " + std::to_string(expected_num_binaries) + " binaries but have "
                    + std::to_string(this->binaries_.size()) + " for " + identifier.str() + " kernel " + this->name());
    }
    return this->binaries_;
}

size_t Kernel::compile_time_args_hash() const {
    return tt::utils::vector_hash<uint32_t>{}(this->compile_time_args_);
}

size_t Kernel::define_args_hash() const {
    return KernelDefinesHash{}(defines_);
}

std::vector<uint32_t> const &Kernel::runtime_args(const CoreCoord &logical_core) {
    log_assert(this->is_on_logical_core(logical_core), "Cannot get runtime args for kernel {} that is not placed on core {}", this->name(), logical_core.str());
    return this->core_to_runtime_args_[logical_core];
}

void Kernel::set_runtime_args(const CoreCoord &logical_core, const std::vector<uint32_t> &runtime_args) {
    auto validate_runtime_args_size = [&]() {
        uint32_t runtime_args_size = runtime_args.size() * sizeof(uint32_t);
        uint64_t l1_arg_base;
        uint64_t result_base;
        switch (this->kernel_type_) {
            case KernelType::DataMovement:
                l1_arg_base = BRISC_L1_ARG_BASE;
                result_base = BRISC_L1_RESULT_BASE;
                break;
            default:
                log_assert(false, "Only data movement kernels have runtime arg support");
        }
        std::stringstream identifier;
        identifier << this->kernel_type_;
        if (l1_arg_base + runtime_args_size >= result_base) {
            TT_THROW(std::to_string(runtime_args_size / 1024) + "KB " + identifier.str()  + " runtime args targeting " + logical_core.str() + " are too large.\
                Cannot be written as they will run into memory region reserved for result. Max allowable size is " + std::to_string((result_base - l1_arg_base)/1024) + " KB.");
        }
    };

    log_assert(this->is_on_logical_core(logical_core), "Cannot set runtime args for core {} since kernel {} is not placed on it!", logical_core.str(), this->name());
    validate_runtime_args_size();
    auto &set_rt_args = this->core_to_runtime_args_[logical_core];
    log_assert(set_rt_args.empty() or set_rt_args.size() == runtime_args.size(), "Illegal Runtime Args: Number of runtime args cannot be modified!");
    set_rt_args = runtime_args;
}

void Kernel::read_binaries() {
    std::vector<ll_api::memory> binaries;
    TT_ASSERT ( !binary_path_.empty(), "Path to Kernel binaries not set!" );
    switch (this->kernel_type_) {
        case KernelType::Compute: {
            for (int trisc_id = 0; trisc_id <= 2; trisc_id++) {
                std::string trisc_id_str = std::to_string(trisc_id);
                std::string hex_path = binary_path_ + "/tensix_thread" + trisc_id_str + "/tensix_thread" + trisc_id_str + ".hex";
                ll_api::memory binary_mem = llrt::get_risc_binary(hex_path, false);
                binaries.push_back(binary_mem);
            }
        }
        break;
        case KernelType::DataMovement: {
            auto dm_kernel = dynamic_cast<DataMovementKernel *>(this);
            TT_ASSERT(dm_kernel != nullptr);
            uint32_t riscv_id;
            std::string binary_path_suffix;
            switch (dm_kernel->data_movement_processor()) {
                case (DataMovementProcessor::RISCV_0): {
                    riscv_id = 0;
                    binary_path_suffix = "/brisc/brisc.hex";
                }
                break;
                case (DataMovementProcessor::RISCV_1): {
                    riscv_id = 1;
                    binary_path_suffix = "/ncrisc/ncrisc.hex";
                }
                break;
                default:
                    TT_ASSERT(false, "Unsupported data movement processor!");
            }
            ll_api::memory binary_mem = llrt::get_risc_binary(binary_path_ + binary_path_suffix, false);
            binaries.push_back(binary_mem);
        }
        break;
        default:
            TT_ASSERT(false, "Unsupported kernel type");
    };
    if (not this->binaries_.empty()) {
        TT_ASSERT(this->binaries_ == binaries);
    } else {
    this->binaries_ = std::move(binaries);
    }
}

RISCV Kernel::processor() const {
    switch (this->kernel_type_) {
        case KernelType::Compute: return RISCV::COMPUTE;
        case KernelType::DataMovement: {
            auto dm_kernel = dynamic_cast<const DataMovementKernel *>(this);
            log_assert(dm_kernel != nullptr, "Expected data movement kernel");
            switch (dm_kernel->data_movement_processor()) {
                case DataMovementProcessor::RISCV_0: return RISCV::BRISC;
                case DataMovementProcessor::RISCV_1: return RISCV::NCRISC;
                default:
                    log_assert(false, "Unsupported data movement processor");
            }
        }
        default:
            log_assert(false, "Unsupported kernel type");
    }
    return RISCV::BRISC;
}

void init_test_mailbox(Device *device, const CoreCoord &core, uint64_t test_mailbox_addr) {
    std::vector<uint32_t> test_mailbox_init_val = {INIT_VALUE};
    tt::llrt::write_hex_vec_to_core(
        device->cluster(), device->pcie_slot(), core, test_mailbox_init_val, test_mailbox_addr);

    std::vector<uint32_t> test_mailbox_init_val_check;
    test_mailbox_init_val_check = tt::llrt::read_hex_vec_from_core(
        device->cluster(), device->pcie_slot(), core, test_mailbox_addr, sizeof(uint32_t));  // read a single uint32_t
    TT_ASSERT(test_mailbox_init_val_check[0] == INIT_VALUE);
}

bool DataMovementKernel::configure(Device *device, const CoreCoord &logical_core) const {
    bool pass = true;
    if (not is_on_logical_core(logical_core)) {
        TT_THROW("Cannot configure kernel because it is not on core " + logical_core.str());
    }
    auto cluster = device->cluster();
    auto pcie_slot = device->pcie_slot();
    auto worker_core = device->worker_core_from_logical_core(logical_core);
    ll_api::memory binary_mem = this->binaries().at(0);

    int riscv_id;
    uint64_t test_mailbox_addr;
    switch (processor_) {
        case (DataMovementProcessor::RISCV_0): {
            riscv_id = 0;
            test_mailbox_addr = TEST_MAILBOX_ADDR;
        }
        break;
        case (DataMovementProcessor::RISCV_1): {
            riscv_id = 1;
            test_mailbox_addr = TEST_MAILBOX_ADDR_NCRISC;
        }
        break;
        default:
            TT_ASSERT(false, "Unsupported data movement processor!");
    }

    pass &= tt::llrt::test_load_write_read_risc_binary(cluster, binary_mem, pcie_slot, worker_core, riscv_id);
    init_test_mailbox(device, worker_core, test_mailbox_addr);
    if (processor_ == DataMovementProcessor::RISCV_1) {
        tt::llrt::enable_ncrisc(cluster, pcie_slot, worker_core);
    }
    return pass;
}

bool ComputeKernel::configure(Device *device, const CoreCoord &logical_core) const {
    bool pass = true;
    if (not is_on_logical_core(logical_core)) {
        TT_THROW("Cannot configure kernel because it is not on core " + logical_core.str());
    }
    auto cluster = device->cluster();
    auto pcie_slot = device->pcie_slot();
    auto worker_core = device->worker_core_from_logical_core(logical_core);
    std::vector<ll_api::memory> binaries = this->binaries();

    for (int trisc_id = 0; trisc_id <= 2; trisc_id++) {
        pass &= tt::llrt::test_load_write_read_trisc_binary(
            cluster,
            binaries.at(trisc_id),
            pcie_slot,
            worker_core,
            trisc_id);
        init_test_mailbox(device, worker_core, trisc_mailbox_addresses[trisc_id]);
    }
    tt::llrt::enable_triscs(cluster, pcie_slot, worker_core);

    return pass;
}

std::ostream& operator<<(std::ostream& os, const DataMovementProcessor& processor) {
    switch (processor) {
        case DataMovementProcessor::RISCV_0: os << "RISCV_0"; break;
        case DataMovementProcessor::RISCV_1: os << "RISCV_1"; break;
        default: TT_THROW("Unknown data movement processor");
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const KernelType& type) {
    switch (type) {
        case KernelType::DataMovement: os << "DataMovement"; break;
        case KernelType::Compute: os << "Compute"; break;
        default: TT_THROW("Unknown kernel type");
    }
    return os;
}

std::string kernel_attributes_str(Kernel *kernel) {
    std::string attr_str = "{";
    if (not kernel->compile_time_args().empty()) {
        attr_str += "Compile args: [";
        for (const auto compile_arg : kernel->compile_time_args()) {
            attr_str += std::to_string(compile_arg) + " ";
        }
        attr_str += "] ";
    }
    if (not kernel->defines().empty()) {
        attr_str += "Defines: {";
        for (const auto &[k, v] : kernel->defines()) {
            attr_str += "{ " + k + " - " + v + " } ";
        }
        attr_str += "} ";
    }
    if (kernel->kernel_type() == KernelType::DataMovement) {
        auto data_movement_kernel = dynamic_cast<DataMovementKernel *>(kernel);
        attr_str += "NOC: " + std::to_string(data_movement_kernel->noc()) + " ";
    } else {
        auto compute_kernel = dynamic_cast<ComputeKernel *>(kernel);
        std::stringstream math_fidel_str;
        math_fidel_str << compute_kernel->math_fidelity();
        attr_str += "Math fidelity: " + math_fidel_str.str() + " ";
        string fp32_en_str = compute_kernel->fp32_dest_acc_en() ? "Y" : "N";
        attr_str += "FP32 dest accumulate enabled: " + fp32_en_str + " ";
        string math_approx_str = compute_kernel->math_approx_mode() ? "Y" : "N";
        attr_str += "Math approx mode enabled: " + math_approx_str + " ";
    }

    attr_str += "}";
    return attr_str;
}

size_t KernelDefinesHash::operator()(const std::map<std::string, std::string> &c_defines) const {
    size_t hash_value = 0;
    for (auto it = c_defines.begin(); it != c_defines.end(); ++it)
        boost::hash_combine(hash_value, std::hash<std::string>{}(it->first + it->second));
    return hash_value;
}

}  // namespace tt_metal

}  // namespace tt
