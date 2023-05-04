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

std::string Kernel::binary_path(const CoreCoord &logical_core) const {
    if (not is_on_logical_core(logical_core)) {
        TT_THROW("Cannot access binary for " + name() + " because it is not on core " + logical_core.str());
    }
    return binary_path_.at(logical_core);
}

std::vector<ll_api::memory> const &Kernel::binaries() const {
    const static std::map<KernelType, int> kernel_type_to_expected_num_binaries = {
        {KernelType::Compute, 3},
        {KernelType::DataMovement, 1}
    };
    int expected_num_binaries = kernel_type_to_expected_num_binaries.at(this->kernel_type_);
    if (this->binaries_.size() != expected_num_binaries) {
        std::stringstream identifier;
        identifier << this->kernel_type_;
        TT_THROW("Expected " + std::to_string(expected_num_binaries) + " binaries but have "
                    + std::to_string(this->binaries_.size()) + " for " + identifier.str() + " kernel " + this->name());
    }
    return this->binaries_;
}

std::vector<uint32_t> Kernel::runtime_args(const CoreCoord &logical_core) const {
    if (not is_on_logical_core(logical_core)) {
        TT_THROW("Cannot access runtime args for " + name() + " because it is not on core " + logical_core.str());
    }
    return this->core_to_runtime_args_.at(logical_core);
}

void Kernel::set_runtime_args(const CoreCoord &logical_core, const std::vector<uint32_t> runtime_args) {
    this->core_to_runtime_args_.insert_or_assign(logical_core, runtime_args);
}

size_t Kernel::compile_time_args_hash() const {
    return tt::utils::vector_hash<uint32_t>{}(this->compile_time_args_);
}

size_t Kernel::define_args_hash(const CoreCoord& logical_core) const {
    if (not is_on_logical_core(logical_core)) {
        TT_THROW("Cannot hash compile time args for " + name() + " because it is not on core " + logical_core.str());
    }
    return KernelDefinesHash{logical_core}(defines_);
}

void Kernel::set_binaries(const std::string &binary_path) {
    std::vector<ll_api::memory> binaries;
    switch (this->kernel_type_) {
        case KernelType::Compute: {
            for (int trisc_id = 0; trisc_id <= 2; trisc_id++) {
                std::string trisc_id_str = std::to_string(trisc_id);
                std::string hex_path = binary_path + "/tensix_thread" + trisc_id_str + "/tensix_thread" + trisc_id_str + ".hex";
                ll_api::memory binary_mem = llrt::get_risc_binary(hex_path);
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
            ll_api::memory binary_mem = llrt::get_risc_binary(binary_path + binary_path_suffix);
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

void init_test_mailbox(Device *device, const CoreCoord &core, uint64_t test_mailbox_addr) {
    std::vector<uint32_t> test_mailbox_init_val = {INIT_VALUE};
    tt::llrt::write_hex_vec_to_core(
        device->cluster(), device->pcie_slot(), core, test_mailbox_init_val, test_mailbox_addr);

    std::vector<uint32_t> test_mailbox_init_val_check;
    test_mailbox_init_val_check = tt::llrt::read_hex_vec_from_core(
        device->cluster(), device->pcie_slot(), core, test_mailbox_addr, sizeof(uint32_t));  // read a single uint32_t
    TT_ASSERT(test_mailbox_init_val_check[0] == INIT_VALUE);
}

void DataMovementKernel::write_runtime_args_to_device(Device *device, const CoreCoord &logical_core) const {
    auto cluster = device->cluster();
    auto pcie_slot = device->pcie_slot();
    auto worker_core = device->worker_core_from_logical_core(logical_core);

    auto rt_args = this->runtime_args(logical_core);
    uint32_t core_x = logical_core.x;
    uint32_t core_y = logical_core.y;
    rt_args.push_back(core_x);
    rt_args.push_back(core_y);
    uint32_t runtime_args_size = rt_args.size() * sizeof(uint32_t);

    uint64_t l1_arg_base;
    uint64_t result_base;
    switch (processor_) {
        case DataMovementProcessor::RISCV_0:
            l1_arg_base = BRISC_L1_ARG_BASE;
            result_base = BRISC_L1_RESULT_BASE;
            break;
        case DataMovementProcessor::RISCV_1:
            l1_arg_base = NCRISC_L1_ARG_BASE;
            result_base = NCRISC_L1_RESULT_BASE;
            break;
        default:
            TT_THROW("Unexpected data movement processor type");
    }

    std::stringstream identifier;
    identifier << processor_;
    if (l1_arg_base + runtime_args_size >= result_base) {
        TT_THROW(std::to_string(runtime_args_size / 1024) + "KB " + identifier.str()  + " runtime args targeting " + logical_core.str() + " are too large.\
            Cannot be written as they will run into memory region reserved for result. Max allowable size is " + std::to_string((result_base - l1_arg_base)/1024) + " KB.");
    }

    tt::llrt::write_hex_vec_to_core(cluster, pcie_slot, worker_core, rt_args, l1_arg_base);
}

bool DataMovementKernel::configure(Device *device, const CoreCoord &logical_core) const {
    bool pass = true;
    if (not is_on_logical_core(logical_core)) {
        TT_THROW("Cannot configure kernel because it is not on core " + logical_core.str());
    }
    auto cluster = device->cluster();
    auto pcie_slot = device->pcie_slot();
    auto worker_core = device->worker_core_from_logical_core(logical_core);
    auto binary_path = binary_path_.at(logical_core);

    int riscv_id;
    std::string binary_path_suffix;
    uint64_t test_mailbox_addr;
    switch (processor_) {
        case (DataMovementProcessor::RISCV_0): {
            riscv_id = 0;
            binary_path_suffix = "/brisc/brisc.hex";
            test_mailbox_addr = TEST_MAILBOX_ADDR;
        }
        break;
        case (DataMovementProcessor::RISCV_1): {
            riscv_id = 1;
            binary_path_suffix = "/ncrisc/ncrisc.hex";
            test_mailbox_addr = TEST_MAILBOX_ADDR_NCRISC;
        }
        break;
        default:
            TT_ASSERT(false, "Unsupported data movement processor!");
    }

    pass &= tt::llrt::test_load_write_read_risc_binary(
        cluster, binary_path + binary_path_suffix, pcie_slot, worker_core, riscv_id);
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

size_t KernelDefinesHash::operator()(const std::map<std::string, std::string> &c_defines) const {
    size_t hash_value = 0;
    for (auto it = c_defines.begin(); it != c_defines.end(); ++it)
        boost::hash_combine(hash_value, std::hash<std::string>{}(it->first + it->second));
    return hash_value;
}

}  // namespace tt_metal

}  // namespace tt
