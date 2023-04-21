#include "llrt.hpp"
#include "hostdevcommon/common_runtime_address_map.h"
#include "hostdevcommon/common_values.hpp"

#include <unordered_set>
#include <mutex>

#include "tools/cpuprof/cpuprof.h"

namespace tt {

// llrt = lower-level runtime
namespace llrt {

namespace fs = std::filesystem;

using std::endl;
using std::move;
using std::string;
using std::to_string;
using std::uint32_t;
using std::unordered_map;
using std::vector;

bool llrt_enable_binary_cache = true;

struct HexNameToMemVectorCache {
    using lock = std::unique_lock<std::mutex>;
    // maps from RisckCacheMapKey to hex file path
    static HexNameToMemVectorCache &inst() {
        static HexNameToMemVectorCache inst_;
        return inst_;
    }

    bool exists(const string &path) {
        lock l(mutex_);
        return cache_.find(path) != cache_.end();
    }
    vector<uint32_t> &get(const string &path) {
        lock l(mutex_);
        return cache_[path];
    }
    void add(const string &path, vector<uint32_t> &&mem) {
        lock l(mutex_);
        cache_[path] = move(mem);
    }

    unordered_map<string, vector<uint32_t>> cache_;
    std::mutex mutex_;
};

// made these free functions -- they're copy/paste of the member functions
// TODO: clean-up epoch_loader / epoch_binary -- a bunch of functions there should not be member functions
vector<uint32_t> get_risc_binary(string path, uint32_t id, bool id_is_trisc) {
    if (llrt_enable_binary_cache && HexNameToMemVectorCache::inst().exists(path)) {
        // std::cout << "-- HEX2MEM CACHE HIT FOR " << path << std::endl;
        return HexNameToMemVectorCache::inst().get(path);
    } else {
        // std::cout << "-- HEX2MEM CACHE MISS FOR " << path << std::endl;
    }

    string path_to_bin = path;
    fs::path bin_file(path_to_bin);
    if (!fs::exists(bin_file)) {
        string tt_metal_home = string(getenv("TT_METAL_HOME"));
        // try loading from home in case cwd isn't home
        path_to_bin = tt_metal_home + "/" + path_to_bin;
        fs::path bin_file_h(path_to_bin);
        if (!fs::exists(bin_file_h)) {
            std::cout << " Error: " << bin_file.c_str() << " doesn't exist" << endl;
            TT_ASSERT(false);
        }
    }

    std::ifstream hex_istream(path_to_bin);
    ll_api::memory mem;
    if (id_is_trisc)
        mem = std::move(ll_api::memory::from_discontiguous_risc_hex(
            hex_istream, (ll_api::memory::risc_type_t)((int)ll_api::memory::TRISC0 + id)));
    else
        mem = std::move(ll_api::memory::from_discontiguous_risc_hex(
            hex_istream, id == 1 ? ll_api::memory::NCRISC : ll_api::memory::BRISC));

    if (llrt_enable_binary_cache)
        HexNameToMemVectorCache::inst().add(path, mem.get_content());

    return mem.get_content();
}

// This deasserts reset for all BRISCs (on all devices, all cores), but not other RISC processors (NCRISC, TRISC)
// Every core gets valid FW (blank kernel if nothing is running on the core) before being taken out ot reset
// This avoids the issue of cores running garbahe out of their L1
// TODO: deassert reset only for used BRISCs (needs a new deassert function w/ a list of core to de-assert)
void deassert_brisc_reset_for_all_chips_all_cores(tt_cluster *cluster, bool stagger_start) {
    cluster->deassert_risc_reset(stagger_start);
    log_debug(tt::LogLLRuntime, "deasserted reset for all BRISCs");
}

// TODO: try using "stop" method from device instead, it's the proper way of asserting reset
void assert_reset_for_all_chips(tt_cluster *cluster) {
    TT_ASSERT(cluster->type == tt::TargetDevice::Silicon);

    log_debug(tt::LogLLRuntime, "Starting resets for {} chips", cluster->get_num_chips());
    for (const chip_id_t &chip_id : cluster->get_all_chips()) {
        cluster->broadcast_remote_tensix_risc_reset(chip_id, TENSIX_ASSERT_SOFT_RESET);
    }
}

// tt_xy_pair core --> NOC coordinates ("functional workers" from the SOC descriptor)
// NOC coord is also synonymous to routing / physical coord
// dram_channel id (0..7) for GS is also mapped to NOC coords in the SOC descriptor

void write_hex_vec_to_core(
    tt_cluster *cluster, int chip, const tt_xy_pair &core, std::vector<uint32_t> hex_vec, uint64_t addr) {
    // the API is named "write_dram_vec", and its overloaded variant is taking (chip, core) pair, ie. it can write to
    // core's L1
    cluster->write_dram_vec(hex_vec, tt_cxy_pair(chip, core), addr);
}

std::vector<std::uint32_t> read_hex_vec_from_core(
    tt_cluster *cluster, int chip, const tt_xy_pair &core, uint64_t addr, uint32_t size) {
    vector<std::uint32_t> read_hex_vec;
    cluster->read_dram_vec(read_hex_vec, tt_cxy_pair(chip, core), addr, size);
    return read_hex_vec;
}

void print_worker_cores(tt_cluster *cluster, chip_id_t chip_id) {
    std::cout << std::endl << "worker cores: " << std::endl;
    for (const tt_xy_pair &core : cluster->get_soc_desc(chip_id).workers) {
        std::cout << core.str() << " ";
    }
    std::cout << std::endl << std::endl;
}

bool is_worker_core(tt_cluster *cluster, const tt_xy_pair &core, chip_id_t chip_id) {
    return std::find(
               cluster->get_soc_desc(chip_id).workers.begin(), cluster->get_soc_desc(chip_id).workers.end(), core) !=
           cluster->get_soc_desc(chip_id).workers.end();
}

CircularBufferConfigVec create_circular_buffer_config_vector() {
    CircularBufferConfigVec circular_buffer_config_vec(
        NUM_CIRCULAR_BUFFERS * UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG, 0);  // init to 0's
    return circular_buffer_config_vec;
}

void set_config_for_circular_buffer(
    CircularBufferConfigVec &circular_buffer_config_vec,
    uint32_t circular_buffer_index,
    uint32_t addr_in_bytes,
    uint32_t size_in_bytes,
    uint32_t size_in_tiles) {
    circular_buffer_config_vec.at(UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * circular_buffer_index) =
        addr_in_bytes >> 4;  // convert to addr in 16B words
    circular_buffer_config_vec.at(UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * circular_buffer_index + 1) =
        size_in_bytes >> 4;  // convert to addr in 16B words
    circular_buffer_config_vec.at(UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * circular_buffer_index + 2) = size_in_tiles;
}

void write_circular_buffer_config_vector_to_core(
    tt_cluster *cluster, int chip, const tt_xy_pair &core, CircularBufferConfigVec circular_buffer_config_vec) {
    write_hex_vec_to_core(cluster, chip, core, circular_buffer_config_vec, CIRCULAR_BUFFER_CONFIG_BASE);
}

void write_graph_interpreter_op_info_to_core(
    tt_cluster *cluster, int chip, const tt_xy_pair &core, op_info_t op_info, int op_idx) {
    vector<uint32_t> op_info_vec = {
        op_info.op_code,
        op_info.cb_in0_id,
        op_info.cb_in1_id,
        op_info.cb_out_id,
        op_info.pop0,
        op_info.pop1,
        op_info.unary};
    uint32_t offset = op_info_vec.size() * sizeof(uint32_t) * op_idx;

    write_hex_vec_to_core(cluster, chip, core, op_info_vec, OP_INFO_BASE_ADDR + offset);
}

// for BRISC and NCRISC
bool test_load_write_read_risc_binary(
    tt_cluster *cluster, std::string hex_file_path, int chip_id, const tt_xy_pair &core, int riscv_id) {
    // PROF_BEGIN("get_risc")
    assert(riscv_id == 0 || riscv_id == 1);

    assert(is_worker_core(cluster, core, chip_id));

    std::vector<uint32_t> hex_vec = get_risc_binary(hex_file_path, riscv_id);  // 0 = BRISC, 1 = NCRISC

    log_debug(tt::LogLLRuntime, "hex_file_path = {}", hex_file_path);
    log_debug(
        tt::LogLLRuntime, "hex_vec size = {}, size_in_bytes = {}", hex_vec.size(), hex_vec.size() * sizeof(uint32_t));
    // PROF_END("get_risc")

    uint64_t addr = 0;
    switch (riscv_id) {
        case 0: addr = l1_mem::address_map::FIRMWARE_BASE; break;         //  BRISC binary addr in L1
        case 1: addr = l1_mem::address_map::NCRISC_FIRMWARE_BASE; break;  // NCRISC binary addr in L1
        default: std::cout << "Unknown rsicv_id = " << riscv_id << std::endl; exit(1);
    }

    write_hex_vec_to_core(cluster, chip_id, core, hex_vec, addr);  // PROF_BEGIN("write_risc")
    log_debug(tt::LogLLRuntime, "wrote hex to the core");          // PROF_END("write_risc")

    if (std::getenv("TT_KERNEL_READBACK_DISABLE") == nullptr) {
        std::vector<uint32_t> read_hex_vec;  // PROF_BEGIN("read_risc")
        read_hex_vec = read_hex_vec_from_core(
            cluster, chip_id, core, addr, hex_vec.size() * sizeof(uint32_t));  // size to read in Bytes
        log_debug(tt::LogLLRuntime, "read hex back from the core");            // PROF_END("read_risc")
        return hex_vec == read_hex_vec;
    }
    return true;
}

// for TRISCs
bool test_load_write_read_trisc_binary(
    tt_cluster *cluster, std::string hex_file_path, int chip_id, const tt_xy_pair &core, int triscv_id) {
    assert(triscv_id >= 0 and triscv_id <= 2);

    assert(is_worker_core(cluster, core, chip_id));

    // PROF_BEGIN("trisc_get")
    std::vector<uint32_t> hex_vec = get_trisc_binary(hex_file_path, triscv_id);  // TRISC 0, 1, 2
    // PROF_END("trisc_get")

    log_debug(tt::LogLLRuntime, "hex_file_path = {}", hex_file_path);
    log_debug(
        tt::LogLLRuntime, "hex_vec size = {}, size_in_bytes = {}", hex_vec.size(), hex_vec.size() * sizeof(uint32_t));

    uint64_t addr = 0;
    switch (triscv_id) {
        case 0: addr = l1_mem::address_map::TRISC0_BASE; break;
        case 1: addr = l1_mem::address_map::TRISC1_BASE; break;
        case 2: addr = l1_mem::address_map::TRISC2_BASE; break;
        default: std::cout << "Unknown triscv_id = " << triscv_id << std::endl; exit(1);
    }

    write_hex_vec_to_core(cluster, chip_id, core, hex_vec, addr);       // PROF_BEGIN("trisc_write")
    log_debug(tt::LogLLRuntime, "wrote trisc binary hex to the core");  // PROF_END("trisc_write")

    if (std::getenv("TT_KERNEL_READBACK_DISABLE") == nullptr) {
        std::vector<uint32_t> read_hex_vec;  // PROF_BEGIN("trisc_read_back")
        read_hex_vec = read_hex_vec_from_core(
            cluster, chip_id, core, addr, hex_vec.size() * sizeof(uint32_t));     // size to read in Bytes
        log_debug(tt::LogLLRuntime, "read trisc binary hex back from the core");  // PROF_END("trisc_read_back")
        return hex_vec == read_hex_vec;
    }
    return true;
}

void disable_ncrisc(tt_cluster *cluster, int chip_id, const tt_xy_pair &core) {
    // disable NCRISC
    uint64_t use_ncrisc_addr = RUNTIME_CONFIG_BASE;
    write_hex_vec_to_core(cluster, chip_id, core, {0}, use_ncrisc_addr);
    log_debug(tt::LogLLRuntime, "disabled ncrisc");
}

void enable_ncrisc(tt_cluster *cluster, int chip_id, const tt_xy_pair &core) {
    // enable NCRISC
    uint64_t use_ncrisc_addr = RUNTIME_CONFIG_BASE;
    write_hex_vec_to_core(cluster, chip_id, core, {1}, use_ncrisc_addr);
    log_debug(tt::LogLLRuntime, "enabled ncrisc");
}

void enable_triscs(tt_cluster *cluster, int chip_id, const tt_xy_pair &core) {
    // enable TRISCs
    uint64_t use_triscs_addr = RUNTIME_CONFIG_BASE + 4;  // TODO: need this as a dedicted const
    write_hex_vec_to_core(cluster, chip_id, core, {1}, use_triscs_addr);
    log_debug(tt::LogLLRuntime, "enabled triscs");
}

void disable_triscs(tt_cluster *cluster, int chip_id, const tt_xy_pair &core) {
    // disable TRISCs
    uint64_t use_triscs_addr = RUNTIME_CONFIG_BASE + 4;  // TODO: need this as a dedicted const
    write_hex_vec_to_core(cluster, chip_id, core, {0}, use_triscs_addr);
    log_debug(tt::LogLLRuntime, "disabled triscs");
}

WorkerCores get_worker_cores_from_cluster(tt_cluster *cluster, int chip_id) {
    WorkerCores worker_cores;
    for (tt_xy_pair raw_core : cluster->get_soc_desc(chip_id).workers) {
        TT_ASSERT(cluster->get_soc_desc(chip_id).is_worker_core(raw_core));
        worker_cores.emplace_back(chip_id, raw_core);
    }
    return worker_cores;
}

// subchannel hard-coded to 0 for now
tt_xy_pair get_core_for_dram_channel(tt_cluster *cluster, int dram_channel_id, chip_id_t chip_id) {
    return cluster->get_soc_desc(chip_id).get_core_for_dram_channel(dram_channel_id, /*sub*/ 0);
}

namespace utils {
void log_current_ai_clk(tt_cluster *cluster) {
    for (const chip_id_t &chip_id : cluster->get_all_chips()) {
        int ai_clk = cluster->get_device_aiclk(chip_id);
        log_info(tt::LogLLRuntime, "AI CLK for device {} is:   {} MHz", chip_id, ai_clk);
    }
}
}  // namespace utils

namespace internal_ {
// This loads to briscs and ncriscs - we may want to add TensixRiscsOptions here
void load_blank_kernel_to_cores(
    tt_cluster *cluster, int chip_id, const TensixRiscsOptions &riscs_to_load, std::vector<tt_xy_pair> cores) {
    TT_ASSERT(riscs_to_load != TensixRiscsOptions::NONE, "You must specify a non-NONE RISC to load blank kernels to");

    for (const tt_xy_pair &core : cores) {
        bool pass = true;

        // PROF_BEGIN("write_brisc")
        pass = test_load_write_read_risc_binary(cluster, "built_kernels/blank_op/brisc/brisc.hex", chip_id, core, 0);
        if (!pass) {
            throw std::runtime_error("Initial testing read/write of brisc to core failed");
        }  // PROF_END("write_brisc")

        if (deduce_if_involves_ncrisc(riscs_to_load)) {  // PROF_BEGIN("ncrisc")
            pass =
                test_load_write_read_risc_binary(cluster, "built_kernels/blank_op/ncrisc/ncrisc.hex", chip_id, core, 1);
            if (!pass) {
                throw std::runtime_error("Initial testing read/write of ncrisc to core failed");
            }
        }  // PROF_END("ncrisc")

        if (deduce_if_involves_triscs(riscs_to_load)) {  // PROF_BEGIN("trisc")
            string op_path = "built_kernels/blank_op";
            pass &= test_load_write_read_trisc_binary(
                cluster, op_path + "/tensix_thread0/tensix_thread0.hex", chip_id, core, 0);
            pass &= test_load_write_read_trisc_binary(
                cluster, op_path + "/tensix_thread1/tensix_thread1.hex", chip_id, core, 1);
            pass &= test_load_write_read_trisc_binary(
                cluster, op_path + "/tensix_thread2/tensix_thread2.hex", chip_id, core, 2);
            if (!pass) {
                throw std::runtime_error("Initial testing read/write of blank to trisc to core failed");
            }
        }  // PROF_END("trisc")
    }
}

void load_blank_kernel_to_all_worker_cores_with_exceptions(
    tt_cluster *cluster, int chip_id, const TensixRiscsOptions &riscs_to_load, std::vector<tt_xy_pair> exceptions) {
    std::vector<tt_xy_pair> cores_to_load_with_blanks;  // PROF_BEGIN("set_diff")
    std::set_difference(
        cluster->get_soc_desc(chip_id).workers.begin(),
        cluster->get_soc_desc(chip_id).workers.end(),
        exceptions.begin(),
        exceptions.end(),
        std::inserter(cores_to_load_with_blanks, cores_to_load_with_blanks.begin()));
    // PROF_END("set_diff")

    for (const tt_xy_pair &core : cores_to_load_with_blanks) {  // PROF_BEGIN("log_blank")
        log_debug(tt::LogLLRuntime, "loading blank to core - {}", core.str());
    }  // PROF_END("log_blank")

    load_blank_kernel_to_cores(cluster, chip_id, riscs_to_load, cores_to_load_with_blanks);
}

void enable_core(tt_cluster *cluster, int chip_id, const tt_xy_pair &core) {
    std::vector<uint32_t> enable_core_enable_val = {ENABLE_CORE_ENABLE_VALUE};
    write_hex_vec_to_core(cluster, chip_id, core, enable_core_enable_val, ENABLE_CORE_MAILBOX_ADDR);
    std::vector<uint32_t> enable_core_mailbox_init_val_check;
    enable_core_mailbox_init_val_check = read_hex_vec_from_core(
        cluster, chip_id, core, ENABLE_CORE_MAILBOX_ADDR, sizeof(uint32_t));  // read a single uint32_t
    TT_ASSERT(
        enable_core_mailbox_init_val_check[0] == ENABLE_CORE_ENABLE_VALUE,
        "val: " + std::to_string(enable_core_mailbox_init_val_check[0]));
}

void enable_cores(tt_cluster *cluster, int chip_id, const std::vector<tt_xy_pair> &cores) {
    for (const tt_xy_pair &core : cores) {
        enable_core(cluster, chip_id, core);
    }
}

void assert_enable_core_mailbox_is_valid_for_core(tt_cluster *cluster, int chip_id, const tt_xy_pair &core) {
    std::vector<uint32_t> enable_core_mailbox_init_val_check = {0};
    enable_core_mailbox_init_val_check = read_hex_vec_from_core(
        cluster, chip_id, core, ENABLE_CORE_MAILBOX_ADDR, sizeof(uint32_t));  // read a single uint32_t
    TT_ASSERT(
        enable_core_mailbox_init_val_check[0] == ENABLE_CORE_ENABLE_VALUE ||
        enable_core_mailbox_init_val_check[0] == ENABLE_CORE_DONE_VALUE);
}

void setup_riscs_on_specified_core(
    tt_cluster *cluster, int chip_id, const TensixRiscsOptions riscs_options, const tt_xy_pair &core) {
    if (riscs_options == TensixRiscsOptions::NONE) {
        TT_THROW("You can't run nothing on the riscs on core " + core.str());
    }

    bool involves_triscs = deduce_if_involves_triscs(riscs_options);
    bool involves_ncrisc = deduce_if_involves_ncrisc(riscs_options);

    std::vector<uint32_t> test_mailbox_init_val = {INIT_VALUE};

    std::function<void(uint64_t)> initialize_and_check_test_mailbox = [&](uint64_t test_mailbox_address_) {
        write_hex_vec_to_core(cluster, chip_id, core, test_mailbox_init_val, test_mailbox_address_);
        std::vector<uint32_t> test_mailbox_init_val_check;
        test_mailbox_init_val_check = read_hex_vec_from_core(
            cluster, chip_id, core, test_mailbox_address_, sizeof(uint32_t));  // read a single uint32_t
        TT_ASSERT(test_mailbox_init_val_check[0] == INIT_VALUE);
        log_debug(
            tt::LogLLRuntime,
            "checked test_mailbox is correctly initialized to value = {} for core {}",
            test_mailbox_init_val_check[0],
            core.str());
    };

    initialize_and_check_test_mailbox(TEST_MAILBOX_ADDRESS);

    if (!involves_ncrisc) {
        disable_ncrisc(cluster, chip_id, core);
    } else {
        enable_ncrisc(cluster, chip_id, core);
        initialize_and_check_test_mailbox(TEST_MAILBOX_ADDR_NCRISC);
    }

    if (!involves_triscs) {
        disable_triscs(cluster, chip_id, core);
    } else {
        enable_triscs(cluster, chip_id, core);
        // I wonder if I can loop through the addresses like this...
        for (uint32_t trisc_mailbox_address : trisc_mailbox_addresses) {
            initialize_and_check_test_mailbox(trisc_mailbox_address);
        }
    }

    enable_core(cluster, chip_id, core);
}

void setup_riscs_on_specified_cores(
    tt_cluster *cluster, int chip_id, const TensixRiscsOptions riscs_options, const std::vector<tt_xy_pair> &cores) {
    for (const tt_xy_pair &core : cores) {
        setup_riscs_on_specified_core(cluster, chip_id, riscs_options, core);
    }
}

bool check_if_riscs_on_specified_core_done(
    tt_cluster *cluster, int chip_id, const TensixRiscsOptions riscs_options, const tt_xy_pair &core) {
    bool core_is_done = true;

    std::function<bool(uint64_t)> get_mailbox_is_done = [&](uint64_t test_mailbox_address_) {
        std::vector<uint32_t> test_mailbox_read_val = {0};
        test_mailbox_read_val = read_hex_vec_from_core(
            cluster, chip_id, core, test_mailbox_address_, sizeof(uint32_t));  // read a single uint32_t
        TT_ASSERT(
            test_mailbox_read_val[0] == INIT_VALUE || test_mailbox_read_val[0] == DONE_VALUE);  // ensure no corruption
        return test_mailbox_read_val[0] == DONE_VALUE;
    };

    // brisc
    core_is_done &= get_mailbox_is_done(TEST_MAILBOX_ADDR);
    assert_enable_core_mailbox_is_valid_for_core(cluster, chip_id, core);

    if (!core_is_done) {
        return core_is_done;
    }

    // ncrisc
    bool involves_ncrisc = deduce_if_involves_ncrisc(riscs_options);
    if (involves_ncrisc) {
        core_is_done &= get_mailbox_is_done(TEST_MAILBOX_ADDR_NCRISC);
    }

    if (!core_is_done) {
        return core_is_done;
    }

    // triscs
    bool involves_triscs = deduce_if_involves_triscs(riscs_options);
    if (involves_triscs) {
        int trisc_id = 0;
        for (trisc_id = 0; trisc_id <= 2; trisc_id++) {
            core_is_done &= get_mailbox_is_done(trisc_mailbox_addresses[trisc_id]);
            if (!core_is_done) {
                return core_is_done;
            }
        }
    }

    return core_is_done;
}

void cleanup_risc_on_specified_core(
    tt_cluster *cluster, int chip_id, const TensixRiscsOptions riscs_options, const tt_xy_pair &core) {
    bool involves_triscs = deduce_if_involves_triscs(riscs_options);
    bool involves_ncrisc = deduce_if_involves_ncrisc(riscs_options);

    if (!involves_ncrisc) {
        enable_ncrisc(cluster, chip_id, core);
    }

    if (!involves_triscs) {
        enable_triscs(cluster, chip_id, core);
    }
}

void run_riscs_on_specified_cores(
    tt_cluster *cluster, int chip_id, const TensixRiscsOptions riscs_option, const std::vector<tt_xy_pair> &cores, const std::vector<uint32_t> &hugepage_done_addrs) {

    bool write_to_huge_page = hugepage_done_addrs.size() > 0;
    if (write_to_huge_page) {
        uint32_t dispatch_done_addr = 0;
        vector<uint32_t> reset = {0};
        cluster->write_sysmem_vec(reset, dispatch_done_addr, chip_id);
    }

    for (const tt_xy_pair &core_ : cores) {
        tt_cxy_pair core = tt_cxy_pair(chip_id, core_);
        cluster->set_remote_tensix_risc_reset(core, TENSIX_DEASSERT_SOFT_RESET);
    }

    if (write_to_huge_page) {
        // In this path, host polls hugepage memory rather than the cores
        // to check that they're done
        bool riscs_are_done = false;
        uint32_t dispatch_done_addr = 0;
        vector<uint32_t> reset = {0};

        vector<uint32_t> riscs_are_done_vec;
        while (not riscs_are_done) {
            riscs_are_done = true;
            // Poll hugepage to see that dispatch has completed
            uint32_t idx = 0;
            for (const tt_xy_pair &core : cores) {
                uint32_t hugepage_done_addr = hugepage_done_addrs.at(idx++);
                cluster->read_sysmem_vec(riscs_are_done_vec, dispatch_done_addr, 4, chip_id);
                riscs_are_done &= riscs_are_done_vec.at(0) == NOTIFY_HOST_KERNEL_COMPLETE_VALUE;
            }
        }
        cluster->write_sysmem_vec(reset, dispatch_done_addr, chip_id);
    } else {
        // In this path, host polls core L1 to check whether they're done
        bool riscs_are_done = false;
        while (!riscs_are_done) {
            riscs_are_done = true;
            for (const tt_xy_pair &core : cores) {
                riscs_are_done &= check_if_riscs_on_specified_core_done(cluster, chip_id, riscs_option, core);
            }
        }
    }

    for (const tt_xy_pair &core : cores) {
        cleanup_risc_on_specified_core(cluster, chip_id, riscs_option, core);
    }

    assert_reset_for_all_chips(cluster);
}


}  // namespace internal_

}  // namespace llrt

}  // namespace tt
