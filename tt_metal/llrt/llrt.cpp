#include "llrt.hpp"
#include "hostdevcommon/common_runtime_address_map.h"

#include "tools/cpuprof/cpuprof.h"

namespace tt {

// llrt = lower-level runtime
namespace llrt {

namespace fs = std::filesystem;
using std::endl;
using std::to_string;
using std::uint32_t;


bool llrt_enable_binary_cache = false; // TODO(AP): temporary
std::map<string, std::vector<uint32_t>> risc_binary_cache; // TODO(AP): temporary optimization

// made these free functions -- they're copy/paste of the member functions
// TODO: clean-up epoch_loader / epoch_binary -- a bunch of functions there should not be member functions
vector <uint32_t> get_risc_binary(string path, uint32_t id) {

    if (llrt_enable_binary_cache && risc_binary_cache.find(path) != risc_binary_cache.end()) {
        //std::cout << "-- RISC CACHE HIT FOR " << path << std::endl;
        return risc_binary_cache[path];
    } else {
        //std::cout << "-- RISC CACHE MISS FOR " << path << std::endl;
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
    ll_api::memory mem = ll_api::memory::from_discontiguous_risc_hex(hex_istream, id==1 ? ll_api::memory::NCRISC: ll_api::memory::BRISC);

    if (llrt_enable_binary_cache)
        risc_binary_cache[path] = mem.get_content(); // TODO(AP): temporary optimization

    return mem.get_content();
}

vector<uint32_t> get_trisc_binary(string path, uint32_t trisc_id) {

    //string path_to_bin = path + "/tensix_thread" + to_string(id) +
    //    "/tensix_thread" + to_string(id) + ".hex";
    if (llrt_enable_binary_cache && risc_binary_cache.find(path) != risc_binary_cache.end()) {
        //std::cout << "-- TRISC CACHE HIT FOR " << path << std::endl;
        return risc_binary_cache[path];
    } else {
        //std::cout << "-- TRISC CACHE MISS FOR " << path << std::endl;
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
    ll_api::memory mem = ll_api::memory::from_discontiguous_risc_hex(hex_istream, (ll_api::memory::risc_type_t)((int)ll_api::memory::TRISC0+trisc_id));

    if (llrt_enable_binary_cache)
        risc_binary_cache[path] = mem.get_content(); // TODO(AP): temporary optimization

    return mem.get_content();
}

// TODO: de-asserting reset properly
//  this deasserts reset for all BRISCs (on all devices, all cores), but not other RISC processors (NCRISC, TRISC)
// even though it deasserts reset for all the BRISCs, we are only loading  BRISC for a single core ("core")
// this is unsafe, since BRISCs for which we haven't loaded FW are now running garbage out of their L1
// proper solution:
// a) load dummy BRISC FW to unused cores, and keep using the function that de-asserts all BRISCs (easier, we can load blank kernel and disable NCRISC loading)
// b) de-assert reset only for used BRISCs (needs a new deassert function w/ a list of core to de-assert) (harder)
void deassert_brisc_reset_for_all_chips_all_cores(tt_cluster *cluster, bool stagger_start) {
    cluster->deassert_risc_reset(stagger_start);
    log_debug(tt::LogLLRuntime, "deasserted reset for all BRISCs");
}

// TODO: try using "stop" method from device instead, it's the proper way of asserting reset
void assert_reset_for_all_chips(tt_cluster *cluster) {
    TT_ASSERT(cluster->type == tt::TargetDevice::Silicon);

    log_debug(tt::LogLLRuntime, "Starting resets for {} chips", cluster->get_num_chips());
    for (const chip_id_t& chip_id : cluster->get_all_chips()) {
        cluster->broadcast_remote_tensix_risc_reset(chip_id, TENSIX_ASSERT_SOFT_RESET);
    }
}

// tt_xy_pair core --> NOC coordinates ("functional workers" from the SOC descriptor)
// NOC coord is also synonymous to routing / physical coord
// dram_channel id (0..7) for GS is also mapped to NOC coords in the SOC descriptor

void write_hex_vec_to_core(tt_cluster *cluster, int chip, const tt_xy_pair& core, std::vector<uint32_t> hex_vec, uint64_t addr) {

    // the API is named "write_dram_vec", and its overloaded variant is taking (chip, core) pair, ie. it can write to core's L1
    cluster->write_dram_vec(hex_vec, tt_cxy_pair(chip, core), addr);
}

std::vector<std::uint32_t> read_hex_vec_from_core(tt_cluster *cluster, int chip, const tt_xy_pair& core, uint64_t addr, uint32_t size) {
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
    return std::find(cluster->get_soc_desc(chip_id).workers.begin(), cluster->get_soc_desc(chip_id).workers.end(), core) != cluster->get_soc_desc(chip_id).workers.end();
}

CircularBufferConfigVec create_circular_buffer_config_vector() {
    CircularBufferConfigVec circular_buffer_config_vec(NUM_CIRCULAR_BUFFERS*UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG, 0); // init to 0's
    return circular_buffer_config_vec;
}

void set_config_for_circular_buffer(CircularBufferConfigVec& circular_buffer_config_vec, uint32_t circular_buffer_index, uint32_t addr_in_bytes, uint32_t size_in_bytes, uint32_t size_in_tiles) {
    circular_buffer_config_vec.at(UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * circular_buffer_index)   = addr_in_bytes >> 4; // convert to addr in 16B words
    circular_buffer_config_vec.at(UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * circular_buffer_index + 1) = size_in_bytes >> 4; // convert to addr in 16B words
    circular_buffer_config_vec.at(UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * circular_buffer_index + 2) = size_in_tiles;
}

void write_circular_buffer_config_vector_to_core(tt_cluster *cluster, int chip, const tt_xy_pair& core, CircularBufferConfigVec circular_buffer_config_vec) {
    write_hex_vec_to_core(cluster, chip, core, circular_buffer_config_vec, CIRCULAR_BUFFER_CONFIG_BASE);
}

void write_graph_interpreter_op_info_to_core(tt_cluster *cluster, int chip, const tt_xy_pair& core, op_info_t op_info, int op_idx) {
    vector<uint32_t> op_info_vec = {op_info.op_code, op_info.cb_in0_id, op_info.cb_in1_id, op_info.cb_out_id, op_info.pop0, op_info.pop1, op_info.unary};
    uint32_t offset = op_info_vec.size() * sizeof(uint32_t) * op_idx;

    write_hex_vec_to_core(cluster, chip, core, op_info_vec, OP_INFO_BASE_ADDR + offset);
}


struct RiscCacheMapKey {
    int chip_id;
    tt_xy_pair core;
    int riscv_id;
    RiscCacheMapKey(int ch, tt_xy_pair cr, int risc) : chip_id(ch), core(cr), riscv_id(risc) {}
    bool operator== (const RiscCacheMapKey& rhs) const {
        return (chip_id == rhs.chip_id) && (core == rhs.core) && (riscv_id == rhs.riscv_id);
    }
};
struct RiscCacheMapHash
{
    std::size_t operator() (const RiscCacheMapKey &k) const
    {
        using std::hash;
        return (hash<int>()(k.chip_id)
                ^ hash<uint64_t>()(k.core.x)
                ^ hash<uint64_t>()(k.core.y)
                ^ hash<uint64_t>()(k.riscv_id));
    }
};
std::unordered_map<RiscCacheMapKey, std::string, RiscCacheMapHash> last_loaded_cache; // cache what was last loaded to a specific risc

// for BRISC and NCRISC
bool test_load_write_read_risc_binary(tt_cluster *cluster, std::string hex_file_path, int chip_id, const tt_xy_pair& core, int riscv_id) {

    // PROF_BEGIN("get_risc")
    assert(riscv_id == 0 || riscv_id == 1);

    assert(is_worker_core(cluster, core, chip_id));

    // TODO(AP): temporary cache, need a proper implementation
    RiscCacheMapKey cache_key{.chip_id = chip_id, .core = core, .riscv_id = riscv_id};
    if (llrt_enable_binary_cache && last_loaded_cache.find(cache_key) != last_loaded_cache.end() && last_loaded_cache[cache_key] == hex_file_path) {
        //std::cout << "Skipping loading of " << hex_file_path << " to core " << core.x << "," << core.y << std::endl;
        return true;
    }
    if (llrt_enable_binary_cache)
        last_loaded_cache[cache_key] = hex_file_path;
    //std::cout << "Loading " << hex_file_path << " to core " << core.x << "," << core.y << std::endl;

    std::vector<uint32_t> hex_vec = get_risc_binary(hex_file_path, riscv_id); // 0 = BRISC, 1 = NCRISC

    log_debug(tt::LogLLRuntime, "hex_file_path = {}", hex_file_path);
    log_debug(tt::LogLLRuntime, "hex_vec size = {}, size_in_bytes = {}", hex_vec.size(), hex_vec.size()*sizeof(uint32_t));
    // PROF_END("get_risc")

    uint64_t addr = 0;
    if (riscv_id == 0) {
        addr = l1_mem::address_map::FIRMWARE_BASE; // BRISC binary addr in L1
    } else if (riscv_id == 1) {
        addr = l1_mem::address_map::NCRISC_FIRMWARE_BASE; // NCRISC binary addr in L1
    } else {
        std::cout << "rsicv_id = " << riscv_id << " not yet implemented" << std::endl;
        exit(1);
    }

    write_hex_vec_to_core(cluster, chip_id, core, hex_vec, addr); // PROF_BEGIN("write_risc")
    log_debug(tt::LogLLRuntime, "wrote hex to the core"); // PROF_END("write_risc")

    std::vector<uint32_t> read_hex_vec; // PROF_BEGIN("read_risc")
    read_hex_vec = read_hex_vec_from_core(cluster, chip_id, core, addr, hex_vec.size()*sizeof(uint32_t));  // size to read in Bytes
    log_debug(tt::LogLLRuntime, "read hex back from the core"); // PROF_END("read_risc")

    return hex_vec == read_hex_vec;
}

// for TRISCs
bool test_load_write_read_trisc_binary(tt_cluster *cluster, std::string hex_file_path, int chip_id, const tt_xy_pair& core, int triscv_id) {

    assert(triscv_id >=0 and triscv_id <=2);

    assert(is_worker_core(cluster, core, chip_id));

    // TODO(AP): temporary cache, need a proper implementation
    RiscCacheMapKey cache_key{.chip_id = chip_id, .core = core, .riscv_id = triscv_id+2}; // +2 to separate from NC/BR
    if (llrt_enable_binary_cache && last_loaded_cache.find(cache_key) != last_loaded_cache.end() && last_loaded_cache[cache_key] == hex_file_path) {
        //std::cout << "Skipping loading of " << hex_file_path << " to core " << core.x << "," << core.y << std::endl;
        return true;
    }
    if (llrt_enable_binary_cache)
        last_loaded_cache[cache_key] = hex_file_path;
    //std::cout << "Loading " << hex_file_path << " to core " << core.x << "," << core.y << std::endl;

    // PROF_BEGIN("trisc_get")
    std::vector<uint32_t> hex_vec = get_trisc_binary(hex_file_path, triscv_id); // TRISC 0, 1, 2
    // PROF_END("trisc_get")

    log_debug(tt::LogLLRuntime, "hex_file_path = {}", hex_file_path);
    log_debug(tt::LogLLRuntime, "hex_vec size = {}, size_in_bytes = {}", hex_vec.size(), hex_vec.size()*sizeof(uint32_t));

    uint64_t addr = 0;
    if (triscv_id == 0) {
        addr = l1_mem::address_map::TRISC0_BASE;
    } else if (triscv_id == 1) {
        addr = l1_mem::address_map::TRISC1_BASE;
    } else if (triscv_id == 2) {
        addr = l1_mem::address_map::TRISC2_BASE;
    } else {
        std::cout << "triscv_id = " << triscv_id << " is not valid" << std::endl;
        exit(1);
    }

    write_hex_vec_to_core(cluster, chip_id, core, hex_vec, addr); // PROF_BEGIN("trisc_write")
    log_debug(tt::LogLLRuntime, "wrote trisc binary hex to the core"); // PROF_END("trisc_write")

    std::vector<uint32_t> read_hex_vec; // PROF_BEGIN("trisc_read_back")
    read_hex_vec = read_hex_vec_from_core(cluster, chip_id, core, addr, hex_vec.size()*sizeof(uint32_t));  // size to read in Bytes
    log_debug(tt::LogLLRuntime, "read trisc binary hex back from the core"); // PROF_END("trisc_read_back")

    return hex_vec == read_hex_vec;
}

void disable_ncrisc(tt_cluster *cluster, int chip_id, const tt_xy_pair& core) {

    // disable NCRISC
    uint64_t use_ncrisc_addr = RUNTIME_CONFIG_BASE;
    write_hex_vec_to_core(cluster, chip_id, core, {0}, use_ncrisc_addr);
    log_debug(tt::LogLLRuntime, "disabled ncrisc");

}

void enable_ncrisc(tt_cluster *cluster, int chip_id, const tt_xy_pair& core) {

    // enable NCRISC
    uint64_t use_ncrisc_addr = RUNTIME_CONFIG_BASE;
    write_hex_vec_to_core(cluster, chip_id, core, {1}, use_ncrisc_addr);
    log_debug(tt::LogLLRuntime, "enabled ncrisc");

}

void enable_triscs(tt_cluster *cluster, int chip_id, const tt_xy_pair& core) {
    // enable TRISCs
    uint64_t use_triscs_addr = RUNTIME_CONFIG_BASE + 4; // TODO: need this as a dedicted const
    write_hex_vec_to_core(cluster, chip_id, core, {1}, use_triscs_addr);
    log_debug(tt::LogLLRuntime, "enabled triscs");
}

void disable_triscs(tt_cluster *cluster, int chip_id, const tt_xy_pair& core) {
    // disable TRISCs
    uint64_t use_triscs_addr = RUNTIME_CONFIG_BASE + 4; // TODO: need this as a dedicted const
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
    return cluster->get_soc_desc(chip_id).get_core_for_dram_channel(dram_channel_id, /*sub*/0);
}

namespace utils {
    void log_current_ai_clk(tt_cluster *cluster) {
        for (const chip_id_t& chip_id : cluster->get_all_chips()) {
            int ai_clk = cluster->get_device_aiclk(chip_id);
            log_info(tt::LogLLRuntime, "AI CLK for device {} is:   {} MHz", chip_id, ai_clk);
        }
    }
}  // namespace utils

namespace internal_ {
    // This loads to briscs and ncriscs - we may want to add TensixRiscsOptions here
    void load_blank_kernel_to_cores(tt_cluster *cluster, int chip_id, const TensixRiscsOptions &riscs_to_load, std::vector<tt_xy_pair> cores) {
        TT_ASSERT(riscs_to_load != TensixRiscsOptions::NONE, "You must specify a non-NONE RISC to load blank kernels to");

        for (const tt_xy_pair& core : cores) {
            bool pass  = true;

            // PROF_BEGIN("write_brisc")
            pass = test_load_write_read_risc_binary(cluster, "built_kernels/blank_op/brisc/brisc.hex", chip_id, core, 0);
            if (!pass) {
                throw std::runtime_error("Initial testing read/write of brisc to core failed");
            } // PROF_END("write_brisc")

            if (deduce_if_involves_ncrisc(riscs_to_load)) { // PROF_BEGIN("ncrisc")
                pass = test_load_write_read_risc_binary(cluster, "built_kernels/blank_op/ncrisc/ncrisc.hex", chip_id, core, 1);
                if (!pass) {
                    throw std::runtime_error("Initial testing read/write of ncrisc to core failed");
                }
            } // PROF_END("ncrisc")

            if (deduce_if_involves_triscs(riscs_to_load)) { // PROF_BEGIN("trisc")
                string op_path = "built_kernels/blank_op";
                pass &= test_load_write_read_trisc_binary(cluster, op_path + "/tensix_thread0/tensix_thread0.hex", chip_id, core, 0);
                pass &= test_load_write_read_trisc_binary(cluster, op_path + "/tensix_thread1/tensix_thread1.hex", chip_id, core, 1);
                pass &= test_load_write_read_trisc_binary(cluster, op_path + "/tensix_thread2/tensix_thread2.hex", chip_id, core, 2);
                if (!pass) {
                    throw std::runtime_error("Initial testing read/write of blank to trisc to core failed");
                }
            } // PROF_END("trisc")
        }
    }

    void load_blank_kernel_to_all_worker_cores_with_exceptions(tt_cluster *cluster, int chip_id, const TensixRiscsOptions &riscs_to_load, std::vector<tt_xy_pair> exceptions) {
        std::vector<tt_xy_pair> cores_to_load_with_blanks; // PROF_BEGIN("set_diff")
        std::set_difference(cluster->get_soc_desc(chip_id).workers.begin(), cluster->get_soc_desc(chip_id).workers.end(), exceptions.begin(), exceptions.end(), std::inserter(cores_to_load_with_blanks, cores_to_load_with_blanks.begin()));
        // PROF_END("set_diff")

        for (const tt_xy_pair &core : cores_to_load_with_blanks) { // PROF_BEGIN("log_blank")
            log_debug(tt::LogLLRuntime, "loading blank to core - {}", core.str());
        } // PROF_END("log_blank")

        load_blank_kernel_to_cores(cluster, chip_id, riscs_to_load, cores_to_load_with_blanks);
    }

    void enable_core(tt_cluster *cluster, int chip_id, const tt_xy_pair &core) {
        std::vector<uint32_t> enable_core_enable_val = {ENABLE_CORE_ENABLE_VALUE};
        write_hex_vec_to_core(cluster, chip_id, core, enable_core_enable_val, ENABLE_CORE_MAILBOX_ADDR);
        std::vector<uint32_t> enable_core_mailbox_init_val_check;
        enable_core_mailbox_init_val_check = read_hex_vec_from_core(cluster, chip_id, core, ENABLE_CORE_MAILBOX_ADDR, sizeof(uint32_t));  // read a single uint32_t
        TT_ASSERT(enable_core_mailbox_init_val_check[0] == ENABLE_CORE_ENABLE_VALUE, "val: " + std::to_string(enable_core_mailbox_init_val_check[0]));
    }

    void enable_cores(tt_cluster *cluster, int chip_id, const std::vector<tt_xy_pair> &cores) {
        for (const tt_xy_pair &core : cores) {
            enable_core(cluster, chip_id, core);
        }
    }

    void assert_enable_core_mailbox_is_valid_for_core(tt_cluster *cluster, int chip_id, const tt_xy_pair &core) {
        std::vector<uint32_t> enable_core_mailbox_init_val_check = {0};
        enable_core_mailbox_init_val_check = read_hex_vec_from_core(cluster, chip_id, core, ENABLE_CORE_MAILBOX_ADDR, sizeof(uint32_t));  // read a single uint32_t
        TT_ASSERT(enable_core_mailbox_init_val_check[0] == ENABLE_CORE_ENABLE_VALUE || enable_core_mailbox_init_val_check[0] == ENABLE_CORE_DONE_VALUE);
    }

    void setup_riscs_on_specified_core(tt_cluster *cluster, int chip_id, const TensixRiscsOptions riscs_options, const tt_xy_pair &core) {
        if (riscs_options == TensixRiscsOptions::NONE) {
            TT_THROW("You can't run nothing on the riscs on core " + core.str());
        }

        bool involves_triscs = deduce_if_involves_triscs(riscs_options);
        bool involves_ncrisc = deduce_if_involves_ncrisc(riscs_options);

        std::vector<uint32_t> test_mailbox_init_val = {INIT_VALUE};

        std::function<void(uint64_t)> initialize_and_check_test_mailbox = [&](uint64_t test_mailbox_address_) {
            write_hex_vec_to_core(cluster, chip_id, core, test_mailbox_init_val, test_mailbox_address_);
            std::vector<uint32_t> test_mailbox_init_val_check;
            test_mailbox_init_val_check = read_hex_vec_from_core(cluster, chip_id, core, test_mailbox_address_, sizeof(uint32_t));  // read a single uint32_t
            TT_ASSERT(test_mailbox_init_val_check[0] == INIT_VALUE);
            log_debug(tt::LogLLRuntime, "checked test_mailbox is correctly initialized to value = {} for core {}", test_mailbox_init_val_check[0], core.str());
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

    void setup_riscs_on_specified_cores(tt_cluster *cluster, int chip_id, const TensixRiscsOptions riscs_options, const std::vector<tt_xy_pair> &cores) {
        for (const tt_xy_pair &core : cores) {
            setup_riscs_on_specified_core(cluster, chip_id, riscs_options, core);
        }
    }

    bool check_if_riscs_on_specified_core_done(tt_cluster *cluster, int chip_id, const TensixRiscsOptions riscs_options, const tt_xy_pair &core) {
        bool core_is_done = true;


        std::function<bool(uint64_t)> get_mailbox_is_done = [&](uint64_t test_mailbox_address_) {
            std::vector<uint32_t> test_mailbox_read_val = {0};
            test_mailbox_read_val = read_hex_vec_from_core(cluster, chip_id, core, test_mailbox_address_, sizeof(uint32_t));  // read a single uint32_t
            TT_ASSERT(test_mailbox_read_val[0] == INIT_VALUE || test_mailbox_read_val[0] == DONE_VALUE); // ensure no corruption
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

    void cleanup_risc_on_specified_core(tt_cluster *cluster, int chip_id, const TensixRiscsOptions riscs_options, const tt_xy_pair &core) {
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
        tt_cluster *cluster, int chip_id, const TensixRiscsOptions riscs_option, const std::vector<tt_xy_pair> &cores) {

        deassert_brisc_reset_for_all_chips_all_cores(cluster);

        bool riscs_are_done = false;
        while (!riscs_are_done) {
            riscs_are_done = true;
            for (const tt_xy_pair &core : cores) {
                riscs_are_done &= check_if_riscs_on_specified_core_done(cluster, chip_id, riscs_option, core);
            }
        }

        for (const tt_xy_pair &core : cores) {
            cleanup_risc_on_specified_core(cluster, chip_id, riscs_option, core);
        }

        assert_reset_for_all_chips(cluster);

    }

    void run_briscs_on_specified_cores(tt_cluster *cluster, int chip_id, const std::vector<tt_xy_pair> &cores) {
        static const TensixRiscsOptions riscs_option = TensixRiscsOptions::BRISC_ONLY;

        setup_riscs_on_specified_cores(cluster, chip_id, riscs_option, cores);

        run_riscs_on_specified_cores(cluster, chip_id, riscs_option, cores);
    }

} // namespace internal_

/*
 * DRAM COPY
 */

std::vector<uint32_t> get_arg_hex_from_dram_copy_kernel_spec(tt_cluster *cluster, int chip_id, const DramCopySpec &spec) {
    SrcChannelId dram_src_channel_id;
    DstChannelId dram_dst_channel_id;
    tt_xy_pair core;
    DramBufferSize dram_buffer_size;
    DramSrcAddr dram_buffer_src_addr;
    DramDstAddr dram_buffer_dst_addr;
    LoadFirmwareFlag load_firmware_flag;

    std::tie(core, dram_src_channel_id, dram_dst_channel_id, dram_buffer_size, dram_buffer_src_addr, dram_buffer_dst_addr, load_firmware_flag) = spec;

    // Kernel arguments
    std::uint32_t l1_buffer_addr = 300 * 1024;

    tt_xy_pair dram_src_noc_xy = cluster->get_soc_desc(chip_id).get_core_for_dram_channel(dram_src_channel_id, /*sub*/0);

    tt_xy_pair dram_dst_noc_xy = cluster->get_soc_desc(chip_id).get_core_for_dram_channel(dram_dst_channel_id, /*sub*/0);

    std::uint32_t chunk_size = 32 * 1024;

    // blast the src buffer to DRAM
    TT_ASSERT(dram_buffer_size % sizeof(std::uint32_t) == 0);

    return {
        l1_buffer_addr,
        dram_buffer_src_addr,
        (std::uint32_t)dram_src_noc_xy.x,
        (std::uint32_t)dram_src_noc_xy.y,

        dram_buffer_dst_addr,
       (std::uint32_t) dram_dst_noc_xy.x,
       (std::uint32_t) dram_dst_noc_xy.y,

        dram_buffer_size,
        chunk_size
    };
}

void run_dram_copy_kernel_with_specs(tt_cluster *cluster, int chip_id, std::vector<DramCopySpec> specs, bool load_blanks) {
    // Hard-code this for now, but will be an option from user functions
    const TensixRiscsOptions riscs_options = TensixRiscsOptions::BRISC_ONLY;

    const std::function<bool(DramCopySpec)> spec_requires_loading = [](const DramCopySpec& spec) { return std::get<6>(spec); };

    std::vector<DramCopySpec> specs_to_load;
    std::copy_if(specs.begin(), specs.end(), std::back_inserter(specs_to_load), spec_requires_loading);

    const std::function<tt_xy_pair(DramCopySpec)> get_core_from_dram_copy_spec = [](const DramCopySpec& spec) { return std::get<0>(spec); };

    std::vector<tt_xy_pair> cores_to_load;
    std::transform(specs_to_load.begin(), specs_to_load.end(), std::inserter(cores_to_load, cores_to_load.begin()), get_core_from_dram_copy_spec);

    bool pass;
    for (const tt_xy_pair& core : cores_to_load) {
        pass = test_load_write_read_risc_binary(cluster, "built_kernels/dram_copy_looped/brisc/brisc.hex", chip_id, core, 0);
        if (!pass) {
            throw std::runtime_error("Initial testing read/write of brisc to core failed");
        }
    }

    // The set of cores to load blanks to needs to be calculated one
    // better... we'll need previous state of blank cores as well So we can
    // get the set of cores that need to be loaded with blank for current
    // set of specs, then take away cores that already have a blank on them
    std::vector<tt_xy_pair> cores_that_will_execute;
    std::transform(specs.begin(), specs.end(), std::back_inserter(cores_that_will_execute), get_core_from_dram_copy_spec);
    if (load_blanks) {
        internal_::load_blank_kernel_to_all_worker_cores_with_exceptions(cluster, chip_id, riscs_options, cores_that_will_execute);
    }

    for (const DramCopySpec& spec : specs) {
        const tt_xy_pair core = get_core_from_dram_copy_spec(spec);

        // blast kernel arguments to L1 in one-shot
        write_hex_vec_to_core(cluster, chip_id, core,
            get_arg_hex_from_dram_copy_kernel_spec(cluster, chip_id, spec), BRISC_L1_ARG_BASE);
    }

    // Hard-code brisc only for now
    internal_::run_briscs_on_specified_cores(cluster, chip_id, cores_that_will_execute);
}

/*
 * RAM COPY
 */

RamCopySpec create_ram_copy_spec(
    tt_xy_pair core,
    SrcL1Cores src_l1_cores,
    DstL1Cores dst_l1_cores,
    DramBufferSize dram_buffer_size,
    DramSrcAddr dram_src_addr,
    DramDstAddr dram_dst_addr,
    tiles_test::TileSize tile_size,
    tiles_test::TileIndex tile_index,
    CountOffset src_core_count_offset,
    CountOffset dst_core_count_offset,
    LoadFirmwareFlag load_firmware_flag
) {
    return std::make_tuple(core, src_l1_cores, dst_l1_cores, dram_buffer_size, dram_src_addr, dram_dst_addr, tile_size, tile_index, src_core_count_offset, dst_core_count_offset, load_firmware_flag);
}

std::vector<uint32_t> get_arg_hex_from_ram_copy_kernel_spec(tt_cluster *cluster, int chip_id, const RamCopySpec &spec) {
    tt_xy_pair core;
    SrcL1Cores src_l1_cores;
    DstL1Cores dst_l1_cores;
    DramBufferSize dram_buffer_size;
    DramSrcAddr dram_buffer_src_addr;
    DramDstAddr dram_buffer_dst_addr;
    tiles_test::TileSize tile_size;
    tiles_test::TileIndex tile_index;
    CountOffset src_core_count_offset;
    CountOffset dst_core_count_offset;
    LoadFirmwareFlag load_firmware_flag;

    std::tie(core, src_l1_cores, dst_l1_cores, dram_buffer_size, dram_buffer_src_addr, dram_buffer_dst_addr, tile_size, tile_index, src_core_count_offset, dst_core_count_offset, load_firmware_flag) = spec;

    int num_of_src_cores = src_l1_cores.size();
    int num_of_dst_cores = dst_l1_cores.size();

    SrcL1Core src_l1_core = src_l1_cores[tiles_test::get_src_core_index_from_tile_index(tile_index, num_of_src_cores, src_core_count_offset)];
    DstL1Core dst_l1_core = dst_l1_cores[tiles_test::get_src_core_index_from_tile_index(tile_index, num_of_dst_cores, dst_core_count_offset)];

    // Kernel arguments
    std::uint32_t l1_buffer_addr = 200 * 1024;

    std::uint32_t chunk_size = 1 * 1024;

    // blast the src buffer to DRAM
    TT_ASSERT(dram_buffer_size % sizeof(std::uint32_t) == 0);

    return {
        l1_buffer_addr,
        dram_buffer_src_addr,
        (std::uint32_t)src_l1_core.x,
        (std::uint32_t)src_l1_core.y,

        dram_buffer_dst_addr,
       (std::uint32_t) dst_l1_core.x,
       (std::uint32_t) dst_l1_core.y,

        dram_buffer_size,
        chunk_size,
    };
}

void run_ram_copy_kernel_with_specs(tt_cluster *cluster, int chip_id, std::vector<RamCopySpec> specs, bool load_blanks) {
    // Hard-code this for now, but will be an option from user functions
    const TensixRiscsOptions riscs_options = TensixRiscsOptions::BRISC_ONLY;

    const std::function<LoadFirmwareFlag(RamCopySpec)> spec_requires_loading = [](const RamCopySpec& spec) { return std::get<10>(spec); };

    std::vector<RamCopySpec> specs_to_load;
    std::copy_if(specs.begin(), specs.end(), std::back_inserter(specs_to_load), spec_requires_loading);

    const std::function<tt_xy_pair(RamCopySpec)> get_core_from_ram_copy_spec = [](const RamCopySpec& spec) { return std::get<0>(spec); };

    std::vector<tt_xy_pair> cores_to_load;
    std::transform(specs_to_load.begin(), specs_to_load.end(), std::inserter(cores_to_load, cores_to_load.begin()), get_core_from_ram_copy_spec);

    bool pass;
    for (const tt_xy_pair& core : cores_to_load) {
        pass = test_load_write_read_risc_binary(cluster, "built_kernels/dram_copy_looped/brisc/brisc.hex", chip_id, core, 0);
        if (!pass) {
            throw std::runtime_error("Initial testing read/write of brisc to core failed");
        }
        log_warning(tt::LogLLRuntime, "loaded dram_copy_looped as ram_copy kernel for core - {}", core.str());
    }

    // The set of cores to load blanks to needs to be calculated one
    // better... we'll need previous state of blank cores as well So we can
    // get the set of cores that need to be loaded with blank for current
    // set of specs, then take away cores that already have a blank on them
    std::vector<tt_xy_pair> cores_that_will_execute;
    std::transform(specs.begin(), specs.end(), std::back_inserter(cores_that_will_execute), get_core_from_ram_copy_spec);
    if (load_blanks) {
        internal_::load_blank_kernel_to_all_worker_cores_with_exceptions(cluster, chip_id, riscs_options, cores_that_will_execute);
    }

    for (const RamCopySpec& spec : specs) {
        const tt_xy_pair core = get_core_from_ram_copy_spec(spec);

        // blast kernel arguments to L1 in one-shot
        write_hex_vec_to_core(cluster, chip_id, core,
            get_arg_hex_from_ram_copy_kernel_spec(cluster, chip_id, spec), BRISC_L1_ARG_BASE);
    }

    // Hard-code brisc only for now
    internal_::run_briscs_on_specified_cores(cluster, chip_id, cores_that_will_execute);
}

/*
 * DRAM TO L1 COPY
 */

DramToL1CopySpec create_dram_to_l1_copy_spec(
    tt_xy_pair core,
    SrcChannelId src_channel_id,
    DramBufferSize dram_buffer_size,
    DramSrcAddr dram_src_addr,
    L1Addr l1_addr,
    LoadFirmwareFlag load_firmware_flag
) {
    return std::make_tuple(core, src_channel_id, dram_buffer_size, dram_src_addr, l1_addr, load_firmware_flag);
}

std::vector<uint32_t> get_arg_hex_from_dram_to_l1_copy_kernel_spec(tt_cluster *cluster, int chip_id, const DramToL1CopySpec &spec) {
    tt_xy_pair core;
    SrcChannelId src_channel_id;
    DramBufferSize dram_buffer_size;
    DramSrcAddr dram_src_addr;
    L1Addr l1_addr;
    LoadFirmwareFlag load_firmware_flag;

    std::tie(core, src_channel_id, dram_buffer_size, dram_src_addr, l1_addr, load_firmware_flag) = spec;

    tt_xy_pair dram_src_noc_xy = cluster->get_soc_desc(chip_id).get_core_for_dram_channel(src_channel_id, /*sub*/0);

    // blast the src buffer to DRAM
    TT_ASSERT(dram_buffer_size % sizeof(std::uint32_t) == 0);

    return {
        dram_src_addr,
        (std::uint32_t)dram_src_noc_xy.x,
        (std::uint32_t)dram_src_noc_xy.y,

        l1_addr,
        dram_buffer_size,
    };
}

void run_dram_to_l1_copy_kernel_with_specs(tt_cluster *cluster, int chip_id, std::vector<DramToL1CopySpec> specs, bool load_blanks) {
    // Hard-code this for now, but will be an option from user functions
    const TensixRiscsOptions riscs_options = TensixRiscsOptions::BRISC_ONLY;

    const std::function<LoadFirmwareFlag(DramToL1CopySpec)> spec_requires_loading = [](const DramToL1CopySpec& spec) { return std::get<5>(spec); };

    std::vector<DramToL1CopySpec> specs_to_load;
    std::copy_if(specs.begin(), specs.end(), std::back_inserter(specs_to_load), spec_requires_loading);

    const std::function<tt_xy_pair(DramToL1CopySpec)> get_core_from_ram_copy_spec = [](const DramToL1CopySpec& spec) { return std::get<0>(spec); };

    std::vector<tt_xy_pair> cores_to_load;
    std::transform(specs_to_load.begin(), specs_to_load.end(), std::inserter(cores_to_load, cores_to_load.begin()), get_core_from_ram_copy_spec);

    bool pass;
    for (const tt_xy_pair& core : cores_to_load) {
        pass = test_load_write_read_risc_binary(cluster, "built_kernels/dram_to_l1_copy/brisc/brisc.hex", chip_id, core, 0);
        if (!pass) {
            throw std::runtime_error("Initial testing read/write of brisc to core failed");
        }
    }

    // The set of cores to load blanks to needs to be calculated one
    // better... we'll need previous state of blank cores as well So we can
    // get the set of cores that need to be loaded with blank for current
    // set of specs, then take away cores that already have a blank on them
    std::vector<tt_xy_pair> cores_that_will_execute;
    std::transform(specs.begin(), specs.end(), std::back_inserter(cores_that_will_execute), get_core_from_ram_copy_spec);
    if (load_blanks) {
        internal_::load_blank_kernel_to_all_worker_cores_with_exceptions(cluster, chip_id, riscs_options, cores_that_will_execute);
    }

    for (const DramToL1CopySpec& spec : specs) {
        const tt_xy_pair core = get_core_from_ram_copy_spec(spec);

        // blast kernel arguments to L1 in one-shot
        write_hex_vec_to_core(cluster, chip_id, core,
            get_arg_hex_from_dram_to_l1_copy_kernel_spec(cluster, chip_id, spec), BRISC_L1_ARG_BASE);
    }

    // Hard-code brisc only for now
    internal_::run_briscs_on_specified_cores(cluster, chip_id, cores_that_will_execute);
}

/*
 * DRAM TO L1 COPY PATTERN
*/

CopyPatternSpec create_copy_pattern_spec(
    tt_xy_pair dest_core,
    DestAddr dest_addr,
    tt_xy_pair src_core,
    SrcAddr src_addr,
    NCHW nchw,
    RSUV rsuv,
    BYTES_PER_DATUM bytes_per_datum,
    NUM_REPETITIONS num_repetitions,
    LoadFirmwareFlag load_firmware_flag
) {
    return std::make_tuple(dest_core, dest_addr, src_core, src_addr, nchw, rsuv, bytes_per_datum, num_repetitions, load_firmware_flag);
}

std::vector<uint32_t> get_arg_hex_from_copy_pattern_kernel_spec(tt_cluster *cluster, int chip_id, const CopyPatternSpec &spec) {
    tt_xy_pair dest_core;
    DestAddr dest_addr;
    tt_xy_pair src_core;
    SrcAddr src_addr;
    NCHW nchw;
    RSUV rsuv;
    BYTES_PER_DATUM bytes_per_datum;
    NUM_REPETITIONS num_repetitions;
    LoadFirmwareFlag load_firmware_flag;

    std::tie(dest_core, dest_addr, src_core, src_addr, nchw, rsuv, bytes_per_datum, num_repetitions, load_firmware_flag) = spec;
    TT_ASSERT(rsuv[2] > 0  and rsuv[3] > 0);
    // We only support moving sticks of byte size multiple of 32B
    // since each datum is 4B (uint32_t), then channels must be multiple of 8
    TT_ASSERT(nchw[1] % 8 == 0);
    std::uint32_t c_bits = log2(nchw[1]);
    std::uint32_t byte_shift = log2(bytes_per_datum);
    return {
        src_addr,
        (std::uint32_t)src_core.x,
        (std::uint32_t)src_core.y,
        dest_addr,
        nchw[0], //N
        nchw[1], //C
        nchw[2], //H
        nchw[3], //W
        rsuv[0], //R
        rsuv[1], //S
        rsuv[2], //U (vertical stride)
        rsuv[3], //V (horizontal stride)
        nchw[3] * rsuv[2], //W * U
        c_bits,
        byte_shift,
        num_repetitions
    };
}

void run_copy_pattern_kernel_with_specs(tt_cluster *cluster, int chip_id, std::vector<CopyPatternSpec> specs, bool load_blanks) {
    // Hard-code this for now, but will be an option from user functions
    const TensixRiscsOptions riscs_options = TensixRiscsOptions::BRISC_ONLY;

    const std::function<LoadFirmwareFlag(CopyPatternSpec)> spec_requires_loading = [](const CopyPatternSpec& spec) { return std::get<8>(spec); };

    std::vector<CopyPatternSpec> specs_to_load;
    std::copy_if(specs.begin(), specs.end(), std::back_inserter(specs_to_load), spec_requires_loading);

    const std::function<tt_xy_pair(CopyPatternSpec)> get_core_from_copy_pattern_spec = [](const CopyPatternSpec& spec) { return std::get<0>(spec); };

    std::vector<tt_xy_pair> cores_to_load;
    std::transform(specs_to_load.begin(), specs_to_load.end(), std::inserter(cores_to_load, cores_to_load.begin()), get_core_from_copy_pattern_spec);

    bool pass;
    for (const tt_xy_pair& core : cores_to_load) {
        pass = test_load_write_read_risc_binary(cluster, "built_kernels/copy_pattern/brisc/brisc.hex", chip_id, core, 0);
        if (!pass) {
            throw std::runtime_error("Initial testing read/write of brisc to core failed");
        }
    }

    // The set of cores to load blanks to needs to be calculated one
    // better... we'll need previous state of blank cores as well So we can
    // get the set of cores that need to be loaded with blank for current
    // set of specs, then take away cores that already have a blank on them
    std::vector<tt_xy_pair> cores_that_will_execute;
    std::transform(specs.begin(), specs.end(), std::back_inserter(cores_that_will_execute), get_core_from_copy_pattern_spec);
    if (load_blanks) {
        internal_::load_blank_kernel_to_all_worker_cores_with_exceptions(cluster, chip_id, riscs_options, cores_that_will_execute);
    }

    for (const CopyPatternSpec& spec : specs) {
        const tt_xy_pair core = get_core_from_copy_pattern_spec(spec);

        // blast kernel arguments to L1 in one-shot
        write_hex_vec_to_core(cluster, chip_id, core,
            get_arg_hex_from_copy_pattern_kernel_spec(cluster, chip_id, spec), BRISC_L1_ARG_BASE);
    }

    // Hard-code brisc only for now
    internal_::run_briscs_on_specified_cores(cluster, chip_id, cores_that_will_execute);
}

/*
 * L1 TO DRAM COPY
 */

L1ToDramCopySpec create_l1_to_dram_copy_spec(
    tt_xy_pair core,
    DstChannelId dst_channel_id,
    DramBufferSize dram_buffer_size,
    DramDstAddr dram_dst_addr,
    L1Addr l1_addr,
    LoadFirmwareFlag load_firmware_flag
) {
    return std::make_tuple(core, dst_channel_id, dram_buffer_size, dram_dst_addr, l1_addr, load_firmware_flag);
}

std::vector<uint32_t> get_arg_hex_from_l1_to_dram_copy_kernel_spec(tt_cluster *cluster, int chip_id, const L1ToDramCopySpec &spec) {
    tt_xy_pair core;
    DstChannelId dst_channel_id;
    DramBufferSize dram_buffer_size;
    DramSrcAddr dram_dst_addr;
    L1Addr l1_addr;
    LoadFirmwareFlag load_firmware_flag;

    std::tie(core, dst_channel_id, dram_buffer_size, dram_dst_addr, l1_addr, load_firmware_flag) = spec;

    tt_xy_pair dram_dst_noc_xy = cluster->get_soc_desc(chip_id).get_core_for_dram_channel(dst_channel_id, /*sub*/0);

    // blast the src L1 buffer to DRAM in kernel
    TT_ASSERT(dram_buffer_size % sizeof(std::uint32_t) == 0);

    return {
        dram_dst_addr,
        (std::uint32_t)dram_dst_noc_xy.x,
        (std::uint32_t)dram_dst_noc_xy.y,

        l1_addr,
        dram_buffer_size,
    };
}

void run_l1_to_dram_copy_kernel_with_specs(tt_cluster *cluster, int chip_id, std::vector<L1ToDramCopySpec> specs, bool load_blanks) {
    // Hard-code this for now, but will be an option from user functions
    const TensixRiscsOptions riscs_options = TensixRiscsOptions::BRISC_NCRISC;

    const std::function<LoadFirmwareFlag(L1ToDramCopySpec)> spec_requires_loading = [](const L1ToDramCopySpec& spec) { return std::get<5>(spec); };

    std::vector<L1ToDramCopySpec> specs_to_load;
    std::copy_if(specs.begin(), specs.end(), std::back_inserter(specs_to_load), spec_requires_loading);

    const std::function<tt_xy_pair(L1ToDramCopySpec)> get_core_from_ram_copy_spec = [](const L1ToDramCopySpec& spec) { return std::get<0>(spec); };

    std::vector<tt_xy_pair> cores_to_load;
    std::transform(specs_to_load.begin(), specs_to_load.end(), std::inserter(cores_to_load, cores_to_load.begin()), get_core_from_ram_copy_spec);

    // This is being done first because of the issue of loading blanks to all briscs and ncriscs
    std::vector<tt_xy_pair> cores_that_will_execute;
    std::transform(specs.begin(), specs.end(), std::back_inserter(cores_that_will_execute), get_core_from_ram_copy_spec);
    if (load_blanks) {
        // load blank to brisc on this worker core so that it executes nothing, but will trigger ncrisc
        internal_::load_blank_kernel_to_all_worker_cores_with_exceptions(cluster, chip_id, riscs_options, {});
    }

    bool pass;
    for (const tt_xy_pair& core : cores_to_load) {
        const int NCRISC_ID = 1;
        pass = test_load_write_read_risc_binary(cluster, "built_kernels/l1_to_dram_copy/ncrisc/ncrisc.hex", chip_id, core, NCRISC_ID);
        if (!pass) {
            throw std::runtime_error("Initial testing read/write of ncrisc to core failed");
        }
    }

    for (const L1ToDramCopySpec& spec : specs) {
        const tt_xy_pair core = get_core_from_ram_copy_spec(spec);

        // blast kernel arguments to L1 in one-shot TO NCRISC!!!
        write_hex_vec_to_core(cluster, chip_id, core,
            get_arg_hex_from_l1_to_dram_copy_kernel_spec(cluster, chip_id, spec), NCRISC_L1_ARG_BASE);
    }

    // Hard-code brisc + ncrisc only for now
    internal_::setup_riscs_on_specified_cores(cluster, chip_id, riscs_options, cores_that_will_execute);

    internal_::run_riscs_on_specified_cores(cluster, chip_id, riscs_options, cores_that_will_execute);
}

} // namespace runtime

} // namespace tt
