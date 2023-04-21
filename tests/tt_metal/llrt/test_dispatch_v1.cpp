#include "dispatch/dispatch_helper_functions.hpp"
#include "llrt/tt_debug_print_server.hpp"

uint32_t nearest_multiple_of_32(uint32_t addr) { return ceil(float(addr) / 32) * 32; }

void assert_32B_alignment(uint32_t addr) { TT_ASSERT((addr % 32) == 0, "Address must be 32B-aligned"); }

// ________ __________    _____      _____
// \______ \\______   \  /  _  \    /     \
//  |    |  \|       _/ /  /_\  \  /  \ /  \
//  |    `   \    |   \/    |    \/    Y    \
// /_______  /____|_  /\____|__  /\____|__  /
//         \/       \/         \/         \/
void assert_valid_dram_data_for_datacopy(const DramConfig &data) {
    tt::log_info(tt::LogTest, "Performing asserts on validity of dram data");

    TT_ASSERT(data.kernels.size() == 5, "Expected 5 kernels to be present.");
    TT_ASSERT(
        data.runtime_args.size(), "Expected two sets of runtime args for the reader and writer kernels respectively.");
    TT_ASSERT(data.cb_configs.size() == 2, "Expected there to be two CB configs (one for input, one for output).");

    TT_ASSERT(
        data.kernels.size() == data.kernel_addrs_and_sizes.size(), "Num kernels needs to match num kernel addresses.");
    TT_ASSERT(
        data.runtime_args.size() == data.runtime_args_addrs_and_sizes.size(),
        "Num runtime args vectors needs to match num runtime vector addresses.");
    TT_ASSERT(
        data.cb_configs.size() == data.cb_configs_addrs_and_sizes.size(),
        "Num cb configs needs to match num cb config addresses.");

    vector<pair<uint32_t, uint32_t>> addrs_and_sizes;
    addrs_and_sizes.insert(
        addrs_and_sizes.end(), data.kernel_addrs_and_sizes.begin(), data.kernel_addrs_and_sizes.end());
    addrs_and_sizes.insert(
        addrs_and_sizes.end(), data.runtime_args_addrs_and_sizes.begin(), data.runtime_args_addrs_and_sizes.end());
    addrs_and_sizes.insert(
        addrs_and_sizes.end(), data.cb_configs_addrs_and_sizes.begin(), data.cb_configs_addrs_and_sizes.end());

    uint32_t total_size = 0;
    for (const auto &[dram_addr, size] : addrs_and_sizes) {
        assert_32B_alignment(dram_addr);
        total_size += size;
    }

    TT_ASSERT(
        total_size < 1024 * 1024,
        "For this test, we require all kernel binaries to fit into L1, therefore the number of bytes must be less than "
        "1MB");
}

void write_to_dram(tt_cluster *cluster, int chip_id, const DramConfig &data) {
    assert_valid_dram_data_for_datacopy(data);

    auto send_to_dram = [&](vector<uint32_t> vec, uint32_t addr) {
        cluster->write_dram_vec(vec, tt_target_dram{chip_id, 0, 0}, addr);
    };

    for (uint32_t kernel_id = 0; kernel_id < data.kernels.size(); kernel_id++) {
        send_to_dram(data.kernels.at(kernel_id), data.kernel_addrs_and_sizes.at(kernel_id).first);
    }

    for (uint32_t rt_args_id = 0; rt_args_id < data.runtime_args.size(); rt_args_id++) {
        send_to_dram(data.runtime_args.at(rt_args_id), data.runtime_args_addrs_and_sizes.at(rt_args_id).first);
    }

    for (uint32_t cb_config_id = 0; cb_config_id < data.cb_configs.size(); cb_config_id++) {
        send_to_dram(data.cb_configs.at(cb_config_id), data.cb_configs_addrs_and_sizes.at(cb_config_id).first);
    }
}

vector<uint32_t> allocate(
    uint32_t &addr, uint32_t &size, const vector<uint32_t> &data, vector<pair<uint32_t, uint32_t>> &addr_and_size) {
    size = data.size() * sizeof(uint32_t);
    addr_and_size.push_back(pair(addr, size));
    addr = nearest_multiple_of_32(addr + size);

    return data;
}

DramConfig construct_dram_config(string op) {
    string op_path = "built_kernels/" + op;
    std::array<string, 5> datacopy_hex_files = {
        op_path + "/brisc/brisc.hex",
        op_path + "/tensix_thread0/tensix_thread0.hex",
        op_path + "/tensix_thread1/tensix_thread1.hex",
        op_path + "/tensix_thread2/tensix_thread2.hex",
        op_path + "/ncrisc/ncrisc.hex",
    };
    const auto [brisc_hex_path, trisc0_hex_path, trisc1_hex_path, trisc2_hex_path, ncrisc_hex_path] =
        datacopy_hex_files;

    uint32_t dram_addr = 0;
    uint32_t size = 0;
    vector<pair<uint32_t, uint32_t>> kernel_addrs_and_sizes;

    const auto allocate_helper = [&dram_addr, &size](
                                     const vector<uint32_t> &vec, vector<pair<uint32_t, uint32_t>> &addrs_and_sizes) {
        return allocate(dram_addr, size, vec, addrs_and_sizes);
    };

    vector<uint32_t> brisc_hex = allocate_helper(tt::llrt::get_risc_binary(brisc_hex_path, 0), kernel_addrs_and_sizes);
    vector<uint32_t> ncrisc_hex =
        allocate_helper(tt::llrt::get_risc_binary(ncrisc_hex_path, 1), kernel_addrs_and_sizes);
    vector<uint32_t> trisc0_hex =
        allocate_helper(tt::llrt::get_trisc_binary(trisc0_hex_path, 0), kernel_addrs_and_sizes);
    vector<uint32_t> trisc1_hex =
        allocate_helper(tt::llrt::get_trisc_binary(trisc1_hex_path, 1), kernel_addrs_and_sizes);
    vector<uint32_t> trisc2_hex =
        allocate_helper(tt::llrt::get_trisc_binary(trisc2_hex_path, 2), kernel_addrs_and_sizes);

    uint32_t num_tiles = NUM_TILES;
    uint32_t num_bytes_per_tile = NUM_BYTES_PER_TILE;
    uint32_t datacopy_src = ACTIVATIONS_DRAM_SRC;
    uint32_t datacopy_dst = datacopy_src + num_tiles * num_bytes_per_tile;
    vector<pair<uint32_t, uint32_t>> runtime_args_addrs_and_sizes;

    vector<uint32_t> writer_runtime_args =
        allocate_helper({datacopy_dst, 1, 0, num_tiles}, runtime_args_addrs_and_sizes);
    vector<uint32_t> reader_runtime_args =
        allocate_helper({datacopy_src, 1, 0, num_tiles}, runtime_args_addrs_and_sizes);

    std::cout << "Writer runtime args" << std::endl;
    for (uint32_t v : writer_runtime_args) {
        std::cout << v << ", ";
    }
    std::cout << std::endl;

    std::cout << "Reader runtime args" << std::endl;
    for (uint32_t v : reader_runtime_args) {
        std::cout << v << ", ";
    }
    std::cout << std::endl;

    uint32_t num_cb_tiles = NUM_CB_TILES;
    vector<pair<uint32_t, uint32_t>> cb_configs_addrs_and_sizes;
    vector<uint32_t> cb_in_config =
        allocate_helper({800 * 1024, num_cb_tiles * NUM_BYTES_PER_TILE, num_cb_tiles}, cb_configs_addrs_and_sizes);
    vector<uint32_t> cb_out_config =
        allocate_helper({900 * 1024, num_cb_tiles * NUM_BYTES_PER_TILE, num_cb_tiles}, cb_configs_addrs_and_sizes);

    vector<vector<uint32_t>> kernels;
    kernels.push_back(brisc_hex);
    kernels.push_back(ncrisc_hex);
    kernels.push_back(trisc0_hex);
    kernels.push_back(trisc1_hex);
    kernels.push_back(trisc2_hex);

    vector<vector<uint32_t>> runtime_args;
    runtime_args.push_back(writer_runtime_args);
    runtime_args.push_back(reader_runtime_args);

    vector<vector<uint32_t>> cb_configs;
    cb_configs.push_back(cb_in_config);
    cb_configs.push_back(cb_out_config);

    return {
        kernels,
        runtime_args,
        cb_configs,
        kernel_addrs_and_sizes,
        runtime_args_addrs_and_sizes,
        cb_configs_addrs_and_sizes};
}

// .____     ____
// |    |   /_   |
// |    |    |   |
// |    |___ |   |
// |_______  \___|
//          \/
tuple<uint32_t, uint64_t, uint32_t> create_kernel_transfer_info(
    uint32_t kernel_id, DramConfig &dram_config, uint32_t &l1_src) {
    uint32_t write_addr;
    switch (kernel_id) {
        case 0: write_addr = l1_mem::address_map::FIRMWARE_BASE; break;
        case 1: write_addr = l1_mem::address_map::NCRISC_FIRMWARE_BASE; break;
        case 2: write_addr = l1_mem::address_map::TRISC0_BASE; break;
        case 3: write_addr = l1_mem::address_map::TRISC1_BASE; break;
        case 4: write_addr = l1_mem::address_map::TRISC2_BASE; break;
        default: TT_ASSERT(false, "Invalid kernel_id");
    }

    std::cout << "Kernel write addr: " << write_addr << std::endl;

    uint32_t kernel_size = std::get<1>(dram_config.kernel_addrs_and_sizes.at(kernel_id));
    tuple<uint32_t, uint64_t, uint32_t> kernel_transfer_info =
        make_tuple(l1_src, NOC_XY_ADDR(NOC_X(1), NOC_Y(1), write_addr), kernel_size);
    l1_src = nearest_multiple_of_32(l1_src + kernel_size);
    return kernel_transfer_info;
}

tuple<uint32_t, uint64_t, uint32_t> create_rt_args_transfer_info(
    uint32_t kernel_id, DramConfig &dram_config, uint32_t &l1_src) {
    uint32_t write_addr;
    switch (kernel_id) {
        case 0: write_addr = BRISC_L1_ARG_BASE; break;
        case 1: write_addr = NCRISC_L1_ARG_BASE; break;
        default: TT_ASSERT(false, "Invalid kernel_id");
    }
        std::cout << "Rt args addr: " << write_addr << std::endl;


    uint32_t rt_args_size = std::get<1>(dram_config.runtime_args_addrs_and_sizes.at(kernel_id));
    tuple<uint32_t, uint64_t, uint32_t> runtime_args_transfer_info =
        make_tuple(l1_src, NOC_XY_ADDR(NOC_X(1), NOC_Y(1), write_addr), rt_args_size);
    l1_src = nearest_multiple_of_32(l1_src + rt_args_size);
    return runtime_args_transfer_info;
}

tuple<uint32_t, uint64_t, uint32_t> create_cb_transfer_info(uint32_t cb_id, uint32_t &l1_src) {
    uint32_t num_bytes_per_cb = UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * sizeof(uint32_t);
    uint32_t write_addr = CIRCULAR_BUFFER_CONFIG_BASE + cb_id * num_bytes_per_cb;
    std::cout << "CB config write addr: " << write_addr << std::endl;

    tuple<uint32_t, uint64_t, uint32_t> cb_transfer_info =
        make_tuple(l1_src, NOC_XY_ADDR(NOC_X(1), NOC_Y(1), write_addr), num_bytes_per_cb);
    l1_src = nearest_multiple_of_32(l1_src + num_bytes_per_cb);
    return cb_transfer_info;
}

CopyDescriptor construct_copy_descriptor_from_dram_config(
    DramConfig &config, uint32_t copy_desc_l1_start, uint32_t read_to_addr) {
    // Since our data in DRAM is contiguous and 32B aligned, we can
    // just issue one read
    uint32_t start_dram_addr = config.kernel_addrs_and_sizes.at(0).first;
    const auto &[last_dram_addr, last_dram_size] =
        config.cb_configs_addrs_and_sizes.at(config.cb_configs_addrs_and_sizes.size() - 1);
    uint32_t transfer_size = nearest_multiple_of_32(last_dram_addr + last_dram_size) - start_dram_addr;
    uint32_t num_reads = 1;
    tuple<uint64_t, uint32_t, uint32_t> read =
        std::make_tuple(NOC_XY_ADDR(NOC_X(1), NOC_Y(0), start_dram_addr), read_to_addr, transfer_size);
    vector<tuple<uint64_t, uint32_t, uint32_t>> reads = {read};

    vector<tuple<uint32_t, uint64_t, uint32_t>> writes;
    tuple<uint32_t, uint64_t, uint32_t> brisc_kernel_write = create_kernel_transfer_info(0, config, read_to_addr);
    tuple<uint32_t, uint64_t, uint32_t> ncrisc_kernel_write = create_kernel_transfer_info(1, config, read_to_addr);
    tuple<uint32_t, uint64_t, uint32_t> trisc0_kernel_write = create_kernel_transfer_info(2, config, read_to_addr);
    tuple<uint32_t, uint64_t, uint32_t> trisc1_kernel_write = create_kernel_transfer_info(3, config, read_to_addr);
    tuple<uint32_t, uint64_t, uint32_t> trisc2_kernel_write = create_kernel_transfer_info(4, config, read_to_addr);

    tuple<uint32_t, uint64_t, uint32_t> brisc_rt_args_write = create_rt_args_transfer_info(0, config, read_to_addr);
    tuple<uint32_t, uint64_t, uint32_t> ncrisc_rt_args_write = create_rt_args_transfer_info(1, config, read_to_addr);

    tuple<uint32_t, uint64_t, uint32_t> cb_in_write = create_cb_transfer_info(0, read_to_addr);
    tuple<uint32_t, uint64_t, uint32_t> cb_out_write = create_cb_transfer_info(16, read_to_addr);

    writes.push_back(brisc_kernel_write);
    writes.push_back(ncrisc_kernel_write);
    writes.push_back(trisc0_kernel_write);
    writes.push_back(trisc1_kernel_write);
    writes.push_back(trisc2_kernel_write);

    writes.push_back(brisc_rt_args_write);
    writes.push_back(ncrisc_rt_args_write);

    writes.push_back(cb_in_write);
    writes.push_back(cb_out_write);

    uint32_t num_writes = writes.size();

    // For this test, we are only running single-core datacopy
    uint32_t num_resets = 1;
    uint32_t num_workers = 1;

    vector<uint64_t> notifies = {NOC_XY_ADDR(NOC_X(1), NOC_Y(1), DISPATCH_MESSAGE_ADDR)};
    vector<uint64_t> resets = {NOC_XY_ADDR(NOC_X(1), NOC_Y(1), DEVICE_DATA.TENSIX_SOFT_RESET_ADDR)};

    return CopyDescriptor{copy_desc_l1_start, num_reads, reads, num_writes, writes, num_resets, num_workers, notifies, resets};
}

vector<uint32_t> convert_copy_desc_to_flat_vec(const CopyDescriptor &copy_desc) {
    vector<uint32_t> flat_desc;
    flat_desc.push_back(copy_desc.num_reads);
    for (const tuple<uint64_t, uint32_t, uint32_t> read : copy_desc.reads) {
        uint64_t src_noc_addr = std::get<0>(read);
        uint32_t src_noc_info_part = src_noc_addr >> 32;
        uint32_t src_addr_part = src_noc_addr & 0xffffffff;
        uint32_t dst_addr = std::get<1>(read);
        uint32_t transfer_size = std::get<2>(read);

        // Push least significant bytes first
        flat_desc.push_back(src_addr_part);
        flat_desc.push_back(src_noc_info_part);
        flat_desc.push_back(dst_addr);
        flat_desc.push_back(transfer_size);
    }

    flat_desc.push_back(copy_desc.num_writes);
    for (const tuple<uint32_t, uint64_t, uint32_t> write : copy_desc.writes) {
        uint32_t src_addr = std::get<0>(write);
        uint64_t dst_noc_addr = std::get<1>(write);
        uint32_t dst_noc_info_part = dst_noc_addr >> 32;
        uint32_t dst_addr_part = dst_noc_addr & 0xffffffff;
        uint32_t transfer_size = std::get<2>(write);

        // Push least significant bytes first
        flat_desc.push_back(src_addr);
        flat_desc.push_back(dst_addr_part);
        flat_desc.push_back(dst_noc_info_part);
        flat_desc.push_back(transfer_size);
    }

    flat_desc.push_back(copy_desc.num_resets);
    flat_desc.push_back(copy_desc.num_workers);
    for (uint64_t notify_noc_addr : copy_desc.notifies) {
        uint32_t notify_noc_info_part = notify_noc_addr >> 32;
        uint32_t notify_addr_part = notify_noc_addr & 0xffffffff;

        // Push least significant bytes first
        flat_desc.push_back(notify_addr_part);
        flat_desc.push_back(notify_noc_info_part);
    }

    for (uint64_t reset_noc_addr : copy_desc.resets) {
        uint32_t reset_noc_info_part = reset_noc_addr >> 32;
        uint32_t reset_addr_part = reset_noc_addr & 0xffffffff;

        // Push least significant bytes first
        flat_desc.push_back(reset_addr_part);
        flat_desc.push_back(reset_noc_info_part);
    }

    std::cout << "FLAT DESC SIZE: " << flat_desc.size() * sizeof(flat_desc.at(0)) << std::endl;

    return flat_desc;
}

void write_copy_desc_to_l1(
    tt_cluster *cluster, int chip_id, tt_xy_pair dispatch_core, const CopyDescriptor &copy_desc) {
    uint32_t l1_addr = copy_desc.l1_addr;
    vector<uint32_t> flat_desc = convert_copy_desc_to_flat_vec(copy_desc);

    tt::llrt::write_hex_vec_to_core(cluster, chip_id, dispatch_core, flat_desc, l1_addr);

    tt::llrt::write_hex_vec_to_core(cluster, chip_id, dispatch_core, {l1_addr}, BRISC_L1_ARG_BASE);
}

void host_dispatch(tt_cluster *cluster, int chip_id, string op, tt_xy_pair dispatch_core, tt_xy_pair worker_core) {
    // Write dispatch binary to core
    tt::llrt::test_load_write_read_risc_binary(
        cluster, "built_kernels/dispatch/brisc/brisc.hex", chip_id, dispatch_core, 0);

    DramConfig dram_config = construct_dram_config(op);
    std::cout << dram_config << std::endl;
    write_to_dram(cluster, chip_id, dram_config);
    CopyDescriptor copy_desc = construct_copy_descriptor_from_dram_config(dram_config, 150 * 1024, 300 * 1024);
    std::cout << copy_desc << std::endl;
    write_copy_desc_to_l1(cluster, chip_id, dispatch_core, copy_desc);

    // Deassert dispatch core
    // tt_start_debug_print_server(cluster, {chip_id}, {dispatch_core});

    tt::llrt::internal_::setup_riscs_on_specified_cores(
        cluster, chip_id, tt::llrt::TensixRiscsOptions::BRISC_ONLY, {dispatch_core});
    tt::llrt::internal_::setup_riscs_on_specified_cores(
        cluster, chip_id, tt::llrt::TensixRiscsOptions::ALL_RISCS, {worker_core});

    uint32_t dispatch_done_addr = 0;
    vector<uint32_t> hugepage_done_addrs = {dispatch_done_addr};
    tt::llrt::internal_::run_riscs_on_specified_cores(
        cluster, chip_id, tt::llrt::TensixRiscsOptions::BRISC_ONLY, {dispatch_core}, hugepage_done_addrs);
}

bool test_dispatch_v1(tt_cluster *cluster, int chip_id, string op) {
    uint32_t dram_buffer_size = NUM_TILES * NUM_BYTES_PER_TILE;  // 1 tile
    std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(
        dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());

    cluster->write_dram_vec(src_vec, tt_target_dram{chip_id, 0, 0}, ACTIVATIONS_DRAM_SRC);
    tt_xy_pair dispatch_core = {11, 1};
    tt_xy_pair worker_core = {1, 1};
    host_dispatch(cluster, chip_id, op, dispatch_core, worker_core);

    vector<uint32_t> dst_vec;
    cluster->read_dram_vec(
        dst_vec,
        tt_target_dram{chip_id, 0, 0},
        ACTIVATIONS_DRAM_SRC + NUM_TILES * NUM_BYTES_PER_TILE,
        dram_buffer_size);

    bool pass = (src_vec == dst_vec);


    if (not pass) {
        print_vec_of_uint32_as_packed_bfloat16(src_vec, NUM_TILES, "Golden");
        print_vec_of_uint32_as_packed_bfloat16(dst_vec, NUM_TILES, "Result");
    }
    TT_ASSERT(pass);
    return pass;
}

int main(int argc, char **argv) {
    bool pass = true;

    const TargetDevice target_type = TargetDevice::Silicon;
    const tt::ARCH arch = tt::ARCH::GRAYSKULL;
    const std::string sdesc_file = get_soc_description_file(arch, target_type);

    try {
        tt_device_params default_params;
        tt_cluster *cluster = new tt_cluster;
        cluster->open_device(arch, target_type, {0}, sdesc_file);
        cluster->start_device(default_params);  // use default params
        tt::llrt::utils::log_current_ai_clk(cluster);
        test_dispatch_v1(cluster, 0, "datacopy_op_dispatch");

    } catch (const std::exception &e) {
        pass = false;
        // Capture the exception error message
        log_error(tt::LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(tt::LogTest, "System error message: {}", std::strerror(errno));
    }

    return 0;
}
