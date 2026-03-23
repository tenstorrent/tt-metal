// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "multi_device_fixture.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "dm_common.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <distributed/mesh_device_impl.hpp>
#include <algorithm>

namespace tt::tt_metal {

using namespace std;
using namespace tt;
using namespace test_utils;

namespace unit_tests::dm::noc_estimator {

// ============ ENUMS ============

enum class NocPattern : uint32_t {
    ONE_TO_ONE = 0,
    ONE_FROM_ONE = 1,
    ONE_TO_ALL = 2,
    ONE_FROM_ALL = 3,
    ALL_TO_ALL = 4,
    ALL_FROM_ALL = 5,
    ONE_TO_ROW = 6,
    ROW_TO_ROW = 7,
    ONE_TO_COLUMN = 8,
    COLUMN_TO_COLUMN = 9,
};

enum class NocMechanism : uint32_t {
    UNICAST = 0,
    MULTICAST = 1,
    MULTICAST_LINKED = 2,
};

enum class MemoryType : uint32_t {
    L1 = 0,
    DRAM_INTERLEAVED = 1,
    DRAM_SHARDED = 2,
};

// Writer kernel modes (must match log_helpers.hpp constants)
constexpr uint32_t WRITER_MODE_UNICAST_SINGLE = 0;
constexpr uint32_t WRITER_MODE_UNICAST_MULTI = 1;
constexpr uint32_t WRITER_MODE_MULTICAST = 2;
constexpr uint32_t WRITER_MODE_MULTICAST_LINKED = 3;

// Reader kernel modes
constexpr uint32_t READER_MODE_SINGLE = 0;
constexpr uint32_t READER_MODE_MULTI = 1;

// ============ CONFIG ============

struct NocEstimatorConfig {
    uint32_t test_id = 0;

    NocPattern pattern = NocPattern::ONE_TO_ONE;
    NocMechanism mechanism = NocMechanism::UNICAST;
    MemoryType memory_type = MemoryType::L1;

    // Transaction parameters (the sweep axes)
    uint32_t num_of_transactions = 1;
    uint32_t pages_per_transaction = 1;
    uint32_t bytes_per_page = 32;

    // Grid parameters
    CoreCoord master_start_coord = {0, 0};
    CoreCoord master_grid_size = {1, 1};
    CoreCoord sub_start_coord = {0, 0};
    CoreCoord sub_grid_size = {1, 1};

    // Behavioral flags
    bool same_axis = true;
    bool stateful = false;
    bool loopback = false;
    NOC noc_id = NOC::NOC_0;
    uint32_t num_virtual_channels = 1;

    // DRAM-specific
    uint32_t dram_channel = 0;

    // Future extensibility:
    // bool posted = false;
};

// ============ HELPERS ============

static bool is_write_pattern(NocPattern p) {
    return p == NocPattern::ONE_TO_ONE || p == NocPattern::ONE_TO_ALL || p == NocPattern::ALL_TO_ALL ||
           p == NocPattern::ONE_TO_ROW || p == NocPattern::ROW_TO_ROW || p == NocPattern::ONE_TO_COLUMN ||
           p == NocPattern::COLUMN_TO_COLUMN;
}

static const std::string KERNELS_DIR = "tests/tt_metal/tt_metal/data_movement/noc_estimator_tests/kernels/";

static vector<uint32_t> make_writer_compile_args(
    const NocEstimatorConfig& cfg,
    uint32_t src_addr,
    uint32_t dst_addr,
    uint32_t bytes_per_txn,
    uint32_t writer_mode,
    uint32_t num_subs,
    uint32_t packed_dest = 0,
    uint32_t mcast_sx = 0,
    uint32_t mcast_sy = 0,
    uint32_t mcast_ex = 0,
    uint32_t mcast_ey = 0) {
    return {
        src_addr,                   // 0
        dst_addr,                   // 1
        cfg.num_of_transactions,    // 2
        bytes_per_txn,              // 3
        cfg.test_id,                // 4
        writer_mode,                // 5
        num_subs,                   // 6
        (uint32_t)cfg.stateful,     // 7
        cfg.num_virtual_channels,   // 8
        (uint32_t)cfg.loopback,     // 9
        mcast_sx,                   // 10
        mcast_sy,                   // 11
        mcast_ex,                   // 12
        mcast_ey,                   // 13
        packed_dest,                // 14
        (uint32_t)cfg.memory_type,  // 15
        (uint32_t)cfg.mechanism,    // 16
        (uint32_t)cfg.pattern,      // 17
        (uint32_t)cfg.same_axis,    // 18
        (uint32_t)cfg.loopback,     // 19
    };
}

static vector<uint32_t> make_reader_compile_args(
    const NocEstimatorConfig& cfg,
    uint32_t local_addr,
    uint32_t bytes_per_txn,
    uint32_t reader_mode,
    uint32_t num_subs) {
    return {
        local_addr,                 // 0
        cfg.num_of_transactions,    // 1
        bytes_per_txn,              // 2
        cfg.test_id,                // 3
        reader_mode,                // 4
        num_subs,                   // 5
        (uint32_t)cfg.stateful,     // 6
        cfg.num_virtual_channels,   // 7
        (uint32_t)cfg.memory_type,  // 8
        (uint32_t)cfg.mechanism,    // 9
        (uint32_t)cfg.pattern,      // 10
        (uint32_t)cfg.same_axis,    // 11
        (uint32_t)cfg.loopback,     // 12
    };
}

static void execute_program(const shared_ptr<distributed::MeshDevice>& mesh_device, Program program) {
    auto mesh_workload = distributed::MeshWorkload();
    vector<uint32_t> coord_data = {0, 0};
    auto target_devices = distributed::MeshCoordinateRange(distributed::MeshCoordinate(coord_data));
    mesh_workload.add_program(target_devices, std::move(program));

    auto& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, mesh_workload, false);
    Finish(cq);
}

static vector<uint32_t> make_test_data(size_t bytes) {
    uint32_t num_elements = bytes / sizeof(bfloat16);
    return generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -100.0f, 100.0f, num_elements, chrono::system_clock::now().time_since_epoch().count());
}

// ============ PATTERN-SPECIFIC RUN FUNCTIONS ============

// Handles ONE_TO_ONE (write) and ONE_FROM_ONE (read)
static bool run_single_core(const shared_ptr<distributed::MeshDevice>& mesh_device, const NocEstimatorConfig& cfg) {
    IDevice* device = mesh_device->impl().get_device(0);
    Program program = CreateProgram();

    const size_t bytes_per_txn = cfg.pages_per_transaction * cfg.bytes_per_page;

    // Core coordinates based on same_axis
    CoreCoord master_coord = {0, 0};
    CoreCoord sub_coord = cfg.same_axis ? CoreCoord{0, 1} : CoreCoord{1, 1};

    // L1 address validation
    L1AddressInfo master_l1 = unit_tests::dm::get_l1_address_and_size(mesh_device, master_coord);
    L1AddressInfo sub_l1 = unit_tests::dm::get_l1_address_and_size(mesh_device, sub_coord);
    if (master_l1.base_address != sub_l1.base_address || master_l1.size != sub_l1.size) {
        log_error(LogTest, "Mismatch in L1 address or size between master and subordinate cores");
        return false;
    }
    if (master_l1.size < bytes_per_txn) {
        log_error(LogTest, "Insufficient L1 size for the test configuration");
        return false;
    }
    uint32_t l1_base = master_l1.base_address;

    CoreCoord phys_sub = device->worker_core_from_logical_core(sub_coord);
    uint32_t packed_sub = (phys_sub.x << 16) | (phys_sub.y & 0xFFFF);

    bool is_write = is_write_pattern(cfg.pattern);

    if (is_write) {
        auto compile_args = make_writer_compile_args(
            cfg, l1_base, l1_base, (uint32_t)bytes_per_txn, WRITER_MODE_UNICAST_SINGLE, 0, packed_sub);
        CreateKernel(
            program,
            KERNELS_DIR + "writer.cpp",
            master_coord,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0, .noc = cfg.noc_id, .compile_args = compile_args});
    } else {
        auto compile_args = make_reader_compile_args(cfg, l1_base, (uint32_t)bytes_per_txn, READER_MODE_SINGLE, 0);
        CoreRangeSet master_set({CoreRange(master_coord)});
        auto kernel = CreateKernel(
            program,
            KERNELS_DIR + "reader.cpp",
            master_set,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
                .compile_args = compile_args});
        SetRuntimeArgs(program, kernel, master_set, {phys_sub.x, phys_sub.y});
    }

    log_info(LogTest, "Running Test ID: {}, Run ID: {}", cfg.test_id, unit_tests::dm::runtime_host_id);
    program.set_runtime_id(unit_tests::dm::runtime_host_id++);

    auto packed_input = make_test_data(bytes_per_txn);
    auto packed_golden = packed_input;

    if (is_write) {
        detail::WriteToDeviceL1(device, master_coord, l1_base, packed_input);
    } else {
        detail::WriteToDeviceL1(device, sub_coord, l1_base, packed_input);
    }
    MetalContext::instance().get_cluster().l1_barrier(device->id());

    execute_program(mesh_device, std::move(program));

    vector<uint32_t> packed_output;
    if (is_write) {
        detail::ReadFromDeviceL1(device, sub_coord, l1_base, bytes_per_txn, packed_output);
    } else {
        detail::ReadFromDeviceL1(device, master_coord, l1_base, bytes_per_txn, packed_output);
    }

    bool is_equal = (packed_output == packed_golden);
    if (!is_equal) {
        log_error(LogTest, "Equality Check failed for test_id {}", cfg.test_id);
    }
    return is_equal;
}

// Handles ONE_TO_ALL (write, unicast/multicast) and ONE_FROM_ALL (read)
static bool run_one_to_many(const shared_ptr<distributed::MeshDevice>& mesh_device, const NocEstimatorConfig& cfg) {
    IDevice* device = mesh_device->impl().get_device(0);
    Program program = CreateProgram();

    const size_t bytes_per_txn = cfg.pages_per_transaction * cfg.bytes_per_page;
    bool is_write = is_write_pattern(cfg.pattern);

    // Master core
    CoreCoord mst_coord = cfg.master_start_coord;
    CoreRangeSet mst_set({CoreRange(mst_coord)});

    // Subordinate grid
    CoreCoord sub_start = cfg.sub_start_coord;
    CoreCoord sub_end = CoreCoord(sub_start.x + cfg.sub_grid_size.x - 1, sub_start.y + cfg.sub_grid_size.y - 1);
    CoreRangeSet sub_set({CoreRange(sub_start, sub_end)});

    bool is_multicast = (cfg.mechanism == NocMechanism::MULTICAST || cfg.mechanism == NocMechanism::MULTICAST_LINKED);
    if (!is_multicast) {
        // For unicast: master sends to each subordinate individually, exclude itself
        sub_set = sub_set.subtract(mst_set);
    }
    // For multicast: master placement determines loopback behavior.
    //   loopback=true  → master inside rectangle, INCLUDE_SRC
    //   loopback=false → master outside rectangle, EXCLUDE_SRC
    uint32_t num_subs = sub_set.num_cores();
    auto sub_core_list = corerange_to_cores(sub_set);

    // L1 addresses
    L1AddressInfo mst_l1 = unit_tests::dm::get_l1_address_and_size(mesh_device, mst_coord);
    // Need space for both source and destination regions in non-loopback case.
    uint32_t required_l1_bytes = (uint32_t)bytes_per_txn * (cfg.loopback ? 1u : 2u);
    if (mst_l1.size < required_l1_bytes) {
        log_error(LogTest, "Insufficient L1 size on master core");
        return false;
    }
    uint32_t mst_l1_addr = mst_l1.base_address;
    uint32_t sub_l1_addr = cfg.loopback ? mst_l1_addr : mst_l1_addr + (uint32_t)bytes_per_txn;
    // Validate that a representative subordinate core also has enough L1.
    if (!cfg.loopback && !sub_core_list.empty()) {
        L1AddressInfo sub_l1 = unit_tests::dm::get_l1_address_and_size(mesh_device, sub_core_list.front());
        if (sub_l1.size < required_l1_bytes) {
            log_error(LogTest, "Insufficient L1 size on subordinate core");
            return false;
        }
    }

    if (is_write) {
        uint32_t writer_mode;

        if (is_multicast) {
            writer_mode = (cfg.mechanism == NocMechanism::MULTICAST_LINKED) ? WRITER_MODE_MULTICAST_LINKED
                                                                            : WRITER_MODE_MULTICAST;
            CoreCoord sub_phys_start = device->worker_core_from_logical_core(sub_start);
            CoreCoord sub_phys_end = device->worker_core_from_logical_core(sub_end);

            auto compile_args = make_writer_compile_args(
                cfg,
                mst_l1_addr,
                sub_l1_addr,
                (uint32_t)bytes_per_txn,
                writer_mode,
                num_subs,
                0,
                sub_phys_start.x,
                sub_phys_start.y,
                sub_phys_end.x,
                sub_phys_end.y);

            CreateKernel(
                program,
                KERNELS_DIR + "writer.cpp",
                mst_set,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0, .noc = cfg.noc_id, .compile_args = compile_args});
        } else {
            // Unicast multi: pack subordinate physical coordinates as runtime args
            writer_mode = WRITER_MODE_UNICAST_MULTI;
            auto compile_args =
                make_writer_compile_args(cfg, mst_l1_addr, sub_l1_addr, (uint32_t)bytes_per_txn, writer_mode, num_subs);

            auto kernel = CreateKernel(
                program,
                KERNELS_DIR + "writer.cpp",
                mst_set,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0, .noc = cfg.noc_id, .compile_args = compile_args});

            vector<uint32_t> rt_args;
            for (auto& core : sub_core_list) {
                CoreCoord phys = device->worker_core_from_logical_core(core);
                rt_args.push_back((phys.x << 16) | (phys.y & 0xFFFF));
            }
            SetRuntimeArgs(program, kernel, mst_set, rt_args);
        }
    } else {
        // ONE_FROM_ALL: read from all subordinates
        auto compile_args =
            make_reader_compile_args(cfg, mst_l1.base_address, (uint32_t)bytes_per_txn, READER_MODE_MULTI, num_subs);
        auto kernel = CreateKernel(
            program,
            KERNELS_DIR + "reader.cpp",
            mst_set,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
                .compile_args = compile_args});

        vector<uint32_t> rt_args;
        for (auto& core : sub_core_list) {
            CoreCoord phys = device->worker_core_from_logical_core(core);
            rt_args.push_back(phys.x);
            rt_args.push_back(phys.y);
        }
        SetRuntimeArgs(program, kernel, mst_set, rt_args);
    }

    log_info(LogTest, "Running Test ID: {}, Run ID: {}", cfg.test_id, unit_tests::dm::runtime_host_id);
    program.set_runtime_id(unit_tests::dm::runtime_host_id++);

    auto packed_input = make_test_data(bytes_per_txn);
    auto packed_golden = packed_input;

    if (is_write) {
        detail::WriteToDeviceL1(device, mst_coord, mst_l1_addr, packed_input);
    } else {
        for (auto& core : sub_core_list) {
            detail::WriteToDeviceL1(device, core, mst_l1.base_address, packed_input);
        }
    }
    MetalContext::instance().get_cluster().l1_barrier(device->id());

    execute_program(mesh_device, std::move(program));

    if (is_write) {
        for (auto& core : sub_core_list) {
            vector<uint32_t> packed_output;
            detail::ReadFromDeviceL1(device, core, sub_l1_addr, bytes_per_txn, packed_output);
            if (packed_output != packed_golden) {
                log_error(LogTest, "Equality Check failed for test_id {}", cfg.test_id);
                return false;
            }
        }
    } else {
        vector<uint32_t> packed_output;
        detail::ReadFromDeviceL1(device, mst_coord, mst_l1.base_address, bytes_per_txn, packed_output);
        if (packed_output != packed_golden) {
            log_error(LogTest, "Equality Check failed for test_id {}", cfg.test_id);
            return false;
        }
    }
    return true;
}

// Handles ALL_TO_ALL (write) and ALL_FROM_ALL (read)
static bool run_all_pattern(const shared_ptr<distributed::MeshDevice>& mesh_device, const NocEstimatorConfig& cfg) {
    IDevice* device = mesh_device->impl().get_device(0);
    Program program = CreateProgram();

    const size_t bytes_per_txn = cfg.pages_per_transaction * cfg.bytes_per_page;
    bool is_write = is_write_pattern(cfg.pattern);

    // Master grid
    CoreCoord mst_start = cfg.master_start_coord;
    CoreCoord mst_end = CoreCoord(mst_start.x + cfg.master_grid_size.x - 1, mst_start.y + cfg.master_grid_size.y - 1);
    CoreRangeSet mst_set({CoreRange(mst_start, mst_end)});
    auto mst_core_list = corerange_to_cores(mst_set);

    // Subordinate grid
    CoreCoord sub_start = cfg.sub_start_coord;
    CoreCoord sub_end = CoreCoord(sub_start.x + cfg.sub_grid_size.x - 1, sub_start.y + cfg.sub_grid_size.y - 1);
    CoreRangeSet sub_set({CoreRange(sub_start, sub_end)});
    uint32_t num_subs = sub_set.num_cores();
    auto sub_core_list = corerange_to_cores(sub_set);

    // L1 addresses (offset to avoid overlap between master source and subordinate destination)
    L1AddressInfo l1_info = unit_tests::dm::get_l1_address_and_size(mesh_device, mst_start);
    if (l1_info.size < 2 * bytes_per_txn) {
        log_error(LogTest, "Insufficient L1 size");
        return false;
    }
    uint32_t mst_l1_addr = l1_info.base_address;
    uint32_t sub_l1_addr = mst_l1_addr + (uint32_t)bytes_per_txn;

    // Subordinate physical coordinates
    vector<uint32_t> sub_worker_coords;
    for (auto& core : sub_core_list) {
        CoreCoord phys = device->worker_core_from_logical_core(core);
        if (is_write) {
            sub_worker_coords.push_back((phys.x << 16) | (phys.y & 0xFFFF));
        } else {
            sub_worker_coords.push_back(phys.x);
            sub_worker_coords.push_back(phys.y);
        }
    }

    if (is_write) {
        auto compile_args = make_writer_compile_args(
            cfg, mst_l1_addr, sub_l1_addr, (uint32_t)bytes_per_txn, WRITER_MODE_UNICAST_MULTI, num_subs);
        auto kernel = CreateKernel(
            program,
            KERNELS_DIR + "writer.cpp",
            mst_set,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0, .noc = cfg.noc_id, .compile_args = compile_args});
        SetRuntimeArgs(program, kernel, mst_set, sub_worker_coords);
    } else {
        auto compile_args =
            make_reader_compile_args(cfg, mst_l1_addr, (uint32_t)bytes_per_txn, READER_MODE_MULTI, num_subs);
        auto kernel = CreateKernel(
            program,
            KERNELS_DIR + "reader.cpp",
            mst_set,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
                .compile_args = compile_args});
        SetRuntimeArgs(program, kernel, mst_set, sub_worker_coords);
    }

    log_info(LogTest, "Running Test ID: {}, Run ID: {}", cfg.test_id, unit_tests::dm::runtime_host_id);
    program.set_runtime_id(unit_tests::dm::runtime_host_id++);

    auto packed_input = make_test_data(bytes_per_txn);
    auto packed_golden = packed_input;

    if (is_write) {
        // Write source data to master cores at mst_l1_addr
        for (auto& core : mst_core_list) {
            detail::WriteToDeviceL1(device, core, mst_l1_addr, packed_input);
        }
    } else {
        // Write source data to subordinate cores at mst_l1_addr
        // (reader kernel reads from local_addr = mst_l1_addr on remote cores)
        for (auto& core : sub_core_list) {
            detail::WriteToDeviceL1(device, core, mst_l1_addr, packed_input);
        }
    }
    MetalContext::instance().get_cluster().l1_barrier(device->id());

    execute_program(mesh_device, std::move(program));

    if (is_write) {
        for (auto& core : sub_core_list) {
            vector<uint32_t> packed_output;
            detail::ReadFromDeviceL1(device, core, sub_l1_addr, bytes_per_txn, packed_output);
            if (packed_output != packed_golden) {
                log_error(LogTest, "Equality Check failed for test_id {}", cfg.test_id);
                return false;
            }
        }
    } else {
        // Reader writes to local_addr = mst_l1_addr on each master core
        for (auto& core : mst_core_list) {
            vector<uint32_t> packed_output;
            detail::ReadFromDeviceL1(device, core, mst_l1_addr, bytes_per_txn, packed_output);
            if (packed_output != packed_golden) {
                log_error(LogTest, "Equality Check failed for test_id {}", cfg.test_id);
                return false;
            }
        }
    }
    return true;
}

// Handles ROW_TO_ROW / COLUMN_TO_COLUMN with multicast:
// All masters in the row/column multicast to the same rectangle (the row/column).
// Masters and subordinates overlap; loopback should be true (INCLUDE_SRC).
static bool run_many_to_many_multicast(
    const shared_ptr<distributed::MeshDevice>& mesh_device, const NocEstimatorConfig& cfg) {
    IDevice* device = mesh_device->impl().get_device(0);
    Program program = CreateProgram();

    const size_t bytes_per_txn = cfg.pages_per_transaction * cfg.bytes_per_page;

    // Master grid
    CoreCoord mst_start = cfg.master_start_coord;
    CoreCoord mst_end = CoreCoord(mst_start.x + cfg.master_grid_size.x - 1, mst_start.y + cfg.master_grid_size.y - 1);
    CoreRangeSet mst_set({CoreRange(mst_start, mst_end)});
    auto mst_core_list = corerange_to_cores(mst_set);

    // Subordinate grid (same as master for row_to_row / column_to_column)
    CoreCoord sub_start = cfg.sub_start_coord;
    CoreCoord sub_end = CoreCoord(sub_start.x + cfg.sub_grid_size.x - 1, sub_start.y + cfg.sub_grid_size.y - 1);
    CoreRangeSet sub_set({CoreRange(sub_start, sub_end)});
    uint32_t num_subs = sub_set.num_cores();
    auto sub_core_list = corerange_to_cores(sub_set);

    // L1 addresses
    L1AddressInfo l1_info = unit_tests::dm::get_l1_address_and_size(mesh_device, mst_start);
    if (l1_info.size < 2 * bytes_per_txn) {
        log_error(LogTest, "Insufficient L1 size");
        return false;
    }
    uint32_t mst_l1_addr = l1_info.base_address;
    uint32_t sub_l1_addr = cfg.loopback ? mst_l1_addr : mst_l1_addr + (uint32_t)bytes_per_txn;

    // All masters multicast to the same sub rectangle
    uint32_t writer_mode =
        (cfg.mechanism == NocMechanism::MULTICAST_LINKED) ? WRITER_MODE_MULTICAST_LINKED : WRITER_MODE_MULTICAST;

    CoreCoord sub_phys_start = device->worker_core_from_logical_core(sub_start);
    CoreCoord sub_phys_end = device->worker_core_from_logical_core(sub_end);

    auto compile_args = make_writer_compile_args(
        cfg,
        mst_l1_addr,
        sub_l1_addr,
        (uint32_t)bytes_per_txn,
        writer_mode,
        num_subs,
        0,
        sub_phys_start.x,
        sub_phys_start.y,
        sub_phys_end.x,
        sub_phys_end.y);

    CreateKernel(
        program,
        KERNELS_DIR + "writer.cpp",
        mst_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = cfg.noc_id, .compile_args = compile_args});

    log_info(LogTest, "Running Test ID: {}, Run ID: {}", cfg.test_id, unit_tests::dm::runtime_host_id);
    program.set_runtime_id(unit_tests::dm::runtime_host_id++);

    auto packed_input = make_test_data(bytes_per_txn);
    auto packed_golden = packed_input;

    // Write source data to all master cores
    for (auto& core : mst_core_list) {
        detail::WriteToDeviceL1(device, core, mst_l1_addr, packed_input);
    }
    MetalContext::instance().get_cluster().l1_barrier(device->id());

    execute_program(mesh_device, std::move(program));

    // Verify all subordinate cores
    for (auto& core : sub_core_list) {
        vector<uint32_t> packed_output;
        detail::ReadFromDeviceL1(device, core, sub_l1_addr, bytes_per_txn, packed_output);
        if (packed_output != packed_golden) {
            log_error(LogTest, "Equality Check failed for test_id {}", cfg.test_id);
            return false;
        }
    }
    return true;
}

enum class DramDirection { READ, WRITE };

// Unified DRAM run function using TensorAccessor + experimental Noc 2.0 API.
// Launches reader OR writer kernel in isolation (no dual-kernel L1 contention).
// For interleaved mode: each core cycles through all banks via sequential page IDs.
// For sharded mode: each core reads/writes only the page(s) mapping to its dedicated bank.
static bool run_dram_accessor(
    const shared_ptr<distributed::MeshDevice>& mesh_device,
    const NocEstimatorConfig& cfg,
    const vector<CoreCoord>& cores,
    const vector<uint32_t>& bank_assignments,
    DramDirection direction) {
    IDevice* device = mesh_device->impl().get_device(0);
    Program program = CreateProgram();

    const size_t bytes_per_txn = cfg.pages_per_transaction * cfg.bytes_per_page;
    uint32_t num_dram_banks = (uint32_t)device->num_dram_channels();
    uint32_t num_cores = (uint32_t)cores.size();

    uint32_t num_pages = num_dram_banks;

    InterleavedBufferConfig buf_config{
        .device = device,
        .size = (size_t)num_pages * bytes_per_txn,
        .page_size = (uint32_t)bytes_per_txn,
        .buffer_type = BufferType::DRAM};
    auto dram_buffer = CreateBuffer(buf_config);

    std::set<CoreRange> core_ranges;
    for (const auto& c : cores) {
        core_ranges.insert(CoreRange(c));
    }
    CoreRangeSet core_set(core_ranges);

    L1AddressInfo l1_info = unit_tests::dm::get_l1_address_and_size(mesh_device, cores[0]);
    uint32_t l1_addr = l1_info.base_address;

    uint32_t barrier_sem_id = CreateSemaphore(program, core_set, 0);

    CoreCoord coordinator_phys = device->worker_core_from_logical_core(cores[0]);
    uint32_t local_scratch_addr = l1_addr + (uint32_t)bytes_per_txn;

    uint32_t mem_type = (uint32_t)cfg.memory_type;
    uint32_t mech = (uint32_t)NocMechanism::UNICAST;

    bool is_read = (direction == DramDirection::READ);
    string kernel_file = is_read ? "dram_reader.cpp" : "dram_writer.cpp";
    auto processor = is_read ? DataMovementProcessor::RISCV_1 : DataMovementProcessor::RISCV_0;

    vector<uint32_t> compile_args = {
        cfg.num_of_transactions,  // 0
        (uint32_t)bytes_per_txn,  // 1
        cfg.test_id,              // 2
        0,                        // 3 (sem_id unused)
        mem_type,                 // 4
        mech,                     // 5
        (uint32_t)cfg.pattern,    // 6
        num_pages,                // 7
    };
    TensorAccessorArgs(*dram_buffer).append_to(compile_args);

    auto kernel = CreateKernel(
        program,
        KERNELS_DIR + kernel_file,
        core_set,
        DataMovementConfig{.processor = processor, .noc = cfg.noc_id, .compile_args = compile_args});

    for (size_t i = 0; i < cores.size(); i++) {
        uint32_t start_page = bank_assignments[i];
        vector<uint32_t> rt_args = {
            dram_buffer->address(),
            l1_addr,
            start_page,
            barrier_sem_id,
            coordinator_phys.x,
            coordinator_phys.y,
            num_cores,
            local_scratch_addr};
        SetRuntimeArgs(program, kernel, cores[i], rt_args);
    }

    log_info(LogTest, "Running Test ID: {}, Run ID: {}", cfg.test_id, unit_tests::dm::runtime_host_id);
    program.set_runtime_id(unit_tests::dm::runtime_host_id++);

    // For reads: fill DRAM buffer with test data so the kernel has something to read.
    // For writes: fill L1 source region via DRAM buffer init (data will be overwritten by kernel).
    auto page_data = make_test_data(bytes_per_txn);
    vector<uint32_t> full_input;
    full_input.reserve(num_pages * page_data.size());
    for (uint32_t p = 0; p < num_pages; p++) {
        full_input.insert(full_input.end(), page_data.begin(), page_data.end());
    }
    detail::WriteToBuffer(dram_buffer, full_input);
    MetalContext::instance().get_cluster().dram_barrier(device->id());

    execute_program(mesh_device, std::move(program));

    return true;
}

// ============ DISPATCH ============

static bool run_dm(const shared_ptr<distributed::MeshDevice>& mesh_device, const NocEstimatorConfig& cfg) {
    switch (cfg.pattern) {
        case NocPattern::ONE_TO_ONE:
        case NocPattern::ONE_FROM_ONE: return run_single_core(mesh_device, cfg);
        case NocPattern::ONE_TO_ALL:
        case NocPattern::ONE_FROM_ALL:
        case NocPattern::ONE_TO_ROW:
        case NocPattern::ONE_TO_COLUMN: return run_one_to_many(mesh_device, cfg);
        case NocPattern::ALL_TO_ALL:
        case NocPattern::ALL_FROM_ALL: return run_all_pattern(mesh_device, cfg);
        case NocPattern::ROW_TO_ROW:
        case NocPattern::COLUMN_TO_COLUMN: {
            bool is_multicast =
                (cfg.mechanism == NocMechanism::MULTICAST || cfg.mechanism == NocMechanism::MULTICAST_LINKED);
            if (is_multicast) {
                return run_many_to_many_multicast(mesh_device, cfg);
            }
            return run_all_pattern(mesh_device, cfg);
        }
        default: log_error(LogTest, "Unknown pattern"); return false;
    }
}

// ============ SWEEP HELPERS ============

// Returns NOC packet size: 8KB for WH, 16KB for BH
// Stateful unicast writes are only valid for transfers smaller than one packet
static uint32_t get_noc_packet_size(IDevice* device) {
    return device->arch() == ARCH::BLACKHOLE ? 16 * 1024 : 8 * 1024;
}

static void packet_sizes_sweep(const shared_ptr<distributed::MeshDevice>& mesh_device, NocEstimatorConfig base_cfg) {
    auto [bytes_per_page, max_bytes, max_pages] = unit_tests::dm::compute_physical_constraints(mesh_device);
    IDevice* device = mesh_device->impl().get_device(0);

    uint32_t max_transactions = 256;
    uint32_t max_pages_per_txn = device->arch() == ARCH::BLACKHOLE ? 1024 : 2048;

    base_cfg.bytes_per_page = bytes_per_page;

    for (uint32_t num_txn = 1; num_txn <= max_transactions; num_txn *= 4) {
        for (uint32_t pages = 1; pages <= max_pages_per_txn; pages *= 2) {
            if (pages > max_pages) {
                continue;
            }
            base_cfg.num_of_transactions = num_txn;
            base_cfg.pages_per_transaction = pages;
            EXPECT_TRUE(run_dm(mesh_device, base_cfg));
        }
    }

    // Flush profiler DRAM buffer to prevent overflow across many sweep iterations
    ReadMeshDeviceProfilerResults(*mesh_device);
}

// Stateful sweep for unicast writes - constrained to packets smaller than max stateful size
static void packet_sizes_sweep_stateful_write(
    const shared_ptr<distributed::MeshDevice>& mesh_device, NocEstimatorConfig base_cfg) {
    auto [bytes_per_page, max_bytes, max_pages] = unit_tests::dm::compute_physical_constraints(mesh_device);
    IDevice* device = mesh_device->impl().get_device(0);

    uint32_t max_transactions = 256;
    uint32_t noc_packet_size = get_noc_packet_size(device);
    uint32_t max_stateful_pages = noc_packet_size / bytes_per_page;

    base_cfg.bytes_per_page = bytes_per_page;
    base_cfg.stateful = true;

    for (uint32_t num_txn = 1; num_txn <= max_transactions; num_txn *= 4) {
        for (uint32_t pages = 1; pages < max_stateful_pages; pages *= 2) {
            if (pages > max_pages) {
                continue;
            }
            base_cfg.num_of_transactions = num_txn;
            base_cfg.pages_per_transaction = pages;
            EXPECT_TRUE(run_dm(mesh_device, base_cfg));
        }
    }

    // Flush profiler DRAM buffer to prevent overflow across many sweep iterations
    ReadMeshDeviceProfilerResults(*mesh_device);
}

// ============ SWEEP FUNCTIONS ============

static void sweep_one_to_one(const shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t test_id) {
    for (bool same_axis : {true, false}) {
        for (bool stateful : {true, false}) {
            NocEstimatorConfig cfg = {
                .test_id = test_id,
                .pattern = NocPattern::ONE_TO_ONE,
                .mechanism = NocMechanism::UNICAST,
                .memory_type = MemoryType::L1,
                .same_axis = same_axis,
                .stateful = stateful,
            };
            packet_sizes_sweep(mesh_device, cfg);
        }
    }
}

static void sweep_one_from_one(const shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t test_id) {
    for (bool same_axis : {true, false}) {
        for (bool stateful : {true, false}) {
            NocEstimatorConfig cfg = {
                .test_id = test_id,
                .pattern = NocPattern::ONE_FROM_ONE,
                .mechanism = NocMechanism::UNICAST,
                .memory_type = MemoryType::L1,
                .same_axis = same_axis,
                .stateful = stateful,
            };
            packet_sizes_sweep(mesh_device, cfg);
        }
    }
}

static void sweep_one_to_all(const shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t test_id) {
    IDevice* device = mesh_device->impl().get_device(0);
    CoreCoord device_grid = device->compute_with_storage_grid_size();

    struct GridConfig {
        CoreCoord size;
    };
    vector<GridConfig> grids = {{{2, 2}}, {{3, 3}}, {{5, 5}}, {{8, 8}}, {device_grid}};

    for (auto& grid : grids) {
        // Skip grid sizes larger than device grid
        if (grid.size.x > device_grid.x || grid.size.y > device_grid.y) {
            continue;
        }
        bool is_full_grid = (grid.size.x >= device_grid.x && grid.size.y >= device_grid.y);

        // Unicast: master outside the sub grid (no loopback concept for unicast)
        // For full device grid, master must be inside since there's no core outside
        {
            NocEstimatorConfig cfg = {
                .test_id = test_id,
                .pattern = NocPattern::ONE_TO_ALL,
                .mechanism = NocMechanism::UNICAST,
                .memory_type = MemoryType::L1,
                .master_start_coord = is_full_grid ? CoreCoord(0, 0) : CoreCoord(grid.size.x, 0),
                .sub_start_coord = {0, 0},
                .sub_grid_size = grid.size,
            };
            packet_sizes_sweep(mesh_device, cfg);

            // Stateful unicast (only valid for small packets)
            packet_sizes_sweep_stateful_write(mesh_device, cfg);
        }

        // Multicast and multicast linked: sweep loopback
        for (auto mechanism : {NocMechanism::MULTICAST, NocMechanism::MULTICAST_LINKED}) {
            // Loopback = true: master inside the multicast rectangle (INCLUDE_SRC)
            {
                NocEstimatorConfig cfg = {
                    .test_id = test_id,
                    .pattern = NocPattern::ONE_TO_ALL,
                    .mechanism = mechanism,
                    .memory_type = MemoryType::L1,
                    .master_start_coord = {0, 0},
                    .sub_start_coord = {0, 0},
                    .sub_grid_size = grid.size,
                    .loopback = true,
                };
                packet_sizes_sweep(mesh_device, cfg);
            }

            // Loopback = false: master outside the multicast rectangle (EXCLUDE_SRC)
            // Skip for full device grid since there is no core outside the rectangle
            if (!is_full_grid) {
                NocEstimatorConfig cfg = {
                    .test_id = test_id,
                    .pattern = NocPattern::ONE_TO_ALL,
                    .mechanism = mechanism,
                    .memory_type = MemoryType::L1,
                    .master_start_coord = CoreCoord(grid.size.x, 0),
                    .sub_start_coord = {0, 0},
                    .sub_grid_size = grid.size,
                    .loopback = false,
                };
                packet_sizes_sweep(mesh_device, cfg);
            }
        }
    }
}

static void sweep_one_from_all(const shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t test_id) {
    IDevice* device = mesh_device->impl().get_device(0);
    CoreCoord device_grid = device->compute_with_storage_grid_size();

    // Sweep stateful for reads (reads can always use stateful with UNICAST)
    for (bool stateful : {false, true}) {
        NocEstimatorConfig cfg = {
            .test_id = test_id,
            .pattern = NocPattern::ONE_FROM_ALL,
            .mechanism = NocMechanism::UNICAST,
            .memory_type = MemoryType::L1,
            .sub_start_coord = {0, 0},
            .sub_grid_size = device_grid,
            .stateful = stateful,
        };
        packet_sizes_sweep(mesh_device, cfg);
    }
}

static void sweep_all_to_all(const shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t test_id) {
    IDevice* device = mesh_device->impl().get_device(0);
    CoreCoord device_grid = device->compute_with_storage_grid_size();

    vector<CoreCoord> grid_sizes = {{2, 2}, {3, 3}, {5, 5}, {8, 8}, device_grid};

    for (auto& grid_size : grid_sizes) {
        // Skip grid sizes larger than device grid
        if (grid_size.x > device_grid.x || grid_size.y > device_grid.y) {
            continue;
        }

        NocEstimatorConfig cfg = {
            .test_id = test_id,
            .pattern = NocPattern::ALL_TO_ALL,
            .mechanism = NocMechanism::UNICAST,
            .memory_type = MemoryType::L1,
            .master_start_coord = {0, 0},
            .master_grid_size = grid_size,
            .sub_start_coord = {0, 0},
            .sub_grid_size = grid_size,
        };
        packet_sizes_sweep(mesh_device, cfg);
    }
}

static void sweep_all_from_all(const shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t test_id) {
    IDevice* device = mesh_device->impl().get_device(0);
    CoreCoord device_grid = device->compute_with_storage_grid_size();

    vector<CoreCoord> grid_sizes = {{2, 2}, {3, 3}, {5, 5}, {8, 8}, device_grid};

    for (auto& grid_size : grid_sizes) {
        // Skip grid sizes larger than device grid
        if (grid_size.x > device_grid.x || grid_size.y > device_grid.y) {
            continue;
        }

        NocEstimatorConfig cfg = {
            .test_id = test_id,
            .pattern = NocPattern::ALL_FROM_ALL,
            .mechanism = NocMechanism::UNICAST,
            .memory_type = MemoryType::L1,
            .master_start_coord = {0, 0},
            .master_grid_size = grid_size,
            .sub_start_coord = {0, 0},
            .sub_grid_size = grid_size,
        };
        packet_sizes_sweep(mesh_device, cfg);
    }
}

// ============ DRAM ACCESSOR SWEEP FUNCTIONS ============

// Helper: run a DRAM accessor sweep with the given cores, bank assignments, and direction.
// Sweeps both NOC_0 and NOC_1 for each configuration.
static void dram_accessor_sweep(
    const shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t test_id,
    NocPattern pattern,
    MemoryType memory_type,
    const vector<CoreCoord>& cores,
    const vector<uint32_t>& bank_assignments,
    DramDirection direction) {
    auto [bytes_per_page, max_bytes, max_pages] = unit_tests::dm::compute_physical_constraints(mesh_device);

    uint32_t max_transactions = 256;
    uint32_t max_pages_per_txn = 256;

    for (NOC noc : {NOC::NOC_0, NOC::NOC_1}) {
        for (uint32_t num_txn = 1; num_txn <= max_transactions; num_txn *= 4) {
            for (uint32_t pages = 1; pages <= max_pages_per_txn; pages *= 2) {
                if (pages * bytes_per_page >= max_bytes) {
                    continue;
                }

                NocEstimatorConfig cfg = {
                    .test_id = test_id,
                    .pattern = pattern,
                    .mechanism = NocMechanism::UNICAST,
                    .memory_type = memory_type,
                    .num_of_transactions = num_txn,
                    .pages_per_transaction = pages,
                    .bytes_per_page = bytes_per_page,
                    .noc_id = noc,
                };
                EXPECT_TRUE(run_dram_accessor(mesh_device, cfg, cores, bank_assignments, direction));
            }
        }
    }

    ReadMeshDeviceProfilerResults(*mesh_device);
}

static void get_all_cores(
    const shared_ptr<distributed::MeshDevice>& mesh_device,
    vector<CoreCoord>& cores,
    vector<uint32_t>& bank_assignments) {
    IDevice* device = mesh_device->impl().get_device(0);
    CoreCoord device_grid = device->compute_with_storage_grid_size();
    uint32_t num_dram_banks = (uint32_t)device->num_dram_channels();
    for (uint32_t y = 0; y < device_grid.y; y++) {
        for (uint32_t x = 0; x < device_grid.x; x++) {
            cores.push_back(CoreCoord(x, y));
            bank_assignments.push_back((y * device_grid.x + x) % num_dram_banks);
        }
    }
}

// ---- READ sweeps ----

static void sweep_dram_sharded_one_from_one(const shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t test_id) {
    vector<CoreCoord> cores = {{0, 0}};
    vector<uint32_t> bank_assignments = {0};
    dram_accessor_sweep(
        mesh_device,
        test_id,
        NocPattern::ONE_FROM_ONE,
        MemoryType::DRAM_SHARDED,
        cores,
        bank_assignments,
        DramDirection::READ);
}

static void sweep_dram_sharded_all_from_all(const shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t test_id) {
    vector<CoreCoord> cores;
    vector<uint32_t> bank_assignments;
    get_all_cores(mesh_device, cores, bank_assignments);
    dram_accessor_sweep(
        mesh_device,
        test_id,
        NocPattern::ALL_FROM_ALL,
        MemoryType::DRAM_SHARDED,
        cores,
        bank_assignments,
        DramDirection::READ);
}

static void sweep_dram_interleaved_one_from_all(
    const shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t test_id) {
    vector<CoreCoord> cores = {{0, 0}};
    vector<uint32_t> bank_assignments = {0};
    dram_accessor_sweep(
        mesh_device,
        test_id,
        NocPattern::ONE_FROM_ALL,
        MemoryType::DRAM_INTERLEAVED,
        cores,
        bank_assignments,
        DramDirection::READ);
}

static void sweep_dram_interleaved_all_from_all(
    const shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t test_id) {
    vector<CoreCoord> cores;
    vector<uint32_t> bank_assignments;
    get_all_cores(mesh_device, cores, bank_assignments);
    dram_accessor_sweep(
        mesh_device,
        test_id,
        NocPattern::ALL_FROM_ALL,
        MemoryType::DRAM_INTERLEAVED,
        cores,
        bank_assignments,
        DramDirection::READ);
}

// ---- WRITE sweeps ----

static void sweep_dram_sharded_one_to_one(const shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t test_id) {
    vector<CoreCoord> cores = {{0, 0}};
    vector<uint32_t> bank_assignments = {0};
    dram_accessor_sweep(
        mesh_device,
        test_id,
        NocPattern::ONE_TO_ONE,
        MemoryType::DRAM_SHARDED,
        cores,
        bank_assignments,
        DramDirection::WRITE);
}

static void sweep_dram_sharded_all_to_all(const shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t test_id) {
    vector<CoreCoord> cores;
    vector<uint32_t> bank_assignments;
    get_all_cores(mesh_device, cores, bank_assignments);
    dram_accessor_sweep(
        mesh_device,
        test_id,
        NocPattern::ALL_TO_ALL,
        MemoryType::DRAM_SHARDED,
        cores,
        bank_assignments,
        DramDirection::WRITE);
}

static void sweep_dram_interleaved_one_to_all(
    const shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t test_id) {
    vector<CoreCoord> cores = {{0, 0}};
    vector<uint32_t> bank_assignments = {0};
    dram_accessor_sweep(
        mesh_device,
        test_id,
        NocPattern::ONE_TO_ALL,
        MemoryType::DRAM_INTERLEAVED,
        cores,
        bank_assignments,
        DramDirection::WRITE);
}

static void sweep_dram_interleaved_all_to_all(
    const shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t test_id) {
    vector<CoreCoord> cores;
    vector<uint32_t> bank_assignments;
    get_all_cores(mesh_device, cores, bank_assignments);
    dram_accessor_sweep(
        mesh_device,
        test_id,
        NocPattern::ALL_TO_ALL,
        MemoryType::DRAM_INTERLEAVED,
        cores,
        bank_assignments,
        DramDirection::WRITE);
}

// ============ ROW / COLUMN SWEEP FUNCTIONS ============

static void sweep_one_to_row(const shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t test_id) {
    IDevice* device = mesh_device->impl().get_device(0);
    CoreCoord device_grid = device->compute_with_storage_grid_size();

    // Unicast: master at (0, 0), sends individually to each core in the row
    {
        NocEstimatorConfig cfg = {
            .test_id = test_id,
            .pattern = NocPattern::ONE_TO_ROW,
            .mechanism = NocMechanism::UNICAST,
            .memory_type = MemoryType::L1,
            .master_start_coord = {0, 0},
            .sub_start_coord = {0, 0},
            .sub_grid_size = {device_grid.x, 1},
        };
        packet_sizes_sweep(mesh_device, cfg);

        // Stateful unicast (only valid for small packets)
        packet_sizes_sweep_stateful_write(mesh_device, cfg);
    }

    // Multicast and multicast linked: sweep loopback
    for (auto mechanism : {NocMechanism::MULTICAST, NocMechanism::MULTICAST_LINKED}) {
        // Loopback = true: master inside the row (INCLUDE_SRC)
        {
            NocEstimatorConfig cfg = {
                .test_id = test_id,
                .pattern = NocPattern::ONE_TO_ROW,
                .mechanism = mechanism,
                .memory_type = MemoryType::L1,
                .master_start_coord = {0, 0},
                .sub_start_coord = {0, 0},
                .sub_grid_size = {device_grid.x, 1},
                .loopback = true,
            };
            packet_sizes_sweep(mesh_device, cfg);
        }

        // Loopback = false: master outside the row (EXCLUDE_SRC)
        // Master at row 1, multicasting to row 0
        if (device_grid.y > 1) {
            NocEstimatorConfig cfg = {
                .test_id = test_id,
                .pattern = NocPattern::ONE_TO_ROW,
                .mechanism = mechanism,
                .memory_type = MemoryType::L1,
                .master_start_coord = {0, 1},
                .sub_start_coord = {0, 0},
                .sub_grid_size = {device_grid.x, 1},
                .loopback = false,
            };
            packet_sizes_sweep(mesh_device, cfg);
        }
    }
}

static void sweep_row_to_row(const shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t test_id) {
    IDevice* device = mesh_device->impl().get_device(0);
    CoreCoord device_grid = device->compute_with_storage_grid_size();

    // Unicast: every core in the row sends individually to every core in the row
    {
        NocEstimatorConfig cfg = {
            .test_id = test_id,
            .pattern = NocPattern::ROW_TO_ROW,
            .mechanism = NocMechanism::UNICAST,
            .memory_type = MemoryType::L1,
            .master_start_coord = {0, 0},
            .master_grid_size = {device_grid.x, 1},
            .sub_start_coord = {0, 0},
            .sub_grid_size = {device_grid.x, 1},
        };
        packet_sizes_sweep(mesh_device, cfg);
    }

    // Multicast and multicast linked: sweep loopback
    for (auto mechanism : {NocMechanism::MULTICAST, NocMechanism::MULTICAST_LINKED}) {
        // Loopback = true: masters inside the multicast row (INCLUDE_SRC)
        {
            NocEstimatorConfig cfg = {
                .test_id = test_id,
                .pattern = NocPattern::ROW_TO_ROW,
                .mechanism = mechanism,
                .memory_type = MemoryType::L1,
                .master_start_coord = {0, 0},
                .master_grid_size = {device_grid.x, 1},
                .sub_start_coord = {0, 0},
                .sub_grid_size = {device_grid.x, 1},
                .loopback = true,
            };
            packet_sizes_sweep(mesh_device, cfg);
        }

        // Loopback = false: masters in row 1, multicasting to row 0 (EXCLUDE_SRC)
        if (device_grid.y > 1) {
            NocEstimatorConfig cfg = {
                .test_id = test_id,
                .pattern = NocPattern::ROW_TO_ROW,
                .mechanism = mechanism,
                .memory_type = MemoryType::L1,
                .master_start_coord = {0, 1},
                .master_grid_size = {device_grid.x, 1},
                .sub_start_coord = {0, 0},
                .sub_grid_size = {device_grid.x, 1},
                .loopback = false,
            };
            packet_sizes_sweep(mesh_device, cfg);
        }
    }
}

static void sweep_one_to_column(const shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t test_id) {
    IDevice* device = mesh_device->impl().get_device(0);
    CoreCoord device_grid = device->compute_with_storage_grid_size();

    // Unicast: master at (0, 0), sends individually to each core in the column
    {
        NocEstimatorConfig cfg = {
            .test_id = test_id,
            .pattern = NocPattern::ONE_TO_COLUMN,
            .mechanism = NocMechanism::UNICAST,
            .memory_type = MemoryType::L1,
            .master_start_coord = {0, 0},
            .sub_start_coord = {0, 0},
            .sub_grid_size = {1, device_grid.y},
        };
        packet_sizes_sweep(mesh_device, cfg);

        // Stateful unicast (only valid for small packets)
        packet_sizes_sweep_stateful_write(mesh_device, cfg);
    }

    // Multicast and multicast linked: sweep loopback
    for (auto mechanism : {NocMechanism::MULTICAST, NocMechanism::MULTICAST_LINKED}) {
        // Loopback = true: master inside the column (INCLUDE_SRC)
        {
            NocEstimatorConfig cfg = {
                .test_id = test_id,
                .pattern = NocPattern::ONE_TO_COLUMN,
                .mechanism = mechanism,
                .memory_type = MemoryType::L1,
                .master_start_coord = {0, 0},
                .sub_start_coord = {0, 0},
                .sub_grid_size = {1, device_grid.y},
                .loopback = true,
            };
            packet_sizes_sweep(mesh_device, cfg);
        }

        // Loopback = false: master outside the column (EXCLUDE_SRC)
        // Master at column 1, multicasting to column 0
        if (device_grid.x > 1) {
            NocEstimatorConfig cfg = {
                .test_id = test_id,
                .pattern = NocPattern::ONE_TO_COLUMN,
                .mechanism = mechanism,
                .memory_type = MemoryType::L1,
                .master_start_coord = {1, 0},
                .sub_start_coord = {0, 0},
                .sub_grid_size = {1, device_grid.y},
                .loopback = false,
            };
            packet_sizes_sweep(mesh_device, cfg);
        }
    }
}

static void sweep_column_to_column(const shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t test_id) {
    IDevice* device = mesh_device->impl().get_device(0);
    CoreCoord device_grid = device->compute_with_storage_grid_size();

    // Unicast: every core in the column sends individually to every core in the column
    {
        NocEstimatorConfig cfg = {
            .test_id = test_id,
            .pattern = NocPattern::COLUMN_TO_COLUMN,
            .mechanism = NocMechanism::UNICAST,
            .memory_type = MemoryType::L1,
            .master_start_coord = {0, 0},
            .master_grid_size = {1, device_grid.y},
            .sub_start_coord = {0, 0},
            .sub_grid_size = {1, device_grid.y},
        };
        packet_sizes_sweep(mesh_device, cfg);
    }

    // Multicast and multicast linked: sweep loopback
    for (auto mechanism : {NocMechanism::MULTICAST, NocMechanism::MULTICAST_LINKED}) {
        // Loopback = true: masters inside the multicast column (INCLUDE_SRC)
        {
            NocEstimatorConfig cfg = {
                .test_id = test_id,
                .pattern = NocPattern::COLUMN_TO_COLUMN,
                .mechanism = mechanism,
                .memory_type = MemoryType::L1,
                .master_start_coord = {0, 0},
                .master_grid_size = {1, device_grid.y},
                .sub_start_coord = {0, 0},
                .sub_grid_size = {1, device_grid.y},
                .loopback = true,
            };
            packet_sizes_sweep(mesh_device, cfg);
        }

        // Loopback = false: masters in column 1, multicasting to column 0 (EXCLUDE_SRC)
        if (device_grid.x > 1) {
            NocEstimatorConfig cfg = {
                .test_id = test_id,
                .pattern = NocPattern::COLUMN_TO_COLUMN,
                .mechanism = mechanism,
                .memory_type = MemoryType::L1,
                .master_start_coord = {1, 0},
                .master_grid_size = {1, device_grid.y},
                .sub_start_coord = {0, 0},
                .sub_grid_size = {1, device_grid.y},
                .loopback = false,
            };
            packet_sizes_sweep(mesh_device, cfg);
        }
    }
}

}  // namespace unit_tests::dm::noc_estimator

// ============ TEST CASES ============

// ---- L1 tests (800-809) ----
TEST_F(GenericMeshDeviceFixture, NocEstimatorL1OneToOne) {
    unit_tests::dm::noc_estimator::sweep_one_to_one(get_mesh_device(), 800);
}

TEST_F(GenericMeshDeviceFixture, NocEstimatorL1OneFromOne) {
    unit_tests::dm::noc_estimator::sweep_one_from_one(get_mesh_device(), 801);
}

TEST_F(GenericMeshDeviceFixture, NocEstimatorL1OneToAll) {
    unit_tests::dm::noc_estimator::sweep_one_to_all(get_mesh_device(), 802);
}

TEST_F(GenericMeshDeviceFixture, NocEstimatorL1OneFromAll) {
    unit_tests::dm::noc_estimator::sweep_one_from_all(get_mesh_device(), 803);
}

TEST_F(GenericMeshDeviceFixture, NocEstimatorL1AllToAll) {
    unit_tests::dm::noc_estimator::sweep_all_to_all(get_mesh_device(), 804);
}

TEST_F(GenericMeshDeviceFixture, NocEstimatorL1AllFromAll) {
    unit_tests::dm::noc_estimator::sweep_all_from_all(get_mesh_device(), 805);
}

TEST_F(GenericMeshDeviceFixture, NocEstimatorL1OneToRow) {
    unit_tests::dm::noc_estimator::sweep_one_to_row(get_mesh_device(), 806);
}

TEST_F(GenericMeshDeviceFixture, NocEstimatorL1RowToRow) {
    unit_tests::dm::noc_estimator::sweep_row_to_row(get_mesh_device(), 807);
}

TEST_F(GenericMeshDeviceFixture, NocEstimatorL1OneToColumn) {
    unit_tests::dm::noc_estimator::sweep_one_to_column(get_mesh_device(), 808);
}

TEST_F(GenericMeshDeviceFixture, NocEstimatorL1ColumnToColumn) {
    unit_tests::dm::noc_estimator::sweep_column_to_column(get_mesh_device(), 809);
}

// ---- DRAM Read tests (810-813) ----
TEST_F(GenericMeshDeviceFixture, NocEstimatorDRAMShardedOneFromOne) {
    unit_tests::dm::noc_estimator::sweep_dram_sharded_one_from_one(get_mesh_device(), 810);
}

TEST_F(GenericMeshDeviceFixture, NocEstimatorDRAMShardedAllFromAll) {
    unit_tests::dm::noc_estimator::sweep_dram_sharded_all_from_all(get_mesh_device(), 811);
}

TEST_F(GenericMeshDeviceFixture, NocEstimatorDRAMInterleavedOneFromAll) {
    unit_tests::dm::noc_estimator::sweep_dram_interleaved_one_from_all(get_mesh_device(), 812);
}

TEST_F(GenericMeshDeviceFixture, NocEstimatorDRAMInterleavedAllFromAll) {
    unit_tests::dm::noc_estimator::sweep_dram_interleaved_all_from_all(get_mesh_device(), 813);
}

// ---- DRAM Write tests (814-817) ----
TEST_F(GenericMeshDeviceFixture, NocEstimatorDRAMShardedOneToOne) {
    unit_tests::dm::noc_estimator::sweep_dram_sharded_one_to_one(get_mesh_device(), 814);
}

TEST_F(GenericMeshDeviceFixture, NocEstimatorDRAMShardedAllToAll) {
    unit_tests::dm::noc_estimator::sweep_dram_sharded_all_to_all(get_mesh_device(), 815);
}

TEST_F(GenericMeshDeviceFixture, NocEstimatorDRAMInterleavedOneToAll) {
    unit_tests::dm::noc_estimator::sweep_dram_interleaved_one_to_all(get_mesh_device(), 816);
}

TEST_F(GenericMeshDeviceFixture, NocEstimatorDRAMInterleavedAllToAll) {
    unit_tests::dm::noc_estimator::sweep_dram_interleaved_all_to_all(get_mesh_device(), 817);
}

}  // namespace tt::tt_metal
