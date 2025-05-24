// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "device_fixture.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "dm_common.hpp"

namespace tt::tt_metal {

using namespace std;
using namespace tt;
using namespace tt::test_utils;

namespace unit_tests::dm::all_to_all {

constexpr uint32_t START_ID = 60;

// TO-DO: Rename "physical" to "worker"? Do they mean the same thing?

// Test Config (i.e. test parameters)
struct AllToAllConfig {
    /* Test ID */
    uint32_t test_id = START_ID;

    /* Grid configurations */
    CoreCoord mst_logical_start_coord =
        CoreCoord();  // Commenting out for now. For now, the assumed master and subordinate starting coordiantes are
                      // both assumed to be (0,0). Maybe if we do 2x2 and 5x5, we can have different starting
                      // coordinates for each (or both).
    CoreCoord sub_logical_start_coord = CoreCoord();
    CoreCoord mst_grid_size = CoreCoord();
    CoreCoord sub_grid_size = CoreCoord();

    /* Transaction size configurations */
    uint32_t num_of_transactions = 0;
    uint32_t transaction_size_pages = 0;

    /* Write configurations */
    DataFormat l1_data_format = DataFormat::Invalid;
    bool loopback = false;
    NOC noc_id = NOC::NOC_0;
    bool is_multicast = false;
    bool is_linked = false;

    // TODO: Add the following parameters
    //  1. Virtual Channel (only useful for unicast)
    //  2. Posted flag (posted multicast has much better performance at larger grid sizes, than non-posted due to
    //  response packets) (60, 45, 23, vs 60, 60, 60 at posted)
};

/// @brief Performs communication from L1 Sender Cores to L1 Receiver Cores.
/// @param device The device on which the test is executed.
/// @param test_config Configuration of the test, defined by a specific struct.
/// @return Status of the test execution (e.g., success or failure).
bool run_dm(IDevice* device, const OneToAllConfig& test_config) {
    // Program
    Program program = CreateProgram();

    /* INITIALIZATION */

    const uint32_t page_size_bytes =
        arch_ == tt::ARCH::BLACKHOLE
            ? 64
            : 32;  // =Flit size: 32 bytes for WH, 64 for BH // This could probably go in the common file

    // Initialize core sets //
    /*
        - CoreRange: Represents a single rectangular range of cores in a 2D grid. It is defined by a starting coordinate
        (`start_coord`) and an ending coordinate (`end_coord`). For example, a CoreRange from (0, 0) to (3, 3) would
        include all cores in that rectangular area.

        - CoreRangeSet: Represents a collection of CoreRange objects. It is used to manage multiple ranges of cores
        and provides functionality for operations like merging, subtracting, and finding intersections between ranges.

        - If `loopback` is disabled in the test configuration, the subordinate cores are adjusted to exclude any
        overlap with the master cores by subtracting `mst_core_set` from `sub_core_set`.
    */

    // Logical Coordinate Ranges

    // Master
    CoreCoord mst_logical_start_coord = test_config.mst_logical_start_coord;
    CoreCoord mst_logical_end_coord = CoreCoord(
        mst_logical_start_coord.x + test_config.mst_grid_size.x - 1,
        mst_logical_start_coord.y + test_config.mst_grid_size.y - 1);
    CoreRangeSet mst_logical_core_set({CoreRange(mst_logical_start_coord, mst_logical_end_coord)});

    // Subordinate
    CoreCoord sub_logical_start_coord = test_config.sub_logical_start_coord;
    CoreCoord sub_logical_end_coord = CoreCoord(
        sub_logical_start_coord.x + test_config.sub_grid_size.x - 1,
        sub_logical_start_coord.y + test_config.sub_grid_size.y - 1);
    CoreRangeSet sub_logical_core_set({CoreRange(sub_logical_start_coord, sub_logical_end_coord)});

    // Subordinate CoreRangeSet must not contain any subset of master cores
    // IDEA: Consider performing this check in the kernel itself instead of here. It would have to be performed per
    // master core
    if (!test_config.loopback) {
        sub_logical_core_set = sub_logical_core_set.subtract(mst_logical_core_set);
    }

    // Keep track of the number of master and subordinate cores
    uint32_t num_masters = mst_logical_core_set.num_cores();
    uint32_t num_subordinates = sub_logical_core_set.num_cores();

    // -------------------- //

    // Determine total size of the sender and receiver buffers // REVISIT: This may have to be modified since we have
    // multiple senders now
    const uint32_t transaction_size_pages = test_config.transaction_size_pages;
    const uint32_t transaction_size_bytes = transaction_size_pages * page_size_bytes;
    const uint32_t total_sender_size_pages = transaction_size_pages * num_subordinates;
    const uint32_t total_sender_size_bytes = transaction_size_bytes * num_subordinates;

    // Initialize shard parameters and buffers

    // Shard Parameters
    /*
        Represents the configuration for a shard buffer, which defines how data is distributed
        across a set of cores in a 2D grid.

        CoreRangeSet& core_sets_:
            A set of cores (processing units) used for the shard grid. This defines the
            physical or logical cores that participate in the data movement operation.

        std::array<uint32_t, 2>& shard_shape_:
            The logical shape of the shard in terms of rows and columns. This determines
    *       how the data is logically divided across the cores.

        ShardOrientation& shard_orientation_:
            The layout order of cores in the shard grid. For example, this could be
            row-major or column-major, which defines how data is traversed or accessed.

        std::array<uint32_t, 2>& page_shape:
            The shape of a single page in the shard, specified as rows and columns. This
            determines the granularity of data distribution within the shard.

        std::array<uint32_t, 2>& tensor2d_shape_in_pages):
            The shape of the entire 2D tensor in terms of pages. This defines the overall
            size of the data being distributed across the cores.

    */

    auto mst_shard_parameters = ShardSpecBuffer(
        mst_core_set,
        {1, transaction_size_bytes / 2},
        ShardOrientation::ROW_MAJOR,
        {1, page_size_bytes / 2},
        {1, transaction_size_pages});
    auto sub_shard_parameters = ShardSpecBuffer(
        sub_core_set,
        {1, transaction_size_bytes / 2},
        ShardOrientation::ROW_MAJOR,
        {1, page_size_bytes / 2},
        {1, total_sender_size_pages});

    // Buffers
    /*
        ShardedBufferConfig is a structure that defines the configuration for a sharded buffer.
        It contains the following members:

        IDevice* device:
            A pointer to the device on which the buffer is allocated.

        DeviceAddr size:
            The size of the buffer in bytes.

        DeviceAddr page_size:
            The size of the unit being interleaved. For non-interleaved buffers, size == page_size.

        BufferType buffer_type:
            The type of buffer. Defaults to BufferType::L1.

        TensorMemoryLayout buffer_layout:
            The memory layout of the buffer. Defaults to TensorMemoryLayout::HEIGHT_SHARDED.

        ShardSpecBuffer shard_parameters:
            The parameters that define the sharding configuration of the buffer.
    */

    auto mst_l1_buffer = CreateBuffer(ShardedBufferConfig{
        .device = device,
        .size = transaction_size_bytes,
        .page_size = page_size_bytes,
        .buffer_type = BufferType::L1,
        .buffer_layout = TensorMemoryLayout::WIDTH_SHARDED,
        .shard_parameters = std::move(master_shard_parameters),
    });
    auto sub_l1_buffer = CreateBuffer(ShardedBufferConfig{
        .device = device,
        .size = total_sender_size_bytes,
        .page_size = page_size_bytes,
        .buffer_type = BufferType::L1,
        .buffer_layout = TensorMemoryLayout::WIDTH_SHARDED,
        .shard_parameters = std::move(subordinate_shard_parameters),
    });

    // Get byte addresses
    uint32_t mst_l1_byte_address = mst_l1_buffer->address();
    uint32_t sub_l1_byte_address = sub_l1_buffer->address();

    // Semaphores
    CoreRangeSet sem_core_set = sub_core_set.merge<CoreRangeSet>(mst_core_set);
    const uint32_t sem_id = CreateSemaphore(program, sem_core_set, 0);

    // QUESTION: Is there a single semaphore per ID per core? Or is it a single semaphore for all cores?

    // Physical Sub Coordinates, Needed for passing in the coordintes for the multicast function
    CoreCoord sub_physical_start_coord = device->worker_core_from_logical_core(sub_logical_start_coord);
    CoreCoord sub_physical_end_coord = device->worker_core_from_logical_core(sub_logical_end_coord);
    std::vector<uint32_t> sub_physical_coords = {
        sub_physical_start_coord.x, sub_physical_start_coord.y, sub_physical_end_coord.x, sub_physical_end_coord.y};

// Check this part. Do we really need each x and y coordinate to take up 32 bits of space on the runtime arguments?

// Obtaining every set of physical coordinates for the master and subordinate cores
// NOTE: very minor but this could technically go in a helper function

// Masters
std:;
    vector<uint32_t> mst_coordinates = {};

    for (auto& mst_logical_core : corerange_to_cores(mst_logical_core_set)) {
        CoreCoord mst_physical_core = device->worker_core_from_logical_core(mst_logical_core);
        mst_coordinates.push_back(mst_logical_core.x);
        mst_coordinates.push_back(mst_logical_core.y);
        insert_index += 2;  // Update the index for the next insertion
    }
    // Subordinates
    std::vector<uint32_t> sub_coordinates = {};

    for (auto& sub_logical_core : corerange_to_cores(sub_logical_core_set)) {
        // REVISIT: Loopback check, deal with this later // WE COULD maybe implement this check in the kernel itself
        /*if (!test_config.loopback &&
            (sub_logical_core.x == test_config.master_core_coord.x && sub_logical_core.y ==
        test_config.master_core_coord.y)) { continue;
        }*/ // LOOPBACK CHECK, TO BE REVISITED
        CoreCoord sub_physical_core = device->worker_core_from_logical_core(sub_logical_core);  // This part namely
        sub_coordinates.push_back(sub_logical_core.x);
        sub_coordinates.push_back(sub_logical_core.y);
    }

    // Kernels

    // Compile-time arguments for kernels

    // Sender Kernel
    std::vector<uint32_t> sender_compile_args = {
        (uint32_t)mst_l1_byte_address,
        (uint32_t)sub_l1_byte_address,
        (uint32_t)test_config.num_of_transactions,
        (uint32_t)test_config.transaction_size_pages,
        (uint32_t)test_config.page_size_bytes,
        (uint32_t)test_config.test_id,
        (uint32_t)num_subordinates,
        (uint32_t)total_sender_size_bytes,
        (uint32_t)test_config.is_linked,
        (uint32_t)semaphore_id};
    // Make it so that the contents of sub_coords are appended to sender_compile_args
    sender_compile_args.insert(sender_compile_args.end(), sub_physical_coords.begin(), sub_physical_coords.end());
    size_t mst_coord_index =
        20  // NOTE: arbitrarily chosen
        sender_compile_args.insert(
            sender_compile_args.begin() + mst_coord_index, mst_coordinates.begin(), mst_coordinates.end());
    size_t sub_coord_index =
        120  // OR have this be the halfway point betewen 5 and 256
        sender_compile_args.insert(
            sender_compile_args.begin() + sub_coord_index, sub_coordinates.begin(), sub_coordinates.end());

    // Create kernels // QUESTION: Is this the point of compilation of the kernel?
    auto sender_kernel = CreateKernel(
        program,
        test_config.is_multicast ? "tests/tt_metal/tt_metal/data_movement/one_to_all/kernels/"
                                   "sender_all_multicast.cpp"  // Defining the base directory in one string and
                                                               // appending to it in another? Is that possible?
                                 : "tests/tt_metal/tt_metal/data_movement/one_to_all/kernels/sender_all.cpp",
        master_core_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = test_config.noc_id,
            .compile_args = sender_compile_args});

    // Define runtime arguments for the kernels

    // Assign runtime arguments to the kernels
    SetRuntimeArgs(program, sender_kernel, master_core_set, master_run_args);

    /* RUNNING THE TEST PROGRAM */

    // Assign unique id
    log_info("Running Test ID: {}, Run ID: {}", test_config.test_id, unit_tests::dm::runtime_host_id);
    program.set_runtime_id(unit_tests::dm::runtime_host_id++);

    // Input
    vector<uint32_t> packed_input = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -100.0f,
        100.0f,
        transaction_size_bytes / bfloat16::SIZEOF,
        chrono::system_clock::now().time_since_epoch().count());

    // Golden output
    vector<uint32_t> packed_golden = packed_input;
    for (uint32_t i = 1; i < num_subordinates; i++) {
        packed_golden.insert(packed_golden.end(), packed_input.begin(), packed_input.end());
    }

    // Launch program and record outputs
    detail::WriteToBuffer(master_l1_buffer, packed_input);
    MetalContext::instance().get_cluster().l1_barrier(device->id());
    detail::LaunchProgram(device, program);

    vector<uint32_t> packed_output;
    detail::ReadFromBuffer(subordinate_l1_buffer, packed_output);

    // Results comparison
    bool pcc = is_close_packed_vectors<bfloat16, uint32_t>(
        packed_output, packed_golden, [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b); });
    if (!pcc) {
        log_error("PCC Check failed");
        log_info("Golden vector");
        print_vector<uint32_t>(packed_golden);
        log_info("Output vector");
        print_vector<uint32_t>(packed_output);
    }

    return pcc;
}
}  // namespace unit_tests::dm::all_to_all

/* =============================================================  /
/  ========== TEST CASES FOR ALL-TO-ALL DATA MOVEMENT ==========  /
/  ============================================================= */

/* ======== No Multicast ======== */
TEST_F(DeviceFixture, TensixDataMovementAllToAll) {}

/* ======== Multicast ========= */
/* ======== 2x2 to 2x2 ======== */
TEST_F(DeviceFixture, TensixDataMovementAllToAllMulticast11x10PacketSizes) {
    uint32_t test_case_id = 1;

    /* Parameters */

    uint32_t max_transactions = 256;
    uint32_t max_transaction_size_pages = 64;

    // IDEA: Implement a for loop that shuffles through several of each of these coordinates to test grids of different
    // locations
    CoreCoord mst_start_coord = {0, 0};
    CoreCorrd sub_start_coord = {0, 0};

    CoreCoord mst_grid_size = {2, 2};
    CoreCoord sub_grid_size = {2, 2};

    NOC noc_id = NOC::NOC_0;
    bool is_linked = false;

    /* Running the Test */

    // Number of Transactions: 1, 4, 16, 64, 256
    for (uint32_t num_of_transactions = 1; num_of_transactions <= max_transactions; num_of_transactions *= 4) {
        // Transaction Size: 1, 2, 4, 8, 16, 32, 64
        for (uint32_t transaction_size_pages = 1; transaction_size_pages <= max_transaction_size_pages;
             transaction_size_pages *= 2) {
            // Test config
            unit_tests::dm::core_to_all::AllToAllConfig test_config = {
                .test_id = unit_tests::dm::all_to_all::START_ID + test_case_id,

                .mst_start_coord = mst_start_coord,
                .sub_start_coord = sub_start_coord,
                .mst_grid_size = mst_grid_size,
                .sub_grid_size = sub_grid_size,

                .num_of_transactions = num_of_transactions,
                .transaction_size_pages = transaction_size_pages,

                .l1_data_format = DataFormat::Float16_b,
                .loopback = true,
                .noc_id = noc_id,
                .is_multicast = true,
                .is_linked = is_linked,
            };

            // Run
            for (unsigned int id = 0; id < num_devices_; id++) {
                EXPECT_TRUE(run_dm(devices_.at(id), test_config));
            }
        }
    }
}

/* ======== Multicast ======== */
TEST_F(DeviceFixture, TensixDataMovementAllToAllMulticast11x10PacketSizes) {
    uint32_t test_case_id = 3;

    /* Parameters */

    uint32_t max_transactions = 256;
    uint32_t max_transaction_size_pages = 64;

    // CoreCoord mst_core_coord = {0, 0};
    // CoreCorrd sub_core_coord = {0, 0};

    CoreCoord mst_grid_size = {
        devices_.at(0)->compute_with_storage_grid_size().x, devices_.at(0)->compute_with_storage_grid_size().y};
    CoreCoord sub_grid_size = {
        devices_.at(0)->compute_with_storage_grid_size().x, devices_.at(0)->compute_with_storage_grid_size().y};

    NOC noc_id = NOC::NOC_0;
    bool is_linked = false;

    // REVISIT THIS!!
    // Maybe also something for mst_grid_size
    // ...
    // Limit the grid size to 100 cores because the max allowed kernel args is 256 uint32_t's
    if (sub_grid_size.x * sub_grid_size.y > 100) {
        uint32_t smaller_dim = sub_grid_size.x > sub_grid_size.y ? sub_grid_size.y : sub_grid_size.x;
        sub_grid_size = {smaller_dim, smaller_dim};
    }

    /* Running the Test */

    // Number of Transactions: 1, 4, 16, 64, 256
    for (uint32_t num_of_transactions = 1; num_of_transactions <= max_transactions; num_of_transactions *= 4) {
        // Transaction Size: 1, 2, 4, 8, 16, 32, 64
        for (uint32_t transaction_size_pages = 1; transaction_size_pages <= max_transaction_size_pages;
             transaction_size_pages *= 2) {
            // Test config
            unit_tests::dm::core_to_all::AllToAllConfig test_config = {
                .test_id = unit_tests::dm::all_to_all::START_ID + test_case_id,

                // .master_core_coord = master_core_coord,
                // .subordinate_core_coord = subordinate_core_coord,
                .mst_grid_size = mst_grid_size,
                .sub_grid_size = sub_grid_size,

                .num_of_transactions = num_of_transactions,
                .transaction_size_pages = transaction_size_pages,

                .l1_data_format = DataFormat::Float16_b,
                .loopback = true,
                .noc_id = noc_id,
                .is_multicast = true,
                .is_linked = is_linked,
            };

            // Run
            for (unsigned int id = 0; id < num_devices_; id++) {
                EXPECT_TRUE(run_dm(devices_.at(id), test_config));
            }
        }
    }
}

/* ======== Multicast Linked ======== */

}  // namespace tt::tt_metal
