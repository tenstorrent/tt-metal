#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/host_api.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"


//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;

CoreCoord get_logical_coord_from_noc_coord(tt::tt_metal::Device *device, const CoreCoord &noc_coord) {
    auto soc_desc = device->cluster()->get_soc_desc(device->pcie_slot());
    auto logical_coord_x = soc_desc.routing_x_to_worker_x.at(noc_coord.x);
    auto logical_coord_y = soc_desc.routing_y_to_worker_y.at(noc_coord.y);
    return CoreCoord(logical_coord_x, logical_coord_y);
}

std::vector<uint32_t> get_logical_compute_and_storage_core_bank_ids(tt_metal::Device *device) {
    auto soc_desc = device->cluster()->get_soc_desc(device->pcie_slot());
    auto compute_and_storage_core_noc_coord = soc_desc.compute_and_storage_cores.at(0);
    auto logical_core = get_logical_coord_from_noc_coord(device, compute_and_storage_core_noc_coord);
    return device->bank_ids_from_logical_core(logical_core);
}

std::vector<uint32_t> get_logical_storage_core_bank_ids(tt_metal::Device *device) {
    auto soc_desc = device->cluster()->get_soc_desc(device->pcie_slot());
    auto storage_core_noc_coord = soc_desc.storage_cores.at(0);
    auto logical_core = get_logical_coord_from_noc_coord(device, storage_core_noc_coord);
    return device->bank_ids_from_logical_core(logical_core);
}

bool test_l1_buffers_allocated_top_down(tt_metal::Device *device) {
    bool pass = true;

    auto logical_compute_and_storage_bank_ids = get_logical_compute_and_storage_core_bank_ids(device);
    TT_ASSERT(logical_compute_and_storage_bank_ids.size() == 1);
    auto compute_and_storage_bank_id = logical_compute_and_storage_bank_ids.at(0);
    auto logical_storage_core_bank_ids = get_logical_storage_core_bank_ids(device);
    TT_ASSERT(logical_storage_core_bank_ids.size() == 2);

    uint32_t buffer_0_size_bytes = 128 * 1024;
    auto l1_buffer_0 = tt_metal::Buffer(device, buffer_0_size_bytes, compute_and_storage_bank_id, buffer_0_size_bytes, tt_metal::BufferType::L1);
    pass &= l1_buffer_0.address() == (device->l1_size() - buffer_0_size_bytes);

    uint32_t buffer_1_size_bytes = 64 * 1024;
    auto l1_buffer_1 = tt_metal::Buffer(device, buffer_1_size_bytes, compute_and_storage_bank_id, buffer_1_size_bytes, tt_metal::BufferType::L1);
    pass &= l1_buffer_1.address() == (device->l1_size() - buffer_0_size_bytes - buffer_1_size_bytes);

    uint32_t buffer_2_size_bytes = 256 * 1024;
    auto l1_buffer_2 = tt_metal::Buffer(device, buffer_2_size_bytes, compute_and_storage_bank_id, buffer_2_size_bytes, tt_metal::BufferType::L1);
    pass &= l1_buffer_2.address() == (device->l1_size() - buffer_0_size_bytes - buffer_1_size_bytes - buffer_2_size_bytes);

    // storage_l1_buffer_0, storage_l1_buffer_1, and storage_l1_buffer_2 are stored in bank 0
    auto storage_l1_buffer_0 = tt_metal::Buffer(device, buffer_0_size_bytes, logical_storage_core_bank_ids.at(0), buffer_0_size_bytes, tt_metal::BufferType::L1);
    pass &= storage_l1_buffer_0.address() == ((device->l1_size()/2) - buffer_0_size_bytes);

    auto storage_l1_buffer_1 = tt_metal::Buffer(device, buffer_1_size_bytes, logical_storage_core_bank_ids.at(0), buffer_1_size_bytes, tt_metal::BufferType::L1);
    pass &= storage_l1_buffer_1.address() == ((device->l1_size()/2) - buffer_0_size_bytes - buffer_1_size_bytes);

    auto storage_l1_buffer_2 = tt_metal::Buffer(device, buffer_2_size_bytes, logical_storage_core_bank_ids.at(0), buffer_2_size_bytes, tt_metal::BufferType::L1);
    pass &= storage_l1_buffer_2.address() == ((device->l1_size()/2) - buffer_0_size_bytes - buffer_1_size_bytes - buffer_2_size_bytes);

    // storage_l1_buffer_3 is stored in bank 1
    auto storage_l1_buffer_3 = tt_metal::Buffer(device, buffer_2_size_bytes, logical_storage_core_bank_ids.at(1), buffer_2_size_bytes, tt_metal::BufferType::L1);
    pass &= storage_l1_buffer_3.address() == (device->l1_size() - buffer_2_size_bytes);

    return pass;
}

bool test_circular_buffers_allocated_bottom_up(tt_metal::Device *device, tt_metal::Program &program) {
    bool pass = true;

    auto logical_compute_and_storage_bank_ids = get_logical_compute_and_storage_core_bank_ids(device);
    TT_ASSERT(logical_compute_and_storage_bank_ids.size() == 1);
    auto compute_and_storage_bank_id = logical_compute_and_storage_bank_ids.at(0);
    auto logical_core = device->logical_core_from_bank_id(compute_and_storage_bank_id);

    uint32_t single_tile_size = 2 * 1024;
    constexpr uint32_t src0_cb_index = CB::c_in0;
    constexpr uint32_t num_input_tiles = 2;
    tt_metal::CircularBuffer *cb_src0 = tt_metal::CreateCircularBuffer(
        program,
        device,
        src0_cb_index,
        logical_core,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        tt::DataFormat::Float16_b
    );
    pass &= cb_src0->address() == UNRESERVED_BASE;

    constexpr uint32_t src1_cb_index = CB::c_in1;
    tt_metal::CircularBuffer *cb_src1 = tt_metal::CreateCircularBuffer(
        program,
        device,
        src1_cb_index,
        logical_core,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        tt::DataFormat::Float16_b
    );
    pass &= (cb_src1->address() == (cb_src0->address() + cb_src0->size()));

    constexpr uint32_t output_cb_index = CB::c_out0;
    constexpr uint32_t num_output_tiles = 2;
    tt_metal::CircularBuffer *cb_output = tt_metal::CreateCircularBuffer(
        program,
        device,
        output_cb_index,
        logical_core,
        num_output_tiles,
        num_output_tiles * single_tile_size,
        tt::DataFormat::Float16_b
    );
    pass &= (cb_output->address() == (cb_src1->address() + cb_src1->size()));

    return pass;
}

bool test_l1_buffer_do_not_grow_beyond_512KB(tt_metal::Device *device) {
    bool pass = true;

    auto logical_compute_and_storage_bank_ids = get_logical_compute_and_storage_core_bank_ids(device);
    TT_ASSERT(logical_compute_and_storage_bank_ids.size() == 1);
    auto compute_and_storage_bank_id = logical_compute_and_storage_bank_ids.at(0);

    try {
        uint32_t buffer_size_bytes = 128 * 1024;
        auto l1_buffer = tt_metal::Buffer(device, buffer_size_bytes, compute_and_storage_bank_id, buffer_size_bytes, tt_metal::BufferType::L1);
    } catch (const std::exception &e) {
        pass = true;
    }

    return pass;
}

bool test_circular_buffers_allowed_to_grow_past_512KB(tt_metal::Device *device, tt_metal::Program &program) {
    bool pass = true;

    auto logical_compute_and_storage_bank_ids = get_logical_compute_and_storage_core_bank_ids(device);
    TT_ASSERT(logical_compute_and_storage_bank_ids.size() == 1);
    auto compute_and_storage_bank_id = logical_compute_and_storage_bank_ids.at(0);
    auto logical_core = device->logical_core_from_bank_id(compute_and_storage_bank_id);

    uint32_t single_tile_size = 2 * 1024;
    constexpr uint32_t src0_cb_index = CB::c_in0;
    constexpr uint32_t num_input_tiles = 176;
    tt_metal::CircularBuffer *cb_src0 = tt_metal::CreateCircularBuffer(
        program,
        device,
        src0_cb_index,
        logical_core,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        tt::DataFormat::Float16_b
    );
    pass &= cb_src0->address() == (UNRESERVED_BASE + ((4 * 1024) * 3));

    return pass;
}

int main(int argc, char **argv) {
    bool pass = true;

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Grayskull Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int pci_express_slot = 0;
        tt_metal::Device *device =
            tt_metal::CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);

        tt_metal::Program program = tt_metal::Program();

        // NOTE: diagrams are NOT to scale
        // Running on compute and storage core: (0, 0) and storage core: (1, 9)
        // compute and storage core L1: one 1 MB bank (120 KB is reserved)
        // ----------------------------------------
        // | 120 KB |               904 KB        |
        // ----------------------------------------
        //
        // storage core L1: two 512 KB banks
        // ----------------------------------------
        // |        bank 0    |        bank 1     |
        // |   512 KB         |    512 KB         |
        // ----------------------------------------
        pass &= tt_metal::InitializeDevice(device, tt_metal::MemoryAllocator::L1_BANKING);

        // Resulting L1 banks after test_l1_buffers_allocated_top_down:
        // compute and storage core L1
        // ----------------------------------------------------------------
        // | 120 KB |                           | 256 KB | 64 KB | 128 KB |
        // ----------------------------------------------------------------
        //
        // storage core L1
        // --------------------------------------------------------------------
        // |               bank 0              |            bank 1            |
        // |        | 256 KB | 64 KB | 128 KB  |                     | 256 KB |
        // --------------------------------------------------------------------
        pass &= test_l1_buffers_allocated_top_down(device);

        // Resulting L1 banks after test_circular_buffers_allocated_bottom_up:
        // compute and storage core L1
        // -------------------------------------------------------------------------
        // | 120 KB | 4 KB | 4 KB | 4 KB |               | 256 KB | 64 KB | 128 KB |
        // -------------------------------------------------------------------------
        //
        // storage core L1
        // --------------------------------------------------------------------
        // |               bank 0              |            bank 1            |
        // |        | 256 KB | 64 KB | 128 KB  |                     | 256 KB |
        // --------------------------------------------------------------------
        pass &= test_circular_buffers_allocated_bottom_up(device, program);

        // tries to allocate a buffer larger than 512 KB - (256 + 64 + 128) KB in compute and storage core
        // this is expected to fail
        pass &= test_l1_buffer_do_not_grow_beyond_512KB(device);

        // Resulting L1 banks after test_circular_buffers_allowed_to_grow_past_512KB:
        // compute and storage core L1
        // -----------------------------------------------------------------------------
        // | 120 KB | 4 KB | 4 KB | 4 KB | 352 KB        |   | 256 KB | 64 KB | 128 KB |
        // -----------------------------------------------------------------------------
        //
        // storage core L1
        // --------------------------------------------------------------------
        // |               bank 0              |            bank 1            |
        // |        | 256 KB | 64 KB | 128 KB  |                     | 256 KB |
        // --------------------------------------------------------------------
        pass &= test_circular_buffers_allowed_to_grow_past_512KB(device, program);

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        pass &= tt_metal::CloseDevice(device);

    } catch (const std::exception &e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        log_fatal(LogTest, "Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}
