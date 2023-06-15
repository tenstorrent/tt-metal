#include "tt_metal/host_api.hpp"

using namespace tt;

void zero_out_sysmem(Device *device) {
    // Prior to running anything, need to clear out system memory
    // to prevent anything being stale. Potentially make it a static
    // method on command queue
    vector<u32> zeros(1024 * 1024 * 1024 / sizeof(u32), 0);
    device->cluster()->write_sysmem_vec(zeros, 0, 0);
}

bool test_enqueue_write_buffer(const ARCH& arch, BufferType buftype) {
    tt_metal::Device *device = tt_metal::CreateDevice(arch, 0);

    tt_metal::InitializeDevice(device, tt_metal::MemoryAllocator::L1_BANKING);

    bool pass = true;
    // Need to scope the following code since there is a lifetime issue where buffers created need to be destroyed prior
    // to deleting a device
    vector<u32> src;
    vector<u32> res;
    {
        int num_tiles = 500;
        Buffer bufa(device, 2048 * num_tiles, 0, 2048, buftype);

        for (u32 i = 0; i < 512 * num_tiles; i++) {
            src.push_back(i);
        }

        // Prior to running anything, need to clear out system memory
        // to prevent anything being stale
        zero_out_sysmem(device);

        // tt_start_debug_print_server(device->cluster(), {0}, {{1, 11}});

        {
            CommandQueue cq(device);
            EnqueueWriteBuffer(cq, bufa, src, false);
            // EnqueueReadBuffer(cq, bufa, res, true);
        }
        ReadFromBuffer(bufa, res);

        pass = src == res;

    }

    // for (int i = 0; i < src.size(); i++) {
    //     if (res.at(i) != 0) {

    //         std::cout << "i: " << i << ", " << src.at(i) << ", " << res.at(i) << std::endl;
    //     }
    // }


    TT_ASSERT(pass);
    delete device;

    return pass;
}

bool test_enqueue_read_buffer(const ARCH& arch, BufferType buftype) {
    tt_metal::Device *device = tt_metal::CreateDevice(arch , 0);

    tt_metal::InitializeDevice(device, tt_metal::MemoryAllocator::L1_BANKING);
    bool pass = true;
    // Need to scope the following code since there is a lifetime issue where buffers created need to be destroyed prior
    // to deleting a device
    {
        int num_tiles = 500;

        Buffer bufa(device, 2048 * num_tiles, 0, 2048, buftype);

        // Prior to running anything, need to clear out system memory
        // to prevent anything being stale. Potentially make it a static
        // method on command queue
        zero_out_sysmem(device);

        vector<u32> src;
        for (u32 i = 0; i < 512 * num_tiles; i++) {
            src.push_back(i);
        }

        vector<u32> res;
        WriteToBuffer(bufa, src);

        {
            CommandQueue cq(device);
            EnqueueReadBuffer(cq, bufa, res, true);
        }

        // for (int i = 0; i < src.size(); i++) {
        //     std::cout << "I: " << i << ", " << src.at(i) << ", " << res.at(i) << std::endl;
        // }

        pass = src == res;
        TT_ASSERT(pass);
    }
    delete device;
    return pass;
}

bool test_enqueue_blocking_read_stress(const ARCH& arch, BufferType buftype) {
    tt_metal::Device *device = tt_metal::CreateDevice(arch, 0);

    tt_metal::InitializeDevice(device, tt_metal::MemoryAllocator::L1_BANKING);
    zero_out_sysmem(device);

    bool pass = true;
    {
        u32 num_tiles_total = 2000;

        // Each buffer stores a single tile
        u32 k = 0;
        CommandQueue cq(device);
        for (int i = 0; i < num_tiles_total; i++) {
            vector<u32> src;
            for (u32 j = 0; j < 512; j++) {
                src.push_back(k);
                k++;
            }
            Buffer buf(device, 2048, 0, 2048, buftype);
            WriteToBuffer(buf, src);

            vector<u32> res;
            EnqueueReadBuffer(cq, buf, res, true);
            pass &= src == res;
        }
    }
    TT_ASSERT(pass);
    delete device;
    return pass;
}

bool test_enqueue_write_stress(const ARCH& arch, BufferType buftype) {
    tt_metal::Device *device = tt_metal::CreateDevice(arch, 0);

    tt_metal::InitializeDevice(device, tt_metal::MemoryAllocator::L1_BANKING);
    zero_out_sysmem(device);

    bool pass = true;
    u32 num_tiles_total = 256 * 10;
    u32 num_pages = (buftype == BufferType::DRAM) ? 16 : 256;
    {  // This scope exists so that buffers are destroyed prior to device
        vector<Buffer> buffers;
        vector<vector<u32>> srcs;
        {  // This scope exists for the command queue destructor to auto-call finish

            // Each buffer stores a single tile
            u32 k = 0;
            CommandQueue cq(device);
            for (int i = 0; i < num_tiles_total / num_pages; i++) {
                buffers.push_back(Buffer(device, 2048 * num_pages, 0, 2048, buftype));
            }
            for (int i = 0; i < num_tiles_total / num_pages; i++) {
                vector<u32> src;
                for (u32 j = 0; j < 512 * num_pages; j++) {
                    src.push_back(k);
                    k++;
                }
                srcs.push_back(src);
                EnqueueWriteBuffer(cq, buffers.at(i), src, false);
            }
        }
        for (int i = 0; i < num_tiles_total / num_pages; i++) {
            vector<u32> res;
            ReadFromBuffer(buffers.at(i), res);
            pass &= res == srcs.at(i);
        }
        std::cout << "Done reading" << std::endl;
    }
    TT_ASSERT(pass);
    delete device;
    return pass;
}

bool test_enqueue_write_and_read_pair_stress(const ARCH& arch, BufferType buftype) {
    /*
        This test performs back to back writes and reads (hence pair)
    */

    bool pass = true;

    tt_metal::Device *device = tt_metal::CreateDevice(arch, 0);

    tt_metal::InitializeDevice(device, tt_metal::MemoryAllocator::L1_BANKING);
    zero_out_sysmem(device);

    {
        u32 num_tiles_total = 2000;

        // Each buffer stores a single tile
        u32 k = 0;
        CommandQueue cq(device);
        for (int i = 0; i < num_tiles_total; i++) {
            vector<u32> src;
            for (u32 j = 0; j < 512; j++) {
                src.push_back(k);
                k++;
            }
            Buffer buf(device, 2048, 0, 2048, buftype);
            EnqueueWriteBuffer(cq, buf, src, false);

            vector<u32> res;
            EnqueueReadBuffer(cq, buf, res, true);
            pass &= src == res;
        }
    }
    TT_ASSERT(pass);
    delete device;
    return pass;

    return pass;
}

bool test_chained_enqueue_writes_then_reads_stress(const ARCH& arch) {
    /*
        This test performs back to back writes and reads (hence pair)
    */

    BufferType buftype;

    bool pass = true;

    tt_metal::Device *device = tt_metal::CreateDevice(arch, 0);

    tt_metal::InitializeDevice(device, tt_metal::MemoryAllocator::L1_BANKING);

    u32 num_tiles_left = 30000;
    {
        // Each buffer stores a single tile
        CommandQueue cq(device);
        vector<vector<u32>> srcs;
        vector<unique_ptr<Buffer>> buffers;

        while (num_tiles_left > 0) {
            u32 num_tiles_in_buffer;
            if ((rand() % 2) == 0) {
                buftype = BufferType::DRAM;
                num_tiles_in_buffer = std::min(int(num_tiles_left), 16);

            } else {
                buftype = BufferType::L1;
                num_tiles_in_buffer = std::min(int(num_tiles_left), 256);
            }

            vector<u32> src;
            for (u32 j = 0; j < 512 * num_tiles_in_buffer; j++) {
                src.push_back(j);
            }
            srcs.push_back(src);

            buffers.push_back(std::make_unique<Buffer>(device, num_tiles_in_buffer * 2048, 0, 2048, buftype));
            EnqueueWriteBuffer(cq, *buffers.at(buffers.size() - 1), src, false);

            num_tiles_left -= num_tiles_in_buffer;
        }

        Finish(cq);

        for (int i = 0; i < buffers.size(); i++) {
            vector<u32> res;
            EnqueueReadBuffer(cq, *buffers.at(i), res, true);
            pass &= srcs.at(i) == res;
        }
    }
    TT_ASSERT(pass);
    delete device;
    return pass;
}

int main(int argc, char **argv) {

    ////////////////////////////////////////////////////////////////////////////
    //                      Initial Runtime Args Parse
    ////////////////////////////////////////////////////////////////////////////
    std::vector<std::string> input_args(argv, argv + argc);
    string arch_name = "";
    try {
        std::tie(arch_name, input_args) =
            test_args::get_command_option_and_remaining_args(input_args, "--arch", "grayskull");
    } catch (const std::exception& e) {
        log_fatal(tt::LogTest, "Command line arguments found exception", e.what());
    }
    const tt::ARCH arch = tt::get_arch_from_string(arch_name);
    // Most basic unit tests
    test_enqueue_write_buffer(arch, BufferType::DRAM);
    test_enqueue_read_buffer(arch, BufferType::DRAM);
    test_enqueue_write_buffer(arch, BufferType::L1);
    test_enqueue_read_buffer(arch, BufferType::L1);

    // Stress testing (just up to 1GB of system memory, no wrapping)
    test_enqueue_blocking_read_stress(arch, BufferType::DRAM);
    test_enqueue_blocking_read_stress(arch, BufferType::L1);

    test_enqueue_write_stress(arch, BufferType::DRAM);
    test_enqueue_write_stress(arch, BufferType::L1);

    test_enqueue_write_and_read_pair_stress(arch, BufferType::DRAM);
    test_enqueue_write_and_read_pair_stress(arch, BufferType::L1);

    test_chained_enqueue_writes_then_reads_stress(arch);
}
