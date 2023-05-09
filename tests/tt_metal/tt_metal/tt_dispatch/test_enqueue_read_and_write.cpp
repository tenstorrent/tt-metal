#include <algorithm>

#include "frameworks/tt_dispatch/impl/command_queue.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/tools/profiler/profiler.hpp"
// #include "llrt/tt_debug_print_server.hpp"

using namespace tt;

void zero_out_sysmem(Device *device) {
    // Prior to running anything, need to clear out system memory
    // to prevent anything being stale. Potentially make it a static
    // method on command queue
    vector<uint> zeros(1024 * 1024 * 1024 / sizeof(uint), 0);
    device->cluster()->write_sysmem_vec(zeros, 0, 0);
}

bool test_enqueue_write_buffer(BufferType buftype) {
    tt_metal::Device *device = tt_metal::CreateDevice(tt::ARCH::GRAYSKULL, 0);

    tt_metal::InitializeDevice(device, tt_metal::MemoryAllocator::L1_BANKING);

    bool pass = true;
    // Need to scope the following code since there is a lifetime issue where buffers created need to be destroyed prior
    // to deleting a device
    {
        int num_tiles = 500;
        Buffer bufa(device, 2048 * num_tiles, 0, 2048, buftype);

        vector<uint> src;
        for (uint i = 0; i < 512 * num_tiles; i++) {
            src.push_back(i);
        }

        // Prior to running anything, need to clear out system memory
        // to prevent anything being stale
        zero_out_sysmem(device);

        // tt_start_debug_print_server(device->cluster(), {0}, {{1, 11}});
        vector<uint> res;
        {
            CommandQueue cq(device);
            EnqueueWriteBuffer(cq, bufa, src, false);
        }

        ReadFromBuffer(bufa, res);
        pass = src == res;
        TT_ASSERT(pass);
    }
    delete device;

    return pass;
}

bool test_enqueue_read_buffer(BufferType buftype) {
    tt_metal::Device *device = tt_metal::CreateDevice(tt::ARCH::GRAYSKULL, 0);

    tt_metal::InitializeDevice(device, tt_metal::MemoryAllocator::L1_BANKING);
    bool pass = true;
    // Need to scope the following code since there is a lifetime issue where buffers created need to be destroyed prior
    // to deleting a device
    {
        int num_tiles = 1;

        Buffer bufa(device, 2048 * num_tiles, 0, 2048, buftype);

        // Prior to running anything, need to clear out system memory
        // to prevent anything being stale. Potentially make it a static
        // method on command queue
        zero_out_sysmem(device);

        vector<uint> src;
        for (uint i = 0; i < 512 * num_tiles; i++) {
            src.push_back(i);
        }

        vector<uint> res;
        WriteToBuffer(bufa, src);

        {
            CommandQueue cq(device);
            EnqueueReadBuffer(cq, bufa, res, true);
        }

        pass = src == res;
        TT_ASSERT(pass);
    }
    delete device;
    return pass;
}

bool test_enqueue_blocking_read_stress(BufferType buftype) {
    tt_metal::Device *device = tt_metal::CreateDevice(tt::ARCH::GRAYSKULL, 0);

    tt_metal::InitializeDevice(device, tt_metal::MemoryAllocator::L1_BANKING);
    zero_out_sysmem(device);

    bool pass = true;
    {
        u32 num_tiles_total = 2000;

        // Each buffer stores a single tile
        u32 k = 0;
        CommandQueue cq(device);
        for (int i = 0; i < num_tiles_total; i++) {
            vector<uint> src;
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

bool test_enqueue_write_stress(BufferType buftype) {
    tt_metal::Device *device = tt_metal::CreateDevice(tt::ARCH::GRAYSKULL, 0);

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
                vector<uint> src;
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

bool test_enqueue_write_and_read_pair_stress(BufferType buftype) {
    /*
        This test performs back to back writes and reads (hence pair)
    */

    bool pass = true;

    tt_metal::Device *device = tt_metal::CreateDevice(tt::ARCH::GRAYSKULL, 0);

    tt_metal::InitializeDevice(device, tt_metal::MemoryAllocator::L1_BANKING);
    zero_out_sysmem(device);

    {
        u32 num_tiles_total = 2000;

        // Each buffer stores a single tile
        u32 k = 0;
        CommandQueue cq(device);
        for (int i = 0; i < num_tiles_total; i++) {
            vector<uint> src;
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

bool test_chained_enqueue_writes_then_reads_stress() {
    /*
        This test performs back to back writes and reads (hence pair)
    */

    BufferType buftype;

    bool pass = true;

    tt_metal::Device *device = tt_metal::CreateDevice(tt::ARCH::GRAYSKULL, 0);

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

            vector<uint> src;
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
    int pci_express_slot = 0;

    // Most basic unit tests
    test_enqueue_write_buffer(BufferType::DRAM);
    test_enqueue_read_buffer(BufferType::DRAM);
    test_enqueue_write_buffer(BufferType::L1);
    test_enqueue_read_buffer(BufferType::L1);

    // Stress testing (just up to 1GB of system memory, no wrapping)
    test_enqueue_blocking_read_stress(BufferType::DRAM);
    test_enqueue_blocking_read_stress(BufferType::L1);

    test_enqueue_write_stress(BufferType::DRAM);
    test_enqueue_write_stress(BufferType::L1);

    test_enqueue_write_and_read_pair_stress(BufferType::DRAM);
    test_enqueue_write_and_read_pair_stress(BufferType::L1);

    test_chained_enqueue_writes_then_reads_stress();
}
