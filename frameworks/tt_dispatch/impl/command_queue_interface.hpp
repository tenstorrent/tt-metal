#include "frameworks/tt_dispatch/impl/command.hpp"
#include "tt_metal/common/base.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/llrt/llrt.hpp"
#include "tt_metal/src/firmware/riscv/grayskull/noc/noc_overlay_parameters.h"
#include "tt_metal/src/firmware/riscv/grayskull/stream_io_map.h"

using namespace tt::tt_metal;

struct SystemMemoryCQWriteInterface {
    // Equation for fifo size is
    // | fifo_wr_ptr + command size B - fifo_rd_ptr |
    // Space available would just be fifo_limit - fifo_size
    u32 fifo_size = 0;
    u32 fifo_wr_ptr = 0;
    const u32 fifo_limit = 1024 * 1024 * 1024 - 1;  // Last possible FIFO address
};

u32 get_cq_rd_ptr(Device* device);

class SystemMemoryWriter {
   public:
    SystemMemoryCQWriteInterface cq_write_interface;
    SystemMemoryWriter();

    void cq_reserve_back(Device* device, u32 cmd_size_B);

    // Ideally, data should be an array or pointer, but vector for time-being
    void cq_write(Device* device, vector<uint>& data, uint write_ptr);

    void send_write_ptr(Device* device);

    void cq_push_back(Device* device, uint push_size_B);
};
