#pragma once

#include "fw_debug.h"

#include "noc_overlay_parameters.h"

struct stream_tile_info_t
{
    uint32_t base_address;
    TileHeader tile_header;
};
// Functions for accessing NOC overlay registers
namespace ckernel
{

typedef volatile uint32_t tt_reg_ptr *regp;

// Only perform the calculation once, as it's expensive to multiply numbers
inline regp get_stream_reg(uint32_t stream_id)
{
    constexpr uint32_t NOC_REGISTER_MMIO_BASE = 0xFFB40000;
    constexpr uint32_t PER_STREAM_REG_SIZE = 0x1000;
    return (regp) (NOC_REGISTER_MMIO_BASE + PER_STREAM_REG_SIZE * stream_id);
}

inline uint32_t get_stream_reg_addr(uint32_t stream_id, uint32_t index)
{
    constexpr uint32_t NOC_REGISTER_MMIO_BASE = 0xFFB40000;
    constexpr uint32_t PER_STREAM_REG_SIZE = 0x1000;
    return (NOC_REGISTER_MMIO_BASE + PER_STREAM_REG_SIZE * stream_id + (index << 2));
}

inline void write_stream_register(regp p_stream_reg, uint32_t index, uint32_t value)
{
    p_stream_reg[index] = value;
}

inline uint32_t read_stream_register(const regp p_stream_reg, uint32_t index)
{
    return p_stream_reg[index];
}

inline uint32_t read_stream_register_field(const regp p_stream_reg, uint32_t index, uint32_t shift, uint32_t width)
{
    return (read_stream_register(p_stream_reg, index) >> shift) & ((1 << width) - 1);
}

// Wait until stream has at least 'count' tiles ready
inline void wait_for_stream_messages(const regp p_stream_reg, const uint count)
{
    uint c = 0;
    do
    {
        c = read_stream_register(p_stream_reg, STREAM_NUM_MSGS_RECEIVED_REG_INDEX);
        FWLOG2("Waiting for %d stream_messages; Current messages: %d", count, c);
    } while (c < count);
}

inline void wait_for_N_stream_messages(const regp p_stream_reg, const uint num_messages) {
    
    uint c = 0;
    do {
        uint32_t msg_info_wr = read_stream_register(p_stream_reg, STREAM_MSG_INFO_WR_PTR_REG_INDEX);
        uint32_t msg_info = read_stream_register(p_stream_reg, STREAM_MSG_INFO_PTR_REG_INDEX);
        uint32_t num_msg = read_stream_register(p_stream_reg, STREAM_NUM_MSGS_RECEIVED_REG_INDEX);
        c = num_msg + (msg_info_wr - msg_info);
        // wait while we receive all the tiles from this stream
        FWLOG2("Waiting for %d stream_messages; Current messages: %d", num_messages, c);
    } while (c < num_messages);
}

inline void wait_for_stream_phase(const regp p_stream_reg, const uint phase_id)
{
    if (phase_id == 0)
    {
        FWLOG0("Warning: Skipping phase_id check!!");
        return;
    }
    uint p = 0;
    do
    {
        p = read_stream_register(p_stream_reg, STREAM_CURR_PHASE_REG_INDEX);
        FWLOG2("curr phase: %d, waiting for stream_phase: %d", p, phase_id);
    } while (p != phase_id);
}

inline void update_stream_read_pointer(regp p_stream_reg, const uint amount)
{
    write_stream_register(p_stream_reg, STREAM_MSG_INFO_CLEAR_REG_INDEX, amount);
}

inline uint read_stream_base_address(const regp p_stream_reg, const uint tile_n)
{
    return read_stream_register(p_stream_reg, STREAM_RECEIVER_MSG_INFO_REG_INDEX + tile_n * 6 + 0); //-> activations base address for tile n
}

inline uint read_stream_zero_mask(const regp p_stream_reg, const uint tile_n)
{
    return read_stream_register(p_stream_reg, STREAM_RECEIVER_MSG_INFO_REG_INDEX + tile_n * 6 + 4); //-> 32-bit zero mask
}

// Read tile info from a stream
inline stream_tile_info_t read_stream_info(const uint tile_index, const regp p_stream_reg)
{
    const uint n = tile_index;
    const uint base_address = read_stream_base_address(p_stream_reg, n);

    TileHeader_u header;

    FWLOG1("[0 ] activations base address: 0x%x", base_address);
    FWLOG1("[1 ] tile size w/o header    : 0x%x", read_stream_register(p_stream_reg, STREAM_RECEIVER_MSG_INFO_REG_INDEX + n * 6 + 1)); //-> message size (tile n size without header)
    header.val[0] = read_stream_register(p_stream_reg, STREAM_RECEIVER_MSG_INFO_REG_INDEX + n * 6 + 2); //-> tile n size including header and tile id (15:0 size, 31:16 tile id)
    FWLOG1("[2a] tile size with header   : 0x%x", (header.val[0] & 0xFFFF));
    FWASSERT("Tile size must be != 0", (header.val[0] & 0xFFFF) != 0);
    FWLOG1("[2b] tile id                 : 0x%x", header.val[0] >> 16);
    header.val[1] = read_stream_register(p_stream_reg, STREAM_RECEIVER_MSG_INFO_REG_INDEX + n * 6 + 3); //-> tile n meta data size and format
    FWLOG1("[3a] meta data size          : 0x%x", (header.val[1] & 0xFFFF));
    FWLOG1("[3b] data format             : 0x%x", ((header.val[1] >> 16) & 0xF));
    FWLOG1("[3c] uncompressed            : 0x%x", ((header.val[1] >> 20) & 0x1));
    header.val[2] = read_stream_zero_mask(p_stream_reg, n); //-> 32-bit zero mask
    FWLOG1("[4 ] zero mask               : 0x%x", header.val[2]);
    //read_stream_register(p_stream_reg, STREAM_RECEIVER_MSG_INFO_REG_INDEX + n * 6 + 5); //-> Reserved

    return stream_tile_info_t{base_address, header.header};
}

inline uint read_dis_zero_compress_group_info(const regp p_stream_reg)
{
    return read_stream_register(p_stream_reg, STREAM_MSG_GROUP_COMPRESS_REG_INDEX);
}

// Return the offset of a tile given it's tile id and table address
inline uint32_t get_indexed_offset(const uint tile_id, const uint weights_offset, const uint table_addr)
{
    //FWLOG1("Weight base address: 0x%x", params[2]);
    //FWLOG1("Weight base address: 0x%x", params[2] << 4);
    const uint16_t *weight_offset_table = reinterpret_cast<uint16_t *>(table_addr << 4);
    uint weight_offset = weight_offset_table[tile_id + weights_offset];
    FWLOG1("Weight table index: %d", tile_id + weights_offset);
    FWLOG1("Weight offset: 0x%x", weight_offset);
    return weight_offset;
}

inline void unpacker_config(const regp p_stream_reg, const uint unpacker_id, const uint fifo_size_factor = 1)
{
    uint fifo_base_addr = read_stream_register(p_stream_reg, STREAM_BUF_START_REG_INDEX);
    uint fifo_size = fifo_size_factor * read_stream_register(p_stream_reg, STREAM_BUF_SIZE_REG_INDEX);
    cfg_write(unpacker_id ? THCON_SEC1_REG2_Unpack_limit_address_ADDR32 : THCON_SEC0_REG2_Unpack_limit_address_ADDR32,
        (fifo_base_addr + fifo_size - 1) | (fifo_size << THCON_SEC0_REG2_Unpack_fifo_size_SHAMT));
}

// Optimized function that reads base addresses and programs registers for one context
inline void program_halo_strips_cntx0(
    volatile uint *cfg, const regp p_stream_reg, const uint first_active_tile, const uint unpack_halo_mask, uint *group_dis_zero_compress)
{
    //const uint strip1_addr = read_stream_base_address(p_stream_reg, 1);
    //const uint strip2_addr = read_stream_base_address(p_stream_reg, 2);
    //const uint strip3_addr = read_stream_base_address(p_stream_reg, 3);
    //cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = strip1_addr;
    //cfg[THCON_SEC0_REG3_Base_cntx2_address_ADDR32] = strip2_addr;
    //cfg[THCON_SEC0_REG3_Base_cntx3_address_ADDR32] = strip3_addr;
    uint dis_zero_compress_group_info = read_dis_zero_compress_group_info(p_stream_reg);       // Get uncompress flag for all 4 tiles
    uint dis_zero_compress_mask = ((dis_zero_compress_group_info & 0x1) << first_active_tile); // Get mask for first active tile

    uint index = 0;
    uint tile = 0;
    ;
    for (uint i = 1; i <= 3; i++)
    {
        if (i == first_active_tile)
            continue;

        if ((unpack_halo_mask >> i) & 0x1)
        {
            index++;
            tile++;
            const uint strip_addr = read_stream_base_address(p_stream_reg, index);
            switch (i)
            {
            case 1: cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = strip_addr; break;
            case 2: cfg[THCON_SEC0_REG3_Base_cntx2_address_ADDR32] = strip_addr; break;
            case 3: cfg[THCON_SEC0_REG3_Base_cntx3_address_ADDR32] = strip_addr; break;
            }
            dis_zero_compress_mask |= (((dis_zero_compress_group_info >> tile) & 0x1) << i);
        }
    }
    *group_dis_zero_compress &= (~(0xf)); // Clear 4 uncompress flags for context 0
    *group_dis_zero_compress |= dis_zero_compress_mask;
}

// Optimized function that reads base addresses and programs registers for one context
// FIXME: this is probably pretty slow.... need to evaluate, and maybe make a separate one for the 'common' case
// where the unpack halo mask is 0xF
inline void program_halo_strips_cntx1(
    volatile uint *cfg, const regp p_stream_reg, const uint first_active_tile, const uint unpack_halo_mask, uint *group_dis_zero_compress)
{
    uint dis_zero_compress_group_info = read_dis_zero_compress_group_info(p_stream_reg);       // Get uncompress flag for all 4 tiles
    uint dis_zero_compress_mask = ((dis_zero_compress_group_info & 0x1) << first_active_tile); // Get mask for first active tile

    uint index = 0;
    uint tile = 0;
    for (uint i = 1; i <= 3; i++)
    {
        if (i == first_active_tile)
            continue;

        if ((unpack_halo_mask >> i) & 0x1)
        {
            index++;
            tile++;
            const uint strip_addr = read_stream_base_address(p_stream_reg, index);
            switch (i)
            {
            case 1: cfg[THCON_SEC0_REG4_Base_cntx5_address_ADDR32] = strip_addr; break;
            case 2: cfg[THCON_SEC0_REG4_Base_cntx6_address_ADDR32] = strip_addr; break;
            case 3: cfg[THCON_SEC0_REG4_Base_cntx7_address_ADDR32] = strip_addr; break;
            }
            dis_zero_compress_mask |= (((dis_zero_compress_group_info >> tile) & 0x1) << i);
        }
    }
    *group_dis_zero_compress &= (~(0xf0000)); // Clear 4 uncompress flags for context 1
    *group_dis_zero_compress |= (dis_zero_compress_mask << 16);
}
} // namespace ckernel

namespace ckernel::stream
{
    // Only perform the calculation once, as it's expensive to multiply numbers
    inline regp get_reg(uint32_t stream_id)
    {
        return ckernel::get_stream_reg(stream_id);
    }

    inline void wait_for_phase(const regp stream_reg, const uint phase_id)
    {
        ckernel::wait_for_stream_phase(stream_reg, phase_id);
    }

    // Wait until stream has at least 'count' tiles ready
    inline void wait_for_messages(const regp stream_reg, const uint count)
    {
        uint c = 0;
        do
        {
            c = read_stream_register(stream_reg, STREAM_NUM_MSGS_RECEIVED_REG_INDEX);
            FWLOG2("Waiting for %d stream_messages; Current messages: %d", count, c);
        } while (c < count);
    }

    // Wait until stream has any messages ready
    template <bool FastPop = false>
    inline void wait_for_token(const regp stream_reg)
    {
        wait_for_messages(stream_reg, 1);

        if constexpr (FastPop) {
            write_stream_register(stream_reg, STREAM_MSG_INFO_CLEAR_REG_INDEX, 1);
        }
    }

    // Wait for a tile for streaming unpacker. Make sure to get address before updating pointer.
    inline uint32_t wait_for_tile(const regp stream_reg)
    {
        constexpr auto tile_count = 1;
        stream::wait_for_messages(stream_reg, tile_count);
        auto tile_l1_addr = read_stream_base_address(stream_reg, 0);
        update_stream_read_pointer(stream_reg, tile_count);
        return tile_l1_addr;
    }

    inline void pop_messages(const regp stream_reg, const uint count) {
        for (uint j = 0; j < count; j++) {
            // TODO: Change to do 2 or 4 (only for stream 4/5) pops at each instruction?
            uint32_t num_msgs = 1;
            // Wait for stream to load tiles into the msg info fifo so that we can pop them
            while (read_stream_register(stream_reg, STREAM_NUM_MSGS_RECEIVED_REG_INDEX) == 0) {}
            write_stream_register(stream_reg, STREAM_MSG_INFO_CLEAR_REG_INDEX, num_msgs);
            write_stream_register(stream_reg, STREAM_MSG_DATA_CLEAR_REG_INDEX, num_msgs);
        }
    }

    inline void release_token(const regp stream_reg)
    {
        write_stream_register(stream_reg, STREAM_MSG_INFO_CLEAR_REG_INDEX, 1);
        write_stream_register(stream_reg, STREAM_MSG_DATA_CLEAR_REG_INDEX, 1);
    }

    // Wait until specific stream register index contains specific value.
    inline void wait_for_reg_value(const regp p_stream_reg, const uint reg_index, const uint reg_value)
    {
        uint rd_value = reg_value - 1; // Initial non matching value
        do
        {
            rd_value = read_stream_register(p_stream_reg, reg_index);
        } while (rd_value != reg_value);
    }

    inline std::uint8_t* get_stream_buf_base_ptr(const regp stream_reg) {
      auto base_addr = read_stream_register(stream_reg, STREAM_BUF_START_REG_INDEX) << 4;
      return reinterpret_cast<std::uint8_t*>(base_addr);
    }
    
    inline std::uint8_t* get_stream_msg_info_wr_ptr(const regp stream_reg) {
      auto base_addr = read_stream_register(stream_reg, STREAM_MSG_INFO_WR_PTR_REG_INDEX) << 4;
      return reinterpret_cast<std::uint8_t*>(base_addr);
    }

    inline std::uint8_t* get_stream_buf_limit_ptr(const regp stream_reg) {
      auto base_addr = read_stream_register(stream_reg, STREAM_BUF_START_REG_INDEX) << 4;;
      auto size = read_stream_register(stream_reg, STREAM_BUF_SIZE_REG_INDEX) << 4;
      auto limit_addr = base_addr + size;
      return reinterpret_cast<std::uint8_t*>(limit_addr);
    }

    inline std::uint8_t* get_stream_msg_ptr(const regp stream_reg) {
      auto base_addr = read_stream_register(stream_reg, STREAM_BUF_START_REG_INDEX) << 4;
      auto rdptr = read_stream_register(stream_reg, STREAM_RD_PTR_REG_INDEX) << 4;
      FWLOG2("base_addr: %d, rdptr: %d", base_addr, rdptr);
      auto tile_addr = base_addr + rdptr;
      return reinterpret_cast<std::uint8_t*>(tile_addr);
    }

    inline std::uint8_t* get_stream_msg_wr_ptr(const regp stream_reg) {
      auto base_addr = read_stream_register(stream_reg, STREAM_BUF_START_REG_INDEX) << 4;
      auto wrptr = read_stream_register(stream_reg, STREAM_WR_PTR_REG_INDEX) << 4;
      auto tile_addr = base_addr + wrptr;
      return reinterpret_cast<std::uint8_t*>(tile_addr);
    }


} // namespace ckernel::stream
