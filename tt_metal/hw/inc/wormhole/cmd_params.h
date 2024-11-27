// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <array>
#include <cassert>
#include <limits>
#include <cstring>
#ifndef DISABLE_CMD_DEBUG
#include <iostream>
#endif

#include "tensix.h"
#include "tensix_types.h"

// [[deprecated("There should be no more traditional fifos.")]]
inline std::uint32_t unpack_fifo_address(std::uint32_t fifo_address) {
    return (fifo_address << FIFO_BASE_ADDRESS_ALIGN_BITS);
}

inline std::uint32_t unpack_address(std::uint32_t address) { return (address << FIFO_BASE_ADDRESS_ALIGN_BITS); }

inline std::uint16_t pack_address(std::uint32_t address) {
#ifdef ASSERT
    ASSERT(
        !(address & bitmask<std::uint32_t>(FIFO_BASE_ADDRESS_ALIGN_BITS)), "Address not aligned and cannot be packed");
#else
    assert(
        !(address & bitmask<std::uint32_t>(FIFO_BASE_ADDRESS_ALIGN_BITS)) &&
        "Address not aligned and cannot be packed");
#endif
    return (address >> FIFO_BASE_ADDRESS_ALIGN_BITS);
}

inline std::uint32_t pack_32b_field(std::uint32_t x, unsigned int bits, unsigned int to_shift) {
    assert(bits + to_shift <= std::numeric_limits<std::uint32_t>::digits);
    assert((x & ~bitmask<std::uint32_t>(bits)) == 0);

    return x << to_shift;
}

inline std::uint32_t unpack_field(std::uint32_t x, unsigned int bits, unsigned int to_shift) {
    return ((x >> to_shift) & bitmask<std::uint32_t>(bits));
}

constexpr int MAX_NUM_PACKS = 4;

/*
  The following are TEST envionrment only legacy commands
  PRODCTION flag is used to exclude their compilation
*/

#ifndef PRODUCTION

#include "cmd_defs.h"

struct PackOperation {
    std::uint32_t config_blob[8];  // This is really a FirmwareCommand<8>

    std::uint8_t stream_ids[4];
    std::uint8_t y_start;
    std::uint8_t y_dim;
    std::uint8_t strip_y_dim;
    std::uint8_t strip_mask : 4;  // Which strips / stream IDs are valid
    bool skip_strip_setup : 1;    // Workaround for tb_tensix tests which don't use ExtendedMegaConfig
    bool force_max_xy : 1;
    bool strip_yz_transposed : 1;  // Only used for fused pack-pack_z top/bottom pack
    std::uint8_t unused : 1;

    std::uint32_t row_mask_select;
    std::uint32_t dest_start_offset;

    // last bit of each phase id (20-bits) is lplc (LocalProducer&LocalConsumer)
    std::uint32_t phase_ids[4];

    void print() const {
#ifndef DISABLE_CMD_DEBUG
        std::cout << "\tPackOperation:" << std::endl;
        std::cout << "\t\tconfig_blob=";
        for (unsigned i = 0; i < 8; i++) {
            std::cout << config_blob[i] << ", ";
        }
        std::cout << std::endl;

        std::cout << "\t\tstream_ids=";
        for (unsigned i = 0; i < 4; i++) {
            std::cout << (unsigned)stream_ids[i] << ", ";
        }
        std::cout << std::endl;

        std::cout << "\t\ty_start=" << (unsigned)y_start << std::endl;
        std::cout << "\t\ty_dim=" << (unsigned)y_dim << std::endl;
        std::cout << "\t\tstrip_y_dim=" << (unsigned)strip_y_dim << std::endl;
        std::cout << "\t\tstrip_mask=" << (unsigned)strip_mask << std::endl;
        std::cout << "\t\trow_mask_select=" << row_mask_select << std::endl;
        std::cout << "\t\tdest_start_offset=" << dest_start_offset << std::endl;

        std::cout << "\t\tphase_ids=";
        for (unsigned i = 0; i < 4; i++) {
            std::cout << (phase_ids[i] & 0x7FFFFFFF) << ", ";
        }
        std::cout << std::endl;

        std::cout << "\t\tlplc=";
        for (unsigned i = 0; i < 4; i++) {
            std::cout << (phase_ids[i] >> 31) << ", ";
        }
        std::cout << std::endl;

#endif
    }
};
static_assert(sizeof(PackOperation) % 16 == 0, "PackOperation must be packable into a 16-byte aligned array");

struct SfpuParams {
    std::uint16_t leaky_slope;
    std::uint32_t sfpu_scale_val;
    std::uint8_t data_format_src;
    std::uint8_t sfpu_rnd_fmt;
    bool fp32_acc;
    bool sfpu_rnd_unsigned_int;
    bool sfpu_stoch_rnd;
    bool sfpu_sort_rand_indx;
    bool sfpu_signed_int16;
    bool z_first_layout;
    bool sfpu_sort_col_major;
    std::uint32_t num_output_faces;
    bool sfpu_half_rows;
    std::uint32_t log_mult_const;
    std::uint32_t dropout_prob;
};

struct PackParams {
    PackOperation pack_ops[MAX_NUM_PACKS];

    bool output_tile_id_passthrough;
    bool unpack_to_dest;
    std::uint8_t num_packs;
    std::uint16_t kernel_id_packer;
    std::uint16_t kernel_id_sfunc;
    std::uint16_t output_tile_id_offset;
    std::uint16_t num_output_tiles;
    std::uint16_t tile_id_offset_by_packer;

    std::uint32_t bias_section_addr;
    std::uint16_t result32b;

    SfpuParams sfpu_params;

    bool pool_idx;
    bool pool_chk_data;
    std::uint32_t start_row;

    // PackParams() {
    //   std::memset(this, 0, sizeof(*this));
    // }

    void SetPackConfigBlob(int idx, std::array<std::uint32_t, 8> pack_config) {
        const auto size = sizeof(PackOperation::config_blob);
        std::memcpy(&pack_ops[idx].config_blob, pack_config.data(), size);
    }

    void parse(const std::uint32_t* command_data) {
        num_packs = unpack_field(command_data[0], 8, 16);
        kernel_id_packer = unpack_field(command_data[0], 16, 0);

        output_tile_id_offset = unpack_field(command_data[1], 16, 16);
        num_output_tiles = unpack_field(command_data[1], 16, 0);

        bias_section_addr = unpack_field(command_data[2], 16, 0);

        output_tile_id_passthrough = unpack_field(command_data[3], 1, 0);
        unpack_to_dest = unpack_field(command_data[3], 1, 1);
        result32b = unpack_field(command_data[3], 1, 2);
        tile_id_offset_by_packer = unpack_field(command_data[3], 16, 16);

        sfpu_params.leaky_slope = unpack_field(command_data[4], 16, 0);

        sfpu_params.sfpu_scale_val = unpack_field(command_data[5], 32, 0);

        sfpu_params.sfpu_half_rows = unpack_field(command_data[6], 1, 25);
        pool_chk_data = unpack_field(command_data[6], 1, 24);
        pool_idx = unpack_field(command_data[6], 1, 23);
        sfpu_params.sfpu_sort_col_major = unpack_field(command_data[6], 1, 22);
        sfpu_params.z_first_layout = unpack_field(command_data[6], 1, 21);
        sfpu_params.sfpu_signed_int16 = unpack_field(command_data[6], 1, 20);
        sfpu_params.sfpu_sort_rand_indx = unpack_field(command_data[6], 1, 19);
        sfpu_params.sfpu_stoch_rnd = unpack_field(command_data[6], 1, 18);
        sfpu_params.sfpu_rnd_unsigned_int = unpack_field(command_data[6], 1, 17);
        sfpu_params.fp32_acc = unpack_field(command_data[6], 1, 16);
        sfpu_params.sfpu_rnd_fmt = unpack_field(command_data[6], 8, 8);
        sfpu_params.data_format_src = unpack_field(command_data[6], 8, 0);

        sfpu_params.num_output_faces = unpack_field(command_data[7], 9, 0);
        sfpu_params.log_mult_const = unpack_field(command_data[8], 32, 0);
        sfpu_params.dropout_prob = unpack_field(command_data[9], 32, 0);
        start_row = unpack_field(command_data[10], 32, 0);
    }

    std::array<std::uint32_t, 96> create_pack_cmd() const {
        std::array<std::uint32_t, 96> cmd;
        cmd.fill(0);

        cmd[0] = pack_32b_field(CMD_PACK, 8, 24) | pack_32b_field(num_packs, 8, 16) |
                 pack_32b_field(kernel_id_packer, 16, 0);

        cmd[1] = pack_32b_field(output_tile_id_offset, 16, 16) | pack_32b_field(num_output_tiles, 16, 0);

        cmd[2] = bias_section_addr;

        cmd[3] = pack_32b_field(tile_id_offset_by_packer, 16, 16) |

                 pack_32b_field(unpack_to_dest, 1, 1) |

                 pack_32b_field(result32b, 1, 2) |

                 pack_32b_field(output_tile_id_passthrough, 1, 0);

        cmd[4] = pack_32b_field(sfpu_params.leaky_slope, 16, 0);

        cmd[5] = sfpu_params.sfpu_scale_val;

        cmd[6] =
            pack_32b_field(sfpu_params.sfpu_half_rows, 1, 25) | pack_32b_field(pool_chk_data, 1, 24) |
            pack_32b_field(pool_idx, 1, 23) | pack_32b_field(sfpu_params.sfpu_sort_col_major, 1, 22) |
            pack_32b_field(sfpu_params.z_first_layout, 1, 21) | pack_32b_field(sfpu_params.sfpu_signed_int16, 1, 20) |
            pack_32b_field(sfpu_params.sfpu_sort_rand_indx, 1, 19) | pack_32b_field(sfpu_params.sfpu_stoch_rnd, 1, 18) |
            pack_32b_field(sfpu_params.sfpu_rnd_unsigned_int, 1, 17) | pack_32b_field(sfpu_params.fp32_acc, 1, 16) |
            pack_32b_field(sfpu_params.sfpu_rnd_fmt, 8, 8) | pack_32b_field(sfpu_params.data_format_src, 8, 0);

        cmd[7] = pack_32b_field(sfpu_params.num_output_faces, 9, 0);
        cmd[8] = sfpu_params.log_mult_const;
        cmd[9] = sfpu_params.dropout_prob;
        cmd[10] = start_row;

        // cmd[31:8] - Reserved

        std::memcpy(cmd.data() + 32, pack_ops, sizeof(pack_ops));
        assert(sizeof(pack_ops) == MAX_NUM_PACKS * sizeof(PackOperation));
        assert(128 + sizeof(pack_ops) == cmd.size() * 4);

        return cmd;
    }

    void print() const {
#ifndef DISABLE_CMD_DEBUG
        std::cout << "PackParams:" << std::endl;
        for (unsigned i = 0; i < MAX_NUM_PACKS; i++) {
            pack_ops[i].print();
        }

        std::cout << "\toutput_tile_id_passthrough=" << output_tile_id_passthrough << std::endl;
        std::cout << "\tunpack_to_dest=" << unpack_to_dest << std::endl;
        std::cout << "\tnum_packs=" << (unsigned)num_packs << std::endl;
        std::cout << "\tkernel_id_packer=" << kernel_id_packer << std::endl;
        std::cout << "\toutput_tile_id_offset=" << output_tile_id_offset << std::endl;
        std::cout << "\tnum_output_tiles=" << num_output_tiles << std::endl;
        std::cout << "\tbias_section_addr=" << bias_section_addr << std::endl;
        std::cout << "\ttile_id_offset_by_packer=" << std::hex << tile_id_offset_by_packer << std::endl;
        std::cout << "\tresult32b=" << std::hex << result32b << std::endl;
        std::cout << "\tstart_row=" << start_row << std::endl;

        std::cout << "\tleaky_slope=" << std::hex << sfpu_params.leaky_slope << std::endl;

        std::cout << "\tsfpu_scale_val=" << std::hex << sfpu_params.sfpu_scale_val << std::endl;

        std::cout << "\tsfpu_params.sfpu_sort_col_major=" << sfpu_params.sfpu_sort_col_major << std::endl;
        std::cout << "\tsfpu_params.sfpu_sort_rand_indx=" << sfpu_params.sfpu_sort_rand_indx << std::endl;
        std::cout << "\tsfpu_params.sfpu_stoch_rnd=" << sfpu_params.sfpu_stoch_rnd << std::endl;
        std::cout << "\tsfpu_params.sfpu_rnd_unsigned_int=" << sfpu_params.sfpu_rnd_unsigned_int << std::endl;
        std::cout << "\tsfpu_params.fp32_acc=" << sfpu_params.fp32_acc << std::endl;
        std::cout << "\tsfpu_params.sfpu_half_rows" << sfpu_params.sfpu_half_rows << std::endl;
        std::cout << "\tsfpu_params.sfpu_rnd_fmt=" << std::hex << sfpu_params.sfpu_rnd_fmt << std::endl;
        std::cout << "\tsfpu_params.data_format_src=" << std::hex << sfpu_params.data_format_src << std::endl;
        std::cout << "\tsfpu_params.num_output_faces=" << std::hex << sfpu_params.num_output_faces << std::endl;
#endif
    }
};

struct StreamConvParams {
    std::uint16_t kernel_id_unpacker;
    std::uint16_t kernel_id_math;
    std::uint16_t kernel_id_packer;

    std::uint16_t input_stream_id;
    std::uint32_t input_phase_id;
    std::uint32_t weight_section_addr;
    std::uint32_t bias_section_addr;

    bool unpack_halo_strips[4];

    std::uint16_t unpack_weights_offset;
    std::uint16_t math_Z_dim_ratio_log2;

    std::uint16_t num_input_tiles;
    std::uint16_t num_output_tiles;
    std::uint16_t output_tile_id_offset;

    std::uint16_t math_fidelity;

    std::uint16_t y_start;
    std::uint16_t halo_dim;
    std::uint8_t halo_y_top = 0;
    std::uint8_t halo_y_bot = 0;
    std::uint8_t halo_y_offset = 0;

    // StreamConvParams() { std::memset(this, 0, sizeof(StreamConvParams));}

    void parse(std::uint32_t* command_data) {
        kernel_id_unpacker = unpack_field(command_data[0], 8, 16);
        kernel_id_math = unpack_field(command_data[0], 8, 8);
        kernel_id_packer = unpack_field(command_data[0], 8, 0);

        input_stream_id = unpack_field(command_data[1], 8, 24);

        unpack_halo_strips[3] = unpack_field(command_data[1], 1, 23);
        unpack_halo_strips[2] = unpack_field(command_data[1], 1, 22);
        unpack_halo_strips[1] = unpack_field(command_data[1], 1, 21);
        unpack_halo_strips[0] = unpack_field(command_data[1], 1, 20);

        math_fidelity = unpack_field(command_data[1], 4, 12);

        weight_section_addr = unpack_address(unpack_field(command_data[3], 16, 0));
        bias_section_addr = unpack_address(unpack_field(command_data[3], 16, 16));

        unpack_weights_offset = unpack_field(command_data[4], 16, 0);
        math_Z_dim_ratio_log2 = unpack_field(command_data[4], 16, 16);

        num_output_tiles = unpack_field(command_data[5], 16, 16);
        output_tile_id_offset = unpack_field(command_data[5], 16, 0);

        y_start = unpack_field(command_data[6], 16, 0);
        num_input_tiles = unpack_field(command_data[6], 16, 16);

        halo_y_top = unpack_field(command_data[6], 6, 0);
        halo_y_bot = unpack_field(command_data[6], 6, 6);
        halo_y_offset = unpack_field(command_data[6], 4, 12);

        halo_dim = unpack_field(command_data[7], 4, 20);
        input_phase_id = unpack_field(command_data[7], 20, 0);

        if (halo_y_bot == 0) {
            halo_y_top = 16 - y_start;
            halo_y_bot = 2 * halo_dim + y_start;
        }
    }

    void print() const {
#ifndef DISABLE_CMD_DEBUG
        std::cout << "StreamConvParams:" << std::endl;
        std::cout << "\tkernel_id_unpacker=" << kernel_id_unpacker << std::endl;
        std::cout << "\tkernel_id_math=" << kernel_id_math << std::endl;
        std::cout << "\tkernel_id_packer=" << kernel_id_packer << std::endl;

        std::cout << "\tinput_stream_id=" << input_stream_id << std::endl;
        std::cout << "\tweight_section_addr=" << weight_section_addr << std::endl;
        std::cout << "\tbias_section_addr=" << bias_section_addr << std::endl;

        std::cout << "\tunpack_halo_strips[0]=" << (unsigned)unpack_halo_strips[0] << std::endl;
        std::cout << "\tunpack_halo_strips[1]=" << (unsigned)unpack_halo_strips[1] << std::endl;
        std::cout << "\tunpack_halo_strips[2]=" << (unsigned)unpack_halo_strips[2] << std::endl;
        std::cout << "\tunpack_halo_strips[3]=" << (unsigned)unpack_halo_strips[3] << std::endl;

        std::cout << "\tunpack_weights_offset=" << unpack_weights_offset << std::endl;
        std::cout << "\tmath_Z_dim_ratio_log2=" << math_Z_dim_ratio_log2 << std::endl;

        std::cout << "\tnum_input_tiles=" << num_input_tiles << std::endl;
        std::cout << "\tnum_output_tiles=" << num_output_tiles << std::endl;
        std::cout << "\toutput_tile_id_offset=" << output_tile_id_offset << std::endl;

        std::cout << "\tmath_fidelity=" << math_fidelity << std::endl;

        std::cout << "\ty_start=" << y_start << std::endl;
        std::cout << "\thalo_y_top=" << halo_y_top << std::endl;
        std::cout << "\thalo_y_bot=" << halo_y_bot << std::endl;
        std::cout << "\thalo_y_offset=" << halo_y_offset << std::endl;
        std::cout << "\thalo_dim=" << halo_dim << std::endl;
#endif
    }

    std::uint32_t halo_y_spec() {
        if ((halo_y_top + halo_y_bot) > 0) {
            return pack_32b_field((uint32_t)halo_y_top, 6, 0) | pack_32b_field((uint32_t)halo_y_bot, 6, 6) |
                   pack_32b_field((uint32_t)halo_y_offset, 4, 12);
        }

        return pack_32b_field((uint32_t)y_start, 12, 0) | pack_32b_field((uint32_t)halo_y_offset, 4, 12);
    }

    std::array<std::uint32_t, 8> create_cmd_stream_conv() {
        std::array<std::uint32_t, 8> cmd;
        cmd.fill(0);

        cmd[0] = pack_32b_field(CMD_STREAM_CONV_2, 8, 24) | pack_32b_field(kernel_id_unpacker, 16, 0);

        // XXX: Unclear if casting the bool to uint32_t is the best way
        cmd[1] = pack_32b_field(input_stream_id, 8, 24) | pack_32b_field((uint32_t)unpack_halo_strips[3], 1, 23) |
                 pack_32b_field((uint32_t)unpack_halo_strips[2], 1, 22) |
                 pack_32b_field((uint32_t)unpack_halo_strips[1], 1, 21) |
                 pack_32b_field((uint32_t)unpack_halo_strips[0], 1, 20) | pack_32b_field(math_fidelity, 4, 12);

        cmd[2] = pack_32b_field(kernel_id_math, 16, 16) | pack_32b_field(kernel_id_packer, 16, 0);

        cmd[3] = pack_32b_field(pack_address(bias_section_addr), 16, 16) |
                 pack_32b_field(pack_address(weight_section_addr), 16, 0);

        cmd[4] = pack_32b_field(unpack_weights_offset, 16, 0) | pack_32b_field(math_Z_dim_ratio_log2, 16, 16);

        cmd[5] = pack_32b_field(num_output_tiles, 16, 16) | pack_32b_field(output_tile_id_offset, 16, 0);

        cmd[6] = pack_32b_field(num_input_tiles, 16, 16) | pack_32b_field(halo_y_spec(), 16, 0);

        cmd[7] = pack_32b_field(halo_dim, 4, 20) | pack_32b_field(input_phase_id, 20, 0);

        return cmd;
    }
};

struct UnaryOperationParams {
    // UnaryOperationParams() { std::memset(this, 0, sizeof(UnaryOperationParams));}
    std::uint32_t math_fidelity;
    std::uint32_t result32b;
    std::uint32_t repack_Z_dim_ratio_log2;

    std::uint32_t kernel_id_unpacker;
    std::uint32_t kernel_id_math;
    std::uint32_t kernel_id_packer;

    std::uint32_t num_activation_tiles;

    std::uint32_t input_A_stream_id;
    // std::uint32_t input_A_phase_id;

    std::uint32_t output_stream_id[4];
    // std::uint32_t output_phase_id[4];

    // TODO: The following fields are deprecated
    std::uint32_t unpacker_kernel_address;
    std::uint32_t math_kernel_address;
    std::uint32_t packer_kernel_address;

    std::uint32_t input_A_section_id;
    std::uint32_t input_A_fifo_address;

    std::uint32_t output_section_id;
    std::uint32_t output_fifo_address;

    void parse(std::uint32_t* command_data) {
        kernel_id_math = unpack_field(command_data[0], 16, 0);
        math_fidelity = unpack_field(command_data[0], 3, 16);
        result32b = unpack_field(command_data[0], 1, 19);
        repack_Z_dim_ratio_log2 = unpack_field(command_data[0], 3, 20);

        kernel_id_unpacker = unpack_field(command_data[1], 16, 0);
        kernel_id_packer = unpack_field(command_data[1], 16, 16);

        output_stream_id[0] = unpack_field(command_data[2], 8, 0);
        output_stream_id[1] = unpack_field(command_data[2], 8, 8);
        output_stream_id[2] = unpack_field(command_data[2], 8, 16);
        output_stream_id[3] = unpack_field(command_data[2], 8, 24);

        input_A_stream_id = unpack_field(command_data[3], 8, 24);
        num_activation_tiles = unpack_field(command_data[3], 16, 0);
    }
};

struct StreamUnaryBinaryCommonParams {
    // StreamUnaryBinaryCommonParams() { std::memset(this, 0, sizeof(StreamUnaryBinaryCommonParams));}
    std::uint32_t math_fidelity;
    std::uint32_t result32b = 0x0;
    std::uint32_t repack_Z_dim_ratio_log2;

    std::uint32_t kernel_id_unpacker;
    std::uint32_t kernel_id_math;
    std::uint32_t kernel_id_packer;

    std::uint32_t num_activation_tiles;

    std::uint32_t input_A_stream_id;
    std::uint32_t input_A_phase_id;

    std::uint32_t realign_start_y;
    std::uint32_t realign_strip_mask;
    std::uint8_t realign_y_top = 0;
    std::uint8_t realign_y_bot = 0;
    std::uint8_t realign_y_offset = 0;

    bool tile_id_passthrough = false;

    // Used for all math kernels that concats in Z whenever there is multiple
    // tiles unpacked per math iteration.
    std::uint16_t math_Z_dim_ratio_log2;

public:
    std::uint32_t realign_y_spec() {
        if ((realign_y_top + realign_y_bot) > 0) {
            return pack_32b_field((uint32_t)realign_y_top, 6, 0) | pack_32b_field((uint32_t)realign_y_bot, 6, 6) |
                   pack_32b_field((uint32_t)realign_y_offset, 4, 12);
        }

        return pack_32b_field((uint32_t)realign_start_y, 12, 0) | pack_32b_field((uint32_t)realign_y_offset, 4, 12);
    }

protected:
    void parse(std::uint32_t* command_data) {
        kernel_id_math = unpack_field(command_data[0], 16, 0);
        math_fidelity = unpack_field(command_data[0], 3, 16);
        result32b = unpack_field(command_data[0], 1, 19);
        repack_Z_dim_ratio_log2 = unpack_field(command_data[0], 3, 20);

        kernel_id_unpacker = unpack_field(command_data[1], 16, 0);
        kernel_id_packer = unpack_field(command_data[1], 16, 16);

        input_A_stream_id = unpack_field(command_data[2], 8, 24);
        num_activation_tiles = unpack_field(command_data[2], 16, 0);

        realign_start_y = unpack_field(command_data[3], 4, 28);
        realign_strip_mask = unpack_field(command_data[3], 4, 24);

        realign_y_top = unpack_field(command_data[3], 6, 0);
        realign_y_bot = unpack_field(command_data[3], 6, 6);
        realign_y_offset = unpack_field(command_data[3], 4, 12);

        tile_id_passthrough = unpack_field(command_data[3], 1, 20);

        input_A_phase_id = unpack_field(command_data[4], 16, 0);
        math_Z_dim_ratio_log2 = unpack_field(command_data[4], 16, 16);
    }

    std::array<std::uint32_t, 8> create_cmd_stream_unary_binary_common() {
        std::array<std::uint32_t, 8> cmd;
        cmd.fill(0);

        cmd[0] = pack_32b_field(kernel_id_math, 16, 0) | pack_32b_field(math_fidelity, 3, 16) |
                 pack_32b_field(result32b, 1, 19) | pack_32b_field(repack_Z_dim_ratio_log2, 3, 20) |
                 pack_32b_field(CMD_STREAM_UNARY_OPERATION, 8, 24);

        cmd[1] = pack_32b_field(kernel_id_unpacker, 16, 0) | pack_32b_field(kernel_id_packer, 16, 16);

        cmd[2] = pack_32b_field(input_A_stream_id, 8, 24) | pack_32b_field(num_activation_tiles, 16, 0);

        cmd[3] = pack_32b_field(realign_y_spec(), 16, 0) | pack_32b_field(realign_start_y, 4, 28) |
                 pack_32b_field(realign_strip_mask, 4, 24) | pack_32b_field(tile_id_passthrough, 1, 20);

        cmd[4] = pack_32b_field(input_A_phase_id, 16, 0) | pack_32b_field(math_Z_dim_ratio_log2, 16, 16);
        // cmd[5-8] will be used by StreamBinaryParams or StreamUnaryParams

        return cmd;
    };
};

struct StreamUnaryParams : StreamUnaryBinaryCommonParams {
    std::uint32_t math_kernel_parameter = 0;

    void parse(std::uint32_t* command_data) {
        StreamUnaryBinaryCommonParams::parse(command_data);
        math_kernel_parameter = command_data[5];
    }

    std::array<std::uint32_t, 8> create_cmd_stream_unary() {
        std::array<std::uint32_t, 8> cmd = create_cmd_stream_unary_binary_common();
        cmd[5] = math_kernel_parameter;
        return cmd;
    }
};

struct StreamBinaryParams : StreamUnaryBinaryCommonParams {
    // StreamBinaryParams() { std::memset(this, 0, sizeof(StreamBinaryParams)); }

    std::uint32_t input_B_stream_id;
    std::uint32_t input_B_phase_id;
    std::uint32_t scaler_section_addr = 0;
    std::uint32_t scaler_config = 0;
    std::uint32_t math_kernel_parameter = 0;
    std::uint32_t sfpu_scale_val;
    std::uint8_t sfpu_rnd_fmt;
    bool sfpu_rnd_unsigned_int;
    bool sfpu_stoch_rnd;
    bool sfpu_sort_rand_indx;
    bool sfpu_signed_int16;
    bool sfpu_sort_col_major;
    std::uint32_t log_mult_const;
    std::uint32_t dropout_prob;

    void parse(std::uint32_t* command_data) {
        StreamUnaryBinaryCommonParams::parse(command_data);
        input_B_stream_id = unpack_field(command_data[5], 8, 0);
        scaler_config = unpack_field(command_data[5], 24, 8);
        input_B_phase_id = unpack_field(command_data[6], 16, 0);
        scaler_section_addr = unpack_address(unpack_field(command_data[6], 16, 16));
        math_kernel_parameter = command_data[7];
    }

    std::array<std::uint32_t, 16> create_cmd_stream_binary() {
        std::array<std::uint32_t, 16> cmd;
        std::array<std::uint32_t, 8> cmd_unary_binary_common = create_cmd_stream_unary_binary_common();
        for (int i = 0; i < 8; i++) {
            cmd[i] = cmd_unary_binary_common[i];
        }

        cmd[0] &= ~(pack_32b_field(0xFF, 8, 24));
        cmd[0] |= pack_32b_field(CMD_STREAM_BINARY_OPERATION, 8, 24);

        cmd[5] = pack_32b_field(input_B_stream_id, 8, 0) | pack_32b_field(scaler_config, 24, 8);
        cmd[6] = pack_32b_field(input_B_phase_id, 16, 0) | pack_32b_field(pack_address(scaler_section_addr), 16, 16);
        cmd[7] = math_kernel_parameter;
        cmd[8] = sfpu_scale_val;
        cmd[9] = pack_32b_field(sfpu_sort_col_major, 1, 12) | pack_32b_field(sfpu_signed_int16, 1, 11) |
                 pack_32b_field(sfpu_sort_rand_indx, 1, 10) | pack_32b_field(sfpu_stoch_rnd, 1, 9) |
                 pack_32b_field(sfpu_rnd_unsigned_int, 1, 8) | pack_32b_field(sfpu_rnd_fmt, 8, 0);
        cmd[10] = log_mult_const;
        cmd[11] = dropout_prob;

        return cmd;
    }
};

struct BinaryOperationParams : UnaryOperationParams {
    // BinaryOperationParams() { std::memset(this, 0, sizeof(BinaryOperationParams)); }

    std::uint32_t math_fidelity;
    std::uint32_t repack_Z_dim_ratio_log2;

    std::uint32_t kernel_id_unpacker;
    std::uint32_t kernel_id_math;
    std::uint32_t kernel_id_packer;

    std::uint32_t num_activation_tiles;

    std::uint32_t input_A_stream_id;

    std::uint32_t output_stream_id[4];

    std::uint32_t input_B_stream_id;

    void parse(std::uint32_t* command_data) {
        kernel_id_math = unpack_field(command_data[0], 16, 0);
        math_fidelity = unpack_field(command_data[0], 4, 16);
        repack_Z_dim_ratio_log2 = unpack_field(command_data[0], 3, 20);

        kernel_id_unpacker = unpack_field(command_data[1], 16, 0);
        kernel_id_packer = unpack_field(command_data[1], 16, 16);

        output_stream_id[0] = unpack_field(command_data[2], 8, 0);
        output_stream_id[1] = unpack_field(command_data[2], 8, 8);
        output_stream_id[2] = unpack_field(command_data[2], 8, 16);
        output_stream_id[3] = unpack_field(command_data[2], 8, 24);

        input_A_stream_id = unpack_field(command_data[3], 8, 24);
        input_B_stream_id = unpack_field(command_data[3], 8, 16);
        num_activation_tiles = unpack_field(command_data[3], 16, 0);
    }

    std::array<std::uint32_t, 4> create_cmd_stream_binary() {
        std::array<std::uint32_t, 4> cmd;
        cmd.fill(0);

        cmd[0] = pack_32b_field(kernel_id_math, 16, 0) | pack_32b_field(math_fidelity, 4, 16) |
                 pack_32b_field(repack_Z_dim_ratio_log2, 3, 20) | pack_32b_field(CMD_STREAM_BINARY_OPERATION, 8, 24);

        cmd[1] = pack_32b_field(kernel_id_unpacker, 16, 0) | pack_32b_field(kernel_id_packer, 16, 16);

        cmd[2] = pack_32b_field(output_stream_id[0], 8, 0) | pack_32b_field(output_stream_id[1], 8, 8) |
                 pack_32b_field(output_stream_id[2], 8, 16) | pack_32b_field(output_stream_id[3], 8, 24);

        cmd[3] = pack_32b_field(input_A_stream_id, 8, 24) | pack_32b_field(input_B_stream_id, 8, 16) |
                 pack_32b_field(num_activation_tiles, 16, 0);

        return cmd;
    }
};

struct StreamPoolParams {
    // StreamPoolParams() { std::memset(this, 0, sizeof(StreamPoolParams)); }
    std::uint32_t kernel_id_unpacker;
    std::uint32_t kernel_id_math;
    std::uint32_t kernel_id_packer;

    std::uint16_t math_fidelity;
    std::uint16_t result32b;
    std::uint32_t repack_Z_dim_ratio_log2;
    std::uint16_t neginf_srca;
    std::uint16_t avg_pool_coeff;
    std::uint8_t int8_mode;

    std::uint32_t input_A_stream_id;
    std::uint32_t input_A_phase_id;

    std::uint32_t num_activation_tiles;

    bool unpack_halo_strips[4];

    std::uint32_t input_B_l1_addr;

    std::uint16_t y_start;
    std::uint16_t halo_dim;
    std::uint8_t halo_y_top = 0;
    std::uint8_t halo_y_bot = 0;
    std::uint8_t halo_y_offset = 0;

    void parse(std::uint32_t* command_data) {
        kernel_id_math = unpack_field(command_data[0], 16, 0);
        math_fidelity = unpack_field(command_data[0], 4, 16);
        result32b = unpack_field(command_data[0], 1, 19);
        repack_Z_dim_ratio_log2 = unpack_field(command_data[0], 3, 20);
        neginf_srca = unpack_field(command_data[0], 1, 23);

        kernel_id_unpacker = unpack_field(command_data[1], 16, 0);
        kernel_id_packer = unpack_field(command_data[1], 16, 16);

        input_A_stream_id = unpack_field(command_data[2], 16, 0);
        num_activation_tiles = unpack_field(command_data[2], 16, 16);

        unpack_halo_strips[3] = unpack_field(command_data[3], 1, 23);
        unpack_halo_strips[2] = unpack_field(command_data[3], 1, 22);
        unpack_halo_strips[1] = unpack_field(command_data[3], 1, 21);
        unpack_halo_strips[0] = unpack_field(command_data[3], 1, 20);
        halo_dim = unpack_field(command_data[3], 4, 28);

        input_B_l1_addr = unpack_address(unpack_field(command_data[3], 16, 0));
        input_A_phase_id = command_data[4];

        y_start = unpack_field(command_data[5], 16, 0);
        halo_y_top = unpack_field(command_data[5], 6, 0);
        halo_y_bot = unpack_field(command_data[5], 6, 6);
        halo_y_offset = unpack_field(command_data[5], 4, 12);

        avg_pool_coeff = unpack_field(command_data[6], 16, 0);
        int8_mode = unpack_field(command_data[6], 1, 16);

        if (halo_y_bot == 0) {
            halo_y_top = 16 - y_start;
            halo_y_bot = 2 * halo_dim + y_start;
        }
    }

    std::uint32_t halo_y_spec() {
        if ((halo_y_top + halo_y_bot) > 0) {
            return pack_32b_field((uint32_t)halo_y_top, 6, 0) | pack_32b_field((uint32_t)halo_y_bot, 6, 6) |
                   pack_32b_field((uint32_t)halo_y_offset, 4, 12);
        }

        return pack_32b_field((uint32_t)y_start, 12, 0) | pack_32b_field((uint32_t)halo_y_offset, 4, 12);
    }

    std::array<std::uint32_t, 8> create_cmd_stream_pool() {
        std::array<std::uint32_t, 8> cmd;
        cmd.fill(0);

        cmd[0] = pack_32b_field(kernel_id_math, 16, 0) | pack_32b_field(math_fidelity, 3, 16) |
                 pack_32b_field(result32b, 1, 19) | pack_32b_field(repack_Z_dim_ratio_log2, 3, 20) |
                 pack_32b_field(neginf_srca, 1, 23) | pack_32b_field(CMD_STREAM_POOL, 8, 24);

        cmd[1] = pack_32b_field(kernel_id_unpacker, 16, 0) | pack_32b_field(kernel_id_packer, 16, 16);

        cmd[2] = pack_32b_field(input_A_stream_id, 16, 0) | pack_32b_field(num_activation_tiles, 16, 16);

        cmd[3] = pack_32b_field(halo_dim, 4, 28) | pack_32b_field((uint32_t)unpack_halo_strips[3], 1, 23) |
                 pack_32b_field((uint32_t)unpack_halo_strips[2], 1, 22) |
                 pack_32b_field((uint32_t)unpack_halo_strips[1], 1, 21) |
                 pack_32b_field((uint32_t)unpack_halo_strips[0], 1, 20) |
                 pack_32b_field(pack_address(input_B_l1_addr), 16, 0);

        cmd[4] = pack_32b_field(input_A_phase_id, 32, 0);

        cmd[5] = pack_32b_field(halo_y_spec(), 16, 0);

        cmd[6] = pack_32b_field(avg_pool_coeff, 16, 0) | pack_32b_field(int8_mode, 1, 16);

        return cmd;
    }
};

struct BinaryOperationParams_added {
    std::uint32_t input_B_fifo_address;
    std::uint16_t int_avg_pool_coeff;
    bool int8_mode;
};

struct TernaryOperationParams_added {
    std::uint32_t input_B_section_id;
    std::uint32_t input_C_section_id;  // Used for Bias, for example
    std::uint32_t input_C_fifo_address;
};

struct TernaryOperationWithBiasParams_added {
    std::uint32_t bias_kernel_address;
};

struct QuinaryOperationParams_added {
    std::uint32_t input_D_section_id;
    std::uint32_t input_D_fifo_address;

    std::uint32_t input_E_section_id;
    std::uint32_t input_E_fifo_address;
};

struct SliceZParams_added {
    std::uint32_t start_index;
    std::uint32_t length;
    std::uint32_t output_size_16B;
};

struct StreamFullConnParams {
    std::uint32_t kernel_id_unpacker;
    std::uint32_t kernel_id_math;
    std::uint32_t kernel_id_packer;

    std::uint8_t activation_stream_id;
    std::uint32_t activation_phase_id;
    std::uint32_t weight_l1_address;
    std::uint32_t weight_offset;

    std::uint8_t iterations;
    std::uint32_t batch_size;

    void parse(const std::uint32_t* command_data) {
        kernel_id_math = unpack_field(command_data[0], 16, 0);
        kernel_id_unpacker = unpack_field(command_data[1], 16, 0);
        kernel_id_packer = unpack_field(command_data[1], 16, 16);

        weight_l1_address = unpack_address(unpack_field(command_data[2], 16, 0));
        activation_stream_id = unpack_field(command_data[2], 8, 16);
        activation_phase_id = command_data[3];

        weight_offset = command_data[4];

        iterations = unpack_field(command_data[0], 8, 16);
        batch_size = unpack_field(command_data[2], 8, 24);
    }

    std::array<std::uint32_t, 8> create_cmd_stream_fullconn() {
        std::array<std::uint32_t, 8> cmd;
        cmd.fill(0);

        cmd[0] = pack_32b_field(kernel_id_math, 16, 0) | pack_32b_field(iterations, 8, 16) |
                 pack_32b_field(CMD_STREAM_FULLCONN, 8, 24);

        cmd[1] = pack_32b_field(kernel_id_unpacker, 16, 0) | pack_32b_field(kernel_id_packer, 16, 16);

        cmd[2] = pack_32b_field(pack_address(weight_l1_address), 16, 0) | pack_32b_field(activation_stream_id, 8, 16) |
                 pack_32b_field(batch_size, 8, 24);

        cmd[3] = pack_32b_field(activation_phase_id, 32, 0);

        cmd[4] = pack_32b_field(weight_offset, 32, 0);

        return cmd;
    }
};

#endif
