#pragma once
#include <random>
#include <algorithm>
#include <functional>
#include <tuple>
#include <iostream>
#include <filesystem>

#include "llrt/tt_cluster.hpp"
#include "tensix.h"
#include "device/device_api.h"
#include "llrt_common/tiles.hpp"
#include "hostdevcommon/common_runtime_address_map.h"

constexpr static uint64_t TEST_MAILBOX_ADDR = l1_mem::address_map::FIRMWARE_BASE + TEST_MAILBOX_ADDRESS;
constexpr static uint64_t ENABLE_CORE_MAILBOX_ADDR = l1_mem::address_map::FIRMWARE_BASE + ENABLE_CORE_MAILBOX;
constexpr static uint64_t TEST_MAILBOX_ADDR_NCRISC = l1_mem::address_map::NCRISC_FIRMWARE_BASE + TEST_MAILBOX_ADDRESS;
constexpr static int INIT_VALUE = 42;
constexpr static uint32_t ENABLE_CORE_ENABLE_VALUE = 1;
constexpr static uint32_t ENABLE_CORE_DONE_VALUE = 0;
constexpr static int DONE_VALUE = 1;

constexpr static uint32_t TRISC_BASE = l1_mem::address_map::TRISC_BASE;
constexpr static uint32_t TRISC_L1_MAILBOX_OFFSET = l1_mem::address_map::TRISC_L1_MAILBOX_OFFSET;

constexpr static uint32_t trisc_sizes[3] = {
    l1_mem::address_map::TRISC0_SIZE, l1_mem::address_map::TRISC1_SIZE, l1_mem::address_map::TRISC2_SIZE};

constexpr static uint32_t trisc_mailbox_addresses[3] = {
    TRISC_BASE + TRISC_L1_MAILBOX_OFFSET,
    TRISC_BASE + trisc_sizes[0] + TRISC_L1_MAILBOX_OFFSET,
    TRISC_BASE + trisc_sizes[0] + trisc_sizes[1] + TRISC_L1_MAILBOX_OFFSET};

namespace tt {

// llrt = lower-level runtime
namespace llrt {

using RamSrcAddr = unsigned int;
using RamDstAddr = unsigned int;
using SrcL1Core = tt_xy_pair;
using SrcL1Cores = std::vector<SrcL1Core>;
using DstL1Core = tt_xy_pair;
using DstL1Cores = std::vector<DstL1Core>;
using SrcChannelId = int;
using DstChannelId = int;
using DramBufferSize = unsigned int;
using DramSrcAddr = unsigned int;
using DramDstAddr = unsigned int;
using L1Addr = std::uint32_t;
using SrcAddr = std::uint32_t;
using DestAddr = std::uint32_t;
using LoadFirmwareFlag = bool;
using CountOffset = unsigned int;
using NCHW = std::array<std::uint32_t, 4>;
using RSUV = std::array<std::uint32_t, 4>;
using BYTES_PER_DATUM = std::uint32_t;
using TRANSACTION_SIZE = std::uint32_t;
using NUM_TRANSACTIONS = std::uint32_t;
using NUM_REPETITIONS = std::uint32_t;

using DramCopySpec = std::tuple<tt_xy_pair, SrcChannelId, DstChannelId, DramBufferSize, DramSrcAddr, DramDstAddr, LoadFirmwareFlag>;
using RamCopySpec = std::tuple<tt_xy_pair, SrcL1Cores, DstL1Cores, DramBufferSize, DramSrcAddr, DramDstAddr, tiles_test::TileSize, tiles_test::TileIndex, CountOffset, CountOffset, LoadFirmwareFlag>;
using DramToL1CopySpec = std::tuple<tt_xy_pair, SrcChannelId, DramBufferSize, DramSrcAddr, L1Addr, LoadFirmwareFlag>;
using CopyPatternSpec = std::tuple<tt_xy_pair, DestAddr, tt_xy_pair, SrcAddr, NCHW, RSUV, BYTES_PER_DATUM, NUM_REPETITIONS, LoadFirmwareFlag>;
using L1ToDramCopySpec = std::tuple<tt_xy_pair, DstChannelId, DramBufferSize, DramDstAddr, L1Addr, LoadFirmwareFlag>;

using WorkerCore = tt_cxy_pair;
using WorkerCores = std::vector<WorkerCore>;
using CircularBufferConfigVec = std::vector<uint32_t>;


// made these free functions -- they're copy/paste of the member functions
// TODO: clean-up epoch_loader / epoch_binary -- a bunch of functions there should not be member functions
vector <uint32_t> get_risc_binary(string path, uint32_t id);
vector<uint32_t> get_trisc_binary(string path, uint32_t trisc_id);

// TODO: de-asserting reset properly
//  this deasserts reset for all BRISCs (on all devices, all cores), but not other RISC processors (NCRISC, TRISC)
// even though it deasserts reset for all the BRISCs, we are only loading  BRISC for a single core ("core")
// this is unsafe, since BRISCs for which we haven't loaded FW are now running garbage out of their L1
// proper solution:
// a) load dummy BRISC FW to unused cores, and keep using the function that de-asserts all BRISCs (easier, we can load blank kernel and disable NCRISC loading)
// b) de-assert reset only for used BRISCs (needs a new deassert function w/ a list of core to de-assert) (harder)
void deassert_brisc_reset_for_all_chips_all_cores(tt_cluster *cluster, bool stagger_start = false);

// TODO: try using "stop" method from device instead, it's the proper way of asserting reset
void assert_reset_for_all_chips(tt_cluster *cluster);

// tt_xy_pair core --> NOC coordinates ("functional workers" from the SOC descriptor)
// NOC coord is also synonymous to routing / physical coord
// dram_channel id (0..7) for GS is also mapped to NOC coords in the SOC descriptor
void write_hex_vec_to_core(tt_cluster *cluster, int chip, const tt_xy_pair& core, std::vector<uint32_t> hex_vec, uint64_t addr);

std::vector<std::uint32_t> read_hex_vec_from_core(tt_cluster *cluster, int chip, const tt_xy_pair& core, uint64_t addr, uint32_t size);

void print_worker_cores(tt_cluster *cluster, chip_id_t chip_id = 0);

bool is_worker_core(tt_cluster *cluster, const tt_xy_pair &core, chip_id_t chip_id = 0);

CircularBufferConfigVec create_circular_buffer_config_vector();

void set_config_for_circular_buffer(CircularBufferConfigVec& circular_buffer_config_vec, uint32_t circular_buffer_index, uint32_t addr_in_bytes, uint32_t size_in_bytes, uint32_t size_in_tiles);

void write_circular_buffer_config_vector_to_core(tt_cluster *cluster, int chip, const tt_xy_pair& core, CircularBufferConfigVec circular_buffer_config_vec);

void write_graph_interpreter_op_info_to_core(tt_cluster *cluster, int chip, const tt_xy_pair& core, op_info_t op_info, int op_idx);

// for BRISC and NCRISC
bool test_load_write_read_risc_binary(tt_cluster *cluster, std::string hex_file_path, int chip_id, const tt_xy_pair& core, int riscv_id);

// for TRISCs
bool test_load_write_read_trisc_binary(tt_cluster *cluster, std::string hex_file_path, int chip_id, const tt_xy_pair& core, int triscv_id);

void disable_ncrisc(tt_cluster *cluster, int chip_id, const tt_xy_pair& core);

void enable_ncrisc(tt_cluster *cluster, int chip_id, const tt_xy_pair& core);

void enable_triscs(tt_cluster *cluster, int chip_id, const tt_xy_pair& core);

void disable_triscs(tt_cluster *cluster, int chip_id, const tt_xy_pair& core);

WorkerCores get_worker_cores_from_cluster(tt_cluster *cluster, int chip_id);

// subchannel hard-coded to 0 for now
tt_xy_pair get_core_for_dram_channel(tt_cluster *cluster, int dram_channel_id, chip_id_t chip_id = 0);

enum class TensixRiscsOptions : std::uint32_t {
    NONE = 0,
    BRISC_ONLY = static_cast<std::uint32_t>(1 << 1),
    BRISC_NCRISC = static_cast<std::uint32_t>(1 << 2),
    BRISC_TRISCS = static_cast<std::uint32_t>(1 << 3),
    ALL_RISCS = static_cast<std::uint32_t>(1 << 4)
};

inline bool operator!=(const TensixRiscsOptions lhs, const TensixRiscsOptions rhs) {
    return
        static_cast<std::underlying_type<TensixRiscsOptions>::type>(lhs) !=
        static_cast<std::underlying_type<TensixRiscsOptions>::type>(rhs);
}

inline bool deduce_if_involves_triscs(const TensixRiscsOptions &riscs_options) {
    return riscs_options == TensixRiscsOptions::BRISC_TRISCS || riscs_options == TensixRiscsOptions::ALL_RISCS;
}

inline bool deduce_if_involves_ncrisc(const TensixRiscsOptions &riscs_options) {
    return riscs_options == TensixRiscsOptions::BRISC_NCRISC || riscs_options == TensixRiscsOptions::ALL_RISCS;
}

namespace utils {
    void log_current_ai_clk(tt_cluster *cluster);
}  // namespace utils

namespace internal_ {
    // This loads to briscs and ncriscs - we may want to add TensixRiscsOptions here
    void load_blank_kernel_to_cores(tt_cluster *cluster, int chip_id, const TensixRiscsOptions &riscs_to_load, std::vector<tt_xy_pair> cores);

    void load_blank_kernel_to_all_worker_cores_with_exceptions(tt_cluster *cluster, int chip_id, const TensixRiscsOptions &riscs_to_load, std::vector<tt_xy_pair> exceptions);

    void enable_core(tt_cluster *cluster, int chip_id, const tt_xy_pair &core);

    void enable_cores(tt_cluster *cluster, int chip_id, const std::vector<tt_xy_pair> &cores);

    void assert_enable_core_mailbox_is_valid_for_core(tt_cluster *cluster, int chip_id, const tt_xy_pair &core);

    void setup_riscs_on_specified_core(tt_cluster *cluster, int chip_id, const TensixRiscsOptions riscs_options, const tt_xy_pair &core);

    void setup_riscs_on_specified_cores(tt_cluster *cluster, int chip_id, const TensixRiscsOptions riscs_options, const std::vector<tt_xy_pair> &core);

    bool check_if_riscs_on_specified_core_done(tt_cluster *cluster, int chip_id, const TensixRiscsOptions riscs_options, const tt_xy_pair &core);

    void cleanup_risc_on_specified_core(tt_cluster *cluster, int chip_id, const TensixRiscsOptions riscs_options, const tt_xy_pair &core);

    void run_riscs_on_specified_cores(tt_cluster *cluster, int chip_id, const TensixRiscsOptions riscs_option,
                                        const std::vector<tt_xy_pair> &cores);

    void run_briscs_on_specified_cores(tt_cluster *cluster, int chip_id, const std::vector<tt_xy_pair> &cores);

} // namespace internal_

/*
 * DRAM COPY
 */

inline DramCopySpec create_dram_copy_spec(tt_xy_pair core, SrcChannelId src_channel_id, DstChannelId dst_channel_id, DramBufferSize dram_buffer_size, DramSrcAddr dram_src_addr, DramDstAddr dram_dst_addr, LoadFirmwareFlag load_firmware_flag = true) {
    return std::make_tuple(core, src_channel_id, dst_channel_id, dram_buffer_size, dram_src_addr, dram_dst_addr, load_firmware_flag);
}

std::vector<uint32_t> get_arg_hex_from_dram_copy_kernel_spec(tt_cluster *cluster, int chip_id, const DramCopySpec &spec);

void run_dram_copy_kernel_with_specs(tt_cluster *cluster, int chip_id, std::vector<DramCopySpec> specs, bool load_blanks = true);

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
    LoadFirmwareFlag load_firmware_flag = true
);

std::vector<uint32_t> get_arg_hex_from_ram_copy_kernel_spec(tt_cluster *cluster, int chip_id, const RamCopySpec &spec);

void run_ram_copy_kernel_with_specs(tt_cluster *cluster, int chip_id, std::vector<RamCopySpec> specs, bool load_blanks = true);

/*
 * DRAM TO L1 COPY
 */

DramToL1CopySpec create_dram_to_l1_copy_spec(
    tt_xy_pair core,
    SrcChannelId src_channel_id,
    DramBufferSize dram_buffer_size,
    DramSrcAddr dram_src_addr,
    L1Addr l1_addr,
    LoadFirmwareFlag load_firmware_flag = true
);

std::vector<uint32_t> get_arg_hex_from_dram_to_l1_copy_kernel_spec(tt_cluster *cluster, int chip_id, const DramToL1CopySpec &spec);

void run_dram_to_l1_copy_kernel_with_specs(tt_cluster *cluster, int chip_id, std::vector<DramToL1CopySpec> specs, bool load_blanks = true);

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
    LoadFirmwareFlag load_firmware_flag = true
);

std::vector<uint32_t> get_arg_hex_from_copy_pattern_kernel_spec(tt_cluster *cluster, int chip_id, const CopyPatternSpec &spec);

void run_copy_pattern_kernel_with_specs(tt_cluster *cluster, int chip_id, std::vector<CopyPatternSpec> specs, bool load_blanks = true);

/*
 * L1 TO DRAM COPY
 */

L1ToDramCopySpec create_l1_to_dram_copy_spec(
    tt_xy_pair core,
    DstChannelId dst_channel_id,
    DramBufferSize dram_buffer_size,
    DramDstAddr dram_dst_addr,
    L1Addr l1_addr,
    LoadFirmwareFlag load_firmware_flag = true
);

std::vector<uint32_t> get_arg_hex_from_l1_to_dram_copy_kernel_spec(tt_cluster *cluster, int chip_id, const L1ToDramCopySpec &spec);

void run_l1_to_dram_copy_kernel_with_specs(tt_cluster *cluster, int chip_id, std::vector<L1ToDramCopySpec> specs, bool load_blanks = true);

} // namespace llrt

} // namespace tt
