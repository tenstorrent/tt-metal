/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include "third_party/umd/device/device_api.h"
#include "tt_metal/third_party/umd/device/tt_xy_pair.h"
#include "tt_metal/third_party/umd/device/tt_cluster_descriptor.h"
#include "common/test_common.hpp"
#include "common/base.hpp"
#include "common/tt_backend_api_types.hpp"
#include "common/metal_soc_descriptor.h"
#include "host_mem_address_map.h"
#include "tt_metal/third_party/umd/src/firmware/riscv/wormhole/eth_interface.h"
#include "dev_mem_map.h"

static constexpr std::uint32_t SW_VERSION = 0x00020000;
using tt_cluster_description = tt_ClusterDescriptor;
using std::chrono::high_resolution_clock;

using tt_target_dram = std::tuple<int, int, int>;
using tt::TargetDevice;
using tt::DEVICE;

struct tt_cluster;
using tt_cluster_on_destroy_callback = std::function<void (tt_cluster*)>;
using tt_cluster_on_close_device_callback = std::function<void (tt_cluster*, int)>;

struct tt_cluster
{
    private:
    std::unique_ptr<tt_device> device;
    std::unordered_map<chip_id_t, metal_SocDescriptor> sdesc_per_chip = {};
    std::unique_ptr<tt_cluster_description> ndesc;
    high_resolution_clock::time_point device_reset_time;
    std::set<chip_id_t> target_device_ids;
    vector<tt_cluster_on_destroy_callback> on_destroy_callbacks;
    vector<tt_cluster_on_close_device_callback> on_close_device_callbacks;

    tt_device_l1_address_params l1_fw_params = {
        (int32_t)20480, (int32_t)0, (int32_t)20480, (int32_t)16384, (int32_t)20480, (int32_t)53248
    };

    tt_driver_host_address_params host_address_params = {host_mem::address_map::ETH_ROUTING_BLOCK_SIZE, host_mem::address_map::ETH_ROUTING_BUFFERS_START};

    tt_driver_eth_interface_params eth_interface_params = {NOC_ADDR_LOCAL_BITS, NOC_ADDR_NODE_ID_BITS, ETH_RACK_COORD_WIDTH, CMD_BUF_SIZE_MASK, MAX_BLOCK_SIZE,
                                    REQUEST_CMD_QUEUE_BASE, RESPONSE_CMD_QUEUE_BASE, CMD_COUNTERS_SIZE_BYTES, REMOTE_UPDATE_PTR_SIZE_BYTES,
                                    CMD_DATA_BLOCK, CMD_WR_REQ, CMD_WR_ACK, CMD_RD_REQ, CMD_RD_DATA, CMD_BUF_SIZE, CMD_DATA_BLOCK_DRAM, ETH_ROUTING_DATA_BUFFER_ADDR,
                                    REQUEST_ROUTING_CMD_QUEUE_BASE, RESPONSE_ROUTING_CMD_QUEUE_BASE, CMD_BUF_PTR_MASK};

    int remote_arc_msg(const chip_id_t &chip, uint32_t msg_code, bool wait_for_done, uint32_t arg0, uint32_t arg1, int timeout, uint32_t *return_3, uint32_t *return_4);
    void configure_static_tlbs(const std::uint32_t& chip);
    void set_dram_barrier(chip_id_t chip_id, uint32_t barrier_value);
    void set_l1_barrier(chip_id_t chip_id, uint32_t barrier_value);

    public:
    TargetDevice type;
    int target_ai_clk = 0;
    bool deasserted_risc_reset;
    tt_cluster() : device(nullptr), type(TargetDevice::Invalid), deasserted_risc_reset(false) {};
    ~tt_cluster() { for (auto cb: on_destroy_callbacks) cb(this); }

    // adds a specified callback to the list of callbacks to be called on destroy of this cluster instance
    void on_destroy(tt_cluster_on_destroy_callback callback);
    void on_close_device(tt_cluster_on_close_device_callback callback);

    std::chrono::seconds get_device_timeout();
    std::chrono::seconds get_device_duration();

    int get_num_chips();
    std::unordered_set<chip_id_t> get_all_chips();

    std::set<chip_id_t> get_all_mmio_chips();

    metal_SocDescriptor& get_soc_desc(chip_id_t chip) { return sdesc_per_chip.at(chip); }
    uint32_t get_harvested_rows(chip_id_t chip) { return device->harvested_rows_per_target.at(chip); }

    tt_cluster_description *get_cluster_desc() { return ndesc.get(); }

    void dump_wall_clock_mailbox(std::string output_dir);

    //! device driver and misc apis
    static std::vector<tt::ARCH> detect_available_devices(const TargetDevice &target_type);
    void clean_system_resources();

    void open_device(
        const tt::ARCH &arch,
        const TargetDevice &target_type,
        const std::set<int> &target_devices,
        const std::string &sdesc_path = "",
        const std::string &ndesc_path = "",
        const bool &skip_driver_allocs = false);

    void start_device(const tt_device_params &device_params);
    void close_device();
    void verify_eth_fw();

    void assert_risc_reset(const chip_id_t &chip);
    void set_remote_tensix_risc_reset(const tt_cxy_pair &core, const TensixSoftResetOptions &soft_resets);
    void deassert_risc_reset(const chip_id_t &target_device_id, bool start_stagger = false);
    void reset_remote_chip(const chip_id_t &chip_id);
    void stop_remote_chip(const chip_id_t &chip);
    void check_timeout(std::string output_dir);
    void dump_debug_mailbox(std::string output_dir);
    void verify_sw_fw_versions(int device_id, std::uint32_t sw_version, std::vector<std::uint32_t> &fw_versions);

    uint32_t reserve_non_mmio_block(bool reserve, tt_cxy_pair core, uint64_t address);

    void write_dram_vec(vector<uint32_t> &vec, tt_target_dram dram, uint64_t addr, bool small_access = false);
    void read_dram_vec(vector<uint32_t> &vec, tt_target_dram dram, uint64_t addr, uint32_t size, bool small_access = false);

    void write_dram_vec(vector<uint32_t> &vec, tt_cxy_pair dram_core, uint64_t addr, bool small_access = false);
    void write_dram_vec(const uint32_t *mem_ptr, uint32_t len, tt_cxy_pair dram_core, uint64_t addr, bool small_access = false);
    void read_dram_vec(vector<uint32_t> &vec, tt_cxy_pair dram_core, uint64_t addr, uint32_t size, bool small_access = false);
    void read_dram_vec(uint32_t *mem_ptr, tt_cxy_pair dram_core, uint64_t addr, uint32_t size, bool small_access = false);

    void write_sysmem_vec(vector<uint32_t> &vec, uint64_t addr, chip_id_t src_device_id);
    void read_sysmem_vec(vector<uint32_t> &vec, uint64_t addr, uint32_t size, chip_id_t src_device_id);

    //! address translation
    void *channel_0_address(std::uint32_t offset, std::uint32_t device_id) const;
    //void *host_dma_address(std::uint64_t offset, chip_id_t src_device_id) const;

    std::map<int, int> get_all_device_aiclks();
    int get_device_aiclk(const chip_id_t &chip_id);
    // void set_power_state(tt_DevicePowerState state);

    // will write a value for each core+hart's debug buffer, indicating that by default
    // any prints will be ignored unless specifically enabled for that core+hart
    // (using tt_start_debug_print_server)
    void reset_debug_print_server_buffers();

    // Writes BARRIER_RESET to all DRAM banks
    void initialize_dram_barrier(chip_id_t chip_id);
    // Writes BARRIER_RESET to all L1 banks
    void initialize_l1_barrier(chip_id_t chip_id);

    void dram_barrier(chip_id_t chip_id);
    void l1_barrier(chip_id_t chip_id);

};

std::ostream &operator<<(std::ostream &os, tt_target_dram const &dram);
bool check_dram_core_exists(const std::vector<std::vector<CoreCoord>> &all_dram_cores, CoreCoord target_core);
