#pragma once

#include <chrono>
#include "device/device_api.h"
#include "common/test_common.hpp"
#include "common/base.hpp"
#include "common/tt_backend_api_types.hpp"

static constexpr std::uint32_t SW_VERSION = 0x00020000;
using tt_soc_description = tt_SocDescriptor;
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
    // std::unique_ptr<tt_soc_description> sdesc;  // SOC desc per chip level now
    std::unordered_map<chip_id_t, tt_soc_description> sdesc_per_chip = {};
    std::unique_ptr<tt_cluster_description> ndesc;
    high_resolution_clock::time_point device_reset_time;
    std::set<chip_id_t> target_device_ids;
    vector<tt_cluster_on_destroy_callback> on_destroy_callbacks;
    vector<tt_cluster_on_close_device_callback> on_close_device_callbacks;

    void enable_ethernet_queue(const chip_id_t &chip, int timeout);
    int remote_arc_msg(const chip_id_t &chip, uint32_t msg_code, bool wait_for_done, uint32_t arg0, uint32_t arg1, int timeout, uint32_t *return_3, uint32_t *return_4);
    public:
    TargetDevice type;
    int target_ai_clk = 0;
    bool deasserted_risc_reset;
    bool performed_harvesting = false;
    std::unordered_map<chip_id_t, uint32_t> harvested_rows_per_target = {};
    tt_cluster() : device(nullptr), type(TargetDevice::Invalid), deasserted_risc_reset(false) {};
    ~tt_cluster() { for (auto cb: on_destroy_callbacks) cb(this); }

    // adds a specified callback to the list of callbacks to be called on destroy of this cluster instance
    void on_destroy(tt_cluster_on_destroy_callback callback);
    void on_close_device(tt_cluster_on_close_device_callback callback);

    std::chrono::seconds get_device_timeout();
    std::chrono::seconds get_device_duration();

    int get_num_chips();
    std::unordered_set<chip_id_t> get_all_chips();

    tt_soc_description& get_soc_desc(chip_id_t chip) { return sdesc_per_chip.at(chip); }

    // tt_soc_description *get_soc_desc() { return sdesc.get(); }
    tt_cluster_description *get_cluster_desc() { return ndesc.get(); }

    tt_xy_pair get_routing_coordinate(int core_r, int core_c, chip_id_t device_id) const;
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

    void broadcast_remote_tensix_risc_reset(const chip_id_t &chip, const TensixSoftResetOptions &soft_resets);
    void set_remote_tensix_risc_reset(const tt_cxy_pair &core, const TensixSoftResetOptions &soft_resets);
    void deassert_risc_reset(bool start_stagger = false);
    void deassert_risc_reset_remote_chip(const chip_id_t &chip_id, bool start_stagger = false);
    void reset_remote_chip(const chip_id_t &chip_id);
    void stop_remote_chip(const chip_id_t &chip);
    void wait_for_completion(std::string output_dir);
    void check_timeout(std::string output_dir);
    void dump_debug_mailbox(std::string output_dir);
    void verify_sw_fw_versions(int device_id, std::uint32_t sw_version, std::vector<std::uint32_t> &fw_versions);

    uint32_t reserve_non_mmio_block(bool reserve, tt_cxy_pair core, uint64_t address);
    void write_to_non_mmio_device(vector<uint32_t> &mem_vector, tt_cxy_pair core, uint64_t address);
    void read_from_non_mmio_device(vector<uint32_t> &mem_vector, tt_cxy_pair core, uint64_t address, uint32_t size_in_bytes);

    void write_dram_vec(vector<uint32_t> &vec, tt_target_dram dram, uint64_t addr, bool small_access = false);
    void read_dram_vec(vector<uint32_t> &vec, tt_target_dram dram, uint64_t addr, uint32_t size, bool small_access = false);

    void write_dram_vec(vector<uint32_t> &vec, tt_cxy_pair dram_core, uint64_t addr, bool small_access = false);
    void read_dram_vec(vector<uint32_t> &vec, tt_cxy_pair dram_core, uint64_t addr, uint32_t size, bool small_access = false);

    void write_sysmem_vec(vector<uint32_t> &vec, uint64_t addr, chip_id_t src_device_id);
    void read_sysmem_vec(vector<uint32_t> &vec, uint64_t addr, uint32_t size, chip_id_t src_device_id);

    //! address translation
    void *channel_0_address(std::uint32_t offset, std::uint32_t device_id) const;
    void *host_dma_address(std::uint64_t offset, chip_id_t src_device_id) const;

    std::map<int, int> get_all_device_aiclks();
    int get_device_aiclk(const chip_id_t &chip_id);
    void set_device_aiclk();
    void reset_device_aiclk();
    void set_power_state(tt_DevicePowerState state);

    // will write a value for each core+hart's debug buffer, indicating that by default
    // any prints will be ignored unless specifically enabled for that core+hart
    // (using tt_start_debug_print_server)
    void reset_debug_print_server_buffers();
};

std::ostream &operator<<(std::ostream &os, tt_target_dram const &dram);
std::unique_ptr<tt_soc_description> load_soc_descriptor_from_file(const tt::ARCH &arch, std::string file_path);
bool check_dram_core_exists(const std::vector<std::vector<tt_xy_pair>> &all_dram_cores, tt_xy_pair target_core);
