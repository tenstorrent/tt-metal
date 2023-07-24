#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "tt_device.h"

tt_VersimDevice::tt_VersimDevice(const tt_SocDescriptor &soc_descriptor_) : tt_device(soc_descriptor_) {
    throw std::runtime_error("tt_VersimDevice() -- VERSIM is not supported in this build\n");
}

tt_VersimDevice::~tt_VersimDevice() {}

int tt_VersimDevice::get_number_of_chips() { return detect_number_of_chips(); }

int tt_VersimDevice::detect_number_of_chips() { return 0; }

void tt_VersimDevice::start(
    std::vector<std::string> plusargs,
    std::vector<std::string> dump_cores,
    bool no_checkers,
    bool /*init_device*/,
    bool /*skip_driver_allocs*/
) {}

void tt_VersimDevice::deassert_risc_reset(bool start_stagger) {}

void tt_VersimDevice::assert_risc_reset() {}
void tt_VersimDevice::write_vector(
    const std::uint32_t *mem_ptr,
    uint32_t len,
    tt_cxy_pair target,
    std::uint32_t address,
    bool host_resident,
    bool small_access,
    chip_id_t src_device_id) {}
void tt_VersimDevice::write_vector(
    std::vector<uint32_t> &mem_vector,
    tt_cxy_pair target,
    std::uint32_t address,
    bool host_resident,
    bool small_access,
    chip_id_t src_device_id) {}
void tt_VersimDevice::read_vector(
    std::uint32_t *mem_ptr,
    tt_cxy_pair target,
    std::uint32_t address,
    std::uint32_t size_in_bytes,
    bool host_resident,
    bool small_access,
    chip_id_t src_device_id) {}
void tt_VersimDevice::read_vector(
    std::vector<uint32_t> &mem_vector,
    tt_cxy_pair target,
    std::uint32_t address,
    std::uint32_t size_in_bytes,
    bool host_resident,
    bool small_access,
    chip_id_t src_device_id) {}
std::map<int, int> tt_VersimDevice::get_clocks() { return {}; }

void tt_VersimDevice::dump_debug_mailbox(std::string output_path, int device_id) {}
void tt_VersimDevice::dump_wall_clock_mailbox(std::string output_path, int device_id) {}

bool versim_check_dram_core_exists(
    const std::vector<std::vector<CoreCoord>> &dram_core_channels, CoreCoord target_core) {
    return false;
}

bool tt_VersimDevice::stop() { return true; }
