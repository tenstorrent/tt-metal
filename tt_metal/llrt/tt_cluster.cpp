#include "tt_cluster.hpp"
#include "eth_interface.h"
#include "device/tt_silicon_driver_common.hpp"
#include "dev_mem_map.h"
#include <immintrin.h>
#include <string>
#include <iomanip>
#include <iostream>
#include "hostdevcommon/common_runtime_address_map.h"
#include "hostdevcommon/debug_print_common.h"

using std::to_string;
using std::cout;
using std::endl;


std::chrono::seconds tt_cluster::get_device_timeout() {
    int device_timeout = 3600; // seconds
    const char* timeout_override = std::getenv("TT_BACKEND_TIMEOUT");
    if (timeout_override) {
        device_timeout = atoi(timeout_override);
    }
    return std::chrono::seconds{device_timeout};
}

std::chrono::seconds tt_cluster::get_device_duration() {
    high_resolution_clock::time_point device_current_time;
    device_current_time = high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(device_current_time - device_reset_time);
    return duration;
}

int tt_cluster::get_num_chips() {
    return get_all_chips().size();
}

std::unordered_set<chip_id_t> tt_cluster::get_all_chips() {
    return ndesc->get_all_chips();
}

void tt_cluster::dump_wall_clock_mailbox(std::string output_dir) {
    bool is_output_dir_populated = output_dir.find("tt_build") != std::string::npos;
    if (is_output_dir_populated) {
        for (auto device_id : target_device_ids){
            string output_path = output_dir + "/wall_clock_device_";
            output_path += to_string(device_id) + ".yaml";
            tt::log_info(tt::LogLLRuntime, "Reading wall-clock mailbox for device {}, output yaml path {}", device_id, output_path);
            std::ofstream output_file(output_path);
            const int mailbox_base_addr = MEM_WALL_CLOCK_MAILBOX_ADDRESS;
            const int num_mailbox_32_regs = 4;
            const int mailbox_size = num_mailbox_32_regs * 4;
            for (auto &worker_core : get_soc_desc(device_id).workers) {
                int core_x = worker_core.x;
                int core_y = worker_core.y;
                std::string core_id = std::to_string(core_x) + "-" + std::to_string(core_y);
                output_file << core_id << ":" << std::endl;

                std::vector<uint32_t> mailbox_events;
                read_dram_vec(mailbox_events, tt_cxy_pair(device_id, core_x, core_y), mailbox_base_addr, mailbox_size);
                assert(mailbox_events.size() == num_mailbox_32_regs);
                uint64_t start_time = (uint64_t(mailbox_events[1]) << 32) + mailbox_events[0];
                uint64_t end_time = (uint64_t(mailbox_events[3]) << 32) + mailbox_events[2];
                output_file << "        " << std::left << std::setw(12) << "start: " << start_time << std::endl;
                output_file << "        " << std::left << std::setw(12) << "end: " << end_time << std::endl;
                output_file << "        " << std::left << std::setw(12) << "runtime: " << end_time - start_time
                            << std::endl;
            }
            output_file.close();
        }
    }
}

// Return vector of device names detected, used by API function. Assumes all devices are the same arch.
std::vector<tt::ARCH> tt_cluster::detect_available_devices(const TargetDevice &target_type){
    static std::vector<tt::ARCH> available_devices = {}; // Static to act as cache for repeat queries to avoid device interation.

    TT_ASSERT(target_type == TargetDevice::Versim or target_type == TargetDevice::Silicon);

    if (target_type == TargetDevice::Silicon) {
        if (available_devices.size() == 0){
            log_debug(tt::LogDevice, "Going to query silicon device for detect_available_devices()");

            std::vector<chip_id_t> available_device_ids = tt_SiliconDevice::detect_available_device_ids(true, true);
            int num_available_devices = available_device_ids.size();

            if (num_available_devices > 0) {
                auto detected_arch_name = detect_arch(available_device_ids.at(0));
                if (detected_arch_name != tt::ARCH::Invalid){
                    tt::ARCH device_name = detected_arch_name;
                    int num_devices = tt_SiliconDevice::detect_number_of_chips(true);
                    available_devices.insert(available_devices.end(), num_devices, device_name);
                }else{
                    tt::log_info(tt::LogDevice, "Silicon device arch name was detected as Invalid");
                }
            } else {
                tt::log_info(tt::LogDevice, "No silicon devices detected");
            }
        }
    }else{
        throw std::runtime_error(
            "We must be using a silicon device");
    }

    return available_devices;
}

// clean up bad system resource state that may be carried over
void tt_cluster::clean_system_resources() {
    TT_ASSERT(device != nullptr ,  "Device not initialized, make sure compile is done before running!");
    device->clean_system_resources();
}

void tt_cluster::verify_eth_fw() {
    const std::unordered_set<chip_id_t> &all_chips = ndesc->get_all_chips();
    for (const chip_id_t &chip : all_chips) {
        std::vector<uint32_t> mem_vector;
        std::vector<uint32_t> fw_versions;

        for (CoreCoord &eth_core : get_soc_desc(chip).ethernet_cores) {
            read_dram_vec(mem_vector, tt_cxy_pair(chip, eth_core), eth_l1_mem::address_map::FW_VERSION_ADDR, 4);
            fw_versions.push_back(mem_vector.at(0));
        }
        verify_sw_fw_versions(chip, SW_VERSION, fw_versions);
    }
}

int extract_chip_id_from_sdesc_path(std::filesystem::path sdesc_path) {
    string file = sdesc_path.filename().string();
    return atoi(file.substr(0, file.find(".")).c_str());
}

void tt_cluster::open_device(
    const tt::ARCH &arch,
    const TargetDevice &target_type,
    const std::set<chip_id_t> &target_devices,
    const std::string &sdesc_path,
    const std::string &ndesc_path,
    const bool &skip_driver_allocs) {
#ifdef ARCH_GRAYSKULL
    tt::log_assert(arch == tt::ARCH::GRAYSKULL, "Arch={} doesn't match compile-time build for GRAYSKULL", get_string(arch));
#endif
#ifdef ARCH_WORMHOLE
    tt::log_assert((arch == tt::ARCH::WORMHOLE_B0) || (arch == tt::ARCH::WORMHOLE), "Arch={} doesn't match compile-time build for WORMHOLE", get_string(arch));
#endif
    target_device_ids = target_devices;

    if (!std::filesystem::is_directory(sdesc_path)) {
        tt_SocDescriptor sdesc = *load_soc_descriptor_from_file(arch, sdesc_path);
        for(auto it : target_devices) {
            sdesc_per_chip[it] = sdesc;
        }
        tt::log_info(tt::LogDevice, "SOC descriptors loaded {}", sdesc_path);
    }
    else {
        for (const auto& entry: std::filesystem::directory_iterator(sdesc_path)) {
            sdesc_per_chip[extract_chip_id_from_sdesc_path(entry.path())] = *load_soc_descriptor_from_file(arch, entry.path());
        }
    }

    if (ndesc_path == "") {
        ndesc = tt_cluster_description::create_for_grayskull_cluster(target_devices);
    } else {
        ndesc = tt_cluster_description::create_from_yaml(ndesc_path);
    }
    tt::log_info(tt::LogDevice, "Network descriptor loaded {}", ndesc_path);

    TT_ASSERT(sdesc_per_chip.size());
    TT_ASSERT(ndesc != nullptr);

    if (target_type == TargetDevice::Silicon) {
        // For silicon driver, filter mmio devices to use only mmio chips required by netlist workload, to allow sharing
        // of resource (reservation/virtualization) like GS where cluster desc only contains netlist workload devices.
        std::unordered_set<chip_id_t> mmio_chips;
        for (auto &d: target_devices){
            if (ndesc->is_chip_mmio_capable(d)){
                mmio_chips.insert(d);
            }
        }

        device = std::make_unique<tt_SiliconDevice>(this->sdesc_per_chip, mmio_chips, skip_driver_allocs);
        if(!std::filesystem::is_directory(sdesc_path)) {
            if (arch == tt::ARCH::WORMHOLE_B0 or arch == tt::ARCH::WORMHOLE) {
                for(auto chip_id = target_devices.begin(); chip_id != target_devices.end(); chip_id++){
                    harvested_rows_per_target[*chip_id] =  device->get_harvested_noc_rows(*mmio_chips.begin()); // The harvesting mask is shared across all devices in the cluster for WH.
                    if(harvested_rows_per_target[*chip_id]) {
                        performed_harvesting = true;
                    }
                }
            } else if (arch == tt::ARCH::GRAYSKULL) {
                // Multichip harvesting is supported for GS.
                for(auto chip_id = target_devices.begin(); chip_id != target_devices.end(); chip_id++){
                    harvested_rows_per_target[*chip_id] =  device->get_harvested_noc_rows(*chip_id);
                    if(harvested_rows_per_target[*chip_id]) {
                        performed_harvesting = true;
                    }
                }
            }
        }
    }

    type = target_type;
    TT_ASSERT(type == TargetDevice::Versim or type == TargetDevice::Silicon);

    if (device) {
        // device->assert_risc_reset();
        if (get_num_chips() != tt::MAX_AVAILABLE_CHIPS) {
            if (arch == tt::ARCH::WORMHOLE || arch == tt::ARCH::WORMHOLE_B0) {
                // TODO: device->get_number_of_chips() currently does not account for non-MMIO chips, if device api is
                // updated to include these chips we can use the same device count check for WH/WH_B0 Have to trust
                // cluster descriptor to specify correct number of chips
                TT_ASSERT(device->get_number_of_chips() >= 1, "Must have at least one detected chip available on device!");
            } else {
                //TT_ASSERT(get_num_chips() <= device->get_number_of_chips(), "Requested number of chips through machine descriptor is bigger than number of chips available on device!");
            }
        }
    }
}

std::map<int, int> tt_cluster::get_all_device_aiclks(){
    return device->get_clocks();
}

int tt_cluster::get_device_aiclk(const chip_id_t &chip_id) {
    if (target_device_ids.find(chip_id) != target_device_ids.end()) {
        return device->get_clocks().at(chip_id);
    }
    return 0;
}

void tt_cluster::set_device_aiclk() {
    if (target_ai_clk != 0) {
        tt::log_info(tt::LogLLRuntime, "Setting the device AICLK to {}", target_ai_clk);
        device->set_device_aiclk(target_ai_clk);
    }
}

void tt_cluster::reset_device_aiclk() {
    tt::log_info(tt::LogLLRuntime, "Resetting device AICLK");
    device->reset_device_aiclk();
}

void tt_cluster::set_power_state(tt_DevicePowerState device_state) {
    std::stringstream ss;
    ss << "Setting silicon device power state to " << device_state;
    tt::log_info(tt::LogLLRuntime, "{}", ss.str());
    device->set_power_state(device_state);
}

void tt_cluster::reset_debug_print_server_buffers() {
    for (const int device_id : this->target_device_ids) {
        auto workers = get_soc_desc(device_id).workers;
        for (const CoreCoord &core : workers)
        for (int hart_id = 0; hart_id < 5; hart_id++) { // TODO(AP): must match DPRINT_NHARTS, magic
            // compute the buffer address for the requested hart
            uint32_t base_addr = PRINT_BUFFER_NC + hart_id*PRINT_BUFFER_SIZE;

            // The way this works is we first initialize all dprint buffers
            // for all cores and all harts to this magic number.
            // This way the device DPRINT will know that this core hasn't been initialized
            // from tt_start_debug_print_server
            // If we didn't have this mechanism them DPRINT in the kernel would have no way of knowing
            // Whether the host is polling it's buffer and flushing it.
            // If kernel code (in debug_print.h) detects that this magic value is present
            // Then it simply skips the print entirely. It prevents the kernel code
            // from hanging in a stall waiting for the host to flush the buffer
            // and removes the requirement that the host must listen on device's buffer.
            vector<uint32_t> initbuf = { uint32_t(DEBUG_PRINT_SERVER_DISABLED_MAGIC) };
            write_dram_vec(initbuf, {uint32_t(device_id), core}, base_addr);
        }
    }
}

void tt_cluster::start_device(const tt_device_params &device_params) {
    TT_ASSERT(sdesc_per_chip.size(), "Descriptor must be loaded. Try open_device()");
    TT_ASSERT(device != nullptr ,  "Device not initialized, make sure compile is done before running!");

    if (device_params.init_device) {
        device->init_system(device_params, get_soc_desc(*target_device_ids.begin()).grid_size); //pass in the same grid size for all (first one)
        set_device_aiclk();
        if (ndesc != nullptr) {
            const std::set<chip_id_t> &all_chips = target_device_ids;
            for (const chip_id_t &chip : all_chips) {
                if (!ndesc->is_chip_mmio_capable(chip)) {
                    reset_remote_chip(chip);
                }
            }
        }
    } else {
        device->start(device_params.expand_plusargs(), {}, false, false, device_params.skip_driver_allocs);
    }

    reset_debug_print_server_buffers();

    device_reset_time = high_resolution_clock::now();
}

void tt_cluster::close_device() {

    for (auto cb: on_close_device_callbacks) {
        // presumably we will have multiple devices per cluster in the future
        // so we pass a device index here
        // currently this is only used for shutting down the debug print server
        cb(this, 0);
    }

    reset_device_aiclk();
    set_power_state(tt_DevicePowerState::LONG_IDLE);
    if (device) {
        device->shutdown_system();
        const std::set<chip_id_t> &all_chips = target_device_ids;
        for (const chip_id_t &chip : all_chips) {
            if (!ndesc->is_chip_mmio_capable(chip)) {
                stop_remote_chip(chip);
            }
        }
        device.reset();
    }
    sdesc_per_chip.clear();
    ndesc.reset();
}

void tt_cluster::wait_for_completion(std::string output_dir) {
    vector<uint32_t> mem_vector;
    std::map<int, std::unordered_set<CoreCoord>> device_idle_cores;
    bool all_worker_cores_done;

    // initially assume no cores are idle
    for (const int device_id : target_device_ids) {
        device_idle_cores.insert({device_id, {}});
    }
    do {
        all_worker_cores_done = true;
        for (const int device_id : target_device_ids) {
            for (const CoreCoord &core : get_soc_desc(device_id).workers) {
                // check for core busy
                bool is_core_busy = device_idle_cores.at(device_id).find(core) == device_idle_cores.at(device_id).end();
                if (is_core_busy) {
                    // check for core done
                    read_dram_vec(mem_vector, tt_cxy_pair(device_id, core), MEM_TEST_MAILBOX_ADDRESS + MEM_MAILBOX_NCRISC_OFFSET, 4);
                    bool is_core_done = (mem_vector.at(0) == 1) or (mem_vector.at(0) == 0xabcd1234);
                    if (is_core_done) {
                        device_idle_cores.at(device_id).insert(core);
                        // check for stream assertions
                        read_dram_vec(mem_vector, tt_cxy_pair(device_id, core), MEM_TEST_MAILBOX_ADDRESS + MEM_MAILBOX_BRISC_OFFSET, 4);
                        if (mem_vector.at(0) == 0xdeeeaaad) {
                            log_fatal(tt::LogDevice, "Device {} stream assertions detected from core {}-{}", device_id, core.x, core.y);
                        }
                    } else {
                        log_trace(tt::LogDevice, "Device {} completion signal not received from core {}-{}", device_id, core.x, core.y);
                        all_worker_cores_done = false;
                    }
                }
            }
        }
        check_timeout(output_dir);
    } while (!all_worker_cores_done);
}

// Returns 0 if everything was OK
int tt_cluster::remote_arc_msg(const chip_id_t &chip, uint32_t msg_code, bool wait_for_done, uint32_t arg0, uint32_t arg1, int timeout, uint32_t *return_3, uint32_t *return_4) {
    constexpr uint64_t ARC_RESET_SCRATCH_ADDR = 0x880030060;
    constexpr uint64_t ARC_RESET_MISC_CNTL_ADDR = 0x880030100;

    auto core = tt_cxy_pair(chip, get_soc_desc(chip).arc_cores.at(0));

    if ((msg_code & 0xff00) != 0xaa00) {
        tt::log_error(tt::LogLLRuntime, "Malformed message. msg_code is 0x{:x} but should be 0xaa..\n", msg_code);
    }
    assert (arg0 <= 0xffff and arg1 <= 0xffff); // Only 16 bits are allowed

    const uint32_t MSG_ERROR_REPLY = 0xffffffff;

    uint32_t fw_arg = arg0 | (arg1<<16);
    int exit_code = 0;

    {
        std::vector<uint32_t> fw_vec = {fw_arg};
        write_dram_vec(fw_vec, core, ARC_RESET_SCRATCH_ADDR + 3 * 4, true);
    }

    {
        std::vector<uint32_t> msg_vec = {msg_code};
        write_dram_vec(msg_vec, core, ARC_RESET_SCRATCH_ADDR + 5 * 4, true);
    }

    std::vector<uint32_t> read_data;
    read_dram_vec(read_data, core, ARC_RESET_MISC_CNTL_ADDR, 4, true);
    uint32_t misc = read_data[0];

    if (misc & (1 << 16)) {
        log_error(tt::LogDevice, "trigger_fw_int failed on device {}", chip);
        return 1;
    } else {
        std::vector<uint32_t> misc_vec = {misc | (1 << 16)};
        write_dram_vec(misc_vec, core, ARC_RESET_MISC_CNTL_ADDR, true);
    }

    if (wait_for_done) {
        uint32_t status = 0xbadbad;
        auto timeout_seconds = std::chrono::seconds(timeout);
        auto start = std::chrono::system_clock::now();
        while (true) {
            if (std::chrono::system_clock::now() - start > timeout_seconds) {
                throw std::runtime_error("Timed out after waiting " + std::to_string(timeout) + " seconds for device " + std::to_string(chip) + " ARC to respond");
            }

            read_data.clear();
            read_dram_vec(read_data, core, ARC_RESET_SCRATCH_ADDR + 5 * 4, 4, true);
            status = read_data[0];

            if ((status & 0xffff) == (msg_code & 0xff)) {
                if (return_3 != nullptr) {
                    read_data.clear();
                    read_dram_vec(read_data, core, ARC_RESET_SCRATCH_ADDR + 3 * 4, 4, true);
                    *return_3 = read_data[0];
                }

                if (return_4 != nullptr) {
                    read_data.clear();
                    read_dram_vec(read_data, core, ARC_RESET_SCRATCH_ADDR + 4 * 4, 4, true);
                    *return_4 = read_data[0];
                }

                exit_code = (status & 0xffff0000) >> 16;
                break;
            } else if (status == MSG_ERROR_REPLY) {
                log_warning(tt::LogDevice, "On device {}, message code 0x{:x} not recognized by FW", chip, msg_code);
                exit_code = MSG_ERROR_REPLY;
                break;
            }
        }
    }

    return exit_code;
}

void tt_cluster::enable_ethernet_queue(const chip_id_t &chip, int timeout) {
    const uint32_t MSG_ERROR_REPLY = 0xffffffff;

    if (type == TargetDevice::Silicon) {
        for (const chip_id_t &chip : target_device_ids) {
            auto arch = get_soc_desc(chip).arch;
            switch (arch) {
                case tt::ARCH::WORMHOLE:
                case tt::ARCH::WORMHOLE_B0: {
                    if (ndesc->is_chip_mmio_capable(chip)) {
                        device->enable_ethernet_queue(chip, timeout);
                    } else {
                        uint32_t msg_success = 0x0;
                        auto timeout_seconds = std::chrono::seconds(timeout);
                        auto start = std::chrono::system_clock::now();
                        while (msg_success != 1) {
                            if (std::chrono::system_clock::now() - start > timeout_seconds) {
                                throw std::runtime_error("Timed out after waiting " + std::to_string(timeout) + " seconds for DRAM to finish training");
                            }

                            int msg_rt = remote_arc_msg(chip, 0xaa58, true, 0xFFFF, 0xFFFF, 1, &msg_success, NULL);
                            if (msg_rt == MSG_ERROR_REPLY) {
                                break;
                            }
                        }
                    }

                    break;
                }
                default: {
                    break;
                }
            }
        }
    }
}

void tt_cluster::broadcast_remote_tensix_risc_reset(const chip_id_t &chip, const TensixSoftResetOptions &soft_resets) {
    auto valid = soft_resets & ALL_TENSIX_SOFT_RESET;

    for (const CoreCoord &worker_core : sdesc_per_chip.at(chip).workers) {
        set_remote_tensix_risc_reset(tt_cxy_pair(chip, worker_core), valid);
    }
}

void tt_cluster::set_remote_tensix_risc_reset(const tt_cxy_pair &core, const TensixSoftResetOptions &soft_resets) {
    auto valid = soft_resets & ALL_TENSIX_SOFT_RESET;

    std::vector<uint32_t> vec = {(std::underlying_type<TensixSoftResetOptions>::type) valid};
    write_dram_vec(vec, core, 0xFFB121B0 /* Should get this value from the device */);
    _mm_sfence();
}

void tt_cluster::deassert_risc_reset(bool start_stagger) {
    if (type == TargetDevice::Versim) {
        // Not running silicon multichip test
         device->deassert_risc_reset();
    } else if (type == TargetDevice::Silicon) {
        // On silicon, we might have num_mmio_chips < total_chips, in this case we manually write data to all the worker
        // cores on remote chips
        // TODO: for now assume that chip ids for MMIO chips are 0 ~ (num_mmio_chips-1)
        // Need to change m_pci_device object in silicon driver to support a generic subset of chip ids with MMIO
        log_debug(tt::LogLLRuntime, "Stagger start : {}", start_stagger);
        device->deassert_risc_reset(start_stagger);
        const std::unordered_set<chip_id_t> &all_chips = ndesc->get_all_chips();
        for (const chip_id_t &chip : all_chips) {
            if (!ndesc->is_chip_mmio_capable(chip)) {
                deassert_risc_reset_remote_chip(chip, start_stagger);
            }
        }
    }
    device_reset_time = high_resolution_clock::now();
    deasserted_risc_reset = false;
}

void tt_cluster::deassert_risc_reset_remote_chip(const chip_id_t &chip, bool start_stagger) {
    if (start_stagger){
        broadcast_remote_tensix_risc_reset(chip, TENSIX_DEASSERT_SOFT_RESET);
    }else{
        broadcast_remote_tensix_risc_reset(chip, TENSIX_DEASSERT_SOFT_RESET_NO_STAGGER);
    }
}

void tt_cluster::reset_remote_chip(const chip_id_t &chip) {
    constexpr uint64_t DEASSERT_ARC_WRITE_ADDR = 0x880030040;

    broadcast_remote_tensix_risc_reset(chip, TENSIX_ASSERT_SOFT_RESET);

    // NOTE(drosen): In wormhole this write should not be done as it can trigger a timing violation.
    //               It remains for backwards compatibility until the new reset sequence is finallized
    //               and broadly distriuted.
    // What if there are multiple arc cores? Revisit this
    TT_ASSERT(sdesc_per_chip.at(chip).arc_cores.size() == 1, "Multiple arc cores specified in soc descriptor, update this reset function");
    // deassert arc reset
    std::vector<uint32_t> RISCV_RESET_DEASSERT = {0xffffffff, 0xffffffff, 0xffff, 0x0, 0x0, 0x0, 0x0, 0x0};
    write_dram_vec(RISCV_RESET_DEASSERT, tt_cxy_pair(chip, sdesc_per_chip.at(chip).arc_cores.at(0)), DEASSERT_ARC_WRITE_ADDR);
}

void tt_cluster::stop_remote_chip(const chip_id_t &chip) {
    auto arch = get_soc_desc(chip).arch;
    switch (arch) {
        case tt::ARCH::GRAYSKULL: {
            // NOTE(drosen): Running this at maximum clocks violates timing contraints for wormhole.
            //               resets are entirely controlled via soft reset regs.
            constexpr uint64_t ASSERT_ARC_WRITE_ADDR = 0x880030040;
            // What if there are multiple arc cores? Revisit this
            TT_ASSERT(get_soc_desc(chip).arc_cores.size() == 1, "Multiple arc cores specified in soc descriptor, update this reset function");
            // assert arc reset
            std::vector<uint32_t> vec = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0};
            write_dram_vec(vec, tt_cxy_pair(chip, get_soc_desc(chip).arc_cores.at(0)), ASSERT_ARC_WRITE_ADDR);
            break;
        }
        case tt::ARCH::WORMHOLE:
        case tt::ARCH::WORMHOLE_B0: {
            broadcast_remote_tensix_risc_reset(chip, TENSIX_ASSERT_SOFT_RESET);
            break;
        }
        case tt::ARCH::Invalid: {
            log_warning(tt::LogLLRuntime, "Tried to stop device with tt::ARCH::Invalid, skipping riscv reset assertion.");
            break;
        }
        default: {
            TT_ASSERT(false, "Unexpected arch %s detected when stopping remote chip!", tt::get_string_lowercase(arch));
            break;
        }
    }
}

void tt_cluster::check_timeout(std::string output_dir) {
    if (get_device_duration() > get_device_timeout()) {
        cout << __FUNCTION__ << " @ " << get_device_duration().count() << "s exceeded TIMEOUT " << get_device_timeout().count() << "s" << endl;
        dump_debug_mailbox(output_dir);
        TT_ASSERT(false ,  "Device TIMEOUT reached, possible hang is detected!");
    }
}

void tt_cluster::dump_debug_mailbox(std::string output_dir) {
    TT_ASSERT(device != nullptr, "Device not initialized, make sure compile is done before running!");
    if (output_dir.find("tt_build") != std::string::npos) {
        for (auto device_id: target_device_ids) {
            string output_path = output_dir + "/debug_mailbox_device_";
            output_path += to_string(device_id) + ".yaml";
            tt::log_info(tt::LogLLRuntime, "Reading debug mailbox for device {}, output yaml path {}", device_id, output_path);
            std::ofstream output_file(output_path);

            std::vector<std::string> debug_mailboxes = {"T0", "T1", "T2", "Ncrisc"};

            const int mailbox_base_addr = MEM_DEBUG_MAILBOX_ADDRESS;
            const int mailbox_size = MEM_DEBUG_MAILBOX_SIZE;
            for (auto &worker_core : get_soc_desc(device_id).workers) {
                int core_x = worker_core.x;
                int core_y = worker_core.y;
                std::string core_id = std::to_string(core_x) + "-" + std::to_string(core_y);
                output_file << core_id << ":" << std::endl;
                int thread_idx = 0;
                for (auto thread : debug_mailboxes) {
                    output_file << "    " << thread << ":" << std::endl;
                    const int mailbox_thread_base_addr = mailbox_base_addr + thread_idx * mailbox_size;
                    std::vector<uint32_t> mailbox_events;
                    read_dram_vec(
                        mailbox_events, tt_cxy_pair(device_id, core_x, core_y), mailbox_thread_base_addr, mailbox_size);
                    thread_idx++;
                    // Number of events returned must be the mailbox size divided by event size (4B)
                    assert(mailbox_events.size() == mailbox_size / 4);
                    for (auto event : mailbox_events) {
                        // The debug mailbox registers are 16b each
                        output_file << "        - " << (event & 0xffff) << std::endl;
                        output_file << "        - " << ((event >> 16) & 0xffff) << std::endl;
                    }
                }
            }
        }
    }
}

inline uint64_t get_sys_addr(uint32_t chip_x, uint32_t chip_y, uint32_t noc_x, uint32_t noc_y, uint64_t offset) {
    uint64_t result = chip_y;
    uint64_t noc_addr_local_bits_mask = (1UL << NOC_ADDR_LOCAL_BITS) - 1;
    result <<= NOC_ADDR_NODE_ID_BITS;
    result |= chip_x;
    result <<= NOC_ADDR_NODE_ID_BITS;
    result |= noc_y;
    result <<= NOC_ADDR_NODE_ID_BITS;
    result |= noc_x;
    result <<= NOC_ADDR_LOCAL_BITS;
    result |= (noc_addr_local_bits_mask & offset);
    return result;
}

void tt_cluster::write_to_non_mmio_device(const uint32_t *mem_ptr, uint32_t len, tt_cxy_pair core, uint64_t address) {
    using data_word_t = uint32_t;
    constexpr int DATA_WORD_SIZE = sizeof(data_word_t);
    constexpr int COMMAND_QUEUE_SIZE = sizeof(cmd_q_t);
    constexpr uint32_t REQUEST_CMD_QUEUE_BASE = ETH_ROUTING_STRUCT_ADDR;
    constexpr uint32_t REQUEST_ROUTING_CMD_QUEUE_BASE = REQUEST_CMD_QUEUE_BASE + sizeof(remote_update_ptr_t) + sizeof(remote_update_ptr_t);
    constexpr uint32_t RESPONSE_CMD_QUEUE_BASE = ETH_ROUTING_STRUCT_ADDR + sizeof(cmd_q_t);

    const chip_id_t &mmio_capable_chip = ndesc->get_closest_mmio_capable_chip(core.chip);
    const tt_cxy_pair remote_transfer_ethernet_core = tt_cxy_pair(mmio_capable_chip, get_soc_desc(core.chip).ethernet_cores.at(0).x, get_soc_desc(core.chip).ethernet_cores.at(0).y);
    const CoreCoord target_chip = ndesc->get_chip_locations().at(core.chip);
    // tt::log_debug(tt::LogDevice, "Writing to non-mmio device {}: tt_cxy_pair {}, addr {}", target_chip.str(), core.str(), address);

    std::vector<std::uint32_t> erisc_req_q;
    std::vector<std::uint32_t> erisc_resp_q;
    std::vector<std::uint32_t> erisc_command;
    std::vector<std::uint32_t> erisc_q_ptr;
    std::vector<std::uint32_t> data_block;

    cmd_q_t *request_command_q;
    cmd_q_t *response_command_q;

    uint32_t size_in_bytes = len * DATA_WORD_SIZE;

    device->read_vector(erisc_req_q, remote_transfer_ethernet_core, ETH_ROUTING_STRUCT_ADDR, COMMAND_QUEUE_SIZE);
    device->read_vector(erisc_resp_q, remote_transfer_ethernet_core, ETH_ROUTING_STRUCT_ADDR + COMMAND_QUEUE_SIZE, COMMAND_QUEUE_SIZE);
    request_command_q = (cmd_q_t *)&erisc_req_q[0];
    response_command_q = (cmd_q_t *)&erisc_resp_q[0];

    uint32_t offset = 0;
    uint32_t block_size;
    while (offset < size_in_bytes) {
        uint32_t req_wr_ptr = request_command_q->wrptr.ptr & CMD_BUF_SIZE_MASK;
        if ((address + offset) & 0x1F) { // address not 32-byte aligned
            block_size = DATA_WORD_SIZE;
        } else {
            block_size = offset + MAX_BLOCK_SIZE > size_in_bytes ? size_in_bytes - offset : MAX_BLOCK_SIZE;
        }
        uint32_t req_flags = block_size > DATA_WORD_SIZE ? (CMD_DATA_BLOCK | CMD_WR_REQ) : CMD_WR_REQ;
        uint32_t resp_flags = block_size > DATA_WORD_SIZE ? (CMD_DATA_BLOCK | CMD_WR_ACK) : CMD_WR_ACK;
        if (req_flags & CMD_DATA_BLOCK) {
            uint32_t buf_address = ETH_ROUTING_DATA_BUFFER_ADDR + req_wr_ptr * MAX_BLOCK_SIZE;
            data_block.resize(block_size/DATA_WORD_SIZE);
            memcpy(&data_block[0], mem_ptr + offset/DATA_WORD_SIZE, block_size);
            device->write_vector (
                data_block,
                remote_transfer_ethernet_core,
                buf_address
            );
            _mm_sfence();
      }
        // Send the read request
        TT_ASSERT((req_flags == CMD_WR_REQ) || (((address + offset) & 0x1F) == 0)); // Block mode address must be 32-byte aligned.
        request_command_q->cmd[req_wr_ptr].sys_addr =
            get_sys_addr(target_chip.x, target_chip.y, core.x, core.y, address + offset);
        request_command_q->cmd[req_wr_ptr].data = req_flags & CMD_DATA_BLOCK ? block_size : *(mem_ptr + offset/DATA_WORD_SIZE);
        request_command_q->cmd[req_wr_ptr].flags = req_flags;

        erisc_command.resize(sizeof(routing_cmd_t)/DATA_WORD_SIZE);
        memcpy(&erisc_command[0], &request_command_q->cmd[req_wr_ptr], sizeof(routing_cmd_t));

        device->write_vector (
            erisc_command,
            remote_transfer_ethernet_core,
            REQUEST_ROUTING_CMD_QUEUE_BASE + (sizeof(routing_cmd_t) * req_wr_ptr)
        );
        _mm_sfence();
        request_command_q->wrptr.ptr = (request_command_q->wrptr.ptr + 1) & CMD_BUF_PTR_MASK;
        erisc_q_ptr.resize(sizeof(remote_update_ptr_t)/DATA_WORD_SIZE);
        memcpy(&erisc_q_ptr[0], &request_command_q->wrptr, sizeof(remote_update_ptr_t));
        device->write_vector (
            erisc_q_ptr,
            remote_transfer_ethernet_core,
            REQUEST_CMD_QUEUE_BASE
        );
        _mm_sfence();

        // Wait for read request completion and extract the data into the `ptr`
        uint32_t resp_rd_ptr;
        do {
            device->read_vector (
                erisc_resp_q,
                remote_transfer_ethernet_core,
                RESPONSE_CMD_QUEUE_BASE,
                sizeof(cmd_q_t)
            );
            response_command_q = (cmd_q_t *)&erisc_resp_q[0];
            resp_rd_ptr = response_command_q->rdptr.ptr & CMD_BUF_SIZE_MASK;
        } while (response_command_q->cmd[resp_rd_ptr].flags != resp_flags);

        // Finally increment the rdptr for the response command q
        response_command_q->rdptr.ptr = (response_command_q->rdptr.ptr + 1) & CMD_BUF_PTR_MASK;
        erisc_q_ptr.resize(sizeof(remote_update_ptr_t)/DATA_WORD_SIZE);
        memcpy(&erisc_q_ptr[0], &response_command_q->rdptr, sizeof(remote_update_ptr_t));
        device->write_vector (
            erisc_q_ptr,
            remote_transfer_ethernet_core,
            RESPONSE_CMD_QUEUE_BASE + sizeof(remote_update_ptr_t)
        );
        _mm_sfence();

        offset += block_size;
    }
}

void tt_cluster::read_from_non_mmio_device(uint32_t *mem_ptr, tt_cxy_pair core, uint64_t address, uint32_t size_in_bytes) {
    using data_word_t = uint32_t;
    constexpr int DATA_WORD_SIZE = sizeof(data_word_t);
    constexpr int COMMAND_QUEUE_SIZE = sizeof(cmd_q_t);
    constexpr uint32_t REQUEST_CMD_QUEUE_BASE = ETH_ROUTING_STRUCT_ADDR;
    constexpr uint32_t REQUEST_ROUTING_CMD_QUEUE_BASE = REQUEST_CMD_QUEUE_BASE + sizeof(remote_update_ptr_t) + sizeof(remote_update_ptr_t);
    constexpr uint32_t RESPONSE_CMD_QUEUE_BASE = ETH_ROUTING_STRUCT_ADDR + sizeof(cmd_q_t);

    const chip_id_t &mmio_capable_chip = ndesc->get_closest_mmio_capable_chip(core.chip);
    const tt_cxy_pair remote_transfer_ethernet_core = tt_cxy_pair(mmio_capable_chip, get_soc_desc(core.chip).ethernet_cores.at(0).x, get_soc_desc(core.chip).ethernet_cores.at(0).y);
    const CoreCoord target_chip = ndesc->get_chip_locations().at(core.chip);
    // tt::log_debug(tt::LogDevice, "Reading from non-mmio device {}: tt_cxy_pair {}, addr {}", target_chip.str(), core.str(), address);

    std::vector<std::uint32_t> erisc_req_q;
    std::vector<std::uint32_t> erisc_resp_q;
    std::vector<std::uint32_t> erisc_command;
    std::vector<std::uint32_t> erisc_q_ptr;
    std::vector<std::uint32_t> data_block;

    cmd_q_t *request_command_q;
    cmd_q_t *response_command_q;

    device->read_vector(erisc_req_q, remote_transfer_ethernet_core, ETH_ROUTING_STRUCT_ADDR, COMMAND_QUEUE_SIZE);
    device->read_vector(erisc_resp_q, remote_transfer_ethernet_core, ETH_ROUTING_STRUCT_ADDR + COMMAND_QUEUE_SIZE, COMMAND_QUEUE_SIZE);
    request_command_q = (cmd_q_t *)&erisc_req_q[0];
    response_command_q = (cmd_q_t *)&erisc_resp_q[0];


    uint32_t offset = 0;
    uint32_t block_size;
    while (offset < size_in_bytes) {
        uint32_t req_wr_ptr = request_command_q->wrptr.ptr & CMD_BUF_SIZE_MASK;
        if ((address + offset) & 0x1F) { // address not 32-byte aligned
            block_size = DATA_WORD_SIZE;
        } else {
            block_size = offset + MAX_BLOCK_SIZE > size_in_bytes ? size_in_bytes - offset : MAX_BLOCK_SIZE;
        }

        uint32_t req_flags = block_size > DATA_WORD_SIZE ? (CMD_DATA_BLOCK | CMD_RD_REQ) : CMD_RD_REQ;
        uint32_t resp_flags = block_size > DATA_WORD_SIZE ? (CMD_DATA_BLOCK | CMD_RD_DATA) : CMD_RD_DATA;
        // Send the read request
        TT_ASSERT((req_flags == CMD_RD_REQ) || (((address + offset) & 0x1F) == 0)); // Block mode offset must be 32-byte aligned.
        request_command_q->cmd[req_wr_ptr].sys_addr =
            get_sys_addr(target_chip.x, target_chip.y, core.x, core.y, address + offset);
        request_command_q->cmd[req_wr_ptr].data = block_size;
        request_command_q->cmd[req_wr_ptr].flags = req_flags;

        erisc_command.resize(sizeof(routing_cmd_t)/DATA_WORD_SIZE);
        memcpy(&erisc_command[0], &request_command_q->cmd[req_wr_ptr], sizeof(routing_cmd_t));

        device->write_vector (
            erisc_command,
            remote_transfer_ethernet_core,
            REQUEST_ROUTING_CMD_QUEUE_BASE + (sizeof(routing_cmd_t) * req_wr_ptr)
        );
        _mm_sfence();

        request_command_q->wrptr.ptr = (request_command_q->wrptr.ptr + 1) & CMD_BUF_PTR_MASK;
        erisc_q_ptr.resize(sizeof(remote_update_ptr_t)/DATA_WORD_SIZE);
        memcpy(&erisc_q_ptr[0], &request_command_q->wrptr, sizeof(remote_update_ptr_t));

        device->write_vector (
            erisc_q_ptr,
            remote_transfer_ethernet_core,
            REQUEST_CMD_QUEUE_BASE
        );
        _mm_sfence();
        // Wait for read request completion and extract the data into the `mem_ptr`
        uint32_t resp_rd_ptr;
        do {
            device->read_vector (
                erisc_resp_q,
                remote_transfer_ethernet_core,
                RESPONSE_CMD_QUEUE_BASE,
                sizeof(cmd_q_t)
            );
            response_command_q = (cmd_q_t *)&erisc_resp_q[0];
            resp_rd_ptr = response_command_q->rdptr.ptr & CMD_BUF_SIZE_MASK;
        } while (response_command_q->cmd[resp_rd_ptr].flags != resp_flags);
        if (block_size == DATA_WORD_SIZE) {
            mem_ptr[offset/DATA_WORD_SIZE] = response_command_q->cmd[resp_rd_ptr].data;
        } else {
            uint32_t buf_address = ETH_ROUTING_DATA_BUFFER_ADDR + resp_rd_ptr * MAX_BLOCK_SIZE;
            device->read_vector(data_block, remote_transfer_ethernet_core, buf_address, block_size);
            memcpy(&mem_ptr[offset/DATA_WORD_SIZE], data_block.data(), block_size);
        }

        // Finally increment the rdptr for the response command q
        response_command_q->rdptr.ptr = (response_command_q->rdptr.ptr + 1) & CMD_BUF_PTR_MASK;
        erisc_q_ptr.resize(sizeof(remote_update_ptr_t)/DATA_WORD_SIZE);
        memcpy(&erisc_q_ptr[0], &response_command_q->rdptr, sizeof(remote_update_ptr_t));
        device->write_vector (
            erisc_q_ptr,
            remote_transfer_ethernet_core,
            RESPONSE_CMD_QUEUE_BASE + sizeof(remote_update_ptr_t)
        );
        _mm_sfence();

        offset += block_size;

    }
}

void tt_cluster::write_dram_vec(vector<uint32_t> &vec, tt_target_dram dram, uint64_t addr, bool small_access)
{
    int chip_id, d_chan, d_subchannel;
    std::tie(chip_id, d_chan, d_subchannel) = dram;
    tt_SocDescriptor& desc_to_use = get_soc_desc(chip_id);
    TT_ASSERT(d_chan < desc_to_use.dram_cores.size(), "Trying to address dram channel that doesnt exist in the device descriptor");
    TT_ASSERT(d_subchannel < desc_to_use.dram_cores.at(d_chan).size(), "Trying to address dram sub channel that doesnt exist in the device descriptor");
    tt_cxy_pair dram_core = tt_cxy_pair(chip_id, desc_to_use.get_core_for_dram_channel(d_chan, d_subchannel));
    size_t offset = desc_to_use.get_address_offset(d_chan);
    write_dram_vec(vec, dram_core, addr + offset, small_access);
}

void tt_cluster::read_dram_vec(vector<uint32_t> &vec, tt_target_dram dram, uint64_t addr, uint32_t size, bool small_access)
{
    int chip_id, d_chan, d_subchannel;
    std::tie(chip_id, d_chan, d_subchannel) = dram;
    tt_SocDescriptor& desc_to_use = get_soc_desc(chip_id);
    TT_ASSERT(d_chan < desc_to_use.dram_cores.size(), "Trying to address dram channel that doesnt exist in the device descriptor");
    TT_ASSERT(d_subchannel < desc_to_use.dram_cores.at(d_chan).size(), "Trying to address dram sub channel that doesnt exist in the device descriptor");
    tt_cxy_pair dram_core = tt_cxy_pair(chip_id, desc_to_use.get_core_for_dram_channel(d_chan, d_subchannel));
    size_t offset = desc_to_use.get_address_offset(d_chan);
    read_dram_vec(vec, dram_core, addr + offset, size, small_access);
}

void tt_cluster::write_dram_vec(const std::uint32_t *mem_ptr, uint32_t len, tt_cxy_pair dram_core, uint64_t addr, bool small_access)
{
    int chip_id = dram_core.chip;
    bool target_is_mmio_capable = ndesc->is_chip_mmio_capable(dram_core.chip);
    if (target_is_mmio_capable) {
        constexpr bool host_resident = false;
        device->write_vector(mem_ptr, len, dram_core, addr, host_resident, small_access);
    } else {
        TT_ASSERT((get_soc_desc(chip_id).ethernet_cores).size() > 0 && get_num_chips() > 1);
        write_to_non_mmio_device(mem_ptr, len, dram_core, addr);
    }
}

void tt_cluster::write_dram_vec(vector<uint32_t> &vec, tt_cxy_pair dram_core, uint64_t addr, bool small_access)
{
    write_dram_vec(&vec[0], vec.size(), dram_core, addr, small_access);
}

void tt_cluster::read_dram_vec(std::uint32_t *mem_ptr, tt_cxy_pair dram_core, uint64_t addr, uint32_t size_in_bytes, bool small_access)
{
    int chip_id = dram_core.chip;
    bool target_is_mmio_capable = ndesc->is_chip_mmio_capable(dram_core.chip);
    if (target_is_mmio_capable || type == TargetDevice::Versim) {
        constexpr bool host_resident = false;
        device->read_vector(mem_ptr, dram_core, addr, size_in_bytes, host_resident, small_access);
    } else {
        TT_ASSERT((get_soc_desc(chip_id).ethernet_cores).size() > 0 && get_num_chips() > 1);
        read_from_non_mmio_device(mem_ptr, dram_core, addr, size_in_bytes);
    }
}

void tt_cluster::read_dram_vec(vector<uint32_t> &vec, tt_cxy_pair dram_core, uint64_t addr, uint32_t size_in_bytes, bool small_access)
{
    vec.resize(size_in_bytes / sizeof(uint32_t));
    read_dram_vec(&vec[0], dram_core, addr, size_in_bytes, small_access);
}

void tt_cluster::write_sysmem_vec(vector<uint32_t> &vec, uint64_t addr, chip_id_t src_device_id)
{
    constexpr bool host_resident = true;
    constexpr bool small_access = false;
    device->write_vector(vec, {}, addr, host_resident, small_access, src_device_id);
}

void tt_cluster::read_sysmem_vec(vector<uint32_t> &vec, uint64_t addr, uint32_t size, chip_id_t src_device_id)
{
    constexpr bool host_resident = true;
    constexpr bool small_access = false;
    device->read_vector(vec, {}, addr, size, host_resident, small_access, src_device_id);
}

void *tt_cluster::channel_0_address(std::uint32_t offset, std::uint32_t device_id) const {
    TT_ASSERT(ndesc->is_chip_mmio_capable(device_id), "Cannot call channel_0_address for non-MMIO device");
    return device->channel_0_address(offset, device_id);
}

void *tt_cluster::host_dma_address(std::uint64_t offset, chip_id_t src_device_id) const {
    return device->host_dma_address(offset, src_device_id);
}

void tt_cluster::verify_sw_fw_versions(
    int device_id, std::uint32_t sw_version, std::vector<std::uint32_t> &fw_versions) {
    tt_version sw(sw_version), fw_first_eth_core(fw_versions.at(0));
    tt::log_info(
        tt::LogDevice,
        "Software version {}, Ethernet FW version {} (Device {})",
        sw.str(),
        fw_first_eth_core.str(),
        device_id);
    for (std::uint32_t &fw_version : fw_versions) {
        tt_version fw(fw_version);

        TT_ASSERT(fw == fw_first_eth_core, "FW versions are not the same across different ethernet cores");
        TT_ASSERT(sw.major == fw.major, "SW/FW major version number out of sync");
        TT_ASSERT(sw.minor <= fw.minor, "SW version is newer than FW version");
    }
}

std::ostream &operator<<(std::ostream &os, tt_target_dram const &dram) {
    os << "Target DRAM chip = " << std::get<0>(dram) << ", chan = " << std::get<1>(dram) << ", subchan = " << std::get<2>(dram);
    return os;
}

std::unique_ptr<tt_soc_description> load_soc_descriptor_from_file(const tt::ARCH &arch, std::string file_path) {
    TT_ASSERT(file_path != "", "soc-descriptor file path must be populated");
    return load_soc_descriptor_from_yaml(file_path);
}

bool check_dram_core_exists(const std::vector<std::vector<CoreCoord>> &all_dram_cores, CoreCoord target_core) {
    bool dram_core_exists = false;
    for (const auto &dram_cores_in_channel : all_dram_cores) {
        for (auto dram_core : dram_cores_in_channel) {
            if (dram_core.x == target_core.x && dram_core.y == target_core.y) {
                return true;
            }
        }
    }
    return false;
}

void tt_cluster::on_destroy(tt_cluster_on_destroy_callback cb) {
    on_destroy_callbacks.push_back(cb);
}

void tt_cluster::on_close_device(tt_cluster_on_close_device_callback cb) {
    on_close_device_callbacks.push_back(cb);
}
