
#ifdef DEBUG_PRINT
#define DEBUG_LOG(str) do { std::cout << str << std::endl; } while( false )
#else
#define DEBUG_LOG(str) do { } while ( false )
#endif

#include "tt_device.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

// #include "dram_address_map.h" // delete this?
#include "yaml-cpp/yaml.h"

// TODO: Remove dependency on command_assembler + soc
#include "command_assembler/soc.h"

namespace CA = CommandAssembler;


void translate_soc_descriptor_to_ca_soc(CA::Soc &soc, const tt_SocDescriptor soc_descriptor) {
  for (auto &core : soc_descriptor.cores) {
    CA::SocNocNode node;
    CA::xy_pair CA_coord(core.first.x, core.first.y);
    node.noc_coord = CA_coord;
    node.memory_size = core.second.l1_size;
    switch (core.second.type) {
      case CoreType::ARC: node.arc = true; break;
      case CoreType::DRAM: {
        node.dram = true;
        #ifdef EN_DRAM_ALIAS
          node.dram_channel_id = std::get<0>(soc_descriptor.dram_core_channel_map.at(core.first));
        #endif
      } break;
      case CoreType::ETH: node.eth = true; break;
      case CoreType::PCIE: node.pcie = true; break;
      case CoreType::WORKER: node.worker = true; break;
      case CoreType::HARVESTED: node.harvested = true; break;
      case CoreType::ROUTER_ONLY: node.router_only = true; break;
      default: std::cout << " Error: Unsupported CoreType type: " << static_cast<int>(core.second.type) << std::endl; break;
    }
    soc.SetNodeProperties(node.noc_coord, node);
  }
}

////////
// Device Versim
////////

#include "device.h"
#include "sim_interactive.h"
#include <command_assembler/xy_pair.h>

tt_VersimDevice::tt_VersimDevice(const tt_SocDescriptor &soc_descriptor_) : tt_device(soc_descriptor_) {}

void tt_VersimDevice::start(
    std::vector<std::string> plusargs,
    std::vector<std::string> dump_cores,
    bool no_checkers,
    bool /*init_device*/,
    bool /*skip_driver_allocs*/
    ) {

     std::cout << "Start Versim Device " << std::endl;
     std::string device_descriptor_dir = "./";

     std::optional<std::string> vcd_suffix;
     if (dump_cores.size() > 0) {
       vcd_suffix = "core_dump.vcd";
     }

     std::vector<std::string> vcd_cores;

     // TODO: For now create a temporary stuff from CA and populate from descriptor before passing back to versim-core
     // interface. mainly bypasses arch_configs etc from llir.  We can populate soc directly
     // MT: have to preserve ca_soc_descriptor object since versim references it at runtime
     CA::xy_pair CA_grid_size(soc_descriptor.grid_size.x, soc_descriptor.grid_size.y);
     // CA::Soc ca_soc_manager(CA_grid_size);
     std::unique_ptr<CA::Soc> p_ca_soc_manager_unique = std::make_unique<CA::Soc>(CA_grid_size);
     translate_soc_descriptor_to_ca_soc(*p_ca_soc_manager_unique, soc_descriptor);
     // TODO: End

     std::cout << "Versim Device: turn_on_device ";
     std::vector<std::uint32_t> trisc_sizes = {l1_mem::address_map::TRISC0_SIZE, l1_mem::address_map::TRISC1_SIZE, l1_mem::address_map::TRISC2_SIZE};
     std::unique_ptr<versim::VersimSimulator> versim_unique = versim::turn_on_device(CA_grid_size, *p_ca_soc_manager_unique, plusargs, vcd_suffix, dump_cores, no_checkers,
        l1_mem::address_map::TRISC_BASE, trisc_sizes);
     versim = versim_unique.release();

     std::cout << "Versim Device: write info to tvm db " << std::endl;
     versim::write_info_to_tvm_db(l1_mem::address_map::TRISC_BASE, trisc_sizes);
     versim::build_and_connect_tvm_phase();

     versim->spin_threads(*p_ca_soc_manager_unique, false);
     versim::assert_reset(*versim);

     p_ca_soc_manager = (void*)(p_ca_soc_manager_unique.release());

     std::cout << "Versim Device: Done start " << std::endl;
}

tt_VersimDevice::~tt_VersimDevice () {}

// bool tt_VersimDevice::run() {
//   std::cout << "Versim Device: Run " << std::endl;

//   // Run Versim main_loop
//   versim::startup_versim_main_loop(*versim);

//   return true;
// }

void tt_VersimDevice::deassert_risc_reset(bool start_stagger) {
  std::cout << "Versim Device: Deassert risc resets start" << std::endl;
  versim::handle_resetting_triscs(*versim);
  std::cout << "Versim Device: Start main loop " << std::endl;
  versim::startup_versim_main_loop(*versim);
}

void tt_VersimDevice::assert_risc_reset() {
  std::cout << "Pause all the cores" << std::endl;
  versim::pause(*versim);

  std::cout << "Wait for cores to go to paused state" << std::endl;
  versim::sleep_wait_for_paused (*versim);

  std::cout << "Assert riscv reset" << std::endl;
  versim::assert_riscv_reset(*versim);
}
// TODO(pgk): move these to ptrs instead of vectors
void tt_VersimDevice::write_vector(std::vector<uint32_t> &mem_vector, tt_cxy_pair target, std::uint32_t address, bool host_resident, bool small_access, chip_id_t src_device_id) {
  // std::cout << "Versim Device: Write vector at target core " << target.str() << ", address: " << std::hex << address << std::dec << std::endl;
  DEBUG_LOG("Versim Device (" << get_sim_time(*versim) << "): Write vector at target core " << target.str() << ", address: " << std::hex << address << std::dec);

  bool aligned_32B = soc_descriptor.cores.at(target).type == CoreType::DRAM;
  // MT: Remove these completely
  CommandAssembler::xy_pair CA_target(target.x, target.y);
  CommandAssembler::memory CA_tensor_memory(address, mem_vector);

  nuapi::device::write_memory_to_core(*versim, CA_target, CA_tensor_memory);
}

void tt_VersimDevice::read_vector(
    std::vector<uint32_t> &mem_vector, tt_cxy_pair target, std::uint32_t address, std::uint32_t size_in_bytes, bool host_resident, bool small_access, chip_id_t src_device_id) {
  // std::cout << "Versim Device: Read vector from target address: 0x" << std::hex << address << std::dec << ", with size: " << size_in_bytes << " Bytes" << std::endl;
  DEBUG_LOG("Versim Device (" << get_sim_time(*versim) << "): Read vector from target address: 0x" << std::hex << address << std::dec << ", with size: " << size_in_bytes << " Bytes");

  CommandAssembler::xy_pair CA_target(target.x, target.y);

  size_t size_in_words = size_in_bytes / 4;
  auto result = nuapi::device::read_memory_from_core(*versim, CA_target, address, size_in_words);
  mem_vector = result;
}

void tt_VersimDevice::dump_debug_mailbox(std::string output_path, int device_id) {
  std::ofstream output_file(output_path);
  printf("-Debug: Reading debug mailbox for device %d\n", device_id);

  std::vector<std::string> debug_mailboxes = {"T0", "T1", "T2", "FW"};

  const int mailbox_base_addr = l1_mem::address_map::DEBUG_MAILBOX_BUF_BASE;
  const int mailbox_size = l1_mem::address_map::DEBUG_MAILBOX_BUF_SIZE;
  for (auto &worker_core: soc_descriptor.workers) {
    int core_x = worker_core.x;
    int core_y = worker_core.y;
    std::string core_id = std::to_string(core_x) + "-" + std::to_string(core_y);
    output_file << core_id << ":" << std::endl;
    int thread_idx = 0;
    for (auto thread: debug_mailboxes) {
      output_file << "    " << thread << ":" << std::endl;
      const int mailbox_thread_base_addr = mailbox_base_addr + thread_idx * mailbox_size;
      std::vector<uint32_t> mailbox_events;
      read_vector(mailbox_events, tt_cxy_pair(device_id, core_x, core_y), mailbox_thread_base_addr, mailbox_size);
      thread_idx++;
      // Number of events returned must be the mailbox size divided by event size (4B)
      assert(mailbox_events.size() == mailbox_size / 4);
      for (auto event: mailbox_events) {
        // The debug mailbox registers are 16b each
        output_file << "        - " << (event & 0xffff) << std::endl;
        output_file << "        - " << ((event >> 16) & 0xffff) << std::endl;
      }
    }
  }
}

void tt_VersimDevice::dump_wall_clock_mailbox(std::string output_path, int device_id) {

  std::ofstream output_file(output_path);
  const int mailbox_base_addr = l1_mem::address_map::WALL_CLOCK_MAILBOX_BASE;
  const int num_mailbox_32_regs = 4;
  const int mailbox_size = num_mailbox_32_regs * 4;
  for (auto &worker_core: soc_descriptor.workers) {
      int core_x = worker_core.x;
      int core_y = worker_core.y;
      std::string core_id = std::to_string(core_x) + "-" + std::to_string(core_y);
      output_file << core_id << ":" << std::endl;

      std::vector<uint32_t> mailbox_events;
      read_vector(mailbox_events, tt_cxy_pair(device_id, core_x, core_y), mailbox_base_addr, mailbox_size);
      assert(mailbox_events.size() == num_mailbox_32_regs);
      uint64_t start_time = (uint64_t(mailbox_events[1]) << 32) + mailbox_events[0];
      uint64_t end_time = (uint64_t(mailbox_events[3]) << 32) + mailbox_events[2];
      output_file << "        " << std::left << std::setw(12) << "start: " << start_time << std::endl;
      output_file << "        " << std::left << std::setw(12) << "end: " << end_time << std::endl;
      output_file << "        " << std::left << std::setw(12) << "runtime: " << end_time - start_time << std::endl;
  }
  output_file.close();
}

bool versim_check_dram_core_exists(const std::vector<std::vector<CoreCoord>> &dram_core_channels, CoreCoord target_core) {
    bool dram_core_exists = false;
    for (const auto &dram_cores_in_channel: dram_core_channels) {
      for (const auto &dram_core : dram_cores_in_channel) {
        if (dram_core.x == target_core.x && dram_core.y == target_core.y) {
            return true;
        }
      }
    }
    return false;
}

int tt_VersimDevice::get_number_of_chips() { return detect_number_of_chips(); }

int tt_VersimDevice::detect_number_of_chips() { return 1; }

// Meant to breakout running functions for simulator
bool tt_VersimDevice::stop() {
  std::cout << "Versim Device: Stop " << std::endl;

  versim::turn_off_device(*versim);
  versim->shutdown();
  // Force free of all versim cores
  for (auto x = 0; x < versim->grid_size.x; x++) {
    for (auto y = 0; y < versim->grid_size.y; y++) {
      delete versim->core_grid.at(x).at(y);
    }
  }
  std::cout << "Versim Device: Stop completed " << std::endl;
  delete versim;
  return true;
}
