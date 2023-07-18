#pragma once

#include <boost/interprocess/sync/named_mutex.hpp>
#include <cassert>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

#include "common/tt_cluster_descriptor.h"
#include "common/tt_soc_descriptor.h"
#include "common/core_coord.h"
#include "device/kmdif.h"
#include "eth_l1_address_map.h"
#include "tt_silicon_driver_common.hpp"
#include "common/logger.hpp"

enum tt_DevicePowerState {
    BUSY,
    SHORT_IDLE,
    LONG_IDLE
};

enum tt_MutexType {
    LARGE_READ_TLB,
    LARGE_WRITE_TLB,
    SMALL_READ_WRITE_TLB,
    ARC_MSG
};

inline std::ostream &operator <<(std::ostream &os, const tt_DevicePowerState power_state) {
    switch (power_state) {
        case tt_DevicePowerState::BUSY: os << "Busy"; break;
        case tt_DevicePowerState::SHORT_IDLE: os << "SHORT_IDLE"; break;
        case tt_DevicePowerState::LONG_IDLE: os << "LONG_IDLE"; break;
        default: throw ("Unknown DevicePowerState");
    }
    return os;
}

struct tt_device_params {
    bool register_monitor = false;
    bool enable_perf_scoreboard = false;
    std::vector<std::string> vcd_dump_cores;
    std::vector<std::string> plusargs;
    bool init_device = true;
    bool early_open_device = false;
    bool skip_driver_allocs = false;

    // The command-line input for vcd_dump_cores can have the following format:
    // {"*-2", "1-*", "*-*", "1-2"}
    // '*' indicates we must dump all the cores in that dimension.
    // This function takes the vector above and unrolles the coords with '*' in one or both dimensions.
    std::vector<std::string> unroll_vcd_dump_cores(CoreCoord grid_size) const {
        std::vector<std::string> unrolled_dump_core;
        for (auto &dump_core: vcd_dump_cores) {
            // If the input is a single *, then dump all cores.
            if (dump_core == "*") {
                for (size_t x = 0; x < grid_size.x; x++) {
                for (size_t y = 0; y < grid_size.y; y++) {
                    std::string current_core_coord(std::to_string(x) + "-" + std::to_string(y));
                    if (std::find(std::begin(unrolled_dump_core), std::end(unrolled_dump_core), current_core_coord) == std::end(unrolled_dump_core)) {
                        unrolled_dump_core.push_back(current_core_coord);
                    }
                }
                }
                continue;
            }
            // Each core coordinate must contain three characters: "core.x-core.y".
            assert(dump_core.size() <= 5);
            size_t delimiter_pos = dump_core.find('-');
            assert (delimiter_pos != std::string::npos); // y-dim should exist in core coord.

            std::string core_dim_x = dump_core.substr(0, delimiter_pos);
            size_t core_dim_y_start = delimiter_pos + 1;
            std::string core_dim_y = dump_core.substr(core_dim_y_start, dump_core.length() - core_dim_y_start);

            if (core_dim_x == "*" && core_dim_y == "*") {
                for (size_t x = 0; x < grid_size.x; x++) {
                    for (size_t y = 0; y < grid_size.y; y++) {
                        std::string current_core_coord(std::to_string(x) + "-" + std::to_string(y));
                        if (std::find(std::begin(unrolled_dump_core), std::end(unrolled_dump_core), current_core_coord) == std::end(unrolled_dump_core)) {
                            unrolled_dump_core.push_back(current_core_coord);
                        }
                    }
                }
            } else if (core_dim_x == "*") {
                for (size_t x = 0; x < grid_size.x; x++) {
                    std::string current_core_coord(std::to_string(x) + "-" + core_dim_y);
                    if (std::find(std::begin(unrolled_dump_core), std::end(unrolled_dump_core), current_core_coord) == std::end(unrolled_dump_core)) {
                        unrolled_dump_core.push_back(current_core_coord);
                    }
                }
            } else if (core_dim_y == "*") {
                for (size_t y = 0; y < grid_size.y; y++) {
                    std::string current_core_coord(core_dim_x + "-" + std::to_string(y));
                    if (std::find(std::begin(unrolled_dump_core), std::end(unrolled_dump_core), current_core_coord) == std::end(unrolled_dump_core)) {
                        unrolled_dump_core.push_back(current_core_coord);
                    }
                }
            } else {
                unrolled_dump_core.push_back(dump_core);
            }
        }
        return unrolled_dump_core;
    }

    std::vector<std::string> expand_plusargs() const {
        std::vector<std::string> all_plusargs {
            "+enable_perf_scoreboard=" + std::to_string(enable_perf_scoreboard),
            "+register_monitor=" + std::to_string(register_monitor)
        };

        all_plusargs.insert(all_plusargs.end(), plusargs.begin(), plusargs.end());

        return all_plusargs;
    }
};

class tt_device
{
    public:

    tt_device(const tt_SocDescriptor &soc_descriptor);
    tt_device(std::unordered_map<chip_id_t, tt_SocDescriptor> soc_descriptor_per_chip);
    virtual ~tt_device();
    //! Starts device
    /*!
    \param device Reference to device object that is to be started
    \param root root folder (git root)
    \param dump_cores vector of cores to dump in simulation
    */
    virtual void start(
        std::vector<std::string> plusargs,
        std::vector<std::string> dump_cores,
        bool no_checkers,
        bool init_device,
        bool skip_driver_allocs);

    //! Deassert brisc reset so it's starts fetching the code
    virtual void deassert_risc_reset(bool start_stagger = false);

    //! Assert brisc reset
    virtual void assert_risc_reset();

    //! Stops and teardowns device
    virtual bool stop();

    //! write vector to specific core -- address is byte address
    /*!
        \param mem_ptr is a pointer into the vector to write.
        \param target is xy coordinate that is the target of the read or write
        \param address is byte address
        \param len is length in words
        \param host_resident is true if this write should be directed towards host dma buffer
    */
    virtual void write_vector(const std::uint32_t *mem_ptr, std::uint32_t len, tt_cxy_pair target, std::uint32_t address, bool host_resident = false, bool small_access = false, chip_id_t src_device_id = -1);

    //! write vector to specific core -- address is byte address
    /*!
        \param mem_vector is the vector to write.
        \param target is xy coordinate that is the target of the read or write
        \param address is byte address
        \param host_resident is true if this write should be directed towards host dma buffer
    */
    virtual void write_vector(std::vector<std::uint32_t> &mem_vector, tt_cxy_pair target, std::uint32_t address, bool host_resident = false, bool small_access = false, chip_id_t src_device_id = -1);

    //! read vector from specific core -- address is byte address
    /*!
        \param mem_ptr is a pointer into the vector to read
        \param target is xy coordinate that is the target of the read or write
        \param address is byte address
        \param size_in_bytes is number of bytes to read
        \param host_resident is true if this write should be directed towards host dma buffer
    */
    virtual void read_vector(
        std::uint32_t *mem_ptr, tt_cxy_pair target, std::uint32_t address, std::uint32_t size_in_bytes, bool host_resident = false, bool small_access = false, chip_id_t src_device_id = -1);

    //! read vector from specific core -- address is byte address
    /*!
        \param mem_vector is the vector to read
        \param target is xy coordinate that is the target of the read or write
        \param address is byte address
        \param size_in_bytes is number of bytes to read
        \param host_resident is true if this write should be directed towards host dma buffer
    */
    virtual void read_vector(
        std::vector<std::uint32_t> &mem_vector, tt_cxy_pair target, std::uint32_t address, std::uint32_t size_in_bytes, bool host_resident = false, bool small_access = false, chip_id_t src_device_id = -1);

    //! return the size of host allocated dma buffers
    virtual uint32_t dma_allocation_size(chip_id_t src_device_id = -1);

    //! Simple test of communication to device/target.  true if it passes.
    virtual bool test_write_read(tt_cxy_pair target);


    //! converts a device address user space pointer if possible; if not, returns a nullptr
    virtual uint32_t * get_usr_ptr(uint32_t d_addr, chip_id_t src_device_id = -1);

    const tt_SocDescriptor *get_soc_descriptor() const;

    virtual void dump_debug_mailbox(std::string output_path, int device_id);
    virtual void dump_wall_clock_mailbox(std::string output_path, int device_id);
    virtual int get_number_of_chips() = 0;
    virtual uint32_t get_harvested_noc_rows(int logical_device_id){
        std::runtime_error("---- tt_device:get_harvested_noc_rows is not implemented\n");
        return 0;
    }
    virtual bool get_dma_buffer(void **mapping, std::uint64_t *physical, std::size_t *size, chip_id_t src_device_id) const;

    virtual void *channel_0_address(std::uint32_t offset, std::uint32_t device_id) const;
    virtual void *host_dma_address(std::uint64_t offset, chip_id_t src_device_id) const;

    virtual void set_power_state(tt_DevicePowerState state) {
        std::runtime_error("---- tt_device::set_power_state is not implemented\n");
    }
    // Get the minimum and maximum values of aiclk (in MHz).
    virtual std::pair<float, float> get_clock_range() {
        std::runtime_error("---- tt_device::get_clock_range is not implemented\n");
        return {0.0, 0.0};
    }
    // Force the chip aiclk to the given value within the tolerance (in MHz).
    virtual void set_device_aiclk(int clock, float tolerance = 1.0) {
        std::runtime_error("---- tt_device::set_device_aiclk is not implemented\n");
    }
    // Restore ARC control of the aiclk.
    virtual void reset_device_aiclk() {
        std::runtime_error("---- tt_device::reset_device_aiclk is not implemented\n");
    }
    virtual std::map<int,int> get_clocks() {
        std::runtime_error("---- tt_device::get_clocks is not implemented\n");
        return std::map<int,int>();
    }
    virtual void init_system(const tt_device_params &device_params, const CoreCoord &grid_size) {
        bool no_checkers = true;
        std::vector<std::string> dump_cores = device_params.unroll_vcd_dump_cores(grid_size);
        start(device_params.expand_plusargs(), dump_cores, no_checkers, device_params.init_device, device_params.skip_driver_allocs);
        set_power_state(tt_DevicePowerState::BUSY);
    }

    virtual void shutdown_system() {
        stop();
    }

    virtual void clean_system_resources(){
        std::runtime_error("---- tt_device::clean_system_resources is not implemented\n");
    }

    virtual void enable_ethernet_queue(const chip_id_t &logical_device_id, int timeout) {
        throw std::runtime_error("---- tt_device::enable_ethernet_queue is not implemented\n");
    }

    protected:

    const tt_SocDescriptor soc_descriptor;
    std::unordered_map<chip_id_t, tt_SocDescriptor> soc_descriptor_per_chip = {};

};

class c_versim_core;
#ifndef TT_METAL_VERSIM_DISABLED
namespace nuapi {namespace device {template <typename, typename>class Simulator;}}
#endif
namespace versim {
  struct VersimSimulatorState;
  #ifndef TT_METAL_VERSIM_DISABLED
  using VersimSimulator = nuapi::device::Simulator<c_versim_core *, VersimSimulatorState>;
  #endif
}

class tt_VersimDevice: public tt_device
{
    public:
     tt_VersimDevice(const tt_SocDescriptor &soc_descriptor_);

     virtual void start(std::vector<std::string> plusargs, std::vector<std::string> dump_cores, bool no_checkers, bool init_device, bool skip_driver_allocs);

     virtual void deassert_risc_reset(bool start_stagger = false);
     virtual void assert_risc_reset();
     virtual bool stop();
     virtual void write_vector(
        const std::uint32_t *mem_ptr,
        uint32_t len,
        tt_cxy_pair target,
        std::uint32_t address,
        bool host_resident = false,
        bool small_access = false,
        chip_id_t src_device_id = -1);

     virtual void write_vector(
        std::vector<std::uint32_t> &mem_vector,
        tt_cxy_pair target,
        std::uint32_t address,
        bool host_resident = false,
        bool small_access = false,
        chip_id_t src_device_id = -1);

     virtual void read_vector(
         std::uint32_t *mem_ptr,
         tt_cxy_pair target,
         std::uint32_t address,
         std::uint32_t size_in_bytes,
         bool host_resident = false,
         bool small_access = false,
         chip_id_t src_device_id = -1);
     virtual void read_vector(
         std::vector<std::uint32_t> &mem_vector,
         tt_cxy_pair target,
         std::uint32_t address,
         std::uint32_t size_in_bytes,
         bool host_resident = false,
         bool small_access = false,
         chip_id_t src_device_id = -1);

     virtual ~tt_VersimDevice();
     virtual void dump_debug_mailbox(std::string output_path, int device_id);
     virtual void dump_wall_clock_mailbox(std::string output_path, int device_id);
     virtual int get_number_of_chips();
     static int detect_number_of_chips();
     virtual std::map<int,int> get_clocks();
    private:

    #ifndef TT_METAL_VERSIM_DISABLED
    versim::VersimSimulator* versim;
    #endif
    void* p_ca_soc_manager;
};


class tt_SiliconDevice: public tt_device
{
    friend class DebudaIFC; // Allowing access for Debuda
    public:
    tt_SiliconDevice(const tt_SocDescriptor &soc_descriptor_, const std::unordered_set<chip_id_t> &target_mmio_device_ids, const bool skip_driver_allocs = false);
    tt_SiliconDevice(std::unordered_map<chip_id_t, tt_SocDescriptor> soc_descriptor_map_, const std::unordered_set<chip_id_t> &target_mmio_device_ids, const bool skip_driver_allocs = false);
    void create_device(const std::unordered_set<chip_id_t> &target_mmio_device_ids, const bool skip_driver_allocs);
    virtual void start(
        std::vector<std::string> plusargs,
        std::vector<std::string> dump_cores,
        bool no_checkers,
        bool init_device,
        bool skip_driver_allocs);

    virtual void deassert_risc_reset(bool start_stagger = false);
    virtual void assert_risc_reset();

    virtual bool stop();

    virtual void write_vector(
        std::vector<std::uint32_t> &mem_vector,
        tt_cxy_pair target,
        std::uint32_t address,
        bool host_resident = false,
        bool small_access = false,
        chip_id_t src_device_id = -1);

    virtual void write_vector(
        const uint32_t *mem_ptr,
        std::uint32_t len,
        tt_cxy_pair target,
        std::uint32_t address,
        bool host_resident = false,
        bool small_access = false,
        chip_id_t src_device_id = -1);

    virtual void read_vector (
        std::vector<std::uint32_t> &mem_vector,
        tt_cxy_pair target,
        std::uint32_t address,
        std::uint32_t size_in_bytes,
        bool host_resident = false,
        bool small_access = false,
        chip_id_t src_device_id = -1);

    virtual void read_vector (
        uint32_t *mem_ptr,
        tt_cxy_pair target,
        std::uint32_t address,
        std::uint32_t size_in_bytes,
        bool host_resident = false,
        bool small_access = false,
        chip_id_t src_device_id = -1);



    virtual void set_power_state(tt_DevicePowerState state);
    virtual std::pair<float, float> get_clock_range();
    virtual void set_device_aiclk(int clock, float tolerance = 1.0);
    virtual void reset_device_aiclk();
    int get_clock(int logical_device_id);
    std::map<int,int> get_clocks();

    virtual bool get_dma_buffer(void **mapping, std::uint64_t *physical, std::size_t *size, chip_id_t src_device_id) const;

    //! Simple test of communication to device/target.  true if it passes.
    virtual bool test_write_read(tt_cxy_pair target);
    virtual void dump_debug_mailbox(std::string output_path, int device_id);
    virtual void dump_wall_clock_mailbox(std::string output_path, int device_id);
    virtual int get_number_of_chips();
    static int detect_number_of_chips(bool respect_reservations = false);
    static std::vector<chip_id_t> detect_available_device_ids(bool respect_reservations = false, bool verbose = false);
    static std::vector<chip_id_t> get_available_devices_from_reservations(std::vector<chip_id_t> device_ids, bool verbose);
    static std::map<chip_id_t, chip_id_t> get_logical_to_physical_mmio_device_id_map(std::vector<chip_id_t> physical_device_ids);
    static std::map<chip_id_t, std::string> get_physical_device_id_to_bus_id_map(std::vector<chip_id_t> physical_device_ids);

    virtual uint32_t dma_allocation_size(chip_id_t src_device_id = -1);

    virtual void *channel_0_address(std::uint32_t offset, std::uint32_t device_id) const;
    virtual void *host_dma_address(std::uint64_t offset, chip_id_t src_device_id) const;

    virtual uint32_t * get_usr_ptr(uint32_t d_addr, chip_id_t src_device_id = -1);

    virtual void clean_system_resources();

    virtual void enable_ethernet_queue(const chip_id_t &logical_device_id, int timeout);

    virtual void init_system(const tt_device_params &device_params, const CoreCoord &grid_size);
    virtual void shutdown_system();
    void set_tensix_risc_reset(struct PCIdevice *device, const CoreCoord &core, const TensixSoftResetOptions &cores);
    void broadcast_tensix_risc_reset(struct PCIdevice *device, const TensixSoftResetOptions &cores);
    virtual uint32_t get_harvested_noc_rows(int logical_device_id);
    virtual ~tt_SiliconDevice ();

    private:
        tt_SocDescriptor& get_soc_descriptor(chip_id_t chip_id);
        std::map<chip_id_t, struct PCIdevice*> m_pci_device_map;    // Map of enabled pci devices
        int m_num_pci_devices;                                      // Number of pci devices in system (enabled or disabled)

        // Level of printouts. Controlled by env var TT_PCI_LOG_LEVEL
        // 0: no debugging messages, 1: less verbose, 2: more verbose
        int m_pci_log_level;

        // Size of the PCIE DMA buffer
        // The setting should not exceed MAX_DMA_BYTES
        std::uint32_t m_dma_buf_size;

        // Per-Device Interprocess mutex used to lock dynamic TLBs, arc_msg(). pci_interface_id -> {name, mutex}
        std::map<tt_MutexType, std::map<int, std::pair<std::string, boost::interprocess::named_mutex*>>> m_per_device_mutexes_map;

        std::map<chip_id_t, void *> hugepage_mapping;
        std::map<chip_id_t, std::size_t> hugepage_mapping_size;
        std::map<chip_id_t, std::uint64_t> hugepage_physical_address;

        std::uint64_t buf_physical_addr = 0;
        void * buf_mapping = nullptr;
        int driver_id;


        // Utility Functions
        std::optional<std::uint64_t> get_tlb_data(std::uint32_t tlb_index, TLB_DATA data);
        std::int32_t get_static_tlb_index(CoreCoord coord);
        std::optional<std::tuple<std::uint32_t, std::uint32_t>> describe_tlb(std::int32_t tlb_index);
        std::optional<std::tuple<std::uint32_t, std::uint32_t>> describe_tlb(CoreCoord coord);

        // Setup functions
        void init_pcie_tlb(struct PCIdevice* pci_device);
        void init_pcie_iatus();
        bool init_hugepage(chip_id_t device_id);
        bool init_dmabuf(chip_id_t device_id);
        void init_device(int device_id);

        bool init_dma_turbo_buf(struct PCIdevice* pci_device);
        bool uninit_dma_turbo_buf(struct PCIdevice* pci_device);

        // Test functions. Returning 0 if all is well.
        int test_pcie_tlb_setup (struct PCIdevice* pci_device);
        int test_setup_interface ();
        int test_broadcast (int logical_device_id);
        bool test_write_speed (struct PCIdevice* pci_device);

        // Communication functions
        void bar_write32 (int logical_device_id, uint32_t addr, uint32_t data);
        uint32_t bar_read32 (int logical_device_id, uint32_t addr);
        int arc_msg(int logical_device_id, uint32_t msg_code, bool wait_for_done = true, uint32_t arg0 = 0, uint32_t arg1 = 0, int timeout=1, uint32_t *return_3 = nullptr, uint32_t *return_4 = nullptr);
        int iatu_configure_peer_region (int logical_device_id, uint32_t peer_region_id, uint64_t bar_addr_64, uint32_t region_size);
        uint32_t get_harvested_rows (int logical_device_id);

        struct PCIdevice* get_pci_device(int pci_intf_id) const;

        int open_hugepage_file(const std::string &dir, chip_id_t device_id);

        boost::interprocess::named_mutex* get_mutex(tt_MutexType type, int pci_interface_id);

        //DMA reader/writers
        void read_dma_buffer(
          std::uint32_t *mem_ptr,
          std::uint32_t address,
          std::uint32_t size_in_bytes,
          chip_id_t src_device_id);

        void write_dma_buffer(
          const std::uint32_t *mem_ptr,
          std::uint32_t len,
          std::uint32_t address,
          chip_id_t src_device_id);


        //Device memory reader/writer
        void write_device_memory(
          const std::uint32_t *mem_ptr,
          std::uint32_t len,
          tt_cxy_pair target,
          std::uint32_t address,
          bool small_access = false);

        void read_device_memory(
          std::uint32_t *mem_ptr,
          tt_cxy_pair target,
          std::uint32_t address,
          std::uint32_t size_in_bytes,
          bool small_access = false);
};

tt::ARCH detect_arch(uint16_t device_id = 0);

struct tt_version {
    std::uint16_t major;
    std::uint8_t minor;
    std::uint8_t patch;

    tt_version(std::uint32_t version) {
        major = (version >> 16) & 0xffff;
        minor = (version >> 8) & 0xff;
        patch = version & 0xff;
    }
    std::string str() const {
        return std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch);
    }
};

constexpr inline bool operator==(const tt_version &a, const tt_version &b) {
    return a.major == b.major && a.minor == b.minor && a.patch == b.patch;
}
std::string format_node(CoreCoord xy);
CoreCoord format_node(std::string str);
