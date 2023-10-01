// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/common/concurrency_interface.hpp"

#include <fmt/ranges.h>

#include <iostream>

#include "tt_metal/common/logger.hpp"
#include "tt_metal/hostdevcommon/common_values.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"

namespace tt::concurrent {

size_t shared_memory_size() {
    size_t num_devices = tt::Cluster::instance().number_of_devices();
    size_t num_mmio_devices = tt::Cluster::instance().number_of_pci_devices();

    const uint32_t one_gb_dram = (1024 * 1024 * 1024);
    const uint32_t two_mb_l1 = (2 * 1024 * 1024);
    uint32_t per_dev_mem_allocator_size = ((two_mb_l1 / MIN_ALLOCATABLE_L1_SIZE_BYTES) * sizeof(block_t)) +
                                          ((one_gb_dram / MIN_ALLOCATABLE_DRAM_SIZE_BYTES) * sizeof(block_t)) +
                                          (2 * sizeof(allocator_t));

    // Overhead to account for variable length pid_vector_t and managed shared memory helper objects (name-object index,
    // internal synchronization objects, internal variables, etc)
    uint32_t overhead = 65536;  // "magic" overhead number taken from boost interprocess docs examples

    return (num_mmio_devices * sizeof(device_driver_t)) + (num_devices * sizeof(device_state_t)) +
           (num_devices * per_dev_mem_allocator_size) + (num_mmio_devices * sizeof(sys_mem_cq_write_interface_t)) +
           overhead;
}

const std::string shared_memory_name() {
    std::string shared_mem_str = "TtMetalSharedMemory";
    return shared_mem_str;
}

// See https://www.boost.org/doc/libs/1_82_0/doc/html/interprocess/managed_memory_segments.html for details on
// constructing managed shared memory
boost::interprocess::managed_shared_memory &get_shared_mem_segment() {
    size_t shm_size_bytes = shared_memory_size();
    static boost::interprocess::managed_shared_memory shared_mem_segment(
        boost::interprocess::open_or_create, shared_memory_name().c_str(), shm_size_bytes);
    if (shared_mem_segment.get_size() < shm_size_bytes) {
        size_t additional_bytes_required = shm_size_bytes - shared_mem_segment.get_size();
        boost::interprocess::managed_shared_memory::grow(shared_memory_name().c_str(), additional_bytes_required);
    }
    return shared_mem_segment;
}

std::string get_shared_mem_object_lock_name(const std::string &shm_object_name) {
    return fmt::format("{}_MUTEX", shm_object_name);
}

device_driver_t::device_driver_t() :
    num_drivers_initialized(0), processes_using_driver(get_shared_mem_segment().get_segment_manager()) {}

driver_control_and_lock_pair_t get_device_driver_controller(chip_id_t mmio_device_id) {
    std::string device_controller_name = get_shared_mem_object_name<device_driver_t>(mmio_device_id);
    device_driver_t *device_controller =
        get_shared_mem_segment().find_or_construct<device_driver_t>(device_controller_name.c_str())();
    std::string driver_controller_mutex_name = get_shared_mem_object_lock_name(device_controller_name);
    static boost::interprocess::named_mutex driver_controller_mutex(
        boost::interprocess::open_or_create, driver_controller_mutex_name.c_str());
    return {device_controller, driver_controller_mutex};
}

boost::interprocess::named_mutex &get_device_driver_sysmem_mutex(chip_id_t mmio_device_id) {
    std::string sysmem_mutex_name = get_shared_mem_object_name<device_driver_t>(mmio_device_id) + "_SysMemMUTEX";
    static boost::interprocess::named_mutex driver_sysmem_mutex(
        boost::interprocess::open_or_create, sysmem_mutex_name.c_str());
    return driver_sysmem_mutex;
}

void remove_device_driver_controller(chip_id_t mmio_device_id) {
    get_shared_mem_segment().destroy<device_driver_t>(
        get_shared_mem_object_name<device_driver_t>(mmio_device_id).c_str());
    std::string driver_mutex_name = get_shared_mem_object_name<device_driver_t>(mmio_device_id) + "_MUTEX";
    boost::interprocess::named_mutex::remove(driver_mutex_name.c_str());
    std::string sysmem_mutex_name = get_shared_mem_object_name<device_driver_t>(mmio_device_id) + "_SysMemMUTEX";
    boost::interprocess::named_mutex::remove(sysmem_mutex_name.c_str());
}

boost::interprocess::named_mutex &get_cluster_desc_yaml_mutex() {
    static boost::interprocess::named_mutex cluster_desc_yaml_mutex(
        boost::interprocess::open_or_create, "ClusterDescYaml_MUTEX");
    return cluster_desc_yaml_mutex;
}

const std::string dram_mem_blocks_name(chip_id_t device_id) {
    std::string dram_block_str = "DramBlocks_" + std::to_string(device_id);
    return dram_block_str;
}

const std::string l1_mem_blocks_name(chip_id_t device_id) {
    std::string l1_block_str = "L1Blocks_" + std::to_string(device_id);
    return l1_block_str;
}

device_state_t::device_state_t(chip_id_t device_id) : num_initializations(0) {}

allocator_t::allocator_t(size_t mem_size) {
    boost::interprocess::offset_ptr<block_t> block =
        static_cast<block_t *>(get_shared_mem_segment().allocate(sizeof(block_t)));
    block->address = 0;
    block->size = mem_size;
    block->prev = 0;
    block->next = 0;
    block->prev_free = 0;
    block->next_free = 0;
    this->block_head = block;
    this->free_block_head = block;
    this->free_block_tail = block;
}

allocator_t::~allocator_t() {
    boost::interprocess::offset_ptr<block_t> prev = 0;
    boost::interprocess::offset_ptr<block_t> current = this->block_head;
    while (current != 0) {
        prev = current;
        current = current->next;
        get_shared_mem_segment().deallocate(prev.get());
    }
    this->block_head = 0;
    this->free_block_head = 0;
    this->free_block_tail = 0;
}

device_state_and_lock_pair_t get_device_state_controller(chip_id_t device_id) {
    std::string device_state_str = get_shared_mem_object_name<device_state_t>(device_id);
    device_state_t *device_state =
        get_shared_mem_segment().find_or_construct<device_state_t>(device_state_str.c_str())(device_id);
    std::string device_state_mutex_name = get_shared_mem_object_lock_name(device_state_str);
    static boost::interprocess::named_mutex device_state_mutex(
        boost::interprocess::open_or_create, device_state_mutex_name.c_str());
    return {device_state, device_state_mutex};
}

boost::interprocess::named_mutex &get_device_lock(chip_id_t device_id) {
    std::string device_lock_name = get_shared_mem_object_name<device_state_t>(device_id) + "_DeviceLockMUTEX";
    static boost::interprocess::named_mutex device_lock(boost::interprocess::open_or_create, device_lock_name.c_str());
    return device_lock;
}

boost::interprocess::named_sharable_mutex &get_launch_program_mutex(chip_id_t device_id) {
    std::string launch_program_mutex_name =
        get_shared_mem_object_name<device_state_t>(device_id) + "_LaunchProgramMUTEX";
    static boost::interprocess::named_sharable_mutex program_launch_mutex(
        boost::interprocess::open_or_create, launch_program_mutex_name.c_str());
    return program_launch_mutex;
}

void initialize_allocator(const std::string &allocator_name, size_t bank_size) {
    allocator_t *allocator = get_shared_mem_segment().find_or_construct<allocator_t>(allocator_name.c_str())(bank_size);
}

allocator_and_lock_pair_t get_allocator(const std::string &allocator_name) {
    std::pair<allocator_t *, size_t> allocator_pair =
        get_shared_mem_segment().find<allocator_t>(allocator_name.c_str());
    if (allocator_pair.second == 0) {
        tt::log_fatal("No {} allocator_t has been constructed in shared memory", allocator_name);
    }
    std::string allocator_mutex_name = get_shared_mem_object_lock_name(allocator_name);
    static boost::interprocess::named_mutex allocator_mutex(
        boost::interprocess::open_or_create, allocator_mutex_name.c_str());
    return {allocator_pair.first, allocator_mutex};
}

void remove_device_state_and_allocators(chip_id_t device_id) {
    std::string device_state_name = get_shared_mem_object_name<device_state_t>(device_id);
    get_shared_mem_segment().destroy<device_state_t>(device_state_name.c_str());
    std::string state_mutex_name = get_shared_mem_object_lock_name(device_state_name);
    boost::interprocess::named_mutex::remove(state_mutex_name.c_str());
    std::string device_lock_name = device_state_name + "_DeviceLockMUTEX";
    boost::interprocess::named_mutex::remove(device_lock_name.c_str());
    std::string launch_program_mutex_name = device_state_name + "_LaunchProgramMUTEX";
    boost::interprocess::named_sharable_mutex::remove(launch_program_mutex_name.c_str());

    std::string dram_blocks_name = dram_mem_blocks_name(device_id);
    get_shared_mem_segment().destroy<allocator_t>(dram_blocks_name.c_str());
    std::string dram_alloc_mutex_name = get_shared_mem_object_lock_name(dram_blocks_name);
    boost::interprocess::named_mutex::remove(dram_alloc_mutex_name.c_str());

    std::string l1_blocks_name = l1_mem_blocks_name(device_id);
    get_shared_mem_segment().destroy<allocator_t>(l1_blocks_name.c_str());
    std::string l1_alloc_mutex_name = get_shared_mem_object_lock_name(l1_blocks_name);
    boost::interprocess::named_mutex::remove(l1_alloc_mutex_name.c_str());
}

sys_mem_cq_write_interface_t::sys_mem_cq_write_interface_t() {
    this->fifo_wr_ptr = CQ_START >> 4;  // In 16B words
    this->fifo_wr_toggle = 0;  // This is used for the edge case where we wrap and our read pointer has not yet moved
}

cq_write_interface_and_lock_pair_t get_cq_write_interface(chip_id_t mmio_device_id) {
    std::string cq_interface_str = get_shared_mem_object_name<sys_mem_cq_write_interface_t>(mmio_device_id);
    sys_mem_cq_write_interface_t *cq_write_interface =
        get_shared_mem_segment().find_or_construct<sys_mem_cq_write_interface_t>(cq_interface_str.c_str())();
    std::string cq_interface_mutex_str = get_shared_mem_object_lock_name(cq_interface_str);
    static boost::interprocess::named_sharable_mutex cq_write_interface_mutex(
        boost::interprocess::open_or_create, cq_interface_mutex_str.c_str());
    return {cq_write_interface, cq_write_interface_mutex};
}

boost::interprocess::named_mutex &get_finish_command_mutex(chip_id_t mmio_device_id) {
    std::string finish_cmd_mutex_name =
        get_shared_mem_object_name<sys_mem_cq_write_interface_t>(mmio_device_id) + "_FinishCommandMUTEX";
    static boost::interprocess::named_mutex finish_cmd_mutex(
        boost::interprocess::open_or_create, finish_cmd_mutex_name.c_str());
    return finish_cmd_mutex;
}

void remove_cq_write_interface(chip_id_t mmio_device_id) {
    std::string cq_interface_str = get_shared_mem_object_name<sys_mem_cq_write_interface_t>(mmio_device_id);
    get_shared_mem_segment().destroy<sys_mem_cq_write_interface_t>(cq_interface_str.c_str());
    std::string cq_wr_mutex_name = get_shared_mem_object_lock_name(cq_interface_str);
    boost::interprocess::named_sharable_mutex::remove(cq_wr_mutex_name.c_str());
    std::string finish_cmd_mutex_name = cq_interface_str + "_FinishCommandMUTEX";
    boost::interprocess::named_mutex::remove(finish_cmd_mutex_name.c_str());
}

void remove_shared_memory() {
    size_t num_devices = tt::Cluster::instance().number_of_devices();
    size_t num_mmio_devices = tt::Cluster::instance().number_of_pci_devices();
    for (chip_id_t mmio_dev_id = 0; mmio_dev_id < num_mmio_devices; mmio_dev_id++) {
        remove_device_driver_controller(mmio_dev_id);
        remove_cq_write_interface(mmio_dev_id);
    }

    for (chip_id_t dev_id = 0; dev_id < num_devices; dev_id++) {
        remove_device_state_and_allocators(dev_id);
    }

    if (not get_shared_mem_segment().all_memory_deallocated()) {
        tt::log_fatal("Memory leak in removing shared memory! Shared memory structures have not been deallocated");
    }

    boost::interprocess::shared_memory_object::remove(shared_memory_name().c_str());
}

}  // namespace tt::concurrent
