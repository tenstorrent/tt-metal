/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <signal.h>

#include <boost/core/demangle.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/offset_ptr.hpp>
#include <boost/interprocess/sync/named_mutex.hpp>
#include <boost/interprocess/sync/named_sharable_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/interprocess/sync/sharable_lock.hpp>
#include <string>
#include <thread>

#include "tt_metal/hostdevcommon/common_runtime_address_map.h"
#include "tt_metal/third_party/umd/device/tt_cluster_descriptor_types.h"

// This file contains APIs to facilitate concurrent access to devices

namespace tt::concurrent {

////////////////////////////////////////////////////////////////////////////////////////////
//                      UMD related shared memory structures
////////////////////////////////////////////////////////////////////////////////////////////

// Alias an STL compatible allocator of pid_ts that allocates pid_t from the managed shared memory segment
// This allocator will allow to place containers holding pids in managed shared memory segments to allow tracking of
// zombie processes
typedef boost::interprocess::allocator<pid_t, boost::interprocess::managed_shared_memory::segment_manager>
    pid_allocator_t;
typedef boost::interprocess::vector<pid_t, pid_allocator_t> pid_vector_t;

// This allows multiple processes/threads to safely initialize, interface with, and close the UMD
// Host reads and writes do not need to be protected because UMD used named mutexes for synchronization
// There are num MMIO devices device_driver_t structs in shared memory
struct device_driver_t {
    device_driver_t();

    // Counts the number of times the device driver for a particular MMIO device has been initialized, count decrements
    // on closing device driver
    uint32_t num_drivers_initialized;

    // Tracks pids that have initialized device_driver_t for a given MMIO device
    // Process adds its pid when initializing a device driver and removes it on closing device driver
    pid_vector_t processes_using_driver;
};

////////////////////////////////////////////////////////////////////////////////////////////
//                      tt_metal::Device related shared memory structures
////////////////////////////////////////////////////////////////////////////////////////////

// This holds shared state and synchronization primitives for a single device (1:1 with tt_metal::Device)
// Currently, multiple processes cannot use the same device concurrently
// There are total num devices (sum of MMIO and remote as seen by tt_ClusterDescriptor) device_state_t structs in shared
// memory
struct device_state_t {
    device_state_t(chip_id_t device_id);

    // Counts the number of times the tt_metal::Device with a given ID has been created, count decrements on closing
    // device Currently this can be at most 1 because only one process can use a device at a given time and device is
    // intiialized once per process
    uint32_t num_initializations;

    // Indicates whether this device was used in fast dispatch
    // Device can only be in one mode. To switch modes, all processes must close device and reopen it
    bool fast_dispatch_mode;
};

// Global memory allocator shared memory structures
struct block_t {
    uint64_t address;
    uint64_t size;
    boost::interprocess::offset_ptr<block_t> prev = 0;
    boost::interprocess::offset_ptr<block_t> next = 0;
    boost::interprocess::offset_ptr<block_t> prev_free = 0;
    boost::interprocess::offset_ptr<block_t> next_free = 0;
};

struct allocator_t {
    allocator_t(size_t mem_size);
    ~allocator_t();
    boost::interprocess::offset_ptr<block_t> block_head = 0;
    boost::interprocess::offset_ptr<block_t> free_block_head = 0;
    boost::interprocess::offset_ptr<block_t> free_block_tail = 0;
};

// Command queue shared memory structure
// Tracks system memory Command Queue write pointer, which needs to be synchronized across all processes/threads
// targeting one device Currently there is one sys_mem_cq_write_interface_t per MMIO device
struct sys_mem_cq_write_interface_t {
    sys_mem_cq_write_interface_t();
    // Equation for fifo size is
    // | fifo_wr_ptr + command size B - fifo_rd_ptr |
    // Space available would just be fifo_limit - fifo_size
    const uint32_t fifo_size = ((1024 * 1024 * 1024) - CQ_START) >> 4;
    const uint32_t fifo_limit = ((1024 * 1024 * 1024) >> 4) - 1;  // Last possible FIFO address

    uint32_t fifo_wr_ptr;
    bool fifo_wr_toggle;
};

typedef std::pair<device_driver_t *, boost::interprocess::named_mutex &> driver_control_and_lock_pair_t;
typedef std::pair<device_state_t *, boost::interprocess::named_mutex &> device_state_and_lock_pair_t;
typedef std::pair<allocator_t *, boost::interprocess::named_mutex &> allocator_and_lock_pair_t;
typedef std::pair<sys_mem_cq_write_interface_t *, boost::interprocess::named_sharable_mutex &>
    cq_write_interface_and_lock_pair_t;

template <typename ShmT>
std::string get_shared_mem_object_name(chip_id_t device_id) {
    return boost::core::demangle(typeid(ShmT).name()) + "_" + std::to_string(device_id);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                    Mutexes that protect shared memory structures and other critical sections
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// All mutexes are implemented using `named_mutex` over  `interprocess_mutex` to ensure we can recover after ungraceful
// termination where mutexes may not get unlocked. This is because `named_mutex` allows a thread/process that did not
// lock the mutex to remove it, whereas `interprocess_mutex` throws if the non-locking thread/mutex tries to remove it
// `named_mutex` cannot be allocated in shared memory
// Note: limitation on `interprocess_mutex` is addressed in Boost 1.78 release which switches to using robust mutex:
//  https://www.boost.org/doc/libs/1_83_0/doc/html/interprocess/acknowledgements_notes.html#interprocess.acknowledgements_notes.release_notes.release_notes_boost_1_78_00
//      - see https://github.com/boostorg/interprocess/issues/65 and https://github.com/boostorg/interprocess/pull/67

boost::interprocess::managed_shared_memory &get_shared_mem_segment();

// Get `device_driver_t` for associated MMIO device and named mutex that protected modifications to device_driver_t
// control variables
driver_control_and_lock_pair_t get_device_driver_controller(chip_id_t mmio_device_id);

// This protects concurrent writes/reads from system memory. System memory can only be accessed by MMIO device
boost::interprocess::named_mutex &get_device_driver_sysmem_mutex(chip_id_t mmio_device_id);

void remove_device_driver_controller(chip_id_t mmio_device_id);

// No ability to change where cluster desc yaml gets generated, for now protect races in writing cluster desc
boost::interprocess::named_mutex &get_cluster_desc_yaml_mutex();

const std::string dram_mem_blocks_name(chip_id_t device_id);

const std::string l1_mem_blocks_name(chip_id_t device_id);

// Get `device_state_t` for associated device and named mutex that protected modifications to device_state_t control
// variables
device_state_and_lock_pair_t get_device_state_controller(chip_id_t device_id);

// Named mutex used to lock a given device for the duration of a process
// When multiple processes can run concurrently on a device, this can be removed
boost::interprocess::named_mutex &get_device_lock(chip_id_t device_id);

// Writes to L1 acquire shareable lock because L1 buffers are guranteed to not overlap since there is a global allocator
// per device Running a program acquires the exclusive lock because L1 buffers could clash with circular buffers (CBs)
// since CB allocation is local to a program
boost::interprocess::named_sharable_mutex &get_launch_program_mutex(chip_id_t device_id);

void initialize_allocator(const std::string &allocator_name, size_t bank_size);

// Get `allocator_t` and named mutex that protected modifications to free/allocated block list
allocator_and_lock_pair_t get_allocator(const std::string &allocator_name);

void remove_device_state_and_allocators(chip_id_t device_id);

// Get `sys_mem_cq_write_interface_t` and named mutex that protected modifications to `sys_mem_cq_write_interface_t`
// write pointer
cq_write_interface_and_lock_pair_t get_cq_write_interface(chip_id_t mmio_device_id);

// Finish command polls until done signal is propagated. The poll needs to be protected to ensure the finish command is
// atomic.
boost::interprocess::named_mutex &get_finish_command_mutex(chip_id_t mmio_device_id);

void remove_cq_write_interface(chip_id_t mmio_device_id);

// Destory managed_shared_memory object to unmap and free resoures
// All allocated structures need to be removed
void remove_shared_memory();

}  // namespace tt::concurrent
