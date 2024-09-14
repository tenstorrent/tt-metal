// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <mutex>

#include "hostdevcommon/common_values.hpp"
#include "impl/dispatch/work_executor.hpp"
#include "tt_metal/impl/allocator/basic_allocator.hpp"
#include "tt_metal/impl/allocator/l1_banking_allocator.hpp"
#include "tt_metal/impl/kernels/data_types.hpp"
#include "tt_metal/impl/trace/trace_buffer.hpp"
#include "tt_metal/impl/program/program_device_map.hpp"
#include "tt_metal/jit_build/build.hpp"
#include "llrt/tt_cluster.hpp"
#include "llrt/hal.hpp"
#include "dev_msgs.h"
#include "tt_metal/impl/dispatch/command_queue_interface.hpp"
#include "program_cache.hpp"

namespace tt {

namespace tt_metal {

// Fwd declares
enum class BufferType;
class Buffer;
class Program;
class JitBuildEnv;
class HWCommandQueue;
class CommandQueue;

namespace detail {
// TODO(agrebenisan): Need device to hold onto command queue programs,
// but the Program type is incomplete by this point. I can have
// a unique_ptr of incomplete type as long as I override the default
// delete function.
struct ProgramDeleter {
    void operator()(Program* p);
};

class TraceDescriptor;

}

using on_close_device_callback = std::function<void ()>;

// TODO: These should be moved into arch specific host files that get exported here
static constexpr float  EPS_GS = 0.001953125f;
static constexpr float  EPS_WHB0 = 1.19209e-7f;
static constexpr float  EPS_BH = EPS_WHB0;

static constexpr float  NAN_GS = 6.9752e19;
static constexpr float  NAN_WHB0 = 7.0040e+19;
static constexpr float  NAN_BH = NAN_WHB0;

static constexpr float  INF_GS = 1.6948e38;
static constexpr float  INF_WHB0 = 1.7014e+38;
static constexpr float  INF_BH = INF_WHB0;

// A physical PCIexpress Tenstorrent device
class Device;

ARCH DeviceArch(const Device *device);

chip_id_t DeviceId(const Device *device);

bool DeviceIsInitialized(Device *device);

int DeviceNumDramChannels(Device *device);

uint32_t DeviceL1SizePerCore(Device *device);

CoreCoord DeviceLogicalGridSize(Device *device);

CoreCoord DeviceComputeWithStorageGridSize(const Device *device);

CoreCoord DeviceDramGridSize(Device *device);

CoreCoord DevicePhysicalCoreFromLogicalCore(const Device *device, const CoreCoord &logical_core, CoreType core_type);

CoreCoord DeviceWorkerCoreFromLogicalCore(const Device *device, const CoreCoord &logical_core);

CoreCoord DeviceEthernetCoreFromLogicalCore(const Device *device, const CoreCoord &logical_core);

std::vector<CoreCoord> DeviceGetEthernetSockets(const Device *device, chip_id_t connected_chip_id);

uint32_t DeviceNumBanks(Device *device, BufferType buffer_type);

CoreCoord DeviceDramCoreFromDramChannel(Device *device, uint32_t dram_channel);

int32_t DeviceBankOffset(Device *device, BufferType buffer_type, uint32_t bank_id);

const std::vector<uint32_t> &DeviceBankIdsFromLogicalCore(Device *device, BufferType buffer_type, const CoreCoord &logical_core);

void DeviceDeallocateBuffers(Device *device);

float DeviceSfpuEps(Device *device);
float DeviceSfpuNan(Device *device);
float DeviceSfpuInf(Device *device);

CommandQueue &DeviceCommandQueue(Device *device, size_t cq_id = 0);

void DeviceBeginTrace(Device *device, uint8_t cq_id, uint32_t tid);
void DeviceEndTrace(Device *device, uint8_t cq_id, uint32_t tid);
void DeviceReplayTrace(Device *device, uint8_t cq_id, uint32_t tid, bool blocking);
void DeviceReleaseTrace(Device *device, uint32_t tid);

void DevicePushWork(Device *device, std::function<void()> &&work, bool blocking = false);

void DeviceSynchronize(Device *device);

void DeviceEnableAsync(Device *device, bool enable);

WorkExecutorMode DeviceGetWorkerMode(Device *device);

program_cache::detail::ProgramCache &DeviceGetProgramCache(Device *device);

void DeviceEnableProgramCache(Device *device);
void DeviceDisableAndClearProgramCache(Device *device);

size_t DeviceNumProgramCacheEntries(Device *device);

bool DeviceInMainThread(Device *device);

Allocator *DeviceGetAllocator(Device *device);

std::set<CoreCoord> &DeviceGetComputeCores(Device *device);

WorkExecutor &DeviceGetWorkExecutor(Device *device);

}  // namespace tt_metal

}  // namespace tt
