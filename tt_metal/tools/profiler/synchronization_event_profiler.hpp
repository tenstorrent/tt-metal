// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(PROFILE_KERNEL)

#include "kernel_profiler.hpp"

struct SynchronizationEventSignal {
    uint64_t cb_id;
};

struct SynchronizationEventWaitStart {
    uint64_t cb_id;
};

struct SynchronizationEventWaitEnd {
    uint64_t cb_id;
};

//////////////////////////////
// CB events
//////////////////////////////

#define RECORD_CB_PUSH_BACK(cb_id) \
    kernel_profiler::              \
        timeStampedData<1000, kernel_profiler::DoingDispatch::DISPATCH, kernel_profiler::PacketTypes::TS_DATA>(cb_id)

#define RECORD_CB_WAIT_FRONT_START(cb_id) \
    kernel_profiler::                     \
        timeStampedData<1001, kernel_profiler::DoingDispatch::DISPATCH, kernel_profiler::PacketTypes::TS_DATA>(cb_id)

#define RECORD_CB_WAIT_FRONT_END(cb_id) \
    kernel_profiler::                   \
        timeStampedData<1002, kernel_profiler::DoingDispatch::DISPATCH, kernel_profiler::PacketTypes::TS_DATA>(cb_id)

//////////////////////////////
// Semaphore events
//////////////////////////////
#define RECORD_SEMAPHORE_SET(semaphore_address)                                                                 \
    kernel_profiler::                                                                                           \
        timeStampedData<1003, kernel_profiler::DoingDispatch::DISPATCH, kernel_profiler::PacketTypes::TS_DATA>( \
            semaphore_address)

#define RECORD_SEMAPHORE_SET_REMOTE(semaphore_address)                                                          \
    kernel_profiler::                                                                                           \
        timeStampedData<1004, kernel_profiler::DoingDispatch::DISPATCH, kernel_profiler::PacketTypes::TS_DATA>( \
            semaphore_address)

#define RECORD_SEMAPHORE_WAIT_START(semaphore_address)                                                          \
    kernel_profiler::                                                                                           \
        timeStampedData<1005, kernel_profiler::DoingDispatch::DISPATCH, kernel_profiler::PacketTypes::TS_DATA>( \
            semaphore_address)

#define RECORD_SEMAPHORE_WAIT_END(semaphore_address)                                                            \
    kernel_profiler::                                                                                           \
        timeStampedData<1006, kernel_profiler::DoingDispatch::DISPATCH, kernel_profiler::PacketTypes::TS_DATA>( \
            semaphore_address)

#else
// Null macros, so this header is safe to include from a non-profiled build.
// kernel_profiler.hpp defines the same nulls in its own disabled branch, since it
// only includes this header when PROFILE_KERNEL is set; these cover a direct
// include (e.g. cb_api.h, which uses the macros but includes no profiler header).

#define RECORD_CB_PUSH_BACK(cb_id) (void(sizeof(cb_id)))
#define RECORD_CB_WAIT_FRONT_START(cb_id) (void(sizeof(cb_id)))
#define RECORD_CB_WAIT_FRONT_END(cb_id) (void(sizeof(cb_id)))

#define RECORD_SEMAPHORE_SET(semaphore_address) (void(sizeof(semaphore_address)))
#define RECORD_SEMAPHORE_SET_REMOTE(semaphore_address) (void(sizeof(semaphore_address)))
#define RECORD_SEMAPHORE_WAIT_START(semaphore_address) (void(sizeof(semaphore_address)))
#define RECORD_SEMAPHORE_WAIT_END(semaphore_address) (void(sizeof(semaphore_address)))

#endif
