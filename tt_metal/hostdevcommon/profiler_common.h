// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
*
* Ennums and defines shared between host and device profiler.
*
*/
#pragma once

#define CC_MAIN_START          1U
#define CC_KERNEL_MAIN_START   2U
#define CC_KERNEL_MAIN_END     3U
#define CC_MAIN_END            4U

namespace kernel_profiler{
/**
 * L1 buffer structure for profiler markers
 * _____________________________________________________________________________________________________
 *|                  |                        |              |             |             |              |
 *| Buffer end index | Dropped marker counter | 1st timer_id | 1st timer_L | 1st timer_H | 2nd timer_id | ...
 *|__________________|________________________|______________|_____________|_____________|______________|
 *
 * */

enum BufferIndex {BUFFER_END_INDEX, DROPPED_MARKER_COUNTER, MARKER_DATA_START};

enum TimerDataIndex {TIMER_ID, TIMER_VAL_L, TIMER_VAL_H, TIMER_DATA_UINT32_SIZE};

}
