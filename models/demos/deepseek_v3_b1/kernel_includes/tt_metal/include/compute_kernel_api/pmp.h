// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common.h"
#include "api/debug/dprint.h"

namespace ckernel {

struct pmp_zone {
    uint64_t start = 0;
    uint64_t end = 0;
    uint64_t min = 0;
    uint64_t max = 0;
    uint64_t sum = 0;
    uint64_t count = 0;
    uint64_t inner = 0;
};

struct pmp_sync {
    uint64_t count = 0;
    int64_t t0_min = 0;
    int64_t t0_max = 0;
    uint64_t t0_sum = 0;
    int64_t t1_min = 0;
    int64_t t1_max = 0;
    uint64_t t1_sum = 0;
    int64_t t2_min = 0;
    int64_t t2_max = 0;
    uint64_t t2_sum = 0;
    int64_t time_min = 0;
    int64_t time_max = 0;
    uint64_t time_sum = 0;
};

ALWI void _pmp_barrier() {
    volatile std::uint32_t* base_address = (std::uint32_t*)MEM_LLK_DEBUG_BASE;
    tensix_sync();
    UNPACK((base_address[1] = 1));
    MATH((base_address[2] = 2));
    PACK((base_address[3] = 3));
    while (base_address[1] != 1) {
        asm("nop");
    }
    while (base_address[2] != 2) {
        asm("nop");
    }
    while (base_address[3] != 3) {
        asm("nop");
    }
    UNPACK((base_address[5] = 5));
    MATH((base_address[6] = 6));
    PACK((base_address[7] = 7));
    while (base_address[5] != 5) {
        asm("nop");
    }
    while (base_address[6] != 6) {
        asm("nop");
    }
    while (base_address[7] != 7) {
        asm("nop");
    }
    UNPACK((base_address[1] = 0));
    MATH((base_address[2] = 0));
    PACK((base_address[3] = 0));
    while (base_address[1] != 0) {
        asm("nop");
    }
    while (base_address[2] != 0) {
        asm("nop");
    }
    while (base_address[3] != 0) {
        asm("nop");
    }
    UNPACK((base_address[5] = 0));
    MATH((base_address[6] = 0));
    PACK((base_address[7] = 0));
}

ALWI volatile pmp_zone* _pmp_get_zone(uint32_t zone_id = 0, uint32_t offset = 0) {
    volatile pmp_zone* zones = (volatile pmp_zone*)MEM_LLK_DEBUG_BASE;
    volatile pmp_zone* zone = nullptr;
    if (offset != 0) {
        zone = &zones[1 + 3 * zone_id + offset];
    } else {
        UNPACK((zone = &zones[1 + 3 * zone_id]));
        MATH((zone = &zones[2 + 3 * zone_id]));
        PACK((zone = &zones[3 + 3 * zone_id]));
    }
    return zone;
}

ALWI void pmp_init_zone(uint32_t zone_id = 0) {
#if defined(DEBUG_PRINT_ENABLED)
    volatile pmp_zone* zone = _pmp_get_zone(zone_id);
    zone->start = 0;
    zone->end = 0;
    zone->min = 0;
    zone->max = 0;
    zone->sum = 0;
    zone->count = 0;
    zone->inner = 0;
#endif
}

ALWI void pmp_start_zone(uint32_t zone_id = 0) {
#if defined(DEBUG_PRINT_ENABLED)
    volatile pmp_zone* zone = _pmp_get_zone(zone_id);
    _pmp_barrier();
    zone->start = ckernel::read_wall_clock();
#endif
}

ALWI void pmp_end_zone(uint32_t zone_id = 0, uint32_t inner_loops = 1) {
#if defined(DEBUG_PRINT_ENABLED)
    _pmp_barrier();
    uint64_t end = ckernel::read_wall_clock();
    volatile pmp_zone* zone_ptr = _pmp_get_zone(zone_id);
    uint64_t duration = end - zone_ptr->start;
    if (zone_ptr->count == 0) {
        zone_ptr->min = duration;
        zone_ptr->max = duration;
    } else {
        zone_ptr->min = zone_ptr->min < duration ? zone_ptr->min : duration;
        zone_ptr->max = zone_ptr->max > duration ? zone_ptr->max : duration;
    }
    zone_ptr->end = end;
    zone_ptr->sum += duration;
    zone_ptr->count += 1;
    zone_ptr->inner += inner_loops;
    if (zone_ptr->count * inner_loops != zone_ptr->inner) {
        zone_ptr->inner = 0;
    }
#endif
}

ALWI void pmp_update_sync(pmp_sync& sync, uint32_t zone_id = 0) {
#if defined(DEBUG_PRINT_ENABLED)
    _pmp_barrier();
    volatile pmp_zone* t0 = _pmp_get_zone(zone_id, 0);
    volatile pmp_zone* t1 = _pmp_get_zone(zone_id, 1);
    volatile pmp_zone* t2 = _pmp_get_zone(zone_id, 2);
    uint64_t avg_start = (t0->start + t1->start + t2->start) / 3;
    int64_t t0_offset = t0->start - avg_start;
    int64_t t1_offset = t1->start - avg_start;
    int64_t t2_offset = t2->start - avg_start;
    int64_t time = (t1->end - t1->start) - (t0->end - t0->start);
    if (sync.count == 0) {
        sync.t0_min = t0_offset;
        sync.t0_max = t0_offset;
        sync.t1_min = t1_offset;
        sync.t1_max = t1_offset;
        sync.t2_min = t2_offset;
        sync.t2_max = t2_offset;
        sync.time_min = time;
        sync.time_max = time;
    } else {
        sync.t0_min = t0_offset < sync.t0_min ? t0_offset : sync.t0_min;
        sync.t0_max = t0_offset > sync.t0_max ? t0_offset : sync.t0_max;
        sync.t1_min = t1_offset < sync.t1_min ? t1_offset : sync.t1_min;
        sync.t1_max = t1_offset > sync.t1_max ? t1_offset : sync.t1_max;
        sync.t2_min = t2_offset < sync.t2_min ? t2_offset : sync.t2_min;
        sync.t2_max = t2_offset > sync.t2_max ? t2_offset : sync.t2_max;
        sync.time_min = time < sync.time_min ? time : sync.time_min;
        sync.time_max = time > sync.time_max ? time : sync.time_max;
    }
    sync.count += 1;
    sync.t0_sum += t0_offset > 0 ? t0_offset : -t0_offset;
    sync.t1_sum += t1_offset > 0 ? t1_offset : -t1_offset;
    sync.t2_sum += t2_offset > 0 ? t2_offset : -t2_offset;
    sync.time_sum += time > 0 ? time : -time;
#endif
}

ALWI void pmp_print_time(uint32_t zone_id = 0) {
#if defined(DEBUG_PRINT_ENABLED)
    volatile pmp_zone* zone_ptr = _pmp_get_zone(zone_id);
    DEVICE_PRINT_UNPACK(
        "Zone: min={} max={} sum={} count={} inner={}\n",
        (uint64_t)zone_ptr->min,
        (uint64_t)zone_ptr->max,
        (uint64_t)zone_ptr->sum,
        (uint64_t)zone_ptr->count,
        (uint64_t)zone_ptr->inner);
#endif
}

ALWI void pmp_print_sync(pmp_sync& sync) {
#if defined(DEBUG_PRINT_ENABLED)
    DEVICE_PRINT_UNPACK(
        "Sync: t0_offset_min={} t0_offset_max={} t0_offset_sum={} t1_offset_min={} t1_offset_max={} t1_offset_sum={} "
        "t2_offset_min={} t2_offset_max={} t2_offset_sum={} time_min={} time_max={} time_sum={} count={}\n",
        (int64_t)sync.t0_min,
        (int64_t)sync.t0_max,
        (uint64_t)sync.t0_sum,
        (int64_t)sync.t1_min,
        (int64_t)sync.t1_max,
        (uint64_t)sync.t1_sum,
        (int64_t)sync.t2_min,
        (int64_t)sync.t2_max,
        (uint64_t)sync.t2_sum,
        (int64_t)sync.time_min,
        (int64_t)sync.time_max,
        (uint64_t)sync.time_sum,
        (uint64_t)sync.count);
#endif
}

template <typename ExecuteCallable, typename LogCallable>
ALWI void pmp_run(
    ExecuteCallable execute,
    LogCallable log,
    uint32_t zone_id = 0,
    uint32_t inner_loops = 1024,
    uint32_t outer_loops = 1024) {
    pmp_init_zone(zone_id);
    pmp_sync sync;

#if not defined(DEBUG_PRINT_ENABLED)
    inner_loops = 1;
    outer_loops = 1;
#endif

    for (uint32_t i = 0; i < outer_loops; i++) {
        pmp_start_zone(zone_id);
        for (uint32_t j = 0; j < inner_loops; j++) {
            execute();
        }
        pmp_end_zone(zone_id, inner_loops);
        pmp_update_sync(sync, zone_id);
    }

#if defined(DEBUG_PRINT_ENABLED)
    log();
#endif
    pmp_print_time(zone_id);
    pmp_print_sync(sync);
}

}  // namespace ckernel
